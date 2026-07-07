import numpy as np

# A synapse counts as having "participated" in the current integration window if
# its (leaky) trace accumulator carries any residual charge above this floor.
PARTICIPATION_EPS = 1e-6

# ---------------------------------------------------------------------------
# Fixed-point convention for gate weights, leak controls, and thresholds.
#
# Per Claude_Integer_Defaults_Prompt.md: the DEFAULT numeric representation for
# these three categories is defined in integer / integer-rational form rather
# than as bare float literals, so a default-configured model carries no
# floating-point constants in those categories. UNIT is the fixed-point scale
# for weight/threshold magnitudes (1 model unit == 1/UNIT); LEAK_SCALE is the
# denominator for leak-rate numerators (a leak of 0.01 is stored as the integer
# numerator 10 over LEAK_SCALE).
#
# SCOPE (this is the "whether model state is fully integer" deliverable): the
# CONTROLS -- thresholds, gate init/cap, and leak rates -- are integer-derived.
# Runtime membrane potentials and weights are still stored as float; a full
# integer-state conversion of the plasticity arithmetic (quadratic saturation,
# budget renormalization, homeostatic scaling) was deliberately DEFERRED so the
# hand-tuned dynamics are preserved bit-for-bit. The leak UPDATE below is
# rewritten to the fixed-point num/scale form, and the leak control is stored
# as the integer numerator self._leak_num (see the leak_rate property).
UNIT = 1000
LEAK_SCALE = 1000


def to_units(x):
    """Float magnitude -> integer fixed-point units (nearest). Use for
    human-supplied gate/threshold values that need an integer representation."""
    return int(round(x * UNIT))


def to_float(u):
    """Integer fixed-point units -> float magnitude (explicit display helper)."""
    return u / UNIT


def leak_num(rate):
    """Leak fraction (float) -> integer leak numerator over LEAK_SCALE."""
    return int(round(rate * LEAK_SCALE))


def div_round(n, d):
    """Deterministic integer rounding division (round-half-away-from-zero),
    provided per the prompt for a future full integer-state conversion where
    truncation drift would matter. Not needed on the current float-state path."""
    return (n + d // 2) // d if n >= 0 else -((-n + d // 2) // d)


def _distribution_entropy(x):
    """Shannon entropy (nats) of the non-negative values in x, treated as a
    distribution after normalization. Silent/zero entries are dropped so a
    perfectly selective synapse set has entropy 0 and a flat one has log(n)."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    s = x.sum()
    if x.size == 0 or s <= 0:
        return 0.0
    p = x / s
    return float(-(p * np.log(p)).sum())


def _concentration(x):
    """Herfindahl-Hirschman concentration of the non-negative values in x,
    normalized to a distribution: sum of squared shares. Ranges from 1/n
    (perfectly spread) to 1 (all mass on one synapse) -- the higher, the more
    the budget/confidence has concentrated onto a few synapses."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    s = x.sum()
    if x.size == 0 or s <= 0:
        return 0.0
    p = x / s
    return float((p * p).sum())


class Neuron:
    """
    A basic spiking neuron model with charge accumulation, refractory period,
    and activity-dependent weight updates.
    
    Both excitatory and inhibitory neurons use the same dynamics;
    the difference is in the sign of their synaptic weights.
    """
    
    def __init__(self, n_inputs, threshold=1000 / UNIT, refractory_period=2,
                 weight_init_range=(-500 / UNIT, 500 / UNIT), learning_rate=0.1,
                 weight_cap=1000 / UNIT, leak_rate=10 / LEAK_SCALE,
                 inhibitory_learning_rate=0.05, inhibitory_weight_cap=None,
                 excitatory_saturation_cap=None,
                 confidence_init=0.10,
                 homeostasis=False, ca_rate=0.01, ca_target=0.02, ca_band=0.5,
                 homeo_up=0.01, homeo_down=0.01,
                 homeo_budget_min=None, homeo_budget_max=None):
        """
        Initialize a neuron.

        Args:
            n_inputs (int): Number of input connections
            threshold (float): Firing threshold (constant)
            refractory_period (int): Refractory period in time steps (1ms each)
            weight_init_range (tuple): Min and max for random weight initialization
            learning_rate (float): Amount weights increase when neuron fires (excitatory plasticity)
            weight_cap (float): Maximum absolute value for weights (weights clipped to [-weight_cap, weight_cap]).
                Also serves as w_max, the saturating magnitude ceiling for inhibitory plasticity.
            leak_rate (float): Fraction of potential that leaks away per time step (0 = no leak, 1 = full leak)
            inhibitory_learning_rate (float): eta for the inhibitory-discharge plasticity rule
                (see apply_inhibition). 0 disables inhibitory learning.
            confidence_init (float): initial value of the per-synapse confidence
                array, which the ACTIVE confidence-consolidation system reads and
                updates (see confidence_consolidation / _update_weights). The old
                trace_mode / confidence_beta / confidence_gamma params of the
                archived credit-splitting rule were removed (see git history and
                Weight_Update_Unification.md for that rule).
            homeostasis (bool): enable homeostatic synaptic scaling -- a THIRD
                local plasticity system, independent of the two above (see
                _homeostatic_scaling). Off by default (preserves prior behavior).
            ca_rate (float): EMA rate of the neuron's own firing-rate sensor
                ("calcium"); small = slow, long-timescale average.
            ca_target (float): homeostatic firing-rate set-point. Scaling nudges
                the neuron toward firing this fraction of steps.
            ca_band (float): deadband half-width as a fraction of ca_target. No
                scaling while the sensor is within [target*(1-band), target*(1+band)].
            homeo_up / homeo_down (float): the FIXED multiplicative step applied to
                the excitatory resource when the neuron is chronically silent
                (grow) or chronically over-active (shrink). Deliberately a constant,
                not proportional to any error -- see _homeostatic_scaling.
            homeo_budget_min / homeo_budget_max (float|None): clamps on the
                homeostatic synaptic resource R.
        """
        # Neuron properties
        self.threshold = threshold          # Constant firing threshold
        self.resting_potential = 0.0        # Resting potential at 0
        self.refractory_period = refractory_period  # Refractory period
        self.refractory_timer = 0           # Counts down refractory period
        self.potential = self.resting_potential     # Current membrane potential
        
        # Weight properties
        self.weights = np.random.uniform(
            weight_init_range[0], weight_init_range[1], n_inputs
        )  # Random weight initialization
        # Ensure initial weights are within caps
        self.weights = np.clip(self.weights, -weight_cap, weight_cap)
        self.learning_rate = learning_rate  # Weight increase amount when firing (excitatory)
        self.weight_cap = weight_cap        # Maximum absolute weight value (also w_max for inhibition)
        self.leak_rate = leak_rate          # stored as integer numerator self._leak_num (see property)
        self.inhibitory_learning_rate = inhibitory_learning_rate  # eta for inhibitory plasticity
        # Saturation ceiling (w_max) for inhibitory gates. Kept separate from the
        # feedforward weight_cap so a saturated gate need not be strong enough to
        # fully reset the membrane (which would recreate hard-WTA collapse).
        # Defaults to weight_cap when None.
        self.inhibitory_weight_cap = inhibitory_weight_cap
        # Saturation ceiling (w_max) for the EXCITATORY quadratic term, kept
        # separate from the hard clip weight_cap -- mirrors inhibitory_weight_cap
        # above but for positive weights. The natural equilibrium of
        # dw = eta*p*(1-w^2/w_max) is w* = sqrt(w_max): if this stays at its
        # default (None -> weight_cap), a synapse can never approach the hard
        # cap whenever weight_cap > 1 (equilibrium undershoots it). Setting
        # this to weight_cap**2 makes sqrt(w_max) == weight_cap exactly, so a
        # habitually-participating synapse CAN reach full self-sufficiency
        # while still using the same saturating shape. Defaults to weight_cap
        # when None (old behavior, e.g. L2E's feedforward weights).
        self.excitatory_saturation_cap = excitatory_saturation_cap

        # Debug record of the inhibitory-discharge events from the most recent
        # apply_inhibition() call (one dict per event). See apply_inhibition.
        self.last_inhibitory_events = []

        # ARCHIVED: per-synapse eligibility trace. No longer read by
        # _update_weights (see that method) -- kept only so anything still
        # inspecting .trace keeps working; still accumulated/decayed as before.
        self.trace = np.zeros(n_inputs)

        # Per-synapse confidence array, seeded at confidence_init and updated by
        # the active confidence-consolidation system (see _update_weights).
        self.confidence = np.full(n_inputs, float(confidence_init))

        # Instantaneous participation signal for the charge-based excitatory
        # rule (see _update_weights): which lines delivered a spike in the
        # most recent receive_input() call. Set there; read in _update_weights.
        self._last_input_spikes = np.zeros(n_inputs)

        # Debug record of the most recent excitatory (_update_weights) event, in
        # "confidence" mode -- participation, confidence before/after, credits,
        # weight deltas, and budget before/after normalization. See _update_weights.
        self.last_excitatory_event = {}

        # Confidence-gated consolidation (opt-in; see Claude_Confidence_Consolidation_Plan.md).
        # A local, label-free consolidation system for positive feedforward gates:
        # gates that repeatedly participate in successful firing grow CONFIDENT and
        # (a) learn more slowly (confidence-gated potentiation, _update_weights),
        # (b) resist loser depression (_depress_losers), while (c) confidence itself
        # decays only after genuinely long disuse (_decay_confidence), so stale
        # neurons stay reusable. All flags default OFF so bare neurons, L1, and the
        # inhibitory neurons are unaffected; the engine enables it on L2E only.
        self.confidence_consolidation = False  # revive confidence, gate eta, decay
        self.loser_depression = False          # depress active gates on inhibitory loss
        self.conf_cap = None                   # effective mature gate value w_conf_cap (None -> weight_cap)
        self.conf_beta = 0.05                  # confidence EMA rate toward maturity (beta_C)
        self.eta_min = 0.05                    # plasticity floor fraction for mature gates
        self.eta_loss = 0.01                   # loser-depression rate eta_- (dimensionless)
        self.loss_gamma = 2                    # protect-small / punish-large exponent
        self.conf_rho_active = 1e-5            # confidence decay while recently active
        self.conf_rho_dead = 1e-3              # confidence decay once dead past the grace window
        self.conf_ca_dead = 0.002              # ca below this counts the neuron as inactive
        self.conf_grace = 5000                 # T_grace: inactive steps before dead-decay engages
        self.inactive_steps = 0                # long grace counter (see update())
        self.loser_depression_events = 0       # diagnostic counter

        # Signed-spike depression ("4a"; opt-in). On this neuron's own fire, the
        # spike carries +1 (pixel on / participating) or -1 (pixel off). Only ON
        # pixels deliver charge and potentiate (unchanged); OFF pixels deliver NO
        # charge but their positive gates are DEPRESSED, giving explicit negative
        # pressure for specialization instead of relying on decay/renormalization.
        # See _update_weights for the rule. Default OFF -> baseline unchanged.
        self.signed_depression = False         # enable OFF-gate depression on fire
        self.eta_off = 0.0                     # depression rate for inactive gates
        self.signed_depression_events = 0      # diagnostic counter

        # Homeostatic synaptic scaling (third local system; see _homeostatic_scaling).
        self.homeostasis = homeostasis
        self.ca_rate = ca_rate
        self.ca_target = ca_target
        self.ca_band = ca_band
        self.homeo_up = homeo_up
        self.homeo_down = homeo_down
        self.homeo_budget_min = homeo_budget_min
        self.homeo_budget_max = homeo_budget_max
        self.ca = 0.0                 # slow EMA of the neuron's OWN firing ("calcium")
        self.homeo_budget = None      # homeostatic excitatory resource R (lazy-init)

        # Optional homeostatic weight budget (uniform per-neuron): if set, positive
        # (excitatory) afferent weights are renormalized to sum to this value after
        # each update, so strengthening one synapse weakens the others.
        self.weight_budget = None

        # Optional floor for positive weights after renormalization (see
        # _apply_budget_and_cap). None = no floor (old behavior).
        self.min_positive_weight = None

        # Spike tracking
        self.last_spike_time = -np.inf      # Time of last spike
        self.spiked = False                 # Did neuron spike in current step?

    @property
    def leak_rate(self):
        """Leak fraction per step. The stored CONTROL is the integer numerator
        self._leak_num over LEAK_SCALE (fixed-point leak); this float view is
        used for the trace decay and for display. See the module header."""
        return self._leak_num / LEAK_SCALE

    @leak_rate.setter
    def leak_rate(self, value):
        self._leak_num = leak_num(value)

    def receive_input(self, input_spikes):
        """
        Accumulate charge based on weighted inputs.

        Args:
            input_spikes (array-like): Binary array indicating which inputs spiked (1) or not (0)
        """
        # Only accumulate charge if not in refractory period
        if self.refractory_timer <= 0:
            input_spikes = np.asarray(input_spikes, dtype=float)
            # Charge accumulation: sum of weight * input for all connections
            input_current = np.dot(self.weights, input_spikes)
            self.potential += input_current
            # Track which lines delivered the charge (un-summed membrane).
            # ARCHIVED: no longer read by _update_weights (see that method) --
            # kept only so anything still inspecting .trace keeps working.
            self.trace += input_spikes
            # Instantaneous participation signal for the charge-based excitatory
            # rule: which lines delivered a spike in THIS receive_input call,
            # i.e. the immediate triggering event, not an accumulated window.
            # This is what replaced the trace-based participation check.
            self._last_input_spikes = input_spikes

    def apply_inhibition(self, inhibitory_spikes):
        """
        Deliver inhibitory-discharge events and run the inhibitory plasticity rule.

        This is the second, INDEPENDENT learning system (the excitatory rule in
        _update_weights is untouched and fires only on a postsynaptic spike). An
        inhibitory synapse is any afferent whose weight is negative; here it acts
        as an adaptive suppression gate. For every inhibitory synapse that carries
        a spike this step we, per the algorithm:

            1. V_pre  = V                       (how charged the neuron was)
            2. V      = V - w   (w = |weight|)  (linear inhibitory discharge)
               V_post = V
            3. p      = V_pre / theta           (normalized closeness to firing)
            4. dw     = eta * p * (1 - w^2 / w_max)   (saturating; w_max = inhibitory_weight_cap
                                                    or weight_cap if that is None)
            5. w      = w + dw                   (gate strengthens toward w_max)

        Note: the quadratic term means the *natural* equilibrium (where
        dw = 0) is w* = sqrt(w_max), not w_max itself, whenever w_max != 1 --
        growth reverses (dw < 0) for w > sqrt(w_max), so the gate settles
        below the nominal ceiling rather than saturating at it. See
        neuron_flexible.py's identical implementation for the same note.

        The gate strengthens most when it suppressed a neuron that was *close to
        firing* (p near 1) and saturates as w -> w_max (finite synaptic resource),
        so no global normalization is needed. Weights are stored with their sign
        (inhibitory = negative), so internally we work on the magnitude w = |weight|
        and write back the negated result, keeping |w| growing (sign-preserving).

        Learning is local (uses only this neuron's V, theta, and the synapse's own
        weight), event-driven (only on an inhibitory discharge), and gradient-free.

        Args:
            inhibitory_spikes (array-like): per-synapse spike flags (aligned to
                self.weights). Only entries on negative-weight synapses matter.

        Returns:
            list[dict]: one debug record per inhibitory event, with keys
            v_pre, v_post, theta, p, w_before, delta_w, w_after (and index).
        """
        events = []
        self.last_inhibitory_events = events
        # Refractory neurons are clamped to rest and do not integrate input, so
        # they also do not undergo inhibitory discharge/learning (parity with
        # receive_input's refractory gate).
        if self.refractory_timer > 0:
            return events

        spikes = np.asarray(inhibitory_spikes, dtype=float)
        theta = self.threshold
        w_max = self.inhibitory_weight_cap if self.inhibitory_weight_cap is not None else self.weight_cap
        # Loser-depression closeness signal: how close to firing this neuron was
        # BEFORE any inhibitory discharge this call (see _depress_losers).
        v_entry = float(self.potential)
        active = np.nonzero((self.weights < 0) & (spikes > 0.5))[0]
        for idx in active:
            w = -float(self.weights[idx])          # magnitude of the inhibitory gate
            v_pre = float(self.potential)
            # Linear discharge, FLOORED at rest: inhibition removes only the charge
            # actually present -- it cannot drive the membrane below resting
            # potential (no negative "charge"). v_pre was captured above, so the
            # weight update below is unchanged; p is already 0 when v_pre <= 0, so a
            # neuron with nothing to lose gets no weight update.
            self.potential = max(self.potential - w, self.resting_potential)
            v_post = float(self.potential)
            # Normalized closeness to firing at inhibition time, clamped to [0, 1]:
            # a hyperpolarized membrane never drives negative learning, and a
            # neuron already at/above threshold caps the drive at p = 1.
            p = min(max(v_pre / theta, 0.0), 1.0) if theta > 0 else 0.0
            if w_max > 0:
                dw = self.inhibitory_learning_rate * p * (1.0 - (w * w) / w_max)
            else:
                dw = 0.0
            w_new = min(max(w + dw, 0.0), w_max)   # saturate at the finite ceiling
            self.weights[idx] = -w_new             # keep the inhibitory sign
            events.append(dict(index=int(idx), v_pre=v_pre, v_post=v_post,
                               theta=theta, p=p, w_before=w,
                               delta_w=w_new - w, w_after=w_new))
        # Loser depression: a real inhibitory discharge (events non-empty) means
        # this neuron was a suppressed near-winner; depress the active positive
        # feedforward gates that made it one. Opt-in; see _depress_losers.
        if self.loser_depression and events:
            self._depress_losers(v_entry)
        return events

    def _depress_losers(self, v_pre_loss):
        """Depress the ACTIVE positive feedforward gates of a neuron just suppressed
        by an inhibitory (L2I->L2E) discharge -- the local loser-depression rule
        (Claude_Confidence_Consolidation_Plan.md):

            p_loss = clamp(V_pre / theta, 0, 1)
            dw_i   = eta_loss * p_loss * x_i * (1 - C_i) * m_i^gamma * (w_i - w_min)
            w_i   <- w_i - dw_i     (then the floor/budget/cap tail)

        Intentionally nonlinear in the protect-small / punish-large direction:
        small gates near the floor barely move (m_i^gamma -> 0), large still-
        unconfident active gates take the largest hit, confident mature gates are
        protected (1 - C_i), and far-away suppressed neurons learn little (small
        p_loss). Inactive gates and the negative gate are never touched."""
        theta = self.threshold
        p_loss = min(max(v_pre_loss / theta, 0.0), 1.0) if theta > 0 else 0.0
        if p_loss <= 0.0:
            return
        participating = self._last_input_spikes > 0.5
        active = np.nonzero((self.weights > 0) & participating)[0]
        if active.size == 0:
            return
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        w = self.weights[active]
        C = self.confidence[active]
        ratio = self._maturity(w)
        dw_minus = (self.eta_loss * p_loss * (1.0 - C)
                    * (ratio ** self.loss_gamma) * (w - w_min))
        self.weights[active] = w - dw_minus
        self.loser_depression_events += 1
        self._apply_budget_and_cap()

    def update(self):
        """
        Update neuron state for next time step.
        Handles refractory period and potential leak to resting state.
        """
        # Homeostatic firing-rate sensor: a slow EMA of the neuron's OWN spiking
        # this step (self.spiked is still set from fire() and is reset at the end
        # of this method). Purely local -- reads nothing but its own output.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)

        # Activity-dependent confidence decay (long-term memory; opt-in).
        if self.confidence_consolidation:
            self._decay_confidence()

        # Handle refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            # Clamp potential to resting during refractory period
            self.potential = self.resting_potential
        else:
            # Apply leak: potential decays toward resting potential. Fixed-point
            # leak control -- the leak amount is the integer numerator
            # self._leak_num over the integer LEAK_SCALE (see the module header),
            # so no float leak constant enters here. For example _leak_num=10
            # over LEAK_SCALE=1000 closes 1% of the distance to rest each ms.
            # (Potential itself is still float on this scope's deferred path.)
            self.potential += self._leak_num * (self.resting_potential - self.potential) / LEAK_SCALE
            # Decay the eligibility trace with the same leak fraction.
            self.trace *= (1.0 - self._leak_num / LEAK_SCALE)

        # Homeostatic synaptic scaling (slow, activity-driven, non-Hebbian).
        if self.homeostasis:
            self._homeostatic_scaling()

        # Reset spike flag for next time step
        self.spiked = False
    
    def check_threshold(self):
        """
        Check if neuron should fire based on threshold.

        Returns:
            bool: True if neuron fires, False otherwise
        """
        # Only check threshold if not in refractory period
        if self.refractory_timer <= 0 and self.potential >= self.threshold:
            return True
        return False
    
    def fire(self):
        """
        Handle spike event: capture charge, discharge (reset potential), start
        refractory period, and update weights from the captured charge.
        """
        # Record spike time
        self.last_spike_time = 0  # Current time step

        # Capture charge BEFORE discharging -- this is what the weight update
        # below is computed from. Mirrors apply_inhibition's V_pre capture.
        v_pre = float(self.potential)

        # Discharge: reset potential after firing (the "subtract charge" step).
        self.potential = self.resting_potential

        # Start refractory period
        self.refractory_timer = self.refractory_period

        # Mark that we spiked this time step
        self.spiked = True

        # Update weights from the charge captured before discharge.
        self._update_weights(v_pre)

        # Evidence consumed: clear the trace so the next cycle starts fresh.
        # ARCHIVED bookkeeping only -- trace is no longer read by _update_weights.
        self.trace = np.zeros_like(self.trace)

    def _update_weights(self, v_pre):
        """
        Charge-based excitatory weight update -- the SAME algorithm as
        apply_inhibition (see that method), applied to positive synapses and
        triggered by this neuron's own fire instead of an incoming inhibitory
        spike. Both rules now share one structure:

            capture charge (v_pre) -> discharge -> p from v_pre/theta ->
            dw = eta * p * (1 - w^2 / w_max) -> w += dw, clipped to weight_cap

        For excitation, p is INVERTED relative to inhibition -- a neuron that
        fires with only just enough charge (v_pre close to theta) gets the
        LARGEST update; one that fires with much more charge than it needed
        gets the SMALLEST:

            p = clamp(theta / v_pre, 0, 1)     (v_pre >= theta always here,
                                                 since check_threshold() already
                                                 required potential >= threshold)

        w_max here is excitatory_saturation_cap (defaults to weight_cap when
        None) -- kept separate from the hard clip weight_cap exactly like
        apply_inhibition's inhibitory_weight_cap. This matters because the
        natural equilibrium of this formula is w* = sqrt(w_max): if w_max ==
        weight_cap and weight_cap > 1, a synapse can never approach the hard
        cap (equilibrium undershoots it, e.g. weight_cap=8 -> equilibrium
        ~2.83). Setting excitatory_saturation_cap = weight_cap**2 makes
        sqrt(w_max) == weight_cap exactly, so growth CAN reach the hard clip
        for a habitually-participating synapse, while still saturating
        smoothly rather than hitting a hard wall.

        Only synapses whose presynaptic line delivered a spike in the most
        recent receive_input() call are updated (self._last_input_spikes) --
        the instantaneous analogue of apply_inhibition's own spike-gated
        synapses. This REPLACES the old accumulated eligibility trace and the
        confidence-weighted credit-splitting rule (both ARCHIVED -- see git
        history prior to this change, and Weight_Update_Unification.md).

        Only positive (excitatory) weights move here; negative (inhibitory)
        synapses never move in this method -- they only ever move in
        apply_inhibition, on an inhibitory discharge into this neuron. After
        the per-synapse update, the existing budget/cap tail
        (_apply_budget_and_cap) runs exactly as before: neurons with a
        weight_budget or homeostasis get renormalized to that resource;
        neurons with neither (weight_budget=None, homeostasis=False) are
        simply clipped to weight_cap, same as apply_inhibition's own
        no-renormalization policy. This rule only changes how dw is computed,
        not what happens to the result afterward.
        """
        theta = self.threshold
        p = min(max(theta / v_pre, 0.0), 1.0) if v_pre > 0 else 0.0
        participating = self._last_input_spikes > 0.5
        active = np.nonzero((self.weights > 0) & participating)[0]
        w_max = self.excitatory_saturation_cap if self.excitatory_saturation_cap is not None else self.weight_cap
        if w_max > 0 and active.size > 0:
            w = self.weights[active]
            if self.confidence_consolidation:
                # Confidence-gated potentiation: mature (confident) gates learn
                # less, with a floor eta_min so no gate freezes:
                #   eta_i = eta_0 * [eta_min + (1 - eta_min)(1 - C_i)]
                C = self.confidence[active]
                eta = self.learning_rate * (self.eta_min + (1.0 - self.eta_min) * (1.0 - C))
                self.weights[active] = w + eta * p * (1.0 - (w * w) / w_max)
                # Mature confidence toward the (pre-update) gate's local maturity;
                # only active gates (x_i = 1) move: C_i += beta_C (m_i - C_i).
                self.confidence[active] = C + self.conf_beta * (self._maturity(w) - C)
            else:
                dw = self.learning_rate * p * (1.0 - (w * w) / w_max)
                self.weights[active] = w + dw
        # Signed-spike depression ("4a"): OFF pixels (positive gates whose input
        # did NOT spike this fire) are pushed down. Confidence-gated when
        # consolidation is on (mature gates resist via (1 - C_i)); shaped by
        # (w_i - w_min) so a gate decelerates smoothly into the floor rather than
        # crossing it -- the _apply_budget_and_cap() tail then clamps to the
        # min_positive_weight floor, so a depressed gate never goes deaf or
        # negative. Uses the same event closeness p as potentiation (an efficient
        # fire depresses OFF gates as hard as it potentiates ON ones).
        if self.signed_depression and self.eta_off > 0.0:
            inactive = np.nonzero((self.weights > 0) & ~participating)[0]
            if inactive.size > 0:
                w_off = self.weights[inactive]
                w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
                gate = (1.0 - self.confidence[inactive]) if self.confidence_consolidation else 1.0
                self.weights[inactive] = w_off - self.eta_off * p * gate * (w_off - w_min)
                self.signed_depression_events += 1
        self._apply_budget_and_cap()

    def _maturity(self, w):
        """Local instantaneous maturity m in [0,1] of positive gate weights w:
        m = clamp((w - w_min) / (w_conf_cap - w_min), 0, 1), where w_min is
        min_positive_weight (or 0) and w_conf_cap is conf_cap -- the effective
        reachable mature synapse value (NOT the hard clip, which budget
        normalization can make unreachable). Falls back to weight_cap when
        conf_cap is unset. See Claude_Confidence_Consolidation_Plan.md."""
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        cap = self.conf_cap if self.conf_cap is not None else self.weight_cap
        if cap <= w_min:
            return np.zeros_like(w)
        return np.clip((w - w_min) / (cap - w_min), 0.0, 1.0)

    def _apply_budget_and_cap(self):
        """Shared tail of both excitatory rules: renormalize positive (excitatory)
        weights to a fixed total so strengthening one input weakens the others (no
        runaway growth), then clip to [-weight_cap, weight_cap]. When homeostasis
        is on, the target total is the homeostatic resource R (self.homeo_budget)
        rather than the fixed weight_budget -- so the resource is regulated by the
        neuron's own activity instead of being a hard constant.

        When min_positive_weight is set, a positive weight is floored there
        after renormalization -- a budgeted neuron that trains heavily on one
        input otherwise erodes EVERY other positive synapse toward 0 (the
        renormalization ratio is applied to participating and
        non-participating synapses alike, every single update, so an unused
        synapse shrinks a little on every event even though it never itself
        gets credited). Left unbounded, a neuron can go permanently deaf to
        any pattern that doesn't touch its currently-favored synapses. The
        floor means the post-floor sum can exceed target by a small amount
        (at most n_inputs * min_positive_weight) -- an intentional, bounded
        relaxation of the exact budget in exchange for guaranteed baseline
        responsiveness to every input. None (default) preserves old behavior."""
        target = self._resource_target()
        if target is not None:
            pos = self.weights > 0
            total = float(self.weights[pos].sum())
            if total > 1e-9:
                self.weights[pos] *= target / total
        if self.min_positive_weight is not None:
            pos = self.weights > 0
            self.weights[pos] = np.maximum(self.weights[pos], self.min_positive_weight)
        self.weights = np.clip(self.weights, -self.weight_cap, self.weight_cap)

    def _resource_target(self):
        """The total positive-weight budget to renormalize to. Under homeostasis
        this is the homeostatic resource R (lazy-initialized from the current
        positive-weight sum on first use); otherwise the fixed weight_budget."""
        if self.homeostasis:
            if self.homeo_budget is None:
                pos = self.weights > 0
                self.homeo_budget = float(self.weights[pos].sum()) if pos.any() else None
            return self.homeo_budget
        return self.weight_budget

    def _homeostatic_scaling(self):
        """
        Homeostatic synaptic scaling -- a third local plasticity system, distinct
        from the on-fire Hebbian rule and the inhibitory-discharge rule.

        Biology: this is Turrigiano-style synaptic scaling. It is NOT triggered by
        firing (that is the Hebbian rule) and NOT by inhibition. It is triggered by
        the neuron's OWN long-run firing rate (a slow "calcium" average, self.ca)
        leaving a target band around a set-point. Chronically silent -> the neuron
        multiplicatively grows its excitatory resource; chronically over-active ->
        it shrinks it. It carries NO pattern information (the scaling is uniform /
        multiplicative, so relative weights are preserved), so it never fakes the
        associative learning -- it just returns a starved neuron toward the firing
        regime, after which the on-fire Hebbian rule carves the actual receptive
        field. "Learning happens on fire" is therefore preserved for the meaningful
        (pattern) learning; this only sets the gain.

        The scaling factor is a FIXED multiplicative constant (1 + homeo_up when
        starved, 1 - homeo_down when saturated), gated by a deadband. It is not
        proportional to any error, not a gradient, and depends on nothing but this
        neuron's own ca vs its own ca_target -- no global signal, no supervision.
        """
        pos = self.weights > 0
        if not pos.any():
            return
        if self.homeo_budget is None:
            self.homeo_budget = float(self.weights[pos].sum())

        lo = self.ca_target * (1.0 - self.ca_band)
        hi = self.ca_target * (1.0 + self.ca_band)
        if self.ca < lo:
            self.homeo_budget *= (1.0 + self.homeo_up)      # chronically silent -> grow
        elif self.ca > hi:
            self.homeo_budget *= (1.0 - self.homeo_down)    # chronically over-active -> shrink
        # else: within the homeostatic band -> no scaling
        if self.homeo_budget_min is not None:
            self.homeo_budget = max(self.homeo_budget, self.homeo_budget_min)
        if self.homeo_budget_max is not None:
            self.homeo_budget = min(self.homeo_budget, self.homeo_budget_max)

        total = float(self.weights[pos].sum())
        if total > 1e-9:
            self.weights[pos] *= self.homeo_budget / total
        self.weights = np.clip(self.weights, -self.weight_cap, self.weight_cap)

    def _decay_confidence(self):
        """Activity-dependent confidence decay -- long-term memory for gate
        maturity (Claude_Confidence_Consolidation_Plan.md). Reuses the neuron's
        own calcium/activity trace (self.ca) as an activity-memory signal, NOT for
        weight scaling. A long grace counter means confidence survives normal
        interleaved training (a specialist may only fire once per full pattern
        sweep) and decays only after GENUINELY long disuse, so stale neurons
        eventually become reusable:

            ca_j >= ca_dead                         -> rho = rho_active (tiny)
            ca_j <  ca_dead and inactive >= T_grace -> rho = rho_dead
            otherwise (inside the grace window)     -> rho = 0
            C_i <- C_i (1 - rho)

        Purely local to this neuron and its synapses; no global signal."""
        if self.ca >= self.conf_ca_dead:
            self.inactive_steps = 0
            rho = self.conf_rho_active
        else:
            self.inactive_steps += 1
            rho = self.conf_rho_dead if self.inactive_steps >= self.conf_grace else 0.0
        if rho > 0.0:
            self.confidence *= (1.0 - rho)

    def plasticity_stats(self):
        """
        Summary statistics over this neuron's excitatory (positive-weight)
        afferents, for diagnosing whether selective receptive fields are forming.

        Returns a dict with, over the positive weights and their confidence:
          weight_entropy / confidence_entropy       -- Shannon entropy (nats);
              lower = more selective (mass on fewer synapses).
          weight_concentration / confidence_concentration -- HHI in [1/n, 1];
              higher = more concentrated.
          budget_used -- current sum of positive weights (vs weight_budget).
        """
        pos_mask = self.weights > 0
        pos_w = self.weights[pos_mask]
        pos_c = self.confidence[pos_mask]
        return dict(
            weight_entropy=_distribution_entropy(pos_w),
            confidence_entropy=_distribution_entropy(pos_c),
            weight_concentration=_concentration(pos_w),
            confidence_concentration=_concentration(pos_c),
            budget_used=float(pos_w.sum()),
        )
        
    def get_state(self):
        """
        Get current neuron state for monitoring/debugging.
        
        Returns:
            dict: Dictionary containing key state variables
        """
        return {
            'potential': self.potential,
            'refractory_timer': self.refractory_timer,
            'spiked': self.spiked,
            'weights': self.weights.copy(),
            'confidence': self.confidence.copy(),
            'ca': self.ca,
            'homeo_budget': self.homeo_budget,
            'last_spike_time': self.last_spike_time
        }