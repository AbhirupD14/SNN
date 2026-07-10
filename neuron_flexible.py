"""
Single Neuron implementation for the SNN.

The class supports both construction styles used in the repo:

* fixed fan-in via ``Neuron(n_inputs=...)`` for simple layers/tests;
* dynamic fan-in via ``add_input_connection()`` + ``finalize_connections()`` for
  column construction where wiring is assembled in stages.
"""

import numpy as np

# A synapse counts as having "participated" in the current integration window if
# its archived trace accumulator carries any residual charge above this floor.
PARTICIPATION_EPS = 1e-6

# ---------------------------------------------------------------------------
# Fixed-point convention for gate weights, leak controls, and thresholds.
#
# The default numeric representation for these three categories is defined in
# integer / integer-rational form rather than as bare float literals. UNIT is the
# fixed-point scale for weight/threshold magnitudes (1 model unit == 1/UNIT);
# LEAK_SCALE is the denominator for leak-rate numerators.
UNIT = 1000
LEAK_SCALE = 1000


def to_units(x):
    """Float magnitude -> integer fixed-point units (nearest)."""
    return int(round(x * UNIT))


def to_float(u):
    """Integer fixed-point units -> float magnitude."""
    return u / UNIT


def leak_num(rate):
    """Leak fraction (float) -> integer leak numerator over LEAK_SCALE."""
    return int(round(rate * LEAK_SCALE))


def div_round(n, d):
    """Deterministic integer rounding division (round-half-away-from-zero)."""
    return (n + d // 2) // d if n >= 0 else -((-n + d // 2) // d)


def _distribution_entropy(x):
    """Shannon entropy (nats) of non-negative values treated as a distribution."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    s = x.sum()
    if x.size == 0 or s <= 0:
        return 0.0
    p = x / s
    return float(-(p * np.log(p)).sum())


def _concentration(x):
    """Herfindahl-Hirschman concentration of non-negative values."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    s = x.sum()
    if x.size == 0 or s <= 0:
        return 0.0
    p = x / s
    return float((p * p).sum())


class Neuron:
    """
    A flexible spiking neuron model that allows dynamic specification of 
    input connections during layer construction, then locks connections 
    for simulation.
    
    Both excitatory and inhibitory neurons use the same dynamics;
    the difference is in the sign of their synaptic weights.
    """
    
    def __init__(self, n_inputs=None, threshold=1000 / UNIT, refractory_period=2,
                 weight_init_range=(-500 / UNIT, 500 / UNIT),
                 learning_rate=0.1, weight_cap=1000 / UNIT, leak_rate=10 / LEAK_SCALE,
                 inhibitory_learning_rate=0.05, inhibitory_weight_cap=None,
                 excitatory_saturation_cap=None,
                 confidence_init=0.10,
                 homeostasis=False, ca_rate=0.01, ca_target=0.02, ca_band=0.5,
                 homeo_up=0.01, homeo_down=0.01,
                 homeo_budget_min=None, homeo_budget_max=None):
        """
        Initialize a flexible neuron.

        Args:
            n_inputs (int|None): optional fixed fan-in. If provided, connections
                are initialized and finalized immediately. If omitted, call
                add_input_connection() and finalize_connections().
            threshold (float): Firing threshold (constant)
            refractory_period (int): Refractory period in time steps (1ms each)
            weight_init_range (tuple): initialization range for fixed fan-in use
            learning_rate (float): Amount weights increase when neuron fires (excitatory plasticity)
            weight_cap (float): Maximum absolute value for weights (weights clipped to [-weight_cap, weight_cap]).
                Also serves as w_max, the saturating magnitude ceiling for inhibitory plasticity.
            leak_rate (float): Fraction of potential that leaks away per time step (0 = no leak, 1 = full leak)
            inhibitory_learning_rate (float): eta for the inhibitory-discharge plasticity rule
                (see apply_inhibition). 0 disables inhibitory learning.
            confidence_init (float): initial value of the active per-synapse
                confidence array. The archived confidence-beta / confidence-gamma
                params were removed.
            homeostasis, ca_rate, ca_target, ca_band, homeo_up, homeo_down,
            homeo_budget_min, homeo_budget_max: homeostatic synaptic scaling.
        """
        # Neuron properties
        self.threshold = threshold          # Firing threshold (constant)
        self.resting_potential = 0.0        # Resting potential at 0
        self.refractory_period = refractory_period  # Refractory period
        self.refractory_timer = 0           # Counts down refractory period
        self.potential = self.resting_potential     # Current membrane potential
        
        # Weight properties - stored as list during construction, numpy array after
        self._weights_list = []        # List of synaptic weights during construction
        self._weights_array = None     # Numpy array after finalization
        self._trace = None             # Per-synapse eligibility trace (set at finalize)
        # Confidence state. Allocated at finalize once the
        # fan-in is known; seeded at confidence_init and updated by the active
        # confidence-consolidation system.
        self._confidence = None
        self.confidence_init = float(confidence_init)
        self.last_excitatory_event = {}
        # Confidence-gated consolidation (opt-in; see
        # Claude_Confidence_Consolidation_Plan.md). All flags off by default so
        # bare neurons / L1 / inhibitory neurons are unaffected.
        self.confidence_consolidation = False  # revive confidence, gate eta, decay
        self.loser_depression = False          # depress active gates on inhibitory loss
        self.conf_cap = None                   # effective mature gate value (None -> weight_cap)
        self.conf_beta = 0.05                  # confidence EMA rate toward maturity
        self.eta_min = 0.05                    # plasticity floor fraction for mature gates
        self.eta_loss = 0.01                   # loser-depression rate (dimensionless)
        self.loss_gamma = 2                    # protect-small / punish-large exponent
        self.conf_rho_active = 1e-5            # confidence decay while recently active
        self.conf_rho_dead = 1e-3              # confidence decay once dead past the grace window
        self.conf_ca_dead = 0.002              # ca below this counts the neuron as inactive
        self.conf_grace = 5000                 # inactive steps before dead-decay engages
        self.inactive_steps = 0                # long grace counter (see update())
        self.loser_depression_events = 0       # diagnostic counter
        # Flow-proportional assembly credit (opt-in; for the E->I "assembly
        # evidence" neurons L2I / L1I -- see _update_weights). On this neuron's
        # OWN fire, credit every incoming positive synapse in proportion to the
        # flow it delivered over the retention window (the per-synapse leaky
        # _trace), normalized so the DOMINANT driver gets the full learning rate;
        # synapses that delivered no flow are depressed toward the floor. This
        # fixes the last-volley-only credit that let a habitual winner's E->I
        # synapse stall below threshold (the L2I firing deadlock). Default OFF.
        self.assembly_flow_credit = False      # flow-proportional E->I credit on fire
        self.assembly_decay_frac = 0.5         # down-pressure on non-contributing gates
        # Signed-spike depression ("4a"; opt-in; see _update_weights). OFF
        # pixels deliver no charge but their positive gates
        # are depressed on this neuron's own fire. Default OFF -> baseline intact.
        self.signed_depression = False         # enable OFF-gate depression on fire
        self.eta_off = 0.0                     # depression rate for inactive gates
        self.signed_depression_events = 0      # diagnostic counter
        # Minimal SIGNED-SPIKE feedforward learning (opt-in; see
        # Claude_Minimal_Signed_Spike_Learning_Prompt.md and _update_weights).
        # On fire, every POSITIVE feedforward synapse gets one local signed update
        # -- active inputs (+1) potentiate, inactive inputs (-1) depress -- with no
        # global weight budget. Replaces the whole potentiation / OFF-depression /
        # confidence / budget stack with a single equation. Default OFF.
        self.signed_spike_learning = False
        # Reset-by-subtraction on fire (opt-in; see fire()). Default OFF reproduces
        # the full reset-to-rest. When ON, a fired neuron keeps its residual
        # overshoot above threshold (standard LIF), floored at rest -- this attacks
        # the discharge asymmetry vs partially-inhibited losers (a fully-reset
        # winner would otherwise be overtaken by losers that ratcheted charge up).
        self.subtractive_reset = False
        # Homeostatic synaptic scaling (third local system).
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
        self.weight_budget = None      # optional homeostatic budget for positive weights
        self.min_positive_weight = None  # optional floor for positive weights (see _apply_budget_and_cap)
        self.learning_rate = learning_rate  # Weight increase amount when neuron fires (excitatory)
        self.weight_cap = weight_cap        # Maximum absolute value for weights (also w_max for inhibition)
        self.leak_rate = leak_rate          # stored as integer numerator self._leak_num (see property)
        self.inhibitory_learning_rate = inhibitory_learning_rate  # eta for inhibitory plasticity
        # Saturation ceiling (w_max) for inhibitory gates, kept separate from the
        # feedforward weight_cap (defaults to weight_cap when None). See apply_inhibition.
        self.inhibitory_weight_cap = inhibitory_weight_cap
        # Saturation ceiling (w_max) for the EXCITATORY quadratic term, kept
        # separate from the hard clip weight_cap -- see this class for the
        # full rationale (sqrt(w_max) equilibrium; set to weight_cap**2 so a
        # habitually-participating synapse can reach the hard cap exactly).
        self.excitatory_saturation_cap = excitatory_saturation_cap
        # Membrane saturation ceiling (absolute, in the same units as potential;
        # None = unbounded, the baseline). A finite driving force / reversal
        # potential: charge accumulation cannot push the membrane above this, so
        # the potential can't ratchet far past threshold between cycles. That is
        # what lets the (small, capped) inhibitory gate actually regulate firing
        # frequency -- against a membrane pinned near threshold a ~0.15*theta
        # discharge is meaningful; against a 3*theta pile-up it is not. Local and
        # per-neuron; see receive_input. Off by default so the baseline is intact.
        self.v_sat = None
        # Sparse excitatory FLOW-RATE accumulation (opt-in; see receive_input,
        # advance_trace, and Current_Implementation_Methodology_Equations.md).
        # Default OFF -> instantaneous V += dot(weights, spikes) baseline. When on,
        # an input spike opens a decaying excitatory CURRENT trace (exc_trace) that
        # is integrated into the membrane over time using closed-form skipped-time
        # math, so weight behaves as a current/gate amplitude rather than a charge
        # packet deposited instantly.
        self.excitatory_flow_rate = False   # enable current-trace accumulation
        self.exc_trace_decay = 0.8          # per-timestep current decay d in [0, 1)
        self.exc_trace_normalized = True    # inject drive*(1-d) so total delivered ~= drive
        self.exc_trace = 0.0                # I: current trace amplitude
        self.exc_trace_last_t = 0           # last outer timestep the trace was advanced to
        # Inhibitory FLOW (opt-in; symmetric to the excitatory flow). When on, a real
        # inhibitory discharge injects the gate magnitude into a decaying inhibitory
        # CURRENT (inh_trace) that drains charge out of the membrane over several
        # steps (in update(), floored at rest), instead of subtracting it all at once
        # -- sustained suppression to counteract the continuous excitatory inflow.
        # Normalized injection w*(1-d) makes the TOTAL drained ~= w (same as the
        # one-shot hit, just spread over time); un-normalized injects w (total w/(1-d),
        # a stronger sustained bite). Does not change the stored gate weight.
        self.inhibitory_flow_rate = False
        self.inh_trace_decay = 0.8
        self.inh_trace_normalized = True
        self.inh_trace = 0.0                # pending inhibitory current (drains in update)
        # Inhibitory-gate plasticity rule (see apply_inhibition). inhibitory_delta_rule
        # False = legacy SATURATING rule (dw = eta*p*(1 - w^2/w_max); every gate
        # converges to the same ceiling sqrt(w_max), so gates end up uniform). True =
        # a differentiating local rule, selected by inhibitory_rule_mode. All rules
        # normalize the gate against a sub-threshold ceiling G = sqrt(w_max) (the
        # scale the saturating rule converges to) and share the same linear
        # floored-at-rest delivery. Neuron default stays legacy so bare-neuron tests
        # are unchanged; SimulationEngine defaults delta_rule True, mode "turnover".
        self.inhibitory_delta_rule = False
        # "turnover" (default): event-local strengthen/turnover on the NORMALIZED gate
        #   u = w/G:  du = eta_up*p_t*(1 - u) - eta_down*u,  p_t = clamp(v_pre/theta,
        #   0, p_max). High-charge targets strengthen more (eta_up term); every gate
        #   turns over proportional to size (eta_down term). No target voltage, no
        #   averages -- purely this event's v_pre, theta, w and spike.
        # "margin": relax the gate toward s = clamp(v_pre - margin_frac*theta, 0, G)
        #   (a hand-set post-inhibition target level); kept as a diagnostic.
        self.inhibitory_rule_mode = "turnover"
        self.inhibitory_eta_up = 0.02       # turnover: charge-driven strengthening
        self.inhibitory_eta_down = 0.005    # turnover: size-proportional decay
        self.inhibitory_p_max = 1.0         # turnover: cap on p_t = v_pre/theta
        self.inhibitory_margin_frac = 0.5   # margin mode: target_post = frac * theta
        self.inhibitory_delta_eta = 0.05    # margin mode: EMA rate toward s
        # Distance attenuation of DELIVERED excitatory drive (opt-in; see
        # receive_input). The weight is the learned gate strength; distance is a
        # fixed per-synapse DELIVERY attenuation that multiplies the injected drive
        # amplitude -- it does NOT change the stored weight, and does NOT enter the
        # trace decay/integration math. Per afferent:
        #   factor_i = (distance_ref / max(d_i, distance_min)) ** distance_power
        # so the effective drive is sum_i spike_i * w_i * factor_i. _distance holds
        # the per-afferent d_i (allocated to ones at finalize = no attenuation until
        # real functional distances are provided). Default OFF preserves behavior.
        self.distance_weighting = False
        self.distance_power = 2.0
        self.distance_ref = 1.0
        self.distance_min = 1.0
        self._distance = None               # per-afferent d_i (set at finalize)
        self._connections_finalized = False  # Flag to prevent changes after finalization
        self.last_inhibitory_events = []    # debug records from the most recent apply_inhibition()
        
        # Spike tracking
        self.last_spike_time = -np.inf      # Time of last spike
        self.spiked = False                 # Did neuron spike in current step?

        if n_inputs is not None:
            weights = np.random.uniform(weight_init_range[0], weight_init_range[1], int(n_inputs))
            self._set_weights(weights, finalized=True)

    @property
    def leak_rate(self):
        """Leak fraction per step, backed by the integer numerator
        self._leak_num over LEAK_SCALE (fixed-point leak control; see
        the fixed-point module constants above)."""
        return self._leak_num / LEAK_SCALE

    @leak_rate.setter
    def leak_rate(self, value):
        self._leak_num = leak_num(value)

    def add_input_connection(self, weight):
        """
        Add an input connection with the specified weight.
        Can only be called before connections are finalized.
        
        Args:
            weight (float): Weight of the connection (positive for excitatory, 
                          negative for inhibitory)
        """
        if self._connections_finalized:
            raise RuntimeError("Cannot add connections after finalization. "
                             "Call finalize_connections() to lock connections.")
        
        self._weights_list.append(float(weight))
        
    def finalize_connections(self):
        """
        Lock the connections after all have been added.
        Converts weights list to numpy array and prepares for simulation.
        Must be called before running simulation.
        """
        if not self._connections_finalized:
            weights = self._weights_list if self._weights_list else []
            self._set_weights(weights, finalized=True)

    def _set_weights(self, weights, finalized=None):
        """Replace the afferent weight vector and resize aligned local state."""
        arr = np.asarray(weights, dtype=float)
        self._weights_array = np.clip(arr, -self.weight_cap, self.weight_cap)
        n = len(self._weights_array)
        self._trace = np.zeros(n)
        if self._confidence is None or len(self._confidence) != n:
            self._confidence = np.full(n, self.confidence_init)
        # Per-afferent delivery distance; default 1.0 (no attenuation). Preserved
        # across weight re-assignments of the same fan-in so setting distances then
        # weights (or vice versa) does not wipe them.
        if self._distance is None or len(self._distance) != n:
            self._distance = np.ones(n)
        self._last_input_spikes = np.zeros(n)
        if finalized is not None:
            self._connections_finalized = finalized
            
    def _ensure_finalized(self):
        """Internal method to check if connections are ready for simulation."""
        if not self._connections_finalized:
            raise RuntimeError("Connections not finalized. "
                             "Call finalize_connections() before simulation.")
        
    def advance_trace(self, t):
        """Lazily integrate the excitatory current trace forward to outer timestep
        `t` (closed-form skipped-time), adding the integrated current to the
        membrane and decaying the trace. This is the sparse equivalent of running
        `V += I; I *= d` once per timestep from exc_trace_last_t up to t, but in
        O(1). No-op outside flow-rate mode or when already at/past t. See the
        methodology doc's flow-rate section for the derivation."""
        if not self.excitatory_flow_rate:
            return
        dt = t - self.exc_trace_last_t
        if dt <= 0:
            return
        d = self.exc_trace_decay
        if self.exc_trace != 0.0:
            # Geometric sum of dt discrete `V += I; I *= d` steps. As d -> 1 the
            # closed form (1 - d^dt)/(1 - d) -> dt (its limit), which also avoids
            # the 1/(1-d) blow-up; d == 0 gives geom == 1 (contributes once).
            geom = float(dt) if abs(1.0 - d) < 1e-9 else (1.0 - d ** dt) / (1.0 - d)
            self.potential += self.exc_trace * geom
            self.exc_trace *= d ** dt
            if self.v_sat is not None and self.potential > self.v_sat:
                self.potential = self.v_sat
        self.exc_trace_last_t = t

    def receive_input(self, input_spikes, charge_scale=1.0, t=None):
        """
        Accumulate charge based on weighted inputs.

        Args:
            input_spikes (array-like): Binary array indicating which inputs spiked (1) or not (0)
            charge_scale (float): scales ONLY the delivered charge (potential
                increment and eligibility trace), not the participation signal --
                _last_input_spikes still records the binary input vector so the
                signed-spike learning rule sees the true participating synapses.
                Used by the chunked-charge L2 competition (l2_charge_chunks) to
                deliver weight_ji/K per inner sub-step; the default 1.0 leaves
                every existing caller unchanged.
            t (int|None): current outer timestep. Required for flow-rate mode
                (excitatory_flow_rate); ignored in the instantaneous baseline.
        """
        self._ensure_finalized()

        # Only accumulate charge if not in refractory period (flow mode too: a
        # refractory neuron neither advances its trace nor injects, so no hidden
        # current builds up while it is clamped -- consistent with instantaneous).
        if self.refractory_timer > 0 or len(self._weights_array) == 0:
            return
        input_spikes = np.asarray(input_spikes, dtype=float)

        # Distance attenuation of DELIVERED drive (opt-in): scale each afferent
        # weight by its fixed per-synapse delivery factor before it drives the
        # membrane. factor_i = (distance_ref / max(d_i, distance_min))^distance_power.
        # This changes ONLY the delivered amplitude (both flow-rate and
        # instantaneous paths), never the stored weight, and it is NOT part of the
        # trace decay/integration math. OFF leaves w_eff pointing at the stored
        # weights (no copy) so the baseline is byte-identical.
        if self.distance_weighting and self._distance is not None:
            factor = (self.distance_ref / np.maximum(self._distance, self.distance_min)) ** self.distance_power
            w_eff = self._weights_array * factor
        else:
            w_eff = self._weights_array

        if self.excitatory_flow_rate and t is not None:
            # Flow-rate: the spike opens a decaying excitatory current trace that is
            # integrated into V over time, instead of depositing all charge at once.
            d = self.exc_trace_decay
            # 1. Advance residual current through the gap timesteps up to t-1.
            self.advance_trace(t - 1)
            self._dbg_v_after_advance = self.potential   # phase diagnostic (see step())
            # 2. Inject new EXCITATORY (positive-weight) drive as current, using the
            #    distance-attenuated effective weights (delivery, not stored weight).
            drive = float(np.dot(np.maximum(w_eff, 0.0), input_spikes)) * charge_scale
            self.exc_trace += drive * (1.0 - d) if self.exc_trace_normalized else drive
            # 3. Same-timestep contribution: one integration step at t (inject then
            #    integrate, so a fresh spike still moves V this timestep).
            self.potential += self.exc_trace
            self.exc_trace *= d
            self.exc_trace_last_t = t
            if self.v_sat is not None and self.potential > self.v_sat:
                self.potential = self.v_sat
            # Only a REAL input volley refreshes the participation mask; residual-
            # flow steps (no spike) leave it, so a neuron that crosses threshold on
            # a later no-input timestep still learns the volley that drove it.
            if input_spikes.any():
                self._trace += input_spikes * charge_scale
                self._last_input_spikes = input_spikes
            return

        # Instantaneous baseline: V += dot(weights, spikes) (distance-attenuated
        # effective weights when distance weighting is on; not entangled with
        # chunking -- charge_scale = 1/K still just scales the resulting drive).
        input_current = np.dot(w_eff, input_spikes) * charge_scale
        self.potential += input_current
        # Saturating membrane: bound accumulated charge at a finite ceiling so it
        # can't ratchet far past threshold (keeps the membrane in a range where the
        # inhibitory gate can regulate it).
        if self.v_sat is not None and self.potential > self.v_sat:
            self.potential = self.v_sat
        # ARCHIVED: no longer read by _update_weights. Kept only so anything still
        # inspecting ._trace keeps working.
        self._trace += input_spikes * charge_scale
        # Instantaneous participation signal for the charge-based excitatory rule
        # -- see _update_weights. Stays BINARY (unscaled) so chunked delivery does
        # not corrupt the participation mask.
        self._last_input_spikes = input_spikes

    def apply_inhibition(self, inhibitory_spikes):
        """
        Deliver inhibitory-discharge events and run the inhibitory plasticity rule.

        Independent second learning system (the excitatory rule in _update_weights
        is untouched and fires only on a postsynaptic spike). An inhibitory synapse
        is any afferent with a negative weight, acting as an adaptive suppression
        gate. For every inhibitory synapse carrying a spike this step, per the
        algorithm (w = |weight|, w_max = inhibitory_weight_cap or weight_cap,
        theta = threshold):

            V_pre  = V ; V = V - w ; V_post = V
            p      = V_pre / theta
            dw     = eta * p * (1 - w^2 / w_max)   (saturating, local, gradient-free)
            w      = w + dw

        The gate strengthens most when it suppressed a near-threshold neuron and
        saturates as w -> w_max, so no global normalization is needed. Sign is
        preserved (weights stay negative; |w| grows). Note the quadratic term
        means the natural equilibrium (dw = 0) is w* = sqrt(w_max), not w_max
        itself, whenever w_max != 1 -- growth reverses for w > sqrt(w_max).

        Returns: list[dict] debug records (v_pre, v_post, theta, p, w_before,
        delta_w, w_after, index) -- also stored on self.last_inhibitory_events.
        """
        self._ensure_finalized()
        events = []
        self.last_inhibitory_events = events
        if self.refractory_timer > 0 or len(self._weights_array) == 0:
            return events

        spikes = np.asarray(inhibitory_spikes, dtype=float)
        theta = self.threshold
        w_max = self.inhibitory_weight_cap if self.inhibitory_weight_cap is not None else self.weight_cap
        # Loser-depression closeness signal: how close to firing this neuron was
        # BEFORE any inhibitory discharge this call (see _depress_losers).
        v_entry = float(self.potential)
        active = np.nonzero((self._weights_array < 0) & (spikes > 0.5))[0]
        for idx in active:
            w = -float(self._weights_array[idx])   # magnitude of the inhibitory gate
            v_pre = float(self.potential)
            if self.inhibitory_flow_rate:
                # Inhibitory FLOW: inject the gate into a decaying inhibitory current
                # that drains the membrane over subsequent steps (see update()),
                # rather than subtracting it all now. No instantaneous change here;
                # the learning rule below still uses v_pre.
                self.inh_trace += w * (1.0 - self.inh_trace_decay) if self.inh_trace_normalized else w
                v_post = v_pre
            else:
                # Linear one-shot discharge FLOORED at rest:
                # inhibition cannot push the membrane below resting potential.
                self.potential = max(self.potential - w, self.resting_potential)
                v_post = float(self.potential)
            # Normalized closeness to firing, clamped to [0, 1] (saturating rule).
            p = min(max(v_pre / theta, 0.0), 1.0) if theta > 0 else 0.0
            if self.inhibitory_delta_rule:
                # Differentiating rules operate on the NORMALIZED gate u = w/G, where
                # the ceiling G = sqrt(w_max) is the same sub-threshold scale the
                # saturating rule converges to (so switching rules doesn't jump the
                # gate scale, and l2_gate_eq_frac still sets it). All variables are
                # LOCAL to this event: this synapse's w, this target's v_pre/theta,
                # the arriving spike. No averages, no global rank/winner identity.
                G = w_max ** 0.5 if w_max > 0 else 0.0
                if self.inhibitory_rule_mode == "margin":
                    # Diagnostic: relax toward the magnitude that brings the target to
                    # a fixed post-inhibition level target_post = margin_frac*theta.
                    target_post = self.inhibitory_margin_frac * theta
                    s = min(max(v_pre - target_post, 0.0), G)
                    dw = self.inhibitory_delta_eta * (s - w)
                else:
                    # Default TURNOVER: event-local strengthen (charge-driven, larger
                    # for high v_pre and small u) minus size-proportional turnover.
                    # Frequently high-charge rivals accumulate a stronger gate; weak/
                    # dead targets drift down -- with NO desired post-inhibition
                    # voltage and NO stored average.
                    u = w / G if G > 0 else 0.0
                    p_t = min(max(v_pre / theta, 0.0), self.inhibitory_p_max) if theta > 0 else 0.0
                    du = self.inhibitory_eta_up * p_t * (1.0 - u) - self.inhibitory_eta_down * u
                    dw = du * G
                w_new = min(max(w + dw, 0.0), G)
            elif w_max > 0:
                dw = self.inhibitory_learning_rate * p * (1.0 - (w * w) / w_max)
                w_new = min(max(w + dw, 0.0), w_max)   # clip to the finite ceiling
            else:
                dw = 0.0
                w_new = w
            self._weights_array[idx] = -w_new      # keep the inhibitory sign
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
        """Depress the ACTIVE positive feedforward gates of a neuron that was just
        suppressed by an L2I->L2E discharge (see Claude_Confidence_Consolidation_Plan.md).
        Nonlinear protect-small / punish-large, confidence-protected, and scaled by
        how close the neuron was to firing (p_loss). Inactive and mature gates are
        spared; the negative gate is never touched (positive weights only)."""
        theta = self.threshold
        p_loss = min(max(v_pre_loss / theta, 0.0), 1.0) if theta > 0 else 0.0
        if p_loss <= 0.0:
            return
        participating = self._last_input_spikes > 0.5
        active = np.nonzero((self._weights_array > 0) & participating)[0]
        if active.size == 0:
            return
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        w = self._weights_array[active]
        C = self._confidence[active]
        ratio = self._maturity(w)
        dw_minus = (self.eta_loss * p_loss * (1.0 - C)
                    * (ratio ** self.loss_gamma) * (w - w_min))
        # Clamp at the floor: the step is meant to decelerate INTO w_min, but a
        # large eta_loss (slider reaches 20) makes the multiplier exceed 1 and
        # overshoot below w_min into negatives. active is the positive-gate set,
        # so flooring here leaves the legitimate negative gate untouched.
        self._weights_array[active] = np.maximum(w - dw_minus, w_min)
        self.loser_depression_events += 1
        self._apply_budget_and_cap()

    def update(self):
        """
        Update neuron state for next time step.
        Handles refractory period and potential leak to resting state.
        """
        self._ensure_finalized()

        # Homeostatic firing-rate sensor: slow EMA of this neuron's own spiking.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)

        # Activity-dependent confidence decay (long-term memory; opt-in). See
        # update() for the full rationale.
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
            # self._leak_num over the integer LEAK_SCALE, so no float leak
            # constant enters here.
            # (Potential itself is still float on this scope's deferred path.)
            self.potential += self._leak_num * (self.resting_potential - self.potential) / LEAK_SCALE
            # Decay the eligibility trace with the same leak fraction.
            self._trace *= (1.0 - self._leak_num / LEAK_SCALE)
            # Inhibitory FLOW: drain the pending inhibitory current out of the
            # membrane this step (floored at rest), then decay it. Runs every step,
            # so a single discharge keeps suppressing the target for ~1/(1-decay)
            # steps instead of only on the step L2I fired. Not reset on fire -- the
            # pending inhibition drains fully regardless.
            if self.inhibitory_flow_rate and self.inh_trace > 0.0:
                self.potential = max(self.potential - self.inh_trace, self.resting_potential)
                self.inh_trace *= self.inh_trace_decay

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
        self._ensure_finalized()

        # Only check threshold if not in refractory period
        if self.refractory_timer <= 0 and self.potential >= self.threshold:
            return True
        return False
    
    def fire(self):
        """
        Handle spike event: capture charge, discharge (reset potential), start
        refractory period, and update weights from the captured charge.
        """
        self._ensure_finalized()

        # Record spike time
        self.last_spike_time = 0  # Current time step

        # Capture charge BEFORE discharging (mirrors apply_inhibition's V_pre).
        # Weight learning below uses this v_pre unchanged, so reset semantics do
        # not alter what the fire teaches -- only the post-fire membrane state.
        v_pre = float(self.potential)

        # Discharge: reset potential after firing. Default is a full reset to
        # rest; with subtractive_reset (opt-in) reset by SUBTRACTION instead --
        # leave the residual overshoot above threshold (standard LIF), floored at
        # rest so inhibition/leak conventions still hold.
        if self.subtractive_reset:
            self.potential = max(self.potential - self.threshold, self.resting_potential)
        else:
            self.potential = self.resting_potential

        # Start refractory period
        self.refractory_timer = self.refractory_period

        # Mark that we spiked this time step
        self.spiked = True

        # Flow-rate: discharge the excitatory current trace along with the membrane,
        # so no residual current keeps re-charging a just-fired neuron (and none
        # back-integrates when it leaves the refractory window). Consistent with the
        # full membrane reset above.
        if self.excitatory_flow_rate:
            self.exc_trace = 0.0

        # Update weights from the charge captured before discharge.
        self._update_weights(v_pre)

        # Evidence consumed: clear the trace for the next cycle (ARCHIVED bookkeeping).
        if self._trace is not None:
            self._trace = np.zeros_like(self._trace)

    def _update_weights(self, v_pre):
        """
        Charge-based excitatory weight update -- same algorithm and rationale for
        fixed and dynamic fan-in construction (same structure as
        apply_inhibition: capture charge, discharge, then
        dw = eta * p * (1 - w^2/w_max), with p = clamp(theta/v_pre, 0, 1) for
        excitation. w_max here is excitatory_saturation_cap (defaults to
        weight_cap when None), kept separate from the hard clip weight_cap so
        the sqrt(w_max) equilibrium can be made to land exactly on weight_cap
        -- see this class for the full rationale. Only synapses active in
        the most recent receive_input() call are updated
        (self._last_input_spikes), replacing the ARCHIVED trace-based
        participation and confidence-weighted credit-splitting rules.
        """
        if self._weights_array is None or len(self._weights_array) == 0:
            return
        theta = self.threshold
        p = min(max(theta / v_pre, 0.0), 1.0) if v_pre > 0 else 0.0
        participating = self._last_input_spikes > 0.5

        # Minimal signed-spike feedforward rule (opt-in). Every POSITIVE synapse
        # updates with a signed local signal -- +1 if its input participated in
        # this firing volley, -1 if not -- through the same saturating term
        # written with a LINEAR cap: dw = eta * p * (1 - (w/w_cap)^2) * signal.
        # Active inputs potentiate toward the cap; inactive inputs depress toward
        # the floor. No weight budget: the -1 signal on inactive inputs supplies
        # the downward pressure the budget used to impose. Bounded locally to
        # [min_positive_weight, weight_cap]. Negative inhibitory gates are NOT
        # touched here (they learn only via apply_inhibition). This fully replaces
        # the confidence / OFF-depression / budget path below, so return early.
        if self.signed_spike_learning:
            pos = self._weights_array > 0
            if pos.any() and self.weight_cap > 0:
                w = self._weights_array[pos]
                signal = np.where(participating[pos], 1.0, -1.0)
                dw = self.learning_rate * p * (1.0 - (w / self.weight_cap) ** 2) * signal
                w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
                self._weights_array[pos] = np.clip(w + dw, w_min, self.weight_cap)
            return

        # Flow-proportional assembly credit (opt-in; the E->I integrators L2I/L1I).
        # On this neuron's OWN fire, split credit across the incoming positive
        # synapses by the flow each delivered over the retention window -- the
        # per-synapse leaky _trace, which decays at this neuron's own leak so it
        # spans exactly the same window its membrane integrates. Normalizing by the
        # MAX flow (not the sum) gives the dominant driver the full learning rate,
        # so a habitual winner's synapse climbs to self-sufficiency instead of the
        # last spike to cross threshold taking all the credit. Synapses that
        # delivered no flow are depressed toward the floor. This is the whole fix
        # for the L2I firing deadlock; it replaces the last-volley path below.
        if self.assembly_flow_credit:
            pos = self._weights_array > 0
            w_max = self.excitatory_saturation_cap if self.excitatory_saturation_cap is not None else self.weight_cap
            if pos.any() and w_max > 0 and self._trace is not None:
                w = self._weights_array[pos]
                flow = self._trace[pos]
                fmax = float(flow.max())
                w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
                if fmax > 0.0:
                    fhat = flow / fmax                      # dominant driver -> 1.0
                    contributed = flow > 0.0
                    # Contributors potentiate toward the cap, scaled by flow share;
                    # non-contributors decelerate into the floor (down-pressure).
                    dw = np.where(
                        contributed,
                        self.learning_rate * p * fhat * (1.0 - (w * w) / w_max),
                        -self.learning_rate * p * self.assembly_decay_frac * (w - w_min),
                    )
                    self._weights_array[pos] = np.clip(w + dw, w_min, self.weight_cap)
            return

        active = np.nonzero((self._weights_array > 0) & participating)[0]
        w_max = self.excitatory_saturation_cap if self.excitatory_saturation_cap is not None else self.weight_cap
        if w_max > 0 and active.size > 0:
            w = self._weights_array[active]
            if self.confidence_consolidation:
                # Confidence-gated potentiation: mature (confident) gates learn
                # less, with a floor eta_min so no gate freezes.
                C = self._confidence[active]
                eta = self.learning_rate * (self.eta_min + (1.0 - self.eta_min) * (1.0 - C))
                self._weights_array[active] = w + eta * p * (1.0 - (w * w) / w_max)
                # Mature confidence toward local instantaneous maturity of the
                # (pre-update) gate; only active gates (x_i = 1) move.
                self._confidence[active] = C + self.conf_beta * (self._maturity(w) - C)
            else:
                dw = self.learning_rate * p * (1.0 - (w * w) / w_max)
                self._weights_array[active] = w + dw
        # Signed-spike depression ("4a"): OFF pixels (positive gates whose input
        # did NOT spike this fire) are pushed down. Confidence-gated when
        # consolidation is on (mature gates resist via (1 - C_i)); shaped by
        # (w_i - w_min) so a gate decelerates into the min_positive_weight floor
        # applied by _apply_budget_and_cap() -- it never goes deaf or negative.
        # Same event closeness p as potentiation.
        if self.signed_depression and self.eta_off > 0.0:
            inactive = np.nonzero((self._weights_array > 0) & ~participating)[0]
            if inactive.size > 0:
                w_off = self._weights_array[inactive]
                w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
                gate = (1.0 - self._confidence[inactive]) if self.confidence_consolidation else 1.0
                self._weights_array[inactive] = w_off - self.eta_off * p * gate * (w_off - w_min)
                self.signed_depression_events += 1
        self._apply_budget_and_cap()

    def _maturity(self, w):
        """Local instantaneous maturity m in [0,1] of positive gate weights w:
        (w - w_min) / (w_conf_cap - w_min), clamped. w_min is min_positive_weight
        (or 0), w_conf_cap is conf_cap (the effective reachable mature value; falls
        back to weight_cap). See Claude_Confidence_Consolidation_Plan.md."""
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        cap = self.conf_cap if self.conf_cap is not None else self.weight_cap
        if cap <= w_min:
            return np.zeros_like(w)
        return np.clip((w - w_min) / (cap - w_min), 0.0, 1.0)

    def _apply_budget_and_cap(self):
        """Shared tail: renormalize positive weights to the resource target, then
        the absolute cap. Under homeostasis the target is the homeostatic resource
        R; otherwise the fixed weight_budget. See _apply_budget_and_cap details,
        including the min_positive_weight floor applied below."""
        target = self._resource_target()
        if target is not None:
            pos = self._weights_array > 0
            total = float(self._weights_array[pos].sum())
            if total > 1e-9:
                self._weights_array[pos] *= target / total
        if self.min_positive_weight is not None:
            pos = self._weights_array > 0
            self._weights_array[pos] = np.maximum(self._weights_array[pos], self.min_positive_weight)
        self._weights_array = np.clip(self._weights_array, -self.weight_cap, self.weight_cap)

    def _resource_target(self):
        """Positive-weight budget to renormalize to: homeostatic resource R under
        homeostasis (lazy-init from current positive sum), else weight_budget."""
        if self.homeostasis:
            if self.homeo_budget is None:
                pos = self._weights_array > 0
                self.homeo_budget = float(self._weights_array[pos].sum()) if pos.any() else None
            return self.homeo_budget
        return self.weight_budget

    def _homeostatic_scaling(self):
        """Turrigiano-style synaptic scaling. Triggered by the neuron's OWN chronic firing rate
        (self.ca) leaving a target band, applies a FIXED multiplicative step (not a
        gradient, not global) to the excitatory resource, preserving relative
        weights so no pattern information is injected."""
        pos = self._weights_array > 0
        if not pos.any():
            return
        if self.homeo_budget is None:
            self.homeo_budget = float(self._weights_array[pos].sum())

        lo = self.ca_target * (1.0 - self.ca_band)
        hi = self.ca_target * (1.0 + self.ca_band)
        if self.ca < lo:
            self.homeo_budget *= (1.0 + self.homeo_up)      # chronically silent -> grow
        elif self.ca > hi:
            self.homeo_budget *= (1.0 - self.homeo_down)    # chronically over-active -> shrink
        if self.homeo_budget_min is not None:
            self.homeo_budget = max(self.homeo_budget, self.homeo_budget_min)
        if self.homeo_budget_max is not None:
            self.homeo_budget = min(self.homeo_budget, self.homeo_budget_max)

        total = float(self._weights_array[pos].sum())
        if total > 1e-9:
            self._weights_array[pos] *= self.homeo_budget / total
        self._weights_array = np.clip(self._weights_array, -self.weight_cap, self.weight_cap)

    def _decay_confidence(self):
        """Activity-dependent confidence decay."""
        if self._confidence is None:
            return
        if self.ca >= self.conf_ca_dead:
            self.inactive_steps = 0
            rho = self.conf_rho_active
        else:
            self.inactive_steps += 1
            rho = self.conf_rho_dead if self.inactive_steps >= self.conf_grace else 0.0
        if rho > 0.0:
            self._confidence *= (1.0 - rho)

    def plasticity_stats(self):
        """Summary statistics over the excitatory (positive-weight) afferents --
        weight/confidence entropy and concentration, plus current budget usage.
        See plasticity_stats on this class."""
        self._ensure_finalized()
        pos_mask = self._weights_array > 0
        pos_w = self._weights_array[pos_mask]
        pos_c = self._confidence[pos_mask]
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
        self._ensure_finalized()
        
        return {
            'potential': self.potential,
            'refractory_timer': self.refractory_timer,
            'spiked': self.spiked,
            'weights': self._weights_array.copy() if self._weights_array is not None else np.array([]),
            'confidence': self._confidence.copy() if self._confidence is not None else np.array([]),
            'last_spike_time': self.last_spike_time
        }

    @property
    def weights(self):
        """Get the weights array."""
        self._ensure_finalized()
        return self._weights_array.copy()

    @weights.setter
    def weights(self, value):
        """Replace weights for fixed-fan-in setup and tests."""
        self._set_weights(value, finalized=True)

    @property
    def trace(self):
        """Archived eligibility trace, kept for inspection/test compatibility."""
        self._ensure_finalized()
        return self._trace.copy()

    @trace.setter
    def trace(self, value):
        arr = np.asarray(value, dtype=float)
        if self._weights_array is not None and len(arr) != len(self._weights_array):
            raise ValueError("trace length must match weights length")
        self._trace = arr.copy()

    @property
    def distance(self):
        """Per-afferent delivery distance d_i (aligned to weights). Used only for
        distance attenuation of the delivered drive; never changes stored weights."""
        self._ensure_finalized()
        return self._distance.copy() if self._distance is not None else None

    @distance.setter
    def distance(self, value):
        arr = np.asarray(value, dtype=float)
        if self._weights_array is not None and len(arr) != len(self._weights_array):
            raise ValueError("distance length must match weights length")
        self._distance = arr.copy()

    @property
    def confidence(self):
        """Get the per-synapse confidence array (aligned to weights)."""
        self._ensure_finalized()
        return self._confidence.copy()

    @confidence.setter
    def confidence(self, value):
        arr = np.asarray(value, dtype=float)
        if self._weights_array is not None and len(arr) != len(self._weights_array):
            raise ValueError("confidence length must match weights length")
        self._confidence = arr.copy()
        
    @property
    def n_inputs(self):
        """Get the number of input connections."""
        self._ensure_finalized()
        return len(self._weights_array)
