import numpy as np

# A synapse counts as having "participated" in the current integration window if
# its (leaky) trace accumulator carries any residual charge above this floor.
PARTICIPATION_EPS = 1e-6


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
    
    def __init__(self, n_inputs, threshold=1.0, refractory_period=2,
                 weight_init_range=(-0.5, 0.5), learning_rate=0.1, weight_cap=1.0, leak_rate=0.01,
                 inhibitory_learning_rate=0.05, inhibitory_weight_cap=None,
                 trace_mode="activity", confidence_init=0.10,
                 confidence_beta=0.30, confidence_gamma=0.02,
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
            trace_mode (str): meaning of the excitatory eligibility trace when the
                neuron fires (see _update_weights):
                  "activity"   -- original rule: dw = lr * trace * sign(w), where
                                  trace is recent per-synapse charge. Every active
                                  synapse gets near-equal credit.
                  "confidence" -- trace is instead a slowly-changing per-synapse
                                  CONFIDENCE (how useful this gate has historically
                                  been for producing this neuron's spikes). Learning
                                  is allocated by confidence-weighted credit; see the
                                  confidence_* args. Weight = gate size, confidence =
                                  trust in that gate: two separate quantities.
            confidence_init (float): starting confidence for every synapse in
                "confidence" mode (small but non-zero, so credit is well-defined).
            confidence_beta (float): growth rate toward 1 for synapses that
                participated in a successful spike: c += beta * (1 - c).
            confidence_gamma (float): slow decay for synapses that did NOT
                participate in a successful spike: c *= (1 - gamma).
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
        if trace_mode not in ("activity", "confidence"):
            raise ValueError(f"trace_mode must be 'activity' or 'confidence', got {trace_mode!r}")
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
        self.leak_rate = leak_rate          # Leak rate (fraction of potential lost per ms)
        self.inhibitory_learning_rate = inhibitory_learning_rate  # eta for inhibitory plasticity
        # Saturation ceiling (w_max) for inhibitory gates. Kept separate from the
        # feedforward weight_cap so a saturated gate need not be strong enough to
        # fully reset the membrane (which would recreate hard-WTA collapse).
        # Defaults to weight_cap when None.
        self.inhibitory_weight_cap = inhibitory_weight_cap

        # Debug record of the inhibitory-discharge events from the most recent
        # apply_inhibition() call (one dict per event). See apply_inhibition.
        self.last_inhibitory_events = []

        # Per-synapse eligibility trace: integrates which input lines delivered
        # charge, with the same leak as the membrane. In "activity" mode it IS
        # the learning signal; in "confidence" mode it is used only to decide
        # which synapses PARTICIPATED in the current integration window.
        self.trace = np.zeros(n_inputs)

        # Excitatory-trace semantics and confidence state (see __init__ docs).
        self.trace_mode = trace_mode
        self.confidence_beta = confidence_beta
        self.confidence_gamma = confidence_gamma
        # Confidence = the neuron's trust that opening each gate contributes to a
        # successful spike. A separate biological quantity from the weight (gate
        # size). Only consulted in "confidence" mode. Starts small but non-zero
        # so credit_i = c_i / sum(c_active) is always well-defined.
        self.confidence = np.full(n_inputs, float(confidence_init))

        # Debug record of the most recent excitatory (_update_weights) event, in
        # "confidence" mode -- participation, confidence before/after, credits,
        # weight deltas, and budget before/after normalization. See _update_weights.
        self.last_excitatory_event = {}

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

        # Spike tracking
        self.last_spike_time = -np.inf      # Time of last spike
        self.spiked = False                 # Did neuron spike in current step?
        
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
            # Track which lines delivered the charge (un-summed membrane)
            self.trace += input_spikes

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
            4. dw     = eta * p * (1 - w / w_max)   (saturating; w_max = inhibitory_weight_cap
                                                    or weight_cap if that is None)
            5. w      = w + dw                   (gate strengthens toward w_max)

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
        active = np.nonzero((self.weights < 0) & (spikes > 0.5))[0]
        for idx in active:
            w = -float(self.weights[idx])          # magnitude of the inhibitory gate
            v_pre = float(self.potential)
            self.potential -= w                    # linear discharge: V = V - w
            v_post = float(self.potential)
            # Normalized closeness to firing at inhibition time, clamped to [0, 1]:
            # a hyperpolarized membrane never drives negative learning, and a
            # neuron already at/above threshold caps the drive at p = 1.
            p = min(max(v_pre / theta, 0.0), 1.0) if theta > 0 else 0.0
            if w_max > 0:
                dw = self.inhibitory_learning_rate * p * (1.0 - w / w_max)
            else:
                dw = 0.0
            w_new = min(max(w + dw, 0.0), w_max)   # saturate at the finite ceiling
            self.weights[idx] = -w_new             # keep the inhibitory sign
            events.append(dict(index=int(idx), v_pre=v_pre, v_post=v_post,
                               theta=theta, p=p, w_before=w,
                               delta_w=w_new - w, w_after=w_new))
        return events

    def update(self):
        """
        Update neuron state for next time step.
        Handles refractory period and potential leak to resting state.
        """
        # Homeostatic firing-rate sensor: a slow EMA of the neuron's OWN spiking
        # this step (self.spiked is still set from fire() and is reset at the end
        # of this method). Purely local -- reads nothing but its own output.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)

        # Handle refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            # Clamp potential to resting during refractory period
            self.potential = self.resting_potential
        else:
            # Apply leak: potential decays toward resting potential
            # leak_rate is the fraction of distance to resting potential that is closed each time step
            # For example, leak_rate=0.01 means 1% of the way to resting potential each ms
            leak_current = self.leak_rate * (self.resting_potential - self.potential)
            self.potential += leak_current
            # Decay the eligibility trace with the same leak as the membrane
            self.trace *= (1.0 - self.leak_rate)

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
        Handle spike event: reset potential, start refractory period,
        and update weights.
        """
        # Record spike time
        self.last_spike_time = 0  # Current time step
        
        # Reset potential after firing
        self.potential = self.resting_potential
        
        # Start refractory period
        self.refractory_timer = self.refractory_period
        
        # Mark that we spiked this time step
        self.spiked = True
        
        # Update weights (only happens when neuron fires)
        self._update_weights()

        # Evidence consumed: clear the trace so the next cycle starts fresh
        self.trace = np.zeros_like(self.trace)

    def _update_weights(self):
        """
        Hebbian weight update, applied only when the neuron fires. Dispatches on
        trace_mode; both variants end with the SAME budget normalization + cap.
        """
        if self.trace_mode == "confidence":
            self._update_weights_confidence()
        else:
            self._update_weights_activity()

    def _update_weights_activity(self):
        """
        Original rule: strengthen each synapse in proportion to its eligibility
        trace -- how much that input line contributed charge over the recent
        window. Silent lines have trace ~0 and are left essentially unchanged.

        Sign-preserving: excitatory inputs grow more positive, inhibitory inputs
        grow more negative (|w| increases either way). A neuron is excitatory or
        inhibitory purely by the sign of the weight it lands on in its target.
        """
        self.weights += self.learning_rate * self.trace * np.sign(self.weights)
        self._apply_budget_and_cap()

    def _update_weights_confidence(self):
        """
        Confidence-weighted excitatory allocation (trace_mode="confidence").

        Weight = gate size; confidence = the neuron's trust that opening that
        gate helps it fire. These evolve on different timescales:

          1. Participation is read from the eligibility trace (which synapses
             delivered charge in the window that led to THIS spike). Participation
             alone does not permanently raise confidence.
          2. Confidence updates only on a successful spike:
               participating:      c += beta * (1 - c)      (fast, saturating -> 1)
               non-participating:   c *= (1 - gamma)         (slow forgetting)
          3. The fixed learning budget is then allocated across the ACTIVE
             EXCITATORY synapses in proportion to their (updated) confidence:
               credit_i = c_i / sum(c over active excitatory)
               dw_i     = learning_rate * credit_i           (>= 0, excitatory only)
          4. The existing weight-budget normalization + cap run exactly as in the
             activity rule, preserving the finite-resource interpretation.

        Inhibitory synapses receive no excitatory credit here (their plasticity is
        the separate apply_inhibition rule), so the two systems stay disjoint.
        Everything is local: only this neuron's trace, confidence, and weights.
        """
        participated = self.trace > PARTICIPATION_EPS
        conf_before = self.confidence.copy()

        # 2. Confidence dynamics (only on this spike).
        self.confidence[participated] += self.confidence_beta * (1.0 - self.confidence[participated])
        self.confidence[~participated] *= (1.0 - self.confidence_gamma)

        # 3. Confidence-weighted credit over active EXCITATORY synapses.
        exc_active = participated & (self.weights > 0)
        credit = np.zeros_like(self.weights)
        denom = float(self.confidence[exc_active].sum())
        if denom > PARTICIPATION_EPS:
            credit[exc_active] = self.confidence[exc_active] / denom
        dw = self.learning_rate * credit
        self.weights += dw

        # Instrumentation: capture the full event before budget normalization.
        pos = self.weights > 0
        budget_before = float(self.weights[pos].sum())

        # 4. Existing budget normalization + cap.
        self._apply_budget_and_cap()

        pos = self.weights > 0
        budget_after = float(self.weights[pos].sum())
        idx = np.nonzero(participated)[0]
        self.last_excitatory_event = dict(
            participating=idx.tolist(),
            confidence_before=conf_before[idx].tolist(),
            confidence_after=self.confidence[idx].tolist(),
            credits=credit[idx].tolist(),
            delta_w=dw[idx].tolist(),
            budget_before=budget_before,
            budget_after=budget_after,
        )

    def _apply_budget_and_cap(self):
        """Shared tail of both excitatory rules: renormalize positive (excitatory)
        weights to a fixed total so strengthening one input weakens the others (no
        runaway growth), then clip to [-weight_cap, weight_cap]. When homeostasis
        is on, the target total is the homeostatic resource R (self.homeo_budget)
        rather than the fixed weight_budget -- so the resource is regulated by the
        neuron's own activity instead of being a hard constant."""
        target = self._resource_target()
        if target is not None:
            pos = self.weights > 0
            total = float(self.weights[pos].sum())
            if total > 1e-9:
                self.weights[pos] *= target / total
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