"""
Flexible Neuron implementation that supports dynamic connection specification
during layer construction, while maintaining all required functionality.
"""

import numpy as np

from neuron import _distribution_entropy, _concentration


class Neuron:
    """
    A flexible spiking neuron model that allows dynamic specification of 
    input connections during layer construction, then locks connections 
    for simulation.
    
    Both excitatory and inhibitory neurons use the same dynamics;
    the difference is in the sign of their synaptic weights.
    """
    
    def __init__(self, threshold=1.0, refractory_period=2,
                 learning_rate=0.1, weight_cap=1.0, leak_rate=0.01,
                 inhibitory_learning_rate=0.05, inhibitory_weight_cap=None,
                 excitatory_saturation_cap=None,
                 trace_mode="activity", confidence_init=0.10,
                 confidence_beta=0.30, confidence_gamma=0.02,
                 homeostasis=False, ca_rate=0.01, ca_target=0.02, ca_band=0.5,
                 homeo_up=0.01, homeo_down=0.01,
                 homeo_budget_min=None, homeo_budget_max=None):
        """
        Initialize a flexible neuron.

        Args:
            threshold (float): Firing threshold (constant)
            refractory_period (int): Refractory period in time steps (1ms each)
            learning_rate (float): Amount weights increase when neuron fires (excitatory plasticity)
            weight_cap (float): Maximum absolute value for weights (weights clipped to [-weight_cap, weight_cap]).
                Also serves as w_max, the saturating magnitude ceiling for inhibitory plasticity.
            leak_rate (float): Fraction of potential that leaks away per time step (0 = no leak, 1 = full leak)
            inhibitory_learning_rate (float): eta for the inhibitory-discharge plasticity rule
                (see apply_inhibition). 0 disables inhibitory learning.
            trace_mode, confidence_init, confidence_beta, confidence_gamma: ARCHIVED.
                See neuron.Neuron's __init__ docstring -- these are still
                accepted and self.trace_mode/self._confidence are still set,
                but no longer affect learning; every neuron now uses the
                single charge-based rule (see _update_weights).
            homeostasis, ca_rate, ca_target, ca_band, homeo_up, homeo_down,
            homeo_budget_min, homeo_budget_max: homeostatic synaptic scaling; see
                neuron.Neuron for the full description (identical semantics).
        """
        if trace_mode not in ("activity", "confidence"):
            raise ValueError(f"trace_mode must be 'activity' or 'confidence', got {trace_mode!r}")
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
        # Confidence state (see neuron.Neuron). Allocated at finalize once the
        # fan-in is known. In "activity" mode it is left untouched.
        self._confidence = None
        self.trace_mode = trace_mode
        self.confidence_init = float(confidence_init)
        self.confidence_beta = confidence_beta
        self.confidence_gamma = confidence_gamma
        self.last_excitatory_event = {}
        # Homeostatic synaptic scaling (third local system; see neuron.Neuron).
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
        self.leak_rate = leak_rate          # Leak rate (fraction of potential lost per ms)
        self.inhibitory_learning_rate = inhibitory_learning_rate  # eta for inhibitory plasticity
        # Saturation ceiling (w_max) for inhibitory gates, kept separate from the
        # feedforward weight_cap (defaults to weight_cap when None). See apply_inhibition.
        self.inhibitory_weight_cap = inhibitory_weight_cap
        # Saturation ceiling (w_max) for the EXCITATORY quadratic term, kept
        # separate from the hard clip weight_cap -- see neuron.Neuron for the
        # full rationale (sqrt(w_max) equilibrium; set to weight_cap**2 so a
        # habitually-participating synapse can reach the hard cap exactly).
        self.excitatory_saturation_cap = excitatory_saturation_cap
        self.last_inhibitory_events = []    # debug records from the most recent apply_inhibition()
        self._connections_finalized = False  # Flag to prevent changes after finalization
        
        # Spike tracking
        self.last_spike_time = -np.inf      # Time of last spike
        self.spiked = False                 # Did neuron spike in current step?
        
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
            if self._weights_list:
                self._weights_array = np.array(self._weights_list, dtype=float)
            else:
                # Handle case of zero connections
                self._weights_array = np.array([], dtype=float)
            # ARCHIVED: eligibility trace, kept only for inspection compatibility.
            self._trace = np.zeros(len(self._weights_array))
            # ARCHIVED: confidence, allocated but never updated (see __init__ docs).
            self._confidence = np.full(len(self._weights_array), self.confidence_init)
            # Instantaneous participation signal for the charge-based rule.
            self._last_input_spikes = np.zeros(len(self._weights_array))
            self._connections_finalized = True
            
    def _ensure_finalized(self):
        """Internal method to check if connections are ready for simulation."""
        if not self._connections_finalized:
            raise RuntimeError("Connections not finalized. "
                             "Call finalize_connections() before simulation.")
        
    def receive_input(self, input_spikes):
        """
        Accumulate charge based on weighted inputs.
        
        Args:
            input_spikes (array-like): Binary array indicating which inputs spiked (1) or not (0)
        """
        self._ensure_finalized()
        
        # Only accumulate charge if not in refractory period
        if self.refractory_timer <= 0:
            # Charge accumulation: sum of weight * input for all connections
            if len(self._weights_array) > 0:
                input_spikes = np.asarray(input_spikes, dtype=float)
                input_current = np.dot(self._weights_array, input_spikes)
                self.potential += input_current
                # ARCHIVED: no longer read by _update_weights. Kept only so
                # anything still inspecting ._trace keeps working.
                self._trace += input_spikes
                # Instantaneous participation signal for the charge-based
                # excitatory rule -- see neuron.Neuron._update_weights.
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
        active = np.nonzero((self._weights_array < 0) & (spikes > 0.5))[0]
        for idx in active:
            w = -float(self._weights_array[idx])   # magnitude of the inhibitory gate
            v_pre = float(self.potential)
            self.potential -= w                    # linear discharge: V = V - w
            v_post = float(self.potential)
            # Normalized closeness to firing, clamped to [0, 1].
            p = min(max(v_pre / theta, 0.0), 1.0) if theta > 0 else 0.0
            if w_max > 0:
                dw = self.inhibitory_learning_rate * p * (1.0 - (w * w) / w_max)
            else:
                dw = 0.0
            w_new = min(max(w + dw, 0.0), w_max)   # saturate at the finite ceiling
            self._weights_array[idx] = -w_new      # keep the inhibitory sign
            events.append(dict(index=int(idx), v_pre=v_pre, v_post=v_post,
                               theta=theta, p=p, w_before=w,
                               delta_w=w_new - w, w_after=w_new))
        return events

    def update(self):
        """
        Update neuron state for next time step.
        Handles refractory period and potential leak to resting state.
        """
        self._ensure_finalized()

        # Homeostatic firing-rate sensor: slow EMA of this neuron's own spiking.
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
            self._trace *= (1.0 - self.leak_rate)

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
        v_pre = float(self.potential)

        # Discharge: reset potential after firing.
        self.potential = self.resting_potential

        # Start refractory period
        self.refractory_timer = self.refractory_period

        # Mark that we spiked this time step
        self.spiked = True

        # Update weights from the charge captured before discharge.
        self._update_weights(v_pre)

        # Evidence consumed: clear the trace for the next cycle (ARCHIVED bookkeeping).
        if self._trace is not None:
            self._trace = np.zeros_like(self._trace)

    def _update_weights(self, v_pre):
        """
        Charge-based excitatory weight update -- identical algorithm and
        rationale to neuron.Neuron._update_weights (same structure as
        apply_inhibition: capture charge, discharge, then
        dw = eta * p * (1 - w^2/w_max), with p = clamp(theta/v_pre, 0, 1) for
        excitation. w_max here is excitatory_saturation_cap (defaults to
        weight_cap when None), kept separate from the hard clip weight_cap so
        the sqrt(w_max) equilibrium can be made to land exactly on weight_cap
        -- see neuron.Neuron for the full rationale. Only synapses active in
        the most recent receive_input() call are updated
        (self._last_input_spikes), replacing the ARCHIVED trace-based
        participation and confidence-weighted credit-splitting rules.
        """
        if self._weights_array is None or len(self._weights_array) == 0:
            return
        theta = self.threshold
        p = min(max(theta / v_pre, 0.0), 1.0) if v_pre > 0 else 0.0
        participating = self._last_input_spikes > 0.5
        active = np.nonzero((self._weights_array > 0) & participating)[0]
        w_max = self.excitatory_saturation_cap if self.excitatory_saturation_cap is not None else self.weight_cap
        if w_max > 0 and active.size > 0:
            w = self._weights_array[active]
            dw = self.learning_rate * p * (1.0 - (w * w) / w_max)
            self._weights_array[active] = w + dw
        self._apply_budget_and_cap()

    def _apply_budget_and_cap(self):
        """Shared tail: renormalize positive weights to the resource target, then
        the absolute cap. Under homeostasis the target is the homeostatic resource
        R; otherwise the fixed weight_budget. See neuron.Neuron for details,
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
        """Turrigiano-style synaptic scaling; see neuron.Neuron._homeostatic_scaling
        for the full rationale. Triggered by the neuron's OWN chronic firing rate
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

    def plasticity_stats(self):
        """Summary statistics over the excitatory (positive-weight) afferents --
        weight/confidence entropy and concentration, plus current budget usage.
        See neuron.Neuron.plasticity_stats."""
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

    @property
    def confidence(self):
        """Get the per-synapse confidence array (aligned to weights)."""
        self._ensure_finalized()
        return self._confidence.copy()
        
    @property
    def n_inputs(self):
        """Get the number of input connections."""
        self._ensure_finalized()
        return len(self._weights_array)