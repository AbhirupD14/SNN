"""
Flexible Neuron implementation that supports dynamic connection specification
during layer construction, while maintaining all required functionality.
"""

import numpy as np

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
                 inhibitory_learning_rate=0.05):
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
        self.weight_budget = None      # optional homeostatic budget for positive weights
        self.learning_rate = learning_rate  # Weight increase amount when neuron fires (excitatory)
        self.weight_cap = weight_cap        # Maximum absolute value for weights (also w_max for inhibition)
        self.leak_rate = leak_rate          # Leak rate (fraction of potential lost per ms)
        self.inhibitory_learning_rate = inhibitory_learning_rate  # eta for inhibitory plasticity
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
            # Eligibility trace: one accumulator per synapse, mirrors the membrane
            self._trace = np.zeros(len(self._weights_array))
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
                # Track which lines delivered the charge (un-summed membrane)
                self._trace += input_spikes

    def apply_inhibition(self, inhibitory_spikes):
        """
        Deliver inhibitory-discharge events and run the inhibitory plasticity rule.

        Independent second learning system (the excitatory rule in _update_weights
        is untouched and fires only on a postsynaptic spike). An inhibitory synapse
        is any afferent with a negative weight, acting as an adaptive suppression
        gate. For every inhibitory synapse carrying a spike this step, per the
        algorithm (w = |weight|, w_max = weight_cap, theta = threshold):

            V_pre  = V ; V = V - w ; V_post = V
            p      = V_pre / theta
            dw     = eta * p * (1 - w / w_max)   (saturating, local, gradient-free)
            w      = w + dw

        The gate strengthens most when it suppressed a near-threshold neuron and
        saturates as w -> w_max, so no global normalization is needed. Sign is
        preserved (weights stay negative; |w| grows).

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
        w_max = self.weight_cap
        active = np.nonzero((self._weights_array < 0) & (spikes > 0.5))[0]
        for idx in active:
            w = -float(self._weights_array[idx])   # magnitude of the inhibitory gate
            v_pre = float(self.potential)
            self.potential -= w                    # linear discharge: V = V - w
            v_post = float(self.potential)
            p = max(v_pre / theta, 0.0) if theta > 0 else 0.0
            if w_max > 0:
                dw = self.inhibitory_learning_rate * p * (1.0 - w / w_max)
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
        Handle spike event: reset potential, start refractory period,
        and update weights.
        """
        self._ensure_finalized()
        
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

        # Evidence consumed: clear the trace for the next cycle
        if self._trace is not None:
            self._trace = np.zeros_like(self._trace)

    def _update_weights(self):
        """
        Hebbian weight update, applied only when the neuron fires.

        Each synapse is strengthened in proportion to its eligibility trace --
        how much that input line contributed charge over the recent window.
        Silent lines have trace ~0 and are left essentially unchanged. Weights
        are then clipped to [-weight_cap, weight_cap].

        The update is sign-preserving: each synapse is strengthened in the
        direction of its own sign, so excitatory inputs grow more positive and
        inhibitory inputs (negative weights, which live in the target neuron's
        array) grow more negative -- i.e. |w| increases either way. A neuron is
        excitatory or inhibitory purely by the sign of the weight it lands on in
        its target; the neuron itself just fires, and trains its own afferents
        with this identical rule.
        """
        # Credit only the synapses that contributed charge (pre_i * post),
        # strengthening each in the direction of its existing sign.
        if self._weights_array is not None and len(self._weights_array) > 0:
            self._weights_array += self.learning_rate * self._trace * np.sign(self._weights_array)

            # Homeostatic budget: renormalize excitatory weights to a fixed total, so
            # strengthening one input weakens the others (no runaway growth).
            if self.weight_budget is not None:
                pos = self._weights_array > 0
                total = float(self._weights_array[pos].sum())
                if total > 1e-9:
                    self._weights_array[pos] *= self.weight_budget / total

            # Apply weight cap to prevent infinite growth
            self._weights_array = np.clip(self._weights_array, -self.weight_cap, self.weight_cap)
        
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
            'last_spike_time': self.last_spike_time
        }
        
    @property
    def weights(self):
        """Get the weights array."""
        self._ensure_finalized()
        return self._weights_array.copy()
        
    @property
    def n_inputs(self):
        """Get the number of input connections."""
        self._ensure_finalized()
        return len(self._weights_array)