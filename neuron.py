import numpy as np

class Neuron:
    """
    A basic spiking neuron model with charge accumulation, refractory period,
    and activity-dependent weight updates.
    
    Both excitatory and inhibitory neurons use the same dynamics;
    the difference is in the sign of their synaptic weights.
    """
    
    def __init__(self, n_inputs, threshold=1.0, refractory_period=2,
                 weight_init_range=(-0.5, 0.5), learning_rate=0.1, weight_cap=1.0, leak_rate=0.01,
                 inhibitory_learning_rate=0.05, inhibitory_weight_cap=None):
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
        # charge, with the same leak as the membrane. Used to credit only the
        # synapses that contributed when the neuron fires.
        self.trace = np.zeros(n_inputs)

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
        Hebbian weight update, applied only when the neuron fires.

        Each synapse is strengthened in proportion to its eligibility trace --
        how much that input line contributed charge over the recent window.
        Lines that were silent have trace ~0 and are left essentially unchanged.
        Weights are then clipped to [-weight_cap, weight_cap].

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
        self.weights += self.learning_rate * self.trace * np.sign(self.weights)

        # Homeostatic budget: renormalize excitatory weights to a fixed total, so a
        # neuron that strengthens one input must weaken others (no runaway growth).
        if self.weight_budget is not None:
            pos = self.weights > 0
            total = float(self.weights[pos].sum())
            if total > 1e-9:
                self.weights[pos] *= self.weight_budget / total

        # Apply weight cap to prevent infinite growth
        self.weights = np.clip(self.weights, -self.weight_cap, self.weight_cap)
        
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
            'last_spike_time': self.last_spike_time
        }