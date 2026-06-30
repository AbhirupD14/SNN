import numpy as np

class Neuron:
    """
    A basic spiking neuron model with charge accumulation, refractory period,
    and activity-dependent weight updates.
    
    Both excitatory and inhibitory neurons use the same dynamics;
    the difference is in the sign of their synaptic weights.
    """
    
    def __init__(self, n_inputs, threshold=1.0, refractory_period=2, 
                 weight_init_range=(-0.5, 0.5), learning_rate=0.1, weight_cap=1.0, leak_rate=0.01):
        """
        Initialize a neuron.
        
        Args:
            n_inputs (int): Number of input connections
            threshold (float): Firing threshold (constant)
            refractory_period (int): Refractory period in time steps (1ms each)
            weight_init_range (tuple): Min and max for random weight initialization
            learning_rate (float): Amount weights increase when neuron fires
            weight_cap (float): Maximum absolute value for weights (weights clipped to [-weight_cap, weight_cap])
            leak_rate (float): Fraction of potential that leaks away per time step (0 = no leak, 1 = full leak)
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
        self.learning_rate = learning_rate  # Weight increase amount when firing
        self.weight_cap = weight_cap        # Maximum absolute weight value
        self.leak_rate = leak_rate          # Leak rate (fraction of potential lost per ms)
        
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
            # Charge accumulation: sum of weight * input for all connections
            input_current = np.dot(self.weights, input_spikes)
            self.potential += input_current
            
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
        
    def _update_weights(self):
        """
        Update synaptic weights when neuron fires.
        Weights increase by learning_rate amount, then are clipped to weight_cap.
        Note: For inhibitory neurons, weights are negative so this makes them more negative
        (up to the negative weight_cap limit).
        """
        # Simple rule: weights increase by learning_rate when neuron fires
        # This implements #4: "Weights increase when firing"
        self.weights += self.learning_rate
        
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