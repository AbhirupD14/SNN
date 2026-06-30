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
                 learning_rate=0.1, weight_cap=1.0, leak_rate=0.01):
        """
        Initialize a flexible neuron.
        
        Args:
            threshold (float): Firing threshold (constant)
            refractory_period (int): Refractory period in time steps (1ms each)
            learning_rate (float): Amount weights increase when neuron fires
            weight_cap (float): Maximum absolute value for weights (weights clipped to [-weight_cap, weight_cap])
            leak_rate (float): Fraction of potential that leaks away per time step (0 = no leak, 1 = full leak)
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
        self.learning_rate = learning_rate  # Weight increase amount when neuron fires
        self.weight_cap = weight_cap        # Maximum absolute value for weights
        self.leak_rate = leak_rate          # Leak rate (fraction of potential lost per ms)
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
                input_current = np.dot(self._weights_array, input_spikes)
                self.potential += input_current
           
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
        
    def _update_weights(self):
        """
        Update synaptic weights when neuron fires.
        Weights increase by learning_rate amount, then are clipped to weight_cap.
        Note: For inhibitory neurons, weights are negative so this makes them more negative
        (up to the negative weight_cap limit).
        """
        # Simple rule: weights increase by learning_rate when neuron fires
        # This implements #4: "Weights increase when firing"
        if self._weights_array is not None and len(self._weights_array) > 0:
            self._weights_array += self.learning_rate
            
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