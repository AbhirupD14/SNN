"""
Modified CorticalColumn class to support L3 to L2 stacking for feedback inhibition.
"""

import numpy as np
from neuron_flexible import Neuron

class CorticalColumn:
    """
    Cortical Column with flexible connectivity to support hierarchical stacking.
    Supports standard lateral inhibition plus feedback connections from higher layers.
    """
    
    def __init__(self, n_neurons, threshold=1.0, refractory_period=2, 
                 learning_rate=0.05, weight_cap=1.0, leak_rate=0.01):
        """
        Initialize cortical column.
        
        Args:
            n_neurons: Number of excitatory neurons in the column
            threshold: Firing threshold for neurons
            refractory_period: Refractory period in time steps
            learning_rate: Weight increase when neuron fires
            weight_cap: Maximum absolute value for weights
            leak_rate: Leak rate (fraction of potential lost per ms)
        """
        self.n_neurons = n_neurons
        # Create excitatory neurons - will receive [from_local_I, from_below]
        self.excitatory_neurons = [
            Neuron(threshold=threshold, refractory_period=refractory_period,
                   learning_rate=learning_rate, weight_cap=weight_cap, leak_rate=leak_rate)
            for _ in range(n_neurons)
        ]
        
        # Create inhibitory neuron - will receive [from_local_E, from_above_E]
        self.inhibitory_neuron = Neuron(
            threshold=threshold, 
            refractory_period=refractory_period,
            learning_rate=learning_rate,
            weight_cap=weight_cap,
            leak_rate=leak_rate
        )
        
        # Track connection patterns
        self._connections_finalized = False
        self.n_feedback_inputs = 0  # Number of feedback inputs to inhibitory neuron
        
    def setup_connectivity(self, n_feedback_inputs=0):
        """
        Set up connectivity patterns for hierarchical stacking:
        - Each E neuron: [from_local_I, from_below] 
        - I neuron: [from_all_local_E, from_feedback_sources]
        
        Args:
            n_feedback_inputs: Number of feedback inputs from layer above (default 0)
        """
        self.n_feedback_inputs = n_feedback_inputs
        
        # Set up excitatory neurons: [from_local_I, from_below]
        for e_neuron in self.excitatory_neurons:
            e_neuron.add_input_connection(0.0)  # Placeholder for from_local_I
            e_neuron.add_input_connection(0.0)  # Placeholder for from_below
            
        # Set up inhibitory neuron: [from_all_local_E, from_feedback_sources]
        for i in range(self.n_neurons):
            self.inhibitory_neuron.add_input_connection(0.0)  # Placeholder for from_E_i
        for i in range(n_feedback_inputs):
            self.inhibitory_neuron.add_input_connection(0.0)  # Placeholder for feedback input i
            
    def finalize_connections(self):
        """Finalize all connections - must be called before simulation."""
        for e_neuron in self.excitatory_neurons:
            e_neuron.finalize_connections()
        self.inhibitory_neuron.finalize_connections()
        self._connections_finalized = True
        
    def _ensure_finalized(self):
        """Internal method to check if connections are ready for simulation."""
        if not self._connections_finalized:
            raise RuntimeError("Connections not finalized. "
                             "Call finalize_connections() before simulation.")
        
    def set_local_inhibition_weights(self, weight):
        """Set the weights for local inhibition (I -> E connections)."""
        self._ensure_finalized()
        for i, e_neuron in enumerate(self.excitatory_neurons):
            # Weight from I to E_i is at index 0
            current_weights = e_neuron._weights_array.copy()
            current_weights[0] = weight
            e_neuron._weights_array = current_weights
            
    def set_feedforward_weights(self, weight):
        """Set the weights for feedforward connections (below E -> E connections)."""
        self._ensure_finalized()
        for i, e_neuron in enumerate(self.excitatory_neurons):
            # Weight from below to E_i is at index 1
            current_weights = e_neuron._weights_array.copy()
            current_weights[1] = weight
            e_neuron._weights_array = current_weights
            
    def set_lateral_inhibition_weights(self, weight):
        """Set the weights for lateral inhibition within the layer (E -> I connections)."""
        self._ensure_finalized()
        # Weights from E_j to I are at indices 0 to n_neurons-1
        for j in range(self.n_neurons):
            current_weights = self.inhibitory_neuron._weights_array.copy()
            current_weights[j] = weight
            self.inhibitory_neuron._weights_array = current_weights
            
    def set_feedback_weights(self, weight):
        """Set the weights for feedback connections (above E -> I connections)."""
        self._ensure_finalized()
        # Weights from above E to I are at indices n_neurons to n_neurons+n_feedback_inputs-1
        start_idx = self.n_neurons
        end_idx = start_idx + self.n_feedback_inputs
        for j in range(start_idx, end_idx):
            current_weights = self.inhibitory_neuron._weights_array.copy()
            current_weights[j] = weight
            self.inhibitory_neuron._weights_array = current_weights
            
    def get_state(self):
        """
        Get current state of the cortical column.
        
        Returns:
            dict: Dictionary containing state of all neurons
        """
        self._ensure_finalized()
        
        return {
            'excitatory_potentials': [n.potential for n in self.excitatory_neurons],
            'excitatory_spiked': [n.spiked for n in self.excitatory_neurons],
            'excitatory_weights': [n._weights_array.copy() for n in self.excitatory_neurons],
            'inhibitory_potential': self.inhibitory_neuron.potential,
            'inhibitory_spiked': self.inhibitory_neuron.spiked,
            'inhibitory_weights': self.inhibitory_neuron._weights_array.copy(),
            'refractory_timers_E': [n.refractory_timer for n in self.excitatory_neurons],
            'refractory_timer_I': self.inhibitory_neuron.refractory_timer
        }