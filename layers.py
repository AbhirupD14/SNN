"""
Implementation of neural layers: InputLayer and CorticalColumn
based on the specified architecture.
"""

import numpy as np
from neuron_flexible import Neuron, UNIT, LEAK_SCALE

class InputLayer:
    """
    Input Layer with n excitatory neurons, each having a dedicated inhibitory neuron.
    Connections: I1_i -> E1_i (inhibitory, 1:1)
    """
    
    def __init__(self, n_neurons, threshold=1000 / UNIT, refractory_period=2,
                 learning_rate=0.05, weight_cap=1000 / UNIT, leak_rate=10 / LEAK_SCALE,
                 n_feedback_inputs=0):
        """
        Initialize input layer.

        Args:
            n_neurons: Number of E/I pairs
            threshold: Firing threshold for neurons
            refractory_period: Refractory period in time steps
            learning_rate: Weight increase when neuron fires
            weight_cap: Maximum absolute weight value
            leak_rate: Leak rate (fraction of potential lost per ms)
            n_feedback_inputs: Number of top-down feedback synapses on each
                inhibitory neuron (e.g. one per higher-layer E neuron, for the
                E2->I1 "quiet the inputs" loop). 0 keeps a single inert input.
        """
        self.n_neurons = n_neurons
        self.n_feedback_inputs = n_feedback_inputs

        # Create excitatory neurons (each receives input from its inhibitory neuron + external)
        self.excitatory_neurons = [
            Neuron(n_inputs=2, threshold=threshold, refractory_period=refractory_period,
                   learning_rate=learning_rate, weight_cap=weight_cap, leak_rate=leak_rate)
            for _ in range(n_neurons)
        ]

        # Create inhibitory neurons. Each carries one afferent synapse per feedback
        # source (from the layer above); each in turn projects to its paired E neuron.
        self.inhibitory_neurons = [
            Neuron(n_inputs=max(1, n_feedback_inputs), threshold=threshold,
                   refractory_period=refractory_period, learning_rate=learning_rate,
                   weight_cap=weight_cap, leak_rate=leak_rate)
            for _ in range(n_neurons)
        ]
        
        # Set up the 1:1 inhibitory connections: I1_i -> E1_i
        # For inhibitory neurons: they have 1 input (from their corresponding E neuron when we set up feedback later)
        # But for now, we'll set their input to come from somewhere - we'll handle connections in the network
        
        # For excitatory neurons: input[0] = from inhibitory neuron, input[1] = external/input from L2
        for i in range(n_neurons):
            # E1_i receives from I1_i (inhibitory) and will receive from L2 later
            # We'll set the weights when we establish connections
            pass
            
        # Storage for monitoring
        self.history = {
            'E_potentials': [], 'E_spiked': [],
            'I_potentials': [], 'I_spiked': [],
            'E_weights': [], 'I_weights': []
        }
    
    def receive_inputs(self, external_inputs, l2_to_l1_inputs=None):
        """
        Receive inputs to the layer.
        
        Args:
            external_inputs: Array of external inputs to each E neuron (length n_neurons)
            l2_to_l1_inputs: Array of inputs from L2 to inhibitory neurons (length n_neurons) 
                           representing E2_j -> I1_i connections
        """
        if len(external_inputs) != self.n_neurons:
            raise ValueError(f"external_inputs must have length {self.n_neurons}")
            
        # Process each E/I pair
        for i in range(self.n_neurons):
            # Excitatory neuron receives:
            # Input 0: from its inhibitory neuron (I1_i)
            # Input 1: external input
            
            # Inhibitory neuron state from previous time step
            # For now, we'll assume we have access to previous spike state
            # In a full network simulation, this would be handled by the network controller
            
            # For now, we'll set this up to receive external input and 
            # the inhibitory connection will be handled when we know the I neuron's state
            E_input = np.array([0.0, external_inputs[i]])  # [from_I, external]
            self.excitatory_neurons[i].receive_input(E_input)
            
            # Inhibitory neuron receives input (will be set by L2 feedback connections)
            if l2_to_l1_inputs is not None and len(l2_to_l1_inputs) == self.n_neurons:
                I_input = np.array([l2_to_l1_inputs[i]])  # From L2
                self.inhibitory_neurons[i].receive_input(I_input)
            else:
                # No L2 input, just zero
                self.inhibitory_neurons[i].receive_input(np.array([0.0]))
    
    def update_states(self):
        """Update all neurons' states (check threshold, fire, then update)."""
        # Record states before update
        E_potentials = [n.potential for n in self.excitatory_neurons]
        E_spiked = [n.check_threshold() for n in self.excitatory_neurons]
        I_potentials = [n.potential for n in self.inhibitory_neurons]
        I_spiked = [n.check_threshold() for n in self.inhibitory_neurons]
        
        # Fire neurons that reached threshold
        for i, neuron in enumerate(self.excitatory_neurons):
            if E_spiked[i]:
                neuron.fire()
                
        for i, neuron in enumerate(self.inhibitory_neurons):
            if I_spiked[i]:
                neuron.fire()
        
        # Store history
        self.history['E_potentials'].append(E_potentials)
        self.history['E_spiked'].append(E_spiked)
        self.history['I_potentials'].append(I_potentials)
        self.history['I_spiked'].append(I_spiked)
        
        # Store weights
        self.history['E_weights'].append([n.weights.copy() for n in self.excitatory_neurons])
        self.history['I_weights'].append([n.weights.copy() for n in self.inhibitory_neurons])
        
        # Update all neurons (handle refractory, leak, etc.)
        for neuron in self.excitatory_neurons:
            neuron.update()
            
        for neuron in self.inhibitory_neurons:
            neuron.update()
    
    def get_layer_activity(self):
        """Get current activity of the layer."""
        if not self.history['E_spiked']:
            return None
            
        return {
            'E_potentials': self.history['E_potentials'][-1],
            'E_spiked': self.history['E_spiked'][-1],
            'I_potentials': self.history['I_potentials'][-1],
            'I_spiked': self.history['I_spiked'][-1]
        }


class CorticalColumn:
    """
    Cortical Column with m excitatory neurons sharing one inhibitory neuron.
    Connections: 
    - E2_j -> I2 (excitatory)
    - I2 -> E2_k (inhibitory for all k)  [lateral inhibition]
    """
    
    def __init__(self, n_neurons, threshold=1000 / UNIT, refractory_period=2,
                 learning_rate=0.05, weight_cap=1000 / UNIT, leak_rate=10 / LEAK_SCALE):
        """
        Initialize cortical column.
        
        Args:
            n_neurons: Number of excitatory neurons
            threshold: Firing threshold for neurons
            refractory_period: Refractory period in time steps
            learning_rate: Weight increase when neuron fires
            weight_cap: Maximum absolute weight value
            leak_rate: Leak rate (fraction of potential lost per ms)
        """
        self.n_neurons = n_neurons
        
        # Create excitatory neurons
        self.excitatory_neurons = [
            Neuron(n_inputs=2, threshold=threshold, refractory_period=refractory_period,
                   learning_rate=learning_rate, weight_cap=weight_cap, leak_rate=leak_rate)
            for _ in range(n_neurons)
        ]
        
        # Create single shared inhibitory neuron
        # It receives from all E neurons, and projects to all E neurons
        self.inhibitory_neuron = Neuron(
            n_inputs=n_neurons,  # Receives from all E neurons
            threshold=threshold, 
            refractory_period=refractory_period,
            learning_rate=learning_rate,
            weight_cap=weight_cap,
            leak_rate=leak_rate
        )
        
        # Storage for monitoring
        self.history = {
            'E_potentials': [], 'E_spiked': [],
            'I_potential': [], 'I_spiked': [],
            'E_weights': [], 'I_weights': []
        }
    
    def receive_inputs(self, l1_to_l2_inputs):
        """
        Receive inputs from L1 to the E neurons in this column.
        
        Args:
            l1_to_l2_inputs: 2D array [n_L1_neurons, n_L2_neurons] 
                           representing connections from L1 to L2 E neurons
        """
        # Each E neuron in L2 receives from:
        # Input 0: from the shared inhibitory neuron (I2) 
        # Input 1: aggregated input from L1
        
        # For now, we'll just set the L1 input - the inhibitory connection 
        # will be handled when we know the I neuron's state
        for j in range(self.n_neurons):
            # Sum inputs from all L1 neurons to this E2_j
            # In a real implementation, we'd have specific weights
            l1_input_sum = np.sum(l1_to_l2_inputs[:, j]) if l1_to_l2_inputs.size > 0 else 0.0
            E_input = np.array([0.0, l1_input_sum])  # [from_I2, from_L1]
            self.excitatory_neurons[j].receive_input(E_input)
    
    def receive_lateral_feedback(self, e_spiked):
        """
        Receive lateral feedback from the inhibitory neuron based on E neuron spikes.
        This implements the I2 -> E2_k connections.
        
        Args:
            e_spiked: Boolean array indicating which E neurons spiked in previous step
        """
        # The inhibitory neuron's output affects all E neurons
        # If I2 spiked, it sends inhibitory signal to all E neurons
        if np.any(e_spiked):
            # Actually, we need to know if I2 spiked, not if E spiked
            # This will be handled in the update cycle
            pass
    
    def update_states(self):
        """Update all neurons' states (check threshold, fire, then update)."""
        # Record states before update
        E_potentials = [n.potential for n in self.excitatory_neurons]
        E_spiked = [n.check_threshold() for n in self.excitatory_neurons]
        I_potential = self.inhibitory_neuron.potential
        I_spiked = self.inhibitory_neuron.check_threshold()
        
        # Fire neurons that reached threshold
        for i, neuron in enumerate(self.excitatory_neurons):
            if E_spiked[i]:
                neuron.fire()
                
        if I_spiked:
            self.inhibitory_neuron.fire()
        
        # Store history
        self.history['E_potentials'].append(E_potentials)
        self.history['E_spiked'].append(E_spiked)
        self.history['I_potential'].append(I_potential)
        self.history['I_spiked'].append(I_spiked)
        
        # Store weights
        self.history['E_weights'].append([n.weights.copy() for n in self.excitatory_neurons])
        self.history['I_weights'].append(self.inhibitory_neuron.weights.copy())
        
        # Update all neurons
        for neuron in self.excitatory_neurons:
            neuron.update()
            
        self.inhibitory_neuron.update()
    
    def get_column_activity(self):
        """Get current activity of the column."""
        if not self.history['E_spiked']:
            return None
            
        return {
            'E_potentials': self.history['E_potentials'][-1],
            'E_spiked': self.history['E_spiked'][-1],
            'I_potential': self.history['I_potential'][-1],
            'I_spiked': self.history['I_spiked'][-1]
        }


def test_layers():
    """Test the layer implementations."""
    print("Testing InputLayer and CorticalColumn...")
    
    # Create layers
    input_layer = InputLayer(n_neurons=3, threshold=1.0, learning_rate=0.05)
    cortical_column = CorticalColumn(n_neurons=2, threshold=1.0, learning_rate=0.05)
    
    print(f"Created InputLayer with {input_layer.n_neurons} E/I pairs")
    print(f"Created CorticalColumn with {cortical_column.n_neurons} E neurons + 1 I neuron")
    
    # Test InputLayer
    print("\n--- Testing InputLayer ---")
    external_inputs = np.array([0.5, 0.0, 0.3])
    l2_to_l1_inputs = np.array([0.0, 0.0, 0.0])  # No feedback initially
    
    input_layer.receive_inputs(external_inputs, l2_to_l1_inputs)
    input_layer.update_states()
    
    activity = input_layer.get_layer_activity()
    if activity:
        print(f"E potentials: {activity['E_potentials']}")
        print(f"E spiked: {activity['E_spiked']}")
        print(f"I potentials: {activity['I_potentials']}")
        print(f"I spiked: {activity['I_spiked']}")
    
    # Test CorticalColumn
    print("\n--- Testing CorticalColumn ---")
    # Simulate L1 -> L2 connections: 3 L1 neurons to 2 L2 neurons
    l1_to_l2_weights = np.array([
        [0.2, 0.1],  # L1_0 -> L2
        [0.1, 0.3],  # L1_1 -> L2  
        [0.3, 0.2]   # L1_2 -> L2
    ])
    
    cortical_column.receive_inputs(l1_to_l2_weights)
    cortical_column.update_states()
    
    activity = cortical_column.get_column_activity()
    if activity:
        print(f"E potentials: {activity['E_potentials']}")
        print(f"E spiked: {activity['E_spiked']}")
        print(f"I potential: {activity['I_potential']}")
        print(f"I spiked: {activity['I_spiked']}")
    
    print("\nLayer tests completed!")


if __name__ == "__main__":
    test_layers()
