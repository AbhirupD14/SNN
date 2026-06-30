#!/usr/bin/env python3
"""
Debug the exact first time step of L1 processing
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layers import InputLayer
from cortical_column_flexible import CorticalColumn

def debug_first_timestep():
    """Debug exactly what happens in the first time step"""
    print("=== Debugging First Time Step ===")
    
    # Same setup as network test
    pattern = np.array([1,1,1,0,0,0,0,0,0], dtype=float)  # row0
    
    n_L1_E = 9
    n_L2_E = 8
    threshold = 0.15
    learning_rate = 0.12
    weight_cap = 1.0
    leak_rate = 0.008
    
    # Initialize network
    input_layer = InputLayer(n_neurons=n_L1_E, threshold=threshold,
                             refractory_period=2,
                             learning_rate=learning_rate,
                             weight_cap=weight_cap,
                             leak_rate=leak_rate)
    
    # Set up L1 E neurons
    for i in range(n_L1_E):
        input_layer.excitatory_neurons[i].weights = np.array([-0.08, 0.9])
        input_layer.inhibitory_neurons[i].weights = np.array([0.0])
    
    # Cortical Column
    cortical_column = CorticalColumn(n_neurons=n_L2_E, threshold=threshold,
                                     refractory_period=2,
                                     learning_rate=learning_rate,
                                     weight_cap=weight_cap,
                                     leak_rate=leak_rate)
    cortical_column.setup_connectivity(n_feedback_inputs=0)
    cortical_column.finalize_connections()
    cortical_column.set_local_inhibition_weights(-0.25)
    cortical_column.set_feedforward_weights(0.22)
    cortical_column.set_lateral_inhibition_weights(-0.28)
    
    W_L2_to_L1_I = np.random.uniform(0.15, 0.35, size=(n_L2_E, n_L1_E))
    
    # Helper functions
    def compute_L1_external_inputs(pv):
        return pv
    
    def compute_L2_to_L1_I_input(L2_E_spiked):
        if np.sum(L2_E_spiked) == 0:
            return np.zeros(n_L1_E)
        return np.dot(L2_E_spiked, W_L2_to_L1_I)
    
    def compute_L1_to_L2_input(L1_E_spiked):
        if np.sum(L1_E_spiked) == 0:
            return np.zeros(n_L2_E)
        return np.dot(L1_E_spiked, np.full((n_L1_E, n_L2_E), 0.22))
    
    # Simulate FIRST TIME STEP only
    print("--- Initial State ---")
    for i in range(3):  # Show first 3 neurons
        e_neuron = input_layer.excitatory_neurons[i]
        i_neuron = input_layer.inhibitory_neurons[i]
        print(f"Neuron {i}: E_weights={e_neuron.weights}, I_weights={i_neuron.weights}")
        print(f"  E: potential={e_neuron.potential}, refractory={e_neuron.refractory_timer}, spiked={e_neuron.spiked}")
        print(f"  I: potential={i_neuron.potential}, refractory={i_neuron.refractory_timer}, spiked={i_neuron.spiked}")
    
    external_input = compute_L1_external_inputs(pattern)
    print(f"\nExternal input: {external_input}")
    
    # Time step 0 - no previous spikes
    L1_E_spiked_prev = np.zeros(n_L1_E, dtype=int)
    L1_I_spiked_prev = np.zeros(n_L1_E, dtype=int)
    L2_E_spiked_prev = np.zeros(n_L2_E, dtype=int)
    L2_I_spiked_prev = False
    
    print(f"Previous spikes: L1_E={L1_E_spiked_prev}, L1_I={L1_I_spiked_prev}, L2_E={L2_E_spiked_prev}, L2_I={L2_I_spiked_prev}")
    
    # L2 to L1 feedback
    L2_to_L1_I_input = compute_L2_to_L1_I_input(L2_E_spiked_prev)
    print(f"L2 to L1 I input: {L2_to_L1_I_input}")
    
    # Process each neuron
    print("\n--- Processing Neurons ---")
    for i in range(3):  # First 3 neurons
        print(f"\nNeuron {i}:")
        # E neuron: [from_I, external]
        spike0 = 1.0 if L1_I_spiked_prev[i] else 0.0
        spike1 = 1.0 if external_input[i] > 0.5 else 0.0
        print(f"  E neuron inputs: [{spike0}, {spike1}] (from_I={L1_I_spiked_prev[i]}, external={external_input[i]})")
        input_layer.excitatory_neurons[i].receive_input(np.array([spike0, spike1]))
        
        # I neuron input: from L2 E neurons (feedback)
        feedback_val = L2_to_L1_I_input[i]
        spike_I = 1.0 if feedback_val > 0 else 0.0
        print(f"  I neuron input: [{spike_I}] (feedback_val={feedback_val})")
        input_layer.inhibitory_neurons[i].receive_input(np.array([spike_I]))
        
        # Check state after receive_input
        e_neuron = input_layer.excitatory_neurons[i]
        i_neuron = input_layer.inhibitory_neurons[i]
        print(f"  After receive_input - E: potential={e_neuron.potential}, I: potential={i_neuron.potential}")
        
        # Check threshold
        e_should_fire = e_neuron.check_threshold()
        i_should_fire = i_neuron.check_threshold()
        print(f"  Check threshold - E: {e_should_fire} (potential {e_neuron.potential} >= {threshold}), I: {i_should_fire} (potential {i_neuron.potential} >= {threshold})")
    
    # Now fire and update like the network does
    print("\n--- Firing and Updating ---")
    L1_E_spiked = [n.check_threshold() for n in input_layer.excitatory_neurons]
    L1_I_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
    
    print(f"L1 E spiked: {L1_E_spiked[:3]}...")  # First 3
    print(f"L1 I spiked: {L1_I_spiked[:3]}...")
    
    # Fire neurons
    for i, neuron in enumerate(input_layer.excitatory_neurons):
        if L1_E_spiked[i]:
            neuron.fire()
            print(f"  Fired E neuron {i}")
    for i, neuron in enumerate(input_layer.inhibitory_neurons):
        if L1_I_spiked[i]:
            neuron.fire()
            print(f"  Fired I neuron {i}")
    
    print("After firing:")
    for i in range(3):
        e_neuron = input_layer.excitatory_neurons[i]
        i_neuron = input_layer.inhibitory_neurons[i]
        print(f"  Neuron {i}: E: potential={e_neuron.potential}, spiked={e_neuron.spiked}, refractory={e_neuron.refractory_timer}")
        print(f"           I: potential={i_neuron.potential}, spiked={i_neuron.spiked}, refractory={i_neuron.refractory_timer}")
    
    # Update
    for neuron in input_layer.excitatory_neurons:
        neuron.update()
    for neuron in input_layer.inhibitory_neurons:
        neuron.update()
    
    print("\nAfter update:")
    for i in range(3):
        e_neuron = input_layer.excitatory_neurons[i]
        i_neuron = input_layer.inhibitory_neurons[i]
        print(f"  Neuron {i}: E: potential={e_neuron.potential}, spiked={e_neuron.spiked}, refractory={e_neuron.refractory_timer}")
        print(f"           I: potential={i_neuron.potential}, spiked={i_neuron.spiked}, refractory={i_neuron.refractory_timer}")

if __name__ == "__main__":
    debug_first_timestep()