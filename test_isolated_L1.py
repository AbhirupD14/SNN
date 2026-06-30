#!/usr/bin/env python3
"""
Simple test to isolate the L1 firing issue
"""
import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layers import InputLayer

def test_isolated_L1_layer():
    """Test the L1 layer in isolation with simple inputs"""
    print("=== Isolated L1 Layer Test ===")
    
    # Create layer with 3 neurons for simplicity
    input_layer = InputLayer(n_neurons=3, threshold=0.15, refractory_period=2,
                            learning_rate=0.12, weight_cap=1.0, leak_rate=0.008)
    
    # Set up L1 E neurons: [from_I, from_external]
    # Weights: inhibitory weight = -0.08, external weight = 0.9
    for i in range(3):
        input_layer.excitatory_neurons[i].weights = np.array([-0.08, 0.9])
        input_layer.inhibitory_neurons[i].weights = np.array([0.0])  # No L2 feedback yet
    
    print("Neuron weights set to [-0.08, 0.9] for all E neurons")
    print("Threshold: 0.15")
    print()
    
    # Test input: [1, 1, 1] (all pixels on)
    external_input = np.array([1.0, 1.0, 1.0])
    l2_to_l1_input = np.array([0.0, 0.0, 0.0])  # No L2 feedback
    
    print(f"External input: {external_input}")
    print(f"L2 to L1 input: {l2_to_l1_input}")
    print()
    
    # Run for 5 time steps
    for step in range(5):
        print(f"--- Time Step {step} ---")
        
        # Record state before processing
        E_potentials_before = [n.potential for n in input_layer.excitatory_neurons]
        E_refractory_before = [n.refractory_timer for n in input_layer.excitatory_neurons]
        I_potentials_before = [n.potential for n in input_layer.inhibitory_neurons]
        I_refractory_before = [n.refractory_timer for n in input_layer.inhibitory_neurons]
        
        print(f"Before receive_inputs:")
        print(f"  E potentials: {E_potentials_before}")
        print(f"  E refractory: {E_refractory_before}")
        print(f"  I potentials: {I_potentials_before}")
        print(f"  I refractory: {I_refractory_before}")
        
        # Process inputs
        input_layer.receive_inputs(external_input, l2_to_l1_input)
        
        # Check thresholds
        E_spiked_check = [n.check_threshold() for n in input_layer.excitatory_neurons]
        I_spiked_check = [n.check_threshold() for n in input_layer.inhibitory_neurons]
        
        print(f"After receive_inputs, before fire/update:")
        print(f"  E check_threshold(): {E_spiked_check}")
        print(f"  I check_threshold(): {I_spiked_check}")
        
        # Fire and update
        input_layer.update_states()
        
        # Record state after processing
        activity = input_layer.get_layer_activity()
        if activity:
            print(f"After update_states:")
            print(f"  E potentials: {activity['E_potentials']}")
            print(f"  E spiked: {activity['E_spiked']}")
            print(f"  I potentials: {activity['I_potentials']}")
            print(f"  I spiked: {activity['I_spiked']}")
        print()

if __name__ == "__main__":
    test_isolated_L1_layer()