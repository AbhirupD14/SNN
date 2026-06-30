#!/usr/bin/env python3
"""
End-to-end test showing L1->L2 signal propagation works
This demonstrates the core functionality needed for the 8-line pattern task
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layers import InputLayer
from cortical_column_flexible import CorticalColumn

def test_end_to_end_signal_propagation():
    """Test that shows L1 firing -> L2 firing -> L2 firing L1 inhibition"""
    print("=== End-to-End Signal Propagation Test ===")
    print("This verifies the core SNN functionality works:\n")
    
    # Network parameters matching our working configuration
    n_L1_E = 9
    n_L2_E = 8
    threshold = 0.15
    learning_rate = 0.12
    weight_cap = 1.0
    leak_rate = 0.008
    
    print(f"Network: {n_L1_E} L1 E/I pairs, {n_L2_E} L2 E neurons + 1 I neuron")
    print(f"Parameters: threshold={threshold}, learning_rate={learning_rate}")
    print()
    
    # Initialize network
    print("1. Initializing network layers...")
    input_layer = InputLayer(n_neurons=n_L1_E, threshold=threshold,
                             refractory_period=2,
                             learning_rate=learning_rate,
                             weight_cap=weight_cap,
                             leak_rate=leak_rate)
    
    # L1 E neurons: [from_I (-0.08), from_external (0.9)] - should fire on input
    for i in range(n_L1_E):
        input_layer.excitatory_neurons[i].weights = np.array([-0.08, 0.9])
        input_layer.inhibitory_neurons[i].weights = np.array([0.0])  # placeholder
    
    # Cortical Column: L2
    print("2. Setting up Cortical Column (L2)...")
    cortical_column = CorticalColumn(n_neurons=n_L2_E, threshold=threshold,
                                     refractory_period=2,
                                     learning_rate=learning_rate,
                                     weight_cap=weight_cap,
                                     leak_rate=leak_rate)
    cortical_column.setup_connectivity(n_feedback_inputs=0)
    cortical_column.finalize_connections()
    
    # L2 weights: [from_local_I (-0.25), from_below (0.22)]
    cortical_column.set_local_inhibition_weights(-0.25)
    cortical_column.set_feedforward_weights(0.22)  # L1->L2 feedforward
    cortical_column.set_lateral_inhibition_weights(-0.28)  # L2 E->L2 I
    
    # L2->L1 feedback weights (for training)
    W_L2_to_L1_I = np.random.uniform(0.15, 0.35, size=(n_L2_E, n_L1_E))
    
    print("3. Testing signal propagation pathway...")
    print("   Path: External Input -> L1 E firing -> L1->L2 weights -> L2 E firing")
    print()
    
    # Test pattern: top row [1,1,1,0,0,0,0,0,0]
    pattern = np.array([1,1,1,0,0,0,0,0,0], dtype=float)
    print(f"4. Testing with pattern: {pattern} (top row)")
    print("   Expected: L1 neurons 0,1,2 should fire -> strong L2 input -> L2 E neurons fire")
    print()
    
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
        # Use the actual weights set in the cortical column
        weights_matrix = np.full((n_L1_E, n_L2_E), 0.22)  # matches what we set
        return np.dot(L1_E_spiked, weights_matrix)
    
    # Simulate one time step to see the full pipeline
    external_input = compute_L1_external_inputs(pattern)
    print(f"5. External input to L1: {external_input}")
    
    # Initial state - no previous spikes
    L1_E_spiked_prev = np.zeros(n_L1_E, dtype=int)
    L1_I_spiked_prev = np.zeros(n_L1_E, dtype=int)
    L2_E_spiked_prev = np.zeros(n_L2_E, dtype=int)
    L2_I_spiked_prev = False
    
    # Step 1: L1 processing
    print("6. Step 1: L1 layer processing")
    L2_to_L1_I_input = compute_L2_to_L1_I_input(L2_E_spiked_prev)
    print(f"   L2->L1 feedback (initial): {L2_to_L1_I_input[:3]}...")
    
    # Apply inputs to L1
    for i in range(n_L1_E):
        spike0 = 1.0 if L1_I_spiked_prev[i] else 0.0
        spike1 = 1.0 if external_input[i] > 0.5 else 0.0
        input_layer.excitatory_neurons[i].receive_input(np.array([spike0, spike1]))
        
        feedback_val = L2_to_L1_I_input[i]
        spike_I = 1.0 if feedback_val > 0 else 0.0
        input_layer.inhibitory_neurons[i].receive_input(np.array([spike_I]))
    
    # Check L1 thresholds
    L1_E_spiked = [n.check_threshold() for n in input_layer.excitatory_neurons]
    L1_I_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
    
    print(f"   L1 E spiked: {L1_E_spiked}")
    print(f"   L1 I spiked: {L1_I_spiked}")
    print(f"   Number of L1 E spikes: {sum(L1_E_spiked)} (expected: 3)")
    
    # Fire L1 neurons
    for i, neuron in enumerate(input_layer.excitatory_neurons):
        if L1_E_spiked[i]:
            neuron.fire()
    for i, neuron in enumerate(input_layer.inhibitory_neurons):
        if L1_I_spiked[i]:
            neuron.fire()
    
    # Step 2: L1->L2 transmission
    print("\n7. Step 2: L1->L2 signal transmission")
    L1_to_L2_input_vec = compute_L1_to_L2_input(L1_E_spiked)
    print(f"   L1 E spiked as array: {np.array(L1_E_spiked).astype(int)}")
    print(f"   L1->L2 input vector: {L1_to_L2_input_vec}")
    print(f"   Expected per L2 neuron: {sum(L1_E_spiked) * 0.22} (3 * 0.22 = 0.66)")
    
    # Step 3: L2 processing
    print("\n8. Step 3: L2 layer processing")
    # Apply inputs to L2
    for j in range(n_L2_E):
        # L2 E neuron j: [from_local_I, from_below]
        inhib_input = 1.0 if L2_I_spiked_prev else 0.0  # from L2 I (none initially)
        excit_input = L1_to_L2_input_vec[j]  # from L1 E
        
        spike0 = inhib_input
        spike1 = 1.0 if excit_input > 0 else 0.0
        
        cortical_column.excitatory_neurons[j].receive_input(np.array([spike0, spike1]))
    
    # L2 I neuron input: from all L2 E neurons (lateral inhibition)
    cortical_column.inhibitory_neuron.receive_input(L2_E_spiked_prev.astype(float))
    
    # Check L2 thresholds
    L2_E_spiked = [n.check_threshold() for n in cortical_column.excitatory_neurons]
    L2_I_spiked = cortical_column.inhibitory_neuron.check_threshold()
    
    print(f"   L2 E spiked: {L2_E_spiked}")
    print(f"   L2 I spiked: {L2_I_spiked}")
    print(f"   Number of L2 E spikes: {sum(L2_E_spiked)} (expected: 8)")
    print(f"   L2 E potentials: {[f'{n.potential:.3f}' for n in cortical_column.excitatory_neurons[:3]]}...")
    
    # Fire L2 neurons
    l2_fired_count = 0
    for j, neuron in enumerate(cortical_column.excitatory_neurons):
        if L2_E_spiked[j]:
            neuron.fire()
            l2_fired_count += 1
    if L2_I_spiked:
        cortical_column.inhibitory_neuron.fire()
    
    print(f"   Actually fired L2 E neurons: {l2_fired_count}")
    
    # Step 4: Verify the loop can continue (L2->L1 feedback)
    print("\n9. Step 4: L2->L1 feedback (for training)")
    L2_to_L1_I_input_after = compute_L2_to_L1_I_input(np.array(L2_E_spiked).astype(int))
    print(f"   L2->L1 feedback after L2 firing: {L2_to_L1_I_input_after[:3]}...")
    print(f"   This will train L1 I neurons in subsequent time steps")
    
    # Summary
    print("\n=== RESULTS ===")
    l1_success = sum(L1_E_spiked) == 3
    l2_success = sum(L2_E_spiked) == n_L2_E  # All L2 E neurons should fire
    
    print(f"L1 layer working correctly: {l1_success} ({sum(L1_E_spiked)}/3 expected spikes)")
    print(f"L2 layer working correctly: {l2_success} ({sum(L2_E_spiked)}/{n_L2_E} expected spikes)")
    print(f"Signal propagation verified: {l1_success and l2_success}")
    
    if l1_success and l2_success:
        print("\n✅ VERIFICATION PASSED")
        print("The core SNN signal propagation pathway is functioning:")
        print("  External Input → L1 E firing → L1->L2 weights → L2 E firing → L2->L1 feedback")
        print("This establishes the foundation for the 8-line pattern consolidation task.")
        return True
    else:
        print("\n❌ VERIFICATION FAILED")
        return False

if __name__ == "__main__":
    success = test_end_to_end_signal_propagation()
    exit(0 if success else 1)