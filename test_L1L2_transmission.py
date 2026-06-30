#!/usr/bin/env python3
"""
Test L1->L2 signal transmission
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layers import InputLayer
from cortical_column_flexible import CorticalColumn

def test_L1_to_L2_transmission():
    """Test that L1 firing properly transmits to L2"""
    print("=== Testing L1->L2 Transmission ===")
    
    # Network setup
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
    
    # Set up L1 E neurons: [-0.08, 0.9] 
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
    cortical_column.set_feedforward_weights(0.22)  # L1->L2
    cortical_column.set_lateral_inhibition_weights(-0.28)
    
    # Weights for helper functions (matching what's set above)
    W_feedforward = np.full((n_L1_E, n_L2_E), 0.22)
    W_L2_to_L1_I = np.zeros((n_L2_E, n_L1_E))  # No feedback for this test
    
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
        return np.dot(L1_E_spiked, W_feedforward)
    
    # Test input: top row pattern [1,1,1,0,0,0,0,0,0]
    pattern = np.array([1,1,1,0,0,0,0,0,0], dtype=float)
    external_input = compute_L1_external_inputs(pattern)
    print(f"Input pattern (top row): {external_input}")
    
    # Simulate what happens when L1 neurons fire
    # Manually set L1 E neurons to fired state (as they should be)
    L1_E_spiked = np.array([True, True, True, False, False, False, False, False, False])
    L1_I_spiked = np.array([False, False, False, False, False, False, False, False, False])  # No I spikes initially
    
    print(f"L1 E spiked: {L1_E_spiked}")
    print(f"L1 I spiked: {L1_I_spiked}")
    
    # Compute L1->L2 input
    L1_to_L2_input_vec = compute_L1_to_L2_input(L1_E_spiked)
    print(f"L1->L2 input vector: {L1_to_L2_input_vec}")
    
    # Expected: 3 * 0.22 = 0.66 for each L2 neuron
    expected = 3 * 0.22
    print(f"Expected input per L2 neuron: {expected}")
    print(f"Match: {np.allclose(L1_to_L2_input_vec, expected)}")
    
    # Now test what happens in L2 when it receives this input
    print("\n--- Testing L2 Response ---")
    
    # Reset L2 neurons
    for neuron in cortical_column.excitatory_neurons:
        neuron.potential = 0.0
        neuron.refractory_timer = 0
        neuron.spiked = False
    cortical_column.inhibitory_neuron.potential = 0.0
    cortical_column.inhibitory_neuron.refractory_timer = 0
    cortical_column.inhibitory_neuron.spiked = False
    
    # Previous states (for time step 0)
    L2_I_spiked_prev = False
    L2_E_spiked_prev = np.zeros(n_L2_E, dtype=int)  # No previous L2 spikes
    
    # Give L2 the input from L1
    print("Applying L1->L2 input to L2 E neurons...")
    for j in range(n_L2_E):
        # E neuron j: [from_local_I, from_below]
        inhib_input = 1.0 if L2_I_spiked_prev else 0.0  # from L2 I (none initially)
        excit_input = L1_to_L2_input_vec[j]  # from L1 E
        
        # Convert to spikes for the two inputs:
        spike0 = inhib_input  # 0 or 1
        spike1 = 1.0 if excit_input > 0 else 0.0  # treat positive input as spike
        
        print(f"L2 E{j}: inhib_input={inhib_input}, excit_input={excit_input:.3f} -> spikes=[{spike0}, {spike1}]")
        cortical_column.excitatory_neurons[j].receive_input(np.array([spike0, spike1]))
    
    # L2 I neuron input: from all L2 E neurons (lateral inhibition) - none initially
    cortical_column.inhibitory_neuron.receive_input(L2_E_spiked_prev.astype(float))
    print(f"L2 I input: {L2_E_spiked_prev.astype(float)} (from L2 E spikes)")
    
    # Check thresholds
    L2_E_spiked = [n.check_threshold() for n in cortical_column.excitatory_neurons]
    L2_I_spiked = cortical_column.inhibitory_neuron.check_threshold()
    
    print(f"L2 E check_threshold(): {L2_E_spiked}")
    print(f"L2 I check_threshold(): {L2_I_spiked}")
    
    # Check potentials
    L2_potentials = [n.potential for n in cortical_column.excitatory_neurons]
    L2_I_potential = cortical_column.inhibitory_neuron.potential
    print(f"L2 E potentials: {[f'{p:.3f}' for p in L2_potentials]}")
    print(f"L2 I potential: {L2_I_potential:.3f}")
    
    # Fire if threshold reached
    print("\n--- Firing ---")
    for j, neuron in enumerate(cortical_column.excitatory_neurons):
        if L2_E_spiked[j]:
            neuron.fire()
            print(f"Fired L2 E{j}")
    if L2_I_spiked:
        cortical_column.inhibitory_neuron.fire()
        print("Fired L2 I")
    
    # Update states
    for neuron in cortical_column.excitatory_neurons:
        neuron.update()
    cortical_column.inhibitory_neuron.update()
    
    print("\n--- After Update ---")
    L2_potentials_after = [n.potential for n in cortical_column.excitatory_neurons]
    L2_I_potential_after = cortical_column.inhibitory_neuron.potential
    L2_E_spiked_after = [n.spiked for n in cortical_column.excitatory_neurons]  # Should be False after update
    L2_I_spiked_after = cortical_column.inhibitory_neuron.spiked
    
    print(f"L2 E potentials after update: {[f'{p:.3f}' for p in L2_potentials_after]}")
    print(f"L2 I potential after update: {L2_I_potential_after:.3f}")
    print(f"L2 E spiked after update: {L2_E_spiked_after}")
    print(f"L2 I spiked after update: {L2_I_spiked_after}")

if __name__ == "__main__":
    test_L1_to_L2_transmission()