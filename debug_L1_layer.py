"""
Debug just the L1 layer to see why neurons aren't firing
"""
import numpy as np
from layers import InputLayer

# Test the InputLayer directly
print("=== Testing InputLayer directly ===")
input_layer = InputLayer(n_neurons=9, threshold=0.15, refractory_period=2, 
                        learning_rate=0.12, weight_cap=1.0, leak_rate=0.008)

# Set up the L1 E neurons to have [-0.08, 0.9] weights (should fire on external input)
for i in range(9):
    input_layer.excitatory_neurons[i].weights = np.array([-0.08, 0.9])  # [I_weight, external_weight]
    input_layer.inhibitory_neurons[i].weights = np.array([0.0])  # placeholder for L2 feedback

print("L1 E neuron weights set to [-0.08, 0.9]")
print("Threshold: 0.15")
print()

# Test pattern row0: [1,1,1,0,0,0,0,0,0]
external_input = np.array([1,1,1,0,0,0,0,0,0], dtype=float)
l2_to_l1_input = np.zeros(9)  # No L2 feedback initially

print(f"External input: {external_input}")
print(f"L2 to L1 input: {l2_to_l1_input}")
print()

# Call receive_inputs
input_layer.receive_inputs(external_input, l2_to_l1_input)

# Check states before update
print("States BEFORE update:")
E_potentials = [n.potential for n in input_layer.excitatory_neurons]
E_spiked_check = [n.check_threshold() for n in input_layer.excitatory_neurons]
I_potentials = [n.potential for n in input_layer.inhibitory_neurons]
I_spiked_check = [n.check_threshold() for n in input_layer.inhibitory_neurons]

print(f"E potentials: {E_potentials}")
print(f"E check_threshold(): {E_spiked_check}")
print(f"I potentials: {I_potentials}")
print(f"I check_threshold(): {I_spiked_check}")
print()

# Update states
input_layer.update_states()

# Check states after update
print("States AFTER update:")
activity = input_layer.get_layer_activity()
if activity:
    print(f"E potentials: {activity['E_potentials']}")
    print(f"E spiked: {activity['E_spiked']}")
    print(f"I potentials: {activity['I_potentials']}")
    print(f"I spiked: {activity['I_spiked']}")
print()

# Let's also manually test what SHOULD happen
print("=== Manual calculation for first neuron (should fire) ===")
w_I, w_ext = -0.08, 0.9
I_spike, ext_spike = 0.0, 1.0  # No inhibitory spike, external spike present
potential_change = w_I * I_spike + w_ext * ext_spike
print(f"Potential change: {w_I} * {I_spike} + {w_ext} * {ext_spike} = {potential_change}")
print(f"This should exceed threshold 0.15: {potential_change >= 0.15}")