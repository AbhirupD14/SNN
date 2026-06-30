"""
Demo: Line pattern recognition with 2-layer network
3x3 grid, 8 line patterns (3 rows, 3 columns, 2 diagonals)
Input layer: 9 E/I pairs (18 neurons)
Cortical column: 8 E neurons + 1 I neuron
"""
import numpy as np
from layers import InputLayer
from cortical_column_flexible import CorticalColumn

# Define the 8 patterns as 9-element vectors (row-major order 3x3)
patterns = {
    'row0':    [1,1,1, 0,0,0, 0,0,0],
    'row1':    [0,0,0, 1,1,1, 0,0,0],
    'row2':    [0,0,0, 0,0,0, 1,1,1],
    'col0':    [1,0,0, 1,0,0, 1,0,0],
    'col1':    [0,1,0, 0,1,0, 0,1,0],
    'col2':    [0,0,1, 0,0,1, 0,0,1],
    'diag0':   [1,0,0, 0,1,0, 0,0,1],  # top-left to bottom-right
    'diag1':   [0,0,1, 0,1,0, 1,0,0],  # top-right to bottom-left
}
pattern_names = list(patterns.keys())
pattern_vectors = [np.array(patterns[name], dtype=float) for name in pattern_names]
n_patterns = len(pattern_vectors)

# Network parameters
n_L1_E = 9  # number of excitatory neurons in input layer (matches pixels)
n_L2_E = 8  # number of excitatory neurons in cortical column (matches patterns)
threshold = 0.5
refractory_period = 2
learning_rate = 0.05
weight_cap = 1.0
leak_rate = 0.02

print("Building 2-layer network for line pattern recognition...")
print(f"Input layer: {n_L1_E} E/I pairs ({n_L1_E*2} total neurons)")
print(f"Cortical column: {n_L2_E} E neurons + 1 I neuron ({n_L2_E+1} total neurons)")
print(f"Patterns to learn: {n_patterns} (3 rows, 3 columns, 2 diagonals)")
print()

# Initialize network
input_layer = InputLayer(n_neurons=n_L1_E, threshold=threshold,
                         refractory_period=refractory_period,
                         learning_rate=learning_rate,
                         weight_cap=weight_cap,
                         leak_rate=leak_rate)
# Set up InputLayer internal connections:
# Each E neuron: [from_I (inhibitory), from_external (pixel input)]
# Each I neuron: [from_L2_E_feedback] (placeholder)
for i in range(n_L1_E):
    input_layer.excitatory_neurons[i].weights = np.array([-0.5, 0.0])  # [I_weight, external_weight]
    input_layer.inhibitory_neurons[i].weights = np.array([0.0])  # placeholder

# Cortical Column: 8 E neurons + 1 I neuron
cortical_column = CorticalColumn(n_neurons=n_L2_E, threshold=threshold,
                                 refractory_period=refractory_period,
                                 learning_rate=learning_rate,
                                 weight_cap=weight_cap,
                                 leak_rate=leak_rate)
# Setup connectivity for L2:
# Each E neuron: [from_local_I, from_below (aggregated L1 E input)]
# Inhibitory neuron: [from_all_local_E (lateral inhibition), from_feedback (none for 2-layer)]
cortical_column.setup_connectivity(n_feedback_inputs=0)
cortical_column.finalize_connections()

# Set up L2 internal weights:
# Local inhibition: I -> E (negative)
cortical_column.set_local_inhibition_weights(-0.5)
# Feedforward weights from L1 E to L2 E: initialize small random
W_feedforward = np.random.uniform(0.1, 0.3, size=(n_L1_E, n_L2_E))
# Lateral inhibition: E -> I within L2 (negative)
cortical_column.set_lateral_inhibition_weights(-0.4)

# Set up L1 I neuron feedback weights from L2 E
# Each L1 I neuron receives from all L2 E neurons
W_L2_to_L1_I = np.random.uniform(0.1, 0.3, size=(n_L2_E, n_L1_E))

print("Network initialized with random weights.")
print("Sample feedforward weights (L1 E0 -> L2 E0..2):")
for j in range(min(3, n_L2_E)):
    print(f"  to L2 E{j}: {W_feedforward[0,j]:.3f}")
print()

# Helper functions
def compute_L1_external_inputs(pattern_vector):
    return pattern_vector

def compute_L2_to_L1_I_input(L2_E_spiked):
    if np.sum(L2_E_spiked) == 0:
        return np.zeros(n_L1_E)
    return np.dot(L2_E_spiked, W_L2_to_L1_I)

def compute_L1_to_L2_input(L1_E_spiked):
    if np.sum(L1_E_spiked) == 0:
        return np.zeros(n_L2_E)
    return np.dot(L1_E_spiked, W_feedforward)

print("Network is ready for training.")
print("To run the full training simulation, execute the demo_line_patterns.py script.")
print("That script implements the full learning loop with pattern consolidation detection.")