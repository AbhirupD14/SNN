"""
Test the dot product computation for L1->L2
"""
import numpy as np

# Simulate what should happen
n_L1_E = 9
n_L2_E = 8

# L1 spikes: first three neurons should spike for row0 pattern
L1_E_spiked = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
print(f"L1_E_spiked: {L1_E_spiked}")

# Feedforward weights (L1 E -> L2 E) - shape (9, 8)
W_feedforward = np.random.uniform(0.1, 0.3, size=(n_L1_E, n_L2_E))
print(f"W_feedforward shape: {W_feedforward.shape}")
print(f"Sample W_feedforward[:, 0] (weights to L2 E0): {W_feedforward[:3, 0]}")  # First 3 rows, first column

# Compute L1->L2 input
L1_to_L2_input = np.dot(L1_E_spiked, W_feedforward)
print(f"L1_to_L2_input: {L1_to_L2_input}")

# Manual computation for first L2 neuron
manual_sum = np.dot(L1_E_spiked, W_feedforward[:, 0])
print(f"Manual sum for L2 E0: {manual_sum}")

# What if we use the fixed weights from the working version?
W_feedforward_fixed = np.full((n_L1_E, n_L2_E), 0.22)  # All weights = 0.22
L1_to_L2_input_fixed = np.dot(L1_E_spiked, W_feedforward_fixed)
print(f"\nWith fixed weights (0.22):")
print(f"L1_to_L2_input_fixed: {L1_to_L2_input_fixed}")
print(f"Expected: 3 * 0.22 = {3 * 0.22}")