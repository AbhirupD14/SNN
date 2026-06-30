"""
8 Line Pattern Recognition with Consolidation - Fixed Version
3x3 grid, 8 line patterns (3 rows, 3 columns, 2 diagonals)
Input layer: 9 E/I pairs (18 neurons total)
Cortical column: 8 E neurons + 1 I neuron (9 neurons total)

Goal: Present each pattern until one L2 E neuron consolidates as the winner for that pattern.
L1 E neurons start trained (fire when presented input).
All other neurons (L1 I, L2 E, L2 I) are trained during simulation.
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

# Network parameters - adjusted for better response
n_L1_E = 9   # number of excitatory neurons in input layer (matches pixels)
n_L2_E = 8   # number of excitatory neurons in cortical column (matches patterns)
threshold = 0.15       # Lower threshold for easier firing
refractory_period = 2
learning_rate = 0.12   # Good learning rate
weight_cap = 1.0
leak_rate = 0.008

# Initialize weight matrices that will be accessible to helper functions
W_feedforward = np.random.uniform(0.1, 0.3, size=(n_L1_E, n_L2_E))
W_L2_to_L1_I = np.random.uniform(0.15, 0.35, size=(n_L2_E, n_L1_E))

print("Building 2-layer network for line pattern consolidation...")
print(f"Input layer: {n_L1_E} E/I pairs ({n_L1_E*2} total neurons)")
print(f"Cortical column: {n_L2_E} E neurons + 1 I neuron ({n_L2_E+1} total neurons)")
print(f"Patterns to learn: {n_patterns} (3 rows, 3 columns, 2 diagonals)")
print(f"Parameters: threshold={threshold}, learning_rate={learning_rate}, leak_rate={leak_rate}")
print()

# Initialize network
input_layer = InputLayer(n_neurons=n_L1_E, threshold=threshold,
                         refractory_period=refractory_period,
                         learning_rate=learning_rate,
                         weight_cap=weight_cap,
                         leak_rate=leak_rate)

# Set up InputLayer internal connections for trained L1 E neurons:
# Each E neuron: [from_I (inhibitory), from_external (pixel input)]
# We want L1 E neurons to fire when input is presented, so:
# - Inhibitory weight: small negative (-0.08) to allow external input to dominate
# - External weight: positive (0.9) so external input drives firing but leaves room for learning
for i in range(n_L1_E):
    input_layer.excitatory_neurons[i].weights = np.array([-0.08, 0.9])  # [I_weight, external_weight]
    # I neuron: single weight from L2 E feedback (will be trained)
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
# Local inhibition: I -> E (negative, moderate for competition)
cortical_column.set_local_inhibition_weights(-0.25)
# Feedforward weights from L1 E to L2 E: initialize to small positive values
# THIS IS THE KEY FIX: Actually setting the feedforward weights on the neurons
initial_feedforward_weight = 0.22
cortical_column.set_feedforward_weights(initial_feedforward_weight)
# Lateral inhibition: E -> I within L2 (negative, moderate for competition)
cortical_column.set_lateral_inhibition_weights(-0.28)
# No feedback inputs in this 2-layer network

# Note: W_feedforward and W_L2_to_L1_I are already defined above for helper functions
print("Network initialized with FIXED feedforward weight setting.")
print(f"L1 E neurons: pre-trained to fire on external input (weights: [-0.08, 0.9])")
print(f"L2 E neurons: local inhibition weights set to -0.25")
print(f"L2 E neurons: feedforward weights initialized to {initial_feedforward_weight}")
print(f"L2 I neuron: lateral inhibition weights set to -0.28")
print("L1 I, L2 E, L2 I neurons: to be trained during simulation")
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
    return np.dot(L1_E_spiked, W_feedforward)  # Fixed: use W_feedforward, not W_L2_to_L1_I.T

# Debug function to check L1 and L2 activity
def debug_activity(pattern_vec, presentation_num, pattern_name):
    """Debug function to verify activity at each layer"""
    external_input = compute_L1_external_inputs(pattern_vec)
    if presentation_num == 0:  # Only show debug for first presentation
        print(f"    DEBUG Pattern {pattern_name}: External input = {external_input}")
        
        # Check L1 activity
        L1_E_spiked = [n.check_threshold() for n in input_layer.excitatory_neurons]
        L1_I_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
        print(f"    DEBUG L1 E spikes (should be [T,T,T,F,F,F,F,F,F] for row0): {L1_E_spiked}")
        
        # Check what input L2 would get
        L1_to_L2_input = compute_L1_to_L2_input(L1_E_spiked)
        print(f"    DEBUG L1->L2 input (weighted sum): {L1_to_L2_input}")
        
        # Check L2 E neuron thresholds and weights
        for j in range(min(3, n_L2_E)):  # Show first 3 L2 neurons
            weights = cortical_column.excitatory_neurons[j]._weights_array
            potential = cortical_column.excitatory_neurons[j].potential
            print(f"    DEBUG L2 E{j}: weights={weights}, potential={potential:.3f}, threshold={threshold}")

# Simulation parameters for consolidation
n_presentations_per_pattern = 20  # Good number for learning
n_time_steps_per_presentation = 25  # Reasonable simulation time
consolidation_threshold = 0.55  # Need to fire >55% of time to win

print(f"Starting consolidation training...")
print(f"Each pattern presented for {n_presentations_per_pattern} presentations × {n_time_steps_per_presentation} steps")
print(f"Consolidation threshold: {consolidation_threshold*100}% of time steps")
print("-" * 70)

# Track consolidation progress
pattern_winner_history = {name: [] for name in pattern_names}
pattern_consolidated = {name: False for name in pattern_names}
consolidation_order = []

for pres_idx in range(n_presentations_per_pattern):
    for p_idx, pattern_name in enumerate(pattern_names):
        if pattern_consolidated[pattern_name]:
            # Skip already consolidated patterns
            continue
            
        pattern_vec = pattern_vectors[p_idx]
        
        # Debug first presentation of each pattern
        if pres_idx == 0:
            debug_activity(pattern_vec, pres_idx, pattern_name)
        
        # Reset neuron states at start of each presentation
        for neuron in input_layer.excitatory_neurons:
            neuron.potential = 0.0
            neuron.refractory_timer = 0
            neuron.spiked = False
        for neuron in input_layer.inhibitory_neurons:
            neuron.potential = 0.0
            neuron.refractory_timer = 0
            neuron.spiked = False
        for neuron in cortical_column.excitatory_neurons:
            neuron.potential = 0.0
            neuron.refractory_timer = 0
            neuron.spiked = False
        cortical_column.inhibitory_neuron.potential = 0.0
        cortical_column.inhibitory_neuron.refractory_timer = 0
        cortical_column.inhibitory_neuron.spiked = False
        
        # Track firing counts for this presentation
        L2_E_fire_counts = np.zeros(n_L2_E, dtype=int)
        
        # Initialize previous spike arrays
        L1_E_spiked_prev = np.zeros(n_L1_E, dtype=int)
        L1_I_spiked_prev = np.zeros(n_L1_E, dtype=int)
        L2_E_spiked_prev = np.zeros(n_L2_E, dtype=int)
        L2_I_spiked_prev = False
        
        for t in range(n_time_steps_per_presentation):
            # Step 1: Compute inputs to L1
            external_input = compute_L1_external_inputs(pattern_vec)
            
            # L2 to L1 feedback input for I neurons (based on previous L2 E spikes)
            L2_to_L1_I_input = compute_L2_to_L1_I_input(L2_E_spiked_prev)
            
            # Set L1 inputs
            for i in range(n_L1_E):
                # E neuron: [from_I, external]
                spike0 = 1.0 if L1_I_spiked_prev[i] else 0.0
                spike1 = 1.0 if external_input[i] > 0.5 else 0.0  # Binary spike based on pixel value
                input_layer.excitatory_neurons[i].receive_input(np.array([spike0, spike1]))
                
                # I neuron input: from L2 E neurons (feedback)
                feedback_val = L2_to_L1_I_input[i]
                spike_I = 1.0 if feedback_val > 0 else 0.0
                input_layer.inhibitory_neurons[i].receive_input(np.array([spike_I]))
            
            # Step 2: Compute inputs to L2
            # Need L1 E spikes from previous step
            L1_to_L2_input_vec = compute_L1_to_L2_input(L1_E_spiked_prev)
            
            for j in range(n_L2_E):
                # E neuron j: [from_local_I, from_below]
                # Local inhibitory input: if L2 I neuron spiked previous step
                inhib_input = 1.0 if L2_I_spiked_prev else 0.0
                # Feedforward input: weighted sum from L1 E spikes
                excit_input = L1_to_L2_input_vec[j]
                # Convert to spikes for the two inputs:
                spike0 = inhib_input
                spike1 = 1.0 if excit_input > 0 else 0.0  # Treat positive input as spike
                cortical_column.excitatory_neurons[j].receive_input(np.array([spike0, spike1]))
            
            # L2 I neuron input: from all L2 E neurons (lateral inhibition)
            cortical_column.inhibitory_neuron.receive_input(L2_E_spiked_prev.astype(float))
            
            # Step 3: Check thresholds and fire
            L1_E_spiked = [n.check_threshold() for n in input_layer.excitatory_neurons]
            L1_I_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
            L2_E_spiked = [n.check_threshold() for n in cortical_column.excitatory_neurons]
            L2_I_spiked = cortical_column.inhibitory_neuron.check_threshold()
            
            # Fire neurons
            for i, neuron in enumerate(input_layer.excitatory_neurons):
                if L1_E_spiked[i]:
                    neuron.fire()
            for i, neuron in enumerate(input_layer.inhibitory_neurons):
                if L1_I_spiked[i]:
                    neuron.fire()
            for j, neuron in enumerate(cortical_column.excitatory_neurons):
                if L2_E_spiked[j]:
                    neuron.fire()
            if L2_I_spiked:
                cortical_column.inhibitory_neuron.fire()
            
            # Record firing counts for L2 E neurons
            for j in range(n_L2_E):
                if L2_E_spiked[j]:
                    L2_E_fire_counts[j] += 1
            
            # Step 4: Update all neurons (handle refractory, leak, etc.)
            for neuron in input_layer.excitatory_neurons:
                neuron.update()
            for neuron in input_layer.inhibitory_neurons:
                neuron.update()
            for neuron in cortical_column.excitatory_neurons:
                neuron.update()
            cortical_column.inhibitory_neuron.update()
            
            # Store previous spikes for next iteration
            L1_E_spiked_prev = L1_E_spiked
            L1_I_spiked_prev = L1_I_spiked
            L2_E_spiked_prev = np.array(L2_E_spiked, dtype=int)
            L2_I_spiked_prev = L2_I_spiked
        
        # After time steps for this presentation, check for activity
        total_L2_spikes = np.sum(L2_E_fire_counts)
        if total_L2_spikes > 0:
            winner_idx = np.argmax(L2_E_fire_counts)
            firing_fraction = L2_E_fire_counts[winner_idx] / n_time_steps_per_presentation
            
            pattern_winner_history[pattern_name].append((winner_idx, firing_fraction))
            
            # Print activity for this presentation (first few and periodic)
            if pres_idx < 4 or (pres_idx+1) % 5 == 0:
                winner_name = pattern_names[winner_idx]
                print(f"Presentation {pres_idx+1:2d}, Pattern {pattern_name:6s}: "
                      f"L2 E neuron {winner_idx} fired {L2_E_fire_counts[winner_idx]:2d} times "
                      f"({firing_fraction*100:5.1f}%)")
            
            # Check if consolidated (winner fires consistently above threshold)
            if firing_fraction >= consolidation_threshold:
                winner_name = pattern_names[winner_idx]
                if not pattern_consolidated[pattern_name]:
                    pattern_consolidated[pattern_name] = True
                    consolidation_order.append((pattern_name, winner_idx, pres_idx+1))
                    print(f"✓ CONSOLIDATED: Pattern {pattern_name:6s} → L2 E neuron {winner_idx} "
                          f"(fired {L2_E_fire_counts[winner_idx]:2d}/{n_time_steps_per_presentation} steps "
                          f"= {firing_fraction*100:5.1f}%) at presentation {pres_idx+1:2d}")
        else:
            # No L2 spikes - print occasionally for early debugging
            if pres_idx < 3:
                print(f"   Presentation {pres_idx+1:2d}, Pattern {pattern_name:6s}: No L2 E spikes")
    
    # Optional: print weight changes every few presentations
    if (pres_idx+1) % 5 == 0 and pres_idx+1 < n_presentations_per_pattern:
        print(f"\n--- After {pres_idx+1} presentations ---")
        print("Sample feedforward weights (L1 E0 -> L2 E0..2):")
        for j in range(min(3, n_L2_E)):
            weights = cortical_column.excitatory_neurons[j]._weights_array
            print(f"  to L2 E{j}: [{weights[0]:.3f}, {weights[1]:.3f}] (I_weights, FF_weights)")
        print()

print("\n" + "=" * 70)
print("=== CONSOLIDATION TRAINING COMPLETE ===")
print("=" * 70)

print("\nConsolidation Results:")
for pattern_name, winner_idx, pres_num in consolidation_order:
    # Find the firing fraction for this consolidation
    firing_fraction = None
    for winner, frac in pattern_winner_history[pattern_name]:
        if winner == winner_idx:
            firing_fraction = frac
            break
    if firing_fraction is not None:
        print(f"  {pattern_name:6s}: consolidated to L2 E neuron {winner_idx} "
              f"at presentation {pres_num} (firing {firing_fraction*100:5.1f}% of time)")
    else:
        print(f"  {pattern_name:6s}: consolidated to L2 E neuron {winner_idx} "
              f"at presentation {pres_num}")

print("\nPatterns not yet consolidated:")
for pattern_name in pattern_names:
    if not pattern_consolidated[pattern_name]:
        if pattern_winner_history[pattern_name]:
            # Show the best candidate so far
            best_winner, best_fraction = max(pattern_winner_history[pattern_name], key=lambda x: x[1])
            print(f"  {pattern_name:6s}: best so far L2 E neuron {best_winner} "
                  f"(firing {best_fraction*100:5.1f}% of time)")
        else:
            print(f"  {pattern_name:6s}: no L2 E spikes recorded")

print("\nFinal network state:")
print("- L1 E neurons: remain pre-trained (fire on external input)")
print("- L1 I neurons: trained via L2 feedback during simulation")
print("- L2 E neurons: trained via spike-timing plasticity during simulation") 
print("- L2 I neuron: trained via lateral inhibition during simulation")
print("- Connectivity: maintained as defined by classes")
print("  * L2 E neurons: [from_local_I, from_below]")
print("  * L2 I neuron: [from_all_local_E]")

print("\n" + "=" * 70)