#!/usr/bin/env python3
"""
Demo: Training I neurons in L1 and all neurons in L2, assuming L1 E neurons are pretrained.
Assumes L1 E neurons instantly fire when their preferred pixel is ON.
Focuses on learning in L1 I neurons (feedback from L2) and L2 neurons (feedforward from L1, lateral inhibition).
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

# Network parameters - optimized for learning in I neurons and L2
n_L1_E = 9  # number of excitatory neurons in input layer (matches pixels)
n_L2_E = 8  # number of excitatory neurons in cortical column (matches patterns)
threshold = 0.25  # Lower threshold to make firing easier
refractory_period = 2
learning_rate = 0.15  # Higher learning rate for faster learning
weight_cap = 1.0
leak_rate = 0.005  # Very low leak to preserve activity for learning

print("Building network for L1 I and L2 training (L1 E pretrained)...")
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

# Set up InputLayer internal connections:
# Each E neuron: [from_I (inhibitory), from_external (pixel input)]
# Each I neuron: [from_L2_E_feedback] (feedback from L2 E neurons)
for i in range(n_L1_E):
    # E neuron: inhibitory weight negative, external weight starts small
    input_layer.excitatory_neurons[i].weights = np.array([-0.3, 0.2])  # [I_weight, external_weight]
    # I neuron: single weight from L2 E feedback (to be learned)
    input_layer.inhibitory_neurons[i].weights = np.array([0.0])  # placeholder, will be learned

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
# Local inhibition: I -> E (negative, stronger to create competition)
cortical_column.set_local_inhibition_weights(-0.4)
# Feedforward weights from L1 E to L2 E: initialize small random
W_feedforward = np.random.uniform(0.2, 0.5, size=(n_L1_E, n_L2_E))  # Store for reference
# Lateral inhibition: E -> I within L2 (stronger for better competition)
cortical_column.set_lateral_inhibition_weights(-0.3)
# No feedback inputs in this 2-layer network

# Set up L1 I neuron feedback weights from L2 E
# Each L1 I neuron receives from all L2 E neurons
W_L2_to_L1_I = np.random.uniform(0.2, 0.4, size=(n_L2_E, n_L1_E))  # Store for reference

print("Network initialized with learning-focused weights.")
print("Sample feedforward weights (L1 E0 -> L2 E0..2):")
for j in range(min(3, n_L2_E)):
    print(f"  to L2 E{j}: {W_feedforward[0,j]:.3f}")
print()

# Helper functions
def compute_L1_external_inputs(pattern_vector):
    """Return external input to each L1 E neuron (pixel values)."""
    return pattern_vector  # length 9

def compute_L2_to_L1_I_input(L2_E_spiked):
    """
    Compute input to each L1 I neuron from L2 E neurons.
    Each L1 I neuron receives from all L2 E neurons with weight matrix W_L2_to_L1_I.
    L2_E_spiked is binary array (0/1) of length n_L2_E.
    Returns vector length n_L1_E.
    """
    if np.sum(L2_E_spiked) == 0:
        return np.zeros(n_L1_E)
    return np.dot(L2_E_spiked, W_L2_to_L1_I)  # shape (n_L1_E,)

def compute_L1_to_L2_input(L1_E_spiked):
    """
    Compute aggregated input from L1 E neurons to each L2 E neuron.
    We'll use a weighted sum: for each L2 E neuron j, sum_i L1_E_spiked[i] * W_feedforward[i,j]
    L1_E_spiked is binary array (0/1) of whether each L1 E neuron spiked in current time step.
    """
    if np.sum(L1_E_spiked) == 0:
        return np.zeros(n_L2_E)
    return np.dot(L1_E_spiked, W_feedforward)  # shape (n_L2_E,)

# Simulation parameters
n_presentations_per_pattern = 12  # Number of times to present each pattern
n_time_steps_per_presentation = 20  # Simulation steps per pattern presentation

print(f"Starting training ({n_presentations_per_pattern} presentations × {n_time_steps_per_presentation} steps each)...")
print("-" * 70)

# We'll track which L2 E neuron wins each pattern over time
pattern_winner_history = {name: [] for name in pattern_names}

# Track weight changes for analysis
initial_W_L2_to_L1_I = W_L2_to_L1_I.copy()
initial_W_feedforward = W_feedforward.copy()

# We'll need to keep track of the previous spike states for lateral inhibition and feedback
# Initialize previous spike states
prev_L1_E_spiked = np.zeros(n_L1_E, dtype=int)  # From previous time step
prev_L1_I_spiked = np.zeros(n_L1_E, dtype=int)
prev_L2_E_spiked = np.zeros(n_L2_E, dtype=int)
prev_L2_I_spiked = False

for pres_idx in range(n_presentations_per_pattern):
    for p_idx, pattern_name in enumerate(pattern_names):
        pattern_vec = pattern_vectors[p_idx]
        # Convert pattern to binary spike pattern for L1 E neurons (1 if pixel ON)
        # Since L1 E neurons are pretrained, they will fire when their pixel is ON.
        l1_e_spike_pattern = (np.array(pattern_vec) > 0.5).astype(int)
        
        # Reset neuron states at start of each presentation for plastic neurons (L1 I, L2 E, L2 I)
        # Note: We do not reset L1 E neurons because we are clamping their spiked state based on pattern.
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
        L1_I_fire_counts = np.zeros(n_L1_E, dtype=int)  # For L1 I neurons
        L2_E_fire_counts = np.zeros(n_L2_E, dtype=int)  # For L2 E neurons
        L2_I_fire_count = 0  # For L2 I neuron
        
        for t in range(n_time_steps_per_presentation):
            # === STEP 1: Set L1 E neurons to fire based on pretraining ===
            # Since L1 E neurons are pretrained, we set their spiked state to match the pattern.
            # We do not update their potentials or weights here (we assume they are fixed).
            # However, we need to provide their spiked state as input to L2.
            l1_e_spiked_current = l1_e_spike_pattern.copy()  # This is the spiked state of L1 E neurons for this time step
            
            # === STEP 2: Compute inputs to L1 I neurons from L2 E feedback ===
            # L1 I neurons receive feedback from L2 E neurons (from previous time step)
            L2_to_L1_I_input = compute_L2_to_L1_I_input(prev_L2_E_spiked)
            
            # Set L1 I neuron inputs
            for i in range(n_L1_E):
                # I neuron has single input: weighted sum of L2 E spikes from previous time step
                input_layer.inhibitory_neurons[i].receive_input(np.array([L2_to_L1_I_input[i]]))
            
            # === STEP 3: Compute inputs to L2 neurons ===
            # L2 E neurons receive [from_local_I, from_below (L1 E input)]
            # L2 I neuron receives [from_all_L2_E (lateral inhibition)]
            
            # L1 E input to L2 E neurons: use the current L1 E spiked state (from pattern)
            l1_to_l2_input_vec = compute_L1_to_L2_input(l1_e_spiked_current)
            
            # L2 I neuron input: from all L2 E neurons (lateral inhibition) from previous time step
            # We use previous time step to avoid instantaneous feedback loops
            l2_i_spiked_prev = prev_L2_I_spiked  # This is a boolean
            
            # Set L2 E neuron inputs
            for j in range(n_L2_E):
                # E neuron j: [from_local_I, from_below]
                # Local inhibitory input: if L2 I neuron spiked in previous time step
                inhib_input = 1.0 if l2_i_spiked_prev else 0.0
                # Feedforward input: weighted sum from L1 E spikes (current time step)
                excit_input = l1_to_l2_input_vec[j]
                cortical_column.excitatory_neurons[j].receive_input(np.array([inhib_input, excit_input]))
            
            # L2 I neuron input: from all L2 E neurons (lateral inhibition) from current time step? 
            # To avoid feedback loops, we use the current L2 E spiked state? 
            # Actually, for lateral inhibition, it's often modeled as instantaneous or from the same time step.
            # However, to keep it simple and stable, we'll use the current L2 E spiked state for the I neuron input.
            # But note: if we use the current state, then the I neuron's state depends on the E neurons' state in the same time step,
            # which is fine as long as we compute the I neuron's state after we have the E neurons' spikes.
            # We will compute the E neurons' spikes first (based on inputs from L1 E and previous L2 I), then compute the I neuron's input.
            # So we need to compute the E neurons' spikes first.
            
            # Let's restructure:
            # 1. We have L1 E spiked state (from pattern) -> l1_e_spiked_current
            # 2. We have previous L2 I spiked state -> prev_L2_I_spiked
            # 3. We compute L2 E neuron inputs and then their spiked state.
            # 4. Then we compute L2 I neuron input from the current L2 E spiked state.
            # 5. Then we compute L2 I neuron spiked state.
            # 6. Then we update all plastic neurons.
            
            # So we need to compute L2 E spikes first.
            
            # Set L2 E neuron inputs (as above)
            for j in range(n_L2_E):
                # E neuron j: [from_local_I, from_below]
                # Local inhibitory input: if L2 I neuron spiked in previous time step
                inhib_input = 1.0 if prev_L2_I_spiked else 0.0
                # Feedforward input: weighted sum from L1 E spikes (current time step)
                excit_input = l1_to_l2_input_vec[j]
                cortical_column.excitatory_neurons[j].receive_input(np.array([inhib_input, excit_input]))
            
            # Now check thresholds for L2 E neurons to get their spiked state
            l2_e_spiked = [n.check_threshold() for n in cortical_column.excitatory_neurons]
            
            # Now compute L2 I neuron input: from all L2 E neurons (lateral inhibition) using the current L2 E spiked state
            l2_i_input = np.array([np.sum(l2_e_spiked)])  # Sum of spikes as input
            cortical_column.inhibitory_neuron.receive_input(l2_i_input)
            
            # Now check threshold for L2 I neuron
            l2_i_spiked = cortical_column.inhibitory_neuron.check_threshold()
            
            # Now set L1 I neuron inputs (we already did this above using prev_L2_E_spiked)
            # But note: we used prev_L2_E_spiked for L1 I input, which is correct (feedback from previous time step)
            # We already set the L1 I neuron inputs above.
            
            # Now we have:
            #   l1_e_spiked_current: spiked state of L1 E neurons (from pattern)
            #   l1_i_spiked: to be computed from L1 I neurons (we haven't set their inputs yet? Wait we did above)
            #   l2_e_spiked: spiked state of L2 E neurons
            #   l2_i_spiked: spiked state of L2 I neuron
            
            # Actually, we set L1 I neuron inputs above using prev_L2_E_spiked. That is correct.
            # Now we need to check the threshold for L1 I neurons.
            # But we have not yet set their inputs? We did set them above. Let's verify:
            # In the loop for i in range(n_L1_E): we did input_layer.inhibitory_neurons[i].receive_input(np.array([L2_to_L1_I_input[i]]))
            # So yes, we have set the L1 I neuron inputs.
            
            # Now check thresholds for L1 I neurons
            l1_i_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
            
            # L1 E neurons: we are clamping their spiked state to l1_e_spiked_current, but we still need to update their state?
            # Since we are assuming they are pretrained and we are overriding their spiked state, we will not update their state.
            # However, for completeness, we could update their state and then override the spiked state.
            # But to keep it simple, we will not update the L1 E neurons at all (skip their update step).
            # We will only update the plastic neurons: L1 I, L2 E, L2 I.
            
            # Record firing for plasticity analysis
            for i in range(n_L1_E):
                if l1_i_spiked[i]:
                    L1_I_fire_counts[i] += 1
            for j in range(n_L2_E):
                if l2_e_spiked[j]:
                    L2_E_fire_counts[j] += 1
            if l2_i_spiked:
                L2_I_fire_count += 1
            
            # Update plastic neurons: L1 I, L2 E, L2 I
            for neuron in input_layer.inhibitory_neurons:
                neuron.update()
            for neuron in cortical_column.excitatory_neurons:
                neuron.update()
            cortical_column.inhibitory_neuron.update()
            
            # Store current spike states as previous for next time step
            prev_L1_E_spiked = l1_e_spiked_current.copy()
            prev_L1_I_spiked = l1_i_spiked.copy()
            prev_L2_E_spiked = np.array(l2_e_spiked, dtype=int)
            prev_L2_I_spiked = l2_i_spiked
        
        # After time steps for this pattern, determine winning L2 E neuron
        if np.sum(L2_E_fire_counts) > 0:
            winner_idx = np.argmax(L2_E_fire_counts)
            winner_name = pattern_names[winner_idx]
            pattern_winner_history[pattern_name].append(winner_idx)
            print(f"Presentation {pres_idx+1:2d}, Pattern {pattern_name:6s}: L2 E neuron {winner_idx} fired {L2_E_fire_counts[winner_idx]:2d} times")
        else:
            print(f"Presentation {pres_idx+1:2d}, Pattern {pattern_name:6s}: No L2 E spikes")
    
    # Optional: print weight changes every few presentations
    if (pres_idx+1) % 4 == 0:
        print(f"\n--- After {pres_idx+1} presentations ---")
        print("Sample L1 I feedback weights (from L2 E0, E1, E2 to L1 I0):")
        for j in range(min(3, n_L2_E)):
            print(f"  from L2 E{j}: {input_layer.inhibitory_neurons[0].weights[j]:.3f}")
        print("Sample L2 E feedforward weights (from L1 E0, E1, E2 to L2 E0):")
        for i in range(min(3, n_L1_E)):
            print(f"  from L1 E{i}: {cortical_column.excitatory_neurons[0].weights[i]:.3f}")
        print()

print("\n" + "=" * 70)
print("=== Training Complete ===")
print("Winner history per pattern:")
for pattern_name in pattern_names:
    winners = pattern_winner_history[pattern_name]
    if winners:
        # most common winner
        from collections import Counter
        most_common = Counter(winners).most_common(1)[0]
        print(f"  {pattern_name:6s}: most often won by L2 E neuron {most_common[0]} (count {most_common[1]} / {len(winners)})")
    else:
        print(f"  {pattern_name:6s}: no wins recorded")

print("\n=== Weight Change Analysis ===")
print("Average change in L1 I feedback weights (from L2 E to L1 I):")
W_L2_to_L1_I_final = np.zeros((n_L2_E, n_L1_E))
for i in range(n_L1_E):
    for j in range(n_L2_E):
        W_L2_to_L1_I_final[j, i] = input_layer.inhibitory_neurons[i].weights[j]
weight_change_L1_I = W_L2_to_L1_I_final - initial_W_L2_to_L1_I
print(f"  Mean absolute change: {np.mean(np.abs(weight_change_L1_I)):.4f}")
print(f"  Max increase: {np.max(weight_change_L1_I):.4f}")
print(f"  Max decrease: {np.min(weight_change_L1_I):.4f}")

print("\nAverage change in L2 E feedforward weights (from L1 E to L2 E):")
# We need to extract the final weights from the cortical column
W_feedforward_final = np.zeros((n_L1_E, n_L2_E))
for j in range(n_L2_E):
    for i in range(n_L1_E):
        W_feedforward_final[i, j] = cortical_column.excitatory_neurons[j].weights[i]
weight_change_L2_E = W_feedforward_final - initial_W_feedforward
print(f"  Mean absolute change: {np.mean(np.abs(weight_change_L2_E)):.4f}")
print(f"  Max increase: {np.max(weight_change_L2_E):.4f}")
print(f"  Max decrease: {np.min(weight_change_L2_E):.4f}")

print("\n=== Testing after training ===")
for p_idx, pattern_name in enumerate(pattern_names):
    pattern_vec = pattern_vectors[p_idx]
    # Reset states for plastic neurons (L1 I, L2 E, L2 I)
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
    # Note: We do not reset L1 E neurons because we are clamping them based on pattern.
    
    # Run a short simulation to see which L2 E neuron fires most
    fire_counts = np.zeros(n_L2_E, dtype=int)
    for t in range(15):
        # Set L1 E neurons to fire based on pattern (manual)
        l1_e_spike_pattern = (np.array(pattern_vec) > 0.5).astype(int)
        
        # Compute inputs to L1 I from L2 E feedback (using previous L2 E spikes)
        if t == 0:
            prev_L2_E_spiked = np.zeros(n_L2_E, dtype=int)
        else:
            prev_L2_E_spiked = prev_L2_E_spiked  # from previous step
        L2_to_L1_I_input = compute_L2_to_L1_I_input(prev_L2_E_spiked)
        
        # Set L1 I neuron inputs
        for i in range(n_L1_E):
            input_layer.inhibitory_neurons[i].receive_input(np.array([L2_to_L1_I_input[i]]))
        
        # Compute L1 -> L2 input
        l1_to_l2_input_vec = compute_L1_to_L2_input(l1_e_spike_pattern)
        
        # Compute L2 I neuron input from L2 E spikes (using previous L2 E spikes? or current? We'll use current for simplicity in test)
        # We'll compute L2 E spikes first.
        # Set L2 E neuron inputs
        for j in range(n_L2_E):
            # E neuron j: [from_local_I, from_below]
            # Local inhibitory input: if L2 I neuron spiked in previous time step
            # We need prev_L2_I_spiked from previous time step
            if t == 0:
                prev_L2_I_spiked = False
            else:
                prev_L2_I_spiked = prev_L2_I_spiked  # from previous step
            inhib_input = 1.0 if prev_L2_I_spiked else 0.0
            # Feedforward input: weighted sum from L1 E spikes (current time step)
            excit_input = l1_to_l2_input_vec[j]
            cortical_column.excitatory_neurons[j].receive_input(np.array([inhib_input, excit_input]))
        
        # Check thresholds for L2 E neurons
        l2_e_spiked = [n.check_threshold() for n in cortical_column.excitatory_neurons]
        
        # Compute L2 I neuron input: from all L2 E neurons (lateral inhibition) using current L2 E spiked state
        l2_i_input = np.array([np.sum(l2_e_spiked)])
        cortical_column.inhibitory_neuron.receive_input(l2_i_input)
        l2_i_spiked = cortical_column.inhibitory_neuron.check_threshold()
        
        # Set L1 I neuron inputs from L2 E feedback (using previous L2 E spikes)
        for i in range(n_L1_E):
            input_layer.inhibitory_neurons[i].receive_input(np.array([L2_to_L1_I_input[i]]))
        
        # Check thresholds for L1 I neurons
        l1_i_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
        
        # Record firing for L2 E neurons
        for j in range(n_L2_E):
            if l2_e_spiked[j]:
                fire_counts[j] += 1
        
        # Update plastic neurons: L1 I, L2 E, L2 I
        for neuron in input_layer.inhibitory_neurons:
            neuron.update()
        for neuron in cortical_column.excitatory_neurons:
            neuron.update()
        cortical_column.inhibitory_neuron.update()
        
        # Store current spike states as previous for next time step
        prev_L2_E_spiked = np.array(l2_e_spiked, dtype=int)
        prev_L2_I_spiked = l2_i_spiked
    
    winner = np.argmax(fire_counts)
    print(f"Pattern {pattern_name:6s}: L2 E neuron {winner} fired {fire_counts[winner]:2d} times (total fires {np.sum(fire_counts)})")

print("\n" + "=" * 70)
print("Demo complete. Observe:")
print("1. Which L2 E neurons consistently win for each pattern (should be selective)")
print("2. How much the L1 I feedback weights changed (should show learning from L2 feedback)")
print("3. How much the L2 E feedforward weights changed (should show learning from L1 input)")
print("If learning worked:")
print("- Each pattern should have a clear winning L2 E neuron")
print("- L1 I neurons should develop selective feedback weights from L2 E neurons")
print("- L2 E neurons should develop selective feedforward weights from L1 E pixels")
print("=" * 70)