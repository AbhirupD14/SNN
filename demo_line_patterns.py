"""
Simulation of a 2-layer network for line pattern recognition on a 3x3 grid.
Patterns: 3 rows, 3 columns, 2 diagonals (8 total).
Input layer: 9 E/I pairs (9x3 grid pixels).
Cortical column: 8 E neurons (one per pattern) + 1 I neuron (lateral inhibition).
Goal: Present each pattern repeatedly until one L2 E neuron consolidates as the winner for that pattern.
"""
import numpy as np
import random
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

print("Building network...")
# Input Layer: 9 E/I pairs
input_layer = InputLayer(n_neurons=n_L1_E, threshold=threshold,
                         refractory_period=refractory_period,
                         learning_rate=learning_rate,
                         weight_cap=weight_cap,
                         leak_rate=leak_rate)
# Set up InputLayer internal connections:
# Each E neuron: [from_I (inhibitory), from_external (pixel input)]
# Each I neuron: [from_L2_E_feedback] (we'll set later)
for i in range(n_L1_E):
    # E neuron: inhibitory weight negative, external weight to be learned
    input_layer.excitatory_neurons[i].weights = np.array([-0.5, 0.0])  # [I_weight, external_weight]
    # I neuron: single weight from L2 E feedback (to be set after L2 created)
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
# Weights shape: (n_L1_E, n_L2_E) - we'll store as list of lists for convenience
W_feedforward = np.random.uniform(0.1, 0.3, size=(n_L1_E, n_L2_E))
# Assign to each L2 E neuron: second input (index 1) gets weighted sum from L1 E
# We'll handle this in simulation by computing the sum each time step.
# For simplicity, we'll store the weight matrix and compute dot product.
# Lateral inhibition: E -> I within L2 (negative)
cortical_column.set_lateral_inhibition_weights(-0.4)
# No feedback inputs in this 2-layer network

# Set up L1 I neuron feedback weights from L2 E (to be set after L2 created)
# Each L1 I neuron receives from all L2 E neurons (we'll use same weight for simplicity)
# We'll store a matrix W_L2_to_L1_I of shape (n_L2_E, n_L1_E)
W_L2_to_L1_I = np.random.uniform(0.1, 0.3, size=(n_L2_E, n_L1_E))

print("Network built.")

# Helper functions to compute inputs
def compute_L1_external_inputs(pattern_vector):
    """Return external input to each L1 E neuron (pixel values)."""
    return pattern_vector  # length 9

def compute_L1_to_L2_input(L1_E_spiked):
    """
    Compute aggregated input from L1 E neurons to each L2 E neuron.
    We'll use a weighted sum: for each L2 E neuron j, sum_i L1_E_spiked[i] * W_feedforward[i,j]
    L1_E_spiked is binary array (0/1) of whether each L1 E neuron spiked in previous step.
    """
    # If no spikes, return zeros
    if np.sum(L1_E_spiked) == 0:
        return np.zeros(n_L2_E)
    # Otherwise compute weighted sum
    return np.dot(L1_E_spiked, W_feedforward)  # shape (n_L2_E,)

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

# Simulation parameters
n_presentations_per_pattern = 20  # how many times to present each pattern before moving on
n_time_steps_per_presentation = 30  # simulation steps per pattern presentation
consolidation_threshold = 0.8  # fraction of time steps a neuron must fire to be considered winner

print("\nStarting training...")
# We'll track which L2 E neuron wins each pattern over time
pattern_winner_history = {name: [] for name in pattern_names}

for pres_idx in range(n_presentations_per_pattern):
    for p_idx, pattern_name in enumerate(pattern_names):
        pattern_vec = pattern_vectors[p_idx]
        # Reset neuron states? We'll let them carry over, but we can reset potentials to 0 for simplicity each presentation.
        # However, to avoid interference, we'll reset potentials at start of each presentation.
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
        
        for t in range(n_time_steps_per_presentation):
            # Step 1: Compute inputs to L1
            external_input = compute_L1_external_inputs(pattern_vec)
            # L2 to L1 feedback input for I neurons (based on previous L2 E spikes)
            # For t=0, use zeros
            if t == 0:
                L2_E_spiked_prev = np.zeros(n_L2_E, dtype=int)
            else:
                L2_E_spiked_prev = L2_E_spiked_prev  # from previous step
            L2_to_L1_I_input = compute_L2_to_L1_I_input(L2_E_spiked_prev)
            
            # Set L1 inputs
            for i in range(n_L1_E):
                # E neuron: [from_I, external]
                # We need the I neuron's spike from previous step to compute inhibitory input
                # We'll approximate using previous I spike; we'll store previous I spikes
                if t == 0:
                    L1_I_spiked_prev = np.zeros(n_L1_E, dtype=int)
                else:
                    L1_I_spiked_prev = L1_I_spiked_prev
                # Inhibitory input: if I neuron spiked previous step, send -1 (or weight * spike)
                # We'll use the weight from I to E (which is stored in E neuron's weight[0])
                inhib_input = input_layer.excitatory_neurons[i].weights[0] * (1.0 if L1_I_spiked_prev[i] else 0.0)
                excit_input = external_input[i] * input_layer.excitatory_neurons[i].weights[1]  # external weight * pixel value
                total_E_input = inhib_input + excit_input
                # We'll directly set the neuron's potential increment? Actually receive_input expects spikes.
                # We'll convert total input to a spike equivalent? Better to use the receive_input mechanism:
                # Each E neuron has two inputs: input 0 from I neuron, input 1 from external.
                # So we need to provide spikes for each input.
                # Input 0 spike: 1 if L1 I neuron spiked previous step else 0
                # Input 1 spike: 1 if external pixel is ON? Actually external input is graded; we can treat pixel value as probability of spike.
                # For simplicity, we'll convert external input to a spike with probability = external_input (since external_input is 0 or 1).
                # We'll do:
                #   spike0 = 1 if L1_I_spiked_prev[i] else 0
                #   spike1 = 1 if external_input[i] > 0.5 else 0  (since external_input is 0 or 1)
                spike0 = 1.0 if L1_I_spiked_prev[i] else 0.0
                spike1 = 1.0 if external_input[i] > 0.5 else 0.0
                input_layer.excitatory_neurons[i].receive_input(np.array([spike0, spike1]))
                
                # I neuron input: from L2 E neurons (feedback)
                # I neuron has single input: weighted sum of L2 E spikes
                # We'll provide a spike probability equal to the weighted sum? We'll just use the weighted sum as a spike? 
                # Instead, we'll treat the I neuron's input as a single channel where we give a spike if the weighted sum > 0.
                # We'll compute the total feedback input and convert to a spike probability.
                # For simplicity, we'll give a spike if the weighted sum > 0.
                feedback_val = L2_to_L1_I_input[i]
                spike_I = 1.0 if feedback_val > 0 else 0.0
                input_layer.inhibitory_neurons[i].receive_input(np.array([spike_I]))
            
            # Step 2: Compute inputs to L2
            # Need L1 E spikes from previous step (we'll use L1_E_spiked_prev)
            if t == 0:
                L1_E_spiked_prev = np.zeros(n_L1_E, dtype=int)
            else:
                L1_E_spiked_prev = L1_E_spiked_prev
            L1_to_L2_input_vec = compute_L1_to_L2_input(L1_E_spiked_prev)
            
            for j in range(n_L2_E):
                # E neuron j: [from_local_I, from_below]
                # Local inhibitory input: if L2 I neuron spiked previous step
                if t == 0:
                    L2_I_spiked_prev = False
                else:
                    L2_I_spiked_prev = L2_I_spiked_prev
                inhib_input = cortical_column.excitatory_neurons[j].weights[0] * (1.0 if L2_I_spiked_prev else 0.0)
                # Feedforward input: weighted sum from L1 E spikes
                excit_input = L1_to_L2_input_vec[j] * cortical_column.excitatory_neurons[j].weights[1]
                # Convert to spikes for the two inputs:
                spike0 = 1.0 if L2_I_spiked_prev else 0.0
                spike1 = 1.0 if L1_to_L2_input_vec[j] > 0 else 0.0  # simplify: treat positive input as spike
                cortical_column.excitatory_neurons[j].receive_input(np.array([spike0, spike1]))
            
            # L2 I neuron input: from all L2 E neurons (lateral inhibition) and feedback (none)
            # We'll compute spikes from L2 E neurons and give as input to I neuron.
            # I neuron has n_L2_E inputs: each from one L2 E neuron (lateral inhibition)
            # We'll provide spikes: 1 if L2 E neuron spiked previous step else 0
            L2_E_spikes_for_I = L2_E_spiked_prev  # Use L2 E spikes from previous step
            cortical_column.inhibitory_neuron.receive_input(L2_E_spikes_for_I.astype(float))
            
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
            L1_I_spiked_prev = L1_I_spiked
            L2_E_spiked_prev = np.array(L2_E_spiked, dtype=int)
            L2_I_spiked_prev = L2_I_spiked
        
        # After time steps for this pattern, determine winning L2 E neuron
        if np.sum(L2_E_fire_counts) > 0:
            winner_idx = np.argmax(L2_E_fire_counts)
            winner_name = pattern_names[winner_idx]
            pattern_winner_history[pattern_name].append(winner_idx)
            print(f"Presentation {pres_idx+1}, Pattern {pattern_name}: L2 E neuron {winner_idx} fired {L2_E_fire_counts[winner_idx]} times")
        else:
            print(f"Presentation {pres_idx+1}, Pattern {pattern_name}: No L2 E spikes")
    
    # Optional: print weight changes every few presentations
    if (pres_idx+1) % 5 == 0:
        print(f"\n--- After {pres_idx+1} presentations ---")
        print("Sample feedforward weights (L1 E0 -> L2 E0..2):")
        for j in range(min(3, n_L2_E)):
            print(f"  to L2 E{j}: {W_feedforward[0,j]:.3f}")
        print()

print("\n=== Training Complete ===")
print("Winner history per pattern:")
for pattern_name in pattern_names:
    winners = pattern_winner_history[pattern_name]
    if winners:
        # most common winner
        from collections import Counter
        most_common = Counter(winners).most_common(1)[0]
        print(f"  {pattern_name}: most often won by L2 E neuron {most_common[0]} (count {most_common[1]} / {len(winners)})")
    else:
        print(f"  {pattern_name}: no wins recorded")

# Test each pattern after training to see which L2 E neuron wins
print("\n=== Testing after training ===")
for p_idx, pattern_name in enumerate(pattern_names):
    pattern_vec = pattern_vectors[p_idx]
    # Reset states
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
    
    # Run a short simulation to see which L2 E neuron fires most
    fire_counts = np.zeros(n_L2_E, dtype=int)
    for t in range(20):
        external_input = compute_L1_external_inputs(pattern_vec)
        # Use zero feedback for simplicity in test
        L2_E_spiked_prev = np.zeros(n_L2_E, dtype=int)
        L1_I_spiked_prev = np.zeros(n_L1_E, dtype=int)
        L2_I_spiked_prev = False
        
        # Set inputs (simplified)
        for i in range(n_L1_E):
            spike0 = 1.0 if L1_I_spiked_prev[i] else 0.0
            spike1 = 1.0 if external_input[i] > 0.5 else 0.0
            input_layer.excitatory_neurons[i].receive_input(np.array([spike0, spike1]))
            input_layer.inhibitory_neurons[i].receive_input(np.array([1.0 if compute_L2_to_L1_I_input(L2_E_spiked_prev)[i] > 0 else 0.0]))
        # L2 inputs
        L1_to_L2_input_vec = compute_L1_to_L2_input(L1_I_spiked_prev)  # Actually should use L1 E spikes, but we approximate
        for j in range(n_L2_E):
            spike0 = 1.0 if L2_I_spiked_prev else 0.0
            spike1 = 1.0 if L1_to_L2_input_vec[j] > 0 else 0.0
            cortical_column.excitatory_neurons[j].receive_input(np.array([spike0, spike1]))
        cortical_column.inhibitory_neuron.receive_input(L2_E_spiked_prev.astype(float))
        
        # Check and fire
        L1_E_spiked = [n.check_threshold() for n in input_layer.excitatory_neurons]
        L1_I_spiked = [n.check_threshold() for n in input_layer.inhibitory_neurons]
        L2_E_spiked = [n.check_threshold() for n in cortical_column.excitatory_neurons]
        L2_I_spiked = cortical_column.inhibitory_neuron.check_threshold()
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
            for j in range(n_L2_E):
                if L2_E_spiked[j]:
                    fire_counts[j] += 1
            # Update
            for neuron in input_layer.excitatory_neurons:
                neuron.update()
            for neuron in input_layer.inhibitory_neurons:
                neuron.update()
            for neuron in cortical_column.excitatory_neurons:
                neuron.update()
            cortical_column.inhibitory_neuron.update()
            
            L1_I_spiked_prev = L1_I_spiked
            L2_E_spiked_prev = np.array(L2_E_spiked, dtype=int)
            L2_I_spiked_prev = L2_I_spiked
    
    winner = np.argmax(fire_counts)
    print(f"Pattern {pattern_name}: L2 E neuron {winner} fired {fire_counts[winner]} times (total fires {np.sum(fire_counts)})")

print("\nDone.")