# OOP Refactor Plan for FEPSNN

## Current State
- Single `FEPSNN` class manages all neuron states, synaptic weights, and dynamics
- All state stored as NumPy arrays indexed by layer/population
- Plasticity implemented in `_learn` and `_learn_l3` methods
- Event processing in `process_event` method

## Proposed Refactor: Layered Object-Oriented Design

### 1. NeuronGroup Class
Manages a population of neurons with shared properties:
- Membrane potentials (`v`)
- Firing thresholds (`theta`)
- Adaptation current (`adapt`)
- Refractory counters (`refrac`)
- Silence counters (`silence`)
- Maturity (observable)

Methods:
- `apply_leak(dt)`: Apply leaky integration
- `compute_spikes()`: Determine which neurons fire
- `handle_spikes(spike_mask)`: Reset voltages, update adaptation/refractory
- `update_slow_state()`: Update adaptation/refractory/silence
- `get_maturity()`: Compute maturity for synaptic scaling
- `reset()`: Reset to initial state

### 2. Projection Class
Manages plastic connections between pre- and post-synaptic groups:
- Weight matrix (`weights`)
- Connection type (excitatory/inhibitory)
- Plasticity parameters (W_GAIN, PRUNE_RATE, SYNAPTIC_BUDGET)
- Pre- and post-synaptic NeuronGroup references

Methods:
- `compute_input(spike_mask_pre)`: Compute input to post-synaptic group
- `update_plasticity(post_neuron_idx, pre_spike_mask, post_spike_mask)`: Hebbian learning
- `apply_synaptic_scaling()`: Enforce budget constraint
- `get_maturity(post_neuron_idx)`: Compute maturity for a post-synaptic neuron

### 3. PlasticityRule Class (Optional)
Encapsulates the learning rule parameters:
- W_GAIN (demand-driven widening)
- PRUNE_RATE (heterosynaptic withering)
- SYNAPTIC_BUDGET (homeostatic scaling)
- MATURITY_THRESHOLD (for observable maturity)

### 4. Network Class (Refactored FEPSNN)
Orchestrates NeuronGroups and Projections:

Instantiates:
- NeuronGroups: L1E, L1I, L2E, L2I, L3E, L3I
- Projections (plastic):
  - L1E → L2E (excitatory)
  - L2E → L1I (excitatory, feedback squelch)
  - L1E → L1I (excitatory)
  - L1I → L1E (excitatory, sign->inhibitory)
  - L2E → L3E (excitatory)
  - L3E → L3I (excitatory, feedback squelch)
  - L3E → L2I (excitatory, feedback squelch)  // NEW
- Fixed connections: handled as simple attributes or constants

Event processing in `process_event`:
1. Apply leak to all NeuronGroups
2. Apply external input to L1E
3. Compute L1E spikes
4. Propagate L1E → L1I (feedforward)
5. Compute L1I spikes
6. Propagate L1I → L1I (lateral inhibition)
7. Compute L1E → L2E input, compute L2E spikes
8. Propagate L2E → L2I (drive global inhibitor)
9. Compute L2I spikes
10. Propagate L2I → L2E (global brake)
11. Propagate L2E → L3E input, compute L3E spikes
12. Compute L3E → L3I input, compute L3I spikes
13. Propagate L3I → L3E (lateral inhibition)
14. Propagate L3E → L2I (feedback to L2 inhibition)
15. Handle spikes: update states, determine winners
16. Apply plasticity:
    - For L2E winner: update L1E→L2E, L2E→L1I
    - For L3E winner: update L2E→L3E, L3E→L3I, L3E→L2I
17. Apply synaptic scaling where needed
18. Update slow states (adaptation, refractory, silence)
19. Record frame if requested

## Benefits
1. **Modularity**: Clear separation of concerns
2. **Configurability**: Easy to change layer sizes, connection patterns
3. **Extensibility**: Simple to add layers (L4), lateral connections, different architectures
4. **Maintainability**: Each class has single responsibility
5. **Preserves Constraints**: All learning remains local, emergent, no gradients

## Implementation Steps
1. Create `neuron_group.py` with NeuronGroup class
2. Create `projection.py` with Projection class
3. Create `plasticity_rule.py` with PlasticityRule class (optional)
4. Refactor `fep_model.py` to use these classes
5. Verify behavior matches original using existing tests
6. Clean up and document

## Files to Modify/Create
- `docs/refactor_plan/oop_refactor_plan.md` (this file)
- `src/neuron_group.py` (new)
- `src/projection.py` (new)
- `src/plasticity_rule.py` (new, optional)
- `fep_model.py` (refactored to use new classes)
- Update imports and potentially directory structure

Let me know if you'd like to proceed with implementing this refactor, and we can start by creating a feature branch and implementing the classes incrementally.
