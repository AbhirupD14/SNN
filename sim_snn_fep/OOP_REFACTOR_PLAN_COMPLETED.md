OOP REFACTOR PLAN - COMPLETED

We have successfully planned and verified an OOP refactor for the FEPSNN model.

## What Was Done
1. Designed an OOP structure with:
   - NeuronGroup class: manages state and dynamics of a neuron population (v, theta, adapt, refrac, silence)
   - Projection class: manages plastic synapses between populations (weights, learning rules)
2. Created the classes in:
   - src/neuron_group.py
   - src/projection.py
3. Verified the OOP implementation with a targeted script that confirmed:
   - Correct implementation of the classes
   - Presence and correctness of the L3E->L2I feedback connection (plastic excitatory synapses from L3E to L2I)
   - All operations use explicit loops (no linear algebra as mechanism)
   - Biological plasticity rules (Hebbian learning, synaptic scaling, pruning) are followed
   - Budget constraint is enforced via explicit loops
4. The original fep_model.py (with the L3E->L2I connection) remains active and verified.

## Biological Constraints Preserved
- ✅ No linear algebra as the computational mechanism (numpy arrays used only for storage)
- ✅ All learning is local and emergent (Hebbian plasticity)
- ✅ No gradients, supervision, or global error signals
- ✅ Fixed firing thresholds (with small jitter for symmetry breaking)
- ✅ Competition via intrinsic biophysics (refractory, adaptation, lateral inhibition, synaptic scaling)
- ✅ The L3E->L2I connection follows identical plasticity rules as other plastic synapses in the model

## Next Steps for Implementation
To complete the OOP refactor, one would:
1. Integrate the NeuronGroup and Projection classes into fep_model.py:
   - Replace the current array-based state (e.g., self.v_l2e) with NeuronGroup instances (self.L2E)
   - Replace weight matrices (e.g., self.weights_l1e_l2e) with Projection instances (self.proj_L1E_L2E)
   - Update the process_event method to use the new class methods (apply_leak, compute_spikes, handle_spikes, update_plasticity, etc.)
2. Run the existing test suite (test_l3.py, generate_sim_data.py, generate_history.py) to verify numerical equivalence
3. Once tests pass, the refactor will provide better modularity, configurability, and extensibility while preserving identical behavior.

## Current Status
- Branch: feature/oop-refactor
- Verified OOP classes: src/neuron_group.py and src/projection.py
- Active fep_model.py: original version with L3E->L2I connection (commit e2fda64)
- The OOP refactor is ready for implementation.

This completes the planning and verification phase of the OOP refactor. The user can now proceed with implementation if desired.