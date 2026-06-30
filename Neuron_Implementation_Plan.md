# Neuron Class Implementation Plan

## Requirements Summary
1. Charge accumulation based on weights (positive for exc, negative for inh)
2. Random weight initialization for all neurons
3. Weight updates only occur when neuron fires
4. Weights increase when firing (amount to be decided)
5. Same rules for inhibitory and excitatory neurons (only weight sign differs)
6. Constant firing thresholds
7. Resting potential at 0
8. Refractory period implementation
9. 1ms time steps for granularity

## Class Design

### Neuron Class
Attributes:
- `threshold`: firing threshold (constant)
- `resting_potential`: always 0
- `refractory_period`: time steps neuron cannot fire after spike
- `refractory_timer`: counts down refractory period
- `potential`: current membrane potential
- `weights`: list of synaptic weights (excitatory=positive, inhibitory=negative)
- `last_spike_time`: track when neuron last fired
- `spiked`: boolean for current time step spike status

Methods:
- `__init__()`: initialize with random weights, set resting potential
- `receive_input(input_spikes)`: accumulate charge based on weighted inputs
- `update()`: update state for next time step (refractory, potential decay if needed)
- `check_threshold()`: determine if neuron should fire
- `fire()`: handle spike event (reset potential, start refractory, update weights)
- `update_weights()`: modify weights when firing (only happens when neuron fires)

### Weight Update Rule (to be decided)
Options:
- Fixed increment: weights += learning_rate when firing
- Hebbian-like: weights += learning_rate * pre_synaptic_activity
- STDP-inspired: depends on timing of pre/post spikes
- Homeostatic: weights adjust to maintain target firing rate

For initial implementation, I'll use a simple fixed increment approach where weights increase by a small amount when the neuron fires.

### Implementation Notes
- Excitatory neurons: weights are positive values
- Inhibitory neurons: weights are negative values (same magnitude range)
- All neurons follow same update rules
- Charge accumulation: potential += sum(weight * input_spike)
- Refractory period: potential clamped to resting potential during refractory
- Time step: 1ms as specified