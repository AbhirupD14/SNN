"""
Minimal test to debug L1 E neuron firing
"""
import numpy as np
from neuron_flexible import Neuron

# Create a single L1 E neuron with the same parameters
neuron = Neuron(threshold=0.15, refractory_period=2, learning_rate=0.12, weight_cap=1.0, leak_rate=0.008)

# Set up connections: [from_I, from_external] 
neuron.add_input_connection(-0.08)  # inhibitory weight
neuron.add_input_connection(0.9)    # external weight
neuron.finalize_connections()

print(f"Neuron weights: {neuron.weights}")
print(f"Threshold: {neuron.threshold}")
print(f"Initial potential: {neuron.potential}")
print(f"Initial refractory_timer: {neuron.refractory_timer}")

# Simulate receiving input: [I_spike, external_spike] = [0.0, 1.0] 
input_spikes = np.array([0.0, 1.0])
print(f"\nInput spikes: {input_spikes}")

# Receive input
neuron.receive_input(input_spikes)
print(f"After receive_input - potential: {neuron.potential}")

# Check threshold
should_fire = neuron.check_threshold()
print(f"Check threshold: {should_fire} (potential {neuron.potential} >= threshold {neuron.threshold})")

if should_fire:
    neuron.fire()
    print(f"After fire - potential: {neuron.potential}, spiked: {neuron.spiked}, refractory_timer: {neuron.refractory_timer}")
    
    # Update
    neuron.update()
    print(f"After update - potential: {neuron.potential}, spiked: {neuron.spiked}, refractory_timer: {neuron.refractory_timer}")

# Try again next time step (should be in refractory period)
print(f"\n--- Next time step ---")
print(f"Before receive_input - potential: {neuron.potential}, refractory_timer: {neuron.refractory_timer}")
neuron.receive_input(input_spikes)
print(f"After receive_input - potential: {neuron.potential}")
should_fire = neuron.check_threshold()
print(f"Check threshold: {should_fire}")
if should_fire:
    neuron.fire()
neuron.update()
print(f"After update - potential: {neuron.potential}, refractory_timer: {neuron.refractory_timer}")

# And again
print(f"\n--- Next time step ---")
print(f"Before receive_input - potential: {neuron.potential}, refractory_timer: {neuron.refractory_timer}")
neuron.receive_input(input_spikes)
print(f"After receive_input - potential: {neuron.potential}")
should_fire = neuron.check_threshold()
print(f"Check threshold: {should_fire}")
if should_fire:
    neuron.fire()
neuron.update()
print(f"After update - potential: {neuron.potential}, refractory_timer: {neuron.refractory_timer}")