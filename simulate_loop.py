"""
Simulation of a simple loop: E1 <- I <- E2 <- E1
Where:
- E1: Excitatory neuron 1
- I: Inhibitory neuron  
- E2: Excitatory neuron 2

Connections:
- I -> E1 (inhibitory)
- E2 -> I (excitatory) 
- E1 -> E2 (excitatory)

We'll present a spike to E1 and observe the looping behavior.
"""

import numpy as np
from neuron import Neuron

def simulate_loop():
    """Simulate the E1 <- I <- E2 <- E1 loop"""
    print("Setting up E1 <- I <- E2 <- E1 loop...")
    
    # Create neurons
    # E1 receives input from I (inhibitory) and external
    E1 = Neuron(n_inputs=2, threshold=1.5, refractory_period=2, learning_rate=0.05, weight_cap=5.0, leak_rate=0.01)
    
    # I receives input from E2 (excitatory)
    I = Neuron(n_inputs=1, threshold=1.0, refractory_period=2, learning_rate=0.05, weight_cap=5.0, leak_rate=0.01)
    
    # E2 receives input from E1 (excitatory)
    E2 = Neuron(n_inputs=1, threshold=1.5, refractory_period=2, learning_rate=0.05, weight_cap=5.0, leak_rate=0.01)
    
    # Set up weights to create strong connections
    # E1: [weight_from_I, weight_from_external]
    E1.weights = np.array([-3.0, 3.0])  # Strong inhibitory from I, strong excitatory external
    
    # I: [weight_from_E2] 
    I.weights = np.array([3.0])  # Strong excitatory from E2
    
    # E2: [weight_from_E1]
    E2.weights = np.array([3.0])  # Strong excitatory from E1
    
    print(f"Initial weights:")
    print(f"  E1 weights [I, external]: {E1.weights}")
    print(f"  I weights [E2]: {I.weights}")  
    print(f"  E2 weights [E1]: {E2.weights}")
    print(f"  Thresholds - E1: {E1.threshold}, I: {I.threshold}, E2: {E2.threshold}")
    print(f"  Leak rates - E1: {E1.leak_rate}, I: {I.leak_rate}, E2: {E2.leak_rate}")
    print()
    
    # Storage for monitoring
    history = {
        'E1_potential': [], 'E1_spiked': [],
        'I_potential': [], 'I_spiked': [],
        'E2_potential': [], 'E2_spiked': [],
        'E1_weights': [], 'I_weights': [], 'E2_weights': []
    }
    
    # Run simulation for 50 time steps (50ms)
    print("Running simulation...")
    for t in range(50):
        # Present external spike to E1 at t=0
        external_spike = 1 if t == 0 else 0
        
        # Prepare inputs for each neuron
        # E1 gets: [I_spike, external_spike]
        # I gets: [E2_spike] 
        # E2 gets: [E1_spike]
        
        E1_input = np.array([0, external_spike])  # Will update with actual I spike below
        I_input = np.array([0])                   # Will update with actual E2 spike below
        E2_input = np.array([0])                  # Will update with actual E1 spike below
        
        # Get spike states from end of previous time step
        # For t=0, use initial conditions (no spikes yet)
        if t == 0:
            prev_E1_spiked = False
            prev_I_spiked = False  
            prev_E2_spiked = False
        else:
            prev_E1_spiked = history['E1_spiked'][-1]
            prev_I_spiked = history['I_spiked'][-1]
            prev_E2_spiked = history['E2_spiked'][-1]
        
        # Set inputs based on previous time step spikes
        E1_input = np.array([
            -1.0 if prev_I_spiked else 0.0,  # I->E1: inhibitory (-1 when I spiked)
            external_spike                    # External input (weight handled by E1.weights[1])
        ])
        I_input = np.array([
            1.0 if prev_E2_spiked else 0.0   # E2->I: excitatory (1 when E2 spiked)
        ])
        E2_input = np.array([
            1.0 if prev_E1_spiked else 0.0   # E1->E2: excitatory (1 when E1 spiked)
        ])
        
        # Step 1: Receive inputs
        E1.receive_input(E1_input)
        I.receive_input(I_input)
        E2.receive_input(E2_input)
        
        # Step 2: Check thresholds and fire
        E1_spiked_this_step = E1.check_threshold()
        I_spiked_this_step = I.check_threshold()
        E2_spiked_this_step = E2.check_threshold()
        
        if E1_spiked_this_step:
            E1.fire()
        if I_spiked_this_step:
            I.fire()
        if E2_spiked_this_step:
            E2.fire()
            
        # Step 3: Record state before update
        history['E1_potential'].append(E1.potential)
        E1_spiked_record = E1.spiked  # This will be True if fired this step
        history['E1_spiked'].append(E1_spiked_record)
        history['E1_weights'].append(E1.weights.copy())
        
        history['I_potential'].append(I.potential)
        I_spiked_record = I.spiked
        history['I_spiked'].append(I_spiked_record)
        history['I_weights'].append(I.weights.copy())
        
        history['E2_potential'].append(E2.potential)
        E2_spiked_record = E2.spiked
        history['E2_spiked'].append(E2_spiked_record)
        history['E2_weights'].append(E2.weights.copy())
        
        # Step 4: Update all neurons (handle refractory, etc.)
        E1.update()
        I.update()
        E2.update()
        
        # Print progress every 10 steps
        if t % 10 == 0 or t < 5:
            print(f"t={t:2d}: E1(V={E1.potential:5.2f}, spiked={E1_spiked_record}), "
                  f"I(V={I.potential:5.2f}, spiked={I_spiked_record}), "
                  f"E2(V={E2.potential:5.2f}, spiked={E2_spiked_record})")
    
    print("\nSimulation complete.")
    
    # Analyze results
    print("\n=== Activity Summary ===")
    E1_spike_count = sum(history['E1_spiked'])
    I_spike_count = sum(history['I_spiked']) 
    E2_spike_count = sum(history['E2_spiked'])
    print(f"E1 spikes: {E1_spike_count}")
    print(f"I spikes:  {I_spike_count}")
    print(f"E2 spikes: {E2_spike_count}")
    
    # Show final weights
    print(f"\n=== Final Weights ===")
    print(f"E1 weights [I, external]: {history['E1_weights'][-1]}")
    print(f"I weights [E2]:           {history['I_weights'][-1]}")
    print(f"E2 weights [E1]:          {history['E2_weights'][-1]}")
    
    # Show first few time steps in detail for debugging
    print(f"\n=== First 5 Time Steps Detail ===")
    for t in range(min(5, len(history['E1_potential']))):
        print(f"t={t}: E1(V={history['E1_potential'][t]:5.2f}, s={history['E1_spiked'][t]}), "
              f"I(V={history['I_potential'][t]:5.2f}, s={history['I_spiked'][t]}), "
              f"E2(V={history['E2_potential'][t]:5.2f}, s={history['E2_spiked'][t]})")
    
    # Determine if we got sustained activity
    if E1_spike_count > 1 or I_spike_count > 1 or E2_spike_count > 1:
        print("\n✓ Loop produced sustained/spiking activity!")
    else:
        print("\n✗ Loop did not sustain activity (may have died out)")
        
    return history

if __name__ == "__main__":
    simulate_loop()