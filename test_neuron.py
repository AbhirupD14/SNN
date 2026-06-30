"""
Test script for Neuron class implementation
"""
import numpy as np
from neuron import Neuron

def test_neuron_basics():
    """Test basic neuron functionality"""
    print("Testing Neuron class...")
    
    # Create a neuron with 5 inputs
    neuron = Neuron(n_inputs=5, threshold=1.0, refractory_period=2)
    
    print(f"Initial weights: {neuron.weights}")
    print(f"Initial potential: {neuron.potential}")
    print(f"Threshold: {neuron.threshold}")
    print(f"Refractory period: {neuron.refractory_period}")
    print(f"Leak rate: {neuron.leak_rate}")
    
    # Test receiving input
    input_spikes = np.array([1, 0, 1, 0, 1])  # Some inputs spike
    neuron.receive_input(input_spikes)
    print(f"After input - potential: {neuron.potential}")
    
    # Test threshold checking
    should_fire = neuron.check_threshold()
    print(f"Should fire: {should_fire}")
    
    if should_fire:
        neuron.fire()
        print(f"After firing - potential: {neuron.potential}")
        print(f"After firing - refractory timer: {neuron.refractory_timer}")
        print(f"Weights after firing: {neuron.weights}")
        
    # Test refractory period
    print("\nTesting refractory period...")
    for i in range(5):
        neuron.update()
        print(f"Time step {i+1}: potential={neuron.potential:.3f}, refractory={neuron.refractory_timer}, spiked={neuron.spiked}")
        
    # Test that weights only update when firing
    print("\nTesting weight update rule...")
    initial_weights = neuron.weights.copy()
    
    # Give input but don't let it fire (keep potential low)
    neuron.potential = 0.0
    neuron.receive_input(np.array([1, 1, 1, 1, 1]))
    print(f"Potential after input (no fire): {neuron.potential}")
    
    # Update without firing
    neuron.update()
    print(f"Weights after update (no fire): {neuron.weights}")
    weights_unchanged = np.allclose(neuron.weights, initial_weights)
    print(f"Weights unchanged when not firing: {weights_unchanged}")
    
    # Now make it fire and check weights change
    neuron.potential = 2.0  # Above threshold
    if neuron.check_threshold():
        neuron.fire()
        print(f"Weights after firing: {neuron.weights}")
        weights_changed = not np.allclose(neuron.weights, initial_weights)
        print(f"Weights changed when firing: {weights_changed}")
    
    print("\nBasic neuron tests completed!")

def test_leak_functionality():
    """Test that the leak functionality works correctly"""
    print("\nTesting leak functionality...")
    
    # Create neuron with significant leak to observe the effect
    neuron = Neuron(n_inputs=1, threshold=1.0, refractory_period=1, leak_rate=0.5)
    neuron.weights = np.array([1.0])  # Weight of 1.0
    
    print(f"Leak rate: {neuron.leak_rate}")
    print(f"Weight: {neuron.weights[0]}")
    
    # Start with some potential
    neuron.potential = 1.0
    print(f"Initial potential: {neuron.potential}")
    
    # Apply leak over several time steps with no input
    for i in range(5):
        neuron.update()  # No input, just leak
        print(f"After update {i+1}: potential = {neuron.potential:.3f}")
        
        # With leak_rate=0.5, potential should halve each step toward resting (0)
        # Step 0: 1.0
        # Step 1: 1.0 + 0.5*(0-1.0) = 1.0 - 0.5 = 0.5
        # Step 2: 0.5 + 0.5*(0-0.5) = 0.5 - 0.25 = 0.25
        # etc.
    
    # Test that leak works with input too
    print("\nTesting leak with input...")
    neuron.potential = 0.0
    # Apply input that would raise potential to 0.8 without leak
    neuron.receive_input(np.array([0.8]))  # weight * input = 1.0 * 0.8 = 0.8
    print(f"After input (before leak): {neuron.potential}")
    
    # Now apply update with leak
    neuron.update()
    # With leak: 0.8 + 0.5*(0-0.8) = 0.8 - 0.4 = 0.4
    print(f"After update (with leak): {neuron.potential:.3f}")
    
    expected = 0.4
    if abs(neuron.potential - expected) < 0.001:
        print("Leak calculation correct!")
    else:
        print(f"Leak calculation incorrect. Expected ~{expected}, got {neuron.potential}")
    
    print("\nLeak test completed!")

def test_weight_cap():
    """Test that weights are properly capped"""
    print("\nTesting weight cap functionality...")
    
    # Create neuron with low weight cap
    neuron = Neuron(n_inputs=3, threshold=0.1, learning_rate=0.3, weight_cap=0.5)
    
    print(f"Initial weights: {neuron.weights}")
    print(f"Weight cap: {neuron.weight_cap}")
    
    # Fire multiple times to test weight cap
    for i in range(5):
        # Make sure neuron fires
        neuron.potential = 1.0  # Well above threshold
        if neuron.check_threshold():
            neuron.fire()
        neuron.update()  # Clear refractory
    
    print(f"Weights after 5 firings: {neuron.weights}")
    
    # Check that no weight exceeds the cap
    max_weight = np.max(np.abs(neuron.weights))
    within_cap = max_weight <= neuron.weight_cap + 1e-10  # Small tolerance for floating point
    print(f"Max absolute weight: {max_weight}")
    print(f"Weights within cap: {within_cap}")
    
    return within_cap

def test_inhibitory_neuron():
    """Test that inhibitory neurons work with negative weights"""
    print("\nTesting inhibitory neuron concept...")
    
    # Create neuron
    neuron = Neuron(n_inputs=3, threshold=0.5, refractory_period=1, weight_cap=1.0)
    
    # Manually set some weights to negative (simulating inhibitory inputs)
    neuron.weights = np.array([-0.3, -0.2, -0.1])
    print(f"Inhibitory weights: {neuron.weights}")
    
    # Test that negative weights subtract from potential
    input_spikes = np.array([1, 1, 1])  # All inhibitory inputs active
    neuron.receive_input(input_spikes)
    print(f"Potential after inhibitory input: {neuron.potential}")
    
    # Test weight update still works (becomes more negative, but capped)
    initial_weights = neuron.weights.copy()
    neuron.potential = 1.0  # Above threshold to make it fire
    if neuron.check_threshold():
        neuron.fire()
        print(f"Weights after firing (should be more negative): {neuron.weights}")
        print(f"Weight change: {neuron.weights - initial_weights}")
    
    print("Inhibitory neuron test completed!")

if __name__ == "__main__":
    test_neuron_basics()
    test_leak_functionality()
    test_weight_cap()
    test_inhibitory_neuron()