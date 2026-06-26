
import numpy as np
from fep_model import FEPSNN, MAX_CONDUCTANCE, MATURITY_THRESHOLD, RESTING_POTENTIAL

def test_leakage_transition():
    print("--- Starting Leakage Transition Test ---")
    snn = FEPSNN()
    
    # 1. Test Nascent State (High Leak)
    # Manually set weights to 0 to ensure MAX_CONDUCTANCE
    snn.weights = np.zeros((9, 9))
    snn.v_e = np.full(9, 100.0)
    # Set threshold high so the neuron doesn't fire and reset V to 0
    snn.theta_e = np.full(9, 200.0)
    
    # Process an empty event (just to trigger the leak step)
    snn.process_event_pattern([])
    
    v_after_leak = snn.v_e[0]
    expected_leak = MAX_CONDUCTANCE * (100.0 - RESTING_POTENTIAL)
    expected_v = 100.0 - expected_leak
    
    print(f"Nascent state V: {v_after_leak:.2f} (Expected: {expected_v:.2f})")
    assert np.isclose(v_after_leak, expected_v), f"Nascent leak failed: {v_after_leak} != {expected_v}"
    print("✅ Nascent state (LIF) leak verified.")

    # 2. Test Mature State (No Leak)
    # Manually set weights above MATURITY_THRESHOLD
    snn.weights = np.full((9, 9), MATURITY_THRESHOLD / 9 + 10.0)
    snn.v_e = np.full(9, 100.0)
    snn.theta_e = np.full(9, 200.0)
    
    snn.process_event_pattern([])
    
    v_mature = snn.v_e[0]
    print(f"Mature state V: {v_mature:.2f} (Expected: 100.00)")
    assert np.isclose(v_mature, 100.0), f"Mature leak failed: {v_mature} != 100.0"
    print("✅ Mature state (Pure Integrator) leak verified.")

    # 3. Test Intermediate State
    # Set weights to exactly half of maturity
    snn.weights = np.full((9, 9), (MATURITY_THRESHOLD / 2) / 9)
    snn.v_e = np.full(9, 100.0)
    snn.theta_e = np.full(9, 200.0)
    
    snn.process_event_pattern([])
    
    v_inter = snn.v_e[0]
    # g_l should be MAX_CONDUCTANCE * (1.0 - 0.5) = 0.5 * MAX_CONDUCTANCE
    expected_g_l = MAX_CONDUCTANCE * 0.5
    expected_v_inter = 100.0 - (expected_g_l * (100.0 - RESTING_POTENTIAL))
    
    print(f"Intermediate state V: {v_inter:.2f} (Expected: {expected_v_inter:.2f})")
    assert np.isclose(v_inter, expected_v_inter), f"Intermediate leak failed: {v_inter} != {expected_v_inter}"
    print("✅ Intermediate state leak verified.")

    print("--- All Leakage Tests Passed! ---")

if __name__ == '__main__':
    test_leakage_transition()
