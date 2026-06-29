import numpy as np
from fep_model import FEPSNN, N_L1_E, N_L2_E, LEAK_FRACTION, MATURITY_THRESHOLD, RESTING_POTENTIAL, T_STEPS, BASELINE_THETA

def test_leakage_transition():
    print("--- Starting Leakage Transition Test ---")
    snn = FEPSNN()
    
    # 1. Test Nascent State (High Leak)
    # Manually set weights to 0 to ensure no input drive
    snn.weights_l1e_l2e = np.zeros((N_L2_E, N_L1_E))
    # Set threshold high so the neuron doesn't fire
    snn.theta_l2e = np.full(N_L2_E, 200.0)
    # Set initial voltage to 100 via adapt (since v = RESTING - adapt)
    snn.adapt[:] = RESTING_POTENTIAL - 100.0  # adapt = -100 gives v = 0 - (-100) = 100
    # Ensure no refractory or other states
    snn.refrac[:] = 0
    snn.silence[:] = 0
    
    # Process an empty event (just to trigger the leak steps)
    snn.process_event_pattern([])
    
    v_after_leak = snn.v_l2e[0]
    # Expected after T_STEPS leak applications: V0 * (1-LEAK)^T_STEPS
    expected_v = 100.0 * ((1.0 - LEAK_FRACTION) ** T_STEPS)
    
    print(f"Nascent state V: {v_after_leak:.2f} (Expected: {expected_v:.2f})")
    assert np.isclose(v_after_leak, expected_v, atol=1.0), f"Nascent leak failed: {v_after_leak} != {expected_v}"
    print("✅ Nascent state (LIF) leak verified.")
    
    # 2. Test that with high weights and a full pattern, neuron fires (winner not None)
    # Reset adapt etc.
    snn.adapt[:] = 0.0
    snn.refrac[:] = 0
    snn.silence[:] = 0
    # Set weights high enough that one volley drives neuron to threshold.
    snn.weights_l1e_l2e = np.full((N_L2_E, N_L1_E), MATURITY_THRESHOLD / N_L1_E + 10.0)
    snn.theta_l2e = np.full(N_L2_E, BASELINE_THETA)  # use baseline threshold
    # Present a pattern that activates all L1_E neurons (max drive)
    pattern = [1]*N_L1_E
    winner = snn.process_event_pattern(pattern)
    print(f"Winner with high weights and full pattern: {winner}")
    assert winner is not None, "Expected a winner with high weights and full pattern"
    print("✅ High weights produce firing (winner not None).")
    
    # 3. Test Intermediate State (leak still works)
    # Reset adapt etc.
    snn.adapt[:] = 0.0
    snn.refrac[:] = 0
    snn.silence[:] = 0
    # Set weights to exactly half of maturity
    snn.weights_l1e_l2e = np.full((N_L2_E, N_L1_E), (MATURITY_THRESHOLD / 2) / N_L1_E)
    snn.theta_l2e = np.full(N_L2_E, 200.0)  # high threshold to avoid firing
    snn.adapt[:] = RESTING_POTENTIAL - 100.0  # start at 100 mV
    
    snn.process_event_pattern([])
    
    v_inter = snn.v_l2e[0]
    expected_inter = 100.0 * ((1.0 - LEAK_FRACTION) ** T_STEPS)
    
    print(f"Intermediate state V: {v_inter:.2f} (Expected: {expected_inter:.2f})")
    assert np.isclose(v_inter, expected_inter, atol=1.0), f"Intermediate leak failed: {v_inter} != {expected_inter}"
    print("✅ Intermediate state leak verified.")
    
    print("--- All Leakage Tests Passed! ---")

if __name__ == '__main__':
    test_leakage_transition()