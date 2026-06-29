import numpy as np
from fep_model import FEPSNN

def test_l3_basic():
    print("Testing basic L3 functionality...")
    snn = FEPSNN()
    
    # Test that we can create the network without errors
    print(f"L2_E neurons: {snn.v_l2e.shape[0]}")
    print(f"L3_E neurons: {snn.v_l3e.shape[0]}")
    print(f"L2E->L3E weights shape: {snn.weights_l2e_l3e.shape}")
    print(f"L3E->L3I weights shape: {snn.weights_l3e_l3i.shape}")
    print(f"L3I->L3E weights shape: {snn.weights_l3i_l3e.shape}")
    
    # Test a simple pattern
    pattern = [0, 1, 2, 3, 4]  # Some arbitrary pattern
    active = [i for i in range(9) if i < len(pattern)]  # Simple mapping
    
    print(f"Testing with pattern: {pattern}")
    winner = snn.process_event(active)
    print(f"Winner: {winner}")
    
    # Test with record=True to see if we get frames
    trace = snn.process_event_pattern([1, 0, 1, 0, 1, 0, 1, 0, 1])  # Alternating pattern
    print(f"Trace length: {len(trace)}")
    if trace:
        print(f"First frame keys: {list(trace[0].keys())}")
        # Check if L3 fields are present
        l3_fields = ['l3_fired', 'l3_brake', 'l3_free_energy', 'v_l3e', 'v_l3i']
        present_fields = [f for f in l3_fields if f in trace[0]]
        print(f"L3 fields present: {present_fields}")
    
    print("L3 basic test completed successfully!")

if __name__ == "__main__":
    test_l3_basic()