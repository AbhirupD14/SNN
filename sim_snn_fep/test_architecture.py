import numpy as np
from fep_model import FEPSNN

def main():
    snn = FEPSNN()
    
    # Define 8 distinct patterns (3x3 grid)
    patterns = {
        "Row 0": [0, 1, 2],
        "Row 1": [3, 4, 5],
        "Row 2": [6, 7, 8],
        "Col 0": [0, 3, 6],
        "Col 1": [1, 4, 7],
        "Col 2": [2, 5, 8],
        "Diag 1": [0, 4, 8],
        "Diag 2": [2, 4, 6]
    }
    
    pattern_names = list(patterns.keys())
    pattern_indices = list(patterns.values())
    
    # Training loop
    epochs = 500
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        for indices in pattern_indices:
            snn.process_event(indices)
            
    # Evaluation
    print("\nNiche Mapping Evaluation:")
    print("========================================")
    results = {}
    for name, indices in patterns.items():
        # Clear potentials before test
        snn.v_l1e.fill(0)
        snn.v_l2e.fill(0)
        winner = snn.process_event(indices)
        results[name] = winner
        print(f"{name:12} -> Claimed by Neuron {winner}")

    # Check for Tyrant State
    winners = [v for v in results.values() if v is not None]
    if len(set(winners)) == 1 and len(winners) > 1:
        print("\n!!! ALERT: Tyrant State Detected !!!")
    elif len(set(winners)) == len(winners):
        print("\nSUCCESS: Perfect Symmetry Breaking. Every pattern has a unique neuron.")
    else:
        print(f"\nPartial Convergence: {len(set(winners))}/{len(pattern_names)} neurons specialized.")

if __name__ == "__main__":
    main()
