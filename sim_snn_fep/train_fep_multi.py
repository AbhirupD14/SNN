import numpy as np
import json
from fep_model import FEPSNN

def main():
    # 1. Initialize SNN and training parameters
    snn = FEPSNN()
    
    # 1. Define the 8 main lines in a 3x3 grid
    patterns = {
        "Row 0 (Top)":    np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),
        "Row 1 (Mid)":    np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
        "Row 2 (Bot)":    np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
        "Col 0 (Left)":   np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
        "Col 1 (Mid)":    np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
        "Col 2 (Right)":  np.array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        "Diag 1 (Main)":  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        "Diag 2 (Anti)":  np.array([0, 0, 1, 0, 1, 0, 1, 0, 0]),
    }
    
    # 2. Training Loop
    epochs = 100
    for epoch in range(epochs):
        # Shuffle patterns to prevent order-bias
        p_list = list(patterns.keys())
        np.random.shuffle(p_list)
        for p_name in p_list:
            snn.process_event_pattern(patterns[p_name])

    # 3. Final Evaluation & Niche Mapping
    print("\nNiche Mapping Evaluation (8 Patterns):")
    print("========================================")
    mapping = {}
    
    # Freeze plasticity for test
    # Note: In our current fep_model, we just don't call the learning logic 
    # or we can just record the winner.
    
    for p_name, p_array in patterns.items():
        # Reset membrane potentials before testing each pattern
        snn.v_e = np.zeros(9)
        snn.v_iff = np.zeros(4)
        
        # We modify the model call slightly or just use the output to see who spikes
        # Since process_event_pattern contains learning, we'll use a "test" mode 
        # by overriding W_GAIN/LOSS temporarily if they were global, 
        # but for now, we'll just record who wins the race.
        trace = snn.process_event_pattern(p_array)
        
        # Find the spike event in the trace
        winner = None
        for step in trace:
            if step['event'] == 'e_spike':
                winner = step['winner']
                break
        
        mapping[p_name] = winner
        print(f"{p_name:<25} -> Claimed by Neuron {winner}")

    # 4. Export Convergence Data
    # We include the binary patterns in the JSON so the HTML doesn't have to hardcode them
    export_data = {
        "weights": snn.weights.tolist(),
        "mapping": {name: int(winner) if winner is not None else None for name, winner in mapping.items()},
        "patterns": {name: arr.tolist() for name, arr in patterns.items()}
    }
    
    with open("/home/adasgup/projects/sim_snn_fep/convergence.json", "w") as f:
        json.dump(export_data, f, indent=4)
    
    print("========================================\nConvergence data written to convergence.json")

if __name__ == "__main__":
    main()
