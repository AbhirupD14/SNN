import numpy as np
import json
from fep_model import FEPSNN

def train_and_export_trace():
    snn = FEPSNN()
    patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
        [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]
    ]
    
    epochs = 100
    full_history = []

    print(f"Training for {epochs} epochs to capture transition...")

    for epoch in range(epochs):
        epoch_data = {
            "epoch": epoch,
            "events": []
        }
        
        for p_idx, pattern in enumerate(patterns):
            trace = snn.process_event_pattern(pattern)
            
            # We only care about the outcome of the event for the big trace
            final_step = trace[-1]
            
            # Sanitize numpy types
            def sanitize(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.int64, np.int32, np.int8)): return int(obj)
                if isinstance(obj, (np.float64, np.float32)): return float(obj)
                if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list): return [sanitize(i) for i in obj]
                return obj

            event_summary = {
                "pattern_idx": p_idx,
                "pattern": pattern,
                "event": final_step['event'],
                "winner": sanitize(final_step.get('winner')),
                "v_e": sanitize(final_step['v_e']),
                "theta_e": sanitize(final_step['theta_e']),
                "iff_spikes": sanitize(final_step.get('iff_metrics', {}).get('spikes', 0)),
                "ifb_triggered": sanitize(final_step.get('ifb_metrics', {}).get('triggered', False))
            }
            epoch_data["events"].append(event_summary)
            
        full_history.append(epoch_data)

    # Export to JSON for the HTML frontend
    with open('training_history.json', 'w') as f:
        json.dump(full_history, f)
    
    print("Training complete. History exported to training_history.json")

if __name__ == "__main__":
    train_and_export_trace()
