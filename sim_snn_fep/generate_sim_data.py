import numpy as np
import json
from fep_model import FEPSNN

def generate_simulation_data():
    snn = FEPSNN()
    # Define the 8 line patterns (as used in training)
    patterns = [
        [1,1,1,0,0,0,0,0,0],  # top row
        [0,0,0,1,1,1,0,0,0],  # middle row
        [0,0,0,0,0,0,1,1,1],  # bottom row
        [1,0,0,1,0,0,1,0,0],  # left column
        [0,1,0,0,1,0,0,1,0],  # middle column
        [0,0,1,0,0,1,0,0,1],  # right column
        [1,0,0,0,1,0,0,0,1],  # diagonal \
        [0,0,1,0,1,0,1,0,0],  # diagonal /
        [0,1,0,1,1,1,0,1,0],  # plus sign (vertical + horizontal middle)
        [1,0,1,0,1,0,1,0,1]   # X shape (both diagonals)
    ]
    epochs = 20  # enough to see learning
    print(f'Generating simulation data for {epochs} epochs...')
    for epoch in range(epochs):
        for pattern in patterns:
            active = [i for i, bit in enumerate(pattern) if bit == 1]
            snn.process_event(active, record=True)
        if (epoch+1) % 5 == 0:
            print(f'  Completed epoch {epoch+1}')
    print(f'Total frames recorded: {len(snn._frames)}')
    # Save to JSON
    with open('sim_data.json', 'w') as f:
        json.dump(snn._frames, f)
    print('Saved sim_data.json')

if __name__ == '__main__':
    generate_simulation_data()