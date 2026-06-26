import numpy as np
import json

# --- Configuration Constants ---
# Populations
N_L1_E = 9
N_L1_I = 9
N_L2_E = 8
N_L2_I = 1

# Synaptic Weights
W_L1E_L1I = -5.0  # L1_E excites L1_I, but L1_I inhibits L1_E (we'll handle signs in logic)
W_L1I_L1E = -10.0 # Local L1 Inhibition
W_L1E_L2E_MIN = 10.0
W_L1E_L2E_MAX = 50.0
W_L2E_L2I = 20.0   # L2_E excites L2_I
W_L2I_L2E = -100.0 # Global L2 Inhibition (The Brake)
W_L2E_L1I = 15.0   # Feedback to L1_I (Squelch)

# Thresholds & Dynamics
BASELINE_THETA = 50.0
THETA_UP = 2.0
THETA_DOWN = 1.0
W_GAIN = 2.0
LEAK_RATE = 0.1

class FEPSNN:
    def __init__(self):
        # --- State Variables ---
        # Layer 1
        self.v_l1e = np.zeros(N_L1_E)
        self.theta_l1e = np.full(N_L1_E, BASELINE_THETA)
        self.v_l1i = np.zeros(N_L1_I)
        self.theta_l1i = np.full(N_L1_I, BASELINE_THETA * 0.5)
        
        # Layer 2
        self.v_l2e = np.zeros(N_L2_E)
        self.theta_l2e = np.full(N_L2_E, BASELINE_THETA) + np.random.uniform(-5, 5, N_L2_E)
        self.v_l2i = np.zeros(N_L2_I)
        self.theta_l2i = np.full(N_L2_I, BASELINE_THETA * 0.3)
        
        # --- Connectivity ---
        # L1_E -> L2_E (The learned weights)
        self.weights_l1e_l2e = np.random.uniform(W_L1E_L2E_MIN, W_L1E_L2E_MAX, (N_L2_E, N_L1_E))
        
        # L2_E -> L1_I (Feedback squelch)
        # Initially random, but we'll keep them dense
        self.weights_l2e_l1i = np.random.uniform(0, W_L2E_L1I, (N_L1_I, N_L2_E))

    def process_event(self, active_indices):
        """
        Biological sequence:
        1. External Input -> L1_E
        2. L1_E <-> L1_I (Local competition)
        3. L1_E -> L2_E (Pattern integration)
        4. L2_E -> L2_I -> L2_E (Global winner-take-all)
        5. L2_E (Winner) -> L1_I (Source squelch)
        """
        # --- Step 1: External Stimulus ---
        # No base_charge. Pure synaptic input.
        self.v_l1e[active_indices] += 50.0 
        
        # --- Step 2: L1 Local Dynamics ---
        # L1_E -> L1_I
        for i in range(N_L1_E):
            if self.v_l1e[i] >= self.theta_l1e[i]:
                self.v_l1i[i] += 20.0 # Excite paired inhibitor
        
        # L1_I -> L1_E (Squelch)
        for i in range(N_L1_I):
            if self.v_l1i[i] >= self.theta_l1i[i]:
                self.v_l1e[i] += W_L1I_L1E
                self.v_l1i[i] = 0 # Reset
        
        # --- Step 3: L1_E -> L2_E ---
        # Only L1 neurons that are currently 'active' (above some threshold) drive L2
        l1_active = np.where(self.v_l1e > 10)[0]
        if len(l1_active) > 0:
            drive = np.sum(self.weights_l1e_l2e[:, l1_active], axis=1)
            self.v_l2e += drive

        # --- Step 4: L2 Winner-Take-All ---
        # Instead of "first to hit threshold," we find the neuron with the 
        # highest "resonance" (voltage relative to threshold) to ensure the 
        # most conceptually aligned neuron wins.
        
        resonance = self.v_l2e / self.theta_l2e
        winner = np.argmax(resonance)
        
        if resonance[winner] >= 1.0:
            # Trigger L2_I Global Brake
            self.v_l2i += W_L2E_L2I
            if self.v_l2i[0] >= self.theta_l2i[0]:
                self.v_l2e += W_L2I_L2E
                self.v_l2i[0] = 0
            
            # 1. Hebbian update: Strengthen alignment
            self.weights_l1e_l2e[winner, l1_active] += W_GAIN
            
            # 2. Soft Weight Divorce: Prevent "Generalist" Tyrants
            # Gently weaken the winner's connections to the active set 
            # so that it must continually specialize to stay the winner.
            divorce_factor = 0.1 
            self.weights_l1e_l2e[winner, l1_active] *= (1.0 - divorce_factor)
            
            # 3. Zero-Sum Structural Constraint
            self.weights_l1e_l2e[winner] -= (np.sum(self.weights_l1e_l2e[winner]) - self.theta_l2e[winner]) / N_L1_E
            
            # 4. Symmetry Breaking: Fatigue the winner
            self.theta_l2e[winner] += THETA_UP
            
            # --- Step 5: L2_E (Winner) -> L1_I (Source Squelch) ---
            self.v_l1i += self.weights_l2e_l1i[:, winner]
            
            self.v_l2e[winner] = 0
            return winner
        else:
            # Homeostatic scaling: Lower thresholds for the starved
            self.theta_l2e = np.maximum(self.theta_l2e - THETA_DOWN, BASELINE_THETA * 0.5)
            return None

        # Decay potentials
        self.v_l1e *= (1 - LEAK_RATE)
        self.v_l2e *= (1 - LEAK_RATE)
