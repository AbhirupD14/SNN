import numpy as np
import json

# --- Configuration Constants ---
# Geometry
N_INPUT = 9
N_E = 9
N_IFF = 4
N_IFB = 1  # Global blanket

# Arithmetic Values
INPUT_CHARGE = 10.0
SHUNT_ABSORPTION_VALUE = 10.0
W_GAIN = 2.0
W_LOSS = 0.5
GLOBAL_BALANCE_FORCE = 1000.0
THETA_UP = 2.0
THETA_DOWN = 1.0
BASELINE_THRESHOLD = 50.0

# Developmental Transition Constants
MAX_CONDUCTANCE = 0.2      # High leak conductance at start
MIN_CONDUCTANCE = 0.0      # Zero conductance at maturity
MATURITY_THRESHOLD = 400.0 # Total weight sum for maturation
RESTING_POTENTIAL = 0.0    # E_L: The value V leaks toward

class FEPSNN:
    def __init__(self):
        # State Variables
        self.v_e = np.zeros(N_E)
        self.theta_e = np.full(N_E, BASELINE_THRESHOLD)
        # Moderate initial symmetry breaking to prevent "killing" half the neurons
        self.theta_e = np.full(N_E, BASELINE_THRESHOLD) + np.random.uniform(-10, 10, N_E)
        
        self.weights = np.random.uniform(10.0, 50.0, (N_E, N_INPUT))
        
        self.v_iff = np.zeros(N_IFF)
        self.theta_iff = np.full(N_IFF, BASELINE_THRESHOLD * 0.5)

    def _capture_state(self, event_type, input_pattern, winner=None):
        return {
            "event": event_type,
            "input": input_pattern.tolist(),
            "v_e": self.v_e.tolist(),
            "theta_e": self.theta_e.tolist(),
            "weights": self.weights.tolist(),
            "winner": winner
        }

    def process_event_pattern(self, active_indices):
        """
        active_indices: A list or array of indices of neurons that spiked.
        Example: [0, 2, 5] instead of [1, 0, 1, 0, 0, 1...]
        """
        trace = []
        
        # For tracing, we'll convert indices back to a pattern just for the log
        input_pattern = np.zeros(N_INPUT)
        input_pattern[active_indices] = 1.0
        
        # Capture Initial State
        trace.append(self._capture_state("initial", input_pattern))
        
        # --- STEP 0: Developmental Leakage (LIF -> Pure Integrator) ---
        # Leakage is driven by conductance g_L. g_L is high when weights are low.
        weight_sums = np.sum(self.weights, axis=1)
        
        # Calculate current conductance g_L for each neuron based on maturity
        g_l = np.clip(
            MAX_CONDUCTANCE * (1.0 - (weight_sums / MATURITY_THRESHOLD)), 
            MIN_CONDUCTANCE, 
            MAX_CONDUCTANCE
        )
        
        # LIF Leakage: dV = g_L * (V - E_L)
        # We subtract this leakage current from the current membrane potential
        leakage_current = g_l * (self.v_e - RESTING_POTENTIAL)
        self.v_e -= leakage_current
        
        # Ensure V doesn't drop below resting potential
        self.v_e = np.maximum(self.v_e, RESTING_POTENTIAL)
        
        # --- STEP A: Local Free Energy Assessment & Input Drive ---
        if len(active_indices) > 0:

            drive = np.sum(self.weights[:, active_indices], axis=1)
            base_charge = len(active_indices) * INPUT_CHARGE
            self.v_e += (drive + base_charge)
        
        # --- STEP B: Feedforward Absorption (I_FF Shunting Gate) ---
        iff_drive = len(active_indices) * (INPUT_CHARGE / N_INPUT)
        self.v_iff += iff_drive
        
        spiking_iff = np.sum(self.v_iff >= self.theta_iff)
        self.v_e -= (spiking_iff * SHUNT_ABSORPTION_VALUE)
        self.v_e = np.maximum(self.v_e, 0)
        
        # Metrics for I_FF
        iff_metrics = {
            "v_iff": self.v_iff.tolist(),
            "spikes": int(spiking_iff),
            "drive": float(iff_drive)
        }
        
        # Reset I_FF
        self.v_iff = np.zeros(N_IFF)
        
        # --- STEP C: Evaluation & Learning ---
        winner = np.argmax(self.v_e)
        
        if self.v_e[winner] >= self.theta_e[winner]:
            # 1. Fire the neuron
            self.v_e[winner] = 0
            self.theta_e[winner] += THETA_UP
            
            # 2. Event-Driven Learning (FEP)
            self.weights[winner, active_indices] += W_GAIN
            
            # Zero-Sum Structural Constraint
            current_sum = np.sum(self.weights[winner])
            target_sum = self.theta_e[winner]
            diff = (current_sum - target_sum) / N_INPUT
            self.weights[winner] -= diff
            
            # --- STEP D: Feedback Balance (I_FB) ---
            # Metrics for I_FB: we treat this as a virtual population spike
            ifb_metrics = {
                "force_applied": float(GLOBAL_BALANCE_FORCE),
                "triggered": True
            }
            self.v_e -= GLOBAL_BALANCE_FORCE
            self.v_e = np.maximum(self.v_e, 0) 
            
            # Symmetry Breaking: Penalty to winner
            self.theta_e[winner] += 25.0 
            
            # Merge inhibitory metrics into the trace
            state = self._capture_state("e_spike", input_pattern, winner=winner)
            state["iff_metrics"] = iff_metrics
            state["ifb_metrics"] = ifb_metrics
            trace.append(state)
        else:
            # No one fired
            self.theta_e = np.maximum(self.theta_e - THETA_DOWN, BASELINE_THRESHOLD)
            state = self._capture_state("no_spike", input_pattern)
            state["iff_metrics"] = iff_metrics
            state["ifb_metrics"] = {"force_applied": 0.0, "triggered": False}
            trace.append(state)
            
        return trace
