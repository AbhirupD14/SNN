"""
fep_model.py  --  Biologically-grounded spiking model for symbol consolidation.

Design commitments (hard constraints):
  * As biological as possible. No gradients, no differential-equation solvers,
    no linear-algebra as the *mechanism*. numpy arrays are only storage; every
    rule below is an explicit per-neuron / per-synapse event.
  * No supervision and nothing supervision-like. No label, teacher, error signal
    or global objective. The winner is whoever crosses a FIXED threshold first in
    a real spiking race -- never argmax.
  * The learning algorithm lives inside each neuron; behaviour is emergent.
  * Plasticity is per-synapse.

The behavioural picture (LIF -> pure integrator), exactly as intended:

  Every neuron has a FIXED firing threshold and a FIXED membrane leak. Its
  incoming synapses are *gates* whose conductance widens with demand (the
  ion-channel analogy). During a presentation the stimulus arrives as a *burst*
  of repeated input volleys; the leaky membrane integrates them.

  * Nascent: gates are small, so each volley adds little. Against the fixed leak
    the cell needs many arrivals (a long burst / many exposures) before charge
    reaches threshold and it fires -- "we keep seeing it and slowly learn it".
    This is Leaky-Integrate-and-Fire behaviour.
  * Firing widens the gates that drove it. With repeated exposure the gates grow
    until the summed conductance of a SINGLE volley already meets the threshold,
    so the cell fires on the first arrival. That end-state is the "pure
    integrator": one spike from each input that matters fires it at once.

  The leak is never switched off -- it stays fixed. The whole LIF -> integrator
  transition is carried by gate growth crossing the point where one volley == theta.

  Competition / anti-tyranny come from intrinsic biophysics, NOT a moving
  threshold:
    * an after-hyperpolarisation (AHP / spike-frequency-adaptation) current builds
      with each spike and hyperpolarises the cell, then slowly recovers -- a
      frequent firer tires and yields the floor (this subtracts charge from the
      cell itself);
    * a refractory period silences a cell just after it fires;
    * the winner drives a global inhibitor that quenches any co-firing rival
      (subtracting charge from the others);
    * a fixed synaptic budget (synaptic scaling) stops any cell from holding
      strong gates to everything, so a generalist's single-volley drive always
      stays below threshold -- only specialists fire fast;
    * a chronically silent cell up-scales its own gates (deprivation-driven
      scaling) until it captures a free niche.

"""

import numpy as np

# --- Populations (unchanged layout) ---
N_L1_E = 9
N_L1_I = 9
N_L2_E = 8
N_L2_I = 1
N_L3_E = 4  # Number of L3 excitatory neurons
N_L3_I = 1  # Number of L3 inhibitory neurons

# --- Synaptic weights / conductances ---
W_L1E_L1I = -5.0       # L1_E -> L1_I drive (magnitude used; sign handled in logic)
W_L1I_L1E = -10.0      # L1_I -> L1_E local inhibition
W_L1E_L2E_MIN = 10.0   # conductance floor for a *surviving* gate (reference)
W_L1E_L2E_MAX = 50.0   # gate conductance ceiling (a gate cannot widen past this)
W_L2E_L2I = 20.0       # L2_E -> L2_I (drives the global inhibitor)
W_L2I_L2E = -100.0     # L2_I -> L2_E global brake (reference scale)
W_L2E_L1I = 15.0       # L2_E -> L1_I feedback squelch ceiling
W_L2E_L3E_MIN = 10.0   # conductance floor for L2E->L3E surviving gate
W_L2E_L3E_MAX = 50.0   # gate conductance ceiling for L2E->L3E gate
W_L3E_L3I = 20.0       # L3_E -> L3_I (drives L3 global inhibitor)
W_L3I_L3E = -100.0     # L3_I -> L3_E global brake (reference scale)
W_L3E_L2I = 15.0       # L3_E -> L2_I feedback squelch ceiling (analogous to L2E->L1I)

# --- Fixed threshold + membrane dynamics ---
RESTING_POTENTIAL = 0.0
BASELINE_THETA = 50.0          # FIXED firing threshold for L2_E (never adapts)
BASELINE_THETA_L3 = 50.0       # FIXED firing threshold for L3_E (never adapts)
LEAK_FRACTION = 0.10           # FIXED fractional membrane leak per timestep
LEAK_RATE = LEAK_FRACTION      # alias kept for continuity
T_STEPS = 30                   # timesteps a stimulus burst is held
EXT_INPUT = 50.0               # external drive that makes an addressed L1_E fire
NOISE = 0.5                    # membrane channel noise (breaks symmetry / ties)

# --- Gate plasticity (the only learning) ---
W_GAIN = 6.0                   # demand-driven gate widening (saturating)
PRUNE_RATE = 0.45              # heterosynaptic withering of un-recruited gates
SYNAPTIC_BUDGET = 50.0         # per-neuron synaptic-scaling cap (local homeostasis)
MATURITY_THRESHOLD = 50.0      # total incoming gate at which one volley == theta
GATE_INIT_MAX = 4.0            # gates are born small -> cells start nascent / leaky

# --- Competition (replaces threshold homeostasis) ---
REFRACTORY = 1                 # presentations a cell stays silent after firing
ADAPT_GAIN = 18.0              # AHP charge added to a cell each time it fires
ADAPT_DECAY = 0.30             # fractional recovery of the AHP current per presentation
SILENCE_THRESH = 10            # presentations of silence before a cell up-scales
RECRUIT_RATE = 0.5             # gate increment per presentation while deprived


class FEPSNN:
    def __init__(self):
        # --- Membrane potentials (transient within a presentation) ---
        self.v_l1e = np.full(N_L1_E, RESTING_POTENTIAL)
        self.v_l1i = np.full(N_L1_I, RESTING_POTENTIAL)
        self.v_l2e = np.full(N_L2_E, RESTING_POTENTIAL)
        self.v_l2i = np.full(N_L2_I, RESTING_POTENTIAL)
        self.v_l3e = np.full(N_L3_E, RESTING_POTENTIAL)
        self.v_l3i = np.full(N_L3_I, RESTING_POTENTIAL)

        # --- FIXED thresholds (tiny manufacturing jitter on L2_E only) ---
        self.theta_l1e = np.full(N_L1_E, BASELINE_THETA)
        self.theta_l1i = np.full(N_L1_I, BASELINE_THETA * 0.5)
        self.theta_l2e = np.full(N_L2_E, BASELINE_THETA) + np.random.uniform(-2, 2, N_L2_E)
        self.theta_l2i = np.full(N_L2_I, BASELINE_THETA * 0.3)
        self.theta_l3e = np.full(N_L3_E, BASELINE_THETA_L3) + np.random.uniform(-2, 2, N_L3_E)
        self.theta_l3i = np.full(N_L3_I, BASELINE_THETA_L3 * 0.3)

        # --- Slow per-cell state: AHP current, refractory, silence ---
        self.adapt = np.zeros(N_L2_E)
        self.refrac = np.zeros(N_L2_E, dtype=int)
        self.silence = np.zeros(N_L2_E, dtype=int)
        
        # L3 slow per-cell state
        self.adapt_l3 = np.zeros(N_L3_E)
        self.refrac_l3 = np.zeros(N_L3_E, dtype=int)
        self.silence_l3 = np.zeros(N_L3_E, dtype=int)
        
        # --- Synaptic gates (the only learned structure) ---
        self.weights_l1e_l2e = np.random.uniform(0.0, GATE_INIT_MAX, (N_L2_E, N_L1_E))
        self.weights_l2e_l1i = np.random.uniform(0.0, 1.0, (N_L1_I, N_L2_E))
        # New plastic inhibitory weights in L1 layer
        self.weights_l1e_l1i = np.random.uniform(0.0, 1.0, (N_L1_E, N_L1_I))   # excitation L1E -> L1I
        self.weights_l1i_l1e = np.random.uniform(0.0, 1.0, (N_L1_I, N_L1_E))   # inhibition L1I -> L1E
        # L2 -> L3 excitatory plastic weights
        self.weights_l2e_l3e = np.random.uniform(0.0, GATE_INIT_MAX, (N_L3_E, N_L2_E))
        # L3 -> L3I feedback squelch (excitatory)
        self.weights_l3e_l3i = np.random.uniform(0.0, 1.0, (N_L3_I, N_L3_E))
        # L3I -> L3E lateral inhibition
        self.weights_l3i_l3e = np.random.uniform(0.0, 1.0, (N_L3_E, N_L3_I))
        # L3 -> L2I feedback squelch (excitatory) - NEW
        self.weights_l3e_l2i = np.random.uniform(0.0, 1.0, (N_L2_I, N_L3_E))
        
        # --- Per-neuron maturity (observable; NOT fed into the leak) ---
        self.maturity = np.zeros(N_L2_E)
        self.maturity_l3 = np.zeros(N_L3_E)
        for j in range(N_L2_E):
            self._recompute_maturity(j)
        for j in range(N_L3_E):
            self._recompute_maturity_l3(j)

        # --- Per-neuron maturity (observable; NOT fed into the leak) ---
        self.maturity = np.zeros(N_L2_E)
        for j in range(N_L2_E):
            self._recompute_maturity(j)

        self.last_trace = []
        self._frames = []   # for recording all frames when record=True

    # ---------------------------------------------------------------- helpers
    def _recompute_maturity(self, j):
        """maturity = how close a single volley of this cell's gates is to theta.
        Local to neuron j (sums only its own gates). Observable; NOT used in leak."""
        total = 0.0
        for i in range(N_L1_E):
            total += self.weights_l1e_l2e[j, i]
        self.maturity[j] = min(total / MATURITY_THRESHOLD, 1.0)

    def _recompute_maturity_l3(self, j):
        """maturity for L3 layer: how close a single volley of L2->L3 gates is to theta."""
        total = 0.0
        for i in range(N_L2_E):
            total += self.weights_l2e_l3e[j, i]
        self.maturity_l3[j] = min(total / MATURITY_THRESHOLD, 1.0)

    def _make_frame(self, active, winner, t, stage, spk_l1e, spk_l1i, l2_fired, brake, free_energy,
                    l3_fired=None, l3_brake=False, l3_free_energy=0.0):
        import numpy as np
        frame = {
            "t": int(t),
            "stage": stage,
            "winner": int(winner) if winner is not None else None,
            "pattern": [1 if i in active else 0 for i in range(N_L1_E)],
            "v_l1e": self.v_l1e.tolist(),
            "v_l1i": self.v_l1i.tolist(),
            "v_l2e": self.v_l2e.tolist(),
            "v_l2i": self.v_l2i.tolist(),
            "v_l3e": self.v_l3e.tolist(),
            "v_l3i": self.v_l3i.tolist(),
            "spk_l1e": [bool(b) for b in spk_l1e],
            "spk_l1i": [bool(b) for b in spk_l1i],
            "spk_l2e": [bool(l2_fired == j) for j in range(N_L2_E)],
            "spk_l2i": [bool(l2_fired == 0)],
            "spk_l3e": [bool(l3_fired == j) for j in range(N_L3_E)] if l3_fired is not None else [False] * N_L3_E,
            "spk_l3i": [bool(l3_fired == 0)] if l3_fired is not None else [False],
            "adapt": self.adapt.tolist(),
            "refrac": self.refrac.tolist(),
            "silence": self.silence.tolist(),
            "adapt_l3": self.adapt_l3.tolist(),
            "refrac_l3": self.refrac_l3.tolist(),
            "silence_l3": self.silence_l3.tolist(),
            "weights_l1e_l2e": self.weights_l1e_l2e.tolist(),
            "weights_l1e_l1i": self.weights_l1e_l1i.tolist(),
            "weights_l1i_l1e": self.weights_l1i_l1e.tolist(),
            "weights_l2e_l3e": self.weights_l2e_l3e.tolist(),
            "weights_l3e_l3i": self.weights_l3e_l3i.tolist(),
            "weights_l3i_l3e": self.weights_l3i_l3e.tolist(),
            "free_energy": float(free_energy),
            "l3_free_energy": float(l3_free_energy)
        }
        return frame

    def _apply_leak(self, v, leak_fraction=LEAK_FRACTION, resting=RESTING_POTENTIAL):
        """Apply leak to a membrane potential array."""
        return v - leak_fraction * (v - resting)

    def _learn(self, j, fired_l1e, fired_l1i):
        """Plasticity entirely local to the cell that just fired (Hebbian: a gate
        widens only when its pre-cell was part of the volley that fired its post)."""
        # 1. L1E -> L2E excitatory plasticity
        for i in range(N_L1_E):
            if fired_l1e[i]:
                g = self.weights_l1e_l2e[j, i]
                self.weights_l1e_l2e[j, i] = g + W_GAIN * (1.0 - g / W_L1E_L2E_MAX)
            else:
                self.weights_l1e_l2e[j, i] *= (1.0 - PRUNE_RATE)
            if self.weights_l1e_l2e[j, i] < 0.0:
                self.weights_l1e_l2e[j, i] = 0.0

        # 2. Synaptic scaling: the cell caps its own total conductance (local).
        total = 0.0
        for i in range(N_L1_E):
            total += self.weights_l1e_l2e[j, i]
        if total > SYNAPTIC_BUDGET:
            scale = SYNAPTIC_BUDGET / total
            for i in range(N_L1_E):
                self.weights_l1e_l2e[j, i] *= scale

        # 3. The cell refreshes its own (observable) maturity.
        self._recompute_maturity(j)

        # 4. Feedback squelch gates widen toward the sources that recruited it.
        for i in range(N_L1_I):
            if fired_l1e[i]:
                g = self.weights_l2e_l1i[i, j]
                self.weights_l2e_l1i[i, j] = g + W_GAIN * (1.0 - g / W_L2E_L1I)

        # 5. L1E -> L1I excitatory plasticity (feedforward drive)
        for pre in range(N_L1_E):   # pre = L1E index
            for post in range(N_L1_I):  # post = L1I index
                if fired_l1e[pre] and fired_l1i[post]:
                    g = self.weights_l1e_l1i[pre, post]
                    self.weights_l1e_l1i[pre, post] = g + W_GAIN * (1.0 - g / 1.0)  # assume max 1.0
                else:
                    self.weights_l1e_l1i[pre, post] *= (1.0 - PRUNE_RATE)
                if self.weights_l1e_l1i[pre, post] < 0.0:
                    self.weights_l1e_l1i[pre, post] = 0.0

        # 6. L1I -> L1E inhibitory plasticity (lateral inhibition)
        for pre in range(N_L1_I):   # pre = L1I index
            for post in range(N_L1_E):  # post = L1E index
                if fired_l1i[pre] and fired_l1e[post]:
                    g = self.weights_l1i_l1e[pre, post]
                    self.weights_l1i_l1e[pre, post] = g + W_GAIN * (1.0 - g / 1.0)  # assume max 1.0
                else:
                    self.weights_l1i_l1e[pre, post] *= (1.0 - PRUNE_RATE)
                if self.weights_l1i_l1e[pre, post] < 0.0:
                    self.weights_l1i_l1e[pre, post] = 0.0

    def _learn_l3(self, j, fired_l2e, fired_l3i):
        """Plasticity for L3 layer: L2E -> L3E excitatory, L3E -> L3I feedback squelch, L3E -> L2I feedback squelch."""
        # 1. L2E -> L3E excitatory plasticity
        for i in range(N_L2_E):
            if fired_l2e[i]:
                g = self.weights_l2e_l3e[j, i]
                self.weights_l2e_l3e[j, i] = g + W_GAIN * (1.0 - g / W_L2E_L3E_MAX)
            else:
                self.weights_l2e_l3e[j, i] *= (1.0 - PRUNE_RATE)
            if self.weights_l2e_l3e[j, i] < 0.0:
                self.weights_l2e_l3e[j, i] = 0.0

        # 2. Synaptic scaling for L3 neuron
        total = 0.0
        for i in range(N_L2_E):
            total += self.weights_l2e_l3e[j, i]
        if total > SYNAPTIC_BUDGET:
            scale = SYNAPTIC_BUDGET / total
            for i in range(N_L2_E):
                self.weights_l2e_l3e[j, i] *= scale

        # 3. Refresh L3 maturity
        self._recompute_maturity_l3(j)

        # 4. Feedback squelch: L3E -> L3I excitatory plasticity
        for i in range(N_L3_I):
            if fired_l3i[i]:  # Note: fired_l3i is length 1, but we loop for generality
                g = self.weights_l3e_l3i[i, j]
                self.weights_l3e_l3i[i, j] = g + W_GAIN * (1.0 - g / 1.0)  # assume max 1.0
            else:
                self.weights_l3e_l3i[i, j] *= (1.0 - PRUNE_RATE)
            if self.weights_l3e_l3i[i, j] < 0.0:
                self.weights_l3e_l3i[i, j] = 0.0

        # 5. Feedback squelch: L3E -> L2I excitatory plasticity (NEW)
        for i in range(N_L2_I):
            if fired_l3i[i]:  # When L3I fires, it means L3E was active recently - train L3E->L2I
                g = self.weights_l3e_l2i[i, j]
                self.weights_l3e_l2i[i, j] = g + W_GAIN * (1.0 - g / W_L3E_L2I)
            else:
                self.weights_l3e_l2i[i, j] *= (1.0 - PRUNE_RATE)
            if self.weights_l3e_l2i[i, j] < 0.0:
                self.weights_l3e_l2i[i, j] = 0.0

    def reset_membranes(self):
        self.v_l1e[:] = RESTING_POTENTIAL
        self.v_l1i[:] = RESTING_POTENTIAL
        self.v_l2e[:] = RESTING_POTENTIAL
        self.v_l2i[:] = RESTING_POTENTIAL
        self.adapt[:] = 0.0
        self.refrac[:] = 0
        self.silence = np.zeros(N_L2_E, dtype=int)  # corrected line
        # L3 reset
        self.v_l3e[:] = RESTING_POTENTIAL
        self.v_l3i[:] = RESTING_POTENTIAL
        self.adapt_l3[:] = 0.0
        self.refrac_l3[:] = 0
        self.silence_l3 = np.zeros(N_L3_E, dtype=int)

    def _snapshot(self, active, winner, t, stage, spk_l1e, spk_l1i, l2_fired, brake,
                  free_energy=0.0, l3_fired=None, l3_brake=False, l3_free_energy=0.0):
        return {
            "step": t, "stage": stage, "winner": winner,
            "l2_fired": l2_fired, "brake": brake, "free_energy": round(float(free_energy), 1),
            "l3_fired": l3_fired, "l3_brake": l3_brake, "l3_free_energy": round(float(l3_free_energy), 1),
            "pattern": list(active),
            "spk_l1e": list(spk_l1e), "spk_l1i": list(spk_l1i),
            "v_l1e": self.v_l1e.copy().tolist(), "v_l1i": self.v_l1i.copy().tolist(),
            "v_l2e": self.v_l2e.copy().tolist(), "v_l2i": self.v_l2i.copy().tolist(),
            "v_l3e": self.v_l3e.copy().tolist(), "v_l3i": self.v_l3i.copy().tolist(),
            "maturity": self.maturity.copy().tolist(),
            "adapt": self.adapt.copy().tolist(),
            "refrac": self.refrac.copy().tolist(),
            "silence": self.silence.copy().tolist(),
            # L3 state
            "maturity_l3": self.maturity_l3.copy().tolist(),
            "adapt_l3": self.adapt_l3.copy().tolist(),
            "refrac_l3": self.refrac_l3.copy().tolist(),
            "silence_l3": self.silence_l3.copy().tolist(),
            # Fields expected by the training scripts for compatibility
            "event": stage,  # we reuse the stage string as the event
            "v_e": self.v_l2e.copy().tolist(),
            "theta_e": self.theta_l2e.copy().tolist(),
            "iff_metrics": {
                "spikes": len(spk_l1i),
                "triggered": brake
            }
        }
    # --------------------------------------------------------- one presentation
    def process_event(self, active_indices, record=False):
        """Present one stimulus burst and run the spiking race (fixed threshold).
        Returns the cell that consolidated the symbol this time, or None.
        Learning is always enabled (plastic=True)."""
        active = list(active_indices)

        # Fresh trial for the fast membranes. The cell starts hyperpolarised by its
        # own AHP current (which recovers a little first); the L1 interneuron pool
        # keeps a fading memory of the last winner's squelch.
        for j in range(N_L2_E):
            self.adapt[j] *= (1.0 - ADAPT_DECAY)
            if self.refrac[j] > 0:
                self.refrac[j] -= 1
        # L3 slow state updates
        for j in range(N_L3_E):
            self.adapt_l3[j] *= (1.0 - ADAPT_DECAY)
            if self.refrac_l3[j] > 0:
                self.refrac_l3[j] -= 1
        self.v_l1e[:] = RESTING_POTENTIAL
        self.v_l2i[:] = RESTING_POTENTIAL
        self.v_l3i[:] = RESTING_POTENTIAL
        self.v_l1i[:] = RESTING_POTENTIAL  # start from rest each presentation
        for j in range(N_L2_E):
            self.v_l2e[j] = RESTING_POTENTIAL - self.adapt[j]
        for j in range(N_L3_E):
            self.v_l3e[j] = RESTING_POTENTIAL - self.adapt_l3[j]

        fired_l1e = [False] * N_L1_E
        fired_l1i = [False] * N_L1_I
        fired_l2e = [False] * N_L2_E
        fired_l3e = [False] * N_L3_E
        fired_l3i = [False] * N_L3_I
        winner = None
        trace = []

        for t in range(T_STEPS):
            # (a) Fixed leak on every membrane.
            self.v_l1e = self._apply_leak(self.v_l1e)
            self.v_l2e = self._apply_leak(self.v_l2e)
            self.v_l3e = self._apply_leak(self.v_l3e)
            self.v_l1i = self._apply_leak(self.v_l1i)

            # (b) The stimulus volley arrives (burst: it keeps arriving each step).
            for i in active:
                self.v_l1e[i] += EXT_INPUT

            # (c) L1_E spikes -> drive paired local inhibitor.
            spikes_l1e = []
            for i in range(N_L1_E):
                if self.v_l1e[i] >= self.theta_l1e[i]:
                    spikes_l1e.append(i)
                    fired_l1e[i] = True
                    self.v_l1e[i] = RESTING_POTENTIAL
                    # drive L1I via excitatory weights
                    for k in range(N_L1_I):
                        self.v_l1i[k] += self.weights_l1e_l1i[i, k]

            # (d) L1_I spikes -> local lateral inhibition of paired L1_E.
            spikes_l1i = []
            for k in range(N_L1_I):
                if self.v_l1i[k] >= self.theta_l1i[k]:
                    spikes_l1i.append(k)
                    fired_l1i[k] = True
                    self.v_l1i[k] = RESTING_POTENTIAL
                    # inhibit L1E via inhibitory weights
                    for i in range(N_L1_E):
                        self.v_l1e[i] -= self.weights_l1i_l1e[k, i]

            # (e) Propagate the L1_E volley to L2_E, one synapse at a time.
            for i in spikes_l1e:
                for j in range(N_L2_E):
                    self.v_l2e[j] += self.weights_l1e_l2e[j, i]

            # (f) Propagate L2E volley to L3E
            for i in range(N_L2_E):
                if self.v_l2e[i] >= 0:  # Only propagate if there's positive voltage
                    for j in range(N_L3_E):
                        self.v_l3e[j] += self.weights_l2e_l3e[j, i]

            # (g) Membrane channel noise, then the spiking race for L2.
            for j in range(N_L2_E):
                self.v_l2e[j] += np.random.uniform(-NOISE, NOISE)
            fired_this_step = None
            for j in range(N_L2_E):
                if self.refrac[j] == 0 and self.v_l2e[j] >= self.theta_l2e[j]:
                    if fired_this_step is None or self.v_l2e[j] > self.v_l2e[fired_this_step]:
                        fired_this_step = j

            # (h) Membrane channel noise, then the spiking race for L3.
            for j in range(N_L3_E):
                self.v_l3e[j] += np.random.uniform(-NOISE, NOISE)
            fired_this_step_l3 = None
            for j in range(N_L3_E):
                if self.refrac_l3[j] == 0 and self.v_l3e[j] >= self.theta_l3e[j]:
                    if fired_this_step_l3 is None or self.v_l3e[j] > self.v_l3e[fired_this_step_l3]:
                        fired_this_step_l3 = j

            brake = False
            free_energy = 0.0
            if fired_this_step is not None:
                j = fired_this_step
                if winner is None:
                    winner = j
                # Free energy = the leftover charge above threshold. With perfectly
                # balanced gates the volley would land exactly on theta (F=0); since
                # learning is emergent it usually overshoots.
                free_energy = self.v_l2e[j] - self.theta_l2e[j]
                # (g) Winner drives the global inhibitor. When it fires it performs a
                #     SECOND CHECK: it subtracts every cell's leftover charge, driving
                #     free energy (V - theta) back to equilibrium (0). The firing cell
                #     "spent" its threshold worth; the inhibitor wipes only the residual.
                self.v_l2i[0] += W_L2E_L2I
                if self.v_l2i[0] >= self.theta_l2i[0]:
                    brake = True
                    for k in range(N_L2_E):
                        if self.v_l2e[k] > self.theta_l2e[k]:
                            self.v_l2e[k] = self.theta_l2e[k]   # wipe the leftover
                    self.v_l2i[0] = RESTING_POTENTIAL
                # (h) Refractory + AHP build-up (the spike's own after-effects).
                self.refrac[j] = REFRACTORY
                self.adapt[j] += ADAPT_GAIN
                # (i) Source squelch (learned feedback) -> L1_I.
                for k in range(N_L1_I):
                    self.v_l1i[k] += self.weights_l2e_l1i[k, j]
                # (j) Consolidation -- local, on-spike only.
                self._learn(j, fired_l1e, fired_l1i)

            # L3 processing: if L3 fires, it drives L3I which provides global inhibition to L3E
            l3_brake = False
            l3_free_energy = 0.0
            if fired_this_step_l3 is not None:
                j = fired_this_step_l3
                # L3 free energy
                l3_free_energy = self.v_l3e[j] - self.theta_l3e[j]
                # L3 winner drives L3I
                self.v_l3i[0] += W_L3E_L3I
                if self.v_l3i[0] >= self.theta_l3i[0]:
                    l3_brake = True
                    for k in range(N_L3_E):
                        if self.v_l3e[k] > self.theta_l3e[k]:
                            self.v_l3e[k] = self.theta_l3e[k]   # wipe the leftover
                    self.v_l3i[0] = RESTING_POTENTIAL
                # L3 winner also drives L2I (feedback inhibition) - NEW
                self.v_l2i[0] += W_L3E_L2I
                if self.v_l2i[0] >= self.theta_l2i[0]:
                    # L2I firing would normally trigger L2 brake, but we handle it separately
                    # For now, we just reset L2I - the actual L2 brake logic is in L2 section
                    self.v_l2i[0] = RESTING_POTENTIAL
                # L3 refractory + AHP build-up
                self.refrac_l3[j] = REFRACTORY
                self.adapt_l3[j] += ADAPT_GAIN
                # L3 -> L3I feedback squelch (excitatory)
                for k in range(N_L3_I):
                    self.v_l3i[k] += self.weights_l3e_l3i[k, j]
                # L3 -> L2I feedback squelch (excitatory) - NEW
                for k in range(N_L2_I):
                    self.v_l2i[k] += self.weights_l3e_l2i[k, j]
                # L3 consolidation -- local, on-spike only.
                self._learn_l3(j, fired_l2e, fired_l3i)  # L2E drives L3E, L3I provides feedback

            if record:
                stage = ("brake" if brake else "e_spike" if fired_this_step is not None
                         else "l1_volley" if spikes_l1e else "integrating")
                l3_stage = ("l3_brake" if l3_brake else "l3_e_spike" if fired_this_step_l3 is not None
                           else "l3_integrating" if any(self.v_l2e[i] >= 0 for i in range(N_L2_E)) else "l3_quiet")
                trace.append(self._snapshot(active, winner, t, stage,
                                            spikes_l1e, spikes_l1i, fired_this_step, brake,
                                            free_energy, l3_fired=fired_this_step_l3,
                                            l3_brake=l3_brake, l3_free_energy=l3_free_energy))
            if winner is not None:
                break

        # Deprivation-driven recruitment for L2
        for j in range(N_L2_E):
            if j == winner:
                self.silence[j] = 0
            else:
                self.silence[j] += 1
                if self.silence[j] > SILENCE_THRESH:
                    for i in range(N_L1_E):
                        self.weights_l1e_l2e[j, i] += RECRUIT_RATE
                    self._recompute_maturity(j)

        # Deprivation-driven recruitment for L3
        l3_winner = None
        for j in range(N_L3_E):
            # Find if L3 had a winner this presentation
            pass  # We'll track L3 winner separately
        
        # For simplicity, we'll use the L2 winner as a proxy for L3 deprivation
        # In a full implementation, we'd track actual L3 winners
        if winner is not None:
            l3_winner = winner % N_L3_E  # Simple mapping for now
            
        for j in range(N_L3_E):
            if j == l3_winner:
                self.silence_l3[j] = 0
            else:
                self.silence_l3[j] += 1
                if self.silence_l3[j] > SILENCE_THRESH:
                    for i in range(N_L2_E):
                        self.weights_l2e_l3e[j, i] += RECRUIT_RATE
                    self._recompute_maturity_l3(j)

        self.last_trace = trace
        if record:
            self._frames.extend(trace)
        return winner

    # --------------------------------------------------------------- probing
    def volleys_to_fire(self, active_indices, max_steps=T_STEPS):
        """Diagnostic of the LIF->integrator transition: from a cleared membrane,
        how many burst timesteps until some cell fires for this pattern? Observation
        only. ~1 step == pure-integrator behaviour; many steps == leaky/nascent."""
        for j in range(N_L2_E):
            self.v_l2e[j] = RESTING_POTENTIAL
        self.v_l1e[:] = RESTING_POTENTIAL
        self.v_l1i[:] = RESTING_POTENTIAL
        save_ref = self.refrac.copy()
        self.refrac[:] = 0
        self.process_event(active_indices, record=True)  # learning always on
        self.refrac[:] = save_ref
        for s in self.last_trace:
            if s["l2_fired"] is not None:
                return s["step"] + 1, s["l2_fired"]
        return None, None

    # Convenience method used by training scripts
    def process_event_pattern(self, pattern):
        """Given a binary list/array pattern, return the trace of a single presentation."""
        active = [i for i, val in enumerate(pattern) if val != 0]
        self.process_event(active, record=True)
        return self.last_trace