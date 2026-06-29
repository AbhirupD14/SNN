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

# --- Synaptic weights / conductances ---
W_L1E_L1I = -5.0       # L1_E -> L1_I drive (magnitude used; sign handled in logic)
W_L1I_L1E = -10.0      # L1_I -> L1_E local inhibition
W_L1E_L2E_MIN = 10.0   # conductance floor for a *surviving* gate (reference)
W_L1E_L2E_MAX = 50.0   # gate conductance ceiling (a gate cannot widen past this)
W_L2E_L2I = 20.0       # L2_E -> L2_I (drives the global inhibitor)
W_L2I_L2E = -100.0     # L2_I -> L2_E global brake (reference scale)
W_L2E_L1I = 15.0       # L2_E -> L1_I feedback squelch ceiling

# --- Fixed threshold + membrane dynamics ---
RESTING_POTENTIAL = 0.0
BASELINE_THETA = 50.0          # FIXED firing threshold for L2_E (never adapts)
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

        # --- FIXED thresholds (tiny manufacturing jitter on L2_E only) ---
        self.theta_l1e = np.full(N_L1_E, BASELINE_THETA)
        self.theta_l1i = np.full(N_L1_I, BASELINE_THETA * 0.5)
        self.theta_l2e = np.full(N_L2_E, BASELINE_THETA) + np.random.uniform(-2, 2, N_L2_E)
        self.theta_l2i = np.full(N_L2_I, BASELINE_THETA * 0.3)

        # --- Slow per-cell state: AHP current, refractory, silence ---
        self.adapt = np.zeros(N_L2_E)
        self.refrac = np.zeros(N_L2_E, dtype=int)
        self.silence = np.zeros(N_L2_E, dtype=int)

        # --- Synaptic gates (the only learned structure) ---
        self.weights_l1e_l2e = np.random.uniform(0.0, GATE_INIT_MAX, (N_L2_E, N_L1_E))
        self.weights_l2e_l1i = np.random.uniform(0.0, 1.0, (N_L1_I, N_L2_E))

        # --- Per-neuron maturity (observable; NOT fed into the leak) ---
        self.maturity = np.zeros(N_L2_E)
        for j in range(N_L2_E):
            self._recompute_maturity(j)

        self.last_trace = []

    # ---------------------------------------------------------------- helpers
    def _recompute_maturity(self, j):
        """maturity = how close a single volley of this cell's gates is to theta.
        Local to neuron j (sums only its own gates). Observable; NOT used in leak."""
        total = 0.0
        for i in range(N_L1_E):
            total += self.weights_l1e_l2e[j, i]
        self.maturity[j] = min(total / MATURITY_THRESHOLD, 1.0)

    def _learn(self, j, fired_l1e):
        """Plasticity entirely local to the cell that just fired (Hebbian: a gate
        widens only when its pre-cell was part of the volley that fired its post)."""
        # 1. Demand-driven widening / heterosynaptic withering, per synapse.
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

    def reset_membranes(self):
        self.v_l1e[:] = RESTING_POTENTIAL
        self.v_l1i[:] = RESTING_POTENTIAL
        self.v_l2e[:] = RESTING_POTENTIAL
        self.v_l2i[:] = RESTING_POTENTIAL
        self.adapt[:] = 0.0
        self.refrac[:] = 0
        self.silence[:] = 0

    def _snapshot(self, active, winner, t, stage, spk_l1e, spk_l1i, l2_fired, brake,
                  free_energy=0.0):
        return {
            "step": t, "stage": stage, "winner": winner,
            "l2_fired": l2_fired, "brake": brake, "free_energy": round(float(free_energy), 1),
            "pattern": list(active),
            "spk_l1e": list(spk_l1e), "spk_l1i": list(spk_l1i),
            "v_l1e": self.v_l1e.copy().tolist(), "v_l1i": self.v_l1i.copy().tolist(),
            "v_l2e": self.v_l2e.copy().tolist(), "v_l2i": self.v_l2i.copy().tolist(),
            "maturity": self.maturity.copy().tolist(),
            "adapt": self.adapt.copy().tolist(),
            "refrac": self.refrac.copy().tolist(),
        }

    # --------------------------------------------------------- one presentation
    def process_event(self, active_indices, record=False, plastic=True):
        """Present one stimulus burst and run the spiking race (fixed threshold).
        Returns the cell that consolidated the symbol this time, or None.
        plastic=False freezes all learning so the network can be observed."""
        active = list(active_indices)

        # Fresh trial for the fast membranes. The cell starts hyperpolarised by its
        # own AHP current (which recovers a little first); the L1 interneuron pool
        # keeps a fading memory of the last winner's squelch.
        for j in range(N_L2_E):
            self.adapt[j] *= (1.0 - ADAPT_DECAY)
            if self.refrac[j] > 0:
                self.refrac[j] -= 1
        self.v_l1e[:] = RESTING_POTENTIAL
        self.v_l2i[:] = RESTING_POTENTIAL
        self.v_l1i *= 0.5
        for j in range(N_L2_E):
            self.v_l2e[j] = RESTING_POTENTIAL - self.adapt[j]

        fired_l1e = [False] * N_L1_E
        winner = None
        trace = []

        for t in range(T_STEPS):
            # (a) Fixed leak on every membrane.
            for i in range(N_L1_E):
                self.v_l1e[i] -= LEAK_FRACTION * (self.v_l1e[i] - RESTING_POTENTIAL)
            for j in range(N_L2_E):
                self.v_l2e[j] -= LEAK_FRACTION * (self.v_l2e[j] - RESTING_POTENTIAL)

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
                    self.v_l1i[i] += abs(W_L1E_L1I)

            # (d) L1_I spikes -> local lateral inhibition of paired L1_E.
            spikes_l1i = []
            for k in range(N_L1_I):
                if self.v_l1i[k] >= self.theta_l1i[k]:
                    spikes_l1i.append(k)
                    self.v_l1e[k] += W_L1I_L1E
                    self.v_l1i[k] = RESTING_POTENTIAL

            # (e) Propagate the L1_E volley to L2_E, one synapse at a time.
            for i in spikes_l1e:
                for j in range(N_L2_E):
                    self.v_l2e[j] += self.weights_l1e_l2e[j, i]

            # (f) Membrane channel noise, then the spiking race. The winner is
            #     simply the cell that reaches threshold FIRST. Within one discrete
            #     tick several cells may already be supra-threshold; the one whose
            #     membrane sits furthest past theta is the one that crossed earliest
            #     (it was integrating fastest), so it fired first. This is ordering
            #     of real spikes by crossing time -- not a god's-eye pick over cells
            #     that never fired. The per-step noise makes the leader unique, so
            #     there is no tie to resolve procedurally.
            for j in range(N_L2_E):
                self.v_l2e[j] += np.random.uniform(-NOISE, NOISE)
            fired_this_step = None
            for j in range(N_L2_E):
                if self.refrac[j] == 0 and self.v_l2e[j] >= self.theta_l2e[j]:
                    if fired_this_step is None or self.v_l2e[j] > self.v_l2e[fired_this_step]:
                        fired_this_step = j

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
                if plastic:
                    self._learn(j, fired_l1e)

            if record:
                stage = ("brake" if brake else "e_spike" if fired_this_step is not None
                         else "l1_volley" if spikes_l1e else "integrating")
                trace.append(self._snapshot(active, winner, t, stage,
                                            spikes_l1e, spikes_l1i, fired_this_step, brake,
                                            free_energy))
            if winner is not None:
                break

        # Deprivation-driven recruitment: each cell tracks its own silence and, if
        # chronically quiet, scales its gates up until it captures a free niche.
        if plastic:
            for j in range(N_L2_E):
                if j == winner:
                    self.silence[j] = 0
                else:
                    self.silence[j] += 1
                    if self.silence[j] > SILENCE_THRESH:
                        for i in range(N_L1_E):
                            self.weights_l1e_l2e[j, i] += RECRUIT_RATE
                        self._recompute_maturity(j)

        self.last_trace = trace
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
        self.process_event(active_indices, record=True, plastic=False)
        self.refrac[:] = save_ref
        for s in self.last_trace:
            if s["l2_fired"] is not None:
                return s["step"] + 1, s["l2_fired"]
        return None, None
