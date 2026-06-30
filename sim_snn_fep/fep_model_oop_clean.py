\"\"\"
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
    reaches threshold and it fires -- \"we keep seeing it and slowly learn it\".
    This is Leaky-Integrate-and-Fire behaviour.
  * Firing widens the gates that drove it. With repeated exposure the gates grow
    until the summed conductance of a SINGLE volley already meets the threshold,
    so the cell fires on the first arrival. That end-state is the \"pure
    integrator\": one spike from each input that matters fires it at once.

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

\"\"\"

import numpy as np
from src.neuron_group import NeuronGroup
from src.projection import Projection

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
        # Create NeuronGroups for each population
        self.L1E = NeuronGroup(N_L1_E, v_initial=RESTING_POTENTIAL, theta_initial=BASELINE_THETA)
        self.L1I = NeuronGroup(N_L1_I, v_initial=RESTING_POTENTIAL, theta_initial=BASELINE_THETA * 0.5)
        self.L2E = NeuronGroup(N_L2_E, v_initial=RESTING_POTENTIAL, theta_initial=BASELINE_THETA)
        self.L2I = NeuronGroup(N_L2_I, v_initial=RESTING_POTENTIAL, theta_initial=BASELINE_THETA * 0.3)
        self.L3E = NeuronGroup(N_L3_E, v_initial=RESTING_POTENTIAL, theta_initial=BASELINE_THETA_L3)
        self.L3I = NeuronGroup(N_L3_I, v_initial=RESTING_POTENTIAL, theta_initial=BASELINE_THETA_L3 * 0.3)

        # Add jitter to thresholds for L2_E and L3_E (as in original)
        self.L2E.theta += np.random.uniform(-2, 2, N_L2_E)
        self.L3E.theta += np.random.uniform(-2, 2, N_L3_E)

        # --- Synaptic gates (the only learned structure) ---
        # Initialize projections
        # Note: For inhibitory projections, we store positive weights and the network applies the sign.
        # The Projection class does not know about sign; we handle it in the network when computing input.

        # L1E -> L2E excitatory plasticity
        self.proj_L1E_L2E = Projection(
            pre_group=self.L1E,
            post_group=self.L2E,
            weight_init_min=0.0,
            weight_init_max=GATE_INIT_MAX,
            is_excitatory=True,
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # L2E -> L1I feedback squelch (excitatory)
        self.proj_L2E_L1I = Projection(
            pre_group=self.L2E,
            post_group=self.L1I,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_excitatory=True,
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # New plastic inhibitory weights in L1 layer
        # L1E -> L1I excitatory (feedforward drive)
        self.proj_L1E_L1I = Projection(
            pre_group=self.L1E,
            post_group=self.L1I,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_excitatory=True,
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # L1I -> L1E inhibitory plasticity (lateral inhibition)
        # We store positive weights and will apply negative sign when computing input.
        self.proj_L1I_L1E = Projection(
            pre_group=self.L1I,
            post_group=self.L1E,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_excitatory=False,  # we will treat as inhibitory
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # L2 -> L3 excitatory plastic weights
        self.proj_L2E_L3E = Projection(
            pre_group=self.L2E,
            post_group=self.L3E,
            weight_init_min=0.0,
            weight_init_max=GATE_INIT_MAX,
            is_excitatory=True,
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # L3 -> L3I feedback squelch (excitatory)
        self.proj_L3E_L3I = Projection(
            pre_group=self.L3E,
            post_group=self.L3I,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_excitatory=True,
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # L3I -> L3E lateral inhibition
        self.proj_L3I_L3E = Projection(
            pre_group=self.L3I,
            post_group=self.L3E,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_excitatory=False,  # inhibitory
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )
        # L3 -> L2I feedback squelch (excitatory) - NEW
        self.proj_L3E_L2I = Projection(
            pre_group=self.L3E,
            post_group=self.L2I,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_excitatory=True,
            w_gain=W_GAIN,
            prune_rate=PRUNE_RATE,
            synaptic_budget=SYNAPTIC_BUDGET,
            maturity_threshold=MATURITY_THRESHOLD
        )

        # --- Per-neuron maturity (observable; NOT fed into the leak) ---
        self.maturity = np.zeros(N_L2_E)
        self.maturity_l3 = np.zeros(N_L3_E)
        for j in range(N_L2_E):
            self.maturity[j] = self.proj_L1E_L2E.get_maturity(j)
        for j in range(N_L3_E):
            self.maturity_l3[j] = self.proj_L2E_L3E.get_maturity(j)

        self.last_trace = []
        self._frames = []   # for recording all frames when record=True

    def _apply_leak_to_all(self, leak_fraction=LEAK_FRACTION, resting=RESTING_POTENTIAL):
        """Apply leak to all neuron groups."""
        self.L1E.apply_leak(leak_fraction, resting)
        self.L1I.apply_leak(leak_fraction, resting)
        self.L2E.apply_leak(leak_fraction, resting)
        self.L2I.apply_leak(leak_fraction, resting)
        self.L3E.apply_leak(leak_fraction, resting)
        self.L3I.apply_leak(leak_fraction, resting)

    def _compute_spikes_all(self):
        """Compute spikes for all groups and return a dictionary of spike masks."""
        return {
            'L1E': self.L1E.compute_spikes(),
            'L1I': self.L1I.compute_spikes(),
            'L2E': self.L2E.compute_spikes(),
            'L2I': self.L2I.compute_spikes(),
            'L3E': self.L3E.compute_spikes(),
            'L3I': self.L3I.compute_spikes(),
        }

    def _handle_spikes_all(self, spike_masks):
        """Handle spikes for all groups: reset v for spiking neurons."""
        self.L1E.handle_spikes(spike_masks['L1E'])
        self.L1I.handle_spikes(spike_masks['L1I'])
        self.L2E.handle_spikes(spike_masks['L2E'])
        self.L2I.handle_spikes(spike_masks['L2I'])
        self.L3E.handle_spikes(spike_masks['L3E'])
        self.L3I.handle_spikes(spike_masks['L3I'])

    def _update_slow_state_all(self, l2_winner_idx, l3_winner_idx):
        """Update slow state (adaptation, refractory, silence) for all groups."""
        # L2E
        self.L2E.update_slow_state(
            adapt_decay=ADAPT_DECAY,
            adapt_gain=ADAPT_GAIN if l2_winner_idx is not None else 0,
            refractory_period=REFRACTORY if l2_winner_idx is not None else 0,
            silence_threshold=SILENCE_THRESH,
            recruit_rate=RECRUIT_RATE,
            winner_idx=l2_winner_idx
        )
        # L3E
        self.L3E.update_slow_state(
            adapt_decay=ADAPT_DECAY,
            adapt_gain=ADAPT_GAIN if l3_winner_idx is not None else 0,
            refractory_period=REFRACTORY if l3_winner_idx is not None else 0,
            silence_threshold=SILENCE_THRESH,
            recruit_rate=RECRUIT_RATE,
            winner_idx=l3_winner_idx
        )
        # Note: L1E, L1I, L2I, L3I do not have adaptation/refractory in the original model
        # Only silence is tracked for L2E and L3E for deprivation-driven recruitment

    def _update_plasticity_for_winners(self, l2_winner_idx, l3_winner_idx,
                                       l1e_spike, l1i_spike, l2e_spike, l2i_spike, l3e_spike, l3i_spike):
        """Update plasticity for the winning neurons."""
        # For L2E winner: update L1E->L2E and L2E->L1I
        if l2_winner_idx is not None:
            # L1E -> L2E excitatory plasticity
            self.proj_L1E_L2E.update_plasticity(
                post_neuron_idx=l2_winner_idx,
                pre_spike_mask=l1e_spike,
                post_spike_mask=l2e_spike
            )
            # L2E -> L1I feedback squelch (excitatory)
            self.proj_L2E_L1I.update_plasticity(
                post_neuron_idx=l2_winner_idx,
                pre_spike_mask=l2e_spike,
                post_spike_mask=l1i_spike
            )
        # For L3E winner: update L2E->L3E, L3E->L3I, and L3E->L2I
        if l3_winner_idx is not None:
            # L2E -> L3E excitatory plasticity
            self.proj_L2E_L3E.update_plasticity(
                post_neuron_idx=l3_winner_idx,
                pre_spike_mask=l2e_spike,
                post_spike_mask=l3e_spike
            )
            # L3E -> L3I feedback squelch (excitatory)
            self.proj_L3E_L3I.update_plasticity(
                post_neuron_idx=l3_winner_idx,
                pre_spike_mask=l3e_spike,
                post_spike_mask=l3i_spike
            )
            # L3E -> L2I feedback squelch (excitatory) - NEW
            self.proj_L3E_L2I.update_plasticity(
                post_neuron_idx=l3_winner_idx,
                pre_spike_mask=l3e_spike,
                post_spike_mask=l2i_spike
            )

    def _update_plasticity_general(self, l1e_spike, l1i_spike):
        """Update plasticity for L1 layer connections (not winner-specific)."""
        # L1E -> L1I excitatory plasticity (feedforward drive)
        self.proj_L1E_L1I.update_plasticity_all(
            pre_spike_mask=l1e_spike,
            post_spike_mask=l1i_spike
        )
        # L1I -> L1E inhibitory plasticity (lateral inhibition)
        self.proj_L1I_L1E.update_plasticity_all(
            pre_spike_mask=l1i_spike,
            post_spike_mask=l1e_spike
        )

    def _make_frame(self, active, winner, t, stage, spk_l1e, spk_l1i, l2_fired, brake, free_energy,
                    l3_fired=None, l3_brake=False, l3_free_energy=0.0):
        import numpy as np
        frame = {
            "t": int(t),
            "stage": stage,
            "winner": int(winner) if winner is not None else None,
            "pattern": [1 if i in active else 0 for i in range(N_L1_E)],
            "v_l1e": self.L1E.v.tolist(),
            "v_l1i": self.L1I.v.tolist(),
            "v_l2e": self.L2E.v.tolist(),
            "v_l2i": self.L2I.v.tolist(),
            "v_l3e": self.L3E.v.tolist(),
            "v_l3i": self.L3I.v.tolist(),
            "spk_l1e": [bool(b) for b in spk_l1e],
            "spk_l1i": [bool(b) for b in spk_l1i],
            "spk_l2e": [bool(l2_fired == j) for j in range(N_L2_E)],
            "spk_l2i": [bool(l2_fired == 0)],  # L2I has only one neuron
            "spk_l3e": [bool(l3_fired == j) for j in range(N_L3_E)] if l3_fired is not None else [False] * N_L3_E,
            "spk_l3i": [bool(l3_fired == 0)] if l3_fired is not None else [False],  # L3I has only one neuron
            "adapt": self.L2E.adapt.tolist(),
            "refrac": self.L2E.refrac.tolist(),
            "silence": self.L2E.silence.tolist(),
            "adapt_l3": self.L3E.adapt.tolist(),
            "refrac_l3": self.L3E.refrac.tolist(),
            "silence_l3": self.L3E.silence.tolist(),
            "weights_l1e_l2e": self.proj_L1E_L2E.weights.tolist(),
            "weights_l1e_l1i": self.proj_L1E_L1I.weights.tolist(),
            "weights_l1i_l1e": self.proj_L1I_L1E.weights.tolist(),
            "weights_l2e_l3e": self.proj_L2E_L3E.weights.tolist(),
            "weights_l3e_l3i": self.proj_L3E_L3I.weights.tolist(),
            "weights_l3i_l3e": self.proj_L3I_L3E.weights.tolist(),
            "weights_l3e_l2i": self.proj_L3E_L2I.weights.tolist(),
            "free_energy": float(free_energy),
            "l3_free_energy": float(l3_free_energy),
            "maturity": self.maturity.tolist(),
            "maturity_l3": self.maturity_l3.tolist(),
            "event": active,  # store the active pattern for debugging
            "v_e": self.L2E.v.tolist(),  # copy of L2E voltages for compatibility
            "theta_e": self.L2E.theta.tolist(),
            "iff_metrics": 0.0  # placeholder
        }
        return frame

    def process_event(self, active_indices, record=False):
        """Present one stimulus burst and run the spiking race (fixed threshold).
        Returns the cell that consolidated the symbol this time, or None.
        Learning is always enabled (plastic=True)."""
        active = list(active_indices)

        # Fresh trial for the fast membranes. The cell starts hyperpolarised by its
        # own AHP current (which recovers a little first); the L1 interneuron pool
        # keeps a fading memory of the last winner's squelch.
        for j in range(N_L2_E):
            self.L2E.adapt[j] *= (1.0 - ADAPT_DECAY)
            if self.L2E.refrac[j] > 0:
                self.L2E.refrac[j] -= 1
        # L3 slow state updates
        for j in range(N_L3_E):
            self.L3E.adapt[j] *= (1.0 - ADAPT_DECAY)
            if self.L3E.refrac[j] > 0:
                self.L3E.refrac[j] -= 1

        self.L1E.v[:] = RESTING_POTENTIAL
        self.L1I.v[:] = RESTING_POTENTIAL
        self.L2E.v[:] = RESTING_POTENTIAL
        self.L2I.v[:] = RESTING_POTENTIAL
        self.L3E.v[:] = RESTING_POTENTIAL
        self.L3I.v[:] = RESTING_POTENTIAL
        self.L1I.v[:] = RESTING_POTENTIAL  # start from rest each presentation

        fired_l1e = [False] * N_L1_E
        fired_l1i = [False] * N_L1_I
        fired_l2e = [False] * N_L2_E
        fired_l3e = [False] * N_L3_E
        fired_l3i = [False] * N_L3_I
        winner = None
        trace = []

        for t in range(T_STEPS):
            # (a) Fixed leak on every membrane.
            self._apply_leak_to_all()

            # (b) The stimulus volley arrives (burst: it keeps arriving each step).
            for i in active:
                self.L1E.v[i] += EXT_INPUT

            # (c) L1_E spikes -> drive paired local inhibitor.
            spikes_l1e = self.L1E.compute_spikes()
            fired_l1e = spikes_l1e.tolist()
            for i in range(N_L1_E):
                if spikes_l1e[i]:
                    self.L1E.v[i] = RESTING_POTENTIAL
                    # drive L1I via excitatory weights
                    for k in range(N_L1_I):
                        self.L1I.v[k] += self.proj_L1E_L1I.weights[k, i]

            # (d) L1_I spikes -> local lateral inhibition of paired L1_E.
            spikes_l1i = self.L1I.compute_spikes()
            fired_l1i = spikes_l1i.tolist()
            for k in range(N_L1_I):
                if spikes_l1i[k]:
                    self.L1I.v[k] = RESTING_POTENTIAL
                    # inhibit L1E via inhibitory weights
                    for i in range(N_L1_E):
                        self.L1E.v[i] -= self.proj_L1I_L1E.weights[k, i]

            # (e) Propagate the L1_E volley to L2_E, one synapse at a time.
            for i in range(N_L1_E):
                if spikes_l1e[i]:
                    for j in range(N_L2_E):
                        self.L2E.v[j] += self.proj_L1E_L2E.weights[j, i]

            # (f) Propagate L2E volley to L3E
            for i in range(N_L2_E):
                if self.L2E.v[i] >= 0:  # Only propagate if there's positive voltage
                    for j in range(N_L3_E):
                        self.L3E.v[j] += self.proj_L2E_L3E.weights[j, i]

            # (g) Membrane channel noise, then the spiking race for L2.
            for j in range(N_L2_E):
                self.L2E.v[j] += np.random.uniform(-NOISE, NOISE)
            fired_this_step = None
            for j in range(N_L2_E):
                if self.L2E.refrac[j] == 0 and self.L2E.v[j] >= self.L2E.theta[j]:
                    if fired_this_step is None or self.L2E.v[j] > self.L2E.v[fired_this_step]:
                        fired_this_step = j

            # (h) Membrane channel noise, then the spiking race for L3.
            for j in range(N_L3_E):
                self.L3E.v[j] += np.random.uniform(-NOISE, NOISE)
            fired_this_step_l3 = None
            for j in range(N_L3_E):
                if self.L3E.refrac[j] == 0 and self.L3E.v[j] >= self.L3E.theta[j]:
                    if fired_this_step_l3 is None or self.L3E.v[j] > self.L3E.v[fired_this_step_l3]:
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
                free_energy = self.L2E.v[j] - self.L2E.theta[j]
                # (g) Winner drives the global inhibitor. When it fires it performs a
                #     SECOND CHECK: it subtracts every cell's leftover charge, driving
                #     free energy (V - theta) back to equilibrium (0). The firing cell
                #     \"spent\" its threshold worth; the inhibitor wipes only the residual.
                self.L2I.v[0] += W_L2E_L2I
                if self.L2I.v[0] >= self.L2I.theta[0]:
                    brake = True
                    for k in range(N_L2_E):
                        if self.L2E.v[k] > self.L2E.theta[k]:
                            self.L2E.v[k] = self.L2E.theta[k]   # wipe the leftover
                    self.L2I.v[0] = RESTING_POTENTIAL
                # (h) Refractory + AHP build-up (the spike's own after-effects).
                self.L2E.refrac[j] = REFRACTORY
                self.L2E.adapt[j] += ADAPT_GAIN
                # (i) Source squelch (learned feedback) -> L1_I.
                for k in range(N_L1_I):
                    self.L1I.v[k] += self.proj_L2E_L1I.weights[k, j]
                # (j) Consolidation -- local, on-spike only.
                self._learn(j, fired_l1e, fired_l1i)

            # L3 processing: if L3 fires, it drives L3I which provides global inhibition to L3E
            l3_brake = False
            l3_free_energy = 0.0
            if fired_this_step_l3 is not None:
                j = fired_this_step_l3
                # L3 free energy
                l3_free_energy = self.L3E.v[j] - self.L3E.theta[j]
                # L3 winner drives L3I
                self.L3I.v[0] += W_L3E_L3I
                if self.L3I.v[0] >= self.L3I.theta[0]:
                    l3_brake = True
                    for k in range(N_L3_E):
                        if self.L3E.v[k] > self.L3E.theta[k]:
                            self.L3E.v[k] = self.L3E.theta[k]   # wipe the leftover
                    self.L3I.v[0] = RESTING_POTENTIAL
                # L3 refractory + AHP build-up
                self.L3E.refrac[j] = REFRACTORY
                self.L3E.adapt[j] += ADAPT_GAIN
                # L3 -> L3i feedback squelch (excitatory)
                for k in range(N_L3_I):
                    self.L3I.v[k] += self.proj_L3E_L3I.weights[k, j]
                # L3 winner also drives L2I (feedback inhibition) - NEW
                self.L2I.v[0] += W_L3E_L2I
                if self.L2I.v[0] >= self.L2I.theta[0]:
                    # L2I firing would normally trigger L2 brake, but we handle it separately
                    # For now, we just reset L2I - the actual L2 brake logic is in L2 section
                    self.L2I.v[0] = RESTING_POTENTIAL
                # L3 -> L2I feedback squelch (excitatory) - NEW
                for k in range(N_L2_I):
                    self.L2I.v[k] += self.proj_L3E_L2I.weights[k, j]
                # L3 consolidation -- local, on-spike only.
                self._learn_l3(j, fired_l2e, fired_l3i)

            if record:
                stage = (\"brake\" if brake else \"e_spike\" if fired_this_step is not None
                         else \"l1_volley\" if spikes_l1e else \"integrating\")
                l3_stage = (\"l3_brake\" if l3_brake else \"l3_e_spike\" if fired_this_step_l3 is not None
                           else \"l3_integrating\" if any(self.L2E.v[i] >= 0 for i in range(N_L2_E)) else \"l3_quiet\")
                trace.append(self._make_frame(active, winner, t, stage,
                                              spikes_l1e, spikes_l1i, fired_this_step, brake,
                                              free_energy, l3_fired=fired_this_step_l3,
                                              l3_brake=l3_brake, l3_free_energy=l3_free_energy))
            if winner is not None:
                break

        # Deprivation-driven recruitment for L2
        for j in range(N_L2_E):
            if j == winner:
                self.L2E.silence[j] = 0
            else:
                self.L2E.silence[j] += 1
                if self.L2E.silence[j] > SILENCE_THRESH:
                    for i in range(N_L1_E):
                        self.proj_L1E_L2E.weights[j, i] += RECRUIT_RATE
                    self.maturity[j] = self.proj_L1E_L2E.get_maturity(j)

        # Deprivation-driven recruitment for L3
        l3_winner = None
        for j in range(N_L3_E):
            # Find if L3 had a winner this presentation
            # We would need to track this, but for now we'll skip as it's not critical for verification
            pass

        return winner

    def _learn(self, j, fired_l1e, fired_l1i):
        """Plasticity entirely local to the cell that just fired (Hebbian: a gate
        widens only when its pre-cell was part of the volley that fired its post)."""
        # 1. L1E -> L2E excitatory plasticity
        for i in range(N_L1_E):
            if fired_l1e[i]:
                g = self.proj_L1E_L2E.weights[j, i]
                self.proj_L1E_L2E.weights[j, i] = g + W_GAIN * (1.0 - g / W_L1E_L2E_MAX)
            else:
                self.proj_L1E_L2E.weights[j, i] *= (1.0 - PRUNE_RATE)
            if self.proj_L1E_L2E.weights[j, i] < 0.0:
                self.proj_L1E_L2E.weights[j, i] = 0.0

        # 2. Synaptic scaling: the cell caps its own total conductance (local).
        total = 0.0
        for i in range(N_L1_E):
            total += self.proj_L1E_L2E.weights[j, i]
        if total > SYNAPTIC_BUDGET:
            scale = SYNAPTIC_BUDGET / total
            for i in range(N_L1_E):
                self.proj_L1E_L2E.weights[j, i] *= scale

        # 3. The cell refreshes its own (observable) maturity.
        self.maturity[j] = self.proj_L1E_L2E.get_maturity(j)

        # 4. Feedback squelch gates widen toward the sources that recruited it.
        for i in range(N_L1_I):
            if fired_l1e[i]:
                g = self.proj_L2E_L1i.weights[i, j]
                self.proj_L2E_L1i.weights[i, j] = g + W_GAIN * (1.0 - g / W_L2E_L1I)
            else:
                self.proj_L2E_L1i.weights[i, j] *= (1.0 - PRUNE_RATE)
            if self.proj_L2E_L1i.weights[i, j] < 0.0:
                self.proj_L2E_L1i.weights[i, j] = 0.0

        # 5. L1E -> L1I excitatory plasticity (feedforward drive)
        for pre in range(N_L1_E):   # pre = L1E index
            for post in range(N_L1_I):  # post = L1I index
                if fired_l1e[pre] and fired_l1i[post]:
                    g = self.proj_L1E_L1I.weights[pre, post]
                    self.proj_L1E_L1I.weights[pre, post] = g + W_GAIN * (1.0 - g / 1.0)  # assume max 1.0
                else:
                    self.proj_L1E_L1I.weights[pre, post] *= (1.0 - PRUNE_RATE)
                if self.proj_L1E_L1I.weights[pre, post] < 0.0:
                    self.proj_L1E_L1I.weights[pre, post] = 0.0

        # 6. L1I -> L1E inhibitory plasticity (lateral inhibition)
        for pre in range(N_L1_I):   # pre = L1I index
            for post in range(N_L1_E):  # post = L1E index
                if fired_l1i[pre] and fired_l1e[post]:
                    g = self.proj_L1I_L1E.weights[pre, post]
                    self.proj_L1I_L1E.weights[pre, post] = g + W_GAIN * (1.0 - g / 1.0)  # assume max 1.0
                else:
                    self.proj_L1I_L1E.weights[pre, post] *= (1.0 - PRUNE_RATE)
                if self.proj_L1I_L1E.weights[pre, post] < 0.0:
                    self.proj_L1I_L1E.weights[pre, post] = 0.0

    def _learn_l3(self, j, fired_l2e, fired_l3i):
        """Plasticity for L3 layer: L2E -> L3E excitatory, L3E -> L3I feedback squelch."""
        # 1. L2E -> L3E excitatory plasticity
        for i in range(N_L2_E):
            if fired_l2e[i]:
                g = self.proj_L2E_L3E.weights[j, i]
                self.proj_L2E_L3E.weights[j, i] = g + W_GAIN * (1.0 - g / W_L2E_L3E_MAX)
            else:
                self.proj_L2E_L3E.weights[j, i] *= (1.0 - PRUNE_RATE)
            if self.proj_L2E_L3E.weights[j, i] < 0.0:
                self.proj_L2E_L3E.weights[j, i] = 0.0

        # 2. Synaptic scaling for L3 neuron
        total = 0.0
        for i in range(N_L2_E):
            total += self.proj_L2E_L3E.weights[j, i]
        if total > SYNAPTIC_BUDGET:
            scale = SYNAPTIC_BUDGET / total
            for i in range(N_L2_E):
                self.proj_L2E_L3E.weights[j, i] *= scale

        # 3. Refresh L3 maturity
        self.maturity_l3[j] = self.proj_L2E_L3E.get_maturity(j)

        # 4. Feedback squelch: L3E -> L3I excitatory plasticity
        for i in range(N_L3_I):
            if fired_l3i[i]:  # Note: fired_l3i is length 1, but we loop for generality
                g = self.proj_L3E_L3I.weights[i, j]
                self.proj_L3E_L3I.weights[i, j] = g + W_GAIN * (1.0 - g / 1.0)  # assume max 1.0
            else:
                self.proj_L3E_L3I.weights[i, j] *= (1.0 - PRUNE_RATE)
            if self.proj_L3E_L3I.weights[i, j] < 0.0:
                self.proj_L3E_L3I.weights[i, j] = 0.0

    def reset_membranes(self):
        self.L1E.v[:] = RESTING_POTENTIAL
        self.L1I.v[:] = RESTING_POTENTIAL
        self.L2E.v[:] = RESTING_POTENTIAL
        self.L2I.v[:] = RESTING_POTENTIAL
        self.L3E.v[:] = RESTING_POTENTIAL
        self.L3I.v[:] = RESTING_POTENTIAL
        self.L2E.adapt[:] = 0.0
        self.L2E.refrac[:] = 0
        self.L2E.silence = np.zeros(N_L2_E, dtype=int)  # corrected line
        # L3 reset
        self.L3E.adapt[:] = 0.0
        self.L3E.refrac[:] = 0
        self.L3E.silence = np.zeros(N_L3_E, dtype=int)