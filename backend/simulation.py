"""
SimulationEngine -- steppable wrapper around the spiking network.

Spikes are delivered immediately (no conduction delays). Every neuron has a
3D position for display; the L2 positions in L2_HOMES remain so the layout
is meaningful in the viewport, but they no longer imply any timing.

Learning architecture:
  - L1E neurons are treated as pre-trained pixel encoders: weights are fixed
    at [-1.0, 1.0] and learning_rate = 0.  They fire whenever the external
    pixel is active and they are not suppressed by their paired L1I neuron.
  - L2E neurons carry a homeostatic weight budget equal to the threshold.
    Only their positive (feedforward) incoming weights are counted; the
    inhibitory index-0 weight is excluded.  As learning strengthens active
    synapses, the budget normalisation weakens the others, producing
    competitive receptive-field emergence.
  - L1I / L2I (inhibitory) neurons carry NO budget.  Instead, each
    individual incoming weight is capped at the threshold.  This lets timing
    dynamics train freely without distorting receptive fields.

With slow leak_l2 (~0.01) and small initial feedforward weights, L2E neurons
require many volleys to fire at first (classic LIF accumulation).  As
synapses specialise, they fire from a single volley (pattern integrator).
The charge-ring visualisation makes this transition directly observable.

L2 competition (source of winner-take-all):
  Competition is produced by the shared inhibitory neuron L2I, not by a
  procedural reset.  When several L2E cross threshold in the same volley one
  fires and drives L2I, which laterally inhibits the other threshold-crossers
  (the near-winners) through the L2I->L2E gate.  Subthreshold neurons are left
  untouched, so charge accumulated across volleys is preserved and every unit
  can eventually win.  Each gate's strength is learned per neuron by the
  inhibitory-plasticity rule (Neuron.apply_inhibition) and saturates below
  threshold, so competition self-organizes and cannot collapse to a permanent
  single winner.  (An earlier version reset all non-winners to rest each step,
  which destroyed subthreshold evidence and locked the network to one neuron.)
"""

from __future__ import annotations

import os
import sys
from collections import deque, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import InputLayer                       # noqa: E402
from cortical_column_flexible import CorticalColumn  # noqa: E402


PATTERNS = {
    'row 0':    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    'row 1':    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    'row 2':    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    'col 0':    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    'col 1':    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    'col 2':    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    'diag \\':  [1, 0, 0, 0, 1, 0, 0, 0, 1],
    'diag /':   [0, 0, 1, 0, 1, 0, 1, 0, 0],
}

# 3D layout for the 8 L2 output neurons — evenly spaced ring in the XY plane.
import math as _math
_R, _Z = 3.2, 4.0
L2_HOMES = [
    (round(_R * _math.cos(k * _math.pi / 4), 4),
     round(_R * _math.sin(k * _math.pi / 4), 4),
     _Z)
    for k in range(8)
]

N_PIX = 9
N_OUT = 8
GRID = 2.2
L2E_FANIN = 1 + N_PIX     # [local_I_placeholder, *pixels]
FREQ_WINDOW = 40
WEIGHT_EPS = 1e-6
LOG_MAX = 400

# Episode-based competition window (interpretation only -- see _update_episode).
# An episode groups the L2 spikes produced across one or more volley bursts, and
# the "winner" is resolved from that spike history only when the episode ends,
# instead of an instantaneous per-step argmax. These two knobs are the episode
# end conditions; neither touches learning, WTA, or membrane dynamics.
EPISODE_QUIET_K = 5    # Condition A: end after this many consecutive L2-silent steps (spec: 3-5)
EPISODE_MAX_LEN = 12   # Condition B: hard cap on episode length in steps (spec: 8-12)

# L2 lateral inhibition ("adaptive gate") parameters. Competition in L2 is
# produced by the shared inhibitory neuron L2I suppressing near-winners through
# the L2I->L2E synapse, NOT by a procedural hard reset. Each L2E owns one gate
# from L2I whose strength is learned by the inhibitory-plasticity rule
# (Neuron.apply_inhibition): it grows when it suppresses a neuron that was close
# to firing and saturates at L2_GATE_WMAX. These values were chosen by a
# parameter sweep (see the competition investigation) as the point giving the
# broadest participation with the most per-pattern differentiation; crucially
# L2_GATE_WMAX < thr_l2 so a saturated gate can't fully reset the membrane.
L2_GATE_INIT = -0.5    # initial gate weight (magnitude 0.5)
L2_GATE_WMAX = 1.5     # saturation ceiling for the gate magnitude (< thr_l2)
L2_GATE_ETA = 0.1      # inhibitory-plasticity learning rate for the gate


class SimulationEngine:
    def __init__(self, seed: int = 1,
                 threshold: float = 1.0,
                 threshold_l2: float = 4.0,
                 leak_l1: float = 0.10,
                 leak_l2: float = 0.01,
                 learning_rate: float = 0.05,
                 weight_cap: float = 1.0,
                 refractory: int = 2,
                 volley_period: int = 4):
        self.params = dict(seed=seed, threshold=threshold, threshold_l2=threshold_l2,
                           leak_l1=leak_l1, leak_l2=leak_l2,
                           learning_rate=learning_rate, weight_cap=weight_cap,
                           refractory=refractory, volley_period=volley_period)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        p = self.params
        rng = np.random.default_rng(p['seed'])
        thr_l1 = p['threshold']      # L1 neurons fire on a single pixel hit
        thr_l2 = p['threshold_l2']   # L2 neurons must accumulate many volleys

        self.l1 = InputLayer(n_neurons=N_PIX, threshold=thr_l1,
                             refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                             weight_cap=thr_l1, leak_rate=p['leak_l1'],
                             n_feedback_inputs=N_OUT)
        # L1E: pre-trained pixel encoders — fixed weights, no learning.
        for e in self.l1.excitatory_neurons:
            e.weights = np.array([-1.0, 1.0])
            e.learning_rate = 0.0
            e.weight_budget = None
        # L1I: weights pre-set to cap (thr_l1) so a single L2E winner reliably
        # fires them in one step, producing one-step feedback inhibition of L1E.
        for inh in self.l1.inhibitory_neurons:
            inh.weights = np.ones(N_OUT) * thr_l1

        self.l2 = CorticalColumn(n_neurons=N_OUT, threshold=thr_l2,
                                 refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                                 weight_cap=thr_l2, leak_rate=p['leak_l2'])
        self.l2.setup_connectivity(n_feedforward_inputs=N_PIX, n_feedback_inputs=0)
        self.l2.finalize_connections()
        # L2I->L2E gates start weak; they are the real source of L2 competition
        # (see step 2c) and self-tune via inhibitory plasticity.
        self.l2.set_local_inhibition_weights(L2_GATE_INIT)
        # E→I weight = thr_l2 so a single L2E winner immediately fires L2I.
        self.l2.set_lateral_excitation_weights(thr_l2)
        # Small random feedforward weights: neurons must accumulate across many
        # volleys initially (LIF phase), then specialise toward single-volley
        # firing (pattern integrator phase).
        ff_weights = rng.uniform(0.05, 0.20, size=(N_OUT, N_PIX))
        self.l2.set_feedforward_weights(ff_weights)
        self.l2.inhibitory_neuron.refractory_period = 0

        self.neurons: dict[str, object] = {}
        self.meta: dict[str, dict] = {}
        self._register_neurons()

        # Budget / cap assignment:
        #   L2E → budget = thr_l2 (positive feedforward weights only).
        #   L1E → no budget (weights fixed, learning disabled).
        #   L1I → cap = thr_l1; L2I → cap = thr_l2.
        for nid, n in self.neurons.items():
            if self.meta[nid]['type'] == 'E' and nid.startswith('L2'):
                n.weight_budget = thr_l2
                # Adaptive lateral-inhibition gate: dedicated (lower) saturation
                # ceiling and its own learning rate, independent of feedforward.
                n.inhibitory_weight_cap = L2_GATE_WMAX
                n.inhibitory_learning_rate = L2_GATE_ETA
            else:
                n.weight_budget = None
                if self.meta[nid]['type'] == 'I':
                    n.weight_cap = thr_l2 if nid.startswith('L2') else thr_l1

        self.l1i_hold = np.zeros(N_PIX)   # L1I spike latch: held until next volley
        self.input_vec = np.array(PATTERNS['row 0'], dtype=float)
        self.timestep = 0
        self.spiked = defaultdict(bool)
        self.freq = {nid: deque(maxlen=FREQ_WINDOW) for nid in self.neurons}
        self.emitted: list[str] = []   # synapse IDs that carried a spike this step
        self._pulses: dict[str, float] = {}
        self._holds: dict[str, float] = {}
        self.event_log: deque = deque(maxlen=LOG_MAX)
        self._log_seq = 0
        self._weights_snapshot = self._all_weights()
        self.changed_synapses: list[dict] = []
        self.l2_drive: dict[str, float] = {}
        self.winner: str | None = None
        self._inh_events: list[tuple] = []   # (neuron_id, event) from this step's discharges

        # Episode-based competition window (interpretation only; see _update_episode).
        self.episode_active = False
        self.episode_timer = 0
        self.episode_last_spike_time = -1
        self.episode_l2_spikes: list[tuple] = []   # list of (timestep, neuron_id)

        self._log('backend', f'network built (seed={p["seed"]}, immediate delivery, '
                             f'{len(self.neurons)} neurons, {len(self.synapses)} synapses)')

    def _register_neurons(self):
        for i in range(N_PIX):
            r, c = divmod(i, 3)
            nid = f'L1E{i}'
            self.neurons[nid] = self.l1.excitatory_neurons[i]
            self.meta[nid] = dict(id=nid, label=f'in {i}', layer='L1', type='E',
                                  threshold=self.params['threshold'],
                                  pos=[(c - 1) * GRID, (1 - r) * GRID, 0.0])
        for i in range(N_PIX):
            r, c = divmod(i, 3)
            nid = f'L1I{i}'
            self.neurons[nid] = self.l1.inhibitory_neurons[i]
            self.meta[nid] = dict(id=nid, label=f'inh {i}', layer='L1', type='I',
                                  threshold=self.params['threshold'],
                                  pos=[(c - 1) * GRID, (1 - r) * GRID, -2.0])
        for j in range(N_OUT):
            nid = f'L2E{j}'
            self.neurons[nid] = self.l2.excitatory_neurons[j]
            self.meta[nid] = dict(id=nid, label=f'out {j}', layer='L2', type='E',
                                  threshold=self.params['threshold_l2'], pos=list(L2_HOMES[j]))
        self.neurons['L2I'] = self.l2.inhibitory_neuron
        self.meta['L2I'] = dict(id='L2I', label='inhib', layer='L2', type='I',
                                threshold=self.params['threshold_l2'], pos=[0.0, 0.0, 6.0])

        self.synapses: list[dict] = []
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.synapses.append(dict(id=f'ff{i}->{j}', source=f'L1E{i}', target=f'L2E{j}', kind='feedforward'))
        for j in range(N_OUT):
            self.synapses.append(dict(id=f'inh->{j}', source='L2I', target=f'L2E{j}', kind='inhibition'))
        for j in range(N_OUT):
            self.synapses.append(dict(id=f'{j}->inh', source=f'L2E{j}', target='L2I', kind='excitation'))
        for i in range(N_PIX):
            self.synapses.append(dict(id=f'li{i}', source=f'L1I{i}', target=f'L1E{i}', kind='inhibition'))
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.synapses.append(dict(id=f'fb{j}->{i}', source=f'L2E{j}', target=f'L1I{i}', kind='feedback'))

    # --------------------------------------------------------------- controls
    def reset(self):
        self._build()

    def set_pattern(self, name: str):
        if name not in PATTERNS:
            raise KeyError(name)
        self.input_vec = np.array(PATTERNS[name], dtype=float)
        self._log('control', f'pattern set: {name}')

    def set_input(self, vec):
        self.input_vec = np.array(vec, dtype=float).reshape(N_PIX)
        self._log('control', 'input vector set')

    def toggle_pixel(self, i: int):
        self.input_vec[i] = 0.0 if self.input_vec[i] > 0.5 else 1.0

    def clear_input(self):
        self.input_vec = np.zeros(N_PIX)
        self._log('control', 'input cleared')

    def random_pattern(self):
        self.input_vec = (np.random.default_rng().random(N_PIX) > 0.5).astype(float)
        self._log('control', 'random input')

    def inject_noise(self, prob: float = 0.15):
        flip = np.random.default_rng().random(N_PIX) < prob
        self.input_vec = np.where(flip, 1.0 - self.input_vec, self.input_vec)
        self._log('control', f'noise injected (p={prob:.2f})')

    def stimulate(self, neuron_id: str, magnitude: float = 1.0, continuous: bool = False):
        if neuron_id not in self.neurons:
            raise KeyError(neuron_id)
        if continuous:
            self._holds.pop(neuron_id, None) if magnitude == 0 else self._holds.__setitem__(neuron_id, magnitude)
        else:
            self._pulses[neuron_id] = self._pulses.get(neuron_id, 0.0) + magnitude
        self._log('control', f'stimulate {neuron_id} (+{magnitude:g}{", hold" if continuous else ""})')

    # ------------------------------------------------------------------- step
    def step(self) -> dict:
        l1, l2 = self.l1, self.l2
        t = self.timestep

        # 1. L1E: [paired I1's previous spike (local inhibition), external pixel].
        #    Pixels fire in synchronized volleys so charge arrives in bursts.
        #    Excitatory pixel drive goes through receive_input; the inhibitory
        #    discharge is delivered as its own event via apply_inhibition, which
        #    also runs the inhibitory-gate plasticity rule. The net membrane before
        #    threshold (ext - |w_inh|) is identical to the old summed delivery, so
        #    spike timing and event ordering are unchanged.
        volley = (t % self.params['volley_period'] == 0)
        self._inh_events = []
        for i, e in enumerate(l1.excitatory_neurons):
            ext = 1.0 if (volley and self.input_vec[i] > 0.5) else 0.0
            e.receive_input(np.array([0.0, ext]))
            # Inhibition is only relevant on volley steps; the hold persists from
            # the previous volley's L1I activity so refractory doesn't swallow it.
            inh = float(self.l1i_hold[i]) if volley else 0.0
            if inh > 0.5:
                for ev in e.apply_inhibition(np.array([1.0, 0.0])):
                    self._inh_events.append((f'L1E{i}', ev))

        self._apply_stim()

        # 2a. L1E fires (no competition).
        l1e = np.array([1.0 if e.check_threshold() else 0.0 for e in l1.excitatory_neurons])
        for k, e in enumerate(l1.excitatory_neurons):
            if l1e[k]:
                e.fire()

        # 2b. Deliver L1E spikes immediately to all L2E neurons.
        ff_vec = np.zeros(L2E_FANIN)
        for i in range(N_PIX):
            if l1e[i]:
                ff_vec[1 + i] = 1.0
        for j, e in enumerate(l2.excitatory_neurons):
            e.receive_input(ff_vec)

        # Capture pre-WTA potential for the charge visualisation.
        self.l2_drive = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}

        # 2c. L2 competition via adaptive lateral inhibition (NOT a hard reset).
        #     When several L2E cross threshold in the same volley, one fires; the
        #     shared inhibitory neuron L2I then suppresses the other threshold
        #     crossers (the near-winners) through its L2I->L2E gate. Subthreshold
        #     neurons are left alone, so charge accumulated across volleys is
        #     preserved and every unit can eventually win -- this is what breaks
        #     the single-winner collapse the hard reset caused. The gate strength
        #     is learned per neuron by Neuron.apply_inhibition (grows when it
        #     suppresses a near-winner, saturates at L2_GATE_WMAX), so competition
        #     self-organizes instead of relying on a hand-tuned constant.
        l2e = np.zeros(N_OUT)
        eligible = [j for j, e in enumerate(l2.excitatory_neurons) if e.check_threshold()]
        inhibited = []
        if eligible:
            winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
            l2.excitatory_neurons[winner].fire()
            l2e[winner] = 1.0
            # The winner drives L2I, which fires (E->I weight = thr_l2) and
            # laterally inhibits the near-winners it beat.
            l2.inhibitory_neuron.receive_input(l2e)
            l2i = 1.0 if l2.inhibitory_neuron.check_threshold() else 0.0
            if l2i:
                l2.inhibitory_neuron.fire()
                inh_spk = np.zeros(L2E_FANIN)
                inh_spk[0] = 1.0                       # index 0 = the L2I->L2E gate
                for j in eligible:
                    if j == winner:
                        continue                        # winner already fired / refractory
                    for ev in l2.excitatory_neurons[j].apply_inhibition(inh_spk):
                        self._inh_events.append((f'L2E{j}', ev))
                    inhibited.append(j)
        else:
            l2.inhibitory_neuron.receive_input(l2e)
            l2i = 1.0 if l2.inhibitory_neuron.check_threshold() else 0.0
            if l2i:
                l2.inhibitory_neuron.fire()

        # 2d. Deliver L2E winner spike immediately to all L1I neurons (feedback).
        #     l2e is length N_OUT with a 1 at the winner index, matching each
        #     L1I neuron's N_OUT-dimensional afferent weight vector.
        for inh in l1.inhibitory_neurons:
            inh.receive_input(l2e)

        # 2e. L1I fires after receiving L2E feedback.
        l1i = np.array([1.0 if n.check_threshold() else 0.0 for n in l1.inhibitory_neurons])
        for k, n in enumerate(l1.inhibitory_neurons):
            if l1i[k]:
                n.fire()

        # 3. Collect synapse IDs that carried a spike this step (for edge flash).
        self.emitted = []
        for i in range(N_PIX):
            if l1e[i]:
                for j in range(N_OUT):
                    self.emitted.append(f'ff{i}->{j}')
        for j in range(N_OUT):
            if l2e[j]:
                for i in range(N_PIX):
                    self.emitted.append(f'fb{j}->{i}')
                self.emitted.append(f'{j}->inh')
        for j in inhibited:
            self.emitted.append(f'inh->{j}')
        for i in range(N_PIX):
            if l1i[i]:
                self.emitted.append(f'li{i}')

        # 4. Advance membrane state (leak + refractory countdown).
        for e in l1.excitatory_neurons:
            e.update()
        for n in l1.inhibitory_neurons:
            n.update()
        for e in l2.excitatory_neurons:
            e.update()
        l2.inhibitory_neuron.update()

        # 5. Bookkeeping.
        self._record_spikes(l1e, l1i, l2e, l2i)
        # Latch L1I activity so it blocks L1E on the NEXT volley, not the next
        # time step (where refractory would silently swallow the inhibition).
        if volley:
            self.l1i_hold = l1i
        self.timestep += 1
        self._detect_weight_changes()
        self._log_inhibitory_events()
        self._update_episode(l2e, t)
        return self.dynamic_state()

    def _log_inhibitory_events(self):
        """Surface inhibitory-discharge plasticity into the event log for the
        dashboard. Only events that actually moved a gate (|delta_w| > eps) are
        logged, so a saturated gate (delta_w == 0) doesn't flood the panel; the
        full per-event debug record always lives on each neuron's
        last_inhibitory_events."""
        for nid, ev in self._inh_events:
            if abs(ev['delta_w']) > WEIGHT_EPS:
                self._log('inhibition',
                          f"{nid} gate: Vpre={ev['v_pre']:.3f} θ={ev['theta']:.2f} "
                          f"p={ev['p']:.2f} |w| {ev['w_before']:.3f}->{ev['w_after']:.3f} "
                          f"(Δ={ev['delta_w']:+.4f})")

    def _apply_stim(self):
        for nid, mag in list(self._pulses.items()):
            self.neurons[nid].potential += mag
        self._pulses.clear()
        for nid, mag in self._holds.items():
            self.neurons[nid].potential += mag

    def _record_spikes(self, l1e, l1i, l2e, l2i):
        for i in range(N_PIX):
            self.spiked[f'L1E{i}'] = bool(l1e[i]); self.freq[f'L1E{i}'].append(l1e[i])
            self.spiked[f'L1I{i}'] = bool(l1i[i]); self.freq[f'L1I{i}'].append(l1i[i])
        for j in range(N_OUT):
            self.spiked[f'L2E{j}'] = bool(l2e[j]); self.freq[f'L2E{j}'].append(l2e[j])
        self.spiked['L2I'] = bool(l2i); self.freq['L2I'].append(l2i)

    def _update_episode(self, l2e, t):
        """
        Episode-based competition interpretation. This is the ONLY thing that
        changed relative to the old instantaneous winner readout: it decides
        *when* competition is considered resolved and *which* neuron is reported
        as the winner. It reads only l2e (this step's L2E spikes) and t, and
        writes only the episode_* fields and self.winner. It never touches a
        neuron, a weight, a potential, WTA, or the learning rule -- so LIF and
        plasticity are byte-for-byte unchanged.

        Structure:
          - An episode STARTS on a volley tick, but only if one is not already
            running (so a single episode can span several volleys up to T_max
            instead of being reset every volley).
          - While active, every L2E spike this step is appended to the history
            and the last-spike time is updated.
          - The episode ENDS on Condition A (K consecutive L2-silent steps) or
            Condition B (episode_timer reaches T_max), whichever comes first.
          - The winner is then resolved from the spike history alone
            (latest-spike, then most-spikes tiebreak) -- no argmax over membrane
            potentials, no global ranking.
        """
        volley = (t % self.params['volley_period'] == 0)
        if volley and not self.episode_active:
            self.episode_active = True
            self.episode_timer = 0
            self.episode_last_spike_time = -1
            self.episode_l2_spikes = []

        if not self.episode_active:
            return

        # Record this step's L2E spikes. WTA fires at most one L2E per step, but
        # we record generally so any co-firing would also be captured.
        for j in range(N_OUT):
            if l2e[j]:
                self.episode_l2_spikes.append((t, f'L2E{j}'))
                self.episode_last_spike_time = t
        self.episode_timer += 1

        # Condition A: silent for K consecutive steps (counted from the last
        # spike, or from episode start if nothing has fired yet).
        if self.episode_last_spike_time >= 0:
            silent = t - self.episode_last_spike_time
        else:
            silent = self.episode_timer - 1
        # Condition B: episode length cap.
        if silent >= EPISODE_QUIET_K or self.episode_timer >= EPISODE_MAX_LEN:
            self._resolve_episode()
            self.episode_active = False

    def _resolve_episode(self):
        """Resolve the episode winner from spike history only.

        Rule 1 (primary): the neuron with the LATEST spike time wins.
        Rule 2 (tiebreak): if several neurons share that latest spike time, the
        one with the MOST spikes over the whole episode wins.
        An episode with no L2 spikes leaves the previous winner untouched.
        """
        if not self.episode_l2_spikes:
            return
        latest_t = max(ts for ts, _ in self.episode_l2_spikes)
        last_spikers = [nid for ts, nid in self.episode_l2_spikes if ts == latest_t]
        if len(set(last_spikers)) > 1:
            counts: dict[str, int] = {}
            for _, nid in self.episode_l2_spikes:
                counts[nid] = counts.get(nid, 0) + 1
            winner = max(set(last_spikers), key=lambda n: counts[n])
        else:
            winner = last_spikers[0]
        if winner != self.winner:
            self._log('learning', f'episode winner -> {winner} '
                                  f'(spikes={len(self.episode_l2_spikes)}, last_t={latest_t})')
        self.winner = winner

    # ------------------------------------------------------------- weight diff
    def _all_weights(self) -> dict:
        w = {}
        for j in range(N_OUT):
            arr = self.l2.excitatory_neurons[j]._weights_array
            w[f'inh->{j}'] = float(arr[0])
            for i in range(N_PIX):
                w[f'ff{i}->{j}'] = float(arr[1 + i])
        iw = self.l2.inhibitory_neuron._weights_array
        for j in range(N_OUT):
            w[f'{j}->inh'] = float(iw[j])
        for i in range(N_PIX):
            w[f'li{i}'] = float(self.l1.excitatory_neurons[i].weights[0])
            fbw = self.l1.inhibitory_neurons[i].weights
            for j in range(N_OUT):
                w[f'fb{j}->{i}'] = float(fbw[j])
        return w

    def _detect_weight_changes(self):
        now = self._all_weights()
        self.changed_synapses = [dict(id=sid, weight=round(v, 4))
                                 for sid, v in now.items()
                                 if abs(v - self._weights_snapshot.get(sid, v)) > WEIGHT_EPS]
        self._weights_snapshot = now

    # ----------------------------------------------------------------- access
    def firing_freq(self, nid: str) -> float:
        d = self.freq[nid]
        return float(sum(d) / len(d)) if d else 0.0

    def activation(self, nid: str) -> float:
        thr = self.meta[nid]['threshold'] or 1.0
        return float(self.neurons[nid].potential / thr)

    def _log(self, kind: str, message: str):
        self._log_seq += 1
        self.event_log.append(dict(seq=self._log_seq, t=self.timestep, kind=kind, message=message))

    # ------------------------------------------------------------ serialization
    def topology(self) -> dict:
        weights = self._all_weights()
        neurons = [dict(**self.meta[nid]) for nid in self.neurons]
        synapses = [dict(**s, weight=round(weights.get(s['id'], 0.0), 4)) for s in self.synapses]
        return dict(neurons=neurons, synapses=synapses, layers=['L1', 'L2'],
                    patterns=list(PATTERNS.keys()),
                    pattern_vectors={k: list(map(int, v)) for k, v in PATTERNS.items()},
                    grid=dict(rows=3, cols=3), params=self.params)

    def dynamic_state(self) -> dict:
        neurons = []
        for nid, n in self.neurons.items():
            pot = self.l2_drive.get(nid, float(n.potential))
            thr = self.meta[nid]['threshold'] or 1.0
            neurons.append(dict(id=nid, potential=round(pot, 4),
                                activation=round(pot / thr, 4),
                                spiked=self.spiked[nid], freq=round(self.firing_freq(nid), 4),
                                refractory=int(n.refractory_timer),
                                assembly=(self.winner if nid == self.winner else None)))
        return dict(timestep=self.timestep, running=False, neurons=neurons,
                    changed_synapses=self.changed_synapses,
                    emitted=self.emitted,
                    input=self.input_vec.astype(int).tolist(), winner=self.winner,
                    episode=dict(active=self.episode_active, timer=self.episode_timer,
                                 spikes=len(self.episode_l2_spikes),
                                 participants=sorted({nid for _, nid in self.episode_l2_spikes})),
                    stats=self.stats(), log=list(self.event_log)[-12:])

    def stats(self) -> dict:
        active = sum(1 for nid in self.neurons if abs(self.neurons[nid].potential) > 1e-3)
        firing = sum(1 for nid in self.neurons if self.spiked[nid])
        pots = [self.neurons[nid].potential for nid in self.neurons]
        weights = list(self._all_weights().values())
        rate = float(np.mean([self.firing_freq(nid) for nid in self.neurons]))
        return dict(total=len(self.neurons), active=active, firing=firing,
                    avg_activation=round(float(np.mean(np.abs(pots))), 4),
                    firing_rate=round(rate, 4), avg_weight=round(float(np.mean(weights)), 4),
                    winner=self.winner)
