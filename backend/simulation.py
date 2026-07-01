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
        self.l2.set_local_inhibition_weights(-1.0)
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

        # 2c. L2 winner-take-all (same step as feedforward arrival).
        l2e = np.zeros(N_OUT)
        eligible = [j for j, e in enumerate(l2.excitatory_neurons) if e.check_threshold()]
        if eligible:
            winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
            l2.excitatory_neurons[winner].fire()
            l2e[winner] = 1.0
            for j, e in enumerate(l2.excitatory_neurons):
                if j != winner:
                    e.potential = e.resting_potential
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
        if l2i:
            for j in range(N_OUT):
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
        self._update_winner(l2e)
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

    def _update_winner(self, l2e):
        freqs = [self.firing_freq(f'L2E{j}') for j in range(N_OUT)]
        best = int(np.argmax(freqs))
        new = f'L2E{best}' if freqs[best] > 0 else None
        if new and new != self.winner:
            self._log('learning', f'winner -> {new} (freq {freqs[best]:.2f})')
        self.winner = new

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
