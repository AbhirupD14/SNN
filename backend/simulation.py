"""
SimulationEngine -- a steppable wrapper around the (unmodified) spiking network in
neuron.py / layers.py / cortical_column_flexible.py.

This version adds distance-based CONDUCTION DELAYS. Every neuron has a 3D position;
a spike travels from source to target over `round(distance / SPEED)` timesteps.
The long feedforward (L1E->L2E) and feedback (L2E->L1I) axons are delayed; local
inhibition and the WTA loop stay same-step. Because the L2 neurons integrate with a
fast leak, they act as COINCIDENCE DETECTORS: a pattern fires the L2 neuron whose
delays make that pattern's pixels arrive together. The L2 positions were chosen so
each of the 8 line patterns has a distinct coincidence home -- this breaks the
single-winner tyranny without any adaptation term.

The engine is the only contact point between the neural computation and the
dashboard: it owns cross-timestep state, exposes control verbs, and produces
plain-Python snapshots of the static topology and the per-timestep dynamic state
(including the charges currently in flight, for the traveling-orb animation).
"""

from __future__ import annotations

import os
import sys
from collections import deque, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import InputLayer                       # noqa: E402
from cortical_column_flexible import CorticalColumn  # noqa: E402


# --- the first experiment: 8 straight lines on a 3x3 grid (row-major) ----------
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

# Coincidence-home position for each L2 neuron (found offline so that each line
# pattern's active pixels arrive coincidentally at exactly one neuron). Index order
# matches the pattern order above, but the neurons are not "told" their pattern --
# the mapping emerges from the delays.
L2_HOMES = [
    (-0.40, -3.20, 3.09),   # 0
    (0.00, -2.00, 3.56),    # 1
    (-0.40, 3.20, 3.09),    # 2
    (3.20, -0.40, 3.09),    # 3
    (-2.00, 0.00, 3.56),    # 4
    (-3.20, -0.40, 3.09),   # 5
    (-3.60, -3.60, 3.09),   # 6
    (3.60, -3.60, 3.09),    # 7
]

N_PIX = 9
N_OUT = 8
GRID = 2.2
SPEED = 0.30              # distance units travelled per timestep (sets conduction delays)
L2E_FANIN = 1 + N_PIX     # [local_I, *pixels]
FREQ_WINDOW = 40
WEIGHT_EPS = 1e-6
LOG_MAX = 400


class SimulationEngine:
    def __init__(self, seed: int = 1,
                 threshold: float = 0.5, leak_l1: float = 0.10, leak_l2: float = 0.40,
                 learning_rate: float = 0.05, weight_cap: float = 1.0, refractory: int = 2,
                 volley_period: int = 32):
        self.params = dict(seed=seed, threshold=threshold, leak_l1=leak_l1, leak_l2=leak_l2,
                           learning_rate=learning_rate, weight_cap=weight_cap,
                           refractory=refractory, volley_period=volley_period, speed=SPEED)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        p = self.params
        rng = np.random.default_rng(p['seed'])

        self.l1 = InputLayer(n_neurons=N_PIX, threshold=p['threshold'],
                             refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                             weight_cap=p['weight_cap'], leak_rate=p['leak_l1'],
                             n_feedback_inputs=N_OUT)
        for e in self.l1.excitatory_neurons:
            e.weights = np.array([-1.0, 1.0])            # [from paired I1 (quiet), external]
        for inh in self.l1.inhibitory_neurons:
            inh.weights = rng.uniform(0.35, 0.55, size=N_OUT)   # E2_j -> I1_i feedback

        self.l2 = CorticalColumn(n_neurons=N_OUT, threshold=p['threshold'],
                                 refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                                 weight_cap=p['weight_cap'], leak_rate=p['leak_l2'])
        self.l2.setup_connectivity(n_feedforward_inputs=N_PIX, n_feedback_inputs=0)
        self.l2.finalize_connections()
        self.l2.set_local_inhibition_weights(-0.6)
        self.l2.set_lateral_excitation_weights(0.5)
        # Feedforward init UNIFORM: every L2 neuron starts identical, so the initial
        # winner for a pattern is decided purely by conduction-delay coincidence, not by
        # weight noise. Learning then sharpens each neuron's own receptive field.
        self.l2.set_feedforward_weights(0.42)
        self.l2.inhibitory_neuron.refractory_period = 0   # fast interneuron gates every volley

        self.neurons: dict[str, object] = {}
        self.meta: dict[str, dict] = {}
        self._register_neurons()
        self._compute_delays()

        # Uniform homeostatic budget: every neuron's positive (excitatory) afferents
        # share a fixed total equal to their initial sum. Applied to all neurons the
        # same way -- no layer is special-cased.
        for n in self.neurons.values():
            pos = n.weights[n.weights > 0]
            n.weight_budget = float(pos.sum()) if pos.size else None

        self.l1i_prev = np.zeros(N_PIX)
        self.input_vec = np.array(PATTERNS['row 0'], dtype=float)
        self.timestep = 0
        self.spiked = defaultdict(bool)
        self.freq = {nid: deque(maxlen=FREQ_WINDOW) for nid in self.neurons}
        self.inbox: dict[int, dict[str, np.ndarray]] = defaultdict(dict)  # step -> {target: vec}
        self.in_flight: list[dict] = []          # charges currently travelling (for the viz)
        self.emitted: list[dict] = []
        self._pulses: dict[str, float] = {}
        self._holds: dict[str, float] = {}
        self.event_log: deque = deque(maxlen=LOG_MAX)
        self._log_seq = 0
        self._weights_snapshot = self._all_weights()
        self.changed_synapses: list[dict] = []
        self.l2_drive: dict[str, float] = {}
        self.winner: str | None = None

        self._log('backend', f'network built (seed={p["seed"]}, delays via SPEED={SPEED}, '
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
                                  threshold=self.params['threshold'], pos=list(L2_HOMES[j]))
        self.neurons['L2I'] = self.l2.inhibitory_neuron
        self.meta['L2I'] = dict(id='L2I', label='inhib', layer='L2', type='I',
                                threshold=self.params['threshold'], pos=[0.0, 0.0, 6.0])

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

    def _dist(self, a: str, b: str) -> float:
        return float(np.linalg.norm(np.array(self.meta[a]['pos']) - np.array(self.meta[b]['pos'])))

    def _compute_delays(self):
        """Conduction delay (in timesteps) for the delayed axons: feedforward + feedback."""
        self.ff_delay = np.array([[max(1, round(self._dist(f'L1E{i}', f'L2E{j}') / SPEED))
                                   for j in range(N_OUT)] for i in range(N_PIX)])
        self.fb_delay = np.array([[max(1, round(self._dist(f'L2E{j}', f'L1I{i}') / SPEED))
                                   for i in range(N_PIX)] for j in range(N_OUT)])
        self.syn_delay = {}
        for i in range(N_PIX):
            for j in range(N_OUT):
                self.syn_delay[f'ff{i}->{j}'] = int(self.ff_delay[i, j])
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.syn_delay[f'fb{j}->{i}'] = int(self.fb_delay[j, i])

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
    def _schedule(self, step: int, target: str, idx: int, fanin: int):
        box = self.inbox[step]
        if target not in box:
            box[target] = np.zeros(fanin)
        box[target][idx] += 1.0

    def step(self) -> dict:
        l1, l2 = self.l1, self.l2
        t = self.timestep
        arrivals = self.inbox.pop(t, {})

        # 1. L1 E: [paired I1's previous spike (local inhibition), external pixel drive].
        #    The input is delivered as periodic synchronized VOLLEYS (a spike every
        #    volley_period steps) rather than a constant DC drive, so the active pixels
        #    of a pattern fire together and their charges arrive coincidentally at L2,
        #    with silent gaps for the fast leak to clear between volleys.
        volley = (t % self.params['volley_period'] == 0)
        for i, e in enumerate(l1.excitatory_neurons):
            ext = 1.0 if (volley and self.input_vec[i] > 0.5) else 0.0
            e.receive_input(np.array([self.l1i_prev[i], ext]))
        # L1 I: delayed top-down feedback arrivals (E2_j -> I1_i).
        for i, inh in enumerate(l1.inhibitory_neurons):
            inh.receive_input(arrivals.get(f'L1I{i}', np.zeros(N_OUT)))
        # L2 E: delayed feedforward arrivals; index 0 (local inhibition) is left to the
        # same-step blanket, so it stays zero here.
        for j, e in enumerate(l2.excitatory_neurons):
            e.receive_input(arrivals.get(f'L2E{j}', np.zeros(L2E_FANIN)))

        self._apply_stim()

        # 2a. L1 fires with no competition.
        l1e = np.array([1.0 if e.check_threshold() else 0.0 for e in l1.excitatory_neurons])
        l1i = np.array([1.0 if n.check_threshold() else 0.0 for n in l1.inhibitory_neurons])
        for k, e in enumerate(l1.excitatory_neurons):
            if l1e[k]:
                e.fire()
        for k, n in enumerate(l1.inhibitory_neurons):
            if l1i[k]:
                n.fire()

        # capture the coincidence drive (membrane just before the blanket resets losers),
        # so the dashboard can show L2 neurons charging up and competing.
        self.l2_drive = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}

        # 2b. L2 winner-take-all resolved within the step (first to threshold wins;
        #     shared inhibitor fires the same step and blankets the pool).
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

        # 3. launch this step's spikes down the delayed axons (feedforward + feedback).
        self.emitted = []
        for i in range(N_PIX):
            if l1e[i]:
                for j in range(N_OUT):
                    d = int(self.ff_delay[i, j])
                    self._schedule(t + d, f'L2E{j}', 1 + i, L2E_FANIN)
                    self.emitted.append(dict(syn=f'ff{i}->{j}', delay=d))
        for j in range(N_OUT):
            if l2e[j]:
                for i in range(N_PIX):
                    d = int(self.fb_delay[j, i])
                    self._schedule(t + d, f'L1I{i}', j, N_OUT)
                    self.emitted.append(dict(syn=f'fb{j}->{i}', delay=d))

        # 4. advance membrane state.
        for e in l1.excitatory_neurons:
            e.update()
        for n in l1.inhibitory_neurons:
            n.update()
        for e in l2.excitatory_neurons:
            e.update()
        l2.inhibitory_neuron.update()

        # 5. bookkeeping.
        self._record_spikes(l1e, l1i, l2e, l2i)
        self.l1i_prev = l1i
        self.timestep += 1
        self._advance_in_flight()
        self._detect_weight_changes()
        self._update_winner(l2e)
        return self.dynamic_state()

    def _apply_stim(self):
        for nid, mag in list(self._pulses.items()):
            self.neurons[nid].potential += mag
        self._pulses.clear()
        for nid, mag in self._holds.items():
            self.neurons[nid].potential += mag

    def _advance_in_flight(self):
        """Track charges still travelling so a reconnecting client can render them."""
        for c in self.emitted:
            self.in_flight.append(dict(syn=c['syn'], t0=self.timestep - 1, delay=c['delay']))
        self.in_flight = [c for c in self.in_flight if self.timestep - c['t0'] < c['delay']]

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
        synapses = [dict(**s, weight=round(weights.get(s['id'], 0.0), 4),
                         delay=self.syn_delay.get(s['id'], 0)) for s in self.synapses]
        return dict(neurons=neurons, synapses=synapses, layers=['L1', 'L2'],
                    patterns=list(PATTERNS.keys()),
                    pattern_vectors={k: list(map(int, v)) for k, v in PATTERNS.items()},
                    grid=dict(rows=3, cols=3), params=self.params)

    def dynamic_state(self) -> dict:
        neurons = []
        for nid, n in self.neurons.items():
            # For L2 excitatory neurons show the coincidence drive (pre-blanket) rather
            # than the post-reset 0, so they visibly charge and compete.
            pot = self.l2_drive.get(nid, float(n.potential))
            thr = self.meta[nid]['threshold'] or 1.0
            neurons.append(dict(id=nid, potential=round(pot, 4),
                                activation=round(pot / thr, 4),
                                spiked=self.spiked[nid], freq=round(self.firing_freq(nid), 4),
                                refractory=int(n.refractory_timer),
                                assembly=(self.winner if nid == self.winner else None)))
        return dict(timestep=self.timestep, running=False, neurons=neurons,
                    changed_synapses=self.changed_synapses,
                    emitted=self.emitted,                 # charges launched this step
                    in_flight=len(self.in_flight),
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
                    in_flight=len(self.in_flight), winner=self.winner)
