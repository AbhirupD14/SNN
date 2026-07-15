"""The one active SNN model: construction, one deterministic step, and state
snapshots for the dashboard.

There is a single scientific model here. No experiment-mode selectors, no legacy
ablations. Neuron behaviour lives in ``snn.neurons``; this module owns topology,
the causal event order, and serialization.

Topology (36 neurons)::

    9 external pixels
           |                       (sensory, frozen, subthreshold)
           v
      9 x L1E_s  ===============>  8 x L2E  ------>  1 x L2I
           ^        dense acc         |   ^             |
           |   frozen subtractive     |   +-------------+
      9 x L1I     hard wipe           |   frozen subtractive hard wipe
           ^                          |
           | instant paired relay     |
      9 x L1E_new  <==================+
                     dense acc feedback

Internal edges: 72 (L1E_s->L2E) + 72 (L2E->L1E_new) + 9 (L1E_new->L1I)
              + 8 (L2E->L2I) + 9 (L1I->L1E_s) + 8 (L2I->L2E) = 178.
Sensory pixel->L1E_s afferents are not serialized as edges (their source is not a
neuron); the pixel grid UI shows the input instead.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn.neurons import (  # noqa: E402
    ExcitatoryNeuron,
    InhibitoryNeuron,
    E_THRESHOLD,
    I_THRESHOLD,
    E_WEIGHT_CAP,
    SUBTRACTIVE_SIGN,
)
from backend.layout import generate_layout  # noqa: E402


# --- The four center-crossing patterns on the 3x3, 9-pixel surface ----------
PATTERNS = {
    'row 1':   [0, 0, 0, 1, 1, 1, 0, 0, 0],
    'col 1':   [0, 1, 0, 0, 1, 0, 0, 1, 0],
    'diag \\': [1, 0, 0, 0, 1, 0, 0, 0, 1],
    'diag /':  [0, 0, 1, 0, 1, 0, 1, 0, 0],
}
N_PIX = 9
N_OUT = 8

# --- Projection-specific initialization -------------------------------------
# One universal per-synapse cap (E_WEIGHT_CAP) is the biological bound. Each
# learned projection is *initialized* around a documented mean chosen so that the
# whole afferent vector sums BELOW threshold. That subthreshold total matters for
# the sign of the learning rule: with p = threshold - sum(acc_weights) > 0, a
# participating (+1) afferent potentiates and an absent (-1) afferent decays, so a
# selective receptive field forms; as the participating weights climb, the total
# approaches threshold and p -> 0, making the rule self-limiting. A uniform init
# near the mature active weight (theta / N_active) would instead put the *total*
# well above threshold (p < 0) and invert the rule -- so the init is deliberately
# a small uniform seed, not the mature target, and the rule sparsifies it.
#
#   sensory (L1E_s, 1 afferent, frozen): a subthreshold weight so a held pixel
#     integrates over three zero-leak steps before its source fires -- and a paired
#     L1I wipe resets that integration, which is what lets feedback modulate the
#     source presentation frequency.
#   feedforward (L1E_s->L2E, 9 afferents, ~3 active): total ~0.55*theta at init, so
#     an undifferentiated competitor integrates a 3-active volley over several steps
#     and sparsifies toward the ~3 active pixels as it wins.
#
# L1E_new is a local COINCIDENCE detector with nine accumulating afferents:
#   index 0    -- one paired local sensory afferent from L1E_s[i]
#   index 1..8 -- dense feedback from L2E[0..7]
# It should fire only when its paired L1E_s[i] AND an L2E winner spike coincide.
# With the shared cap at theta/2, exact two-input coincidence is calibrated:
# a mature paired-sensory weight (~500) plus a mature winning-L2E weight (~500)
# sums to theta on a coincident step, while either branch alone (~500) stays
# below theta. Init keeps the total below theta so p = theta - sum(acc_weights)
# starts positive (Hebbian): a strong-but-subthreshold paired sensory seed and a
# small nonzero seed on each feedback afferent.
SENSORY_WEIGHT = E_THRESHOLD / 3.0                 # frozen, subthreshold; ~3 steps/source spike
FF_INIT_TOTAL_FRAC = 0.55                          # L2E ff sum(acc_weights) at init, / theta
FF_INIT_MEAN = FF_INIT_TOTAL_FRAC * E_THRESHOLD / N_PIX   # ~61
ENEW_SENSORY_INIT_FRAC = 0.49                      # paired local sensory afferent, / theta -> 490
ENEW_FB_INIT_FRAC = 0.06                           # each L2E feedback afferent, / theta -> 60
INIT_JITTER_FRAC = 0.04                            # deterministic narrow seeded jitter
DISTANCE_POWER = 2.0                               # fixed exponent for the learning-rate factor

FREQ_WINDOW = 40
LOG_MAX = 400

# The shared default excitatory weight cap is theta/2 = 500, which calibrates the
# two-input L1E_new coincidence (500 + 500 = theta). Zero leak is NOT a valid
# default for this circuit (two lone-branch events could accumulate and falsely
# fire L1E_new); the default leak below is the value selected by
# experiments/frequency_experiment.py. e_threshold is fixed at 1000 (inspectability)
# and is NOT part of the editable allowlist.
LEAK_DEFAULT = 0.03
DEFAULTS = dict(
    seed=1,
    e_threshold=E_THRESHOLD,
    e_weight_cap=E_THRESHOLD / 2.0,                # 500 (shared)
    eta=0.01,
    leak_rate=LEAK_DEFAULT,
    refractory_steps=0,
    input_period=1,
)
# Keys a browser config-apply is allowed to change. Everything else is derived or
# structural and is rejected.
EDITABLE_KEYS = {'eta', 'leak_rate', 'refractory_steps', 'e_weight_cap', 'input_period'}


class SimulationEngine:
    """Owns the network and advances it one deterministic timestep at a time."""

    def __init__(self, seed: int = 1, **overrides):
        params = dict(DEFAULTS)
        params['seed'] = int(seed)
        for k, v in overrides.items():
            if k not in DEFAULTS:
                raise KeyError(f'unknown config key: {k!r}')
            params[k] = v
        self.params = params
        self._build()

    # ================================================================ build
    def _build(self):
        p = self.params
        rng = np.random.default_rng(p['seed'])

        self.pos = generate_layout(rng, N_PIX, N_OUT)
        thr = float(p['e_threshold'])
        cap = float(p['e_weight_cap'])
        leak = float(p['leak_rate'])
        refr = int(p['refractory_steps'])
        eta = float(p['eta'])

        def jitter(mean, size):
            j = rng.uniform(1.0 - INIT_JITTER_FRAC, 1.0 + INIT_JITTER_FRAC, size=size)
            return np.clip(mean * j, 0.0, cap)

        # Distance-factor rows (learning-rate multiplier), normalized per projection
        # so the closest synapse has factor 1.
        ff_factor = self._distance_factors(
            [f'L1E{i}' for i in range(N_PIX)], [f'L2E{j}' for j in range(N_OUT)])
        fb_factor = self._distance_factors(
            [f'L2E{j}' for j in range(N_OUT)], [f'L1Enew{i}' for i in range(N_PIX)])
        # Paired local sensory projection L1E_s[i] -> L1E_new[i], one afferent each,
        # normalized across the nine pairs.
        local_factor = self._distance_factors(
            [f'L1E{i}' for i in range(N_PIX)], [f'L1Enew{i}' for i in range(N_PIX)])

        # ---- populations ----
        self.l1e_s = [
            ExcitatoryNeuron(
                f'L1E{i}', 'source',
                acc_weights=np.array([SENSORY_WEIGHT]), acc_distance_factor=np.array([1.0]),
                threshold=thr, w_max=cap, leak_rate=leak, refractory_steps=refr,
                eta=eta, learn=False, subt_magnitude=thr)     # wiped by paired L1I (delayed)
            for i in range(N_PIX)]

        # L1E_new afferent layout: [0] = paired local sensory, [1..8] = L2E feedback.
        self.l1e_new = [
            ExcitatoryNeuron(
                f'L1Enew{i}', 'supervisor',
                acc_weights=self._enew_init(rng, cap),
                acc_distance_factor=np.concatenate(([local_factor[i][i]], fb_factor[i])),
                threshold=thr, w_max=cap, leak_rate=leak, refractory_steps=refr,
                eta=eta, learn=True, subt_magnitude=0.0)      # not inhibited
            for i in range(N_PIX)]

        self.l2e = [
            ExcitatoryNeuron(
                f'L2E{j}', 'competitor',
                acc_weights=jitter(FF_INIT_MEAN, N_PIX), acc_distance_factor=ff_factor[j],
                threshold=thr, w_max=cap, leak_rate=leak, refractory_steps=refr,
                eta=eta, learn=True, subt_magnitude=thr)      # wiped by L2I
            for j in range(N_OUT)]

        self.l1i = [InhibitoryNeuron(f'L1I{i}', 'relay', threshold=thr / 3.0) for i in range(N_PIX)]
        self.l2i = InhibitoryNeuron('L2I', 'relay', threshold=thr / 3.0)

        self.exc = {n.id: n for n in (*self.l1e_s, *self.l1e_new, *self.l2e)}
        self.inh = {n.id: n for n in (*self.l1i, self.l2i)}
        self.neurons = {**self.exc, **self.inh}
        self.order = ([n.id for n in self.l1e_s] + [n.id for n in self.l1e_new]
                      + [n.id for n in self.l1i] + [n.id for n in self.l2e] + ['L2I'])

        self._build_meta_and_edges(thr)

        # ---- runtime state ----
        self.timestep = 0
        self.winner = None
        first = next(iter(PATTERNS))
        self.current_pattern = first
        self.input_vec = np.array(PATTERNS[first], dtype=float)
        self.spiked = {nid: False for nid in self.neurons}
        self._spike_hist = {nid: deque(maxlen=FREQ_WINDOW) for nid in self.neurons}
        self.changed_synapses = []
        self.emitted = []
        self.applied_inhibition = []
        self.event_log = deque(maxlen=LOG_MAX)
        self._log_seq = 0
        self._pending_charge = {}       # manual single-pulse charge injections
        self._continuous = {}           # manual continuous charge injections
        self._force_relay = set()       # manual firing of inhibitory relays
        # Queued L1I->L1E_s inhibition: set when L1E_new[i] fires, delivered on the
        # NEXT timestep (the one-step delay lives in the I->E synapse, not the relay).
        self._pending_inh = np.zeros(N_PIX, dtype=bool)

    def _distance_factors(self, sources, targets):
        ds = np.array([[np.linalg.norm(self.pos[s] - self.pos[t]) for s in sources]
                       for t in targets])
        positive = ds[ds > 0]
        d_ref = float(positive.min()) if positive.size else 1.0
        return (d_ref / np.maximum(ds, d_ref)) ** DISTANCE_POWER

    def _enew_init(self, rng, cap):
        """Deterministic seeded init for one L1E_new: a strong-but-subthreshold paired
        sensory seed at index 0 and a small nonzero seed on each of the eight L2E
        feedback afferents. The sub-threshold-total invariant (sum < theta) is
        enforced after jitter so p = theta - sum(acc_weights) starts positive."""
        thr = float(self.params['e_threshold'])
        w = np.empty(1 + N_OUT)
        w[0] = ENEW_SENSORY_INIT_FRAC * thr
        w[1:] = ENEW_FB_INIT_FRAC * thr
        w *= rng.uniform(1.0 - INIT_JITTER_FRAC, 1.0 + INIT_JITTER_FRAC, size=w.shape)
        np.clip(w, 0.0, 0.98 * cap, out=w)     # paired sensory stays strictly below the cap
        total = w.sum()
        if total >= thr:                       # enforce sub-threshold total invariant
            w *= 0.97 * thr / total
        return w

    def _build_meta_and_edges(self, thr):
        i_thr = thr / 3.0
        meta = {}

        def add(nid, label, layer, ntype, role, threshold):
            meta[nid] = dict(id=nid, label=label, layer=layer, type=ntype, role=role,
                             threshold=round(float(threshold), 4),
                             pos=[round(float(x), 4) for x in self.pos[nid]])

        for i in range(N_PIX):
            add(f'L1E{i}', f'L1E_s{i}', 'L1', 'E', 'source', thr)
            add(f'L1Enew{i}', f'L1E_new{i}', 'L1', 'E', 'supervisor', thr)
            add(f'L1I{i}', f'L1I{i}', 'L1', 'I', 'relay', i_thr)
        for j in range(N_OUT):
            add(f'L2E{j}', f'L2E{j}', 'L2', 'E', 'competitor', thr)
        add('L2I', 'L2I', 'L2', 'I', 'relay', i_thr)
        self.meta = meta

        # Static edge list. Weights are read live from neuron arrays at serialize
        # time; the ``weight`` here is only a construction seed / kind tag.
        edges = []

        def edge(eid, src, tgt, kind, weight=None, sign=None):
            e = dict(id=eid, source=src, target=tgt, kind=kind)
            if sign is not None:
                e['sign'] = sign
            edges.append(e)

        for j in range(N_OUT):
            for i in range(N_PIX):
                edge(f'ff{i}->{j}', f'L1E{i}', f'L2E{j}', 'feedforward')
        for i in range(N_PIX):
            for j in range(N_OUT):
                edge(f'fb{j}->{i}', f'L2E{j}', f'L1Enew{i}', 'feedback')
        for i in range(N_PIX):
            # Paired local sensory afferent: L1E_s[i] -> L1E_new[i] (coincidence input).
            edge(f'cl{i}', f'L1E{i}', f'L1Enew{i}', 'coincidence_local')
        for i in range(N_PIX):
            edge(f're_l1_{i}', f'L1Enew{i}', f'L1I{i}', 'relay_excitation')
        for j in range(N_OUT):
            edge(f're_l2_{j}', f'L2E{j}', 'L2I', 'relay_excitation')
        for i in range(N_PIX):
            edge(f'inh_l1_{i}', f'L1I{i}', f'L1E{i}', 'inhibition', sign=SUBTRACTIVE_SIGN)
        for j in range(N_OUT):
            edge(f'inh_l2_{j}', 'L2I', f'L2E{j}', 'inhibition', sign=SUBTRACTIVE_SIGN)
        self.synapses = edges

    # ============================================================== stepping
    def _begin_step(self):
        for nid in self.neurons:
            self.spiked[nid] = False
        for n in self.exc.values():
            n.spiked = False
        for n in self.inh.values():
            n.clear()
        self.changed_synapses = []
        self.emitted = []
        self.applied_inhibition = []

    def _mark_spike(self, nid):
        self.spiked[nid] = True

    def _record_wipe(self, nid, removed):
        if removed <= 0:
            return
        self.applied_inhibition.append(dict(target=nid, v_pre=round(float(removed), 4),
                                            charge_removed=round(float(removed), 4),
                                            reached_rest=True))

    def _emit_weight_changes(self, neuron, edge_of):
        for k, w in enumerate(neuron.acc_weights):
            self.changed_synapses.append(dict(id=edge_of(k), weight=round(float(w), 4)))

    def _emit_enew_weight_changes(self, i, neuron):
        # L1E_new afferents: [0] -> cl{i} (paired sensory), [1+j] -> fb{j}->{i}.
        w = neuron.acc_weights
        self.changed_synapses.append(dict(id=f'cl{i}', weight=round(float(w[0]), 4)))
        for j in range(N_OUT):
            self.changed_synapses.append(dict(id=f'fb{j}->{i}', weight=round(float(w[1 + j]), 4)))

    def step(self) -> dict:
        self.timestep += 1
        t = self.timestep
        self._begin_step()

        input_arrives = (t % max(1, int(self.params['input_period'])) == 0)

        # 1. deposit external sensory charge (+ manual injections) into L1E_s.
        for i, n in enumerate(self.l1e_s):
            if input_arrives and self.input_vec[i] > 0.5:
                n.receive_acc(n.acc_weights[0])
        self._apply_manual_charge()

        # 2. deliver queued inhibition from the previous timestep to L1E_s -- AFTER
        #    the new sensory deposit and BEFORE the L1E_s threshold check, so it
        #    removes real accumulating charge and lowers the source cadence.
        for i in np.nonzero(self._pending_inh)[0]:
            removed = self.l1e_s[i].hard_wipe()
            self.emitted.append(f'inh_l1_{i}')
            self._record_wipe(self.l1e_s[i].id, removed)
        self._pending_inh = np.zeros(N_PIX, dtype=bool)

        # 3. resolve L1E_s crossings (frozen sensory -> no weight update).
        l1s_spikes = np.zeros(N_PIX, dtype=bool)
        for i, n in enumerate(self.l1e_s):
            if n.can_fire():
                n.fire()
                l1s_spikes[i] = True
                self._mark_spike(n.id)

        # 4. deliver each L1E_s spike densely to all L2E AND locally to its paired
        #    L1E_new[i] (afferent index 0). Geometry never scales delivered charge.
        active_pix = np.nonzero(l1s_spikes)[0]
        if active_pix.size:
            for j, n in enumerate(self.l2e):
                charge = float((n.acc_weights * l1s_spikes).sum())
                if charge:
                    n.receive_acc(charge)
                    for i in active_pix:
                        self.emitted.append(f'ff{i}->{j}')
            for i in active_pix:
                self.l1e_new[i].receive_acc(self.l1e_new[i].acc_weights[0])
                self.emitted.append(f'cl{i}')

        # 5. deterministic L2 competition.
        winner_j = None
        crossers = [j for j, n in enumerate(self.l2e) if n.can_fire()]
        if crossers:
            winner_j = max(crossers, key=lambda j: (self.l2e[j].V, -j))
            wn = self.l2e[winner_j]
            wn.fire()
            wn.update_acc_weights(l1s_spikes)
            self._emit_weight_changes(wn, lambda i, j=winner_j: f'ff{i}->{j}')
            self._mark_spike(wn.id)
            self.winner = wn.id
            # structural relay -> L2I fires immediately -> hard-wipe every L2E
            self.l2i.receive()
            if self.l2i.resolve():
                self._mark_spike('L2I')
                self.emitted.append(f're_l2_{winner_j}')
                for j, n in enumerate(self.l2e):
                    removed = n.hard_wipe()
                    self.emitted.append(f'inh_l2_{j}')
                    self._record_wipe(n.id, removed)

        # 6. deliver the winning L2E spike through dense feedback to all L1E_new
        #    (afferent index 1 + winner_j).
        if winner_j is not None:
            for i, n in enumerate(self.l1e_new):
                n.receive_acc(n.acc_weights[1 + winner_j])
                self.emitted.append(f'fb{winner_j}->{i}')

        # 7-9. resolve L1E_new coincidence crossings; each fire learns from its nine
        #      real afferents ([0]=paired sensory, [1..8]=L2E feedback) and triggers
        #      its paired L1I, which fires now but QUEUES its wipe for next step.
        l2_part = np.zeros(N_OUT, dtype=bool)
        if winner_j is not None:
            l2_part[winner_j] = True
        for i, n in enumerate(self.l1e_new):
            if n.can_fire():
                n.fire()
                part = np.concatenate(([bool(l1s_spikes[i])], l2_part))
                n.update_acc_weights(part)
                self._emit_enew_weight_changes(i, n)
                self._mark_spike(n.id)
                relay = self.l1i[i]
                relay.receive()
                if relay.resolve():
                    self._mark_spike(relay.id)
                    self.emitted.append(f're_l1_{i}')
                    self._pending_inh[i] = True          # delayed I->E: delivered next step

        self._apply_manual_relays()

        # 10. leak + refractory countdown, once, on excitatory neurons.
        for n in self.exc.values():
            n.advance()

        # 11. history + frequency
        for nid in self.neurons:
            self._spike_hist[nid].append(1 if self.spiked[nid] else 0)

        return self.dynamic_state()

    # -------------------------------------------------- manual firing helpers
    def _apply_manual_charge(self):
        for nid, mag in list(self._pending_charge.items()):
            n = self.exc.get(nid)
            if n is not None:
                n.receive_acc(mag)
        self._pending_charge = {}
        for nid, mag in self._continuous.items():
            n = self.exc.get(nid)
            if n is not None:
                n.receive_acc(mag)

    def _apply_manual_relays(self):
        # Debug-only: force an inhibitory relay to fire and apply its wipe.
        for nid in list(self._force_relay):
            if nid == 'L2I':
                for j, n in enumerate(self.l2e):
                    self._record_wipe(n.id, n.hard_wipe())
                    self.emitted.append(f'inh_l2_{j}')
                self._mark_spike('L2I')
            elif nid.startswith('L1I'):
                i = int(nid[3:])
                self._record_wipe(self.l1e_s[i].id, self.l1e_s[i].hard_wipe())
                self.emitted.append(f'inh_l1_{i}')
                self._mark_spike(nid)
        self._force_relay = set()

    def firing_freq(self, nid):
        h = self._spike_hist[nid]
        return sum(h) / len(h) if h else 0.0

    # ================================================================ control
    def set_pattern(self, name: str):
        if name not in PATTERNS:
            raise KeyError(name)
        self.current_pattern = name
        self.input_vec = np.array(PATTERNS[name], dtype=float)

    def set_input(self, vec):
        self.input_vec = np.array(vec, dtype=float).reshape(N_PIX)

    def toggle_pixel(self, i: int):
        self.input_vec[i] = 0.0 if self.input_vec[i] > 0.5 else 1.0

    def clear_input(self):
        self.input_vec = np.zeros(N_PIX)

    def random_pattern(self):
        self.input_vec = (np.random.default_rng().random(N_PIX) > 0.5).astype(float)

    def inject_noise(self, prob: float = 0.15):
        flip = np.random.default_rng().random(N_PIX) < prob
        self.input_vec = np.where(flip, 1.0 - self.input_vec, self.input_vec)

    def stimulate(self, neuron_id: str, magnitude: float = 1.0, continuous: bool = False):
        if neuron_id not in self.neurons:
            raise KeyError(neuron_id)
        if neuron_id in self.inh:
            if magnitude > 0:
                self._force_relay.add(neuron_id)
            return
        charge = float(magnitude) * self.params['e_threshold']
        if continuous:
            if magnitude <= 0:
                self._continuous.pop(neuron_id, None)
            else:
                self._continuous[neuron_id] = charge
        else:
            self._pending_charge[neuron_id] = charge

    def set_feedforward_weight(self, j: int, i: int, weight: float) -> float:
        if not (0 <= j < N_OUT and 0 <= i < N_PIX):
            raise IndexError(f'feedforward index out of range: j={j}, i={i}')
        w = float(np.clip(weight, 0.0, self.params['e_weight_cap']))
        self.l2e[j].acc_weights[i] = w
        return w

    def apply_config(self, overrides: dict):
        """Merge editable overrides and rebuild. Unknown/legacy keys are rejected."""
        applied = []
        for k, v in (overrides or {}).items():
            if k not in EDITABLE_KEYS:
                continue                       # reject deleted/legacy keys silently
            self.params[k] = v
            applied.append(k)
        if applied:
            self._build()
            self._log('config', f'applied {applied}; network rebuilt')
        return applied

    def reset(self):
        """Rebuild the same network (same seed) -> wipes learned weights."""
        self._build()
        self._log('control', 'reset: network rebuilt from seed')

    def reseed(self):
        """New random seed -> fresh initial weights under the same config."""
        self.params['seed'] = int(np.random.SeedSequence().generate_state(1)[0])
        self._build()
        self._log('control', f'reseed: seed={self.params["seed"]}')
        return self.params['seed']

    def _log(self, kind: str, message: str):
        self._log_seq += 1
        self.event_log.append(dict(seq=self._log_seq, t=self.timestep, kind=kind, message=message))

    # ============================================================ serialize
    def _live_weight(self, edge):
        eid, kind = edge['id'], edge['kind']
        if kind == 'feedforward':
            i, j = eid[2:].split('->')
            return float(self.l2e[int(j)].acc_weights[int(i)])
        if kind == 'feedback':
            j, i = eid[2:].split('->')
            return float(self.l1e_new[int(i)].acc_weights[1 + int(j)])   # [0] is local sensory
        if kind == 'coincidence_local':
            i = int(eid[2:])
            return float(self.l1e_new[i].acc_weights[0])
        if kind == 'inhibition':
            return float(self.params['e_threshold'])       # frozen subtractive magnitude
        return None                                          # structural relay: no weight

    def topology(self) -> dict:
        neurons = [dict(**self.meta[nid]) for nid in self.order]
        synapses = []
        for e in self.synapses:
            w = self._live_weight(e)
            synapses.append(dict(**e, weight=(None if w is None else round(w, 4))))
        return dict(neurons=neurons, synapses=synapses, layers=['L1', 'L2'],
                    patterns=list(PATTERNS.keys()),
                    pattern_vectors={k: list(map(int, v)) for k, v in PATTERNS.items()},
                    grid=dict(rows=3, cols=3), params=self._public_params())

    def _public_params(self):
        p = self.params
        thr = float(p['e_threshold'])
        return dict(seed=p['seed'], e_threshold=thr, e_weight_cap=float(p['e_weight_cap']),
                    eta=float(p['eta']), leak_rate=float(p['leak_rate']),
                    refractory_steps=int(p['refractory_steps']),
                    input_period=int(p['input_period']),
                    i_threshold=round(thr / 3.0, 4),
                    # Fields the receptive-field / weights charts read to scale by cap.
                    threshold_l2=thr,
                    l2e_weight_cap_frac=float(p['e_weight_cap']) / thr if thr else 1.0)

    def dynamic_state(self) -> dict:
        neurons = []
        for nid in self.order:
            n = self.neurons[nid]
            thr = self.meta[nid]['threshold'] or 1.0
            pot = float(n.potential)
            neurons.append(dict(
                id=nid, potential=round(pot, 4), activation=round(pot / thr, 4),
                spiked=bool(self.spiked[nid]), freq=round(self.firing_freq(nid), 4),
                refractory=int(n.refractory_timer),
                assembly=(self.winner if nid == self.winner else None)))
        return dict(timestep=self.timestep, running=False, neurons=neurons,
                    changed_synapses=self.changed_synapses,
                    emitted=self.emitted,
                    applied_inhibition=self.applied_inhibition,
                    input=self.input_vec.astype(int).tolist(),
                    winner=self.winner,
                    stats=self.stats(), log=list(self.event_log)[-12:])

    def stats(self) -> dict:
        active = sum(1 for nid in self.neurons if abs(self.neurons[nid].potential) > 1e-3)
        firing = sum(1 for v in self.spiked.values() if v)
        rate = float(np.mean([self.firing_freq(nid) for nid in self.neurons]))
        return dict(total=len(self.neurons), active=active, firing=firing,
                    firing_rate=round(rate, 4), winner=self.winner)
