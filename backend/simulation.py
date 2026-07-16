"""Conductance-based predictive-inhibition SNN: construction, a synchronous
double-buffered timestep with explicit integer synaptic delays, and state
snapshots for the dashboard.

Two topologies share one synchronous engine, selected by ``enew_enabled``:

``enew_enabled=False`` -- the predictive-inhibition (PI) experiment (default here
for the scientific question)::

        external -> L1E_s[i] ==ff==> L2E[j] --relay--> PI[j]
                        ^                 |
                        |     predictive inhibitory conductance (locally plastic)
                        +-----------------+
                                          L2E[j] --relay--> L2I_WTA --> all L2E (conductance)

    9 L1E_s, 8 L2E, 8 PI (paired 1:1 with L2E), 1 L2I_WTA  = 26 neurons.
    Each PI[j] owns 9 candidate inhibitory output synapses onto L1E_s (72 total).

``enew_enabled=True`` -- the retained L1E_new coincidence comparison topology
(36 neurons). Its inhibition is ALSO conductance now (no hard wipes anywhere).

Timestep semantics (see ``step``): every internal excitatory projection has an
integer delay of 1; relays (L2I_WTA, PI, L1I) fire in the same boundary as their
source spike and schedule their inhibitory *conductance* output for the next
boundary. External input arrives at the current boundary. All targets integrate
once per boundary from double-buffered arrivals, so no behaviour depends on Python
neuron-iteration order.
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
    PredictiveInterneuron,
    E_THRESHOLD,
    I_THRESHOLD,
    E_WEIGHT_CAP,
    INHIBITORY_SIGN,
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

# --- Excitatory initialization (unchanged learning rule) --------------------
SENSORY_WEIGHT = E_THRESHOLD / 3.0                 # frozen sensory afferent weight
FF_INIT_TOTAL_FRAC = 0.55                          # L2E ff sum(acc_weights) at init, / theta
FF_INIT_MEAN = FF_INIT_TOTAL_FRAC * E_THRESHOLD / N_PIX   # ~61
ENEW_SENSORY_INIT_FRAC = 0.49                      # paired local sensory afferent, / theta
ENEW_FB_INIT_FRAC = 0.06                           # each L2E feedback afferent, / theta
INIT_JITTER_FRAC = 0.04                            # deterministic narrow seeded jitter
DISTANCE_POWER = 2.0                               # fixed exponent for the learning-rate factor

FREQ_WINDOW = 40
LOG_MAX = 400
SYNAPTIC_DELAY = 1                                 # integer delay on every internal projection

LEAK_DEFAULT = 0.03
DEFAULTS = dict(
    seed=1,
    e_threshold=E_THRESHOLD,
    e_weight_cap=E_THRESHOLD / 2.0,                # 500 (shared accumulating cap)
    eta=0.01,                                      # excitatory accumulating learning rate
    leak_rate=LEAK_DEFAULT,                        # -> baseline leak conductance g_L
    refractory_steps=0,
    input_period=1,
    enew_enabled=True,                             # topology switch (comparison flag)

    # --- conductance / trace (membrane) ---
    # Inhibitory-conductance retention is deliberately SPLIT by target population so
    # the two roles have independent timescales (a confound otherwise: a persistent
    # WTA pulse alone can cause turnover with no predictive inhibition at all):
    #   alpha_inh    -- decay on L2E (the L2I_WTA target). Kept FAST so WTA is a clean
    #                   single-winner suppressor that does not itself drive turnover.
    #   alpha_inh_l1 -- decay on L1E_s / L1E_new (the predictive-PI / legacy-L1I
    #                   target). This is the symmetry-breaking lever: the overlap
    #                   experiment shows the shared-feature shunt must persist across
    #                   the rival's accumulation window. 0.95 gives robust (8/8-seed)
    #                   PI-driven turnover with recovery; the fast regime never breaks.
    alpha_inh=0.6,                                 # inhibitory-conductance retention on L2E (WTA)
    alpha_inh_l1=0.95,                             # retention on L1E_s / L1E_new (predictive)
    alpha_a=0.85,                                  # activity-trace retention/step
    beta_v=0.30,                                   # trace gain on sub-threshold depolarization
    beta_s=1.00,                                   # trace gain on a spike
    a_max=1.0,                                     # activity-trace bound
    e_inh=0.0,                                     # inhibitory reversal (<= V_rest); 0 = shunting

    # --- predictive inhibition (PI) ---
    pi_eta=0.02,                                   # SLOW local inhibitory association rate
    pi_w_max=1.0,                                  # per-candidate-synapse weight cap
    pi_lt_decay=0.001,                             # slow passive weight decay (recovery)
    pi_g_scale=6.0,                                # conductance per unit PI weight (fast expression)
    pi_conductance_enabled=True,                   # express PI conductance onto L1E_s
    pi_plasticity_enabled=True,                    # learn PI output synapses

    # --- L2 winner-take-all inhibition ---
    l2i_g_scale=12.0,                              # global inhibitory conductance pulse magnitude
)
# Keys a browser/experiment config-apply may change. Everything else is derived.
EDITABLE_KEYS = {
    'eta', 'leak_rate', 'refractory_steps', 'e_weight_cap', 'input_period',
    'enew_enabled', 'alpha_inh', 'alpha_inh_l1', 'alpha_a', 'beta_v', 'beta_s',
    'a_max', 'e_inh', 'pi_eta', 'pi_w_max', 'pi_lt_decay', 'pi_g_scale',
    'pi_conductance_enabled', 'pi_plasticity_enabled', 'l2i_g_scale',
}


class SimulationEngine:
    """Owns the network and advances it one synchronous timestep at a time."""

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
    def _mkE(self, nid, role, acc_weights, acc_distance_factor, *, learn, alpha_inh):
        p = self.params
        return ExcitatoryNeuron(
            nid, role, acc_weights=acc_weights, acc_distance_factor=acc_distance_factor,
            threshold=float(p['e_threshold']), w_max=float(p['e_weight_cap']),
            leak_rate=float(p['leak_rate']), refractory_steps=int(p['refractory_steps']),
            eta=float(p['eta']), learn=learn,
            e_inh=float(p['e_inh']), alpha_inh=float(alpha_inh),
            alpha_a=float(p['alpha_a']), beta_v=float(p['beta_v']),
            beta_s=float(p['beta_s']), a_max=float(p['a_max']))

    def _build(self):
        p = self.params
        rng = np.random.default_rng(p['seed'])

        self.pos = generate_layout(rng, N_PIX, N_OUT)
        thr = float(p['e_threshold'])
        cap = float(p['e_weight_cap'])
        self.enew_enabled = bool(p['enew_enabled'])

        def jitter(mean, size):
            j = rng.uniform(1.0 - INIT_JITTER_FRAC, 1.0 + INIT_JITTER_FRAC, size=size)
            return np.clip(mean * j, 0.0, cap)

        ff_factor = self._distance_factors(
            [f'L1E{i}' for i in range(N_PIX)], [f'L2E{j}' for j in range(N_OUT)])
        alpha_l1 = float(p['alpha_inh_l1'])            # slow, predictive-target decay
        alpha_l2 = float(p['alpha_inh'])               # fast, WTA-target decay

        # ---- shared excitatory populations ----
        self.l1e_s = [
            self._mkE(f'L1E{i}', 'source', np.array([SENSORY_WEIGHT]), np.array([1.0]),
                      learn=False, alpha_inh=alpha_l1)
            for i in range(N_PIX)]
        self.l2e = [
            self._mkE(f'L2E{j}', 'competitor', jitter(FF_INIT_MEAN, N_PIX), ff_factor[j],
                      learn=True, alpha_inh=alpha_l2)
            for j in range(N_OUT)]
        self.l2i = InhibitoryNeuron('L2I', 'relay', threshold=thr / 3.0)

        if self.enew_enabled:
            # Retained L1E_new coincidence comparison topology.
            fb_factor = self._distance_factors(
                [f'L2E{j}' for j in range(N_OUT)], [f'L1Enew{i}' for i in range(N_PIX)])
            local_factor = self._distance_factors(
                [f'L1E{i}' for i in range(N_PIX)], [f'L1Enew{i}' for i in range(N_PIX)])
            self.l1e_new = [
                self._mkE(f'L1Enew{i}', 'supervisor', self._enew_init(rng, cap),
                          np.concatenate(([local_factor[i][i]], fb_factor[i])), learn=True,
                          alpha_inh=alpha_l1)
                for i in range(N_PIX)]
            self.l1i = [InhibitoryNeuron(f'L1I{i}', 'relay', threshold=thr / 3.0)
                        for i in range(N_PIX)]
            self.pi = []
        else:
            # Predictive-inhibition experiment topology: one PI per L2E.
            self.l1e_new = []
            self.l1i = []
            self.pi = [
                PredictiveInterneuron(
                    f'PI{j}', 'predictor', N_PIX,
                    w_init=0.0, w_max=float(p['pi_w_max']), eta=float(p['pi_eta']),
                    lt_decay=float(p['pi_lt_decay']), g_scale=float(p['pi_g_scale']),
                    threshold=thr / 3.0)
                for j in range(N_OUT)]

        self.exc = {n.id: n for n in (*self.l1e_s, *self.l1e_new, *self.l2e)}
        relays = (*self.l1i, self.l2i, *self.pi)
        self.inh = {n.id: n for n in relays}
        self.neurons = {**self.exc, **self.inh}
        self.order = (
            [n.id for n in self.l1e_s]
            + [n.id for n in self.l1e_new]
            + [n.id for n in self.l1i]
            + [n.id for n in self.pi]
            + [n.id for n in self.l2e]
            + ['L2I'])

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
        self.inhibitory_pulses = []
        self.event_log = deque(maxlen=LOG_MAX)
        self._log_seq = 0
        self._continuous = {}

        # ---- delivery double-buffers (arrivals scheduled for the NEXT boundary) ----
        self._exc_next = {}            # nid -> excitatory charge for next boundary
        self._inh_next = []            # list of pulse dicts for next boundary
        self._ff_part_next = np.zeros(N_PIX, dtype=bool)   # which L1E_s fired -> L2E next
        self._winner_next = None       # winner index this boundary -> feedback/relay next
        # arrivals landing at the CURRENT boundary (filled from *_next at step start)
        self._ff_part_now = np.zeros(N_PIX, dtype=bool)
        self._winner_now = None

    def _distance_factors(self, sources, targets):
        ds = np.array([[np.linalg.norm(self.pos[s] - self.pos[t]) for s in sources]
                       for t in targets])
        positive = ds[ds > 0]
        d_ref = float(positive.min()) if positive.size else 1.0
        return (d_ref / np.maximum(ds, d_ref)) ** DISTANCE_POWER

    def _enew_init(self, rng, cap):
        thr = float(self.params['e_threshold'])
        w = np.empty(1 + N_OUT)
        w[0] = ENEW_SENSORY_INIT_FRAC * thr
        w[1:] = ENEW_FB_INIT_FRAC * thr
        w *= rng.uniform(1.0 - INIT_JITTER_FRAC, 1.0 + INIT_JITTER_FRAC, size=w.shape)
        np.clip(w, 0.0, 0.98 * cap, out=w)
        total = w.sum()
        if total >= thr:
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
            if self.enew_enabled:
                add(f'L1Enew{i}', f'L1E_new{i}', 'L1', 'E', 'supervisor', thr)
                add(f'L1I{i}', f'L1I{i}', 'L1', 'I', 'relay', i_thr)
        for j in range(N_OUT):
            add(f'L2E{j}', f'L2E{j}', 'L2', 'E', 'competitor', thr)
            if not self.enew_enabled:
                add(f'PI{j}', f'PI{j}', 'L2', 'I', 'predictor', i_thr)
        add('L2I', 'L2I', 'L2', 'I', 'relay', i_thr)
        self.meta = meta

        edges = []

        def edge(eid, src, tgt, kind, sign=None):
            e = dict(id=eid, source=src, target=tgt, kind=kind)
            if sign is not None:
                e['sign'] = sign
            edges.append(e)

        for j in range(N_OUT):
            for i in range(N_PIX):
                edge(f'ff{i}->{j}', f'L1E{i}', f'L2E{j}', 'feedforward')
        for j in range(N_OUT):
            edge(f're_l2_{j}', f'L2E{j}', 'L2I', 'relay_excitation')
        for j in range(N_OUT):
            edge(f'inh_l2_{j}', 'L2I', f'L2E{j}', 'inhibition', sign=INHIBITORY_SIGN)

        if self.enew_enabled:
            for i in range(N_PIX):
                for j in range(N_OUT):
                    edge(f'fb{j}->{i}', f'L2E{j}', f'L1Enew{i}', 'feedback')
            for i in range(N_PIX):
                edge(f'cl{i}', f'L1E{i}', f'L1Enew{i}', 'coincidence_local')
            for i in range(N_PIX):
                edge(f're_l1_{i}', f'L1Enew{i}', f'L1I{i}', 'relay_excitation')
            for i in range(N_PIX):
                edge(f'inh_l1_{i}', f'L1I{i}', f'L1E{i}', 'inhibition', sign=INHIBITORY_SIGN)
        else:
            for j in range(N_OUT):
                edge(f're_pi_{j}', f'L2E{j}', f'PI{j}', 'relay_excitation')   # paired 1:1
            for j in range(N_OUT):
                for i in range(N_PIX):
                    edge(f'pi{j}->{i}', f'PI{j}', f'L1E{i}', 'predictive_inhibition',
                         sign=INHIBITORY_SIGN)
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
        self.inhibitory_pulses = []

    def _record_pulse(self, target, dg, source, kind, weight, g_before, g_after):
        self.inhibitory_pulses.append(dict(
            source=source, target=target, kind=kind,
            synaptic_weight=round(float(weight), 6),
            conductance_increment=round(float(dg), 6),
            g_inh_before=round(float(g_before), 6),
            g_inh_after=round(float(g_after), 6),
            boundary=int(self.timestep)))

    def _deliver_inhibition(self, pulse):
        n = self.exc.get(pulse['target'])
        if n is None:
            return
        g_before = n.g_inh
        n.add_inhibition(pulse['dg'])
        self._record_pulse(pulse['target'], pulse['dg'], pulse['source'],
                           pulse['kind'], pulse['weight'], g_before, n.g_inh)

    def step(self) -> dict:
        self.timestep += 1
        t = self.timestep
        self._begin_step()
        p = self.params

        # ---- subphase 1: gather scheduled arrivals (emitted at t-1, delay 1) ----
        exc_now = self._exc_next
        inh_now = self._inh_next
        self._ff_part_now = self._ff_part_next
        self._winner_now = self._winner_next
        self._exc_next = {}
        self._inh_next = []
        self._ff_part_next = np.zeros(N_PIX, dtype=bool)
        self._winner_next = None

        # inhibitory conductance arrivals (persistent; added before integration)
        for pulse in inh_now:
            self._deliver_inhibition(pulse)

        # excitatory internal arrivals
        for nid, q in exc_now.items():
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(q)

        # external sensory input (delay 0) + manual continuous injections
        input_arrives = (t % max(1, int(p['input_period'])) == 0)
        for i, n in enumerate(self.l1e_s):
            if input_arrives and self.input_vec[i] > 0.5:
                n.gather_exc(n.acc_weights[0])
        for nid, mag in self._continuous.items():
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(mag)

        # ---- subphase 2: integrate every excitatory neuron once ----
        for n in self.exc.values():
            n.integrate()

        # ---- subphase 3: threshold test + fire ----
        # L1E_s: every crosser fires (sensory sources).
        l1s_spikes = np.zeros(N_PIX, dtype=bool)
        for i, n in enumerate(self.l1e_s):
            if n.can_fire():
                n.fire()
                l1s_spikes[i] = True
                self.spiked[n.id] = True

        # L1E_new (enew branch): every crosser fires + learns.
        enew_spikes = np.zeros(N_PIX, dtype=bool)
        if self.enew_enabled:
            l2_part = np.zeros(N_OUT, dtype=bool)
            if self._winner_now is not None:
                l2_part[self._winner_now] = True
            for i, n in enumerate(self.l1e_new):
                if n.can_fire():
                    n.fire()
                    enew_spikes[i] = True
                    self.spiked[n.id] = True
                    part = np.concatenate(([bool(self._ff_part_now[i])], l2_part))
                    n.update_acc_weights(part)
                    self._emit_enew_weight_changes(i, n)

        # L2E: deterministic single-winner WTA among the crossers (selection, NOT
        # charge removal). Only the winner fires and learns.
        winner_j = None
        crossers = [j for j, n in enumerate(self.l2e) if n.can_fire()]
        if crossers:
            winner_j = max(crossers, key=lambda j: (self.l2e[j].V, -j))
            wn = self.l2e[winner_j]
            wn.fire()
            wn.update_acc_weights(self._ff_part_now)     # causal volley arrived this boundary
            self._emit_weight_changes(wn, lambda i, j=winner_j: f'ff{i}->{j}')
            self.spiked[wn.id] = True
            self.winner = wn.id

        # ---- subphase 4: update local activity traces (survive reset) ----
        for n in self.exc.values():
            n.update_trace()

        # ---- subphase 5: emit spikes into delay-1 queues; run local relays/plasticity ----
        # L1E_s spikes -> L2E feedforward (delay 1); enew: paired local sensory too.
        if l1s_spikes.any():
            active_pix = np.nonzero(l1s_spikes)[0]
            for j, n in enumerate(self.l2e):
                q = float((n.acc_weights * l1s_spikes).sum())
                if q:
                    self._sched_exc(n.id, q)
                    for i in active_pix:
                        self.emitted.append(f'ff{i}->{j}')
            self._ff_part_next = l1s_spikes.copy()
            if self.enew_enabled:
                for i in active_pix:
                    self._sched_exc(self.l1e_new[i].id, float(self.l1e_new[i].acc_weights[0]))
                    self.emitted.append(f'cl{i}')

        # L2 winner -> L2I_WTA relay (same boundary); conductance onto all L2E (delay 1).
        if winner_j is not None:
            self.l2i.receive()
            if self.l2i.resolve():
                self.spiked['L2I'] = True
                self.emitted.append(f're_l2_{winner_j}')
                g = float(p['l2i_g_scale'])
                for j in range(N_OUT):
                    self._sched_inh(f'L2E{j}', g, 'L2I', 'wta', weight=g)
            self._winner_next = winner_j

            if self.enew_enabled:
                # dense feedback L2E[winner] -> L1E_new (delay 1)
                for i, n in enumerate(self.l1e_new):
                    self._sched_exc(n.id, float(n.acc_weights[1 + winner_j]))
                    self.emitted.append(f'fb{winner_j}->{i}')
            else:
                # paired PI[winner] relay (same boundary); learns locally now; its
                # conductance onto L1E_s is scheduled for the next boundary.
                self._fire_pi(winner_j)

        # enew: L1E_new spikes -> paired L1I relay (same boundary); conductance
        # onto paired L1E_s (delay 1).
        if self.enew_enabled and enew_spikes.any():
            for i in np.nonzero(enew_spikes)[0]:
                relay = self.l1i[i]
                relay.receive()
                if relay.resolve():
                    self.spiked[relay.id] = True
                    self.emitted.append(f're_l1_{i}')
                    g = float(p['l2i_g_scale'])           # legacy L1I uses the same fixed scale
                    self._sched_inh(f'L1E{i}', g, relay.id, 'legacy_l1i', weight=g)

        # ---- subphase 6: decay conductance once; subphase 7: refractory + PI decay ----
        for n in self.exc.values():
            n.decay_conductance()
            n.advance_refractory()
        for pi in self.pi:
            pi.passive_decay()

        # ---- subphase 8: history + serialize ----
        for nid in self.neurons:
            self._spike_hist[nid].append(1 if self.spiked[nid] else 0)
        return self.dynamic_state()

    # --------------------------------------------------------- emission helpers
    def _sched_exc(self, nid, q):
        self._exc_next[nid] = self._exc_next.get(nid, 0.0) + float(q)

    def _sched_inh(self, target, dg, source, kind, weight):
        self._inh_next.append(dict(target=target, dg=float(dg), source=source,
                                   kind=kind, weight=float(weight)))

    def _fire_pi(self, winner_j):
        """Fire the winner's paired PI relay: emit its (pre-update) inhibitory
        conductance onto every L1E_s for the next boundary, then update its output
        synapses STRICTLY LOCALLY from each target's own activity trace."""
        p = self.params
        pi = self.pi[winner_j]
        pi.receive()
        if not pi.resolve():
            return
        self.spiked[pi.id] = True
        # conductance from PRE-update weights, delivered next boundary
        if p['pi_conductance_enabled']:
            pulse = pi.conductance_pulse()               # g_scale * w (pre-update)
            for i in range(N_PIX):
                if pulse[i] > 0.0:
                    self._sched_inh(f'L1E{i}', pulse[i], pi.id, 'predictive',
                                    weight=float(pi.w[i]))
                    self.emitted.append(f'pi{winner_j}->{i}')
        # strictly-local plasticity: each output synapse reads only its own target's
        # trace and its own weight. Traces were finalized in subphase 4.
        if p['pi_plasticity_enabled']:
            traces = np.array([n.a for n in self.l1e_s])
            pi.learn(traces)
            self._emit_pi_weight_changes(winner_j, pi)

    def _emit_weight_changes(self, neuron, edge_of):
        for k, w in enumerate(neuron.acc_weights):
            self.changed_synapses.append(dict(id=edge_of(k), weight=round(float(w), 4)))

    def _emit_enew_weight_changes(self, i, neuron):
        w = neuron.acc_weights
        self.changed_synapses.append(dict(id=f'cl{i}', weight=round(float(w[0]), 4)))
        for j in range(N_OUT):
            self.changed_synapses.append(dict(id=f'fb{j}->{i}', weight=round(float(w[1 + j]), 4)))

    def _emit_pi_weight_changes(self, j, pi):
        for i in range(N_PIX):
            self.changed_synapses.append(dict(id=f'pi{j}->{i}', weight=round(float(pi.w[i]), 6)))

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
            return                                       # relays are event-driven
        charge = float(magnitude) * self.params['e_threshold']
        if continuous:
            if magnitude <= 0:
                self._continuous.pop(neuron_id, None)
            else:
                self._continuous[neuron_id] = charge
        else:
            self._sched_exc(neuron_id, charge)           # lands next boundary

    def set_feedforward_weight(self, j: int, i: int, weight: float) -> float:
        if not (0 <= j < N_OUT and 0 <= i < N_PIX):
            raise IndexError(f'feedforward index out of range: j={j}, i={i}')
        w = float(np.clip(weight, 0.0, self.params['e_weight_cap']))
        self.l2e[j].acc_weights[i] = w
        return w

    def apply_config(self, overrides: dict):
        applied = []
        for k, v in (overrides or {}).items():
            if k not in EDITABLE_KEYS:
                continue
            self.params[k] = v
            applied.append(k)
        if applied:
            self._build()
            self._log('config', f'applied {applied}; network rebuilt')
        return applied

    def reset(self):
        self._build()
        self._log('control', 'reset: network rebuilt from seed')

    def reseed(self):
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
            return float(self.l1e_new[int(i)].acc_weights[1 + int(j)])
        if kind == 'coincidence_local':
            i = int(eid[2:])
            return float(self.l1e_new[i].acc_weights[0])
        if kind == 'predictive_inhibition':
            j, i = eid[2:].split('->')
            return float(self.pi[int(j)].w[int(i)])
        return None                                       # structural relay / conductance gate

    def topology(self) -> dict:
        neurons = [dict(**self.meta[nid]) for nid in self.order]
        synapses = []
        for e in self.synapses:
            w = self._live_weight(e)
            synapses.append(dict(**e, weight=(None if w is None else round(w, 6))))
        return dict(neurons=neurons, synapses=synapses, layers=['L1', 'L2'],
                    patterns=list(PATTERNS.keys()),
                    pattern_vectors={k: list(map(int, v)) for k, v in PATTERNS.items()},
                    grid=dict(rows=3, cols=3), params=self._public_params())

    def _public_params(self):
        p = self.params
        thr = float(p['e_threshold'])
        out = dict(seed=p['seed'], e_threshold=thr, e_weight_cap=float(p['e_weight_cap']),
                   eta=float(p['eta']), leak_rate=float(p['leak_rate']),
                   refractory_steps=int(p['refractory_steps']),
                   input_period=int(p['input_period']),
                   enew_enabled=bool(p['enew_enabled']),
                   alpha_inh=float(p['alpha_inh']), alpha_inh_l1=float(p['alpha_inh_l1']),
                   alpha_a=float(p['alpha_a']),
                   beta_v=float(p['beta_v']), beta_s=float(p['beta_s']),
                   a_max=float(p['a_max']), e_inh=float(p['e_inh']),
                   pi_eta=float(p['pi_eta']), pi_w_max=float(p['pi_w_max']),
                   pi_lt_decay=float(p['pi_lt_decay']), pi_g_scale=float(p['pi_g_scale']),
                   pi_conductance_enabled=bool(p['pi_conductance_enabled']),
                   pi_plasticity_enabled=bool(p['pi_plasticity_enabled']),
                   l2i_g_scale=float(p['l2i_g_scale']),
                   synaptic_delay=SYNAPTIC_DELAY,
                   i_threshold=round(thr / 3.0, 4),
                   threshold_l2=thr,
                   l2e_weight_cap_frac=float(p['e_weight_cap']) / thr if thr else 1.0)
        return out

    def dynamic_state(self) -> dict:
        neurons = []
        for nid in self.order:
            n = self.neurons[nid]
            thr = self.meta[nid]['threshold'] or 1.0
            pot = float(n.potential)
            rec = dict(
                id=nid, potential=round(pot, 4), activation=round(pot / thr, 4),
                spiked=bool(self.spiked[nid]), freq=round(self.firing_freq(nid), 4),
                refractory=int(n.refractory_timer),
                assembly=(self.winner if nid == self.winner else None))
            if isinstance(n, ExcitatoryNeuron):
                rec['g_inh'] = round(float(n.g_inh), 6)
                rec['trace'] = round(float(n.a), 6)
                rec['v_pre_reset'] = round(float(n.v_pre_reset), 4)
            neurons.append(rec)
        return dict(timestep=self.timestep, running=False, neurons=neurons,
                    changed_synapses=self.changed_synapses,
                    emitted=self.emitted,
                    inhibitory_pulses=self.inhibitory_pulses,
                    input=self.input_vec.astype(int).tolist(),
                    winner=self.winner,
                    stats=self.stats(), log=list(self.event_log)[-12:])

    def stats(self) -> dict:
        active = sum(1 for nid in self.neurons if abs(self.neurons[nid].potential) > 1e-3)
        firing = sum(1 for v in self.spiked.values() if v)
        rate = float(np.mean([self.firing_freq(nid) for nid in self.neurons]))
        return dict(total=len(self.neurons), active=active, firing=firing,
                    firing_rate=round(rate, 4), winner=self.winner)
