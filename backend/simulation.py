"""Conductance-based predictive-inhibition SNN: construction, a synchronous
double-buffered timestep with explicit integer synaptic delays, and state
snapshots for the dashboard.

Two topologies share one synchronous engine, selected by ``topology``:

``topology='pi'`` -- the predictive-inhibition (PI) experiment (default here for the
scientific question)::

        external -> L1E_s[i] ==ff==> L2E[j] --relay--> PI[j]
                        ^                 |
                        |     predictive inhibitory conductance (locally plastic)
                        +-----------------+
                                          L2E[j] --relay--> L2I_WTA --> all L2E (conductance)

    9 L1E_s, 8 L2E, 8 PI (paired 1:1 with L2E), 1 L2I_WTA  = 26 neurons.
    Each PI[j] owns 9 candidate inhibitory output synapses onto L1E_s (72 total).

``topology='old'`` -- the original dense global-inhibition topology (27 neurons)::

        external -> L1E_s[i] ==ff==> L2E[j] --relay--> L2I_WTA --> all L2E (conductance)
                        ^                 |
                        |                 +--relay (DENSE: every L2E -> every L1I)--+
                        |                                                            v
                        +----------- inhibition (paired L1I[i] -> L1E_s[i]) ------ L1I[i]

    9 L1E_s, 9 L1I (paired relays), 8 L2E, 1 L2I_WTA  = 27 neurons. The single L2
    winner drives ALL nine L1I relays (dense feedback), so every L1E_s receives a
    persistent inhibitory conductance pulse on the next boundary -- global inhibition
    gated by the winner. Inhibition is conductance now (no hard wipes anywhere).

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
)
from backend.layout import generate_layout  # noqa: E402
from backend.network_spec import (  # noqa: E402
    preset_spec, validate_spec, ARCHETYPES,
)


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
    topology='pi',                                 # topology selector: 'pi' | 'old'

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
    'topology', 'alpha_inh', 'alpha_inh_l1', 'alpha_a', 'beta_v', 'beta_s',
    'a_max', 'e_inh', 'pi_eta', 'pi_w_max', 'pi_lt_decay', 'pi_g_scale',
    'pi_conductance_enabled', 'pi_plasticity_enabled', 'l2i_g_scale',
}
VALID_TOPOLOGIES = ('pi', 'old')


class SimulationEngine:
    """Owns the network and advances it one synchronous timestep at a time."""

    def __init__(self, seed: int = 1, **overrides):
        params = dict(DEFAULTS)
        params['seed'] = int(seed)
        for k, v in overrides.items():
            if k not in DEFAULTS:
                raise KeyError(f'unknown config key: {k!r}')
            params[k] = v
        if params['topology'] not in VALID_TOPOLOGIES:
            raise ValueError(f'topology must be one of {VALID_TOPOLOGIES}, got {params["topology"]!r}')
        self.params = params
        self._custom_spec = None       # a user/editor NetworkSpec overrides the preset when set
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

        self.pos = generate_layout(rng, N_PIX, N_OUT)  # full functional layout (preset ids)
        thr = float(p['e_threshold'])
        cap = float(p['e_weight_cap'])

        # The active NetworkSpec: an applied custom graph overrides the named preset.
        if self._custom_spec is not None:
            spec = validate_spec(self._custom_spec, N_PIX)
            self.mode = spec.get('name') or 'custom'
        else:
            self.mode = str(p['topology'])             # 'pi' | 'old'
            spec = preset_spec(self.mode, N_PIX, N_OUT)
        self.spec = spec
        self._build_from_spec(spec, rng, thr, cap)

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
        # Feedforward causal volley: the set of sensory source ids that fired at t-1,
        # so a competitor's learning at t reads which of ITS afferents participated.
        self._ff_fired_next = set()
        self._ff_fired_now = set()

    def _distance_factors(self, sources, targets):
        ds = np.array([[np.linalg.norm(self.pos[s] - self.pos[t]) for s in sources]
                       for t in targets])
        positive = ds[ds > 0]
        d_ref = float(positive.min()) if positive.size else 1.0
        return (d_ref / np.maximum(ds, d_ref)) ** DISTANCE_POWER

    def _build_from_spec(self, spec, rng, thr, cap):
        """Construct neurons, meta, edges, and the generic execution adjacency from a
        NetworkSpec. Presets rebuild byte-identically: the RNG draw order (layout, then
        per-competitor feedforward jitter in node order) is preserved.

        Excitatory node positions come from ``node['pos']`` if the spec supplies one
        (editor-placed), else from the seeded functional layout by id (presets)."""
        p = self.params
        i_thr = thr / 3.0
        alpha_l1 = float(p['alpha_inh_l1'])            # slow, predictive-target decay
        alpha_l2 = float(p['alpha_inh'])               # fast, WTA-target decay

        nodes = spec['nodes']
        edges = spec['edges']
        node_by_id = {n['id']: n for n in nodes}

        # ---- positions: spec override, else functional layout by id ----
        pos = {}
        for n in nodes:
            if n.get('pos') is not None:
                pos[n['id']] = np.asarray(n['pos'], dtype=float)
            elif n['id'] in self.pos:
                pos[n['id']] = self.pos[n['id']]
            else:
                pos[n['id']] = np.zeros(3)             # placeholder for an unplaced node
        self.pos = pos

        # ---- expand bidirectional edges into directed delivery edges ----
        # A bidirectional (directed=False) edge delivers BOTH ways; the engine works on
        # directed edges only. The reverse gets a '~r' id and is engine-internal; the
        # editor still sees the single bidirectional edge via ``current_spec``.
        dedges = []
        for e in edges:
            fwd = dict(e)
            fwd['directed'] = True
            dedges.append(fwd)
            if not e.get('directed', True):
                dedges.append(dict(id=f"{e['id']}~r", source=e['target'], target=e['source'],
                                   kind=e['kind'], directed=True,
                                   **({'sign': e['sign']} if 'sign' in e else {})))

        # ---- feedforward wiring first: a competitor's afferent list sizes its weights ----
        ff_by_comp = {}                                # comp_id -> [source_id ...] in edge order
        ff_edge_ids = {}                               # comp_id -> [edge_id ...] aligned
        for e in dedges:
            if e['kind'] == 'feedforward':
                ff_by_comp.setdefault(e['target'], []).append(e['source'])
                ff_edge_ids.setdefault(e['target'], []).append(e['id'])
        # Global distance reference across ALL feedforward pairs (matches the historical
        # _distance_factors: d_ref is the min positive source-target distance overall).
        ff_dists = {e['id']: float(np.linalg.norm(pos[e['source']] - pos[e['target']]))
                    for e in dedges if e['kind'] == 'feedforward'}
        pos_d = [d for d in ff_dists.values() if d > 0]
        d_ref = min(pos_d) if pos_d else 1.0

        def ff_factor(eid):
            return (d_ref / max(ff_dists[eid], d_ref)) ** DISTANCE_POWER

        # ---- construct neurons per archetype (competitors draw jitter in node order) ----
        def jitter(mean, size):
            j = rng.uniform(1.0 - INIT_JITTER_FRAC, 1.0 + INIT_JITTER_FRAC, size=size)
            return np.clip(mean * j, 0.0, cap)

        neurons = {}
        self.sensory, self.competitors, self.relays = [], [], []
        self._sensory_pixel = []
        for n in nodes:
            nid, arch = n['id'], n['archetype']
            if arch == 'e_sensory':
                cell = self._mkE(nid, 'source', np.array([SENSORY_WEIGHT]), np.array([1.0]),
                                 learn=False, alpha_inh=alpha_l1)
                self.sensory.append(cell)
                self._sensory_pixel.append(n.get('pixel'))
                neurons[nid] = cell
            elif arch == 'e_competitor':
                srcs = ff_by_comp.get(nid, [])
                eids = ff_edge_ids.get(nid, [])
                w = jitter(FF_INIT_MEAN, len(srcs)) if srcs else np.zeros(0)
                dfac = np.array([ff_factor(eid) for eid in eids]) if eids else np.zeros(0)
                cell = self._mkE(nid, 'competitor', w, dfac, learn=True, alpha_inh=alpha_l2)
                cell.ff_src = list(srcs)
                cell.ff_edge_ids = list(eids)
                self.competitors.append(cell)
                neurons[nid] = cell
            elif arch == 'i_relay':
                cell = InhibitoryNeuron(nid, 'relay', threshold=i_thr)
                self.relays.append(cell)
                neurons[nid] = cell
            elif arch == 'predictor':
                cell = PredictiveInterneuron(
                    nid, 'predictor', 0, w_init=0.0, w_max=float(p['pi_w_max']),
                    eta=float(p['pi_eta']), lt_decay=float(p['pi_lt_decay']),
                    g_scale=float(p['pi_g_scale']), threshold=i_thr)
                self.relays.append(cell)
                neurons[nid] = cell

        # ---- predictor output wiring: w vector aligns to its predictive targets ----
        pred_out = {}                                  # pred_id -> [(target_id, edge_id) ...]
        for e in dedges:
            if e['kind'] == 'predictive_inhibition':
                pred_out.setdefault(e['source'], []).append((e['target'], e['id']))
        for cell in self.relays:
            if isinstance(cell, PredictiveInterneuron):
                outs = pred_out.get(cell.id, [])
                cell.n_targets = len(outs)
                cell.w = np.zeros(len(outs))
                cell.pred_targets = [t for (t, _) in outs]
                cell.pred_edge_ids = [eid for (_, eid) in outs]

        # ---- registries used across the engine ----
        self.exc = {c.id: c for c in (*self.sensory, *self.competitors)}
        self.inh = {c.id: c for c in self.relays}
        self.neurons = {**self.exc, **self.inh}
        self.order = [n['id'] for n in nodes]
        # Back-compat handles used by tests / experiments (presets only need these):
        self.l1e_s = list(self.sensory)
        self.l2e = list(self.competitors)
        self.pi = [c for c in self.relays if isinstance(c, PredictiveInterneuron)]
        self.l1i = [c for c in self.relays
                    if not isinstance(c, PredictiveInterneuron)
                    and node_by_id[c.id].get('layer') == 'L1']
        self.l2i = self.inh.get('L2I') or next(
            (c for c in self.relays if not isinstance(c, PredictiveInterneuron)), None)
        self._comp_ids = {c.id for c in self.competitors}

        # ---- generic execution adjacency (built once per rebuild) ----
        self._relayexc_out = {}        # source_id -> [(relay_id, edge_id) ...]
        self._inh_out = {}             # relay_id  -> [(target_id, edge_id) ...]
        for e in dedges:
            if e['kind'] == 'relay_excitation':
                self._relayexc_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'inhibition':
                self._inh_out.setdefault(e['source'], []).append((e['target'], e['id']))

        # ---- meta + serialized synapse list ----
        meta = {}
        for n in nodes:
            arch = ARCHETYPES[n['archetype']]
            m = dict(
                id=n['id'], label=n.get('label') or n['id'], layer=n.get('layer', 'L2'),
                type=arch['cls'], role=arch['role'], archetype=n['archetype'],
                threshold=round(thr * arch['thr_frac'], 4),
                pos=[round(float(x), 4) for x in pos[n['id']]])
            if n.get('pixel') is not None:
                m['pixel'] = int(n['pixel'])       # lets the RF map an afferent to a grid cell
            meta[n['id']] = m
        self.meta = meta
        self.synapses = [dict(id=e['id'], source=e['source'], target=e['target'],
                              kind=e['kind'], **({'sign': e['sign']} if 'sign' in e else {}),
                              **({'directed': False} if not e.get('directed', True) else {}))
                         for e in edges]
        # weight lookups for serialization: edge_id -> (cell, weight_index)
        self._ff_weight_ref = {}
        for cell in self.competitors:
            for widx, eid in enumerate(cell.ff_edge_ids):
                self._ff_weight_ref[eid] = (cell, widx)
        self._pred_weight_ref = {}
        for cell in self.pi:
            for widx, eid in enumerate(cell.pred_edge_ids):
                self._pred_weight_ref[eid] = (cell, widx)

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
        self._ff_fired_now = self._ff_fired_next
        self._exc_next = {}
        self._inh_next = []
        self._ff_fired_next = set()

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
        for n, pix in zip(self.sensory, self._sensory_pixel):
            if input_arrives and pix is not None and self.input_vec[pix] > 0.5:
                n.gather_exc(n.acc_weights[0])
        for nid, mag in self._continuous.items():
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(mag)

        # ---- subphase 2: integrate every excitatory neuron once ----
        for n in self.exc.values():
            n.integrate()

        # ---- subphase 3: threshold test + fire ----
        # e_sensory: every threshold crosser fires (sensory sources).
        fired_sensory = set()
        for n in self.sensory:
            if n.can_fire():
                n.fire()
                fired_sensory.add(n.id)
                self.spiked[n.id] = True

        # e_competitor: deterministic single-winner WTA among the crossers (selection,
        # NOT charge removal). Highest membrane wins; tie-break = lowest node order.
        winner = None
        crossers = [(idx, n) for idx, n in enumerate(self.competitors) if n.can_fire()]
        if crossers:
            _, winner = max(crossers, key=lambda kv: (kv[1].V, -kv[0]))
            winner.fire()
            # Learn feedforward: participation = which of THIS competitor's afferents
            # fired in the causal volley that arrived this boundary.
            part = np.array([src in self._ff_fired_now for src in winner.ff_src], dtype=bool)
            winner.update_acc_weights(part)
            self._emit_ff_weight_changes(winner)
            self.spiked[winner.id] = True
            self.winner = winner.id

        # ---- subphase 4: update local activity traces (survive reset) ----
        for n in self.exc.values():
            n.update_trace()

        # ---- subphase 5: emit spikes into delay-1 queues; run local relays/plasticity ----
        # sensory spikes -> feedforward charge onto competitor targets (delay 1).
        if fired_sensory:
            for comp in self.competitors:
                q = 0.0
                for widx, src in enumerate(comp.ff_src):
                    if src in fired_sensory:
                        q += float(comp.acc_weights[widx])
                        self.emitted.append(comp.ff_edge_ids[widx])
                if q:
                    self._sched_exc(comp.id, q)
            self._ff_fired_next = fired_sensory

        # winner competitor drives its relay_excitation targets THIS boundary; each
        # relay that fires schedules its inhibitory / predictive conductance for t+1.
        if winner is not None:
            self._drive_relays(winner)

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

    def _drive_relays(self, winner):
        """The winner competitor drives every relay it projects to (relay_excitation);
        each relay that fires emits its outgoing conductance for the NEXT boundary.

        This one traversal expresses both topologies: whatever relay_excitation edges
        leave the winner (its paired L2I plus paired PI, or L2I plus every L1I) fire
        their targets, and each fired relay's own inhibition / predictive_inhibition
        edges schedule the persistent conductance pulses. An i_relay emits a fixed
        pulse; a predictor emits pre-update pulse[i] AND learns locally."""
        p = self.params
        g = float(p['l2i_g_scale'])
        for rid, re_eid in self._relayexc_out.get(winner.id, []):
            relay = self.inh.get(rid)
            if relay is None:
                continue
            relay.receive()
            if not relay.resolve():
                continue
            self.spiked[rid] = True
            self.emitted.append(re_eid)
            if isinstance(relay, PredictiveInterneuron):
                if p['pi_conductance_enabled']:
                    pulse = relay.conductance_pulse()    # g_scale * w (pre-update)
                    for widx, (tgt, eid) in enumerate(zip(relay.pred_targets, relay.pred_edge_ids)):
                        if pulse[widx] > 0.0:
                            self._sched_inh(tgt, float(pulse[widx]), rid, 'predictive',
                                            weight=float(relay.w[widx]))
                            self.emitted.append(eid)
                if p['pi_plasticity_enabled']:
                    # strictly-local: each output synapse reads only its own target's
                    # trace (finalized in subphase 4) and its own weight.
                    traces = np.array([self.exc[t].a if t in self.exc else 0.0
                                       for t in relay.pred_targets])
                    relay.learn(traces)
                    for widx, eid in enumerate(relay.pred_edge_ids):
                        self.changed_synapses.append(
                            dict(id=eid, weight=round(float(relay.w[widx]), 6)))
            else:
                # i_relay: fixed inhibitory conductance onto each of its targets.
                for tgt, eid in self._inh_out.get(rid, []):
                    kind = 'wta' if tgt in self._comp_ids else 'inhibition'
                    self._sched_inh(tgt, g, rid, kind, weight=g)
                    self.emitted.append(eid)

    def _emit_ff_weight_changes(self, comp):
        for widx, eid in enumerate(comp.ff_edge_ids):
            self.changed_synapses.append(dict(id=eid, weight=round(float(comp.acc_weights[widx]), 4)))

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
        if not (0 <= j < len(self.l2e) and 0 <= i < len(self.l2e[j].acc_weights)):
            raise IndexError(f'feedforward index out of range: j={j}, i={i}')
        w = float(np.clip(weight, 0.0, self.params['e_weight_cap']))
        self.l2e[j].acc_weights[i] = w
        return w

    def set_synapse_weight(self, edge_id: str, weight: float) -> float:
        """Hand-set any plastic synapse weight by its edge id (topology-agnostic):
        feedforward -> competitor.acc_weights (clip [0, e_weight_cap]); predictive
        -> predictor.w (clip [0, pi_w_max]). Raises KeyError for a non-plastic/unknown
        edge. Best used while paused."""
        ref = self._ff_weight_ref.get(edge_id)
        if ref is not None:
            cell, widx = ref
            w = float(np.clip(weight, 0.0, self.params['e_weight_cap']))
            cell.acc_weights[widx] = w
            return w
        ref = self._pred_weight_ref.get(edge_id)
        if ref is not None:
            cell, widx = ref
            w = float(np.clip(weight, 0.0, self.params['pi_w_max']))
            cell.w[widx] = w
            return w
        raise KeyError(edge_id)

    def apply_config(self, overrides: dict):
        applied = []
        for k, v in (overrides or {}).items():
            if k not in EDITABLE_KEYS:
                continue
            if k == 'topology' and v not in VALID_TOPOLOGIES:
                raise ValueError(f'topology must be one of {VALID_TOPOLOGIES}, got {v!r}')
            self.params[k] = v
            applied.append(k)
        # Selecting a named preset topology discards any applied custom graph.
        if 'topology' in applied:
            self._custom_spec = None
        if applied:
            self._build()
            self._log('config', f'applied {applied}; network rebuilt')
        return applied

    # ---------------------------------------------------------- topology editing
    def current_spec(self) -> dict:
        """The active NetworkSpec with live (resolved) positions -- what the editor
        loads. Weights are NOT included; live weights come from ``topology()``."""
        nodes = []
        for n in self.spec['nodes']:
            m = self.meta[n['id']]
            node = dict(id=n['id'], archetype=n['archetype'], layer=m['layer'],
                        label=m['label'], pos=list(m['pos']))
            if n.get('pixel') is not None:
                node['pixel'] = n['pixel']
            nodes.append(node)
        edges = [dict(e) for e in self.spec['edges']]
        return dict(name=self.mode, nodes=nodes, edges=edges,
                    is_custom=self._custom_spec is not None)

    def apply_topology(self, spec: dict):
        """Validate and install a custom NetworkSpec, then rebuild. Raises SpecError
        (a ValueError) if the graph is structurally invalid. Learned state is reset."""
        norm = validate_spec(spec, N_PIX)
        self._custom_spec = norm
        self._build()
        self._log('topology', f"applied topology '{self.mode}': "
                              f"{len(norm['nodes'])} nodes, {len(norm['edges'])} edges")
        return self.current_spec()

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
            ref = self._ff_weight_ref.get(eid)
            return None if ref is None else float(ref[0].acc_weights[ref[1]])
        if kind == 'predictive_inhibition':
            ref = self._pred_weight_ref.get(eid)
            return None if ref is None else float(ref[0].w[ref[1]])
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
                   topology=str(p['topology']),
                   topology_name=str(self.mode),
                   is_custom_topology=bool(self._custom_spec is not None),
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
