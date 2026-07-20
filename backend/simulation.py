"""Conductance-based predictive-inhibition SNN: construction, a synchronous
double-buffered timestep with explicit integer synaptic delays, and state
snapshots for the dashboard.

Five topologies share one engine, selected by ``topology``:

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

``topology='rg'`` -- the `old` cortical topology with an explicit retinal-ganglion
source layer spliced in ahead of L1 (36 neurons)::

        external -> RG[i] ==ff(plastic, paired 1:1)==> L1E[i] ==ff(plastic, dense)==> L2E[j]
                                  ^                                    |
                                  |                                    +--relay--> L2I_WTA --> all L2E
                                  |                                    |
                                  +---- inhibition (paired L1I[i]) --- L1I[i] <--relay (DENSE)--+

    9 RG, 9 L1E, 9 L1I, 8 L2E, 1 L2I = 36 neurons; 178 internal synapses.

    RG cells are exogenous binary spike sources: a held active pixel makes its RG cell
    spike on EVERY ``input_period`` boundary, and no cortical inhibition (L1I, L2I) can
    stop it. The direct external->L1E injection of `old` is removed here: every L1E
    charge is a real, weighted, delay-1 RG spike. L1E is therefore a plastic
    NONCOMPETITIVE ``e_encoder`` -- it learns its single RG afferent with exactly the
    same accumulating rule as L2E, but every crosser fires (no WTA in L1).

    This preset isolates the timing consequences of a persistent source layer and a
    plastic L1 path. It is NOT expected to show contextual explaining-away: the dense
    L2E->L1I feedback erases winner identity because every winner drives every L1I.

``topology='rg_residual'`` -- a 52-cell classification-preserving residual circuit.
RG drives plastic, uninhibited L1E encoders. L1E supplies the full pattern to L2E and
a fixed paired copy to a separate ErrorE sheet. Paired PI cells learn predictive
inhibition onto ErrorE. Residual spikes broadcast to eight SwitchI cells, but only a
cell carrying its paired competitor's pre-existing local eligibility trace fires and
inhibits that incumbent. Shared L2I still enforces a deterministic single L2 winner.
The graph has exactly 274 directed internal projections.

Timestep semantics (see ``step``): every internal excitatory projection has an
integer delay of 1; relays (L2I_WTA, PI, L1I) fire in the same boundary as their
source spike and schedule their inhibitory *conductance* output for the next
boundary. External input arrives at the current boundary. All targets integrate
once per boundary from double-buffered arrivals, so no behaviour depends on Python
neuron-iteration order.

Feedforward dispatch is generic over hops: ANY permitted source spike (rg_source or
any fired E-class cell, including a competitor) schedules weighted charge onto its plastic targets
for the next boundary, and causal participation is recorded **per postsynaptic target
per arrival boundary**. In `rg` that yields the two-hop chain

    t: RG emits -> t+1: L1E integrates, fires, learns its RG afferent, emits
    -> t+2: L2E integrates, WTA picks one winner, it learns its delivered L1E volley
       and drives L2I + all nine L1I -> t+3: inhibitory conductance arrives at L1E/L2E.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn.neurons import (  # noqa: E402
    ConductanceLIFNeuron,
    CoincidencePyramidalNeuron,
    ExcitatoryNeuron,
    InhibitoryNeuron,
    PredictiveInterneuron,
    SwitchInterneuron,
    SourceNeuron,
    E_THRESHOLD,
    I_THRESHOLD,
    E_WEIGHT_CAP,
    leak_to_conductance,
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
FF_INIT_TOTAL_FRAC = 0.55                          # legacy ordinary-E mean scale, / theta
FF_INIT_MEAN = FF_INIT_TOTAL_FRAC * E_THRESHOLD / N_PIX   # ~61
L2_INIT_TOTAL_FRAC = 0.95                          # latency-WTA row total / theta
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
    topology='pi',                                 # preset selector (see VALID_TOPOLOGIES)

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

    # --- plastic encoder (e_encoder) feedforward: the RG -> L1E path in 'rg' ---
    # These are PROJECTION-LEVEL controls, not a hidden L1 code branch: the learning
    # equation, threshold, leak, eta, cap and trace are shared with L2E verbatim. They
    # exist so the timing experiment can run its frozen / equal-init controls without
    # editing the engine. Defaults reproduce "initialized exactly like an L2 afferent".
    enc_plasticity_enabled=True,                   # OFF -> encoder ff weights frozen at init
    enc_init_jitter=True,                          # OFF -> every encoder afferent starts at enc_w_init
    enc_w_init=None,                               # None -> FF_INIT_MEAN (the shared L2 per-afferent scale)

    # --- residual/error pathway ('rg_residual') ---
    # A fixed L1E spike copy must remain supra-threshold after one leaky integration
    # boundary; 1.10*theta yields ~1.08*theta at the default leak from rest.
    residual_exc_scale=1.10,                       # fixed L1E -> ErrorE charge, / theta
    switch_trace_decay=0.97,                       # spans observed L2->ErrorE timing gaps
    switch_trace_threshold=0.50,                   # temporal-AND eligibility threshold
    switch_residual_charge_frac=0.55,              # charge per ErrorE event, / I threshold
    switch_trace_charge_frac=0.90,                 # max paired-trace priming, / I threshold
    switch_branch_cap_frac=0.90,                   # each branch alone stays subthreshold
    switch_g_scale=12.0,                           # paired incumbent inhibitory pulse
    switch_conductance_enabled=True,               # ablation: SwitchI may fire but not inhibit

    # --- coincidence pyramidal cell (event-resolved 'rg_coincidence' topology) ---
    # C threshold reads e_threshold, C leak reads leak_rate, C refractory reads
    # refractory_steps directly (intrinsic parity with E). The basal weight scale is
    # DERIVED from the resolved threshold + leak via the two-coincidence equations
    # unless a headless experiment overrides it; None => derive. C learning is slower
    # than ordinary E learning: the row->column->row sweep found eta_c=0.001 retained
    # the novelty window and produced turnover/recovery in 8/8 seeds.
    c_eta=0.001,                                   # separately controlled basal learning rate
    l2_init_total_frac=L2_INIT_TOTAL_FRAC,          # normalized latency-WTA afferent total / theta
    c_basal_weight_init=None,                      # None -> 1.01 * w_2(T)
    c_basal_weight_max=None,                       # None -> 1.10 * w_2(T)
    c_basal_window_steps=1,                        # basal eligibility window (fixed at 1)
    pretrained_exc_margin=1.05,                    # RG->L1E fixed packet / (theta/kappa)
    crossing_time_tolerance=1e-12,                 # fixed numeric tie tolerance (not a control)
)
# Keys a browser/experiment config-apply may change. Everything else is derived.
EDITABLE_KEYS = {
    'eta', 'leak_rate', 'refractory_steps', 'e_weight_cap', 'input_period',
    'topology', 'alpha_inh', 'alpha_inh_l1', 'alpha_a', 'beta_v', 'beta_s',
    'a_max', 'e_inh', 'pi_eta', 'pi_w_max', 'pi_lt_decay', 'pi_g_scale',
    'pi_conductance_enabled', 'pi_plasticity_enabled', 'l2i_g_scale',
    'enc_plasticity_enabled', 'enc_init_jitter', 'enc_w_init',
    'residual_exc_scale', 'switch_trace_decay', 'switch_trace_threshold',
    'switch_residual_charge_frac', 'switch_trace_charge_frac',
    'switch_g_scale', 'switch_conductance_enabled',
    'c_eta', 'l2_init_total_frac',
}
VALID_TOPOLOGIES = ('pi', 'old', 'rg', 'rg_residual', 'rg_coincidence')


class BoundaryEventScheduler:
    """Engine-owned helper that orders analytic sub-boundary threshold crossings within
    one outer boundary. It owns NO scientific state beyond ``current_tau`` and the tie
    log: every membrane computes its own local crossing time from the same equation, and
    the scheduler merely picks the earliest and advances all membranes to it.

    It never inspects learned weights, a global input pattern, or end-of-boundary
    voltages to choose a winner -- selection is pure first-spike latency. Exact or
    within-tolerance ties fall back to stable node order (recorded as a ``latency_tie``);
    times that are merely close but distinguishable are NOT tie-broken by node order.
    """

    def __init__(self, membranes, tolerance):
        self.membranes = list(membranes)     # fixed stable node order
        self.tolerance = float(tolerance)
        self.current_tau = 0.0
        self.ties = []

    def next_event(self):
        """Return ``(cell, absolute_tau)`` for the earliest finite crossing in
        ``[current_tau, 1]``, or ``(None, None)`` if none remains. Recomputed fresh from
        live membrane state each call, so a hard reset that just changed a cell's state
        automatically invalidates its previously-predicted crossing."""
        remaining = 1.0 - self.current_tau
        cands = []
        for idx, cell in enumerate(self.membranes):
            dtau = cell.crossing_time(remaining)
            if math.isfinite(dtau):
                cands.append((self.current_tau + dtau, idx, cell))
        if not cands:
            return None, None
        min_tau = min(c[0] for c in cands)
        tied = [c for c in cands if abs(c[0] - min_tau) <= self.tolerance]
        winner = min(tied, key=lambda c: c[1])       # stable node order among true ties
        if len(tied) > 1:
            self.ties.append(dict(
                tau=round(min_tau, 12), tolerance=self.tolerance,
                ids=[c[2].id for c in sorted(tied, key=lambda c: c[1])],
                chosen=winner[2].id))
        return winner[2], winner[0]

    def advance_all(self, target_tau):
        """Advance EVERY membrane from ``current_tau`` to ``target_tau`` with the exact
        segment trajectory (each under its own frozen drive)."""
        dt = target_tau - self.current_tau
        if dt < 0.0:
            dt = 0.0
        for cell in self.membranes:
            cell.advance_segment(dt)
        self.current_tau = target_tau

    def advance_to_end(self):
        if self.current_tau < 1.0:
            self.advance_all(1.0)


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
        if not 0.0 < float(params['l2_init_total_frac']) < 1.0:
            raise ValueError('l2_init_total_frac must satisfy 0 < rho < 1')
        if params['c_eta'] is not None and float(params['c_eta']) < 0.0:
            raise ValueError('c_eta must be non-negative or None')
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

    def _resolve_coincidence_params(self):
        """Resolve the C-cell basal weight scale + pretrained packet from the shared
        threshold and leak using the two-coincidence equations, honoring explicit
        headless overrides. Rejects a configuration whose cap can fire on ONE deposit.

            r = e^{-g_L},  kappa = (1 - e^{-g_L}) / g_L  (kappa = 1 at g_L = 0)
            w_2(T) = theta / (kappa (1 + r^T))      # min weight crossing on 2nd deposit
            w_1    = theta / kappa                   # min weight firing on ONE deposit
            Q_pretrained = pretrained_exc_margin * theta / kappa
        """
        p = self.params
        theta = float(p['e_threshold'])
        g_L = leak_to_conductance(float(p['leak_rate']))
        r = math.exp(-g_L)                                # == 1 - leak_rate
        kappa = 1.0 if g_L == 0.0 else (1.0 - math.exp(-g_L)) / g_L
        T = int(p['c_basal_window_steps'])
        if T < 1:
            raise ValueError(f'c_basal_window_steps must be >= 1, got {T}')
        w2 = theta / (kappa * (1.0 + r ** T))
        w1 = theta / kappa
        c_init = (1.01 * w2 if p['c_basal_weight_init'] is None
                  else float(p['c_basal_weight_init']))
        c_max = (1.10 * w2 if p['c_basal_weight_max'] is None
                 else float(p['c_basal_weight_max']))
        if not c_max < w1:
            raise ValueError(
                f'c_basal_weight_max ({c_max:.4f}) must be < the one-deposit firing '
                f'weight w_1 ({w1:.4f}); a C cell would fire on a single coincidence.')
        if c_init > c_max:
            raise ValueError(
                f'c_basal_weight_init ({c_init:.4f}) must be <= c_basal_weight_max '
                f'({c_max:.4f}).')
        c_eta = float(p['eta']) if p['c_eta'] is None else float(p['c_eta'])
        q_pretrained = float(p['pretrained_exc_margin']) * theta / kappa
        return dict(w1=w1, w2=w2, kappa=kappa, c_init=c_init, c_max=c_max,
                    c_eta=c_eta, q_pretrained=q_pretrained)

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
        # Feedforward causal volley, tracked PER POSTSYNAPTIC TARGET PER ARRIVAL
        # BOUNDARY: target_id -> {source ids whose spike is delivered to that target on
        # the next boundary}. A plastic cell's learning at t therefore reads exactly the
        # afferents delivered to IT at t -- one hop's volley can never leak into another
        # target's update, and with two hops the L1E and L2E updates stay independent.
        self._ff_deliv_next = {}
        self._ff_deliv_now = {}
        # Event-path delay-1 dendritic delivery buffers (basal/apical events for t+1).
        self._basal_next = {}          # c_target_id -> [(source_id, signal) ...]
        self._apical_next = {}         # c_target_id -> [source_id ...]
        # Event-path per-boundary diagnostics (exposed via dynamic state in Phase 6).
        self.hard_reset_events = []
        self.latency_ties = []

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

        # ---- feedforward wiring first: a target's afferent list sizes its weights ----
        ff_by_tgt = {}                                 # target_id -> [source_id ...] in edge order
        ff_edge_ids = {}                               # target_id -> [edge_id ...] aligned
        for e in dedges:
            if e['kind'] == 'feedforward':
                ff_by_tgt.setdefault(e['target'], []).append(e['source'])
                ff_edge_ids.setdefault(e['target'], []).append(e['id'])
        # Distance reference: the 1/d^2 learning-rate factor is normalized so the closest
        # connected pair scores 1.0. The reference is taken PER PLASTIC-TARGET POPULATION
        # (per target archetype), not globally across every feedforward edge. With one
        # feedforward population this is identical to the historical global rule, so
        # pi/old are untouched; with two hops it stops the short RG->L1E projection from
        # rescaling every L1E->L2E factor, which would silently make `rg`'s L2 learning
        # non-comparable to `old`'s.
        ff_dists, ff_tgt_arch = {}, {}
        for e in dedges:
            if e['kind'] != 'feedforward':
                continue
            ff_dists[e['id']] = float(np.linalg.norm(pos[e['source']] - pos[e['target']]))
            ff_tgt_arch[e['id']] = node_by_id[e['target']]['archetype']
        d_ref_by_arch = {}
        for eid, d in ff_dists.items():
            if d > 0:
                a = ff_tgt_arch[eid]
                d_ref_by_arch[a] = min(d_ref_by_arch.get(a, d), d)

        def ff_factor(eid):
            d_ref = d_ref_by_arch.get(ff_tgt_arch[eid], 1.0)
            return (d_ref / max(ff_dists[eid], d_ref)) ** DISTANCE_POWER

        # ---- construct neurons per archetype (plastic cells draw jitter in node order) ----
        def jitter(mean, size):
            j = rng.uniform(1.0 - INIT_JITTER_FRAC, 1.0 + INIT_JITTER_FRAC, size=size)
            return np.clip(mean * j, 0.0, cap)

        def normalized_latency_row(raw):
            """Preserve seeded direction while setting a common latency-WTA total.

            A full coincidence bank has nine afferents and reaches exactly
            ``l2_init_total_frac * theta``. A smaller custom bank cannot exceed its
            physical ``n_afferents * cap`` capacity; the bounded proportional fill
            handles that case without making valid editor graphs unbuildable.
            """
            raw = np.asarray(raw, dtype=float)
            if raw.size == 0:
                return raw
            target = min(float(p['l2_init_total_frac']) * thr, raw.size * cap)
            out = np.zeros_like(raw)
            active = np.ones(raw.size, dtype=bool)
            remaining = target
            while np.any(active):
                active_sum = float(raw[active].sum())
                if active_sum <= 0.0:
                    out[active] = remaining / int(active.sum())
                    break
                proposal = raw[active] * (remaining / active_sum)
                over = proposal > cap
                if not np.any(over):
                    out[active] = proposal
                    break
                active_idx = np.flatnonzero(active)
                capped_idx = active_idx[over]
                out[capped_idx] = cap
                remaining -= cap * len(capped_idx)
                active[capped_idx] = False
            return out

        enc_mean = FF_INIT_MEAN if p['enc_w_init'] is None else float(p['enc_w_init'])
        enc_jitter = bool(p['enc_init_jitter'])

        # ---- plastic feedforward initialization, drawn up front -------------------
        # Ordinary competitors and encoders retain the historical per-afferent scale.
        # Latency competitors start from the same seeded narrow jitter directions, then
        # normalize each row to l2_init_total_frac * theta for equal initial free energy.
        #
        # Draw order is BY ARCHETYPE (all competitors in node order, then all encoders
        # in node order) rather than raw node order. Two consequences, both deliberate:
        #   * pi/old are untouched -- they have no encoders, so this is still exactly
        #     "each competitor in node order";
        #   * `rg` gets bit-identical L2E initialization to `old`, so the timing
        #     experiment compares the two topologies at the same L2 seed instead of
        #     confounding the RG layer with a reshuffled competitor init.
        # The equal-init control still DRAWS (then discards) its jitter, so toggling
        # enc_init_jitter changes only the encoder weights and nothing downstream.
        # A third pass initializes e_latency_competitor LAST. It only draws for graphs
        # that contain latency competitors, so presets without them keep a bit-identical
        # RNG draw order (and therefore identical goldens).
        ff_w = {}
        for arch_pass in ('e_competitor', 'e_encoder', 'e_latency_competitor'):
            for n in nodes:
                if n['archetype'] != arch_pass:
                    continue
                srcs = ff_by_tgt.get(n['id'], [])
                if not srcs:
                    ff_w[n['id']] = np.zeros(0)
                    continue
                if arch_pass == 'e_competitor':
                    ff_w[n['id']] = jitter(FF_INIT_MEAN, len(srcs))
                elif arch_pass == 'e_latency_competitor':
                    ff_w[n['id']] = normalized_latency_row(
                        jitter(FF_INIT_MEAN, len(srcs)))
                else:
                    j = jitter(enc_mean, len(srcs))
                    ff_w[n['id']] = j if enc_jitter else np.clip(
                        np.full(len(srcs), enc_mean), 0.0, cap)

        # ---- coincidence dendrite wiring: one basal + >=1 apical afferent per C cell.
        # d_ref for the basal distance influence is the smallest positive basal-edge
        # distance across the whole e_coincidence target population (geometry changes
        # learning rate only, never delivered charge).
        basal_in_map = {}        # c_target_id -> (source_id, edge_id)
        apical_in_map = {}       # c_target_id -> [(source_id, edge_id) ...]
        basal_dist = {}          # c_target_id -> functional L1E->L1C distance
        for e in dedges:
            if e['kind'] == 'basal_excitation':
                basal_in_map[e['target']] = (e['source'], e['id'])
                basal_dist[e['target']] = float(np.linalg.norm(pos[e['source']] - pos[e['target']]))
            elif e['kind'] == 'apical_excitation':
                apical_in_map.setdefault(e['target'], []).append((e['source'], e['id']))
        _basal_positive = [d for d in basal_dist.values() if d > 0]
        d_ref_basal = min(_basal_positive) if _basal_positive else 1.0
        basal_phi = {t: (d_ref_basal / max(d, d_ref_basal)) ** DISTANCE_POWER
                     for t, d in basal_dist.items()}

        # Resolve C basal weight scale + pretrained packet only when this graph uses
        # them; a legacy graph never pays the cost and never risks the invariant check.
        has_coincidence = any(n['archetype'] == 'e_coincidence' for n in nodes)
        has_pretrained = any(n['archetype'] == 'e_pretrained' for n in nodes)
        cpar = (self._resolve_coincidence_params()
                if (has_coincidence or has_pretrained) else None)
        self._q_pretrained = cpar['q_pretrained'] if cpar else 0.0

        neurons = {}
        self.sensory, self.encoders, self.residuals, self.competitors = [], [], [], []
        self.sources, self.relays, self.switches, self.plastic = [], [], [], []
        self.coincidence, self.latency_competitors, self.pretrained = [], [], []
        self._input_sinks = []                         # [(cell, pixel) ...] external drive
        for n in nodes:
            nid, arch = n['id'], n['archetype']
            if arch == 'rg_source':
                cell = SourceNeuron(nid, 'rg_source', threshold=thr)
                self.sources.append(cell)
                self._input_sinks.append((cell, n.get('pixel')))
                neurons[nid] = cell
            elif arch == 'e_sensory':
                cell = self._mkE(nid, 'source', np.array([SENSORY_WEIGHT]), np.array([1.0]),
                                 learn=False, alpha_inh=alpha_l1)
                self.sensory.append(cell)
                self._input_sinks.append((cell, n.get('pixel')))
                neurons[nid] = cell
            elif arch in ('e_encoder', 'e_competitor'):
                is_comp = (arch == 'e_competitor')
                srcs = ff_by_tgt.get(nid, [])
                eids = ff_edge_ids.get(nid, [])
                w = ff_w[nid]
                dfac = np.array([ff_factor(eid) for eid in eids]) if eids else np.zeros(0)
                cell = self._mkE(
                    nid, 'competitor' if is_comp else 'encoder', w, dfac,
                    # An encoder shares L2E's threshold, rest/reset, leak, refractory,
                    # eta, weight cap, trace equation and learning rule. Only its
                    # inhibitory-conductance retention differs, and that is a CIRCUIT
                    # timescale keyed to which relay population targets it (L1I -> L1E
                    # uses alpha_inh_l1, L2I -> L2E uses alpha_inh), not a second
                    # excitatory learning rule.
                    learn=True if is_comp else bool(p['enc_plasticity_enabled']),
                    alpha_inh=alpha_l2 if is_comp else alpha_l1)
                cell.ff_src = list(srcs)
                cell.ff_edge_ids = list(eids)
                (self.competitors if is_comp else self.encoders).append(cell)
                self.plastic.append(cell)
                neurons[nid] = cell
            elif arch == 'e_latency_competitor':
                # Same flat feedforward bank + accumulating rule as a legacy competitor,
                # but it competes by first-spike latency (event-resolved), so it is NOT
                # registered in the deterministic-WTA self.competitors list.
                srcs = ff_by_tgt.get(nid, [])
                eids = ff_edge_ids.get(nid, [])
                w = ff_w[nid]
                dfac = np.array([ff_factor(eid) for eid in eids]) if eids else np.zeros(0)
                cell = self._mkE(nid, 'competitor', w, dfac, learn=True, alpha_inh=alpha_l2)
                cell.ff_src = list(srcs)
                cell.ff_edge_ids = list(eids)
                self.latency_competitors.append(cell)
                self.plastic.append(cell)                # reuse ff delivery bookkeeping
                neurons[nid] = cell
            elif arch == 'e_pretrained':
                # Fixed-input noncompetitive relay: empty plastic afferent arrays,
                # learn=False. Its RG afferent is a fixed pretrained_excitation packet
                # delivered via gather_exc, never an accumulating weight.
                cell = self._mkE(nid, 'pretrained', np.zeros(0), np.zeros(0),
                                 learn=False, alpha_inh=alpha_l1)
                cell.ff_src = []
                cell.ff_edge_ids = []
                self.pretrained.append(cell)
                neurons[nid] = cell
            elif arch == 'e_coincidence':
                basal = basal_in_map.get(nid)
                if basal is None:
                    raise ValueError(f'e_coincidence {nid!r} has no basal_excitation edge')
                apical = apical_in_map.get(nid, [])
                cell = CoincidencePyramidalNeuron(
                    nid, basal[0], basal[1],
                    apical_sources=[s for (s, _) in apical],
                    apical_edge_ids=[eid for (_, eid) in apical],
                    basal_weight=cpar['c_init'], basal_distance_factor=basal_phi.get(nid, 1.0),
                    w_max=cpar['c_max'], eta_c=cpar['c_eta'], learn=True,
                    threshold=thr, leak_rate=float(p['leak_rate']),
                    refractory_steps=int(p['refractory_steps']),
                    e_inh=float(p['e_inh']), alpha_inh=alpha_l1,
                    alpha_a=float(p['alpha_a']), beta_v=float(p['beta_v']),
                    beta_s=float(p['beta_s']), a_max=float(p['a_max']))
                self.coincidence.append(cell)
                neurons[nid] = cell
            elif arch == 'e_residual':
                # ErrorE owns no plastic afferents. Fixed L1 evidence-copy events are
                # gathered generically, while PI conductance is integrated jointly.
                cell = self._mkE(nid, 'residual', np.zeros(0), np.zeros(0),
                                 learn=False, alpha_inh=alpha_l1)
                cell.ff_src = []
                cell.ff_edge_ids = []
                self.residuals.append(cell)
                neurons[nid] = cell
            elif arch == 'i_relay':
                cell = InhibitoryNeuron(nid, 'relay', threshold=i_thr)
                self.relays.append(cell)
                neurons[nid] = cell
            elif arch == 'switch':
                cell = SwitchInterneuron(
                    nid, 'switch', trace_decay=float(p['switch_trace_decay']),
                    trace_threshold=float(p['switch_trace_threshold']),
                    residual_charge_frac=float(p['switch_residual_charge_frac']),
                    trace_charge_frac=float(p['switch_trace_charge_frac']),
                    branch_cap_frac=float(p['switch_branch_cap_frac']), threshold=i_thr)
                self.relays.append(cell)
                self.switches.append(cell)
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
        # C and latency/pretrained cells share the excitatory membrane registry so any
        # membrane-bearing traversal (integration, trace, serialization) includes them.
        self.exc = {c.id: c for c in
                    (*self.sensory, *self.encoders, *self.residuals, *self.competitors,
                     *self.pretrained, *self.latency_competitors, *self.coincidence)}
        self.inh = {c.id: c for c in self.relays}
        self.src = {c.id: c for c in self.sources}
        self.neurons = {**self.exc, **self.inh, **self.src}
        # This graph is event-resolved iff any node is an event-resolved archetype or any
        # edge is an immediate hard reset (metadata-driven; never keyed to node ids or the
        # preset name). Legacy graphs keep the byte-compatible synchronous step path.
        self.event_resolved = (
            any(ARCHETYPES[n['archetype']]['event_resolved'] for n in nodes)
            or any(e['kind'] == 'hard_reset_inhibition' for e in edges))
        self.order = [n['id'] for n in nodes]
        # Back-compat handles used by tests / experiments (presets only need these):
        # l1e_s is "the L1 excitatory cells": the fixed-afferent sources in pi/old, the
        # plastic encoders in rg. rg is the RG source layer (empty in pi/old).
        self.l1e_s = [*self.sensory, *self.encoders]
        self.rg = list(self.sources)
        self.l2e = list(self.competitors)
        self.pi = [c for c in self.relays if isinstance(c, PredictiveInterneuron)]
        self.l1i = [c for c in self.relays
                    if not isinstance(c, PredictiveInterneuron)
                    and node_by_id[c.id].get('layer') == 'L1']
        self.l2i = self.inh.get('L2I') or next(
            (c for c in self.relays if not isinstance(c, PredictiveInterneuron)), None)
        self._comp_ids = {c.id for c in self.competitors}

        # ---- generic execution adjacency (built once per rebuild) ----
        self._relayexc_out = {}        # source_id -> ordinary relay/predictor targets
        self._switch_residual_out = {} # ErrorE source -> SwitchI targets
        self._trace_out = {}           # paired L2E source -> SwitchI trace target
        self._fixedexc_out = {}        # E source -> fixed-charge residual targets
        self._inh_out = {}             # relay_id -> [(target_id, edge_id) ...]
        # ---- event-resolved structured dispatch adjacency (coincidence topology) ----
        self._pretrained_out = {}      # rg source -> [(e_pretrained target, edge_id) ...]
        self._basal_out = {}           # L1E source -> [(e_coincidence target, edge_id) ...]
        self._apical_out = {}          # L2E source -> [(e_coincidence target, edge_id) ...]
        self._hardreset_out = {}       # i_relay -> [(E target, edge_id) ...] immediate reset
        for e in dedges:
            if e['kind'] == 'relay_excitation':
                if node_by_id[e['target']]['archetype'] == 'switch':
                    self._switch_residual_out.setdefault(e['source'], []).append(
                        (e['target'], e['id']))
                else:
                    self._relayexc_out.setdefault(e['source'], []).append(
                        (e['target'], e['id']))
            elif e['kind'] == 'trace_excitation':
                self._trace_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'fixed_excitation':
                self._fixedexc_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'inhibition':
                self._inh_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'pretrained_excitation':
                self._pretrained_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'basal_excitation':
                self._basal_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'apical_excitation':
                self._apical_out.setdefault(e['source'], []).append((e['target'], e['id']))
            elif e['kind'] == 'hard_reset_inhibition':
                self._hardreset_out.setdefault(e['source'], []).append((e['target'], e['id']))

        # ---- meta + serialized synapse list ----
        meta = {}
        for n in nodes:
            arch = ARCHETYPES[n['archetype']]
            m = dict(
                id=n['id'], label=n.get('label') or n['id'], layer=n.get('layer', 'L2'),
                type=arch['cls'], role=arch['role'], archetype=n['archetype'],
                threshold=round(thr * arch['thr_frac'], 4),
                pos=[round(float(x), 4) for x in pos[n['id']]])
            # 'pixel' here is the DISPLAY grid cell this unit stands for, so the
            # receptive-field view can place an afferent. It comes from the node's input
            # ownership (e_sensory / rg_source) or, for a downstream encoder that owns no
            # input, from its display 'grid' tag.
            gridcell = n.get('pixel', n.get('grid'))
            if gridcell is not None:
                m['pixel'] = int(gridcell)
            if n.get('pixel') is not None:
                m['owns_input'] = True             # this cell is the external input sink
            meta[n['id']] = m
        self.meta = meta
        self.synapses = [dict(id=e['id'], source=e['source'], target=e['target'],
                              kind=e['kind'], **({'sign': e['sign']} if 'sign' in e else {}),
                              **({'directed': False} if not e.get('directed', True) else {}))
                         for e in edges]
        # weight lookups for serialization: edge_id -> (cell, weight_index)
        self._ff_weight_ref = {}
        for cell in self.plastic:
            for widx, eid in enumerate(cell.ff_edge_ids):
                self._ff_weight_ref[eid] = (cell, widx)
        self._pred_weight_ref = {}
        for cell in self.pi:
            for widx, eid in enumerate(cell.pred_edge_ids):
                self._pred_weight_ref[eid] = (cell, widx)
        # basal_excitation weight lookup: edge_id -> (c_cell, basal_index=0). Serves
        # serialization and manual editing (clipped with the C-specific basal cap).
        self._basal_weight_ref = {}
        for cell in self.coincidence:
            self._basal_weight_ref[cell.basal.edge_ids[0]] = (cell, 0)
        # Source-indexed feedforward adjacency for the event path: a fired source
        # schedules charge (owned by each target) onto its plastic targets for t+1.
        self._ff_out = {}              # source_id -> [(target_id, widx, edge_id) ...]
        for cell in self.plastic:
            for widx, src in enumerate(cell.ff_src):
                self._ff_out.setdefault(src, []).append(
                    (cell.id, widx, cell.ff_edge_ids[widx]))
        self._latency_ids = {c.id for c in self.latency_competitors}

    # ============================================================== stepping
    def _begin_step(self):
        for nid in self.neurons:
            self.spiked[nid] = False
        for n in self.exc.values():
            n.spiked = False
        for n in self.inh.values():
            n.clear()
        for n in self.src.values():
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
        # Metadata-driven dispatch: an event-resolved graph runs the analytic
        # sub-boundary scheduler; every legacy graph keeps the byte-compatible
        # synchronous path below, unchanged.
        if self.event_resolved:
            return self._event_step()
        self.timestep += 1
        t = self.timestep
        self._begin_step()
        p = self.params

        # ---- subphase 1: gather scheduled arrivals (emitted at t-1, delay 1) ----
        exc_now = self._exc_next
        inh_now = self._inh_next
        self._ff_deliv_now = self._ff_deliv_next
        self._exc_next = {}
        self._inh_next = []
        self._ff_deliv_next = {}

        # inhibitory conductance arrivals (persistent; added before integration)
        for pulse in inh_now:
            self._deliver_inhibition(pulse)

        # excitatory internal arrivals
        for nid, q in exc_now.items():
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(q)

        # external input (delay 0) + manual continuous injections.
        # An e_sensory sink integrates its fixed afferent charge like any other input.
        # An rg_source sink does NOT integrate: its spike is exogenous and is simply
        # asserted for this boundary, which is why a held edge keeps producing RG spikes
        # on every input boundary no matter how hard L1 is being inhibited.
        input_arrives = (t % max(1, int(p['input_period'])) == 0)
        for n, pix in self._input_sinks:
            active = bool(input_arrives and pix is not None and self.input_vec[pix] > 0.5)
            if isinstance(n, SourceNeuron):
                n.present(active)
            elif active:
                n.gather_exc(n.acc_weights[0])
        for nid, mag in self._continuous.items():
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(mag)

        # ---- subphase 2: integrate every excitatory neuron once ----
        for n in self.exc.values():
            n.integrate()

        # ---- subphase 3: threshold test + fire ----
        # ``fired_ff`` collects every cell whose spike THIS boundary may drive a
        # feedforward synapse (rg_source, e_sensory, e_encoder). It is consumed once, in
        # subphase 5, and only ever schedules charge for t+1 -- so no spike can traverse
        # two feedforward hops within one boundary.
        fired_ff = set()
        fired_e = set()       # real excitatory spikes that may drive structural E outputs

        # rg_source: exogenous, already asserted in subphase 1. No membrane, no
        # threshold test, no refractory, nothing for inhibition to gate.
        for n in self.sources:
            if n.spiked:
                fired_ff.add(n.id)
                self.spiked[n.id] = True

        # e_sensory: every threshold crosser fires (sensory sources).
        for n in self.sensory:
            if n.can_fire():
                n.fire()
                fired_ff.add(n.id)
                fired_e.add(n.id)
                self.spiked[n.id] = True

        # e_encoder: plastic but NONCOMPETITIVE -- every threshold crosser fires and
        # learns its own delivered volley. There is no arbitration in L1: two encoders
        # crossing on the same boundary both fire.
        for n in self.encoders:
            if n.can_fire():
                n.fire()
                n.update_acc_weights(self._participation(n))
                self._emit_ff_weight_changes(n)
                fired_ff.add(n.id)
                fired_e.add(n.id)
                self.spiked[n.id] = True

        # e_residual: nonplastic and NONCOMPETITIVE. Every ErrorE threshold crosser
        # fires; its event is later broadcast to all structurally connected switches.
        for n in self.residuals:
            if n.can_fire():
                n.fire()
                fired_e.add(n.id)
                self.spiked[n.id] = True

        # e_competitor: deterministic single-winner WTA among the crossers (selection,
        # NOT charge removal). Highest membrane wins; tie-break = lowest node order.
        winner = None
        crossers = [(idx, n) for idx, n in enumerate(self.competitors) if n.can_fire()]
        if crossers:
            _, winner = max(crossers, key=lambda kv: (kv[1].V, -kv[0]))
            winner.fire()
            # Learn feedforward: participation = which of THIS competitor's afferents
            # were delivered to IT in the causal volley that arrived this boundary.
            winner.update_acc_weights(self._participation(winner))
            self._emit_ff_weight_changes(winner)
            self.spiked[winner.id] = True
            self.winner = winner.id
            fired_ff.add(winner.id)     # supports deeper valid feedforward graphs
            fired_e.add(winner.id)

        # ---- subphase 4: update local activity traces (survive reset) ----
        for n in self.exc.values():
            n.update_trace()

        # ---- subphase 5: emit spikes into delay-1 queues; run local relays/plasticity ----
        # Generic feedforward dispatch: every permitted source spike schedules weighted
        # charge onto each of its plastic targets for the NEXT boundary, and records
        # itself in that target's arrival volley. Works for any number of hops.
        if fired_ff:
            for tgt in self.plastic:
                q = 0.0
                delivered = set()
                for widx, src in enumerate(tgt.ff_src):
                    if src in fired_ff:
                        q += float(tgt.acc_weights[widx])
                        delivered.add(src)
                        self.emitted.append(tgt.ff_edge_ids[widx])
                if delivered:
                    self._ff_deliv_next[tgt.id] = delivered
                if q:
                    self._sched_exc(tgt.id, q)

        # Fixed structural E->ErrorE evidence copies. These are not learned weights:
        # each real source spike schedules one fixed supra-threshold charge pulse.
        if fired_e:
            q_fixed = float(p['residual_exc_scale']) * float(p['e_threshold'])
            for src in self.order:
                if src not in fired_e:
                    continue
                for tgt, eid in self._fixedexc_out.get(src, []):
                    self._sched_exc(tgt, q_fixed)
                    self.emitted.append(eid)

        # winner competitor drives its relay_excitation targets THIS boundary; each
        # relay that fires schedules its inhibitory / predictive conductance for t+1.
        if winner is not None:
            self._drive_relays(winner)

        # Switches gather every residual broadcast, evaluate it against eligibility
        # carried INTO this boundary, and only then record current L2 spikes as future
        # local traces. No global self.winner value is read by the switch mechanism.
        if self.switches:
            self._drive_switches(fired_e)

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

    # ============================================== event-resolved boundary step
    def _event_step(self) -> dict:
        """One outer boundary for an event-resolved graph: setup (deliver frozen
        arrivals, resolve C gates, freeze drive), then an analytic sub-boundary event
        loop (earliest crossing fires, drives zero-latency relays, applies immediate
        hard resets, recomputes), then finalization (trace, conductance, refractory)."""
        self.timestep += 1
        t = self.timestep
        p = self.params

        # ---- outer-boundary setup -------------------------------------------------
        self._begin_event_step()

        # rotate delay-1 buffers (arrivals emitted at t-1 land now).
        exc_now = self._exc_next
        inh_now = self._inh_next
        basal_now = self._basal_next
        apical_now = self._apical_next
        self._ff_deliv_now = self._ff_deliv_next
        self._exc_next, self._inh_next = {}, []
        self._basal_next, self._apical_next = {}, {}
        self._ff_deliv_next = {}

        for pulse in inh_now:                      # legacy conductance arrivals, if any
            self._deliver_inhibition(pulse)
        for nid, q in exc_now.items():             # feedforward + pretrained charge
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(q)
        for cid, events in basal_now.items():      # basal dendritic events
            c = self.exc.get(cid)
            if isinstance(c, CoincidencePyramidalNeuron):
                for src, sig in events:
                    c.gather_basal(src, sig)
        for cid, srcs in apical_now.items():       # apical dendritic events
            c = self.exc.get(cid)
            if isinstance(c, CoincidencePyramidalNeuron):
                for src in srcs:
                    c.gather_apical(src)

        # external input to RG sources (delay 0); a fired RG schedules its fixed
        # pretrained packet + any structural outputs for t+1 (it owns no membrane).
        input_arrives = (t % max(1, int(p['input_period'])) == 0)
        for n, pix in self._input_sinks:
            active = bool(input_arrives and pix is not None and self.input_vec[pix] > 0.5)
            if isinstance(n, SourceNeuron):
                n.present(active)
            elif active:
                n.gather_exc(n.acc_weights[0])
        for nid, mag in self._continuous.items():
            n = self.exc.get(nid)
            if n is not None:
                n.gather_exc(mag)
        for n in self.sources:
            if n.spiked:
                self.spiked[n.id] = True
                self._emit_event_outputs(n.id)

        # resolve every C dendritic gate, then freeze each membrane's gathered drive.
        for c in self.coincidence:
            c.resolve_dendrites()
        for n in self.exc.values():
            n.freeze_drive()

        # ---- sub-boundary event loop ----------------------------------------------
        membranes = [self.exc[nid] for nid in self.order if nid in self.exc]
        sched = BoundaryEventScheduler(membranes, p['crossing_time_tolerance'])
        while sched.current_tau < 1.0:
            cell, tau = sched.next_event()
            if cell is None:
                sched.advance_to_end()
                break
            sched.advance_all(tau)                 # advance ALL membranes to the event
            self._fire_event_cell(cell, tau)       # fire + learn + emit + relays + resets
        self.latency_ties = sched.ties

        # ---- boundary finalization ------------------------------------------------
        for n in self.exc.values():
            n.update_trace()
            n.decay_conductance()
            n.advance_refractory()
        for nid in self.neurons:
            self._spike_hist[nid].append(1 if self.spiked[nid] else 0)
        return self.dynamic_state()

    def _begin_event_step(self):
        """Clear per-boundary reporting + transient membrane state for the event path."""
        for nid in self.neurons:
            self.spiked[nid] = False
        for n in self.exc.values():
            n.begin_event_boundary()
        for n in self.inh.values():
            n.clear()
        for n in self.src.values():
            n.clear()
        self.changed_synapses = []
        self.emitted = []
        self.inhibitory_pulses = []
        self.hard_reset_events = []
        self.latency_ties = []
        self.winner = None

    def _fire_event_cell(self, cell, tau):
        """Fire the selected E/C cell at ``tau``, run its immediate learning, schedule
        its ordinary delay-1 outputs for t+1, then resolve its zero-latency relays and
        apply their immediate hard resets at the same ``tau``."""
        cell.fire(tau)
        self.spiked[cell.id] = True
        if isinstance(cell, CoincidencePyramidalNeuron):
            cell.update_basal_weight()               # causal firing-boundary gate state
            self.changed_synapses.append(
                dict(id=cell.basal.edge_ids[0], weight=round(float(cell.basal_weight), 6)))
        elif cell.id in self._latency_ids:
            cell.update_acc_weights(self._participation(cell))
            self._emit_ff_weight_changes(cell)
            if self.winner is None:                  # report-only first latency spike
                self.winner = cell.id
        self._emit_event_outputs(cell.id)
        self._drive_event_relays(cell.id, tau)

    def _emit_event_outputs(self, source_id):
        """Schedule a fired source's ordinary one-boundary-delay outputs for t+1:
        feedforward charge (owned by the target), fixed pretrained packets, and basal /
        apical dendritic events. These never arrive during the current event loop."""
        for tgt, widx, eid in self._ff_out.get(source_id, []):
            tgtcell = self.exc.get(tgt)
            if tgtcell is None:
                continue
            self._sched_exc(tgt, float(tgtcell.acc_weights[widx]))
            self._ff_deliv_next.setdefault(tgt, set()).add(source_id)
            self.emitted.append(eid)
        for tgt, eid in self._pretrained_out.get(source_id, []):
            self._sched_exc(tgt, float(self._q_pretrained))
            self.emitted.append(eid)
        for tgt, eid in self._basal_out.get(source_id, []):
            self._basal_next.setdefault(tgt, []).append((source_id, 1.0))
            self.emitted.append(eid)
        for tgt, eid in self._apical_out.get(source_id, []):
            self._apical_next.setdefault(tgt, []).append(source_id)
            self.emitted.append(eid)

    def _drive_event_relays(self, source_id, tau):
        """Resolve every stateless inhibitory relay driven by ``source_id`` at the same
        ``tau``, then apply each relay's outgoing hard resets immediately. A relay emits
        at most one spike per boundary; a second same-boundary input creates no burst and
        no second reset event."""
        for rid, re_eid in self._relayexc_out.get(source_id, []):
            relay = self.inh.get(rid)
            if relay is None or relay.spiked:        # already fired this boundary
                continue
            relay.receive()
            if not relay.resolve():
                continue
            self.spiked[rid] = True
            relay.spike_tau = tau                    # relay inherits its driver's tau
            self.emitted.append(re_eid)
            for tgt, hr_eid in self._hardreset_out.get(rid, []):
                tgtcell = self.exc.get(tgt)
                if tgtcell is None:
                    continue
                v_before = float(tgtcell.V)
                drive_before = float(tgtcell.remaining_excitation)
                tgtcell.hard_reset(tau)              # V <- rest, discard remaining drive
                self.hard_reset_events.append(dict(
                    source=rid, target=tgt, edge_id=hr_eid, kind='hard_reset',
                    outer_boundary=int(self.timestep), tau=round(float(tau), 12),
                    v_before=round(v_before, 6), drive_before=round(drive_before, 6)))
                self.emitted.append(hr_eid)

    # --------------------------------------------------------- emission helpers
    def _participation(self, cell):
        """Causal participation for ONE plastic target's update: for each of its own
        afferents, did that afferent's spike arrive at THIS target on THIS boundary?

        Read from the per-target arrival volley, so an update can never see a source
        that fired into a different target, nor one from the preceding/following
        boundary. Afferents that did not participate get the rule's -1 signal, but only
        inside the target that actually fired."""
        delivered = self._ff_deliv_now.get(cell.id, ())
        return np.array([src in delivered for src in cell.ff_src], dtype=bool)

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

    def _drive_switches(self, fired_e):
        """Resolve distributed residual/error gates without a global winner lookup.

        Phase A broadcasts each real ErrorE event along every explicit connection.
        Phase B resolves each SwitchI against only the decayed trace it carried into
        this boundary and, on coincidence, schedules paired inhibition for t+1.
        Phase C records current L2E spikes on the explicit paired trace edges for use
        on future boundaries. A switch that just fired consumes the trace and is not
        immediately re-primed by the same incumbent spike.
        """
        p = self.params

        # A. Gather every residual event before resolving any switch. Iterating the
        # stable node order makes emitted-edge order deterministic.
        for src in self.order:
            if src not in fired_e:
                continue
            for sid, eid in self._switch_residual_out.get(src, []):
                sw = self.inh.get(sid)
                if isinstance(sw, SwitchInterneuron):
                    sw.receive_residual()
                    self.emitted.append(eid)

        # B. Strict AND and paired output. The switch never reads self.winner.
        g = float(p['switch_g_scale'])
        for sw in self.switches:
            if not sw.resolve_residual():
                continue
            self.spiked[sw.id] = True
            if p['switch_conductance_enabled']:
                for tgt, eid in self._inh_out.get(sw.id, []):
                    self._sched_inh(tgt, g, sw.id, 'switch', weight=g)
                    self.emitted.append(eid)

        # C. A current competitor spike writes only its explicitly paired trace. If
        # that switch fired above, leave its consumed trace at zero this boundary.
        for src in self.order:
            if src not in fired_e:
                continue
            for sid, eid in self._trace_out.get(src, []):
                sw = self.inh.get(sid)
                if isinstance(sw, SwitchInterneuron):
                    self.emitted.append(eid)
                    if not sw.spiked:
                        sw.prime()

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
        if neuron_id in self.inh or neuron_id in self.src:
            return          # relays are event-driven; RG is driven only by its pixel
        if isinstance(self.exc.get(neuron_id), CoincidencePyramidalNeuron):
            raise ValueError(
                f'{neuron_id!r} is a coincidence cell; scalar stimulation would bypass '
                f'its basal/apical coincidence gate. Deliver explicit basal/apical '
                f'events instead.')
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
        ref = self._basal_weight_ref.get(edge_id)
        if ref is not None:
            cell, _ = ref
            w = float(np.clip(weight, 0.0, cell.w_max))     # C-specific basal cap
            cell.basal.weights[0] = w
            return w
        raise KeyError(edge_id)                             # apical/pretrained/reset: not editable

    def apply_config(self, overrides: dict):
        applied = []
        for k, v in (overrides or {}).items():
            if k not in EDITABLE_KEYS:
                continue
            if k == 'topology' and v not in VALID_TOPOLOGIES:
                raise ValueError(f'topology must be one of {VALID_TOPOLOGIES}, got {v!r}')
            if k == 'l2_init_total_frac' and not 0.0 < float(v) < 1.0:
                raise ValueError('l2_init_total_frac must satisfy 0 < rho < 1')
            if k == 'c_eta' and v is not None and float(v) < 0.0:
                raise ValueError('c_eta must be non-negative or None')
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
            # Both tags must survive the editor round-trip: 'pixel' is input ownership,
            # 'grid' is the display/receptive-field cell. Dropping 'grid' here would
            # silently unmap a saved rg graph's L1E cells from the RF view.
            if n.get('pixel') is not None:
                node['pixel'] = n['pixel']
            if n.get('grid') is not None:
                node['grid'] = n['grid']
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
        if kind == 'basal_excitation':
            ref = self._basal_weight_ref.get(eid)
            return None if ref is None else float(ref[0].basal_weight)
        if kind == 'pretrained_excitation':
            # Report the fixed delivered charge (a named fixed magnitude, not a learned
            # weight). Apical / hard-reset edges are unweighted -> None below.
            return float(self._q_pretrained) if self._q_pretrained else None
        return None                                       # structural relay / conductance gate

    def topology(self) -> dict:
        neurons = [dict(**self.meta[nid]) for nid in self.order]
        synapses = []
        for e in self.synapses:
            w = self._live_weight(e)
            synapses.append(dict(**e, weight=(None if w is None else round(w, 6))))
        # Layers in upstream->downstream order, restricted to those the active graph
        # actually has (so 'rg' reports the RG source layer and pi/old do not).
        layers = [L for L in ('RG', 'L1', 'ERR', 'L2')
                  if any(m['layer'] == L for m in self.meta.values())]
        layers += sorted({m['layer'] for m in self.meta.values()} - set(layers))
        return dict(neurons=neurons, synapses=synapses, layers=layers,
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
                   enc_plasticity_enabled=bool(p['enc_plasticity_enabled']),
                   enc_init_jitter=bool(p['enc_init_jitter']),
                   enc_w_init=(None if p['enc_w_init'] is None else float(p['enc_w_init'])),
                   residual_exc_scale=float(p['residual_exc_scale']),
                   switch_trace_decay=float(p['switch_trace_decay']),
                   switch_trace_threshold=float(p['switch_trace_threshold']),
                   switch_residual_charge_frac=float(p['switch_residual_charge_frac']),
                   switch_trace_charge_frac=float(p['switch_trace_charge_frac']),
                   switch_branch_cap_frac=float(p['switch_branch_cap_frac']),
                   switch_g_scale=float(p['switch_g_scale']),
                   switch_conductance_enabled=bool(p['switch_conductance_enabled']),
                   c_eta=(float(p['eta']) if p['c_eta'] is None else float(p['c_eta'])),
                   l2_init_total_frac=float(p['l2_init_total_frac']),
                   ff_init_mean=float(FF_INIT_MEAN),
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
            if isinstance(n, ConductanceLIFNeuron):
                # Generalized to the shared membrane base so a sibling coincidence cell
                # does not silently lose its conductance/trace/pre-reset state here.
                rec['g_inh'] = round(float(n.g_inh), 6)
                rec['trace'] = round(float(n.a), 6)
                rec['v_pre_reset'] = round(float(n.v_pre_reset), 4)
                # Optional, additive: an event-resolved E spike carries its sub-boundary
                # tau. Guarded on event_resolved so legacy dynamic payloads are unchanged.
                if self.event_resolved:
                    rec['spike_tau'] = (None if n.spike_tau is None
                                        else round(float(n.spike_tau), 9))
                if isinstance(n, CoincidencePyramidalNeuron):
                    rec['basal_weight'] = round(float(n.basal_weight), 6)
                    rec['basal_received'] = bool(n.basal_received)
                    rec['basal_eligible'] = bool(n.basal_eligible)
                    rec['apical_active'] = bool(n.apical_active)
                    rec['apical_sources'] = sorted(n.apical_sources)
                    rec['coincidence_active'] = bool(n.coincidence_active)
                    rec['coincidence_charge'] = round(float(n.coincidence_charge), 6)
            elif isinstance(n, SwitchInterneuron):
                rec['winner_trace'] = round(float(n.x), 6)
                rec['residual_received'] = bool(n.received_residual)
                rec['residual_events'] = int(n.residual_events)
                rec['residual_charge'] = round(float(n.residual_charge), 6)
                rec['trace_charge'] = round(float(n.trace_charge), 6)
                rec['v_pre_reset'] = round(float(n.v_pre_reset), 4)
            neurons.append(rec)
        return dict(timestep=self.timestep, running=False, neurons=neurons,
                    changed_synapses=self.changed_synapses,
                    emitted=self.emitted,
                    inhibitory_pulses=self.inhibitory_pulses,
                    # Additive event-resolved diagnostics: immediate hard resets are
                    # reported SEPARATELY from conductance pulses (never mislabeled), and
                    # tolerance ties are auditable. Empty on legacy graphs.
                    hard_reset_events=list(self.hard_reset_events),
                    latency_ties=list(self.latency_ties),
                    input=self.input_vec.astype(int).tolist(),
                    winner=self.winner,
                    stats=self.stats(), log=list(self.event_log)[-12:])

    def stats(self) -> dict:
        active = sum(1 for nid in self.neurons if abs(self.neurons[nid].potential) > 1e-3)
        firing = sum(1 for v in self.spiked.values() if v)
        rate = float(np.mean([self.firing_freq(nid) for nid in self.neurons]))
        return dict(total=len(self.neurons), active=active, firing=firing,
                    firing_rate=round(rate, 4), winner=self.winner)
