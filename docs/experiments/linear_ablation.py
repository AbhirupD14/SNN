"""Linear weight-update ablation study (prompts/Claude_Linear_Weight_Update_Ablation_Plan.md).

Tests removing the per-synapse ``1 - (w/w_max)^2`` multiplier from the accumulating
E/L2E rule and the C basal rule. After this study the ordinary E/L2E production default
is `linear_bounded` (the multiplier removed); C stays `c_quadratic_bounded`. Every mode
is headless. Conditions A (historical quadratic E) and B (promoted linear E) set their
modes explicitly, so this driver is independent of the engine default.

Phases (each callable; results are JSON + a Markdown report):
  0  baseline lock + determinism fingerprints
  2  isolated non-WTA trajectories (E) and isolated C cadence
  3  factorial rg_coincidence network ablation (conditions A-H)
  4  plus-sign composition-readiness probe

Run:  PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py <phase>
      (phase in {0,2,3,4,all})
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS
from snn.neurons import CoincidencePyramidalNeuron, ExcitatoryNeuron, leak_to_conductance

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linear_ablation')

# The plan's fixed experiment configuration (frozen scope).
FIXED_CONFIG = dict(topology='rg_coincidence', l2_init_total_frac=0.95, eta=0.01,
                    c_eta=0.001, leak_rate=0.0, refractory_steps=0, e_weight_cap=500.0)
SEEDS = list(range(1, 9))                 # seeds 1..8 for the Phase 0 baseline lock
DASHBOARD_SEED = 4083693835               # plan-specified dashboard seed


def _round(x, nd=6):
    return round(float(x), nd)


def _fingerprint(engine, steps, schedule=None):
    """Deterministic trace digest: per-checkpoint L2E/C weights + spike tallies."""
    frames = []
    plan = schedule or [('row 1', steps)]
    for name, n in plan:
        engine.set_pattern(name)
        for _ in range(n):
            engine.step()
            frames.append((
                tuple(_round(w) for c in engine.latency_competitors for w in c.acc_weights),
                tuple(_round(c.basal_weight) for c in engine.coincidence),
                sum(1 for c in engine.latency_competitors if c.spiked),
                sum(1 for c in engine.coincidence if c.spiked)))
    blob = json.dumps(frames, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ============================================================ Phase 0
def phase0_baseline(steps=400):
    """Lock deterministic baseline fingerprints (default modes, instrumentation off)."""
    results = {}
    for seed in SEEDS + [DASHBOARD_SEED]:
        e = SimulationEngine(seed=seed, **FIXED_CONFIG)
        results[str(seed)] = _fingerprint(e, steps)
    # determinism: a second run must match exactly.
    e2 = SimulationEngine(seed=1, **FIXED_CONFIG)
    deterministic = _fingerprint(e2, steps) == results['1']
    # instrumentation-off byte-identity: enabling record_updates must not change the trace.
    e3 = SimulationEngine(seed=1, **FIXED_CONFIG)
    for c in e3.latency_competitors:
        c.record_updates = True
    for c in e3.coincidence:
        c.record_updates = True
    instr_same = _fingerprint(e3, steps) == results['1']
    out = dict(steps=steps, fixed_config=FIXED_CONFIG,
               fingerprints=results, deterministic=deterministic,
               instrumentation_off_byte_identical=instr_same)
    _save('phase0_baseline.json', out)
    print(f"Phase 0: {len(results)} seed fingerprints; deterministic={deterministic}; "
          f"instrumentation-neutral={instr_same}")
    return out


def _save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), 'w') as f:
        json.dump(obj, f, indent=2)


# ============================================================ Phase 2 (isolated E)
CHECKPOINTS = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
THETA = 1000.0


def _iso_e(mode, participation_fn, n_events=10000, n_aff=9, w_max=500.0, eta=0.01,
           dist=None, seed=1):
    """One isolated non-WTA E cell fed repeated volleys; NO membrane/WTA -- each event is
    a single accumulating update. Returns trajectory + summary metrics per the plan."""
    dist = np.ones(n_aff) if dist is None else np.asarray(dist, float)
    w = np.full(n_aff, 0.95 * THETA / n_aff)                # identical init across modes
    n = ExcitatoryNeuron('E', 'iso', acc_weights=w.copy(), acc_distance_factor=dist,
                         eta=eta, w_max=w_max, threshold=THETA, learn=True, update_mode=mode)
    rng = np.random.default_rng(seed)
    neg_fe = overshoot_ct = cap_hits = floor_hits = 0
    max_overshoot = 0.0
    reach = {90: None, 95: None, 99: None}
    checks = {}
    nan = False
    active_mask_ref = participation_fn(0, rng) if callable(participation_fn) else participation_fn
    for ev in range(1, n_events + 1):
        part = participation_fn(ev, rng)
        fe = THETA - float(n.acc_weights.sum())
        if fe < 0:
            neg_fe += 1
        n.update_acc_weights(part)
        tot = float(n.acc_weights.sum())
        if tot > THETA:
            overshoot_ct += 1
            max_overshoot = max(max_overshoot, tot - THETA)
        if mode != 'linear_nonnegative':
            cap_hits += int((n.acc_weights >= w_max - 1e-9).sum())
        floor_hits += int((n.acc_weights <= 1e-12).sum())
        if not np.all(np.isfinite(n.acc_weights)):
            nan = True
            break
        active = float(n.acc_weights[np.asarray(active_mask_ref, bool)].sum())
        for pct in (90, 95, 99):
            if reach[pct] is None and active >= pct / 100.0 * THETA:
                reach[pct] = ev
        if ev in CHECKPOINTS:
            checks[ev] = dict(total=round(tot, 3), active=round(active, 3),
                              inactive=round(tot - active, 3), max_w=round(float(n.acc_weights.max()), 3))
    return dict(mode=mode, events=n_events, reach_pct=reach, checkpoints=checks,
                neg_fe_count=neg_fe, overshoot_count=overshoot_ct,
                max_overshoot=round(max_overshoot, 3),
                cap_hits=cap_hits, floor_hits=floor_hits, nan_or_inf=nan,
                final_total=round(float(n.acc_weights.sum()), 3),
                final_max_w=round(float(n.acc_weights.max()), 3),
                final_min_w=round(float(n.acc_weights.min()), 3))


def _fixed_part(active):
    mask = np.zeros(9, bool); mask[active] = True
    return lambda ev, rng: mask


def _alt_part(a, b):
    ma = np.zeros(9, bool); ma[a] = True
    mb = np.zeros(9, bool); mb[b] = True
    return lambda ev, rng: (ma if ev % 2 else mb)


def _noisy_part(active, p=0.10):
    base = np.zeros(9, bool); base[active] = True
    return lambda ev, rng: np.logical_xor(base, rng.random(9) < p)


def phase2_isolated(n_events=10000):
    e_patterns = {
        '1_of_9': _fixed_part([0]),
        '3_of_9': _fixed_part([0, 1, 2]),
        '5_of_9': _fixed_part([0, 1, 2, 3, 4]),
        'alt_disjoint_3': _alt_part([0, 1, 2], [3, 4, 5]),
        'noisy_3_10pct': _noisy_part([0, 1, 2], 0.10),
        'unequal_distance_3': _fixed_part([0, 1, 2]),
    }
    dists = {'unequal_distance_3': np.array([1.0, 0.6, 0.3, 1, 1, 1, 1, 1, 1])}
    e_out = {}
    for pat, fn in e_patterns.items():
        e_out[pat] = {m: _iso_e(m, fn, n_events=n_events, dist=dists.get(pat))
                      for m in ('quadratic_bounded', 'linear_bounded', 'linear_nonnegative')}
    c_out = phase2_isolated_c()
    out = dict(n_events=n_events, e=e_out, c=c_out)
    _save('phase2_isolated.json', out)
    _print_phase2(out)
    return out


# ---------------------------------------------------- Phase 2 (isolated C cadence)
def _iso_c(mode, n_coincidences=20000):
    """Drive one valid basal/apical coincidence per boundary through the real analytic
    resolver, over the FULL horizon (FE bounds cap-free C, so no runaway to clamp). The
    two-event cadence is validated over the ENTIRE trajectory (ideal fires[i] == i % 2),
    and every rejection condition is recorded as a first-occurrence event or ``None`` --
    so an OBSERVED invariant failure is distinguishable from a projected long-horizon
    trend. C dynamics are unchanged."""
    eng = SimulationEngine(seed=1, **FIXED_CONFIG)
    cpar = eng._resolve_coincidence_params()
    theta = float(eng.params['e_threshold'])
    w1 = theta                                               # impulse from-reset threshold
    c = CoincidencePyramidalNeuron('L1C', 'L1E', 'b0', apical_sources=['L2E'],
                                   apical_edge_ids=['a0'], basal_weight=cpar['c_init'],
                                   w_max=cpar['c_max'], eta_c=cpar['c_eta'], learn=True,
                                   update_mode=mode, threshold=theta,
                                   leak_rate=float(eng.params['leak_rate']),
                                   refractory_steps=int(eng.params['refractory_steps']))
    fires, weights = [], []
    first_one_from_rest = first_reach_w1 = first_exceed_2theta = first_nonfinite = None
    max_weight = cpar['c_init']
    for i in range(n_coincidences):
        from_rest = (c.V == c.v_rest)                        # state before this impulse
        c.begin_event_boundary(); c.gather_basal('L1E', 1.0); c.gather_apical('L2E')
        c.resolve_dendrites(); c.freeze_drive()
        dtau = c.crossing_time(1.0)
        fired = math.isfinite(dtau)
        if fired:
            c.advance_segment(dtau); c.fire(dtau); c.update_basal_weight()
            if from_rest and first_one_from_rest is None:
                first_one_from_rest = i                      # one coincidence fired C from rest
        else:
            c.advance_segment(1.0)
        c.update_trace(); c.decay_conductance(); c.advance_refractory()
        w = c.basal_weight
        fires.append(int(fired)); weights.append(round(w, 4))
        max_weight = max(max_weight, w)
        if first_reach_w1 is None and w >= w1:
            first_reach_w1 = i
        if first_exceed_2theta is None and w > 2 * theta:
            first_exceed_2theta = i
        if first_nonfinite is None and not math.isfinite(w):
            first_nonfinite = i

    # Full-trajectory cadence: ideal is strict alternation fires[i] == i % 2.
    mismatches = [i for i, f in enumerate(fires) if f != (i % 2)]
    finite = all(math.isfinite(x) for x in weights)
    stayed_below_w1 = (first_reach_w1 is None)
    ever_from_rest = (first_one_from_rest is not None)
    cadence_preserved = (len(mismatches) == 0)
    # An OBSERVED rejection condition (per the plan's C decision gate) actually occurred?
    rejection_observed = any(x is not None for x in
                             (first_one_from_rest, first_reach_w1, first_exceed_2theta,
                              first_nonfinite)) or not cadence_preserved
    ck = [0, 1, 10, 50, 100, 250, 1000, 2500, 5000, 10000, 19999]
    return dict(mode=mode, events=len(weights), w1=round(w1, 3),
                c_init=round(cpar['c_init'], 3), c_max=round(cpar['c_max'], 3),
                fire_pattern_first16=fires[:16],
                two_event_cadence_ok_full=cadence_preserved,
                cadence_mismatch_count=len(mismatches),
                first_cadence_mismatch=(mismatches[0] if mismatches else None),
                first_one_from_rest=first_one_from_rest, first_reach_w1=first_reach_w1,
                first_exceed_2theta=first_exceed_2theta, first_nonfinite=first_nonfinite,
                min_safety_margin=round(w1 - max_weight, 4),   # w1 - basal_weight, min over run
                final_weight=weights[-1] if weights else None,
                max_weight=round(max_weight, 4),
                finite=finite, stayed_below_w1=stayed_below_w1,
                ever_fired_from_rest=ever_from_rest,
                rejection_condition_observed=rejection_observed,
                weight_checkpoints={k: weights[k] for k in ck if k < len(weights)})


def phase2_isolated_c():
    return {m: _iso_c(m) for m in
            ('c_quadratic_bounded', 'c_linear_bounded', 'c_linear_nonnegative')}


# ============================================================ Phase 3 (factorial)
CONDITIONS = {
    'A': ('quadratic_bounded', 'c_quadratic_bounded'),      # HISTORICAL quadratic-E baseline
    'B': ('linear_bounded', 'c_quadratic_bounded'),         # PROMOTED PRODUCTION rule (linear-bounded E)
    'C': ('quadratic_bounded', 'c_linear_bounded'),         # isolate C soft term
    'D': ('linear_bounded', 'c_linear_bounded'),            # remove both soft terms
    'E': ('linear_nonnegative', 'c_quadratic_bounded'),     # cap-free E, safe C
    'F': ('linear_nonnegative', 'c_linear_bounded'),        # candidate linear E + bounded C
    'G': ('quadratic_bounded', 'c_linear_nonnegative'),     # C safety-failure probe
    'H': ('linear_nonnegative', 'c_linear_nonnegative'),    # combined safety probe
}
ROW_PIX, COL_PIX = [3, 4, 5], [1, 4, 7]
SCHEDULE = [('row 1', 2500), ('col 1', 2500), ('row 1', 2500)]


def _owner(engine, pix):
    best, bv = None, -1.0
    for c in engine.latency_competitors:
        v = sum(c.acc_weights[c.ff_src.index(f'L1E{i}')] for i in pix)
        if v > bv:
            bv, best = v, c.id
    return best


def _phase3_run(cond, seed):
    em, cm = CONDITIONS[cond]
    e = SimulationEngine(seed=seed, e_weight_update_mode=em, c_weight_update_mode=cm,
                         **FIXED_CONFIG)
    theta = float(e.params['e_threshold'])
    w1 = theta                                               # C impulse from-reset threshold
    for c in e.latency_competitors:
        c.record_updates = True
    init_totals = {c.id: round(float(c.acc_weights.sum()), 3) for c in e.latency_competitors}
    winners, phase_bounds, b = [], [], 0
    rg = l1e = c_spk = c_coinc = l1_reset = l2_ties = 0
    c_fired_from_rest = 0
    max_c_weight_run = 0.0
    abort = None
    for pi, (name, n) in enumerate(SCHEDULE):
        e.set_pattern(name)
        start = b
        for _ in range(n):
            e.step(); b += 1
            w = [c.id for c in e.latency_competitors if c.spiked]
            winners.append(w[0] if w else None)
            rg += sum(1 for i in ROW_PIX + COL_PIX if e.spiked.get(f'RG{i}'))
            l1e += sum(1 for i in ROW_PIX + COL_PIX if e.spiked.get(f'L1E{i}'))
            for c in e.coincidence:
                if c.coincidence_active:
                    c_coinc += 1
                if c.spiked:
                    c_spk += 1
                    if c.basal_weight >= theta:              # one deposit alone would fire
                        c_fired_from_rest += 1
            max_c_weight_run = max(max_c_weight_run,
                                   max(c.basal_weight for c in e.coincidence))
            l1_reset += sum(1 for h in e.hard_reset_events if h['source'].startswith('L1I'))
            l2_ties += len(e.latency_ties)
            if cm == 'c_linear_nonnegative':
                mx = max(c.basal_weight for c in e.coincidence)
                if mx >= w1 or mx > 2 * theta or not math.isfinite(mx):
                    abort = f'C weight {mx:.1f} hit safety bound (w1={w1:.0f})'
                    break
        phase_bounds.append((start, b))
        if abort:
            break

    # owners + dominance in the last 500 of each completed phase
    def dom(lo, hi):
        seg = [x for x in winners[lo:hi] if x]
        if not seg:
            return None, 0.0
        from collections import Counter
        top, ct = Counter(seg).most_common(1)[0]
        return top, round(ct / len(seg), 3)
    owners = {}
    labels = ['row', 'col', 'return_row']
    for lbl, (lo, hi) in zip(labels, phase_bounds):
        owners[lbl] = dom(max(lo, hi - 500), hi)

    # E weight stats from update logs
    neg_fe = overshoot = cap_hits = above_cap = floor_hits = 0
    most_neg_fe = 0.0
    max_w = overshoot_max = 0.0
    for c in e.latency_competitors:
        for r in c.update_log:
            if r['fe_post'] < 0:
                neg_fe += 1
                most_neg_fe = min(most_neg_fe, r['fe_post'])
            if r['post_sum'] > theta:
                overshoot += 1
                overshoot_max = max(overshoot_max, r['post_sum'] - theta)
            cap_hits += r['n_at_max']
            floor_hits += r['n_zero']
            max_w = max(max_w, r['max_w'])
            if em == 'linear_nonnegative' and r['max_w'] > float(e.params['e_weight_cap']):
                above_cap += 1
    final_totals = {c.id: round(float(c.acc_weights.sum()), 3) for c in e.latency_competitors}

    # turnover timing in the column phase
    first_rival = last_incumbent = None
    if len(phase_bounds) >= 2:
        clo, chi = phase_bounds[1]
        row_owner = owners['row'][0]
        for i in range(clo, chi):
            if winners[i] and winners[i] != row_owner and first_rival is None:
                first_rival = i - clo
            if winners[i] == row_owner:
                last_incumbent = i - clo

    # complete per-phase and final-500 winner counts
    from collections import Counter
    phase_wc, final500_wc = [], []
    for (lo, hi) in phase_bounds:
        phase_wc.append(dict(Counter(x for x in winners[lo:hi] if x)))
        final500_wc.append(dict(Counter(x for x in winners[max(lo, hi - 500):hi] if x)))

    return dict(cond=cond, seed=seed, e_mode=em, c_mode=cm, abort=abort,
                owners=owners, first_rival_col=first_rival, last_incumbent_col=last_incumbent,
                phase_winner_counts=phase_wc, final500_winner_counts=final500_wc,
                init_totals=init_totals, final_totals=final_totals,
                init_total_mean=round(float(np.mean(list(init_totals.values()))), 2),
                final_totals_sample=dict(list(final_totals.items())[:3]),
                overshoot_max=round(overshoot_max, 4), floor_hits=floor_hits,
                min_c_safety_margin=round(w1 - max_c_weight_run, 3), w1=round(w1, 2),
                max_individual_weight=round(max_w, 2),
                neg_fe_count=neg_fe, most_neg_fe=round(most_neg_fe, 3),
                overshoot_count=overshoot, cap_hits=cap_hits, weights_above_cap=above_cap,
                c_spikes=c_spk, c_coincidences=c_coinc, l1_resets=l1_reset,
                c_fired_from_rest=c_fired_from_rest,
                max_c_weight=round(max(c.basal_weight for c in e.coincidence), 3),
                l1e_over_rg=round(l1e / rg, 4) if rg else None, l2_ties=l2_ties)


# --- 32 fresh-seed confirmation of A vs B (the linear_bounded-E production candidate) ---
# Deliberately disjoint from the exploratory seeds (1-16) and the dashboard seed
# 4083693835. Recorded explicitly so the confirmation is reproducible.
CONFIRM_SEEDS = list(range(2001, 2033))            # 32 fresh deterministic seeds


def _apply_gates(r):
    """Original success gates, applied explicitly per seed."""
    row = r['owners'].get('row', (None, 0.0))
    col = r['owners'].get('col', (None, 0.0))
    ret = r['owners'].get('return_row', (None, 0.0))
    gates = dict(
        row_dominance_ge_090=(row[1] >= 0.90),
        col_dominance_ge_080=(col[1] >= 0.80),
        return_dominance_ge_080=(ret[1] >= 0.80),
        col_owner_differs_from_row=(col[0] is not None and col[0] != row[0]),
        return_owner_equals_row=(ret[0] is not None and ret[0] == row[0]),
    )
    return gates, all(gates.values())


def confirm_ab(conds=('A', 'B'), seeds=None):
    """32-fresh-seed confirmation for the recommended linear-E change. Stores every
    plan-required per-seed metric and applies the original success gates explicitly."""
    seeds = list(seeds if seeds is not None else CONFIRM_SEEDS)
    stale = (set(seeds) & set(range(1, 17))) | (set(seeds) & {DASHBOARD_SEED})
    assert not stale, f'must use fresh seeds, not {sorted(stale)}'
    replay = _phase3_replay_ok('B', seeds[0])       # determinism on a fresh seed
    per_cond = {}
    runs = []
    for cond in conds:
        succ = 0
        seed_records = []
        for s in seeds:
            r = _phase3_run(cond, s)
            gates, ok = _apply_gates(r)
            r['gates'] = gates
            r['all_gates_pass'] = ok
            succ += int(ok)
            seed_records.append(r)
            runs.append(r)
        per_cond[cond] = dict(
            success_count=succ, n=len(seeds), success_rate=round(succ / len(seeds), 3),
            mean_row_dom=round(float(np.mean([r['owners']['row'][1] for r in seed_records])), 3),
            mean_col_dom=round(float(np.mean([r['owners'].get('col', (None, 0))[1] for r in seed_records])), 3),
            mean_ret_dom=round(float(np.mean([r['owners'].get('return_row', (None, 0))[1] for r in seed_records])), 3),
            col_turnover_rate=round(float(np.mean([1.0 if r['owners'].get('col', (None,))[0] != r['owners']['row'][0] else 0.0 for r in seed_records])), 3),
            return_recovery_rate=round(float(np.mean([1.0 if r['owners'].get('return_row', (None,))[0] == r['owners']['row'][0] else 0.0 for r in seed_records])), 3),
            max_individual_weight=round(float(np.max([r['max_individual_weight'] for r in seed_records])), 2),
            total_neg_fe=int(sum(r['neg_fe_count'] for r in seed_records)),
            total_overshoot=int(sum(r['overshoot_count'] for r in seed_records)),
            max_overshoot=round(float(np.max([r['overshoot_max'] for r in seed_records])), 4),
            total_cap_hits=int(sum(r['cap_hits'] for r in seed_records)),
            total_c_from_rest=int(sum(r['c_fired_from_rest'] for r in seed_records)),
            aborts=int(sum(1 for r in seed_records if r['abort'])))
    out = dict(seeds=seeds, schedule=SCHEDULE, fixed_config=FIXED_CONFIG,
               deterministic_replay=replay, per_cond=per_cond, runs=runs)
    _save('confirm_ab_32seeds.json', out)
    print(f"\n=== 32 fresh-seed confirmation (seeds {seeds[0]}..{seeds[-1]}); "
          f"replay_ok={replay} ===")
    for cond in conds:
        p = per_cond[cond]
        print(f"  {cond} ({CONDITIONS[cond][0]}/{CONDITIONS[cond][1]}):")
        print(f"     ALL GATES PASS: {p['success_count']}/{p['n']}  "
              f"(row_dom>={0.90}, col_dom>={0.80}, ret_dom>={0.80}, col!=row, ret==row)")
        print(f"     mean dom row/col/ret={p['mean_row_dom']}/{p['mean_col_dom']}/{p['mean_ret_dom']}  "
              f"col_turnover={p['col_turnover_rate']} return_recovery={p['return_recovery_rate']}")
        print(f"     stability: maxW={p['max_individual_weight']} negFE={p['total_neg_fe']} "
              f"overshoot={p['total_overshoot']}(max {p['max_overshoot']}) capHits={p['total_cap_hits']} "
              f"C_from_rest={p['total_c_from_rest']} aborts={p['aborts']}")
    return out


def _phase3_replay_ok(cond, seed):
    def fp(_seed):
        r = _phase3_run(cond, _seed)
        return (r['owners'], r['final_totals_sample'], r['c_spikes'], r['l1e_over_rg'])
    return fp(seed) == fp(seed)


def phase3_factorial(seeds=range(1, 17), dashboard_seed=DASHBOARD_SEED):
    runs = []
    for cond in CONDITIONS:
        for seed in list(seeds) + [dashboard_seed]:
            runs.append(_phase3_run(cond, seed))
    replay = _phase3_replay_ok('A', 1)
    out = dict(schedule=SCHEDULE, conditions=CONDITIONS, deterministic_replay=replay,
               runs=runs)
    _save('phase3_factorial.json', out)
    _print_phase3(out)
    return out


def _agg(runs, cond, key):
    vals = [r[key] for r in runs if r['cond'] == cond and r[key] is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def _print_phase3(out):
    runs = out['runs']
    print(f"\n=== Phase 3: factorial network (seeds mean); replay_ok={out['deterministic_replay']} ===")
    print("cond E-mode/C-mode          row_dom col_dom ret_dom  maxW  negFE overshoot capHits >cap  Cfromrest maxCw  L1E/RG  aborts")
    for cond, (em, cm) in CONDITIONS.items():
        cr = [r for r in runs if r['cond'] == cond]
        aborts = sum(1 for r in cr if r['abort'])
        rd = round(float(np.mean([r['owners']['row'][1] for r in cr if 'row' in r['owners']])), 2)
        cd = [r['owners'].get('col', (None, 0))[1] for r in cr]
        cd = round(float(np.mean(cd)), 2)
        rr = [r['owners'].get('return_row', (None, 0))[1] for r in cr]
        rr = round(float(np.mean(rr)), 2)
        print(f"{cond} {em[:12]:12}/{cm[2:14]:12} {rd:6} {cd:6} {rr:6} "
              f"{_agg(cr,cond,'max_individual_weight'):6} {_agg(cr,cond,'neg_fe_count'):5} "
              f"{_agg(cr,cond,'overshoot_count'):8} {_agg(cr,cond,'cap_hits'):6} "
              f"{_agg(cr,cond,'weights_above_cap'):5} {sum(r['c_fired_from_rest'] for r in cr):8} "
              f"{_agg(cr,cond,'max_c_weight'):6} {_agg(cr,cond,'l1e_over_rg'):6}  {aborts}")


def _print_phase2(out):
    print("\n=== Phase 2: isolated E convergence (events to X% of theta active sum) ===")
    for pat, modes in out['e'].items():
        print(f"  {pat}:")
        for m, r in modes.items():
            print(f"    {m:20s} reach90/95/99={r['reach_pct'][90]}/{r['reach_pct'][95]}/"
                  f"{r['reach_pct'][99]}  neg_fe={r['neg_fe_count']} overshoot_ct={r['overshoot_count']} "
                  f"max_over={r['max_overshoot']} max_w={r['final_max_w']} nan={r['nan_or_inf']}")
    print("\n=== Phase 2: isolated C cadence (full trajectory, 20k events) ===")
    for m, r in out['c'].items():
        print(f"  {m:22s} cadence_full={r['two_event_cadence_ok_full']} "
              f"(mismatches={r['cadence_mismatch_count']}, first@{r['first_cadence_mismatch']}) "
              f"max_w={r['max_weight']} min_margin(w1-w)={r['min_safety_margin']} "
              f"first_from_rest={r['first_one_from_rest']} first_reach_w1={r['first_reach_w1']} "
              f"REJECTION_OBSERVED={r['rejection_condition_observed']}")


# ============================================================ Phase 4 (composition)
PLUS_PIX = sorted(set(ROW_PIX) | set(COL_PIX))              # {1,3,4,5,7} = plus sign


def _freeze_learning(e):
    """Disable ALL plasticity on the engine so a diagnostic probe cannot change learned
    state: E/L2E accumulating weights and C basal weights. Returns a byte-snapshot of
    every plastic array so the caller can assert invariance afterward."""
    for c in e.latency_competitors:
        c.learn = False
    for c in e.coincidence:
        c.learn = False
    snap = {c.id: c.acc_weights.copy() for c in e.latency_competitors}
    snap.update({f'{c.id}#basal': c.basal.weights.copy() for c in e.coincidence})
    return snap


def _weights_unchanged(e, snap):
    for c in e.latency_competitors:
        if not np.array_equal(c.acc_weights, snap[c.id]):
            return False
    for c in e.coincidence:
        if not np.array_equal(c.basal.weights, snap[f'{c.id}#basal']):
            return False
    return True


def _phase4_probe(e, washout):
    """Present the plus union and capture each L2E's frozen boundary-start state at the
    FIRST boundary on which plus L2 drive is actually delivered (frozen_excitation > 0) --
    i.e. before the first winner's reset. Learning must already be frozen by the caller.

    ``washout`` = controlled-state probe: each L2E membrane ``V`` and inhibitory
    conductance ``g_inh`` are zeroed before every boundary, equalizing transient membrane
    state while learned weights are preserved (and asserted unchanged by the caller). WTA
    is unchanged; no diagnostic co-firing. Returns (snapshot, boundary_index) or ({}, None)."""
    e.set_input([1 if i in PLUS_PIX else 0 for i in range(9)])
    e._crossing_capture = []
    chosen, chosen_at = {}, None
    for step_i in range(14):
        if washout:
            for c in e.latency_competitors:
                c.V = 0.0
                c.g_inh = 0.0
        e.step()
        last = e._crossing_capture[-1]
        if any(v['frozen_excitation'] > 0 for v in last.values()):
            chosen, chosen_at = last, step_i             # FIRST plus-driven boundary; stop
            break
    e._crossing_capture = None
    return chosen, chosen_at


def _owner_crossing(snap, cid, active_wsum):
    d = snap.get(cid, {})
    v0 = d.get('v_before_drive', 0.0)
    frozen = d.get('frozen_excitation', 0.0)
    # NOTE: reconstructed projection, not a captured voltage; leak=0 and ignores g_inh.
    projected_end_v = round(v0 + frozen, 4)
    return dict(owner=cid, v_before_drive=round(v0, 4), frozen_excitation=round(frozen, 4),
                g_inh=d.get('g_inh', 0.0), projected_uninhibited_end_v=projected_end_v,
                active_input_wsum=round(active_wsum, 3), refractory=d.get('refractory', 0),
                tau=d.get('tau'), finite=d.get('finite', False))


def _phase4_run(cond, seed, train=2500):
    """Learn row 1 then col 1, then run TWO composition probes on the plus-sign union with
    LEARNING DISABLED: a carried-state probe (real recent membrane) and a controlled-state
    probe (diagnostic membrane washout). Row-owner and column-owner crossing predictions
    are reported separately. WTA is unchanged; no learned weight changes during a probe."""
    em, cm = CONDITIONS[cond]

    def train_engine():
        e = SimulationEngine(seed=seed, e_weight_update_mode=em, c_weight_update_mode=cm,
                             **FIXED_CONFIG)
        for name in ('row 1', 'col 1'):
            e.set_pattern(name)
            for _ in range(train):
                e.step()
        return e

    e = train_engine()
    row_owner, col_owner = _owner(e, ROW_PIX), _owner(e, COL_PIX)

    def wsum(cell, pix):
        return float(sum(cell.acc_weights[cell.ff_src.index(f'L1E{i}')] for i in pix))
    row_wsum = wsum(e.exc[row_owner], PLUS_PIX)
    col_wsum = wsum(e.exc[col_owner], PLUS_PIX)

    def probe(washout):
        eng = train_engine()                               # fresh trained copy per probe
        wsnap = _freeze_learning(eng)                      # disable plasticity + snapshot
        snap, at = _phase4_probe(eng, washout)
        weights_unchanged = _weights_unchanged(eng, wsnap)
        assert weights_unchanged, 'composition probe changed learned weights'
        r = _owner_crossing(snap, row_owner, row_wsum)
        c = _owner_crossing(snap, col_owner, col_wsum)
        n_finite = sum(1 for v in snap.values() if v.get('finite'))
        gap = (None if not (r['finite'] and c['finite'])
               else round(abs(r['tau'] - c['tau']), 6))
        earlier = ('row' if (r['finite'] and (not c['finite'] or r['tau'] < c['tau']))
                   else 'col' if c['finite'] else 'neither')
        return dict(row=r, col=c, n_finite_crossings=n_finite, tau_gap=gap, earlier=earlier,
                    captured_boundary=at, learned_weights_unchanged=weights_unchanged)

    return dict(cond=cond, seed=seed, row_owner=row_owner, col_owner=col_owner,
                distinct_owners=(row_owner != col_owner),
                row_plus_wsum=round(row_wsum, 2), col_plus_wsum=round(col_wsum, 2),
                carried=probe(washout=False), controlled=probe(washout=True))


def phase4_composition(conds=('A', 'B', 'D', 'F'), seeds=range(1, 9)):
    runs = [_phase4_run(cond, s) for cond in conds for s in seeds]
    out = dict(plus_pixels=PLUS_PIX,
               washout_note=('learning is DISABLED during both probes (E/L2E + C weights '
                             'frozen and asserted unchanged); the controlled probe also '
                             'zeros each L2E V and g_inh before every plus boundary; both '
                             'capture the FIRST plus-driven boundary; production untouched'),
               runs=runs)
    _save('phase4_composition.json', out)
    all_frozen = all(r[p]['learned_weights_unchanged'] for r in runs for p in ('carried', 'controlled'))
    print("\n=== Phase 4: composition readiness (carried-state vs controlled-state) ===")
    print(f"  learned weights unchanged during every probe: {all_frozen}")
    for cond in conds:
        cr = [r for r in runs if r['cond'] == cond]
        distinct = sum(r['distinct_owners'] for r in cr)
        for probe in ('carried', 'controlled'):
            both = sum(1 for r in cr if r[probe]['row']['finite'] and r[probe]['col']['finite'])
            row_fin = sum(1 for r in cr if r[probe]['row']['finite'])
            col_fin = sum(1 for r in cr if r[probe]['col']['finite'])
            gaps = [r[probe]['tau_gap'] for r in cr if r[probe]['tau_gap'] is not None]
            mg = round(float(np.mean(gaps)), 5) if gaps else None
            print(f"  {cond} [{probe:10}]: distinct={distinct}/{len(cr)} "
                  f"row_finite={row_fin} col_finite={col_fin} both_finite={both} mean_gap={mg}")
    return out
    return out


def main(argv):
    phase = argv[0] if argv else 'all'
    if phase in ('0', 'all'):
        phase0_baseline()
    if phase in ('2', 'all'):
        phase2_isolated()
    if phase in ('3', 'all'):
        phase3_factorial()
    if phase in ('confirm', 'all'):
        confirm_ab()
    if phase in ('4', 'all'):
        phase4_composition()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
