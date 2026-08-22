"""Dual FE/FES self-regulating direct 3x3 cortical-column experiment.

Causally tests whether the direct four-neuron column (``rg_direct_cc4``: a 3x3 RGC surface
densely feeding four ordinary latency-E competitors + one central WTA I) can assign the
four overlapping 3x3 patterns (row 1, col 1, diag \\, diag /, all sharing the center pixel)
one-to-one through the experimental dual node/synapse free-energy rule -- WITHOUT any
feature relay, coincidence feedback, predictive inhibition, Eor, hierarchy, or input-
frequency halving. The only inhibition is the ordinary within-column WTA I.

This is an implementation-and-execution script: it RUNS the reference experiment, the
flag-off same-topology control, a seed sweep, a long-dwell stress test, the required 16
(B, m) grid, and an isolated coincidence-cell causal probe, and writes durable artifacts
(config, summary, phase/window CSV, per-learning-event FE/FES trace, final weights, source
delivery + WTA counts, and a dashboard-compatible replay) beneath the gitignored
``experiments/runs/dual_fe_cc4/<run-id>/``, with an explicit PASS/FAIL per acceptance
condition.

Reference (seed 1): dual FE/FES ON, e = wte = 0.001, B = 5.0, eta = 0.01, c_eta = 0.005,
leak_rate = 0, refractory_steps = 0. The LR multiplier ``m`` (eta = 0.01*m, c_eta =
0.005*m) is an EXPERIMENT convention for the sensitivity study, never a dashboard control;
e = wte = 0.001 stay fixed throughout.

Run: ``PYTHONPATH=. .venv/bin/python experiments/dual_fe_cc4_consolidation.py``
     (add ``--quick`` for a fast smoke run, ``--full`` for the complete study).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine, PATTERNS                    # noqa: E402
from snn.neurons import (CoincidencePyramidalNeuron, dual_fe, dual_fes,      # noqa: E402
                         E_THRESHOLD)
from experiments.replay_recorder import (                                    # noqa: E402
    ReplayRecorder, STATUS_COMPLETED, FEEDBACK_NOT_APPLICABLE,
)

CANONICAL_ORDER = ('row 1', 'col 1', 'diag \\', 'diag /')
CENTER_PIXEL = 4
THETA = E_THRESHOLD
# Reference parameters (fixed; never overwritten by a sweep result).
REF = dict(e=0.001, wte=0.001, B=5.0, eta=0.01, c_eta=0.005, leak_rate=0.0,
           refractory_steps=0)
# Required LR x B sensitivity grid (seed 1). e = wte = 0.001 fixed throughout.
GRID_B = (1.0, 5.0, 20.0, 50.0)
GRID_M = (1, 10, 100, 500)


def _active_pixels(pattern):
    return [i for i, v in enumerate(PATTERNS[pattern]) if v > 0.5]


def build_engine(seed, *, dual, B=REF['B'], m=1, e=REF['e'], wte=REF['wte'],
                 leak_rate=REF['leak_rate'], refractory_steps=REF['refractory_steps']):
    """The same-topology engine for both flag states. ``m`` scales the LRs (eta = 0.01*m,
    c_eta = 0.005*m); e/wte/B are the dual parameters (ignored when dual is off)."""
    return SimulationEngine(
        seed=int(seed), topology='rg_direct_cc4', leak_rate=leak_rate,
        refractory_steps=refractory_steps, dual_fe_fes=bool(dual),
        eta=REF['eta'] * m, c_eta=REF['c_eta'] * m,
        dual_fe_e=e, dual_fe_wte=wte, dual_fe_B=B)


def _weight_matrix(engine):
    """competitor id -> its nine feedforward weights indexed by pixel."""
    out = {}
    for c in engine.latency_competitors:
        idx = {int(s[3:]): i for i, s in enumerate(c.ff_src)}          # RGC<pixel> -> weight index
        out[c.id] = [round(float(c.acc_weights[idx[p]]), 4) for p in range(engine.n_pix)]
    return out


def _active_total(engine, comp, pattern):
    idx = {int(s[3:]): i for i, s in enumerate(comp.ff_src)}
    return float(sum(comp.acc_weights[idx[p]] for p in _active_pixels(pattern)))


def run_phase(engine, pattern, dwell, *, prev_owner, final_window, dominance,
              collect_events=False, recorder=None):
    """Present one pattern for a fixed dwell; never stop early. Returns the phase metrics
    plus the per-competitor causal evidence (weights, FE, update magnitudes, RGC/WTA
    counts, firing counts, first-spike tau)."""
    engine.set_pattern(pattern)
    active = _active_pixels(pattern)
    novel = [p for p in active if p != CENTER_PIXEL]
    comp_ids = [c.id for c in engine.latency_competitors]
    if collect_events:
        for c in engine.latency_competitors:
            c.record_updates = True
            c.update_log.clear()

    weights_pre = _weight_matrix(engine)
    win_all = Counter()
    win_windows = []                        # dominance over consecutive equal windows
    rgc_fires = {p: 0 for p in active}
    comp_fires = Counter()
    first_tau = {}
    hard_resets = 0
    fe_by_comp = defaultdict(list)
    absdw_by_comp = defaultdict(float)
    nfire_by_comp = Counter()
    events = []

    n_windows = 4
    win_len = max(1, dwell // n_windows)
    win_counter = Counter()
    for step in range(1, dwell + 1):
        engine.step()
        w = engine.winner
        if w is not None:
            win_all[w] += 1
            win_counter[w] += 1
            comp_fires[w] += 1
            if w not in first_tau:
                cell = engine.neurons[w]
                first_tau[w] = (None if cell.spike_tau is None else round(float(cell.spike_tau), 6))
        for p in active:
            if engine.spiked[f'RGC{p}']:
                rgc_fires[p] += 1
        hard_resets += len(engine.hard_reset_events)
        # drain learning events (FE + applied dw) from each competitor
        for c in engine.latency_competitors:
            for ev in c.update_log:
                fe_by_comp[c.id].append(ev['fe'])
                absdw_by_comp[c.id] += float(np.sum(np.abs(ev['applied_dw'])))
                nfire_by_comp[c.id] += 1
                if collect_events:
                    events.append(dict(phase=pattern, cell=c.id, iaccq=round(ev['iaccq'], 4),
                                       theta=ev['theta'], fe=round(ev['fe'], 6),
                                       lr=ev['lr'], pre_sum=round(ev['pre_sum'], 4),
                                       post_sum=round(ev['post_sum'], 4),
                                       max_w=round(ev['max_w'], 4), min_w=round(ev['min_w'], 4),
                                       spike_tau=ev['spike_tau'], nonfinite=ev['nonfinite']))
            c.update_log.clear()
        if step % win_len == 0:
            o, cnt = win_counter.most_common(1)[0] if win_counter else (None, 0)
            tot = sum(win_counter.values())
            win_windows.append((o, round(cnt / tot, 4) if tot else 0.0))
            win_counter = Counter()
        if recorder is not None:
            recorder.record_frame(engine)
            recorder.metrics.append_row({
                'timestep': engine.timestep, 'phase': pattern,
                'winner': engine.winner or '', 'firing': int(engine.dynamic_state()['stats']['firing'])})

    weights_post = _weight_matrix(engine)
    owner, owner_wins = win_all.most_common(1)[0] if win_all else (None, 0)
    total = sum(win_all.values())
    # dominance = fraction over the FINAL window(s)
    final_windows = win_windows[-max(1, final_window):]
    final_dom = min((d for (o, d) in final_windows if o == owner), default=0.0) if owner else 0.0

    return dict(
        pattern=pattern, active_pixels=active, novel_pixels=novel, prev_owner=prev_owner,
        owner=owner, owner_wins=owner_wins, total_wins=total,
        dominance=round(owner_wins / total, 4) if total else 0.0,
        final_window_dominance=round(final_dom, 4),
        consolidated=bool(owner is not None and final_dom >= dominance),
        turnover_from_prev=bool(prev_owner is not None and owner != prev_owner),
        winner_counts=dict(win_all), window_dominance=win_windows,
        rgc_fires={str(p): rgc_fires[p] for p in active},
        rgc_equal_frequency=bool(len(set(rgc_fires.values())) == 1),
        competitor_fires=dict(comp_fires), hard_reset_events=hard_resets,
        first_spike_tau={k: first_tau[k] for k in first_tau},
        # causal evidence: mean FE + total |dw| per competitor (incumbent vs free)
        mean_fe={cid: round(float(np.mean(fe_by_comp[cid])), 6) for cid in comp_ids
                 if fe_by_comp[cid]},
        total_abs_dw={cid: round(absdw_by_comp[cid], 4) for cid in comp_ids},
        n_updates={cid: int(nfire_by_comp[cid]) for cid in comp_ids},
        weights_pre=weights_pre, weights_post=weights_post,
        active_total_pre={cid: round(_active_total(engine, engine.neurons[cid], pattern), 4)
                          for cid in comp_ids},
    ), events


def recall(engine, order):
    """Present each pattern once WITHOUT resetting weights; the recall winner is the first
    competitor to consolidate (mature). Learning stays on (as in the dashboard) but a few
    boundaries do not materially move a matured owner."""
    out = {}
    for pattern in order:
        engine.set_pattern(pattern)
        win = Counter()
        for _ in range(40):
            engine.step()
            if engine.winner is not None:
                win[engine.winner] += 1
        out[pattern] = win.most_common(1)[0][0] if win else None
    return out


def classify_failure(phases, owners, recall_map, order):
    """Return (passed, failure_class|None). Declared BEFORE inspecting outcomes."""
    if all(len(p['competitor_fires']) == 0 for p in phases):
        return False, 'no_competitor_fires_or_learns'
    distinct = len(set(o for o in owners if o)) == len(owners) and None not in owners
    consolidated = all(p['consolidated'] for p in phases)
    turnover = all(p['turnover_from_prev'] for p in phases[1:])
    recall_ok = all(recall_map.get(pat) == own for pat, own in zip(order, owners))
    if distinct and consolidated and turnover and recall_ok:
        return True, None
    # failure taxonomy
    owner_counts = Counter(o for o in owners if o)
    if any(v >= 2 for v in owner_counts.values()):
        return False, 'one_incumbent_absorbs_multiple_patterns'
    if not consolidated:
        return False, 'learning_measurable_but_too_slow'
    if not distinct:
        return False, 'stable_owners_not_one_to_one'
    if not recall_ok:
        return False, 'recall_inconsistent_with_training'
    return False, 'stable_owners_not_one_to_one'


def run_four_pattern(engine, *, dwell, order=CANONICAL_ORDER, final_window=1,
                     dominance=0.95, collect_events=False, recorder=None):
    """Full protocol: train each pattern for the fixed dwell (no early stop), then recall
    all four without resetting weights, then evaluate + classify."""
    phases, all_events, prev = [], [], None
    for pattern in order:
        if recorder is not None:
            recorder.set_annotation(phase=pattern)
            recorder.marker('phase_start', data={'phase': pattern,
                                                 'active_pixels': _active_pixels(pattern)})
        ph, evs = run_phase(engine, pattern, dwell, prev_owner=prev,
                            final_window=final_window, dominance=dominance,
                            collect_events=collect_events, recorder=recorder)
        phases.append(ph)
        all_events.extend(evs)
        prev = ph['owner']
    owners = [p['owner'] for p in phases]
    recall_map = recall(engine, order)
    passed, failure = classify_failure(phases, owners, recall_map, order)
    # RGC frequency integrity across all phases (no feature halved/dropped/inhibited).
    rgc_equal = all(p['rgc_equal_frequency'] for p in phases)
    return dict(
        owners=owners, owner_by_pattern={p['pattern']: p['owner'] for p in phases},
        distinct_owners=len(set(o for o in owners if o)), phases=phases,
        recall=recall_map,
        recall_consistent=all(recall_map.get(pat) == own for pat, own in zip(order, owners)),
        turnover_every_switch=all(p['turnover_from_prev'] for p in phases[1:]),
        all_consolidated=all(p['consolidated'] for p in phases),
        rgc_equal_frequency_all=rgc_equal,
        passed=passed, failure_class=failure,
    ), all_events


# ================================================= isolated coincidence probe
def c_probe(B, m, *, steps=400):
    """Isolated coincidence-cell causal probe for the SAME dual FE/FES parameters. The
    acceptance column has NO C cell, so this proves the C implementation separately: it
    drives one CoincidencePyramidalNeuron with repeated basal+apical coincidences and
    records the basal trajectory, FE/FES, valid coincidences, and whether one vs two
    deposits (from rest, leak 0) can fire it. C activity cannot influence the direct column."""
    theta = THETA
    eta_c = REF['c_eta'] * m
    c = CoincidencePyramidalNeuron('C', 'B', 'b0', apical_sources=['A'], apical_edge_ids=['a0'],
                                   basal_weight=0.25 * theta, w_max=4 * theta, eta_c=eta_c,
                                   use_fe=True, update_mode='c_dual_fe_fes', threshold=theta,
                                   basal_distance_factor=1.0, leak_rate=0.0)
    c.record_updates = True
    # one-deposit-from-rest test: a single basal+apical coincidence at the current weight.
    one_deposit_fires = (0.25 * theta) >= theta
    # two-deposit-from-rest test (leak 0, refractory 0): two consecutive coincidences with
    # no fire in between accumulate 2*w on the membrane.
    two_deposit_fires = (2 * 0.25 * theta) >= theta
    matured_at = None
    traj = []
    for t in range(1, steps + 1):
        c.begin_event_boundary()
        c.deliver_basal('B', 1.0, tau=0.0)          # basal available
        c.deliver_apical('A', tau=0.0)              # apical permission -> deposit w*s
        fired = False
        if c.can_fire():
            c.fire(0.0)
            c.update_basal_weight()                 # dual FE/FES basal learning
            fired = True
            if matured_at is None:
                matured_at = t
        traj.append((t, round(float(c.basal_weight), 4), fired))
    log = c.update_log[-1] if c.update_log else None
    return dict(B=B, m=m, eta_c=eta_c, basal_init=0.25 * theta,
                one_deposit_from_rest_fires=one_deposit_fires,
                two_deposit_from_rest_fires=two_deposit_fires,
                matured_to_one_shot_at=matured_at, final_basal=round(float(c.basal_weight), 4),
                valid_coincidences=int(sum(1 for _, _, f in traj if f)),
                last_fe=(round(log['fe'], 6) if log else None),
                last_fes=(round(log['fes'], 6) if log else None),
                nonfinite=bool(not math.isfinite(c.basal_weight)))


# ============================================================== orchestration
def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _write_csv(path, rows, columns):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def _switch_evidence(phases, order):
    """Per-pattern-switch weight-based evidence: incumbent vs new owner, their FE + total
    |dw|, active-afferent totals for the old and new pattern, and RGC frequency integrity.
    Directly supports the claim that the incumbent's established weights + high charge make
    its updates smaller than the still-plastic competitor's during turnover."""
    rows = []
    for i, p in enumerate(phases):
        if i == 0:
            continue
        inc, new = p['prev_owner'], p['owner']
        prev_pat = order[i - 1]
        rows.append(dict(
            switch=f'{prev_pat} -> {p["pattern"]}', incumbent=inc, new_owner=new,
            turnover=bool(inc != new),
            incumbent_mean_fe=p['mean_fe'].get(inc), new_owner_mean_fe=p['mean_fe'].get(new),
            incumbent_abs_dw=p['total_abs_dw'].get(inc),
            new_owner_abs_dw=p['total_abs_dw'].get(new),
            incumbent_smaller_updates=bool(
                inc != new and p['total_abs_dw'].get(inc, 0.0) < p['total_abs_dw'].get(new, 0.0)),
            incumbent_active_total_new_pattern=round(
                _sum_active(p['weights_pre'].get(inc), p['active_pixels']), 4) if inc else None,
            new_owner_active_total_new_pattern=round(
                _sum_active(p['weights_pre'].get(new), p['active_pixels']), 4) if new else None,
            rgc_equal_frequency=p['rgc_equal_frequency']))
    return rows


def _sum_active(weight_row, active_pixels):
    if weight_row is None:
        return 0.0
    return float(sum(weight_row[p] for p in active_pixels))


def _acceptance(result):
    """Explicit PASS/FAIL per acceptance condition (the primary desired outcome)."""
    ph = result['phases']
    return {
        'four_distinct_owners': result['distinct_owners'] == 4 and None not in result['owners'],
        'each_pattern_one_stable_owner': result['all_consolidated'],
        'turnover_at_every_switch': result['turnover_every_switch'],
        'recall_preserves_one_to_one': result['recall_consistent'],
        'first_owner_does_not_absorb_others': (
            Counter(o for o in result['owners'] if o).most_common(1)[0][1] == 1
            if any(result['owners']) else False),
        'all_rgc_unsuppressed_equal_frequency': result['rgc_equal_frequency_all'],
    }


def reference_and_control(run_dir, *, dwell, seed=1, replay=True):
    """The reference seed-1 run (dual ON, saved with a replay) and the same-topology
    flag-off control (production rule)."""
    # ---- reference (dual ON) with replay + full evidence ----
    engine = build_engine(seed, dual=True, B=REF['B'], m=1)
    rec = None
    if replay:
        rec = ReplayRecorder(
            engine, experiment='dual_fe_cc4_reference',
            output_root=os.path.join(run_dir, 'replay'), record_every=2, checkpoint_every=50,
            hierarchical_feedback=FEEDBACK_NOT_APPLICABLE,
            conditions={'dual_fe_fes': True, **REF, 'm': 1, 'seed': seed},
            schedule={'order': list(CANONICAL_ORDER), 'dwell': dwell},
            metrics_columns=['timestep', 'phase', 'winner', 'firing'],
            optional_columns=['winner', 'firing'])
        rec.__enter__()
    result, events = run_four_pattern(engine, dwell=dwell, collect_events=True, recorder=rec)
    acc = _acceptance(result)
    if rec is not None:
        rec.finish(STATUS_COMPLETED, checks=acc,
                   result={'owners': result['owners'], 'passed': result['passed']})
        rec.__exit__(None, None, None)

    # artifacts
    _write_json(os.path.join(run_dir, 'reference_summary.json'),
                dict(config=dict(seed=seed, dual_fe_fes=True, m=1, **REF, dwell=dwell),
                     acceptance=acc, **{k: result[k] for k in
                     ('owners', 'owner_by_pattern', 'distinct_owners', 'recall',
                      'recall_consistent', 'turnover_every_switch', 'all_consolidated',
                      'rgc_equal_frequency_all', 'passed', 'failure_class')}))
    _write_json(os.path.join(run_dir, 'reference_phases.json'), result['phases'])
    _write_json(os.path.join(run_dir, 'final_weights.json'),
                result['phases'][-1]['weights_post'])
    # per-learning-event FE/FES trace
    if events:
        _write_csv(os.path.join(run_dir, 'learning_events.csv'), events,
                   ['phase', 'cell', 'iaccq', 'theta', 'fe', 'lr', 'pre_sum', 'post_sum',
                    'max_w', 'min_w', 'spike_tau', 'nonfinite'])
    # phase/window metrics CSV
    prows = []
    for p in result['phases']:
        prows.append(dict(pattern=p['pattern'], owner=p['owner'], dominance=p['dominance'],
                          final_window_dominance=p['final_window_dominance'],
                          consolidated=p['consolidated'], turnover=p['turnover_from_prev'],
                          hard_resets=p['hard_reset_events'],
                          rgc_equal_frequency=p['rgc_equal_frequency'],
                          incumbent_mean_fe=(p['mean_fe'].get(p['prev_owner'])
                                             if p['prev_owner'] else None),
                          new_owner_mean_fe=p['mean_fe'].get(p['owner']),
                          incumbent_abs_dw=(p['total_abs_dw'].get(p['prev_owner'])
                                            if p['prev_owner'] else None),
                          new_owner_abs_dw=p['total_abs_dw'].get(p['owner'])))
    _write_csv(os.path.join(run_dir, 'reference_phase_metrics.csv'), prows,
               ['pattern', 'owner', 'dominance', 'final_window_dominance', 'consolidated',
                'turnover', 'hard_resets', 'rgc_equal_frequency', 'incumbent_mean_fe',
                'new_owner_mean_fe', 'incumbent_abs_dw', 'new_owner_abs_dw'])
    # source delivery + WTA counts
    _write_json(os.path.join(run_dir, 'source_delivery.json'),
                {p['pattern']: dict(rgc_fires=p['rgc_fires'],
                                    rgc_equal_frequency=p['rgc_equal_frequency'],
                                    competitor_fires=p['competitor_fires'],
                                    hard_reset_events=p['hard_reset_events'],
                                    first_spike_tau=p['first_spike_tau'])
                 for p in result['phases']})

    # per-switch weight-based evidence (incumbent vs free competitor)
    _write_csv(os.path.join(run_dir, 'reference_switch_evidence.csv'),
               _switch_evidence(result['phases'], CANONICAL_ORDER),
               ['switch', 'incumbent', 'new_owner', 'turnover', 'incumbent_mean_fe',
                'new_owner_mean_fe', 'incumbent_abs_dw', 'new_owner_abs_dw',
                'incumbent_smaller_updates', 'incumbent_active_total_new_pattern',
                'new_owner_active_total_new_pattern', 'rgc_equal_frequency'])

    # ---- flag-off control (same topology, production rule) ----
    control_engine = build_engine(seed, dual=False, m=1)
    control, _ = run_four_pattern(control_engine, dwell=dwell)
    _write_json(os.path.join(run_dir, 'control_flag_off_summary.json'),
                dict(config=dict(seed=seed, dual_fe_fes=False, dwell=dwell),
                     acceptance=_acceptance(control),
                     **{k: control[k] for k in ('owners', 'owner_by_pattern',
                        'distinct_owners', 'recall_consistent', 'passed', 'failure_class')}))
    return result, acc, control


def seed_sweep(run_dir, *, seeds, dwell, dual=True, B=REF['B'], m=1, tag='reference'):
    rows = []
    for s in seeds:
        eng = build_engine(s, dual=dual, B=B, m=m)
        res, _ = run_four_pattern(eng, dwell=dwell)
        rows.append(dict(seed=s, owners=res['owners'], distinct=res['distinct_owners'],
                         turnover=res['turnover_every_switch'],
                         consolidated=res['all_consolidated'],
                         recall_consistent=res['recall_consistent'],
                         rgc_equal=res['rgc_equal_frequency_all'],
                         passed=res['passed'], failure_class=res['failure_class']))
    passed = sum(1 for r in rows if r['passed'])
    out = dict(tag=tag, dual=dual, B=B, m=m, dwell=dwell, n_seeds=len(seeds),
               n_passed=passed, success_fraction=round(passed / max(1, len(seeds)), 3),
               per_seed=rows)
    _write_json(os.path.join(run_dir, f'seed_sweep_{tag}.json'), out)
    return out


def long_dwell_stress(run_dir, *, long_dwell, second_dwell, m, B=REF['B'], seed=1):
    """Train pattern 1 substantially longer than the consolidation window, then present the
    second overlapping pattern. Checks whether slow nonzero tail plasticity eventually locks
    a one-afferent incumbent that then absorbs pattern 2 (a failure after long dwell is a
    scientific result, not permission to shorten the test)."""
    eng = build_engine(seed, dual=True, B=B, m=m)
    p1, _ = run_phase(eng, CANONICAL_ORDER[0], long_dwell, prev_owner=None,
                     final_window=1, dominance=0.95)
    p2, _ = run_phase(eng, CANONICAL_ORDER[1], second_dwell, prev_owner=p1['owner'],
                     final_window=1, dominance=0.95)
    out = dict(seed=seed, B=B, m=m, long_dwell=long_dwell, second_dwell=second_dwell,
               pattern1_owner=p1['owner'], pattern1_final_dominance=p1['final_window_dominance'],
               pattern2_owner=p2['owner'], pattern2_final_dominance=p2['final_window_dominance'],
               turnover_after_long_dwell=bool(p2['owner'] != p1['owner']),
               incumbent_locked=bool(p2['owner'] == p1['owner']),
               pattern1_active_total=p1['active_total_pre'],
               pattern2_active_total=p2['active_total_pre'],
               pattern1_max_weight=max(max(v) for v in p1['weights_post'].values()),
               rgc_equal_frequency=bool(p1['rgc_equal_frequency'] and p2['rgc_equal_frequency']))
    _write_json(os.path.join(run_dir, 'long_dwell_stress.json'), out)
    return out


def bxm_grid(run_dir, *, dwell, seed=1, c_probe_steps=400):
    """The required 16-point (B, m) sensitivity grid on the direct column, plus the same 16
    points through the isolated C probe. Same init/order/dwell/thresholds every point; no
    early stop; e = wte = 0.001 fixed. Saves every attempted configuration incl. failures."""
    grid_rows, c_rows = [], []
    for B in GRID_B:
        for m in GRID_M:
            eng = build_engine(seed, dual=True, B=B, m=m)
            res, _ = run_four_pattern(eng, dwell=dwell)
            wp = res['phases'][-1]['weights_post']
            allw = [v for row in wp.values() for v in row]
            drift = 0.0
            # late-window weight drift: |post - pre| over the last phase
            last = res['phases'][-1]
            for cid in last['weights_post']:
                drift = max(drift, max(abs(a - b) for a, b in
                                       zip(last['weights_post'][cid], last['weights_pre'][cid])))
            inc_dw = [p['total_abs_dw'].get(p['prev_owner']) for p in res['phases'][1:]
                      if p['prev_owner']]
            free_dw = [p['total_abs_dw'].get(p['owner']) for p in res['phases'][1:]]
            grid_rows.append(dict(
                B=B, m=m, eta=REF['eta'] * m, c_eta=REF['c_eta'] * m,
                distinct_owners=res['distinct_owners'],
                per_pattern_dominance=[p['final_window_dominance'] for p in res['phases']],
                recall_consistent=res['recall_consistent'],
                turnover_count=sum(1 for p in res['phases'][1:] if p['turnover_from_prev']),
                first_consolidation_step=next(
                    (i for i, p in enumerate(res['phases']) if p['consolidated']), None),
                min_weight=round(min(allw), 4), max_weight=round(max(allw), 4),
                final_weight_mean=round(float(np.mean(allw)), 4),
                late_weight_drift=round(drift, 4),
                incumbent_abs_dw=round(float(np.mean(inc_dw)), 4) if inc_dw else None,
                free_abs_dw=round(float(np.mean(free_dw)), 4) if free_dw else None,
                floor_hits=int(sum(1 for v in allw if v <= REF['wte'] + 1e-9)),
                nonfinite=bool(not all(math.isfinite(v) for v in allw)),
                rgc_equal_frequency=res['rgc_equal_frequency_all'],
                passed=res['passed'], failure_class=res['failure_class']))
            cp = c_probe(B, m, steps=c_probe_steps)
            c_rows.append(cp)
    _write_csv(os.path.join(run_dir, 'grid_bxm.csv'), grid_rows,
               ['B', 'm', 'eta', 'c_eta', 'distinct_owners', 'per_pattern_dominance',
                'recall_consistent', 'turnover_count', 'first_consolidation_step',
                'min_weight', 'max_weight', 'final_weight_mean', 'late_weight_drift',
                'incumbent_abs_dw', 'free_abs_dw', 'floor_hits', 'nonfinite',
                'rgc_equal_frequency', 'passed', 'failure_class'])
    _write_csv(os.path.join(run_dir, 'c_probe_bxm.csv'), c_rows,
               ['B', 'm', 'eta_c', 'basal_init', 'one_deposit_from_rest_fires',
                'two_deposit_from_rest_fires', 'matured_to_one_shot_at', 'final_basal',
                'valid_coincidences', 'last_fe', 'last_fes', 'nonfinite'])
    _write_json(os.path.join(run_dir, 'grid_bxm.json'), grid_rows)
    return grid_rows, c_rows


def _grid_table(grid_rows):
    """Compact distinct-owner heatmap over B (rows) x m (cols)."""
    by = {(r['B'], r['m']): r for r in grid_rows}
    lines = ['distinct owners (4 = one-to-one) over B x m; * = PASS',
             '        m=' + '   '.join(f'{m:>4}' for m in GRID_M)]
    for B in GRID_B:
        cells = []
        for m in GRID_M:
            r = by[(B, m)]
            cells.append(f"{r['distinct_owners']}{'*' if r['passed'] else ' '}")
        lines.append(f'  B={B:>4}  ' + '   '.join(f'{c:>4}' for c in cells))
    return '\n'.join(lines)


def select_confirmation_candidate(grid_rows):
    """Declared BEFORE inspecting outcomes: among PASSING points, prefer full one-to-one +
    recall, then long-dwell stability (checked separately), then the smallest LR multiplier,
    then the B closest to 5."""
    passing = [r for r in grid_rows if r['passed']]
    if not passing:
        return None
    passing.sort(key=lambda r: (r['m'], abs(r['B'] - 5.0)))
    return dict(B=passing[0]['B'], m=passing[0]['m'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-root', default='experiments/runs/dual_fe_cc4')
    ap.add_argument('--dwell', type=int, default=800)
    ap.add_argument('--seeds', type=int, default=8, help='seeds 1..N for sweeps')
    ap.add_argument('--long-dwell', type=int, default=6000)
    ap.add_argument('--quick', action='store_true', help='fast smoke run (small dwell/grid)')
    ap.add_argument('--no-replay', action='store_true')
    args = ap.parse_args(argv)

    if args.quick:
        args.dwell = 300
        args.seeds = 3
        args.long_dwell = 1500

    from experiments.replay_recorder import _new_run_id
    run_dir = os.path.join(args.output_root, _new_run_id('dual_fe_cc4'))
    os.makedirs(run_dir, exist_ok=True)
    print(f'== dual FE/FES direct CC4 experiment ==\nrun dir: {run_dir}\n')

    _write_json(os.path.join(run_dir, 'config.json'), dict(
        experiment='dual_fe_cc4_consolidation', topology='rg_direct_cc4',
        reference=REF, grid_B=list(GRID_B), grid_m=list(GRID_M),
        dwell=args.dwell, seeds=list(range(1, args.seeds + 1)),
        long_dwell=args.long_dwell, order=list(CANONICAL_ORDER)))
    _write_json(os.path.join(run_dir, 'topology_fingerprint.json'),
                (lambda t: dict(nodes=len(t['neurons']), synapses=len(t['synapses']),
                                node_ids=[n['id'] for n in t['neurons']],
                                params={k: t['params'][k] for k in
                                        ('dual_fe_fes', 'dual_fe_e', 'dual_fe_wte', 'dual_fe_B',
                                         'eta', 'c_eta', 'leak_rate', 'refractory_steps')}))
                (build_engine(1, dual=True).topology()))

    # 1) reference (dual ON) + flag-off control -- saved BEFORE any sweep/tuning.
    print('[1/5] reference (seed 1, dual ON, m=1) + flag-off control ...')
    ref, ref_acc, control = reference_and_control(run_dir, dwell=args.dwell,
                                                  replay=not args.no_replay)
    print(f"    reference owners={ref['owners']} passed={ref['passed']} "
          f"({ref['failure_class']})")
    print(f"    control  owners={control['owners']} passed={control['passed']} "
          f"({control['failure_class']})")

    # 2) required (B, m) grid + isolated C probe (seed 1). Saved every point.
    print('[2/5] 16-point (B, m) grid + isolated C probe ...')
    grid_rows, c_rows = bxm_grid(run_dir, dwell=args.dwell)
    table = _grid_table(grid_rows)
    with open(os.path.join(run_dir, 'grid_table.txt'), 'w') as f:
        f.write(table + '\n')
    print('\n' + table + '\n')

    # 3) confirmation candidate (declared rule) + multi-seed sweep.
    candidate = select_confirmation_candidate(grid_rows)
    print(f'[3/5] confirmation candidate (declared rule): {candidate}')
    seeds = list(range(1, args.seeds + 1))
    if candidate is not None:
        cand_sweep = seed_sweep(run_dir, seeds=seeds, dwell=args.dwell,
                                B=candidate['B'], m=candidate['m'], tag='candidate')
        print(f"    candidate seed sweep: {cand_sweep['n_passed']}/{cand_sweep['n_seeds']} "
              f"pass (success={cand_sweep['success_fraction']})")
        # candidate evidence run (seed 1): full per-switch weight-based evidence where every
        # switch turns over, so the incumbent-vs-free update comparison is available at each.
        cand_eng = build_engine(1, dual=True, B=candidate['B'], m=candidate['m'])
        cand_res, _ = run_four_pattern(cand_eng, dwell=args.dwell, collect_events=True)
        _write_json(os.path.join(run_dir, 'candidate_phases.json'), cand_res['phases'])
        _write_csv(os.path.join(run_dir, 'candidate_switch_evidence.csv'),
                   _switch_evidence(cand_res['phases'], CANONICAL_ORDER),
                   ['switch', 'incumbent', 'new_owner', 'turnover', 'incumbent_mean_fe',
                    'new_owner_mean_fe', 'incumbent_abs_dw', 'new_owner_abs_dw',
                    'incumbent_smaller_updates', 'incumbent_active_total_new_pattern',
                    'new_owner_active_total_new_pattern', 'rgc_equal_frequency'])
    else:
        cand_sweep = None

    # 4) reference-parameter seed sweep (m=1), reported regardless of pass.
    print('[4/5] reference-parameter (m=1) seed sweep ...')
    ref_sweep = seed_sweep(run_dir, seeds=seeds, dwell=args.dwell, m=1, tag='reference')
    print(f"    reference seed sweep: {ref_sweep['n_passed']}/{ref_sweep['n_seeds']} pass")

    # 5) long-dwell stress at the reference params AND at the candidate (if any).
    print('[5/5] long-dwell stress test ...')
    ld_ref = long_dwell_stress(run_dir, long_dwell=args.long_dwell,
                               second_dwell=args.dwell, m=1)
    ld_cand = (long_dwell_stress(run_dir, long_dwell=args.long_dwell, second_dwell=args.dwell,
                                 m=candidate['m'], B=candidate['B'])
               if candidate else None)
    if ld_cand:                     # keep both; write the candidate one under a distinct name
        _write_json(os.path.join(run_dir, 'long_dwell_stress_candidate.json'), ld_cand)
    print(f"    long-dwell (m=1): p1={ld_ref['pattern1_owner']} -> p2={ld_ref['pattern2_owner']} "
          f"turnover={ld_ref['turnover_after_long_dwell']}")

    overall = dict(
        run_dir=run_dir, reference_acceptance=ref_acc, reference_passed=ref['passed'],
        reference_failure_class=ref['failure_class'], control_passed=control['passed'],
        control_failure_class=control['failure_class'],
        grid_pass_count=sum(1 for r in grid_rows if r['passed']),
        grid_points=len(grid_rows), confirmation_candidate=candidate,
        candidate_success_fraction=(cand_sweep['success_fraction'] if cand_sweep else None),
        reference_success_fraction=ref_sweep['success_fraction'],
        long_dwell_incumbent_locked_m1=ld_ref['incumbent_locked'],
        any_nonfinite=any(r['nonfinite'] for r in grid_rows))
    _write_json(os.path.join(run_dir, 'summary.json'), overall)
    print(f'\nwrote artifacts to {run_dir}')
    print('acceptance (reference, seed 1, dual ON, m=1):')
    for k, v in ref_acc.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
