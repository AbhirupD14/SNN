"""Headless acceptance experiment for the Eor-less direct-identity tiled topology.

This is the AUTHORITATIVE scientific evidence for ``tiled_cc_direct_identity``; a dashboard
screenshot is only supplemental. It imports no FastAPI / websocket / DOM module and runs
four deterministic probes:

  A. TURNOVER + RECALL. Train one 3x3 patch on a pattern until an L1 owner is stable,
     switch to a second pattern and require a DIFFERENT ordinary-E owner, then return to
     the first and check recall of the original owner. Records whether each owner emitted
     direct parent evidence on the boundary it won (no Eor recovery process), and whether
     one owner's C basal learning depressed another's.
  B. PATTERN SWEEP. Probe A repeated over all four canonical patterns and several seeds,
     so a turnover claim rests on more than one trajectory.
  C. COMPOSITION. Two active patches; checks that each patch contributes its ACTUAL winner
     identity, that an L2 cell receives distinct simultaneous evidence from both children,
     and that the theta/2 detector ceiling holds per identity afferent.
  D. IDENTITY DISCRIMINATION. The success criterion of the whole topology: hold the patch
     LOCATIONS fixed and change only which local pattern (hence which local winner) is
     active, then ask whether L2 distinguishes the two compositions.

Everything is reported as measured. A probe that fails is recorded, not hidden. Run:

    PYTHONPATH=. .venv/bin/python experiments/direct_identity_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine, PATTERNS               # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'direct_identity_results.json')

TOPOLOGY = 'tiled_cc_direct_identity'
# Frozen fast-maturation acceptance rates, so a column consolidates inside a run instead
# of taking ~40k boundaries. The live dashboard later moved to c_eta=16 and auto pacing;
# this artifact intentionally retains the already-recorded c_eta=2, input_period=1 protocol.
RATES = dict(dual_fe_fes=True, eta=4.0, c_eta=2.0, dual_fe_B=5.0, leak_rate=0.0,
             refractory_steps=0, c_feedback_reset=True, e_weight_cap_frac=0.5)
DWELL = 2500                 # boundaries per pattern phase
TAIL = 400                   # trailing window used to declare the phase owner
DOMINANCE = 0.8              # owner must win >= this fraction of the tail's won boundaries


def _new_engine(seed):
    return SimulationEngine(seed=seed, topology=TOPOLOGY, **RATES)


def _c_weights(engine, column):
    c = engine.exc[f'{column}C']
    return {s: float(w) for s, w in zip(c.basal.source_ids, c.basal.weights)}


def run_phase(engine, patch, pattern, column, dwell=DWELL, tail=TAIL):
    """Present one pattern into one patch for ``dwell`` boundaries.

    Returns the tail-window owner plus the causal evidence checks: on how many of the
    boundaries the owner won did its identity actually reach the parent bank in the SAME
    step (the property Eor could not guarantee once its weights collapsed)."""
    engine.set_patch_pattern(patch[0], patch[1], pattern)
    wins = Counter()
    tail_wins = Counter()
    won_boundaries = 0
    evidence_boundaries = 0
    l2_wins = Counter()
    l2_e = [n for n in engine.exc if engine._column_of.get(n) == 'L2c00'
            and engine._role_of.get(n) == 'E']
    for step in range(dwell):
        st = engine.step()
        w = st['column_winners'].get(column)
        if w:
            wins[w['id']] += 1
            won_boundaries += 1
            if step >= dwell - tail:
                tail_wins[w['id']] += 1
            # direct parent evidence: the winner's own identity is queued to every L2 E
            reached = {pe for pe in l2_e
                       if w['id'] in engine._ff_deliv_next.get(pe, ())}
            if reached == set(l2_e):
                evidence_boundaries += 1
        l2w = st['column_winners'].get('L2c00')
        if l2w:
            l2_wins[l2w['id']] += 1
    owner, owner_wins = (tail_wins.most_common(1)[0] if tail_wins else (None, 0))
    tail_total = sum(tail_wins.values())
    return dict(
        pattern=pattern, owner=owner,
        owner_dominance=round(owner_wins / tail_total, 4) if tail_total else 0.0,
        distinct_owners_in_tail=len(tail_wins),
        won_boundaries=won_boundaries,
        direct_evidence_boundaries=evidence_boundaries,
        direct_evidence_fraction=(round(evidence_boundaries / won_boundaries, 4)
                                  if won_boundaries else 0.0),
        l2_owner=(l2_wins.most_common(1)[0][0] if l2_wins else None),
        l2_win_boundaries=sum(l2_wins.values()),
        c_basal=_c_weights(engine, column))


# ----------------------------------------------------------------- probe A / B
def run_turnover(seed=1, patch=(1, 1), first='row 1', second='col 1', dwell=DWELL):
    """Train -> switch -> return. The declared turnover criterion is: each phase has a
    single dominant tail owner, the switch phase's owner DIFFERS from the first, and the
    return phase recalls the first owner."""
    column = f'L1c{patch[0]}{patch[1]}'
    e = _new_engine(seed)
    phases = [run_phase(e, patch, first, column, dwell=dwell),
              run_phase(e, patch, second, column, dwell=dwell),
              run_phase(e, patch, first, column, dwell=dwell)]
    owners = [p['owner'] for p in phases]
    # Did learning the second owner's association depress the first owner's basal weight?
    w_after_first = phases[0]['c_basal']
    w_after_second = phases[1]['c_basal']
    first_owner = owners[0]
    depression = None
    if first_owner is not None:
        depression = round(w_after_second[first_owner] - w_after_first[first_owner], 9)
    return dict(
        seed=seed, patch=list(patch), patterns=[first, second, first], dwell=dwell,
        owners=owners, phases=phases,
        checks=dict(
            stable_owner_each_phase=all(p['owner'] is not None
                                        and p['owner_dominance'] >= DOMINANCE
                                        for p in phases),
            turnover_on_switch=(owners[0] is not None and owners[1] is not None
                                and owners[0] != owners[1]),
            recall_on_return=(owners[0] is not None and owners[0] == owners[2]),
            every_owner_emits_direct_evidence=all(p['direct_evidence_fraction'] == 1.0
                                                  for p in phases),
            first_owner_basal_not_depressed=(depression is not None and depression >= 0.0),
        ),
        first_owner_basal_delta_during_second_phase=depression)


def run_pattern_sweep(seeds=(1, 2, 3), patch=(1, 1), dwell=DWELL):
    """Probe A over every ordered pair of canonical patterns, for several seeds."""
    names = list(PATTERNS)
    runs = []
    for seed in seeds:
        for i, first in enumerate(names):
            second = names[(i + 1) % len(names)]
            r = run_turnover(seed=seed, patch=patch, first=first, second=second, dwell=dwell)
            runs.append(dict(seed=seed, first=first, second=second,
                             owners=r['owners'], checks=r['checks']))
    tally = {k: sum(1 for r in runs if r['checks'][k]) for k in runs[0]['checks']}
    return dict(seeds=list(seeds), n_runs=len(runs), passed=tally, runs=runs)


# --------------------------------------------------------------------- probe C
def run_composition(seed=1, patches=((0, 0), (2, 2)), patterns=('row 1', 'col 1'),
                    dwell=DWELL):
    """Two active patches: each contributes its OWN winner identity to the parent."""
    e = _new_engine(seed)
    cols = [f'L1c{p[0]}{p[1]}' for p in patches]
    for p, pat in zip(patches, patterns):
        e.set_patch_pattern(p[0], p[1], pat)
    owners = {c: Counter() for c in cols}
    simultaneous = 0
    distinct_pair_events = Counter()
    l2_wins = Counter()
    for _ in range(dwell):
        st = e.step()
        won = {c: st['column_winners'].get(c) for c in cols}
        for c, w in won.items():
            if w:
                owners[c][w['id']] += 1
        if all(won.values()):
            simultaneous += 1
            ids = tuple(won[c]['id'] for c in cols)
            # both identities must be queued to the SAME parent cell, distinctly
            for pe in [n for n in e.exc if e._column_of.get(n) == 'L2c00'
                       and e._role_of.get(n) == 'E']:
                if set(ids) <= set(e._ff_deliv_next.get(pe, ())):
                    distinct_pair_events[ids] += 1
                    break
        l2w = st['column_winners'].get('L2c00')
        if l2w:
            l2_wins[l2w['id']] += 1
    # theta/2 ceiling on every direct-identity afferent of every L2 detector
    theta = float(e.params['e_threshold'])
    cap = 0.5 * theta
    max_identity_w = max(float(e.exc[n].acc_weights.max())
                         for n in e.exc if e._column_of.get(n) == 'L2c00'
                         and e._role_of.get(n) == 'E')
    return dict(
        seed=seed, patches=[list(p) for p in patches], patterns=list(patterns), dwell=dwell,
        per_column_owner={c: (owners[c].most_common(1)[0] if owners[c] else None)
                          for c in cols},
        distinct_column_owners=len({owners[c].most_common(1)[0][0] for c in cols
                                    if owners[c]}) == len(cols),
        simultaneous_boundaries=simultaneous,
        distinct_pair_evidence_boundaries=sum(distinct_pair_events.values()),
        observed_identity_pairs={'+'.join(k): v for k, v in distinct_pair_events.items()},
        l2_owner=(l2_wins.most_common(1)[0][0] if l2_wins else None),
        l2_win_boundaries=sum(l2_wins.values()),
        max_l2_identity_weight=round(max_identity_w, 6),
        identity_cap_respected=bool(max_identity_w <= cap + 1e-6))


# --------------------------------------------------------------------- probe D
def run_identity_discrimination(seed=1, patches=((0, 0), (2, 2)),
                                comp_a=('row 1', 'col 1'), comp_b=('diag \\', 'col 1'),
                                dwell=DWELL, probe=600):
    """THE success criterion. Two compositions occupy the SAME patch locations and differ
    only in which local pattern -- hence which local winner identity -- one patch emits.

    Trains on composition A, then B, then re-presents each and records the L2 winner
    distribution. Success = the two compositions drive DIFFERENT L2 winners, which is only
    possible if the child winner's identity (not merely "this column fired") reached L2."""
    e = _new_engine(seed)
    cols = [f'L1c{p[0]}{p[1]}' for p in patches]

    def present(comp, n):
        for p, pat in zip(patches, comp):
            e.set_patch_pattern(p[0], p[1], pat)
        l2 = Counter()
        local = {c: Counter() for c in cols}
        for _ in range(n):
            st = e.step()
            w = st['column_winners'].get('L2c00')
            if w:
                l2[w['id']] += 1
            for c in cols:
                lw = st['column_winners'].get(c)
                if lw:
                    local[c][lw['id']] += 1
        return l2, {c: (local[c].most_common(1)[0][0] if local[c] else None) for c in cols}

    present(comp_a, dwell)                       # train A
    present(comp_b, dwell)                       # train B
    la, owners_a = present(comp_a, probe)        # probe A
    lb, owners_b = present(comp_b, probe)        # probe B
    top_a = la.most_common(1)[0][0] if la else None
    top_b = lb.most_common(1)[0][0] if lb else None
    changed_col = cols[0]
    return dict(
        seed=seed, patches=[list(p) for p in patches],
        composition_a=list(comp_a), composition_b=list(comp_b),
        dwell=dwell, probe=probe,
        local_owners_a=owners_a, local_owners_b=owners_b,
        local_identity_actually_changed=(owners_a[changed_col] != owners_b[changed_col]),
        l2_distribution_a=dict(la), l2_distribution_b=dict(lb),
        l2_top_a=top_a, l2_top_b=top_b,
        l2_distinguishes_compositions=(top_a is not None and top_b is not None
                                       and top_a != top_b))


# --------------------------------------------------------------------- probe E
def run_boundary_edge_anomaly(seed=1, warm=2500, measure=1000):
    """Regression probe for the former ``tau=1.0`` boundary-edge deadlock.

    With the mandated theta/2 detector ceiling, two coordinated child identity events
    deliver exactly theta, so a reset L2 detector crosses at the boundary edge. The event
    loop now drains dependent crossings already available at that same edge; C must fire,
    learn, and schedule feedback without retaining runaway voltage. One- and two-patch
    regimes remain separate because only the latter consistently exercises ``tau=1.0``."""
    out = {}
    for label, patches, patterns in (('one_patch', [(1, 1)], ['row 1']),
                                     ('two_patches', [(0, 0), (2, 2)], ['row 1', 'col 1'])):
        e = _new_engine(seed)
        for p, pat in zip(patches, patterns):
            e.set_patch_pattern(p[0], p[1], pat)
        for _ in range(warm):
            e.step()
        l2_taus = Counter()
        deposit_taus = Counter()
        c_spikes = 0
        feedback = 0
        for _ in range(measure):
            st = e.step()
            for n in e.exc:
                if e.spiked.get(n) and e._column_of.get(n) == 'L2c00' \
                        and e._role_of.get(n) == 'E':
                    l2_taus[round(float(e.exc[n].spike_tau), 4)] += 1
            for c in e.coincidence:
                if c.coincidence_deposit_tau is not None:
                    deposit_taus[round(float(c.coincidence_deposit_tau), 4)] += 1
                if e.spiked.get(c.id):
                    c_spikes += 1
            feedback += sum(1 for h in st['hard_reset_events']
                            if h['kind'] == 'feedback_hard_reset')
        stuck = {c.id: round(float(c.V), 1) for c in e.coincidence
                 if c.V > 10.0 * c.threshold}
        out[label] = dict(
            boundaries=measure,
            l2_spike_taus={str(k): v for k, v in sorted(l2_taus.items())},
            c_deposit_taus={str(k): v for k, v in sorted(deposit_taus.items())},
            deposits_at_boundary_edge=deposit_taus.get(1.0, 0),
            deposits_total=sum(deposit_taus.values()),
            c_spikes=c_spikes, feedback_resets=feedback,
            c_cells_with_runaway_retained_charge=stuck)
    return out


def main():
    out = dict(
        topology=TOPOLOGY, rates=RATES, dwell=DWELL, tail=TAIL, dominance=DOMINANCE,
        probe_a_turnover=run_turnover(),
        probe_b_pattern_sweep=run_pattern_sweep(),
        probe_c_composition=run_composition(),
        probe_d_identity_discrimination=run_identity_discrimination(),
        probe_e_boundary_edge_anomaly=run_boundary_edge_anomaly(),
    )
    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    a = out['probe_a_turnover']
    print('A turnover     :', a['owners'], a['checks'])
    print('B sweep passed :', out['probe_b_pattern_sweep']['passed'],
          'of', out['probe_b_pattern_sweep']['n_runs'])
    c = out['probe_c_composition']
    print('C composition  : owners', c['per_column_owner'], 'pairs',
          c['observed_identity_pairs'], 'cap ok', c['identity_cap_respected'])
    d = out['probe_d_identity_discrimination']
    print('D discrimination: local changed', d['local_identity_actually_changed'],
          '| L2', d['l2_top_a'], 'vs', d['l2_top_b'],
          '| distinguishes', d['l2_distinguishes_compositions'])
    for label, r in out['probe_e_boundary_edge_anomaly'].items():
        print(f'E {label:12s}: L2 taus {r["l2_spike_taus"]} | deposits at tau=1.0 '
              f'{r["deposits_at_boundary_edge"]}/{r["deposits_total"]} | C spikes '
              f'{r["c_spikes"]} | feedback {r["feedback_resets"]} | runaway C '
              f'{len(r["c_cells_with_runaway_retained_charge"])}')
    print('wrote', RESULTS_PATH)


if __name__ == '__main__':
    main()
