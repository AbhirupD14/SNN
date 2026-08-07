"""Deterministic headless preflight for the four presentation demo recordings.

The recorder MUST NOT hard-code owner ids, cadence numbers, turnover latencies, or graph
counts into its on-screen captions. This script runs each demonstration's protocol headless,
at the SAME seed and the SAME engine configuration the recording server will be driven with,
and writes every number the recorder is allowed to display to

    presentation_assets/preflight_measurements.json

Run:
    PYTHONPATH=. .venv/bin/python scripts/preflight_demo_measurements.py

Configuration note (recorded in the output and in recording_notes.md). The recording uses the
ORDINARY dashboard server, so every engine carries the two dashboard construction overrides
that ``/api/config`` cannot undo: ``e_weight_cap_frac=0.5`` (the theta/2 pattern-detector
ceiling) and the dual FE/FES rule. The published headless experiment
``experiments/dual_fe_cc4_consolidation.py`` leaves ``e_weight_cap_frac`` at the engine
default ``None``. ``--compare-uncapped`` re-runs the CC4 protocols with the experiment's
uncapped construction so the notes can state whether the ceiling changed the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine                       # noqa: E402
from experiments.dual_fe_cc4_consolidation import REF                 # noqa: E402

SEED = 1
CANONICAL_ORDER = ('row 1', 'col 1', 'diag \\', 'diag /')

# The dashboard construction overrides that /api/config cannot reach. Every engine below is
# built with these so the preflight measures exactly what the recorded server will do.
DASHBOARD_CONSTRUCTION = dict(e_weight_cap_frac=0.5)

# Video 2/3 scientific configuration: the dual FE/FES confirmation candidate for
# rg_direct_cc4 measured in experiments/dual_fe_cc4_consolidation.py -- B=5, LR multiplier
# m=100, i.e. eta = 0.01*100 and c_eta = 0.005*100.
CC4_B = REF['B']
CC4_M = 100
CC4_CONFIG = dict(topology='rg_direct_cc4', dual_fe_fes=True,
                  eta=REF['eta'] * CC4_M, c_eta=REF['c_eta'] * CC4_M,
                  leak_rate=REF['leak_rate'], refractory_steps=REF['refractory_steps'],
                  dual_fe_e=REF['e'], dual_fe_wte=REF['wte'], dual_fe_B=CC4_B)

# Video 1: the small 3x3 coincidence/feedback circuit at the dashboard's fast-maturation
# rates, so the cadence transition completes inside a recordable number of boundaries.
COINCIDENCE_CONFIG = dict(topology='rg_coincidence', dual_fe_fes=True, eta=4.0, c_eta=16.0,
                          input_period=0, leak_rate=0.0, refractory_steps=0)

# Video 4: both scaling graphs at the dashboard's own configuration.
SCALING_CONFIG = dict(dual_fe_fes=True, eta=4.0, c_eta=16.0, input_period=0,
                      leak_rate=0.0, refractory_steps=0)


def build(config, **extra):
    params = dict(seed=SEED, **DASHBOARD_CONSTRUCTION, **config)
    params.update(extra)
    return SimulationEngine(**params)


def _ids(engine, prefix):
    return [n for n in engine.order if n.startswith(prefix)]


def _active_input_cells(engine, pattern, prefix):
    """The <prefix>N cells whose index is an active pixel of ``pattern``."""
    vec = engine._active_patterns()[pattern]
    return [f'{prefix}{i}' for i, v in enumerate(vec) if v > 0.5]


# ===================================================== video 1: cadence / uncertainty
def measure_cadence(total=2400, early_window=400, late_window=400, pattern='row 1'):
    """L1E emission per RG volley on the sustained pattern, plus the boundary from which
    every active L1E strictly alternates fire/silent to the end of the run.

    The cadence is an OBSERVABLE signal, not a calibrated confidence score -- see
    docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md and docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md.
    """
    engine = build(COINCIDENCE_CONFIG)
    engine.set_pattern(pattern)
    l1e, rg = _ids(engine, 'L1E'), _ids(engine, 'RG')
    active_l1e = _active_input_cells(engine, pattern, 'L1E')
    seq = {a: [] for a in active_l1e}
    rg_total = l1_total = 0
    per_window, win = [], 50
    rg_w = l1_w = 0
    for t in range(1, total + 1):
        engine.step()
        for a in active_l1e:
            seq[a].append(1 if engine.spiked[a] else 0)
        r = sum(1 for n in rg if engine.spiked[n])
        l = sum(1 for n in l1e if engine.spiked[n])
        rg_total += r
        l1_total += l
        rg_w += r
        l1_w += l
        if t % win == 0:
            per_window.append(dict(boundary=t, ratio=round(l1_w / rg_w, 4) if rg_w else 0.0))
            rg_w = l1_w = 0

    # early / late emission ratio measured directly from the per-neuron sequences
    def window_ratio(a, lo, hi):
        s = seq[a][lo:hi]
        return round(sum(s) / len(s), 4) if s else 0.0

    early = {a: window_ratio(a, 0, early_window) for a in active_l1e}
    late = {a: window_ratio(a, total - late_window, total) for a in active_l1e}

    # First boundary from which EVERY active L1E strictly alternates fire/silent through the
    # end of the run. This is the declared cadence-lock criterion.
    lock = None
    for t0 in range(total - 1):
        ok = True
        for a in active_l1e:
            tail = seq[a][t0:]
            if any(tail[i] == tail[i + 1] for i in range(len(tail) - 1)):
                ok = False
                break
        if ok:
            lock = t0 + 1
            break

    # A compact literal sample the recorder may quote verbatim.
    sample_cell = active_l1e[0]
    return dict(
        topology='rg_coincidence', pattern=pattern, seed=SEED, boundaries=total,
        resolved_input_period=int(engine.resolved_input_period()),
        active_l1e=active_l1e, active_rg=_active_input_cells(engine, pattern, 'RG'),
        early_window=early_window, late_window=late_window,
        early_emission_ratio=early, late_emission_ratio=late,
        early_emission_ratio_mean=round(sum(early.values()) / len(early), 4),
        late_emission_ratio_mean=round(sum(late.values()) / len(late), 4),
        aggregate_l1e_per_rg=round(l1_total / rg_total, 4) if rg_total else 0.0,
        cadence_lock_boundary=lock,
        cadence_lock_criterion=('first boundary from which every active L1E alternates '
                                'fire/silent with no repeat through the end of the run'),
        sample_cell=sample_cell,
        sample_early=''.join(map(str, seq[sample_cell][:40])),
        sample_late=''.join(map(str, seq[sample_cell][-40:])),
        per_window_ratio=per_window,
    )


# ============================================ video 2: continuous sequential learning
def _phase(engine, pattern, dwell):
    """Present one pattern for ``dwell`` boundaries; return the modal owner and the
    per-boundary winner sequence."""
    engine.set_pattern(pattern)
    wins, seq = Counter(), []
    for _ in range(dwell):
        engine.step()
        seq.append(engine.winner)
        if engine.winner is not None:
            wins[engine.winner] += 1
    owner = wins.most_common(1)[0][0] if wins else None
    eligible = [w for w in seq if w is not None]
    tail = eligible[-max(1, len(eligible) // 4):]
    dominance = round(tail.count(owner) / len(tail), 4) if tail and owner else 0.0
    return dict(pattern=pattern, owner=owner, dwell=dwell,
                eligible_presentations=len(eligible),
                final_quarter_dominance=dominance,
                winner_counts=dict(wins)), seq


def measure_continuous(dwell=800, order=CANONICAL_ORDER, **construction):
    """One live network learns the four overlapping patterns in sequence with NO reset,
    reseed, rebuild, or config change between phases."""
    engine = build(CC4_CONFIG, **construction)
    phases, prev = [], None
    for pattern in order:
        ph, _ = _phase(engine, pattern, dwell)
        ph['previous_owner'] = prev
        ph['turnover_from_previous'] = bool(prev is not None and ph['owner'] != prev)
        phases.append(ph)
        prev = ph['owner']
    owners = [p['owner'] for p in phases]

    # Cold recall over the SAME network, no reset: does each pattern still map to its owner?
    recall = {}
    for pattern in order:
        engine.set_pattern(pattern)
        w = Counter()
        for _ in range(40):
            engine.step()
            if engine.winner is not None:
                w[engine.winner] += 1
        recall[pattern] = w.most_common(1)[0][0] if w else None

    return dict(
        topology='rg_direct_cc4', seed=SEED, B=CC4_B, m=CC4_M,
        eta=CC4_CONFIG['eta'], c_eta=CC4_CONFIG['c_eta'],
        e_weight_cap_frac=(construction.get('e_weight_cap_frac',
                                            DASHBOARD_CONSTRUCTION['e_weight_cap_frac'])),
        order=list(order), dwell=dwell, phases=phases,
        owner_by_pattern={p['pattern']: p['owner'] for p in phases},
        owners=owners, distinct_owners=len(set(o for o in owners if o)),
        one_to_one=bool(len(set(o for o in owners if o)) == len(owners) and None not in owners),
        turnover_every_switch=all(p['turnover_from_previous'] for p in phases[1:]),
        recall=recall,
        recall_consistent=all(recall.get(p) == o for p, o in zip(order, owners)),
    )


# ================================================== video 3: long-dwell exposure bias
def measure_long_dwell(train_dwell=800, long_dwell=6000, post_dwell=800,
                       held='row 1', switch_to='col 1', order=CANONICAL_ORDER,
                       **construction):
    """Train the four-pattern network, hold ONE pattern for a much longer dwell, then switch
    to a different pattern WITHOUT reset or rebuild, and measure the turnover latency in
    eligible presentations under a declared criterion."""
    engine = build(CC4_CONFIG, **construction)
    train = []
    prev = None
    for pattern in order:
        ph, _ = _phase(engine, pattern, train_dwell)
        ph['previous_owner'] = prev
        train.append(ph)
        prev = ph['owner']
    trained_owner_by_pattern = {p['pattern']: p['owner'] for p in train}

    # --- the long dwell on one pattern ---
    hold, hold_seq = _phase(engine, held, long_dwell)
    incumbent = hold['owner']
    hold_tail = [w for w in hold_seq[-200:] if w is not None]
    hold['tail_200_dominance'] = (round(hold_tail.count(incumbent) / len(hold_tail), 4)
                                  if hold_tail else 0.0)

    # --- switch, without reset/rebuild/reconfigure ---
    engine.set_pattern(switch_to)
    seq = []
    for _ in range(post_dwell):
        engine.step()
        seq.append(engine.winner)
    eligible = [w for w in seq if w is not None]

    # Declared criterion: the turnover presentation is the FIRST eligible presentation whose
    # winner differs from the incumbent AND which begins an unbroken run of that same winner
    # through the end of the post-switch phase.
    turnover_index = None
    new_owner = None
    for k, w in enumerate(eligible):
        if w != incumbent and all(x == w for x in eligible[k:]):
            turnover_index = k + 1          # 1-based eligible presentation
            new_owner = w
            break
    first_non_incumbent = next((k + 1 for k, w in enumerate(eligible) if w != incumbent), None)
    boundary_of_turnover = None
    if turnover_index is not None:
        seen = 0
        for b, w in enumerate(seq, start=1):
            if w is not None:
                seen += 1
                if seen == turnover_index:
                    boundary_of_turnover = b
                    break

    post_counts = Counter(w for w in eligible)
    return dict(
        topology='rg_direct_cc4', seed=SEED, B=CC4_B, m=CC4_M,
        eta=CC4_CONFIG['eta'], c_eta=CC4_CONFIG['c_eta'],
        e_weight_cap_frac=(construction.get('e_weight_cap_frac',
                                            DASHBOARD_CONSTRUCTION['e_weight_cap_frac'])),
        order=list(order), train_dwell=train_dwell, long_dwell=long_dwell,
        post_dwell=post_dwell, held_pattern=held, switched_to=switch_to,
        train_phases=train, trained_owner_by_pattern=trained_owner_by_pattern,
        long_dwell_phase=hold, incumbent=incumbent,
        post_switch_eligible_presentations=len(eligible),
        post_switch_winner_counts=dict(post_counts),
        first_non_incumbent_eligible_presentation=first_non_incumbent,
        turnover_eligible_presentation=turnover_index,
        turnover_boundary_after_switch=boundary_of_turnover,
        new_owner=new_owner,
        turnover_occurred=bool(new_owner is not None and new_owner != incumbent),
        first_presentation_instantaneous=bool(turnover_index == 1),
        consolidation_criterion=(
            'first eligible presentation (a boundary on which some competitor fired) whose '
            'winner differs from the long-dwell incumbent AND which begins an unbroken run '
            'of that same winner through the end of the post-switch phase'),
        post_switch_head=[w or '-' for w in seq[:24]],
    )


def measure_headless_long_dwell_crosscheck(long_dwell=6000, post_dwell=800, **construction):
    """The published headless protocol for comparison: a FRESH engine holds pattern 1 for the
    long dwell (no four-pattern training first), then sees pattern 2. Mirrors
    experiments/dual_fe_cc4_consolidation.py::long_dwell_stress."""
    engine = build(CC4_CONFIG, **construction)
    p1, _ = _phase(engine, CANONICAL_ORDER[0], long_dwell)
    p2, _ = _phase(engine, CANONICAL_ORDER[1], post_dwell)
    return dict(long_dwell=long_dwell, post_dwell=post_dwell,
                pattern1=CANONICAL_ORDER[0], pattern1_owner=p1['owner'],
                pattern2=CANONICAL_ORDER[1], pattern2_owner=p2['owner'],
                turnover_after_long_dwell=bool(p2['owner'] != p1['owner']),
                incumbent_locked=bool(p2['owner'] == p1['owner']))


# ================================================================= video 4: scaling
def measure_scaling():
    out = {}
    for name in ('tiled_cc', 'two_tower_composition'):
        engine = build(SCALING_CONFIG, topology=name)
        topo = engine.topology()
        meta = engine.tiled_meta
        entry = dict(
            topology=name, seed=SEED,
            nodes=len(topo['neurons']), edges=len(topo['synapses']),
            resolved_input_period=int(engine.resolved_input_period()),
            patterns=list(topo['pattern_vectors']),
            sheet_patterns=list(topo.get('tiling', {}).get('sheet_patterns', []) or []),
            input_shape=dict(meta['input_shape']), patch_shape=dict(meta['patch_shape']),
            grid_shape=dict(meta['grid_shape']),
            column_layers=[dict(l) for l in meta['column_layers']],
            column_count=len(meta['columns']),
            columns_by_layer=dict(Counter(c['layer'] for c in meta['columns'])),
        )
        out[name] = entry
    return out


def measure_two_tower_stimuli(dwell=600, glyphs=('V', 'A', '7')):
    """Drive the whole-sheet glyphs on the two-tower graph and record which L2/L3 ordinary-E
    owner each glyph produces. This is evidence for the DOCUMENTED negative result: the two
    tower L2 columns separate the glyphs, the single L3 column does not."""
    engine = build(SCALING_CONFIG, topology='two_tower_composition')
    # Ordinary competing E cells per column layer, derived from the graph's own tiling
    # metadata rather than guessed from id spelling. Eor/C/I are excluded: the claim is
    # about which ordinary-E owner each glyph recruits.
    by_layer: dict = {}
    for col in engine.tiled_meta['columns']:
        cells = [n for n in engine.order
                 if n.startswith(col['id'] + 'E') and n != col['id'] + 'Eor']
        by_layer.setdefault(col['layer'], []).extend(cells)

    owners = {layer: {} for layer in ('L2', 'L3')}
    for g in glyphs:
        engine.set_pattern(g)
        counts = {layer: Counter() for layer in owners}
        for _ in range(dwell):
            engine.step()
            for layer in owners:
                for nid in by_layer.get(layer, []):
                    if engine.spiked[nid]:
                        counts[layer][nid] += 1
        for layer in owners:
            owners[layer][g] = [n for n, _ in counts[layer].most_common(4)]

    def top(layer, g):
        v = owners[layer][g]
        return v[0] if v else None

    distinct_l2 = len({(top('L2', g)) for g in glyphs})
    distinct_l3 = len({(top('L3', g)) for g in glyphs})
    return dict(dwell=dwell, glyphs=list(glyphs),
                l2_top_owners=owners['L2'], l3_top_owners=owners['L3'],
                l2_top_owner_per_glyph={g: top('L2', g) for g in glyphs},
                l3_top_owner_per_glyph={g: top('L3', g) for g in glyphs},
                distinct_l2_top_owners=distinct_l2,
                distinct_l3_top_owners=distinct_l3,
                l3_separates_glyphs=bool(distinct_l3 == len(glyphs)))


# ===================================================================== orchestration
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default='presentation_assets/preflight_measurements.json')
    ap.add_argument('--compare-uncapped', action='store_true',
                    help="also run the CC4 protocols with the published experiment's "
                         'uncapped construction (e_weight_cap_frac=None)')
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print('[1/4] video 1 -- rg_coincidence cadence transition ...')
    cadence = measure_cadence()
    print(f"    early L1E/RG {cadence['early_emission_ratio_mean']} -> "
          f"late {cadence['late_emission_ratio_mean']}; "
          f"cadence locks at boundary {cadence['cadence_lock_boundary']}")

    print('[2/4] video 2 -- rg_direct_cc4 continuous sequential learning ...')
    continuous = measure_continuous()
    print(f"    owners {continuous['owners']} one_to_one={continuous['one_to_one']} "
          f"recall_consistent={continuous['recall_consistent']}")

    print('[3/4] video 3 -- long-dwell exposure-bias stress ...')
    long_dwell = measure_long_dwell()
    print(f"    incumbent {long_dwell['incumbent']} after {long_dwell['long_dwell']} "
          f"boundaries -> new owner {long_dwell['new_owner']} at eligible presentation "
          f"{long_dwell['turnover_eligible_presentation']}")
    crosscheck = measure_headless_long_dwell_crosscheck()
    print(f"    headless cross-check: {crosscheck['pattern1_owner']} -> "
          f"{crosscheck['pattern2_owner']} turnover={crosscheck['turnover_after_long_dwell']}")

    print('[4/4] video 4 -- scaling graphs ...')
    scaling = measure_scaling()
    for k, v in scaling.items():
        print(f"    {k}: {v['nodes']} nodes / {v['edges']} edges, "
              f"{v['input_shape']['rows']}x{v['input_shape']['cols']} sheet, "
              f"{v['column_count']} columns {v['columns_by_layer']}")
    two_tower = measure_two_tower_stimuli()
    print(f"    two-tower L3 top owners {two_tower['l3_top_owners']} "
          f"-> separates glyphs = {two_tower['l3_separates_glyphs']}")

    payload = dict(
        seed=SEED,
        dashboard_construction=DASHBOARD_CONSTRUCTION,
        cc4_config=CC4_CONFIG, coincidence_config=COINCIDENCE_CONFIG,
        scaling_config=SCALING_CONFIG,
        cadence=cadence, continuous=continuous, long_dwell=long_dwell,
        headless_long_dwell_crosscheck=crosscheck,
        scaling=scaling, two_tower_stimuli=two_tower,
    )

    if args.compare_uncapped:
        print('[+] uncapped comparison (e_weight_cap_frac=None, as in the published '
              'experiment) ...')
        payload['uncapped_comparison'] = dict(
            continuous=measure_continuous(e_weight_cap_frac=None),
            long_dwell=measure_long_dwell(e_weight_cap_frac=None),
            headless_crosscheck=measure_headless_long_dwell_crosscheck(
                e_weight_cap_frac=None))

    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nwrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
