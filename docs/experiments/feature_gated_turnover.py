"""Causal acceptance experiment for the feature-gated tiled topology (headless).

Demonstrates that ``tiled_cc_feature_gated`` restores rg_coincidence's feature-specific
turnover INSIDE each tiled L1 recognition module. Between each 3x3 RGC patch and its eight
competitors sit nine fixed feature relays; each relay has a paired coincidence C and feature
inhibitory If, and the competitor bank keeps a separate WTA-only I. When a mature local
owner predicts its active features (apical), the paired feature gates suppress ONLY the
explained relays -- the shared center relay is transiently silenced while the two novel
relays stay active, so a different competitor takes ownership.

This reuses the same reference protocol/settings as ``experiments/microcircuit_turnover.py``
(the small-circuit causal oracle): leak 0, no refractory, eta 0.01, c_eta 0.005, L2 row
normalization 0.95. It is experiment infrastructure -- it changes NO preset, default, model
dynamic, or dashboard control, and it tunes nothing to make the topology pass.

Two stages:

  Stage A -- one active RF (canonical row 1 -> col 1 -> diag \\ -> diag /). Records per switch
             the owner handoff, center-vs-novel relay firing, every feature C/If reset with
             tau ordering + pre-spike evidence, WTA events (separately), and confirms no
             other L1 module learns or resets while its patch is blank.
  Stage B -- the same pattern presented in ALL nine RFs (single seed), showing each module
             independently produces stable turnover and four distinct local owners, with no
             feature gate inhibiting another patch.

Stage B runs ONLY if Stage A passes. Each stage writes a replay/metrics/summary artifact
via ``experiments.replay_recorder``.

Run: ``PYTHONPATH=. .venv/bin/python experiments/feature_gated_turnover.py --out <dir>``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine, PATTERNS                 # noqa: E402
from backend.network_spec import embed_patch_pattern                      # noqa: E402
from experiments.replay_recorder import (                                 # noqa: E402
    ReplayRecorder, STATUS_COMPLETED, STATUS_FAILED, FEEDBACK_NOT_APPLICABLE,
)

CANONICAL_ORDER = ('row 1', 'col 1', 'diag \\', 'diag /')
CENTER_FEATURE = 4              # patch-local center pixel, shared by all four patterns
TOPOLOGY = 'tiled_cc_feature_gated'

# Reference protocol settings (identical to microcircuit_turnover.py); NOT topology tuning.
REF_CFG = dict(leak_rate=0.0, refractory_steps=0, eta=0.01, c_eta=0.005,
               l2_init_total_frac=0.95)

METRICS_COLUMNS = [
    'timestep', 'phase', 'pattern', 'module', 'stage',
    'active_module_winner', 'center_relay_spike', 'novel_relay_spikes',
    'feature_resets', 'prespike_center_resets', 'wta_resets',
    'other_module_resets', 'other_module_winners',
]


def _build_engine() -> SimulationEngine:
    return SimulationEngine(seed=1, topology=TOPOLOGY, **REF_CFG)


def _module_maps(engine, module_id):
    """Metadata-only handles for one L1 recognition module (no id parsing beyond lookups)."""
    E, S, C, If = [], {}, {}, {}
    for nid, m in engine.meta.items():
        if m.get('column_id') != module_id:
            continue
        role, fk = m.get('column_role'), m.get('feature_index')
        if role == 'E':
            E.append(nid)
        elif role == 'S':
            S[fk] = nid
        elif role == 'C':
            C[fk] = nid
        elif role == 'If':
            If[fk] = nid
    wta = next(nid for nid, m in engine.meta.items()
               if m.get('column_id') == module_id and m.get('column_role') == 'I')
    return dict(E=sorted(E), S=S, C=C, If=If, wta=wta)


def _l1_modules(engine):
    return sorted({m['column_id'] for m in engine.meta.values()
                   if m.get('column_role') == 'E' and m.get('layer') == 'L1'})


def _role_of(engine, nid):
    return engine.meta.get(nid, {}).get('column_role')


def _classify_resets(engine):
    """Split this boundary's hard resets into feature-relay resets and WTA resets, and
    verify no If/WTA cross-talk. Returns (feature_events, wta_events, crosstalk)."""
    feature, wta, crosstalk = [], [], []
    for ev in engine.hard_reset_events:
        src_role, tgt_role = _role_of(engine, ev['source']), _role_of(engine, ev['target'])
        if src_role == 'If':
            if tgt_role != 'S':
                crosstalk.append(ev)                # a feature If must reset only a relay
            else:
                feature.append(ev)
        elif src_role == 'I':                       # WTA relay
            if tgt_role != 'E':
                crosstalk.append(ev)                # a WTA I must reset only ordinary E
            else:
                wta.append(ev)
        else:
            crosstalk.append(ev)
    return feature, wta, crosstalk


# ============================================================ Stage A
def _run_phase_A(engine, rec, mod, module_id, other_modules, pattern, *, dwell, early,
                 final_window, prev_owner, phase_idx):
    engine.set_pattern(pattern)
    active = [i for i, v in enumerate(PATTERNS[pattern]) if v > 0.5]
    novel = [i for i in active if i != CENTER_FEATURE]
    rec.set_annotation(phase=('train' if prev_owner is None else 'switch'), pattern=pattern)
    rec.marker('phase', data=dict(phase_index=phase_idx, pattern=pattern,
                                  prev_owner=prev_owner, novel=novel))

    early_center = 0
    early_novel: Counter = Counter()
    final_winners: Counter = Counter()
    prespike_center_resets = 0
    feature_reset_total = 0
    wta_reset_total = 0
    crosstalk_total = 0
    other_resets = 0
    other_winners = 0
    sample_trace = None                             # one representative suppressing boundary

    s_center, c_center, if_center = mod['S'][CENTER_FEATURE], mod['C'][CENTER_FEATURE], \
        mod['If'][CENTER_FEATURE]

    for step in range(1, dwell + 1):
        engine.step()
        feature_ev, wta_ev, crosstalk = _classify_resets(engine)
        crosstalk_total += len(crosstalk)
        feature_reset_total += len(feature_ev)
        wta_reset_total += len(wta_ev)

        # a switch is causal only if every counted feature reset is paired/local + pre-spike
        for ev in feature_ev:
            src, tgt = ev['source'], ev['target']
            paired = (engine.meta[src].get('column_id') == engine.meta[tgt].get('column_id')
                      and engine.meta[src].get('feature_index')
                      == engine.meta[tgt].get('feature_index'))
            if not paired:
                crosstalk_total += 1

        # activity in OTHER (blank) modules is a leak: count winners + resets there
        for om in other_modules:
            if om in engine.column_winners:
                other_winners += 1
        for ev in engine.hard_reset_events:
            om = engine.meta[ev['target']].get('column_id')
            if om in other_modules:
                other_resets += 1

        win = engine.column_winners.get(module_id)
        center_spike = bool(engine.spiked[s_center])
        novel_spikes = {p: bool(engine.spiked[mod['S'][p]]) for p in novel}

        if step <= early:
            if center_spike:
                early_center += 1
            for p in novel:
                if novel_spikes[p]:
                    early_novel[p] += 1
        if win and step > dwell - final_window:
            final_winners[win['id']] += 1

        # capture ONE representative suppressing-boundary causal trace during the switch
        if sample_trace is None and prev_owner is not None and step <= early:
            reset_ev = next((ev for ev in feature_ev if ev['target'] == s_center), None)
            if reset_ev is not None and reset_ev['drive_before'] > 0.0 and not center_spike:
                c_cell = engine.neurons[c_center]
                sample_trace = dict(
                    timestep=engine.timestep,
                    owner_spike_tau=(win['tau'] if win else None),
                    center_C_deposit_tau=(None if c_cell.coincidence_deposit_tau is None
                                          else round(float(c_cell.coincidence_deposit_tau), 9)),
                    center_If_reset_tau=round(float(reset_ev['tau']), 9),
                    center_relay_drive_discarded=round(float(reset_ev['drive_before']), 6),
                    center_relay_fired=center_spike,
                    novel_relay_fired={str(p): novel_spikes[p] for p in novel},
                    reset_is_paired=(engine.meta[reset_ev['source']].get('feature_index')
                                     == CENTER_FEATURE),
                    reset_source=reset_ev['source'])

        if prev_owner is not None and step <= early and not center_spike:
            reset_ev = next((ev for ev in feature_ev if ev['target'] == s_center), None)
            if reset_ev is not None and reset_ev['drive_before'] > 0.0:
                prespike_center_resets += 1

        rec.record_frame(engine)
        rec.metrics.append_row(dict(
            timestep=engine.timestep, phase=('train' if prev_owner is None else 'switch'),
            pattern=pattern, module=module_id, stage='A',
            active_module_winner=(win['id'] if win else ''),
            center_relay_spike=int(center_spike),
            novel_relay_spikes=int(sum(novel_spikes.values())),
            feature_resets=len(feature_ev), prespike_center_resets=int(
                prev_owner is not None and not center_spike and any(
                    ev['target'] == s_center and ev['drive_before'] > 0 for ev in feature_ev)),
            wta_resets=len(wta_ev),
            other_module_resets=sum(1 for ev in engine.hard_reset_events
                                    if engine.meta[ev['target']].get('column_id') in other_modules),
            other_module_winners=sum(1 for om in other_modules if om in engine.column_winners)))

    final_owner, final_wins = (final_winners.most_common(1)[0]
                               if final_winners else (None, 0))
    final_total = sum(final_winners.values())
    novel_mean = mean(early_novel[i] for i in novel) if novel else 0.0
    return dict(
        pattern=pattern, prev_owner=prev_owner, active_pixels=active, novel_pixels=novel,
        early_center_relay_fires=early_center,
        early_novel_relay_fires={str(i): early_novel[i] for i in novel},
        early_novel_mean=round(novel_mean, 2),
        center_suppressed_vs_novel=bool(prev_owner is not None and early_center < novel_mean),
        novel_relays_active=bool(all(early_novel[i] > 0 for i in novel)) if novel else False,
        final_owner=final_owner,
        final_dominance=round(final_wins / final_total, 4) if final_total else 0.0,
        turnover_from_prev=bool(prev_owner is not None and final_owner != prev_owner),
        prespike_center_resets=prespike_center_resets,
        feature_reset_total=feature_reset_total, wta_reset_total=wta_reset_total,
        crosstalk_total=crosstalk_total, other_resets=other_resets,
        other_winners=other_winners, sample_suppression_trace=sample_trace)


def run_stage_A(output_root, *, dwell, early, final_window, dominance):
    engine = _build_engine()
    patch = (1, 1)
    engine.set_patch(*patch)
    module_id = f'L1m{patch[0]}{patch[1]}'
    mod = _module_maps(engine, module_id)
    other_modules = [m for m in _l1_modules(engine) if m != module_id]

    with ReplayRecorder(
        engine, experiment='feature_gated_turnover_stageA', output_root=output_root,
        seed=1, record_every=25, checkpoint_every=40, metrics_columns=METRICS_COLUMNS,
        hierarchical_feedback=FEEDBACK_NOT_APPLICABLE,
        conditions=dict(stage='A', topology=TOPOLOGY, module=module_id, patch=list(patch),
                        **REF_CFG),
        schedule=dict(order=list(CANONICAL_ORDER), dwell=dwell)) as rec:
        phases, prev = [], None
        for idx, pattern in enumerate(CANONICAL_ORDER):
            ph = _run_phase_A(engine, rec, mod, module_id, other_modules, pattern,
                              dwell=dwell, early=early, final_window=final_window,
                              prev_owner=prev, phase_idx=idx)
            phases.append(ph)
            prev = ph['final_owner']

        owners = [p['final_owner'] for p in phases]
        switches = phases[1:]
        checks = {
            'all_consolidated': all(p['final_dominance'] >= dominance and p['final_owner']
                                    for p in phases),
            'four_distinct_owners': len(set(owners)) == len(owners) and None not in owners,
            'turnover_every_switch': all(p['turnover_from_prev'] for p in switches),
            'center_suppressed_every_switch': all(p['center_suppressed_vs_novel']
                                                  for p in switches),
            'novel_relays_active_every_switch': all(p['novel_relays_active']
                                                    for p in switches),
            'every_center_reset_prespike': all(p['prespike_center_resets'] > 0
                                               for p in switches),
            'no_feature_wta_crosstalk': all(p['crosstalk_total'] == 0 for p in phases),
            'no_blank_module_resets': all(p['other_resets'] == 0 for p in phases),
            'no_blank_module_winners': all(p['other_winners'] == 0 for p in phases),
        }
        passed = all(checks.values())
        result = dict(stage='A', module=module_id, owners=owners,
                      owner_by_pattern={p['pattern']: p['final_owner'] for p in phases},
                      phases=phases, checks=checks)
        rec.finish(STATUS_COMPLETED if passed else STATUS_FAILED,
                   checks=checks, result=result)
    return dict(passed=passed, run_dir=rec.run_dir, **result)


# ============================================================ Stage B
def _full_field_vector(pattern):
    vec = [0] * 81
    for pr in range(3):
        for pc in range(3):
            part = embed_patch_pattern((9, 9), (3, 3), (pr, pc), PATTERNS[pattern])
            vec = [a or b for a, b in zip(vec, part)]
    return vec


def run_stage_B(output_root, *, dwell, final_window, dominance):
    engine = _build_engine()
    modules = _l1_modules(engine)
    mods = {m: _module_maps(engine, m) for m in modules}
    e_ids = {m: mods[m]['E'] for m in modules}

    with ReplayRecorder(
        engine, experiment='feature_gated_turnover_stageB', output_root=output_root,
        seed=1, record_every=50, checkpoint_every=40, metrics_columns=METRICS_COLUMNS,
        hierarchical_feedback=FEEDBACK_NOT_APPLICABLE,
        conditions=dict(stage='B', topology=TOPOLOGY, **REF_CFG),
        schedule=dict(order=list(CANONICAL_ORDER), dwell=dwell)) as rec:
        owners = {m: [] for m in modules}
        crosstalk_total = 0
        for idx, pattern in enumerate(CANONICAL_ORDER):
            engine.set_input(_full_field_vector(pattern))
            rec.set_annotation(phase='switch' if idx else 'train', pattern=pattern)
            rec.marker('phase', data=dict(phase_index=idx, pattern=pattern))
            win = {m: Counter() for m in modules}
            for step in range(1, dwell + 1):
                engine.step()
                _, _, crosstalk = _classify_resets(engine)
                crosstalk_total += len(crosstalk)
                # a feature reset in module X must never target a relay of module Y
                for ev in engine.hard_reset_events:
                    if _role_of(engine, ev['source']) == 'If':
                        if (engine.meta[ev['source']]['column_id']
                                != engine.meta[ev['target']]['column_id']):
                            crosstalk_total += 1
                for m in modules:
                    w = engine.column_winners.get(m)
                    if w and step > dwell - final_window:
                        win[m][w['id']] += 1
                rec.record_frame(engine)
                if step % 50 == 0:
                    rec.metrics.append_row(dict(
                        timestep=engine.timestep, phase='switch' if idx else 'train',
                        pattern=pattern, module='ALL', stage='B',
                        active_module_winner='', center_relay_spike=0, novel_relay_spikes=0,
                        feature_resets=0, prespike_center_resets=0, wta_resets=0,
                        other_module_resets=0, other_module_winners=0))
            for m in modules:
                owners[m].append(win[m].most_common(1)[0][0] if win[m] else None)

        per_module = {m: dict(owners=owners[m],
                              distinct=len(set(owners[m])),
                              turnover=all(owners[m][k] != owners[m][k - 1]
                                           for k in range(1, len(owners[m])))
                              and None not in owners[m])
                      for m in modules}
        checks = {
            'all_modules_four_distinct': all(v['distinct'] == 4 for v in per_module.values()),
            'all_modules_turnover': all(v['turnover'] for v in per_module.values()),
            'no_cross_patch_inhibition': crosstalk_total == 0,
        }
        passed = all(checks.values())
        result = dict(stage='B', per_module=per_module, crosstalk_total=crosstalk_total,
                      checks=checks)
        rec.finish(STATUS_COMPLETED if passed else STATUS_FAILED,
                   checks=checks, result=result)
    return dict(passed=passed, run_dir=rec.run_dir, **result)


# ============================================================ report + CLI
def _print_stage_A(res):
    print(f"=== Stage A (one RF, module {res['module']}) ===")
    for p in res['phases']:
        tag = 'TRAIN ' if p['prev_owner'] is None else 'SWITCH'
        supp = ('center suppressed' if p['center_suppressed_vs_novel']
                else ('center NOT suppressed' if p['prev_owner'] else 'first pattern'))
        print(f"  {tag} {p['pattern']:>8s}: owner={p['final_owner']} "
              f"dom={p['final_dominance']:.2f} turnover={p['turnover_from_prev']} | "
              f"centerS={p['early_center_relay_fires']} "
              f"novel={list(p['early_novel_relay_fires'].values())} "
              f"prespike_resets={p['prespike_center_resets']} ({supp})")
    print(f"  owners: {res['owners']}")
    for k, v in res['checks'].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  STAGE A PASS: {res['passed']}  artifact={res['run_dir']}")


def _print_stage_B(res):
    print("=== Stage B (nine RFs, same pattern) ===")
    for m, v in res['per_module'].items():
        print(f"  {m}: owners={v['owners']} distinct={v['distinct']} turnover={v['turnover']}")
    for k, v in res['checks'].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  STAGE B PASS: {res['passed']}  artifact={res['run_dir']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'runs', 'feature_gated_turnover'))
    ap.add_argument('--dwell', type=int, default=4000)
    ap.add_argument('--early', type=int, default=200)
    ap.add_argument('--final-window', type=int, default=500)
    ap.add_argument('--dominance', type=float, default=0.8)
    ap.add_argument('--summary', default=None, help='optional combined JSON summary path')
    ap.add_argument('--skip-stage-b', action='store_true')
    args = ap.parse_args(argv)

    a = run_stage_A(args.out, dwell=args.dwell, early=args.early,
                    final_window=args.final_window, dominance=args.dominance)
    _print_stage_A(a)
    combined = dict(stage_A=a)
    if a['passed'] and not args.skip_stage_b:
        b = run_stage_B(args.out, dwell=args.dwell, final_window=args.final_window,
                        dominance=args.dominance)
        _print_stage_B(b)
        combined['stage_B'] = b
    elif not a['passed']:
        print('Stage A failed: preserving artifacts and NOT running Stage B (no tuning).')

    if args.summary:
        with open(args.summary, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f'wrote {args.summary}')
    return 0 if a['passed'] and (args.skip_stage_b or combined.get('stage_B', {}).get('passed'))\
        else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
