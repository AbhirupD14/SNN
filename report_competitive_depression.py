"""Multi-seed experimental report for the L2 hard-reset competitive-depression
architecture (L2_Hard_Reset_Competitive_Depression_Spec.md, Section 11).

Runs the DASHBOARD configuration (backend.api engine params) over several fixed
seeds WITHOUT changing the algorithm between seeds, and reports for each seed and
condition:

  - L2E neurons that fired at least once, and the dead-L2E count;
  - distinct sustained winners across the four patterns;
  - sustained dominance per pattern (modal winner's fraction of held cycles);
  - center-pixel (index 4) weight distribution across the 8 L2E;
  - number and mean magnitude of competitive-depression events during measurement.

Two conditions, identical except for the ablation switch:
  - competitive: loser_depression = True  (the canonical L2I-event depression);
  - hard-reset baseline: loser_depression = False (pre-change: reset only, no
    competitive depression).

This is DIAGNOSTIC. It introduces no task-specific weights and no pattern-aware
exceptions; any center-pixel protection must emerge from the same task-independent
rule.

    PYTHONPATH=. .venv/bin/python report_competitive_depression.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.api import engine          # dashboard-configured engine (source of config)
from backend.simulation import PATTERNS, N_OUT, N_PIX

SEEDS = (1, 2, 3, 4, 5)
TRAIN_EPOCHS = 60
HOLD_CYCLES = 40
CENTER_PIX = 4                          # the shared center pixel of the 4 line patterns


def _train_interleaved(steps, epochs):
    for _ in range(epochs):
        for name in PATTERNS:
            engine.set_pattern(name)
            for _ in range(steps):
                engine.step()


def _sustained(name, steps, hold_cycles, dep_acc):
    """Hold `name` for steps*hold_cycles steps; return per-cycle winners. Also
    accumulate competitive-depression stats into dep_acc = [count, sum_abs_delta]."""
    engine.set_pattern(name)
    winners: list[int] = []
    for _ in range(steps * hold_cycles):
        engine.step()
        for nid, rec in engine._reset_events:
            if rec['depressed_indices']:
                dep_acc[0] += len(rec['depressed_indices'])
                dep_acc[1] += float(np.abs(rec['delta_weights']).sum())
        for j in range(N_OUT):
            if engine.spiked[f'L2E{j}']:
                winners.append(j)
                break
    return winners


def measure(seed, competitive):
    engine.apply_config({'seed': seed, 'loser_depression': bool(competitive)})
    steps = max(1, engine.params['cycle_period'])
    _train_interleaved(steps, TRAIN_EPOCHS)

    fired = set()
    per_pattern = {}
    dep_acc = [0, 0.0]          # [event count, summed |delta_w|]
    for name in PATTERNS:
        winners = _sustained(name, steps, HOLD_CYCLES, dep_acc)
        fired.update(winners)
        if winners:
            modal, cnt = Counter(winners).most_common(1)[0]
            per_pattern[name] = (modal, cnt / len(winners))
        else:
            per_pattern[name] = (None, 0.0)

    center = [float(engine.l2.excitatory_neurons[j]._weights_array[CENTER_PIX])
              for j in range(N_OUT)]
    modal_winners = [m for (m, _) in per_pattern.values() if m is not None]
    return dict(
        fired=len(fired),
        dead=N_OUT - len(fired),
        distinct_winners=len(set(modal_winners)),
        per_pattern=per_pattern,
        mean_dominance=float(np.mean([d for (_, d) in per_pattern.values()])),
        center=center,
        dep_events=dep_acc[0],
        dep_mean_mag=(dep_acc[1] / dep_acc[0]) if dep_acc[0] else 0.0,
    )


def _fmt_row(seed, r):
    pp = " ".join(f"{n[:4]}:{(m if m is not None else '-')}"
                  f"/{d:.2f}" for n, (m, d) in r['per_pattern'].items())
    cen = "[" + " ".join(f"{c:.0f}" for c in r['center']) + "]"
    return (f"seed {seed}: fired={r['fired']}/8 dead={r['dead']} "
            f"distinct={r['distinct_winners']}/4 mean_dom={r['mean_dominance']:.2f} "
            f"dep_events={r['dep_events']} dep_mag~{r['dep_mean_mag']:.1f}\n"
            f"          per-pattern(win/dom): {pp}\n"
            f"          center-pixel weights: {cen}")


def main():
    print("=" * 78)
    print("L2 hard-reset competitive-depression -- multi-seed report")
    print(f"seeds={SEEDS} train_epochs={TRAIN_EPOCHS} hold_cycles={HOLD_CYCLES}")
    print("=" * 78)
    for label, competitive in (("COMPETITIVE (loser_depression=True)", True),
                               ("HARD-RESET BASELINE (loser_depression=False)", False)):
        print(f"\n--- {label} ---")
        agg = []
        for seed in SEEDS:
            r = measure(seed, competitive)
            agg.append(r)
            print(_fmt_row(seed, r))
        print(f"  MEAN over seeds: fired={np.mean([a['fired'] for a in agg]):.1f}/8 "
              f"dead={np.mean([a['dead'] for a in agg]):.1f} "
              f"distinct={np.mean([a['distinct_winners'] for a in agg]):.2f}/4 "
              f"mean_dom={np.mean([a['mean_dominance'] for a in agg]):.3f} "
              f"dep_events={np.mean([a['dep_events'] for a in agg]):.0f}")


if __name__ == "__main__":
    main()
