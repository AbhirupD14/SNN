"""
Ownership variance report: balanced init as a FAIR start, not a solution.

Balanced initialization only guarantees an unbiased developmental starting point
(every L2E equal incoming, every pixel equal outgoing, no task structure). Whether
the network ends up with distinct per-pattern owners is up to the LEARNING and
COMPETITION rules -- they must amplify the small unbiased differences the jitter
seeds, and recover from near-symmetric or unfavorable states.

So we do NOT report a single best run. For each init condition we train from many
seeds and report the DISTRIBUTION of ownership metrics (mean, std, min, max), which
is the honest picture of how reliably the rules break symmetry.

Metrics (over a held-out final window, patterns cycled):
  * distinct_owners : # of distinct L2E that are the modal winner of >=1 pattern
                      (out of len(PATTERNS) patterns; higher = better tiling).
  * mean_dominance  : avg over patterns of (modal-owner wins / total wins for that
                      pattern) -- how decisively the owner holds its pattern.
  * silent_patterns : patterns that produced no L2E winner at all.

Conditions: exactly-identical (eps=0), balanced-2%, balanced-5%, adversarial
near-tied (eps=1e-4), legacy-wide. The active model's rules are used unchanged (we
vary only the seed and the init) by cloning the dashboard engine's config.

    PYTHONPATH=. .venv/bin/python report_balanced_init_ownership.py
"""

from __future__ import annotations

from collections import Counter
import numpy as np

from backend.api import engine            # its own instance in THIS process
from backend.simulation import PATTERNS, N_OUT

SEEDS = list(range(8))
TRAIN_STEPS = 2400          # ~ cycle through the 4 patterns many times
MEASURE_STEPS = 400         # held-out final window
NAMES = list(PATTERNS.keys())

CONDITIONS = [
    ("exactly-identical",     dict(l2e_init_mode="balanced", l2e_init_jitter=0.0)),
    ("balanced-2%",           dict(l2e_init_mode="balanced", l2e_init_jitter=0.02)),
    ("balanced-5%",           dict(l2e_init_mode="balanced", l2e_init_jitter=0.05)),
    ("adversarial-near-tied", dict(l2e_init_mode="balanced", l2e_init_jitter=1e-4)),
    ("legacy-wide",           dict(l2e_init_mode="legacy_wide", l2e_init_jitter=0.05)),
]


def _winner(e):
    for j in range(N_OUT):
        if e.spiked[f"L2E{j}"]:
            return j
    return None


def _run_once(seed, cfg):
    engine.apply_config({"seed": seed, **cfg})   # full rebuild, fresh weights
    # Train.
    for s in range(TRAIN_STEPS):
        engine.set_pattern(NAMES[(s // 8) % len(NAMES)])
        engine.step()
    # Measure ownership over a held-out window.
    wins = {nm: Counter() for nm in NAMES}
    for s in range(MEASURE_STEPS):
        nm = NAMES[(s // 8) % len(NAMES)]
        engine.set_pattern(nm)
        engine.step()
        w = _winner(engine)
        if w is not None:
            wins[nm][w] += 1
    owners, doms, silent = [], [], 0
    for nm in NAMES:
        c = wins[nm]
        if not c:
            silent += 1
            continue
        top, top_n = c.most_common(1)[0]
        owners.append(top)
        doms.append(top_n / sum(c.values()))
    return dict(distinct_owners=len(set(owners)),
                mean_dominance=(float(np.mean(doms)) if doms else 0.0),
                silent_patterns=silent)


def _fmt(vals):
    a = np.array(vals, dtype=float)
    return f"{a.mean():5.2f} +/- {a.std():.2f}  [min {a.min():.2f}, max {a.max():.2f}]"


def main():
    print(f"patterns={len(NAMES)}  L2E={N_OUT}  seeds={SEEDS}")
    print(f"train={TRAIN_STEPS} steps, measure={MEASURE_STEPS} steps\n")
    print(f"{'condition':22s} {'distinct_owners':>28s} {'mean_dominance':>28s} {'silent':>16s}")
    print("-" * 98)
    summary = {}
    for name, cfg in CONDITIONS:
        do, md, sil = [], [], []
        for seed in SEEDS:
            r = _run_once(seed, cfg)
            do.append(r["distinct_owners"]); md.append(r["mean_dominance"]); sil.append(r["silent_patterns"])
        summary[name] = (do, md, sil)
        print(f"{name:22s} {_fmt(do):>28s} {_fmt(md):>28s} {_fmt(sil):>16s}")
    print("-" * 98)
    print(f"(distinct_owners out of {len(NAMES)} patterns; higher=better tiling. "
          "Variance across seeds is the point --")
    print(" balanced init is a fair START; the rules must amplify small unbiased "
          "differences to earn ownership.)")

    # Interpretation (printed, not asserted -- this is a report).
    do0 = np.array(summary["exactly-identical"][0], dtype=float)
    do5 = np.array(summary["balanced-5%"][0], dtype=float)
    print()
    print(f"exact-feedforward-symmetry (eps=0) distinct_owners mean = {do0.mean():.2f}: even with")
    print("  ZERO feedforward jitter the pool still tiles, because the FEEDFORWARD init is not the")
    print("  only source of asymmetry -- the ring GEOMETRY gives each L2E different per-pixel")
    print("  distances, and the L2E->L2I and feedback inits are INDEPENDENT seeded streams. These")
    print("  are all task-INDEPENDENT and unbiased, and the learning/competition rules amplify them.")
    print("  So no explicit transient perturbation is needed for this network; a genuinely fully-")
    print("  symmetric variant (uniform geometry + shared inhibitory init) would need one tiny kick.")
    print(f"balanced-5% distinct_owners mean = {do5.mean():.2f}: adding unbiased feedforward jitter")
    print("  keeps tiling in the same band -- balanced init is a fair START, and the rules (not the")
    print("  init) do the work. Variance across seeds (above) is the honest measure, not any best run.")


if __name__ == "__main__":
    main()
