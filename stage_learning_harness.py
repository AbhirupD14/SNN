"""
Staged learning harness for the MINIMAL signed-spike experiment.

See Claude_Minimal_Signed_Spike_Learning_Prompt.md. The minimal model strips the
compensating stack down to the core local loop:

    charge accumulates -> neuron fires -> fire updates local feedforward weights
    (signed: active inputs +1 potentiate, inactive -1 depress; NO budget) ->
    learned L2I lateral inhibition regulates competition -> L1I feedback
    inhibition regulates input -> repeat.  refractory = 0 (inhibition, not a hard
    lockout, regulates frequency).

We do NOT jump straight to 8 patterns; we grow the problem in 5 stages and find
where ownership first collapses. Ownership is measured the honest way -- visit
level, not per-cycle -- and swept over dwell length because a 1-cycle visit gives
a misleadingly perfect first-winner map.

Metric per stage (learning stays ON during eval, so long-dwell corruption shows):
  owner(P)       = most common early (first) winner across P's repeated visits
  consistency(P) = fraction of P's visits the owner won
  + distinct owners / N, collisions, dead L2E, firers/visit, L2I & L1I spike
    counts, and RF-match (cosine of the owner's feedforward weights with P).

Scale convention (Option A -- current fixed-point UNIT=1000, documented):
  threshold_l2 = 8*UNIT, L2E weight_cap = thr_l2, floor = L2E_MIN_WEIGHT_FLOOR,
  and (1 - (w/w_cap)^2) uses the LINEAR cap w_cap (not a squared denominator).
  All charge/threshold/weight magnitudes share the linear UNIT scale; p, signal,
  leaks are dimensionless. Nothing mixes old small-float with UNIT-scaled values.

    PYTHONPATH=. .venv/bin/python stage_learning_harness.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


# The minimal experiment: signed-spike learning ON, every compensating mechanism
# OFF, no budget, no refractory lockout. Lateral (L2I) and feedback (L1I)
# inhibition stay active (they are structural, not toggled here).
MINIMAL = dict(
    signed_spike_learning=True,
    confidence_consolidation=False,
    loser_depression=False,
    signed_depression=False,
    homeostasis=False,
    subtractive_reset=False,
    lasting_inhibition=False,
    event_driven=False,
    v_sat_frac=None,        # membrane saturation off
    l2e_budget=False,       # NO positive-weight budget
    refractory=0,           # inhibition regulates frequency, not a hard lockout
    # Capacity rule: per-afferent cap = thr/3 (three strong active afferents reach
    # threshold), positive floor = 1, each I threshold = its E threshold / 3.
    l2e_weight_cap_frac=1 / 3,
    pos_weight_floor=1,
    l2i_threshold_frac=1 / 3,
    l1i_threshold_frac=1 / 3,
)

STAGES = [
    ("1: one pattern",            ["row 0"]),
    ("2: two disjoint patterns",  ["row 0", "row 2"]),
    ("3: rows only",              ["row 0", "row 1", "row 2"]),
    ("4: rows + columns",         ["row 0", "row 1", "row 2", "col 0", "col 1", "col 2"]),
    ("5: all eight",              list(PATTERNS.keys())),
]

DWELLS = [1, 2, 4, 8, 16, 40]
TRAIN_ROUNDS = 40
EVAL_ROUNDS = 12
SEEDS = (1, 2, 3, 4)


def _ff(engine, j):
    """Owner j's feedforward (pixel) weight vector -- afferent indices 1.. (index
    0 is the L2I->L2E inhibitory gate)."""
    return engine.l2.excitatory_neurons[j]._weights_array[1:1 + len(PATTERNS['row 0'])]


def _rf_match(engine, j, pattern_name):
    """Cosine similarity between owner j's feedforward RF and the pattern's binary
    active-pixel vector. 1.0 == RF sits exactly on the pattern's pixels."""
    w = np.asarray(_ff(engine, j), dtype=float)
    b = np.asarray(PATTERNS[pattern_name], dtype=float)
    nw, nb = np.linalg.norm(w), np.linalg.norm(b)
    return float(w.dot(b) / (nw * nb)) if nw > 0 and nb > 0 else 0.0


def measure(stage_patterns, dwell, seed):
    e = SimulationEngine(seed=seed, **MINIMAL)
    steps = e.params['cycle_period']
    names = stage_patterns
    N = len(names)

    # Train: interleave the stage's patterns.
    for _ in range(TRAIN_ROUNDS):
        for nm in names:
            e.set_pattern(nm)
            for _ in range(steps):
                e.step()

    l2i_keys = [k for k in e.spiked if k.startswith('L2I')]
    l1i_keys = [k for k in e.spiked if k.startswith('L1I')]

    hist = {nm: Counter() for nm in names}
    firers_per_visit = []
    l2i_spikes = l1i_spikes = 0
    for _ in range(EVAL_ROUNDS):
        for nm in names:
            e.set_pattern(nm)
            first = None
            firers = set()
            for _ in range(steps * dwell):
                e.step()
                l2i_spikes += sum(1 for k in l2i_keys if e.spiked[k])
                l1i_spikes += sum(1 for k in l1i_keys if e.spiked[k])
                for j in range(N_OUT):
                    if e.spiked[f'L2E{j}']:
                        firers.add(j)
                        if first is None:
                            first = j
                        break
            firers_per_visit.append(len(firers))
            if first is not None:
                hist[nm][first] += 1

    owner, cons, rf = {}, {}, []
    for nm in names:
        c = hist[nm]
        if c:
            top, top_n = c.most_common(1)[0]
            owner[nm] = top
            cons[nm] = top_n / sum(c.values())
            rf.append(_rf_match(e, top, nm))
        else:
            owner[nm] = None
            cons[nm] = 0.0
            rf.append(0.0)

    owners = [o for o in owner.values() if o is not None]
    distinct = len(set(owners))
    collisions = sum(n for n in Counter(owners).values() if n > 1)
    # "dead" = L2E units that never own a stage pattern AND are not a distinct
    # owner; report units that never fired as a stage owner out of the pool.
    dead = N_OUT - len(set(owners))
    return dict(N=N, owner=owner, cons=cons,
                distinct=distinct, mean_cons=float(np.mean(list(cons.values()))),
                collisions=collisions, dead_vs_pool=dead,
                firers=float(np.mean(firers_per_visit)),
                l2i=l2i_spikes, l1i=l1i_spikes, rf=float(np.mean(rf)))


def _agg(stage_patterns, dwell):
    res = [measure(stage_patterns, dwell, s) for s in SEEDS]
    n = len(res)
    keys = ('distinct', 'mean_cons', 'collisions', 'dead_vs_pool',
            'firers', 'l2i', 'l1i', 'rf')
    return {k: sum(r[k] for r in res) / n for k in keys}, res


def main():
    print("MINIMAL signed-spike experiment | scale: UNIT-scaled (Option A) | "
          "refractory=0 | no budget | seeds 1-4\n")

    # 1) Per-stage snapshot at a representative dwell.
    D = 4
    print(f"================ per-stage @ dwell={D} cycles (mean over seeds) ================")
    print(f"  {'stage':<26} {'distinct':>9} {'consist':>8} {'collis':>7} "
          f"{'dead':>5} {'firers':>7} {'L2I':>6} {'L1I':>6} {'RFmatch':>8}")
    first_distinct_fail = first_cons_fail = None
    for label, pats in STAGES:
        a, _ = _agg(pats, D)
        N = len(pats)
        print(f"  {label:<26} {a['distinct']:>6.2f}/{N:<2} {a['mean_cons']:>8.3f} "
              f"{a['collisions']:>7.2f} {a['dead_vs_pool']:>5.2f} {a['firers']:>7.2f} "
              f"{a['l2i']:>6.0f} {a['l1i']:>6.0f} {a['rf']:>8.3f}")
        if first_distinct_fail is None and a['distinct'] < N - 0.5:
            first_distinct_fail = label           # patterns start sharing owners
        if first_cons_fail is None and a['mean_cons'] < 0.75:
            first_cons_fail = label                # no single stable owner per pattern

    # 2) Dwell sensitivity per stage.
    print(f"\n================ dwell sweep: mean consistency (distinct/N) ================")
    header = "  {:<26}".format("stage") + "".join(f"{('d='+str(d)):>12}" for d in DWELLS)
    print(header)
    for label, pats in STAGES:
        N = len(pats)
        cells = []
        for d in DWELLS:
            a, _ = _agg(pats, d)
            cells.append(f"{a['mean_cons']:.2f}({a['distinct']:.1f}/{N})")
        print("  {:<26}".format(label) + "".join(f"{c:>12}" for c in cells))

    print("\n================ verdict ================")
    print(f"  Distinct-owner collapse (patterns share an owner): "
          f"{first_distinct_fail or 'none through stage 5'}")
    print(f"  Per-pattern owner instability (consistency < 0.75): "
          f"{first_cons_fail or 'none through stage 5'}")
    print("  Note: 'dead' counts pool units that are not stage owners, so it is")
    print("  dominated by N_OUT-N at small stages -- read it together with 'firers'.")
    print("  Does signed depression replace the budget? Compare RFmatch/dead above to")
    print("  the budgeted baseline (visit_consistency.py). Dwell sweep shows whether")
    print("  long holds corrupt ownership under always-on plasticity.")


if __name__ == "__main__":
    main()
