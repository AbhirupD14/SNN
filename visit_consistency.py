"""
Visit-level ownership CONSISTENCY harness -- the honest metric for repeatable
one-to-one pattern ownership.

The goal of this project is NOT that one neuron dominates every cycle while a
pattern is held (that is sustained dominance -- see sustained_dominance.py, and
it is explicitly NOT required). The goal is REPEATABLE OWNERSHIP across
presentations: show P1 -> neuron A wins; later show P1 again -> A wins again;
each of the 8 line patterns should have a stable, distinct owner.

So the metric here is measured PER VISIT, not per cycle:
  - Train by cycling the 8 patterns for many visits (learning stays on).
  - Evaluate by cycling the 8 patterns for EVAL_ROUNDS more rounds (learning
    stays on -- ownership must survive ongoing plasticity, which is the point).
  - For each visit, record the FIRST L2E winner after the pattern switch
    (first L2E to spike within an early window of VISIT_CYCLES cycles).
  - owner(P)       = most common winner across P's visits
    consistency(P) = fraction of P's visits that the owner won.

Reports pattern->owner, per-pattern consistency, mean consistency, distinct
owners/8, collisions (owners shared by >1 pattern), and -- secondary only --
dead L2E (never won any visit).

The model uses NO labels and NO pattern->winner table; ownership is read out
externally here purely for scoring. Learning/competition stay local.

Also reports peak L2E membrane at competition time (as a multiple of threshold):
with an unbounded membrane the accumulated charge ratchets to ~2-3x threshold, so
the small capped L2I->L2E gate cannot regulate it; the v_sat conditions bound it
near threshold so inhibition becomes the frequency regulator.

Compares (seeds 1-4):
  refractory=2 (baseline)      |  refractory=0
  refractory=2 + v_sat         |  refractory=0 + v_sat   (charge scaled near thr,
                                                          inhibition regulates)

    PYTHONPATH=. .venv/bin/python visit_consistency.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


TRAIN_ROUNDS = 60      # training sweeps over all 8 patterns
EVAL_ROUNDS = 25       # evaluation visits per pattern
VISIT_CYCLES = 3       # early window (in cycles) in which the first winner is taken
SEEDS = (1, 2, 3, 4)


def _train_interleaved(engine, names, steps, rounds):
    for _ in range(rounds):
        for name in names:
            engine.set_pattern(name)
            for _ in range(steps):
                engine.step()


def _freeze_plasticity(engine):
    """Turn OFF every local learning rule after training so evaluation is a pure
    readout of the trained network. The COMPETITION dynamics stay fully intact --
    apply_inhibition still discharges the membrane (with eta=0 the gate just does
    not further learn), leak/refractory/v_sat still act -- so who-fires-first is
    still decided by the same mechanism, only the weights no longer move. Without
    this, dwelling on a pattern during eval lets the round-robin teach the pattern
    to rivals (learning is otherwise always on), which corrupts the very ownership
    we are trying to measure."""
    for n in engine.neurons.values():
        n.learning_rate = 0.0
        n.inhibitory_learning_rate = 0.0
        n.signed_depression = False
        n.loser_depression = False
        n.confidence_consolidation = False
        n.homeostasis = False
        n.eta_off = 0.0
        n.eta_loss = 0.0


def _first_winner(engine, name, steps, visit_cycles, peaks):
    """Switch to `name` and return the FIRST L2E to spike within the early window
    (visit_cycles cycles). None if the pool stays silent through the window.
    Learning stays on -- we do not freeze or reset membranes, so this is the
    honest 'which neuron grabs this pattern on presentation' readout. Also record
    the peak competition-time L2E membrane seen (engine.l2_drive) into `peaks`."""
    engine.set_pattern(name)
    winner = None
    for _ in range(steps * visit_cycles):
        engine.step()
        peaks.append(max(engine.l2_drive.values()))
        if winner is None:
            for j in range(N_OUT):
                if engine.spiked[f'L2E{j}']:
                    winner = j
                    break
    return winner


def measure(seed, refractory, v_sat_frac=None, freeze=True,
            train_rounds=TRAIN_ROUNDS, eval_rounds=EVAL_ROUNDS):
    names = list(PATTERNS.keys())
    e = SimulationEngine(seed=seed, refractory=refractory, v_sat_frac=v_sat_frac)
    steps = e.params['cycle_period']
    thr = e.params['threshold_l2']

    _train_interleaved(e, names, steps, train_rounds)
    if freeze:
        _freeze_plasticity(e)

    # Evaluation: cycle the 8 patterns eval_rounds times, first-winner per visit.
    hist: dict[str, Counter] = {n: Counter() for n in names}
    peaks: list[float] = []
    for _ in range(eval_rounds):
        for name in names:
            w = _first_winner(e, name, steps, VISIT_CYCLES, peaks)
            if w is not None:
                hist[name][w] += 1

    owner: dict[str, int | None] = {}
    consistency: dict[str, float] = {}
    for name in names:
        c = hist[name]
        if c:
            top, top_n = c.most_common(1)[0]
            owner[name] = top
            consistency[name] = top_n / sum(c.values())
        else:
            owner[name] = None
            consistency[name] = 0.0

    owners = [o for o in owner.values() if o is not None]
    distinct = len(set(owners))
    # collisions: how many patterns share their owner with another pattern.
    owner_counts = Counter(owners)
    collisions = sum(n for n in owner_counts.values() if n > 1)
    won_any = set(owners)
    dead = N_OUT - len(won_any)
    mean_cons = float(np.mean(list(consistency.values())))
    peak_arr = np.array(peaks) / thr if peaks else np.array([0.0])
    return dict(seed=seed, owner=owner, consistency=consistency,
                distinct=distinct, mean_cons=mean_cons,
                collisions=collisions, dead=dead,
                peak_mean=float(peak_arr.mean()), peak_max=float(peak_arr.max()))


def _print_condition(tag, results):
    print(f"\n===== {tag} =====")
    for m in results:
        per = "  ".join(f"{n}->L2E{m['owner'][n]}:{m['consistency'][n]:.2f}"
                        for n in m['owner'])
        print(f"  seed {m['seed']}: distinct={m['distinct']}/8  "
              f"mean_consistency={m['mean_cons']:.3f}  "
              f"collisions={m['collisions']}  dead={m['dead']}  "
              f"peak_V={m['peak_mean']:.2f}x/{m['peak_max']:.2f}x thr")
        print(f"           {per}")
    n = len(results)
    agg = {k: sum(m[k] for m in results) / n
           for k in ('distinct', 'mean_cons', 'collisions', 'dead',
                     'peak_mean', 'peak_max')}
    print(f"  MEAN: distinct={agg['distinct']:.2f}/8  consistency={agg['mean_cons']:.3f}  "
          f"collisions={agg['collisions']:.2f}  dead={agg['dead']:.2f}  "
          f"peak_V(mean/max)={agg['peak_mean']:.2f}x/{agg['peak_max']:.2f}x thr")
    return agg


def main():
    # PRIMARY: ownership readout with plasticity FROZEN after training (dwell can't
    # corrupt it), so this isolates whether the trained network maps each pattern
    # to a stable, distinct owner and whether refractory / v_sat change that.
    conditions = [
        ("refractory=2 (baseline)",         dict(refractory=2)),
        ("refractory=0",                    dict(refractory=0)),
        ("refractory=2 + v_sat=1.5x",       dict(refractory=2, v_sat_frac=1.5)),
        ("refractory=0 + v_sat=1.5x",       dict(refractory=0, v_sat_frac=1.5)),
    ]
    print("################ FROZEN eval (pure ownership readout) ################")
    summary = []
    for tag, kw in conditions:
        res = [measure(s, freeze=True, **kw) for s in SEEDS]
        summary.append((tag, _print_condition(tag, res)))

    print("\n===== summary: FROZEN eval (mean over seeds 1-4) =====")
    print(f"  {'condition':<30} {'distinct':>9} {'consist':>8} "
          f"{'collis':>7} {'dead':>5} {'peakV(mn/mx)':>16}")
    for tag, a in summary:
        print(f"  {tag:<30} {a['distinct']:>7.2f}/8 {a['mean_cons']:>8.3f} "
              f"{a['collisions']:>7.2f} {a['dead']:>5.2f} "
              f"{a['peak_mean']:>7.2f}x/{a['peak_max']:.2f}x")

    # SECONDARY: same readout but with plasticity left ON during eval, to show the
    # dwell-driven drift -- holding a pattern lets the round-robin teach it to
    # rivals, which is why a live dashboard should present briefly, not dwell.
    print("\n############ PLASTIC eval (learning stays on -> drift) ############")
    plastic = [measure(s, freeze=False, refractory=2) for s in SEEDS]
    a = _print_condition("refractory=2, learning ON during eval", plastic)


if __name__ == "__main__":
    main()
