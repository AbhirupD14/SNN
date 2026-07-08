"""
Inhibitory winner-take-all LOCKOUT experiment (learning stays ON).

Target mechanism (project owner's spec): while a pattern is dwelt on, the owner
fires, drives L2I, and L2I inhibition removes the OTHER L2E neurons' charge so
they do not fire at all. Once a pattern is owned, rivals go silent -> only the
owner keeps learning -> ownership self-reinforces under continual plasticity (no
freezing needed). The lockout must be per-pattern (each pattern's specialist
locks the rest out), NOT a single universal winner.

Two local ingredients, both default-off, both measured here:
  - v_sat_frac     : bound the L2E membrane near threshold so a rival's charge is
                     small enough for the gate to actually clear it.
  - l2_gate_eq_frac: raise the learnable L2I->L2E gate equilibrium so a discharge
                     can FLOOR a rival to rest (gate ~ membrane, not ~0.15x it).

Metric (learning ON throughout; hold each pattern HOLD_CYCLES cycles):
  - firers/hold  : mean number of DISTINCT L2E that fire during a held pattern.
                   1.0 == perfect lockout (only the owner ever fires). This is the
                   direct read of "no other neuron fires".
  - dominance    : owner's share of cycles during the hold (1.0 == owner every cycle).
  - distinct/8   : distinct owners across the 8 patterns (want 8 -> per-pattern,
                   not one universal king).
  - collisions   : patterns sharing an owner (want 0).
  - dead         : L2E that never won any pattern (secondary).
  - peakV        : peak competition-time membrane, xthreshold (shows charge scale).

    PYTHONPATH=. .venv/bin/python lockout_experiment.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


TRAIN_ROUNDS = 60
HOLD_CYCLES = 40
SEEDS = (1, 2, 3, 4)


def _train(engine, names, steps, rounds):
    for _ in range(rounds):
        for name in names:
            engine.set_pattern(name)
            for _ in range(steps):
                engine.step()


def _hold(engine, name, steps, hold_cycles):
    """Hold one pattern (learning ON). Return the per-cycle winner sequence, the
    set of all L2E that fired, and the peak competition-time membrane seen."""
    engine.set_pattern(name)
    winners: list[int] = []
    firers: set[int] = set()
    peak = 0.0
    for _ in range(steps * hold_cycles):
        engine.step()
        peak = max(peak, max(engine.l2_drive.values()))
        for j in range(N_OUT):
            if engine.spiked[f'L2E{j}']:
                winners.append(j)
                firers.add(j)
                break
    return winners, firers, peak


def measure(seed, **kw):
    names = list(PATTERNS.keys())
    e = SimulationEngine(seed=seed, **kw)
    steps = e.params['cycle_period']
    thr = e.params['threshold_l2']
    _train(e, names, steps, TRAIN_ROUNDS)

    owner: dict[str, int | None] = {}
    dom: dict[str, float] = {}
    firers_per_hold: list[int] = []
    won_any: set[int] = set()
    peak = 0.0
    for name in names:
        winners, firers, pk = _hold(e, name, steps, HOLD_CYCLES)
        peak = max(peak, pk)
        firers_per_hold.append(len(firers))
        if winners:
            c = Counter(winners)
            top, top_n = c.most_common(1)[0]
            owner[name] = top
            dom[name] = top_n / len(winners)
            won_any.add(top)
        else:
            owner[name] = None
            dom[name] = 0.0

    owners = [o for o in owner.values() if o is not None]
    distinct = len(set(owners))
    collisions = sum(n for n in Counter(owners).values() if n > 1)
    dead = N_OUT - len(won_any)
    return dict(seed=seed, distinct=distinct,
                mean_firers=float(np.mean(firers_per_hold)),
                mean_dom=float(np.mean(list(dom.values()))),
                collisions=collisions, dead=dead, peakV=peak / thr,
                owner=owner)


def _run(tag, kw):
    res = [measure(s, **kw) for s in SEEDS]
    n = len(res)
    agg = {k: sum(m[k] for m in res) / n
           for k in ('distinct', 'mean_firers', 'mean_dom', 'collisions', 'dead', 'peakV')}
    print(f"\n===== {tag} =====")
    for m in res:
        print(f"  seed {m['seed']}: firers/hold={m['mean_firers']:.2f}  "
              f"dominance={m['mean_dom']:.3f}  distinct={m['distinct']}/8  "
              f"collisions={m['collisions']}  dead={m['dead']}  peakV={m['peakV']:.2f}x")
    print(f"  MEAN: firers/hold={agg['mean_firers']:.2f}  dominance={agg['mean_dom']:.3f}  "
          f"distinct={agg['distinct']:.2f}/8  collisions={agg['collisions']:.2f}  "
          f"dead={agg['dead']:.2f}  peakV={agg['peakV']:.2f}x")
    return tag, agg


def main():
    conditions = [
        ("baseline",                          dict()),
        ("v_sat=1.25",                        dict(v_sat_frac=1.25)),
        ("gate_eq=1.25 (no v_sat)",           dict(l2_gate_eq_frac=1.25)),
        ("v_sat=1.25 + gate_eq=1.25",         dict(v_sat_frac=1.25, l2_gate_eq_frac=1.25)),
        ("v_sat=1.25 + gate_eq=1.5",          dict(v_sat_frac=1.25, l2_gate_eq_frac=1.5)),
    ]
    summary = [_run(tag, kw) for tag, kw in conditions]
    print("\n===== summary (mean over seeds 1-4; learning ON) =====")
    print(f"  {'condition':<28} {'firers/hold':>11} {'dominance':>10} "
          f"{'distinct':>9} {'collis':>7} {'dead':>5} {'peakV':>6}")
    for tag, a in summary:
        print(f"  {tag:<28} {a['mean_firers']:>11.2f} {a['mean_dom']:>10.3f} "
              f"{a['distinct']:>7.2f}/8 {a['collisions']:>7.2f} {a['dead']:>5.2f} "
              f"{a['peakV']:>5.2f}x")
    print("\n  firers/hold -> 1.0 means the owner locked every rival out (goal).")


if __name__ == "__main__":
    main()
