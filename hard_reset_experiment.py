"""
Measurement harness for hard-reset inhibition (Hard_Reset_Inhibition_Plan.md).

Runs the plan's comparison as a CLEAN A/B: identical minimal signed-spike regime
(loser depression / confidence / signed depression / budget / homeostasis /
flow-rate all OFF, no membrane noise), with ONLY `l2i_hard_reset_losers` toggled.
That isolates the hard reset's effect from every compensating mechanism.

Conditions:
    baseline : minimal signed-spike regime, small subtractive L2I gate (flag off)
    hardreset: same regime, non-winners clamped to rest after inhibitory learning

Reported per condition (mean over seeds):
    - sustained-presentation dominance   (the HONEST one-to-one metric: hold each
      pattern for many cycles, take the modal winner's cycle fraction)
    - interleaved all-8 distinct modal owners / 8
    - dead / ownerless L2E
    - single held-pattern dominance      (best pattern's sustained dominance)
    - immediate re-fire rate after an L2I discharge
    - loser V/theta immediately before vs after reset
    - feedforward weight saturation rate  (fraction of L2E positive gates at cap)

    PYTHONPATH=. .venv/bin/python hard_reset_experiment.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


TRAIN_EPOCHS = 60
HOLD_CYCLES = 40
SEEDS = (1, 2, 3, 4)

# Minimal signed-spike regime shared by BOTH conditions (see the plan's Intended
# Minimal Regime). Only l2i_hard_reset_losers differs between conditions.
BASE_REGIME = dict(
    signed_spike_learning=True,
    loser_depression=False,
    confidence_consolidation=False,
    signed_depression=False,
    l2e_budget=False,
    homeostasis=False,
    lasting_inhibition=False,
    inhibitory_flow_rate=False,
    excitatory_flow_rate=False,
)


def _train_interleaved(engine, names, steps, epochs):
    for _ in range(epochs):
        for name in names:
            engine.set_pattern(name)
            for _ in range(steps):
                engine.step()


def _sustained_winners(engine, name, steps, hold_cycles):
    """Hold one pattern; return the per-cycle L2E winner sequence and loser
    V/theta captured before reset (from the competitive-reset record's v_pre). The
    round-robin signature -- how often the winner CHANGES from one cycle to the
    next under a fixed input -- is computed from the winner sequence by the
    caller; a hard reset that removes loser carryover should lower it."""
    engine.set_pattern(name)
    winners: list[int] = []
    loser_vpre_ratios: list[float] = []
    thr = engine.params['threshold_l2']
    for _ in range(steps * hold_cycles):
        engine.step()
        for nid, rec in engine._reset_events:
            if nid.startswith('L2E'):
                loser_vpre_ratios.append(rec['v_pre'] / thr if thr else 0.0)
        for j in range(N_OUT):
            if engine.spiked[f'L2E{j}']:
                winners.append(j)
                break
    return dict(winners=winners, loser_vpre_ratios=loser_vpre_ratios)


def _rotation_rate(winners):
    """Fraction of consecutive cycles where the winner switched -- the direct
    round-robin rate under sustained input (0 = one neuron holds, 1 = alternates
    every cycle)."""
    if len(winners) < 2:
        return 0.0
    switches = sum(1 for a, b in zip(winners, winners[1:]) if a != b)
    return switches / (len(winners) - 1)


def _saturation_rate(engine):
    """Fraction of L2E positive feedforward gates sitting at (>=99% of) their cap
    -- how many co-specialists have raced their weights to the ceiling."""
    at_cap = tot = 0
    for nid, n in engine.neurons.items():
        if not (engine.meta[nid]['type'] == 'E' and nid.startswith('L2')):
            continue
        cap = n.weight_cap
        w = n.weights
        pos = w[w > 0]
        at_cap += int(np.sum(pos >= 0.99 * cap))
        tot += pos.size
    return at_cap / tot if tot else 0.0


def measure(seed, hard_reset, epochs=TRAIN_EPOCHS, hold_cycles=HOLD_CYCLES):
    names = list(PATTERNS.keys())
    e = SimulationEngine(seed=seed, l2i_hard_reset_losers=hard_reset, **BASE_REGIME)
    steps = e.params['cycle_period']
    thr = e.params['threshold_l2']

    _train_interleaved(e, names, steps, epochs)

    modal: dict[str, int | None] = {}
    dominance: dict[str, float] = {}
    rotations: list[float] = []
    ever_fired = set()
    vpre_ratios: list[float] = []
    for name in names:
        d = _sustained_winners(e, name, steps, hold_cycles)
        w = d['winners']
        ever_fired.update(w)
        vpre_ratios.extend(d['loser_vpre_ratios'])
        if w:
            c = Counter(w)
            top, top_n = c.most_common(1)[0]
            modal[name] = top
            dominance[name] = top_n / len(w)
            rotations.append(_rotation_rate(w))
        else:
            modal[name] = None
            dominance[name] = 0.0

    distinct = len({v for v in modal.values() if v is not None})
    dead = N_OUT - len(ever_fired)
    mean_dom = float(np.mean(list(dominance.values())))
    best_dom = float(np.max(list(dominance.values())))
    rotation = float(np.mean(rotations)) if rotations else 0.0
    vpre_mean = float(np.mean(vpre_ratios)) if vpre_ratios else 0.0
    return dict(seed=seed, distinct=distinct, mean_dom=mean_dom, best_dom=best_dom,
                dead=dead, rotation=rotation, vpre=vpre_mean,
                sat=_saturation_rate(e), modal=modal, dominance=dominance)


def _summarize(tag, results):
    print(f"\n===== {tag} =====")
    keys = ('distinct', 'mean_dom', 'best_dom', 'dead', 'rotation', 'vpre', 'sat')
    for m in results:
        print(f"  seed {m['seed']}: distinct={m['distinct']}/{len(PATTERNS)}  "
              f"sustained_dom={m['mean_dom']:.3f}  best={m['best_dom']:.3f}  "
              f"dead={m['dead']}  rotation={m['rotation']:.3f}  "
              f"loserV/θ={m['vpre']:.3f}  sat={m['sat']:.3f}")
    agg = {k: float(np.mean([m[k] for m in results])) for k in keys}
    print(f"  MEAN: distinct={agg['distinct']:.2f}/{len(PATTERNS)}  sustained_dom={agg['mean_dom']:.3f}  "
          f"best_held={agg['best_dom']:.3f}  dead={agg['dead']:.2f}  "
          f"rotation={agg['rotation']:.3f}  loserV/θ={agg['vpre']:.3f}  sat={agg['sat']:.3f}")
    return agg


def main():
    base = [measure(s, False) for s in SEEDS]
    hard = [measure(s, True) for s in SEEDS]
    b = _summarize("baseline (subtractive L2I gate)", base)
    h = _summarize("hard reset (loser -> rest after learning)", hard)

    print("\n===== summary (mean over seeds) =====")
    def line(label, k, fmt="{:+.3f}"):
        print(f"  {label:<26} baseline {b[k]:.3f}  ->  hardreset {h[k]:.3f}   "
              f"(delta {fmt.format(h[k] - b[k])})")
    line("sustained dominance", "mean_dom")
    line("single held-pattern dom", "best_dom")
    line(f"distinct owners /{len(PATTERNS)}", "distinct")
    line("dead L2E", "dead")
    line("winner rotation rate", "rotation")
    line("loser V/theta pre-reset", "vpre")
    line("feedforward saturation", "sat")
    print("\n  (loser V/theta AFTER reset is 0.000 by construction under hard reset;")
    print("   baseline retains max(V - gate, 0), so carryover is nonzero there.)")


if __name__ == "__main__":
    main()
