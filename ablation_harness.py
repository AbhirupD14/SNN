"""
Reusable ablation harness -- the HONEST one-to-one metric, parameterised by
engine configuration and aggregated across seeds.

WHY THIS EXISTS
---------------
`metrics_consolidation.py` reports a MEASUREMENT ARTIFACT (it shows each pattern
for one short window per sweep and takes the modal winner ACROSS sweeps, so it
only ever sees the specialist's reliable first-cycle-after-switch win and never
the round-robin that follows -- see AGENT_HANDOFF.md sec 2). `sustained_dominance.py`
uses the honest sustained-presentation protocol but is hardwired to the
subtractive-reset ablation and reports no seed variance and none of the
receptive-field similarity metrics.

This module factors the honest protocol into a reusable function `evaluate()`
that takes ANY SimulationEngine configuration (a dict of constructor kwargs),
runs it across several seeds, and reports the metrics used by the archived
input-init ablation and the next distance-weighting ablation
(`Input_Vector_Initialization_And_Distance_Weighting.md`):

    - time-to-stable ownership (epochs until per-pattern modal winner stops moving)
    - sustained dominance (fraction of held cycles the modal specialist wins)
    - distinct modal winners / owner collisions
    - dead L2E count
    - initial and final pairwise receptive-field cosine similarity
    - SEED VARIANCE (std across seeds) on every scalar above

The vector-aware initialization ablation is now archived (feedforward init is
plain uniform random again); RF-cosine metrics stay useful for showing whether
distance-weighted geometry changes final receptive field overlap. Build distance
behind flags and compare conditions here.

    PYTHONPATH=. .venv/bin/python ablation_harness.py                 # baseline demo
    PYTHONPATH=. .venv/bin/python ablation_harness.py --seeds 1 2 3   # pick seeds
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


# Honest-protocol defaults (match sustained_dominance.py so numbers are comparable).
TRAIN_EPOCHS = 60      # interleaved training sweeps over all patterns
HOLD_CYCLES = 40       # cycles a pattern is held during sustained measurement
SEEDS = (1, 2, 3)      # >=3 seeds so seed variance is meaningful

# CANONICAL BASELINE for the distance-weighting ablation: the
# minimal signed-spike rule with NO weight budget (neuron_flexible.py:495-503).
# On fire, every positive feedforward synapse updates with a local signed signal
# -- +1 if its input participated, -1 if not -- and the -1 supplies the downward
# pressure the budget used to impose; stored weights are otherwise free.
#
# As of 2026-07-08 this MATCHES the committed SimulationEngine defaults (kept
# explicit here so the baseline is pinned even if the defaults move again). It is
# the regime where seed dependence actually appears: the budget masks it by
# forcing 8/8 distinct on every seed (variance ~0), whereas here distinctness
# collapses to ~4/8 and swings by seed. Measure distance ablations as deltas ON
# TOP OF this condition, e.g.
#   compare({'baseline': dict(BASELINE),
#            'distance_ff': {**BASELINE, 'distance_weighting': True}}, ...)
BASELINE = {'signed_spike_learning': True, 'l2e_budget': False}


# ---------------------------------------------------------------------------
# Protocol primitives
# ---------------------------------------------------------------------------
def _train_interleaved(engine, names, steps, epochs, ownership_log=None):
    """Interleaved training: present every pattern for `steps` steps per sweep.

    If `ownership_log` is provided, after each sweep record the neuron that fired
    most for each pattern DURING that sweep, so time-to-stable-ownership can be
    derived from when the per-pattern modal owner stops changing."""
    for _ in range(epochs):
        sweep_winner = {}
        for name in names:
            engine.set_pattern(name)
            window = np.zeros(N_OUT, dtype=int)
            for _ in range(steps):
                engine.step()
                for j in range(N_OUT):
                    if engine.spiked[f'L2E{j}']:
                        window[j] += 1
            sweep_winner[name] = int(window.argmax()) if window.sum() > 0 else None
        if ownership_log is not None:
            ownership_log.append(sweep_winner)


def _sustained_winners(engine, name, steps, hold_cycles):
    """Hold one pattern for `steps * hold_cycles` steps; return the per-cycle L2E
    winner sequence. At most one L2E resolves per cycle, so each spike is one
    cycle's winner; silent cycles contribute nothing."""
    engine.set_pattern(name)
    winners: list[int] = []
    for _ in range(steps * hold_cycles):
        engine.step()
        for j in range(N_OUT):
            if engine.spiked[f'L2E{j}']:
                winners.append(j)
                break
    return winners


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mean_pairwise_cosine(rf: np.ndarray) -> float:
    """Mean cosine similarity over all unordered pairs of receptive-field rows.
    High => neurons point the same direction (duplicate RFs); this is exactly
    what vector-aware initialization tries to push down."""
    n = rf.shape[0]
    norms = np.linalg.norm(rf, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = rf / safe[:, None]
    sims = []
    for i in range(n):
        for k in range(i + 1, n):
            sims.append(float(np.dot(unit[i], unit[k])))
    return float(np.mean(sims)) if sims else 0.0


def _time_to_stable(ownership_log, patience=5) -> int | None:
    """Earliest epoch after which EVERY pattern's per-sweep modal owner is
    unchanged for `patience` consecutive sweeps (and none is None). Returns the
    epoch index (0-based) at which that stable run begins, or None if never."""
    if len(ownership_log) < patience:
        return None
    names = list(ownership_log[0].keys())
    for start in range(len(ownership_log) - patience + 1):
        window = ownership_log[start:start + patience]
        stable = True
        for name in names:
            vals = [sweep[name] for sweep in window]
            if vals[0] is None or any(v != vals[0] for v in vals):
                stable = False
                break
        if stable:
            return start
    return None


def evaluate_seed(seed, condition, epochs=TRAIN_EPOCHS, hold_cycles=HOLD_CYCLES):
    """One honest run at one seed for one engine configuration (`condition` is a
    dict of SimulationEngine constructor kwargs). Returns per-seed metrics."""
    names = list(PATTERNS.keys())
    e = SimulationEngine(seed=seed, **condition)
    steps = e.params['cycle_period']

    init_rf_cos = _mean_pairwise_cosine(e.l2.feedforward_weights())

    ownership_log: list[dict] = []
    _train_interleaved(e, names, steps, epochs, ownership_log=ownership_log)

    final_rf_cos = _mean_pairwise_cosine(e.l2.feedforward_weights())
    ttos = _time_to_stable(ownership_log)

    modal: dict[str, int | None] = {}
    dominance: dict[str, float] = {}
    ever_fired: set[int] = set()
    for name in names:
        winners = _sustained_winners(e, name, steps, hold_cycles)
        ever_fired.update(winners)
        if winners:
            c = Counter(winners)
            top, top_n = c.most_common(1)[0]
            modal[name] = top
            dominance[name] = top_n / len(winners)
        else:
            modal[name] = None
            dominance[name] = 0.0

    owned = [v for v in modal.values() if v is not None]
    distinct = len(set(owned))
    collisions = len(owned) - distinct       # patterns sharing an owner
    dead = N_OUT - len(ever_fired)
    mean_dom = float(np.mean(list(dominance.values())))

    return dict(seed=seed, distinct=distinct, collisions=collisions,
                mean_dom=mean_dom, dead=dead, init_rf_cos=init_rf_cos,
                final_rf_cos=final_rf_cos, time_to_stable=ttos,
                modal=modal, dominance=dominance)


# ---------------------------------------------------------------------------
# Aggregation across seeds (mean + seed variance)
# ---------------------------------------------------------------------------
_SCALARS = ('mean_dom', 'distinct', 'collisions', 'dead',
            'init_rf_cos', 'final_rf_cos')


def evaluate(condition, seeds=SEEDS, epochs=TRAIN_EPOCHS, hold_cycles=HOLD_CYCLES):
    """Run `condition` across `seeds`; return per-seed rows plus mean and std
    (SEED VARIANCE) for every scalar metric, and mean time-to-stable over the
    seeds that reached stability (with a count of how many did)."""
    rows = [evaluate_seed(s, condition, epochs, hold_cycles) for s in seeds]
    agg = {}
    for k in _SCALARS:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k] = (float(vals.mean()), float(vals.std()))
    reached = [r['time_to_stable'] for r in rows if r['time_to_stable'] is not None]
    agg['time_to_stable'] = (float(np.mean(reached)) if reached else None, len(reached))
    return dict(rows=rows, agg=agg, n_seeds=len(seeds))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_condition(tag, result):
    print(f"\n===== {tag} =====")
    for r in result['rows']:
        tts = r['time_to_stable']
        tts_s = f"{tts}" if tts is not None else "--"
        print(f"  seed {r['seed']}: dom={r['mean_dom']:.3f}  distinct={r['distinct']}/{len(PATTERNS)}  "
              f"collisions={r['collisions']}  dead={r['dead']}  "
              f"rf_cos init={r['init_rf_cos']:.3f}->final={r['final_rf_cos']:.3f}  "
              f"t_stable={tts_s}")
    a = result['agg']
    def ms(k):
        m, s = a[k]
        return f"{m:.3f}+-{s:.3f}"
    tstos_mean, tstos_n = a['time_to_stable']
    tstos_s = f"{tstos_mean:.1f} ({tstos_n}/{result['n_seeds']} reached)" if tstos_mean is not None else "-- (0 reached)"
    print(f"  MEAN+-SD over {result['n_seeds']} seeds:")
    print(f"    sustained_dominance = {ms('mean_dom')}")
    print(f"    distinct/{len(PATTERNS)}          = {ms('distinct')}   collisions = {ms('collisions')}")
    print(f"    dead                = {ms('dead')}")
    print(f"    rf_cos initial      = {ms('init_rf_cos')}   final = {ms('final_rf_cos')}")
    print(f"    time_to_stable      = {tstos_s} epochs")


def compare(conditions: dict, seeds=SEEDS, epochs=TRAIN_EPOCHS, hold_cycles=HOLD_CYCLES):
    """Run several named conditions and print each. `conditions` maps a label to a
    dict of SimulationEngine kwargs (use {} for the current default baseline)."""
    results = {}
    for tag, cond in conditions.items():
        results[tag] = evaluate(cond, seeds, epochs, hold_cycles)
        print_condition(tag, results[tag])
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--epochs', type=int, default=TRAIN_EPOCHS)
    ap.add_argument('--hold', type=int, default=HOLD_CYCLES)
    args = ap.parse_args()

    # Canonical baseline = signed-spike / no-budget (see BASELINE above), the
    # regime distance ablations are measured against. Future ablations plug in as
    # extra entries layered on top.
    compare({'baseline (signed-spike, no budget)': dict(BASELINE)},
            seeds=tuple(args.seeds), epochs=args.epochs, hold_cycles=args.hold)


if __name__ == "__main__":
    main()
