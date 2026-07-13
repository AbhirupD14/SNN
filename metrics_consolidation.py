"""
Focused consolidation metrics for the confidence-gated consolidation mechanism
(Claude_Confidence_Consolidation_Plan.md).

Reports for the center-crossing 3x3 line primitives driven through the L1->L2 engine:

  - distinct winners across the patterns,
  - within-pattern winner dominance (how consistently one neuron owns a pattern),
  - old-pattern preservation after training NEW patterns (continual-learning check),
  - number of L2E neurons that never fired,
  - confidence distribution per L2E (max feedforward confidence),
  - loser-depression event counts.

Runs the mechanism ON vs OFF (ablation) at several seeds so the effect is visible.
This is a diagnostic script, not a pass/fail unit test; it prints a report.

    PYTHONPATH=. .venv/bin/python metrics_consolidation.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT, N_PIX


def _present(engine, name, steps, fired):
    """Drive one pattern for `steps` steps. Accumulate global L2E spike counts in
    `fired`, and return the neuron that fired most during THIS window (or None if
    the pool was silent) -- a clean per-pattern attribution that does not lag
    across patterns the way the episode winner (engine.winner) can."""
    engine.set_pattern(name)
    window = np.zeros(N_OUT, dtype=int)
    for _ in range(steps):
        engine.step()
        for j in range(N_OUT):
            if engine.spiked[f'L2E{j}']:
                fired[j] += 1
                window[j] += 1
    return int(window.argmax()) if window.sum() > 0 else None


def _winner_map(engine, patterns, steps, epochs, record_last):
    """Present `patterns` for `epochs` sweeps; over the last `record_last` sweeps
    record which neuron fired most for each pattern. Returns (modal winner per
    pattern, dominance per pattern, fired counts)."""
    fired = np.zeros(N_OUT, dtype=int)
    hist: dict[str, Counter] = {name: Counter() for name in patterns}
    for ep in range(epochs):
        for name in patterns:
            w = _present(engine, name, steps, fired)
            if ep >= epochs - record_last and w is not None:
                hist[name][w] += 1
    modal = {name: (c.most_common(1)[0][0] if c else None) for name, c in hist.items()}
    dominance = {name: (c.most_common(1)[0][1] / sum(c.values()) if c else 0.0)
                 for name, c in hist.items()}
    return modal, dominance, fired


def measure(seed, consolidation, epochs=60):
    """Full metric run at one seed. Also does an old-pattern-preservation probe:
    train the first half of the patterns, snapshot their winners, train the
    second half, then re-probe the first half and check retention."""
    names = list(PATTERNS.keys())
    steps = None

    e = SimulationEngine(seed=seed, confidence_consolidation=consolidation,
                         loser_depression=consolidation)
    steps = e.params['cycle_period']

    # --- main interleaved training over all patterns ---
    modal, dominance, fired = _winner_map(e, names, steps, epochs, record_last=5)

    distinct = len(set(v for v in modal.values() if v is not None))
    never_fired = int((fired == 0).sum())
    mean_dom = float(np.mean(list(dominance.values())))

    # confidence distribution: max feedforward confidence per L2E neuron
    max_conf = []
    for j in range(N_OUT):
        conf = e.l2.excitatory_neurons[j].confidence          # aligned to weights
        max_conf.append(float(conf[1:1 + N_PIX].max()))       # skip index 0 (inhib gate)
    loss_events = int(sum(e.neurons[f'L2E{j}'].loser_depression_events for j in range(N_OUT)))

    # --- old-pattern preservation (continual learning) ---
    e2 = SimulationEngine(seed=seed, confidence_consolidation=consolidation,
                          loser_depression=consolidation)
    split = max(1, len(names) // 2)
    first, second = names[:split], names[split:]
    old_map, _, _ = _winner_map(e2, first, steps, epochs, record_last=5)
    _winner_map(e2, second, steps, epochs, record_last=5)          # train the new set
    reprobe, _, _ = _winner_map(e2, first, steps, max(8, epochs // 6), record_last=5)
    preserved = sum(1 for n in first
                    if old_map[n] is not None and old_map[n] == reprobe[n])

    return dict(distinct=distinct, winners=modal, dominance=mean_dom,
                never_fired=never_fired, max_conf=max_conf,
                loss_events=loss_events, preserved=preserved, n_old=len(first))


def main():
    seeds = (1, 2, 3, 4)
    for consolidation in (False, True):
        tag = "ON " if consolidation else "OFF"
        print(f"\n===== confidence consolidation {tag} =====")
        dsum = psum = 0
        for seed in seeds:
            m = measure(seed, consolidation)
            dsum += m['distinct']; psum += m['preserved']
            conf_str = "[" + " ".join(f"{c:.2f}" for c in m['max_conf']) + "]"
            print(f"  seed {seed}: distinct_winners={m['distinct']}/{len(PATTERNS)}  "
                  f"within_pattern_dominance={m['dominance']:.2f}  "
                  f"never_fired={m['never_fired']}  "
                  f"old_preserved={m['preserved']}/{m['n_old']}  "
                  f"loser_depressions={m['loss_events']}")
            print(f"           max_feedforward_confidence per L2E {conf_str}")
        print(f"  mean distinct winners: {dsum/len(seeds):.2f}/{len(PATTERNS)}   "
              f"mean old-pattern preservation: {psum/len(seeds):.2f}/{max(1, len(PATTERNS)//2)}")


if __name__ == "__main__":
    main()
