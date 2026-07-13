"""
Sustained-presentation dominance harness -- the HONEST one-to-one metric.

Unlike metrics_consolidation.py (which presents each pattern for a single short
window per sweep and takes the modal winner ACROSS sweeps -- a measurement
artifact that only ever sees the specialist's reliable first-cycle-after-switch
win), this harness HOLDS each pattern continuously for many cycles and records
one winner PER CYCLE. That exposes round-robin behavior even when short visits
specialists exist, rivals can still take most cycles under sustained input.

Protocol (per seed, per condition):
  1. Build SimulationEngine(seed, subtractive_reset=<cond>).
  2. Interleaved-train `epochs` sweeps; each sweep presents every pattern for
     `cycle_period` steps (learning stays on throughout -- the local rules never
     switch off).
  3. For each pattern: hold it for cycle_period * HOLD_CYCLES steps and record the
     L2E that fired each cycle (exactly one competition resolves per cycle).
  4. Report distinct modal winners / 8, mean sustained dominance, per-pattern
     modal winner + dominance, and dead-L2E count.

Compares baseline (subtractive_reset=False) vs experiment (True) across seeds.

FINDING (see AGENT_HANDOFF.md sec 5-6): at the default refractory=2, subtractive
reset is INERT -- the winner fires far above threshold (~19k-26k vs an 8k
threshold) and fire() does leave it a large residual, but update()'s refractory
clamp forces the membrane back to rest for the whole refractory period, erasing
the residual on the same step. Baseline and experiment come out byte-identical.
Only at refractory=0 (no clamp) does the residual survive, and there it trades a
small dominance gain for a large distinctness collapse (8 -> ~5.5 distinct) --
the same margin-vs-distinctness frontier documented in the handoff. So this
mechanism, as specified, does NOT deliver sustained one-to-one ownership. The
refractory sweep at the bottom of the run makes the masking self-documenting.

    PYTHONPATH=. .venv/bin/python sustained_dominance.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


TRAIN_EPOCHS = 60      # interleaved training sweeps over all patterns
HOLD_CYCLES = 40       # cycles a pattern is held during sustained measurement
SEEDS = (1, 2, 3, 4)


def _train_interleaved(engine, names, steps, epochs):
    """Interleaved training: present every pattern for `steps` steps per sweep."""
    for _ in range(epochs):
        for name in names:
            engine.set_pattern(name)
            for _ in range(steps):
                engine.step()


def _sustained_winners(engine, name, steps, hold_cycles):
    """Hold one pattern for `steps * hold_cycles` steps; return the sequence of
    per-cycle L2E winners (the neuron that actually spiked that cycle). At most
    one L2E fires per cycle, so each spike is one cycle's winner; silent cycles
    contribute nothing."""
    engine.set_pattern(name)
    winners: list[int] = []
    for _ in range(steps * hold_cycles):
        engine.step()
        for j in range(N_OUT):
            if engine.spiked[f'L2E{j}']:
                winners.append(j)
                break   # only one L2E resolves per cycle
    return winners


def measure(seed, subtractive_reset, refractory=None,
            epochs=TRAIN_EPOCHS, hold_cycles=HOLD_CYCLES):
    names = list(PATTERNS.keys())
    kw = {} if refractory is None else {'refractory': refractory}
    e = SimulationEngine(seed=seed, subtractive_reset=subtractive_reset, **kw)
    steps = e.params['cycle_period']

    _train_interleaved(e, names, steps, epochs)

    modal: dict[str, int | None] = {}
    dominance: dict[str, float] = {}
    ever_fired = set()
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

    distinct = len({v for v in modal.values() if v is not None})
    dead = N_OUT - len(ever_fired)
    mean_dom = float(np.mean(list(dominance.values())))
    return dict(seed=seed, distinct=distinct, mean_dom=mean_dom, dead=dead,
                modal=modal, dominance=dominance)


def _print_run(tag, results):
    print(f"\n===== subtractive_reset {tag} =====")
    ds = dd = dm = 0.0
    for m in results:
        print(f"  seed {m['seed']}: distinct={m['distinct']}/{len(PATTERNS)}  "
              f"mean_sustained_dominance={m['mean_dom']:.3f}  dead={m['dead']}")
        per = "  ".join(f"{name}->L2E{m['modal'][name]}:{m['dominance'][name]:.2f}"
                        for name in m['modal'])
        print(f"           {per}")
        ds += m['distinct']; dd += m['dead']; dm += m['mean_dom']
    n = len(results)
    print(f"  MEAN: distinct={ds/n:.2f}/{len(PATTERNS)}  sustained_dominance={dm/n:.3f}  dead={dd/n:.2f}")
    return dm / n, ds / n, dd / n


def _refractory_probe():
    """Isolate WHY the headline comparison shows no effect: sweep refractory and
    show that the subtractive-reset residual only survives at refractory=0 (where
    update() does not clamp the winner to rest), and even then it collapses
    distinctness rather than delivering ownership."""
    print("\n===== refractory sweep (mean over seeds) =====")
    print("  (subtractive reset can only matter where the winner is NOT clamped to")
    print("   rest during refractory; that is refractory=0 only.)")
    for refr in (2, 1, 0):
        for sub in (False, True):
            res = [measure(s, sub, refractory=refr) for s in SEEDS]
            n = len(res)
            dist = sum(m['distinct'] for m in res) / n
            dom = sum(m['mean_dom'] for m in res) / n
            dead = sum(m['dead'] for m in res) / n
            tag = "exp " if sub else "base"
            print(f"  refractory={refr}  {tag}: distinct={dist:.2f}/{len(PATTERNS)}  "
                  f"sustained_dominance={dom:.3f}  dead={dead:.2f}")


def main():
    base = [measure(s, False) for s in SEEDS]
    exp = [measure(s, True) for s in SEEDS]
    b_dom, b_dist, b_dead = _print_run("OFF (baseline)", base)
    e_dom, e_dist, e_dead = _print_run("ON  (experiment)", exp)

    print("\n===== summary (mean over seeds, default refractory=2) =====")
    print(f"  sustained dominance : baseline {b_dom:.3f}  ->  experiment {e_dom:.3f}  "
          f"(delta {e_dom - b_dom:+.3f})")
    print(f"  distinct winners /{len(PATTERNS)} : baseline {b_dist:.2f}   ->  experiment {e_dist:.2f}")
    print(f"  dead L2E            : baseline {b_dead:.2f}   ->  experiment {e_dead:.2f}")
    print("  NOTE: identical at default refractory -- the residual is erased by the")
    print("        refractory clamp (see module docstring). See the sweep below.")

    _refractory_probe()


if __name__ == "__main__":
    main()
