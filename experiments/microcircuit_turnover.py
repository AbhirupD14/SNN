"""Single 3x3 microcircuit turnover test (rg_coincidence), evidence-gathering only.

Directly tests the hypothesis that the small circuit turns over BECAUSE it has a
feature-specific inhibitory layer. In ``rg_coincidence`` each pixel/feature i has its own
paired chain::

    L1E_i (pretrained feature relay) -> L1C_i (coincidence) -> L1I_i -> L1E_i (reset)

so when the current L2E owner predicts feature i (apical) and feature i is active (basal),
only L1E_i is suppressed -- not the whole feature layer. tiled_cc replaced this with one
column C -> one column I that hard-resets the ENTIRE ordinary-E bank, which erases feature
identity and (as separately measured) removes turnover.

Protocol for one microcircuit, canonical order row 1 -> col 1 -> diag \\ -> diag /:

  1. Train each pattern for a fixed dwell.
  2. At every switch, all four patterns share the CENTER pixel (4); each adds two novel
     features. The incumbent predicts the shared center, so the center feature relay is
     transiently suppressed while the novel relays stay active, letting a different
     competitor win and consolidate.

This is NOT production; it uses the same experiment-only L2-initialization normalization as
``coincidence_turnover_sweep`` and does not change any preset, default, or dashboard
control. It changes no model dynamic.

Run: ``PYTHONPATH=. .venv/bin/python experiments/microcircuit_turnover.py``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine, PATTERNS               # noqa: E402
from experiments.coincidence_turnover_sweep import (                    # noqa: E402
    normalize_l2_initial_totals,
)

CANONICAL_ORDER = ("row 1", "col 1", "diag \\", "diag /")
CENTER_PIXEL = 4                       # the pixel shared by all four 3x3 patterns

# Known-good turnover regime for the small circuit (see coincidence_turnover_sweep):
# experiment-only L2 row normalization + a C learning rate fast enough to mature the
# coincidence prediction. These are experiment settings, not production defaults.
DEFAULTS = dict(eta=0.01, c_eta=0.005, leak_rate=0.0, refractory_steps=0,
                e_weight_cap=500.0, l2_total_frac=0.95)


def _active_pixels(pattern: str) -> list:
    return [i for i, v in enumerate(PATTERNS[pattern]) if v > 0.5]


def build_microcircuit(seed: int, *, eta: float, c_eta: float, leak_rate: float,
                       refractory_steps: int, e_weight_cap: float,
                       l2_total_frac: float) -> SimulationEngine:
    engine = SimulationEngine(seed=int(seed), topology="rg_coincidence", eta=eta,
                              c_eta=c_eta, leak_rate=leak_rate,
                              refractory_steps=refractory_steps, e_weight_cap=e_weight_cap)
    normalize_l2_initial_totals(engine, l2_total_frac)
    return engine


def _run_phase(engine: SimulationEngine, pattern: str, dwell: int, *,
               early: int, final_window: int, prev_owner) -> dict:
    """Present one pattern; capture the early transient (center vs novel feature-relay
    firing + winner handoff) and the consolidated final owner."""
    engine.set_pattern(pattern)
    active = _active_pixels(pattern)
    novel = [i for i in active if i != CENTER_PIXEL]

    early_center = 0
    early_novel: Counter = Counter()
    early_winners: Counter = Counter()
    final_winners: Counter = Counter()
    c_center_spikes = 0
    hard_resets = 0

    for step in range(1, dwell + 1):
        engine.step()
        fired = [c.id for c in engine.latency_competitors if c.spiked]
        if len(fired) > 1:
            raise AssertionError(f"multiple L2 winners at t={engine.timestep}: {fired}")
        if step <= early:
            if engine.spiked[f"L1E{CENTER_PIXEL}"]:
                early_center += 1
            for pix in novel:
                if engine.spiked[f"L1E{pix}"]:
                    early_novel[pix] += 1
            if fired:
                early_winners[fired[0]] += 1
            if engine.spiked[f"L1C{CENTER_PIXEL}"]:
                c_center_spikes += 1
        if fired and step > dwell - final_window:
            final_winners[fired[0]] += 1
        hard_resets += len(engine.hard_reset_events)

    final_owner, final_wins = (final_winners.most_common(1)[0]
                               if final_winners else (None, 0))
    final_total = sum(final_winners.values())
    novel_mean = mean(early_novel[i] for i in novel) if novel else 0.0

    return {
        "pattern": pattern,
        "active_pixels": active,
        "novel_pixels": novel,
        "prev_owner": prev_owner,
        "early_window": early,
        "early_center_relay_fires": early_center,
        "early_novel_relay_fires": {str(i): early_novel[i] for i in novel},
        "early_novel_mean": round(novel_mean, 2),
        "center_suppressed_vs_novel": bool(prev_owner is not None
                                           and early_center < novel_mean),
        "early_center_C_spikes": c_center_spikes,
        "early_winners": dict(early_winners),
        "final_owner": final_owner,
        "final_dominance": round(final_wins / final_total, 4) if final_total else 0.0,
        "turnover_from_prev": bool(prev_owner is not None and final_owner != prev_owner),
        "hard_reset_events": hard_resets,
    }


def run_microcircuit(seed: int = 1, *, dwell: int = 4000, early: int = 200,
                     final_window: int = 500, order=CANONICAL_ORDER,
                     dominance: float = 0.8, **cfg) -> dict:
    """Run the full four-pattern turnover protocol on one microcircuit and evaluate the
    feature-specific inhibition mechanism."""
    params = {**DEFAULTS, **cfg}
    engine = build_microcircuit(seed, **params)

    phases = []
    prev_owner = None
    for pattern in order:
        ph = _run_phase(engine, pattern, dwell, early=early,
                        final_window=final_window, prev_owner=prev_owner)
        phases.append(ph)
        prev_owner = ph["final_owner"]

    owners = [p["final_owner"] for p in phases]
    switches = phases[1:]                 # every phase after the first is a switch
    checks = {
        # each pattern consolidates a clear owner
        "all_consolidated": all(p["final_dominance"] >= dominance and p["final_owner"]
                                for p in phases),
        # a different competitor wins after every switch (turnover)
        "turnover_every_switch": all(p["turnover_from_prev"] for p in switches),
        # the shared center feature relay is transiently suppressed vs novel relays
        "center_suppressed_every_switch": all(p["center_suppressed_vs_novel"]
                                              for p in switches),
        # the novel feature relays stay active through each switch (not suppressed)
        "novel_relays_active_every_switch": all(
            all(v > 0 for v in p["early_novel_relay_fires"].values()) for p in switches),
        # the four owners are distinct (a real one-to-one mapping)
        "four_distinct_owners": len(set(owners)) == len(owners) and None not in owners,
    }
    return {
        "experiment": "microcircuit_turnover",
        "topology": "rg_coincidence",
        "seed": seed,
        "params": params,
        "protocol": {"order": list(order), "dwell": dwell, "early": early,
                     "final_window": final_window, "shared_center_pixel": CENTER_PIXEL,
                     "dominance_threshold": dominance},
        "owner_by_pattern": {p["pattern"]: p["final_owner"] for p in phases},
        "owners": owners,
        "phases": phases,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _print_report(res: dict) -> None:
    print(f"=== microcircuit turnover (rg_coincidence, seed {res['seed']}) ===")
    print(f"params: c_eta={res['params']['c_eta']} rho={res['params']['l2_total_frac']} "
          f"dwell={res['protocol']['dwell']}")
    for p in res["phases"]:
        tag = "TRAIN " if p["prev_owner"] is None else "SWITCH"
        supp = ("center suppressed" if p["center_suppressed_vs_novel"]
                else ("center NOT suppressed" if p["prev_owner"] else "first pattern"))
        print(f"  {tag} {p['pattern']:>8s}: owner={p['final_owner']} "
              f"dom={p['final_dominance']:.2f} turnover={p['turnover_from_prev']} | "
              f"early center L1E{CENTER_PIXEL}={p['early_center_relay_fires']} "
              f"novel{p['novel_pixels']}={list(p['early_novel_relay_fires'].values())} "
              f"({supp})")
    print(f"owners: {res['owners']}  distinct={res['checks']['four_distinct_owners']}")
    for k, v in res["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"ALL CHECKS PASS: {res['passed']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dwell", type=int, default=4000)
    ap.add_argument("--early", type=int, default=200)
    ap.add_argument("--final-window", type=int, default=500)
    ap.add_argument("--c-eta", type=float, default=DEFAULTS["c_eta"])
    ap.add_argument("--l2-total-frac", type=float, default=DEFAULTS["l2_total_frac"])
    ap.add_argument("--output", default=None, help="optional JSON result path")
    args = ap.parse_args(argv)

    res = run_microcircuit(args.seed, dwell=args.dwell, early=args.early,
                           final_window=args.final_window, c_eta=args.c_eta,
                           l2_total_frac=args.l2_total_frac)
    _print_report(res)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.output}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
