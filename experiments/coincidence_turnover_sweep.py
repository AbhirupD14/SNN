"""Headless L2-initialization/C-learning sweep for ``rg_coincidence``.

The dashboard experiment that motivated this sweep learned ``row 1`` for 2,500
boundaries, switched to the intersecting ``col 1`` for 2,500, then returned to
``row 1``.  Pixel 4 is the shared center.  The desired result is stable ownership
of the first row, turnover to another L2E owner after center suppression, and
recovery of the original row owner when the first pattern returns.

This is intentionally an experiment-only intervention.  Production defaults and
dashboard controls are not changed.  Each L2E cell keeps the direction of its
ordinary seeded +/-4% initialization, but its row is normalized to

    sum_i w_ji = rho * theta,       0 < rho < 1

so all competitors begin with the same positive free energy

    FE_j = theta - sum_i w_ji = (1 - rho) * theta.

That recovers the useful idea behind the archived ``uniform_normalized`` scheme
from commit 21ec95e without restoring its obsolete absolute weight scale. The
pre-promotion baseline ``rho = 0.55`` remains included in the sweep.

Run (full preregistered sweep):

    PYTHONPATH=. .venv/bin/python experiments/coincidence_turnover_sweep.py

The confirmation stage repeats the selected condition with normal and reversed
L2 scheduler order.  Reordering changes only the scheduler's exact-tie fallback;
it does not exchange weights, ids, geometry, or connectivity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from statistics import mean

import numpy as np

from backend.simulation import L2_INIT_TOTAL_FRAC, SimulationEngine


OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "coincidence_turnover_results.json",
)
INIT_FRACS = (0.55, 0.70, 0.80, 0.90, 0.95)
C_ETAS = (0.001, 0.0025, 0.005, 0.01)
DEFAULT_SEEDS = tuple(range(1, 9))
DEFAULT_DWELL = 2500
DEFAULT_FINAL_WINDOW = 500
ACTIVE_BY_PATTERN = {
    "row 1": (3, 4, 5),
    "col 1": (1, 4, 7),
}


def normalize_l2_initial_totals(engine: SimulationEngine, total_frac: float) -> dict:
    """Normalize L2 afferent rows while preserving each seeded row direction.

    This is called immediately after construction and before the first step.  It
    refuses non-positive free energy and any condition that would exceed the
    configured per-synapse cap.
    """
    rho = float(total_frac)
    if not 0.0 < rho < 1.0:
        raise ValueError(f"total_frac must satisfy 0 < rho < 1, got {rho}")

    theta = float(engine.params["e_threshold"])
    cap = float(engine.params["e_weight_cap"])
    target = rho * theta
    before = []
    after = []
    for cell in engine.latency_competitors:
        raw_sum = float(cell.acc_weights.sum())
        if raw_sum <= 0.0:
            raise ValueError(f"{cell.id} has non-positive initial afferent total")
        before.append(raw_sum)
        scaled = cell.acc_weights * (target / raw_sum)
        if float(scaled.max(initial=0.0)) > cap + 1e-12:
            raise ValueError(
                f"rho={rho} would initialize {cell.id} above weight cap {cap}"
            )
        cell.acc_weights[:] = scaled
        after.append(float(cell.acc_weights.sum()))

    return {
        "rho": rho,
        "theta": theta,
        "target_total": target,
        "free_energy": theta - target,
        "totals_before": before,
        "totals_after": after,
    }


def reverse_l2_scheduler_order(engine: SimulationEngine) -> None:
    """Reverse only L2 cells at their existing scheduler-order positions."""
    ids = {cell.id for cell in engine.latency_competitors}
    reversed_ids = iter(reversed([nid for nid in engine.order if nid in ids]))
    engine.order = [next(reversed_ids) if nid in ids else nid for nid in engine.order]


def _dominant(counts: Counter) -> tuple[str | None, int, float]:
    total = sum(counts.values())
    if not total:
        return None, 0, 0.0
    winner, wins = counts.most_common(1)[0]
    return winner, wins, wins / total


def _run_phase(
    engine: SimulationEngine,
    pattern: str,
    steps: int,
    final_window: int,
) -> dict:
    engine.set_pattern(pattern)
    active = ACTIVE_BY_PATTERN[pattern]
    winner_counts: Counter = Counter()
    final_counts: Counter = Counter()
    l1_counts: Counter = Counter()
    c_counts: Counter = Counter()
    first100_l1: Counter = Counter()
    winner_events = []
    l2_tie_events = 0
    all_tie_events = 0
    hard_resets = 0

    for rel_step in range(1, steps + 1):
        engine.step()
        fired = [cell.id for cell in engine.latency_competitors if cell.spiked]
        if len(fired) > 1:
            raise AssertionError(f"more than one L2 winner at boundary {engine.timestep}: {fired}")
        if fired:
            winner = fired[0]
            winner_counts[winner] += 1
            winner_events.append((rel_step, winner))
            if rel_step > steps - final_window:
                final_counts[winner] += 1

        for pix in active:
            if engine.spiked[f"L1E{pix}"]:
                l1_counts[pix] += 1
                if rel_step <= min(100, steps):
                    first100_l1[pix] += 1
        for idx, cell in enumerate(engine.coincidence):
            if cell.spiked:
                c_counts[idx] += 1

        for tie in engine.latency_ties:
            all_tie_events += 1
            if any(nid.startswith("L2E") for nid in tie["ids"]):
                l2_tie_events += 1
        hard_resets += len(engine.hard_reset_events)

    dom, dom_wins, dominance = _dominant(winner_counts)
    final_dom, final_wins, final_dominance = _dominant(final_counts)
    return {
        "pattern": pattern,
        "steps": steps,
        "winner_counts": dict(sorted(winner_counts.items())),
        "winner_events": winner_events,
        "dominant": dom,
        "dominant_wins": dom_wins,
        "dominance": dominance,
        "final_window": final_window,
        "final_winner_counts": dict(sorted(final_counts.items())),
        "final_dominant": final_dom,
        "final_dominant_wins": final_wins,
        "final_dominance": final_dominance,
        "l1_counts": {str(k): l1_counts[k] for k in active},
        "first100_l1_counts": {str(k): first100_l1[k] for k in active},
        "c_spike_counts": {str(k): v for k, v in sorted(c_counts.items())},
        "l2_tie_events": l2_tie_events,
        "all_tie_events": all_tie_events,
        "hard_reset_events": hard_resets,
    }


def _weight_snapshot(engine: SimulationEngine) -> dict:
    return {
        cell.id: {
            "total": float(cell.acc_weights.sum()),
            "weights": [float(w) for w in cell.acc_weights],
        }
        for cell in engine.latency_competitors
    }


def run_condition(
    seed: int,
    total_frac: float,
    c_eta: float,
    *,
    dwell: int = DEFAULT_DWELL,
    final_window: int = DEFAULT_FINAL_WINDOW,
    l2_order: str = "normal",
) -> dict:
    """Run one row -> column -> row condition through the real engine."""
    if dwell < 1:
        raise ValueError("dwell must be positive")
    if not 1 <= final_window <= dwell:
        raise ValueError("final_window must be between 1 and dwell")
    if l2_order not in ("normal", "reverse"):
        raise ValueError("l2_order must be 'normal' or 'reverse'")

    engine = SimulationEngine(
        seed=int(seed),
        topology="rg_coincidence",
        eta=0.01,
        c_eta=float(c_eta),
        leak_rate=0.0,
        refractory_steps=0,
        e_weight_cap=500.0,
    )
    init = normalize_l2_initial_totals(engine, total_frac)
    if l2_order == "reverse":
        reverse_l2_scheduler_order(engine)

    initial_weights = _weight_snapshot(engine)
    row_train = _run_phase(engine, "row 1", dwell, final_window)
    after_row_weights = _weight_snapshot(engine)
    column = _run_phase(engine, "col 1", dwell, final_window)
    after_column_weights = _weight_snapshot(engine)
    row_return = _run_phase(engine, "row 1", dwell, final_window)

    incumbent = row_train["final_dominant"]
    col_events = column.pop("winner_events")
    row_train.pop("winner_events")
    row_return.pop("winner_events")
    rival_steps = [step for step, winner in col_events if winner != incumbent]
    incumbent_steps = [step for step, winner in col_events if winner == incumbent]

    center = column["first100_l1_counts"].get("4", 0)
    novel = [column["first100_l1_counts"].get(str(i), 0) for i in (1, 7)]
    novelty_minus_center = (mean(novel) - center) / min(100, dwell)

    row_stable = bool(
        incumbent is not None and row_train["final_dominance"] >= 0.90
    )
    replaced = bool(
        incumbent is not None
        and column["final_dominant"] is not None
        and column["final_dominant"] != incumbent
        and column["final_dominance"] >= 0.80
    )
    recovered = bool(
        incumbent is not None
        and row_return["final_dominant"] == incumbent
        and row_return["final_dominance"] >= 0.80
    )

    return {
        "seed": int(seed),
        "total_frac": float(total_frac),
        "c_eta": float(c_eta),
        "l2_order": l2_order,
        "dwell": int(dwell),
        "final_window": int(final_window),
        "initialization": init,
        "initial_weights": initial_weights,
        "after_row_weights": after_row_weights,
        "after_column_weights": after_column_weights,
        "final_weights": _weight_snapshot(engine),
        "c_final_basal_weights": [float(c.basal_weight) for c in engine.coincidence],
        "row_train": row_train,
        "column": column,
        "row_return": row_return,
        "incumbent": incumbent,
        "first_rival_step": min(rival_steps) if rival_steps else None,
        "last_incumbent_step": max(incumbent_steps) if incumbent_steps else None,
        "center_suppression_first100": novelty_minus_center,
        "row_stable": row_stable,
        "column_replaced": replaced,
        "row_recovered": recovered,
        "full_success": row_stable and replaced and recovered,
        "negative_initial_fe": init["free_energy"] < -1e-9,
    }


def summarize(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("cannot summarize an empty run list")

    def rate(key):
        return sum(bool(run[key]) for run in runs) / len(runs)

    def avg(values):
        values = [v for v in values if v is not None and math.isfinite(v)]
        return mean(values) if values else None

    def id_counts(values):
        # Keep an empty/no-winner condition serializable and sortable beside ids.
        return dict(sorted(Counter("none" if v is None else v for v in values).items()))

    return {
        "n": len(runs),
        "total_frac": runs[0]["total_frac"],
        "c_eta": runs[0]["c_eta"],
        "row_stable_rate": rate("row_stable"),
        "column_replacement_rate": rate("column_replaced"),
        "row_recovery_rate": rate("row_recovered"),
        "full_success_rate": rate("full_success"),
        "negative_initial_fe_count": sum(run["negative_initial_fe"] for run in runs),
        "mean_first_rival_step": avg([run["first_rival_step"] for run in runs]),
        "mean_last_incumbent_step": avg([run["last_incumbent_step"] for run in runs]),
        "mean_center_suppression_first100": avg(
            [run["center_suppression_first100"] for run in runs]
        ),
        "mean_row_train_final_dominance": avg(
            [run["row_train"]["final_dominance"] for run in runs]
        ),
        "mean_column_final_dominance": avg(
            [run["column"]["final_dominance"] for run in runs]
        ),
        "mean_row_return_final_dominance": avg(
            [run["row_return"]["final_dominance"] for run in runs]
        ),
        "l2_tie_events": sum(
            run[phase]["l2_tie_events"]
            for run in runs
            for phase in ("row_train", "column", "row_return")
        ),
        "incumbent_ids": id_counts(run["incumbent"] for run in runs),
        "column_final_ids": id_counts(
            run["column"]["final_dominant"] for run in runs
        ),
    }


def _selection_key(summary: dict) -> tuple:
    """Predeclared priority: whole protocol, turnover, recovery, then stability."""
    return (
        summary["full_success_rate"],
        summary["column_replacement_rate"],
        summary["row_recovery_rate"],
        summary["row_stable_rate"],
        summary["mean_column_final_dominance"] or 0.0,
        -summary["total_frac"],
        -summary["c_eta"],
    )


@dataclass(frozen=True)
class Job:
    seed: int
    total_frac: float
    c_eta: float
    dwell: int
    final_window: int
    l2_order: str = "normal"


def _run_job(job: Job) -> dict:
    return run_condition(
        job.seed,
        job.total_frac,
        job.c_eta,
        dwell=job.dwell,
        final_window=job.final_window,
        l2_order=job.l2_order,
    )


def _execute(jobs: list[Job], workers: int) -> list[dict]:
    if workers <= 1:
        return [_run_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_job, jobs))


def _group_summaries(runs: list[dict]) -> list[dict]:
    groups = {}
    for run in runs:
        key = (run["total_frac"], run["c_eta"], run["l2_order"])
        groups.setdefault(key, []).append(run)
    return [summarize(group) | {"l2_order": key[2]} for key, group in groups.items()]


def _print_summaries(label: str, summaries: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print("rho   c_eta   stable  replace recover success  col-dom  first-rival  L2-ties")
    for s in sorted(summaries, key=lambda x: (x["total_frac"], x["c_eta"], x["l2_order"])):
        print(
            f"{s['total_frac']:.2f}  {s['c_eta']:<7g} "
            f"{s['row_stable_rate']:.3f}   {s['column_replacement_rate']:.3f}   "
            f"{s['row_recovery_rate']:.3f}   {s['full_success_rate']:.3f}    "
            f"{s['mean_column_final_dominance']:.3f}    "
            f"{str(round(s['mean_first_rival_step'], 1)) if s['mean_first_rival_step'] else '-':>7}"
            f"      {s['l2_tie_events']}"
            + (f"  ({s['l2_order']})" if s["l2_order"] != "normal" else "")
        )


def run_sweep(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    dwell: int = DEFAULT_DWELL,
    final_window: int = DEFAULT_FINAL_WINDOW,
    workers: int = 1,
) -> dict:
    # Stage 1: initialization magnitude at the current C learning rate.
    stage1_jobs = [
        Job(seed, rho, 0.01, dwell, final_window)
        for rho in INIT_FRACS
        for seed in seeds
    ]
    stage1_runs = _execute(stage1_jobs, workers)
    stage1_summaries = _group_summaries(stage1_runs)
    best_init = max(stage1_summaries, key=_selection_key)
    best_rho = best_init["total_frac"]
    _print_summaries("STAGE 1: L2 INITIAL TOTAL", stage1_summaries)
    print(f"selected rho={best_rho:.2f}")

    # Stage 2: slow C-cell maturation at the selected L2 magnitude.
    stage2_jobs = [
        Job(seed, best_rho, c_eta, dwell, final_window)
        for c_eta in C_ETAS
        for seed in seeds
    ]
    stage2_runs = _execute(stage2_jobs, workers)
    stage2_summaries = _group_summaries(stage2_runs)
    best = max(stage2_summaries, key=_selection_key)
    best_c_eta = best["c_eta"]
    _print_summaries("STAGE 2: C LEARNING RATE", stage2_summaries)
    print(f"selected rho={best_rho:.2f}, c_eta={best_c_eta:g}")

    # Stage 3: repeat both tie-break orders for paired, per-seed comparison.
    confirm_jobs = [
        Job(seed, best_rho, best_c_eta, dwell, final_window, order)
        for order in ("normal", "reverse")
        for seed in seeds
    ]
    confirmation_runs = _execute(confirm_jobs, workers)
    confirmation_summaries = _group_summaries(confirmation_runs)
    _print_summaries("STAGE 3: ORDER CONFIRMATION", confirmation_summaries)

    by_order_seed = {
        (run["l2_order"], run["seed"]): run for run in confirmation_runs
    }
    paired_fields = ("incumbent", "column_replaced", "row_recovered", "full_success")
    paired_equal = {
        str(seed): all(
            by_order_seed[("normal", seed)][field]
            == by_order_seed[("reverse", seed)][field]
            for field in paired_fields
        )
        and by_order_seed[("normal", seed)]["column"]["final_dominant"]
        == by_order_seed[("reverse", seed)]["column"]["final_dominant"]
        for seed in seeds
    }
    order_invariant_rate = sum(paired_equal.values()) / len(paired_equal)
    print(f"paired order-invariant outcome rate: {order_invariant_rate:.3f}")

    return {
        "protocol": {
            "patterns": ["row 1", "col 1", "row 1"],
            "shared_center_pixel": 4,
            "dwell": dwell,
            "final_window": final_window,
            "seeds": list(seeds),
            "production_default_total_frac": L2_INIT_TOTAL_FRAC,
            "init_fracs": list(INIT_FRACS),
            "c_etas": list(C_ETAS),
            "success_definition": {
                "row_stable": "original row final-window dominance >= 0.90",
                "column_replaced": "different column owner with final-window dominance >= 0.80",
                "row_recovered": "original row owner returns with final-window dominance >= 0.80",
            },
        },
        "selection": {
            "best_total_frac": best_rho,
            "best_c_eta": best_c_eta,
            "order_invariant_rate": order_invariant_rate,
            "paired_order_equal": paired_equal,
        },
        "stage1": {"summaries": stage1_summaries, "runs": stage1_runs},
        "stage2": {"summaries": stage2_summaries, "runs": stage2_runs},
        "confirmation": {
            "summaries": confirmation_summaries,
            "runs": confirmation_runs,
        },
    }


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--dwell", type=int, default=DEFAULT_DWELL)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", default=OUT)
    args = parser.parse_args()

    results = run_sweep(
        args.seeds,
        dwell=args.dwell,
        final_window=args.final_window,
        workers=args.workers,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
