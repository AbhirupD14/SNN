#!/usr/bin/env python
"""The bounded 3x3 experiment suite for the NEST event-driven prototype.

Runs the eight cases of prompt section 10 with FROZEN learning, writes per-case artifacts
to a timestamped directory under `experiments/runs/nest_3x3/` (gitignored), and prints a
compact summary. The committed narrative lives in
`docs/NEST_EVENT_DRIVEN_3X3_REPORT.md`.

This suite stands alone rather than plugging into the existing `experiments/` cluster
runner: the NEST environment is a separate interpreter by design (prompt section 8.1), so
it cannot share that runner's in-process conventions.

Run:  .nest-env/bin/python experiments/nest_3x3_suite.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nest_backend.engine import Stimulus, run_case  # noqa: E402
from nest_backend.topology import Timescales  # noqa: E402

RUNS_ROOT = REPO_ROOT / "experiments" / "runs" / "nest_3x3"

N_PRESENTATIONS = 8
SEED = 1


# ---------------------------------------------------------------------------------
# Python-engine cross-check (prompt section 5.4, measurement 1)
# ---------------------------------------------------------------------------------
def python_engine_winners(patches: dict, *, seed: int = SEED, n_boundaries: int = 24) -> dict:
    """Per-column winner history from the validated Python engine, same stimulus.

    The engine is the ORACLE, not a co-simulation: it is driven independently with the same
    patch assignment and its own `column_winners` bookkeeping is read out. Only winner
    IDENTITY is compared -- spike times are not expected to match and are not compared.
    """
    from backend.simulation import SimulationEngine

    from nest_backend.topology import reference_overrides

    engine = SimulationEngine(seed=seed, **reference_overrides(seed))
    for (patch_row, patch_col), pattern in patches.items():
        engine.set_patch_pattern(patch_row, patch_col, pattern)

    history: dict = {}
    for _ in range(n_boundaries):
        engine.step()
        for column, winner in engine.column_winners.items():
            history.setdefault(column, []).append(winner["id"])
    return history


def nest_first_winners(result) -> dict:
    """Per-column ordered list of the FIRST ordinary-E to fire in each window."""
    spikes = result.metrics["spikes"]
    period = result.settings["period_ms"]
    t0 = result.settings["t0_ms"]
    seen: dict = {}
    for spike in spikes:
        nid = spike["id"]
        role = _role_of(result, nid)
        if role != "E":
            continue
        column = _column_of(result, nid)
        window = int((spike["t"] - t0) // period)
        key = (column, window)
        if key not in seen:
            seen[key] = nid
    out: dict = {}
    for (column, window), nid in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.setdefault(column, []).append(nid)
    return out


_META_CACHE: dict = {}


def _meta(result):
    if "meta" not in _META_CACHE:
        from backend.network_spec import tiled_cc_spec

        _META_CACHE["meta"] = {n["id"]: n for n in tiled_cc_spec(cc_e_count=8)["nodes"]}
    return _META_CACHE["meta"]


def _role_of(result, nid):
    return _meta(result).get(nid, {}).get("column_role")


def _column_of(result, nid):
    return _meta(result).get(nid, {}).get("column_id")


def winner_agreement(result, patches: dict) -> dict:
    """How often the NEST latency winner is the Python engine's total-drive winner.

    Reported honestly as a MEASUREMENT, per prompt section 5.4: the two arbitrate by
    different quantities (weighted prefix-sum over arrival order vs. total frozen drive),
    so disagreement is a result to report, not a defect to tune away.
    """
    nest_winners = nest_first_winners(result)
    python_winners = python_engine_winners(patches)
    per_column: dict = {}
    for column, nest_seq in nest_winners.items():
        py_seq = python_winners.get(column, [])
        if not py_seq:
            per_column[column] = {"comparable": 0, "agree": 0, "rate": None,
                                  "note": "python engine produced no winner for this column"}
            continue
        py_mode = max(set(py_seq), key=py_seq.count)
        agree = sum(1 for nid in nest_seq if nid == py_mode)
        per_column[column] = {
            "comparable": len(nest_seq),
            "agree": agree,
            "rate": round(agree / len(nest_seq), 4) if nest_seq else None,
            "nest_winners": nest_seq,
            "python_dominant_winner": py_mode,
            "python_distinct_winners": len(set(py_seq)),
        }
    comparable = sum(v["comparable"] for v in per_column.values())
    agreed = sum(v["agree"] for v in per_column.values())
    return {
        "per_column": per_column,
        "overall_rate": round(agreed / comparable, 4) if comparable else None,
        "comparable_windows": comparable,
    }


# ---------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------
def build_cases(ts: Timescales) -> list:
    """The eight bounded cases. Each entry is (name, description, callable -> RunResult)."""
    cases = [
        ("01_center_patch_row",
         "Centre patch (1,1) driven with `row 1`.",
         lambda: run_case("01_center_patch_row", Stimulus({(1, 1): "row 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS)),
        ("02_center_patch_col",
         "Centre patch (1,1) driven with `col 1`. Contrasts with case 1 to expose any "
         "row/column asymmetry caused by the geometric scan order.",
         lambda: run_case("02_center_patch_col", Stimulus({(1, 1): "col 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS)),
        ("03_two_independent_patches",
         "Patch (0,0) with `row 1` and patch (2,2) with `col 1`; tests locality.",
         lambda: run_case("03_two_independent_patches",
                          Stimulus({(0, 0): "row 1", (2, 2): "col 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS)),
        ("04_all_nine_patches",
         "`row 1` in every patch simultaneously.",
         lambda: run_case("04_all_nine_patches",
                          Stimulus({(r, c): "row 1" for r in range(3) for c in range(3)}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS)),
        ("05_pattern_switch",
         "Centre patch `row 1` then `col 1`, WITHOUT rebuilding the network.",
         lambda: run_case("05_pattern_switch", Stimulus({(1, 1): "row 1"}),
                          switch_to=Stimulus({(1, 1): "col 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS)),
        ("06_feedback_disabled",
         "Case 1 repeated with the translated C -> I confirmation pathway disabled. At the "
         "initial C weight this is expected to be INDISTINGUISHABLE from case 1, because a "
         "C frozen at theta/4 can never fire and so never drives the pathway anyway.",
         lambda: run_case("06_feedback_disabled", Stimulus({(1, 1): "row 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS,
                          feedback_enabled=False)),
        ("06b_feedback_on_matured_c",
         "Feedback ON with C frozen at its MATURE theta weight -- the confirmation pathway "
         "is live. Pairs with 06c to make the feedback control a real comparison.",
         lambda: run_case("06b_feedback_on_matured_c", Stimulus({(1, 1): "row 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS,
                          feedback_enabled=True, c_basal_weight=1000.0)),
        ("06c_feedback_off_matured_c",
         "Feedback OFF with C frozen at its MATURE theta weight. The difference against "
         "06b isolates what the top-down confirmation reset actually does.",
         lambda: run_case("06c_feedback_off_matured_c", Stimulus({(1, 1): "row 1"}),
                          seed=SEED, timescales=ts, n_presentations=N_PRESENTATIONS,
                          feedback_enabled=False, c_basal_weight=1000.0)),
    ]
    return cases


def cadence_case(ts: Timescales) -> list:
    """Case 7: the feedback cadence law.

    The reference engine predicts strict fire/silent alternation when the presentation
    period equals the graph's own feedback loop latency `L`, and NO suppression at `L + 1`
    hop. That is a claim about RELATIVE timing, which NEST owns natively, so it should
    survive a faithful port. Failure indicates a translation error, not a native-semantic
    difference.
    """
    from nest_backend.topology import NestTiledNetwork

    probe = NestTiledNetwork(seed=SEED, timescales=ts)
    loop_ms = probe.feedback_loop_latency_ms()

    out = []
    # `C` is initialized at theta/4 and the one-shot condition is `w_basal >= theta`, so
    # under frozen learning `C` cannot fire, the `C -> I` confirmation never triggers, and
    # the cadence law is untestable. Each period is therefore run twice: once at the
    # engine's initial weight (which is what "frozen" ordinarily means here) and once with
    # `C` frozen at its MATURE weight, which is the only condition under which the law it
    # predicts can be exercised at all.
    for label, period in (("at_L", loop_ms), ("above_L", loop_ms + ts.base_ff)):
        # The presentation period must land on the `h` grid: NEST rejects a spike time it
        # cannot represent at the current resolution, and the graph-derived loop latency is
        # a mean over real delays, so it is almost never a whole number of steps.
        period_ms = max(period, ts.spread * 1.05)
        period_ms = round(round(period_ms / ts.h) * ts.h, 10)
        case_ts = Timescales(h=ts.h, base_ff=ts.base_ff, spread=ts.spread,
                             presentation=period_ms)
        which = "graph-derived loop latency L" if label == "at_L" else "L + one hop"
        out.append((
            f"07_cadence_{label}",
            f"Presentation period = {period_ms} ms ({which}); C frozen at its INITIAL "
            "theta/4 weight, so C cannot fire and the confirmation pathway is inert.",
            (lambda t=case_ts, n=f"07_cadence_{label}": run_case(
                n, Stimulus({(1, 1): "row 1"}), seed=SEED, timescales=t,
                n_presentations=12, settle_windows=1.0)),
        ))
        out.append((
            f"07_cadence_{label}_matured_c",
            f"Presentation period = {period_ms} ms ({which}); C frozen at its MATURE "
            "theta weight, so the C -> I confirmation pathway is live. This is the only "
            "condition under which the cadence law can be exercised.",
            (lambda t=case_ts, n=f"07_cadence_{label}_matured_c": run_case(
                n, Stimulus({(1, 1): "row 1"}), seed=SEED, timescales=t,
                n_presentations=12, settle_windows=1.0,
                c_basal_weight=1000.0)),
        ))
    return out


def sweep_cases(ts: Timescales) -> list:
    """Case 8: the S / L_wta timescale-separation sweep.

    Declared hypothesis (prompt section 5.5): winner multiplicity falls toward 1 as the
    ratio grows, and approaches the number of supra-threshold competitors as it approaches
    1. This is a convergence characterisation, NOT a search for the setting that
    reproduces a preferred outcome -- the full curve is reported including the settings
    where WTA fails.
    """
    out = []
    # Ratio 0 is the control: dispersion off entirely, a perfectly simultaneous volley.
    out.append((
        "08_sweep_ratio_0",
        "Dispersion DISABLED: a simultaneous volley, no arrival ordering to arbitrate.",
        lambda: run_case("08_sweep_ratio_0", Stimulus({(1, 1): "row 1"}), seed=SEED,
                         timescales=Timescales(h=ts.h, base_ff=ts.base_ff, spread=ts.spread,
                                               presentation=40.0),
                         n_presentations=N_PRESENTATIONS, dispersion_enabled=False),
    ))
    for spread in (0.2, 0.6, 2.0, 6.0, 12.0):
        ratio = spread / (2 * ts.h)
        case_ts = Timescales(h=ts.h, base_ff=ts.base_ff, spread=spread, presentation=40.0)
        out.append((
            f"08_sweep_ratio_{ratio:g}",
            f"S = {spread} ms, S / L_wta = {ratio:g}.",
            (lambda t=case_ts, n=f"08_sweep_ratio_{ratio:g}": run_case(
                n, Stimulus({(1, 1): "row 1"}), seed=SEED, timescales=t,
                n_presentations=N_PRESENTATIONS)),
        ))
    return out


def resolution_cases(ts: Timescales) -> list:
    """Case 8b: the `h` sweep at FIXED ratio.

    Hypothesis: outcomes are invariant, because only the ratio is physical. A dependence on
    absolute `h` at fixed ratio would indicate a discretization artifact and must be
    explained.
    """
    out = []
    for h in (0.05, 0.1, 0.2):
        case_ts = Timescales(h=h, base_ff=10 * h, spread=20 * h, presentation=200 * h)
        out.append((
            f"09_resolution_h{h:g}",
            f"h = {h} ms with L_wta : S : D held at 1 : 10 : 100.",
            (lambda t=case_ts, n=f"09_resolution_h{h:g}": run_case(
                n, Stimulus({(1, 1): "row 1"}), seed=SEED, timescales=t,
                n_presentations=N_PRESENTATIONS)),
        ))
    return out


# ---------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: a timestamped run dir)")
    parser.add_argument("--quick", action="store_true",
                        help="run only the six core cases")
    args = parser.parse_args(argv)

    ts = Timescales()
    run_dir = args.out or (RUNS_ROOT / time.strftime("%Y%m%dT%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(ts)
    if not args.quick:
        cases = cases + cadence_case(ts) + sweep_cases(ts) + resolution_cases(ts)

    index: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "seed": SEED,
        "n_presentations": N_PRESENTATIONS,
        "default_timescales": ts.describe(),
        "cases": {},
    }

    print(f"writing artifacts to {run_dir}\n")
    header = f"{'case':<28} {'spikes':>7} {'mult':>6} {'max':>4} {'frac1':>6} {'EorM':>5} {'C dep':>6}"
    print(header)
    print("-" * len(header))

    for name, description, runner in cases:
        result = runner()
        metrics = result.metrics
        overall = metrics["winner_multiplicity_overall"]

        payload = {
            "name": name,
            "description": description,
            "manifest": result.manifest,
            "stimulus": result.stimulus,
            "settings": result.settings,
            "wall_clock_s": result.wall_clock_s,
            "metrics": metrics,
        }
        # The Python-engine winner cross-check only makes sense for the plain cases.
        if name.startswith(("01_", "02_", "03_", "04_")):
            patches = {}
            for key, pattern in result.stimulus.items():
                row, col = (int(x) for x in key.split(","))
                patches[(row, col)] = pattern
            payload["winner_agreement"] = winner_agreement(result, patches)

        (run_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

        eor = [v["last_input_multiplicity"] for v in metrics["eor_input_multiplicity"].values()
               if v["n_spikes"]]
        deposits = sum(v["deposits"] for v in metrics["coincidence"].values())
        print(f"{name:<28} {metrics['spike_count']:>7} "
              f"{str(overall['mean']):>6} {str(overall['max']):>4} "
              f"{str(overall['fraction_single_winner']):>6} "
              f"{(max(eor) if eor else 0):>5} {deposits:>6}")

        index["cases"][name] = {
            "description": description,
            "spike_count": metrics["spike_count"],
            "winner_multiplicity_overall": overall,
            "eor_max_input_multiplicity": max(eor) if eor else None,
            "coincidence_deposits": deposits,
            "wall_clock_s": result.wall_clock_s,
            "timescales": result.manifest["timescales"],
            "realized_spread": result.manifest["delays"]["realized_spread_ms"],
            "firing_pattern": metrics["firing_pattern"],
            "agreement": payload.get("winner_agreement", {}).get("overall_rate"),
        }

    (run_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nindex written to {run_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
