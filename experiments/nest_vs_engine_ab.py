#!/usr/bin/env python
"""Produce a MATCHED PAIR of replays -- same topology, seed and stimulus -- one recorded
from the validated Python engine, one from the NEST prototype.

Both land in the same directory and both open with the dashboard's ordinary Load Test
control, so the two engines can be compared frame by frame on identical input.

    .nest-env/bin/python experiments/nest_vs_engine_ab.py --shape 3x3 --pattern "row 1"

Why one script and not two
++++++++++++++++++++++++++

The comparison is only meaningful if NOTHING differs except the engine. Driving both from
one entry point is what guarantees that: identical spec, identical seed, identical pattern,
identical presentation count. Anything a reader might otherwise have to take on trust is
written into `pair.json` next to the artifacts.

What the pair CANNOT be
+++++++++++++++++++++++

Not a trace comparison. The engines have different execution models -- the reference
resolves crossings at continuous sub-boundary `tau`, NEST quantises to an `h` grid -- so
spike TIMES are not expected to match and are not compared. What is comparable is the
invariant outcome: which competitor wins, how many win per presentation, and whether the
column's firing pattern alternates.

Requires the NEST environment (it drives both engines from one process; the Python engine
has no NEST dependency and runs fine there).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.replay_recorder import (  # noqa: E402
    FEEDBACK_NOT_APPLICABLE,
    STATUS_COMPLETED,
    ReplayRecorder,
)

DEFAULT_OUT = REPO_ROOT / "experiments" / "runs" / "nest_3x3" / "ab"


# ------------------------------------------------------------------ python engine
def record_python(shape, pattern, seed, presentations, out_dir):
    """Drive the validated Python engine and record it with the existing recorder.

    The engine is stepped at its OWN cadence -- `input_period = 0` auto-matches the graph's
    feedback loop latency -- because forcing it onto NEST's millisecond schedule would
    change the very thing under comparison.
    """
    from backend.network_spec import tiled_cc_spec
    from backend.simulation import SimulationEngine

    from nest_backend.topology import reference_overrides

    engine = SimulationEngine(seed=seed, **reference_overrides(seed))
    rows, cols = shape
    engine.apply_topology(tiled_cc_spec(cc_e_count=8, input_rows=rows, input_cols=cols,
                                        patch_rows=rows, patch_cols=cols))
    engine.set_patch_pattern(0, 0, pattern)

    period = engine.resolved_input_period()
    boundaries = presentations * period

    with ReplayRecorder(
        engine,
        experiment="ab_python_engine",
        output_root=str(out_dir),
        run_id="python_engine",
        seed=seed,
        record_every=1,
        checkpoint_every=50,
        conditions={"engine": "python", "shape": f"{rows}x{cols}", "pattern": pattern},
        schedule={"presentations": presentations, "input_period": period,
                  "boundaries": boundaries},
        hierarchical_feedback=FEEDBACK_NOT_APPLICABLE,
    ) as rec:
        rec.set_annotation(phase="ab", pattern=pattern)
        rec.marker("presentation", data={"pattern": pattern, "patch": [0, 0]})
        winners = []
        for _ in range(boundaries):
            engine.step()
            rec.record_frame(engine)
            if engine.column_winners:
                winners.append({k: v["id"] for k, v in engine.column_winners.items()})
        rec.finish(STATUS_COMPLETED, result={"winners": winners})

    return {
        "path": str(out_dir / "python_engine" / "replay.snn.jsonl"),
        "boundaries": boundaries,
        "input_period": period,
        "loop_latency_boundaries": engine.feedback_loop_latency,
        "winner_samples": winners[:12],
        "distinct_winners": sorted({w for d in winners for w in d.values()}),
    }


# ------------------------------------------------------------------ pair report shape
def build_nest_section(net, spikes, *, t0, period, python_owners=None):
    """Assemble the NEST half of `pair.json`. PURE: no simulation, no file IO.

    Split out of `record_nest` so the REPORT SHAPE can be tested directly. The audit's
    finding was not that the numbers were wrong but that the report answered the wrong
    question: it carried winner multiplicity and no owner identity, so a run whose owner
    rotated across every window scored identically to one with a single stable owner and
    was labelled "matched".

    `python_owners` is `{column_id: owner_node_id}` from the compatibility oracle. Agreement
    is computed HERE, in the reporting layer, from the modal owner the metrics layer
    reports -- so when the metrics layer cannot name an owner, agreement is `None`
    (unknown) rather than a default that would read as a passing comparison.
    """
    from nest_backend.recording import (  # noqa: PLC0415
        aggregate_multiplicity, winner_multiplicity,
    )

    summary = winner_multiplicity(net, spikes, t0, period)
    owners = python_owners or {}
    columns = {}
    for column, stats in summary.items():
        entry = dict(stats)
        owner = entry.get("owner")
        entry["python_owner"] = owners.get(column)
        entry["agrees_with_python"] = (
            None if owner is None or column not in owners
            else owner == owners[column]
        )
        columns[column] = entry
    return {
        "winner_multiplicity": aggregate_multiplicity(summary),
        "columns": columns,
    }


# --------------------------------------------------------------------------- nest
def record_nest(shape, pattern, seed, presentations, out_dir, h, spread, jitter,
                presentation_ms, python_owners=None):
    from nest_backend.engine import Stimulus, run_case
    from nest_backend.replay_adapter import write_replay
    from nest_backend.topology import Timescales

    ts = Timescales(h=h, base_ff=1.0, spread=spread, presentation=presentation_ms)
    result = run_case("ab_nest", Stimulus({(0, 0): pattern}), seed=seed, timescales=ts,
                      n_presentations=presentations, shape=shape, jitter_ms=jitter)
    path = out_dir / "nest" / "replay.snn.jsonl"
    info = write_replay(result, path)
    # The emitted section goes through `build_nest_section`, the same pure path the Phase 0
    # probe exercises, so the tested report shape and the written one cannot drift apart.
    # `winner_multiplicity` reads column metadata off the network; the run has finished and
    # released it, so the recorded topology stands in as the metadata view.
    class _MetaView:
        node_meta = {n["id"]: n for n in result.topology["neurons"]}

    spikes = [(row["t"], row["id"]) for row in result.metrics["spikes"]]
    section = build_nest_section(
        _MetaView(), spikes,
        t0=float(result.settings.get("t0_ms") or 0.0),
        period=float(result.settings.get("period_ms") or presentation_ms),
        python_owners=python_owners,
    )
    return {
        "path": info["path"],
        "frames": info["frames"],
        "timescales": result.manifest["timescales"],
        "engine_profile": result.manifest.get("engine_profile"),
        "jitter_ms": jitter,
        "winner_multiplicity": section["winner_multiplicity"],
        "columns": section["columns"],
        "firing_pattern": result.metrics["firing_pattern"],
        "spike_count": result.metrics["spike_count"],
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="3x3", help="input/patch shape, e.g. 3x3 or 9x9")
    parser.add_argument("--pattern", default="row 1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--presentations", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    # NEST-side timing. The defaults are the configuration under which single-winner WTA
    # holds on 8/8 seeds; pass --h 0.1 --jitter 0 to reproduce the failing regime instead.
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--spread", type=float, default=2.0)
    parser.add_argument("--jitter", type=float, default=3.0)
    parser.add_argument("--presentation-ms", type=float, default=60.0)
    args = parser.parse_args(argv)

    rows, cols = (int(x) for x in args.shape.lower().split("x"))
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"recording matched pair -> {out_dir}\n")
    python_info = record_python((rows, cols), args.pattern, args.seed,
                                args.presentations, out_dir)
    print(f"  python engine: {python_info['path']}")
    # The Python owner per column, so the NEST section can report agreement rather than
    # leaving the reader to eyeball two lists.
    python_owners = {}
    for sample in python_info.get("winner_samples", []):
        for column, owner in sample.items():
            python_owners.setdefault(column, owner)
    nest_info = record_nest((rows, cols), args.pattern, args.seed, args.presentations,
                            out_dir, args.h, args.spread, args.jitter,
                            args.presentation_ms, python_owners=python_owners)
    print(f"  nest:          {nest_info['path']}")

    pair = {
        "matched_on": {
            "topology": f"tiled_cc input {rows}x{cols}, patch {rows}x{cols}, cc_e_count 8",
            "seed": args.seed,
            "pattern": args.pattern,
            "patch": [0, 0],
            "presentations": args.presentations,
        },
        "not_compared": (
            "Spike TIMES. The engines have different execution models -- the reference "
            "resolves crossings at continuous sub-boundary tau, NEST quantises to the h "
            "grid -- so exact timing agreement is neither expected nor meaningful."
        ),
        "python_engine": python_info,
        "nest": nest_info,
    }
    (out_dir / "pair.json").write_text(json.dumps(pair, indent=2, default=str) + "\n")

    print("\n--- comparison -------------------------------------------------------")
    print(f"  python : loop latency L = {python_info['loop_latency_boundaries']} boundaries, "
          f"input_period = {python_info['input_period']}")
    print(f"           distinct column winners: {python_info['distinct_winners']}")
    print(f"  nest   : h = {nest_info['timescales']['h_ms']} ms, "
          f"jitter = {nest_info['jitter_ms']} ms")
    m = nest_info["winner_multiplicity"]
    print(f"           winners/window mean {m['mean']}, max {m['max']}, "
          f"single-winner {m['fraction_single_winner']}")
    print(f"           firing pattern {nest_info['firing_pattern']}")
    print(f"\npair.json written to {out_dir / 'pair.json'}")
    print("Open BOTH with the dashboard's Load Test control to compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
