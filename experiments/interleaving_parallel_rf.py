"""Interleaving test run across every RF network in parallel, recorded as a replay.

The 9x9 ``tiled_cc`` preset has a 3x3 grid of cortical columns, one per 3x3 input patch
-- i.e. nine independent receptive-field (RF) networks. This experiment runs the SAME
interleaving schedule in every RF at once: each phase drives ALL nine patches with one
pattern, cycling ``row 1 -> col 1 -> diag \\ -> diag /`` and repeating for several cycles.
Because every column sees the identical interleaved sequence simultaneously, the replay
lets you watch how each independent RF specializes under interleaved presentation and
compare them side by side.

Learning uses the experimental dual FE/FES rule (``dual_fe_fes=True``): both the
ordinary/latency-E feedforward learning and the coincidence basal learning switch to the
inverse-quadratic dual node/synapse free-energy equations.

The run writes a versioned directory under ``--output-root`` (default
``experiments/runs/``, which is gitignored); load the ``replay.snn.jsonl`` in the
dashboard via the "Load test" button to watch it unfold.

Run::

    PYTHONPATH=. .venv/bin/python experiments/interleaving_parallel_rf.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine                       # noqa: E402
from experiments.replay_recorder import (                            # noqa: E402
    ReplayRecorder, STATUS_COMPLETED, FEEDBACK_NOT_APPLICABLE,
)

# The four named 3x3 input patterns, presented in this interleaved order.
INTERLEAVE = ["row 1", "col 1", "diag \\", "diag /"]


def _all_patches(engine: SimulationEngine):
    """Every (row, col) patch coordinate of the tiled grid."""
    grid = engine.tiled_meta["grid_shape"]
    return [(r, c) for r in range(int(grid["rows"])) for c in range(int(grid["cols"]))]


def run(output_root: str, *, seed: int = 1, dwell: int = 200, cycles: int = 75,
        leak_rate: float = 0.0, record_every: int = 40) -> str:
    engine = SimulationEngine(seed=seed, topology="tiled_cc", dual_fe_fes=True,
                              leak_rate=leak_rate)
    patches = _all_patches(engine)
    n_patches = len(patches)
    boundaries_total = len(INTERLEAVE) * dwell * cycles

    with ReplayRecorder(
        engine,
        experiment="interleaving_parallel_rf",
        output_root=output_root,
        record_every=record_every,
        # One full weight snapshot per RECORDED frame. With record_every > 1 the frames'
        # changed_synapses undercount the skipped steps, so the player snaps each frame to
        # its checkpoint (replay.js advanceWeights) -- making the RF/weights view exact at
        # every displayed frame despite the sampling. checkpoint_every counts frames, so
        # 1 == checkpoint after every recorded frame.
        checkpoint_every=1,
        hierarchical_feedback=FEEDBACK_NOT_APPLICABLE,
        conditions={
            "preset": "tiled_cc",
            "dual_fe_fes": True,
            "leak_rate": leak_rate,
            "drive": "synchronized: all patches driven with the same pattern each phase",
            "rf_networks": n_patches,
            "interleave_order": INTERLEAVE,
            "dwell_boundaries": dwell,
            "cycles": cycles,
            "note": "observation artifact; not a scientific claim of consolidation",
        },
        schedule={"interleave": INTERLEAVE, "dwell": dwell, "cycles": cycles,
                  "boundaries": boundaries_total, "patches": n_patches},
        metrics_columns=["timestep", "cycle", "phase", "pattern", "firing", "winner"],
        optional_columns=["winner"],
    ) as rec:
        spikes = 0
        for cycle in range(cycles):
            for pattern in INTERLEAVE:
                # Drive every RF patch with this phase's pattern (synchronized parallel).
                for (r, c) in patches:
                    engine.set_patch_pattern(r, c, pattern)
                rec.set_annotation(phase=f"cycle {cycle + 1}", pattern=pattern)
                rec.marker("phase_start", data={"cycle": cycle + 1, "pattern": pattern,
                                                 "patches": n_patches})
                for _ in range(dwell):
                    dyn = engine.step()
                    rec.record_frame(engine)
                    firing = int(dyn["stats"]["firing"])
                    spikes += firing
                    rec.metrics.append_row({
                        "timestep": engine.timestep,
                        "cycle": cycle + 1,
                        "phase": f"cycle {cycle + 1}",
                        "pattern": pattern,
                        "firing": firing,
                        "winner": dyn.get("winner") or "",
                    })

        rec.finish(
            STATUS_COMPLETED,
            checks={"ran_to_completion": True,
                    "boundaries": rec.frames_written > 0},
            result={"total_firing_events": spikes,
                    "boundaries": boundaries_total,
                    "rf_networks": n_patches},
        )
        return rec.run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    ap.add_argument("--output-root", default=default_root,
                    help="where to write <run-id>/ (default: experiments/runs/)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dwell", type=int, default=200,
                    help="boundaries per interleaved phase (default 200)")
    ap.add_argument("--cycles", type=int, default=75,
                    help="times to repeat the full interleave order (default 75). "
                         "Total boundaries = 4 * dwell * cycles; per-pattern exposure = "
                         "dwell * cycles. Defaults => 60000 total, 15000/pattern.")
    ap.add_argument("--record-every", type=int, default=40, dest="record_every",
                    help="sample one frame every N boundaries for the replay (default 40). "
                         "Weights stay exact at every recorded frame (per-frame checkpoint).")
    ap.add_argument("--leak", type=float, default=0.0,
                    help="membrane leak rate (default 0.0)")
    args = ap.parse_args(argv)

    per_pattern = args.dwell * args.cycles
    total = 4 * per_pattern
    print(f"schedule: dwell={args.dwell} x cycles={args.cycles} => "
          f"{per_pattern} boundaries/pattern, {total} total; "
          f"~{total // args.record_every} recorded frames (record_every={args.record_every})")
    run_dir = run(args.output_root, seed=args.seed, dwell=args.dwell,
                  cycles=args.cycles, leak_rate=args.leak,
                  record_every=args.record_every)
    print(f"wrote run directory: {run_dir}")
    for name in sorted(os.listdir(run_dir)):
        size = os.path.getsize(os.path.join(run_dir, name))
        print(f"  {name:22s} {size:10d} bytes")
    print(f"\nReplay: {os.path.join(run_dir, 'replay.snn.jsonl')}")
    print("Load it in the dashboard via the 'Load test' button to watch it unfold.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
