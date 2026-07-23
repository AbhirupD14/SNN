"""Minimal example: wrap an engine loop with the replay recorder.

This is a DEMONSTRATION of the recording contract, not a scientific result. It runs a very
short tiled_cc simulation and writes a versioned run directory. It declares no ownership
thresholds and makes no consolidation claim.

Run (writes to a temp dir by default so it never pollutes the repo)::

    PYTHONPATH=. .venv/bin/python experiments/replay_recorder_example.py

or against a durable output root::

    PYTHONPATH=. .venv/bin/python experiments/replay_recorder_example.py \
        --output-root experiments/runs
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine                       # noqa: E402
from experiments.replay_recorder import (                            # noqa: E402
    ReplayRecorder, STATUS_COMPLETED, FEEDBACK_NOT_APPLICABLE,
)


def run(output_root: str, boundaries: int = 40, record_every: int = 2) -> str:
    engine = SimulationEngine(seed=1, topology="tiled_cc", cc_e_count=8, leak_rate=0.0)
    engine.set_patch(1, 1)
    engine.set_pattern("row 1")

    with ReplayRecorder(
        engine,
        experiment="replay_recorder_example",
        output_root=output_root,
        record_every=record_every,
        checkpoint_every=10,
        # Declared truthfully; the recorder never infers this from neuron IDs.
        hierarchical_feedback=FEEDBACK_NOT_APPLICABLE,
        conditions={"note": "demonstration only, not a scientific result"},
        schedule={"patch": [1, 1], "pattern": "row 1", "boundaries": boundaries},
        metrics_columns=["timestep", "phase", "pattern", "firing", "winner"],
        optional_columns=["winner"],
    ) as rec:
        rec.set_annotation(phase="present", pattern="row 1")
        rec.marker("phase_start", data={"phase": "present"})

        spikes = 0
        for _ in range(boundaries):
            dyn = engine.step()
            rec.record_frame(engine)
            firing = int(dyn["stats"]["firing"])
            spikes += firing
            rec.metrics.append_row({
                "timestep": engine.timestep,
                "phase": "present",
                "pattern": "row 1",
                "firing": firing,
                "winner": dyn.get("winner") or "",
            })

        rec.finish(
            STATUS_COMPLETED,
            checks={"ran_to_completion": True},
            result={"total_firing_events": spikes},
        )
        return rec.run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", default=None,
                    help="where to write <run-id>/; default: a temporary directory")
    args = ap.parse_args(argv)

    tmp = None
    root = args.output_root
    if root is None:
        tmp = tempfile.mkdtemp(prefix="replay-example-")
        root = tmp
    run_dir = run(root)
    print(f"wrote run directory: {run_dir}")
    for name in sorted(os.listdir(run_dir)):
        size = os.path.getsize(os.path.join(run_dir, name))
        print(f"  {name:20s} {size:8d} bytes")
    if tmp is not None:
        print(f"(temporary; remove with: rm -rf {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
