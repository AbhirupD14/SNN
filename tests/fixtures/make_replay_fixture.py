"""Generate the tiny deterministic replay fixture used by the dashboard replay
player tests, THROUGH THE REAL RECORDER (never hand-maintained).

Writes two committed artifacts next to this script:

    replay_fixture.snn.jsonl      a short, self-contained recorder replay
    replay_fixture.expected.json  {frame_index: {syn_id: weight}} reference
                                  reconstruction at every frame, computed by the
                                  recorder's own pure reconstruct_weights_at

The JS parser test (tests/replay.parser.test.mjs) parses the .jsonl with the
browser player's parser and asserts its reconstruction matches expected.json, so
the JS and Python weight-reconstruction agree bit-for-bit off the same file.

Regenerate after any schema change::

    PYTHONPATH=. .venv/bin/python tests/fixtures/make_replay_fixture.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.simulation import SimulationEngine                       # noqa: E402
from experiments.replay_recorder import (                            # noqa: E402
    ReplayRecorder, STATUS_COMPLETED, FEEDBACK_INTACT,
    read_records, reconstruct_weights_at, REC_FRAME,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "replay_fixture.snn.jsonl")
EXPECTED = os.path.join(HERE, "replay_fixture.expected.json")


def build(tmp_root: str) -> str:
    # Deterministic: fixed seed, learning on so the run produces real
    # changed_synapses across multiple weight checkpoints. rg_coincidence keeps
    # the fixture tiny (45 neurons) while still exercising every record kind.
    engine = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    engine.set_pattern("row 1")

    with ReplayRecorder(
        engine,
        experiment="replay_fixture",
        output_root=tmp_root,
        run_id="replay-fixture",
        record_every=2,
        checkpoint_every=6,
        hierarchical_feedback=FEEDBACK_INTACT,
        conditions={"note": "test fixture only, not a scientific result"},
        schedule={"pattern": "row 1"},
        metrics_columns=["timestep", "phase", "pattern", "firing"],
    ) as rec:
        rec.set_annotation(phase="present-A", pattern="row 1")
        rec.marker("phase_start", data={"phase": "present-A"})
        for i in range(24):
            if i == 12:                       # a mid-run phase/pattern transition
                engine.set_pattern("col 1")
                rec.set_annotation(phase="present-B", pattern="col 1")
                rec.marker("pattern_change", data={"pattern": "col 1"})
            dyn = engine.step()
            rec.record_frame(engine)
            rec.metrics.append_row({
                "timestep": engine.timestep, "phase": rec._annotation["phase"],
                "pattern": rec._annotation["pattern"], "firing": int(dyn["stats"]["firing"]),
            })
        rec.marker("recall_start", data={"phase": "recall"})
        rec.finish(STATUS_COMPLETED, checks={"generated": True},
                   result={"note": "fixture"})
        return rec.run_dir


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="replay-fixture-")
    try:
        run_dir = build(tmp)
        src = os.path.join(run_dir, "replay.snn.jsonl")
        os.makedirs(HERE, exist_ok=True)
        shutil.copyfile(src, FIXTURE)

        records = read_records(FIXTURE)
        expected = {}
        for r in records:
            if r.get("record") == REC_FRAME:
                fi = r["frame_index"]
                expected[str(fi)] = reconstruct_weights_at(records, fi)
        with open(EXPECTED, "w", encoding="utf-8") as f:
            json.dump(expected, f, indent=0, sort_keys=True)

        n_frames = len(expected)
        size = os.path.getsize(FIXTURE)
        print(f"wrote {FIXTURE} ({size} bytes, {n_frames} frames)")
        print(f"wrote {EXPECTED} ({len(expected)} frame reconstructions)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
