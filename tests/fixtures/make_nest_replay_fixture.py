"""Generate the committed NEST replay fixture THROUGH THE REAL ADAPTER.

Writes two artifacts next to this script:

    nest_replay_fixture.snn.jsonl      a short, self-contained NEST replay
    nest_replay_fixture.expected.json  the source-of-truth measurements the JS parser
                                       test asserts against (spike multiset, counts,
                                       provenance), taken from the NEST RunResult itself
                                       rather than from the artifact -- so the JS test
                                       compares the artifact against the run, not against
                                       itself.

Never hand-maintained. Regenerate after any adapter or schema change:

    .nest-env/bin/python tests/fixtures/make_nest_replay_fixture.py

Requires the NEST environment (see nest_backend/README.md). The fixture is deliberately
short (5 presentations) so it stays small enough to commit while still containing
multi-winner windows, coincidence activity and a pattern marker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nest_backend.engine import Stimulus, run_case  # noqa: E402
from nest_backend.replay_adapter import build_records, write_replay  # noqa: E402

CASE = "fixture_center_patch_row"
PRESENTATIONS = 5
SEED = 1


def main() -> int:
    result = run_case(CASE, Stimulus({(1, 1): "row 1"}), seed=SEED,
                      n_presentations=PRESENTATIONS)

    replay_path = HERE / "nest_replay_fixture.snn.jsonl"
    info = write_replay(result, replay_path)

    records = build_records(result)
    frames = [r for r in records if r["record"] == "frame"]

    expected = {
        "case": CASE,
        "seed": SEED,
        "presentations": PRESENTATIONS,
        # Structure
        "neuron_count": len(result.topology["neurons"]),
        "synapse_count": len(result.topology["synapses"]),
        "weighted_synapse_count": sum(1 for s in result.topology["synapses"]
                                      if s["weight"] is not None),
        # The authoritative event record, straight from the NEST run. Cortical spikes and
        # retinal input events come from two DIFFERENT NEST recorders (generators are
        # devices, model neurons are not) but both are firing cells in the renderer, so
        # both are expected to carry a `spiked` flag.
        "cortical_spike_count": result.metrics["spike_count"],
        "cortical_spikes": sorted([round(float(s["t"]), 6), s["id"]]
                                  for s in result.metrics["spikes"]),
        "input_event_count": len(result.metrics["input_schedule"]),
        "input_events": sorted([round(float(e["t"]), 6), e["id"]]
                               for e in result.metrics["input_schedule"]),
        "all_events": sorted(
            [[round(float(s["t"]), 6), s["id"]] for s in result.metrics["spikes"]]
            + [[round(float(e["t"]), 6), e["id"]]
               for e in result.metrics["input_schedule"]]),
        "lit_rgc_ids": sorted({e["id"] for e in result.metrics["input_schedule"]}),
        # Frame shape
        "frame_count": len(frames),
        "frame_ticks": [f["timestep"] for f in frames],
        "frame_t_ms": [f["dynamic"]["nest"]["t_ms"] for f in frames],
        # Provenance the artifact must carry
        "resolution_h_ms": result.manifest["timescales"]["h_ms"],
        "learning_mode": "frozen",
        "mpi_available": False,
        "nest_version": result.manifest["versions"]["nest"],
        "nestml_version": result.manifest["versions"]["nestml"],
        # The headline scientific observation
        "winner_multiplicity_overall": result.metrics["winner_multiplicity_overall"],
        "artifact": {"records": info["records"], "frames": info["frames"],
                     "markers": info["markers"]},
    }

    expected_path = HERE / "nest_replay_fixture.expected.json"
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"replay": str(replay_path), **info,
                      "expected": str(expected_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
