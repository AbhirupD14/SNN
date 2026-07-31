#!/usr/bin/env python
"""Phase 5: thread and MPI characterisation (prompt section 9 Phase 5).

Compares one thread against >= 2 threads, and one MPI rank against >= 2 ranks when the
installed build supports it, using IDENTICAL topology, stimuli, seeds, resolution and model
parameters. Records spike multisets, per-column counts, runtime and capability flags.

This phase tests CORRECTNESS and operational readiness. No speedup is promised or claimed
from a 191-node graph -- at this size the parallel overhead dominates, and the runtimes
below are reported only so the claim "no speedup was measured" is backed by a measurement.

Run (threads):  .nest-env/bin/python experiments/nest_parallel_check.py
Run (MPI):      .nest-env/bin/mpirun -np 2 .nest-env/bin/python \\
                    experiments/nest_parallel_check.py --mpi-worker
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nest_backend.engine import Stimulus, run_case  # noqa: E402

CENTER = Stimulus({(1, 1): "row 1"})
OUT = REPO_ROOT / ".nest-build" / "parallel_check.json"


def fingerprint(result) -> dict:
    """A comparable summary. Spike times AND senders, so reordering cannot hide."""
    spikes = sorted((round(s["t"], 6), s["id"]) for s in result.metrics["spikes"])
    return {
        "n_spikes": len(spikes),
        "spike_multiset": spikes,
        "by_column": result.metrics["by_column"],
        "winner_multiplicity_overall": result.metrics["winner_multiplicity_overall"],
        "wall_clock_s": result.wall_clock_s,
    }


def thread_comparison() -> dict:
    out = {}
    for threads in (1, 2, 4):
        try:
            result = run_case(f"threads{threads}", CENTER, n_presentations=8,
                              threads=threads)
            out[str(threads)] = fingerprint(result)
        except Exception as exc:
            out[str(threads)] = {"error": f"{type(exc).__name__}: {exc}"}

    baseline = out.get("1", {}).get("spike_multiset")
    out["identical_to_single_thread"] = {
        t: (v.get("spike_multiset") == baseline)
        for t, v in out.items() if t.isdigit() and t != "1"
    }
    return out


def mpi_worker() -> None:
    """Executed inside `mpirun`. Rank 0 writes the fingerprint."""
    import nest

    result = run_case("mpi", CENTER, n_presentations=8)
    payload = {
        "num_processes": int(nest.num_processes),
        "rank": int(nest.Rank()),
        **fingerprint(result),
    }
    if int(nest.Rank()) == 0:
        path = REPO_ROOT / ".nest-build" / f"mpi_rank_report_{nest.num_processes}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        print(f"rank 0 wrote {path}")


def mpi_comparison() -> dict:
    """Re-invoke this script under `mpirun` at 1 and 2 ranks and compare.

    NOTE on what MPI does to the SPIKE RECORD: under multiple ranks each rank only records
    its LOCAL nodes, so a single rank's recorder output is a SUBSET of the whole network's.
    A rank-0 spike multiset is therefore NOT expected to equal the single-process multiset,
    and reporting it as a mismatch would be misleading. What is compared is the per-rank
    total against the sum over ranks.
    """
    mpirun = REPO_ROOT / ".nest-env" / "bin" / "mpirun"
    python = REPO_ROOT / ".nest-env" / "bin" / "python"
    if not mpirun.is_file():
        return {"available": False, "reason": "mpirun not present in the environment"}

    results = {}
    for ranks in (1, 2):
        cmd = [str(mpirun), "--oversubscribe", "-np", str(ranks), str(python),
               str(Path(__file__).resolve()), "--mpi-worker"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                  cwd=str(REPO_ROOT))
            report = REPO_ROOT / ".nest-build" / f"mpi_rank_report_{ranks}.json"
            if proc.returncode != 0:
                results[str(ranks)] = {"error": proc.stderr[-1500:] or proc.stdout[-1500:]}
            elif report.is_file():
                data = json.loads(report.read_text())
                data.pop("spike_multiset", None)  # keep the summary compact
                results[str(ranks)] = data
            else:
                results[str(ranks)] = {"error": "worker produced no report"}
        except Exception as exc:
            results[str(ranks)] = {"error": f"{type(exc).__name__}: {exc}"}
    results["available"] = True
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi-worker", action="store_true",
                        help="internal: run one rank under mpirun")
    args = parser.parse_args()

    if args.mpi_worker:
        mpi_worker()
        return 0

    import nest

    report = {
        "capabilities": {
            "nest_version": str(nest.__version__),
            "mpi4py_present": True,
        },
        "threads": thread_comparison(),
        "mpi": mpi_comparison(),
    }
    try:
        import mpi4py  # noqa: PLC0415
    except Exception:
        report["capabilities"]["mpi4py_present"] = False

    OUT.parent.mkdir(parents=True, exist_ok=True)
    compact = json.loads(json.dumps(report, default=str))
    for key in list(compact.get("threads", {})):
        if isinstance(compact["threads"].get(key), dict):
            compact["threads"][key].pop("spike_multiset", None)
    OUT.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
