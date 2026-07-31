#!/usr/bin/env python
"""Stage B gate: is chunked NEST execution identical to one-shot execution?

Live viewing would advance NEST in finite observation chunks. The dashboard prompt
(section 5.3) forbids shipping live mode unless chunking is proven inert:

  1. preload an identical stimulus schedule;
  2. run the case once with a single `Simulate(total_duration)`;
  3. run it again as repeated `Simulate(chunk_duration)` calls;
  4. assert identical `(timestamp, repository sender id)` spike multisets;
  5. assert identical final node counters and connection weights;
  6. repeat at 1, 2 and 4 threads where supported.

If any of that fails, live mode is not shipped and offline replay remains the delivered
feature. This script is the evidence either way.

Run:  .nest-env/bin/python experiments/nest_chunked_equivalence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nest_backend.engine import Stimulus, build_schedule  # noqa: E402
from nest_backend.recording import collect_spikes  # noqa: E402
from nest_backend.topology import NestTiledNetwork, Timescales  # noqa: E402

CENTER = Stimulus({(1, 1): "row 1"})
OUT = REPO_ROOT / ".nest-build" / "chunked_equivalence.json"


def _build(threads: int, ts: Timescales):
    net = NestTiledNetwork(seed=1, timescales=ts, threads=threads)
    schedule = build_schedule(net, CENTER, t0=ts.presentation, period=ts.presentation,
                              n_presentations=6)
    net.set_input_schedule(schedule)
    return net


def _observe(net) -> dict:
    """Everything that must be identical: spikes, counters and weights."""
    import nest  # noqa: PLC0415

    spikes = sorted((round(t, 6), nid) for t, nid in collect_spikes(net))
    counters = {}
    for nid in sorted(net.gid_of):
        if net.node_meta[nid]["archetype"] == "rg_source":
            continue
        cell = net.gid_of[nid]
        entry = {}
        for field in ("n_spikes", "n_resets", "n_deposits", "n_suppressed",
                      "n_confirm_spikes", "q", "q_pre"):
            try:
                entry[field] = float(cell.get(field))
            except Exception:
                pass
        counters[nid] = entry

    weights = {}
    for edge_id, (src_gid, tgt_gid) in sorted(net.conn_of.items()):
        conns = nest.GetConnections(source=nest.NodeCollection([src_gid]),
                                    target=nest.NodeCollection([tgt_gid]))
        try:
            weights[edge_id] = round(float(conns.get("weight")), 9)
        except Exception:
            pass
    return {"spikes": spikes, "counters": counters, "weights": weights}


def run_one_shot(threads: int, ts: Timescales, duration: float) -> dict:
    net = _build(threads, ts)
    net.simulate(duration)
    return _observe(net)


def run_chunked(threads: int, ts: Timescales, duration: float, chunk: float) -> dict:
    net = _build(threads, ts)
    remaining = duration
    calls = 0
    while remaining > 1e-9:
        step = min(chunk, remaining)
        net.simulate(step)
        remaining = round(remaining - step, 10)
        calls += 1
    out = _observe(net)
    out["simulate_calls"] = calls
    return out


def compare(a: dict, b: dict) -> dict:
    spikes_equal = a["spikes"] == b["spikes"]
    counters_equal = a["counters"] == b["counters"]
    weights_equal = a["weights"] == b["weights"]
    diff_examples = []
    if not spikes_equal:
        only_a = [s for s in a["spikes"] if s not in b["spikes"]][:5]
        only_b = [s for s in b["spikes"] if s not in a["spikes"]][:5]
        diff_examples = [{"only_one_shot": only_a, "only_chunked": only_b}]
    if not counters_equal:
        for nid, values in a["counters"].items():
            if b["counters"].get(nid) != values:
                diff_examples.append({"node": nid, "one_shot": values,
                                      "chunked": b["counters"].get(nid)})
                if len(diff_examples) > 5:
                    break
    return {
        "spike_multiset_identical": spikes_equal,
        "counters_identical": counters_equal,
        "weights_identical": weights_equal,
        "identical": spikes_equal and counters_equal and weights_equal,
        "n_spikes_one_shot": len(a["spikes"]),
        "n_spikes_chunked": len(b["spikes"]),
        "differences": diff_examples,
    }


def main() -> int:
    ts = Timescales()
    duration = ts.presentation * 9        # 6 presentations + settle
    # The observation cadence: an integer multiple of h (NEST rejects anything else), and
    # deliberately NOT a divisor of the presentation interval, so chunk boundaries land
    # mid-volley -- the case most likely to expose a difference if one exists. 6.7 ms is
    # 67 ticks at h = 0.1 and never aligns with the 20 ms presentation.
    chunk = round(round(ts.presentation / 3.0 / ts.h) * ts.h, 10)

    if abs(chunk / ts.h - round(chunk / ts.h)) > 1e-9:
        raise SystemExit(f"chunk {chunk} is not a multiple of h={ts.h}")

    report = {
        "duration_ms": duration,
        "chunk_ms": chunk,
        "chunk_is_multiple_of_h": abs(chunk / ts.h - round(chunk / ts.h)) < 1e-9,
        "timescales": ts.describe(),
        "threads": {},
    }

    for threads in (1, 2, 4):
        try:
            one = run_one_shot(threads, ts, duration)
            many = run_chunked(threads, ts, duration, chunk)
            result = compare(one, many)
            result["simulate_calls_chunked"] = many.get("simulate_calls")
            report["threads"][str(threads)] = result
        except Exception as exc:
            report["threads"][str(threads)] = {"error": f"{type(exc).__name__}: {exc}"}

    report["all_identical"] = all(
        v.get("identical") is True for v in report["threads"].values()
    )
    report["verdict"] = (
        "PASS: chunked execution is identical to one-shot at every tested thread count; "
        "live observation chunks are safe."
        if report["all_identical"] else
        "FAIL: chunked execution changes results. Live mode must NOT be shipped; offline "
        "replay remains the delivered feature."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nwritten to {OUT}")
    return 0 if report["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
