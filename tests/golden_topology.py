"""Golden regression oracle for the topology-generalization refactor.

Captures a bit-exact fingerprint of a deterministic run for a given topology so
that the Phase 2 graph-driven engine refactor can be proven behaviour-preserving.

Usage:
    PYTHONPATH=. .venv/bin/python tests/golden_topology.py capture <name> <topo>
    PYTHONPATH=. .venv/bin/python tests/golden_topology.py verify  <name> <topo>

where <topo> is a topology selector understood by SimulationEngine. The capture
writes tests/golden/<name>.json; verify rebuilds and compares byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from backend.simulation import SimulationEngine, PATTERNS

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# A deterministic schedule that exercises learning, WTA turnover, and inhibition:
# dwell on each of the four patterns in turn, then revisit the first.
SCHEDULE = [("row 1", 120), ("col 1", 120), ("diag \\", 80), ("diag /", 80), ("row 1", 100)]


def _round(x, nd=6):
    return round(float(x), nd)


def _frame_fingerprint(dyn):
    """A stable, order-independent-per-field digest of one dynamic frame."""
    neurons = []
    for n in dyn["neurons"]:
        rec = [n["id"], _round(n["potential"]), _round(n["activation"]),
               int(bool(n["spiked"])), _round(n["freq"]), int(n["refractory"])]
        if "g_inh" in n:
            rec += [_round(n["g_inh"]), _round(n.get("trace", 0.0)),
                    _round(n.get("v_pre_reset", 0.0))]
        if "winner_trace" in n:
            rec += [_round(n["winner_trace"]), int(bool(n.get("residual_received", False))),
                    int(n.get("residual_events", 0)), _round(n.get("residual_charge", 0.0)),
                    _round(n.get("trace_charge", 0.0)),
                    _round(n.get("v_pre_reset", 0.0))]
        neurons.append(rec)
    changed = sorted((c["id"], _round(c["weight"])) for c in dyn.get("changed_synapses", []))
    return {
        "t": dyn["timestep"],
        "winner": dyn["winner"],
        "neurons": neurons,
        "changed": changed,
        "emitted": sorted(dyn.get("emitted", [])),
        "pulses": sorted((p["source"], p["target"], _round(p["conductance_increment"]))
                         for p in dyn.get("inhibitory_pulses", [])),
    }


def run(topo_kwargs):
    e = SimulationEngine(seed=1, **topo_kwargs)
    topo = e.topology()
    frames = []
    for name, steps in SCHEDULE:
        e.set_pattern(name)
        for _ in range(steps):
            frames.append(_frame_fingerprint(e.step()))
    # Topology fingerprint: neuron ids/meta + synapse ids/kinds/initial weights.
    topo_fp = {
        "neurons": [[n["id"], n["layer"], n["type"], n["role"], _round(n["threshold"])]
                    for n in topo["neurons"]],
        "synapses": sorted((s["id"], s["source"], s["target"], s["kind"],
                            None if s.get("weight") is None else _round(s["weight"]))
                           for s in topo["synapses"]),
        "params": {k: (_round(v) if isinstance(v, float) else v)
                   for k, v in topo["params"].items()},
    }
    return {"topology": topo_fp, "frames": frames}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def digests(blob):
    """Separate digests so intended non-behavioural changes are distinguished from a
    real dynamics change:

      * ``frames``   -- the neural INVARIANT: per-neuron potential/spike/g_inh/trace,
                        winner, learned-weight changes, and inhibitory conductance
                        pulses. This is what must never change for a fixed seed.
      * ``emitted``  -- the edge-flash visualization list (display only).
      * ``topology`` -- serialized ids/params/edge order.
    """
    dyn = [{k: v for k, v in f.items() if k != "emitted"} for f in blob["frames"]]
    emitted = [f.get("emitted", []) for f in blob["frames"]]
    return {"frames": digest(dyn), "emitted": digest(emitted),
            "topology": digest(blob["topology"])}


TOPOS = {
    "pi": dict(topology="pi"),
    "old": dict(topology="old"),
    # 'rg' bootstraps through TWO plastic hops (RG->L1E, then L1E->L2E) and its L1E
    # must accumulate ~23 RG events before its first spike, so the shared SCHEDULE's
    # per-pattern dwell is long enough to reach L2 development but this golden is
    # deliberately the same schedule as the others for comparability.
    "rg": dict(topology="rg"),
    "rg_residual": dict(topology="rg_residual"),
}


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    action, name, topo = argv[0], argv[1], argv[2]
    kwargs = TOPOS.get(topo)
    if kwargs is None:
        print(f"unknown topo {topo!r}; known: {list(TOPOS)}")
        return 1
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    blob = run(kwargs)
    if action == "capture":
        with open(path, "w") as f:
            json.dump(blob, f)
        d = digests(blob)
        print(f"captured {name} ({topo}) -> {path}  frames={d['frames'][:16]} topo={d['topology'][:16]}")
        return 0
    if action == "verify":
        with open(path) as f:
            ref = json.load(f)
        got, want = digests(blob), digests(ref)
        frames_ok = got["frames"] == want["frames"]
        topo_ok = got["topology"] == want["topology"]
        if frames_ok and topo_ok:
            print(f"OK {name} ({topo}) bit-exact  frames={got['frames'][:16]} topo={got['topology'][:16]}")
            return 0
        # Dynamics are the behavioural invariant; the topology fingerprint includes
        # serialized params/edge-ids that intended refactors may legitimately change.
        print(f"{'FRAMES OK' if frames_ok else 'FRAMES DIFFER'} / "
              f"{'TOPO OK' if topo_ok else 'TOPO DIFFERS'}  {name} ({topo})")
        if not frames_ok:
            print(f"  frames ref={want['frames'][:16]} got={got['frames'][:16]}")
            for i, (a, b) in enumerate(zip(ref["frames"], blob["frames"])):
                if a != b:
                    print(f"  first frame diff at index {i} (t={b.get('t')})")
                    break
        if not topo_ok:
            print(f"  topology ref={want['topology'][:16]} got={got['topology'][:16]}")
        # Non-zero only when the behavioural invariant (frames) broke.
        return 0 if frames_ok else 2
    print(f"unknown action {action!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
