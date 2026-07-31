#!/usr/bin/env python
"""Phase 1 feasibility spike: can a supported NEST model recover the CIPP winner race?

    .nest-env/bin/python experiments/nest_cipp_phase1_feasibility.py

Per `prompts/Claude_NEST_CIPP_Semantic_Repair_Prompt.md` Phase 1. This is a FEASIBILITY
SPIKE, not a port: two competitors, one shared source, nothing else. The canonical topology
is deliberately not translated -- the question here is narrow enough to answer on a
microcircuit, and answering it on the full graph would confound it with tiling.

The question
++++++++++++

The impulse prototype applies each arrival as an instantaneous jump and fires in the
receipt handler, so simultaneous deltas cross on the same step and the current backend
invents per-edge delay dispersion to break the tie. That replaces the decision rule::

    CIPP / Python oracle:  earliest crossing under total frozen drive
    impulse prototype:     earliest weighted prefix sum in DELAY order

So: with UNIFORM conduction delay and no jitter, can total supported drive alone decide
who fires first, and can a fast-but-nonzero inhibitory loop commit that decision?

What is deliberately absent
+++++++++++++++++++++++++++

* geometry -- every equivalent afferent has the same delay;
* random jitter -- there is none, at any point;
* zero-delay synapses -- NEST has no such thing; the E->I->E loop is explicitly nonzero
  and its latency is MEASURED, not assumed away.

Candidate order (the brief's, not negotiable): a built-in precise model first, then a
NESTML continuous model, then a minimal custom C++ extension. Precise models are not to be
rejected merely for lacking zero-delay synapses -- the question is whether they recover the
membrane-latency race.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "experiments" / "runs" / "cipp_phase1"

# The reference candidate. `iaf_psc_exp_ps` integrates its subthreshold dynamics EXACTLY
# between events and locates the off-grid threshold crossing by regula-falsi root finding
# (see the NEST model documentation for `iaf_psc_exp_ps`). Continuous postsynaptic current
# plus off-grid crossing is exactly the pair of properties the impulse accumulator lacks --
# it applies each arrival as an instantaneous jump and fires in the receipt handler.
CANDIDATE = "iaf_psc_exp_ps"

# One shared configuration for every measurement below, so no gate can pass because it was
# quietly given a different circuit. Weights are in pA; the model's defaults put threshold
# 15 mV above rest with C_m = 250 pF and tau_m = 10 ms.
CIRCUIT = {
    "neuron": {"V_th": -55.0, "E_L": -70.0, "V_reset": -70.0, "C_m": 250.0,
               "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0, "t_ref": 2.0},
    # The inhibitory relay fires as soon as any competitor does: one competitor spike is
    # far more than enough to cross it.
    "relay": {"V_th": -69.0, "E_L": -70.0, "V_reset": -70.0, "C_m": 250.0,
              "tau_m": 10.0, "tau_syn_ex": 0.5, "tau_syn_in": 2.0, "t_ref": 0.5},
    "w_e_to_i": 60000.0,      # drives the relay across threshold immediately
    "w_i_to_e": -900000.0,    # a decisive local veto, not a graded nudge
    "source_delay": 1.0,      # UNIFORM: identical for every equivalent afferent
    "d_ei": 0.1,              # nonzero by necessity -- NEST has no zero delay
    "d_ie": 0.1,
    "t_source": 10.0,
}


def _build(nest, w_strong, w_weak, *, h, source_delay, wta, d_ei=None, d_ie=None,
           unit_weight=None, reverse_creation_order=False):
    """Two competitors on synchronous sources. `wta` toggles the inhibitory loop.

    Competitors are addressed by LOGICAL ROLE (`"strong"` / `"weak"`), never by index or
    GID. Each is created by its own `Create` call, and `reverse_creation_order` flips which
    role is created first -- so the strong competitor genuinely holds the higher GID in the
    reversed case. Creating both in one `Create(model, 2)` call, as an earlier version did,
    made the flag reverse only connection insertion order while the strong cell kept the
    lower GID; that tested less than it appeared to.

    `unit_weight` decomposes each competitor's drive into that many equal afferents, EACH
    FROM ITS OWN spike generator firing at the same instant. Distinct source nodes are what
    the real mechanism has; parallel connections from a single generator would leave the
    multi-source summation path unexercised.
    """
    cfg = CIRCUIT
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": h, "rng_seed": 1, "local_num_threads": 1})

    weight_of = {"strong": float(w_strong), "weak": float(w_weak)}
    creation_order = ["weak", "strong"] if reverse_creation_order else ["strong", "weak"]

    node_of = {}
    for role in creation_order:
        node_of[role] = nest.Create(CANDIDATE, 1, cfg["neuron"])

    recorder = nest.Create("spike_recorder")
    for role in creation_order:
        nest.Connect(node_of[role], recorder)

    for role in creation_order:
        total = weight_of[role]
        if unit_weight:
            count, remainder = divmod(total, float(unit_weight))
            if remainder:
                raise ValueError(
                    f"weight {total} is not a whole multiple of unit_weight {unit_weight}"
                )
            for _ in range(int(count)):
                source = nest.Create("spike_generator", 1,
                                     {"spike_times": [cfg["t_source"]]})
                nest.Connect(source, node_of[role],
                             syn_spec={"weight": float(unit_weight),
                                       "delay": float(source_delay)})
        else:
            source = nest.Create("spike_generator", 1, {"spike_times": [cfg["t_source"]]})
            nest.Connect(source, node_of[role],
                         syn_spec={"weight": total, "delay": float(source_delay)})

    relay_recorder = None
    if wta:
        relay = nest.Create(CANDIDATE, 1, cfg["relay"])
        relay_recorder = nest.Create("spike_recorder")
        nest.Connect(relay, relay_recorder)
        for role in creation_order:
            nest.Connect(node_of[role], relay,
                         syn_spec={"weight": cfg["w_e_to_i"],
                                   "delay": float(d_ei if d_ei is not None
                                                  else cfg["d_ei"])})
            nest.Connect(relay, node_of[role],
                         syn_spec={"weight": cfg["w_i_to_e"],
                                   "delay": float(d_ie if d_ie is not None
                                                  else cfg["d_ie"])})

    role_of_gid = {int(node_of[role].global_id): role for role in node_of}
    return node_of, role_of_gid, recorder, relay_recorder


def _spikes(nest, role_of_gid, recorder):
    """`{role: [precise spike times]}` -- RAW, unrounded, keyed by LOGICAL ROLE.

    Keying by role rather than by GID or creation index is what makes the creation-order
    permutation meaningful: the same logical competitor is compared across both builds.

    Rounding here would manufacture agreement -- two crossings differing in the 15th
    decimal would compare equal and be reported as identical. Comparisons downstream state
    their own tolerance explicitly.
    """
    events = recorder.get("events")
    out: dict = {"strong": [], "weak": []}
    for sender, t in zip(events["senders"], events["times"]):
        role = role_of_gid.get(int(sender))
        if role is not None:
            out[role].append(float(t))
    return {role: sorted(times) for role, times in out.items()}


def crossing_times(nest, w_strong, w_weak, *, h=0.01, source_delay=None,
                   unit_weight=None, reverse_creation_order=False):
    """Counterfactual crossing times with the WTA loop DISABLED, raw and unrounded."""
    delay = CIRCUIT["source_delay"] if source_delay is None else source_delay
    _nodes, role_of_gid, recorder, _ = _build(
        nest, w_strong, w_weak, h=h, source_delay=delay, wta=False,
        unit_weight=unit_weight, reverse_creation_order=reverse_creation_order)
    nest.Simulate(CIRCUIT["t_source"] + delay + 60.0)
    spikes = _spikes(nest, role_of_gid, recorder)
    return {
        "strong": spikes["strong"][0] if spikes["strong"] else None,
        "weak": spikes["weak"][0] if spikes["weak"] else None,
        "arrival": CIRCUIT["t_source"] + delay,
        "gid_of_strong": next(g for g, r in role_of_gid.items() if r == "strong"),
        "gid_of_weak": next(g for g, r in role_of_gid.items() if r == "weak"),
    }


def wta_outcome(nest, w_strong, w_weak, *, h=0.01, source_delay=None, d_ei=None, d_ie=None,
                unit_weight=None, tie_tolerance_ms=0.0):
    """Committed winners with the inhibitory loop ENABLED, plus the measured loop latency.

    `winner` is the ROLE the circuit actually committed to, and is `None` unless exactly
    one competitor was allowed to fire. An earlier version took `min((time, index))`, which
    silently resolved an exact tie by node index -- manufacturing a winner from creation
    order, the same class of error as manufacturing one from delay jitter.
    """
    delay = CIRCUIT["source_delay"] if source_delay is None else source_delay
    _nodes, role_of_gid, recorder, relay_recorder = _build(
        nest, w_strong, w_weak, h=h, source_delay=delay, wta=True, d_ei=d_ei, d_ie=d_ie,
        unit_weight=unit_weight)
    nest.Simulate(CIRCUIT["t_source"] + delay + 60.0)
    spikes = _spikes(nest, role_of_gid, recorder)
    relay = sorted(float(t) for t in relay_recorder.get("events")["times"])

    committed = [role for role in ("strong", "weak") if spikes[role]]
    earliest = min((spikes[role][0] for role in committed), default=None)
    tied = [role for role in committed
            if earliest is not None and abs(spikes[role][0] - earliest) <= tie_tolerance_ms]

    winner = committed[0] if len(committed) == 1 else None
    classification = (
        "no_spike" if not committed
        else "committed" if len(committed) == 1
        else "exact_tie" if len(tied) > 1
        else "unresolved_race"
    )

    loop_latency = None
    if relay and earliest is not None:
        loop_latency = relay[0] - earliest + (d_ie if d_ie is not None else CIRCUIT["d_ie"])
    return {
        "spikes": spikes,
        "relay_spikes": relay,
        "committed": committed,
        "n_committed": len(committed),
        "winner": winner,
        "first_to_cross": None if len(tied) != 1 else tied[0],
        "earliest_time": earliest,
        "classification": classification,
        "loop_latency_ms": loop_latency,
    }


# ------------------------------------------------------------------------------- gates
def gate_drive_orders_winner(nest, h=0.01):
    """GATE 1: under uniform arrival, the stronger total supported drive fires first."""
    pairs = [(8000.0, 6000.0), (8000.0, 7000.0), (12000.0, 4000.0), (20000.0, 19000.0)]
    rows = []
    for strong, weak in pairs:
        times = crossing_times(nest, strong, weak, h=h)
        ok = (times["strong"] is not None and times["weak"] is not None
              and times["strong"] < times["weak"])
        rows.append({"w_strong": strong, "w_weak": weak, **times, "ordered": ok})
    return {"passed": all(r["ordered"] for r in rows), "measurements": rows}


def gate_multi_afferent_decomposition(nest, h=0.01, unit=1000.0):
    """GATE 6: the same TOTAL drive decides identically however it is decomposed.

    The real mechanism sums many afferents; a single aggregate event would never exercise
    that path. 8000 vs 6000 pA is rebuilt as eight versus six synchronous 1000 pA
    afferents, then rebuilt again with the competitors created in the opposite order.
    Crossing times must be unchanged in every case -- otherwise the decision depends on
    event insertion or node creation order, which is exactly what this profile must not do.
    """
    aggregate = crossing_times(nest, 8000.0, 6000.0, h=h)
    decomposed = crossing_times(nest, 8000.0, 6000.0, h=h, unit_weight=unit)
    permuted = crossing_times(nest, 8000.0, 6000.0, h=h, unit_weight=unit,
                              reverse_creation_order=True)

    def same(a, b):
        return (a["strong"] == b["strong"]) and (a["weak"] == b["weak"])

    return {
        "passed": (same(aggregate, decomposed) and same(decomposed, permuted)
                   and decomposed["strong"] < decomposed["weak"]),
        "unit_weight_pa": unit,
        "afferent_counts": {"strong": int(8000.0 / unit), "weak": int(6000.0 / unit)},
        "aggregate": aggregate,
        "decomposed": decomposed,
        "decomposed_reverse_creation_order": permuted,
        "aggregate_equals_decomposed": same(aggregate, decomposed),
        "creation_order_invariant": same(decomposed, permuted),
        "comparison": "raw float equality, no rounding",
    }


def gate_delay_scaling_preserves_winner(nest, h=0.01):
    """GATE 2: scaling ALL equivalent conduction delays together must not reverse it.

    A winner that survives 1 ms afferents but flips at 4 ms was decided by delay, not by
    drive -- which is precisely the failure mode the impulse prototype has.
    """
    rows = []
    for delay in (0.5, 1.0, 2.0, 4.0, 8.0):
        times = crossing_times(nest, 8000.0, 6000.0, h=h, source_delay=delay)
        # Raw latencies decide; rounded copies are for the table only.
        latency_strong = (None if times["strong"] is None
                          else times["strong"] - times["arrival"])
        latency_weak = (None if times["weak"] is None
                        else times["weak"] - times["arrival"])
        rows.append({"source_delay_ms": delay, **times,
                     "latency_strong": latency_strong, "latency_weak": latency_weak,
                     "latency_strong_display": None if latency_strong is None
                                               else round(latency_strong, 6),
                     "latency_weak_display": None if latency_weak is None
                                             else round(latency_weak, 6),
                     "strong_first": times["strong"] < times["weak"]})
    # How far the raw latency actually varies across a 16x range of conduction delay.
    raw = [r["latency_strong"] for r in rows]
    return {
        "passed": all(r["strong_first"] for r in rows),
        "latency_spread_ms": max(raw) - min(raw),
        "latency_spread_ms_display": f"{max(raw) - min(raw):.3e}",
        "measurements": rows,
    }


def gate_resolution_convergence(nest):
    """GATE 3: reducing `h` must converge both spike time and winner identity.

    Convergence, not bit-identity. The crossing is found by regula-falsi root finding over
    exactly integrated subthreshold dynamics, so the finest resolutions agree to the last
    float ULPs (~3e-15 ms measured) rather than to every bit. Raw unrounded times are kept
    in the report so that distinction stays visible.
    """
    rows = []
    for h in (0.1, 0.05, 0.01, 0.005, 0.001):
        times = crossing_times(nest, 8000.0, 6000.0, h=h)
        rows.append({"h": h, **times, "strong_first": times["strong"] < times["weak"]})
    strong = [r["strong"] for r in rows]
    # RAW arithmetic. Rounding before the comparison reported a 3.55e-15 ms spread as
    # exactly 0.0, which overstated convergence as bit-identity. Display fields are
    # rounded separately, and never fed back into a decision.
    spread = max(strong) - min(strong)
    finest_gap = abs(rows[-1]["strong"] - rows[-2]["strong"])
    return {
        "passed": all(r["strong_first"] for r in rows) and finest_gap < 1e-9,
        "spike_time_spread_ms": spread,
        "spike_time_spread_ms_display": f"{spread:.3e}",
        "finest_two_gap_ms": finest_gap,
        "finest_two_gap_ms_display": f"{finest_gap:.3e}",
        "comparison": "raw floats; tolerance 1e-9 ms",
        "identity_stable": len({r["strong_first"] for r in rows}) == 1,
        "measurements": rows,
    }


def gate_single_winner_within_margin(nest, h=0.01):
    """GATE 4: when the counterfactual gap exceeds the loop latency, exactly one commits.

    Reported as an ENVELOPE, not as an absolute claim. NEST cannot retract an emitted
    spike, so a nonpositive margin must be surfaced as an unresolved race rather than
    quietly counted as a win.
    """
    rows = []
    for strong, weak in [(8000.0, 6000.0), (8000.0, 7000.0), (8000.0, 7900.0),
                         (12000.0, 4000.0), (20000.0, 19000.0)]:
        counterfactual = crossing_times(nest, strong, weak, h=h)
        outcome = wta_outcome(nest, strong, weak, h=h)
        # RAW gap and margin decide the case; rounded copies exist only for the table.
        gap = (None if None in (counterfactual["strong"], counterfactual["weak"])
               else counterfactual["weak"] - counterfactual["strong"])
        latency = outcome["loop_latency_ms"]
        margin = None if None in (gap, latency) else gap - latency
        resolved = margin is not None and margin > 0
        rows.append({
            "w_strong": strong, "w_weak": weak,
            "counterfactual_strong": counterfactual["strong"],
            "counterfactual_weak": counterfactual["weak"],
            "runner_up_gap_ms": gap, "loop_latency_ms": latency, "margin_ms": margin,
            "runner_up_gap_ms_display": None if gap is None else round(gap, 6),
            "margin_ms_display": None if margin is None else round(margin, 6),
            "n_committed": outcome["n_committed"], "winner": outcome["winner"],
            "classification": outcome["classification"],
            "positive_margin": resolved,
            "contract_met": (outcome["winner"] == "strong") if resolved else None,
        })
    positive = [r for r in rows if r["positive_margin"]]
    return {
        "passed": bool(positive) and all(r["contract_met"] for r in positive),
        "positive_margin_cases": len(positive),
        "unresolved_cases": [r for r in rows if not r["positive_margin"]],
        "measurements": rows,
    }


def gate_exact_tie_is_reported(nest, h=0.01):
    """GATE 5: exactly equal drive must be REPORTED as a tie, never silently broken.

    There is no local arbitration mechanism in this spike, so the honest outcome is a
    declared unresolved race. Inventing a winner here is what delay jitter did.
    """
    counterfactual = crossing_times(nest, 8000.0, 8000.0, h=h)
    outcome = wta_outcome(nest, 8000.0, 8000.0, h=h)
    identical = (counterfactual["strong"] is not None
                 and counterfactual["strong"] == counterfactual["weak"])
    return {
        # A tie must be DETECTED and must not name a winner. Reporting competitor 0 here
        # would be indistinguishable from a real result.
        "passed": identical and outcome["winner"] is None,
        "tie_detected": identical,
        "winner": outcome["winner"],
        "counterfactual": counterfactual,
        "n_committed": outcome["n_committed"],
        "classification": outcome["classification"],
        "note": ("equal drive crosses at an identical time; with no declared local "
                 "arbitration this is reported as an unresolved tie with winner=None, "
                 "never resolved by node index"),
    }


def measure_operating_envelope(nest, h=0.01):
    """Where the resolvable/unresolvable boundary sits, as an INTERVAL, per loop latency.

    Not a pass/fail gate -- an input to Phase 2. Loop latency is
    `d_ei + relay crossing + d_ie`, and it sets how similar two drives may be and still
    commit to one winner.

    Naming matters here: a LOWER weak/strong ratio is an EASIER case (the drives are
    further apart), so the interesting quantity is the LARGEST sampled ratio that still
    resolves, together with the adjacent sampled ratio that does not. Reporting a single
    number as a "minimum" would inverte the ordering and read as the opposite claim.
    """
    rows = []
    for d in (0.1, 0.05, 0.02, 0.01):
        outcome = wta_outcome(nest, 8000.0, 6000.0, h=h, d_ei=d, d_ie=d)
        loop = outcome["loop_latency_ms"]
        largest_resolvable = None
        smallest_unresolved = None
        # Walk from similar drives towards separated ones; the first ratio whose
        # counterfactual gap clears the loop latency is the boundary, and the sample just
        # before it is the unresolved side of that boundary.
        previous = None
        for weak in range(7950, 5000, -50):
            counterfactual = crossing_times(nest, 8000.0, float(weak), h=h)
            gap = counterfactual["weak"] - counterfactual["strong"]
            if gap > loop:
                largest_resolvable = weak / 8000.0
                smallest_unresolved = previous
                break
            previous = weak / 8000.0
        rows.append({
            "d_ei_ms": d, "d_ie_ms": d, "loop_latency_ms": loop,
            "largest_sampled_resolvable_weak_over_strong": largest_resolvable,
            "smallest_sampled_unresolved_weak_over_strong": smallest_unresolved,
            "boundary_interval": [largest_resolvable, smallest_unresolved],
            "sample_step_pa": 50.0,
        })
    return {
        "measurements": rows,
        "reading": ("a LOWER weak/strong ratio means drives further apart and is easier to "
                    "resolve; the boundary is bracketed between the largest resolvable and "
                    "the smallest unresolved sample, at a 50 pA sampling step"),
        "note": ("loop latency = d_ei + relay crossing + d_ie; the relay's own crossing "
                 "time is the floor and cannot be removed"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import nest
    nest.set_verbosity("M_ERROR")

    out_dir = args.out or (OUT_ROOT / time.strftime("%Y%m%dT%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CIPP Phase 1 feasibility spike -- candidate 1: {CANDIDATE}")
    print(f"uniform conduction, no geometry, no jitter, nonzero E->I->E loop, h={args.h}\n")

    started = time.perf_counter()
    gates = {
        "1_drive_orders_winner": gate_drive_orders_winner(nest, args.h),
        "2_delay_scaling_preserves_winner": gate_delay_scaling_preserves_winner(nest, args.h),
        "3_resolution_convergence": gate_resolution_convergence(nest),
        "4_single_winner_within_margin": gate_single_winner_within_margin(nest, args.h),
        "5_exact_tie_reported": gate_exact_tie_is_reported(nest, args.h),
        "6_multi_afferent_decomposition": gate_multi_afferent_decomposition(nest, args.h),
    }
    for name, result in gates.items():
        print(f"  {'PASS' if result['passed'] else 'FAIL'}  {name}")

    envelope = measure_operating_envelope(nest, args.h)
    print("\n  operating envelope (input to Phase 2):")
    for row in envelope["measurements"]:
        lo = row["largest_sampled_resolvable_weak_over_strong"]
        hi = row["smallest_sampled_unresolved_weak_over_strong"]
        print(f"    loop {row['loop_latency_ms']:.4f} ms -> boundary between "
              f"{lo} (resolves) and {hi} (does not)")

    report = {
        "candidate": CANDIDATE,
        "operating_envelope": envelope,
        "candidate_rank": 1,
        "phase": "1_feasibility_spike",
        "circuit": CIRCUIT,
        "h": args.h,
        "geometry_enabled": False,
        "jitter_ms": 0.0,
        "nest_version": nest.__version__,
        "wall_clock_s": round(time.perf_counter() - started, 2),
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates.values()),
    }
    (out_dir / "feasibility.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nall gates passed: {report['all_passed']}")
    print(f"report -> {out_dir / 'feasibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
