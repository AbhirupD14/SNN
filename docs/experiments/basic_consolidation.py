"""Basic L1 consolidation headless experiment (tiled_cc and tiled_cc_l1_4).

Asks whether each of the nine 3x3 receptive fields can consolidate the four canonical
local patterns onto a stable, unique one-to-one mapping between patterns and ordinary L1 E
competitors — under two matched, *separately reported* conditions:

  * ``intact``            -- the complete preset (parent apical, C firing, C->I, I reset);
  * ``feedback_disabled`` -- the predictive ``column_c_to_i`` consequence disabled by
                             explicit edge metadata, local E->I->E WTA preserved.

Runs without FastAPI/WebSockets/browser. Emits, per (topology, condition, seed), a shared
replay-recorder artifact directory (manifest / replay.snn.jsonl / metrics.csv / summary),
plus batch-level aggregate summary.json + metrics.csv. See docs/BASIC_CONSOLIDATION.md.

This module records evidence; it does not modify any network dynamic, learning equation,
threshold, delay, scheduler rule, or topology preset. A timeout is never reinterpreted as
consolidation, and intact vs feedback-disabled results are never pooled.

CLI: ``PYTHONPATH=. .venv/bin/python experiments/basic_consolidation.py --help``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation import SimulationEngine, PATTERNS                 # noqa: E402
from experiments.replay_recorder import (                                # noqa: E402
    ReplayRecorder, STATUS_COMPLETED, STATUS_FAILED,
    FEEDBACK_INTACT, FEEDBACK_DISABLED,
)
from experiments.consolidation_analysis import (                         # noqa: E402
    CANONICAL_PATTERN_ORDER, DEFAULT_WINDOW, DEFAULT_DOMINANCE, DEFAULT_STABLE_WINDOWS,
    all_nine_patch_input, verify_all_nine_input, ordinary_e_ids_by_column,
    assess_ownership, column_mapping_checks,
)

TOPOLOGIES = ("tiled_cc", "tiled_cc_l1_4")
CONDITIONS = ("intact", "feedback_disabled")
_CONDITION_FEEDBACK = {"intact": FEEDBACK_INTACT, "feedback_disabled": FEEDBACK_DISABLED}

# CLI-friendly aliases for the backslash/space pattern names.
PATTERN_ALIASES = {
    "row1": "row 1", "row 1": "row 1",
    "col1": "col 1", "col 1": "col 1",
    "diag_back": "diag \\", "diag\\": "diag \\", "diag \\": "diag \\",
    "diag_fwd": "diag /", "diag/": "diag /", "diag /": "diag /",
}


# ============================================================ engine construction
def disable_column_c_to_i(engine: SimulationEngine) -> list:
    """Experiment-local feedback ablation: remove ONLY the ``column_c_to_i`` relay drive so
    a C spike no longer recruits its local I / hard reset. Selected purely from edge
    ``projection`` metadata (never id parsing). Preserves ``column_e_to_i`` (E->I) and
    ``column_i_to_e`` (I->E) — the local WTA path. Returns the exact disabled edge ids.
    """
    disabled = []
    targets = [(s["source"], s["target"], s["id"]) for s in engine.synapses
               if s.get("projection") == "column_c_to_i"]
    for src, _tgt, eid in targets:
        lst = engine._relayexc_out.get(src, [])
        engine._relayexc_out[src] = [(t, e) for (t, e) in lst if e != eid]
        disabled.append(eid)
    return sorted(disabled)


def make_engine(topology: str, seed: int, condition: str, *,
                cc_e_count: int = 8, leak_rate: float = 0.0):
    """Fresh deterministic engine for one run. Applies the feedback ablation only for the
    ``feedback_disabled`` condition; ``intact`` is byte-identical to an unmodified engine.
    """
    engine = SimulationEngine(seed=seed, topology=topology,
                              cc_e_count=cc_e_count, leak_rate=leak_rate)
    disabled_edges: list = []
    if condition == "feedback_disabled":
        disabled_edges = disable_column_c_to_i(engine)
    return engine, disabled_edges


# ============================================================ freeze / transfer
def plastic_edge_weights(engine: SimulationEngine) -> dict:
    """All live plastic weights keyed by stable edge id (ordinary/Eor/L2 E feedforward and
    C basal). This is the complete transferable/freezable plastic state for these presets.
    """
    out: dict = {}
    for eid, (cell, widx) in engine._ff_weight_ref.items():
        out[eid] = float(cell.acc_weights[widx])
    for eid, (cell, _widx) in engine._basal_weight_ref.items():
        out[eid] = float(cell.basal.weights[0])
    return out


def freeze_learning(engine: SimulationEngine) -> dict:
    """Freeze EVERY plastic object in the preset and return a byte-snapshot for invariance
    assertions. Covers ordinary L1 E, Eor, L2 E (all latency competitors) and every C basal
    learner; also defensively neutralizes optional plastic paths if a preset ever grows one.
    """
    for cell in engine.plastic:
        if hasattr(cell, "learn"):
            cell.learn = False
    for cell in getattr(engine, "coincidence", []):
        cell.learn = False
    # Defensive: any predictor/switch/encoder plasticity flags (absent in these presets).
    engine.params["pi_plasticity_enabled"] = False
    engine.params["enc_plasticity_enabled"] = False
    for sw in getattr(engine, "switches", []):
        if hasattr(sw, "learn"):
            sw.learn = False
    return plastic_edge_weights(engine)


def assert_weights_unchanged(engine: SimulationEngine, snapshot: dict) -> None:
    current = plastic_edge_weights(engine)
    if set(current) != set(snapshot):
        raise AssertionError("plastic edge id set changed during a frozen probe")
    for eid, w in snapshot.items():
        if current[eid] != w:
            raise AssertionError(f"plastic weight {eid} changed under freeze: "
                                 f"{w} -> {current[eid]}")


def transfer_plastic_weights(src: SimulationEngine, dst: SimulationEngine) -> dict:
    """Copy every plastic weight from ``src`` to ``dst`` by stable edge id. Rejects any
    missing/misaligned id. Membrane/refractory/queue/conductance state is NOT transferred.
    """
    src_w = plastic_edge_weights(src)
    dst_w = plastic_edge_weights(dst)
    if set(src_w) != set(dst_w):
        missing = sorted(set(src_w) ^ set(dst_w))
        raise ValueError(f"plastic edge id mismatch between engines: {missing[:8]}...")
    for eid, w in src_w.items():
        dst.set_synapse_weight(eid, w)
    # verify exact transfer
    after = plastic_edge_weights(dst)
    for eid, w in src_w.items():
        if after[eid] != w:
            raise AssertionError(f"weight transfer inexact for {eid}: {w} -> {after[eid]}")
    return after


# ============================================================ per-run context
class RunContext:
    """Static per-run structure derived once from the engine (columns, candidate ids)."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.tiled_meta = engine.tiled_meta
        layer_of = {nid: engine.meta[nid].get("layer") for nid in engine.order}
        self.l1_owners = ordinary_e_ids_by_column(engine._role_of, engine._column_of,
                                                  layer_of, layer="L1")
        self.l1_columns = sorted(self.l1_owners)
        self.n_competitors = {c: len(v) for c, v in self.l1_owners.items()}
        self.rgc_nodes = [dict(id=nid, patch_id=engine.meta[nid].get("patch_id"),
                               pixel=engine.meta[nid].get("pixel"))
                          for nid in engine.order
                          if engine._role_of.get(nid) is None
                          and engine.meta[nid].get("archetype") == "rg_source"]

    def input_for(self, pattern: str) -> list:
        return all_nine_patch_input(self.tiled_meta, pattern, PATTERNS)

    def verify_input(self, vec, pattern: str) -> dict:
        return verify_all_nine_input(self.tiled_meta, vec, pattern, PATTERNS, self.rgc_nodes)


def _column_winner_events(dyn: dict, l1_columns) -> dict:
    """Extract this boundary's ordinary-E winner event (neuron id) for each L1 column that
    produced one. Uses the engine's ``column_winners`` (role 'E' only, already filtered)."""
    cw = dyn.get("column_winners", {})
    return {c: cw[c]["id"] for c in l1_columns if c in cw}


# ============================================================ training
def train_phase(ctx: RunContext, rec: ReplayRecorder, pattern: str, *,
                window: int, dominance: float, stable_windows: int, timeout: int,
                progress) -> dict:
    """Present ``pattern`` in all nine patches until all nine L1 columns are simultaneously
    consolidated, or ``timeout`` boundaries elapse. Returns the phase result."""
    engine = ctx.engine
    vec = ctx.input_for(pattern)
    ctx.verify_input(vec, pattern)
    engine.set_input(vec)

    start_ts = int(engine.timestep) + 1
    rec.set_annotation(phase="train", pattern=pattern)
    rec.marker("train_start", data={"pattern": pattern, "start_timestep": start_ts})

    events: dict = {c: [] for c in ctx.l1_columns}
    acquisition: dict = {c: None for c in ctx.l1_columns}
    consolidated_now: dict = {c: False for c in ctx.l1_columns}
    boundaries = 0
    all_stable_at = None

    for b in range(1, timeout + 1):
        dyn = engine.step()
        boundaries = b
        rec.record_frame(engine)      # recorder honors its own record_every stride
        for c, nid in _column_winner_events(dyn, ctx.l1_columns).items():
            events[c].append(nid)

        # reassess every column on trailing events (no permanent latch)
        all_stable = True
        for c in ctx.l1_columns:
            v = assess_ownership(events[c], window=window, dominance=dominance,
                                 stable_windows=stable_windows)
            consolidated_now[c] = v.consolidated
            if v.consolidated and acquisition[c] is None:
                acquisition[c] = int(engine.timestep)
                rec.marker("column_consolidated",
                           data={"column": c, "pattern": pattern, "owner": v.owner,
                                 "timestep": int(engine.timestep)})
            if not v.consolidated:
                all_stable = False

        if b % 200 == 0:
            n_stable = sum(consolidated_now.values())
            progress(f"train {pattern:>8s} t={engine.timestep} stable={n_stable}/9 "
                     f"events~{min(len(e) for e in events.values())}")
        if all_stable:
            all_stable_at = int(engine.timestep)
            break

    timed_out = all_stable_at is None
    verdicts = {c: assess_ownership(events[c], window=window, dominance=dominance,
                                    stable_windows=stable_windows).as_dict()
                for c in ctx.l1_columns}
    owners = {c: verdicts[c]["owner"] for c in ctx.l1_columns}

    if timed_out:
        rec.marker("train_timeout", data={"pattern": pattern, "boundaries": boundaries,
                                          "stable_columns": sum(consolidated_now.values())})
    else:
        rec.marker("train_consolidated", data={"pattern": pattern, "timestep": all_stable_at})

    return {
        "pattern": pattern, "start_timestep": start_ts, "end_timestep": int(engine.timestep),
        "boundaries": boundaries, "timed_out": timed_out, "all_stable_at": all_stable_at,
        "owners": owners, "acquisition": acquisition, "verdicts": verdicts,
        "event_counts": {c: len(events[c]) for c in ctx.l1_columns},
    }


def _owner_weight_report(engine: SimulationEngine, owner_id: Optional[str]) -> dict:
    """Final incoming feedforward weight total and per-afferent weights for a pattern owner
    (distinguishes mature weights from stable firing)."""
    if owner_id is None:
        return {"total": None, "per_afferent": None}
    total = 0.0
    per = {}
    for eid, (cell, widx) in engine._ff_weight_ref.items():
        if cell.id == owner_id:
            w = float(cell.acc_weights[widx])
            per[eid] = round(w, 6)
            total += w
    return {"total": round(total, 6), "per_afferent": per}


# ============================================================ recall
def cold_recall(topology: str, seed: int, condition: str, ctx_trained: RunContext,
                trained_weights: dict, pattern: str, *, window: int, dominance: float,
                stable_windows: int, timeout: int, cc_e_count: int, leak_rate: float) -> dict:
    """Independent cold-state recall for one pattern: fresh engine, transferred trained
    plastic state by edge id, learning frozen, only that pattern presented from a fresh
    dynamic state. Returns owner + verdict for every column."""
    engine, _ = make_engine(topology, seed, condition, cc_e_count=cc_e_count,
                            leak_rate=leak_rate)
    transfer_plastic_weights(ctx_trained.engine, engine)
    snap = freeze_learning(engine)
    ctx = RunContext(engine)
    engine.set_input(ctx.input_for(pattern))

    events = {c: [] for c in ctx.l1_columns}
    need = window * stable_windows
    for _b in range(1, timeout + 1):
        dyn = engine.step()
        for c, nid in _column_winner_events(dyn, ctx.l1_columns).items():
            events[c].append(nid)
        if all(len(events[c]) >= need for c in ctx.l1_columns):
            break
    assert_weights_unchanged(engine, snap)      # recall changed no weight

    verdicts = {c: assess_ownership(events[c], window=window, dominance=dominance,
                                    stable_windows=stable_windows) for c in ctx.l1_columns}
    return {c: {"owner": verdicts[c].owner, "consolidated": verdicts[c].consolidated,
                "n_events": verdicts[c].n_events,
                "dominance": verdicts[c].final_window_dominance}
            for c in ctx.l1_columns}


def sequential_recall(engine: SimulationEngine, ctx: RunContext, pattern_order, *,
                      window: int, dominance: float, stable_windows: int,
                      recall_block: int) -> dict:
    """Sequential frozen recall on one already-frozen engine: present all patterns in order
    without resetting dynamic state between them. Reported separately (carryover-prone)."""
    result = {}
    for pattern in pattern_order:
        engine.set_input(ctx.input_for(pattern))
        events = {c: [] for c in ctx.l1_columns}
        for _b in range(recall_block):
            dyn = engine.step()
            for c, nid in _column_winner_events(dyn, ctx.l1_columns).items():
                events[c].append(nid)
        verdicts = {c: assess_ownership(events[c], window=window, dominance=dominance,
                                        stable_windows=stable_windows) for c in ctx.l1_columns}
        result[pattern] = {c: {"owner": verdicts[c].owner,
                               "consolidated": verdicts[c].consolidated,
                               "n_events": verdicts[c].n_events} for c in ctx.l1_columns}
    return result


# ============================================================ one full run
def run_one(topology: str, seed: int, condition: str, *, pattern_order,
            output_root: str, window: int, dominance: float, stable_windows: int,
            train_timeout: int, recall_timeout: int, recall_block: int,
            record_stride: int, checkpoint_every: int, cc_e_count: int, leak_rate: float,
            quiet: bool = False) -> dict:
    """Execute the complete Basic protocol for one (topology, condition, seed)."""
    t0 = time.perf_counter()

    def progress(msg):
        if not quiet:
            print(f"[{topology}/{condition}/seed{seed}] {msg}", flush=True)

    engine, disabled_edges = make_engine(topology, seed, condition,
                                         cc_e_count=cc_e_count, leak_rate=leak_rate)
    ctx = RunContext(engine)

    conditions_meta = {
        "feedback_condition": condition,
        "disabled_c_to_i_edges": disabled_edges,
        "pattern_order": list(pattern_order),
        "ownership": {"window": window, "dominance": dominance,
                      "stable_windows": stable_windows,
                      "train_timeout": train_timeout, "recall_timeout": recall_timeout},
        "n_competitors_per_column": ctx.n_competitors,
    }
    metrics_columns = [
        "run_id", "seed", "topology", "condition", "pattern_order", "phase", "pattern",
        "timestep", "column_id", "neuron_id", "winner_events", "dominance", "is_owner",
        "is_runner_up", "stable_window_count", "consolidated", "acquisition_timestep",
        "timeout", "collision", "assigned_competitors", "unassigned_competitors",
        "recall_owner_match", "incoming_weight_total", "per_afferent_weights",
    ]

    rec = ReplayRecorder(
        engine, experiment=f"basic_consolidation.{topology}.{condition}",
        output_root=output_root, seed=seed, record_every=record_stride,
        checkpoint_every=checkpoint_every,
        hierarchical_feedback=_CONDITION_FEEDBACK[condition],
        conditions=conditions_meta,
        schedule={"pattern_order": list(pattern_order), "train_timeout": train_timeout},
        metrics_columns=metrics_columns,
        optional_columns=["recall_owner_match", "acquisition_timestep",
                          "incoming_weight_total", "per_afferent_weights", "dominance"],
    )
    run_id = rec.run_id

    def base_row(**kw):
        row = {"run_id": run_id, "seed": seed, "topology": topology, "condition": condition,
               "pattern_order": "|".join(pattern_order)}
        row.update(kw)
        return row

    try:
        # ---------------- training ----------------
        phase_results = {}
        for pattern in pattern_order:
            pr = train_phase(ctx, rec, pattern, window=window, dominance=dominance,
                             stable_windows=stable_windows, timeout=train_timeout,
                             progress=progress)
            phase_results[pattern] = pr
            # emit per-column ownership rows
            for c in ctx.l1_columns:
                v = pr["verdicts"][c]
                owner = pr["owners"][c]
                wr = _owner_weight_report(engine, owner)
                rec.metrics.append_row(base_row(
                    phase="train", pattern=pattern, timestep=pr["end_timestep"],
                    column_id=c, neuron_id=owner or "",
                    winner_events=pr["event_counts"][c],
                    dominance=("" if v["final_window_dominance"] is None
                               else v["final_window_dominance"]),
                    is_owner=bool(owner), is_runner_up=False,
                    stable_window_count=len(v["windows"]),
                    consolidated=v["consolidated"],
                    acquisition_timestep=("" if pr["acquisition"][c] is None
                                          else pr["acquisition"][c]),
                    timeout=pr["timed_out"], collision="",
                    assigned_competitors="", unassigned_competitors="",
                    incoming_weight_total=("" if wr["total"] is None else wr["total"]),
                    per_afferent_weights=json.dumps(wr["per_afferent"]) if wr["per_afferent"] else "",
                ))

        # ---------------- mapping / capacity ----------------
        column_results = {}
        for c in ctx.l1_columns:
            owners_by_pattern = {p: phase_results[p]["owners"][c] for p in pattern_order}
            mc = column_mapping_checks(owners_by_pattern, ctx.n_competitors[c])
            column_results[c] = {"owners_by_pattern": owners_by_pattern, **mc}
            if mc["collisions"]:
                rec.marker("mapping_collision", data={"column": c,
                           "collisions": mc["collisions"]})
        basic_gate = all(column_results[c]["passed"] for c in ctx.l1_columns)
        any_timeout = any(phase_results[p]["timed_out"] for p in pattern_order)

        # ---------------- freeze + recall ----------------
        rec.marker("learning_freeze", data={})
        trained_weights = plastic_edge_weights(engine)
        freeze_snap = freeze_learning(engine)

        rec.marker("cold_recall_start", data={})
        cold = {}
        for pattern in pattern_order:
            cold[pattern] = cold_recall(topology, seed, condition, ctx, trained_weights,
                                        pattern, window=window, dominance=dominance,
                                        stable_windows=stable_windows, timeout=recall_timeout,
                                        cc_e_count=cc_e_count, leak_rate=leak_rate)
        # cold recall owner-match per column/pattern
        cold_match = {}
        for c in ctx.l1_columns:
            per_pat = {}
            for pattern in pattern_order:
                train_owner = phase_results[pattern]["owners"][c]
                recall_owner = cold[pattern][c]["owner"]
                per_pat[pattern] = (train_owner is not None and recall_owner == train_owner)
                rec.metrics.append_row(base_row(
                    phase="cold_recall", pattern=pattern, timestep=int(engine.timestep),
                    column_id=c, neuron_id=recall_owner or "",
                    winner_events=cold[pattern][c]["n_events"],
                    dominance=("" if cold[pattern][c]["dominance"] is None
                               else cold[pattern][c]["dominance"]),
                    is_owner=bool(recall_owner), is_runner_up=False,
                    stable_window_count="", consolidated=cold[pattern][c]["consolidated"],
                    acquisition_timestep="", timeout=False, collision="",
                    assigned_competitors="", unassigned_competitors="",
                    recall_owner_match=per_pat[pattern],
                ))
            # cold recall unique across patterns for this column
            recalled = [cold[p][c]["owner"] for p in pattern_order]
            unique = (all(o is not None for o in recalled)
                      and len(set(recalled)) == len(recalled))
            cold_match[c] = {"per_pattern": per_pat, "all_match": all(per_pat.values()),
                             "unique": unique}

        # sequential frozen recall on the same frozen engine
        rec.marker("sequential_recall_start", data={})
        seq = sequential_recall(engine, ctx, pattern_order, window=window,
                                dominance=dominance, stable_windows=stable_windows,
                                recall_block=recall_block)
        assert_weights_unchanged(engine, freeze_snap)   # no learned weight moved in recall
        for pattern in pattern_order:
            for c in ctx.l1_columns:
                rec.metrics.append_row(base_row(
                    phase="sequential_recall", pattern=pattern, timestep=int(engine.timestep),
                    column_id=c, neuron_id=seq[pattern][c]["owner"] or "",
                    winner_events=seq[pattern][c]["n_events"], dominance="",
                    is_owner=bool(seq[pattern][c]["owner"]), is_runner_up=False,
                    stable_window_count="", consolidated=seq[pattern][c]["consolidated"],
                    acquisition_timestep="", timeout=False, collision="",
                    assigned_competitors="", unassigned_competitors="",
                ))

        # ---------------- summary ----------------
        cold_retention = all(cold_match[c]["all_match"] and cold_match[c]["unique"]
                             for c in ctx.l1_columns)
        checks = {
            "basic_gate_all_columns": basic_gate,
            "no_training_timeout": not any_timeout,
            "cold_recall_retention": cold_retention,
        }
        status = STATUS_COMPLETED
        failure_reason = None
        if any_timeout:
            # a timed-out phase is a confounded/failed run, never silent consolidation
            failure_reason = "training timeout in at least one phase"

        owner_matrix = {c: column_results[c]["owners_by_pattern"] for c in ctx.l1_columns}
        result = {
            "topology": topology, "seed": seed, "condition": condition,
            "pattern_order": list(pattern_order),
            "basic_gate_passed": basic_gate,
            "any_training_timeout": any_timeout,
            "owner_matrix": owner_matrix,
            "per_column": {c: {
                "passed": column_results[c]["passed"],
                "checks": column_results[c]["checks"],
                "capacity": column_results[c]["capacity"],
                "collisions": column_results[c]["collisions"],
                "confusion": {p: phase_results[p]["verdicts"][c] for p in pattern_order},
            } for c in ctx.l1_columns},
            "cold_recall": {c: cold_match[c] for c in ctx.l1_columns},
            "cold_recall_retention": cold_retention,
            "sequential_recall": seq,
            "acquisition_durations": {
                p: {c: (None if phase_results[p]["acquisition"][c] is None
                        else phase_results[p]["acquisition"][c] - phase_results[p]["start_timestep"])
                    for c in ctx.l1_columns} for p in pattern_order},
            "disabled_c_to_i_edges": disabled_edges,
            "ownership_defaults": {"window": window, "dominance": dominance,
                                   "stable_windows": stable_windows},
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
        }
        rec.marker("final_result", data={"basic_gate_passed": basic_gate,
                                         "any_timeout": any_timeout})
        summary = rec.finish(status, checks=checks, result=result,
                             failure_reason=failure_reason)
        progress(f"done basic_gate={basic_gate} timeout={any_timeout} "
                 f"({result['elapsed_seconds']}s) -> {rec.run_dir}")
        return {"run_id": run_id, "run_dir": rec.run_dir, "status": status,
                "checks": checks, "result": result, "summary_path": rec.summary_path}
    except BaseException as exc:      # noqa: BLE001 -- recorder finalizes, then re-raise
        if not rec._finalized:
            rec.finish(STATUS_FAILED, checks={}, result={"error": repr(exc)},
                       failure_reason=repr(exc))
        raise


# ============================================================ batch orchestration
def _run_dir_completed(output_root: str, topology: str, condition: str, seed: int) -> Optional[str]:
    """Return an existing completed run dir matching this cell, else None (for --resume)."""
    if not os.path.isdir(output_root):
        return None
    for name in os.listdir(output_root):
        d = os.path.join(output_root, name)
        summ = os.path.join(d, "summary.json")
        if not os.path.isfile(summ):
            continue
        try:
            s = json.load(open(summ))
        except (OSError, json.JSONDecodeError):
            continue
        r = s.get("result", {})
        if (r.get("topology") == topology and r.get("condition") == condition
                and r.get("seed") == seed and s.get("status") == STATUS_COMPLETED):
            return d
    return None


def run_batch(*, topologies, conditions, seeds, pattern_order, output_root, window,
              dominance, stable_windows, train_timeout, recall_timeout, recall_block,
              record_stride, checkpoint_every, cc_e_count, leak_rate, resume: bool,
              quiet: bool = False) -> dict:
    os.makedirs(output_root, exist_ok=True)
    cells = [(t, c, s) for t in topologies for c in conditions for s in seeds]
    print(f"batch: {len(cells)} runs -> {output_root}", flush=True)
    runs = []
    for (topology, condition, seed) in cells:
        if resume:
            done = _run_dir_completed(output_root, topology, condition, seed)
            if done:
                print(f"skip (resume) {topology}/{condition}/seed{seed} -> {done}", flush=True)
                runs.append({"topology": topology, "condition": condition, "seed": seed,
                             "run_dir": done, "skipped": True})
                continue
        out = run_one(topology, seed, condition, pattern_order=pattern_order,
                      output_root=output_root, window=window, dominance=dominance,
                      stable_windows=stable_windows, train_timeout=train_timeout,
                      recall_timeout=recall_timeout, recall_block=recall_block,
                      record_stride=record_stride, checkpoint_every=checkpoint_every,
                      cc_e_count=cc_e_count, leak_rate=leak_rate, quiet=quiet)
        runs.append({"topology": topology, "condition": condition, "seed": seed,
                     "run_dir": out["run_dir"], "run_id": out["run_id"],
                     "checks": out["checks"], "basic_gate": out["result"]["basic_gate_passed"],
                     "any_timeout": out["result"]["any_training_timeout"]})
    agg = write_aggregate(output_root)
    return {"runs": runs, "aggregate": agg}


# ============================================================ aggregation / summarize
def write_aggregate(output_root: str) -> dict:
    """Scan all run dirs and write batch-level aggregate summary.json + metrics.csv.
    Intact and feedback_disabled are kept separate; incomplete runs are labeled, never
    scored as pass/fail. Safe to run while runs are still active."""
    rows = []
    per_condition = {c: {"completed": 0, "basic_pass": 0, "timeout": 0} for c in CONDITIONS}
    incomplete = []
    if os.path.isdir(output_root):
        for name in sorted(os.listdir(output_root)):
            d = os.path.join(output_root, name)
            summ = os.path.join(d, "summary.json")
            man = os.path.join(d, "manifest.json")
            if not os.path.isfile(man):
                continue
            try:
                m = json.load(open(man))
            except (OSError, json.JSONDecodeError):
                continue
            status = m.get("status")
            if status != STATUS_COMPLETED or not os.path.isfile(summ):
                incomplete.append({"run_dir": d, "status": status})
                continue
            s = json.load(open(summ))
            r = s.get("result", {})
            cond = r.get("condition")
            row = {"run_dir": name, "topology": r.get("topology"), "condition": cond,
                   "seed": r.get("seed"), "basic_gate_passed": r.get("basic_gate_passed"),
                   "any_training_timeout": r.get("any_training_timeout"),
                   "cold_recall_retention": r.get("cold_recall_retention")}
            rows.append(row)
            if cond in per_condition:
                per_condition[cond]["completed"] += 1
                per_condition[cond]["basic_pass"] += int(bool(r.get("basic_gate_passed")))
                per_condition[cond]["timeout"] += int(bool(r.get("any_training_timeout")))

    agg = {"output_root": os.path.abspath(output_root),
           "n_completed": len(rows), "n_incomplete": len(incomplete),
           "incomplete_runs": incomplete, "per_condition": per_condition, "runs": rows}
    with open(os.path.join(output_root, "aggregate_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    # aggregate long CSV
    import csv
    cols = ["run_dir", "topology", "condition", "seed", "basic_gate_passed",
            "any_training_timeout", "cold_recall_retention"]
    with open(os.path.join(output_root, "aggregate_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return agg


# ============================================================ CLI
def _parse_patterns(spec: str) -> list:
    out = []
    for tok in spec.split(","):
        key = tok.strip()
        if key in PATTERN_ALIASES:
            out.append(PATTERN_ALIASES[key])
        elif key in PATTERNS:
            out.append(key)
        else:
            raise argparse.ArgumentTypeError(f"unknown pattern {key!r}")
    if sorted(out) != sorted(CANONICAL_PATTERN_ORDER):
        raise argparse.ArgumentTypeError(
            f"pattern order must be a permutation of the four canonical patterns, got {out}")
    return out


def _parse_seeds(spec: str) -> list:
    seeds = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(tok))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", default="both",
                   help="tiled_cc | tiled_cc_l1_4 | both (default both)")
    p.add_argument("--condition", default="both",
                   help="intact | feedback_disabled | both (default both)")
    p.add_argument("--seeds", type=_parse_seeds, default=[1],
                   help="seed list/range, e.g. '1' or '1-8' or '1,3,5' (default 1)")
    p.add_argument("--pattern-order", type=_parse_patterns,
                   default=list(CANONICAL_PATTERN_ORDER),
                   help="comma list of patterns (aliases: row1,col1,diag_back,diag_fwd)")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--dominance", type=float, default=DEFAULT_DOMINANCE)
    p.add_argument("--stable-windows", type=int, default=DEFAULT_STABLE_WINDOWS)
    p.add_argument("--train-timeout", type=int, default=3000,
                   help="max boundaries per training pattern (default 3000)")
    p.add_argument("--recall-timeout", type=int, default=1500,
                   help="max boundaries per cold-recall pattern (default 1500)")
    p.add_argument("--recall-block", type=int, default=400,
                   help="boundaries per sequential-recall pattern block (default 400)")
    p.add_argument("--record-stride", type=int, default=20,
                   help="replay state-frame stride (default 20)")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="weight-checkpoint interval in recorded frames (default 50)")
    p.add_argument("--cc-e-count", type=int, default=8,
                   help="ordinary-E bank size for tiled_cc (ignored by tiled_cc_l1_4)")
    p.add_argument("--leak-rate", type=float, default=0.0)
    p.add_argument("--output-root", default="experiments/runs/basic_consolidation")
    p.add_argument("--resume", action="store_true",
                   help="skip cells with a completed run dir already present")
    p.add_argument("--workers", type=int, default=1,
                   help="worker count (currently sequential; reserved, default 1)")
    p.add_argument("--summarize-only", action="store_true",
                   help="only (re)write the aggregate over an existing output root")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize_only:
        agg = write_aggregate(args.output_root)
        print(json.dumps(agg["per_condition"], indent=2))
        print(f"completed={agg['n_completed']} incomplete={agg['n_incomplete']}")
        return 0

    topologies = TOPOLOGIES if args.topology == "both" else (args.topology,)
    conditions = CONDITIONS if args.condition == "both" else (args.condition,)
    for t in topologies:
        if t not in TOPOLOGIES:
            raise SystemExit(f"unknown topology {t!r}")
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}")

    out = run_batch(
        topologies=topologies, conditions=conditions, seeds=args.seeds,
        pattern_order=args.pattern_order, output_root=args.output_root,
        window=args.window, dominance=args.dominance, stable_windows=args.stable_windows,
        train_timeout=args.train_timeout, recall_timeout=args.recall_timeout,
        recall_block=args.recall_block, record_stride=args.record_stride,
        checkpoint_every=args.checkpoint_every, cc_e_count=args.cc_e_count,
        leak_rate=args.leak_rate, resume=args.resume, quiet=args.quiet)
    agg = out["aggregate"]
    print("=== per-condition (separate; never pooled) ===")
    print(json.dumps(agg["per_condition"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
