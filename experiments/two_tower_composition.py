"""Task 4 — two 9x9 towers feeding one L3 composition column (bounded V/A/7 probe).

Asks whether two independently learned halves of a ``9x18`` glyph can be composed by ONE
final classic L3 cortical column under the same dual-FE/FES rule and the same generic
child->parent connectivity used everywhere else in the fabric::

    Tower 0 (left):   9x9 RGC -> 9 L1 classic CC -> T0L2c00 -.
                                                              >-- L3c00
    Tower 1 (right):  9x9 RGC -> 9 L1 classic CC -> T1L2c00 -'

It introduces no feature gate, no direct-identity relay and no new neural mechanism. The
graph is built by :func:`backend.network_spec.two_tower_composition_spec`. It was first
run through the ordinary custom-topology path and, once the result was in, promoted to the
``two_tower_composition`` built-in preset at the user's request so the same graph can be
driven live from the dashboard. The two construction paths are bit-identical (same node
order, seeded layout and initial weights), so promoting it did not change any result.

Live engine contract (asserted at construction, recorded in every artifact):

    dual FE/FES on, e = wte = 0.001, B = 5 (reference)
    eta = 4.0, c_eta = 16.0
    leak_rate = 0.0, refractory_steps = 0
    input_period = 0            (AUTO: one volley per resolved causal chain)
    e_weight_cap_frac = 0.5     (theta/2 pattern-detector ceiling)
    relay_weight_cap_frac = 1.0 (theta one-afferent ceiling)
    eor_w_init_frac = 1.0 + eor_plasticity_enabled = False
                                (Eor is a FIXED non-plastic relay at theta)
    c_feedback_reset = True     (top-down delay-1 feedback reset enabled)

Phases:

``structure``   build + validate the graph and the glyphs; write the topology/glyph artifacts.
``preflight``   representation-separability probe: L3 *learning only* disabled, lower towers
                trained on V/A/7, then a frozen cold clone per glyph records exactly what
                L3 can observe (the two L2-Eor event streams).
``composition`` the required composition training conditions (L3 learning enabled), plus
                independent cold-state recall per glyph.

CLI: ``PYTHONPATH=. .venv/bin/python experiments/two_tower_composition.py --help``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from typing import Any, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.network_spec import two_tower_composition_spec              # noqa: E402
from backend.simulation import SimulationEngine                          # noqa: E402
from experiments.basic_consolidation import (                            # noqa: E402
    assert_weights_unchanged, freeze_learning, plastic_edge_weights,
)
from experiments.consolidation_analysis import (                         # noqa: E402
    DEFAULT_DOMINANCE, DEFAULT_STABLE_WINDOWS, DEFAULT_WINDOW, assess_ownership,
)
from experiments.replay_recorder import (                                # noqa: E402
    FEEDBACK_INTACT, STATUS_COMPLETED, STATUS_FAILED, ReplayRecorder,
)
from experiments.two_tower_analysis import (                             # noqa: E402
    CANONICAL_GLYPH_ORDER, GLYPH_COLS, L3Signature, SlugRegistry, ThroughputTracker,
    c_readiness, classify_signatures, condition_slug, failure_record, glyph_vector,
    one_event_maturity, verify_glyphs,
)

SCHEMA_VERSION = 1

# ---------------------------------------------------------------- engine contract
# The CURRENT live contract. Every value is asserted on the constructed engine before a
# single boundary is stepped; ``dual_fe_B`` is the one declared per-condition parameter.
ENGINE_CONTRACT: dict = dict(
    cc_e_count=8,
    dual_fe_fes=True,
    dual_fe_e=0.001,
    dual_fe_wte=0.001,
    eta=4.0,
    c_eta=16.0,
    leak_rate=0.0,
    refractory_steps=0,
    input_period=0,
    e_weight_cap_frac=0.5,
    relay_weight_cap_frac=1.0,
    eor_w_init_frac=1.0,
    eor_plasticity_enabled=False,
    c_feedback_reset=True,
)
REFERENCE_B = 5.0

# Ownership criterion (identical to the rest of the fabric's experiments).
OWNERSHIP = dict(window=DEFAULT_WINDOW, dominance=DEFAULT_DOMINANCE,
                 stable_windows=DEFAULT_STABLE_WINDOWS)
# Observed Eor throughput readiness.
EOR_MIN_DELIVERIES = 50
EOR_MIN_RELIABILITY = 0.95
# Full-gate confirmation: the Boolean conjunction must hold on this many consecutive
# boundaries, so a one-boundary coincidence cannot stop a phase.
GATE_CONFIRM = 3
# The complete composition gate, and the LOWER-TOWER subset the separability preflight
# stops on (its L3 plasticity is frozen, so ``G`` is unreachable there BY CONSTRUCTION and
# must not be allowed to hold the preflight hostage). Every milestone is always evaluated.
GATE_KEYS = ("A", "B", "C", "D", "E", "F", "G")
PREFLIGHT_GATE_KEYS = ("A", "B", "C", "D", "E", "F")
# Replay budget: at most this many written frames per condition.
REPLAY_FRAME_BUDGET = 1500
CHECKPOINT_EVERY_FRAMES = 50


# ============================================================ engine construction
TOPOLOGY = "two_tower_composition"


def make_two_tower_engine(*, seed: int, B: float = REFERENCE_B,
                          cc_e_count: int = 8) -> SimulationEngine:
    """One fresh two-tower engine under the live contract.

    Built through the ``two_tower_composition`` built-in preset, which is bit-identical to
    the earlier ``apply_topology(two_tower_composition_spec(...))`` path (same node order,
    same seeded layout, same initial weights) -- so the dashboard loads exactly the graph
    this experiment measures. Instrumentation is enabled by the caller afterwards.
    """
    contract = dict(ENGINE_CONTRACT)
    contract["cc_e_count"] = int(cc_e_count)
    engine = SimulationEngine(seed=int(seed), topology=TOPOLOGY,
                              dual_fe_B=float(B), **contract)
    assert engine.mode == TOPOLOGY, f"engine mode {engine.mode!r}, expected {TOPOLOGY!r}"
    assert_engine_contract(engine, B=B, cc_e_count=cc_e_count)
    return engine


def assert_engine_contract(engine: SimulationEngine, *, B: float,
                           cc_e_count: int = 8) -> dict:
    """Assert every declared parameter, learning mode and structural ceiling. Raises
    ``AssertionError`` on any drift; returns the audited report."""
    p = engine.params
    for key, want in ENGINE_CONTRACT.items():
        if key == "cc_e_count":
            want = int(cc_e_count)
        got = p[key]
        assert got == want, f"engine param {key}: expected {want!r}, got {got!r}"
    assert float(p["dual_fe_B"]) == float(B), "dual_fe_B drifted from the condition value"

    meta = engine.tiled_meta
    assert meta is not None, "two-tower graph must carry tiled topology metadata"
    assert (meta["input_shape"]["rows"], meta["input_shape"]["cols"]) == (9, 18)
    assert engine.n_pix == 9 * 18, f"engine input surface is {engine.n_pix}, expected 162"
    layers = [L["layer"] for L in meta["column_layers"]]
    assert layers == ["L1", "L2", "L3"], f"unexpected column layers {layers}"
    n_cols = {L: 0 for L in layers}
    for c in meta["columns"]:
        n_cols[c["layer"]] += 1
    assert n_cols == {"L1": 18, "L2": 2, "L3": 1}, f"unexpected column census {n_cols}"

    thr = float(p["e_threshold"])
    detector_cap = float(p["e_weight_cap_frac"]) * thr
    relay_cap = float(p["relay_weight_cap_frac"]) * thr
    audited = {"ordinary": 0, "eor": 0, "c": 0}
    for cell in engine.plastic:
        role = engine._role_of.get(cell.id)
        assert cell.update_mode == "dual_fe_fes", \
            f"{cell.id} update_mode {cell.update_mode!r}, expected 'dual_fe_fes'"
        if role == "Eor":
            assert cell.learn is False, f"Eor {cell.id} must be a FIXED non-plastic relay"
            assert cell.w_cap == relay_cap, f"Eor {cell.id} ceiling is not theta"
            assert all(abs(float(w) - thr) < 1e-9 for w in cell.acc_weights), \
                f"Eor {cell.id} bank is not fixed at theta"
            audited["eor"] += 1
        elif role == "E":
            assert cell.learn is True, f"ordinary E {cell.id} must be plastic"
            assert cell.w_cap == detector_cap, \
                f"ordinary E {cell.id} ceiling {cell.w_cap} is not theta/2"
            audited["ordinary"] += 1
    for cell in engine.coincidence:
        assert cell.update_mode == "c_dual_fe_fes", \
            f"{cell.id} update_mode {cell.update_mode!r}, expected 'c_dual_fe_fes'"
        assert cell.w_cap == relay_cap, f"C {cell.id} basal ceiling is not theta"
        audited["c"] += 1
    assert audited["eor"] == 21 and audited["c"] == 21, f"population audit {audited}"
    assert audited["ordinary"] == 21 * int(cc_e_count), f"population audit {audited}"

    return {"params_ok": True, "populations": audited,
            "detector_cap": detector_cap, "relay_cap": relay_cap, "theta": thr,
            "feedback_loop_latency": engine.feedback_loop_latency,
            "resolved_input_period": engine.resolved_input_period()}


# ============================================================ metadata index
class FabricIndex:
    """One metadata-derived index of the built graph. Every scientific selection below
    goes through this object; ids are parsed ONLY in the builder-naming assertion."""

    def __init__(self, engine: SimulationEngine):
        meta = engine.tiled_meta
        if meta is None:
            raise ValueError("FabricIndex requires a tiled graph")
        self.engine = engine
        self.columns = {c["id"]: dict(c) for c in meta["columns"]}
        self.layer_of = {nid: engine.meta[nid].get("layer") for nid in engine.order}

        self.ordinary: dict = {}
        self.eor: dict = {}
        self.c: dict = {}
        self.i: dict = {}
        for nid in engine.order:
            role = engine._role_of.get(nid)
            cid = engine._column_of.get(nid)
            if role is None or cid is None:
                continue
            if cid not in self.columns:
                raise ValueError(f"node {nid!r} names undeclared column {cid!r}")
            layer = self.columns[cid]["layer"]
            if role == "E":
                self.ordinary.setdefault(layer, {}).setdefault(cid, []).append(nid)
            elif role == "Eor":
                if cid in self.eor:
                    raise ValueError(f"column {cid!r} has more than one Eor")
                self.eor[cid] = nid
            elif role == "C":
                self.c[cid] = nid
            elif role == "I":
                self.i[cid] = nid
            else:
                raise ValueError(f"unexpected column_role {role!r} on {nid!r}")
        for layer, by_col in self.ordinary.items():
            for cid, ids in by_col.items():
                ids.sort()
                if len(ids) != self.columns[cid]["e_count"]:
                    raise ValueError(f"column {cid!r} ordinary-E census mismatch")
        for cid in self.columns:
            for slot, table in (("Eor", self.eor), ("C", self.c), ("I", self.i)):
                if cid not in table:
                    raise ValueError(f"column {cid!r} is missing its {slot}")

        self.parents = {cid: list(c["parent_ids"]) for cid, c in self.columns.items()}
        self.children: dict = {cid: [] for cid in self.columns}
        for cid, pids in self.parents.items():
            for pid in pids:
                self.children[pid].append(cid)
        for cid in self.children:
            self.children[cid].sort()

        # towers derived from the PARENT CHAIN (never the id prefix): one tower per L2
        # column, holding the L1 columns that declare it as parent.
        self.l2_columns = sorted(cid for cid, c in self.columns.items() if c["layer"] == "L2")
        self.l3_columns = sorted(cid for cid, c in self.columns.items() if c["layer"] == "L3")
        if len(self.l3_columns) != 1:
            raise ValueError(f"expected exactly one L3 column, got {self.l3_columns}")
        self.l3 = self.l3_columns[0]
        self.towers = {l2: sorted(self.children[l2]) for l2 in self.l2_columns}
        for l2 in self.l2_columns:
            if self.parents[l2] != [self.l3]:
                raise ValueError(f"L2 column {l2!r} does not declare the L3 column as parent")
        if self.parents[self.l3]:
            raise ValueError("the L3 column must have no parent")
        self.l1_columns = sorted(cid for cid, c in self.columns.items() if c["layer"] == "L1")

        # RGC surface, per global patch, ordered by patch-local position.
        self.rgc_by_patch: dict = {}
        self.patch_of_column: dict = {}
        for nid in engine.order:
            m = engine.meta[nid]
            if m.get("archetype") != "rg_source":
                continue
            key = (int(m["patch_row"]), int(m["patch_col"]))
            self.rgc_by_patch.setdefault(key, []).append(
                (int(m["patch_local_row"]), int(m["patch_local_col"]), nid, int(m["pixel"])))
        for key, entries in self.rgc_by_patch.items():
            entries.sort()
            self.rgc_by_patch[key] = [e[2] for e in entries]
        self.pixel_of_rgc = {engine.meta[nid]["id"]: int(engine.meta[nid]["pixel"])
                             for nid in engine.order
                             if engine.meta[nid].get("archetype") == "rg_source"}
        for cid in self.l1_columns:
            c = self.columns[cid]
            key = (int(c["row"]), int(c["col"]))
            if key not in self.rgc_by_patch:
                raise ValueError(f"L1 column {cid!r} has no RGC patch at {key}")
            self.patch_of_column[cid] = key
        self.column_of_patch = {v: k for k, v in self.patch_of_column.items()}

        self.ff_edge: dict = {}
        self.basal_edge: dict = {}
        for e in engine.synapses:
            if e["kind"] == "feedforward":
                self.ff_edge[(e["source"], e["target"])] = e["id"]
            elif e["kind"] == "basal_excitation":
                cid = engine._column_of.get(e["target"])
                self.basal_edge[cid] = e["id"]
        for cid in self.columns:
            if cid not in self.basal_edge:
                raise ValueError(f"column {cid!r} has no Eor->C basal edge")

        # C cells excluded from any readiness gate: declared by metadata, never by id.
        self.dormant_c = sorted(cid for cid in self.columns
                                if not bool(engine.meta[self.c[cid]].get("has_parent")))

    # -------------------------------------------------------------- helpers
    def tower_of(self, column_id: str) -> Optional[str]:
        """The L2 column that owns ``column_id`` (itself, for an L2 column)."""
        if column_id in self.towers:
            return column_id
        for l2, kids in self.towers.items():
            if column_id in kids:
                return l2
        return None

    def active_rgc_ids(self, column_id: str, vector: Sequence[int]) -> list:
        """The RGC ids of this column's own patch that the input vector actually drives."""
        return [nid for nid in self.rgc_by_patch[self.patch_of_column[column_id]]
                if vector[self.pixel_of_rgc[nid]]]

    def active_l1_columns(self, vector: Sequence[int]) -> list:
        return sorted(cid for cid in self.l1_columns if self.active_rgc_ids(cid, vector))

    def active_child_eors(self, l2_column: str, vector: Sequence[int]) -> list:
        """The child L1 Eor ids of one tower whose columns are driven by this glyph."""
        active = set(self.active_l1_columns(vector))
        return sorted(self.eor[cid] for cid in self.towers[l2_column] if cid in active)

    def cell(self, nid: str):
        return self.engine.exc[nid]

    def assert_builder_naming(self) -> None:
        """The ONE place id parsing is allowed: a cheap check that the deterministic
        naming contract in the builder still holds. Scientific selection never uses it."""
        assert self.l3 == "L3c00", f"L3 column id contract broken: {self.l3!r}"
        assert self.l2_columns == ["T0L2c00", "T1L2c00"], self.l2_columns
        for l2, kids in self.towers.items():
            prefix = l2[:2]
            assert all(k.startswith(prefix) for k in kids), (l2, kids)
            assert len(kids) == 9, (l2, kids)


# ============================================================ per-boundary state
class GlyphTrackers:
    """All bounded per-glyph trackers. Nothing unbounded is retained: ownership keeps only
    its trailing ``window*stable_windows`` winner events, and update logs are drained into
    per-cell aggregates every boundary."""

    def __init__(self, index: FabricIndex, vector: Sequence[int], *,
                 window: int, stable_windows: int):
        self.index = index
        self.vector = list(vector)
        need = int(window) * int(stable_windows)
        cols = list(index.columns)

        # Glyph-static selections, resolved ONCE from metadata (the input vector cannot
        # change inside a phase), so the per-boundary gate does no rescanning.
        self.active_l1 = index.active_l1_columns(self.vector)
        self.active_rgc = {cid: index.active_rgc_ids(cid, self.vector)
                           for cid in self.active_l1}
        self.active_child_eors = {l2: index.active_child_eors(l2, self.vector)
                                  for l2 in index.l2_columns}
        self.gated_c = [cid for cid in (self.active_l1 + index.l2_columns)
                        if cid not in index.dormant_c]
        self.participating_l2_eors = sorted(
            index.eor[l2] for l2 in index.l2_columns if self.active_child_eors[l2])
        self.events = {cid: deque(maxlen=need) for cid in cols}
        self.event_counts = {cid: 0 for cid in cols}
        self.verdict: dict = {cid: None for cid in cols}
        self.owner_first_at: dict = {cid: None for cid in cols}
        self.owner_lost_at: dict = {cid: [] for cid in cols}
        self.winner_counts: dict = {cid: {} for cid in cols}

        # observed owner -> Eor throughput, one tracker per column
        self.eor_paths = {cid: ThroughputTracker(window=EOR_MIN_DELIVERIES) for cid in cols}

        # C diagnostics
        self.c_stats = {cid: dict(apical_boundaries=0, basal_boundaries=0,
                                  gate_open=0, deposits=0, spikes=0, updates=0,
                                  opportunities=0, first_spike=None,
                                  first_one_shot_at=None) for cid in cols}
        # learning-event aggregates, per cell
        self.learning: dict = {}
        self.spike_counts: dict = {}
        self.hard_resets = 0
        self.feedback_resets = 0
        self.latency_ties = 0
        self.nonfinite = 0
        self.milestone_first: dict = {}

    # -------------------------------------------------------- ownership
    def note_winner(self, column_id: str, neuron_id: str, timestep: int, *,
                    window: int, dominance: float, stable_windows: int) -> None:
        self.events[column_id].append(neuron_id)
        self.event_counts[column_id] += 1
        self.winner_counts[column_id][neuron_id] = \
            self.winner_counts[column_id].get(neuron_id, 0) + 1
        v = assess_ownership(list(self.events[column_id]), window=window,
                             dominance=dominance, stable_windows=stable_windows)
        prev = self.verdict[column_id]
        self.verdict[column_id] = v
        if v.consolidated and self.owner_first_at[column_id] is None:
            self.owner_first_at[column_id] = int(timestep)
        if prev is not None and prev.consolidated and not v.consolidated:
            self.owner_lost_at[column_id].append(int(timestep))

    def owner(self, column_id: str) -> Optional[str]:
        v = self.verdict[column_id]
        return v.owner if (v is not None and v.consolidated) else None

    def consolidated(self, column_id: str) -> bool:
        v = self.verdict[column_id]
        return bool(v is not None and v.consolidated)

    # -------------------------------------------------------- learning events
    def drain_updates(self, engine: SimulationEngine, timestep: int) -> None:
        for cell in list(engine.plastic) + list(engine.coincidence):
            log = cell.update_log
            if not log:
                continue
            agg = self.learning.get(cell.id)
            if agg is None:
                agg = self.learning[cell.id] = dict(
                    cell_id=cell.id, update_events=0, positive_synapses=0,
                    negative_synapses=0, floor_hits=0, sum_abs_raw_dw=0.0,
                    sum_abs_applied_dw=0.0, fe_sum=0.0, fes_sum=0.0, fes_n=0,
                    nonfinite=0, first_update_at=int(timestep), last_update_at=int(timestep))
            for rec in log:
                agg["update_events"] += 1
                agg["last_update_at"] = int(timestep)
                raw = rec.get("raw_dw")
                applied = rec.get("applied_dw")
                if isinstance(raw, list):
                    agg["sum_abs_raw_dw"] += float(sum(abs(x) for x in raw))
                    agg["positive_synapses"] += sum(1 for x in applied if x > 0)
                    agg["negative_synapses"] += sum(1 for x in applied if x < 0)
                    agg["sum_abs_applied_dw"] += float(sum(abs(x) for x in applied))
                    agg["floor_hits"] += int(rec.get("n_floor", 0))
                    fes = rec.get("fes") or []
                    agg["fes_sum"] += float(sum(fes))
                    agg["fes_n"] += len(fes)
                else:
                    agg["sum_abs_raw_dw"] += abs(float(raw or 0.0))
                    agg["sum_abs_applied_dw"] += abs(float(applied or 0.0))
                    agg["positive_synapses"] += int(float(applied or 0.0) > 0)
                    agg["negative_synapses"] += int(float(applied or 0.0) < 0)
                    agg["floor_hits"] += int(bool(rec.get("at_floor")))
                    if rec.get("fes") is not None:
                        agg["fes_sum"] += float(rec["fes"])
                        agg["fes_n"] += 1
                if rec.get("fe") is not None:
                    agg["fe_sum"] += float(rec["fe"])
                if rec.get("nonfinite"):
                    agg["nonfinite"] += 1
                    self.nonfinite += 1
            log.clear()


# ============================================================ milestone evaluation
def evaluate_gate(index: FabricIndex, tr: GlyphTrackers, *, timestep: int) -> dict:
    """Evaluate the declared composition phase gate on THIS boundary.

    Gate items (all for the currently presented glyph):

    ``A`` every active L1 column stably owned
    ``B`` every active L1 owner one-event mature from its own patch's active RGCs
    ``C`` every active L1 owner -> Eor path ready (structural + observed reliability)
    ``D`` both tower L2 columns stably owned and one-event mature
    ``E`` both L2 owner -> Eor paths ready
    ``F`` every active non-top C cell one-shot ready
    ``G`` L3 stably owned and structurally one-event mature from its participating L2 Eors

    Blank patches cannot satisfy an ownership criterion and never hold the phase hostage;
    they are still recorded. L3 Eor and L3 C are reported but excluded (L3 has no parent
    and no higher consumer here).
    """
    active_l1 = tr.active_l1
    detail: dict = {"active_l1_columns": active_l1, "l1": {}, "l2": {}, "l3": {}, "c": {}}

    a_ok = b_ok = c_ok = True
    for cid in active_l1:
        owner = tr.owner(cid)
        row: dict = {"owner": owner, "consolidated": tr.consolidated(cid),
                     "events": tr.event_counts[cid]}
        if owner is None:
            a_ok = b_ok = c_ok = False
        else:
            mat = one_event_maturity(index.cell(owner), tr.active_rgc[cid])
            row["maturity"] = mat
            b_ok &= mat["mature"]
            path = tr.eor_paths[cid].report(min_boundaries=EOR_MIN_DELIVERIES,
                                            threshold=EOR_MIN_RELIABILITY)
            eor_mat = one_event_maturity(index.cell(index.eor[cid]), [owner])
            row["eor_path"] = path
            row["eor_maturity"] = eor_mat
            c_ok &= bool(path["ready"] and eor_mat["mature"])
        detail["l1"][cid] = row

    d_ok = e_ok = True
    for l2 in index.l2_columns:
        owner = tr.owner(l2)
        child_eors = tr.active_child_eors[l2]
        row = {"owner": owner, "consolidated": tr.consolidated(l2),
               "events": tr.event_counts[l2], "active_child_eors": child_eors}
        if owner is None or not child_eors:
            d_ok = e_ok = False
        else:
            mat = one_event_maturity(index.cell(owner), child_eors)
            row["maturity"] = mat
            d_ok &= mat["mature"]
            path = tr.eor_paths[l2].report(min_boundaries=EOR_MIN_DELIVERIES,
                                           threshold=EOR_MIN_RELIABILITY)
            eor_mat = one_event_maturity(index.cell(index.eor[l2]), [owner])
            row["eor_path"] = path
            row["eor_maturity"] = eor_mat
            e_ok &= bool(path["ready"] and eor_mat["mature"])
        detail["l2"][l2] = row

    f_ok = True
    gated_c = tr.gated_c
    for cid in gated_c:
        cell = index.cell(index.c[cid])
        st = tr.c_stats[cid]
        rep = c_readiness(cell.basal_weight, cell.threshold, deposits=st["deposits"],
                          spikes=st["spikes"], updates=st["updates"],
                          opportunities=st["opportunities"])
        detail["c"][cid] = rep
        f_ok &= rep["ready"]

    l3 = index.l3
    l3_owner = tr.owner(l3)
    participating = tr.participating_l2_eors
    l3row = {"owner": l3_owner, "consolidated": tr.consolidated(l3),
             "events": tr.event_counts[l3], "participating_l2_eors": participating}
    g_ok = l3_owner is not None and bool(participating)
    if g_ok:
        mat = one_event_maturity(index.cell(l3_owner), participating)
        l3row["maturity"] = mat
        g_ok = mat["mature"]
    # reported, never gated
    l3row["eor_path"] = tr.eor_paths[l3].report(min_boundaries=EOR_MIN_DELIVERIES,
                                                threshold=EOR_MIN_RELIABILITY)
    l3c = index.cell(index.c[l3])
    st = tr.c_stats[l3]
    l3row["c_reported_excluded"] = c_readiness(
        l3c.basal_weight, l3c.threshold, deposits=st["deposits"], spikes=st["spikes"],
        updates=st["updates"], opportunities=st["opportunities"])
    detail["l3"] = l3row

    milestones = {"A": bool(a_ok and active_l1), "B": bool(b_ok and active_l1),
                  "C": bool(c_ok and active_l1), "D": bool(d_ok), "E": bool(e_ok),
                  "F": bool(f_ok and gated_c), "G": bool(g_ok)}
    detail["milestones"] = milestones
    detail["gated_c_columns"] = gated_c
    detail["excluded_dormant_c"] = index.dormant_c
    detail["timestep"] = int(timestep)
    return detail


# ============================================================ training loop
def train_glyph(engine: SimulationEngine, index: FabricIndex, tr: GlyphTrackers,
                rec: Optional[ReplayRecorder], *, glyph: str, vector: Sequence[int],
                timeout: int, ownership: Mapping[str, Any], phase: str,
                gate_keys: Sequence[str] = GATE_KEYS,
                l3_signature: Optional[L3Signature] = None,
                progress=None) -> dict:
    """Present one glyph until the declared stop gate holds for ``GATE_CONFIRM``
    consecutive boundaries, or ``timeout`` boundaries elapse. A timeout is an UNFINISHED
    phase, never evidence of learning: training continues to the next glyph from the
    preserved state.

    ``gate_keys`` selects which milestones may STOP the phase. Every milestone is still
    evaluated and recorded on every boundary; the preflight simply cannot stop on ``G``
    (L3 maturity) because its L3 plasticity is deliberately frozen."""
    engine.set_input(list(vector))
    start_ts = int(engine.timestep) + 1
    if rec is not None:
        rec.set_annotation(phase=phase, pattern=glyph)
        rec.marker("glyph_start", data={"glyph": glyph, "start_timestep": start_ts,
                                        "active_l1_columns": index.active_l1_columns(vector)})

    eor_of = index.eor
    l3_sources = [index.eor[l2] for l2 in index.l2_columns]
    l3_cells = [index.cell(nid) for nid in index.ordinary["L3"][index.l3]]
    consecutive = 0
    outcome = "timeout"
    last_detail: dict = {}
    boundaries = 0

    for b in range(1, int(timeout) + 1):
        # 1. step
        dyn = engine.step()
        boundaries = b
        t = int(engine.timestep)

        # 2. ordinary winner events
        for cid, w in dyn.get("column_winners", {}).items():
            tr.note_winner(cid, w["id"], t, **ownership)

        # 3. spikes / relays / C diagnostics (unsampled)
        for cid in index.columns:
            eor_id = eor_of[cid]
            spiked_eor = bool(engine.spiked.get(eor_id))
            tau = index.cell(eor_id).spike_tau if spiked_eor else None
            tr.eor_paths[cid].note_boundary(t, spiked_eor, tau)
            if cid in dyn.get("column_winners", {}):
                tr.eor_paths[cid].note_source_event(t)

            c_cell = index.cell(index.c[cid])
            st = tr.c_stats[cid]
            apical = c_cell.apical_delivery_count > 0
            basal = c_cell.basal_delivery_count > 0
            gate = bool(c_cell.coincidence_active)
            st["apical_boundaries"] += int(apical)
            st["basal_boundaries"] += int(basal)
            st["gate_open"] += int(gate)
            if gate or (apical and basal):
                st["opportunities"] += 1
            st["deposits"] += int(c_cell.coincidence_deposit_count)
            if engine.spiked.get(index.c[cid]):
                st["spikes"] += 1
                if st["first_spike"] is None:
                    st["first_spike"] = t
            if (st["first_one_shot_at"] is None
                    and float(c_cell.basal_weight) >= float(c_cell.threshold) - 1e-9):
                st["first_one_shot_at"] = t
        tr.hard_resets += len(dyn.get("hard_reset_events", []))
        tr.feedback_resets += sum(1 for e in dyn.get("hard_reset_events", [])
                                  if e.get("kind") == "feedback_hard_reset")
        tr.latency_ties += len(dyn.get("latency_ties", []))
        for nid, s in engine.spiked.items():
            if s:
                tr.spike_counts[nid] = tr.spike_counts.get(nid, 0) + 1

        # L3 input signature (frozen probes only; the two L2-Eor delivery streams)
        if l3_signature is not None:
            for src in l3_sources:
                if engine.spiked.get(src):
                    q = sum(float(w) for cell in l3_cells
                            for s, w in zip(cell.ff_src, cell.acc_weights) if s == src)
                    l3_signature.events.append(
                        (t - start_ts + 2, src, round(q, 6)))     # delivery lands at t+1
                    tau = index.cell(src).spike_tau
                    l3_signature.taus.append(None if tau is None else round(float(tau), 9))

        # 4. drain update logs immediately
        tr.drain_updates(engine, t)
        for cid in index.columns:
            n_new = tr.c_stats[cid]["updates"]
            agg = tr.learning.get(index.c[cid])
            if agg is not None:
                tr.c_stats[cid]["updates"] = agg["update_events"]
            else:
                tr.c_stats[cid]["updates"] = n_new

        # 5. milestones from this unsampled boundary
        detail = evaluate_gate(index, tr, timestep=t)
        last_detail = detail
        for name, ok in detail["milestones"].items():
            if ok and name not in tr.milestone_first:
                tr.milestone_first[name] = t
                if rec is not None:
                    rec.marker("milestone", data={"glyph": glyph, "milestone": name,
                                                  "timestep": t})
        consecutive = (consecutive + 1
                       if all(detail["milestones"][k] for k in gate_keys) else 0)

        # 6/7. replay frame (sampling never controls analysis)
        if rec is not None:
            rec.record_frame(engine)

        if progress is not None and b % 2000 == 0:
            progress(f"{phase}/{glyph} t={t} gate={detail['milestones']} "
                     f"L3owner={detail['l3']['owner']}")

        if consecutive >= GATE_CONFIRM:
            outcome = "full_gate_ready"
            break

    if rec is not None:
        rec.record_frame(engine, force=True)
        rec.marker("glyph_end", data={"glyph": glyph, "outcome": outcome,
                                      "boundaries": boundaries,
                                      "milestone_first": dict(tr.milestone_first),
                                      "milestone_final": last_detail.get("milestones", {})})
    return {"glyph": glyph, "outcome": outcome, "boundaries": boundaries,
            "stop_gate_keys": list(gate_keys),
            "start_timestep": start_ts, "end_timestep": int(engine.timestep),
            "milestone_first": dict(tr.milestone_first),
            "milestone_final": last_detail.get("milestones", {}),
            "detail": last_detail}


# ============================================================ cold-state probes
def cold_probe(*, seed: int, B: float, cc_e_count: int, weights: Mapping[str, float],
               glyph: str, boundaries: int, ownership: Mapping[str, Any],
               settle_after_consolidation: int = 200,
               capture_signature: bool = True) -> dict:
    """Independent frozen probe for ONE glyph on a FRESH identical engine and a cold
    dynamic state. Never mutates the training engine and never reuses a probe engine
    across glyphs (residual charge would make the probe order-dependent)."""
    engine = make_two_tower_engine(seed=seed, B=B, cc_e_count=cc_e_count)
    index = FabricIndex(engine)
    transfer_plastic_weights_from_mapping(engine, weights)
    snap = freeze_learning(engine)

    vector = glyph_vector(glyph)
    engine.set_input(list(vector))
    tr = GlyphTrackers(index, vector, window=ownership["window"],
                       stable_windows=ownership["stable_windows"])
    sig = L3Signature(glyph=glyph,
                      source_ids=[index.eor[l2] for l2 in index.l2_columns]) \
        if capture_signature else None

    start_ts = int(engine.timestep) + 1
    l3_sources = [index.eor[l2] for l2 in index.l2_columns]
    l3_cells = [index.cell(nid) for nid in index.ordinary["L3"][index.l3]]
    settle_left = None
    observed = 0
    for _b in range(int(boundaries)):
        engine.step()
        t = int(engine.timestep)
        observed = t - start_ts + 1
        for cid, w in engine.column_winners.items():
            tr.note_winner(cid, w["id"], t, **ownership)
        if sig is not None:
            for src in l3_sources:
                if engine.spiked.get(src):
                    q = sum(float(w) for cell in l3_cells
                            for s, w in zip(cell.ff_src, cell.acc_weights) if s == src)
                    sig.events.append((t - start_ts + 2, src, round(q, 6)))
                    tau = index.cell(src).spike_tau
                    sig.taus.append(None if tau is None else round(float(tau), 9))
        if settle_left is None:
            if all(tr.consolidated(l2) for l2 in index.l2_columns):
                settle_left = int(settle_after_consolidation)
        elif settle_left > 0:
            settle_left -= 1
            if settle_left == 0:
                break
    assert_weights_unchanged(engine, snap)
    if sig is not None:
        sig.observed_boundaries = observed

    owners = {cid: tr.owner(cid) for cid in index.columns}
    verdicts = {cid: (tr.verdict[cid].as_dict() if tr.verdict[cid] is not None else None)
                for cid in index.columns}
    return {"glyph": glyph, "observed_boundaries": observed, "owners": owners,
            "verdicts": verdicts, "signature": sig,
            "l3_owner": owners[index.l3],
            "l2_owners": {l2: owners[l2] for l2 in index.l2_columns},
            "event_counts": dict(tr.event_counts)}


def transfer_plastic_weights_from_mapping(engine: SimulationEngine,
                                          weights: Mapping[str, float]) -> None:
    """Install a plastic-weight snapshot (edge id -> weight) into ``engine`` exactly."""
    live = plastic_edge_weights(engine)
    if set(live) != set(weights):
        missing = sorted(set(live) ^ set(weights))
        raise ValueError(f"plastic edge id mismatch: {missing[:8]}...")
    for eid, w in weights.items():
        engine.set_synapse_weight(eid, float(w))
    after = plastic_edge_weights(engine)
    for eid, w in weights.items():
        if after[eid] != float(w):
            raise AssertionError(f"weight transfer inexact for {eid}: {w} -> {after[eid]}")


# ============================================================ hashes / artifacts
def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def topology_fingerprint(engine: SimulationEngine) -> tuple:
    """SHA-256 of the validated ``current_spec()`` with only live display ``pos``
    removed, serialized with sorted keys and compact separators."""
    spec = engine.current_spec()
    nodes = []
    for n in spec["nodes"]:
        node = {k: v for k, v in n.items() if k != "pos"}
        nodes.append(node)
    payload = {"name": spec.get("name"), "topology": spec.get("topology"),
               "nodes": nodes, "edges": spec["edges"]}
    text = _canonical(payload)
    return _sha256(text), payload


def config_hash(config: Mapping[str, Any]) -> tuple:
    text = _canonical(config)
    return _sha256(text), text


def _load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _dedupe_failures(records: Sequence[Mapping[str, Any]]) -> list:
    """Stable de-duplication so re-running a phase never inflates the failure catalog."""
    out, seen = [], set()
    for r in records:
        key = _canonical({k: r.get(k) for k in
                          ("taxon", "first_seed", "first_glyph", "first_layer",
                           "first_column", "smallest_reproducing_condition")})
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(r))
    return out


def _atomic_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_csv(path: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _git_state(repo_dir: str) -> dict:
    import subprocess
    def run(*args):
        try:
            return subprocess.run(args, cwd=repo_dir, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:                                  # noqa: BLE001
            return ""
    commit = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {"commit": commit or None, "dirty": dirty}


# ============================================================ cell markers / resume
def cell_dir(run_dir: str, slug: str) -> str:
    return os.path.join(run_dir, "cells", slug)


def read_cell_status(path: str) -> Optional[dict]:
    p = os.path.join(path, "status.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_cell_status(path: str, **fields) -> None:
    _atomic_json(os.path.join(path, "status.json"), fields)


def cell_is_complete(path: str, chash: str) -> bool:
    """A cell may be skipped ONLY when its atomic marker says completed AND its recorded
    config hash equals the requested one. Directory existence alone is never enough."""
    st = read_cell_status(path)
    return bool(st and st.get("status") == STATUS_COMPLETED
                and st.get("config_hash") == chash)


def next_attempt_dir(base: str) -> str:
    """A failed/interrupted cell stays inspectable; a rerun opens a new attempt dir."""
    if not os.path.isdir(base) or read_cell_status(base) is None:
        return base
    n = 2
    while os.path.isdir(f"{base}.attempt{n}"):
        n += 1
    return f"{base}.attempt{n}"


# ============================================================ phases
def phase_structure(run_dir: str, *, cc_e_count: int, seed: int, B: float) -> dict:
    """Build + validate the graph and glyphs; write the structural artifacts."""
    engine = make_two_tower_engine(seed=seed, B=B, cc_e_count=cc_e_count)
    index = FabricIndex(engine)
    index.assert_builder_naming()
    fp, payload = topology_fingerprint(engine)
    spec = two_tower_composition_spec(cc_e_count=cc_e_count)

    counts = {"nodes": len(spec["nodes"]), "edges": len(spec["edges"]),
              "l1_columns": len(index.l1_columns), "l2_columns": len(index.l2_columns),
              "l3_columns": len(index.l3_columns),
              "rgc": sum(1 for n in spec["nodes"] if n["archetype"] == "rg_source")}
    audit = assert_engine_contract(engine, B=B, cc_e_count=cc_e_count)
    _atomic_json(os.path.join(run_dir, "composition_topology.json"), {
        "schema_version": SCHEMA_VERSION,
        "name": spec["name"], "counts": counts,
        "topology_fingerprint": fp,
        "canonical_payload_sha256": fp,
        "engine_audit": audit,
        "columns": index.columns,
        "towers": {l2: index.towers[l2] for l2 in index.l2_columns},
        "dormant_c_columns": index.dormant_c,
        "spec": payload,
    })

    glyph_report = verify_glyphs(CANONICAL_GLYPH_ORDER)
    for g, rep in glyph_report.items():
        rep["vector"] = glyph_vector(g)
        rep["active_l1_columns"] = index.active_l1_columns(rep["vector"])
        rep["active_columns_by_tower"] = {
            l2: sorted(set(rep["active_l1_columns"]) & set(index.towers[l2]))
            for l2 in index.l2_columns}
    _atomic_json(os.path.join(run_dir, "composition_glyphs.json"), {
        "schema_version": SCHEMA_VERSION,
        "sheet": {"rows": 9, "cols": GLYPH_COLS, "seam_column": GLYPH_COLS // 2},
        "glyph_order": list(CANONICAL_GLYPH_ORDER),
        "glyphs": glyph_report,
    })
    return {"counts": counts, "topology_fingerprint": fp,
            "engine_audit": audit, "glyphs": glyph_report}


def _disable_l3_plasticity(engine: SimulationEngine, index: FabricIndex) -> list:
    """Preflight intervention: disable LEARNING ONLY on L3 ordinary E and Eor, selected
    by metadata. Firing and every graph path stay intact -- L3 is never disconnected and
    its top-down feedback is never suppressed."""
    frozen = []
    for nid in index.ordinary["L3"][index.l3] + [index.eor[index.l3]]:
        cell = engine.exc[nid]
        cell.learn = False
        frozen.append(nid)
    return sorted(frozen)


def run_training_cell(*, run_dir: str, slug: str, seed: int, B: float, cc_e_count: int,
                      glyphs: Sequence[str], timeout: int, freeze_l3: bool,
                      ownership: Mapping[str, Any], chash: str, config: Mapping[str, Any],
                      record_every: int, quiet: bool) -> dict:
    """One complete training cell (preflight pretraining or a composition condition),
    followed by frozen cold-state probes for every glyph."""
    path = next_attempt_dir(cell_dir(run_dir, slug))
    os.makedirs(path, exist_ok=True)
    write_cell_status(path, status="running", config_hash=chash, slug=slug,
                      started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def progress(msg):
        if not quiet:
            print(f"[{slug}] {msg}", flush=True)

    t0 = time.perf_counter()
    engine = make_two_tower_engine(seed=seed, B=B, cc_e_count=cc_e_count)
    index = FabricIndex(engine)
    fp, _payload = topology_fingerprint(engine)
    frozen_l3 = _disable_l3_plasticity(engine, index) if freeze_l3 else []
    l3_weights_before = {eid: w for eid, w in plastic_edge_weights(engine).items()
                         if engine._ff_weight_ref.get(eid)
                         and engine._ff_weight_ref[eid][0].id in set(
                             index.ordinary["L3"][index.l3] + [index.eor[index.l3]])}

    for cell in engine.plastic:
        cell.record_updates = True
    for cell in engine.coincidence:
        cell.record_updates = True

    rec = ReplayRecorder(
        engine, experiment=f"two_tower_composition.{slug}",
        output_root=os.path.join(run_dir, "replays"), run_id=slug, seed=seed,
        record_every=record_every, checkpoint_every=CHECKPOINT_EVERY_FRAMES,
        hierarchical_feedback=FEEDBACK_INTACT,
        conditions=dict(config), schedule={"glyph_order": list(glyphs),
                                           "glyph_timeout": int(timeout)},
        metrics_columns=[
            "slug", "seed", "B", "phase", "glyph", "timestep", "layer", "column_id",
            "neuron_id", "role", "winner_events", "dominance", "consolidated",
            "owner_first_at", "owner_lost_count", "active", "one_event_mature",
            "active_charge", "threshold", "margin", "eor_reliability",
            "eor_delivery_boundaries", "c_basal_weight", "c_one_shot", "c_deposits",
            "c_spikes", "c_updates", "outcome",
        ],
        optional_columns=["dominance", "one_event_mature", "active_charge", "margin",
                          "eor_reliability", "eor_delivery_boundaries", "c_basal_weight",
                          "c_one_shot", "c_deposits", "c_spikes", "c_updates",
                          "owner_first_at", "threshold"],
    )

    try:
        phase_name = "preflight_pretrain" if freeze_l3 else "composition_train"
        gate_keys = PREFLIGHT_GATE_KEYS if freeze_l3 else GATE_KEYS
        glyph_results = {}
        learning_rows = []
        trackers = {}
        for glyph in glyphs:
            vector = glyph_vector(glyph)
            tr = GlyphTrackers(index, vector, window=ownership["window"],
                               stable_windows=ownership["stable_windows"])
            res = train_glyph(engine, index, tr, rec, glyph=glyph, vector=vector,
                              timeout=timeout, ownership=ownership, phase=phase_name,
                              gate_keys=gate_keys, progress=progress)
            glyph_results[glyph] = res
            trackers[glyph] = tr
            progress(f"glyph {glyph}: {res['outcome']} after {res['boundaries']} boundaries "
                     f"(milestones {res['milestone_final']})")
            _emit_metric_rows(rec, index, tr, res, slug=slug, seed=seed, B=B,
                              phase=phase_name, glyph=glyph)
            for cell_id, agg in sorted(tr.learning.items()):
                learning_rows.append(_learning_row(index, agg, phase=phase_name,
                                                   slug=slug, seed=seed, B=B, glyph=glyph,
                                                   interval=res["boundaries"],
                                                   timestep=res["end_timestep"]))

        # ---- frozen snapshot + independent cold-state probes ----
        rec.marker("learning_freeze", data={"frozen_l3_cells": frozen_l3})
        weights = plastic_edge_weights(engine)
        l3_frozen_invariant = None
        if freeze_l3:
            after = {eid: weights[eid] for eid in l3_weights_before}
            drift = {eid: [l3_weights_before[eid], after[eid]]
                     for eid in after if after[eid] != l3_weights_before[eid]}
            if drift:
                raise AssertionError(f"L3 plastic weights moved under the freeze: {drift}")
            l3_frozen_invariant = {"n_edges": len(after), "drifted": 0}

        probes = {}
        for glyph in glyphs:
            pr = cold_probe(seed=seed, B=B, cc_e_count=cc_e_count, weights=weights,
                            glyph=glyph, boundaries=config["recall_boundaries"],
                            ownership=ownership)
            probes[glyph] = pr
            progress(f"cold probe {glyph}: L3 owner={pr['l3_owner']} "
                     f"L2 owners={pr['l2_owners']}")

        signatures = {g: probes[g]["signature"] for g in glyphs}
        separability = classify_signatures(signatures)
        rec.marker("separability", data={k: v for k, v in separability.items()
                                         if k != "source_participation"})

        result = {
            "slug": slug, "seed": seed, "B": B, "freeze_l3": freeze_l3,
            "frozen_l3_cells": frozen_l3,
            "l3_frozen_weight_invariant": l3_frozen_invariant,
            "topology_fingerprint": fp,
            "glyph_order": list(glyphs),
            "glyph_results": {g: {k: v for k, v in r.items() if k != "detail"}
                              for g, r in glyph_results.items()},
            "glyph_detail": {g: r["detail"] for g, r in glyph_results.items()},
            "cold_probes": {g: {k: v for k, v in pr.items() if k != "signature"}
                            for g, pr in probes.items()},
            "l3_signatures": {g: {"events": sig.events, "taus": sig.taus,
                                  "observed_boundaries": sig.observed_boundaries,
                                  "source_ids": sig.source_ids,
                                  "summary": sig.summary()}
                              for g, sig in signatures.items()},
            "separability": separability,
            "trained_weights_sha256": _sha256(_canonical(
                {k: repr(float(v)) for k, v in sorted(weights.items())})),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "learning_rows": learning_rows,
            "nonfinite_total": sum(tr.nonfinite for tr in trackers.values()),
        }
        checks = {
            "no_nonfinite": result["nonfinite_total"] == 0,
            "all_glyphs_full_gate": all(r["outcome"] == "full_gate_ready"
                                        for r in glyph_results.values()),
            "three_distinct_cold_l3_owners": _distinct_l3(probes, glyphs),
        }
        if freeze_l3:
            checks["l3_weights_unchanged_under_freeze"] = True
        rec.finish(STATUS_COMPLETED, checks=checks, result={
            k: v for k, v in result.items() if k != "learning_rows"})
        # Replay accounting, recorded AFTER the stream is closed: planned vs actual
        # boundaries, calculated stride, frames actually written, and bytes on disk.
        replay_path = os.path.join(rec.run_dir, "replay.snn.jsonl")
        result["recording"] = {
            "record_every": rec.record_every,
            "checkpoint_every_frames": rec.checkpoint_every,
            "planned_max_boundaries": len(glyphs) * int(timeout),
            "actual_boundaries": sum(r["boundaries"] for r in glyph_results.values()),
            "frames_written": rec.frames_written,
            "replay_bytes": (os.path.getsize(replay_path)
                             if os.path.isfile(replay_path) else None),
            "replay_path": replay_path,
        }
        _atomic_json(os.path.join(path, "result.json"), result)
        write_cell_status(path, status=STATUS_COMPLETED, config_hash=chash, slug=slug,
                          checks=checks, run_dir=path, replay_dir=rec.run_dir,
                          frames=rec.frames_written,
                          completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return result
    except BaseException as exc:                            # noqa: BLE001
        if not rec._finalized:
            rec.finish(STATUS_FAILED, checks={}, result={"error": repr(exc)},
                       failure_reason=repr(exc))
        write_cell_status(path, status=STATUS_FAILED, config_hash=chash, slug=slug,
                          error=repr(exc))
        raise


def _distinct_l3(probes: Mapping[str, dict], glyphs: Sequence[str]) -> bool:
    owners = [probes[g]["l3_owner"] for g in glyphs]
    return all(o is not None for o in owners) and len(set(owners)) == len(owners)


def _learning_row(index: FabricIndex, agg: Mapping[str, Any], *, phase: str, slug: str,
                  seed: int, B: float, glyph: str, interval: int, timestep: int) -> dict:
    cid = index.engine._column_of.get(agg["cell_id"])
    return {
        "phase": phase, "condition": slug, "seed": seed, "B": B, "glyph": glyph,
        "timestep": timestep, "interval_boundaries": interval,
        "layer": index.columns[cid]["layer"] if cid else "",
        "column_id": cid or "", "role": index.engine._role_of.get(agg["cell_id"]) or "",
        "cell_id": agg["cell_id"], "update_events": agg["update_events"],
        "positive_synapses": agg["positive_synapses"],
        "negative_synapses": agg["negative_synapses"],
        "floor_hits": agg["floor_hits"],
        "sum_abs_raw_dw": round(agg["sum_abs_raw_dw"], 6),
        "sum_abs_applied_dw": round(agg["sum_abs_applied_dw"], 6),
        "mean_FE": (round(agg["fe_sum"] / agg["update_events"], 6)
                    if agg["update_events"] else ""),
        "mean_FES": (round(agg["fes_sum"] / agg["fes_n"], 6) if agg["fes_n"] else ""),
        "first_update_at": agg["first_update_at"], "last_update_at": agg["last_update_at"],
        "nonfinite": agg["nonfinite"],
    }


def _emit_metric_rows(rec: ReplayRecorder, index: FabricIndex, tr: GlyphTrackers,
                      res: Mapping[str, Any], *, slug: str, seed: int, B: float,
                      phase: str, glyph: str) -> None:
    detail = res["detail"]
    active = set(detail.get("active_l1_columns", []))
    for cid in sorted(index.columns):
        layer = index.columns[cid]["layer"]
        owner = tr.owner(cid)
        v = tr.verdict[cid]
        row_detail = (detail["l1"].get(cid) or detail["l2"].get(cid)
                      or (detail["l3"] if cid == index.l3 else {}) or {})
        mat = row_detail.get("maturity") or {}
        path = row_detail.get("eor_path") or {}
        c_cell = index.cell(index.c[cid])
        st = tr.c_stats[cid]
        rec.metrics.append_row({
            "slug": slug, "seed": seed, "B": B, "phase": phase, "glyph": glyph,
            "timestep": res["end_timestep"], "layer": layer, "column_id": cid,
            "neuron_id": owner or "", "role": "E",
            "winner_events": tr.event_counts[cid],
            "dominance": ("" if v is None or v.final_window_dominance is None
                          else v.final_window_dominance),
            "consolidated": bool(owner is not None),
            "owner_first_at": ("" if tr.owner_first_at[cid] is None
                               else tr.owner_first_at[cid]),
            "owner_lost_count": len(tr.owner_lost_at[cid]),
            "active": bool(cid in active or layer in ("L2", "L3")),
            "one_event_mature": mat.get("mature", ""),
            "active_charge": mat.get("active_charge", ""),
            "threshold": mat.get("threshold", ""),
            "margin": mat.get("margin", ""),
            "eor_reliability": path.get("response_reliability", ""),
            "eor_delivery_boundaries": path.get("eligible_delivery_boundaries", ""),
            "c_basal_weight": round(float(c_cell.basal_weight), 6),
            "c_one_shot": bool(float(c_cell.basal_weight)
                               >= float(c_cell.threshold) - 1e-9),
            "c_deposits": st["deposits"], "c_spikes": st["spikes"],
            "c_updates": st["updates"], "outcome": res["outcome"],
        })


# ============================================================ orchestration
LEARNING_CSV_COLUMNS = [
    "phase", "condition", "seed", "B", "glyph", "timestep", "interval_boundaries",
    "layer", "column_id", "role", "cell_id", "update_events", "positive_synapses",
    "negative_synapses", "floor_hits", "sum_abs_raw_dw", "sum_abs_applied_dw",
    "mean_FE", "mean_FES", "first_update_at", "last_update_at", "nonfinite",
]
RESULTS_CSV_COLUMNS = [
    "condition", "seed", "B", "glyph", "outcome", "boundaries", "left_L2_owner",
    "right_L2_owner", "left_L2_Eor_ready", "right_L2_Eor_ready",
    "L3_observed_source_signature", "L3_owner", "L3_dominance", "L3_distinct_mapping",
    "cold_recall_owner", "cold_recall_match", "representation_identifiable",
    "first_failure_taxon", "nonfinite_count",
]


def run_all(*, run_dir: str, seeds: Sequence[int], B: float, cc_e_count: int,
            glyph_timeout: int, recall_boundaries: int, phases: Sequence[str],
            resume: bool, quick: bool, quiet: bool) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    slugs = SlugRegistry()
    ownership = dict(OWNERSHIP)
    git = _git_state(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    planned = len(CANONICAL_GLYPH_ORDER) * int(glyph_timeout)
    record_every = max(1, math.ceil(planned / REPLAY_FRAME_BUDGET))

    base_config = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "two_tower_composition",
        "task": "Claude_Final_Experiments_Prompt.md Task 4 (only)",
        "engine_contract": dict(ENGINE_CONTRACT),
        "dual_fe_B": float(B),
        "cc_e_count": int(cc_e_count),
        "glyph_order": list(CANONICAL_GLYPH_ORDER),
        "glyph_timeout": int(glyph_timeout),
        "recall_boundaries": int(recall_boundaries),
        "ownership": ownership,
        "eor_readiness": {"min_delivery_boundaries": EOR_MIN_DELIVERIES,
                          "min_reliability": EOR_MIN_RELIABILITY},
        "gate_confirm_boundaries": GATE_CONFIRM,
        "recording": {"record_every": record_every,
                      "checkpoint_every_frames": CHECKPOINT_EVERY_FRAMES,
                      "planned_max_boundaries": planned,
                      "frame_budget": REPLAY_FRAME_BUDGET},
        "seeds": list(seeds),
        "quick": bool(quick),
        "git": git,
        # Task 1 was not executed in this bounded run, so no low-B candidate exists.
        "low_B_candidate": None,
        "low_B_candidate_reason": (
            "Task 1 (adaptive hold-out B sweep) was not executed in this bounded run; "
            "the prompt's second composition condition is therefore not evaluable and is "
            "recorded as null rather than substituted with an undeclared B."),
    }
    _atomic_json(os.path.join(run_dir, "config.json"), base_config)

    # A later-phase invocation on an existing parent run MERGES into that run's aggregate:
    # a phase that did not run this time keeps whatever the earlier invocation recorded and
    # is never silently reset to null.
    status = _load_json(os.path.join(run_dir, "status.json")) or {}
    status.setdefault("phases", {})
    status.setdefault("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    aggregate: dict = _load_json(os.path.join(run_dir, "aggregate_summary.json")) or {
        "structure": None, "preflight": None, "composition": None, "failures": []}
    aggregate.update({"schema_version": SCHEMA_VERSION,
                      "run_id": os.path.basename(run_dir),
                      "repository": git, "config": base_config})
    aggregate.setdefault("artifacts", {})
    aggregate.setdefault("completion", {})
    failures: list = list(aggregate.get("failures") or [])
    learning_rows: list = []
    results_rows: list = []

    # ---------------- structure ----------------
    if "structure" in phases:
        struct = phase_structure(run_dir, cc_e_count=cc_e_count, seed=seeds[0], B=B)
        aggregate["structure"] = {k: v for k, v in struct.items() if k != "glyphs"}
        aggregate["structure"]["glyph_patch_summary"] = {
            g: {"n_active": rep["n_active"], "half_counts": rep["half_counts"],
                "patches": rep["patches"], "active_l1_columns": rep["active_l1_columns"]}
            for g, rep in struct["glyphs"].items()}
        status["phases"]["structure"] = STATUS_COMPLETED
        _atomic_json(os.path.join(run_dir, "status.json"), status)

    # ---------------- preflight ----------------
    if "preflight" in phases:
        seed = seeds[0]
        cfg = dict(base_config, phase="preflight", seed=seed, freeze_l3=True,
                   recall_boundaries=int(recall_boundaries))
        chash, payload = config_hash(cfg)
        slug = slugs.claim(condition_slug("preflight", seed=seed, b=B))
        path = cell_dir(run_dir, slug)
        if resume and cell_is_complete(path, chash):
            with open(os.path.join(path, "result.json")) as f:
                res = json.load(f)
            print(f"skip (resume) {slug}", flush=True)
        else:
            res = run_training_cell(
                run_dir=run_dir, slug=slug, seed=seed, B=B, cc_e_count=cc_e_count,
                glyphs=CANONICAL_GLYPH_ORDER, timeout=glyph_timeout, freeze_l3=True,
                ownership=ownership, chash=chash, config=cfg, record_every=record_every,
                quiet=quiet)
            _atomic_json(os.path.join(path, "config_payload.json"),
                         {"config_hash": chash, "canonical": payload})
        learning_rows += res.get("learning_rows", [])
        aggregate["preflight"] = {
            "slug": slug, "seed": seed,
            "frozen_l3_cells": res["frozen_l3_cells"],
            "l3_frozen_weight_invariant": res.get("l3_frozen_weight_invariant"),
            "recording": res.get("recording"),
            "glyph_results": res["glyph_results"],
            "separability": res["separability"],
            "l3_signature_summaries": {g: s["summary"]
                                       for g, s in res["l3_signatures"].items()},
            "cold_probe_l2_owners": {g: p["l2_owners"]
                                     for g, p in res["cold_probes"].items()},
        }
        _atomic_json(os.path.join(run_dir, "preflight_signatures.json"), {
            "schema_version": SCHEMA_VERSION, "slug": slug, "seed": seed,
            "declared_l3_sources": res["separability"]["declared_l3_source_ids"],
            "separability": res["separability"],
            "signatures": res["l3_signatures"],
        })
        if not res["separability"]["identifiable"]:
            failures.append(failure_record(
                "representation_not_identifiable",
                first_seed=seed, first_layer="L3", first_column="L3c00",
                first_glyph=None,
                smallest_reproducing_condition=(
                    f"two-tower graph, seed {seed}, B={B}, frozen cold probes of "
                    f"{list(CANONICAL_GLYPH_ORDER)}"),
                milestone_before_failure="lower-tower L1/L2 milestones (see glyph_results)",
                evidence=res["separability"],
                structural=True,
                parameter_sensitive=False,
                narrowest_future_action=(
                    "expose more than one source identity per tower to L3 (e.g. a second "
                    "output channel per column) -- NOT attempted here"),
            ))
        status["phases"]["preflight"] = STATUS_COMPLETED
        _atomic_json(os.path.join(run_dir, "status.json"), status)

    # ---------------- composition ----------------
    if "composition" in phases:
        comp: dict = {"conditions": {}, "declared_conditions": [
            {"name": "reference", "B": float(B), "wipe": False, "evaluable": True},
            {"name": "low_B", "B": None, "wipe": False, "evaluable": False,
             "reason": base_config["low_B_candidate_reason"]},
        ]}
        for seed in seeds:
            cfg = dict(base_config, phase="composition", seed=seed, freeze_l3=False,
                       condition="reference", recall_boundaries=int(recall_boundaries))
            chash, payload = config_hash(cfg)
            slug = slugs.claim(condition_slug("composition", condition="reference",
                                              b=B, seed=seed))
            path = cell_dir(run_dir, slug)
            if resume and cell_is_complete(path, chash):
                with open(os.path.join(path, "result.json")) as f:
                    res = json.load(f)
                print(f"skip (resume) {slug}", flush=True)
            else:
                res = run_training_cell(
                    run_dir=run_dir, slug=slug, seed=seed, B=B, cc_e_count=cc_e_count,
                    glyphs=CANONICAL_GLYPH_ORDER, timeout=glyph_timeout, freeze_l3=False,
                    ownership=ownership, chash=chash, config=cfg,
                    record_every=record_every, quiet=quiet)
                _atomic_json(os.path.join(path, "config_payload.json"),
                             {"config_hash": chash, "canonical": payload})
            learning_rows += res.get("learning_rows", [])
            comp["conditions"][slug] = {
                "seed": seed, "B": B,
                "recording": res.get("recording"),
                "glyph_results": res["glyph_results"],
                "separability": res["separability"],
                "cold_probes": res["cold_probes"],
                "distinct_cold_l3_owners": _distinct_l3_from_result(res),
            }
            results_rows += _results_rows(res, condition="reference", seed=seed, B=B)
            for taxon, extra in _classify_composition_failures(res, seed=seed, B=B):
                failures.append(failure_record(taxon, **extra))
            status["phases"][f"composition/{slug}"] = STATUS_COMPLETED
            _atomic_json(os.path.join(run_dir, "status.json"), status)

            # Declared gate: a structural representation failure at seed 1 stops the
            # seed sweep rather than spending time confirming an unidentifiable code.
            if not res["separability"]["identifiable"] and len(seeds) > 1:
                comp["seed_sweep_stopped"] = {
                    "after_seed": seed,
                    "reason": ("seed-1 failed for a structural representation reason; the "
                               "prompt's declared gate forbids spending a seed sweep on it"),
                    "remaining_seeds": [s for s in seeds if s > seed]}
                break
        aggregate["composition"] = comp

    # ---------------- artifacts ----------------
    if learning_rows:
        _write_csv(os.path.join(run_dir, "learning_event_counts.csv"),
                   LEARNING_CSV_COLUMNS, learning_rows)
    if results_rows:
        _write_csv(os.path.join(run_dir, "composition_results.csv"),
                   RESULTS_CSV_COLUMNS, results_rows)
    failures = _dedupe_failures(failures)
    _atomic_json(os.path.join(run_dir, "failure_catalog.json"), failures)
    aggregate["failures"] = failures
    aggregate["artifacts"] = {
        name: os.path.isfile(os.path.join(run_dir, name)) for name in
        ("config.json", "status.json", "composition_topology.json",
         "composition_glyphs.json", "preflight_signatures.json",
         "composition_results.csv", "learning_event_counts.csv", "failure_catalog.json")}
    aggregate["completion"] = {
        "phases_run": sorted(set(aggregate["completion"].get("phases_run") or [])
                             | set(phases)),
        "phases_run_this_invocation": list(phases),
        "quick": bool(quick),
        "not_evaluable": {
            "low_B_condition": base_config["low_B_candidate_reason"],
            "tasks_1_3": "Not executed: this run implements Task 4 only, as instructed.",
        },
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(os.path.join(run_dir, "aggregate_summary.json"), aggregate)
    status["finished_utc"] = aggregate["completion"]["finished_utc"]
    _atomic_json(os.path.join(run_dir, "status.json"), status)
    return aggregate


def _distinct_l3_from_result(res: Mapping[str, Any]) -> bool:
    owners = [res["cold_probes"][g]["l3_owner"] for g in res["glyph_order"]]
    return all(o is not None for o in owners) and len(set(owners)) == len(owners)


def _results_rows(res: Mapping[str, Any], *, condition: str, seed: int,
                  B: float) -> list:
    rows = []
    sep = res["separability"]
    l2_ids = sorted({k for g in res["glyph_order"]
                     for k in res["cold_probes"][g]["l2_owners"]})
    for glyph in res["glyph_order"]:
        gr = res["glyph_results"][glyph]
        detail = res["glyph_detail"][glyph]
        probe = res["cold_probes"][glyph]
        sig = res["l3_signatures"][glyph]["summary"]
        l2_rows = detail.get("l2", {})
        rows.append({
            "condition": condition, "seed": seed, "B": B, "glyph": glyph,
            "outcome": gr["outcome"], "boundaries": gr["boundaries"],
            "left_L2_owner": (l2_rows.get(l2_ids[0], {}) or {}).get("owner"),
            "right_L2_owner": (l2_rows.get(l2_ids[1], {}) or {}).get("owner")
            if len(l2_ids) > 1 else None,
            "left_L2_Eor_ready": ((l2_rows.get(l2_ids[0], {}) or {}).get("eor_path")
                                  or {}).get("ready"),
            "right_L2_Eor_ready": (((l2_rows.get(l2_ids[1], {}) or {}).get("eor_path")
                                    or {}).get("ready") if len(l2_ids) > 1 else None),
            "L3_observed_source_signature": _canonical(sig["events_per_source"]),
            "L3_owner": (detail.get("l3", {}) or {}).get("owner"),
            "L3_dominance": "",
            "L3_distinct_mapping": _distinct_l3_from_result(res),
            "cold_recall_owner": probe["l3_owner"],
            "cold_recall_match": (probe["l3_owner"] is not None
                                  and probe["l3_owner"] == (detail.get("l3", {}) or {}).get("owner")),
            "representation_identifiable": sep["identifiable"],
            "first_failure_taxon": ("representation_not_identifiable"
                                    if not sep["identifiable"] else ""),
            "nonfinite_count": res.get("nonfinite_total", 0),
        })
    return rows


def _classify_composition_failures(res: Mapping[str, Any], *, seed: int, B: float) -> list:
    out = []
    if not res["separability"]["identifiable"]:
        out.append(("representation_not_identifiable", dict(
            first_seed=seed, first_layer="L3", first_column="L3c00",
            smallest_reproducing_condition=f"composition training, seed {seed}, B={B}",
            evidence=res["separability"], structural=True, parameter_sensitive=False,
            narrowest_future_action=("more than one L3-visible source identity per tower "
                                     "-- NOT attempted here"))))
    if not _distinct_l3_from_result(res):
        owners = {g: res["cold_probes"][g]["l3_owner"] for g in res["glyph_order"]}
        out.append(("L2_mapping_collision", dict(
            first_seed=seed, first_layer="L3", first_column="L3c00",
            smallest_reproducing_condition=f"cold recall, seed {seed}, B={B}",
            evidence={"cold_recall_l3_owners": owners},
            structural=True, parameter_sensitive=False,
            narrowest_future_action="see representation_not_identifiable")))
    for glyph, gr in res["glyph_results"].items():
        if gr["outcome"] != "full_gate_ready":
            missing = [k for k, v in (gr.get("milestone_final") or {}).items() if not v]
            taxon = "timeout_unclassified"
            if missing == ["G"] or missing == ["G", "F"]:
                taxon = "representation_not_identifiable"
            elif "F" in missing:
                taxon = "C_coincidence_starvation"
            elif "D" in missing or "E" in missing:
                taxon = "L2_evidence_collapse"
            elif "A" in missing or "B" in missing:
                taxon = "learning_event_starvation"
            out.append((taxon, dict(
                first_seed=seed, first_glyph=glyph, first_layer="composition_gate",
                smallest_reproducing_condition=(
                    f"glyph {glyph}, seed {seed}, B={B}, timeout {gr['boundaries']}"),
                milestone_before_failure=gr.get("milestone_first"),
                evidence={"milestone_final": gr.get("milestone_final"),
                          "unmet": missing},
                structural=None, parameter_sensitive=None,
                narrowest_future_action="inspect the per-column detail in result.json")))
    return out


# ============================================================ CLI
def _parse_seeds(spec: str) -> list:
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out or [1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="all",
                   choices=("structure", "preflight", "composition", "all"))
    p.add_argument("--seed", type=int, default=None, help="single seed (alias for --seeds)")
    p.add_argument("--seeds", type=_parse_seeds, default=None,
                   help="seed list/range, e.g. '1' or '1-4' or '1,3' (default 1)")
    p.add_argument("--B", type=float, default=REFERENCE_B,
                   help=f"dual FE/FES sharpness (default {REFERENCE_B}, the reference)")
    p.add_argument("--cc-e-count", type=int, default=8)
    p.add_argument("--glyph-timeout", type=int, default=30000,
                   help="max boundaries per glyph (default 30000)")
    p.add_argument("--recall-boundaries", type=int, default=1000,
                   help="max boundaries per frozen cold probe (default 1000)")
    p.add_argument("--output-root", default=None,
                   help="create a NEW parent run under this root and print its path")
    p.add_argument("--run-dir", default=None,
                   help="open one EXISTING parent run for a later phase")
    p.add_argument("--resume", action="store_true",
                   help="skip a cell only when its atomic marker says completed AND its "
                        "recorded config hash matches")
    p.add_argument("--quick", action="store_true",
                   help="implementation smoke test; NEVER a scientific result")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_root and args.run_dir:
        raise SystemExit("supply --output-root OR --run-dir, never both")

    seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed else [1])
    glyph_timeout, recall = args.glyph_timeout, args.recall_boundaries
    if args.quick:
        glyph_timeout = min(glyph_timeout, 2000)
        recall = min(recall, 400)

    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--run-dir {run_dir!r} does not exist")
    else:
        root = args.output_root or "experiments/runs/two_tower_composition"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        rid = f"{stamp}-two_tower{'-quick' if args.quick else ''}"
        run_dir = os.path.join(root, rid)
        if os.path.exists(run_dir):
            raise SystemExit(f"run dir {run_dir!r} already exists; never overwrite a run")
        os.makedirs(run_dir)
    print(f"RUN_DIR={os.path.abspath(run_dir)}", flush=True)

    phases = (("structure", "preflight", "composition") if args.phase == "all"
              else (args.phase,))
    agg = run_all(run_dir=run_dir, seeds=seeds, B=args.B, cc_e_count=args.cc_e_count,
                  glyph_timeout=glyph_timeout, recall_boundaries=recall, phases=phases,
                  resume=args.resume, quick=args.quick, quiet=args.quiet)
    print(json.dumps({"run_dir": os.path.abspath(run_dir),
                      "phases": list(phases),
                      "separability": (agg.get("preflight") or {}).get("separability", {})
                      .get("verdict"),
                      "failures": [f["taxon"] for f in agg["failures"]]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
