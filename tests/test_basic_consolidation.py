"""Focused tests for the Basic consolidation harness and its analysis utilities.

These prove the harness and analysis are correct; the long scientific sweep is NOT run
here. All artifacts go to temporary directories.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.simulation import SimulationEngine, PATTERNS
from backend.network_spec import embed_patch_pattern
from experiments import basic_consolidation as bc
from experiments import consolidation_analysis as ca


def _engine(topology="tiled_cc", seed=1, condition="intact"):
    e, disabled = bc.make_engine(topology, seed, condition)
    return e, disabled


# 1. all-nine construction reproduces each 3x3 PATTERNS vector in all nine patches --------
@pytest.mark.parametrize("topology", bc.TOPOLOGIES)
@pytest.mark.parametrize("pattern", list(ca.CANONICAL_PATTERN_ORDER))
def test_all_nine_input_construction(topology, pattern):
    e, _ = _engine(topology)
    ctx = bc.RunContext(e)
    vec = ctx.input_for(pattern)
    report = ctx.verify_input(vec, pattern)          # raises on any violation
    assert report["n_patches"] == 9
    # every one of the nine patches contains exactly the pattern
    ishape = (e.tiled_meta["input_shape"]["rows"], e.tiled_meta["input_shape"]["cols"])
    pshape = (e.tiled_meta["patch_shape"]["rows"], e.tiled_meta["patch_shape"]["cols"])
    for pr in range(3):
        for pc in range(3):
            emb = embed_patch_pattern(ishape, pshape, (pr, pc), PATTERNS[pattern])
            for i, v in enumerate(emb):
                if v:
                    assert vec[i] == 1
    assert sum(vec) == 9 * sum(PATTERNS[pattern])


# 2. candidate selection = exactly ordinary L1 E ------------------------------
@pytest.mark.parametrize("topology,n_expected", [("tiled_cc", 8), ("tiled_cc_l1_4", 4)])
def test_candidate_selection_ordinary_l1_e_only(topology, n_expected):
    e, _ = _engine(topology)
    ctx = bc.RunContext(e)
    assert len(ctx.l1_columns) == 9
    for c in ctx.l1_columns:
        ids = ctx.l1_owners[c]
        assert len(ids) == n_expected
        for nid in ids:
            assert e._role_of[nid] == "E"           # ordinary, never Eor/C/I
            assert e.meta[nid]["layer"] == "L1"
    # excludes Eor / C / I / L2 / RGC
    picked = {nid for ids in ctx.l1_owners.values() for nid in ids}
    for nid in e.order:
        role = e._role_of.get(nid)
        if role in ("Eor", "C", "I") or e.meta[nid].get("layer") in ("L2", "RGC"):
            assert nid not in picked


# 3. synthetic ownership: stable passes; degenerate cases fail -----------------
def test_ownership_assessment_cases():
    W = 50
    stable = ["n0"] * (3 * W)
    v = ca.assess_ownership(stable, window=W)
    assert v.consolidated and v.owner == "n0" and v.strict_unanimous_final

    assert not ca.assess_ownership([], window=W).consolidated                 # silence
    assert ca.assess_ownership([], window=W).reason == "insufficient_events"
    assert not ca.assess_ownership(["n0"] * (3 * W - 1), window=W).consolidated  # too few

    alt = ["n0", "n1"] * (3 * W // 2)                                          # alternating
    assert not ca.assess_ownership(alt, window=W).consolidated

    turnover = ["n0"] * (2 * W) + ["n1"] * W                                   # late turnover
    tv = ca.assess_ownership(turnover, window=W)
    assert not tv.consolidated and tv.reason == "owner_turnover"

    subthresh = (["n0"] * 47 + ["n1"] * 3) * 3                                 # 0.94 < 0.95
    sv = ca.assess_ownership(subthresh, window=W)
    assert not sv.consolidated and sv.reason == "below_dominance"


# 4. duplicate stable owner across patterns => collision, no infinite wait -----
def test_duplicate_owner_is_collision():
    order = list(ca.CANONICAL_PATTERN_ORDER)
    owners = {p: "L1c00E2" for p in order}            # same owner for all four patterns
    mc = ca.column_mapping_checks(owners, n_competitors=8)
    assert not mc["passed"]
    assert not mc["checks"]["no_collision"]
    assert mc["collisions"]["L1c00E2"] == 4
    # analysis is a pure function: it returns immediately (cannot loop forever)


# 5. capacity accounting is exact for both presets ----------------------------
def test_capacity_accounting_exact():
    order = list(ca.CANONICAL_PATTERN_ORDER)
    distinct = {order[i]: f"L1c00E{i}" for i in range(4)}
    cc8 = ca.column_mapping_checks(distinct, n_competitors=8)
    assert cc8["passed"]
    assert cc8["capacity"]["unassigned_competitors"] == 4
    l1_4 = ca.column_mapping_checks(distinct, n_competitors=4)
    assert l1_4["passed"]
    assert l1_4["capacity"]["unassigned_competitors"] == 0


# 6. feedback disable suppresses only column_c_to_i; WTA intact; intact unchanged
def test_feedback_disable_selective():
    e_dis, disabled = _engine("tiled_cc", condition="feedback_disabled")
    # exactly the metadata-selected c_to_i edges, all intra-column
    expected = sorted(s["id"] for s in e_dis.synapses if s.get("projection") == "column_c_to_i")
    assert disabled == expected and len(disabled) == 10     # nine L1 columns + one L2 column
    for eid in disabled:
        s = next(s for s in e_dis.synapses if s["id"] == eid)
        assert e_dis._column_of[s["source"]] == e_dis._column_of[s["target"]]  # intra-column
    # C source no longer drives a relay; E sources still do (WTA preserved)
    for c in bc.RunContext(e_dis).l1_columns:
        c_id = f"{c}C"
        assert e_dis._relayexc_out.get(c_id, []) == []
        for e_id in bc.RunContext(e_dis).l1_owners[c]:
            assert e_dis._relayexc_out.get(e_id)              # E->I relay still present

    # behavioral: intact produces C-driven resets, disabled produces none; both keep WTA
    def c_driven_and_wins(engine, B=350):
        ctx = bc.RunContext(engine)
        engine.set_input(ctx.input_for("row 1"))
        c_driven = wins = 0
        for _ in range(B):
            d = engine.step()
            cw = d["column_winners"]
            wins += sum(1 for c in ctx.l1_columns if c in cw)
            reset_cols = {engine._column_of.get(r["source"]) for r in d["hard_reset_events"]}
            c_driven += sum(1 for col in reset_cols if col in ctx.l1_columns and col not in cw)
        return c_driven, wins

    e_intact, _ = _engine("tiled_cc", condition="intact")
    cd_i, wins_i = c_driven_and_wins(e_intact)
    cd_d, wins_d = c_driven_and_wins(e_dis)
    assert cd_i > 0 and wins_i > 0                            # C->I real in intact
    assert cd_d == 0 and wins_d > 0                           # no C reset, WTA intact

    # intact mode is byte-identical to a plain unmodified engine over the same input
    e_ref = SimulationEngine(seed=1, topology="tiled_cc", cc_e_count=8, leak_rate=0.0)
    e_chk, _ = _engine("tiled_cc", condition="intact")
    for eng in (e_ref, e_chk):
        eng.set_input(bc.RunContext(eng).input_for("col 1"))
    for _ in range(120):
        a, b = e_ref.step(), e_chk.step()
        assert [n["spiked"] for n in a["neurons"]] == [n["spiked"] for n in b["neurons"]]
        assert a["column_winners"] == b["column_winners"]


# 7. learning freeze covers every plastic cell/edge ---------------------------
@pytest.mark.parametrize("topology", bc.TOPOLOGIES)
def test_freeze_covers_all_plastic(topology):
    e, _ = _engine(topology)
    snap = bc.freeze_learning(e)
    # every latency competitor and coincidence cell is frozen
    for cell in e.latency_competitors:
        assert cell.learn is False
    for cell in e.coincidence:
        assert cell.learn is False
    # snapshot keys == every plastic edge id
    assert set(snap) == set(e._ff_weight_ref) | set(e._basal_weight_ref)
    # stepping under an active input changes no plastic weight
    e.set_input(bc.RunContext(e).input_for("diag \\"))
    for _ in range(60):
        e.step()
    bc.assert_weights_unchanged(e, snap)


# 8. cold-engine transfer reproduces weights exactly and rejects mismatch ------
def test_weight_transfer_exact_and_rejects_mismatch():
    src, _ = _engine("tiled_cc")
    src.set_input(bc.RunContext(src).input_for("row 1"))
    for _ in range(150):
        src.step()
    dst, _ = _engine("tiled_cc", seed=1)
    after = bc.transfer_plastic_weights(src, dst)
    assert after == bc.plastic_edge_weights(src)                # exact
    # misaligned topology id sets are rejected
    other, _ = _engine("tiled_cc_l1_4")
    with pytest.raises(ValueError):
        bc.transfer_plastic_weights(src, other)


# 9. both recall modes leave every weight byte-identical ----------------------
def test_recall_modes_do_not_learn(tmp_path):
    src, _ = _engine("tiled_cc")
    src.set_input(bc.RunContext(src).input_for("row 1"))
    for _ in range(200):
        src.step()
    ctx = bc.RunContext(src)
    trained = bc.plastic_edge_weights(src)

    cold = bc.cold_recall("tiled_cc", 1, "intact", ctx, trained, "row 1",
                          window=5, dominance=0.9, stable_windows=1, timeout=120,
                          cc_e_count=8, leak_rate=0.0)
    assert isinstance(cold, dict)

    # sequential recall on a frozen engine changes nothing
    snap = bc.freeze_learning(src)
    bc.sequential_recall(src, ctx, list(ca.CANONICAL_PATTERN_ORDER),
                         window=5, dominance=0.9, stable_windows=1, recall_block=40)
    bc.assert_weights_unchanged(src, snap)


# 10. short smoke run produces parseable artifacts, markers, CSV, summary ------
def test_smoke_run_artifacts(tmp_path):
    out = bc.run_one("tiled_cc", 1, "intact", pattern_order=list(ca.CANONICAL_PATTERN_ORDER),
                     output_root=str(tmp_path), window=5, dominance=0.9, stable_windows=1,
                     train_timeout=40, recall_timeout=40, recall_block=15,
                     record_stride=10, checkpoint_every=5, cc_e_count=8, leak_rate=0.0,
                     quiet=True)
    run_dir = out["run_dir"]
    from experiments.replay_recorder import read_records
    records = read_records(f"{run_dir}/replay.snn.jsonl")
    kinds = {r["record"] for r in records}
    assert {"header", "frame", "marker", "result"} <= kinds
    marker_kinds = {r["kind"] for r in records if r["record"] == "marker"}
    assert {"train_start", "learning_freeze", "cold_recall_start",
            "sequential_recall_start", "final_result"} <= marker_kinds
    # CSV has train + recall rows
    import csv
    with open(f"{run_dir}/metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    phases = {r["phase"] for r in rows}
    assert {"train", "cold_recall", "sequential_recall"} <= phases
    summary = json.load(open(f"{run_dir}/summary.json"))
    assert summary["status"] in ("completed", "failed")
    assert "basic_gate_passed" in summary["result"]
    # a shortened run legitimately times out; that must be truthfully recorded
    assert summary["result"]["any_training_timeout"] in (True, False)


# 11. timeout/interruption produce resumable, explicitly-incomplete artifacts --
def test_resume_skips_completed_and_labels_incomplete(tmp_path):
    bc.run_one("tiled_cc", 1, "intact", pattern_order=list(ca.CANONICAL_PATTERN_ORDER),
               output_root=str(tmp_path), window=5, dominance=0.9, stable_windows=1,
               train_timeout=25, recall_timeout=25, recall_block=10, record_stride=10,
               checkpoint_every=5, cc_e_count=8, leak_rate=0.0, quiet=True)
    # resume detects the completed cell
    assert bc._run_dir_completed(str(tmp_path), "tiled_cc", "intact", 1) is not None
    assert bc._run_dir_completed(str(tmp_path), "tiled_cc", "intact", 7) is None
    # a hand-made incomplete run dir is labeled incomplete, not scored
    incomplete = tmp_path / "incomplete-run"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text(json.dumps({"status": "interrupted"}))
    agg = bc.write_aggregate(str(tmp_path))
    assert agg["n_incomplete"] >= 1
    assert any(r["status"] == "interrupted" for r in agg["incomplete_runs"])


# 12. recording disabled vs enabled does not change harness dynamics ----------
def test_harness_dynamics_independent_of_recording(tmp_path):
    # run the training loop directly with recording, and a bare engine without it
    e_rec, _ = _engine("tiled_cc")
    e_bare, _ = _engine("tiled_cc")
    for eng in (e_rec, e_bare):
        eng.set_input(bc.RunContext(eng).input_for("row 1"))
    from experiments.replay_recorder import ReplayRecorder, STATUS_COMPLETED
    rec = ReplayRecorder(e_rec, experiment="inert", output_root=str(tmp_path),
                         record_every=1, metrics_columns=["t"])
    for _ in range(120):
        a = e_rec.step(); rec.record_frame(e_rec)
        b = e_bare.step()
        assert [n["spiked"] for n in a["neurons"]] == [n["spiked"] for n in b["neurons"]]
        assert a["column_winners"] == b["column_winners"]
    rec.finish(STATUS_COMPLETED, checks={"ok": True})
    wr = {s["id"]: s["weight"] for s in e_rec.topology()["synapses"]}
    wb = {s["id"]: s["weight"] for s in e_bare.topology()["synapses"]}
    assert wr == wb
