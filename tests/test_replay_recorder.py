"""Focused tests for the headless replay recorder and its versioned artifact contract.

These prove the recorder is a truthful, streaming, schema-valid, and behaviorally inert
recording layer. They use temporary directories only and never assert on dashboard winner
labels or visual styling.
"""

from __future__ import annotations

import json
import math

import pytest

from backend.simulation import SimulationEngine
from experiments import replay_recorder as rr
from experiments.replay_recorder import (
    ReplayRecorder, MetricsWriter, STATUS_COMPLETED, STATUS_INTERRUPTED,
    REPLAY_SCHEMA_NAME, REPLAY_SCHEMA_VERSION, REC_HEADER, REC_FRAME, REC_MARKER,
    REC_CHECKPOINT, REC_RESULT, reconstruct_weights_at, read_records, iter_records,
)


def _engine(seed=1):
    e = SimulationEngine(seed=seed, topology="tiled_cc", cc_e_count=8, leak_rate=0.0)
    e.set_patch(1, 1)
    e.set_pattern("row 1")
    return e


def _short_run(root, *, boundaries=30, record_every=1, checkpoint_every=10,
               metrics_columns=("timestep", "firing"), finish=True, engine=None):
    e = engine or _engine()
    rec = ReplayRecorder(e, experiment="unit", output_root=str(root),
                         record_every=record_every, checkpoint_every=checkpoint_every,
                         metrics_columns=list(metrics_columns))
    rec.set_annotation(phase="present", pattern="row 1")
    rec.marker("phase_start", data={"phase": "present"})
    for _ in range(boundaries):
        dyn = e.step()
        wrote = rec.record_frame(e)
        if wrote and "firing" in metrics_columns:
            rec.metrics.append_row({"timestep": e.timestep, "firing": int(dyn["stats"]["firing"])})
    if finish:
        rec.finish(STATUS_COMPLETED, checks={"ok": True}, result={"n": boundaries})
    return e, rec


# 1. all four artifacts with declared schema/version --------------------------
def test_creates_all_four_artifacts(tmp_path):
    e, rec = _short_run(tmp_path)
    for name in ("manifest.json", "replay.snn.jsonl", "metrics.csv", "summary.json"):
        assert (tmp_path / rec.run_id / name).exists(), name
    manifest = json.loads((tmp_path / rec.run_id / "manifest.json").read_text())
    assert manifest["replay_schema"] == REPLAY_SCHEMA_NAME
    assert manifest["replay_schema_version"] == REPLAY_SCHEMA_VERSION
    assert manifest["status"] == STATUS_COMPLETED
    assert manifest["seed"] == 1
    assert manifest["engine_params"]["topology"] == "tiled_cc"


# 2. header is first, self-contained, carries a valid topology ----------------
def test_header_is_first_and_self_contained(tmp_path):
    e, rec = _short_run(tmp_path)
    records = read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
    assert records[0]["record"] == REC_HEADER
    assert sum(1 for r in records if r["record"] == REC_HEADER) == 1
    hdr = records[0]
    assert hdr["schema"] == REPLAY_SCHEMA_NAME and hdr["schema_version"] == REPLAY_SCHEMA_VERSION
    topo = hdr["topology"]
    assert topo["neurons"] and topo["synapses"] and topo["params"]
    assert hdr["neuron_order"] == [n["id"] for n in topo["neurons"]]
    assert hdr["synapse_order"] == [s["id"] for s in topo["synapses"]]
    # initial live weights are present in the header topology
    assert any(s.get("weight") is not None for s in topo["synapses"])


# 3. monotonic frames + dashboard-required dynamic fields ---------------------
def test_frames_monotonic_and_dynamic_fields(tmp_path):
    e, rec = _short_run(tmp_path, boundaries=25)
    frames = [r for r in read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
              if r["record"] == REC_FRAME]
    idx = [f["frame_index"] for f in frames]
    ts = [f["timestep"] for f in frames]
    assert idx == list(range(len(frames)))
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    required = {"neurons", "input", "winner", "column_winners", "changed_synapses",
                "emitted", "inhibitory_pulses", "hard_reset_events", "latency_ties",
                "stats", "log", "timestep"}
    for f in frames:
        assert required <= set(f["dynamic"]), required - set(f["dynamic"])
        assert isinstance(f["dynamic"]["neurons"], list) and f["dynamic"]["neurons"]
        assert f["record_every"] == 1


# 4. markers preserve caller annotations --------------------------------------
def test_markers_preserve_annotations(tmp_path):
    e = _engine()
    rec = ReplayRecorder(e, experiment="unit", output_root=str(tmp_path))
    rec.set_annotation(phase="train", pattern="col 2", tags=["a"], notes="hello")
    e.step()
    rec.record_frame(e)
    rec.marker("pattern_change", data={"from": "row 1", "to": "col 2"})
    rec.finish(STATUS_COMPLETED, checks={"ok": True})
    markers = [r for r in read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
               if r["record"] == REC_MARKER]
    assert len(markers) == 1
    m = markers[0]
    assert m["kind"] == "pattern_change"
    assert m["data"] == {"from": "row 1", "to": "col 2"}
    assert m["annotation"] == {"phase": "train", "pattern": "col 2",
                               "tags": ["a"], "notes": "hello"}


# 5. weight reconstruction (forward + random/backward access) -----------------
def test_weight_reconstruction_and_random_access(tmp_path):
    # record_every=1 so every changed_synapses is captured. Between checkpoints the
    # authority is the production 4-dp ``changed_synapses`` payload (unchanged here);
    # reconstruction therefore reproduces the live weights to that recorded precision.
    e = _engine()
    rec = ReplayRecorder(e, experiment="unit", output_root=str(tmp_path),
                         record_every=1, checkpoint_every=7)
    live_by_frame = {}
    for _ in range(40):
        e.step()
        rec.record_frame(e)
        live_by_frame[rec.frames_written - 1] = {
            edge["id"]: float(e._live_weight(edge))
            for edge in e.synapses if e._live_weight(edge) is not None}
    rec.finish(STATUS_COMPLETED, checks={"ok": True})

    records = read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
    # random / backward access order
    for target in (39, 0, 20, 6, 7, 13, 5):
        got = reconstruct_weights_at(records, target)
        live = live_by_frame[target]
        assert set(got) == set(live), f"frame {target} key mismatch"
        for k, v in live.items():
            assert got[k] == pytest.approx(v, abs=1e-4), f"frame {target} synapse {k}"


# 6. record_every honored and skipped timesteps represented honestly ----------
def test_record_every_is_honored(tmp_path):
    e, rec = _short_run(tmp_path, boundaries=20, record_every=3, checkpoint_every=100,
                        metrics_columns=("timestep", "firing"))
    frames = [r for r in read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
              if r["record"] == REC_FRAME]
    # 20 steps, stride 3 -> frames at observed calls 1,4,7,... -> ceil(20/3)=7
    assert len(frames) == 7
    ts = [f["timestep"] for f in frames]
    assert ts == [1, 4, 7, 10, 13, 16, 19]      # actual engine timesteps, gaps visible
    assert all(f["record_every"] == 3 for f in frames)


# 7. CSV round-trips with commas and quoted text ------------------------------
def test_csv_quoting_round_trip(tmp_path):
    path = tmp_path / "m.csv"
    w = MetricsWriter(str(path), columns=["a", "text", "n"], optional_columns=["n"])
    w.append_row({"a": 1, "text": 'has, comma and "quote"', "n": 2})
    w.append_row({"a": 2, "text": "line\nbreak"})     # optional n omitted
    w.close()
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["text"] == 'has, comma and "quote"'
    assert rows[0]["n"] == "2"
    assert rows[1]["text"] == "line\nbreak"
    assert rows[1]["n"] == ""


# 8. interruption leaves parseable JSONL and marks status; frames survive -----
def test_interruption_preserves_frames_and_marks_status(tmp_path):
    e = _engine()
    run_id = None
    with pytest.raises(KeyboardInterrupt):
        with ReplayRecorder(e, experiment="unit", output_root=str(tmp_path)) as rec:
            run_id = rec.run_id
            for i in range(10):
                e.step()
                rec.record_frame(e)
                if i == 6:
                    raise KeyboardInterrupt("simulated stop")
    path = tmp_path / run_id / "replay.snn.jsonl"
    records = read_records(path)                 # must parse cleanly
    frames = [r for r in records if r["record"] == REC_FRAME]
    assert len(frames) == 7                       # frames 0..6 preserved, none erased
    assert records[-1]["record"] == REC_RESULT
    manifest = json.loads((tmp_path / run_id / "manifest.json").read_text())
    summary = json.loads((tmp_path / run_id / "summary.json").read_text())
    assert manifest["status"] == STATUS_INTERRUPTED
    assert summary["status"] == STATUS_INTERRUPTED
    assert "KeyboardInterrupt" in summary["failure_reason"]


# 9. non-finite values and schema/ID mismatches fail clearly ------------------
def test_non_finite_and_mismatch_fail(tmp_path):
    # non-finite metric value
    w = MetricsWriter(str(tmp_path / "m.csv"), columns=["x"])
    with pytest.raises(ValueError):
        w.append_row({"x": math.inf})
    # unknown / missing columns
    with pytest.raises(ValueError):
        w.append_row({"y": 1})
    w2 = MetricsWriter(str(tmp_path / "m2.csv"), columns=["a", "b"])
    with pytest.raises(ValueError):
        w2.append_row({"a": 1})                  # missing required b
    w.close(); w2.close()
    # non-finite inside a replay record (marker data)
    e = _engine()
    rec = ReplayRecorder(e, experiment="unit", output_root=str(tmp_path))
    with pytest.raises(ValueError):
        rec.marker("bad", data={"v": float("nan")})
    rec.finish(STATUS_COMPLETED, checks={"ok": True})
    # reconstruction with a bad frame index / no header
    records = read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
    with pytest.raises(ValueError):
        reconstruct_weights_at(records, 9999)
    with pytest.raises(ValueError):
        rr.header_initial_weights([{"record": REC_FRAME, "frame_index": 0}])
    # bad recorder configuration
    with pytest.raises(ValueError):
        ReplayRecorder(e, experiment="x", output_root=str(tmp_path), record_every=0)
    with pytest.raises(ValueError):
        ReplayRecorder(e, experiment="x", output_root=str(tmp_path),
                       hierarchical_feedback="maybe")


# 10. recording is behaviorally inert ----------------------------------------
def test_recording_is_behaviorally_inert(tmp_path):
    e_rec = _engine(seed=1)
    e_ref = _engine(seed=1)
    rec = ReplayRecorder(e_rec, experiment="unit", output_root=str(tmp_path),
                         record_every=1, checkpoint_every=5)
    for _ in range(60):
        d_rec = e_rec.step()
        rec.record_frame(e_rec)
        rec.checkpoint(e_rec)                     # extra reads must also be inert
        d_ref = e_ref.step()
        # identical spikes, winners, inputs, stats every step
        assert [n["spiked"] for n in d_rec["neurons"]] == [n["spiked"] for n in d_ref["neurons"]]
        assert d_rec["winner"] == d_ref["winner"]
        assert d_rec["column_winners"] == d_ref["column_winners"]
        assert d_rec["stats"] == d_ref["stats"]
    rec.finish(STATUS_COMPLETED, checks={"ok": True})
    # identical final live weights and neuron potentials
    wr = {s["id"]: s["weight"] for s in e_rec.topology()["synapses"]}
    wf = {s["id"]: s["weight"] for s in e_ref.topology()["synapses"]}
    assert wr == wf
    pr = [n["potential"] for n in e_rec.dynamic_state()["neurons"]]
    pf = [n["potential"] for n in e_ref.dynamic_state()["neurons"]]
    assert pr == pf


# 11. bounded memory: recorder does not retain frames ------------------------
def _container_footprint(rec):
    total = 0
    for v in vars(rec).values():
        if isinstance(v, (list, dict, set, tuple)):
            total += len(v)
    return total


def test_recorder_memory_is_bounded(tmp_path):
    e_small = _engine()
    rec_small = ReplayRecorder(e_small, experiment="unit", output_root=str(tmp_path),
                               metrics_columns=["timestep"])
    for _ in range(10):
        e_small.step(); rec_small.record_frame(e_small)
        rec_small.metrics.append_row({"timestep": e_small.timestep})
    small_footprint = _container_footprint(rec_small)
    rec_small.finish(STATUS_COMPLETED, checks={"ok": True})

    e_big = _engine()
    rec_big = ReplayRecorder(e_big, experiment="unit", output_root=str(tmp_path),
                             metrics_columns=["timestep"])
    for _ in range(400):
        e_big.step(); rec_big.record_frame(e_big)
        rec_big.metrics.append_row({"timestep": e_big.timestep})
    big_footprint = _container_footprint(rec_big)
    rec_big.finish(STATUS_COMPLETED, checks={"ok": True})

    assert rec_big.frames_written == 400
    # 40x more frames must not grow retained container state.
    assert big_footprint == small_footprint


# checkpoint reconstruction is exact even with a coarse stride (via checkpoints)
def test_checkpoint_frame_reconstruction_exact_with_stride(tmp_path):
    e = _engine()
    rec = ReplayRecorder(e, experiment="unit", output_root=str(tmp_path),
                         record_every=1, checkpoint_every=5)
    live = {}
    for _ in range(30):
        e.step()
        rec.record_frame(e)
        live[rec.frames_written - 1] = {edge["id"]: round(float(e._live_weight(edge)), 6)
                                        for edge in e.synapses if e._live_weight(edge) is not None}
    rec.finish(STATUS_COMPLETED, checks={"ok": True})
    records = read_records(tmp_path / rec.run_id / "replay.snn.jsonl")
    # frames that coincide with checkpoints reconstruct exactly from the checkpoint alone
    for target in (4, 9, 14, 19, 24, 29):
        assert reconstruct_weights_at(records, target) == live[target]
