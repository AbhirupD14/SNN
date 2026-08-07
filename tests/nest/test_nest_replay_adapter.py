"""NEST -> `snn.replay` adapter tests (dashboard prompt sections 6, 7).

These assert the two things the adapter exists to guarantee: that the artifact is a
faithful, losslessly-identified copy of what NEST recorded, and that everything NEST did
NOT record is reported as unavailable rather than as zero.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from experiments.replay_recorder import (
    REPLAY_SCHEMA_NAME,
    REPLAY_SCHEMA_VERSION,
    header_initial_weights,
    read_records,
    reconstruct_weights_at,
)
from nest_backend.engine import Stimulus, run_case
from nest_backend.replay_adapter import (
    AdapterError,
    build_records,
    to_ms,
    to_tick,
    write_replay,
)
from nest_backend.topology import Timescales

CENTER = Stimulus({(1, 1): "row 1"})


@pytest.fixture(scope="module")
def result():
    return run_case("01_center_patch_row", CENTER, n_presentations=6)


@pytest.fixture(scope="module")
def records(result):
    return build_records(result)


@pytest.fixture(scope="module")
def parts(records):
    header = records[0]
    frames = [r for r in records if r["record"] == "frame"]
    markers = [r for r in records if r["record"] == "marker"]
    final = [r for r in records if r["record"] == "result"]
    return header, frames, markers, final


# ------------------------------------------------------------------ time mapping


@pytest.mark.parametrize("h", [0.05, 0.1, 0.2])
def test_tick_conversion_round_trips(h):
    for k in range(0, 200):
        t_ms = round(k * h, 10)
        assert to_tick(t_ms, h) == k
        assert to_ms(to_tick(t_ms, h), h) == pytest.approx(t_ms)


@pytest.mark.parametrize("h", [0.05, 0.1, 0.2])
def test_off_grid_timestamp_fails_loudly(h):
    """A silently rounded timestamp would move a spike into a neighbouring frame."""
    with pytest.raises(AdapterError, match="not on the"):
        to_tick(h * 3.5, h)


def test_non_finite_timestamp_is_rejected():
    with pytest.raises(AdapterError):
        to_tick(float("inf"), 0.1)


# -------------------------------------------------------------------- structure


def test_header_carries_the_full_canonical_topology(parts):
    header, _frames, _markers, _final = parts
    assert header["schema"] == REPLAY_SCHEMA_NAME
    assert header["schema_version"] == REPLAY_SCHEMA_VERSION
    topo = header["topology"]
    assert len(topo["neurons"]) == 191
    assert len(topo["synapses"]) == 1052
    assert len(header["neuron_order"]) == 191
    assert len(header["synapse_order"]) == 1052


def test_repository_ids_round_trip_exactly(parts):
    """The dashboard selects by repository id; a NEST GID must never become identity."""
    from backend.network_spec import tiled_cc_spec

    header, frames, _markers, _final = parts
    spec = tiled_cc_spec(cc_e_count=8)
    expected_nodes = {n["id"] for n in spec["nodes"]}
    expected_edges = {e["id"] for e in spec["edges"]}

    assert {n["id"] for n in header["topology"]["neurons"]} == expected_nodes
    assert {s["id"] for s in header["topology"]["synapses"]} == expected_edges
    for frame in frames:
        assert {n["id"] for n in frame["dynamic"]["neurons"]} == expected_nodes


def test_nest_gids_are_provenance_only_not_identity(parts, result):
    header, _frames, _markers, _final = parts
    id_map = result.manifest["id_map"]
    assert len(set(id_map.values())) == len(id_map), "GID mapping must be injective"
    # No frame or topology record may key anything by a GID.
    for node in header["topology"]["neurons"]:
        assert not str(node["id"]).isdigit()


def test_node_metadata_needed_by_the_dashboard_survives(parts):
    header, _f, _m, _r = parts
    by_id = {n["id"]: n for n in header["topology"]["neurons"]}
    e_cell = by_id["L1c11E0"]
    for field in ("layer", "type", "role", "archetype", "pos", "column_id", "column_role"):
        assert field in e_cell, f"{field} missing from node metadata"
    assert by_id["L2c00C"]["has_parent"] is False, "the dormant C must stay identifiable"
    assert "tiling" in header["topology"], "tiled metadata drives column grouping"
    assert header["topology"]["grid"]["rows"] == 9


def test_edges_carry_kind_projection_and_realized_delay(parts):
    header, _f, _m, _r = parts
    by_id = {s["id"]: s for s in header["topology"]["synapses"]}
    sample = by_id["L1c00_eor_c"]
    assert sample["kind"] == "basal_excitation"
    assert sample["projection"] == "column_eor_to_c_basal"
    assert sample["nest_delay_ms"] is not None
    assert all(s.get("nest_delay_ms") is not None for s in header["topology"]["synapses"])


# ---------------------------------------------------------------------- weights


def test_weighted_edges_carry_frozen_nest_weights(parts, result):
    header, _f, _m, _r = parts
    weighted = [s for s in header["topology"]["synapses"] if s["weight"] is not None]
    assert len(weighted) == 810, "800 feedforward + 10 basal edges are weighted"
    eor = [s for s in header["topology"]["synapses"]
           if s.get("projection") == "column_e_to_eor"]
    assert all(s["weight"] == pytest.approx(1000.0) for s in eor), (
        "Eor is frozen at theta so it relays on a single afferent"
    )


def test_unweighted_edge_kinds_stay_null_not_zero(parts):
    """A placeholder NEST delivers on an unweighted connection is not a weight."""
    header, _f, _m, _r = parts
    for synapse in header["topology"]["synapses"]:
        if synapse["kind"] in ("apical_excitation", "hard_reset_inhibition"):
            assert synapse["weight"] is None, (
                f"{synapse['id']} is unweighted in the reference model and must not "
                "report the NEST delivery placeholder as a weight"
            )


def test_frozen_runner_emits_no_weight_changes(parts):
    _header, frames, _m, _r = parts
    for frame in frames:
        assert frame["dynamic"]["changed_synapses"] == []


def test_learning_runner_records_updates_and_exports_materialized_weights():
    """The ordinary runner must not silently turn ``learning=True`` into frozen output."""
    learning = run_case(
        "learning_artifact_probe",
        Stimulus({(0, 0): "row 1"}),
        shape=(3, 3),
        n_presentations=4,
        learning=True,
    )

    assert learning.manifest["learning"] is True
    assert learning.manifest["weight_state"] == {
        "initial_source": "reference engine deterministic initialization",
        "available_snapshots": ["initial", "materialized_current"],
        "dashboard_default_snapshot": "materialized_current",
        "updates_recorded": True,
        "logical_weights_flushed": False,
    }
    assert learning.weight_changes, "a firing learning run must carry recorder updates"
    assert learning.topology["nest"]["weight_state"] == "initial"

    # The header's PROSE must name the same snapshot as its `weight_state`. This payload is
    # the pre-run baseline; describing it as materialized is the audit's mislabel with the
    # two numbers swapped, and the prose is what a dashboard reader actually sees.
    nest_block = learning.topology["nest"]
    assert "active" in nest_block["learning"]
    assert "materialized" not in nest_block["learning"], (
        f"header topology holds {nest_block['weight_state']!r} weights but says "
        f"{nest_block['learning']!r}"
    )
    assert not any("materialized" in note for note in nest_block["known_differences"]
                   if "topology weights" in note or "weights are" in note), (
        "known_differences claims a materialized snapshot for an initial-weight payload"
    )

    by_edge = {synapse["id"]: synapse for synapse in learning.topology["synapses"]}
    changed_edge = learning.weight_changes[-1][1]
    exported = by_edge[changed_edge]
    assert exported["weight_state"] == "initial"
    assert "initial_weight" in exported

    records = build_records(learning)
    header = records[0]
    assert header["conditions"]["learning"] == "recorded"
    assert header["nest"]["learning_mode"] == "recorded"
    assert header["nest"]["weight_state"]["logical_weights_flushed"] is False
    assert any(
        frame["dynamic"]["changed_synapses"]
        for frame in records if frame["record"] == "frame"
    )
    assert all(
        "learning" in frame["annotation"]["tags"]
        for frame in records if frame["record"] == "frame"
    )

    final_record = records[-1]
    weights = final_record["measurements"]["weights"]
    assert weights["state"] == "materialized_final"
    assert weights["header_weight_state"] == "initial"
    assert weights["values"] == learning.final_weights

    # Reconstruction has to CROSS a real distance, or the assertion below would hold on a
    # run in which nothing learned. Some plastic edges never materialize (their afferent
    # went quiet); those legitimately end where they started.
    header_weight = {synapse["id"]: synapse["weight"]
                     for synapse in learning.topology["synapses"]}
    moved = [edge for edge, final in learning.final_weights.items()
             if abs(final - header_weight[edge]) > 1e-3]
    assert moved, "no plastic weight moved: the reconstruction check would be vacuous"

    frames = [record for record in records if record["record"] == "frame"]
    reconstructed = reconstruct_weights_at(records, len(frames) - 1)
    for edge_id, final_weight in learning.final_weights.items():
        assert reconstructed[edge_id] == pytest.approx(final_weight, abs=1e-6)


def test_a_learning_run_without_captured_finals_is_not_labelled_frozen():
    """A learning artifact that carries no final snapshot must say so.

    The saturation DEMO pass is exactly this shape: its header topology is a
    materialized-current snapshot of the trained network and it attaches neither recorder
    deltas nor a separate final set. Labelling that artifact's weight block
    ``initial_frozen`` states the opposite of both halves -- the weights are neither
    initial nor frozen.
    """
    learning = run_case(
        "learning_without_finals",
        Stimulus({(0, 0): "row 1"}),
        shape=(3, 3),
        n_presentations=2,
        learning=True,
    )
    no_finals = dataclasses.replace(learning, weight_changes=[], final_weights={})

    weights = build_records(no_finals)[-1]["measurements"]["weights"]
    assert weights["state"] == "not_captured", (
        f"a learning run with no captured finals reports {weights['state']!r}"
    )
    assert weights["values"] is None
    assert weights["header_weight_state"] == "initial"


# ------------------------------------------------------------------- honesty


def test_unsampled_state_is_absent_never_zero(parts):
    """The core invariant: unavailable is not zero, and not a repeated final value.

    Unsampled fields are omitted, not sent as 0. Reading an absent field yields
    `undefined`, which every consumer treats as unknown through `!= null` -- the same path
    an explicit null takes. What must never appear is a numeric value.
    """
    header, frames, _m, _r = parts
    declared = header["recording"]["state_availability"]
    for frame in frames:
        for node in frame["dynamic"]["neurons"]:
            # Membrane state needs a multimeter; this run had none, so it must be ABSENT.
            for field in ("potential", "activation", "refractory"):
                assert declared[field] == "unavailable"
                assert node.get(field) is None, (
                    f"{field} must be unavailable, got {node.get(field)!r}"
                )


def test_freq_is_derived_from_the_spike_record_not_sampled(parts):
    """`freq` is DERIVED, so unlike membrane state it is present without a multimeter.

    It is a count over the recorded spike train against a declared window, which makes 0.0
    a real measurement ("this cell did not fire recently") rather than a stand-in for
    unknown. The distinction is carried in `state_availability`, not left to the reader.
    """
    header, frames, _m, _r = parts
    assert header["recording"]["state_availability"]["freq"] == "derived"

    seen = [n["freq"] for f in frames for n in f["dynamic"]["neurons"] if "freq" in n]
    assert seen, "freq must be present on a run that declares a presentation period"
    assert all(0.0 <= v <= 1.0 for v in seen), "freq is a normalised fraction"
    assert any(v > 0.0 for v in seen), "a run with spikes must show a non-zero rate"


def test_every_node_appears_in_every_frame(parts):
    """A node omitted from a frame would retain the previous frame's spiked flag."""
    _header, frames, _m, _r = parts
    for frame in frames:
        assert len(frame["dynamic"]["neurons"]) == 191


def test_availability_is_declared_on_header_and_frames(parts):
    header, frames, _m, _r = parts
    declared = header["recording"]["state_availability"]
    assert declared["spiked"] == "recorded"
    assert declared["emitted"] == "derived"
    assert declared["potential"] == "unavailable"
    assert "not zero" in header["recording"]["unavailable_note"]
    for frame in frames:
        assert frame["dynamic"]["nest"]["state_availability"] == declared


def test_spikes_are_preserved_exactly(parts, result):
    """Cortical spikes AND retinal input events both reach the `spiked` flag.

    They arrive on two different NEST recorders -- generators are devices, model neurons
    are not -- but both are firing cells as far as the renderer is concerned.
    """
    _header, frames, _m, _r = parts
    recorded = sorted(
        [(round(s["t"], 6), s["id"]) for s in result.metrics["spikes"]]
        + [(round(e["t"], 6), e["id"]) for e in result.metrics["input_schedule"]]
    )
    replayed = []
    for frame in frames:
        t_ms = frame["dynamic"]["nest"]["t_ms"]
        for node in frame["dynamic"]["neurons"]:
            if node["spiked"]:
                replayed.append((round(t_ms, 6), node["id"]))
    assert sorted(replayed) == recorded, "every recorded event appears exactly once"


def test_rgc_sources_light_up_on_their_input_events(parts, result):
    """Regression: the 9x9 input surface must not stay dark.

    RGC events are recorded on a separate NEST recorder from cortical spikes, and an
    earlier adapter set `spiked` only from the cortical one -- so every retinal cell
    stayed dark in the renderer despite its events being present in the artifact.
    """
    header, frames, _m, _r = parts
    rgc = {n["id"] for n in header["topology"]["neurons"]
           if n.get("archetype") == "rg_source"}
    assert len(rgc) == 81

    lit = {n["id"] for f in frames for n in f["dynamic"]["neurons"]
           if n["spiked"] and n["id"] in rgc}
    expected = {e["id"] for e in result.metrics["input_schedule"]}
    assert lit == expected, "every RGC that emitted must be flagged as spiked"
    assert lit, "the driven patch must produce lit RGC cells"


def test_rgc_spikes_pulse_their_feedforward_edges(parts, result):
    """A lit RGC must also pulse its rg_to_column edges, or the input looks disconnected."""
    _header, frames, _m, _r = parts
    outgoing = result.outgoing_edges
    for frame in frames:
        fired = {n["id"] for n in frame["dynamic"]["neurons"] if n["spiked"]}
        expected = sorted({e for nid in fired for e in outgoing.get(nid, ())})
        assert frame["dynamic"]["emitted"] == expected


def test_simultaneous_winners_are_never_collapsed(parts):
    """Invariant 3: multiple winners remain multiple winners."""
    _header, frames, _m, _r = parts
    multi = 0
    for frame in frames:
        for _column, winners in frame["dynamic"]["nest"]["column_winners"].items():
            if len(winners) > 1:
                multi += 1
                fired = {n["id"] for n in frame["dynamic"]["neurons"] if n["spiked"]}
                assert set(winners) <= fired, "every winner keeps its own spiked flag"
    assert multi > 0, (
        "the default NEST configuration produces multi-winner windows; if this ever "
        "becomes zero the WTA finding has changed"
    )


def test_emitted_edges_are_exactly_the_outgoing_edges_of_spiking_nodes(parts, result):
    """The one derived field, checked against its documented exact rule."""
    _header, frames, _m, _r = parts
    outgoing = result.outgoing_edges
    for frame in frames:
        fired = {n["id"] for n in frame["dynamic"]["neurons"] if n["spiked"]}
        expected = sorted({e for nid in fired for e in outgoing.get(nid, ())})
        assert frame["dynamic"]["emitted"] == expected


# ------------------------------------------------------------------- ordering


def test_frames_are_strictly_monotonic_in_index_and_tick(parts):
    """The schema-1 parser rejects a non-monotonic timestep, so merging must be complete."""
    _header, frames, _m, _r = parts
    indices = [f["frame_index"] for f in frames]
    ticks = [f["timestep"] for f in frames]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks)


def test_same_timestamp_events_share_one_frame(parts, result):
    _header, frames, _m, _r = parts
    h = result.manifest["timescales"]["h_ms"]
    ticks_from_spikes = {to_tick(float(s["t"]), h) for s in result.metrics["spikes"]}
    frame_ticks = {f["timestep"] for f in frames}
    assert ticks_from_spikes <= frame_ticks


def test_conversion_is_deterministic_for_the_same_artifact(result):
    a = json.dumps(build_records(result), sort_keys=True)
    b = json.dumps(build_records(result), sort_keys=True)
    assert a == b


# ---------------------------------------------------------------- provenance


def test_provenance_covers_everything_required(parts):
    header, _f, _m, _r = parts
    prov = header["nest"]
    for field in ("nest_version", "nestml_version", "python_version", "git", "spec_hash",
                  "seed", "resolution_h_ms", "L_wta_ms", "S_target_ms", "D_ms",
                  "threads", "mpi_ranks", "dispersion_enabled", "feedback_enabled",
                  "learning_mode", "adapter_version", "case"):
        assert field in prov, f"provenance is missing {field}"
    assert prov["learning_mode"] == "frozen"
    assert prov["mpi_available"] is False


def test_validity_envelope_is_carried_into_the_artifact(parts):
    header, _f, _m, _r = parts
    envelope = header["topology"]["nest"]["validity_envelope"]
    assert envelope["leak_rate"] == 0.0
    assert envelope["persistent_inhibitory_conductance"] is False
    known = header["topology"]["nest"]["known_differences"]
    assert any("MULTIPLE ordinary-E winners" in d for d in known)


def test_presentation_markers_are_recorded(parts):
    _header, _frames, markers, _r = parts
    assert markers, "each presentation must be markable in the transport"
    assert all(m["kind"] in ("presentation", "pattern_switch") for m in markers)


def test_final_result_record_carries_the_case_summary(parts, result):
    _header, _frames, _m, final = parts
    assert len(final) == 1
    assert final[0]["measurements"]["spike_count"] == result.metrics["spike_count"]


# ------------------------------------------------------------------ round-trip


def test_written_artifact_parses_with_the_python_reader(result, tmp_path):
    path = tmp_path / "replay.snn.jsonl"
    info = write_replay(result, path)
    records = read_records(str(path))
    assert len(records) == info["records"]

    initial = header_initial_weights(records)
    assert len(initial) == 810
    frames = [r for r in records if r["record"] == "frame"]
    final_weights = reconstruct_weights_at(records, frames[-1]["frame_index"])
    assert final_weights == initial, "frozen learning: weights never move"


def test_unknown_repository_id_fails_loudly(result):
    import copy

    broken = copy.deepcopy(result)
    broken.metrics["spikes"] = list(broken.metrics["spikes"]) + [{"t": 25.0, "id": "NOPE"}]
    with pytest.raises(AdapterError, match="unknown repository node id"):
        build_records(broken)


def test_artifact_stays_small_enough_for_browser_replay(result, tmp_path):
    info = write_replay(result, tmp_path / "r.snn.jsonl")
    assert info["bytes"] < 8_000_000, (
        f"{info['bytes']} bytes over {info['frames']} frames is too large for the browser "
        "player; frame density or per-frame payload needs revisiting"
    )


# --------------------------------------------------- scientific noninterference


def test_conversion_does_not_change_the_run(result):
    """Invariant 7: visualization never changes NEST results.

    The adapter is a pure reader, so the check is that converting an artifact leaves the
    artifact identical -- and that a second identical run still matches after a conversion
    has happened in between.
    """
    before = json.dumps(result.metrics, sort_keys=True, default=str)
    build_records(result)
    after = json.dumps(result.metrics, sort_keys=True, default=str)
    assert before == after

    fresh = run_case("01_center_patch_row", CENTER, n_presentations=6)
    assert fresh.metrics["spikes"] == result.metrics["spikes"]
    assert fresh.metrics["winner_multiplicity_overall"] == \
        result.metrics["winner_multiplicity_overall"]


def test_recording_a_run_matches_an_unrecorded_run():
    """Same seed and schedule, with and without producing a replay: identical results."""
    plain = run_case("plain", CENTER, n_presentations=6, timescales=Timescales())
    recorded = run_case("recorded", CENTER, n_presentations=6, timescales=Timescales())
    build_records(recorded)

    assert plain.metrics["spikes"] == recorded.metrics["spikes"]
    assert plain.metrics["by_column"] == recorded.metrics["by_column"]
    assert plain.metrics["winner_multiplicity"] == recorded.metrics["winner_multiplicity"]
    assert plain.metrics["coincidence"] == recorded.metrics["coincidence"]
    assert plain.manifest["timescales"] == recorded.manifest["timescales"]
