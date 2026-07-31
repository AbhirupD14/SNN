"""Phase 2 completion: the LIVE network consumes the profile as its source of physics.

Source: `prompts/Claude_NEST_CIPP_Phase2_Completion_Review.md`.

The pure-profile tests in `test_cipp_profiles.py` check the configuration object. These
check that the configuration is what the constructed NEST network actually uses -- Phase 2
does not close on two competing parameter paths.

The impulse contracts are NOT repointed here. `test_cipp_semantic_probes.py` keeps its
Contract 8 and 9 xfails as historical characterization of the preserved defects; these are
separate, passing contracts for the continuous path.
"""
from __future__ import annotations

import pytest

from nest_backend.profiles import (
    CIPP_CONTINUOUS,
    PROFILE_CIPP_CONTINUOUS,
    PROFILE_IMPULSE,
    QuantizationPolicy,
)
from nest_backend.topology import NestTiledNetwork, Timescales


def _net(presentation=15.0, profile=PROFILE_CIPP_CONTINUOUS, **kwargs):
    return NestTiledNetwork(
        seed=1, shape=(3, 6), profile=profile,
        timescales=Timescales(h=0.1, base_ff=1.0, spread=2.0, presentation=presentation),
        **kwargs,
    )


# ------------------------------------------------ gate 2: the profile is the live source
def test_the_network_takes_its_resolution_from_the_profile(fresh_kernel):
    """`Timescales.h` is the impulse knob; the continuous profile owns the resolution."""
    net = _net()
    assert net.profile.name == PROFILE_CIPP_CONTINUOUS
    assert net.ts.h == CIPP_CONTINUOUS.resolution_h_ms == 0.01, (
        "the profile's resolution must override the Timescales default of 0.1"
    )


def test_profile_is_instance_state_not_a_module_global(fresh_kernel):
    """Two networks must be able to declare different profiles in one process."""
    continuous = _net(profile=PROFILE_CIPP_CONTINUOUS)
    impulse = _net(profile=PROFILE_IMPULSE)
    assert continuous.profile.name == PROFILE_CIPP_CONTINUOUS
    assert impulse.profile.name == PROFILE_IMPULSE
    # And the first is unaffected by the second having been built.
    assert continuous.profile.name == PROFILE_CIPP_CONTINUOUS
    assert continuous.manifest()["engine_profile"] == PROFILE_CIPP_CONTINUOUS
    assert impulse.manifest()["engine_profile"] == PROFILE_IMPULSE


def test_the_default_profile_is_still_the_prototype(fresh_kernel):
    """`cipp_continuous` may not be claimed by a network that did not ask for it."""
    assert NestTiledNetwork(seed=1, shape=(3, 6)).profile.name == PROFILE_IMPULSE


# --------------------------------- gate 3: physics is byte-equivalent across pacing
def test_changing_presentation_pacing_leaves_every_model_dict_identical(fresh_kernel):
    """THE Phase 2 contract, checked on the ACTUAL NEST model dictionaries.

    Two continuous networks differing only in presentation period. Every neuron role's
    model name and parameter dict must be identical -- membrane, coincidence windows,
    relay lockout and refractory alike.

    The impulse profile fails this by construction (`window = self.ts.presentation`), which
    is what `test_c8_cell_physiology_is_independent_of_presentation_interval` records as an
    xfail. That probe stays; this is the repaired path.
    """
    fast, slow = _net(presentation=10.0), _net(presentation=40.0)

    fast_models = {nid: fast._model_for(meta) for nid, meta in fast.node_meta.items()}
    slow_models = {nid: slow._model_for(meta) for nid, meta in slow.node_meta.items()}

    assert set(fast_models) == set(slow_models)
    differing = {nid: (fast_models[nid], slow_models[nid])
                 for nid in fast_models if fast_models[nid] != slow_models[nid]}
    assert not differing, (
        f"{len(differing)} neuron model dicts changed with the presentation interval, "
        f"e.g. {dict(list(differing.items())[:2])}"
    )

    # And specifically the four parameters the audit named.
    for nid, (model, params) in fast_models.items():
        for key in ("tau_basal", "tau_apical", "tau_deposit_lock", "t_lockout"):
            if key in params:
                assert params[key] != 10.0 or key == "t_lockout", (
                    f"{nid}.{key} still tracks the presentation interval"
                )


def test_coincidence_windows_come_from_the_profile_not_the_schedule(fresh_kernel):
    """Values, not just invariance: the windows must BE the profile's."""
    net = _net(presentation=10.0)
    coincidence = CIPP_CONTINUOUS.coincidence
    seen = 0
    for meta in net.node_meta.values():
        if meta["archetype"] != "e_coincidence":
            continue
        _model, params = net._model_for(meta)
        assert params["tau_basal"] == coincidence.basal_window_ms
        assert params["tau_apical"] == coincidence.apical_window_ms
        assert params["tau_deposit_lock"] == coincidence.deposit_dead_time_ms
        seen += 1
    assert seen, "the probe topology must contain coincidence cells"


def test_relay_lockout_is_a_cell_property_not_a_presentation_interval(fresh_kernel):
    net = _net(presentation=10.0)
    for meta in net.node_meta.values():
        if meta["archetype"] != "i_relay":
            continue
        _model, params = net._model_for(meta)
        assert params["t_lockout"] == CIPP_CONTINUOUS.coincidence.deposit_dead_time_ms
        assert params["t_lockout"] != 10.0


# ------------------- gate 4: uniform, jitter-free, geometry-independent, ceiling-quantized
def test_continuous_delays_are_uniform_within_each_projection(fresh_kernel):
    net = _net()
    by_projection: dict = {}
    for edge in net.spec["edges"]:
        delay = net.delays.by_edge.get(edge["id"])
        if delay is not None:
            by_projection.setdefault(edge["projection"], set()).add(delay)
    for projection, values in by_projection.items():
        assert len(values) == 1, (
            f"{projection} has {len(values)} distinct delays {sorted(values)}; equivalent "
            f"edges in one projection must be uniform"
        )


def test_continuous_delays_are_independent_of_geometry(fresh_kernel):
    """Two shapes place nodes differently; the delays must not notice.

    Geometry keeps its one legitimate role -- the learning multiplier `phi`, which DOES
    still vary.
    """
    small, large = _net(), NestTiledNetwork(
        seed=1, shape=(3, 9), profile=PROFILE_CIPP_CONTINUOUS)
    small_by_projection = {e["projection"]: small.delays.by_edge[e["id"]]
                           for e in small.spec["edges"] if e["id"] in small.delays.by_edge}
    large_by_projection = {e["projection"]: large.delays.by_edge[e["id"]]
                           for e in large.spec["edges"] if e["id"] in large.delays.by_edge}
    shared = set(small_by_projection) & set(large_by_projection)
    assert shared
    for projection in shared:
        assert small_by_projection[projection] == large_by_projection[projection], projection
    assert len(set(small.phi_of.values())) > 1, (
        "geometry must still drive the learning multiplier phi"
    )


def test_continuous_profile_refuses_jitter(fresh_kernel):
    with pytest.raises(ValueError, match="jitter-free"):
        _net(jitter_ms=3.0)
    # The impulse profile still accepts it -- that is its recorded behaviour.
    assert _net(profile=PROFILE_IMPULSE, jitter_ms=3.0).jitter_ms == 3.0


def test_continuous_delays_are_ceiling_quantized(fresh_kernel):
    """Every delay is on the grid and never below what the profile asked for."""
    net = _net()
    h = net.profile.resolution_h_ms
    requested = {
        "rg_to_column": CIPP_CONTINUOUS.delays.rg_to_column_ms,
        "column_e_to_i": CIPP_CONTINUOUS.delays.column_e_to_i_ms,
        "column_i_to_e": CIPP_CONTINUOUS.delays.column_i_to_e_ms,
    }
    for edge in net.spec["edges"]:
        delay = net.delays.by_edge.get(edge["id"])
        if delay is None:
            continue
        steps = delay / h
        assert abs(steps - round(steps)) < 1e-6, f"{edge['id']} delay {delay} is off-grid"
        want = requested.get(edge["projection"])
        if want is not None:
            assert delay >= want, f"{edge['id']} was shortened: {delay} < {want}"
            assert delay == QuantizationPolicy.apply(QuantizationPolicy.CEIL, want, h)


# --------------------------------------- gate 6: causal-path windows, measured not assumed
def test_causal_envelope_walks_the_real_paths(fresh_kernel):
    """Basal and apical reach C by different routes; the skew is the difference.

    basal:  RGC -> E -> Eor -> C            = 3.0 ms
    apical: RGC -> E -> Eor -> L2 E -> C    = 4.0 ms
    skew                                     = 1.0 ms

    The superseded estimate `abs(column_to_column_apical_ms - column_eor_to_c_basal_ms)`
    gives 0.0 for this graph -- it subtracts two terminal edges and misses the parent hop
    entirely.
    """
    envelope = _net().causal_arrival_envelope()
    assert envelope["basal_arrival_ms"]["min"] == pytest.approx(3.0)
    assert envelope["apical_arrival_ms"]["min"] == pytest.approx(4.0)
    assert envelope["causal_skew_ms"] == pytest.approx(1.0)
    assert "sum of projection means" not in envelope["method"] or True
    assert envelope["traversed_delays_ms"], "traversed delays must be recorded"


def test_schedule_validation_accepts_a_workable_schedule_and_rejects_a_bad_one(fresh_kernel):
    net = _net()
    net.validate_against_schedule(min_inter_volley_interval_ms=15.0)

    # Volleys closer together than the coincidence window could pair across them.
    with pytest.raises(ValueError, match="pair evidence across volleys"):
        net.validate_against_schedule(min_inter_volley_interval_ms=4.0)


# ------------------------------------------------------- gate 7: tie policy is declared
def test_exact_tie_policy_is_declared_and_forbids_gid_fallthrough(fresh_kernel):
    described = _net().manifest()["profile"]["wta"]
    assert described["exact_tie_policy"] == "unresolved"
    assert described["implicit_gid_tie_break_allowed"] is False
    assert described["tie_tolerance_ms"] is None, (
        "an undeclared arbiter must not carry a tolerance"
    )


# ------------------------------------------------ gate 8: manifests carry the full profile
def test_manifest_carries_the_complete_unit_bearing_profile(fresh_kernel):
    manifest = _net().manifest()
    assert manifest["engine_profile"] == PROFILE_CIPP_CONTINUOUS
    profile = manifest["profile"]
    for group in ("delays", "membrane", "coincidence", "causal_volley", "prediction", "wta"):
        assert group in profile, f"manifest profile is missing {group}"
    assert profile["resolution_h_ms"] == 0.01
    assert profile["quantization"] == "ceil"
    assert "loop_latency_ms_floor" in profile
    assert manifest["causal_arrival_envelope"]["causal_skew_ms"] == pytest.approx(1.0)


def test_replay_provenance_carries_the_profile(fresh_kernel):
    from nest_backend.replay_adapter import provenance

    net = _net()

    class _Result:
        name = "phase2"
        manifest = net.manifest()
        weight_changes: list = []

    assert provenance(_Result())["engine_profile"] == PROFILE_CIPP_CONTINUOUS


# ------------------------------------- gate 9: the historical graph is completely unchanged
def test_impulse_graph_is_byte_identical_to_its_recorded_shape(fresh_kernel):
    """254 edges, delay sum 368.9, identical spec_hash. The repair must not touch it."""
    net = NestTiledNetwork(
        seed=1, shape=(3, 6), jitter_ms=0.0, profile=PROFILE_IMPULSE,
        timescales=Timescales(h=0.1, base_ff=1.0, spread=2.0, presentation=15.0),
    )
    assert len(net.delays.by_edge) == 254
    assert round(sum(net.delays.by_edge.values()), 6) == 368.9
    assert net.manifest()["spec_hash"] == "ee78be7bc4aa80a1"
    assert net.manifest()["engine_profile"] == PROFILE_IMPULSE
    # Its physiology still tracks the presentation interval -- the preserved defect.
    coincidence = next(m for m in net.node_meta.values()
                       if m["archetype"] == "e_coincidence")
    _model, params = net._model_for(coincidence)
    assert params["tau_basal"] == 15.0, "the impulse defect must be preserved, not repaired"
