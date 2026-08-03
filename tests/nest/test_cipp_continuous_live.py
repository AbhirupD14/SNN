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
def test_changing_pacing_leaves_every_neuron_AND_synapse_dict_identical(fresh_kernel):
    """THE Phase 2 contract, over neuron AND synapse physiology.

    The first version of this test compared only `_model_for()` while claiming both. It
    also chose 10/40 ms, which happen to sit on the same side of the old
    `presentation * 0.5` clamp, so the leak stayed invisible. Measured on the continuous
    branch before the fix::

        presentation 3.0 ms -> plastic tau_volley 1.5 ms
        presentation 5.0 ms -> plastic tau_volley 2.5 ms

    The pairs below straddle that clamp deliberately, and also vary the legacy
    `Timescales.spread` and `base_ff` the old formula read.
    """
    variants = [
        _net(presentation=3.0, learning=True),
        _net(presentation=5.0, learning=True),
        _net(presentation=40.0, learning=True),
        NestTiledNetwork(seed=1, shape=(3, 6), profile=PROFILE_CIPP_CONTINUOUS,
                         learning=True,
                         timescales=Timescales(h=0.1, base_ff=7.0, spread=0.25,
                                               presentation=9.0)),
    ]

    def physics(net):
        neurons = {nid: net._model_for(meta) for nid, meta in net.node_meta.items()}
        synapses = {}
        for edge in net.spec["edges"]:
            if edge["id"] not in net.delays.by_edge:
                continue
            spec = dict(net._syn_spec(edge))
            # Delay is a conduction property carried by the profile and is compared with
            # everything else; only repository identity is normalised away.
            spec.pop("receptor_type", None)
            synapses[edge["id"]] = spec
        return neurons, synapses

    reference_neurons, reference_synapses = physics(variants[0])
    for net in variants[1:]:
        neurons, synapses = physics(net)
        differing_neurons = {k: (reference_neurons[k], neurons[k])
                             for k in reference_neurons if reference_neurons[k] != neurons[k]}
        assert not differing_neurons, (
            f"neuron physics changed with the schedule: "
            f"{dict(list(differing_neurons.items())[:2])}"
        )
        differing_synapses = {k: (reference_synapses[k], synapses[k])
                              for k in reference_synapses
                              if reference_synapses[k] != synapses[k]}
        assert not differing_synapses, (
            f"SYNAPSE physics changed with the schedule: "
            f"{dict(list(differing_synapses.items())[:2])}"
        )


def test_learning_membership_comes_from_the_profile_not_the_schedule(fresh_kernel):
    """`tau_volley` is the causal-volley separation, full stop."""
    separation = CIPP_CONTINUOUS.causal_volley.separation_ms
    for presentation in (3.0, 5.0, 10.0, 40.0):
        net = _net(presentation=presentation, learning=True)
        assert net.tau_volley_ms == separation, (
            f"presentation {presentation} gave tau_volley {net.tau_volley_ms}"
        )
        edge = next(e for e in net.spec["edges"]
                    if e.get("projection") == "rg_to_column")
        assert net._syn_spec(edge)["tau_volley"] == separation


def test_schedule_validation_runs_on_the_real_path_including_irregular_times(fresh_kernel):
    """Validation is invoked by `set_input_schedule`, not left to a caller to remember.

    An irregular schedule is checked on its SMALLEST actual gap, not a nominal period.
    """
    net = _net(presentation=15.0)
    rgc = sorted(net.generators)[:2]
    # Irregular but legal: smallest gap 6 ms, comfortably above the 5 ms window.
    net.set_input_schedule({rgc[0]: [10.0, 16.0, 40.0], rgc[1]: [10.0, 40.0]})

    # Smallest gap 2 ms -- below the coincidence window, so it must be refused.
    with pytest.raises(ValueError, match="pair evidence across volleys"):
        net.set_input_schedule({rgc[0]: [10.0, 12.0, 40.0]})


def test_a_rejected_schedule_leaves_every_generator_untouched(fresh_kernel):
    """`set_input_schedule` is atomic: snap, validate, THEN install.

    It previously wrote every generator before validating, so a schedule the profile
    rejected stayed installed in the kernel -- the caller saw an exception while NEST held
    the bad times. Reproduced before the fix::

        installed [10.0, 30.0] -> rejected call -> generator held [10.0, 12.0, 40.0]

    Checked across ALL generators, not just the ones the rejected call mentioned.
    """
    net = _net(presentation=15.0)
    driven = sorted(net.generators)[:3]

    good = {driven[0]: [10.0, 30.0], driven[1]: [10.0, 30.0]}
    net.set_input_schedule(good)
    before = {nid: [float(t) for t in gen.get("spike_times")]
              for nid, gen in net.generators.items()}
    assert before[driven[0]] == [10.0, 30.0], "setup: the valid schedule must install"

    # Smallest gap 2 ms, below the 5 ms coincidence window: the profile must refuse it.
    with pytest.raises(ValueError, match="pair evidence across volleys"):
        net.set_input_schedule({driven[0]: [10.0, 12.0, 40.0], driven[2]: [10.0, 12.0]})

    after = {nid: [float(t) for t in gen.get("spike_times")]
             for nid, gen in net.generators.items()}
    assert after == before, (
        "a rejected schedule must leave every generator exactly as it was; "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )
    # Including the generator the rejected call would have cleared.
    assert after[driven[1]] == [10.0, 30.0]
    # And a later valid call still installs normally, so the guard is not sticky.
    net.set_input_schedule({driven[0]: [10.0, 40.0]})
    assert [float(t) for t in net.generators[driven[0]].get("spike_times")] == [10.0, 40.0]


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


def test_relay_lockout_comes_from_the_relay_policy_not_the_C_soma(fresh_kernel):
    """One role per field. The relay lockout had been borrowing the C deposit dead time."""
    net = _net(presentation=10.0)
    seen = 0
    for meta in net.node_meta.values():
        if meta["archetype"] != "i_relay":
            continue
        _model, params = net._model_for(meta)
        assert params["t_lockout"] == CIPP_CONTINUOUS.relay.wta_lockout_ms
        assert params["t_lockout"] != 10.0
        seen += 1
    assert seen, "the probe topology must contain relay cells"


def test_c_refractory_is_the_C_cells_own_value_not_the_competitors(fresh_kernel):
    """`MembranePolicy.t_ref_ms` describes the continuous COMPETITOR, not the C cell."""
    net = _net()
    competitor_t_ref = CIPP_CONTINUOUS.membrane.t_ref_ms
    for meta in net.node_meta.values():
        if meta["archetype"] != "e_coincidence":
            continue
        _model, params = net._model_for(meta)
        assert params["t_ref"] == CIPP_CONTINUOUS.coincidence.c_refractory_ms == 0.0
        assert params["t_ref"] != competitor_t_ref, (
            "the C cell must not inherit the competitor refractory"
        )


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
def test_causal_envelope_walks_connected_edge_sequences(fresh_kernel):
    """Real connectivity, asserted on endpoints -- not projection extrema.

    The superseded version grouped edges by projection and summed independent extrema, and
    its test carried the vacuous assertion
    `assert "sum of projection means" not in envelope["method"] or True`, which is true for
    every possible input. This checks the actual node and edge sequences.
    """
    net = _net()
    envelope = net.causal_arrival_envelope()
    edge_by_id = {e["id"]: e for e in net.spec["edges"]}

    assert envelope["matched_pairs"] > 0
    for pair in envelope["pairs_recorded"]:
        for arm in ("basal", "apical"):
            path = pair[arm]
            nodes, edges = path["nodes"], path["edges"]
            assert len(edges) == len(nodes) - 1
            # Every hop must actually connect the two nodes it sits between.
            for index, edge_id in enumerate(edges):
                edge = edge_by_id[edge_id]
                assert edge["source"] == nodes[index], (
                    f"{edge_id} does not start at {nodes[index]}"
                )
                assert edge["target"] == nodes[index + 1], (
                    f"{edge_id} does not end at {nodes[index + 1]}"
                )
            # The recorded delays are the delays of those edges, and they sum to the total.
            assert path["delays"] == [net.delays.by_edge[e] for e in edges]
            assert path["path_delay_ms"] == pytest.approx(sum(path["delays"]))


def test_causal_envelope_matches_pairs_at_the_same_C_and_branch(fresh_kernel):
    """A coincidence is one C receiving both arms of ONE causal branch.

    Summarising 144 basal and 2304 apical paths independently can subtract an arrival at
    one C from an arrival at another, or from one driven by a different RGC. That
    difference is not a skew.
    """
    envelope = _net().causal_arrival_envelope()
    for pair in envelope["pairs_recorded"]:
        basal, apical = pair["basal"], pair["apical"]
        assert basal["nodes"][-1] == apical["nodes"][-1] == pair["target_c"], (
            "both arms must terminate at the SAME coincidence cell"
        )
        assert basal["nodes"][:3] == apical["nodes"][:3] == pair["shared_prefix_nodes"], (
            "both arms must share the RGC -> competitor -> relay causal prefix"
        )
        assert pair["source_rgc"] == basal["nodes"][0]
        assert pair["conduction_skew_ms"] == pytest.approx(
            apical["path_delay_ms"] - basal["path_delay_ms"])
    assert envelope["unmatched_basal_paths"] == 0, (
        "every basal path in this graph should have a matching apical partner"
    )


def test_causal_envelope_is_labelled_conduction_only_and_defers_processing(fresh_kernel):
    """The reported skew is a LOWER BOUND and must say so.

    It omits the parent competitor's membrane crossing latency on the apical arm, which is
    unknown until Phase 3. Phase 1's pA weights cannot supply it -- that report states
    explicitly that those units are not mapped to repository charge units.
    """
    envelope = _net().causal_arrival_envelope()
    assert envelope["kind"] == "conduction_only"
    assert envelope["processing_latency_included"] is False
    assert envelope["window_validation_status"] == "provisional_deferred_to_phase_3"
    assert "LOWER BOUND" in envelope["deferred"]


def test_full_window_validation_is_not_performed_against_conduction_alone(fresh_kernel):
    """Only the bound that does not need the missing term is checked.

    Validating a 5 ms window against a conduction-only lower bound would produce a pass
    that means nothing, so the skew check is deferred and only the cross-volley bound runs.
    """
    net = _net()
    # A skew far wider than the window would fail the FULL check; the deferred path must
    # not run it, so a workable schedule still validates.
    net.validate_against_schedule(min_inter_volley_interval_ms=15.0)
    # And the full check, when explicitly invoked, does still reject that case.
    with pytest.raises(ValueError, match="measured causal arrival skew"):
        CIPP_CONTINUOUS.coincidence.validate(causal_skew_ms=99.0)


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
    for group in ("relay", "disposition"):
        assert group in profile, f"manifest profile is missing {group}"
    envelope = manifest["causal_arrival_envelope"]
    assert envelope["conduction_skew_ms"] == pytest.approx(1.0)
    assert envelope["kind"] == "conduction_only"
    assert manifest["accepted_differences"], "accepted differences must travel too"


def test_manifest_never_reports_the_target_membrane_model_as_active(fresh_kernel):
    """DEFECT 4: the profile serialises `iaf_psc_exp_ps` while the live cells are not that.

    The Phase 2/3 boundary is legitimate; advertising the target as active physics is not.
    """
    net = _net()
    disposition = net.manifest()["profile"]["disposition"]
    assert disposition["target_membrane_model"] == "iaf_psc_exp_ps"
    assert disposition["active_competitor_model"] == "event_accumulator"
    assert disposition["target_membrane_model"] != disposition["active_competitor_model"]
    assert disposition["scaffold"] is True
    assert "charge units" in disposition["active_units"]

    # And the live cells really are the accumulator, so the disposition is not just a label.
    competitor = next(m for m in net.node_meta.values()
                      if m["archetype"] == "e_latency_competitor"
                      and m.get("column_role") != "Eor")
    model, _params = net._model_for(competitor)
    assert model.startswith("event_accumulator")
    assert model != CIPP_CONTINUOUS.membrane.model


def test_no_artifact_implies_the_mechanical_profile_has_passed(fresh_kernel):
    """`mechanical_profile_promoted` must be False everywhere it appears."""
    net = _net()
    manifest = net.manifest()
    assert manifest["profile"]["disposition"]["mechanical_profile_promoted"] is False

    dashboard = net.dashboard_topology()["nest"]
    assert dashboard["implementation_disposition"]["mechanical_profile_promoted"] is False
    joined = " ".join(dashboard["known_differences"])
    assert "SCAFFOLD" in joined and "mechanical_profile_promoted = False" in joined
    assert "TARGET, not active physics" in joined

    # And the disposition object itself refuses to be promoted by Phase 2.
    import dataclasses
    promoted = dataclasses.replace(CIPP_CONTINUOUS.disposition,
                                   mechanical_profile_promoted=True)
    with pytest.raises(ValueError, match="may not be set until"):
        promoted.validate()


def test_each_profile_has_its_own_accurate_disposition(fresh_kernel):
    """The impulse profile must not inherit the continuous profile's TARGET.

    Inheriting the default made every impulse manifest advertise an `iaf_psc_exp_ps`
    target the prototype has never had.
    """
    impulse = _net(profile=PROFILE_IMPULSE).manifest()["profile"]["disposition"]
    assert impulse["target_membrane_model"] == "event_accumulator", (
        "the prototype has no continuous target; what it runs is what it is"
    )
    assert impulse["scaffold"] is False
    assert impulse["known_defects"], "the preserved defects must be declared"
    joined = " ".join(impulse["known_defects"])
    assert "round()" in joined and "presentation interval" in joined

    impulse_membrane = _net(profile=PROFILE_IMPULSE).manifest()["profile"]["membrane"]
    assert impulse_membrane["applicable"] is False
    assert "model" not in impulse_membrane, (
        "an impulse manifest must not attach iaf_psc_exp_ps mV/pF values to its accumulator"
    )

    continuous = _net().manifest()["profile"]["disposition"]
    assert continuous["target_membrane_model"] == "iaf_psc_exp_ps"
    assert continuous["scaffold"] is True


def test_replay_provenance_carries_the_complete_profile_payload(fresh_kernel):
    """DEFECT 5: the name alone cannot distinguish a scaffold from a promoted engine."""
    from nest_backend.replay_adapter import provenance

    net = _net()
    manifest = net.manifest()

    class _Result:
        name = "phase2"
        weight_changes: list = []

    _Result.manifest = manifest
    prov = provenance(_Result())

    assert prov["engine_profile"] == PROFILE_CIPP_CONTINUOUS
    # Compared against the SOURCE manifest, not merely checked for presence.
    assert prov["profile"] == manifest["profile"]
    assert prov["implementation_disposition"] == manifest["profile"]["disposition"]
    assert prov["causal_arrival_envelope"] == manifest["causal_arrival_envelope"]
    assert prov["accepted_differences"] == manifest["accepted_differences"]
    # The payload has to be enough to tell the status without the manifest.
    assert prov["implementation_disposition"]["mechanical_profile_promoted"] is False
    assert prov["causal_arrival_envelope"]["kind"] == "conduction_only"


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
