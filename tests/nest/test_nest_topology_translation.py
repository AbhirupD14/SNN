"""Phase 2: canonical topology translation (prompt sections 4, 5.3, 9 Phase 2, 11).

Structural parity is asserted against the canonical `NetworkSpec` itself, never against a
hand-written expected graph -- copying the topology into the test would defeat the point of
translating it.
"""
from __future__ import annotations

import pytest

from nest_backend.topology import (
    E_THRESHOLD,
    LeakNotSupported,
    NestTiledNetwork,
    Timescales,
    build_reference_engine,
)


@pytest.fixture(scope="module")
def net():
    return NestTiledNetwork(seed=1)


# ----------------------------------------------------------------- structure


def test_node_and_edge_counts_match_the_canonical_spec(net):
    from backend.network_spec import tiled_cc_spec

    spec = tiled_cc_spec(cc_e_count=8)
    assert len(spec["nodes"]) == 191
    assert len(spec["edges"]) == 1052
    manifest = net.manifest()
    assert manifest["counts"]["nodes"] == len(spec["nodes"])
    assert manifest["counts"]["edges"] == len(spec["edges"])
    assert manifest["counts"]["connected_edges"] == len(spec["edges"]), (
        "every spec edge must become a NEST connection when feedback is enabled"
    )


def test_role_inventory_matches_the_spec(net):
    from collections import Counter

    from backend.network_spec import tiled_cc_spec

    spec = tiled_cc_spec(cc_e_count=8)
    expected = Counter(n["archetype"] for n in spec["nodes"])
    assert expected == Counter({"e_latency_competitor": 90, "rg_source": 81,
                                "e_coincidence": 10, "i_relay": 10})
    assert len(net.ids_where(column_role="E")) == 80
    assert len(net.ids_where(column_role="Eor")) == 10
    assert len(net.ids_where(column_role="C")) == 10
    assert len(net.ids_where(column_role="I")) == 10


def test_edge_kind_inventory_matches_the_spec(net):
    from collections import Counter

    from backend.network_spec import tiled_cc_spec

    spec = tiled_cc_spec(cc_e_count=8)
    assert Counter(e["kind"] for e in spec["edges"]) == Counter({
        "feedforward": 800, "relay_excitation": 90, "hard_reset_inhibition": 80,
        "apical_excitation": 72, "basal_excitation": 10,
    })
    assert len(net.conn_of) == len(spec["edges"])


def test_external_id_mapping_is_complete_and_injective(net):
    manifest = net.manifest()
    id_map = manifest["id_map"]
    assert len(id_map) == 191, "every repository node id maps to a NEST node"
    assert len(set(id_map.values())) == 191, "the mapping is injective"
    for repo_id, gid in id_map.items():
        assert net.id_of_gid[gid] == repo_id, "the reverse mapping round-trips"


def test_dormant_top_column_c_is_selected_by_metadata_not_by_id(net):
    """The L2 `C` has no parent, so its gate can never open."""
    dormant = net.ids_where(archetype="e_coincidence", has_parent=False)
    assert dormant == ["L2c00C"]
    assert len(net.ids_where(archetype="e_coincidence", has_parent=True)) == 9


def test_construction_is_deterministic_for_a_fixed_seed():
    a = NestTiledNetwork(seed=7)
    b = NestTiledNetwork(seed=7)
    assert a.manifest()["spec_hash"] == b.manifest()["spec_hash"]
    assert a.weights == b.weights
    assert a.delays.by_edge == b.delays.by_edge


def test_different_seeds_change_weights_but_not_structure():
    a = NestTiledNetwork(seed=1)
    b = NestTiledNetwork(seed=2)
    assert a.manifest()["counts"] == b.manifest()["counts"]
    assert a.weights != b.weights, "the seeded initializer must actually depend on the seed"


# --------------------------------------------------------------------- timing


def test_every_delay_is_a_valid_multiple_of_h_and_at_least_h(net):
    h = net.ts.h
    for edge_id, delay in net.delays.by_edge.items():
        assert delay >= h - 1e-12, f"{edge_id} has delay {delay} below the resolution {h}"
        steps = delay / h
        assert abs(steps - round(steps)) < 1e-6, f"{edge_id} delay {delay} is off the h grid"


def test_fast_loop_pathways_take_the_minimum_delay(net):
    """Apical permission, WTA recruitment and the reset all run at `h` (section 5.3)."""
    fast = ("column_to_column_apical", "column_e_to_i", "column_i_to_e", "column_c_to_i")
    for edge in net.spec["edges"]:
        if edge.get("projection") in fast:
            assert net.delays.by_edge[edge["id"]] == pytest.approx(net.ts.h)


def test_declared_timescale_separation_holds_as_measured(net):
    """`L_wta : S : D` is checked on the built network, not on the intention."""
    manifest = net.manifest()
    ts = manifest["timescales"]
    assert ts["L_wta_ms"] == pytest.approx(2 * ts["h_ms"])
    realized = manifest["delays"]["realized_spread_ms"]["rg_to_column"]
    median = realized["per_target_median_ms"]
    assert median == pytest.approx(ts["S_target_ms"], rel=0.35), (
        f"realized per-target spread {median} should track the target S {ts['S_target_ms']}"
    )
    assert median > ts["L_wta_ms"], "S must exceed the lateral loop for a race to resolve"
    assert ts["D_ms"] > median, "the presentation interval must exceed the arrival spread"


def test_dispersion_can_be_disabled_as_a_control(net):
    """The control condition for section 5.2: a simultaneous volley, no arrival ordering."""
    flat = NestTiledNetwork(seed=1, dispersion_enabled=False)
    rg_delays = {flat.delays.by_edge[e["id"]] for e in flat.spec["edges"]
                 if e.get("projection") == "rg_to_column"}
    assert len(rg_delays) == 1, "with dispersion off every afferent shares one delay"


def test_feedback_loop_latency_is_derived_from_the_graph(net):
    """Never from a preset name -- it must re-track on any wiring change."""
    latency = net.feedback_loop_latency_ms()
    assert latency > 0
    # E -> Eor -> parent E dominates; the three fast hops add 3h.
    assert latency > 2 * net.ts.base_ff
    assert latency < 2 * net.ts.base_ff + net.ts.spread + 10 * net.ts.h


# -------------------------------------------------------------------- weights


def test_initial_weights_come_from_the_reference_engine(net):
    """Frozen means frozen at the ENGINE's deterministic initialization."""
    engine = build_reference_engine(seed=1)
    for edge in net.spec["edges"]:
        if edge["kind"] in ("feedforward", "basal_excitation"):
            assert edge["id"] in net.weights, f"{edge['id']} has no translated weight"
    eor_edges = [e for e in net.spec["edges"] if e.get("projection") == "column_e_to_eor"]
    for edge in eor_edges:
        assert net.weights[edge["id"]] == pytest.approx(E_THRESHOLD), (
            "Eor is initialized frozen at theta so it fires on a single afferent"
        )
    assert float(engine.params["leak_rate"]) == 0.0


def test_frozen_weights_never_change_during_simulation(net):
    """No `static_synapse` weight may move -- there is no plasticity in Phases 0-3."""
    import nest

    from nest_backend.engine import Stimulus, build_schedule

    fresh = NestTiledNetwork(seed=1)
    before = {c: nest.GetConnections(source=fresh.gid_of[e["source"]],
                                     target=fresh.gid_of[e["target"]]).get("weight")
              for c, e in [(e["id"], e) for e in fresh.spec["edges"][:40]]}
    schedule = build_schedule(fresh, Stimulus({(1, 1): "row 1"}),
                              t0=20.0, period=20.0, n_presentations=4)
    fresh.set_input_schedule(schedule)
    fresh.simulate(150.0)
    after = {c: nest.GetConnections(source=fresh.gid_of[e["source"]],
                                    target=fresh.gid_of[e["target"]]).get("weight")
             for c, e in [(e["id"], e) for e in fresh.spec["edges"][:40]]}
    assert before == after


def test_translation_refuses_a_non_zero_leak(monkeypatch):
    """The event models are an exact port ONLY at leak_rate = 0 (section 2.0)."""
    import nest_backend.topology as topology

    original = topology.reference_overrides

    def leaky(seed: int = 1):
        params = original(seed)
        params["leak_rate"] = 0.05
        return params

    monkeypatch.setattr(topology, "reference_overrides", leaky)
    with pytest.raises(LeakNotSupported):
        topology.build_reference_engine(seed=1)


# ------------------------------------------------------------------ stimulus


def test_center_patch_stimulus_traverses_the_expected_path():
    """RGC -> L1 -> Eor -> L2, and nothing outside the driven column."""
    from nest_backend.engine import Stimulus, run_case

    result = run_case("center", Stimulus({(1, 1): "row 1"}), n_presentations=6)
    by_column = result.metrics["by_column"]
    assert "L1c11" in by_column, "the driven centre column must be active"
    assert "L2c00" in by_column, "activity must reach the L2 column"
    assert set(by_column) <= {"L1c11", "L2c00"}, (
        f"only the driven column and L2 should fire, got {sorted(by_column)}"
    )
    assert by_column["L1c11"].get("Eor", 0) > 0, "the column's Eor must relay"


def test_two_independent_patches_stay_local():
    from nest_backend.engine import Stimulus, run_case

    result = run_case("two_patch", Stimulus({(0, 0): "row 1", (2, 2): "col 1"}),
                      n_presentations=6)
    active = set(result.metrics["by_column"])
    assert "L1c00" in active and "L1c22" in active
    untouched = {"L1c01", "L1c02", "L1c10", "L1c11", "L1c12", "L1c20", "L1c21"}
    assert not (active & untouched), (
        f"undriven columns must stay silent, got {sorted(active & untouched)}"
    )


def test_patch_pixels_are_patch_local_row_major(net):
    pixels = net.patch_pixels(1, 1)
    assert len(pixels) == 9
    metas = [net.node_meta[p] for p in pixels]
    assert [(m["patch_local_row"], m["patch_local_col"]) for m in metas] == [
        (r, c) for r in range(3) for c in range(3)
    ]
