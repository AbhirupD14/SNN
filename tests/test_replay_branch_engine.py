"""Engine-level tests for the replay branch loader (SimulationEngine.branch_from_weights).

A branch installs a recorded weight snapshot into a freshly reset compatible engine. These
tests prove the loader is atomic (every validation failure preserves the complete prior live
state), restores exactly the submitted mutable weights at recorded precision, checks -- never
mutates -- fixed pretrained edges, clears all transient neuron/event/continuous state, and
never learns merely by loading. Reference maps are used, never neuron-id prefixes.
"""

from __future__ import annotations

import math

import pytest

from backend.simulation import SimulationEngine


# --------------------------------------------------------------------- helpers
def snapshot(engine):
    """The complete {edge_id: weight} map the engine reports (every non-null weight)."""
    return {s["id"]: s["weight"] for s in engine.topology()["synapses"]
            if s["weight"] is not None}


def fingerprint(engine):
    """A hashable fingerprint of the complete live weight state (for atomicity checks)."""
    return tuple(sorted(snapshot(engine).items()))


def predictor_spec():
    """A tiny hand-authored graph that CONTAINS predictive_inhibition (no dashboard preset
    does). Two sensory L1E, one competitor L2E, one predictor PI0, one WTA L2I."""
    return dict(name="pred_test", nodes=[
        dict(id="L1E0", archetype="e_sensory", layer="L1", label="L1E0", pixel=0),
        dict(id="L1E1", archetype="e_sensory", layer="L1", label="L1E1", pixel=1),
        dict(id="PI0", archetype="predictor", layer="L2", label="PI0"),
        dict(id="L2E0", archetype="e_competitor", layer="L2", label="L2E0"),
        dict(id="L2I", archetype="i_relay", layer="L2", label="L2I"),
    ], edges=[
        dict(id="ff0->0", source="L1E0", target="L2E0", kind="feedforward", directed=True),
        dict(id="ff1->0", source="L1E1", target="L2E0", kind="feedforward", directed=True),
        dict(id="re0", source="L2E0", target="L2I", kind="relay_excitation", directed=True),
        dict(id="inh0", source="L2I", target="L2E0", kind="inhibition", sign=-1, directed=True),
        dict(id="re_pi0", source="L2E0", target="PI0", kind="relay_excitation", directed=True),
        dict(id="pi0->0", source="PI0", target="L1E0", kind="predictive_inhibition",
             sign=-1, directed=True),
        dict(id="pi0->1", source="PI0", target="L1E1", kind="predictive_inhibition",
             sign=-1, directed=True),
    ])


# --------------------------------------------------------------- happy paths
def test_ordinary_feedforward_restoration():
    e = SimulationEngine(seed=5, topology="rg_direct_cc4")
    snap = snapshot(e)
    # bump a couple of feedforward weights and branch back to the recorded snapshot
    ff = [s["id"] for s in e.topology()["synapses"] if s["kind"] == "feedforward"]
    e.set_synapse_weight(ff[0], 321.0)
    res = e.branch_from_weights(dict(snap), restore_input=False)
    assert res["restored_mutable"] == len(ff)
    assert snapshot(e) == pytest.approx(snap, abs=1e-9)


def test_coincidence_basal_restoration():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    snap = snapshot(e)
    basal = [s["id"] for s in e.topology()["synapses"] if s["kind"] == "basal_excitation"]
    assert basal
    target = snap[basal[0]] * 0.5 + 1.0
    snap2 = dict(snap); snap2[basal[0]] = target
    e.branch_from_weights(dict(snap2), restore_input=False)
    assert snapshot(e)[basal[0]] == pytest.approx(target, abs=1e-9)


def test_predictive_weight_restoration():
    e = SimulationEngine(seed=1)
    e.apply_topology(predictor_spec())
    e.set_synapse_weight("pi0->0", 0.4)
    snap = snapshot(e)
    assert "pi0->0" in snap
    e.step()                                        # move some weights around
    e.branch_from_weights(dict(snap), restore_input=False)
    assert snapshot(e)["pi0->0"] == pytest.approx(0.4, abs=1e-9)


def test_fixed_pretrained_edges_checked_but_not_mutated():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    snap = snapshot(e)
    pre = [s["id"] for s in e.topology()["synapses"] if s["kind"] == "pretrained_excitation"]
    assert pre
    q = e._q_pretrained
    res = e.branch_from_weights(dict(snap), restore_input=False)
    assert res["checked_fixed"] == len(pre)
    # the fixed delivered magnitude is unchanged (it is never a plastic weight)
    for eid in pre:
        assert snapshot(e)[eid] == pytest.approx(round(q, 6), abs=1e-6)


def test_dual_cap_free_weight_above_historical_cap_restores_unchanged():
    e = SimulationEngine(seed=3, topology="rg_direct_cc4", dual_fe_fes=True)
    snap = snapshot(e)
    ff = [s["id"] for s in e.topology()["synapses"] if s["kind"] == "feedforward"][0]
    big = 999999.0                                   # far above the legacy theta/2 = 500 cap
    snap[ff] = big
    e.branch_from_weights(dict(snap), restore_input=False)
    assert snapshot(e)[ff] == pytest.approx(big, abs=1e-3)


# ----------------------------------------------------------------- rejections
@pytest.fixture
def coincidence():
    return SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)


def test_below_floor_rejected_not_clipped(coincidence):
    e = coincidence
    before = fingerprint(e)
    snap = snapshot(e)
    ff = [s["id"] for s in e.topology()["synapses"] if s["kind"] == "feedforward"][0]
    snap[ff] = -5.0
    with pytest.raises(ValueError, match="below the"):
        e.branch_from_weights(dict(snap), restore_input=False)
    assert fingerprint(e) == before                  # untouched (not clipped to floor)


def test_missing_entry_rejected(coincidence):
    e = coincidence
    before = fingerprint(e)
    snap = snapshot(e)
    snap.pop(next(iter(snap)))
    with pytest.raises(ValueError, match="incomplete|missing"):
        e.branch_from_weights(dict(snap), restore_input=False)
    assert fingerprint(e) == before


def test_extra_unknown_entry_rejected(coincidence):
    e = coincidence
    before = fingerprint(e)
    snap = snapshot(e); snap["not_a_real_edge"] = 1.0
    with pytest.raises(ValueError, match="unknown|extra"):
        e.branch_from_weights(dict(snap), restore_input=False)
    assert fingerprint(e) == before


def test_nonfinite_value_rejected(coincidence):
    e = coincidence
    before = fingerprint(e)
    snap = snapshot(e); snap[next(iter(snap))] = math.inf
    with pytest.raises(ValueError, match="not finite"):
        e.branch_from_weights(dict(snap), restore_input=False)
    assert fingerprint(e) == before


def test_nonnumeric_value_rejected(coincidence):
    e = coincidence
    snap = snapshot(e); snap[next(iter(snap))] = "big"
    with pytest.raises(ValueError, match="not a number"):
        e.branch_from_weights(dict(snap), restore_input=False)


def test_fixed_pretrained_mismatch_rejected(coincidence):
    e = coincidence
    before = fingerprint(e)
    snap = snapshot(e)
    pre = [s["id"] for s in e.topology()["synapses"] if s["kind"] == "pretrained_excitation"][0]
    snap[pre] = snap[pre] + 50.0                      # a fixed magnitude is never converted
    with pytest.raises(ValueError, match="fixed pretrained"):
        e.branch_from_weights(dict(snap), restore_input=False)
    assert fingerprint(e) == before


def test_predictive_out_of_range_rejected():
    e = SimulationEngine(seed=1)
    e.apply_topology(predictor_spec())
    snap = snapshot(e); snap["pi0->0"] = 99.0         # pi_w_max default 1.0
    with pytest.raises(ValueError, match="exceeds the"):
        e.branch_from_weights(dict(snap), restore_input=False)


# ------------------------------------------------------- transient state reset
def test_branch_clears_transient_state_and_resets_timestep(coincidence):
    e = coincidence
    e.set_pattern("row 1")
    for _ in range(8):                                # accumulate membrane / event / timestep state
        e.step()
    assert e.timestep > 0
    snap = snapshot(e)
    e.branch_from_weights(dict(snap), restore_input=False)
    assert e.timestep == 0                            # fresh engine at its normal reset timestep
    for n in e.neurons.values():
        assert float(n.potential) == pytest.approx(0.0, abs=1e-9)
        if hasattr(n, "g_inh"):
            assert float(n.g_inh) == pytest.approx(0.0, abs=1e-9)
        if hasattr(n, "refractory_timer"):
            assert int(n.refractory_timer) == 0
        if hasattr(n, "remaining_excitation"):
            assert float(n.remaining_excitation) == pytest.approx(0.0, abs=1e-9)
    assert e._continuous == {}
    assert e.changed_synapses == []
    assert e.hard_reset_events == []


def test_source_timestep_is_provenance_only(coincidence):
    e = coincidence
    for _ in range(5):
        e.step()
    snap = snapshot(e)
    res = e.branch_from_weights(dict(snap), restore_input=False,
                                provenance={"timestep": 840, "frame_index": 42})
    assert res["source_timestep"] == 840
    assert res["source_frame_index"] == 42
    assert res["live_timestep"] == 0                  # the fresh engine starts at reset t


def test_no_learning_occurs_by_loading(coincidence):
    e = coincidence
    snap = snapshot(e)
    e.branch_from_weights(dict(snap), restore_input=False)
    # loading alone must not run the learning rule: weights equal the submitted snapshot exactly
    assert snapshot(e) == pytest.approx(snap, abs=1e-9)


# --------------------------------------------------------------- input restore
def test_input_restoration_on_and_off(coincidence):
    e = coincidence
    snap = snapshot(e)
    vec = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    e.branch_from_weights(dict(snap), input_vector=vec, restore_input=True)
    assert e.input_vec.astype(int).tolist() == vec
    # off -> blank input regardless of a supplied vector
    e.branch_from_weights(dict(snap), input_vector=vec, restore_input=False)
    assert e.input_vec.astype(int).tolist() == [0] * e.n_pix


def test_input_vector_validation(coincidence):
    e = coincidence
    snap = snapshot(e)
    before = fingerprint(e)
    with pytest.raises(ValueError, match="pixels"):
        e.branch_from_weights(dict(snap), input_vector=[0, 1], restore_input=True)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        e.branch_from_weights(dict(snap), input_vector=[2] * e.n_pix, restore_input=True)
    assert fingerprint(e) == before                  # invalid input never resets the engine


def test_repeated_branch_is_deterministic(coincidence):
    e = coincidence
    snap = snapshot(e)
    e.branch_from_weights(dict(snap), restore_input=False)
    first = fingerprint(e)
    e.branch_from_weights(dict(snap), restore_input=False)
    assert fingerprint(e) == first
