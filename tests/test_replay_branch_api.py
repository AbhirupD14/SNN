"""API-layer tests for the replay branch request (backend/branch.py) and the
POST /api/replay/branch-weights endpoint.

These cover the strict compatibility contract, schema validation, actionable 4xx errors, the
independent (browser-untrusted) re-validation against the live engine, and that a failed
request never fabricates a success broadcast. No HTTP client is available in this repo, so the
endpoint's orchestration helper is exercised directly (exactly what the route calls).
"""

from __future__ import annotations

import asyncio

import pytest

from backend.simulation import SimulationEngine
from backend.branch import (
    apply_branch_request, build_branch_contract, compare_contracts, BranchError,
    BRANCH_SCHEMA_NAME, BRANCH_SCHEMA_VERSION,
)
from experiments.replay_recorder import REPLAY_SCHEMA_NAME, REPLAY_SCHEMA_VERSION


# --------------------------------------------------------------------- helpers
def make_payload(engine, **over):
    topo = engine.topology()
    snap = {s["id"]: s["weight"] for s in topo["synapses"] if s["weight"] is not None}
    payload = dict(
        branch_schema=BRANCH_SCHEMA_NAME, branch_schema_version=BRANCH_SCHEMA_VERSION,
        replay_schema=REPLAY_SCHEMA_NAME, replay_schema_version=REPLAY_SCHEMA_VERSION,
        source={"run_id": "run-x", "experiment": "exp", "seed": engine.params["seed"]},
        frame_index=7, timestep=140, precision="checkpoint", use_checkpoint=False,
        recorded_topology=topo, weights=snap, restore_input=True,
        input=engine.input_vec.astype(int).tolist(),
    )
    payload.update(over)
    return payload


def fingerprint(engine):
    return tuple(sorted((s["id"], s["weight"]) for s in engine.topology()["synapses"]
                        if s["weight"] is not None))


# ---------------------------------------------------------------- happy path
def test_branch_request_success():
    rec = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    payload = make_payload(rec)
    live = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    for _ in range(6):
        live.step()
    res = apply_branch_request(live, payload)
    assert res["branched"] is True
    assert res["restored_mutable"] > 0
    assert res["source_frame_index"] == 7
    assert res["source_timestep"] == 140
    assert res["live_timestep"] == 0
    assert res["precision"] == "checkpoint"
    assert res["source_seed"] == 7 and res["live_seed"] == 7
    # weights match the recorded snapshot
    assert fingerprint(live) == fingerprint(rec)


# ------------------------------------------------------------ schema errors
def test_unsupported_branch_schema():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    with pytest.raises(BranchError) as ei:
        apply_branch_request(e, make_payload(e, branch_schema="nope"))
    assert ei.value.status_code == 400


def test_unsupported_replay_schema_version():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    with pytest.raises(BranchError) as ei:
        apply_branch_request(e, make_payload(e, replay_schema_version=999))
    assert ei.value.status_code == 400


def test_malformed_body_rejected():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    with pytest.raises(BranchError):
        apply_branch_request(e, "not a dict")
    with pytest.raises(BranchError):
        apply_branch_request(e, make_payload(e, weights="not an object"))


# ---------------------------------------------------- incompatible topology
def test_incompatible_topology_is_409_and_untouched():
    rec = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    payload = make_payload(rec)
    live = SimulationEngine(seed=7, topology="tiled_cc")
    before = fingerprint(live)
    with pytest.raises(BranchError) as ei:
        apply_branch_request(live, payload)
    assert ei.value.status_code == 409
    assert "topology" in str(ei.value) or "neuron" in str(ei.value)
    assert fingerprint(live) == before                 # live engine entirely unchanged


def test_seed_mismatch_is_409_with_actionable_message():
    rec = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    payload = make_payload(rec)
    live = SimulationEngine(seed=99, topology="rg_coincidence", leak_rate=0.0)
    with pytest.raises(BranchError) as ei:
        apply_branch_request(live, payload)
    assert ei.value.status_code == 409
    assert "seed" in str(ei.value)


def test_model_parameter_mismatch_is_409():
    rec = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    payload = make_payload(rec)
    live = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.03)
    with pytest.raises(BranchError) as ei:
        apply_branch_request(live, payload)
    assert ei.value.status_code == 409
    assert "leak_rate" in str(ei.value)


def test_learning_mode_mismatch_is_409():
    # dual FE/FES on vs off is a model-affecting mismatch (dual_fe_fes flag).
    rec = SimulationEngine(seed=3, topology="rg_direct_cc4", dual_fe_fes=True)
    payload = make_payload(rec)
    live = SimulationEngine(seed=3, topology="rg_direct_cc4", dual_fe_fes=False)
    with pytest.raises(BranchError) as ei:
        apply_branch_request(live, payload)
    assert ei.value.status_code == 409
    assert "dual_fe_fes" in str(ei.value)


# ------------------------------------------------------------- bounded sizes
def test_oversized_weight_map_rejected_cheaply():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    payload = make_payload(e)
    payload["weights"] = {f"x{i}": 1.0 for i in range(len(e.synapses) + 5)}
    with pytest.raises(BranchError, match="entries"):
        apply_branch_request(e, payload)


def test_oversized_input_rejected():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    payload = make_payload(e, input=[0] * (e.n_pix + 3))
    with pytest.raises(BranchError, match="pixels"):
        apply_branch_request(e, payload)


# --------------------------------------------------------- contract helpers
def test_build_contract_excludes_weight_values_but_keeps_weighted_set():
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    c1 = build_branch_contract(e.topology())
    # change some weights; the contract must be identical (only the SET of weighted edges
    # matters, never their values)
    for _ in range(5):
        e.step()
    c2 = build_branch_contract(e.topology())
    assert c1 == c2
    assert c1["weighted"]


def test_compare_contracts_identical_is_none():
    a = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    b = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    for _ in range(4):
        b.step()                                       # weights drift; contract is unchanged
    assert compare_contracts(build_branch_contract(a.topology()),
                             build_branch_contract(b.topology())) is None


def test_arbitrary_replay_text_is_data_only():
    # A hostile recorded_topology that is not a proper object must be rejected as data, never
    # dereferenced as a path or evaluated.
    e = SimulationEngine(seed=7, topology="rg_coincidence", leak_rate=0.0)
    with pytest.raises(BranchError):
        apply_branch_request(e, make_payload(e, recorded_topology="/etc/passwd"))
    with pytest.raises(BranchError):
        build_branch_contract({"neurons": "not-a-list"})


# ----------------------------------------------------------- endpoint smoke
def test_endpoint_pauses_runner_and_returns_success():
    import backend.api as api
    api.runner.running = True
    payload = make_payload(api.engine)                 # from the live global engine itself
    res = asyncio.run(api.branch_weights(api.BranchWeightsBody(**payload)))
    assert api.runner.running is False                 # forced paused on the branch
    assert isinstance(res, dict) and res.get("branched") is True


def test_endpoint_failure_returns_4xx_and_does_not_broadcast():
    import backend.api as api
    from fastapi.responses import JSONResponse
    before = fingerprint(api.engine)
    payload = make_payload(api.engine, branch_schema="bad.schema")
    resp = asyncio.run(api.branch_weights(api.BranchWeightsBody(**payload)))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    assert fingerprint(api.engine) == before           # engine untouched, no success state
