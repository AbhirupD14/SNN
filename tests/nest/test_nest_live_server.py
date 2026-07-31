"""Stage B: the read-only live NEST server (dashboard prompt section 5).

The gating test is `test_chunked_execution_is_identical_to_one_shot`. Live mode is only
legitimate because chunking is inert; if that ever stops being true, this file fails and
the server's justification is gone with it.
"""
from __future__ import annotations

import pytest

from nest_backend.engine import Stimulus, build_schedule
from nest_backend.recording import collect_spikes
from nest_backend.topology import NestTiledNetwork, Timescales

CENTER = Stimulus({(1, 1): "row 1"})


def _prepared(threads: int, ts: Timescales):
    net = NestTiledNetwork(seed=1, timescales=ts, threads=threads)
    net.set_input_schedule(build_schedule(net, CENTER, t0=ts.presentation,
                                          period=ts.presentation, n_presentations=5))
    return net


def _final_state(net):
    import nest

    spikes = sorted((round(t, 6), nid) for t, nid in collect_spikes(net))
    counters = {}
    for nid in sorted(net.gid_of):
        if net.node_meta[nid]["archetype"] == "rg_source":
            continue
        cell = net.gid_of[nid]
        counters[nid] = {f: float(cell.get(f))
                         for f in ("n_spikes", "q", "q_pre")
                         if f in cell.get()}
    weights = {}
    for edge_id, (src, tgt) in sorted(net.conn_of.items()):
        conn = nest.GetConnections(source=nest.NodeCollection([src]),
                                   target=nest.NodeCollection([tgt]))
        weights[edge_id] = round(float(conn.get("weight")), 9)
    return spikes, counters, weights


# ------------------------------------------------------------------- THE GATE


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_chunked_execution_is_identical_to_one_shot(threads):
    """Live mode may only exist while this passes (prompt section 5.3).

    The chunk is an integer multiple of `h` and deliberately does NOT divide the
    presentation interval, so chunk boundaries land mid-volley.
    """
    ts = Timescales()
    duration = ts.presentation * 8
    chunk = round(round(ts.presentation / 3.0 / ts.h) * ts.h, 10)
    assert abs(chunk / ts.h - round(chunk / ts.h)) < 1e-9, "chunk must be a multiple of h"
    assert abs(ts.presentation % chunk) > 1e-9, "chunk must not align with presentations"

    one_shot = _prepared(threads, ts)
    one_shot.simulate(duration)
    expected = _final_state(one_shot)

    chunked = _prepared(threads, ts)
    remaining = duration
    while remaining > 1e-9:
        step = min(chunk, remaining)
        chunked.simulate(step)
        remaining = round(remaining - step, 10)
    actual = _final_state(chunked)

    assert actual[0] == expected[0], "spike multiset differs"
    assert actual[1] == expected[1], "node counters differ"
    assert actual[2] == expected[2], "connection weights differ"


# --------------------------------------------------------------- the server


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from nest_backend.dashboard_api import app

    with TestClient(app) as c:
        yield c


def test_server_identifies_itself_as_nest_live_not_the_legacy_engine(client):
    info = client.get("/api/nest/info").json()
    assert info["engine"] == "nest"
    assert info["mode"] == "live"
    assert info["read_only"] is True
    assert info["learning"] == "frozen"
    assert info["mpi_available"] is False


def test_unsupported_controls_are_declared_with_reasons(client):
    unsupported = client.get("/api/nest/info").json()["unsupported"]
    for control in ("topology_editing", "weight_mutation", "learning_controls",
                    "branch_from_replay", "reseed", "stimulation_in_the_past"):
        assert unsupported.get(control), f"{control} must be refused with a stated reason"


def test_mutating_endpoints_are_refused_not_silently_routed(client):
    for path in ("/api/weight", "/api/reseed", "/api/topology"):
        response = client.post(path)
        assert response.status_code == 409, path
        assert response.json()["engine"] == "nest"


def test_state_returns_the_canonical_topology(client):
    state = client.get("/api/state").json()
    assert len(state["topology"]["neurons"]) == 191
    assert len(state["topology"]["synapses"]) == 1052
    assert state["topology"]["params"]["engine"] == "nest"


def test_step_advances_exactly_one_observation_chunk(client):
    from nest_backend.dashboard_api import CHUNK_TICKS

    before = client.get("/api/state").json()["dynamic"]["nest"]["t_ms"]
    after = client.post("/api/step").json()["dynamic"]["nest"]["t_ms"]
    ts_h = client.get("/api/state").json()["dynamic"]["nest"]["resolution_h_ms"]
    assert round(after - before, 9) == round(CHUNK_TICKS * ts_h, 9)


def test_live_frames_use_the_same_shape_as_replay_frames(client):
    """Live and replay must not drift: same fields, same availability declaration."""
    from nest_backend.replay_adapter import STATE_AVAILABILITY

    dynamic = client.post("/api/step").json()["dynamic"]
    assert len(dynamic["neurons"]) == 191
    assert dynamic["changed_synapses"] == []
    assert dynamic["nest"]["state_availability"] == STATE_AVAILABILITY
    assert dynamic["nest"]["live"] is True
    for node in dynamic["neurons"]:
        assert node.get("potential") is None
        assert node.get("activation") is None


def test_reset_rebuilds_the_same_seeded_network(client):
    first = client.post("/api/reset").json()
    for _ in range(3):
        client.post("/api/step")
    second = client.post("/api/reset").json()
    w1 = {s["id"]: s["weight"] for s in first["topology"]["synapses"]}
    w2 = {s["id"]: s["weight"] for s in second["topology"]["synapses"]}
    assert w1 == w2
    assert second["dynamic"]["nest"]["t_ms"] == 0.0


def test_pattern_change_rebuilds_rather_than_rewriting_the_past(client):
    response = client.post("/api/pattern", json={"pattern": "col 1", "row": 1, "col": 1})
    assert response.status_code == 200
    body = response.json()
    assert "REBUILDS" in body["note"]
    assert body["dynamic"]["nest"]["t_ms"] == 0.0
    assert body["dynamic"]["nest"]["patches"] == {"1,1": "col 1"}


def test_only_one_thread_ever_enters_nest(client):
    """Concurrent requests must serialise, not overlap inside `Simulate`."""
    import concurrent.futures

    client.post("/api/reset")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(client.post, "/api/step") for _ in range(6)]
        responses = [f.result() for f in futures]
    assert all(r.status_code == 200 for r in responses)
    times = sorted(r.json()["dynamic"]["nest"]["t_ms"] for r in responses)
    # Six serialised chunks produce six DISTINCT advancing times. An overlap would
    # duplicate or skip one.
    assert len(set(times)) == 6, f"expected six distinct chunk times, got {times}"


def test_speed_is_a_viewing_cadence_not_a_scientific_timestep(client):
    """Whatever the playback speed, one step is always one chunk of NEST time."""
    from nest_backend.dashboard_api import CHUNK_TICKS

    client.post("/api/reset")
    h = client.get("/api/state").json()["dynamic"]["nest"]["resolution_h_ms"]
    advances = []
    previous = client.get("/api/state").json()["dynamic"]["nest"]["t_ms"]
    for _ in range(3):
        now = client.post("/api/step").json()["dynamic"]["nest"]["t_ms"]
        advances.append(round(now - previous, 9))
        previous = now
    assert advances == [round(CHUNK_TICKS * h, 9)] * 3
