"""Guards for the dashboard replay-player test fixture.

The browser player is validated in JavaScript (tests/replay.parser.test.mjs,
run with `node --test`). These Python checks keep that fixture honest: the
committed replay.snn.jsonl must still parse with the recorder's own readers, and
the committed expected.json must equal the recorder's reconstruction of it (so a
schema change that forgets to regenerate the fixture fails here loudly).
"""

from __future__ import annotations

import json
import os

import pytest

from experiments.replay_recorder import (
    read_records, reconstruct_weights_at, REC_HEADER, REC_FRAME,
    REPLAY_SCHEMA_NAME, REPLAY_SCHEMA_VERSION,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "replay_fixture.snn.jsonl")
EXPECTED = os.path.join(HERE, "fixtures", "replay_fixture.expected.json")


@pytest.fixture(scope="module")
def records():
    assert os.path.exists(FIXTURE), (
        "replay fixture missing; regenerate with "
        "PYTHONPATH=. python tests/fixtures/make_replay_fixture.py")
    return read_records(FIXTURE)


def test_fixture_header_is_supported_schema(records):
    header = records[0]
    assert header["record"] == REC_HEADER
    assert header["schema"] == REPLAY_SCHEMA_NAME
    assert header["schema_version"] == REPLAY_SCHEMA_VERSION
    # self-contained: initial weights live in the header topology
    assert any(s.get("weight") is not None for s in header["topology"]["synapses"])


def test_fixture_has_every_record_kind(records):
    kinds = {r["record"] for r in records}
    assert {"header", "frame", "marker", "weight_checkpoint", "result"} <= kinds


def test_expected_reconstruction_matches_committed_fixture(records):
    """The committed expected.json must equal the recorder's reconstruction of the
    committed .jsonl at every frame (this is what the JS test checks JS against)."""
    with open(EXPECTED, encoding="utf-8") as f:
        expected = json.load(f)
    frame_indices = [r["frame_index"] for r in records if r["record"] == REC_FRAME]
    assert frame_indices, "fixture has no frames"
    assert {str(fi) for fi in frame_indices} == set(expected), \
        "expected.json frame set is stale; regenerate the fixture"
    for fi in frame_indices:
        got = reconstruct_weights_at(records, fi)
        exp = expected[str(fi)]
        assert set(got) == set(exp), f"frame {fi}: synapse id set drifted"
        for sid, w in exp.items():
            assert got[sid] == pytest.approx(w, abs=1e-9), f"frame {fi} synapse {sid}"
