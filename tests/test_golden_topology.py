"""Behavioural regression: all built-in topologies stay bit-exact (per-frame dynamics)
against their captured goldens. This is the oracle protecting the graph-driven
engine refactor -- if the generalized engine changes any neuron's potential, spike,
conductance, trace, or learned weight for a fixed seed+schedule, these fail.

Regenerate intentionally with:
    PYTHONPATH=. .venv/bin/python tests/golden_topology.py capture <name> <topo>
"""

import os

import pytest

from tests.golden_topology import run, digests, TOPOS, GOLDEN_DIR
import json


@pytest.mark.parametrize("name,topo", [("pi_baseline", "pi"), ("old_baseline", "old"),
                                      ("rg_baseline", "rg"),
                                      ("rg_residual_baseline", "rg_residual")])
def test_topology_frames_bit_exact(name, topo):
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    with open(path) as f:
        ref = json.load(f)
    got = digests(run(TOPOS[topo]))
    want = digests(ref)
    assert got["frames"] == want["frames"], (
        f"{name} ({topo}) per-frame dynamics changed; if intended, recapture the golden")
