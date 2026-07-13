"""
Phase 4 (symmetry-breaking plan): inhibitory (negative-weight) plasticity must
update ONLY on a real inhibitory discharge into a NON-refractory target.

A refractory neuron is clamped to rest and does not integrate input, so it must
also not undergo inhibitory discharge or gate learning. apply_inhibition() must,
while refractory: return [] (no events), leave the potential untouched, and leave
the negative gate weight untouched. This preserves the principle that inhibitory
plasticity fires only when inhibition actually reduces charge in an active target.

Covers: fixed-fan-in and dynamic-fan-in construction of a standalone negative gate
(apply_inhibition, still used by L1I->L1E feedback and legacy experiments), each
with an ACTIVE control so the no-op assertions can't pass vacuously. A separate
in-engine test covers the active L2 path, which has NO learned gate: its
competitive reset is unconditional and ignores the refractory timer entirely.

Run:
    PYTHONPATH=. .venv/bin/python test_refractory_gating.py
"""
import numpy as np
from neuron_flexible import Neuron
from backend.simulation import SimulationEngine, L2E_FANIN


def _make_fixed():
    n = Neuron(n_inputs=2, threshold=1.0, inhibitory_learning_rate=0.1)
    n.weights = np.array([-0.5, 0.8])   # index 0 = inhibitory gate (negative)
    return n, "weights"


def _make_dynamic():
    n = Neuron(threshold=1.0, inhibitory_learning_rate=0.1)
    n.add_input_connection(-0.5)
    n.add_input_connection(0.8)
    n.finalize_connections()
    return n, "_weights_array"


def _w(n, attr, i):
    return float(getattr(n, attr)[i])


def _refractory_noop(make, label):
    n, attr = make()
    n.potential = 0.9          # near threshold: would normally drive strong learning
    n.refractory_timer = 2     # refractory
    w_before, p_before = _w(n, attr, 0), float(n.potential)
    events = n.apply_inhibition(np.array([1.0, 0.0]))
    assert events == [], f"{label}: apply_inhibition must return [] during refractory, got {events}"
    assert float(n.potential) == p_before, f"{label}: potential changed during refractory"
    assert _w(n, attr, 0) == w_before, f"{label}: negative weight changed during refractory"
    print(f"  PASS {label}: refractory apply_inhibition is a no-op (V and w unchanged)")


def _active_updates(make, label):
    n, attr = make()
    n.potential = 0.9
    n.refractory_timer = 0     # active
    w_before = _w(n, attr, 0)
    events = n.apply_inhibition(np.array([1.0, 0.0]))
    assert events, f"{label}: expected an inhibitory event when active"
    assert abs(_w(n, attr, 0)) > abs(w_before), f"{label}: gate should strengthen when active"
    print(f"  PASS {label}: active apply_inhibition updates the gate (control)")


def test_fixed_and_dynamic_refractory_gating():
    print("=== unit: fixed + dynamic neuron refractory gating ===")
    _refractory_noop(_make_fixed, "fixed")
    _refractory_noop(_make_dynamic, "dynamic")
    _active_updates(_make_fixed, "fixed")
    _active_updates(_make_dynamic, "dynamic")


def test_in_engine_competitive_reset_ignores_refractory():
    """The active engine has NO learned L2I->L2E gate: L2 competition is the
    unweighted competitive reset. Unlike apply_inhibition (which no-ops on a
    refractory target), a competitive reset is UNCONDITIONAL -- it must clamp even a
    refractory loser to rest and leave its refractory timer untouched (spec Sec 5)."""
    print("=== in-engine: competitive reset is unconditional (refractory too) ===")
    e = SimulationEngine(seed=1)
    n = e.l2.excitatory_neurons[0]
    thr = e.params['threshold_l2']

    # Active L2E has exactly N_PIX positive pixel afferents and no negative gate.
    assert len(n._weights_array) == L2E_FANIN, len(n._weights_array)
    assert not (n._weights_array < 0).any(), "active L2E must have no negative gate"

    # Refractory target: still resets to rest, timer preserved.
    n.refractory_timer = 2
    n.potential = 0.9 * thr
    rec = n.apply_competitive_reset()
    assert float(n.potential) == n.resting_potential, "refractory loser not reset to rest"
    assert n.refractory_timer == 2, "competitive reset must not touch the refractory timer"
    assert rec['v_post'] == n.resting_potential
    print("  PASS refractory: membrane clamped to rest, refractory timer untouched")

    # Active target: also resets to rest.
    n.refractory_timer = 0
    n.potential = 0.9 * thr
    n.apply_competitive_reset()
    assert float(n.potential) == n.resting_potential, "active loser not reset to rest"
    print("  PASS active: membrane clamped to rest")


if __name__ == "__main__":
    test_fixed_and_dynamic_refractory_gating()
    test_in_engine_competitive_reset_ignores_refractory()
    print("ALL REFRACTORY-GATING TESTS PASSED")
