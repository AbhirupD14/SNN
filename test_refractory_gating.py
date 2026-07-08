"""
Phase 4 (symmetry-breaking plan): inhibitory (negative-weight) plasticity must
update ONLY on a real inhibitory discharge into a NON-refractory target.

A refractory neuron is clamped to rest and does not integrate input, so it must
also not undergo inhibitory discharge or gate learning. apply_inhibition() must,
while refractory: return [] (no events), leave the potential untouched, and leave
the negative gate weight untouched. This preserves the principle that inhibitory
plasticity fires only when inhibition actually reduces charge in an active target.

Covers: fixed-fan-in construction, dynamic fan-in construction, and the
in-engine L2I->L2E gate. Each also has an ACTIVE control showing the gate DOES
update when the target is not refractory, so the no-op assertions can't pass
vacuously.

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


def test_in_engine_gate_refractory():
    print("=== in-engine: L2I->L2E gate does not update when target is refractory ===")
    e = SimulationEngine(seed=1)
    n = e.l2.excitatory_neurons[0]
    thr = e.params['threshold_l2']
    inh_spk = np.zeros(L2E_FANIN)
    inh_spk[0] = 1.0                       # index 0 = the L2I->L2E gate line

    # Refractory target: the gate must NOT update.
    n.refractory_timer = 2
    n.potential = 0.9 * thr
    w_before = float(n._weights_array[0])
    ev = n.apply_inhibition(inh_spk)
    assert ev == [], f"in-engine: expected no events on refractory target, got {ev}"
    assert float(n._weights_array[0]) == w_before, "in-engine: gate updated while target refractory"
    print("  PASS refractory: gate weight unchanged, no discharge event")

    # Active target: the gate DOES update (control), proving the no-op is real.
    n.refractory_timer = 0
    n.potential = 0.9 * thr
    w_before2 = float(n._weights_array[0])
    ev2 = n.apply_inhibition(inh_spk)
    assert ev2, "in-engine: expected a discharge event on an active target"
    assert abs(float(n._weights_array[0])) > abs(w_before2), "in-engine: gate should strengthen when active"
    print("  PASS active: gate strengthened on a real discharge (control)")


if __name__ == "__main__":
    test_fixed_and_dynamic_refractory_gating()
    test_in_engine_gate_refractory()
    print("ALL REFRACTORY-GATING TESTS PASSED")
