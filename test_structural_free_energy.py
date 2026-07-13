"""Focused tests for the structural free-energy plasticity gate.

Verifies:
  1. flag OFF preserves the exact signed-spike behavior (p-scaled);
  2. the gate shrinks as the positive-afferent sum approaches theta;
  3. the gate uses ONLY positive afferents (the negative inhibitory gate is
     ignored) and only this neuron's own threshold -- no voltage, no rivals;
  4. a mature neuron (sum >= theta) learns at eta_floor, an under-built one at ~1;
  5. the gate is applied only through the signed-spike (excitatory L2E) path.

Plain-script style (matches test_l2_competition.py etc.):
    PYTHONPATH=. .venv/bin/python test_structural_free_energy.py
"""

import numpy as np

from neuron_flexible import Neuron
from snn.rules import bounded_signed_update


def _make_neuron(weights, theta=8.0, cap=8.0 / 3, floor=0.001, lr=0.02):
    n = Neuron(threshold=theta, refractory_period=0, weight_cap=cap, learning_rate=lr)
    n.weights = np.asarray(weights, dtype=float)
    n.min_positive_weight = floor
    n.signed_spike_learning = True
    return n


def test_gate_monotonic_in_sum():
    """gate = max(eta_floor, 1 - clamp(sum/theta,0,1)) -- decreasing in the sum,
    floored, and computed from POSITIVE afferents only."""
    theta = 8.0
    floor = 0.05
    # sum well below theta -> gate ~ 1
    low = _make_neuron([0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0], theta=theta)
    low.structural_free_energy = True
    low.structural_fe_eta_floor = floor
    g_low = low._structural_free_energy_gate()
    assert abs(g_low - (1.0 - 0.3 / theta)) < 1e-9, g_low

    # sum == theta exactly -> maturity 1 -> gate == floor
    mid = _make_neuron([8.0 / 3, 8.0 / 3, 8.0 / 3, 0, 0, 0, 0, 0, 0], theta=theta)
    mid.structural_free_energy = True
    mid.structural_fe_eta_floor = floor
    assert abs(mid._structural_free_energy_gate() - floor) < 1e-9, mid._structural_free_energy_gate()

    # sum > theta -> still clamped at maturity 1 -> gate == floor
    hi = _make_neuron([8.0 / 3, 8.0 / 3, 8.0 / 3, 2.0, 0, 0, 0, 0, 0], theta=theta)
    hi.structural_free_energy = True
    hi.structural_fe_eta_floor = floor
    assert abs(hi._structural_free_energy_gate() - floor) < 1e-9

    assert g_low > mid._structural_free_energy_gate()
    print(f"PASS: gate monotonic and floored (low={g_low:.4f}, mature={floor:.4f})")


def test_negative_gate_ignored():
    """A negative inhibitory afferent (e.g. L2I->L2E) must not enter the sum."""
    n = _make_neuron([2.0, 2.0, -5.0, 0, 0, 0, 0, 0, 0], theta=8.0)
    n.structural_free_energy = True
    assert abs(n._positive_afferent_weight_sum() - 4.0) < 1e-9
    # sum uses 4.0 (=2+2), not -1.0 (=2+2-5): maturity = 4/8 = 0.5, gate = 0.5
    assert abs(n._structural_free_energy_gate() - 0.5) < 1e-9
    print("PASS: negative inhibitory gate excluded from the structural sum")


def test_flag_off_identical():
    """With the flag off, one fire must produce the exact p-scaled signed update."""
    w0 = [1.0, 1.0, 0.5, 0.2, 0, 0, 0, 0, 0]
    theta, cap, floor, lr = 8.0, 8.0 / 3, 0.001, 0.02
    v_pre = 10.0
    p = min(max(theta / v_pre, 0.0), 1.0)
    participating = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    n = _make_neuron(w0, theta=theta, cap=cap, floor=floor, lr=lr)
    n._last_input_spikes = participating.copy()
    n.structural_free_energy = False
    # drive the update directly with a known v_pre
    n._update_weights(v_pre)
    got = n.weights

    # hand-computed reference via the shared bounded kernel (gain = eta*p)
    ref = np.asarray(w0, dtype=float)
    pos = ref > 0
    sig = np.where(participating[pos] > 0.5, 1.0, -1.0)
    ref[pos] = bounded_signed_update(ref[pos], floor, cap, lr * p, sig)
    assert np.allclose(got, ref), (got, ref)
    print("PASS: flag OFF reproduces the exact p-scaled signed-spike update")


def test_gate_replaces_p():
    """With the flag ON, the update uses eta*gate (NOT eta*p) as the gain."""
    w0 = [1.0, 1.0, 0.5, 0.2, 0, 0, 0, 0, 0]
    theta, cap, floor, lr = 8.0, 8.0 / 3, 0.001, 0.02
    v_pre = 10.0  # p = 0.8; different from the structural gate so we can tell them apart
    participating = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    n = _make_neuron(w0, theta=theta, cap=cap, floor=floor, lr=lr)
    n._last_input_spikes = participating.copy()
    n.structural_free_energy = True
    n.structural_fe_eta_floor = 0.02
    gate = n._structural_free_energy_gate()  # computed on the PRE-update weights
    n._update_weights(v_pre)
    got = n.weights

    ref = np.asarray(w0, dtype=float)
    pos = ref > 0
    sig = np.where(participating[pos] > 0.5, 1.0, -1.0)
    ref[pos] = bounded_signed_update(ref[pos], floor, cap, lr * gate, sig)
    assert np.allclose(got, ref), (got, ref)
    # and confirm it is genuinely different from the p-scaled path
    p = theta / v_pre
    assert abs(gate - p) > 1e-3, (gate, p)
    print(f"PASS: flag ON uses eta*gate (gate={gate:.4f}) not eta*p (p={p:.4f})")


def test_only_signed_path():
    """The gate is inert unless signed_spike_learning is the active rule: on the
    legacy charge path, toggling structural_free_energy must change nothing."""
    w0 = [2.0, 2.0, 2.0, 0, 0, 0, 0, 0, 0]
    part = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)

    off = _make_neuron(w0, theta=8.0)
    off.signed_spike_learning = False        # legacy charge rule
    off.structural_free_energy = False
    off._last_input_spikes = part.copy()
    off._update_weights(10.0)

    on = _make_neuron(w0, theta=8.0)
    on.signed_spike_learning = False         # legacy charge rule
    on.structural_free_energy = True         # must be ignored on this path
    on._last_input_spikes = part.copy()
    on._update_weights(10.0)

    assert np.allclose(off.weights, on.weights), (off.weights, on.weights)
    print("PASS: structural gate does not hijack the non-signed (legacy) path")


if __name__ == "__main__":
    test_gate_monotonic_in_sum()
    test_negative_gate_ignored()
    test_flag_off_identical()
    test_gate_replaces_p()
    test_only_signed_path()
    print("\nALL STRUCTURAL FREE-ENERGY TESTS PASSED")
