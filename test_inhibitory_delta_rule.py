"""
Tests for the inhibitory-gate plasticity rules (Neuron.apply_inhibition).

Three rules, selected by (inhibitory_delta_rule, inhibitory_rule_mode):
  * saturating (delta_rule=False, legacy)      -> every gate -> sqrt(w_max), uniform
  * turnover   (delta_rule=True, "turnover")   -> DEFAULT: event-local strengthen/decay
                                                  on u=w/G, gates differentiate, no
                                                  target voltage and no averages
  * margin     (delta_rule=True, "margin")     -> diagnostic: relax toward
                                                  s = clamp(v_pre - margin*theta, 0, G)

All are local to one discharge event (this synapse's w, the target's v_pre/theta,
the arriving spike) and share the linear floored-at-rest delivery.

    PYTHONPATH=. .venv/bin/python test_inhibitory_delta_rule.py
"""

from __future__ import annotations

import numpy as np

from neuron_flexible import Neuron


def _gate(n):
    return -float(n.weights[0])


def _make(theta=1.0, cap=1.0, rule=True, mode="turnover", w0=0.5, **kw):
    """A one-synapse neuron with a single inhibitory gate of magnitude w0, gate
    ceiling G = sqrt(cap). kw overrides rule params."""
    n = Neuron(n_inputs=1, threshold=theta, refractory_period=2)
    n.inhibitory_weight_cap = cap
    n.inhibitory_delta_rule = rule
    n.inhibitory_rule_mode = mode
    for k, v in kw.items():
        setattr(n, k, v)
    n.weights = np.array([-w0])
    return n


def _discharge(n, v_pre, times=1):
    """Hold the membrane at v_pre and deliver `times` inhibitory discharges."""
    for _ in range(times):
        n.potential = float(v_pre)
        n.apply_inhibition(np.array([1.0]))


# ---------------------------------------------------------------------------
# 1. Legacy saturating rule preserved
# ---------------------------------------------------------------------------
def test_saturating_rule_preserved():
    n = _make(theta=1.0, cap=1.0, rule=False)   # w_max = 1.0
    n.inhibitory_learning_rate = 0.05
    n.weights = np.array([-0.5])
    _discharge(n, v_pre=0.8)
    # p = 0.8; dw = 0.05*0.8*(1 - 0.25/1.0) = 0.03 -> w = 0.53
    assert np.isclose(_gate(n), 0.53, atol=1e-6), _gate(n)
    print("PASS: saturating rule unchanged (dw = eta*p*(1 - w^2/w_max))")


# ---------------------------------------------------------------------------
# 2. Differentiation: different v_pre -> different learned gate
# ---------------------------------------------------------------------------
def test_turnover_differentiates_by_charge():
    hi = _make(w0=0.1); _discharge(hi, v_pre=0.9, times=1500)   # run to convergence
    lo = _make(w0=0.1); _discharge(lo, v_pre=0.4, times=1500)
    assert _gate(hi) > _gate(lo) + 0.05, (_gate(hi), _gate(lo))
    # Offline check: converged gate matches the per-event fixed point
    # u* = eta_up*p / (eta_down + eta_up*p), G=1 (descriptive, not used by the rule).
    for p, g in ((0.9, _gate(hi)), (0.4, _gate(lo))):
        u_star = 0.02 * p / (0.005 + 0.02 * p)
        assert np.isclose(g, u_star, atol=2e-3), (p, g, u_star)
    print(f"PASS: turnover differentiates by charge (v_pre 0.9 -> {_gate(hi):.3f} > "
          f"0.4 -> {_gate(lo):.3f}), matching the offline fixed point")


def test_turnover_strengthens_more_for_higher_charge_per_event():
    """Single event: a higher-charge target's gate grows more (before turnover
    balances it)."""
    a = _make(w0=0.1); _discharge(a, v_pre=0.9)
    b = _make(w0=0.1); _discharge(b, v_pre=0.3)
    assert (_gate(a) - 0.1) > (_gate(b) - 0.1) > 0, (_gate(a), _gate(b))
    print("PASS: one event strengthens a high-charge gate more than a low-charge one")


# ---------------------------------------------------------------------------
# 3. Low-charge target -> gate drifts down (turnover) / s=0 decays (margin)
# ---------------------------------------------------------------------------
def test_turnover_decays_dead_target():
    n = _make(w0=0.8)                 # start with a strong gate
    _discharge(n, v_pre=0.0, times=1500)  # never carries charge -> only turnover acts
    assert _gate(n) < 0.02, _gate(n)  # -eta_down*u drives it toward 0
    print(f"PASS: turnover decays a never-charged target's gate toward 0 ({_gate(n):.4f})")


def test_margin_below_target_decays_to_zero():
    n = _make(mode="margin", w0=0.6, inhibitory_margin_frac=0.5, inhibitory_delta_eta=0.1)
    _discharge(n, v_pre=0.3, times=200)   # v_pre 0.3 < margin 0.5 -> s = 0
    assert _gate(n) < 0.01, _gate(n)
    print(f"PASS: margin rule with v_pre below target_post gives s=0 -> gate -> 0 ({_gate(n):.4f})")


def test_margin_differentiates_by_charge():
    hi = _make(mode="margin", w0=0.1, inhibitory_margin_frac=0.5, inhibitory_delta_eta=0.1)
    lo = _make(mode="margin", w0=0.1, inhibitory_margin_frac=0.5, inhibitory_delta_eta=0.1)
    _discharge(hi, v_pre=0.9, times=200)   # s = 0.9-0.5 = 0.4
    _discharge(lo, v_pre=0.7, times=200)   # s = 0.7-0.5 = 0.2
    assert _gate(hi) > _gate(lo) + 0.05, (_gate(hi), _gate(lo))
    print(f"PASS: margin rule differentiates (v_pre 0.9 -> {_gate(hi):.3f} > 0.7 -> {_gate(lo):.3f})")


# ---------------------------------------------------------------------------
# 4. Refractory target: no discharge, no update
# ---------------------------------------------------------------------------
def test_refractory_target_no_update():
    for mode in ("turnover", "margin"):
        n = _make(mode=mode, w0=0.4)
        n.refractory_timer = 2
        n.potential = 0.9
        ev = n.apply_inhibition(np.array([1.0]))
        assert ev == [], (mode, ev)
        assert np.isclose(_gate(n), 0.4), (mode, _gate(n))
    # Legacy rule too.
    n = _make(rule=False, w0=0.4); n.refractory_timer = 2; n.potential = 0.9
    assert n.apply_inhibition(np.array([1.0])) == [] and np.isclose(_gate(n), 0.4)
    print("PASS: refractory target receives no discharge and no gate update (all rules)")


# ---------------------------------------------------------------------------
# 5. Signed-spike excitatory update is unchanged by the inhibitory rule
# ---------------------------------------------------------------------------
def test_signed_spike_unaffected_by_inhibitory_rule():
    def fire_once(delta, mode):
        n = Neuron(n_inputs=3, threshold=1.0, refractory_period=0, weight_cap=1.0)
        n.signed_spike_learning = True
        n.min_positive_weight = 0.01
        n.inhibitory_delta_rule = delta
        n.inhibitory_rule_mode = mode
        n.weights = np.array([-0.5, 0.3, 0.3])   # idx0 inhibitory gate, idx1/2 positive
        n.receive_input(np.array([0.0, 1.0, 0.0]))   # only synapse 1 participates
        n.potential = 1.5                            # force a suprathreshold fire
        n.fire()
        return n.weights.copy()

    base = fire_once(False, "turnover")
    for delta, mode in ((True, "turnover"), (True, "margin")):
        assert np.allclose(fire_once(delta, mode), base), (delta, mode)
    # Signed-spike potentiates the participating positive synapse and depresses the
    # non-participating one (-1 signal); it NEVER touches the negative gate.
    assert np.isclose(base[0], -0.5), base            # negative gate untouched by fire
    assert base[1] > 0.3 and base[2] < 0.3, base      # +1 grew, -1 depressed
    print("PASS: signed-spike excitatory fire is identical under every inhibitory rule; "
          "the negative gate is never touched by firing")


def main():
    test_saturating_rule_preserved()
    test_turnover_differentiates_by_charge()
    test_turnover_strengthens_more_for_higher_charge_per_event()
    test_turnover_decays_dead_target()
    test_margin_below_target_decays_to_zero()
    test_margin_differentiates_by_charge()
    test_refractory_target_no_update()
    test_signed_spike_unaffected_by_inhibitory_rule()
    print("\nALL INHIBITORY DELTA-RULE TESTS PASSED")


if __name__ == "__main__":
    main()
