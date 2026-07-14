"""Isolated unit test for the Section 6 local predictor rule (Experiment.md).

Tests snn/rules/predictive.py directly -- no engine, no schedule -- so the rule's
correctness is pinned independently of the rest of the network:

  - active local trace (x=1) INCREASES only the delivered feedback weights;
  - zero local trace (x=0) DECREASES only the delivered feedback weights;
  - sources without a delivered spike never move;
  - the local afferent never enters (the rule only ever sees the feedback slice);
  - weights clamp to [0, G] under repeated updates from either extreme;
  - the trace decays by exp(-1/tau) and resets to 1 on a paired spike.

    PYTHONPATH=. .venv/bin/python test_predictor_rule.py
"""
import numpy as np

import neuron_flexible  # noqa: F401  import first: resolves the snn<->neuron_flexible cycle
from snn.rules.predictive import trace_lambda, decay_and_set_trace, predictor_update

G = 2666.6666666666665      # a representative theta_L1I (V_rest=0 so G=theta)
ETA_UP, ETA_DOWN = 0.08, 0.04


def test_active_trace_increases_delivered_only():
    w = np.full(4, 0.10 * G)
    delivered = np.array([1.0, 0.0, 1.0, 0.0])
    w2 = predictor_update(w, x_i=1.0, delivered=delivered, G=G,
                          eta_up=ETA_UP, eta_down=ETA_DOWN)
    # delivered (0, 2) increased; undelivered (1, 3) unchanged.
    assert w2[0] > w[0] and w2[2] > w[2], "delivered weights did not increase under x=1"
    assert w2[1] == w[1] and w2[3] == w[3], "undelivered weights moved"
    # exact value: u=0.10, du=0.08*(1-0.10)=0.072 -> u=0.172 -> w=0.172*G
    assert np.isclose(w2[0], (0.10 + ETA_UP * (1 - 0.10)) * G)
    print("  x=1 increases ONLY delivered feedback weights (exact): OK")


def test_zero_trace_decreases_delivered_only():
    w = np.full(4, 0.10 * G)
    delivered = np.array([1.0, 0.0, 1.0, 0.0])
    w2 = predictor_update(w, x_i=0.0, delivered=delivered, G=G,
                          eta_up=ETA_UP, eta_down=ETA_DOWN)
    assert w2[0] < w[0] and w2[2] < w[2], "delivered weights did not decrease under x=0"
    assert w2[1] == w[1] and w2[3] == w[3], "undelivered weights moved"
    # u=0.10, du=-0.04*0.10=-0.004 -> u=0.096 -> w=0.096*G
    assert np.isclose(w2[0], (0.10 - ETA_DOWN * 0.10) * G)
    print("  x=0 decreases ONLY delivered feedback weights (exact): OK")


def test_clamp_to_0_G():
    # From the top: many x=1 updates never exceed G.
    w = np.full(3, 0.99 * G)
    for _ in range(200):
        w = predictor_update(w, 1.0, np.ones(3), G, ETA_UP, ETA_DOWN)
    assert np.all(w <= G + 1e-9) and np.all(w >= 0.0), "exceeded [0, G] going up"
    assert np.allclose(w, G), "did not converge to the G cap under sustained x=1"
    # From the bottom: many x=0 updates never go below 0.
    w = np.full(3, 0.01 * G)
    for _ in range(500):
        w = predictor_update(w, 0.0, np.ones(3), G, ETA_UP, ETA_DOWN)
    assert np.all(w >= 0.0) and np.all(w <= G), "left [0, G] going down"
    assert np.all(w < 0.01 * G), "did not decay toward 0 under sustained x=0"
    print("  weights clamp to [0, G] from both extremes: OK")


def test_partial_trace_direction():
    # Intermediate trace: sign of du flips at the fixed point u* where
    # eta_up*x*(1-u) == eta_down*(1-x)*u. For x=0.5: eta_up*(1-u)=eta_down*u ->
    # u* = eta_up/(eta_up+eta_down) = 0.08/0.12 = 0.6667.
    x = 0.5
    u_star = ETA_UP / (ETA_UP + ETA_DOWN)
    w_low = np.array([0.3 * G]); w_high = np.array([0.9 * G])
    up = predictor_update(w_low, x, np.ones(1), G, ETA_UP, ETA_DOWN)
    dn = predictor_update(w_high, x, np.ones(1), G, ETA_UP, ETA_DOWN)
    assert up[0] > w_low[0], "below the fixed point should rise"
    assert dn[0] < w_high[0], "above the fixed point should fall"
    assert np.isclose(u_star, 0.6666666666, atol=1e-6)
    print(f"  partial trace x=0.5 converges toward u*={u_star:.4f}: OK")


def test_trace_decay_and_set():
    lam = trace_lambda(2.0)
    assert np.isclose(lam, np.exp(-0.5))
    x = np.array([0.5, 0.5, 0.0])
    x2 = decay_and_set_trace(x, spiked=[1, 0, 0], lam=lam)
    assert x2[0] == 1.0, "paired spike must set trace to 1"
    assert np.isclose(x2[1], 0.5 * lam) and np.isclose(x2[2], 0.0)
    # input not mutated (pure)
    assert x[0] == 0.5
    print("  trace decays by exp(-1/tau) and resets to 1 on a paired spike: OK")


def main():
    test_active_trace_increases_delivered_only()
    test_zero_trace_decreases_delivered_only()
    test_clamp_to_0_G()
    test_partial_trace_direction()
    test_trace_decay_and_set()
    print("PASS: local predictor rule (Section 6)")


if __name__ == "__main__":
    main()
