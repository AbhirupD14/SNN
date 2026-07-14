"""Local predictive-inhibition trace and predictor rule (Experiment.md Section 6).

This is the ONLY learning rule for the predictive L1I feedback weights. It is fully
local: each L1I owns one scalar trace x_i in [0,1] driven only by its paired L1E
spike, and each delivered L2E->L1I feedback synapse moves toward/away from its cap
according to the coincidence between that trace and the arriving feedback spike. No
gradients, labels, winners, or pattern identities enter.

Kept deliberately small and side-effect-free so it can be unit-tested in isolation
(test_predictor_rule.py) and called once per timestep from SimulationEngine.step().
"""

import numpy as np


def trace_lambda(tau_steps):
    """Per-step multiplicative decay for the local trace: lambda = exp(-1/tau)."""
    return float(np.exp(-1.0 / float(tau_steps)))


def decay_and_set_trace(x, spiked, lam):
    """Section 6.1 trace update for a whole L1I population at once.

        x_i <- lam * x_i ; if paired L1E_i spiked this step: x_i <- 1

    `x` and `spiked` are aligned arrays; returns the new trace array. Pure (does not
    mutate the input)."""
    x = np.asarray(x, dtype=float) * lam
    x[np.asarray(spiked, dtype=float) > 0.5] = 1.0
    return x


def predictor_update(w_fb, x_i, delivered, G, eta_up, eta_down):
    """Section 6.2 predictor update for ONE L1I's feedback weight vector.

    For each feedback source j with a DELIVERED spike (delivered[j] > 0.5), on the
    normalized gate u = w/G (G = theta_L1I - V_rest_L1I):

        u <- clip(u + eta_up*x_i*(1 - u) - eta_down*(1 - x_i)*u, 0, 1)
        w <- G * u

    Sources without a delivered spike do NOT update. `x_i` is this L1I's current
    local trace. The caller passes only the feedback slice, so the fixed local
    afferent never enters this rule. Returns the new feedback weight vector (pure).

    Runs for every delivered feedback spike even when the target L1I is refractory:
    refractoriness may block the membrane deposit in receive_input, but the arriving
    presynaptic event must still teach the predictor (Experiment.md Section 6.2).
    """
    w_fb = np.asarray(w_fb, dtype=float)
    delivered = np.asarray(delivered, dtype=float)
    if G <= 0:
        return w_fb.copy()
    u = w_fb / G
    du = eta_up * x_i * (1.0 - u) - eta_down * (1.0 - x_i) * u
    u_new = np.clip(u + du, 0.0, 1.0)
    active = delivered > 0.5
    w_new = w_fb.copy()
    w_new[active] = G * u_new[active]
    return w_new
