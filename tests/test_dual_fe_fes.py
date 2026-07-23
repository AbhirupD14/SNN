"""Focused algebra + state-timing proofs for the experimental dual node/synapse
free-energy (dual FE/FES) learning rule (`dual_fe_fes` for ordinary/latency E,
`c_dual_fe_fes` for a coincidence basal), and the one dashboard flag that switches both.

The equations under test (full precision):

    FE     = e   + (1 - e)   / (1 + B * ((Iaccq/theta)   - 0.5)^2)     # node, shared
    FES_i  = wte + (1 - wte) / (1 + B * ((2*wt_i/theta)   - 0.5)^2)    # per synapse i
    dw_i   = LR * FE * FES_i * input_signal_i * influence_i
    wt_i  <- max(wte, wt_i + dw_i)                                     # floor wte, NO cap

Reference: e = wte = 0.001, B = 5.0.
"""

import math

import numpy as np
import pytest

from snn.neurons import (ExcitatoryNeuron, CoincidencePyramidalNeuron,
                         dual_fe, dual_fes, dual_fe_c, dual_fes_c, E_THRESHOLD,
                         DUAL_E_DEFAULT, DUAL_WTE_DEFAULT, DUAL_B_DEFAULT)
from backend.simulation import SimulationEngine

THETA = E_THRESHOLD
E, WTE, B = DUAL_E_DEFAULT, DUAL_WTE_DEFAULT, DUAL_B_DEFAULT


def make_dual_e(w, *, dist=None, eta=0.01, iaccq=None, theta=THETA,
                dual_e=E, dual_wte=WTE, dual_B=B):
    w = np.asarray(w, dtype=float)
    dist = np.ones_like(w) if dist is None else np.asarray(dist, dtype=float)
    n = ExcitatoryNeuron('E', 'test', acc_weights=w.copy(), acc_distance_factor=dist,
                         eta=eta, threshold=theta, learn=True, update_mode='dual_fe_fes',
                         dual_e=dual_e, dual_wte=dual_wte, dual_B=dual_B, leak_rate=0.0)
    n.iaccq = float(w.sum() if iaccq is None else iaccq)
    return n


# ===================================================================== 1 & 2
def test_fe_is_exactly_one_at_half_theta():
    assert dual_fe(0.5 * THETA, THETA, E, B) == 1.0                # FE(0.5) == 1 exactly
    # ... for any B and any e, since the quadratic term is exactly zero there.
    for bb in (0.0, 1.0, 5.0, 50.0):
        assert dual_fe(0.5 * THETA, THETA, 0.2, bb) == pytest.approx(1.0, abs=0.0)


def test_fes_is_exactly_one_at_quarter_theta():
    assert dual_fes(0.25 * THETA, THETA, WTE, B) == 1.0           # FES(0.25) == 1 exactly
    for bb in (0.0, 1.0, 5.0, 50.0):
        assert dual_fes(0.25 * THETA, THETA, 0.2, bb) == pytest.approx(1.0, abs=0.0)


# ========================================================================= 3
def test_reference_values_at_endpoints_and_overshoot_match_equations():
    def fe(r):
        return E + (1 - E) / (1 + B * (r - 0.5) ** 2)

    def fes(r):
        return WTE + (1 - WTE) / (1 + B * (2 * r - 0.5) ** 2)

    # FE at Iaccq/theta in {0, 0.5, 1, 2 (overshoot)}.
    assert dual_fe(0.0, THETA) == pytest.approx(fe(0.0))
    assert dual_fe(0.5 * THETA, THETA) == pytest.approx(fe(0.5)) == pytest.approx(1.0)
    assert dual_fe(1.0 * THETA, THETA) == pytest.approx(fe(1.0))
    assert dual_fe(2.0 * THETA, THETA) == pytest.approx(fe(2.0))   # overshoot retained
    # FES at w/theta in {0, 0.25, 0.5, 1 (overshoot)}.
    assert dual_fes(0.0, THETA) == pytest.approx(fes(0.0))
    assert dual_fes(0.25 * THETA, THETA) == pytest.approx(1.0)
    assert dual_fes(0.5 * THETA, THETA) == pytest.approx(fes(0.5))
    assert dual_fes(1.0 * THETA, THETA) == pytest.approx(fes(1.0))
    # B=5 does NOT reach the asymptotic floor e/wte at these finite endpoints.
    assert dual_fe(0.0, THETA) > E + 0.1
    assert dual_fes(0.0, THETA) > WTE + 0.1


# ========================================================================= 4
def test_fe_uses_captured_pre_reset_iaccq_not_post_reset_state():
    # Engine event path: the winner's logged Iaccq is the pre-reset frozen drive (the sum of
    # its participating PRE-update afferent weights), NOT the post-reset membrane (0).
    e = SimulationEngine(seed=1, topology='rg_direct_cc4', leak_rate=0.0,
                         refractory_steps=0, dual_fe_fes=True)
    for c in e.latency_competitors:
        c.record_updates = True
    e.set_pattern('row 1')                                        # pixels 3,4,5
    winner_entry = None
    for _ in range(40):
        e.step()
        for c in e.latency_competitors:
            if c.update_log:
                winner_entry = c.update_log[-1]
        if winner_entry is not None and e.winner is not None:
            w = e.neurons[e.winner]
            assert w.V == 0.0                                     # post-reset membrane is rest
            break
    assert winner_entry is not None
    pre_w = np.array(winner_entry['pre_w'])
    signal = np.array(winner_entry['signal'])
    participating = signal > 0
    # Iaccq equals the frozen delivered packet = sum of participating pre-update weights.
    assert winner_entry['iaccq'] == pytest.approx(float(pre_w[participating].sum()), abs=1e-6)
    assert winner_entry['iaccq'] > 0.0                            # not the post-reset 0
    # FE was computed from that captured Iaccq.
    assert winner_entry['fe'] == pytest.approx(
        dual_fe(winner_entry['iaccq'], THETA, E, B))


# ========================================================================= 5
def test_each_synapse_uses_its_own_pre_update_fes_in_a_vector_update():
    # Weights are NOT symmetric about the FES peak (theta/4), so all three FES differ.
    w = np.array([0.10 * THETA, 0.30 * THETA, 0.50 * THETA])
    n = make_dual_e(w, eta=0.02, iaccq=0.5 * THETA)              # FE == 1 so it isolates FES
    n.record_updates = True
    n.fire()
    n.iaccq = 0.5 * THETA
    n.update_acc_weights(np.array([True, True, True]))
    log = n.update_log[-1]
    for i in range(3):
        fes_i = dual_fes(w[i], THETA, WTE, B)
        assert log['fes'][i] == pytest.approx(fes_i)
        exp = 0.02 * 1.0 * fes_i * 1.0 * 1.0                     # eta*FE*FES_i*signal*influence
        assert log['applied_dw'][i] == pytest.approx(exp)
    # The three synapses moved by three different amounts (own FES each).
    assert len(set(np.round(log['applied_dw'], 9))) == 3


# ========================================================================= 6
def test_participating_and_nonparticipating_move_in_opposite_directions():
    n = make_dual_e([0.25 * THETA, 0.25 * THETA], eta=0.02, iaccq=0.5 * THETA)
    before = n.acc_weights.copy()
    n.fire()
    n.iaccq = 0.5 * THETA
    n.update_acc_weights(np.array([True, False]))
    assert n.acc_weights[0] > before[0]                          # participating potentiates
    assert n.acc_weights[1] < before[1]                          # absent depresses (never +)


# ========================================================================= 7
def test_negative_update_clips_at_wte_not_below():
    n = make_dual_e([0.25 * THETA, 0.25 * THETA], eta=5000.0, iaccq=0.5 * THETA)
    n.fire()
    n.iaccq = 0.5 * THETA
    n.update_acc_weights(np.array([True, False]))                # afferent 1 depressed hard
    assert n.acc_weights[1] == pytest.approx(WTE)                # floor is the raw wte
    assert n.acc_weights[1] >= WTE                               # never below wte


# ========================================================================= 8
def test_large_positive_update_is_not_clipped_by_any_historical_cap():
    # theta/2 == the old E ceiling; a big dual step must sail past it (no hidden w_max clip).
    n = make_dual_e([0.25 * THETA], eta=500.0, iaccq=0.5 * THETA)   # FE=1, FES=1, huge step
    n.fire()
    n.iaccq = 0.5 * THETA
    n.update_acc_weights(np.array([True]))
    assert n.acc_weights[0] > 0.5 * THETA                        # exceeds theta/2
    assert n.acc_weights[0] == pytest.approx(0.25 * THETA + 500.0 * 1.0 * 1.0)
    assert math.isfinite(n.acc_weights[0])


# ========================================================================= 9
def make_dual_c(w=0.25 * THETA, *, eta_c=0.005, phi=1.0, theta=THETA):
    return CoincidencePyramidalNeuron('C', 'B', 'b0', apical_sources=['A'],
                                      apical_edge_ids=['a0'], basal_weight=w,
                                      w_max=4 * theta, eta_c=eta_c, use_fe=True,
                                      update_mode='c_dual_fe_fes', threshold=theta,
                                      basal_distance_factor=phi, leak_rate=0.0)


def test_c_basal_uses_coincidence_fe_fes_and_requires_valid_apical_coincidence():
    theta = THETA
    w0, eta_c, phi = 0.30 * theta, 0.01, 0.8
    # No apical coincidence -> A=0 -> no learning.
    c = make_dual_c(w0, eta_c=eta_c, phi=phi)
    c.apical_active = False
    c._deposit_signal = 1.0
    c.v_pre = 1.2 * theta
    assert c.update_basal_weight() == pytest.approx(w0)          # A=0 => dw=0
    # Valid apical coincidence -> the C-specific FE(Iaccq)/FES(w) references (peaks at
    # Iaccq=theta and w=theta/2), NOT the ordinary-E peaks.
    c = make_dual_c(w0, eta_c=eta_c, phi=phi)
    c.record_updates = True
    c.apical_active = True                                       # Boolean apical eligibility
    c._deposit_signal = 1.0
    c.v_pre = 1.2 * theta                                        # pre-reset accumulated basal charge
    c.update_basal_weight()
    fe = dual_fe_c(1.2 * theta, theta, E, B)
    fes = dual_fes_c(w0, theta, WTE, B)
    exp = w0 + eta_c * fe * fes * 1.0 * 1.0 * phi                # eta_c*FE*FES*A*s*phi
    assert c.basal_weight == pytest.approx(exp)
    log = c.update_log[-1]
    assert log['iaccq'] == pytest.approx(1.2 * theta) and log['fe'] == pytest.approx(fe)
    assert log['fes'] == pytest.approx(fes)
    # differs from the ordinary-E references (which would peak elsewhere)
    assert dual_fe_c(theta, theta) == pytest.approx(1.0)         # FE==1 at the firing charge theta
    assert dual_fes_c(0.5 * theta, theta) == pytest.approx(1.0)  # FES==1 at the weight midpoint


def test_c_basal_floor_is_wte_no_cap():
    theta = THETA
    # A hard depression (negative s) floors at wte, never below.
    c = make_dual_c(0.25 * theta, eta_c=5000.0)
    c.apical_active = True
    c._deposit_signal = -1.0                                     # depressing basal signal
    c.v_pre = 0.5 * theta
    c.update_basal_weight()
    assert c.basal_weight == pytest.approx(WTE)
    # A huge potentiation is not clipped by any cap.
    c = make_dual_c(0.25 * theta, eta_c=5000.0)
    c.apical_active = True
    c._deposit_signal = 1.0
    c.v_pre = 0.5 * theta
    c.update_basal_weight()
    assert c.basal_weight > theta                                # well past any historical C cap


# ======================================================================== 10
def test_flag_off_e_and_c_traces_are_the_unchanged_production_rule():
    # Ordinary-E: the default (flag-off) mode is the production budget-FE rule, and its
    # logged trace matches the hand-computed budget update -- adding dual params/iaccq
    # capture perturbs nothing.
    n = ExcitatoryNeuron('E', 't', acc_weights=np.array([100.0, 100.0]),
                         acc_distance_factor=np.array([1.0, 0.5]), eta=0.01,
                         threshold=1000.0, leak_rate=0.0)
    assert n.update_mode == 'linear_fe'
    n.record_updates = True
    n.fire()
    n.update_acc_weights(np.array([True, False]))
    fe = 1.10 * 1000.0 - 200.0
    exp = np.array([100.0 + 0.01 * fe * (+1) * 1.0, 100.0 + 0.01 * fe * (-1) * 0.5])
    assert n.acc_weights == pytest.approx(exp)                   # budget-FE rule, unchanged
    assert 'fe_pre' in n.update_log[-1] and 'iaccq' not in n.update_log[-1]
    # Coincidence: default c_linear_bounded basal update unchanged.
    c = CoincidencePyramidalNeuron('C', 'B', 'b0', apical_sources=['A'], apical_edge_ids=['a0'],
                                   basal_weight=505.0, w_max=550.0, eta_c=0.001,
                                   update_mode='c_linear_bounded', threshold=1000.0,
                                   leak_rate=0.0)
    c.apical_active = True
    c._deposit_signal = 1.0
    fe_c = 1.10 * 1000.0 - 505.0
    assert c.update_basal_weight() == pytest.approx(min(550.0, 505.0 + 0.001 * fe_c))


# ==================================================================== validation
@pytest.mark.parametrize('bad', [dict(dual_e=0.0), dict(dual_e=1.5),
                                 dict(dual_wte=0.0), dict(dual_wte=1.0001),
                                 dict(dual_B=-1.0), dict(dual_B=float('inf'))])
def test_invalid_dual_params_rejected(bad):
    with pytest.raises(ValueError):
        make_dual_e([0.25 * THETA], **bad)


def test_engine_rejects_bad_dual_params():
    for bad in (dict(dual_fe_e=0.0), dict(dual_fe_wte=2.0), dict(dual_fe_B=-1.0)):
        with pytest.raises(ValueError):
            SimulationEngine(topology='rg_direct_cc4', dual_fe_fes=True, **bad)
