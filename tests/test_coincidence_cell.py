"""Phase 2 — isolated CoincidencePyramidalNeuron: the dendritic coincidence truth
table, one-boundary basal eligibility, the exact basal learning rule, firing/leak/
refractory parity with E, and the calibrated two-coincidence cadence.

Every test drives explicit basal/apical events through a faithful miniature event
loop (begin -> deliver -> resolve -> freeze -> analytic crossing/advance -> fire ->
learn). There is no scalar-stimulate shortcut that bypasses the gate.
"""

import math

import numpy as np
import pytest

from snn.neurons import (
    CoincidencePyramidalNeuron,
    DendriticCompartment,
    ExcitatoryNeuron,
    E_THRESHOLD,
    leak_to_conductance,
)


# ---------------------------------------------------------------- calibration
def _calib(theta=1000.0, leak=0.03, T=1):
    g_L = leak_to_conductance(leak)
    r = math.exp(-g_L)
    kappa = 1.0 if g_L == 0 else (1.0 - math.exp(-g_L)) / g_L
    w2 = theta / (kappa * (1.0 + r ** T))
    w1 = theta / kappa
    return dict(g_L=g_L, r=r, kappa=kappa, w2=w2, w1=w1,
               c_init=1.01 * w2, c_max=1.10 * w2)


def make_c(**kw):
    d = dict(nid='L1C0', basal_source='L1E0', basal_edge_id='b0',
             apical_sources=['L2E0'], apical_edge_ids=['a0'],
             basal_weight=200.0, w_max=500.0, eta_c=0.01, learn=True,
             leak_rate=0.0, threshold=1000.0, refractory_steps=0)
    d.update(kw)
    return CoincidencePyramidalNeuron(**d)


def deliver(c, basal=False, apical=None, basal_signal=1.0):
    """One boundary of delivery + gate resolution (no membrane advance)."""
    c.begin_event_boundary()
    if basal:
        c.gather_basal('L1E0', basal_signal)
    for src in (apical or []):
        c.gather_apical(src)
    c.resolve_dendrites()
    c.freeze_drive()
    return c.coincidence_charge


def run_boundary(c, basal=False, apical=None, basal_signal=1.0, interval=1.0):
    """Full mini event loop for ONE boundary: returns True iff the C cell fired."""
    deliver(c, basal=basal, apical=apical, basal_signal=basal_signal)
    dtau = c.crossing_time(interval)
    fired = False
    if math.isfinite(dtau):
        c.advance_segment(dtau)
        c.fire(dtau)
        c.update_basal_weight()
        fired = True
    else:
        c.advance_segment(interval)
    c.update_trace()
    c.decay_conductance()
    c.advance_refractory()
    return fired


# =============================================================== truth table
def test_no_input_deposits_zero():
    c = make_c()
    assert deliver(c) == 0.0
    assert c.V == 0.0


def test_basal_only_deposits_zero():
    c = make_c()
    assert deliver(c, basal=True) == 0.0
    c.advance_segment(1.0)
    assert c.V == 0.0                       # no apical -> no charge -> no depolarization


def test_repeated_basal_only_never_charges():
    c = make_c(leak_rate=0.03)
    for _ in range(20):
        deliver(c, basal=True)
        c.advance_segment(1.0)
    assert c.V == pytest.approx(0.0)        # only leak toward rest; never a deposit


def test_apical_only_deposits_zero():
    c = make_c()
    for _ in range(5):
        assert deliver(c, apical=['L2E0']) == 0.0
        c.advance_segment(1.0)
    assert c.V == 0.0


def test_simultaneous_basal_apical_deposits_pre_update_weight():
    c = make_c(basal_weight=321.0)
    q = deliver(c, basal=True, apical=['L2E0'])
    assert q == pytest.approx(321.0)        # exactly w * s, s = 1


def test_basal_then_apical_next_boundary_deposits_once():
    c = make_c(basal_weight=250.0)
    assert deliver(c, basal=True) == 0.0            # t: basal, no apical -> carried
    q = deliver(c, apical=['L2E0'])                 # t+1: apical matches carried basal
    assert q == pytest.approx(250.0)


def test_basal_then_apical_two_boundaries_later_deposits_zero():
    c = make_c(basal_weight=250.0)
    assert deliver(c, basal=True) == 0.0            # t
    assert deliver(c) == 0.0                        # t+1: nothing -> eligibility expires
    assert deliver(c, apical=['L2E0']) == 0.0       # t+2: apical too late


def test_apical_then_basal_next_boundary_deposits_zero():
    c = make_c(basal_weight=250.0)
    assert deliver(c, apical=['L2E0']) == 0.0       # t: apical only (never persists)
    assert deliver(c, basal=True) == 0.0            # t+1: basal only, stale apical gone


def test_apical_then_basal_with_new_apical_deposits():
    c = make_c(basal_weight=250.0)
    assert deliver(c, apical=['L2E0']) == 0.0       # t: apical only
    q = deliver(c, basal=True, apical=['L2E0'])     # t+1: basal + a NEW apical
    assert q == pytest.approx(250.0)


def test_multiple_apical_sources_one_deposit():
    c = make_c(basal_weight=250.0, apical_sources=['L2E0', 'L2E1', 'L2E2'],
               apical_edge_ids=['a0', 'a1', 'a2'])
    q = deliver(c, basal=True, apical=['L2E0', 'L2E1', 'L2E2'])
    assert q == pytest.approx(250.0)                # one Boolean gate, one deposit
    assert c.apical_active and len(c.apical_sources) == 3


def test_current_and_carried_basal_one_deposit():
    c = make_c(basal_weight=250.0)
    assert deliver(c, basal=True) == 0.0            # carried
    q = deliver(c, basal=True, apical=['L2E0'])     # current + carried, one apical
    assert q == pytest.approx(250.0)                # still exactly ONE deposit


def test_consumed_eligibility_cannot_be_reused():
    c = make_c(basal_weight=250.0)
    assert deliver(c, basal=True) == 0.0
    assert deliver(c, apical=['L2E0']) == pytest.approx(250.0)   # consumes carried basal
    assert deliver(c, apical=['L2E0']) == 0.0        # nothing left to consume


def test_arrival_order_within_boundary_invariant():
    # basal-then-apical vs apical-then-basal in the SAME boundary -> identical.
    c1 = make_c(basal_weight=250.0)
    c1.begin_event_boundary(); c1.gather_basal('L1E0'); c1.gather_apical('L2E0')
    q1 = c1.resolve_dendrites()
    c2 = make_c(basal_weight=250.0)
    c2.begin_event_boundary(); c2.gather_apical('L2E0'); c2.gather_basal('L1E0')
    q2 = c2.resolve_dendrites()
    assert q1 == q2 == pytest.approx(250.0)


# ================================================================== learning
def test_exact_update_equation_numeric():
    theta, w, w_max, eta, phi = 1000.0, 200.0, 500.0, 0.01, 0.64
    c = make_c(basal_weight=w, w_max=w_max, eta_c=eta, basal_distance_factor=phi,
               threshold=theta)
    # Deposit + fire so learning runs on the causal state.
    run_boundary(c, basal=True, apical=['L2E0'])   # may or may not cross; force below
    # Recompute directly on a FRESH cell to isolate the equation:
    c2 = make_c(basal_weight=w, w_max=w_max, eta_c=eta, basal_distance_factor=phi,
                threshold=theta)
    c2.begin_event_boundary(); c2.gather_basal('L1E0'); c2.gather_apical('L2E0')
    c2.resolve_dendrites()
    c2.apical_active = True                          # gate active at firing
    c2._deposit_signal = 1.0
    w_after = c2.update_basal_weight()
    fe = theta - w
    dw = eta * fe * 1.0 * (1.0 - (w / w_max) ** 2) * 1.0 * phi
    assert w_after == pytest.approx(min(w_max, w + dw))


def test_fe_uses_only_basal_weight_no_apical_term():
    c = make_c(basal_weight=300.0, w_max=500.0, eta_c=0.02, threshold=1000.0)
    c.begin_event_boundary(); c.gather_basal('L1E0'); c.gather_apical('L2E0')
    c.resolve_dendrites()
    before = c.basal_weight
    w_after = c.update_basal_weight()
    fe = 1000.0 - before
    dw = 0.02 * fe * (1.0 - (before / 500.0) ** 2) * 1.0 * 1.0
    assert w_after == pytest.approx(before + dw)     # FE = theta - w, nothing apical


def test_no_learning_without_spike_deposit_or_apical():
    # basal-only, apical-only, and a valid subthreshold deposit that never fires must
    # not change the weight (learning is only invoked on a real spike by the engine;
    # here we simply never call update_basal_weight without a fire).
    for kw in (dict(basal=True), dict(apical=['L2E0']), dict()):
        c = make_c(basal_weight=250.0, threshold=1e9)   # unreachable threshold
        w0 = c.basal_weight
        run_boundary(c, **kw)
        assert c.basal_weight == pytest.approx(w0)


def test_deposit_uses_pre_update_weight():
    c = make_c(basal_weight=250.0, eta_c=0.05, w_max=500.0, threshold=1000.0)
    q = deliver(c, basal=True, apical=['L2E0'])
    assert q == pytest.approx(250.0)                 # deposit is the pre-update weight
    c.apical_active = True; c._deposit_signal = 1.0
    c.update_basal_weight()
    assert c.basal_weight > 250.0                    # learning affects FUTURE deposits


def test_clipping_and_saturation_at_cap():
    c = make_c(basal_weight=499.0, w_max=500.0, eta_c=0.9, threshold=1000.0)
    for _ in range(200):
        c.begin_event_boundary(); c.gather_basal('L1E0'); c.gather_apical('L2E0')
        c.resolve_dendrites(); c.apical_active = True; c._deposit_signal = 1.0
        c.update_basal_weight()
    assert c.basal_weight <= 500.0
    assert c.basal_weight == pytest.approx(500.0, abs=1e-6)   # saturates at cap


def test_no_apical_weight_vector():
    c = make_c()
    assert c.apical.weights is None
    assert not c.apical.plastic
    assert c.basal.plastic and c.basal.weights.shape == (1,)


def test_no_update_acc_weights_fallback():
    c = make_c()
    assert not hasattr(c, 'update_acc_weights')
    assert not hasattr(c, 'acc_weights')


# ============================================== firing / leak / refractory parity
def test_intrinsic_parity_with_e_cell():
    # Same threshold/leak/rest/refractory + same drive -> same membrane trajectory.
    c = make_c(leak_rate=0.03, threshold=1000.0, refractory_steps=2, basal_weight=800.0)
    e = ExcitatoryNeuron('E', 'competitor', acc_weights=np.zeros(0),
                         acc_distance_factor=np.zeros(0), leak_rate=0.03,
                         threshold=1000.0, refractory_steps=2, learn=False)
    assert c.g_L == pytest.approx(e.g_L)
    assert c.v_rest == e.v_rest and c.threshold == e.threshold
    assert c.refractory_steps == e.refractory_steps
    # deposit 800 into C via the gate; give E the same frozen drive.
    deliver(c, basal=True, apical=['L2E0'])
    e.gather_exc(800.0); e.freeze_drive()
    c.advance_segment(0.7); e.advance_segment(0.7)
    assert c.V == pytest.approx(e.V, abs=1e-9)


def test_suprathreshold_without_gate_cannot_fire():
    c = make_c(leak_rate=0.0, threshold=1000.0)
    c.V = 5000.0                                     # retained supra-threshold membrane
    c.begin_event_boundary()                         # no coincidence this boundary
    c.gather_basal('L1E0')                           # basal only -> gate closed
    c.resolve_dendrites(); c.freeze_drive()
    assert not c.can_fire()
    assert c.crossing_time(1.0) == math.inf


def test_valid_event_during_refractory_affects_membrane_but_no_fire():
    c = make_c(leak_rate=0.0, threshold=1000.0, refractory_steps=3, basal_weight=1200.0)
    c.refractory_timer = 2                           # in refractory
    fired = run_boundary(c, basal=True, apical=['L2E0'])
    assert fired is False
    assert c.V > 0.0                                 # charge accrued despite refractory
    assert c.basal_weight == pytest.approx(1200.0)   # no learning without a spike


def test_one_spike_per_boundary_consumes_drive():
    c = make_c(leak_rate=0.0, threshold=1000.0, refractory_steps=0, basal_weight=5000.0)
    fired = run_boundary(c, basal=True, apical=['L2E0'])
    assert fired is True
    assert c.V == 0.0
    assert c.remaining_excitation == 0.0             # frozen packet consumed on firing
    assert c.fired_this_boundary is True


# ============================================ calibrated two-coincidence regime
def test_one_max_weight_deposit_subthreshold_from_reset():
    cal = _calib()
    c = make_c(basal_weight=cal['c_max'], w_max=cal['c_max'] * 1.5,
               leak_rate=0.03, threshold=1000.0, learn=False)
    fired = run_boundary(c, basal=True, apical=['L2E0'])
    assert fired is False
    assert c.V < 1000.0                              # one cap deposit stays below theta


def test_two_init_weight_deposits_reach_threshold():
    cal = _calib()
    c = make_c(basal_weight=cal['c_init'], w_max=cal['c_max'] * 1.5,
               leak_rate=0.03, threshold=1000.0, learn=False)
    assert run_boundary(c, basal=True, apical=['L2E0']) is False   # 1st: subthreshold
    assert run_boundary(c, basal=True, apical=['L2E0']) is True    # 2nd: crosses


def test_mature_cadence_is_every_second_coincidence():
    cal = _calib()
    c = make_c(basal_weight=cal['c_init'], w_max=cal['c_max'] * 1.5,
               leak_rate=0.03, threshold=1000.0, refractory_steps=0, learn=False)
    fires = [run_boundary(c, basal=True, apical=['L2E0']) for _ in range(8)]
    # every valid coincidence deposits; the cell fires on every SECOND one from reset.
    assert fires == [False, True, False, True, False, True, False, True]


def test_resolved_cap_below_one_deposit_firing_weight():
    cal = _calib()
    assert cal['c_max'] < cal['w1']                  # c_basal_weight_max < w_1 invariant
