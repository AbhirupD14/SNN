"""Excitatory neuron: charge delivery, geometry-invariance of delivered charge,
threshold/reset, the production linear-bounded accumulating-weight rule, the historical
quadratic mode, signed ``p`` boundary, weight cap, and refractory off-by-one convention.

Conductance / trace / persistent-inhibition dynamics are covered separately in
tests/test_conductance_neuron.py. The accumulating-weight mechanics remain independent
of the conductance rewrite.
"""

import numpy as np
import pytest

from snn.neurons import ExcitatoryNeuron, E_THRESHOLD, E_WEIGHT_CAP


def make_neuron(**kw):
    n = kw.pop('n', 3)
    defaults = dict(acc_weights=np.full(n, 300.0), acc_distance_factor=np.ones(n))
    defaults.update(kw)
    return ExcitatoryNeuron('E', 'test', **defaults)


# ------------------------------------------------------------ delivery / geometry
def test_accumulating_charge_is_raw_weighted_sum():
    n = make_neuron(acc_weights=np.array([100.0, 250.0, 40.0]), leak_rate=0.0, learn=False)
    spikes = np.array([1.0, 0.0, 1.0])
    n.gather_exc(float((n.acc_weights * spikes).sum()))
    n.integrate()
    assert n.V == pytest.approx(140.0)               # 100 + 40, no leak -> pure jump


def test_delivered_charge_is_invariant_to_geometry():
    near = make_neuron(acc_weights=np.array([200.0, 200.0]),
                       acc_distance_factor=np.array([1.0, 1.0]), leak_rate=0.0, learn=False)
    far = make_neuron(acc_weights=np.array([200.0, 200.0]),
                      acc_distance_factor=np.array([0.01, 0.01]), leak_rate=0.0, learn=False)
    near.gather_exc(400.0); near.integrate()
    far.gather_exc(400.0); far.integrate()
    assert near.V == far.V == pytest.approx(400.0)   # geometry never scales delivered charge


def test_geometry_changes_the_learning_delta_not_the_charge():
    part = np.array([True, True])
    near = make_neuron(acc_weights=np.array([100.0, 100.0]),
                       acc_distance_factor=np.array([1.0, 1.0]))
    far = make_neuron(acc_weights=np.array([100.0, 100.0]),
                      acc_distance_factor=np.array([0.25, 0.25]))
    near.fire(); near.update_acc_weights(part)
    far.fire(); far.update_acc_weights(part)
    d_near = near.acc_weights - 100.0
    d_far = far.acc_weights - 100.0
    assert np.all(d_near > 0) and np.all(d_far > 0)
    assert d_near == pytest.approx(4.0 * d_far)      # closer synapse learns 4x as fast


def test_threshold_and_reset():
    n = make_neuron(acc_weights=np.array([E_THRESHOLD]), acc_distance_factor=np.ones(1),
                    leak_rate=0.0, learn=False)
    n.gather_exc(E_THRESHOLD - 1); n.integrate()
    assert not n.can_fire()
    n.gather_exc(1); n.integrate()
    assert n.can_fire()
    v_pre = n.fire()
    assert v_pre == pytest.approx(E_THRESHOLD)
    assert n.V == 0.0 and n.spiked


# --------------------------------------------------------------- the weight rule
def test_weight_rule_participate_potentiates_absent_depresses():
    # Direction of the rule is mode-agnostic (holds for the new linear-bounded DEFAULT).
    w0 = 200.0
    n = make_neuron(acc_weights=np.array([w0, w0, w0]), acc_distance_factor=np.ones(3), eta=0.01)
    n.fire()
    n.update_acc_weights(np.array([True, False, True]))
    assert n.acc_weights[0] > w0 and n.acc_weights[2] > w0    # participating potentiate
    assert n.acc_weights[1] < w0                              # absent depresses


def test_default_update_is_linear_fe_exact():
    # The production default is cap-free `linear_fe`: dw = eta*(B - sum(w))*s*influence,
    # floor 0 only (NO upper cap). No quadratic multiplier. B = maturity_budget_frac*theta.
    w0 = 200.0
    n = make_neuron(acc_weights=np.array([w0, w0, w0]),
                    acc_distance_factor=np.array([1.0, 0.5, 1.0]), eta=0.01)
    assert n.update_mode == 'linear_fe'                       # engine-independent default
    assert n.maturity_budget_frac == 1.10                     # budget-headroom default
    n.fire()
    w_before = n.acc_weights.copy()
    p = 1.10 * E_THRESHOLD - w_before.sum()                   # budget target, NOT firing theta
    n.update_acc_weights(np.array([True, False, True]))
    expected = np.maximum(w_before + 0.01 * p * np.array([1.0, -1.0, 1.0])
                          * np.array([1.0, 0.5, 1.0]), 0.0)   # floor only; NO cap, NO multiplier
    assert n.acc_weights == pytest.approx(expected)


def test_p_is_full_weight_sum_not_active_subset():
    # The fullness error p that scales every delta is (threshold - sum of ALL weights):
    # every weight that can contribute charge toward threshold, NOT just the afferents
    # that participated this boundary. Construct a case where the two readings differ a
    # lot -- a large SILENT afferent -- and pin the delta to the full-sum p.
    w = np.array([100.0, 500.0, 100.0])
    part = np.array([True, False, True])              # the 500 afferent stays silent
    n = make_neuron(acc_weights=w.copy(), acc_distance_factor=np.ones(3), eta=0.001)
    n.fire()
    n.update_acc_weights(part)
    p_all = 1.10 * E_THRESHOLD - w.sum()              # 1100 - 700 = 400  (the rule's p)
    p_active_only = 1.10 * E_THRESHOLD - w[part].sum()  # 1100 - 200 = 900  (the wrong reading)
    assert p_all != p_active_only                     # the fixture actually distinguishes them
    # a participating afferent (signal +1, distance 1) moved by exactly eta * p_all
    assert n.acc_weights[0] - 100.0 == pytest.approx(0.001 * p_all)
    assert n.acc_weights[0] - 100.0 != pytest.approx(0.001 * p_active_only)


def test_historical_quadratic_mode_exact():
    # The historical quadratic rule stays available under an explicit mode request.
    w0 = 200.0
    n = make_neuron(acc_weights=np.array([w0, w0, w0]), acc_distance_factor=np.ones(3),
                    eta=0.01, update_mode='quadratic_bounded')
    n.fire()
    w_before = n.acc_weights.copy()
    p = 1.10 * E_THRESHOLD - w_before.sum()           # budget applies to both update modes
    n.update_acc_weights(np.array([True, False, True]))
    expected = np.clip(
        w_before + 0.01 * p * np.array([1.0, -1.0, 1.0]) * (1 - (w_before / E_WEIGHT_CAP) ** 2),
        0, E_WEIGHT_CAP)
    assert n.acc_weights == pytest.approx(expected)


@pytest.mark.parametrize('total,sign', [(600.0, +1), (1100.0, 0), (1500.0, -1)])
def test_signed_p_boundary(total, sign):
    n = 3
    each = total / n
    neuron = make_neuron(acc_weights=np.full(n, each), acc_distance_factor=np.ones(n), eta=0.01)
    neuron.fire()
    w_before = neuron.acc_weights.copy()
    neuron.update_acc_weights(np.ones(n, dtype=bool))
    delta = neuron.acc_weights - w_before
    if sign > 0:
        assert np.all(delta > 0)
    elif sign == 0:
        assert delta == pytest.approx(np.zeros(n))
    else:
        assert np.all(delta < 0)


def test_production_default_is_cap_free_above_500():
    # The production default (`linear_fe`) never clips to a per-synapse cap: a lone active
    # afferent grows past the historical theta/2 ceiling toward the FE budget B=1.1*theta.
    n = make_neuron(acc_weights=np.array([0.0]), acc_distance_factor=np.ones(1),
                    eta=0.01, learn=True, leak_rate=0.0)
    assert n.update_mode == 'linear_fe'
    for _ in range(5000):
        n.fire(); n.update_acc_weights(np.array([True]))
    assert n.acc_weights[0] > 500.0                          # exceeds the old cap
    assert n.acc_weights[0] == pytest.approx(1.10 * E_THRESHOLD, abs=1.0)   # converges to B


def test_weight_cap_clip_bounded_mode():
    # The HEADLESS `linear_bounded` mode still enforces the [0, w_max] clip (regression).
    n = make_neuron(acc_weights=np.array([E_WEIGHT_CAP - 1]), acc_distance_factor=np.ones(1),
                    eta=10.0, learn=True, update_mode='linear_bounded', w_max=E_WEIGHT_CAP)
    n.acc_weights[0] = 10.0
    n.fire()
    n.update_acc_weights(np.array([True]))
    assert 0.0 <= n.acc_weights[0] <= E_WEIGHT_CAP


def test_weight_floor_clips_depression():
    # A non-participating afferent is depressed; w_floor sets its lower clip.
    # floor=0 (default): can reach exactly 0. floor>0: holds a residual weight.
    for floor, lo in ((0.0, 0.0), (1.0, 1.0)):
        n = make_neuron(acc_weights=np.array([600.0, 5.0]), acc_distance_factor=np.ones(2),
                        eta=10.0, learn=True, w_floor=floor)
        n.fire()
        n.update_acc_weights(np.array([True, False]))   # afferent 1 absent -> depressed hard
        assert n.acc_weights[1] == pytest.approx(lo)     # clipped exactly at the floor
        assert n.acc_weights[0] >= floor


def test_weight_floor_default_is_zero_byte_compatible():
    a = make_neuron(acc_weights=np.array([300.0, 300.0]), acc_distance_factor=np.ones(2),
                    eta=0.1, learn=True)
    b = make_neuron(acc_weights=np.array([300.0, 300.0]), acc_distance_factor=np.ones(2),
                    eta=0.1, learn=True, w_floor=0.0)
    for n in (a, b):
        n.fire(); n.update_acc_weights(np.array([True, False]))
    assert np.array_equal(a.acc_weights, b.acc_weights)  # explicit 0 == default


def test_learn_flag_freezes_weights():
    n = make_neuron(acc_weights=np.array([300.0, 300.0]), acc_distance_factor=np.ones(2),
                    learn=False)
    before = n.acc_weights.copy()
    n.fire()
    n.update_acc_weights(np.array([True, False]))
    assert n.acc_weights == pytest.approx(before)    # frozen sources never learn


# ------------------------------------------------------------- budget headroom
def test_matured_specialist_is_one_step_integrator():
    # A clean single-pattern specialist (3 of 9 afferents participate EVERY step) driven
    # for a few thousand fires should mature so its active afferents sum >= the firing
    # threshold -- i.e. one integration boundary now crosses theta and fires. The
    # budget-headroom default (1.10) is what lifts the active sum above theta; the old
    # theta-target rule asymptoted to theta- and never fired in one step.
    part = np.array([True, True, True] + [False] * 6)
    n = make_neuron(n=9, acc_weights=np.zeros(9), acc_distance_factor=np.ones(9),
                    leak_rate=0.0, eta=0.01, learn=True)
    for _ in range(4000):
        n.fire()
        n.update_acc_weights(part)
    active_sum = float(n.acc_weights[part].sum())
    inactive_sum = float(n.acc_weights[~part].sum())
    assert inactive_sum == pytest.approx(0.0, abs=1e-6)   # silent afferents depressed to floor
    assert active_sum >= E_THRESHOLD                       # matured budget clears theta
    # one integration of ONLY the active afferents now crosses the firing threshold
    n.V = 0.0
    n.gather_exc(active_sum); n.integrate()
    assert n.V >= n.threshold and n.can_fire()             # one-step integrator


def test_maturity_budget_frac_below_one_raises():
    with pytest.raises(ValueError):
        make_neuron(maturity_budget_frac=0.9)


# --------------------------------------------------------------------- refractory
def test_leak_rate_validation():
    with pytest.raises(ValueError):
        make_neuron(leak_rate=1.5)


def test_refractory_off_by_one():
    n = make_neuron(acc_weights=np.array([E_THRESHOLD]), acc_distance_factor=np.ones(1),
                    leak_rate=0.0, learn=False, refractory_steps=2)
    n.gather_exc(E_THRESHOLD); n.integrate()
    assert n.can_fire()
    n.fire()
    n.advance_refractory()                            # end of firing step: does not count
    assert n.refractory_timer == 2
    n.spiked = False
    n.gather_exc(E_THRESHOLD); n.integrate()
    assert not n.can_fire()                           # step +1 blocked
    n.advance_refractory(); assert n.refractory_timer == 1
    assert not n.can_fire()                           # step +2 blocked
    n.advance_refractory(); assert n.refractory_timer == 0
    assert n.can_fire()                               # step +3 free
