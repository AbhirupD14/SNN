"""Excitatory neuron: charge delivery, geometry-invariance of delivered charge,
threshold/reset, the exact nonlinear accumulating-weight rule, the signed ``p``
boundary, weight cap, and the refractory off-by-one convention.

Conductance / trace / persistent-inhibition dynamics are covered separately in
tests/test_conductance_neuron.py. The one accumulating-weight learning rule is
unchanged by the conductance rewrite.
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
    w0 = 200.0
    n = make_neuron(acc_weights=np.array([w0, w0, w0]), acc_distance_factor=np.ones(3), eta=0.01)
    n.fire()
    participation = np.array([True, False, True])
    w_before = n.acc_weights.copy()
    p = E_THRESHOLD - w_before.sum()                 # 1000 - 600 = 400 > 0
    n.update_acc_weights(participation)
    expected = np.clip(
        w_before + 0.01 * p * np.array([1.0, -1.0, 1.0]) * (1 - (w_before / E_WEIGHT_CAP) ** 2),
        0, E_WEIGHT_CAP)
    assert n.acc_weights == pytest.approx(expected)
    assert n.acc_weights[0] > w0 and n.acc_weights[2] > w0
    assert n.acc_weights[1] < w0


@pytest.mark.parametrize('total,sign', [(600.0, +1), (1000.0, 0), (1500.0, -1)])
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


def test_weight_cap_clip():
    n = make_neuron(acc_weights=np.array([E_WEIGHT_CAP - 1]), acc_distance_factor=np.ones(1),
                    eta=10.0, learn=True)
    n.acc_weights[0] = 10.0
    n.fire()
    n.update_acc_weights(np.array([True]))
    assert 0.0 <= n.acc_weights[0] <= E_WEIGHT_CAP


def test_learn_flag_freezes_weights():
    n = make_neuron(acc_weights=np.array([300.0, 300.0]), acc_distance_factor=np.ones(2),
                    learn=False)
    before = n.acc_weights.copy()
    n.fire()
    n.update_acc_weights(np.array([True, False]))
    assert n.acc_weights == pytest.approx(before)    # frozen sources never learn


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
