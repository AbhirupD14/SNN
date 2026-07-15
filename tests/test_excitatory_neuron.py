"""Excitatory neuron: delivery, geometry-invariance, threshold/reset, the exact
nonlinear weight rule, the signed ``p`` boundary, frozen subtractive hard-wipe,
and leak / refractory conventions.
"""

import numpy as np
import pytest

from snn.neurons import (
    ExcitatoryNeuron,
    E_THRESHOLD,
    E_WEIGHT_CAP,
    SUBTRACTIVE_SIGN,
)


def make_neuron(**kw):
    n = kw.pop('n', 3)
    defaults = dict(
        acc_weights=np.full(n, 300.0),
        acc_distance_factor=np.ones(n),
    )
    defaults.update(kw)
    return ExcitatoryNeuron('E', 'test', **defaults)


def test_accumulating_charge_is_raw_weighted_sum():
    n = make_neuron(acc_weights=np.array([100.0, 250.0, 40.0]),
                    acc_distance_factor=np.ones(3))
    spikes = np.array([1.0, 0.0, 1.0])
    n.receive_acc(float((n.acc_weights * spikes).sum()))
    assert n.V == pytest.approx(140.0)


def test_delivered_charge_is_invariant_to_geometry():
    # Two neurons with identical weights but wildly different distance factors must
    # deliver identical charge -- geometry lives in learning only.
    near = make_neuron(acc_weights=np.array([200.0, 200.0]),
                       acc_distance_factor=np.array([1.0, 1.0]))
    far = make_neuron(acc_weights=np.array([200.0, 200.0]),
                      acc_distance_factor=np.array([0.01, 0.01]))
    charge = 400.0
    near.receive_acc(charge)
    far.receive_acc(charge)
    assert near.V == far.V == pytest.approx(400.0)


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
    # Same sign, but the closer synapse learns four times as fast.
    assert np.all(d_near > 0) and np.all(d_far > 0)
    assert d_near == pytest.approx(4.0 * d_far)


def test_threshold_and_reset():
    n = make_neuron(acc_weights=np.array([E_THRESHOLD]), acc_distance_factor=np.ones(1),
                    learn=False)
    n.receive_acc(E_THRESHOLD - 1)
    assert not n.can_fire()
    n.receive_acc(1)
    assert n.can_fire()
    v_pre = n.fire()
    assert v_pre == pytest.approx(E_THRESHOLD)
    assert n.V == 0.0 and n.spiked


def test_weight_rule_participate_potentiates_absent_depresses():
    w0 = 200.0
    n = make_neuron(acc_weights=np.array([w0, w0, w0]), acc_distance_factor=np.ones(3),
                    eta=0.01)
    n.fire()
    participation = np.array([True, False, True])
    w_before = n.acc_weights.copy()
    p = E_THRESHOLD - w_before.sum()                    # 1000 - 600 = 400 > 0
    n.update_acc_weights(participation)
    expected = np.clip(
        w_before + 0.01 * p * np.array([1.0, -1.0, 1.0]) * (1 - (w_before / E_WEIGHT_CAP) ** 2),
        0, E_WEIGHT_CAP)
    assert n.acc_weights == pytest.approx(expected)
    assert n.acc_weights[0] > w0 and n.acc_weights[2] > w0   # participated -> up
    assert n.acc_weights[1] < w0                             # absent -> down


@pytest.mark.parametrize('total,sign', [(600.0, +1), (1000.0, 0), (1500.0, -1)])
def test_signed_p_boundary(total, sign):
    # p = threshold - sum(acc_weights) crosses zero: below threshold a participating
    # afferent potentiates; exactly at threshold it is frozen; above it depresses.
    n = 3
    each = total / n
    neuron = make_neuron(acc_weights=np.full(n, each), acc_distance_factor=np.ones(n),
                         eta=0.01)
    neuron.fire()
    w_before = neuron.acc_weights.copy()
    neuron.update_acc_weights(np.ones(n, dtype=bool))     # all participate (+1)
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
    # Force a large positive delta; clip must hold at the cap.
    n.acc_weights[0] = 10.0                       # far below threshold so p is large positive
    n.fire()
    n.update_acc_weights(np.array([True]))
    assert 0.0 <= n.acc_weights[0] <= E_WEIGHT_CAP


def test_frozen_subtractive_hard_wipe_postcondition():
    n = make_neuron(subt_magnitude=E_THRESHOLD, learn=False)
    n.receive_acc(742.0)                          # subthreshold
    removed = n.receive_subt(True)
    assert n.V == 0.0
    assert removed == pytest.approx(742.0)
    # A crosser above threshold is still floored to exactly rest (no residual).
    n.receive_acc(1500.0)
    n.receive_subt(True)
    assert n.V == 0.0


def test_subtractive_never_updates_acc_weights():
    n = make_neuron(subt_magnitude=E_THRESHOLD)
    before = n.acc_weights.copy()
    n.receive_acc(500.0)
    n.receive_subt(True)
    assert n.acc_weights == pytest.approx(before)


def test_leak_zero_and_nonzero():
    # Zero leak: charge is retained exactly.
    n0 = make_neuron(leak_rate=0.0, learn=False)
    n0.receive_acc(500.0); n0.advance()
    assert n0.V == pytest.approx(500.0)
    # Nonzero leak: decays toward rest by (1 - leak) each step.
    n1 = make_neuron(leak_rate=0.1, learn=False)
    n1.receive_acc(500.0); n1.advance()
    assert n1.V == pytest.approx(450.0)


def test_leak_rate_validation():
    with pytest.raises(ValueError):
        make_neuron(leak_rate=1.5)


def test_refractory_off_by_one():
    # refractory_steps = R blocks exactly R steps after the firing step.
    n = make_neuron(acc_weights=np.array([E_THRESHOLD]), acc_distance_factor=np.ones(1),
                    learn=False, refractory_steps=2)
    n.receive_acc(E_THRESHOLD)
    assert n.can_fire()
    n.fire()
    n.advance()                                   # end of firing step: does not count
    assert n.refractory_timer == 2
    # Step +1: blocked
    n.spiked = False
    n.receive_acc(E_THRESHOLD)
    assert not n.can_fire()
    n.advance(); assert n.refractory_timer == 1
    # Step +2: still blocked
    assert not n.can_fire()
    n.advance(); assert n.refractory_timer == 0
    # Step +3: free again
    assert n.can_fire()
