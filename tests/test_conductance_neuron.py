"""Conductance-based membrane, local activity trace, and local PI plasticity.

These pin the neuron-level contract of the predictive-inhibition model: joint
excitation/inhibition integration before threshold testing, a persistent
inhibitory conductance that survives voltage reset, an activity trace that also
survives reset, and strictly-local PI output-synapse plasticity.
"""

import math

import numpy as np
import pytest

from snn.neurons import (
    ExcitatoryNeuron,
    PredictiveInterneuron,
    E_THRESHOLD,
    leak_to_conductance,
)


def make(**kw):
    n = kw.pop('n', 3)
    d = dict(acc_weights=np.full(n, 100.0), acc_distance_factor=np.ones(n), learn=False)
    d.update(kw)
    return ExcitatoryNeuron('E', 'test', **d)


# ------------------------------------------------------------ membrane dynamics
def test_zero_leak_zero_inhibition_is_pure_integrator():
    n = make(leak_rate=0.0)
    n.gather_exc(400.0); n.integrate()
    assert n.V == pytest.approx(400.0)
    n.gather_exc(400.0); n.integrate()
    assert n.V == pytest.approx(800.0)            # accumulates with no leak/conductance


def test_decay_only_matches_historical_leak_exactly():
    # With no input, integrate() must decay V by exactly (1 - leak_rate) -- the
    # documented migration path from the old per-step leak.
    for leak in (0.03, 0.1, 0.25):
        n = make(leak_rate=leak)
        n.V = 500.0
        n.integrate()                              # no gathered input
        assert n.V == pytest.approx(500.0 * (1.0 - leak))


def test_leak_conductance_mapping():
    assert leak_to_conductance(0.0) == 0.0
    assert math.exp(-leak_to_conductance(0.1)) == pytest.approx(0.9)


def test_joint_integration_matches_closed_form():
    n = make(leak_rate=0.1)
    n.g_inh = 2.0
    n.V = 300.0
    n.gather_exc(1000.0)
    g_L = leak_to_conductance(0.1)
    g_tot = g_L + 2.0
    v_inf = (1000.0) / g_tot                        # E_L = E_inh = 0
    expect = v_inf + (300.0 - v_inf) * math.exp(-g_tot)
    n.integrate()
    assert n.V == pytest.approx(expect)


def test_sufficient_inhibition_suppresses_instant_spike():
    # theta of excitation crosses instantly with no inhibition; the SAME excitation
    # stays sub-threshold when enough g_inh is present in that same boundary.
    hot = make(leak_rate=0.03)
    hot.gather_exc(1.5 * E_THRESHOLD); hot.integrate()
    assert hot.can_fire()                           # would fire instantly

    shunted = make(leak_rate=0.03)
    shunted.add_inhibition(20.0)                    # strong conductance present now
    shunted.gather_exc(1.5 * E_THRESHOLD); shunted.integrate()
    assert not shunted.can_fire()                   # same excitation, suppressed
    assert shunted.V < E_THRESHOLD


def test_same_boundary_order_independence():
    # Excitation then inhibition, vs inhibition then excitation, then repeated
    # gathers in mixed order -- integration reads only the totals, so V is identical.
    a = make(leak_rate=0.05)
    a.gather_exc(300.0); a.add_inhibition(1.0); a.gather_exc(400.0); a.add_inhibition(0.5)
    a.integrate()
    b = make(leak_rate=0.05)
    b.add_inhibition(0.5); b.gather_exc(400.0); b.add_inhibition(1.0); b.gather_exc(300.0)
    b.integrate()
    assert a.V == pytest.approx(b.V)


def test_conductance_nonnegative_and_monotone_decay():
    n = make(alpha_inh=0.5)
    n.add_inhibition(8.0)
    prev = n.g_inh
    for _ in range(10):
        n.decay_conductance()
        assert 0.0 <= n.g_inh <= prev              # monotone non-increasing, nonnegative
        prev = n.g_inh
    assert n.g_inh == pytest.approx(8.0 * 0.5 ** 10)


def test_reset_preserves_conductance_and_trace():
    n = make(leak_rate=0.03)
    n.add_inhibition(1.0)                            # modest conductance; still crosses
    n.gather_exc(E_THRESHOLD * 3); n.integrate()
    n.update_trace()
    g_before, a_before = n.g_inh, n.a
    assert n.can_fire()
    n.fire()
    assert n.V == 0.0                               # voltage reset
    assert n.g_inh == g_before                      # conductance NOT cleared
    assert n.a == a_before                          # trace NOT cleared
    assert a_before > 0.0


def test_zero_conductance_no_nan():
    n = make(leak_rate=0.0)                          # g_L == 0, g_inh == 0 -> g_total == 0
    n.gather_exc(0.0); n.integrate()
    assert n.V == 0.0 and not math.isnan(n.V)


# ---------------------------------------------------------------- activity trace
def test_trace_potentiates_then_decays_without_activity():
    n = make(leak_rate=0.0, alpha_a=0.8, beta_v=0.0, beta_s=1.0, a_max=1.0)
    n.gather_exc(E_THRESHOLD); n.integrate()
    assert n.can_fire(); n.fire()
    n.update_trace()
    assert n.a == pytest.approx(1.0)                # spike bumps trace to the cap
    n.spiked = False
    prev = n.a
    for _ in range(5):
        n.v_pre_reset = 0.0                          # no depolarization, no spike
        n.update_trace()
        assert n.a < prev
        prev = n.a


def test_trace_uses_subthreshold_depolarization():
    n = make(leak_rate=0.0, alpha_a=0.0, beta_v=0.5, beta_s=1.0, threshold=1000.0)
    n.gather_exc(400.0); n.integrate()              # sub-threshold: depol = 0.4
    assert not n.can_fire()
    n.update_trace()
    assert n.a == pytest.approx(0.5 * 0.4)          # beta_v * depol, alpha_a = 0, no spike


# ----------------------------------------------------------- local PI plasticity
def make_pi(n=9, **kw):
    d = dict(w_init=0.0, w_max=1.0, eta=0.2, lt_decay=0.0, g_scale=6.0)
    d.update(kw)
    return PredictiveInterneuron('PI0', 'predictor', n, **d)


def test_pi_potentiates_active_target_only():
    pi = make_pi()
    traces = np.zeros(9); traces[4] = 1.0           # only target 4 recently active
    pi.learn(traces)
    assert pi.w[4] > 0.0
    assert np.all(pi.w[np.arange(9) != 4] == 0.0)   # inactive targets stay at zero


def test_pi_update_is_element_wise_local():
    # Mutating target k's trace changes only w[k]; no other synapse is inspected.
    base = make_pi(eta=0.3)
    t0 = np.zeros(9)
    single = make_pi(eta=0.3)
    t1 = np.zeros(9); t1[6] = 0.7
    base.learn(t0); single.learn(t1)
    diff = np.nonzero(np.abs(single.w - base.w) > 1e-12)[0]
    assert diff.tolist() == [6]                     # exactly the one active target moved


def test_pi_pulse_uses_pre_update_weight():
    pi = make_pi(eta=0.5, g_scale=4.0)
    pi.w[2] = 0.5
    pulse_before = pi.conductance_pulse().copy()    # read pulse from CURRENT (pre-update) weights
    assert pulse_before[2] == pytest.approx(4.0 * 0.5)
    traces = np.zeros(9); traces[2] = 1.0
    pi.learn(traces)                                 # learning changes FUTURE pulses only
    assert pi.w[2] > 0.5


def test_pi_weights_bounded_and_deterministic():
    a = make_pi(eta=0.9)
    b = make_pi(eta=0.9)
    traces = np.linspace(0, 1, 9)
    for _ in range(50):
        a.learn(traces); b.learn(traces)
    assert np.all(a.w >= 0.0) and np.all(a.w <= 1.0)
    assert np.array_equal(a.w, b.w)                 # deterministic


def test_pi_passive_decay_is_local_and_slow():
    pi = make_pi(lt_decay=0.01)
    pi.w[:] = 1.0
    pi.passive_decay()
    assert np.allclose(pi.w, 0.99)                  # each weight decays from itself only
