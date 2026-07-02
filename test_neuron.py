"""
Unit tests for the spiking Neuron (neuron.py) and its flexible-fan-in twin
(neuron_flexible.py), covering the trace-gated, sign-preserving Hebbian rule.
"""
import numpy as np
from neuron import Neuron
from neuron_flexible import Neuron as FlexNeuron


def test_receive_accumulates_potential_and_trace():
    n = Neuron(n_inputs=3, threshold=10.0, leak_rate=0.0)
    n.weights = np.array([0.2, 0.5, 0.3])
    n.receive_input(np.array([1, 0, 1]))
    assert np.isclose(n.potential, 0.5)              # 0.2 + 0.3
    assert np.allclose(n.trace, [1, 0, 1])           # un-summed presynaptic activity
    print("PASS: receive_input accumulates membrane potential and per-synapse trace")


def test_trace_gating_only_active_synapses_grow():
    n = Neuron(n_inputs=4, threshold=0.5, refractory_period=2,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    n.weights = np.array([0.4, 0.4, 0.4, 0.4])
    n.receive_input(np.array([1, 1, 0, 0]))          # only lines 0,1 active -> 0.8 >= 0.5
    assert n.check_threshold()
    n.fire()
    assert np.allclose(n.weights[:2], 0.5)           # active grew
    assert np.allclose(n.weights[2:], 0.4)           # silent untouched
    assert np.allclose(n.trace, 0.0)                 # trace cleared on fire
    print("PASS: only synapses that delivered charge are credited; trace resets on fire")


def test_sign_preserving_update():
    n = Neuron(n_inputs=2, threshold=0.0, refractory_period=2,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    n.weights = np.array([0.4, -0.4])                # one excitatory, one inhibitory synapse
    n.receive_input(np.array([1, 1]))
    n.potential = 1.0                                # force a spike
    assert n.check_threshold()
    n.fire()
    assert np.isclose(n.weights[0], 0.5), n.weights  # excitatory grows more positive
    assert np.isclose(n.weights[1], -0.5), n.weights # inhibitory grows more negative
    print("PASS: update is sign-preserving (|w| grows in each synapse's own direction)")


def test_no_update_without_firing():
    n = Neuron(n_inputs=3, threshold=100.0, learning_rate=0.5, weight_cap=10.0, leak_rate=0.0)
    n.weights = np.array([0.3, 0.3, 0.3])
    before = n.weights.copy()
    n.receive_input(np.array([1, 1, 1]))             # nowhere near threshold
    assert not n.check_threshold()
    n.update()
    assert np.allclose(n.weights, before)            # weights change only on fire
    print("PASS: weights are unchanged when the neuron does not fire")


def test_membrane_and_trace_leak_together():
    n = Neuron(n_inputs=2, threshold=100.0, leak_rate=0.5)
    n.weights = np.array([1.0, 1.0])
    n.receive_input(np.array([1, 0]))                # potential 1.0, trace [1,0]
    n.update()                                       # no fire -> both leak by (1-0.5)
    assert np.isclose(n.potential, 0.5)              # 1.0 + 0.5*(0-1.0)
    assert np.allclose(n.trace, [0.5, 0.0])          # trace decays with the same leak
    print("PASS: membrane potential and eligibility trace leak with the same rate")


def test_weight_cap():
    n = Neuron(n_inputs=2, threshold=0.0, refractory_period=0,
               learning_rate=0.3, weight_cap=0.5, leak_rate=0.0)
    n.weights = np.array([0.4, -0.4])
    for _ in range(10):
        n.receive_input(np.array([1, 1]))
        n.potential = 5.0                            # force firing every step
        if n.check_threshold():
            n.fire()
        n.update()
    assert np.max(np.abs(n.weights)) <= 0.5 + 1e-9   # never exceeds cap
    assert np.isclose(n.weights[0], 0.5) and np.isclose(n.weights[1], -0.5)  # both reached cap
    print("PASS: repeated firing drives active weights to +/-cap, never beyond")


def test_refractory_blocks_accumulation():
    n = Neuron(n_inputs=1, threshold=0.5, refractory_period=2, leak_rate=0.0)
    n.weights = np.array([1.0])
    n.receive_input(np.array([1]))
    n.fire()                                         # enters refractory (timer = 2)
    assert n.refractory_timer == 2
    n.receive_input(np.array([1]))                   # ignored during refractory
    assert np.isclose(n.potential, 0.0)
    assert np.allclose(n.trace, 0.0)
    print("PASS: no charge or trace accumulates during the refractory period")


def test_inhibitory_event_dynamics_and_learning():
    """Exact numeric check of the inhibitory-discharge plasticity rule."""
    n = Neuron(n_inputs=1, threshold=2.0, weight_cap=1.0, leak_rate=0.0,
               inhibitory_learning_rate=0.1)
    n.weights = np.array([-0.4])                     # one inhibitory gate, |w| = 0.4
    n.potential = 1.0                                # V_pre
    events = n.apply_inhibition(np.array([1]))       # deliver the discharge
    ev = events[0]
    # V_pre=1.0, V_post = 1.0 - 0.4 = 0.6
    assert np.isclose(n.potential, 0.6) and np.isclose(ev['v_post'], 0.6)
    assert np.isclose(ev['v_pre'], 1.0)
    # p = V_pre/theta = 1.0/2.0 = 0.5
    assert np.isclose(ev['p'], 0.5)
    # dw = eta*p*(1 - w/w_max) = 0.1*0.5*(1 - 0.4) = 0.03  ->  w = 0.43
    assert np.isclose(ev['delta_w'], 0.03) and np.isclose(ev['w_after'], 0.43)
    assert np.isclose(n.weights[0], -0.43)           # still inhibitory, magnitude grew
    print("PASS: inhibitory event applies V=V-w and strengthens the gate by eta*p*(1-w/w_max)")


def test_inhibitory_learning_prefers_near_threshold():
    """The gate strengthens more for a neuron that was closer to firing."""
    near = Neuron(n_inputs=1, threshold=1.0, weight_cap=1.0, leak_rate=0.0, inhibitory_learning_rate=0.2)
    far = Neuron(n_inputs=1, threshold=1.0, weight_cap=1.0, leak_rate=0.0, inhibitory_learning_rate=0.2)
    near.weights = np.array([-0.3]); near.potential = 0.9   # p = 0.9
    far.weights = np.array([-0.3]);  far.potential = 0.2    # p = 0.2
    d_near = near.apply_inhibition(np.array([1]))[0]['delta_w']
    d_far = far.apply_inhibition(np.array([1]))[0]['delta_w']
    assert d_near > d_far > 0, (d_near, d_far)
    print("PASS: near-threshold suppression drives a larger gate update (specialization)")


def test_inhibitory_saturation_at_wmax():
    """Repeated suppression drives the gate toward w_max and stops (finite resource)."""
    n = Neuron(n_inputs=1, threshold=1.0, weight_cap=0.8, leak_rate=0.0, inhibitory_learning_rate=0.5)
    n.weights = np.array([-0.1])
    last = 0.1
    for _ in range(200):
        n.potential = 1.0                            # keep it maximally near threshold
        w_after = n.apply_inhibition(np.array([1]))[0]['w_after']
        assert w_after >= last - 1e-12               # monotonically non-decreasing
        assert w_after <= 0.8 + 1e-9                 # never exceeds w_max
        last = w_after
    assert np.isclose(-n.weights[0], 0.8, atol=1e-3) # converged to the ceiling
    print("PASS: inhibitory gate saturates at w_max under repeated near-threshold suppression")


def test_inhibitory_refractory_gate():
    """No discharge or learning while refractory."""
    n = Neuron(n_inputs=1, threshold=1.0, weight_cap=1.0, leak_rate=0.0)
    n.weights = np.array([-0.5]); n.potential = 0.9
    n.refractory_timer = 2
    events = n.apply_inhibition(np.array([1]))
    assert events == []
    assert np.isclose(n.potential, 0.9) and np.isclose(n.weights[0], -0.5)
    print("PASS: inhibitory plasticity is gated off during the refractory period")


def test_inhibitory_independent_of_excitatory():
    """apply_inhibition leaves excitatory weights alone; a postsynaptic spike
    delivered via receive_input leaves separately-delivered inhibitory gates alone."""
    n = Neuron(n_inputs=2, threshold=0.5, refractory_period=0,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0, inhibitory_learning_rate=0.3)
    n.weights = np.array([0.6, -0.4])                # [excitatory, inhibitory]
    # Inhibitory event must not touch the excitatory weight.
    n.potential = 0.4
    n.apply_inhibition(np.array([0, 1]))
    assert np.isclose(n.weights[0], 0.6), "excitatory weight changed on an inhibitory event"
    # Excitatory spike (charge only on line 0) must not touch the inhibitory gate,
    # because the inhibitory line delivered no excitatory trace.
    w_inh = n.weights[1]
    n.receive_input(np.array([1, 0]))
    n.potential = 1.0
    n.fire()
    assert np.isclose(n.weights[1], w_inh), "inhibitory gate changed on an excitatory spike"
    assert n.weights[0] > 0.6, "excitatory rule should still strengthen the active E synapse"
    print("PASS: the two learning systems are independent")


def test_flexible_neuron_inhibition_parity():
    """Flexible-fan-in neuron runs the identical inhibitory rule."""
    fn = FlexNeuron(threshold=2.0, weight_cap=1.0, leak_rate=0.0, inhibitory_learning_rate=0.1)
    for w in [0.5, -0.4]:
        fn.add_input_connection(w)
    fn.finalize_connections()
    fn.potential = 1.0
    ev = fn.apply_inhibition(np.array([0, 1]))[0]
    assert np.isclose(fn.potential, 0.6)
    assert np.isclose(ev['p'], 0.5) and np.isclose(ev['delta_w'], 0.03)
    assert np.isclose(fn.weights[1], -0.43) and np.isclose(fn.weights[0], 0.5)
    print("PASS: flexible neuron applies the inhibitory rule identically (excitatory synapse untouched)")


def test_flexible_neuron_parity():
    fn = FlexNeuron(threshold=0.0, refractory_period=2,
                    learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    for w in [0.4, -0.4, 0.4]:
        fn.add_input_connection(w)
    fn.finalize_connections()
    fn.receive_input(np.array([1, 1, 0]))
    fn.potential = 1.0
    assert fn.check_threshold()
    fn.fire()
    assert np.isclose(fn.weights[0], 0.5)            # active excitatory -> more positive
    assert np.isclose(fn.weights[1], -0.5)           # active inhibitory -> more negative
    assert np.isclose(fn.weights[2], 0.4)            # silent -> untouched
    print("PASS: flexible-fan-in neuron behaves identically (trace + sign preserving)")


# ---------------------------------------------------------------------------
# Confidence-mode excitatory plasticity (trace_mode="confidence")
# ---------------------------------------------------------------------------

def _fire_once(n, spikes):
    """Deliver one input vector, fire if threshold reached, advance one step."""
    n.receive_input(np.asarray(spikes, dtype=float))
    fired = n.check_threshold()
    if fired:
        n.fire()
    n.update()
    return fired


def test_activity_is_default_and_confidence_untouched():
    n = Neuron(n_inputs=3, threshold=0.5, refractory_period=0, leak_rate=0.0,
               learning_rate=0.1, weight_cap=10.0)
    assert n.trace_mode == "activity"                 # default preserves old behavior
    n.weights = np.array([0.4, 0.4, 0.4])
    _fire_once(n, [1, 1, 0])
    assert np.allclose(n.weights[:2], 0.5)            # original rule: dw = lr*trace*sign
    assert np.isclose(n.weights[2], 0.4)
    assert np.allclose(n.confidence, 0.10)            # confidence never consulted/changed
    print("PASS: activity mode is the default and leaves confidence untouched")


def test_confidence_starts_small_grows_and_saturates():
    n = Neuron(n_inputs=3, threshold=0.5, refractory_period=0, leak_rate=0.0,
               learning_rate=0.05, weight_cap=10.0, trace_mode="confidence",
               confidence_init=0.10, confidence_beta=0.30, confidence_gamma=0.02)
    n.weights = np.array([0.5, 0.5, 0.5])
    assert np.allclose(n.confidence, 0.10)            # small but non-zero start
    _fire_once(n, [1, 1, 0])                          # 0,1 participate; 2 silent
    assert np.isclose(n.confidence[0], 0.10 + 0.30 * 0.90)   # 0.37 (fast growth)
    assert np.isclose(n.confidence[1], 0.37)
    assert np.isclose(n.confidence[2], 0.10 * (1 - 0.02))    # 0.098 (slow decay)
    for _ in range(80):
        _fire_once(n, [1, 1, 0])
    assert 0.99 < n.confidence[0] < 1.0               # asymptotes toward 1
    print("PASS: confidence starts small, grows fast, saturates <1; idle synapses decay slowly")


def test_confidence_unchanged_without_a_spike():
    n = Neuron(n_inputs=3, threshold=10.0, refractory_period=0, leak_rate=0.0,
               trace_mode="confidence")
    n.weights = np.array([0.5, 0.5, 0.5])
    c0 = n.confidence.copy()
    for _ in range(5):
        n.receive_input(np.array([1, 1, 1]))          # accumulates but never fires
        assert not n.check_threshold()
        n.update()
    assert np.allclose(n.confidence, c0)              # participation alone is not permanent
    print("PASS: confidence changes only on a successful spike, not on mere participation")


def test_confidence_credit_is_normalized_and_budget_preserved():
    n = Neuron(n_inputs=3, threshold=0.5, refractory_period=0, leak_rate=0.0,
               learning_rate=0.1, weight_cap=10.0, trace_mode="confidence")
    n.weights = np.array([0.4, 0.4, 0.4])
    n.weight_budget = 1.2
    n.confidence = np.array([0.8, 0.2, 0.5])          # unequal -> unequal credit
    _fire_once(n, [1, 1, 0])                          # 0,1 active excitatory; 2 silent
    ev = n.last_excitatory_event
    assert ev['participating'] == [0, 1]
    assert np.isclose(sum(ev['credits']), 1.0)        # credit normalized over active exc.
    assert ev['credits'][0] > ev['credits'][1]        # more-trusted gate gets more credit
    assert np.isclose(n.weights[n.weights > 0].sum(), 1.2)   # budget conserved
    print("PASS: credit is confidence-weighted, normalized to 1, and budget is conserved")


def test_confidence_diverges_from_weight():
    # Two synapses share a pixel that is always on; one also carries a pixel that
    # is only sometimes on. The always-on gate should accrue higher confidence.
    n = Neuron(n_inputs=2, threshold=0.5, refractory_period=0, leak_rate=0.0,
               learning_rate=0.05, weight_cap=10.0, trace_mode="confidence")
    n.weights = np.array([0.5, 0.5])
    n.weight_budget = 1.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        common = 1.0
        rare = 1.0 if rng.random() < 0.3 else 0.0
        # ensure it fires: common alone (0.5*budget-share) may be sub-threshold, so
        # drive with both lines' current weights; potential = w0*common + w1*rare.
        n.receive_input(np.array([common, rare]))
        if n.potential < n.threshold:                 # top up so the common line always fires
            n.potential = n.threshold
        if n.check_threshold():
            n.fire()
        n.update()
    assert n.confidence[0] > n.confidence[1]           # always-present gate is trusted more
    print("PASS: confidence diverges from weight (always-present input earns more trust)")


def test_flexible_confidence_parity():
    base = Neuron(n_inputs=3, threshold=0.5, refractory_period=0, leak_rate=0.0,
                  learning_rate=0.1, weight_cap=10.0, trace_mode="confidence")
    base.weights = np.array([0.5, 0.5, 0.4])
    base.weight_budget = 1.0
    fx = FlexNeuron(threshold=0.5, refractory_period=0, leak_rate=0.0,
                    learning_rate=0.1, weight_cap=10.0, trace_mode="confidence")
    for w in [0.5, 0.5, 0.4]:
        fx.add_input_connection(w)
    fx.finalize_connections()
    fx.weight_budget = 1.0
    for _ in range(20):
        for n in (base, fx):
            n.receive_input(np.array([1, 1, 0]))
            if n.check_threshold():
                n.fire()
            n.update()
    assert np.allclose(base.weights, fx.weights)
    assert np.allclose(base.confidence, fx.confidence)
    print("PASS: flexible-fan-in neuron matches base neuron in confidence mode")


# ---------------------------------------------------------------------------
# Homeostatic synaptic scaling (third local system; opt-in via homeostasis=True)
# ---------------------------------------------------------------------------

def test_homeostasis_off_by_default_leaves_weights_alone():
    n = Neuron(n_inputs=3, leak_rate=0.0)
    assert n.homeostasis is False
    n.weights = np.array([0.3, 0.3, 0.3])
    for _ in range(100):
        n.update()                                    # never fires
    assert np.allclose(n.weights, 0.3)                # no scaling when off
    # The calcium sensor is still tracked (cheap, always-on), just unused.
    assert n.ca == 0.0
    print("PASS: homeostasis is off by default and leaves weights untouched")


def test_homeostasis_grows_a_chronically_silent_neuron():
    n = Neuron(n_inputs=3, threshold=1.0, leak_rate=0.0, weight_cap=10.0,
               homeostasis=True, ca_rate=0.1, ca_target=0.10, ca_band=0.5,
               homeo_up=0.02, homeo_down=0.02, homeo_budget_min=0.1, homeo_budget_max=100.0)
    n.weights = np.array([0.3, 0.3, 0.3])
    start = n.weights.sum()
    for _ in range(200):
        n.update()                                    # silent: ca stays 0 < set-point
    assert n.weights.sum() > start * 3                # resource grew substantially
    assert np.allclose(n.weights / n.weights.sum(),
                       [1/3, 1/3, 1/3])               # multiplicative: relative shape preserved
    print("PASS: chronically silent neuron up-scales its resource (recruitment)")


def test_homeostasis_shrinks_a_hyperactive_neuron():
    n = Neuron(n_inputs=3, threshold=1.0, leak_rate=0.0, weight_cap=10.0,
               homeostasis=True, ca_rate=0.1, ca_target=0.10, ca_band=0.5,
               homeo_up=0.02, homeo_down=0.02, homeo_budget_min=0.05, homeo_budget_max=100.0)
    n.weights = np.array([0.3, 0.3, 0.3])
    start = n.weights.sum()
    for _ in range(200):
        n.spiked = True                               # pretend it fired -> ca climbs above set-point
        n.update()
    assert n.weights.sum() < start                    # resource shrank (anti-tyranny)
    print("PASS: chronically hyperactive neuron down-scales its resource (anti-tyranny)")


def test_homeostasis_scalar_is_fixed_not_error_proportional():
    # The multiplicative step must be exactly (1 +/- homeo_up/down) -- a constant,
    # not a value that scales with how far ca is from the set-point (that would be
    # gradient-like). Verify one silent step multiplies the resource by exactly 1+up.
    n = Neuron(n_inputs=2, leak_rate=0.0, weight_cap=100.0,
               homeostasis=True, ca_rate=0.1, ca_target=0.10, ca_band=0.5,
               homeo_up=0.02, homeo_down=0.02)
    n.weights = np.array([0.4, 0.6])                  # sum 1.0
    n.update()                                        # ca=0 < lo -> one grow step
    assert np.isclose(n.homeo_budget, 1.0 * 1.02)     # exactly (1+up), independent of |ca-target|
    assert np.isclose(n.weights.sum(), 1.02)
    print("PASS: homeostatic scalar is a fixed constant, not an error-proportional/gradient value")


def test_flexible_homeostasis_parity():
    base = Neuron(n_inputs=3, threshold=1.0, leak_rate=0.0, weight_cap=10.0,
                  homeostasis=True, ca_rate=0.1, ca_target=0.10,
                  homeo_up=0.02, homeo_down=0.02, homeo_budget_max=100.0)
    base.weights = np.array([0.3, 0.3, 0.3])
    fx = FlexNeuron(threshold=1.0, leak_rate=0.0, weight_cap=10.0,
                    homeostasis=True, ca_rate=0.1, ca_target=0.10,
                    homeo_up=0.02, homeo_down=0.02, homeo_budget_max=100.0)
    for w in [0.3, 0.3, 0.3]:
        fx.add_input_connection(w)
    fx.finalize_connections()
    for _ in range(150):
        base.update(); fx.update()
    assert np.allclose(base.weights, fx.weights)
    assert np.isclose(base.ca, fx.ca) and np.isclose(base.homeo_budget, fx.homeo_budget)
    print("PASS: flexible neuron matches base neuron under homeostatic scaling")


if __name__ == "__main__":
    test_receive_accumulates_potential_and_trace()
    test_trace_gating_only_active_synapses_grow()
    test_sign_preserving_update()
    test_no_update_without_firing()
    test_membrane_and_trace_leak_together()
    test_weight_cap()
    test_refractory_blocks_accumulation()
    test_flexible_neuron_parity()
    test_inhibitory_event_dynamics_and_learning()
    test_inhibitory_learning_prefers_near_threshold()
    test_inhibitory_saturation_at_wmax()
    test_inhibitory_refractory_gate()
    test_inhibitory_independent_of_excitatory()
    test_flexible_neuron_inhibition_parity()
    test_activity_is_default_and_confidence_untouched()
    test_confidence_starts_small_grows_and_saturates()
    test_confidence_unchanged_without_a_spike()
    test_confidence_credit_is_normalized_and_budget_preserved()
    test_confidence_diverges_from_weight()
    test_flexible_confidence_parity()
    test_homeostasis_off_by_default_leaves_weights_alone()
    test_homeostasis_grows_a_chronically_silent_neuron()
    test_homeostasis_shrinks_a_hyperactive_neuron()
    test_homeostasis_scalar_is_fixed_not_error_proportional()
    test_flexible_homeostasis_parity()
    print("\nALL NEURON UNIT TESTS PASSED")
