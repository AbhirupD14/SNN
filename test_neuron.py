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
    print("\nALL NEURON UNIT TESTS PASSED")
