"""
Unit tests for the single spiking Neuron implementation in neuron_flexible.py.

Covers the charge-based weight-update rule shared by both signs of synapse:
positive (excitatory) synapses update on this neuron's own fire, negative
(inhibitory) synapses update event-driven via apply_inhibition -- same
algorithm (capture charge, discharge, dw = eta*p*(1-w^2/w_max)), different
trigger and an inverted p.

The previous trace-gated/sign-preserving "activity" rule and the
confidence-weighted credit-splitting "confidence" rule are ARCHIVED -- see git
history prior to this file's rewrite for their tests.
"""
import numpy as np
from neuron_flexible import Neuron


def test_receive_accumulates_potential_and_trace():
    n = Neuron(n_inputs=3, threshold=10.0, leak_rate=0.0)
    n.weights = np.array([0.2, 0.5, 0.3])
    n.receive_input(np.array([1, 0, 1]))
    assert np.isclose(n.potential, 0.5)              # 0.2 + 0.3
    assert np.allclose(n.trace, [1, 0, 1])           # ARCHIVED bookkeeping, still tracked
    assert np.allclose(n._last_input_spikes, [1, 0, 1])  # instantaneous participation signal
    print("PASS: receive_input accumulates membrane potential, and both trace and the instantaneous spike signal")


def test_only_instantaneously_active_synapses_grow():
    """Only synapses active in the immediately preceding receive_input() call
    are updated -- this replaces the old accumulated-trace participation."""
    n = Neuron(n_inputs=4, threshold=0.5, refractory_period=2,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    n.weights = np.array([0.4, 0.4, 0.4, 0.4])
    n.receive_input(np.array([1, 1, 0, 0]))          # only lines 0,1 active -> v_pre = 0.8
    assert n.check_threshold()
    n.fire()
    # p = clamp(theta/v_pre, 0, 1) = clamp(0.5/0.8, 0, 1) = 0.625
    # dw = eta*p*(1 - w^2/w_max) = 0.1*0.625*(1 - 0.16/10) = 0.0615 -> w = 0.4615
    assert np.allclose(n.weights[:2], 0.4615, atol=1e-4)   # active grew
    assert np.allclose(n.weights[2:], 0.4)                  # silent untouched
    print("PASS: only instantaneously-active synapses are credited")


def test_participation_does_not_persist_across_steps():
    """A synapse that delivered charge several steps ago but not in the step
    that triggered firing is NOT credited -- the old multi-step trace memory
    is gone; only the immediately preceding receive_input() call matters."""
    n = Neuron(n_inputs=2, threshold=1.5, refractory_period=0,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    n.weights = np.array([0.4, 0.4])
    n.receive_input(np.array([1, 0]))                # line 0 contributes 0.4, potential=0.4
    n.receive_input(np.array([0, 1]))                # line 1 contributes 0.4, potential=0.8
    n.potential = 1.5                                 # force threshold crossing on a line-1-only step
    n.receive_input(np.array([0, 1]))                # only line 1 active THIS step
    assert n.check_threshold()
    n.fire()
    assert np.isclose(n.weights[0], 0.4)              # line 0's earlier contribution: NOT credited
    assert n.weights[1] > 0.4                          # line 1 (this step's participant): credited
    print("PASS: only the most recent triggering event's spikes are credited, not a multi-step history")


def test_fire_only_updates_positive_weights():
    """apply_inhibition is the only rule that ever moves a negative synapse;
    firing never touches one, even if it delivered charge this step.
    (threshold=0.0 would make p=theta/v_pre degenerate to 0 for the new
    charge-based rule -- unlike the old trace rule, this one needs theta>0.)"""
    n = Neuron(n_inputs=2, threshold=0.5, refractory_period=2,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    n.weights = np.array([0.4, -0.4])                # one excitatory, one inhibitory synapse
    n.receive_input(np.array([1, 1]))
    n.potential = 1.0                                # force a spike, above threshold
    assert n.check_threshold()
    n.fire()
    assert n.weights[0] > 0.4, n.weights              # excitatory grows
    assert np.isclose(n.weights[1], -0.4), n.weights  # inhibitory UNTOUCHED by fire()
    print("PASS: fire() only ever updates positive weights; negative ones are untouched")


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


def test_weight_cap_and_saturation_equilibrium():
    """Repeated firing drives the active positive weight toward its natural
    equilibrium w* = sqrt(w_max) when w_max > 1 (same quadratic-saturation
    property as apply_inhibition), never past
    w_max. (For w_max < 1, sqrt(w_max) > w_max, so growth never reverses
    within the clip range and the weight simply saturates at the cap instead
    -- see test_inhibitory_saturation_at_wmax for that regime.)"""
    n = Neuron(n_inputs=2, threshold=0.5, refractory_period=0,
               learning_rate=0.3, weight_cap=2.0, leak_rate=0.0)
    n.weights = np.array([0.1, -0.4])
    for _ in range(500):
        n.receive_input(np.array([1, 1]))
        n.potential = 5.0                            # force firing every step
        if n.check_threshold():
            n.fire()
        n.update()
    assert n.weights[0] <= 2.0 + 1e-9                # never exceeds cap
    assert np.isclose(n.weights[0], np.sqrt(2.0), atol=1e-3)   # settles at sqrt(w_max), not w_max
    assert np.isclose(n.weights[1], -0.4)             # inhibitory weight never touched by fire()
    print("PASS: repeated firing drives the active weight to its sqrt(w_max) equilibrium, never beyond w_max")


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


def test_charge_based_excitatory_rule_smaller_for_more_charge():
    """The core property requested: a neuron that fires with LESS charge
    (closer to its own threshold) gets a BIGGER update; one that fires with
    MORE charge gets a SMALLER one -- the inverse of apply_inhibition's p."""
    marginal = Neuron(n_inputs=1, threshold=1.0, weight_cap=10.0, leak_rate=0.0, learning_rate=0.2)
    overshoot = Neuron(n_inputs=1, threshold=1.0, weight_cap=10.0, leak_rate=0.0, learning_rate=0.2)
    marginal.weights = np.array([0.3]); marginal.receive_input(np.array([1])); marginal.potential = 1.0   # v_pre = theta
    overshoot.weights = np.array([0.3]); overshoot.receive_input(np.array([1])); overshoot.potential = 5.0  # v_pre >> theta
    marginal.fire()
    overshoot.fire()
    d_marginal = marginal.weights[0] - 0.3
    d_overshoot = overshoot.weights[0] - 0.3
    assert d_marginal > d_overshoot > 0, (d_marginal, d_overshoot)
    print("PASS: firing with less charge drives a bigger update than firing with more charge")


def test_charge_based_excitatory_exact_numeric():
    """Exact numeric check, mirroring test_inhibitory_event_dynamics_and_learning."""
    n = Neuron(n_inputs=1, threshold=2.0, weight_cap=1.0, leak_rate=0.0, learning_rate=0.1)
    n.weights = np.array([0.4])
    n.receive_input(np.array([1]))
    n.potential = 4.0                                 # v_pre = 4.0 (theta=2.0 -> p=0.5)
    n.fire()
    # p = clamp(theta/v_pre, 0, 1) = clamp(2.0/4.0, 0, 1) = 0.5
    # dw = eta*p*(1 - w^2/w_max) = 0.1*0.5*(1 - 0.16) = 0.042 -> w = 0.442
    assert np.isclose(n.weights[0], 0.442)
    print("PASS: charge-based excitatory update matches eta*p*(1-w^2/w_max) exactly, p=theta/v_pre")


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
    # dw = eta*p*(1 - w^2/w_max) = 0.1*0.5*(1 - 0.16) = 0.042  ->  w = 0.442
    assert np.isclose(ev['delta_w'], 0.042) and np.isclose(ev['w_after'], 0.442)
    assert np.isclose(n.weights[0], -0.442)          # still inhibitory, magnitude grew
    print("PASS: inhibitory event applies V=V-w and strengthens the gate by eta*p*(1-w^2/w_max)")


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
    delivered via fire() leaves separately-delivered inhibitory gates alone."""
    n = Neuron(n_inputs=2, threshold=0.5, refractory_period=0,
               learning_rate=0.1, weight_cap=10.0, leak_rate=0.0, inhibitory_learning_rate=0.3)
    n.weights = np.array([0.6, -0.4])                # [excitatory, inhibitory]
    # Inhibitory event must not touch the excitatory weight.
    n.potential = 0.4
    n.apply_inhibition(np.array([0, 1]))
    assert np.isclose(n.weights[0], 0.6), "excitatory weight changed on an inhibitory event"
    # Excitatory spike (charge only on line 0) must not touch the inhibitory gate.
    w_inh = n.weights[1]
    n.receive_input(np.array([1, 0]))
    n.potential = 1.0
    n.fire()
    assert np.isclose(n.weights[1], w_inh), "inhibitory gate changed on an excitatory spike"
    assert n.weights[0] > 0.6, "excitatory rule should still strengthen the active E synapse"
    print("PASS: the two learning systems are independent")


def test_dynamic_fanin_inhibition():
    """Dynamic fan-in construction runs the inhibitory rule."""
    fn = Neuron(threshold=2.0, weight_cap=1.0, leak_rate=0.0, inhibitory_learning_rate=0.1)
    for w in [0.5, -0.4]:
        fn.add_input_connection(w)
    fn.finalize_connections()
    fn.potential = 1.0
    ev = fn.apply_inhibition(np.array([0, 1]))[0]
    assert np.isclose(fn.potential, 0.6)
    assert np.isclose(ev['p'], 0.5) and np.isclose(ev['delta_w'], 0.042)
    assert np.isclose(fn.weights[1], -0.442) and np.isclose(fn.weights[0], 0.5)
    print("PASS: dynamic fan-in neuron applies the inhibitory rule (excitatory synapse untouched)")


def test_dynamic_fanin_excitatory_rule():
    """Dynamic fan-in construction runs the charge-based excitatory rule."""
    fn = Neuron(threshold=0.5, refractory_period=2,
                learning_rate=0.1, weight_cap=10.0, leak_rate=0.0)
    for w in [0.4, 0.4, -0.4]:
        fn.add_input_connection(w)
    fn.finalize_connections()
    fn.receive_input(np.array([1, 1, 0]))            # v_pre = 0.4 + 0.4 = 0.8; line 2 silent
    assert fn.check_threshold()
    fn.fire()
    # p = clamp(0.5/0.8, 0, 1) = 0.625; dw = 0.1*0.625*(1-0.16/10) = 0.0615
    assert np.isclose(fn.weights[0], 0.4615, atol=1e-4)   # active excitatory grew
    assert np.isclose(fn.weights[1], 0.4615, atol=1e-4)   # active excitatory grew
    assert np.isclose(fn.weights[2], -0.4)                 # inhibitory (and silent) untouched by fire()
    print("PASS: dynamic fan-in neuron runs the excitatory rule")


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


def test_dynamic_fanin_homeostasis():
    base = Neuron(n_inputs=3, threshold=1.0, leak_rate=0.0, weight_cap=10.0,
                  homeostasis=True, ca_rate=0.1, ca_target=0.10,
                  homeo_up=0.02, homeo_down=0.02, homeo_budget_max=100.0)
    base.weights = np.array([0.3, 0.3, 0.3])
    fx = Neuron(threshold=1.0, leak_rate=0.0, weight_cap=10.0,
                homeostasis=True, ca_rate=0.1, ca_target=0.10,
                homeo_up=0.02, homeo_down=0.02, homeo_budget_max=100.0)
    for w in [0.3, 0.3, 0.3]:
        fx.add_input_connection(w)
    fx.finalize_connections()
    for _ in range(150):
        base.update(); fx.update()
    assert np.allclose(base.weights, fx.weights)
    assert np.isclose(base.ca, fx.ca) and np.isclose(base.homeo_budget, fx.homeo_budget)
    print("PASS: fixed and dynamic fan-in construction match under homeostatic scaling")


def test_fixed_point_scale_invariance():
    """The fixed-point convention (linear magnitudes * UNIT, quadratic saturation
    denominators * UNIT**2, dimensionless rates unchanged) must not change the
    dynamics -- only the numeric scale. Runs the SAME neuron (excitatory rule,
    inhibitory gate, confidence consolidation, and weight budget all active) at a
    1x and a 1000x scale with identical NORMALIZED drive, and asserts the
    normalized weight trajectories are bit-for-bit identical. This is the
    regression guard for the UNIT=1000 scaling: if a magnitude is ever mis-scaled
    (linear where it should be quadratic, or a dimensionful knob left unscaled),
    this diverges."""
    def run(U, seed=0):
        n = Neuron()
        n.weight_cap = 1.0 * U
        n.threshold = 1.0 * U
        n.excitatory_saturation_cap = (1.0 * U) ** 2   # quadratic (w/wcap)^2 form
        n.inhibitory_weight_cap = 1.5 * U * U           # quadratic gate w_max
        n.inhibitory_learning_rate = 0.1 * U            # linear eta
        n.learning_rate = 0.01 * U                      # linear eta
        n.confidence_consolidation = True
        n.conf_cap = (1.0 / 3.0) * U
        n.conf_beta = 0.05
        n.eta_min = 0.05
        n.weight_budget = 8.0 * U
        n.min_positive_weight = 0.01 * U
        ff = np.array([0.05, 0.20, 0.08, 0.15]) * U
        n._set_weights(np.concatenate([[-0.5 * U], ff]), finalized=True)
        n._confidence = np.full(5, 0.10)
        n._apply_budget_and_cap()
        spikes = np.array([0.0, 1.0, 1.0, 0.0, 0.0])    # pixels 1,2 active
        traj = []
        for _ in range(40):
            n.potential = 0.0
            for _ in range(6):
                n.receive_input(spikes)
            if n.potential >= n.threshold:
                n.fire()
            else:
                n.apply_inhibition(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
            traj.append(np.abs(n._weights_array).copy() / U)   # NORMALIZED
            n.update()
        return np.array(traj)

    a, b = run(1.0), run(1000.0)
    assert np.array_equal(a, b), \
        f"scale not invariant: max |1x-1000x| = {np.abs(a - b).max():.3e}"
    print("PASS: fixed-point scaling is exactly dynamics-invariant (1x == 1000x normalized)")


if __name__ == "__main__":
    test_fixed_point_scale_invariance()
    test_receive_accumulates_potential_and_trace()
    test_only_instantaneously_active_synapses_grow()
    test_participation_does_not_persist_across_steps()
    test_fire_only_updates_positive_weights()
    test_no_update_without_firing()
    test_membrane_and_trace_leak_together()
    test_weight_cap_and_saturation_equilibrium()
    test_refractory_blocks_accumulation()
    test_charge_based_excitatory_rule_smaller_for_more_charge()
    test_charge_based_excitatory_exact_numeric()
    test_dynamic_fanin_excitatory_rule()
    test_inhibitory_event_dynamics_and_learning()
    test_inhibitory_learning_prefers_near_threshold()
    test_inhibitory_saturation_at_wmax()
    test_inhibitory_refractory_gate()
    test_inhibitory_independent_of_excitatory()
    test_dynamic_fanin_inhibition()
    test_homeostasis_off_by_default_leaves_weights_alone()
    test_homeostasis_grows_a_chronically_silent_neuron()
    test_homeostasis_shrinks_a_hyperactive_neuron()
    test_homeostasis_scalar_is_fixed_not_error_proportional()
    test_dynamic_fanin_homeostasis()
    print("\nALL NEURON UNIT TESTS PASSED")
