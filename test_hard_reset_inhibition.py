"""
Tests for hard-reset inhibition (Hard_Reset_Inhibition_Plan.md).

The idea: once L2I declares a winner, each losing L2E's accumulated membrane
charge is first CONSUMED by the local inhibitory-plasticity rule (which reads the
pre-reset charge as evidence the neuron was a real competitor) and only THEN
cleared back to rest. Weights are never wiped -- only transient membrane charge.

This attacks the round-robin asymmetry directly:

    winner fires  -> winner resets to rest
    losers inhibited -> (baseline) losers KEEP most of their charge -> start ahead
    hard reset       -> losers return to the SAME baseline as the winner

The eight checks below mirror the plan's "Tests" section:

  1. flag off preserves current inhibition behaviour (subtractive/floored gate)
  2. on an L2I fire, a non-winning L2E's V is reset to rest
  3. the winner L2E is not reset by L2I (it is never passed to apply_inhibition)
  4. inhibitory learning uses v_pre captured BEFORE the reset
  5. the reset happens AFTER inhibitory learning, not before
  6. loser depression is disabled in the clean hard-reset preset
  7. the signed-spike L2E feedforward update is unchanged by the flag
  8. flow-rate current traces are cleared under the hard-reset preset
"""

import numpy as np

from neuron_flexible import Neuron
from backend.simulation import SimulationEngine, N_OUT, PATTERNS


# ----------------------------------------------------------------------------
# The "clean hard-reset preset" from the plan's Intended Minimal Regime:
# signed-spike feedforward + hard L2I loser reset, with every compensating
# mechanism (loser depression, confidence, signed depression, budget,
# homeostasis, lasting inhibition, flow-rate) turned OFF and no membrane noise.
# ----------------------------------------------------------------------------
HARD_RESET_PRESET = dict(
    signed_spike_learning=True,
    l2i_hard_reset_losers=True,
    hard_reset_clear_traces=True,
    loser_depression=False,
    confidence_consolidation=False,
    signed_depression=False,
    l2e_budget=False,
    homeostasis=False,
    lasting_inhibition=False,
    inhibitory_flow_rate=False,
    excitatory_flow_rate=False,
)


def _gate_neuron(v0, gate=0.3, theta=1.0, eta=0.2, hard_reset=False,
                 clear_traces=True):
    """A single-gate excitatory neuron sitting at V=v0 with one inhibitory
    afferent (index 0, magnitude `gate`). Mirrors an L2E carrying its L2I->L2E
    gate at synapse 0."""
    n = Neuron(n_inputs=1, threshold=theta, weight_cap=theta, leak_rate=0.0,
               refractory_period=0, inhibitory_learning_rate=eta,
               inhibitory_weight_cap=theta)
    n.weights = np.array([-gate])
    n.potential = v0
    n.l2i_hard_reset_losers = hard_reset
    n.hard_reset_clear_traces = clear_traces
    return n


# --------------------------------------------------------------- test 1
def test_flag_off_preserves_behaviour():
    """With the flag OFF, apply_inhibition still subtracts the gate magnitude and
    floors at rest -- the membrane is NOT clamped to rest."""
    V0, GATE = 0.9, 0.3
    n = _gate_neuron(V0, gate=GATE, hard_reset=False)
    n.apply_inhibition(np.array([1.0]))
    assert np.isclose(n.potential, V0 - GATE), n.potential
    assert n.potential > n.resting_potential
    print(f"PASS test1: flag off -> subtractive discharge kept "
          f"(V {V0} -> {n.potential:.3f}, not rest)")


# --------------------------------------------------------------- test 2
def test_loser_reset_to_rest():
    """With the flag ON, a real inhibitory discharge clamps the loser to rest."""
    V0 = 0.9
    n = _gate_neuron(V0, gate=0.3, hard_reset=True)
    events = n.apply_inhibition(np.array([1.0]))
    assert events, "expected an inhibitory discharge event"
    assert np.isclose(n.potential, n.resting_potential), n.potential
    print(f"PASS test2: loser hard-reset to rest (V {V0} -> {n.potential:.3f})")


def test_no_reset_without_discharge():
    """The flag only fires on a REAL discharge (events non-empty). No inhibitory
    spike this step -> no reset."""
    V0 = 0.9
    n = _gate_neuron(V0, gate=0.3, hard_reset=True)
    events = n.apply_inhibition(np.array([0.0]))   # no inhibitory spike
    assert not events
    assert np.isclose(n.potential, V0), n.potential
    print("PASS test2b: no discharge -> no reset (charge preserved)")


# --------------------------------------------------------------- test 3
def test_winner_not_reset():
    """In the L2 competition the winner fires and is never passed to
    apply_inhibition; only non-winners are inhibited (and hard-reset)."""
    e = SimulationEngine(seed=1, l2i_hard_reset_losers=True)
    l2 = e.l2
    thr = e.params['threshold_l2']
    # Force two L2E over threshold with a clear max, everyone else below.
    for j, n in enumerate(l2.excitatory_neurons):
        n.refractory_timer = 0
        n.last_inhibitory_events = []
        n.potential = 0.0
    winner_j, loser_j = 0, 1
    l2.excitatory_neurons[winner_j].potential = 2.0 * thr    # clear max
    l2.excitatory_neurons[loser_j].potential = 1.5 * thr     # crosses, loses
    # Pre-charge L2I to its threshold so the winner's spike reliably fires it
    # (E->I weights start well below threshold, so a lone winner otherwise won't).
    l2.inhibitory_neuron.refractory_timer = 0
    l2.inhibitory_neuron.potential = l2.inhibitory_neuron.threshold

    l2e = np.zeros(N_OUT)
    e._inh_events = []
    l2i, inhibited, winner = e._resolve_l2_competition(l2, l2e, e.timestep)

    assert winner == winner_j, f"expected winner {winner_j}, got {winner}"
    assert winner_j not in inhibited, "winner must not be inhibited"
    assert not l2.excitatory_neurons[winner_j].last_inhibitory_events, \
        "winner received an inhibitory discharge"
    if l2i:
        # The loser was inhibited and hard-reset to rest.
        assert loser_j in inhibited, inhibited
        assert np.isclose(l2.excitatory_neurons[loser_j].potential,
                          l2.excitatory_neurons[loser_j].resting_potential), \
            "loser was not hard-reset to rest"
        print(f"PASS test3: winner L2E{winner_j} not reset by L2I; "
              f"loser L2E{loser_j} hard-reset to rest")
    else:
        # If L2I didn't fire there is no inhibition at all -- still a valid check
        # that the winner is untouched by inhibition.
        print(f"PASS test3: winner L2E{winner_j} not inhibited (L2I did not fire)")


# --------------------------------------------------------------- tests 4 & 5
def test_learning_uses_pre_reset_charge():
    """Inhibitory-gate learning must read the loser's charge BEFORE the reset.
    Proven two ways:
      (4) the event's v_pre equals the pre-reset membrane, and the gate delta is
          bit-identical to the flag-OFF run (same learning, different post-state).
      (5) had the reset run FIRST, v_pre would be 0 -> p=0 -> dw=0; the gate still
          grows, so learning happened before the clear."""
    V0, GATE = 0.8, 0.3
    off = _gate_neuron(V0, gate=GATE, hard_reset=False)
    on = _gate_neuron(V0, gate=GATE, hard_reset=True)

    ev_off = off.apply_inhibition(np.array([1.0]))[0]
    ev_on = on.apply_inhibition(np.array([1.0]))[0]

    # (4) v_pre captured before reset, and gate learning identical to flag-off.
    assert np.isclose(ev_on['v_pre'], V0), ev_on['v_pre']
    assert np.isclose(ev_on['delta_w'], ev_off['delta_w']), \
        (ev_on['delta_w'], ev_off['delta_w'])
    assert np.isclose(-float(on.weights[0]), -float(off.weights[0])), \
        "hard reset changed what the inhibitory rule learned"
    # (5) ordering: the gate grew, which is impossible if V had been reset first.
    assert ev_on['delta_w'] > 0, "gate did not grow -- reset ran before learning?"
    # ...and the membrane really was cleared afterwards.
    assert np.isclose(on.potential, on.resting_potential), on.potential
    print(f"PASS test4/5: learning read v_pre={ev_on['v_pre']:.3f} (delta_w="
          f"{ev_on['delta_w']:+.5f}, matches flag-off), THEN reset to rest")


# --------------------------------------------------------------- test 6
def test_preset_disables_loser_depression():
    """The clean hard-reset preset turns loser depression off and hard reset on
    for every L2E (the plan forbids combining the two in the first pass)."""
    e = SimulationEngine(seed=1, **HARD_RESET_PRESET)
    l2e = [n for nid, n in e.neurons.items()
           if e.meta[nid]['type'] == 'E' and nid.startswith('L2')]
    assert l2e, "no L2E neurons found"
    for n in l2e:
        assert n.l2i_hard_reset_losers is True
        assert n.loser_depression is False
        assert n.confidence_consolidation is False
        assert n.signed_depression is False
    print(f"PASS test6: preset -> {len(l2e)} L2E with hard reset on, "
          f"loser depression / confidence / signed depression off")


# --------------------------------------------------------------- test 7
def test_signed_spike_update_unchanged():
    """The hard-reset flag lives in apply_inhibition (loser path); it must not
    touch the signed-spike feedforward update on the fire path. Two identical
    neurons -- one with the flag on, one off -- must learn bit-identically."""
    weights = [-0.3, 0.1, 0.2, 0.05, 0.0]   # index 0 = inhibitory gate
    inp = np.array([0.0, 1.0, 1.0, 0.0, 0.0])

    def _fire_once(hard_reset):
        n = Neuron(n_inputs=5, threshold=0.25, weight_cap=0.25, leak_rate=0.0,
                   refractory_period=0, learning_rate=0.02)
        n.weights = np.array(weights, dtype=float)
        n.min_positive_weight = 0.001
        n.signed_spike_learning = True
        n.l2i_hard_reset_losers = hard_reset
        n.receive_input(inp)
        n.fire()
        return n.weights.copy()

    w_off = _fire_once(False)
    w_on = _fire_once(True)
    assert np.array_equal(w_off, w_on), (w_off, w_on)

    # And the signed rule actually moved the feedforward weights (rule is alive).
    assert not np.array_equal(w_on, np.array(weights)), "signed update did nothing"
    print("PASS test7: signed-spike feedforward update bit-identical with the "
          "flag on vs off (and non-trivial)")


# --------------------------------------------------------------- test 8
def test_flow_traces_cleared_on_reset():
    """Under hard_reset_clear_traces, a reset zeroes the excitatory/inhibitory
    current traces so no residual flow refills the membrane after the clamp."""
    n = _gate_neuron(0.9, gate=0.3, hard_reset=True, clear_traces=True)
    n.exc_trace = 5.0
    n.inh_trace = 3.0
    n.apply_inhibition(np.array([1.0]))
    assert np.isclose(n.potential, n.resting_potential)
    assert n.exc_trace == 0.0 and n.inh_trace == 0.0, (n.exc_trace, n.inh_trace)

    # With clearing disabled, the reset still clamps V but leaves traces intact.
    m = _gate_neuron(0.9, gate=0.3, hard_reset=True, clear_traces=False)
    m.exc_trace = 5.0
    m.apply_inhibition(np.array([1.0]))
    assert np.isclose(m.potential, m.resting_potential)
    assert m.exc_trace == 5.0, m.exc_trace
    print("PASS test8: hard_reset_clear_traces zeroes exc/inh traces "
          "(and leaves them when disabled)")


# --------------------------------------------------------------- integration
def test_preset_eliminates_carryover_end_to_end():
    """Sanity end-to-end: under the preset, whenever L2I issues competitive resets
    the reset L2E neurons end the step at rest (zero carryover)."""
    e = SimulationEngine(seed=1, **HARD_RESET_PRESET)
    carryover_steps = 0
    discharge_steps = 0
    names = list(PATTERNS)
    for ep in range(40):
        for name in names:
            e.set_pattern(name)
            for _ in range(25):
                e.step()
                inhibited = [int(nid[3:]) for nid, _ in e._reset_events
                             if nid.startswith('L2E')]
                if inhibited:
                    discharge_steps += 1
                    for j in inhibited:
                        n = e.l2.excitatory_neurons[j]
                        if (n.potential > n.resting_potential + 1e-9
                                and not e.spiked[f'L2E{j}']):
                            carryover_steps += 1
                            break
    assert discharge_steps > 0, "L2I never discharged the pool -- nothing exercised"
    assert carryover_steps == 0, \
        f"{carryover_steps}/{discharge_steps} discharge steps left loser charge"
    print(f"PASS integration: {discharge_steps} L2I discharge steps, "
          f"zero loser carryover")


if __name__ == "__main__":
    test_flag_off_preserves_behaviour()
    test_loser_reset_to_rest()
    test_no_reset_without_discharge()
    test_winner_not_reset()
    test_learning_uses_pre_reset_charge()
    test_preset_disables_loser_depression()
    test_signed_spike_update_unchanged()
    test_flow_traces_cleared_on_reset()
    test_preset_eliminates_carryover_end_to_end()
    print("\nALL HARD-RESET INHIBITION TESTS PASSED")
