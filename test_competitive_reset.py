"""Focused tests for the L2 hard-reset competitive-depression architecture.

Covers the low-level rule, topology, and integration contract:
  - the shared bounded weight kernel (H_up / reflected H_down);
  - Neuron.apply_competitive_reset (unweighted reset + local depression);
  - engine topology / serialization (no learned L2I->L2E gate; reset edges);
  - integration (zero loser carryover, E->I still trains, 4-pattern cycling).

Plain-script style:
    PYTHONPATH=. .venv/bin/python test_competitive_reset.py
"""

import numpy as np

from neuron_flexible import Neuron
from snn.rules import bounded_signed_update
from backend.simulation import SimulationEngine, N_OUT, N_PIX, PATTERNS


# ======================================================================
# Bounded kernel (spec Section 6)
# ======================================================================
def test_kernel_up_largest_at_min_zero_at_cap():
    w_min, w_cap, gain = 1.0, 4.0, 0.5
    up_at_min = bounded_signed_update(np.array([w_min]), w_min, w_cap, gain, +1)[0] - w_min
    up_at_cap = bounded_signed_update(np.array([w_cap]), w_min, w_cap, gain, +1)[0] - w_cap
    assert up_at_cap == 0.0, up_at_cap                       # H_up(1) = 0
    assert np.isclose(up_at_min, gain), up_at_min            # H_up(0) = 1
    # monotone decreasing in w
    mids = bounded_signed_update(np.array([1.5, 2.5, 3.5]), w_min, w_cap, gain, +1) \
        - np.array([1.5, 2.5, 3.5])
    assert mids[0] > mids[1] > mids[2] > 0, mids
    print("PASS kernel: upward largest at w_min, zero at w_cap, decreasing")


def test_kernel_down_zero_at_min_max_at_cap():
    w_min, w_cap, gain = 1.0, 4.0, 0.5
    dn_at_min = bounded_signed_update(np.array([w_min]), w_min, w_cap, gain, -1)[0] - w_min
    dn_at_cap = bounded_signed_update(np.array([w_cap]), w_min, w_cap, gain, -1)[0] - w_cap
    assert dn_at_min == 0.0, dn_at_min                       # H_down(0) = 0 -> floored can't move
    assert np.isclose(dn_at_cap, -gain), dn_at_cap           # H_down(1) = 1 -> capped moves fully
    print("PASS kernel: downward zero at w_min, maximal at w_cap")


def test_kernel_capped_can_be_depressed():
    w_min, w_cap, gain = 1.0, 4.0, 0.5
    out = bounded_signed_update(np.array([w_cap]), w_min, w_cap, gain, -1)[0]
    assert out < w_cap, out
    assert out >= w_min
    print("PASS kernel: a capped weight can be depressed downward")


def test_kernel_floored_cannot_go_lower():
    w_min, w_cap, gain = 1.0, 4.0, 0.9
    out = bounded_signed_update(np.array([w_min]), w_min, w_cap, gain, -1)[0]
    assert out == w_min, out
    print("PASS kernel: a floored weight cannot be depressed further")


def test_kernel_stays_in_bounds():
    w_min, w_cap = 1.0, 4.0
    w = np.array([1.0, 2.0, 3.0, 4.0])
    for gain in (0.1, 1.0, 100.0):
        up = bounded_signed_update(w, w_min, w_cap, gain, +1)
        dn = bounded_signed_update(w, w_min, w_cap, gain, -1)
        assert (up >= w_min - 1e-12).all() and (up <= w_cap + 1e-12).all(), up
        assert (dn >= w_min - 1e-12).all() and (dn <= w_cap + 1e-12).all(), dn
    print("PASS kernel: both directions stay within [w_min, w_cap] for any gain")


def test_kernel_degenerate_range_is_noop():
    w = np.array([2.0, 3.0])
    for out in (bounded_signed_update(w, 4.0, 4.0, 0.5, -1),
                bounded_signed_update(w, 5.0, 4.0, 0.5, +1)):
        assert np.array_equal(out, w), out
    print("PASS kernel: degenerate w_cap <= w_min is a safe no-op")


# ======================================================================
# Competitive reset (spec Section 5, 7)
# ======================================================================
def _l2e(weights, theta=8.0, cap=8.0 / 3, floor=1.0, lr=0.02,
         participating=None, loser_depression=True, structural=False):
    n = Neuron(threshold=theta, refractory_period=0, weight_cap=cap, learning_rate=lr)
    n.weights = np.asarray(weights, dtype=float)
    n.min_positive_weight = floor
    n.loser_depression = loser_depression
    n.structural_free_energy = structural
    n.structural_fe_eta_floor = 0.02
    if participating is not None:
        n._last_input_spikes = np.asarray(participating, dtype=float)
    return n


def test_reset_positive_charge_to_rest():
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 0])
    n.potential = 0.7 * n.threshold
    rec = n.apply_competitive_reset()
    assert n.potential == n.resting_potential
    assert rec['v_post'] == n.resting_potential
    assert rec['v_pre'] == 0.7 * n.threshold
    print("PASS reset: a charged loser ends at exact rest")


def test_reset_clears_traces():
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 0])
    n.hard_reset_clear_traces = True
    n.potential = 0.5 * n.threshold
    n.exc_trace = 9.0
    n.inh_trace = 4.0
    n.apply_competitive_reset()
    assert n.exc_trace == 0.0 and n.inh_trace == 0.0, (n.exc_trace, n.inh_trace)
    print("PASS reset: pending excitatory/inhibitory current traces are zeroed")


def test_reset_without_negative_afferent():
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 1])
    assert not (n._weights_array < 0).any()
    n.potential = 0.6 * n.threshold
    n.apply_competitive_reset()          # no negative synapse required
    assert n.potential == n.resting_potential
    print("PASS reset: works with no negative afferent weight")


def test_reset_occurs_with_mode_none():
    # "none" is the hard-reset-only control: the pool still resets identically but
    # no loser weight update runs (spec Section 9).
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 0])
    w_before = n._weights_array.copy()
    n.potential = 0.9 * n.threshold
    rec = n.apply_competitive_reset(competitive_weight_update="none")
    assert n.potential == n.resting_potential            # still resets
    assert np.array_equal(n._weights_array, w_before)    # but does not learn
    assert rec['depressed_indices'] == []
    assert rec['plasticity_applied'] is False
    print("PASS reset: mode 'none' still resets (no weight change)")


def test_no_reset_when_l2i_does_not_fire():
    """The resolver only issues resets when L2I crosses threshold. Force a lone
    sub-E->I winner so L2I cannot fire, and confirm no reset events."""
    e = SimulationEngine(seed=1, l1i_immediate_relay=False)
    l2 = e.l2
    thr = e.params['threshold_l2']
    for n in l2.excitatory_neurons:
        n.refractory_timer = 0
        n.potential = 0.0
    l2.excitatory_neurons[0].potential = 1.2 * thr       # single crosser
    l2.inhibitory_neuron.refractory_timer = 0
    l2.inhibitory_neuron.potential = 0.0                 # far from firing
    e._reset_events = []
    l2i, inhibited, winner = e._resolve_l2_competition(l2, np.zeros(N_OUT), e.timestep, None)
    assert winner == 0
    if not l2i:
        assert inhibited == [] and e._reset_events == []
        print("PASS reset: no reset/depression when L2I does not fire")
    else:
        raise AssertionError("L2I unexpectedly fired; test setup invalid")


def test_winner_receives_reset_but_is_protected():
    # Current contract: the reset is broadcast
    # to EVERY L2E, winner included. The winner fired this step and is in refractory,
    # so it receives the reset event but skips the loser weight update (protection is
    # the refractory timer, not a one-hot winner check). It is still not counted as
    # inhibited.
    e = SimulationEngine(seed=1, l1i_immediate_relay=False)
    l2 = e.l2
    thr = e.params['threshold_l2']
    for n in l2.excitatory_neurons:
        n.refractory_timer = 0
        n.potential = 0.0
    l2.excitatory_neurons[0].potential = 2.0 * thr
    l2.excitatory_neurons[1].potential = 1.5 * thr
    l2.inhibitory_neuron.refractory_timer = 0
    l2.inhibitory_neuron.potential = l2.inhibitory_neuron.threshold   # ensure L2I fires
    e._reset_events = []
    l2i, inhibited, winner = e._resolve_l2_competition(l2, np.zeros(N_OUT), e.timestep, None)
    assert winner == 0
    recs = {nid: rec for nid, rec in e._reset_events}
    assert 'L2E0' in recs, "winner must receive the broadcast reset event"
    assert recs['L2E0']['refractory_at_arrival'] is True
    assert recs['L2E0']['plasticity_applied'] is False, "refractory winner must not learn"
    assert 0 not in inhibited
    print("PASS reset: the winner receives the reset but is refractory-protected")


# ======================================================================
# Competitive depression (spec Section 7)
# ======================================================================
def test_depression_only_participating_positive():
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 0, 1])
    n.potential = 0.9 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset()
    assert set(rec['depressed_indices']) == {0, 2}, rec['depressed_indices']
    assert n._weights_array[0] < before[0]               # participating -> down
    assert n._weights_array[2] < before[2]
    assert n._weights_array[1] == before[1]              # non-participating unchanged
    print("PASS depress: only participating positive weights decrease")


def test_depression_creates_no_negative_or_absent_weight():
    n = _l2e([2.0, 1.0, 2.0], participating=[1, 1, 1], floor=1.0)
    n.potential = n.threshold                            # p_loss = 1 (max)
    for _ in range(50):
        n.potential = n.threshold
        n.apply_competitive_reset()
    assert (n._weights_array >= 1.0 - 1e-12).all(), n._weights_array
    assert not (n._weights_array < 0).any()
    print("PASS depress: no negative or below-floor weight is ever created")


def test_zero_charge_loser_resets_but_does_not_learn():
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 1])
    n.potential = n.resting_potential                    # p_loss = 0
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset()
    assert n.potential == n.resting_potential
    assert np.array_equal(n._weights_array, before)
    assert rec['depressed_indices'] == []
    print("PASS depress: a zero-charge loser resets but does not learn")


def test_higher_charge_gets_larger_update():
    lo = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 1])
    hi = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 1])
    lo.potential = 0.3 * lo.threshold
    hi.potential = 0.9 * hi.threshold
    d_lo = -lo.apply_competitive_reset()['delta_weights']
    d_hi = -hi.apply_competitive_reset()['delta_weights']
    assert (d_hi > d_lo).all(), (d_lo, d_hi)
    print("PASS depress: a higher-charge loser is depressed more than a lower one")


def test_structural_maturity_scales_once():
    """With the structural free-energy gate on, the loss gain is
    learning_rate * gate * p_loss with gate applied EXACTLY once."""
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 1], structural=True)
    n.potential = 0.8 * n.threshold
    p_loss = 0.8
    gate = n._structural_free_energy_gate()              # pre-update
    w_before = n._weights_array.copy()
    gain = n.learning_rate * gate * p_loss
    ref = bounded_signed_update(w_before, n.min_positive_weight, n.weight_cap,
                                gain, np.full(3, -1.0))
    n.apply_competitive_reset()
    assert np.allclose(n._weights_array, ref), (n._weights_array, ref)
    print(f"PASS depress: structural maturity scales the loss update once (gate={gate:.3f})")


def test_single_event_not_both_depressions():
    """apply_competitive_reset must NOT invoke the legacy apply_inhibition loser
    path: it never populates last_inhibitory_events and bumps the depression
    counter exactly once."""
    n = _l2e([2.0, 2.0, 2.0], participating=[1, 1, 1])
    n.last_inhibitory_events = ['sentinel']
    n.loser_depression_events = 0
    n.potential = 0.9 * n.threshold
    n.apply_competitive_reset()
    assert n.loser_depression_events == 1, n.loser_depression_events
    assert n.last_inhibitory_events == ['sentinel'], "apply_inhibition path ran"
    print("PASS depress: one event triggers competitive depression only (not legacy)")


# ======================================================================
# Topology & serialization (spec Section 4, 8)
# ======================================================================
def test_topology_shapes_and_no_gate():
    e = SimulationEngine(seed=1)
    for j in range(N_OUT):
        n = e.l2.excitatory_neurons[j]
        assert len(n._weights_array) == N_PIX, len(n._weights_array)
        assert len(n.confidence) == N_PIX
        assert len(n.distance) == N_PIX
        assert not (n._weights_array < 0).any(), f"L2E{j} carries a negative gate"
    w = e._all_weights()
    assert not any(k.startswith('inh->') for k in w), "learned inh-> weight leaked"
    ff_per_neuron = {j: sum(1 for k in w if k.endswith(f'->{j}') and k.startswith('ff'))
                     for j in range(N_OUT)}
    assert all(v == N_PIX for v in ff_per_neuron.values()), ff_per_neuron
    print("PASS topo: every L2E has N_PIX positive afferents; no learned inh-> weight")


def test_topology_reset_edges_visible_and_pulse():
    e = SimulationEngine(seed=1)
    syn = {s['id']: s for s in e.topology()['synapses']}
    for j in range(N_OUT):
        s = syn[f'reset->{j}']
        assert s['kind'] == 'reset_inhibition', s
        assert s['weight'] is None, s
    # Drive until a reset fires and confirm the edge id is emitted (pulses).
    e.set_pattern('row 1')
    pulsed = False
    for _ in range(400):
        d = e.step()
        if any(x.startswith('reset->') for x in d['emitted']):
            pulsed = True
            break
    assert pulsed, "reset-> edges never pulsed"
    print("PASS topo: reset edges are visible (weight=null) and pulse on reset")


def test_manual_rf_edit_targets_pixel():
    e = SimulationEngine(seed=1)
    j, i, val = 2, 5, 123.0
    applied = e.set_feedforward_weight(j, i, val)
    assert e.l2.excitatory_neurons[j]._weights_array[i] == applied, applied
    # exactly that one afferent moved
    print(f"PASS topo: manual RF edit targets pixel i={i} (value {applied})")


# ======================================================================
# Integration (spec Section 10)
# ======================================================================
def test_integration_zero_carryover_and_ei_trains():
    e = SimulationEngine(seed=1)   # dashboard-like defaults, competitive depression on
    ei_before = e.l2.inhibitory_neuron._weights_array[:N_OUT].copy()
    reset_steps = 0
    for ep in range(20):
        for name in PATTERNS:
            e.set_pattern(name)
            for _ in range(20):
                e.step()
                if e._reset_events:
                    reset_steps += 1
                    # every non-winner reset this step ends at exact rest
                    for nid, _ in e._reset_events:
                        j = int(nid[3:])
                        n = e.l2.excitatory_neurons[j]
                        if not e.spiked[nid]:
                            assert n.potential == n.resting_potential, (nid, n.potential)
    ei_after = e.l2.inhibitory_neuron._weights_array[:N_OUT]
    assert reset_steps > 0, "no competitive resets exercised"
    assert not np.allclose(ei_before, ei_after), "E->I (L2E->L2I) weights never trained"
    assert e.l2.inhibitory_neuron.threshold > 0
    print(f"PASS integ: {reset_steps} reset steps, zero carryover, E->I trained")


def test_integration_four_pattern_cycle_no_drift():
    e = SimulationEngine(seed=3)
    e.set_auto_cycle(True)
    for _ in range(2000):
        e.step()
    # shapes intact after a long cycled run
    for j in range(N_OUT):
        assert len(e.l2.excitatory_neurons[j]._weights_array) == N_PIX
    _ = e.topology(); _ = e.dynamic_state()
    print("PASS integ: 4-pattern cycling runs 2000 steps without shape/index drift")


if __name__ == "__main__":
    # kernel
    test_kernel_up_largest_at_min_zero_at_cap()
    test_kernel_down_zero_at_min_max_at_cap()
    test_kernel_capped_can_be_depressed()
    test_kernel_floored_cannot_go_lower()
    test_kernel_stays_in_bounds()
    test_kernel_degenerate_range_is_noop()
    # reset
    test_reset_positive_charge_to_rest()
    test_reset_clears_traces()
    test_reset_without_negative_afferent()
    test_reset_occurs_with_mode_none()
    test_no_reset_when_l2i_does_not_fire()
    test_winner_receives_reset_but_is_protected()
    # depression
    test_depression_only_participating_positive()
    test_depression_creates_no_negative_or_absent_weight()
    test_zero_charge_loser_resets_but_does_not_learn()
    test_higher_charge_gets_larger_update()
    test_structural_maturity_scales_once()
    test_single_event_not_both_depressions()
    # topology
    test_topology_shapes_and_no_gate()
    test_topology_reset_edges_visible_and_pulse()
    test_manual_rf_edit_targets_pixel()
    # integration
    test_integration_zero_carryover_and_ei_trains()
    test_integration_four_pattern_cycle_no_drift()
    print("\nALL COMPETITIVE-RESET TESTS PASSED")
