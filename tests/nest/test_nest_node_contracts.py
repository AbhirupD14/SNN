"""Phase 1 event-node microcontracts (prompt sections 6.1-6.4, 9 Phase 1, 11).

Every transition asserted here is caused by a recorded NEST event. No test advances state
by calling a Python helper, because that is precisely the modelling error that let the
reference engine ship a dead eligibility-carry path while its unit tests passed.
"""
from __future__ import annotations

import pytest

THETA = 1000.0
H = 0.1


def _accumulator(nest, **params):
    base = {"theta": THETA, "t_ref": 0.0}
    base.update(params)
    return nest.Create("event_accumulator", 1, base)


def _drive(nest, target, times, weight, port="exc", delay=H):
    receptors = nest.GetDefaults("event_accumulator").get("receptor_types") or {}
    gen = nest.Create("spike_generator", 1, {"spike_times": list(times)})
    spec = {"weight": weight, "delay": delay}
    if receptors:
        spec["receptor_type"] = receptors[port.upper()]
    nest.Connect(gen, target, syn_spec=spec)
    return gen


# ------------------------------------------------------------------ accumulator


def test_one_event_is_counted_once(fresh_kernel):
    nest = fresh_kernel()
    cell = _accumulator(nest)
    _drive(nest, cell, [1.0], THETA / 4)
    nest.Simulate(5.0)
    assert int(cell.get("n_exc_steps")) == 1
    assert float(cell.get("q")) == pytest.approx(THETA / 4)
    assert int(cell.get("n_spikes")) == 0


def test_below_at_and_above_threshold(fresh_kernel):
    """At-threshold must fire: the reference condition is `V >= theta`, not `>`."""
    nest = fresh_kernel()
    below = _accumulator(nest)
    at = _accumulator(nest)
    above = _accumulator(nest)
    _drive(nest, below, [1.0], THETA - 1.0)
    _drive(nest, at, [1.0], THETA)
    _drive(nest, above, [1.0], THETA + 1.0)
    nest.Simulate(5.0)
    assert int(below.get("n_spikes")) == 0
    assert int(at.get("n_spikes")) == 1
    assert int(above.get("n_spikes")) == 1


def test_accumulation_is_a_pure_sum_across_events(fresh_kernel):
    """No leak: sub-threshold charge persists indefinitely between arrivals."""
    nest = fresh_kernel()
    cell = _accumulator(nest)
    _drive(nest, cell, [1.0, 50.0, 100.0], THETA / 4)
    nest.Simulate(150.0)
    assert int(cell.get("n_spikes")) == 0
    assert float(cell.get("q")) == pytest.approx(0.75 * THETA)


def test_threshold_reset_and_pre_reset_snapshot(fresh_kernel):
    """`q_pre` is the unclamped I_accq the dual FE/FES rule reads."""
    nest = fresh_kernel()
    cell = _accumulator(nest)
    _drive(nest, cell, [1.0], 1.5 * THETA)
    nest.Simulate(5.0)
    assert int(cell.get("n_spikes")) == 1
    assert float(cell.get("q")) == pytest.approx(0.0)
    assert float(cell.get("q_pre")) == pytest.approx(1.5 * THETA), (
        "overshoot must be retained, never clamped to theta"
    )


def test_same_time_multiplicity_is_aggregated_and_stable(fresh_kernel):
    """Simultaneous arrivals on one port are delivered as ONE summed value.

    This is a native NEST semantic, not a choice this prototype makes: it is measured and
    recorded rather than worked around. The consequence is that N simultaneous events are
    one handler invocation carrying N-fold charge, so multiplicity is recovered from
    `q_pre / w` rather than from a handler-invocation count.
    """
    nest = fresh_kernel()
    cell = _accumulator(nest)
    for _ in range(4):
        _drive(nest, cell, [1.0], THETA / 8)
    nest.Simulate(5.0)
    assert int(cell.get("n_exc_steps")) == 1, "four simultaneous events = one invocation"
    assert float(cell.get("q")) == pytest.approx(THETA / 2), "all four charges delivered"


def test_refractory_blocks_firing_not_accumulation(fresh_kernel):
    """The declared policy, matching `gather_exc` + `can_fire` in the reference engine."""
    nest = fresh_kernel()
    cell = _accumulator(nest, t_ref=10.0)
    _drive(nest, cell, [1.0, 2.0, 3.0], THETA)
    nest.Simulate(20.0)
    assert int(cell.get("n_spikes")) == 1
    assert int(cell.get("n_blocked_by_refr")) == 2
    assert float(cell.get("q")) == pytest.approx(2 * THETA), (
        "charge arriving during refractory still accumulates"
    )


def test_refractory_expiry_allows_the_next_spike(fresh_kernel):
    nest = fresh_kernel()
    cell = _accumulator(nest, t_ref=5.0)
    _drive(nest, cell, [1.0, 20.0], THETA)
    nest.Simulate(30.0)
    assert int(cell.get("n_spikes")) == 2


def test_reset_clears_charge(fresh_kernel):
    nest = fresh_kernel()
    cell = _accumulator(nest)
    _drive(nest, cell, [1.0], THETA / 2)
    _drive(nest, cell, [5.0], 1.0, port="reset")
    nest.Simulate(10.0)
    assert float(cell.get("q")) == pytest.approx(0.0)
    assert int(cell.get("n_resets")) == 1


def test_reset_cannot_erase_an_already_recorded_spike(fresh_kernel):
    """NEST cannot un-send a spike and this model does not try (prompt section 6.4)."""
    nest = fresh_kernel()
    cell = _accumulator(nest)
    recorder = nest.Create("spike_recorder")
    nest.Connect(cell, recorder)
    _drive(nest, cell, [1.0], THETA)
    _drive(nest, cell, [2.0], 1.0, port="reset")
    nest.Simulate(10.0)
    assert len(recorder.get("events")["times"]) == 1
    assert int(cell.get("n_spikes")) == 1, "the spike count survives the reset"
    assert float(cell.get("q")) == pytest.approx(0.0)


# ----------------------------------------------------------------------- relay


def test_relay_propagates_with_a_nest_valid_delay(fresh_kernel):
    """Eor contract: ONE accepted upstream event produces ONE output spike."""
    nest = fresh_kernel()
    relay = nest.Create("event_relay", 1, {"theta": THETA, "t_lockout": 0.0})
    ports = nest.GetDefaults("event_relay")["receptor_types"]
    recorder = nest.Create("spike_recorder")
    nest.Connect(relay, recorder)
    gen = nest.Create("spike_generator", 1, {"spike_times": [1.0]})
    nest.Connect(gen, relay, syn_spec={"weight": THETA, "delay": H,
                                       "receptor_type": ports["EXC"]})
    nest.Simulate(5.0)
    times = list(recorder.get("events")["times"])
    assert len(times) == 1
    assert times[0] == pytest.approx(1.0 + H), "emission at the arrival timestamp"


def test_relay_lockout_emits_at_most_one_volley_per_window(fresh_kernel):
    """The `I` contract: a second same-window input creates no burst and no second reset."""
    nest = fresh_kernel()
    relay = nest.Create("event_relay", 1,
                        {"theta": THETA / 3, "t_lockout": 20.0})
    ports = nest.GetDefaults("event_relay")["receptor_types"]
    gen = nest.Create("spike_generator", 1, {"spike_times": [1.0, 2.0, 3.0]})
    nest.Connect(gen, relay, syn_spec={"weight": THETA, "delay": H,
                                       "receptor_type": ports["EXC"]})
    nest.Simulate(15.0)
    assert int(relay.get("n_spikes")) == 1
    assert int(relay.get("n_suppressed")) == 2


# ----------------------------------------------------------------- coincidence


def _coincidence(nest, **params):
    base = {"theta": THETA, "t_ref": 0.0,
            "tau_basal": 10.0, "tau_apical": 10.0, "tau_deposit_lock": 10.0}
    base.update(params)
    cell = nest.Create("event_coincidence", 1, base)
    receptors = nest.GetDefaults("event_coincidence")["receptor_types"]
    return cell, receptors


def _send(nest, cell, receptors, port, times, weight=THETA):
    gen = nest.Create("spike_generator", 1, {"spike_times": list(times)})
    nest.Connect(gen, cell, syn_spec={
        "weight": weight, "delay": H, "receptor_type": receptors[port.upper()]})
    return gen


def test_basal_only_is_silent_and_deposits_nothing(fresh_kernel):
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest)
    _send(nest, cell, receptors, "basal", [1.0, 5.0, 9.0])
    nest.Simulate(20.0)
    assert int(cell.get("n_deposits")) == 0
    assert int(cell.get("n_spikes")) == 0
    assert float(cell.get("q")) == pytest.approx(0.0), "basal alone never depolarizes"


def test_apical_only_is_silent_and_deposits_nothing(fresh_kernel):
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest)
    _send(nest, cell, receptors, "apical", [1.0, 5.0, 9.0])
    nest.Simulate(20.0)
    assert int(cell.get("n_deposits")) == 0
    assert int(cell.get("n_spikes")) == 0
    assert float(cell.get("q")) == pytest.approx(0.0)


def test_basal_then_apical_within_ttl_deposits_exactly_once(fresh_kernel):
    """THE CARRY PATH. This is the ordering that was dead in the reference engine.

    A basal event arrives first and must remain eligible until the apical arrives; the
    deposit is then attributed to `deposit_basal_first`, so a dead carry is visible in the
    recording rather than having to be inferred from a silent cell.
    """
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_basal=10.0)
    _send(nest, cell, receptors, "basal", [1.0], weight=THETA)
    _send(nest, cell, receptors, "apical", [5.0])
    nest.Simulate(20.0)
    assert int(cell.get("n_deposits")) == 1
    assert int(cell.get("n_deposit_basal_first")) == 1
    assert int(cell.get("n_deposit_apical_first")) == 0
    assert int(cell.get("n_spikes")) == 1, "w_basal = theta is the one-shot condition"


def test_apical_then_basal_within_ttl_deposits_exactly_once(fresh_kernel):
    """The mirror ordering. Must deposit once and must NOT double-deposit."""
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_apical=10.0)
    _send(nest, cell, receptors, "apical", [1.0])
    _send(nest, cell, receptors, "basal", [5.0], weight=THETA)
    nest.Simulate(20.0)
    assert int(cell.get("n_deposits")) == 1
    assert int(cell.get("n_deposit_apical_first")) == 1
    assert int(cell.get("n_deposit_basal_first")) == 0
    assert int(cell.get("n_spikes")) == 1


def test_same_timestep_basal_and_apical_deposits_once(fresh_kernel):
    """Declared handler priority, not chance, resolves the simultaneous case."""
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest)
    _send(nest, cell, receptors, "basal", [1.0], weight=THETA)
    _send(nest, cell, receptors, "apical", [1.0])
    nest.Simulate(20.0)
    assert int(cell.get("n_deposits")) == 1
    assert int(cell.get("n_spikes")) == 1


def test_expired_basal_eligibility_does_not_deposit(fresh_kernel):
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_basal=2.0)
    _send(nest, cell, receptors, "basal", [1.0], weight=THETA)
    _send(nest, cell, receptors, "apical", [15.0])
    nest.Simulate(30.0)
    assert int(cell.get("n_deposits")) == 0
    assert int(cell.get("n_spikes")) == 0
    assert int(cell.get("n_basal_expired")) == 1, "expiry is counted, not silently dropped"


def test_expired_apical_permission_does_not_deposit(fresh_kernel):
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_apical=2.0)
    _send(nest, cell, receptors, "apical", [1.0])
    _send(nest, cell, receptors, "basal", [15.0], weight=THETA)
    nest.Simulate(30.0)
    assert int(cell.get("n_deposits")) == 0
    assert int(cell.get("n_spikes")) == 0


def test_deposit_is_idempotent_within_one_window(fresh_kernel):
    """"Deposits once per boundary": a second apical in the window adds no charge."""
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_apical=20.0, tau_deposit_lock=20.0)
    _send(nest, cell, receptors, "basal", [1.0], weight=THETA / 2)
    _send(nest, cell, receptors, "apical", [2.0, 4.0, 6.0])
    nest.Simulate(30.0)
    assert int(cell.get("n_deposits")) == 1
    assert float(cell.get("q")) == pytest.approx(THETA / 2), "exactly one deposit of w*s"


def test_sub_threshold_deposits_accumulate_but_cannot_fire_without_a_gate(fresh_kernel):
    """C fires ONLY at a valid coincidence, never from retained charge.

    This is the structural form of the reference rule `crossing_time -> infinity while the
    gate is closed`.
    """
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_basal=5.0, tau_apical=5.0,
                                   tau_deposit_lock=5.0)
    _send(nest, cell, receptors, "basal", [1.0, 21.0], weight=THETA * 0.6)
    _send(nest, cell, receptors, "apical", [2.0, 22.0])
    nest.Simulate(40.0)
    assert int(cell.get("n_deposits")) == 2
    assert int(cell.get("n_spikes")) == 1, "fires on the deposit that crosses, not before"


def test_earliest_pending_basal_stays_causal(fresh_kernel):
    """Multi-basal causality: the first pending event owns the deposit."""
    nest = fresh_kernel()
    cell, receptors = _coincidence(nest, tau_basal=20.0, tau_apical=20.0)
    _send(nest, cell, receptors, "basal", [1.0], weight=THETA / 4)
    _send(nest, cell, receptors, "basal", [3.0], weight=THETA)
    _send(nest, cell, receptors, "apical", [6.0])
    nest.Simulate(30.0)
    assert int(cell.get("n_deposits")) == 1
    assert int(cell.get("n_basal_superseded")) == 1
    assert float(cell.get("q")) == pytest.approx(THETA / 4), (
        "the EARLIEST basal is causal; the later one is observed but does not displace it"
    )


# ----------------------------------------------- two-competitor WTA microcontract


def _wta_pair(nest, inter_arrival_ms, *, n_afferent_events=4, lockout=50.0):
    """Two competitors sharing ONE volley, plus the `E -> I -> E` loop.

    This mirrors how the real column works and is why the arrangement matters. All
    competitors in a column see the SAME afferent spikes at the same times; they differ
    only in their weights. So the race is not "whose evidence arrives first" but "who
    reaches theta on an EARLIER arrival of the shared volley".

    `inter_arrival_ms` is the gap between successive arrivals within the volley -- i.e.
    `S / n_afferent_events`, the quantity a latency race can actually resolve. The
    stronger competitor `a` crosses on the 3rd arrival, the weaker `b` would cross on the
    4th; whether `b` is stopped depends entirely on whether the loop closes in between.

    The volley is finite (`n_afferent_events`), matching the reference engine's
    discarded drive packet: once reset, the loser's remaining evidence in THIS volley is
    sub-threshold, so it cannot recover within the presentation.
    """
    a = _accumulator(nest)
    b = _accumulator(nest)
    relay = nest.Create("event_relay", 1, {"theta": THETA / 3, "t_lockout": lockout})
    receptors = nest.GetDefaults("event_accumulator")["receptor_types"]
    relay_ports = nest.GetDefaults("event_relay")["receptor_types"]
    for cell in (a, b):
        nest.Connect(cell, relay, syn_spec={"weight": THETA, "delay": H,
                                            "receptor_type": relay_ports["EXC"]})
        nest.Connect(relay, cell, syn_spec={
            "weight": 1.0, "delay": H, "receptor_type": receptors["RESET"]})

    times = [round(1.0 + k * inter_arrival_ms, 6) for k in range(n_afferent_events)]
    gen = nest.Create("spike_generator", 1, {"spike_times": times}) if inter_arrival_ms \
        else nest.Create("spike_generator", 1, {"spike_times": [1.0]})
    # With a zero gap NEST would collapse repeated identical times, so the simultaneous
    # case is expressed as one arrival carrying the whole volley's charge instead.
    scale = 1.0 if inter_arrival_ms else float(n_afferent_events)
    nest.Connect(gen, a, syn_spec={"weight": 0.35 * THETA * scale, "delay": H,
                                   "receptor_type": receptors["EXC"]})
    nest.Connect(gen, b, syn_spec={"weight": 0.26 * THETA * scale, "delay": H,
                                   "receptor_type": receptors["EXC"]})
    return a, b, relay


def test_wta_microcontract_separated_arrivals_yield_one_winner(fresh_kernel):
    """Inter-arrival gap 2.0 ms = 10 x L_wta: the loop closes between the two crossings.

    `a` reaches theta on the 3rd arrival (3 x 0.35 = 1.05 theta). The loop resets the bank
    0.2 ms later, erasing `b`'s accumulated 0.78 theta. `b`'s only remaining evidence in
    this volley is the 4th arrival (0.26 theta), so it cannot cross.
    """
    nest = fresh_kernel()
    a, b, _relay = _wta_pair(nest, inter_arrival_ms=2.0)
    nest.Simulate(30.0)
    assert int(a.get("n_spikes")) == 1
    assert int(b.get("n_spikes")) == 0, "the loser was reset before it could cross"
    assert int(b.get("n_resets")) >= 1


def test_wta_microcontract_simultaneous_arrivals_yield_two_winners(fresh_kernel):
    """Gap 0: with no arrival ordering there is nothing to arbitrate.

    This is the mechanical proof of prompt section 5.2 -- fast inhibition alone does NOT
    recover single-winner WTA. `L_wta` cannot be smaller than one resolution step, while a
    simultaneous volley separates the crossings by exactly zero, so both competitors have
    already fired before any reset can be delivered.
    """
    nest = fresh_kernel()
    a, b, _relay = _wta_pair(nest, inter_arrival_ms=0.0)
    nest.Simulate(30.0)
    assert int(a.get("n_spikes")) == 1
    assert int(b.get("n_spikes")) == 1, "both cross before any reset can arrive"


def test_wta_microcontract_gap_below_loop_latency_still_yields_two_winners(fresh_kernel):
    """Inter-arrival gap = h = 0.5 x L_wta: too short for the loop to intervene.

    `a` crosses on the 3rd arrival and `b` on the 4th, but those are only `h` apart while
    the loop needs `2h`, so `b` fires before the reset lands. Single-winner WTA requires
    the gap to exceed `L_wta`, not merely to be non-zero.
    """
    nest = fresh_kernel()
    a, b, _relay = _wta_pair(nest, inter_arrival_ms=H)
    nest.Simulate(30.0)
    assert int(a.get("n_spikes")) == 1
    assert int(b.get("n_spikes")) == 1
