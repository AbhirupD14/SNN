"""Phase 4 — analytic sub-boundary event scheduler and emergent latency WTA.

Two layers: (1) BoundaryEventScheduler unit tests over bare membranes with exactly
controlled crossing times (ordering, ties, tolerance, invalidation); (2) engine
integration on a minimal synthetic C + latency-WTA graph, verifying WTA emerges from
first-spike latency + the L2I reset loop with no forced-winner policy, and that
ordinary excitation keeps its one-boundary delay.
"""

import math

import pytest

from snn.neurons import ConductanceLIFNeuron, CoincidencePyramidalNeuron
from backend.simulation import SimulationEngine, BoundaryEventScheduler
from tests.test_coincidence_spec import _synth_spec


# =============================================================== scheduler unit
def _membrane(V0, drive, thr=1000.0, leak=0.0):
    # leak=0 -> pure integrator -> crossing tau = (thr - V0) / drive (exactly).
    n = ConductanceLIFNeuron('m', 'test', threshold=thr, leak_rate=leak)
    n.V = V0
    n.gather_exc(drive)
    n.freeze_drive()
    return n


def test_scheduler_picks_earliest_finite_crossing():
    a = _membrane(0.0, 2000.0)      # tau = 0.5
    b = _membrane(0.0, 1000.0)      # tau = 1.0
    sched = BoundaryEventScheduler([a, b], 1e-12)
    cell, tau = sched.next_event()
    assert cell is a and tau == pytest.approx(0.5)
    assert sched.ties == []


def test_scheduler_earliest_wins_regardless_of_node_order():
    # Index 0 crosses LATER; the earlier (index 1) still wins, no node-order tie-break.
    a = _membrane(0.0, 1000.0)      # tau = 1.0  (index 0)
    b = _membrane(0.0, 2000.0)      # tau = 0.5  (index 1)
    sched = BoundaryEventScheduler([a, b], 1e-12)
    cell, tau = sched.next_event()
    assert cell is b and tau == pytest.approx(0.5)
    assert sched.ties == []          # distinguishable times are not a tie


def test_scheduler_exact_tie_uses_stable_node_order():
    a = _membrane(0.0, 2000.0)      # tau = 0.5
    b = _membrane(0.0, 2000.0)      # tau = 0.5 (exactly)
    sched = BoundaryEventScheduler([a, b], 1e-12)
    cell, tau = sched.next_event()
    assert cell is a               # lowest node order wins the tie
    assert len(sched.ties) == 1
    assert sched.ties[0]['ids'] == ['m', 'm'] and sched.ties[0]['chosen'] == 'm'


def test_scheduler_within_tolerance_is_a_tie_outside_is_not():
    a = _membrane(0.0, 2000.0)                 # tau = 0.5000
    b = _membrane(0.0, 2000.0 / 1.004)         # tau ~= 0.502 (diff ~0.002)
    within = BoundaryEventScheduler([a, b], 1e-2)
    cell, _ = within.next_event()
    assert cell is a and len(within.ties) == 1           # 0.002 < 1e-2 -> tie
    a2 = _membrane(0.0, 2000.0)
    b2 = _membrane(0.0, 2000.0 / 1.004)
    outside = BoundaryEventScheduler([a2, b2], 1e-6)
    cell2, _ = outside.next_event()
    assert cell2 is a2 and outside.ties == []            # 0.002 > 1e-6 -> not a tie


def test_scheduler_advance_all_moves_every_membrane():
    a = _membrane(0.0, 500.0)
    b = _membrane(100.0, 500.0)
    sched = BoundaryEventScheduler([a, b], 1e-12)
    sched.advance_all(0.4)
    assert sched.current_tau == pytest.approx(0.4)
    assert a.V == pytest.approx(200.0)      # 0 + 500*0.4
    assert b.V == pytest.approx(300.0)      # 100 + 500*0.4


def test_scheduler_hard_reset_invalidates_stale_candidate():
    a = _membrane(0.0, 3000.0)      # would cross at tau ~= 0.333
    sched = BoundaryEventScheduler([a], 1e-12)
    cell, tau = sched.next_event()
    assert cell is a
    a.hard_reset(tau)               # an earlier event wiped its state + drive
    assert sched.next_event() == (None, None)   # its former crossing is invalid


# =============================================================== engine fixture
def _wta_engine(w0=1500.0, w1=1200.0, learn=False, **overrides):
    e = SimulationEngine(seed=1, leak_rate=0.03, e_weight_cap=2000.0, **overrides)
    e.apply_topology(_synth_spec())
    e.set_input([1, 0, 0, 0, 0, 0, 0, 0, 0])       # hold pixel 0 -> RG0 fires each boundary
    for c in e.latency_competitors:
        c.learn = learn
        c.acc_weights[:] = w0 if c.id == 'L2E0' else w1
    return e


def _run_until_l2_spike(e, limit=12):
    for _ in range(limit):
        e.step()
        winners = [c.id for c in e.latency_competitors if c.spiked]
        if winners:
            return winners
    return []


# ------------------------------------------------- emergent latency WTA + reset
def test_higher_drive_competitor_wins_and_resets_the_other():
    e = _wta_engine(w0=1500.0, w1=1200.0)
    winners = _run_until_l2_spike(e)
    assert winners == ['L2E0']                     # earlier latency wins
    l2e1 = next(c for c in e.latency_competitors if c.id == 'L2E1')
    assert l2e1.spiked is False                    # reset before it could fire
    reset_targets = {(h['source'], h['target']) for h in e.hard_reset_events}
    assert ('L2I', 'L2E1') in reset_targets        # L2I hard-reset the loser


def test_reversed_drive_reverses_winner_without_node_reorder():
    e = _wta_engine(w0=1200.0, w1=1500.0)          # L2E1 now the stronger
    winners = _run_until_l2_spike(e)
    assert winners == ['L2E1']
    assert e.order == _wta_engine().order          # node order is identical


def test_winner_spike_and_learning_survive_its_own_reset():
    e = _wta_engine(w0=1500.0, w1=1200.0, learn=True)
    _run_until_l2_spike(e)
    win = next(c for c in e.latency_competitors if c.id == 'L2E0')
    assert win.spiked is True                      # its emitted spike is NOT erased
    assert win.spike_tau is not None
    assert win.V == 0.0                            # membrane cleared by its own L2I reset
    # its ff learning ran this boundary (weight change emitted).
    assert any(cs['id'] in win.ff_edge_ids for cs in e.changed_synapses)


def test_all_to_all_reset_permits_at_most_one_l2_spike_per_boundary():
    e = _wta_engine(w0=1500.0, w1=1490.0)          # near, but resolvable
    for _ in range(12):
        e.step()
        assert sum(1 for c in e.latency_competitors if c.spiked) <= 1


def test_multiple_same_time_resets_recorded_individually():
    e = _wta_engine(w0=1500.0, w1=1200.0)
    _run_until_l2_spike(e)
    l2_resets = [h for h in e.hard_reset_events if h['source'] == 'L2I']
    assert len(l2_resets) == 2                      # both L2E reset...
    assert l2_resets[0]['tau'] == pytest.approx(l2_resets[1]['tau'])   # ...at one tau


def test_reset_discards_loser_drive():
    e = _wta_engine(w0=1500.0, w1=1200.0)
    _run_until_l2_spike(e)
    l2e1 = next(c for c in e.latency_competitors if c.id == 'L2E1')
    assert l2e1.remaining_excitation == 0.0        # loser's frozen packet discarded
    assert l2e1.V == 0.0


def test_no_legacy_wta_path_used():
    e = _wta_engine()
    _run_until_l2_spike(e)
    assert e.competitors == []                      # no deterministic-WTA participants
    assert e.winner in ('L2E0', 'L2E1')             # report-only, set from a latency spike


# ------------------------------------------------- one-boundary delay preserved
def test_ordinary_excitation_keeps_one_boundary_delay():
    e = _wta_engine()
    e.step()                                        # t=1: RG0 fires (exogenous)
    assert e.spiked['RG0'] is True
    assert e.spiked['L1E0'] is False                # L1E cannot fire the same boundary
    e.step()                                        # t=2: pretrained L1E fires now
    assert e.spiked['L1E0'] is True
    assert e.spiked['L2E0'] is False                # cannot cross a 2nd edge same boundary


def test_winner_apical_output_opens_mature_c_at_same_tau():
    e = _wta_engine(w0=1500.0, w1=1200.0)
    c = e.exc['L1C0']
    c.basal.weights[0] = c.w_max
    _run_until_l2_spike(e)
    winner = e.exc['L2E0']
    assert winner.spiked and c.spiked
    assert c.coincidence_deposit_count == 1
    assert c.coincidence_deposit_tau == pytest.approx(winner.spike_tau)
    assert c.spike_tau == pytest.approx(winner.spike_tau)
    assert c.apical_delivery_count == 1
    assert c.apical_duplicate_count == 0
    assert not hasattr(e, '_apical_next')


# ------------------------------------------------------ legacy path untouched
def test_legacy_graph_uses_synchronous_step():
    e = SimulationEngine(seed=1, topology='pi')
    assert e.event_resolved is False
    e.set_pattern('row 1')
    dyn = e.step()
    assert dyn['timestep'] == 1                      # legacy step still works
