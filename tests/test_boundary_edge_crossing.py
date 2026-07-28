"""Sub-boundary event loop: crossings that only become available at exactly ``tau = 1.0``.

The loop drains events rather than stopping at the boundary edge. Without that, a
DEPENDENT crossing -- one enabled by another cell's spike at exactly tau = 1.0 -- was
unschedulable: a coincidence cell whose apical permission came from a parent firing at 1.0
committed its deposit, sat above threshold with an open gate, and then had the gate cleared
by the next boundary. Since a closed gate correctly forbids firing, the cell retained charge
forever, never fired, and therefore never learned (C learning is gated on its own spike).

This is reachable whenever drive lands exactly on theta: with the theta/2 detector ceiling
and zero leak, two coordinated identity events deliver exactly theta, so dtau = 1.0 exactly.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, CoincidencePyramidalNeuron
from experiments.direct_identity_experiment import RATES

THETA = 1000.0
WARM = 2500
MEASURE = 400


@pytest.fixture(scope='module')
def two_patch():
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity', **RATES)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'col 1')
    for _ in range(WARM):
        e.step()
    return e


def test_the_l2_detector_really_does_cross_at_the_boundary_edge(two_patch):
    """Guards the premise: if this stops being exactly 1.0 the rest of the file is vacuous."""
    e = two_patch
    seen = set()
    for _ in range(50):
        e.step()
        for nid in e.exc:
            if (e.spiked.get(nid) and e._column_of.get(nid) == 'L2c00'
                    and e._role_of.get(nid) == 'E'):
                seen.add(round(float(e.exc[nid].spike_tau), 6))
    assert seen == {1.0}, seen


def test_coincidence_cells_fire_despite_edge_timed_permission(two_patch):
    e = two_patch
    spikes = 0
    for _ in range(MEASURE):
        e.step()
        spikes += sum(1 for c in e.coincidence if e.spiked.get(c.id))
    assert spikes > 0, 'a C whose gate opens at tau=1.0 must still be schedulable'


def test_no_coincidence_cell_retains_runaway_charge(two_patch):
    e = two_patch
    for _ in range(MEASURE):
        e.step()
    for c in e.coincidence:
        assert c.V < 10.0 * c.threshold, (
            f'{c.id} retained {c.V:.1f} against theta {c.threshold}: its gate opened but it '
            f'was never scheduled, so it accumulated without firing or learning')


def test_edge_timed_c_cells_actually_learn(two_patch):
    """The consequence that matters: C learning is gated on firing, so a C that cannot be
    scheduled cannot learn no matter how much input it receives."""
    e = two_patch
    init = 0.25 * THETA                       # dual-rule init (FES middle theta/4)
    matured = [c for c in e.coincidence if float(c.basal.weights.max()) > init + 1.0]
    assert matured, 'no C basal weight moved off its initialization'


def test_a_cell_below_threshold_at_the_edge_still_does_not_fire():
    """The drain must admit ONLY cells already at or over threshold at tau = 1.0 -- a
    sub-threshold cell has no interval left to cross in and must report inf."""
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity')
    cell = e.exc['L1c00E0']
    cell.V = 0.5 * THETA                       # half way, with drive but no room left
    cell.remaining_excitation = 100.0
    assert not np.isfinite(cell.crossing_time(0.0))
    cell.V = THETA
    assert cell.crossing_time(0.0) == 0.0      # already there -> fires at the edge


def test_the_loop_terminates_even_when_events_keep_arriving_at_the_edge():
    """Termination does not depend on tau advancing: a fired cell reports inf for the rest
    of the boundary, so the drain runs at most once per membrane."""
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity', **RATES)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'col 1')
    for _ in range(200):
        st = e.step()                          # would hang, not fail, if unbounded
        assert st['timestep'] == e.timestep
    for c in e.exc.values():
        assert c.crossing_time(1.0) == float('inf') or not c.fired_this_boundary


# ------------------------------------------- input pacing locked to the loop latency
@pytest.mark.parametrize('topo,latency', [('tiled_cc_direct_identity', 2),
                                          ('tiled_cc', 3),
                                          ('tiled_cc_double_eor', 4)])
def test_input_period_equal_to_loop_latency_gives_strict_alternation(topo, latency):
    """With a volley every boundary the confirmed column runs L-on/L-off (period 2L), so
    the cadence depends on wiring depth. Pacing the input at exactly the loop latency makes
    each volley's confirmation land on its SUCCESSOR's drive packet, cancelling it -- so
    presentations alternate exactly, independent of L."""
    from backend.dashboard_config import DASHBOARD_OVERRIDES
    p = dict(DASHBOARD_OVERRIDES)
    p['topology'] = topo
    p['input_period'] = latency
    e = SimulationEngine(seed=1, **p)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'col 1')
    for _ in range(12000):
        e.step()
    while e.timestep % latency != 0:
        e.step()
    seq = ''
    for _ in range(20):
        fired = False
        for _ in range(latency):
            if e.step()['column_winners'].get('L1c00'):
                fired = True
        seq += '1' if fired else '0'
    assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1)), seq


def test_input_period_applies_without_wiping_learned_state():
    """Pacing is runtime-only: re-pacing a trained column must not rebuild it."""
    e = SimulationEngine(seed=1, topology='tiled_cc', dual_fe_fes=True, eta=4.0,
                         c_eta=16.0, dual_fe_B=5.0, leak_rate=0.0, e_weight_cap_frac=0.5)
    e.set_patch_pattern(0, 0, 'row 1')
    for _ in range(600):
        e.step()
    before = {c.id: c.acc_weights.copy() for c in e.latency_competitors}
    t_before = e.timestep
    assert e.apply_config({'input_period': 3}) == ['input_period']
    assert e.params['input_period'] == 3
    assert e.timestep == t_before                       # not reset
    for c in e.latency_competitors:
        assert np.array_equal(c.acc_weights, before[c.id])   # learned state preserved


# --------------------------------------- auto pacing: derived from the graph, not the name
@pytest.mark.parametrize('topo,expected', [('tiled_cc_direct_identity', 2),
                                           ('tiled_cc', 3),
                                           ('tiled_cc_l1_4', 3),
                                           ('tiled_cc_double_eor', 4),
                                           ('rg_coincidence', None),
                                           ('rg_direct_cc4', None)])
def test_feedback_loop_latency_is_derived_from_the_graph(topo, expected):
    """Matches the impulse-measured latency, and is None where there is no top-down loop."""
    assert SimulationEngine(seed=1, topology=topo).feedback_loop_latency == expected


def test_auto_input_period_retracks_on_topology_change():
    """A hand-set period silently desyncs when the graph changes; auto must not."""
    e = SimulationEngine(seed=1, topology='tiled_cc', input_period=0)
    assert e.resolved_input_period() == 3
    e.apply_config({'topology': 'tiled_cc_direct_identity'})
    assert e.resolved_input_period() == 2
    e.apply_config({'topology': 'tiled_cc_double_eor'})
    assert e.resolved_input_period() == 4
    # an explicit value is honoured verbatim and does NOT re-track
    m = SimulationEngine(seed=1, topology='tiled_cc', input_period=3)
    m.apply_config({'topology': 'tiled_cc_direct_identity'})
    assert m.resolved_input_period() == 3


def test_auto_pacing_falls_back_to_one_without_a_feedback_loop():
    e = SimulationEngine(seed=1, topology='rg_coincidence', input_period=0)
    assert e.feedback_loop_latency is None and e.resolved_input_period() == 1
