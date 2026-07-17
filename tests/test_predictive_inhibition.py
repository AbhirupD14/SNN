"""Engine-level predictive-inhibition behaviour: local PI learning onto active
pixels, delayed and persistent conductance expression, predicted-vs-unpredicted
suppression, deterministic order-independence, and the causal control that
symmetry breaking requires predictive conductance AND plasticity.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, PATTERNS, N_PIX
from experiments.predictive_inhibition_overlap import run_schedule, active_pixels


def direct(seed=1, **cfg):
    return SimulationEngine(seed=seed, topology='pi', **cfg)


def test_pi_learns_active_pixels_only():
    e = direct()
    e.set_pattern('row 1')                              # active {3,4,5}
    for _ in range(2500):
        e.step()
    j = int(e.winner[3:])
    w = e.pi[j].w
    active = sorted(active_pixels('row 1'))
    inactive = sorted(set(range(N_PIX)) - set(active))
    assert w[active].sum() > 0.5                        # learned onto the row features
    assert np.all(w[inactive] < 1e-6)                  # nothing onto inactive pixels (local)


def test_pi_conductance_pulse_is_delayed_then_persists():
    # A PI event schedules its L1E_s conductance for the NEXT boundary, and (at the
    # slow L1 retention) that conductance persists across several later boundaries.
    e = direct()
    e.set_pattern('row 1')
    for _ in range(2500):
        e.step()                                        # mature the incumbent PI
    # Find a boundary where the incumbent PI fires.
    j = int(e.winner[3:])
    fired = None
    for _ in range(200):
        d = e.step()
        if any(n['id'] == f'PI{j}' and n['spiked'] for n in d['neurons']):
            fired = d
            break
    assert fired is not None
    # No same-boundary conductance from this PI event (delay 1): its pulses are queued.
    assert not any(p['source'] == f'PI{j}' for p in fired['inhibitory_pulses'])
    # Next boundary the predictive pulse lands; then it persists (decays slowly).
    d1 = e.step()
    preds = [p for p in d1['inhibitory_pulses'] if p['kind'] == 'predictive']
    assert preds and any(p['conductance_increment'] > 0 for p in preds)
    target = max(preds, key=lambda p: p['conductance_increment'])['target']
    g0 = next(n['g_inh'] for n in d1['neurons'] if n['id'] == target)
    persisted = []
    for _ in range(4):
        d = e.step()
        persisted.append(next(n['g_inh'] for n in d['neurons'] if n['id'] == target))
    assert g0 > 0 and persisted[0] > 0                 # conductance outlives the single event
    assert persisted[-1] < g0                          # and decays over time


def test_predicted_features_suppressed_more_than_unpredicted():
    # Train the row, switch to the column: the incumbent predicts the shared pixel 4,
    # so its conductance suppresses pixel 4 more than the novel pixels {1,7}.
    r = run_schedule(1, steps=(2500, 2500, 200))
    assert r['g_shared_mean'] > r['g_novel_mean']
    assert r['shared_over_novel_g'] > 1.0


def test_step_is_deterministic_and_order_independent():
    # The synchronous gather-then-integrate design makes a step independent of neuron
    # iteration order; two identical seeded runs must produce identical trajectories.
    def run():
        e = direct(seed=3)
        e.set_pattern('row 1')
        hist = []
        for _ in range(400):
            d = e.step()
            hist.append(d['winner'])
        return hist, np.concatenate([pi.w for pi in e.pi])
    h1, w1 = run()
    h2, w2 = run()
    assert h1 == h2
    assert np.array_equal(w1, w2)


def test_symmetry_break_requires_conductance_and_plasticity():
    # The causal control, as a regression: predictive inhibition drives turnover.
    steps = (2000, 2500, 300)
    full = run_schedule(1, steps=steps)
    no_g = run_schedule(1, steps=steps, pi_conductance_enabled=False)
    no_p = run_schedule(1, steps=steps, pi_plasticity_enabled=False)
    assert full['symmetry_break'] is True              # PI on -> winner changes
    assert no_g['symmetry_break'] is False             # no conductance -> no change
    assert no_p['symmetry_break'] is False             # no learning -> no change
    assert full['incumbent_recovers'] is True          # not catastrophic forgetting


def test_l2i_wta_conductance_does_not_prevent_the_winner():
    # The winner spikes on the boundary it wins; the WTA conductance it triggers lands
    # only on the next boundary, so it cannot cancel its own cause.
    e = direct(seed=1, leak_rate=0.0)
    e.clear_input()
    e.l2e[2].acc_weights[:] = 0.0
    e.l2e[2].acc_weights[4] = 1.3 * e.params['e_threshold']
    e.input_vec[4] = 1.0
    won = False
    for _ in range(10):
        d = e.step()
        if any(n['id'] == 'L2E2' and n['spiked'] for n in d['neurons']):
            won = True
            assert d['winner'] == 'L2E2'
            # no WTA conductance delivered on this same boundary
            assert not any(p['kind'] == 'wta' for p in d['inhibitory_pulses'])
            break
    assert won
