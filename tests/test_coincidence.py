"""L1E_new coincidence comparison branch (enew_enabled=True) under the conductance
engine: nine-afferent construction, delayed paired local-sensory delivery, an
emergent coincidence advantage from leaky integration, and pixel-selective L1I
inhibitory *conductance* (no longer a hard wipe) that stays local, not global.

The one-step feedforward/feedback delay means a detector's paired sensory afferent
(from L1E_s at t-1) and its winner feedback (from the L2 winner at t-1) both arrive
at boundary t, so coincidence is still tested by joint leaky integration.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT
from snn.neurons import ExcitatoryNeuron, E_THRESHOLD


def fresh(leak=0.03, seed=1):
    e = SimulationEngine(seed=seed, leak_rate=leak)
    e.clear_input()
    return e


def detector(leak, sensory=500.0, winner_w=500.0):
    """A mature two-afferent coincidence detector: afferent 0 = paired sensory,
    afferent 1 = the associated winning L2E feedback."""
    return ExcitatoryNeuron('probe', 'supervisor',
                            acc_weights=np.array([sensory, winner_w]),
                            acc_distance_factor=np.ones(2),
                            threshold=E_THRESHOLD, w_max=E_THRESHOLD / 2.0,
                            leak_rate=leak, learn=False)


def drive_detector(mode, leak, T, steps=400, sensory=500.0, winner_w=500.0):
    """First firing step (or None) of a mature detector fed a train every T steps.
    mode in {'sensory', 'winner', 'coincident'}. Integrates (leaks) every step."""
    n = detector(leak, sensory, winner_w)
    for t in range(1, steps + 1):
        if t % T == 0:
            if mode in ('sensory', 'coincident'):
                n.gather_exc(n.acc_weights[0])
            if mode in ('winner', 'coincident'):
                n.gather_exc(n.acc_weights[1])
        n.integrate()
        if n.can_fire():
            n.fire()
            return t
    return None


# --------------------------------------------------------------- construction
def test_l1e_new_has_nine_afferents_and_subthreshold_init():
    e = SimulationEngine(seed=1)
    for n in e.l1e_new:
        assert n.acc_weights.shape == (1 + N_OUT,)
        assert np.all(n.acc_weights >= 0) and np.all(n.acc_weights <= e.params['e_weight_cap'])
        assert n.acc_weights.sum() < e.params['e_threshold']
    assert e.params['e_weight_cap'] == pytest.approx(500.0)


def test_default_leak_is_nonzero():
    assert SimulationEngine(seed=1).params['leak_rate'] > 0.0


# ------------------------------------------------------------- charge delivery
def test_local_sensory_reaches_only_paired_l1e_new_next_boundary():
    e = fresh()
    e.input_vec[:] = 0.0
    e.input_vec[4] = 1.0
    for n in e.l1e_new:
        n.acc_weights[0] = 400.0
    # Step until L1E_s[4] fires and EMITS its paired local-sensory edge.
    for _ in range(8):
        d = e.step()
        if 'cl4' in d['emitted']:
            break
    assert 'cl4' in d['emitted']
    assert not any(f'cl{i}' in d['emitted'] for i in range(N_PIX) if i != 4)
    # The charge lands on the paired detector on the NEXT boundary (delay 1).
    e.step()
    assert e.l1e_new[4].V > 0.0
    assert all(e.l1e_new[i].V == 0.0 for i in range(N_PIX) if i != 4)


# ---------------------------------------------------------------- coincidence
def test_coincident_volley_depolarizes_more_than_a_lone_branch():
    # A single coincident deposit (500 + 500) produces strictly more depolarization
    # than either lone 500 branch in the same boundary.
    lone = detector(0.2); lone.gather_exc(lone.acc_weights[0]); lone.integrate()
    coin = detector(0.2)
    coin.gather_exc(coin.acc_weights[0]); coin.gather_exc(coin.acc_weights[1]); coin.integrate()
    assert coin.V > lone.V


def test_leaky_integration_gives_emergent_coincidence():
    # At a leak where a repeated lone 500 branch saturates below threshold, only the
    # coincident (500 + 500) train crosses -- an emergent AND from leak + integration.
    leak, T = 0.5, 1
    assert drive_detector('sensory', leak, T) is None
    assert drive_detector('winner', leak, T) is None
    assert drive_detector('coincident', leak, T) is not None


# ----------------------------------------------------------- L1I conductance
def test_l1i_relay_fires_and_conductance_is_delayed_and_paired():
    e = SimulationEngine(seed=1)
    e.set_pattern('row 1')
    # Find a boundary where L1E_new[i] fires and its L1I relay emits.
    hit = None
    for _ in range(200):
        d = e.step()
        for i in (3, 4, 5):
            if any(n['id'] == f'L1I{i}' and n['spiked'] for n in d['neurons']):
                hit = (i, d)
                break
        if hit:
            break
    assert hit is not None
    i, d = hit
    assert f're_l1_{i}' in d['emitted']
    # No same-boundary inhibitory conductance on L1E{i}: the pulse is delayed one step.
    assert not any(p['target'] == f'L1E{i}' for p in d['inhibitory_pulses'])
    # Next boundary it lands as a 'legacy_l1i' conductance pulse on the paired L1E_s.
    d2 = e.step()
    landed = [p for p in d2['inhibitory_pulses']
              if p['kind'] == 'legacy_l1i' and p['target'] == f'L1E{i}']
    assert landed and landed[0]['conductance_increment'] > 0.0


def test_held_pattern_inhibition_is_pixel_selective():
    e = SimulationEngine(seed=1)
    e.set_pattern('row 1')                                  # active pixels 3,4,5
    fired = np.zeros(N_PIX, dtype=int)
    hits = np.zeros(N_PIX, dtype=int)
    for _ in range(1500):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        for i in range(N_PIX):
            if by[f'L1Enew{i}']['spiked']:
                fired[i] += 1
        for p in d['inhibitory_pulses']:
            if p['kind'] == 'legacy_l1i':
                hits[int(p['target'][3:])] += 1
    assert set(np.nonzero(fired)[0]) == {3, 4, 5}          # only active pixels' detectors fire
    assert set(np.nonzero(hits)[0]) <= {3, 4, 5}           # inhibition local, never global


def test_switch_inhibition_follows_the_new_active_pixels():
    e = SimulationEngine(seed=1)
    e.set_pattern('row 1')
    for _ in range(1200):
        e.step()
    e.set_pattern('col 1')                                  # active pixels 1,4,7
    hits = np.zeros(N_PIX, dtype=int)
    for _ in range(1500):
        d = e.step()
        for p in d['inhibitory_pulses']:
            if p['kind'] == 'legacy_l1i':
                hits[int(p['target'][3:])] += 1
    hit_pixels = set(np.nonzero(hits)[0].tolist())
    assert hit_pixels                                       # inhibition happened
    assert hit_pixels <= {1, 4, 7}                          # only column-active pixels
