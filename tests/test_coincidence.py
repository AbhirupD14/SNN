"""L1E_new local coincidence detector, delayed I->E inhibition, and pixel-selective
learning -- the corrected L1 feedback circuit.

L1E_new[i] has nine accumulating afferents: index 0 is the paired local sensory
afferent from L1E_s[i]; indices 1..8 are dense L2E feedback. It fires only when its
paired sensory input and an L2E winner coincide (weighted integration + leak, not a
boolean gate). A firing L1E_new[i] triggers L1I[i], whose subtractive output is
queued and delivered to L1E_s[i] on the NEXT timestep.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT
from snn.neurons import ExcitatoryNeuron, E_THRESHOLD


def fresh(leak=0.03, seed=1):
    e = SimulationEngine(seed=seed, leak_rate=leak)
    e.clear_input()
    return e


def mature_enew(leak, winner_j=0, sensory=500.0, winner_w=500.0):
    w = np.zeros(1 + N_OUT)
    w[0] = sensory
    w[1 + winner_j] = winner_w
    return ExcitatoryNeuron('L1Enew', 'supervisor', acc_weights=w,
                            acc_distance_factor=np.ones(1 + N_OUT),
                            threshold=E_THRESHOLD, w_max=E_THRESHOLD / 2.0,
                            leak_rate=leak, learn=False)


# --------------------------------------------------------------- construction
def test_l1e_new_has_nine_afferents_and_subthreshold_init():
    e = SimulationEngine(seed=1)
    for n in e.l1e_new:
        assert n.acc_weights.shape == (1 + N_OUT,)          # 1 sensory + 8 feedback
        assert np.all(n.acc_weights >= 0) and np.all(n.acc_weights <= e.params['e_weight_cap'])
        assert n.acc_weights.sum() < e.params['e_threshold']  # p starts positive
    assert e.params['e_weight_cap'] == pytest.approx(500.0)   # shared cap theta/2


def test_default_leak_is_nonzero():
    # Zero leak is not a valid default for this circuit.
    assert SimulationEngine(seed=1).params['leak_rate'] > 0.0


# ------------------------------------------------------------- charge delivery
def test_local_sensory_reaches_only_paired_l1e_new():
    e = fresh()
    e.input_vec[:] = 0.0
    e.input_vec[4] = 1.0
    for n in e.l1e_new:
        n.acc_weights[0] = 400.0          # visible paired-sensory weight
        n.V = 0.0
    # Step until L1E_s[4] fires and delivers locally.
    for _ in range(6):
        d = e.step()
        if 'cl4' in d['emitted']:
            break
    assert 'cl4' in d['emitted']                              # paired local delivery fired
    assert not any(f'cl{i}' in d['emitted'] for i in range(N_PIX) if i != 4)
    # Only the paired L1E_new received local sensory charge (others got none locally).
    assert e.l1e_new[4].V > 0.0


def test_geometry_does_not_affect_delivered_charge():
    # Two mature detectors with identical weights but different distance factors get
    # identical charge from the same spike.
    near = mature_enew(0.0); far = mature_enew(0.0)
    near.acc_distance_factor[:] = 1.0
    far.acc_distance_factor[:] = 0.01
    near.receive_acc(near.acc_weights[0]); far.receive_acc(far.acc_weights[0])
    assert near.V == far.V == pytest.approx(500.0)


# ---------------------------------------------------------------- coincidence
def _drive(mode, leak, T, winner_j=0, steps=400):
    n = mature_enew(leak, winner_j)
    for t in range(1, steps + 1):
        if t % T == 0:
            if mode in ('sensory', 'coincident'):
                n.receive_acc(n.acc_weights[0])
            if mode in ('winner', 'coincident'):
                n.receive_acc(n.acc_weights[1 + winner_j])
        if n.can_fire():
            n.fire(); return t
        n.advance()
    return None


def test_mature_coincidence_rejects_lone_branches_under_leak():
    # At a leak that rejects a lone 500 branch, only the coincident (500+500) train
    # crosses threshold -- an emergent AND from weighted integration + leak.
    leak, T = 0.20, 5
    assert _drive('sensory', leak, T) is None               # lone sensory subthreshold
    assert _drive('winner', leak, T) is None                # lone winner subthreshold
    assert _drive('coincident', leak, T) is not None        # coincidence fires


def test_coincident_fires_on_first_volley_regardless_of_leak():
    # 500 + 500 = theta arrives in one step, so a coincident volley always fires.
    for leak in (0.0, 0.1, 0.3):
        assert _drive('coincident', leak, T=5) == 5


# ------------------------------------------------------------- local learning
def test_participation_potentiates_active_and_depresses_inactive():
    # A learning L1E_new firing on a coincident (paired sensory + winner j) event:
    # sensory[0] and feedback[1+j] potentiate; the other feedback afferents depress.
    n = ExcitatoryNeuron('L1Enew', 'supervisor',
                         acc_weights=np.array([300.0, 40., 40., 40., 40., 40., 40., 40., 40.]),
                         acc_distance_factor=np.ones(9),
                         threshold=E_THRESHOLD, w_max=E_THRESHOLD / 2.0, eta=0.02)
    before = n.acc_weights.copy()
    n.fire()
    part = np.zeros(9, dtype=bool); part[0] = True; part[1 + 2] = True   # sensory + winner 2
    n.update_acc_weights(part)
    assert n.acc_weights[0] > before[0]                     # paired sensory up
    assert n.acc_weights[1 + 2] > before[1 + 2]             # winning feedback up
    for j in range(N_OUT):
        if j != 2:
            assert n.acc_weights[1 + j] < before[1 + j]     # other feedback down
    assert np.all(n.acc_weights <= E_THRESHOLD / 2.0)       # within [0, 500]


def test_only_participating_pixel_enew_can_learn():
    # Over a held row pattern, only the active-pixel L1E_new fire/learn; inactive-pixel
    # detectors stay quiet (no false training toward the winner) -> selective, not global.
    e = SimulationEngine(seed=1)                            # default leak 0.03
    e.set_pattern('row 1')                                  # active pixels 3,4,5
    active_fired = inactive_fired = 0
    for _ in range(1500):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        for i in range(N_PIX):
            if by[f'L1Enew{i}']['spiked']:
                if i in (3, 4, 5):
                    active_fired += 1
                else:
                    inactive_fired += 1
    assert active_fired > 0
    assert inactive_fired == 0                              # the corrected-bug invariant


# --------------------------------------------------------------------- delay
def test_l1i_relay_fires_on_coincidence_step_but_wipe_is_delayed():
    # Construct a single coincident L1E_new fire and check the relay fires this step
    # while its wipe is queued (no same-step L1E_s wipe), landing next step.
    e = fresh(leak=0.0)
    for n in e.l2e:
        n.acc_weights[:] = 0.0
    e.l2e[0].acc_weights[4] = E_THRESHOLD                   # L2E0 wins on pixel 4
    for n in e.l1e_new:
        n.acc_weights[:] = 0.0
    e.l1e_new[4].acc_weights[0] = 500.0                    # paired sensory
    e.l1e_new[4].acc_weights[1 + 0] = 500.0                # winner-0 feedback -> coincidence
    e.input_vec[:] = 0.0; e.input_vec[4] = 1.0
    for _ in range(6):
        d = e.step()
        if any(n['id'] == 'L1I4' and n['spiked'] for n in d['neurons']):
            break
    # L1I4 fired this step; the wipe was NOT applied to L1E4 this step (queued).
    assert 're_l1_4' in d['emitted']
    assert not any(ev['target'] == 'L1E4' for ev in d['applied_inhibition'])
    assert e._pending_inh[4]                                # queued for next step
    # Next step, after sensory deposit but before the L1E_s crossing, the wipe lands.
    e.l1e_s[4].V = 600.0                                    # subthreshold charge to remove
    d2 = e.step()
    landed = {ev['target']: ev for ev in d2['applied_inhibition']}
    assert 'L1E4' in landed and landed['L1E4']['charge_removed'] > 0.0


def test_delayed_inhibition_lowers_source_cadence():
    # The paired-source cadence in the live circuit is longer than the un-inhibited
    # integrator cadence (a held pixel at zero leak fires every 3 steps).
    e = SimulationEngine(seed=1, leak_rate=0.0)
    e.set_pattern('row 1')
    spikes = 0
    for _ in range(600):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        spikes += by['L1E3']['spiked']
    cadence = 600 / max(1, spikes)
    assert cadence > 3.0                                    # inhibition delayed some spikes


# ------------------------------------------------------------- pattern switch
def test_switch_inhibition_is_pixel_selective_not_global():
    # After training a row and switching to a column, the delayed inhibition lands on
    # the column's active pixels (coincidence), never on all nine pixels globally.
    e = SimulationEngine(seed=1)
    e.set_pattern('row 1')
    for _ in range(1500):
        e.step()
    e.set_pattern('col 1')                                  # active pixels 1,4,7
    inh_hits = np.zeros(N_PIX, dtype=int)
    for _ in range(1500):
        d = e.step()
        for ev in d['applied_inhibition']:
            t = ev['target']
            if t.startswith('L1E') and not t.startswith('L1Enew'):
                inh_hits[int(t[3:])] += 1
    hit_pixels = set(np.nonzero(inh_hits)[0].tolist())
    assert hit_pixels                                       # inhibition happened
    assert hit_pixels <= {1, 4, 7}                          # only column-active pixels
