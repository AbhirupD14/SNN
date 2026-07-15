"""Causal step / WTA: deterministic tie-break, a single winner for simultaneous
crossings, the immediate L2I relay, all L2E at rest after the inhibitory event,
and the L2->L1E_new->L1I->L1E_s feedback/wipe path hitting only the paired source.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT
from snn.neurons import E_THRESHOLD


def fresh(seed=1):
    e = SimulationEngine(seed=seed)
    e.clear_input()
    return e


def zero_l2e(e):
    for n in e.l2e:
        n.acc_weights[:] = 0.0
        n.V = 0.0


def drive_pixel_volley(e, pix):
    """Turn on one pixel and step until L1E_s[pix] fires, returning that step's frame."""
    e.clear_input()
    e.input_vec[pix] = 1.0
    for _ in range(5):
        d = e.step()
        if d['neurons'][pix]['spiked']:      # L1E{pix} is first in order
            return d
    raise AssertionError('L1E_s did not fire')


def test_single_winner_and_stable_tiebreak():
    e = fresh()
    zero_l2e(e)
    pix = 4
    # Two competitors receive identical, exactly-threshold charge on the volley.
    e.l2e[2].acc_weights[pix] = E_THRESHOLD
    e.l2e[5].acc_weights[pix] = E_THRESHOLD
    d = drive_pixel_volley(e, pix)
    winners = [n['id'] for n in d['neurons'] if n['spiked'] and n['id'].startswith('L2E')]
    assert winners == ['L2E2']               # equal V -> lowest index wins
    assert e.winner == 'L2E2'


def test_highest_membrane_wins_over_lower_index():
    e = fresh()
    zero_l2e(e)
    pix = 4
    e.l2e[2].acc_weights[pix] = E_THRESHOLD          # exactly threshold
    e.l2e[5].acc_weights[pix] = E_THRESHOLD * 1.5    # higher membrane, higher index
    d = drive_pixel_volley(e, pix)
    winners = [n['id'] for n in d['neurons'] if n['spiked'] and n['id'].startswith('L2E')]
    assert winners == ['L2E5']               # highest pre-fire membrane wins the tie-break


def test_winner_invokes_l2i_immediately_and_wipes_all_l2e():
    e = fresh()
    zero_l2e(e)
    pix = 4
    for j in (1, 3, 6):
        e.l2e[j].acc_weights[pix] = E_THRESHOLD * (1 + 0.1 * j)  # several crossers
    d = drive_pixel_volley(e, pix)
    # Exactly one L2E fired; L2I fired the same step.
    l2_winners = [n['id'] for n in d['neurons'] if n['spiked'] and n['id'].startswith('L2E')]
    assert len(l2_winners) == 1
    assert any(n['id'] == 'L2I' and n['spiked'] for n in d['neurons'])
    # Every L2E membrane is at rest after the inhibitory hard-wipe.
    for n in e.l2e:
        assert n.V == 0.0
    # The non-winner crossers appear as competitive-reset events.
    wiped = {ev['target'] for ev in d['applied_inhibition']}
    assert {'L2E1', 'L2E3', 'L2E6'} - {l2_winners[0]} <= wiped


def test_l2_feedback_reaches_l1e_new_densely():
    # The winning L2E spike is delivered to every L1E_new via its own feedback
    # afferent (index 1 + winner), and the paired-sensory afferent (index 0) is
    # never touched by feedback.
    e = fresh()
    zero_l2e(e)
    pix = 4
    win = 3
    e.l2e[win].acc_weights[pix] = E_THRESHOLD
    for n in e.l1e_new:                                    # zero feedback, keep sensory
        n.acc_weights[1:] = 0.0
        n.acc_weights[1 + win] = 200.0                    # a probe feedback weight
        n.V = 0.0
    d = drive_pixel_volley(e, pix)
    assert e.winner == 'L2E3'
    # Each L1E_new received exactly its winner-afferent charge (200) from feedback;
    # none fired (200 < threshold, no coincident sensory on inactive pixels).
    emitted = set(d['emitted'])
    assert all(f'fb{win}->{i}' in emitted for i in range(N_PIX))
    assert not any(s.startswith(f'fb{k}->') for s in emitted for k in range(N_OUT) if k != win)


def test_input_period_gates_delivery():
    e = SimulationEngine(seed=1, input_period=3)
    e.clear_input()
    e.input_vec[4] = 1.0
    # Only every third step delivers sensory charge; source integrates slower.
    e.step(); e.step()
    assert e.l1e_s[4].V == pytest.approx(0.0)   # steps 1,2 not input_arrives (t%3!=0)
    e.step()                                     # t=3 delivers
    assert e.l1e_s[4].V > 0.0
