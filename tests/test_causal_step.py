"""Causal step / WTA under the synchronous conductance engine.

Pins: deterministic single-winner arbitration (highest membrane, lowest-index
tie-break), that arbitration is SELECTION not charge removal (losers keep their
charge and are instead suppressed by the L2I_WTA conductance on the NEXT boundary),
that the winner spike precedes the feedback inhibition it causes (L2I cannot cancel
its own winner), and the explicit one-step feedforward delay.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT


def fresh(seed=1):
    e = SimulationEngine(seed=seed, leak_rate=0.0)     # zero leak -> exact charge arithmetic
    e.clear_input()
    return e


def zero_l2e(e):
    for n in e.l2e:
        n.acc_weights[:] = 0.0
        n.V = 0.0


def step_until_l2_spike(e, maxs=12):
    for _ in range(maxs):
        d = e.step()
        l2 = [n['id'] for n in d['neurons'] if n['spiked'] and n['id'].startswith('L2E')]
        if l2:
            return d, l2
    raise AssertionError('no L2E fired within the window')


def test_feedforward_has_one_step_delay():
    # An L1E_s spike drives L2E on the FOLLOWING boundary, not the same one.
    e = fresh()
    zero_l2e(e)
    e.l2e[3].acc_weights[4] = 1.2 * e.params['e_threshold']
    e.input_vec[4] = 1.0
    fired_l1_step = winner_step = None
    for t in range(1, 10):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        if fired_l1_step is None and by['L1E4']['spiked']:
            fired_l1_step = t
            assert not by['L2E3']['spiked']            # L2E does NOT fire the same step
        elif fired_l1_step is not None and by['L2E3']['spiked']:
            winner_step = t
            break
    assert winner_step == fired_l1_step + 1            # exactly one boundary later


def test_single_winner_and_stable_tiebreak():
    e = fresh()
    zero_l2e(e)
    pix = 4
    thr = e.params['e_threshold']
    e.l2e[2].acc_weights[pix] = 1.2 * thr              # equal supra-threshold charge
    e.l2e[5].acc_weights[pix] = 1.2 * thr
    e.input_vec[pix] = 1.0
    d, winners = step_until_l2_spike(e)
    assert winners == ['L2E2']                         # equal V -> lowest index wins
    assert e.winner == 'L2E2'


def test_highest_membrane_wins_over_lower_index():
    e = fresh()
    zero_l2e(e)
    pix = 4
    thr = e.params['e_threshold']
    e.l2e[2].acc_weights[pix] = 1.1 * thr
    e.l2e[5].acc_weights[pix] = 1.6 * thr              # higher membrane, higher index
    e.input_vec[pix] = 1.0
    d, winners = step_until_l2_spike(e)
    assert winners == ['L2E5']


def test_wta_is_selection_and_l2i_conductance_lands_next_boundary():
    e = fresh()
    zero_l2e(e)
    pix = 4
    thr = e.params['e_threshold']
    for j in (1, 3, 6):
        e.l2e[j].acc_weights[pix] = (1.2 + 0.1 * j) * thr   # several crossers
    e.input_vec[pix] = 1.0
    d, winners = step_until_l2_spike(e)
    assert len(winners) == 1                           # exactly one winner fires
    winner = winners[0]
    # L2I_WTA fired the SAME boundary the winner did (relay), but no L2E was wiped to
    # rest by arbitration: losers keep their membrane charge this boundary.
    assert any(n['id'] == 'L2I' and n['spiked'] for n in d['neurons'])
    by = {n['id']: n for n in d['neurons']}
    losers = [f'L2E{j}' for j in (1, 3, 6) if f'L2E{j}' != winner]
    assert all(by[l]['potential'] > 0.0 for l in losers)     # NOT hard-wiped
    assert by[winner]['g_inh'] == 0.0                        # conductance not yet delivered
    # Next boundary: the WTA conductance pulse lands on every L2E (kind 'wta').
    d2 = e.step()
    wta = [p for p in d2['inhibitory_pulses'] if p['kind'] == 'wta']
    assert {p['target'] for p in wta} == {f'L2E{j}' for j in range(N_OUT)}
    assert all(p['conductance_increment'] > 0 for p in wta)
    assert all(by2['g_inh'] > 0 for by2 in
               (n for n in d2['neurons'] if n['id'].startswith('L2E')))


def test_old_winner_relays_densely_to_all_l1i_next_boundary():
    # Old topology: the winning L2E spike is emitted into the dense L2E->L1I relay to
    # every L1I (edge re_l1i_{win}->{i}); no other competitor's relay is emitted.
    e = SimulationEngine(seed=1, topology='old', leak_rate=0.0)
    e.clear_input()
    zero_l2e(e)
    pix, win = 4, 3
    e.l2e[win].acc_weights[pix] = 1.2 * e.params['e_threshold']
    e.input_vec[pix] = 1.0
    d, winners = step_until_l2_spike(e)
    assert winners == ['L2E3']
    emitted = set(d['emitted'])
    assert all(f're_l1i_{win}->{i}' in emitted for i in range(N_PIX))   # dense to all L1I
    assert not any(s.startswith(f're_l1i_{k}->') for s in emitted
                   for k in range(N_OUT) if k != win)
    # Every L1I fired this boundary; the paired inhibition lands next boundary.
    assert all(n['spiked'] for n in d['neurons'] if n['id'].startswith('L1I'))


def test_input_period_gates_delivery():
    e = SimulationEngine(seed=1, input_period=3, leak_rate=0.0)
    e.clear_input()
    e.input_vec[4] = 1.0
    e.step(); e.step()
    assert e.l1e_s[4].V == pytest.approx(0.0)          # t=1,2 deliver nothing (t%3!=0)
    e.step()                                            # t=3 delivers
    assert e.l1e_s[4].V > 0.0
