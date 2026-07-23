"""Causal step / WTA under the synchronous conductance engine.

Pins the SHARED synchronous engine mechanics (no built-in preset uses them any more, but a
custom non-event-resolved NetworkSpec still does): deterministic single-winner arbitration
(highest membrane, lowest-index tie-break), that arbitration is SELECTION not charge removal
(losers keep their charge and are suppressed by the WTA conductance on the NEXT boundary),
and the explicit one-step feedforward delay + input-period gating. Built on a minimal
custom competitor/relay graph rather than a removed built-in preset.
"""

import pytest

from backend.simulation import SimulationEngine

N_PIX, N_OUT = 9, 8


def _sync_spec(n_pix=N_PIX, n_comp=N_OUT):
    """A minimal synchronous WTA graph: n_pix sensory sources fully connected to n_comp
    competitors, each competitor relaying to one shared inhibitory relay that inhibits
    every competitor (global WTA)."""
    nodes = ([{'id': f'S{i}', 'archetype': 'e_sensory', 'pixel': i} for i in range(n_pix)]
             + [{'id': f'C{j}', 'archetype': 'e_competitor'} for j in range(n_comp)]
             + [{'id': 'R', 'archetype': 'i_relay'}])
    edges = ([{'id': f'ff{i}->{j}', 'source': f'S{i}', 'target': f'C{j}', 'kind': 'feedforward'}
              for i in range(n_pix) for j in range(n_comp)]
             + [{'id': f're{j}', 'source': f'C{j}', 'target': 'R', 'kind': 'relay_excitation'}
                for j in range(n_comp)]
             + [{'id': f'inh{j}', 'source': 'R', 'target': f'C{j}', 'kind': 'inhibition'}
                for j in range(n_comp)])
    return {'name': 'sync_wta', 'nodes': nodes, 'edges': edges}


def fresh(seed=1, **cfg):
    e = SimulationEngine(seed=seed, leak_rate=0.0, **cfg)   # zero leak -> exact charge arithmetic
    e.apply_topology(_sync_spec())
    e.clear_input()
    for c in e.competitors:
        c.acc_weights[:] = 0.0
        c.V = 0.0
    return e


def step_until_spike(e, maxs=12):
    for _ in range(maxs):
        d = e.step()
        won = [n['id'] for n in d['neurons'] if n['spiked'] and n['id'].startswith('C')]
        if won:
            return d, won
    raise AssertionError('no competitor fired within the window')


def test_feedforward_has_one_step_delay():
    # A sensory spike drives its competitor on the FOLLOWING boundary, not the same one.
    e = fresh()
    e.competitors[3].acc_weights[4] = 1.2 * e.params['e_threshold']
    e.input_vec[4] = 1.0
    fired_src_step = winner_step = None
    for t in range(1, 10):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        if fired_src_step is None and by['S4']['spiked']:
            fired_src_step = t
            assert not by['C3']['spiked']              # competitor does NOT fire the same step
        elif fired_src_step is not None and by['C3']['spiked']:
            winner_step = t
            break
    assert winner_step == fired_src_step + 1            # exactly one boundary later


def test_single_winner_and_stable_tiebreak():
    e = fresh()
    thr = e.params['e_threshold']
    e.competitors[2].acc_weights[4] = 1.2 * thr        # equal supra-threshold charge
    e.competitors[5].acc_weights[4] = 1.2 * thr
    e.input_vec[4] = 1.0
    d, winners = step_until_spike(e)
    assert winners == ['C2']                            # equal V -> lowest index wins
    assert e.winner == 'C2'


def test_highest_membrane_wins_over_lower_index():
    e = fresh()
    thr = e.params['e_threshold']
    e.competitors[2].acc_weights[4] = 1.1 * thr
    e.competitors[5].acc_weights[4] = 1.6 * thr        # higher membrane, higher index
    e.input_vec[4] = 1.0
    d, winners = step_until_spike(e)
    assert winners == ['C5']


def test_wta_is_selection_not_charge_removal():
    e = fresh()
    thr = e.params['e_threshold']
    for j in (1, 3, 6):
        e.competitors[j].acc_weights[4] = (1.2 + 0.1 * j) * thr   # several crossers
    e.input_vec[4] = 1.0
    d, winners = step_until_spike(e)
    assert len(winners) == 1                            # exactly one winner fires
    winner = winners[0]
    by = {n['id']: n for n in d['neurons']}
    losers = [f'C{j}' for j in (1, 3, 6) if f'C{j}' != winner]
    assert all(by[l]['potential'] > 0.0 for l in losers)     # losers NOT hard-wiped to rest
    assert by[winner]['g_inh'] == 0.0                        # conductance not yet delivered
    # Next boundary: the WTA inhibitory conductance pulse lands (g_inh rises on competitors).
    d2 = e.step()
    by2 = {n['id']: n for n in d2['neurons']}
    assert by2[winner]['g_inh'] > 0.0


def test_input_period_gates_delivery():
    e = fresh(input_period=3)
    e.input_vec[4] = 1.0
    e.step(); e.step()
    assert e.neurons['S4'].V == pytest.approx(0.0)      # t=1,2 deliver nothing (t%3!=0)
    e.step()                                            # t=3 delivers
    assert e.neurons['S4'].V > 0.0
