"""Tiled cortical-column execution: neuron parity, generic event-plastic learning,
local hard WTA, and event timing (Phases 2-3).

A single deterministic center-patch isolation run is captured once (module fixture) and
reused by the causal/WTA/timing assertions; the parity/learning checks build their own
small fixtures.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine
from snn.neurons import ExcitatoryNeuron, E_THRESHOLD


# ---------------------------------------------------------------- parity
def test_ordinary_e_and_eor_are_the_same_class_and_config():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    col = 'L1c11'
    E = next(c for c in e.latency_competitors
             if e._column_of.get(c.id) == col and e._role_of.get(c.id) == 'E')
    Eor = next(c for c in e.latency_competitors
               if e._column_of.get(c.id) == col and e._role_of.get(c.id) == 'Eor')
    assert type(E) is type(Eor) is ExcitatoryNeuron
    for attr in ('threshold', 'w_max', 'w_floor', 'eta', 'update_mode',
                 'maturity_budget_frac', 'leak_rate', 'refractory_steps',
                 'alpha_inh', 'alpha_a', 'beta_v', 'beta_s', 'a_max', 'learn'):
        assert getattr(E, attr) == getattr(Eor, attr), attr


def test_ordinary_e_and_eor_numeric_parity():
    # Two cells built through the SAME class/params with identical afferents + drive must
    # evolve identically (crossing, spike, reset, trace, weight update).
    def mk():
        return ExcitatoryNeuron('x', 'competitor',
                                acc_weights=np.array([300.0, 400.0, 200.0]),
                                acc_distance_factor=np.array([1.0, 0.9, 0.8]),
                                threshold=E_THRESHOLD, leak_rate=0.0, eta=0.01)
    a, b = mk(), mk()
    part = np.array([True, False, True])
    for cell in (a, b):
        cell.gather_exc(1100.0)                       # supra-threshold drive
        cell.freeze_drive()
        cell.advance_segment(1.0)
    assert a.crossing_time(1.0) == b.crossing_time(1.0) == 0.0
    for cell in (a, b):
        cell.fire(tau=0.5)
        cell.update_acc_weights(part)
        cell.update_trace()
    assert a.V == b.V and a.spiked == b.spiked
    assert np.array_equal(a.acc_weights, b.acc_weights)
    assert a.a == b.a and a.spike_tau == b.spike_tau


# ---------------------------------------------------------------- isolation run
def _classify(engine):
    return engine._role_of, engine._column_of


@pytest.fixture(scope='module')
def isolation():
    """Center-patch (1,1) isolation run; capture per-boundary spikes/resets/winners and
    C deposits, plus the engine and the pre-run weights."""
    e = SimulationEngine(seed=1, topology='tiled_cc', leak_rate=0.0)
    e.set_pattern('row 1')                           # center patch -> column L1c11
    role, col = _classify(e)
    eor_cell = next(c for c in e.latency_competitors
                    if col.get(c.id) == 'L1c11' and role.get(c.id) == 'Eor')
    # Only the cell that FIRES learns, so snapshot every L2 ordinary E's init weights and
    # resolve the actual winner after the run.
    l2e_cells = {c.id: c for c in e.latency_competitors if col.get(c.id) == 'L2c00'}
    init = dict(eor=eor_cell.acc_weights.copy(),
                l2e={cid: c.acc_weights.copy() for cid, c in l2e_cells.items()},
                top_c=e.exc['L2c00C'].basal_weight)
    frames = []
    l2_winner_id = None
    for _ in range(500):
        d = e.step()
        spikes = [(role.get(n['id']), col.get(n['id']), n['id'])
                  for n in d['neurons'] if n['spiked']]
        cdep = {n['id']: dict(count=n.get('coincidence_deposit_count', 0),
                              tau=n.get('coincidence_deposit_tau'),
                              spiked=n['spiked'],
                              spike_tau=n.get('spike_tau'))
                for n in d['neurons'] if role.get(n['id']) == 'C'}
        if l2_winner_id is None and 'L2c00' in d['column_winners']:
            l2_winner_id = d['column_winners']['L2c00']['id']
        frames.append(dict(t=d['timestep'], spikes=spikes,
                           winners=d['column_winners'],
                           resets=list(d['hard_reset_events']), cdep=cdep))
    return dict(engine=e, role=role, col=col, frames=frames, init=init,
                l2e_cells=l2e_cells, l2_winner_id=l2_winner_id, eor_cell=eor_cell)


def test_at_most_one_ordinary_e_winner_per_column_per_boundary(isolation):
    for f in isolation['frames']:
        per_col = {}
        for r, c, nid in f['spikes']:
            if r == 'E':
                per_col[c] = per_col.get(c, 0) + 1
        assert all(v <= 1 for v in per_col.values()), (f['t'], per_col)
        # column_winners records at most one per column
        assert all(isinstance(w, dict) for w in f['winners'].values())


def test_only_active_center_column_wins_no_cross_column_reset(isolation):
    role, col = isolation['role'], isolation['col']
    saw_center_winner = False
    for f in isolation['frames']:
        for cid in f['winners']:
            # only the center patch column and (later) L2 may win under isolation
            assert cid in ('L1c11', 'L2c00'), (f['t'], cid)
            if cid == 'L1c11':
                saw_center_winner = True
        # every hard reset stays inside one column; never targets an Eor
        for r in f['resets']:
            assert col.get(r['source']) == col.get(r['target'])
            assert role.get(r['target']) == 'E'      # Eor is never reset by local I
    assert saw_center_winner


def test_local_i_resets_its_whole_bank_including_the_winner(isolation):
    role, col = isolation['role'], isolation['col']
    checked = False
    for f in isolation['frames']:
        win = f['winners'].get('L1c11')
        if not win or not f['resets']:
            continue
        reset_targets = {r['target'] for r in f['resets'] if col.get(r['source']) == 'L1c11'}
        bank = {nid for nid in role if role[nid] == 'E' and col.get(nid) == 'L1c11'}
        assert reset_targets == bank                 # entire local bank reset
        assert win['id'] in reset_targets            # incl. the emitted winner
        checked = True
    assert checked


def test_i_is_one_shot_no_duplicate_reset(isolation):
    # In any boundary, no ordinary E is hard-reset twice (I emits at most once/boundary).
    for f in isolation['frames']:
        targets = [r['target'] for r in f['resets']]
        assert len(targets) == len(set(targets)), (f['t'], targets)


def test_feedforward_chain_matures_e_to_eor_to_l2(isolation):
    role, col = isolation['role'], isolation['col']

    def fired(r, c):
        return any(any(sr == r and sc == c for sr, sc, _ in f['spikes'])
                   for f in isolation['frames'])
    assert fired('E', 'L1c11')                       # ordinary E fires
    assert fired('Eor', 'L1c11')                     # its Eor fires (one hop later, learned)
    assert fired('E', 'L2c00')                       # an L2 ordinary E fires (second hop)


def test_l2_winner_apical_and_child_c_deposit_share_tau(isolation):
    # When an L2 ordinary E wins, every L1 C gets an apical at that tau; only the C with
    # local basal eligibility (the active column) deposits, at the same tau.
    checked = False
    for f in isolation['frames']:
        l2win = f['winners'].get('L2c00')
        if not l2win:
            continue
        active = f['cdep'].get('L1c11C')
        if active and active['count'] > 0:
            assert active['tau'] == pytest.approx(l2win['tau'])    # shared tau
            checked = True
    assert checked


def test_top_c_never_deposits_or_fires(isolation):
    for f in isolation['frames']:
        top = f['cdep']['L2c00C']
        assert top['count'] == 0 and top['spiked'] is False


def test_eor_and_l2e_learn_only_on_their_own_firing(isolation):
    eor = isolation['eor_cell']
    init = isolation['init']
    assert not np.array_equal(eor.acc_weights, init['eor'])       # Eor learned
    win_id = isolation['l2_winner_id']
    assert win_id is not None                                     # an L2 E won
    l2e = isolation['l2e_cells'][win_id]
    assert not np.array_equal(l2e.acc_weights, init['l2e'][win_id])   # the winner learned
    # L2 E owns nine child-Eor afferents; only the active child's weight rose, inactive
    # children were pushed down (per-target participation, no cross-hop leakage).
    idx_active = l2e.ff_src.index('L1c11Eor')
    assert l2e.acc_weights[idx_active] > init['l2e'][win_id][idx_active]
    inactive = [i for i in range(len(l2e.ff_src)) if i != idx_active]
    assert all(l2e.acc_weights[i] <= init['l2e'][win_id][i] + 1e-9 for i in inactive)


def test_top_c_basal_weight_never_changes(isolation):
    assert isolation['engine'].exc['L2c00C'].basal_weight == isolation['init']['top_c']


def test_rgc_owns_no_weight_receiving_e_owns_it(isolation):
    e = isolation['engine']
    # RGC feedforward weight lives on the receiving ordinary E, not the RGC source.
    rg = e.src['RGC40']
    assert not hasattr(rg, 'acc_weights')
    ref = e._ff_weight_ref['RGC40_L1c11E0']          # rg_to_column edge
    cell, widx = ref
    assert cell.id == 'L1c11E0' and e._role_of[cell.id] == 'E'


# ---------------------------------------------------------------- two-patch WTA
def test_two_l1_columns_can_each_win_in_same_boundary():
    # Independent tiled columns: two active patches each produce one local winner while
    # the single L2 column stays hard single-winner. (Not the deferred multi-winner work.)
    from backend.network_spec import embed_patch_pattern
    e = SimulationEngine(seed=1, topology='tiled_cc', leak_rate=0.0)
    role, col = e._role_of, e._column_of
    # activate patch (0,0) row and patch (2,2) row simultaneously
    v = np.array(embed_patch_pattern((9, 9), (3, 3), (0, 0), [0, 0, 0, 1, 1, 1, 0, 0, 0]))
    v += np.array(embed_patch_pattern((9, 9), (3, 3), (2, 2), [0, 0, 0, 1, 1, 1, 0, 0, 0]))
    e.set_input(v)
    both = False
    for _ in range(60):
        d = e.step()
        wcols = set(d['column_winners'])
        if {'L1c00', 'L1c22'} <= wcols:
            both = True
        # L2 stays single-winner: at most one L2 ordinary E winner
        assert sum(1 for c in wcols if c == 'L2c00') <= 1
    assert both
