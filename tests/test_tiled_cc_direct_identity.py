"""Direct-identity tiled cortical columns (``tiled_cc_direct_identity``): builder,
structural validation, and source-addressed identity transmission.

Eor is removed. Each ordinary E projects to EVERY parent ordinary E, so the parent owns a
distinct plastic weight per ``(child column, child winner)`` address rather than one pooled
"this column fired" event, and each column's C owns one learned basal per local ordinary E.
The prior tiled presets are left completely unchanged (asserted here too).
"""

import copy

import numpy as np
import pytest

from backend.network_spec import (
    tiled_cc_direct_identity_spec, tiled_cc_spec, preset_spec, validate_spec, SpecError,
    build_direct_column, connect_direct_identity, DirectColumnHandles,
    TILED_FAMILY, TILED_VARIANT_DIRECT_IDENTITY,
)
from backend.simulation import SimulationEngine, CoincidencePyramidalNeuron, E_THRESHOLD
from backend import presets as ps

N_IN = 81
THETA = E_THRESHOLD


def _counts(spec):
    return len(spec['nodes']), len(spec['edges'])


def _by_projection(spec):
    out = {}
    for e in spec['edges']:
        out.setdefault(e.get('projection'), []).append(e)
    return out


# ---------------------------------------------- 1-3: counts, determinism, roles
def test_default_is_181_nodes_1546_edges():
    spec = tiled_cc_direct_identity_spec()
    assert _counts(spec) == (181, 1546)
    norm = validate_spec(spec, N_IN)                    # survives normalization
    assert _counts(norm) == (181, 1546)
    assert norm['topology']['variant'] == TILED_VARIANT_DIRECT_IDENTITY
    assert norm['topology']['family'] == TILED_FAMILY


def test_edge_component_counts_match_the_declared_contract():
    p = _by_projection(tiled_cc_direct_identity_spec())
    assert len(p['rg_to_column']) == 648                # 81 RGC * 8 local E
    assert len(p['identity_child_e_to_parent_e']) == 576   # 9 * 8 * 8
    assert len(p['column_to_column_apical']) == 72      # 8 L2 E * 9 L1 C
    assert len(p['column_e_to_c_basal']) == 80          # 10 columns * 8 E
    assert len(p['column_e_to_i']) == 80
    assert len(p['column_i_to_e']) == 80
    assert len(p['column_c_to_i']) == 10
    assert sum(len(v) for v in p.values()) == 1546


def test_construction_is_deterministic_and_ids_unique():
    a, b = tiled_cc_direct_identity_spec(), tiled_cc_direct_identity_spec()
    ids = [n['id'] for n in a['nodes']]
    eids = [e['id'] for e in a['edges']]
    assert len(ids) == len(set(ids)) and len(eids) == len(set(eids))
    assert ids == [n['id'] for n in b['nodes']]
    assert eids == [e['id'] for e in b['edges']]


def test_no_eor_anywhere():
    spec = tiled_cc_direct_identity_spec()
    assert not any(n.get('column_role') == 'Eor' for n in spec['nodes'])
    assert not any('Eor' in n['id'] for n in spec['nodes'])
    ids = {n['id'] for n in spec['nodes']}
    assert all(e['source'] in ids and e['target'] in ids for e in spec['edges'])
    # the engine must expose no Eor cell or Eor weight either
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity')
    assert all(e._role_of.get(c.id) != 'Eor' for c in e.latency_competitors)
    assert not any('Eor' in s['id'] for s in e.topology()['synapses'])


def test_exact_population_and_role_counts():
    spec = tiled_cc_direct_identity_spec()
    arch, roles = {}, {}
    for n in spec['nodes']:
        arch[n['archetype']] = arch.get(n['archetype'], 0) + 1
        roles[n.get('column_role')] = roles.get(n.get('column_role'), 0) + 1
    assert arch == {'rg_source': 81, 'e_latency_competitor': 80,
                    'e_coincidence': 10, 'i_relay': 10}
    assert roles[None] == 81 and roles['E'] == 80 and roles['C'] == 10 and roles['I'] == 10


# ------------------------------------------------- 4-6: local structural rules
def test_each_c_has_eight_source_distinct_local_basal_edges():
    spec = tiled_cc_direct_identity_spec()
    meta = {n['id']: n for n in spec['nodes']}
    basal = {}
    for e in spec['edges']:
        if e['kind'] == 'basal_excitation':
            basal.setdefault(e['target'], []).append(e['source'])
    assert len(basal) == 10
    for c_id, srcs in basal.items():
        assert len(srcs) == 8 and len(set(srcs)) == 8      # source-distinct, never pooled
        assert meta[c_id].get('multi_basal') is True
        for s in srcs:                                     # local ordinary E only
            assert meta[s]['column_role'] == 'E'
            assert meta[s]['column_id'] == meta[c_id]['column_id']


def test_no_rgc_edge_reaches_a_c_or_an_i():
    spec = tiled_cc_direct_identity_spec()
    meta = {n['id']: n for n in spec['nodes']}
    for e in spec['edges']:
        if meta[e['source']]['archetype'] == 'rg_source':
            assert meta[e['target']].get('column_role') == 'E'


def test_top_c_is_dormant_and_children_have_apical():
    spec = tiled_cc_direct_identity_spec()
    meta = {n['id']: n for n in spec['nodes']}
    apical = {}
    for e in spec['edges']:
        if e['kind'] == 'apical_excitation':
            apical.setdefault(e['target'], set()).add(e['source'])
    assert meta['L2c00C']['has_parent'] is False
    assert 'L2c00C' not in apical                          # dormant: zero apical sources
    for pr in range(3):
        for pc in range(3):
            c_id = f'L1c{pr}{pc}C'
            assert len(apical[c_id]) == 8                  # every L2 ordinary E
            assert all(meta[s]['column_id'] == 'L2c00' for s in apical[c_id])


# ------------------------------------------------------ 7: identity projection
def test_every_child_e_addresses_every_parent_e_distinctly():
    spec = tiled_cc_direct_identity_spec()
    meta = {n['id']: n for n in spec['nodes']}
    l2_e = {n['id'] for n in spec['nodes']
            if n.get('column_id') == 'L2c00' and n.get('column_role') == 'E'}
    reached = {}
    for e in spec['edges']:
        if e.get('projection') == 'identity_child_e_to_parent_e':
            reached.setdefault(e['source'], set()).add(e['target'])
    children = [n['id'] for n in spec['nodes']
                if n.get('column_role') == 'E' and meta[n['id']]['column_id'] != 'L2c00']
    assert len(children) == 72
    for ce in children:
        assert reached[ce] == l2_e            # each source address reaches every parent E
    # ...and each L2 E therefore owns 72 distinct source addresses
    incoming = {}
    for e in spec['edges']:
        if e.get('projection') == 'identity_child_e_to_parent_e':
            incoming.setdefault(e['target'], set()).add(e['source'])
    for pe in l2_e:
        assert len(incoming[pe]) == 72        # 9 child columns * 8 possible winners


def test_l2_e_weight_index_is_source_addressed_in_the_engine():
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity')
    l2e = e.exc['L2c00E0']
    assert len(l2e.ff_src) == 72 and len(set(l2e.ff_src)) == 72
    # every (child column, child winner) pair appears exactly once, and no pooled source
    assert all(e._role_of.get(s) == 'E' for s in l2e.ff_src)
    assert {e._column_of[s] for s in l2e.ff_src} == {f'L1c{r}{c}' for r in range(3)
                                                     for c in range(3)}
    assert not any('Eor' in s or s.endswith('c00') for s in l2e.ff_src)


# ------------------------------------------- 8: validation rejects malformations
def _mutate_and_expect(mutate, match=None):
    spec = tiled_cc_direct_identity_spec()
    mutate(spec)
    with pytest.raises(SpecError, match=match):
        validate_spec(spec, N_IN)


def test_cross_column_basal_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1c00E0', target='L1c01C', kind='basal_excitation')),
        match='cross-column basal')


def test_duplicate_basal_edge_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1c00E0', target='L1c00C', kind='basal_excitation')),
        match='duplicate basal_excitation')


def test_duplicate_apical_edge_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L2c00E0', target='L1c00C', kind='apical_excitation')),
        match='duplicate apical_excitation')


def test_lateral_e_to_e_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1c00E0', target='L1c00E1', kind='feedforward')),
        match='lateral E->E')


def test_missing_identity_edge_rejected():
    def drop(s):
        s['edges'] = [e for e in s['edges'] if e['id'] != 'L1c00E0_L2c00E0']
    _mutate_and_expect(drop, match='must address every ordinary E')


def test_missing_basal_edge_rejected():
    def drop(s):
        s['edges'] = [e for e in s['edges'] if e['id'] != 'L1c00_E3_c']
    _mutate_and_expect(drop, match='source-distinct basal')


def test_an_eor_role_is_rejected():
    _mutate_and_expect(lambda s: s['nodes'].append(dict(
        id='X', archetype='e_latency_competitor', layer='L1',
        column_id='L1c00', column_role='Eor')), match='has no output relay')


def test_apical_onto_the_dormant_top_c_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L2c00E0', target='L2c00C', kind='apical_excitation')),
        match='must have zero apical_excitation')


def test_multi_basal_capability_is_opt_in_only():
    """A C that has NOT declared multi_basal keeps the historical exactly-one invariant,
    so no existing custom graph is silently relaxed."""
    spec = {'name': 'x', 'nodes': [
        {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
        {'id': 'S1', 'archetype': 'e_sensory', 'pixel': 1},
        {'id': 'P', 'archetype': 'e_sensory', 'pixel': 2},
        {'id': 'C', 'archetype': 'e_coincidence'}],
        'edges': [{'id': 'b0', 'source': 'S0', 'target': 'C', 'kind': 'basal_excitation'},
                  {'id': 'b1', 'source': 'S1', 'target': 'C', 'kind': 'basal_excitation'},
                  {'id': 'a', 'source': 'P', 'target': 'C', 'kind': 'apical_excitation'}]}
    with pytest.raises(SpecError, match='exactly one incoming'):
        validate_spec(copy.deepcopy(spec), 9)
    opted = copy.deepcopy(spec)
    next(n for n in opted['nodes'] if n['id'] == 'C')['multi_basal'] = True
    norm = validate_spec(opted, 9)                       # accepted once declared
    assert norm['nodes'][-1]['multi_basal'] is True      # and preserved through validation


# ---------------------------------- 9: engine construction, serialization, store
def test_engine_builds_multi_basal_cells_with_aligned_state():
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity')
    assert e.mode == 'tiled_cc_direct_identity'
    top = e.topology()
    assert len(top['neurons']) == 181 and len(top['synapses']) == 1546
    for c in e.coincidence:
        assert isinstance(c, CoincidencePyramidalNeuron)
        assert c.n_basal == 8
        assert len(c.basal.edge_ids) == 8 and len(set(c.basal.edge_ids)) == 8
        assert c.basal.weights.shape == c.basal.distance_factors.shape == (8,)
        assert c.w_cap == THETA                          # one-afferent ceiling, not theta/2


def test_every_basal_edge_serializes_its_own_weight():
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity')
    c = e.exc['L1c11C']
    c.basal.weights[:] = [10.0 * i for i in range(8)]
    got = {s['id']: s['weight'] for s in e.topology()['synapses']
           if s['kind'] == 'basal_excitation' and s['target'] == 'L1c11C'}
    assert sorted(got.values()) == [10.0 * i for i in range(8)]
    # ...and each id maps back to the matching source
    for eid, w in got.items():
        cell, bidx = e._basal_weight_ref[eid]
        assert cell is c and c.basal.weights[bidx] == w


def test_manual_edit_and_branch_round_trip_preserve_every_basal_weight():
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity')
    c = e.exc['L1c11C']
    target = {eid: 100.0 + 7.0 * i for i, eid in enumerate(c.basal.edge_ids)}
    for eid, w in target.items():
        assert e.set_synapse_weight(eid, w) == w
    assert [c.basal.weights[i] for i in range(8)] == list(target.values())
    snap = {s['id']: s['weight'] for s in e.topology()['synapses']
            if s['weight'] is not None}
    e.branch_from_weights(snap)                          # full-snapshot branch
    c2 = e.exc['L1c11C']
    for eid, w in target.items():
        cell, bidx = e._basal_weight_ref[eid]
        assert cell.basal.weights[bidx] == w             # every source restored exactly
    assert c2.n_basal == 8


def test_preset_store_lists_and_loads_the_new_topology(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, 'PRESET_DIR', str(tmp_path))
    entry = next(p for p in ps.list_presets(9, 8)
                 if p['name'] == 'tiled_cc_direct_identity')
    assert entry['builtin'] and (entry['nodes'], entry['edges']) == (181, 1546)
    spec = ps.load_spec('tiled_cc_direct_identity', 81, 8)
    assert all(n.get('pos') is not None for n in spec['nodes'])   # real 3D positions
    ps.save_preset('di copy', SimulationEngine(
        seed=1, topology='tiled_cc_direct_identity').current_spec(), 81)
    loaded = ps.load_spec('di copy', 81, 8)
    assert _counts(loaded) == (181, 1546)
    assert loaded['topology']['variant'] == TILED_VARIANT_DIRECT_IDENTITY
    assert next(n for n in loaded['nodes']
                if n['id'] == 'L1c00C')['multi_basal'] is True


# ------------------------------------ 10: reusable builders + prior presets safe
def test_builders_return_fresh_state():
    _, n1, e1 = build_direct_column('A', 'L1', 0, 0, n_e=3, has_parent=True)
    _, n2, e2 = build_direct_column('B', 'L1', 0, 1, n_e=3, has_parent=True)
    assert n1 is not n2 and e1 is not e2
    n1.append({'poison': True})
    assert not any('poison' in n for n in n2)


def test_identity_link_is_the_full_cross_product():
    a, _, _ = build_direct_column('A', 'L1', 0, 0, n_e=3, has_parent=True)
    b, _, _ = build_direct_column('B', 'L2', 0, 0, n_e=4, has_parent=False)
    edges = connect_direct_identity(a, b)
    ff = [e for e in edges if e['kind'] == 'feedforward']
    ap = [e for e in edges if e['kind'] == 'apical_excitation']
    assert len(ff) == 12 and len(ap) == 4                # N_child*N_parent, N_parent
    assert {(e['source'], e['target']) for e in ff} == {
        (s, t) for s in a.e_ids for t in b.e_ids}


def test_prior_tiled_presets_are_structurally_unchanged():
    assert _counts(tiled_cc_spec(cc_e_count=8)) == (191, 1052)
    assert 'variant' not in tiled_cc_spec(cc_e_count=8)['topology']   # classic stays absent
    assert _counts(preset_spec('rg_coincidence', 9, 8)) == (45, 196)
    assert _counts(preset_spec('tiled_cc_l1_4', 81, 8)) == (155, 620)


# ------------------------------------- 11: driven identity transmission (engine)
def _drive_one_l1_winner(seed=1, column='L1c11', winner_index=0, boundaries=4):
    """Force ONE ordinary E of one L1 column to win, then run boundaries so its identity
    event is delivered to L2. Charge is scheduled through the engine's own delay-1 path
    (``_sched_exc``), so the local WTA and hard reset still decide the winner -- the probe
    only biases which cell crosses first.

    Returns (engine, winner_id, {L2 cell: delivered source ids})."""
    e = SimulationEngine(seed=seed, topology='tiled_cc_direct_identity', leak_rate=0.0)
    winner = f'{column}E{winner_index}'
    for _ in range(boundaries):
        e._sched_exc(winner, 2.0 * THETA)
        e.step()
    assert e.spiked[winner], 'probe failed to make the chosen E win'
    delivered = {nid: set(srcs) for nid, srcs in e._ff_deliv_now.items()}
    return e, winner, delivered


def test_a_driven_winner_reaches_every_l2_e_through_its_own_weight_index():
    e, winner, delivered = _drive_one_l1_winner(winner_index=3)
    l2_e = [f'L2c00E{i}' for i in range(8)]
    for pe in l2_e:
        assert winner in delivered.get(pe, set()), pe      # identity reached every parent E
        cell = e.exc[pe]
        widx = cell.ff_src.index(winner)                   # ...at ITS OWN address
        assert cell.ff_edge_ids[widx] == f'{winner}_{pe}'


def test_a_different_local_winner_selects_different_afferents():
    _, w0, d0 = _drive_one_l1_winner(winner_index=0)
    _, w5, d5 = _drive_one_l1_winner(winner_index=5)
    assert w0 != w5
    reached0 = d0.get('L2c00E0', set()) & {f'L1c11E{i}' for i in range(8)}
    reached5 = d5.get('L2c00E0', set()) & {f'L1c11E{i}' for i in range(8)}
    assert reached0 == {w0} and reached5 == {w5}           # disjoint source addresses


def test_no_pooled_source_appears_in_delivery_state():
    e, winner, delivered = _drive_one_l1_winner(winner_index=2)
    for pe, srcs in delivered.items():
        for s in srcs:
            assert e._role_of.get(s) in ('E', None)        # never an Eor / column-level id
            assert 'Eor' not in s


def test_local_wta_still_permits_one_winner_per_child_column():
    e, winner, _ = _drive_one_l1_winner(winner_index=6)
    for _ in range(20):
        e._sched_exc(winner, 2.0 * THETA)
        e.step()
        per_column = {}
        for nid, spiked in e.spiked.items():
            if spiked and e._role_of.get(nid) == 'E':
                per_column.setdefault(e._column_of[nid], []).append(nid)
        for cid, fired in per_column.items():
            assert len(fired) == 1, (cid, fired)           # hard single winner per column


def test_c_learns_only_the_winning_source_in_a_live_column():
    """End-to-end: in a trained column only the OWNER's basal association matures; every
    other local E keeps its initial weight byte-identical."""
    e = SimulationEngine(seed=1, topology='tiled_cc_direct_identity', dual_fe_fes=True,
                         eta=4.0, c_eta=2.0, dual_fe_B=5.0, leak_rate=0.0,
                         refractory_steps=0, c_feedback_reset=True, e_weight_cap_frac=0.5)
    c = e.exc['L1c11C']
    init = float(c.basal.weights[0])
    e.set_patch_pattern(1, 1, 'row 1')
    winners = set()
    for _ in range(3000):
        st = e.step()
        w = st['column_winners'].get('L1c11')
        if w:
            winners.add(w['id'])
    assert len(winners) == 1, winners                       # a single stable local owner
    owner = winners.pop()
    k = c.basal.source_ids.index(owner)
    assert c.basal.weights[k] > init                        # the owner's association matured
    others = [float(w) for i, w in enumerate(c.basal.weights) if i != k]
    assert others == [init] * 7                             # ...and no other was depressed
