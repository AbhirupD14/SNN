"""Tiled cortical-column builder + structural validation (Phase 1).

Covers the spec's "Builder and validation" focused tests: exact node/edge counts at the
default N and arbitrary N, per-column composition, patch RGC completeness, exact internal
edge sets, child/parent link completeness, the dormant top C, and rejection of malformed
missing/extra/cross-column graphs, plus metadata survival through validate/export.
"""

import copy

import pytest

from backend.network_spec import (
    tiled_cc_spec, validate_spec, SpecError, embed_patch_pattern,
    build_cortical_column, connect_rgc_patch, connect_columns, TILED_FAMILY,
)

N_IN = 81


def _counts(spec):
    return len(spec['nodes']), len(spec['edges'])


def test_default_n8_is_191_nodes_1052_edges():
    spec = tiled_cc_spec(cc_e_count=8)
    assert _counts(spec) == (191, 1052)
    # acceptance: exact counts are a contract, re-checked after validation/normalization.
    norm = validate_spec(spec, N_IN)
    assert _counts(norm) == (191, 1052)


@pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 12, 16])
def test_arbitrary_n_follows_formulae(n):
    spec = tiled_cc_spec(cc_e_count=n)
    assert _counts(spec) == (10 * n + 111, 129 * n + 20)
    validate_spec(spec, N_IN)                       # every N is a valid tiled graph


def test_ids_are_unique_and_deterministic():
    a, b = tiled_cc_spec(cc_e_count=8), tiled_cc_spec(cc_e_count=8)
    ids_a = [n['id'] for n in a['nodes']]
    assert len(ids_a) == len(set(ids_a))            # unique node ids
    eids = [e['id'] for e in a['edges']]
    assert len(eids) == len(set(eids))              # unique edge ids
    assert ids_a == [n['id'] for n in b['nodes']]   # deterministic across builds
    assert [e['id'] for e in a['edges']] == [e['id'] for e in b['edges']]


def test_tile_builder_returns_fresh_state():
    # Calling the tile builder twice must not share mutable lists / state.
    _, n1, e1 = build_cortical_column('A', 'L1', 0, 0, n_e=3, has_parent=True)
    _, n2, e2 = build_cortical_column('B', 'L1', 0, 1, n_e=3, has_parent=True)
    assert n1 is not n2 and e1 is not e2
    n1.append({'poison': True})
    assert not any('poison' in n for n in n2)


def test_each_column_composition():
    spec = tiled_cc_spec(cc_e_count=8)
    meta = spec['topology']
    roles = {c['id']: {'E': 0, 'Eor': 0, 'C': 0, 'I': 0} for c in meta['columns']}
    for n in spec['nodes']:
        if n.get('column_role'):
            roles[n['column_id']][n['column_role']] += 1
    for cid, r in roles.items():
        assert r == {'E': 8, 'Eor': 1, 'C': 1, 'I': 1}, (cid, r)
    assert len(meta['columns']) == 10                # 9 L1 + 1 L2


def test_each_patch_has_nine_unique_rgcs_and_no_cross_patch():
    spec = tiled_cc_spec(cc_e_count=8)
    # RGC per patch
    by_patch = {}
    for n in spec['nodes']:
        if n['archetype'] == 'rg_source':
            by_patch.setdefault(n['patch_id'], set()).add(n['pixel'])
    assert len(by_patch) == 9 and all(len(v) == 9 for v in by_patch.values())
    assert sorted(p for s in by_patch.values() for p in s) == list(range(81))
    # each L1 ordinary E gets exactly the nine RGCs of its own patch, none across patches
    rg_pixel = {n['id']: n['pixel'] for n in spec['nodes'] if n['archetype'] == 'rg_source'}
    rg_patch = {n['id']: n['patch_id'] for n in spec['nodes'] if n['archetype'] == 'rg_source'}
    e_patch = {n['id']: n['patch'] for n in spec['nodes'] if n.get('column_role') == 'E'
               and n['layer'] == 'L1'}
    e_afferents = {}
    for e in spec['edges']:
        if e.get('projection') == 'rg_to_column':
            e_afferents.setdefault(e['target'], []).append(e['source'])
    for eid, srcs in e_afferents.items():
        assert len(srcs) == 9 and len(set(srcs)) == 9
        assert all(rg_patch[s] == e_patch[eid] for s in srcs)      # no cross-patch edge


def test_l1_eor_reaches_all_l2_e_and_all_l2e_reach_each_l1_c():
    spec = tiled_cc_spec(cc_e_count=8)
    l2_e = {n['id'] for n in spec['nodes']
            if n.get('column_id') == 'L2c00' and n.get('column_role') == 'E'}
    # each L1 Eor -> every L2 ordinary E
    for pr in range(3):
        for pc in range(3):
            eor = f'L1c{pr}{pc}Eor'
            reached = {e['target'] for e in spec['edges']
                       if e['kind'] == 'feedforward' and e['source'] == eor}
            assert reached == l2_e
    # every L2 ordinary E -> each L1 C apically
    for pr in range(3):
        for pc in range(3):
            c = f'L1c{pr}{pc}C'
            apical_src = {e['source'] for e in spec['edges']
                          if e['kind'] == 'apical_excitation' and e['target'] == c}
            assert apical_src == l2_e


def test_no_eor_is_an_apical_source_and_top_c_is_dormant():
    spec = tiled_cc_spec(cc_e_count=8)
    eor_ids = {n['id'] for n in spec['nodes'] if n.get('column_role') == 'Eor'}
    assert not any(e['source'] in eor_ids for e in spec['edges']
                   if e['kind'] == 'apical_excitation')
    top_c = 'L2c00C'
    assert not any(e['kind'] == 'apical_excitation' and e['target'] == top_c
                   for e in spec['edges'])
    topc_node = next(n for n in spec['nodes'] if n['id'] == top_c)
    assert topc_node['has_parent'] is False
    # valid ONLY because it has no parent
    validate_spec(spec, N_IN)


def test_metadata_survives_validate_and_roundtrip():
    spec = tiled_cc_spec(cc_e_count=8)
    norm = validate_spec(spec, N_IN)
    assert norm['topology']['family'] == TILED_FAMILY
    assert norm['topology']['input_shape'] == {'rows': 9, 'cols': 9}
    e = next(n for n in norm['nodes'] if n.get('column_role') == 'E' and n['layer'] == 'L1')
    for f in ('column_id', 'column_role', 'column_index', 'column_row', 'column_col', 'patch'):
        assert f in e
    rg = next(n for n in norm['nodes'] if n['archetype'] == 'rg_source')
    for f in ('pixel', 'input_row', 'input_col', 'patch_id', 'patch_local_row'):
        assert f in rg
    assert any(ed.get('projection') == 'rg_to_column' for ed in norm['edges'])
    # re-validating the normalized spec is a fixed point (idempotent)
    assert _counts(validate_spec(norm, N_IN)) == (191, 1052)


# --- malformed graphs are rejected ------------------------------------------------
def _mutate_and_expect(mutate):
    spec = tiled_cc_spec(cc_e_count=8)
    mutate(spec)
    with pytest.raises(SpecError):
        validate_spec(spec, N_IN)


def test_missing_internal_edge_rejected():
    def drop(s):
        for i, e in enumerate(s['edges']):
            if e.get('projection') == 'column_i_to_e':
                del s['edges'][i]
                return
    _mutate_and_expect(drop)


def test_cross_column_reset_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1c00I', target='L1c01E0', kind='hard_reset_inhibition', sign=-1)))


def test_lateral_e_to_e_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1c00E0', target='L1c00E1', kind='feedforward')))


def test_apical_into_top_c_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L2c00E0', target='L2c00C', kind='apical_excitation')))


def test_eor_apical_source_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L2c00Eor', target='L1c00C', kind='apical_excitation')))


def test_generic_rg_coincidence_still_requires_apical():
    # A non-tiled coincidence graph keeps the ">= one apical" rule (no dormant exception).
    from backend.network_spec import preset_spec
    spec = preset_spec('rg_coincidence', 9, 8)
    spec['edges'] = [e for e in spec['edges']
                     if not (e['kind'] == 'apical_excitation' and e['target'] == 'L1C0')]
    with pytest.raises(SpecError, match='at least one incoming apical'):
        validate_spec(spec, 9)


# --- embed_patch_pattern helper ---------------------------------------------------
def test_embed_patch_pattern_all_patches():
    row = [0, 0, 0, 1, 1, 1, 0, 0, 0]               # 3x3 center row
    for pr in range(3):
        for pc in range(3):
            vec = embed_patch_pattern((9, 9), (3, 3), (pr, pc), row)
            assert len(vec) == 81 and sum(vec) == 3
            active = {i for i, x in enumerate(vec) if x}
            gr = pr * 3 + 1                          # center local row
            expect = {gr * 9 + (pc * 3 + lc) for lc in range(3)}
            assert active == expect


def test_embed_patch_pattern_validates_bounds():
    with pytest.raises(ValueError):
        embed_patch_pattern((9, 9), (3, 3), (3, 0), [0] * 9)     # patch row out of range
    with pytest.raises(ValueError):
        embed_patch_pattern((9, 9), (3, 3), (0, 0), [0] * 8)     # wrong local length
    with pytest.raises(ValueError):
        embed_patch_pattern((9, 9), (2, 2), (0, 0), [0] * 4)     # patch does not tile
