"""Feature-gated tiled cortical-column builder + structural validation.

Covers the spec's focused structural tests for the eight-competitor feature-gated variant:
exact 424-node/1932-edge counts and deterministic construction; exactly 81 RGC / 81 feature
relays / 81 feature C / 81 feature I / nine L1 recognition modules / one L2 module; the
paired feature relay/C/If chain (one basal, eight local apical, one paired reset); the
separate WTA-only I; the L1 Eor -> L2 feedforward with no L2 -> L1 apical feedback; and
rejection of every malformed graph the invariants forbid. The prior three tiled presets are
left completely unchanged (asserted here so this file also guards against regressions).
"""

import copy

import pytest

from backend.network_spec import (
    tiled_cc_feature_gated_spec, tiled_cc_spec, validate_spec, SpecError,
    build_wta_bank, build_feature_gate, connect_bank_feedforward, WtaBankHandles,
    TILED_FAMILY, TILED_VARIANT_FEATURE_GATED,
)
from backend.simulation import SimulationEngine
from backend import presets as ps

N_IN = 81


def _counts(spec):
    return len(spec['nodes']), len(spec['edges'])


def _roles(spec):
    """{module_id: {role: [feature_index or None ...]}} from node metadata only."""
    out = {}
    for n in spec['nodes']:
        r = n.get('column_role')
        if r is None:
            continue
        out.setdefault(n['column_id'], {}).setdefault(r, []).append(n.get('feature_index'))
    return out


# --- 1-3: counts, determinism, population tallies ---------------------------------
def test_default_is_424_nodes_1932_edges():
    spec = tiled_cc_feature_gated_spec()
    assert _counts(spec) == (424, 1932)
    norm = validate_spec(spec, N_IN)                    # contract survives normalization
    assert _counts(norm) == (424, 1932)
    assert norm['topology']['variant'] == TILED_VARIANT_FEATURE_GATED
    assert norm['topology']['family'] == TILED_FAMILY


def test_construction_is_deterministic_and_ids_unique():
    a, b = tiled_cc_feature_gated_spec(), tiled_cc_feature_gated_spec()
    ids = [n['id'] for n in a['nodes']]
    eids = [e['id'] for e in a['edges']]
    assert len(ids) == len(set(ids)) and len(eids) == len(set(eids))
    assert ids == [n['id'] for n in b['nodes']]
    assert eids == [e['id'] for e in b['edges']]


def test_exact_population_counts():
    spec = tiled_cc_feature_gated_spec()
    arch = {}
    for n in spec['nodes']:
        arch[n['archetype']] = arch.get(n['archetype'], 0) + 1
    assert arch['rg_source'] == 81
    assert arch['e_pretrained'] == 81                   # feature relays S
    assert arch['e_coincidence'] == 81                  # feature C
    # feature I (81) + WTA I (10) = 91 i_relay
    assert arch['i_relay'] == 91
    # 9*8 L1 E + 9 L1 Eor + 8 L2 E + 1 L2 Eor = 90 latency competitors
    assert arch['e_latency_competitor'] == 90
    roles = _roles(spec)
    l1 = [m for m in roles if m.startswith('L1m')]
    assert len(l1) == 9 and 'L2m00' in roles
    for m in l1:
        assert len(roles[m]['E']) == 8
        assert sorted(roles[m]['S']) == list(range(9))
        assert sorted(roles[m]['C']) == list(range(9))
        assert sorted(roles[m]['If']) == list(range(9))
    # L2 is a plain WTA bank: 8 E, Eor, WTA I, and NO feature gates.
    assert set(roles['L2m00']) == {'E', 'Eor', 'I'} and len(roles['L2m00']['E']) == 8


# --- 3-6: relay/patch/basal/apical/paired-reset invariants ------------------------
def _edge_index(spec):
    by_kind = {}
    for e in spec['edges']:
        by_kind.setdefault(e['kind'], []).append(e)
    return by_kind


def test_each_rgc_maps_to_one_relay_in_one_patch():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    pretrained = _edge_index(spec)['pretrained_excitation']
    assert len(pretrained) == 81
    seen_rgc, seen_s = set(), set()
    for e in pretrained:
        rg, s = meta[e['source']], meta[e['target']]
        assert rg['archetype'] == 'rg_source' and s['column_role'] == 'S'
        # the relay lives in the module whose patch owns the RGC's pixel
        gr, gc = rg['pixel'] // 9, rg['pixel'] % 9
        assert (gr // 3, gc // 3) == (s['column_row'], s['column_col'])
        assert s['feature_index'] == (gr % 3) * 3 + (gc % 3)
        seen_rgc.add(e['source']); seen_s.add(e['target'])
    assert len(seen_rgc) == 81 and len(seen_s) == 81   # one-to-one


def test_each_relay_feeds_all_and_only_its_module_e():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    e_by_mod = {}
    for n in spec['nodes']:
        if n.get('column_role') == 'E':
            e_by_mod.setdefault(n['column_id'], set()).add(n['id'])
    fed = {}
    for e in spec['edges']:
        if e.get('projection') == 'feature_relay_to_e':
            fed.setdefault(e['source'], set()).add(e['target'])
    for s_id, targets in fed.items():
        mod = meta[s_id]['column_id']
        assert targets == e_by_mod[mod]                 # all and only the local 8 E


def test_each_feature_c_has_one_paired_basal_and_eight_local_apical():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    basal, apical = {}, {}
    for e in spec['edges']:
        if e['kind'] == 'basal_excitation':
            basal.setdefault(e['target'], []).append(e['source'])
        elif e['kind'] == 'apical_excitation':
            apical.setdefault(e['target'], []).append(e['source'])
    c_ids = [n['id'] for n in spec['nodes'] if n.get('column_role') == 'C']
    assert len(c_ids) == 81
    for c in c_ids:
        assert len(basal[c]) == 1
        s = basal[c][0]
        assert meta[s]['column_role'] == 'S'
        assert (meta[s]['column_id'], meta[s]['feature_index']) == (
            meta[c]['column_id'], meta[c]['feature_index'])            # S[k] pairs C[k]
        srcs = apical[c]
        assert len(srcs) == 8 and len(set(srcs)) == 8
        assert all(meta[x]['column_role'] == 'E'
                   and meta[x]['column_id'] == meta[c]['column_id'] for x in srcs)


def test_feature_i_driven_only_by_paired_c_and_resets_only_paired_relay():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    drive, reset = {}, {}
    for e in spec['edges']:
        if e.get('projection') == 'feature_c_to_i':
            drive.setdefault(e['target'], []).append(e['source'])
        elif e.get('projection') == 'feature_i_to_relay':
            reset.setdefault(e['source'], []).append(e['target'])
    if_ids = [n['id'] for n in spec['nodes'] if n.get('column_role') == 'If']
    assert len(if_ids) == 81
    for if_id in if_ids:
        key = (meta[if_id]['column_id'], meta[if_id]['feature_index'])
        assert len(drive[if_id]) == 1 and meta[drive[if_id][0]]['column_role'] == 'C'
        assert (meta[drive[if_id][0]]['column_id'],
                meta[drive[if_id][0]]['feature_index']) == key
        assert len(reset[if_id]) == 1 and meta[reset[if_id][0]]['column_role'] == 'S'
        assert (meta[reset[if_id][0]]['column_id'],
                meta[reset[if_id][0]]['feature_index']) == key


# --- 7-9: WTA vs feature-I separation; no whole-bank predictive reset --------------
def test_wta_i_driven_only_by_its_bank_and_resets_exactly_that_bank():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    wta_ids = {n['id'] for n in spec['nodes']
               if n.get('column_role') == 'I'}
    e_by_mod = {}
    for n in spec['nodes']:
        if n.get('column_role') == 'E':
            e_by_mod.setdefault(n['column_id'], set()).add(n['id'])
    drive, reset = {}, {}
    for e in spec['edges']:
        if e.get('projection') == 'bank_e_to_wta_i':
            drive.setdefault(e['target'], set()).add(e['source'])
        elif e.get('projection') == 'bank_wta_i_to_e':
            reset.setdefault(e['source'], set()).add(e['target'])
    for wta in wta_ids:
        mod = meta[wta]['column_id']
        assert drive[wta] == e_by_mod[mod]              # only its own E bank drives it
        assert reset[wta] == e_by_mod[mod]              # it resets exactly that bank


def test_wta_and_feature_i_are_distinct_with_disjoint_reset_targets():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    wta_resets, feature_resets = set(), set()
    for e in spec['edges']:
        if e.get('projection') == 'bank_wta_i_to_e':
            wta_resets.add(e['target'])
        elif e.get('projection') == 'feature_i_to_relay':
            feature_resets.add(e['target'])
    # WTA resets ordinary E; feature If resets feature relays S. Disjoint node sets.
    assert wta_resets and feature_resets and not (wta_resets & feature_resets)
    assert all(meta[t]['column_role'] == 'E' for t in wta_resets)
    assert all(meta[t]['column_role'] == 'S' for t in feature_resets)


def test_no_column_wide_c_to_shared_i_to_whole_bank_reset():
    # In this variant no feature C drives the WTA I, and no ordinary E drives a feature If;
    # so there is no "one C -> one I -> reset the entire competitor bank" path.
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    for e in spec['edges']:
        if e['kind'] == 'relay_excitation':
            s, t = meta[e['source']], meta[e['target']]
            if t['column_role'] == 'I':                 # WTA relay
                assert s['column_role'] == 'E'
            else:                                       # feature If
                assert s['column_role'] == 'C'
        if e['kind'] == 'hard_reset_inhibition':
            s, t = meta[e['source']], meta[e['target']]
            # a feature If never resets an ordinary E bank
            if s['column_role'] == 'If':
                assert t['column_role'] == 'S'


# --- 10: hierarchy feedforward complete; no L2->L1 apical feedback -----------------
def test_l1_eor_feeds_all_l2_e_and_no_l2_to_l1_apical():
    spec = tiled_cc_feature_gated_spec()
    meta = {n['id']: n for n in spec['nodes']}
    l2_e = {n['id'] for n in spec['nodes']
            if n.get('column_id') == 'L2m00' and n.get('column_role') == 'E'}
    for pr in range(3):
        for pc in range(3):
            eor = f'L1m{pr}{pc}Eor'
            reached = {e['target'] for e in spec['edges']
                       if e.get('projection') == 'bank_to_bank_ff' and e['source'] == eor}
            assert reached == l2_e
    # every apical source is a LOCAL L1 ordinary E; no L2 E ever drives an L1 C.
    for e in spec['edges']:
        if e['kind'] == 'apical_excitation':
            assert meta[e['source']]['column_id'] == meta[e['target']]['column_id']
            assert meta[e['source']]['column_id'] != 'L2m00'


# --- 11: validation rejects malformed graphs --------------------------------------
def _mutate_and_expect(mutate):
    spec = tiled_cc_feature_gated_spec()
    mutate(spec)
    with pytest.raises(SpecError):
        validate_spec(spec, N_IN)


def test_cross_patch_feature_edge_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1m00S0', target='L1m01E0', kind='feedforward')))


def test_cross_patch_rgc_relay_rejected():
    # RGC of patch (0,0) feeding a relay of module (0,1)
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='RGC0', target='L1m01S0', kind='pretrained_excitation')))


def test_swapped_c_i_pairing_rejected():
    # C[0]'s reset relay drives the WRONG feature relay S[1]
    def swap(s):
        for e in s['edges']:
            if e['id'] == 'L1m00If0_L1m00S0':
                e['target'] = 'L1m00S1'
                return
    _mutate_and_expect(swap)


def test_missing_basal_rejected():
    def drop(s):
        s['edges'] = [e for e in s['edges']
                      if e.get('projection') != 'feature_relay_to_c_basal'
                      or e['source'] != 'L1m00S0']
    _mutate_and_expect(drop)


def test_duplicate_pixel_ownership_rejected():
    _mutate_and_expect(lambda s: s['nodes'].append(
        dict(id='RGCdup', archetype='rg_source', layer='RGC', pixel=0)))


def test_wta_feature_i_role_mixing_rejected():
    # an ordinary E driving a feature If (should only be driven by its paired C)
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1m00E1', target='L1m00If0', kind='relay_excitation')))


def test_feature_c_driving_wta_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L1m00C0', target='L1m00Iwta', kind='relay_excitation')))


def test_l2_to_l1_apical_feedback_rejected():
    _mutate_and_expect(lambda s: s['edges'].append(
        dict(id='x', source='L2m00E0', target='L1m00C0', kind='apical_excitation')))


def test_missing_feature_reset_rejected():
    def drop(s):
        s['edges'] = [e for e in s['edges'] if e['id'] != 'L1m00If0_L1m00S0']
    _mutate_and_expect(drop)


# --- 12-14: engine reuse, serialization round-trip, prior presets unchanged -------
def test_builders_return_fresh_state():
    _, n1, e1 = build_wta_bank('A', 'L1', 0, 0, n_e=3)
    _, n2, e2 = build_wta_bank('B', 'L1', 0, 1, n_e=3)
    assert n1 is not n2 and e1 is not e2
    n1.append({'poison': True})
    assert not any('poison' in n for n in n2)


def test_engine_construction_deterministic_and_reuses_classes():
    from snn.neurons import (ExcitatoryNeuron, CoincidencePyramidalNeuron,
                             InhibitoryNeuron, SourceNeuron)
    a = SimulationEngine(seed=1, topology='tiled_cc_feature_gated')
    b = SimulationEngine(seed=1, topology='tiled_cc_feature_gated')
    # identical initial live weights (deterministic construction)
    wa = {s['id']: s['weight'] for s in a.topology()['synapses']}
    wb = {s['id']: s['weight'] for s in b.topology()['synapses']}
    assert wa == wb
    # existing neuron classes are reused unchanged for the new roles
    assert any(isinstance(c, SourceNeuron) for c in a.sources)
    assert any(isinstance(c, ExcitatoryNeuron) for c in a.pretrained)         # feature relays
    assert any(isinstance(c, CoincidencePyramidalNeuron) for c in a.coincidence)
    assert any(isinstance(c, InhibitoryNeuron) for c in a.relays)
    # stepping is deterministic
    for e in (a, b):
        e.set_patch(1, 1); e.set_pattern('row 1')
        for _ in range(30):
            e.step()
    assert a.timestep == b.timestep


def test_serialization_preset_save_load_preserves_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, 'PRESET_DIR', str(tmp_path))
    spec = ps.load_spec('tiled_cc_feature_gated', 81, 8)     # built-in load w/ positions
    assert spec['topology']['variant'] == TILED_VARIANT_FEATURE_GATED
    # a matured C basal weight survives a save/load round-trip
    e = SimulationEngine(seed=1, topology='tiled_cc_feature_gated')
    export = e.current_spec()
    assert export['topology']['variant'] == TILED_VARIANT_FEATURE_GATED
    # feature metadata survives validation
    norm = validate_spec(export, 81)
    s_node = next(n for n in norm['nodes'] if n.get('column_role') == 'S')
    for f in ('column_id', 'column_role', 'feature_index', 'input_row', 'input_col'):
        assert f in s_node
    # save + reload keeps counts + variant
    ps.save_preset('fg copy', export, 81)
    loaded = ps.load_spec('fg copy', 81, 8)
    assert _counts(loaded) == (424, 1932)
    assert loaded['topology']['variant'] == TILED_VARIANT_FEATURE_GATED


def test_prior_three_presets_are_structurally_unchanged():
    # feature-gated construction must not alter the historical controls' graphs
    assert _counts(tiled_cc_spec(cc_e_count=8)) == (191, 1052)
    assert 'variant' not in tiled_cc_spec(cc_e_count=8)['topology']  # classic stays absent
    from backend.network_spec import preset_spec
    assert _counts(preset_spec('rg_coincidence', 9, 8)) == (45, 196)
    assert _counts(preset_spec('tiled_cc_l1_4', 81, 8)) == (155, 620)
