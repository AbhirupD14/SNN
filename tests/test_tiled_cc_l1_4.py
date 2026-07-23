"""Focused tests for the built-in ``tiled_cc_l1_4`` preset: the tiled cortical-column
hierarchy with a shallower L1 bank (four ordinary competing E per L1 column, eight in
L2). Covers exact node/edge counts, per-column E composition, valid tiled metadata and
connectivity, engine construction/stepping, dashboard preset availability, and that the
existing ``tiled_cc`` preset is unchanged.
"""

import collections

from backend.network_spec import (
    preset_spec, tiled_cc_spec, validate_spec, PRESETS, TILED_PRESETS, TILED_FAMILY,
)
from backend.simulation import SimulationEngine, VALID_TOPOLOGIES
from backend.serializer import full_state
from backend.dashboard_config import CONFIG_SPEC
from backend import presets as ps

N_IN = 81


def _spec():
    return preset_spec('tiled_cc_l1_4', N_IN, 8)


def _ord_e_by_column(nodes):
    byc = collections.Counter()
    for n in nodes:
        if n.get('column_role') == 'E':
            byc[n['column_id']] += 1
    return byc


# --- registry ---------------------------------------------------------------

def test_registered_as_builtin_and_tiled():
    assert 'tiled_cc_l1_4' in PRESETS
    assert 'tiled_cc_l1_4' in TILED_PRESETS
    assert 'tiled_cc_l1_4' in VALID_TOPOLOGIES


# --- exact dimensions -------------------------------------------------------

def test_exact_node_and_edge_counts():
    spec = _spec()
    assert spec['name'] == 'tiled_cc_l1_4'
    assert len(spec['nodes']) == 155
    assert len(spec['edges']) == 620
    # counts are a contract re-checked after validation/normalization.
    norm = validate_spec(spec, N_IN)
    assert len(norm['nodes']) == 155
    assert len(norm['edges']) == 620


def test_81_pixel_input_surface():
    rgc = [n for n in _spec()['nodes'] if n.get('layer') == 'RGC']
    assert len(rgc) == 81
    assert {n['pixel'] for n in rgc} == set(range(81))


def test_nine_l1_columns_and_one_l2():
    cols = _spec()['topology']['columns']
    l1 = [c for c in cols if c['layer'] == 'L1']
    l2 = [c for c in cols if c['layer'] == 'L2']
    assert len(l1) == 9
    assert len(l2) == 1
    assert l2[0]['id'] == 'L2c00'


# --- per-column E composition ----------------------------------------------

def test_four_ordinary_e_in_every_l1_column():
    nodes = _spec()['nodes']
    byc = _ord_e_by_column(nodes)
    l1cols = {k for k in byc if k.startswith('L1')}
    assert len(l1cols) == 9
    assert all(byc[c] == 4 for c in l1cols)
    # per-column metadata agrees.
    l1meta = [c for c in _spec()['topology']['columns'] if c['layer'] == 'L1']
    assert {c['e_count'] for c in l1meta} == {4}


def test_eight_ordinary_e_in_l2():
    byc = _ord_e_by_column(_spec()['nodes'])
    assert byc['L2c00'] == 8
    l2meta = next(c for c in _spec()['topology']['columns'] if c['layer'] == 'L2')
    assert l2meta['e_count'] == 8


def test_each_column_has_one_eor_c_and_i():
    roles = collections.defaultdict(collections.Counter)
    for n in _spec()['nodes']:
        if n.get('column_id'):
            roles[n['column_id']][n['column_role']] += 1
    assert len(roles) == 10                            # 9 L1 + 1 L2
    for cid, rc in roles.items():
        assert rc['Eor'] == 1 and rc['C'] == 1 and rc['I'] == 1
        assert rc['E'] == (8 if cid.startswith('L2') else 4)


# --- metadata + connectivity validity --------------------------------------

def test_valid_tiled_metadata_and_connectivity():
    spec = validate_spec(_spec(), N_IN)
    meta = spec['topology']
    assert meta['family'] == TILED_FAMILY
    assert meta['input_shape'] == {'rows': 9, 'cols': 9}
    assert meta['patch_shape'] == {'rows': 3, 'cols': 3}
    assert meta['grid_shape'] == {'rows': 3, 'cols': 3}
    # all intra/inter-column projection families are present.
    projections = {s.get('projection') for s in spec['synapses'] if s.get('projection')} \
        if 'synapses' in spec else {s.get('projection') for s in spec['edges']}
    assert {'rg_to_column', 'column_e_to_eor', 'column_e_to_i', 'column_i_to_e',
            'column_eor_to_c_basal', 'column_c_to_i', 'column_to_column_ff',
            'column_to_column_apical'} <= projections


# --- engine construction + stepping ----------------------------------------

def test_engine_builds_and_steps():
    e = SimulationEngine(seed=1, topology='tiled_cc_l1_4')
    topo = e.topology()
    assert len(topo['neurons']) == 155
    assert len(topo['synapses']) == 620
    assert topo['tiling']['family'] == TILED_FAMILY
    e.set_pattern('row 1')
    d = None
    for _ in range(20):
        d = e.step()
        assert 'column_winners' in d
    assert d['timestep'] == 20
    assert len(d['input']) == 81


def test_full_state_envelope_reports_l1_4_l2_8():
    e = SimulationEngine(seed=1, topology='tiled_cc_l1_4')
    fs = full_state(e, running=False, speed=12.0)
    assert set(fs) == {'topology', 'dynamic'}
    cols = fs['topology']['tiling']['columns']
    assert {c['e_count'] for c in cols if c['layer'] == 'L1'} == {4}
    assert next(c['e_count'] for c in cols if c['layer'] == 'L2') == 8


# --- dashboard availability -------------------------------------------------

def test_dashboard_selector_and_preset_store_expose_l1_4():
    topo_ctrl = next(c for c in CONFIG_SPEC if c['key'] == 'topology')
    assert any(o['value'] == 'tiled_cc_l1_4' for o in topo_ctrl['options'])
    lst = ps.list_presets(9, 8)
    entry = next(p for p in lst if p['name'] == 'tiled_cc_l1_4')
    assert entry['builtin'] and entry['nodes'] == 155 and entry['edges'] == 620
    spec = ps.load_spec('tiled_cc_l1_4', 9, 8)         # loads with real positions
    assert spec['topology']['family'] == TILED_FAMILY
    assert all(n.get('pos') for n in spec['nodes'])


# --- no regression to tiled_cc ---------------------------------------------

def test_existing_tiled_cc_unchanged():
    spec = tiled_cc_spec()                              # default uniform N=8
    assert spec['name'] == 'tiled_cc'
    assert len(spec['nodes']) == 191 and len(spec['edges']) == 1052
    assert spec['topology']['cc_e_count'] == 8
    byc = _ord_e_by_column(spec['nodes'])
    assert set(byc.values()) == {8}                    # every column still 8 ordinary E
    assert {c['e_count'] for c in spec['topology']['columns']} == {8}
