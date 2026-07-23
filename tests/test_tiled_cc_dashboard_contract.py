"""Dashboard payload / config contract for tiled_cc (Phase 5). Verifies the topology
and dynamic payloads carry exactly the metadata the frontend reads (input grid + patch
separators, pixel ownership for RGC flashing, column grouping/roles, projection families,
per-column winners), and that the selector/control/preset surface exposes tiled_cc. This
is the headless stand-in for the browser smoke checks (no DOM available here). Legacy
payload contracts must be unchanged.
"""

from backend.simulation import SimulationEngine
from backend.serializer import full_state, topology_message, dynamic_message
from backend.dashboard_config import CONFIG_SPEC, config_values
from backend import presets as ps


def test_input_grid_and_pixel_ownership_metadata():
    topo = SimulationEngine(seed=1, topology='tiled_cc').topology()
    # 9x9 input grid dims for the input panel
    assert topo['grid'] == {'rows': 9, 'cols': 9}
    assert topo['tiling']['patch_shape'] == {'rows': 3, 'cols': 3}
    # every input pixel is owned by exactly one flashing input sink (controls.js RGC flash)
    owners = {}
    for n in topo['neurons']:
        if n.get('pixel') is not None and n.get('owns_input'):
            owners.setdefault(n['pixel'], []).append(n['id'])
    assert sorted(owners) == list(range(81))
    assert all(len(v) == 1 for v in owners.values())


def test_legacy_input_grid_unchanged():
    topo = SimulationEngine(seed=1, topology='rg_coincidence').topology()
    assert topo['grid'] == {'rows': 3, 'cols': 3}       # controls.js renders 9 cells
    assert 'tiling' not in topo
    owners = [n for n in topo['neurons'] if n.get('owns_input')]
    assert len(owners) == 9


def test_neuron_and_edge_metadata_for_grouping():
    topo = SimulationEngine(seed=1, topology='tiled_cc').topology()
    roles = {n.get('column_role') for n in topo['neurons'] if n.get('column_role')}
    assert roles == {'E', 'Eor', 'C', 'I'}              # renderer/inspector grouping
    # every column node carries its column id + tile coords
    col_nodes = [n for n in topo['neurons'] if n.get('column_role')]
    assert all({'column_id', 'column_row', 'column_col'} <= set(n) for n in col_nodes)
    # projection families for edge filtering/labels
    projections = {s.get('projection') for s in topo['synapses'] if s.get('projection')}
    assert {'rg_to_column', 'column_e_to_eor', 'column_e_to_i', 'column_i_to_e',
            'column_eor_to_c_basal', 'column_c_to_i', 'column_to_column_ff',
            'column_to_column_apical'} <= projections
    # top C carries a dormant marker (has_parent False) for the inspector
    topc = next(n for n in topo['neurons']
                if n.get('column_id') == 'L2c00' and n.get('column_role') == 'C')
    assert topc['has_parent'] is False


def test_weights_view_targets_are_selectable():
    # weights.js selects role == 'competitor' plastic cells: tiled ordinary E and Eor both
    # qualify (same archetype), so per-target weight charts work for the whole hierarchy.
    topo = SimulationEngine(seed=1, topology='tiled_cc').topology()
    comp = [n for n in topo['neurons'] if n['role'] == 'competitor']
    roles = {n['column_role'] for n in comp}
    assert roles == {'E', 'Eor'}
    assert len(comp) == 10 * (8 + 1)                    # N E + 1 Eor per column


def test_dynamic_payload_has_column_winners_and_legacy_keys():
    e = SimulationEngine(seed=1, topology='tiled_cc', leak_rate=0.0)
    e.set_pattern('row 1')
    seen = {}
    d = None
    for _ in range(40):
        d = e.step()
        assert 'column_winners' in d                     # key present every frame
        seen.update(d['column_winners'])
    assert 'L1c11' in seen                                # active column won on some boundary
    # legacy dynamic keys still present
    assert {'timestep', 'neurons', 'winner', 'input', 'stats', 'hard_reset_events'} <= set(d)
    assert len(d['input']) == 81


def test_config_surface_exposes_tiled_presets_without_cc_e_count():
    # Both tiled presets are selectable, but cc_e_count is no longer a dashboard control
    # (the two tiled presets fix their own population sizes).
    topo_ctrl = next(c for c in CONFIG_SPEC if c['key'] == 'topology')
    opts = {o['value'] for o in topo_ctrl['options']}
    assert {'tiled_cc', 'tiled_cc_l1_4'} <= opts
    keys = {c['key'] for c in CONFIG_SPEC}
    assert 'cc_e_count' not in keys
    assert 'cc_e_count' not in config_values(SimulationEngine(seed=1).params)


def test_preset_store_lists_and_loads_tiled_cc():
    lst = ps.list_presets(9, 8)
    tc = next(p for p in lst if p['name'] == 'tiled_cc')
    assert tc['builtin'] and tc['nodes'] == 191 and tc['edges'] == 1052
    spec = ps.load_spec('tiled_cc', 9, 8)               # loads with real positions + meta
    assert spec['topology']['family'] == 'tiled_cortical_columns'
    assert all(n.get('pos') for n in spec['nodes'])


def test_full_state_and_message_envelopes_well_formed_for_tiled():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    fs = full_state(e, running=False, speed=12.0)
    assert set(fs) == {'topology', 'dynamic'}
    assert fs['topology']['tiling']['cc_e_count'] == 8
    assert topology_message(e)['type'] == 'topology'
    assert dynamic_message(e, True, 5.0)['data']['running'] is True


def test_set_patch_updates_selected_patch_in_topology():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    assert e.topology()['tiling']['selected_patch'] == [1, 1]
    e.set_patch(0, 2)
    assert e.topology()['tiling']['selected_patch'] == [0, 2]
    # switching patch re-embeds the active pattern
    e.set_pattern('row 1')
    e.set_patch(2, 0)
    assert e.topology()['tiling']['selected_patch'] == [2, 0]
