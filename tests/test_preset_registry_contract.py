"""Contract for the built-in preset registry: the project exposes exactly five current
built-in topologies (rg_coincidence, tiled_cc, tiled_cc_l1_4, tiled_cc_feature_gated,
rg_direct_cc4). The feature-gated variant restored rg_coincidence's feature-specific
inhibition inside the tiled L1 layer; the direct rg_direct_cc4 column is the experimental
dual FE/FES acceptance topology (four ordinary E + one central WTA I). The earlier presets
are unchanged historical controls. The four obsolete presets (pi, old, rg, rg_residual) are
rejected as built-ins, the dashboard surface is the retained controls plus the one new dual
FE/FES toggle, and custom/saved graphs still load.
"""

import pytest

from backend.simulation import SimulationEngine, VALID_TOPOLOGIES, EDITABLE_KEYS
from backend.network_spec import PRESETS, preset_spec
from backend.dashboard_config import CONFIG_SPEC, config_values
from backend import presets as ps

RETAINED = ('rg_coincidence', 'tiled_cc', 'tiled_cc_l1_4', 'tiled_cc_feature_gated',
            'rg_direct_cc4')
OBSOLETE = ('pi', 'old', 'rg', 'rg_residual')


def test_builtin_preset_names_are_exactly_the_retained_set():
    assert tuple(PRESETS) == RETAINED
    assert tuple(VALID_TOPOLOGIES) == RETAINED


def test_dashboard_selector_has_exactly_the_retained_set():
    topo = next(c for c in CONFIG_SPEC if c['key'] == 'topology')
    assert [o['value'] for o in topo['options']] == list(RETAINED)


def test_feature_gated_preset_is_the_only_change_to_the_prior_three():
    # The three historical controls keep their exact node/edge structure; the feature-gated
    # variant is additive (424 nodes / 1932 edges, feature_gated variant metadata).
    assert PRESETS[:3] == ('rg_coincidence', 'tiled_cc', 'tiled_cc_l1_4')
    fg = SimulationEngine(topology='tiled_cc_feature_gated').topology()
    assert len(fg['neurons']) == 424 and len(fg['synapses']) == 1932
    assert fg['tiling']['variant'] == 'feature_gated'


def test_preset_store_lists_exactly_the_builtins(tmp_path, monkeypatch):
    # Point the user-preset dir at an empty temp dir so only built-ins are listed.
    monkeypatch.setattr(ps, 'PRESET_DIR', str(tmp_path))
    builtins = [p['name'] for p in ps.list_presets(9, 8) if p['builtin']]
    assert builtins == list(RETAINED)
    # A user-saved preset is listed alongside the four built-ins (not replacing them).
    ps.save_preset('My Graph', {
        'name': 'My Graph',
        'nodes': [{'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
                  {'id': 'C0', 'archetype': 'e_competitor'}],
        'edges': [{'id': 'e', 'source': 'S0', 'target': 'C0', 'kind': 'feedforward'}]}, 9)
    names = {p['name'] for p in ps.list_presets(9, 8)}
    assert set(RETAINED) <= names and 'My Graph' in names


def test_default_engine_is_rg_coincidence():
    assert SimulationEngine().params['topology'] == 'rg_coincidence'
    assert SimulationEngine().mode == 'rg_coincidence'


@pytest.mark.parametrize('name', OBSOLETE)
def test_obsolete_names_rejected_everywhere(name):
    with pytest.raises(ValueError):
        SimulationEngine(topology=name)                  # constructor
    with pytest.raises(ValueError):
        preset_spec(name, 9, 8)                          # spec builder
    with pytest.raises(KeyError):
        ps.load_spec(name, 9, 8)                         # preset store (not a built-in / saved)
    # apply_config silently ignores an invalid topology value only if it is in EDITABLE_KEYS;
    # 'topology' is editable, so an obsolete value must raise.
    e = SimulationEngine(topology='rg_coincidence')
    with pytest.raises(ValueError):
        e.apply_config({'topology': name})


def test_dashboard_config_keys_are_exactly_the_retained_set():
    keys = {c['key'] for c in CONFIG_SPEC}
    assert keys == {'topology', 'leak_rate', 'refractory_steps', 'eta', 'c_eta',
                    'l2_init_total_frac', 'dual_fe_fes'}
    assert keys == set(config_values(SimulationEngine().params))
    assert EDITABLE_KEYS == keys                         # browser apply surface matches


def test_dual_fe_fes_toggle_is_the_one_new_control():
    dual = [c for c in CONFIG_SPEC if c['key'] == 'dual_fe_fes']
    assert len(dual) == 1 and dual[0]['kind'] == 'toggle'
    assert SimulationEngine().params['dual_fe_fes'] is False   # default off (production rule)


def test_rg_direct_cc4_registered_and_runs_flag_off_and_on():
    for dual in (False, True):
        e = SimulationEngine(topology='rg_direct_cc4', leak_rate=0.0, dual_fe_fes=dual)
        assert e.mode == 'rg_direct_cc4'
        top = e.topology()
        assert len(top['neurons']) == 14 and len(top['synapses']) == 44
        assert top['params']['dual_fe_fes'] is dual
        e.set_pattern('row 1')
        for _ in range(6):
            e.step()                                          # same topology, both flag states


def test_tiled_cc_retains_eight_e_structure():
    cols = SimulationEngine(topology='tiled_cc').topology()['tiling']['columns']
    assert {c['e_count'] for c in cols} == {8}
    assert len(SimulationEngine(topology='tiled_cc').topology()['neurons']) == 191


def test_tiled_cc_l1_4_retains_l1_4_l2_8():
    cols = SimulationEngine(topology='tiled_cc_l1_4').topology()['tiling']['columns']
    assert {c['e_count'] for c in cols if c['layer'] == 'L1'} == {4}
    assert next(c['e_count'] for c in cols if c['layer'] == 'L2') == 8
    assert len(SimulationEngine(topology='tiled_cc_l1_4').topology()['neurons']) == 155


def test_saved_custom_topology_still_loads_and_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, 'PRESET_DIR', str(tmp_path))
    spec = {'name': 'custom', 'nodes': [
        {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
        {'id': 'C0', 'archetype': 'e_competitor'},
        {'id': 'R', 'archetype': 'i_relay'}],
        'edges': [{'id': 'ff', 'source': 'S0', 'target': 'C0', 'kind': 'feedforward'},
                  {'id': 're', 'source': 'C0', 'target': 'R', 'kind': 'relay_excitation'},
                  {'id': 'in', 'source': 'R', 'target': 'C0', 'kind': 'inhibition'}]}
    ps.save_preset('custom', spec, 9)
    loaded = ps.load_spec('custom', 9, 8)
    e = SimulationEngine()
    e.apply_topology(loaded)
    assert e.mode == 'custom' and e._custom_spec is not None
    e.set_input([1, 0, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(10):
        e.step()                                         # a saved custom graph still runs
