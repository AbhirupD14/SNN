"""The graph-driven engine: NetworkSpec validation, preset round-trip, and running
an arbitrary custom graph built from the fixed archetype/edge-kind vocabulary.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX
from backend.network_spec import (
    preset_spec, validate_spec, SpecError, ARCHETYPES, EDGE_KINDS,
)


# --------------------------------------------------------------- preset specs
def test_preset_specs_match_engine_counts():
    for name, n in (('pi', 26), ('old', 27), ('rg', 36), ('rg_residual', 52)):
        spec = preset_spec(name, N_PIX, 8)
        assert len(spec['nodes']) == n
        norm = validate_spec(spec, N_PIX)          # presets are valid specs
        assert len(norm['edges']) == len(spec['edges'])


def test_current_spec_round_trips_shape():
    e = SimulationEngine(seed=1, topology='pi')
    spec = e.current_spec()
    assert spec['name'] == 'pi' and spec['is_custom'] is False
    assert len(spec['nodes']) == 26
    # every node carries a resolved position and a valid archetype
    for node in spec['nodes']:
        assert node['archetype'] in ARCHETYPES
        assert len(node['pos']) == 3
    # re-applying the exported spec rebuilds and runs
    e.apply_topology(spec)
    assert e.topology()['params']['is_custom_topology'] is True
    e.set_pattern('row 1')
    for _ in range(50):
        e.step()


# ------------------------------------------------------------------ validation
def _mini_nodes():
    return [
        {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0, 'pos': [-4, 0, 0]},
        {'id': 'C0', 'archetype': 'e_competitor', 'pos': [0, 6, 0]},
        {'id': 'R', 'archetype': 'i_relay', 'pos': [0, 10, 0]},
    ]


def test_validate_rejects_unknown_archetype():
    spec = {'nodes': [{'id': 'X', 'archetype': 'wat'}], 'edges': []}
    with pytest.raises(SpecError):
        validate_spec(spec, N_PIX)


def test_validate_rejects_dangling_edge():
    spec = {'nodes': _mini_nodes(),
            'edges': [{'source': 'S0', 'target': 'GONE', 'kind': 'feedforward'}]}
    with pytest.raises(SpecError):
        validate_spec(spec, N_PIX)


def test_validate_rejects_bad_endpoint_archetype():
    # feedforward must target a competitor, not a relay.
    spec = {'nodes': _mini_nodes(),
            'edges': [{'source': 'S0', 'target': 'R', 'kind': 'feedforward'}]}
    with pytest.raises(SpecError):
        validate_spec(spec, N_PIX)


def test_validate_rejects_duplicate_node_and_pixel():
    dup = _mini_nodes() + [{'id': 'S0', 'archetype': 'e_sensory', 'pixel': 1}]
    with pytest.raises(SpecError):
        validate_spec({'nodes': dup, 'edges': []}, N_PIX)
    clash = [{'id': 'A', 'archetype': 'e_sensory', 'pixel': 0},
             {'id': 'B', 'archetype': 'e_sensory', 'pixel': 0}]
    with pytest.raises(SpecError):
        validate_spec({'nodes': clash, 'edges': []}, N_PIX)


def test_validate_fills_edge_defaults():
    spec = {'nodes': _mini_nodes(),
            'edges': [{'source': 'S0', 'target': 'C0', 'kind': 'feedforward'}]}
    norm = validate_spec(spec, N_PIX)
    e = norm['edges'][0]
    assert e['directed'] is True and e['id'] and e['sign'] == +1


# ------------------------------------------------------------- custom execution
def test_custom_graph_builds_runs_and_learns():
    spec = {
        'name': 'mini',
        'nodes': [
            {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0, 'pos': [-4, 0, 0]},
            {'id': 'S1', 'archetype': 'e_sensory', 'pixel': 1, 'pos': [0, 0, 0]},
            {'id': 'C0', 'archetype': 'e_competitor', 'pos': [-2, 6, 0]},
            {'id': 'C1', 'archetype': 'e_competitor', 'pos': [2, 6, 0]},
            {'id': 'R', 'archetype': 'i_relay', 'pos': [0, 10, 0]},
        ],
        'edges': [
            # C0 has two afferents (both pixels), C1 has one -> C0 is the strong unit.
            {'id': 'f00', 'source': 'S0', 'target': 'C0', 'kind': 'feedforward'},
            {'id': 'f10', 'source': 'S1', 'target': 'C0', 'kind': 'feedforward'},
            {'id': 'f11', 'source': 'S1', 'target': 'C1', 'kind': 'feedforward'},
            {'id': 're0', 'source': 'C0', 'target': 'R', 'kind': 'relay_excitation'},
            {'id': 're1', 'source': 'C1', 'target': 'R', 'kind': 'relay_excitation'},
            {'id': 'inh0', 'source': 'R', 'target': 'C0', 'kind': 'inhibition'},
            {'id': 'inh1', 'source': 'R', 'target': 'C1', 'kind': 'inhibition'},
        ],
    }
    e = SimulationEngine(seed=1, topology='pi')
    e.apply_topology(spec)
    assert e.mode == 'mini' and len(e.topology()['neurons']) == 5
    assert len(e.competitors) == 2 and len(e.sensory) == 2 and len(e.relays) == 1
    e.set_input([1, 1, 0, 0, 0, 0, 0, 0, 0])       # drive pixels 0 and 1 -> C0 (2 afferents)
    won = False
    for _ in range(150):
        d = e.step()
        if d['winner'] == 'C0':
            won = True
    assert won                                          # the generic engine runs the graph
    assert np.any(e.competitors[0].acc_weights > 62)   # C0 learned an afferent up


def test_preset_selection_clears_custom_graph():
    e = SimulationEngine(seed=1, topology='pi')
    e.apply_topology({'name': 'mini', 'nodes': _mini_nodes(),
                      'edges': [{'source': 'S0', 'target': 'C0', 'kind': 'feedforward'}]})
    assert e._custom_spec is not None
    e.apply_config({'topology': 'old'})            # selecting a preset drops the custom graph
    assert e._custom_spec is None
    assert len(e.topology()['neurons']) == 27


def test_bidirectional_edge_valid_only_when_reverse_is_valid():
    # competitor<->competitor feedforward: reverse is also a valid feedforward -> OK.
    ok = {'nodes': [{'id': 'C0', 'archetype': 'e_competitor'},
                    {'id': 'C1', 'archetype': 'e_competitor'}],
          'edges': [{'id': 'e', 'source': 'C0', 'target': 'C1',
                     'kind': 'feedforward', 'directed': False}]}
    assert validate_spec(ok, N_PIX)['edges'][0]['directed'] is False
    # inhibition is inherently one-way (relay->E); bidirectional must be rejected.
    bad = {'nodes': [{'id': 'R', 'archetype': 'i_relay'},
                     {'id': 'C', 'archetype': 'e_competitor'}],
           'edges': [{'id': 'x', 'source': 'R', 'target': 'C',
                      'kind': 'inhibition', 'directed': False}]}
    with pytest.raises(SpecError):
        validate_spec(bad, N_PIX)


def test_bidirectional_edge_delivers_both_directions():
    e = SimulationEngine(seed=1, topology='pi')
    e.apply_topology({
        'name': 'bi',
        'nodes': [{'id': 'S', 'archetype': 'e_sensory', 'pixel': 0},
                  {'id': 'C0', 'archetype': 'e_competitor'},
                  {'id': 'C1', 'archetype': 'e_competitor'}],
        'edges': [{'source': 'S', 'target': 'C0', 'kind': 'feedforward'},
                  {'id': 'lat', 'source': 'C0', 'target': 'C1',
                   'kind': 'feedforward', 'directed': False}]})
    # the bidirectional lateral edge gives each competitor an afferent from the other
    assert 'C1' in e.competitors[0].ff_src        # reverse (C1->C0) delivered
    assert 'C0' in e.competitors[1].ff_src        # forward (C0->C1) delivered
    # the editor still sees ONE bidirectional edge, not two
    lat = [ed for ed in e.current_spec()['edges'] if ed['id'] == 'lat']
    assert len(lat) == 1 and lat[0]['directed'] is False


def test_set_synapse_weight_by_edge_id():
    e = SimulationEngine(seed=1, topology='pi')
    # feedforward weight, clipped to e_weight_cap
    w = e.set_synapse_weight('ff4->0', 321.0)
    assert w == 321.0
    assert e.l2e[0].acc_weights[4] == 321.0
    assert e.set_synapse_weight('ff4->0', 1e9) == e.params['e_weight_cap']   # clipped
    # predictive weight, clipped to pi_w_max
    assert e.set_synapse_weight('pi0->4', 0.5) == 0.5
    assert e.pi[0].w[4] == 0.5
    assert e.set_synapse_weight('pi0->4', 99) == e.params['pi_w_max']        # clipped
    # unknown / non-plastic edges raise
    import pytest as _pt
    with _pt.raises(KeyError):
        e.set_synapse_weight('inh_l2_0', 1.0)        # structural (no learned magnitude)
    with _pt.raises(KeyError):
        e.set_synapse_weight('nope', 1.0)


def test_sensory_pixel_in_meta():
    e = SimulationEngine(seed=1, topology='pi')
    topo = e.topology()
    px = {n['id']: n.get('pixel') for n in topo['neurons'] if n['id'].startswith('L1E')}
    assert px == {f'L1E{i}': i for i in range(9)}    # RF maps afferents by this


def test_edge_kind_vocabulary_is_fixed():
    # The vocabulary is a contract: growing it is a deliberate act, not a side effect.
    # Residual/error behavior deliberately grows the fixed vocabulary with a
    # nonplastic E->ErrorE copy and a paired L2E->SwitchI trace event.
    assert set(EDGE_KINDS) == {'feedforward', 'relay_excitation', 'inhibition',
                               'predictive_inhibition', 'fixed_excitation',
                               'trace_excitation'}
    assert set(ARCHETYPES) == {'rg_source', 'e_sensory', 'e_encoder', 'e_residual',
                               'e_competitor', 'i_relay', 'predictor', 'switch'}


def test_competitor_spike_dispatches_valid_downstream_feedforward_edge():
    """A competitor is an E-class feedforward source; its winning spike must emit."""
    e = SimulationEngine(seed=1, topology='pi', leak_rate=0.0)
    e.apply_topology({
        'name': 'deep',
        'nodes': [{'id': 'S', 'archetype': 'e_sensory', 'pixel': 0},
                  {'id': 'C', 'archetype': 'e_competitor'},
                  {'id': 'E', 'archetype': 'e_encoder'}],
        'edges': [{'id': 's-c', 'source': 'S', 'target': 'C', 'kind': 'feedforward'},
                  {'id': 'c-e', 'source': 'C', 'target': 'E', 'kind': 'feedforward'}]})
    e.clear_input()
    e.competitors[0].V = 1.2 * e.params['e_threshold']
    e.encoders[0].acc_weights[0] = 1.2 * e.params['e_threshold']
    d1 = e.step()
    assert d1['winner'] == 'C' and 'c-e' in d1['emitted']
    d2 = e.step()
    assert next(n for n in d2['neurons'] if n['id'] == 'E')['spiked'] is True
