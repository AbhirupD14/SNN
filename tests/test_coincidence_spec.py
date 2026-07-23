"""Phase 3 — coincidence graph vocabulary, validation, and metadata-driven
construction. Covers valid/invalid minimal graphs, deterministic construction of a
synthetic C + latency-WTA fixture, and the resolved C weight scale. No event-loop
execution here (that is Phase 4); these tests only validate + construct.
"""

import numpy as np
import pytest

from backend.network_spec import (
    validate_spec, SpecError, ARCHETYPES, EDGE_KINDS, EVENT_RESOLVED_ARCHETYPES,
)
from backend.simulation import SimulationEngine


# ----------------------------------------------------------------- fixtures
def _synth_spec():
    """RG->L1E(pretrained)->{basal->L1C, ff->L2E0/1}; L2E apical->L1C; L2E->L2I hard
    reset; L1C->L1I->L1E hard reset. A minimal event-resolved C + latency-WTA graph."""
    return {
        'name': 'synth_coin',
        'nodes': [
            {'id': 'RG0', 'archetype': 'rg_source', 'pixel': 0},
            {'id': 'L1E0', 'archetype': 'e_pretrained', 'layer': 'L1'},
            {'id': 'L1C0', 'archetype': 'e_coincidence', 'layer': 'L1'},
            {'id': 'L1I0', 'archetype': 'i_relay', 'layer': 'L1'},
            {'id': 'L2E0', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
            {'id': 'L2E1', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
            {'id': 'L2I', 'archetype': 'i_relay', 'layer': 'L2'},
        ],
        'edges': [
            {'id': 'pt0', 'source': 'RG0', 'target': 'L1E0', 'kind': 'pretrained_excitation'},
            {'id': 'b0', 'source': 'L1E0', 'target': 'L1C0', 'kind': 'basal_excitation'},
            {'id': 'ff0', 'source': 'L1E0', 'target': 'L2E0', 'kind': 'feedforward'},
            {'id': 'ff1', 'source': 'L1E0', 'target': 'L2E1', 'kind': 'feedforward'},
            {'id': 'ap0', 'source': 'L2E0', 'target': 'L1C0', 'kind': 'apical_excitation'},
            {'id': 'ap1', 'source': 'L2E1', 'target': 'L1C0', 'kind': 'apical_excitation'},
            {'id': 're_l1i', 'source': 'L1C0', 'target': 'L1I0', 'kind': 'relay_excitation'},
            {'id': 'hr_l1', 'source': 'L1I0', 'target': 'L1E0',
             'kind': 'hard_reset_inhibition', 'sign': -1},
            {'id': 're0', 'source': 'L2E0', 'target': 'L2I', 'kind': 'relay_excitation'},
            {'id': 're1', 'source': 'L2E1', 'target': 'L2I', 'kind': 'relay_excitation'},
            {'id': 'hr20', 'source': 'L2I', 'target': 'L2E0',
             'kind': 'hard_reset_inhibition', 'sign': -1},
            {'id': 'hr21', 'source': 'L2I', 'target': 'L2E1',
             'kind': 'hard_reset_inhibition', 'sign': -1},
        ],
    }


def _build(spec, **overrides):
    e = SimulationEngine(seed=1, leak_rate=0.03, **overrides)
    e.apply_topology(spec)
    return e


# ----------------------------------------------------------- vocabulary shape
def test_new_archetypes_and_kinds_present_and_flagged():
    for a in ('e_pretrained', 'e_coincidence', 'e_latency_competitor'):
        assert ARCHETYPES[a]['cls'] == 'E'
        assert ARCHETYPES[a]['event_resolved'] is True
    assert set(EVENT_RESOLVED_ARCHETYPES) == {
        'e_pretrained', 'e_coincidence', 'e_latency_competitor'}
    for k in ('pretrained_excitation', 'basal_excitation', 'apical_excitation',
              'hard_reset_inhibition'):
        assert k in EDGE_KINDS
    assert EDGE_KINDS['basal_excitation']['plastic'] is True
    assert EDGE_KINDS['apical_excitation']['plastic'] is False
    assert EDGE_KINDS['hard_reset_inhibition']['sign'] == -1


# ------------------------------------------------------------ valid minimal graph
def test_valid_minimal_graph_validates_and_builds():
    norm = validate_spec(_synth_spec(), 9)
    assert norm['name'] == 'synth_coin'
    e = _build(_synth_spec())
    assert e.event_resolved is True
    assert [c.id for c in e.coincidence] == ['L1C0']
    assert [c.id for c in e.latency_competitors] == ['L2E0', 'L2E1']
    assert e.competitors == []                       # NOT in the legacy WTA list
    assert [c.id for c in e.pretrained] == ['L1E0']


def test_validate_is_idempotent_round_trip():
    a = validate_spec(_synth_spec(), 9)
    b = validate_spec(a, 9)
    assert a['nodes'] == b['nodes'] and a['edges'] == b['edges']


def test_construction_is_deterministic():
    e1 = _build(_synth_spec())
    e2 = _build(_synth_spec())
    assert e1.order == e2.order
    assert e1.coincidence[0].basal_weight == pytest.approx(e2.coincidence[0].basal_weight)
    w1 = np.concatenate([c.acc_weights for c in e1.latency_competitors])
    w2 = np.concatenate([c.acc_weights for c in e2.latency_competitors])
    assert np.array_equal(w1, w2)


# ---------------------------------------------------- synthetic fixture details
def test_synthetic_fixture_c_dendrites_and_weights():
    e = _build(_synth_spec())
    c = e.coincidence[0]
    assert c.basal.source_ids == ['L1E0']
    assert c.apical.source_ids == ['L2E0', 'L2E1']
    assert c.apical.weights is None                  # apical is unweighted
    cpar = e._resolve_coincidence_params()
    assert c.basal_weight == pytest.approx(cpar['c_init'])
    assert c.w_max == pytest.approx(cpar['c_max'])
    assert c.eta_c == pytest.approx(cpar['c_eta'])
    assert cpar['c_max'] >= cpar['w1']               # conservative budget headroom
    # C intrinsic parity with a latency competitor's membrane.
    assert c.threshold == e.latency_competitors[0].threshold
    assert c.g_L == pytest.approx(e.latency_competitors[0].g_L)


def test_synthetic_fixture_adjacency_and_refs():
    e = _build(_synth_spec())
    assert e._pretrained_out == {'RG0': [('L1E0', 'pt0')]}
    assert e._basal_out == {'L1E0': [('L1C0', 'b0')]}
    assert e._apical_out == {'L2E0': [('L1C0', 'ap0')], 'L2E1': [('L1C0', 'ap1')]}
    assert e._hardreset_out == {'L1I0': [('L1E0', 'hr_l1')],
                                'L2I': [('L2E0', 'hr20'), ('L2E1', 'hr21')]}
    assert e._basal_weight_ref['b0'][0] is e.coincidence[0]
    assert e._q_pretrained > 0.0


def test_basal_distance_influence_bounded():
    e = _build(_synth_spec())
    phi = float(e.coincidence[0].basal.distance_factors[0])
    assert 0.0 < phi <= 1.0                          # normalized inverse-square, safe


# ------------------------------------------------------------- invalid graphs
def _mutate(fn):
    spec = _synth_spec()
    fn(spec)
    return spec


def test_missing_basal_edge_rejected():
    def drop(spec):
        spec['edges'] = [e for e in spec['edges'] if e['id'] != 'b0']
    with pytest.raises(SpecError, match='exactly one incoming basal'):
        validate_spec(_mutate(drop), 9)


def test_multiple_basal_edges_rejected():
    def dup(spec):
        spec['nodes'].append({'id': 'L1Ex', 'archetype': 'e_pretrained', 'layer': 'L1'})
        spec['edges'].append({'id': 'bx', 'source': 'L1Ex', 'target': 'L1C0',
                              'kind': 'basal_excitation'})
    with pytest.raises(SpecError, match='exactly one incoming basal'):
        validate_spec(_mutate(dup), 9)


def test_missing_apical_edge_rejected():
    def drop(spec):
        spec['edges'] = [e for e in spec['edges']
                         if e['kind'] != 'apical_excitation']
    with pytest.raises(SpecError, match='at least one incoming apical'):
        validate_spec(_mutate(drop), 9)


def test_duplicate_apical_edge_rejected():
    def dup(spec):
        spec['edges'].append({'id': 'ap0b', 'source': 'L2E0', 'target': 'L1C0',
                              'kind': 'apical_excitation'})
    with pytest.raises(SpecError, match='duplicate apical_excitation'):
        validate_spec(_mutate(dup), 9)


def test_bidirectional_basal_rejected():
    def flip(spec):
        for e in spec['edges']:
            if e['id'] == 'b0':
                e['directed'] = False
    with pytest.raises(SpecError, match='must be directed'):
        validate_spec(_mutate(flip), 9)


def test_bidirectional_hard_reset_rejected():
    def flip(spec):
        for e in spec['edges']:
            if e['id'] == 'hr_l1':
                e['directed'] = False
    with pytest.raises(SpecError, match='must be directed'):
        validate_spec(_mutate(flip), 9)


def test_hard_reset_onto_legacy_e_rejected():
    # A hard reset targeting a legacy synchronous E cell is illegal.
    spec = {
        'nodes': [
            {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
            {'id': 'R', 'archetype': 'i_relay', 'layer': 'L2'},
            {'id': 'C', 'archetype': 'e_coincidence', 'layer': 'L1'},
            {'id': 'B', 'archetype': 'e_pretrained', 'layer': 'L1'},
            {'id': 'RGx', 'archetype': 'rg_source', 'pixel': 1},
        ],
        'edges': [
            {'id': 'pt', 'source': 'RGx', 'target': 'B', 'kind': 'pretrained_excitation'},
            {'id': 'b', 'source': 'B', 'target': 'C', 'kind': 'basal_excitation'},
            {'id': 'ap', 'source': 'B', 'target': 'C', 'kind': 'apical_excitation'},
            {'id': 'hr', 'source': 'R', 'target': 'S0', 'kind': 'hard_reset_inhibition',
             'sign': -1},
        ],
    }
    with pytest.raises(SpecError, match='requires every excitatory target'):
        validate_spec(spec, 9)


def test_mixing_legacy_competitor_with_event_resolved_rejected():
    # A graph WITHOUT any hard reset, mixing a legacy e_competitor with an
    # event-resolved e_latency_competitor -> the mixing rule (not the hard-reset rule).
    spec = {
        'nodes': [
            {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
            {'id': 'LC', 'archetype': 'e_competitor', 'layer': 'L2'},
            {'id': 'EV', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
        ],
        'edges': [
            {'id': 'ff_lc', 'source': 'S0', 'target': 'LC', 'kind': 'feedforward'},
            {'id': 'ff_ev', 'source': 'S0', 'target': 'EV', 'kind': 'feedforward'},
        ],
    }
    with pytest.raises(SpecError, match='may not mix legacy'):
        validate_spec(spec, 9)


# ------------------------------------------------------- config invariant
def test_high_cap_accepted_init_above_cap_still_rejected():
    # A cap above the impulse threshold is now the intended one-shot regime, not a
    # misconfiguration -- it builds fine (the old c_max < w1 guard is gone). The
    # c_init <= c_max ordering guard is still enforced.
    spec = _synth_spec()
    e = _build(spec, c_basal_weight_max=5000.0)       # high cap accepted (no rejection)
    assert e.coincidence[0].w_max == pytest.approx(5000.0)
    with pytest.raises(ValueError, match='must be <= c_basal_weight_max'):
        _build(spec, c_basal_weight_init=5000.0, c_basal_weight_max=100.0)


# ------------------------------------------------ legacy graphs stay legacy
def test_non_coincidence_custom_graph_is_not_event_resolved():
    # A custom graph with no coincidence/latency archetypes runs the synchronous engine
    # (stand-in for the removed legacy presets): not event-resolved, no C/latency/pretrained.
    e = SimulationEngine(seed=1)
    e.apply_topology({'name': 'sync', 'nodes': [
        {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
        {'id': 'C0', 'archetype': 'e_competitor'},
        {'id': 'R', 'archetype': 'i_relay'}],
        'edges': [{'id': 'ff', 'source': 'S0', 'target': 'C0', 'kind': 'feedforward'},
                  {'id': 're', 'source': 'C0', 'target': 'R', 'kind': 'relay_excitation'},
                  {'id': 'in', 'source': 'R', 'target': 'C0', 'kind': 'inhibition'}]})
    assert e.event_resolved is False
    assert e.coincidence == [] and e.latency_competitors == [] and e.pretrained == []
