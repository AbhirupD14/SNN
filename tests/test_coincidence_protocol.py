"""Phase 6 — public protocol + API for the coincidence topology: additive,
backward-compatible serialization of C/timing fields, live basal weight editing,
scalar-C-stimulation rejection, API execution of the new preset, and deterministic
replay. Legacy payloads must be byte-unchanged.
"""

import asyncio

import pytest

from backend.simulation import SimulationEngine
from backend.serializer import full_state


def _coin(**kw):
    e = SimulationEngine(seed=1, topology='rg_coincidence', **kw)
    e.set_pattern('row 1')
    return e


def _run(e, n):
    for _ in range(n):
        e.step()
    return e


# ---------------------------------------------------- static topology payload
def test_topology_serializes_edge_weights_by_kind():
    e = _run(_coin(), 22)
    syn = {s['id']: s for s in e.topology()['synapses']}
    assert syn['basal0']['weight'] is not None and syn['basal0']['weight'] > 0    # live basal
    assert syn['apical0->0']['weight'] is None                                    # unweighted
    assert syn['hr_l1_0']['weight'] is None                                       # hard reset
    assert syn['rg0->l1e0']['weight'] == pytest.approx(e._q_pretrained)           # fixed magnitude


def test_topology_reports_new_archetypes_and_kinds():
    e = _coin()
    neurons = {n['id']: n for n in e.topology()['neurons']}
    assert neurons['L1E0']['role'] == 'pretrained'
    assert neurons['L1C0']['role'] == 'coincidence'
    assert neurons['L2E0']['role'] == 'competitor'
    kinds = {s['kind'] for s in e.topology()['synapses']}
    assert {'pretrained_excitation', 'basal_excitation', 'apical_excitation',
            'hard_reset_inhibition'} <= kinds


# ----------------------------------------------------- dynamic C/timing fields
def test_dynamic_c_fields_and_diagnostics_present():
    e = _run(_coin(), 22)
    d = e.dynamic_state()
    c = next(n for n in d['neurons'] if n['id'] == 'L1C3')
    assert {'basal_weight', 'basal_received', 'basal_eligible', 'apical_active',
            'apical_sources', 'coincidence_active', 'coincidence_charge',
            'deposit_committed_this_boundary', 'coincidence_deposit_count',
            'coincidence_deposit_tau', 'apical_delivery_count',
            'apical_duplicate_count',
            'spike_tau'} <= set(c)
    assert 'hard_reset_events' in d and 'latency_ties' in d
    l2 = next(n for n in d['neurons'] if n['id'] == 'L2E0')
    assert 'spike_tau' in l2                          # every event-resolved E spike carries tau


def test_hard_reset_events_schema_when_present():
    e = _coin()
    seen = None
    for _ in range(30):
        e.step()
        if e.hard_reset_events:
            seen = e.hard_reset_events[0]
            break
    assert seen is not None
    assert set(seen) == {'source', 'target', 'edge_id', 'kind', 'outer_boundary',
                         'tau', 'v_before', 'drive_before'}
    assert seen['kind'] == 'hard_reset'


# ---- non-coincidence (synchronous custom) payloads carry no coincidence fields ----
def _sync_engine():
    """A minimal non-coincidence, non-event-resolved custom graph (stand-in for the old
    legacy presets): one sensory source, one competitor, one relay."""
    e = SimulationEngine(seed=1)
    e.apply_topology({'name': 'sync', 'nodes': [
        {'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
        {'id': 'C0', 'archetype': 'e_competitor'},
        {'id': 'R', 'archetype': 'i_relay'}],
        'edges': [{'id': 'ff', 'source': 'S0', 'target': 'C0', 'kind': 'feedforward'},
                  {'id': 're', 'source': 'C0', 'target': 'R', 'kind': 'relay_excitation'},
                  {'id': 'in', 'source': 'R', 'target': 'C0', 'kind': 'inhibition'}]})
    return e


def test_legacy_dynamic_payload_has_no_coincidence_fields():
    e = _sync_engine()
    e.set_input([1, 0, 0, 0, 0, 0, 0, 0, 0])
    d = e.step()
    n0 = d['neurons'][0]
    assert 'spike_tau' not in n0
    assert 'basal_weight' not in n0 and 'coincidence_active' not in n0
    # additive top-level diagnostics are present but empty for a non-coincidence graph.
    assert d['hard_reset_events'] == [] and d['latency_ties'] == []


def test_old_payload_field_set_preserved_for_legacy():
    e = _sync_engine()
    e.set_input([0, 0, 0, 1, 0, 0, 0, 0, 0])
    d = e.step()
    assert set(d) >= {'timestep', 'running', 'neurons', 'changed_synapses', 'emitted',
                      'inhibitory_pulses', 'input', 'winner', 'stats', 'log'}


# ----------------------------------------------- manual basal weight editing
def test_set_basal_weight_clips_to_c_cap():
    e = _coin()
    cap = e._resolve_coincidence_params()['c_max']
    assert e.set_synapse_weight('basal3', 9e9) == pytest.approx(cap)
    assert e.set_synapse_weight('basal3', -5.0) == 0.0


def test_apical_pretrained_reset_edges_not_weight_editable():
    e = _coin()
    for eid in ('apical0->0', 'rg0->l1e0', 'hr_l1_0', 're_l1i_0'):
        with pytest.raises(KeyError):
            e.set_synapse_weight(eid, 100.0)


def test_scalar_stimulation_of_coincidence_cell_rejected():
    e = _coin()
    with pytest.raises(ValueError, match='coincidence cell'):
        e.stimulate('L1C0')
    # a latency competitor is still ordinarily stimulable (not a C cell).
    e.stimulate('L2E0')                               # no raise


# ------------------------------------------------------ full-state envelope
def test_full_state_envelope_wellformed_for_preset():
    e = _run(_coin(), 5)
    fs = full_state(e, running=False, speed=10.0)
    assert set(fs) == {'topology', 'dynamic'}
    assert fs['dynamic']['hard_reset_events'] == fs['dynamic']['hard_reset_events']


# -------------------------------------------------------- deterministic replay
def test_deterministic_replay_same_seed_same_frames():
    def trace():
        e = SimulationEngine(seed=1, topology='rg_coincidence')
        e.set_pattern('row 1')
        frames = []
        for _ in range(30):
            d = e.step()
            frames.append(tuple(
                (n['id'], round(n['potential'], 6), int(n['spiked']),
                 round(n.get('spike_tau') or -1.0, 9), round(n.get('basal_weight', -1), 6))
                for n in d['neurons']))
        return frames
    assert trace() == trace()


# ----------------------------------------------------------------- API layer
def test_api_vocabulary_propagates_event_resolved_and_kinds():
    import backend.api as api
    vocab = api._vocabulary()
    assert vocab['archetypes']['e_coincidence']['event_resolved'] is True
    assert vocab['archetypes']['e_sensory']['event_resolved'] is False
    assert {'basal_excitation', 'apical_excitation', 'pretrained_excitation',
            'hard_reset_inhibition'} <= set(vocab['edge_kinds'])


def test_api_engine_runs_new_preset_via_config():
    import backend.api as api
    prior_topology = api.engine.params['topology']
    try:
        api.engine.apply_config({'topology': 'rg_coincidence'})
        assert api.engine.event_resolved is True
        api.engine.set_pattern('row 1')
        for _ in range(6):
            api.engine.step()
        d = api.engine.dynamic_state()
        assert any(n['id'].startswith('L1C') for n in d['neurons'])
    finally:
        api.engine.apply_config({'topology': prior_topology})


def test_api_stimulate_endpoint_rejects_c_cell_with_400():
    import backend.api as api
    from backend.simulation import SimulationEngine as SE

    async def call():
        prior_topology = api.engine.params['topology']
        api.engine.apply_config({'topology': 'rg_coincidence'})
        try:
            body = api.StimulateBody(neuron_id='L1C0', magnitude=1.0)
            resp = await api.stimulate(body)
            return resp
        finally:
            api.engine.apply_config({'topology': prior_topology})

    resp = asyncio.run(call())
    assert getattr(resp, 'status_code', None) == 400
