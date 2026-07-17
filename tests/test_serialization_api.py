"""Serialization / API smoke: the topology and dynamic payloads carry exactly the
fields the frontend consumes, the new population ids/roles are present, config
rejects deleted keys, and reset/reseed plus a full-state cycle work.
"""

import pytest

from backend.simulation import SimulationEngine
from backend.serializer import full_state, topology_message, dynamic_message
from backend.dashboard_config import CONFIG_SPEC, config_values


@pytest.fixture
def engine():
    return SimulationEngine(seed=7)


def test_topology_payload_shape(engine):
    topo = engine.topology()
    assert set(topo) >= {'neurons', 'synapses', 'layers', 'patterns',
                         'pattern_vectors', 'grid', 'params'}
    for n in topo['neurons']:
        assert set(n) >= {'id', 'label', 'layer', 'type', 'threshold', 'pos', 'role'}
        assert len(n['pos']) == 3
    for s in topo['synapses']:
        assert set(s) >= {'id', 'source', 'target', 'kind'}
        assert 'weight' in s
    # RF/weights charts need these params to scale by the cap (now theta/2).
    assert topo['params']['threshold_l2'] == 1000
    assert topo['params']['l2e_weight_cap_frac'] == pytest.approx(0.5)


def test_new_population_ids_and_roles(engine):
    # Default topology is 'pi': 9 L1E_s sources, 8 L2E competitors, 8 PI, 1 L2I.
    ids = {n['id'] for n in engine.topology()['neurons']}
    assert {f'L1E{i}' for i in range(9)} <= ids          # sources (controls.js flashes these)
    assert {f'PI{j}' for j in range(8)} <= ids           # predictive interneurons
    assert {f'L2E{j}' for j in range(8)} <= ids
    assert 'L2I' in ids
    assert not any(i.startswith('L1Enew') for i in ids)  # coincidence topology removed
    ff_ids = {s['id'] for s in engine.topology()['synapses'] if s['kind'] == 'feedforward'}
    assert 'ff0->0' in ff_ids and 'ff8->7' in ff_ids     # weights.js/receptive.js id scheme


def test_dynamic_payload_shape(engine):
    engine.set_pattern('row 1')
    d = engine.step()
    assert set(d) >= {'timestep', 'running', 'neurons', 'changed_synapses',
                      'emitted', 'inhibitory_pulses', 'input', 'winner', 'stats', 'log'}
    assert 'applied_inhibition' not in d                  # renamed: pulses are not charge removal
    for n in d['neurons']:
        assert set(n) >= {'id', 'potential', 'activation', 'spiked', 'freq', 'refractory'}
    # Excitatory neurons expose conductance and activity-trace state.
    exc = [n for n in d['neurons'] if n['id'].startswith(('L1E', 'L2E'))
           and not n['id'].startswith('L1I')]
    for n in exc:
        assert {'g_inh', 'trace', 'v_pre_reset'} <= set(n)
    assert len(d['input']) == 9


def test_inhibitory_pulse_schema():
    # A PI (direct-topology) inhibitory pulse carries conductance-pulse fields and
    # never serializes a 'charge_removed'.
    e = SimulationEngine(seed=1, topology='pi')
    e.set_pattern('row 1')
    pulse = None
    for _ in range(400):
        d = e.step()
        preds = [p for p in d['inhibitory_pulses'] if p['kind'] == 'predictive']
        if preds:
            pulse = preds[0]
            break
    assert pulse is not None
    assert set(pulse) == {'source', 'target', 'kind', 'synaptic_weight',
                          'conductance_increment', 'g_inh_before', 'g_inh_after', 'boundary'}
    assert 'charge_removed' not in pulse
    assert pulse['g_inh_after'] >= pulse['g_inh_before']


def test_config_rejects_deleted_keys(engine):
    applied = engine.apply_config({'confidence_consolidation': True,
                                   'l2_charge_chunks': 20,
                                   'distance_weighting': False})
    assert applied == []                                 # every legacy key rejected
    assert 'confidence_consolidation' not in engine.params
    assert 'l2_charge_chunks' not in engine.params


def test_config_accepts_editable_keys(engine):
    applied = engine.apply_config({'leak_rate': 0.1, 'refractory_steps': 2})
    assert set(applied) == {'leak_rate', 'refractory_steps'}
    assert engine.params['leak_rate'] == 0.1
    assert engine.l2e[0].leak_rate == 0.1                # rebuild propagated it
    # The control surface matches config_values and exposes the separated
    # predictive-inhibition timescales plus the two ablation toggles.
    keys = {c['key'] for c in CONFIG_SPEC}
    assert keys == set(config_values(engine.params))
    assert keys == {'leak_rate', 'refractory_steps', 'eta', 'e_weight_cap', 'topology',
                    'alpha_inh', 'alpha_inh_l1', 'alpha_a', 'pi_eta', 'pi_g_scale',
                    'l2i_g_scale', 'pi_conductance_enabled', 'pi_plasticity_enabled',
                    # 'rg' topology controls: the RG->L1E projection's ablation toggle
                    # and its initialization-jitter control.
                    'enc_plasticity_enabled', 'enc_init_jitter',
                    # residual topology timing/expression controls
                    'residual_exc_scale', 'switch_trace_decay',
                    'switch_trace_threshold', 'switch_residual_charge_frac',
                    'switch_trace_charge_frac', 'switch_g_scale',
                    'switch_conductance_enabled'}


def test_reset_and_reseed_cycle(engine):
    engine.set_pattern('row 1')
    for _ in range(20):
        engine.step()
    engine.reset()
    assert engine.timestep == 0                          # reset rebuilds fresh
    seed_before = engine.params['seed']
    new_seed = engine.reseed()
    assert new_seed != seed_before

    # full_state / message envelopes are well-formed.
    fs = full_state(engine, running=False, speed=12.0)
    assert set(fs) == {'topology', 'dynamic'}
    assert topology_message(engine)['type'] == 'topology'
    assert dynamic_message(engine, True, 5.0)['data']['running'] is True


def test_api_module_imports_and_builds():
    import backend.api as api
    assert api.engine is not None
    assert len(api.engine.topology()['neurons']) == 26   # default topology 'pi'
