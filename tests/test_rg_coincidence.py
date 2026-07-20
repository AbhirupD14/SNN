"""Phase 5 — the full built-in `rg_coincidence` preset: exact topology counts and
roles, pretrained RG->L1E behavior, the end-to-end causal chain, immediate L1/L2
resets sharing their causal spike's tau, and the calibrated timing-sanity crossings.
Runs entirely through the public SimulationEngine entry point.
"""

import math
from collections import Counter

import pytest

from snn.neurons import CoincidencePyramidalNeuron, leak_to_conductance
from backend.simulation import SimulationEngine, N_PIX, N_OUT
from backend.network_spec import preset_spec


def _engine(**kw):
    return SimulationEngine(seed=1, topology='rg_coincidence', **kw)


# ------------------------------------------------------------ topology counts
def test_exact_node_and_edge_counts():
    spec = preset_spec('rg_coincidence', N_PIX, N_OUT)
    assert len(spec['nodes']) == 45
    assert len(spec['edges']) == 196
    arch = Counter(n['archetype'] for n in spec['nodes'])
    assert arch == {'rg_source': 9, 'e_pretrained': 9, 'e_coincidence': 9,
                    'i_relay': 10, 'e_latency_competitor': 8}
    kinds = Counter(e['kind'] for e in spec['edges'])
    assert kinds == {'pretrained_excitation': 9, 'feedforward': 72,
                     'basal_excitation': 9, 'apical_excitation': 72,
                     'relay_excitation': 17, 'hard_reset_inhibition': 17}


def test_registries_and_no_legacy_competitor():
    e = _engine()
    assert e.event_resolved is True
    assert len(e.coincidence) == 9
    assert len(e.latency_competitors) == 8
    assert len(e.pretrained) == 9
    assert e.competitors == []                       # never a legacy WTA participant
    # all L2 nodes are e_latency_competitor
    l2_arch = {n['archetype'] for n in e.spec['nodes'] if n['id'].startswith('L2E')}
    assert l2_arch == {'e_latency_competitor'}


def test_each_c_has_one_basal_and_eight_apical():
    e = _engine()
    for c in e.coincidence:
        assert len(c.basal.source_ids) == 1
        idx = c.id[3:]
        assert c.basal.source_ids == [f'L1E{idx}']   # paired basal
        assert len(c.apical.source_ids) == N_OUT     # all eight L2E
        assert c.apical.source_ids == [f'L2E{j}' for j in range(N_OUT)]


def test_c_has_no_flat_ff_or_apical_weights():
    e = _engine()
    for c in e.coincidence:
        assert isinstance(c, CoincidencePyramidalNeuron)
        assert not hasattr(c, 'acc_weights')
        assert c.apical.weights is None


def test_l1i_one_to_one_and_l2i_all_l2e():
    e = _engine()
    for i in range(N_PIX):
        assert e._hardreset_out[f'L1I{i}'] == [(f'L1E{i}', f'hr_l1_{i}')]
    l2i_targets = {t for (t, _) in e._hardreset_out['L2I']}
    assert l2i_targets == {f'L2E{j}' for j in range(N_OUT)}


def test_topology_order_invariance():
    a, b = _engine(), _engine()
    assert a.order == b.order
    import numpy as np
    wa = np.concatenate([c.acc_weights for c in a.latency_competitors])
    wb = np.concatenate([c.acc_weights for c in b.latency_competitors])
    assert np.array_equal(wa, wb)
    assert [c.basal_weight for c in a.coincidence] == [c.basal_weight for c in b.coincidence]


def test_latency_l2_rows_use_normalized_working_initialization():
    e = _engine()
    target = e.params['l2_init_total_frac'] * e.params['e_threshold']
    assert e.params['l2_init_total_frac'] == pytest.approx(0.95)
    assert e.params['c_eta'] == pytest.approx(0.001)
    for cell in e.latency_competitors:
        assert cell.acc_weights.sum() == pytest.approx(target)
        assert len(set(cell.acc_weights.round(9))) > 1   # seeded direction retained


def test_working_initialization_survives_config_rebuild_and_reset():
    e = _engine()
    e.apply_config({'l2_init_total_frac': 0.8, 'c_eta': 0.0025})
    for cell in e.latency_competitors:
        assert cell.acc_weights.sum() == pytest.approx(800.0)
    assert all(c.eta_c == pytest.approx(0.0025) for c in e.coincidence)
    e.reset()
    for cell in e.latency_competitors:
        assert cell.acc_weights.sum() == pytest.approx(800.0)


# ------------------------------------------------------ pretrained RG -> L1E
def test_pretrained_l1e_fires_one_boundary_after_single_rg_spike():
    e = _engine()
    e.set_input([1, 0, 0, 0, 0, 0, 0, 0, 0])         # pixel 0 -> RG0
    e.step()                                         # t=1: RG0 fires (exogenous)
    assert e.spiked['RG0'] is True
    assert e.spiked['L1E0'] is False                 # not the same boundary
    e.clear_input()                                  # isolate to a single RG spike
    e.step()                                         # t=2: pretrained L1E0 fires
    assert e.spiked['L1E0'] is True
    l1e0 = e.exc['L1E0']
    # No L1E weight exists to change (fixed pretrained relay).
    assert l1e0.acc_weights.shape == (0,)


def test_pretrained_crossing_tau_matches_spec_sanity():
    # theta=1000, leak=0.03 -> pretrained L1E crossing from rest tau ~= 0.952.
    e = _engine()
    cpar = e._resolve_coincidence_params()
    theta = float(e.params['e_threshold'])
    g_L = leak_to_conductance(float(e.params['leak_rate']))
    v_inf = cpar['q_pretrained'] / g_L
    tau = (1.0 / g_L) * math.log(v_inf / (v_inf - theta))
    assert tau == pytest.approx(0.952, abs=0.002)
    # observe it live: isolate one RG spike, read L1E spike_tau.
    e.set_input([1, 0, 0, 0, 0, 0, 0, 0, 0]); e.step(); e.clear_input(); e.step()
    assert e.exc['L1E0'].spike_tau == pytest.approx(tau, abs=1e-6)


def test_resolved_c_weight_scale_and_invariant():
    e = _engine()
    cpar = e._resolve_coincidence_params()
    assert cpar['c_init'] == pytest.approx(520.538, abs=0.01)
    assert cpar['c_max'] == pytest.approx(566.923, abs=0.01)
    assert cpar['c_max'] < cpar['w1']                # cap below one-deposit firing weight
    for c in e.coincidence:
        assert c.basal_weight == pytest.approx(cpar['c_init'])
        assert c.w_max == pytest.approx(cpar['c_max'])


def test_c_second_coincidence_timing_sanity():
    # From rest: one c_init deposit leaks a boundary, the second crosses ~0.980;
    # at the cap the second crosses ~0.813 (spec non-normative sanity values).
    e = _engine()
    cpar = e._resolve_coincidence_params()
    theta = float(e.params['e_threshold'])
    g_L = leak_to_conductance(float(e.params['leak_rate']))

    def second_cross_tau(w):
        v1 = (w / g_L) * (1.0 - math.exp(-g_L))      # after 1st full boundary from rest
        v_inf = w / g_L
        return (1.0 / g_L) * math.log((v_inf - v1) / (v_inf - theta))

    assert second_cross_tau(cpar['c_init']) == pytest.approx(0.980, abs=0.002)
    assert second_cross_tau(cpar['c_max']) == pytest.approx(0.813, abs=0.002)


# --------------------------------------------------------- end-to-end causal
def test_full_causal_chain_c_fires_and_resets_paired_l1e():
    e = _engine()
    e.set_pattern('row 1')                            # active pixels 3, 4, 5
    saw_l2, saw_c, saw_l1_reset = False, set(), set()
    for _ in range(30):
        e.step()
        if any(c.spiked for c in e.latency_competitors):
            saw_l2 = True
        for c in e.coincidence:
            if c.spiked:
                saw_c.add(c.id)
        for h in e.hard_reset_events:
            if h['source'].startswith('L1I'):
                saw_l1_reset.add((h['source'], h['target']))
    assert saw_l2                                     # latency WTA produced a winner
    # only the ACTIVE-pixel C cells (3,4,5) ever fire
    assert saw_c == {'L1C3', 'L1C4', 'L1C5'}
    assert saw_l1_reset == {('L1I3', 'L1E3'), ('L1I4', 'L1E4'), ('L1I5', 'L1E5')}


def test_immediate_resets_share_causal_spike_tau():
    e = _engine()
    e.set_pattern('row 1')
    checked_c, checked_l2 = False, False
    for _ in range(30):
        e.step()
        # L1I reset tau == the C cell's spike tau that drove it.
        for c in e.coincidence:
            if c.spiked:
                idx = c.id[3:]
                for h in e.hard_reset_events:
                    if h['source'] == f'L1I{idx}':
                        assert h['tau'] == pytest.approx(c.spike_tau)
                        checked_c = True
        # L2I reset tau == the winning L2E's spike tau.
        for c in e.latency_competitors:
            if c.spiked:
                for h in e.hard_reset_events:
                    if h['source'] == 'L2I':
                        assert h['tau'] == pytest.approx(c.spike_tau)
                        checked_l2 = True
    assert checked_c and checked_l2


def test_basal_learning_only_on_active_pixel_c_cells():
    e = _engine()
    e.set_pattern('row 1')
    init = e._resolve_coincidence_params()['c_init']
    for _ in range(40):
        e.step()
    for c in e.coincidence:
        idx = int(c.id[3:])
        if idx in (3, 4, 5):
            assert c.basal_weight > init              # active pixels learned
        else:
            assert c.basal_weight == pytest.approx(init)   # inactive never learned
        assert c.basal_weight <= e._resolve_coincidence_params()['c_max'] + 1e-9


def test_preset_runs_through_public_reset_and_config():
    # Public entry points: constructor, apply_config topology switch, reset.
    e = SimulationEngine(seed=1, topology='pi')
    e.apply_config({'topology': 'rg_coincidence'})
    assert e.mode == 'rg_coincidence' and e.event_resolved
    e.set_pattern('col 1')
    for _ in range(5):
        e.step()
    e.reset()
    assert e.timestep == 0 and e.event_resolved
