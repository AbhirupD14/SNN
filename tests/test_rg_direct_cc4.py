"""Structural + causal-symmetry contract for the direct 3x3 cortical-column preset
(``rg_direct_cc4``): the exact graph, its forbidden paths, deterministic construction,
serialization round-trip of the dual FE/FES parameters, and proof that the initial winner
follows active-afferent synaptic strength rather than serialized node order.
"""

import numpy as np
import pytest

from backend.network_spec import rg_direct_cc4_spec, validate_spec
from backend.simulation import SimulationEngine, PATTERNS


# =============================================================== exact graph
def test_exact_node_and_edge_counts_derived_from_the_graph():
    spec = validate_spec(rg_direct_cc4_spec(), 9)
    nodes, edges = spec['nodes'], spec['edges']
    assert len(nodes) == 14                                     # 9 RGC + 4 E + 1 I
    assert len(edges) == 44                                     # 9*4 + 4 + 4
    arch = {}
    for n in nodes:
        arch[n['archetype']] = arch.get(n['archetype'], 0) + 1
    assert arch == {'rg_source': 9, 'e_latency_competitor': 4, 'i_relay': 1}
    kinds = {}
    for e in edges:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    assert kinds == {'feedforward': 36, 'relay_excitation': 4, 'hard_reset_inhibition': 4}


def test_projections_are_exactly_the_three_declared_families():
    spec = validate_spec(rg_direct_cc4_spec(), 9)
    by_proj = {}
    for e in spec['edges']:
        by_proj.setdefault(e.get('projection'), []).append(e)
    assert set(by_proj) == {'rg_to_column', 'column_e_to_i', 'column_i_to_e'}
    e_ids = {n['id'] for n in spec['nodes'] if n.get('column_role') == 'E'}
    rgc_ids = {n['id'] for n in spec['nodes'] if n['archetype'] == 'rg_source'}
    i_id = next(n['id'] for n in spec['nodes'] if n.get('column_role') == 'I')
    # RGC -> every E (dense, plastic feedforward).
    assert {(e['source'], e['target']) for e in by_proj['rg_to_column']} == {
        (r, c) for r in rgc_ids for c in e_ids}
    # Every E -> the single WTA I; the WTA I -> every E (hard reset).
    assert {e['source'] for e in by_proj['column_e_to_i']} == e_ids
    assert all(e['target'] == i_id for e in by_proj['column_e_to_i'])
    assert {e['target'] for e in by_proj['column_i_to_e']} == e_ids
    assert all(e['source'] == i_id for e in by_proj['column_i_to_e'])


def test_forbidden_structures_are_absent():
    spec = validate_spec(rg_direct_cc4_spec(), 9)
    archs = {n['archetype'] for n in spec['nodes']}
    # No feature relay / coincidence C / feature-specific I / Eor / predictor / switch.
    assert archs == {'rg_source', 'e_latency_competitor', 'i_relay'}
    assert not any(n.get('column_role') == 'Eor' for n in spec['nodes'])
    kinds = {e['kind'] for e in spec['edges']}
    forbidden = {'basal_excitation', 'apical_excitation', 'predictive_inhibition',
                 'fixed_excitation', 'trace_excitation', 'pretrained_excitation',
                 'inhibition'}
    assert not (kinds & forbidden)                             # no coincidence/relay/predictive path
    # No hierarchy: exactly one non-RGC layer, and no inhibition targets an RGC.
    layers = {n['layer'] for n in spec['nodes'] if n['archetype'] != 'rg_source'}
    assert layers == {'CC'}                                    # no L2/L3 node
    rgc_ids = {n['id'] for n in spec['nodes'] if n['archetype'] == 'rg_source'}
    for e in spec['edges']:
        assert e['target'] not in rgc_ids                     # nothing drives/inhibits an RGC


def test_construction_uses_metadata_not_id_parsing():
    spec = validate_spec(rg_direct_cc4_spec(), 9)
    for n in spec['nodes']:
        if n['archetype'] == 'e_latency_competitor':
            assert n['column_role'] == 'E' and isinstance(n['column_index'], int)
        elif n['archetype'] == 'i_relay':
            assert n['column_role'] == 'I'
        elif n['archetype'] == 'rg_source':
            assert isinstance(n['pixel'], int) and 0 <= n['pixel'] < 9


# ====================================================== deterministic build
def test_fresh_construction_is_deterministic():
    a = rg_direct_cc4_spec()
    b = rg_direct_cc4_spec()
    assert [n['id'] for n in a['nodes']] == [n['id'] for n in b['nodes']]
    assert [e['id'] for e in a['edges']] == [e['id'] for e in b['edges']]
    assert [n.get('pos') for n in a['nodes']] == [n.get('pos') for n in b['nodes']]
    # Two engines with the same seed initialize identical weights.
    e1 = SimulationEngine(seed=7, topology='rg_direct_cc4', leak_rate=0.0, dual_fe_fes=True)
    e2 = SimulationEngine(seed=7, topology='rg_direct_cc4', leak_rate=0.0, dual_fe_fes=True)
    for c1, c2 in zip(e1.latency_competitors, e2.latency_competitors):
        assert np.array_equal(c1.acc_weights, c2.acc_weights)


def test_the_four_e_are_placed_around_the_i():
    spec = rg_direct_cc4_spec()
    pos = {n['id']: n['pos'] for n in spec['nodes']}
    i_pos = np.array(pos['ccI'])
    e_pos = [np.array(pos[f'ccE{k}']) for k in range(4)]
    # Same layer height, ringed around the central I (equal radius, distinct angles).
    radii = [np.linalg.norm(p[:2] - i_pos[:2]) for p in e_pos]
    assert all(p[2] == i_pos[2] for p in e_pos)               # same column plane
    assert max(radii) - min(radii) < 1e-9 and radii[0] > 0.5  # a real ring around I


# ============================================ serialization / replay round-trip
def test_dual_fe_params_round_trip_in_serialized_params():
    e = SimulationEngine(seed=1, topology='rg_direct_cc4', leak_rate=0.0,
                         dual_fe_fes=True, dual_fe_e=0.002, dual_fe_wte=0.003, dual_fe_B=20.0)
    params = e.topology()['params']
    assert params['dual_fe_fes'] is True
    assert params['dual_fe_e'] == pytest.approx(0.002)
    assert params['dual_fe_wte'] == pytest.approx(0.003)
    assert params['dual_fe_B'] == pytest.approx(20.0)
    # Toggling off through apply_config rebuilds to the production rule and records it.
    e.apply_config({'dual_fe_fes': False})
    assert e.topology()['params']['dual_fe_fes'] is False
    assert e.latency_competitors[0].update_mode == 'linear_fe'


# ==================================== causal symmetry-breaking probes
def _active_pixels(pattern):
    return [i for i, v in enumerate(PATTERNS[pattern]) if v > 0.5]


def _flatten_weights(engine, base, active, advantaged, delta):
    """Overwrite every competitor's feedforward weights to a flat ``base`` on all afferents,
    then add ``delta`` to the ``advantaged`` competitor's ACTIVE-pattern afferents only."""
    for k, c in enumerate(engine.latency_competitors):
        c.acc_weights[:] = base
        if k == advantaged:
            src_index = {s: i for i, s in enumerate(c.ff_src)}
            for pix in active:
                c.acc_weights[src_index[f'RGC{pix}']] += delta


def _run_to_first_winner(engine, pattern, max_steps=50):
    engine.set_pattern(pattern)
    for _ in range(max_steps):
        engine.step()
        if engine.winner is not None:
            return engine.winner
    return None


@pytest.mark.parametrize('advantaged', [2, 1])                # neither is the first node (ccE0)
def test_winner_follows_active_afferent_strength_not_node_order(advantaged):
    e = SimulationEngine(seed=1, topology='rg_direct_cc4', leak_rate=0.0,
                         refractory_steps=0, dual_fe_fes=True)
    pattern = 'row 1'
    active = _active_pixels(pattern)
    # A flat base of theta/4 on all afferents; a small advantage on ONE competitor's active
    # afferents. It is not first in serialized order, so a win proves strength, not order.
    _flatten_weights(e, base=0.25 * e.params['e_threshold'], active=active,
                     advantaged=advantaged, delta=5.0)
    winner = _run_to_first_winner(e, pattern)
    assert winner == e.latency_competitors[advantaged].id
    # The winner is the argmax of active-afferent total (the strongest), not an id special-case.
    totals = []
    for c in e.latency_competitors:
        idx = {s: i for i, s in enumerate(c.ff_src)}
        totals.append(sum(float(c.acc_weights[idx[f'RGC{p}']]) for p in active))
    assert int(np.argmax(totals)) == advantaged


def test_all_active_rgc_and_all_competitors_see_equal_source_events():
    e = SimulationEngine(seed=1, topology='rg_direct_cc4', leak_rate=0.0,
                         refractory_steps=0, dual_fe_fes=True)
    pattern = 'row 1'
    active = set(_active_pixels(pattern))
    e.set_pattern(pattern)
    rgc_fires = {f'RGC{p}': 0 for p in active}
    deliv = {c.id: 0 for c in e.latency_competitors}
    for _ in range(20):
        e.step()
        for p in active:
            if e.spiked[f'RGC{p}']:
                rgc_fires[f'RGC{p}'] += 1
        # every competitor received exactly the active afferents this boundary (dense wiring)
        for c in e.latency_competitors:
            delivered = e._ff_deliv_now.get(c.id, set()) or e._ff_deliv_next.get(c.id, set())
            if delivered:
                assert {int(s[3:]) for s in delivered} == active
                deliv[c.id] += 1
    # Every active RGC emitted the same number of spikes, none suppressed or frequency-halved.
    assert len(set(rgc_fires.values())) == 1 and next(iter(rgc_fires.values())) == 20
    # Every competitor saw the same number of delivery boundaries.
    assert len(set(deliv.values())) == 1


def test_exact_symmetry_is_resolved_by_stable_node_order_honestly():
    # With perfectly equal weights the four crossings tie; the scheduler falls back to stable
    # node order, so ccE0 wins. Reported honestly (this is why the production run uses seeded
    # jitter to break the tie by strength instead).
    e = SimulationEngine(seed=1, topology='rg_direct_cc4', leak_rate=0.0,
                         refractory_steps=0, dual_fe_fes=True)
    flat = 0.25 * e.params['e_threshold']
    for c in e.latency_competitors:
        c.acc_weights[:] = flat
    winner = _run_to_first_winner(e, 'row 1')
    assert winner == e.latency_competitors[0].id                # lowest node order among a tie
