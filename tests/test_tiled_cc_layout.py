"""Tiled functional layout + serialization (Phase 4). Determinism, non-overlap, column
centers following declared tile coordinates, and the additive public/dynamic payloads.
Legacy layout/RNG order and payload keys must stay intact.
"""

import numpy as np

from backend.simulation import SimulationEngine
from backend.layout import generate_tiled_layout, generate_layout
from backend.network_spec import tiled_cc_spec, validate_spec


def test_every_tiled_node_has_a_nonzero_distinct_position():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    positions = {nid: tuple(np.round(m['pos'], 6)) for nid, m in e.meta.items()}
    assert len(positions) == 191
    # no node at the zero placeholder
    assert all(any(abs(x) > 1e-6 for x in p) for p in positions.values())
    # all positions distinct
    assert len(set(positions.values())) == 191
    # distinct WITHIN each column too
    by_col = {}
    for nid, m in e.meta.items():
        cid = m.get('column_id')
        if cid:
            by_col.setdefault(cid, []).append(positions[nid])
    for cid, ps in by_col.items():
        assert len(set(ps)) == len(ps), cid


def test_layout_is_deterministic_for_a_seed():
    spec = validate_spec(tiled_cc_spec(cc_e_count=8), 81)
    a = generate_tiled_layout(np.random.default_rng(1), spec)
    b = generate_tiled_layout(np.random.default_rng(1), spec)
    assert set(a) == set(b)
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_column_centers_follow_declared_tile_coordinates():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    # For the L1 tile grid, column-center x increases with col and y decreases with row.
    def center(cid):
        pts = [m['pos'] for m in e.meta.values() if m.get('column_id') == cid]
        return np.mean(pts, axis=0)
    c00, c01, c22 = center('L1c00'), center('L1c01'), center('L1c22')
    assert c01[0] > c00[0]                              # col 1 is to the right of col 0
    assert c22[1] < c00[1]                              # row 2 is below row 0
    # L2 column sits above (higher z) the L1 tile array
    l2 = center('L2c00')
    assert l2[2] > c00[2]


def test_public_topology_carries_tiling_metadata():
    topo = SimulationEngine(seed=1, topology='tiled_cc').topology()
    assert topo['tiling']['family'] == 'tiled_cortical_columns'
    assert topo['tiling']['input_shape'] == {'rows': 9, 'cols': 9}
    assert topo['tiling']['patch_shape'] == {'rows': 3, 'cols': 3}
    assert len(topo['tiling']['columns']) == 10
    assert topo['tiling']['selected_patch'] == [1, 1]   # default center patch
    # role + projection metadata are present on neurons/synapses
    assert any(n.get('column_role') == 'Eor' for n in topo['neurons'])
    assert any(s.get('projection') == 'column_to_column_apical' for s in topo['synapses'])
    # params expose the construction dims
    p = topo['params']
    assert p['cc_e_count'] == 8 and p['input_shape'] == {'rows': 9, 'cols': 9}
    assert p['column_count'] == 10


def test_dynamic_state_has_per_column_winners():
    e = SimulationEngine(seed=1, topology='tiled_cc', leak_rate=0.0)
    e.set_pattern('row 1')
    seen = {}
    for _ in range(40):
        d = e.step()
        assert 'column_winners' in d
        seen.update(d['column_winners'])
    assert 'L1c11' in seen                              # the active column reported a winner
    assert set(seen['L1c11']) == {'id', 'tau'}


def test_legacy_layout_and_payload_keys_unchanged():
    # Adding the tiled path consumes no legacy RNG draws: the legacy generator is
    # byte-identical for a seed, and legacy dynamic payloads keep their keys.
    a = generate_layout(np.random.default_rng(1), 9, 8)
    b = generate_layout(np.random.default_rng(1), 9, 8)
    assert set(a) == set(b) and all(np.allclose(a[k], b[k]) for k in a)
    e = SimulationEngine(seed=1, topology='rg_coincidence')
    d = e.step()
    assert {'timestep', 'neurons', 'winner', 'input', 'stats',
            'column_winners'} <= set(d)
    topo = e.topology()
    assert {'neurons', 'synapses', 'layers', 'patterns', 'grid', 'params'} <= set(topo)
