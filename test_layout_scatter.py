"""Focused tests for spacious seeded layout and distance-weighted charge delivery.

The topology coordinates are final physical/display coordinates. L1E->L2E model
distances are derived from them with one global unit; the coordinates themselves are
never normalized. Run directly with `.venv/bin/python test_layout_scatter.py`.
"""

from pathlib import Path

import numpy as np

from neuron_flexible import Neuron
from backend.simulation import SimulationEngine, N_OUT, N_PIX
from backend.layout import (
    GRID, _L2E_ANCHORS, _L2I_HOME, _L2_Z, _LAYOUT_VIEW_VECTOR, _PAIR_MIN_SEPARATION,
    _PROJECTED_SEPARATION_FRACTION,
)


def _positions(engine):
    return {nid: np.asarray(m['pos'], dtype=float) for nid, m in engine.meta.items()}


def _l1e(engine):
    return np.array([engine.meta[f'L1E{i}']['pos'] for i in range(N_PIX)])


def _l2e(engine):
    return np.array([engine.meta[f'L2E{j}']['pos'] for j in range(N_OUT)])


def _role(nid):
    return 'L2I' if nid == 'L2I' else nid[:3]


def _required(engine, a, b):
    configured = float(engine.params['layout_min_separation'])
    return max(configured, _PAIR_MIN_SEPARATION.get(
        frozenset((_role(a), _role(b))), configured))


def test_same_seed_identical_layout_and_distances():
    a, b = SimulationEngine(seed=7), SimulationEngine(seed=7)
    for nid in a.meta:
        assert np.allclose(a.meta[nid]['pos'], b.meta[nid]['pos']), nid
    assert np.allclose(a.physical_ff_distances, b.physical_ff_distances)
    for j in range(N_OUT):
        assert np.allclose(a.l2.excitatory_neurons[j].distance,
                           b.l2.excitatory_neurons[j].distance)


def test_different_seeds_differ():
    a, b = SimulationEngine(seed=1), SimulationEngine(seed=2)
    assert sum(not np.allclose(a.meta[n]['pos'], b.meta[n]['pos']) for n in a.meta) >= N_PIX


def test_l1e_is_irregular_but_retinotopic():
    e = SimulationEngine(seed=3)
    p = _l1e(e)
    assert len(set(np.round(p[:, 0], 6))) > 3
    assert len(set(np.round(p[:, 1], 6))) > 3
    assert np.ptp(p[:, 2]) > 1e-6
    for i in range(N_PIX):
        r, c = divmod(i, 3)
        for k in range(N_PIX):
            r2, c2 = divmod(k, 3)
            if r == r2 and c < c2:
                assert p[i, 0] < p[k, 0]
            if c == c2 and r < r2:
                assert p[i, 1] > p[k, 1]


def test_l2e_is_bounded_irregular_cloud():
    e = SimulationEngine(seed=4)
    p = _l2e(e)
    assert np.ptp(p[:, 2]) > 1e-6
    assert np.std(np.hypot(p[:, 0], p[:, 1])) > 0.1
    assert np.max(np.hypot(p[:, 0], p[:, 1])) <= e.params['l2_xy_radius'] + 1e-9
    assert np.all(np.abs(p[:, 2] - _L2_Z) <= e.params['l2_z_jitter'] + 1e-9)


def test_layers_are_locally_separated():
    e = SimulationEngine(seed=5)
    l1e_z = _l1e(e)[:, 2]
    l2e_z = _l2e(e)[:, 2]
    assert l2e_z.min() - l1e_z.max() >= 5.0
    for i in range(N_PIX):
        assert e.meta[f'L1I{i}']['pos'][2] < e.meta[f'L1E{i}']['pos'][2]


def test_final_coordinates_meet_visual_spacing_floors():
    # Exercise several layouts: the guarantee must hold AFTER every transformation.
    for seed in range(20):
        e = SimulationEngine(seed=seed)
        pos = _positions(e)
        ids = list(pos)
        for ix, a in enumerate(ids):
            for b in ids[ix + 1:]:
                actual = float(np.linalg.norm(pos[a] - pos[b]))
                required = _required(e, a, b)
                assert actual + 1e-9 >= required, (seed, a, b, actual, required)


def test_physical_positions_are_not_distance_normalized():
    e = SimulationEngine(seed=1, layout_scatter_enabled=False)
    # Compatibility geometry must remain at its declared physical scale.
    assert np.allclose(e.meta['L1E0']['pos'], [-GRID, GRID, 0.0])
    assert np.allclose(e.meta['L2E0']['pos'], _L2E_ANCHORS[0])
    assert np.allclose(e.meta['L2I']['pos'], _L2I_HOME)


def test_default_view_has_screen_space_clearance():
    # Orthographic plane-gap invariant used during placement. The renderer uses the
    # same camera-target vector, so no pair can be separated only in depth.
    view = _LAYOUT_VIEW_VECTOR / np.linalg.norm(_LAYOUT_VIEW_VECTOR)
    for seed in range(20):
        e = SimulationEngine(seed=seed)
        pos = _positions(e)
        ids = list(pos)
        for ix, a in enumerate(ids):
            for b in ids[ix + 1:]:
                delta = pos[a] - pos[b]
                projected = delta - np.dot(delta, view) * view
                required = _required(e, a, b) * _PROJECTED_SEPARATION_FRACTION
                assert np.linalg.norm(projected) + 1e-9 >= required, \
                    (seed, a, b, np.linalg.norm(projected), required)


def test_model_distances_derive_from_physical_coordinates():
    e = SimulationEngine(seed=8)
    l1, l2 = _l1e(e), _l2e(e)
    physical = np.linalg.norm(l2[:, None, :] - l1[None, :, :], axis=2)
    expected = physical / physical.min()
    assert np.allclose(e.physical_ff_distances, physical)
    assert np.isclose(e.feedforward_distance_unit, physical.min())
    for j in range(N_OUT):
        assert np.allclose(e.l2.excitatory_neurons[j].distance, expected[j])
    assert np.isclose(expected.min(), 1.0)


def test_model_distances_are_positive_and_vary():
    e = SimulationEngine(seed=10)
    model = np.array([e.l2.excitatory_neurons[j].distance for j in range(N_OUT)])
    assert np.isfinite(model).all() and (model >= 1.0 - 1e-12).all()
    assert model.std() / model.mean() > 0.1
    assert np.mean([row.std() / row.mean() for row in model]) > 0.05


def test_equal_weights_and_different_distances_deliver_different_charge():
    n = Neuron(n_inputs=2)
    n.weights = np.array([0.9, 0.9])
    n.distance = np.array([1.0, 2.0])
    n.distance_weighting = True
    n.distance_ref = n.distance_min = 1.0
    n.distance_power = 2.0
    before = n.weights.copy()
    n.receive_input(np.array([1.0, 0.0]))
    near = n.potential
    n.potential = 0.0
    n.receive_input(np.array([0.0, 1.0]))
    far = n.potential
    assert np.isclose(near, 0.9)
    assert np.isclose(far, 0.9 / 4.0)
    assert np.allclose(n.weights, before)


def test_distance_off_delivers_full_weight():
    n = Neuron(n_inputs=1)
    n.weights = np.array([0.9])
    n.distance = np.array([4.0])
    n.distance_weighting = False
    n.receive_input(np.array([1.0]))
    assert np.isclose(n.potential, 0.9)


def test_distance_does_not_rescale_learning_delta():
    def delta(distance):
        n = Neuron(n_inputs=1, threshold=1.0, weight_cap=10.0, learning_rate=0.2)
        n.weights = np.array([0.5])
        n.excitatory_saturation_cap = 100.0
        n.distance = np.array([distance])
        n.distance_weighting = True
        n._last_input_spikes = np.array([1.0])
        n.potential = 1.0
        w0 = n.weights.copy()
        n.fire()
        return n.weights - w0
    assert np.allclose(delta(1.0), delta(4.0))


def test_reset_reproduces_layout():
    e = SimulationEngine(seed=12)
    before = _positions(e)
    before_d = e.physical_ff_distances.copy()
    e.reset()
    for nid in before:
        assert np.allclose(before[nid], e.meta[nid]['pos']), nid
    assert np.allclose(before_d, e.physical_ff_distances)


def test_renderer_fits_camera_to_topology():
    source = Path('frontend/renderer.js').read_text()
    assert 'new THREE.OrthographicCamera' in source
    assert 'WITHIN_LAYER_SPACING = 4.0' in source
    assert 'BETWEEN_LAYER_SPACING = 4.0' in source
    assert 'this.functionalPos.set(m.id, raw)' in source
    assert 'this._fitCameraToTopology();' in source
    assert 'new THREE.Box3().setFromPoints(points)' in source


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print(f'PASS: {test.__name__}')
    print('ALL LAYOUT-SCATTER TESTS PASSED')
