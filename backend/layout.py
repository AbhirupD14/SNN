"""Seeded functional neuron layout.

These coordinates belong to the model: SimulationEngine uses them to derive
feedforward distances. The browser may expand them for display, but must never
feed display positions back into this module.
"""

from __future__ import annotations

import math

import numpy as np


GRID = 3.8
_L2_RING_R = 5.5
_L2_Z = 8.0
_L2E_ANCHORS = [
    (round(_L2_RING_R * math.cos(k * 2 * math.pi / 8), 4),
     round(_L2_RING_R * math.sin(k * 2 * math.pi / 8), 4),
     _L2_Z)
    for k in range(8)
]
_L1I_Z_OFFSET = -2.6
_L2I_HOME = (0.0, 0.0, 12.0)

# Camera-to-target vector used by the default renderer. Rejecting collisions in
# its orthogonal plane prevents neurons from being separated only in depth.
_LAYOUT_VIEW_VECTOR = np.array([14.0, -18.0, 6.0])
_PROJECTED_SEPARATION_FRACTION = 0.75

# Model-space center-distance floors. Renderer-only scaling adds more visual
# space without changing these coordinates or the resulting signal attenuation.
_PAIR_MIN_SEPARATION = {
    frozenset(('L1E',)): 1.5,
    frozenset(('L1I',)): 1.2,
    frozenset(('L2E',)): 2.0,
    frozenset(('L2E', 'L2I')): 2.2,
    frozenset(('L1E', 'L1I')): 1.3,
}


def layer_anchors(n_pix: int, n_out: int) -> dict[str, np.ndarray]:
    """Return the regular reference layout used when scatter is disabled."""
    anchors: dict[str, np.ndarray] = {}
    for i in range(n_pix):
        row, col = divmod(i, 3)
        anchors[f'L1E{i}'] = np.array([(col - 1) * GRID, (1 - row) * GRID, 0.0])
        anchors[f'L1I{i}'] = anchors[f'L1E{i}'] + np.array([0.0, 0.0, _L1I_Z_OFFSET])
    for j in range(n_out):
        if j < len(_L2E_ANCHORS):
            anchor = _L2E_ANCHORS[j]
        else:
            angle = j * 2 * math.pi / n_out
            anchor = (_L2_RING_R * math.cos(angle), _L2_RING_R * math.sin(angle), _L2_Z)
        anchors[f'L2E{j}'] = np.array(anchor, dtype=float)
    anchors['L2I'] = np.array(_L2I_HOME, dtype=float)
    return anchors


def generate_layout(params: dict, rng, n_pix: int, n_out: int) -> dict[str, np.ndarray]:
    """Generate deterministic, irregular functional coordinates for all neurons.

    L1E remains inside retinotopic cells, L1I stays near its paired L1E, and L2E
    is sampled as a bounded cloud. Pairwise 3D and projected separation checks
    reject overlaps rather than silently accepting them.
    """
    anchors = layer_anchors(n_pix, n_out)
    if not params['layout_scatter_enabled']:
        return {neuron_id: anchor.copy() for neuron_id, anchor in anchors.items()}

    min_sep = max(0.0, float(params['layout_min_separation']))
    placed: dict[str, np.ndarray] = {}

    def role(neuron_id):
        return 'L2I' if neuron_id == 'L2I' else neuron_id[:3]

    def required(a, b):
        return max(min_sep, _PAIR_MIN_SEPARATION.get(
            frozenset((role(a), role(b))), min_sep))

    view = _LAYOUT_VIEW_VECTOR / np.linalg.norm(_LAYOUT_VIEW_VECTOR)

    def acceptable(neuron_id, pos):
        for other, other_pos in placed.items():
            gap = required(neuron_id, other)
            delta = pos - other_pos
            if np.linalg.norm(delta) < gap:
                return False
            projected = delta - np.dot(delta, view) * view
            if np.linalg.norm(projected) < gap * _PROJECTED_SEPARATION_FRACTION:
                return False
        return True

    def place(neuron_id, sampler, attempts=1000):
        for _ in range(attempts):
            pos = sampler()
            if acceptable(neuron_id, pos):
                placed[neuron_id] = pos
                return
        raise RuntimeError(f'could not place {neuron_id} without a visual overlap')

    # Keeping jitter below half a cell preserves row/column/diagonal ordering.
    xy_jitter = min(abs(float(params['l1_xy_jitter'])), 0.45 * GRID)
    z_jitter = abs(float(params['l1_z_jitter']))
    pair_jitter = abs(float(params['l1i_pair_jitter']))
    l2_radius = abs(float(params['l2_xy_radius']))
    l2_z_jitter = abs(float(params['l2_z_jitter']))

    for i in range(n_pix):
        anchor = anchors[f'L1E{i}']
        place(f'L1E{i}', lambda anchor=anchor: anchor + np.array([
            rng.uniform(-xy_jitter, xy_jitter),
            rng.uniform(-xy_jitter, xy_jitter),
            rng.uniform(-z_jitter, z_jitter),
        ]))

    for i in range(n_pix):
        base = placed[f'L1E{i}'] + np.array([0.0, 0.0, _L1I_Z_OFFSET])
        place(f'L1I{i}', lambda base=base: base + np.array([
            rng.uniform(-pair_jitter, pair_jitter),
            rng.uniform(-pair_jitter, pair_jitter),
            rng.uniform(-pair_jitter, pair_jitter),
        ]))

    for j in range(n_out):
        def sample_l2e():
            angle = rng.uniform(0.0, 2 * math.pi)
            radius = l2_radius * math.sqrt(rng.uniform(0.0, 1.0))
            return np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                _L2_Z + rng.uniform(-l2_z_jitter, l2_z_jitter),
            ])
        place(f'L2E{j}', sample_l2e)

    place('L2I', lambda: anchors['L2I'] + np.array([
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-0.8, 0.8),
    ]))
    return placed
