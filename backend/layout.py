"""Seeded functional neuron layout for the minimal SNN model.

These coordinates belong to the model: the engine derives per-afferent learning
distances from them (never charge delivery). The browser expands them for display
but must never feed display positions back here.

Populations and their homes:
    L1E_s[i]   3x3 sensory grid at z = 0            (the pixel sources)
    L1E_new[i] same grid lifted to z = +Z_OFFSET    (supervisory partners, above)
    L1I[i]     same grid dropped to z = -Z_OFFSET    (instant relays, below)
    L2E[j]     ring at z = L2_Z                       (eight competitors)
    PI[j]      just outside L2E[j] on the ring        (paired predictive interneurons)
    L2I        above the ring at z = L2I_Z            (one shared relay)

Both direct (PI) and comparison (L1E_new/L1I) populations are laid out here; only
the ones the active topology builds are read by the engine.

Small deterministic jitter (seeded) breaks exact distance ties so the geometric
learning-rate factor varies per synapse, while staying well inside each cell so
retinotopic ordering is preserved.
"""

from __future__ import annotations

import math

import numpy as np

GRID = 3.8
Z_OFFSET = 2.6            # L1E_new sits +Z above, L1I -Z below, each source pixel
L2_RING_R = 5.5
L2_Z = 8.0
L2I_Z = 12.0

_XY_JITTER = 0.35        # < half a cell: keeps row/col/diagonal ordering intact
_Z_JITTER = 0.25
_L2_Z_JITTER = 0.6


def _grid_anchor(i: int) -> np.ndarray:
    row, col = divmod(i, 3)
    return np.array([(col - 1) * GRID, (1 - row) * GRID, 0.0])


def generate_layout(rng, n_pix: int, n_out: int) -> dict[str, np.ndarray]:
    """Return deterministic functional coordinates for every neuron id."""
    pos: dict[str, np.ndarray] = {}

    def jitter(xy=_XY_JITTER, z=_Z_JITTER):
        return np.array([rng.uniform(-xy, xy), rng.uniform(-xy, xy), rng.uniform(-z, z)])

    for i in range(n_pix):
        anchor = _grid_anchor(i)
        pos[f'L1E{i}'] = anchor + jitter()
        pos[f'L1Enew{i}'] = anchor + np.array([0.0, 0.0, Z_OFFSET]) + jitter()
        pos[f'L1I{i}'] = anchor + np.array([0.0, 0.0, -Z_OFFSET]) + jitter()

    for j in range(n_out):
        angle = j * 2 * math.pi / n_out
        anchor = np.array([L2_RING_R * math.cos(angle), L2_RING_R * math.sin(angle), L2_Z])
        pos[f'L2E{j}'] = anchor + np.array([rng.uniform(-0.4, 0.4),
                                            rng.uniform(-0.4, 0.4),
                                            rng.uniform(-_L2_Z_JITTER, _L2_Z_JITTER)])
        # Paired predictive interneuron: just outside its L2E on the same ring.
        pi_anchor = np.array([(L2_RING_R + 2.2) * math.cos(angle),
                              (L2_RING_R + 2.2) * math.sin(angle), L2_Z])
        pos[f'PI{j}'] = pi_anchor + np.array([rng.uniform(-0.3, 0.3),
                                              rng.uniform(-0.3, 0.3),
                                              rng.uniform(-_L2_Z_JITTER, _L2_Z_JITTER)])

    pos['L2I'] = np.array([0.0, 0.0, L2I_Z]) + jitter(0.6, 0.6)
    return pos
