"""Seeded functional neuron layout for the minimal SNN model.

These coordinates belong to the model: the engine derives per-afferent learning
distances from them (never charge delivery). The browser expands them for display
but must never feed display positions back here.

The sensory sheet is a generic near-square grid of ``n_pix`` cells (width =
``ceil(sqrt(n_pix))``, centered), so the per-synapse learning distances are correct
for any input count. The 9-pixel default is exactly the historical 3x3.

Populations and their homes:
    RG[i]      same grid at z = -RG_Z               (retinal source layer, upstream)
    L1E_s[i]   n_pix sensory grid at z = 0          (the pixel sources; 3x3 at n_pix=9)
    ErrorE[i]  same grid at z = ERROR_Z              (parallel residual/error sheet)
    L1E_new[i] same grid lifted to z = +Z_OFFSET    (supervisory partners, above)
    L1I[i]     same grid dropped to z = -Z_OFFSET    (instant relays, below)
    L2E[j]     ring at z = L2_Z                       (n_out competitors)
    PI[j]      just outside L2E[j] on the ring        (paired predictive interneurons)
    SwitchI[j] outside PI[j] on the same ray           (paired local incumbent switch)
    L2I        above the ring at z = L2I_Z            (one shared relay)

Ordering constraint: RG coordinates are computed WITHOUT drawing from ``rng``. The
same generator goes on to draw the per-competitor feedforward jitter, so any extra
draw here would shift every downstream weight and break the bit-exact ``pi``/``old``
goldens. RG therefore sits on an exact grid plane; the RG_i -> L1E_i distances still
vary per synapse because the L1E end is jittered.

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
RG_Z = 6.0                # RG sits this far BELOW its pixel's L1E (upstream of L1I)
ERROR_Z = 3.2             # residual sheet between full-evidence L1 and categorical L2
L2_RING_R = 5.5
SWITCH_RING_R = L2_RING_R + 4.4
L2_Z = 8.0
L2I_Z = 12.0

_XY_JITTER = 0.35        # < half a cell: keeps row/col/diagonal ordering intact
_Z_JITTER = 0.25
_L2_Z_JITTER = 0.6


def grid_dims(n_pix: int) -> tuple[int, int]:
    """Near-square grid (width, height) holding ``n_pix`` cells: width first, so a
    perfect square is square and a non-square count grows the last row. n_pix=9 -> (3, 3)."""
    width = int(math.ceil(math.sqrt(n_pix)))
    height = int(math.ceil(n_pix / width)) if width else 0
    return width, height


def _grid_anchor(i: int, width: int, height: int) -> np.ndarray:
    """Centered grid position for cell ``i`` on a ``width`` x ``height`` sheet. At
    (width, height) == (3, 3) this is exactly the historical centered 3x3 anchor."""
    row, col = divmod(i, width)
    return np.array([(col - (width - 1) / 2.0) * GRID,
                     ((height - 1) / 2.0 - row) * GRID, 0.0])


def generate_layout(rng, n_pix: int, n_out: int) -> dict[str, np.ndarray]:
    """Return deterministic functional coordinates for every neuron id."""
    pos: dict[str, np.ndarray] = {}
    width, height = grid_dims(n_pix)

    def jitter(xy=_XY_JITTER, z=_Z_JITTER):
        return np.array([rng.uniform(-xy, xy), rng.uniform(-xy, xy), rng.uniform(-z, z)])

    for i in range(n_pix):
        anchor = _grid_anchor(i, width, height)
        pos[f'L1E{i}'] = anchor + jitter()
        pos[f'L1Enew{i}'] = anchor + np.array([0.0, 0.0, Z_OFFSET]) + jitter()
        pos[f'L1I{i}'] = anchor + np.array([0.0, 0.0, -Z_OFFSET]) + jitter()
        # L1C reuses the already-generated L1Enew functional position (the coincidence
        # partner sits above its pixel). This is a COPY, not a new RNG draw, so adding
        # the rg_coincidence preset never shifts pi/old/rg/rg_residual layout or goldens.
        pos[f'L1C{i}'] = pos[f'L1Enew{i}'].copy()

    # RG last and UNJITTERED: see the module docstring -- drawing here would shift the
    # competitor feedforward jitter and break the pi/old goldens.
    for i in range(n_pix):
        pos[f'RG{i}'] = _grid_anchor(i, width, height) + np.array([0.0, 0.0, -RG_Z])
        # New residual positions are deterministic and draw no RNG: adding the preset
        # must not shift pi/old/rg layout or weight-init goldens.
        pos[f'ErrorE{i}'] = _grid_anchor(i, width, height) + np.array([0.0, 0.0, ERROR_Z])

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
        # Also unjittered to avoid consuming an extra RNG draw for existing presets.
        pos[f'SwitchI{j}'] = np.array([SWITCH_RING_R * math.cos(angle),
                                       SWITCH_RING_R * math.sin(angle), L2_Z])

    pos['L2I'] = np.array([0.0, 0.0, L2I_Z]) + jitter(0.6, 0.6)
    return pos


# --- tiled cortical-column functional layout --------------------------------
# A SEPARATE, metadata-driven layout path. It never touches the legacy generator above,
# so adding it consumes no RNG draws in legacy construction and cannot shift any golden.
# It is parameterized entirely by column metadata + ordinary-E count N: no fixed "8 E",
# "two layers", or one-character layer-id assumption. Repeated columns reuse one local
# motif translated to a column center.
TILE_PX = 2.2             # RGC pixel spacing on the retinal plane
TILE_PATCH_GAP = 1.7      # extra gap between 3x3 patches (visible patch boundaries)
TILE_LAYER_DZ = 11.0      # z rise per column layer above the RGC plane
TILE_COL_SPREAD = 7.5     # column-center spacing within a layer's tile grid
TILE_E_RING_R = 1.7       # ordinary-E ring radius inside a column
TILE_ROLE_OFF = 2.9       # Eor/C/I offset from the column center
_TILE_JITTER = 0.18       # small seeded jitter (reproducible scientific state)


def _tile_axis(idx, count, patch, spacing, gap):
    """Centered coordinate for grid cell ``idx`` of ``count`` with an added ``gap`` every
    ``patch`` cells (so patch boundaries are visible)."""
    raw = idx * spacing + (idx // patch) * gap
    span = (count - 1) * spacing + ((count - 1) // patch) * gap
    return raw - span / 2.0


TILE_FEATURE_DZ = 4.4     # feature relay plane rise above the RGC surface
TILE_FEATURE_C_OFF = 1.0  # paired feature C offset (feedback side) from its relay
TILE_FEATURE_I_OFF = 1.0  # paired feature If offset (inhibitory side) from its relay


def _place_feature_gates(pos, spec, in_rows, in_cols, p_rows, p_cols, jit) -> None:
    """Place the feature-gated variant's relays/C/I: each feature relay S sits directly above
    its paired RGC pixel (one plane up), with its paired C offset to the feedback (+y) side
    and its paired If to the inhibitory (-z) side, so every relay->C->If->relay loop is
    visually associated yet separated enough to inspect the paired edges. Derived entirely
    from node metadata (input_row/input_col + column_role/feature_index), never from ids."""
    for n in spec['nodes']:
        role = n.get('column_role')
        if role not in ('S', 'C', 'If'):
            continue
        gr, gc = n.get('input_row'), n.get('input_col')
        if gr is None or gc is None:
            # C/If carry no pixel; derive the relay's pixel plane from the paired S below.
            continue
        x = _tile_axis(gc, in_cols, p_cols, TILE_PX, TILE_PATCH_GAP)
        y = -_tile_axis(gr, in_rows, p_rows, TILE_PX, TILE_PATCH_GAP)
        pos[n['id']] = np.array([x, y, TILE_FEATURE_DZ]) + jit(0.08)
    # C/If inherit their paired relay S position (by module + feature_index) plus an offset.
    relay_pos = {}
    for n in spec['nodes']:
        if n.get('column_role') == 'S':
            relay_pos[(n['column_id'], n.get('feature_index'))] = pos[n['id']]
    for n in spec['nodes']:
        role = n.get('column_role')
        if role not in ('C', 'If'):
            continue
        base = relay_pos.get((n.get('column_id'), n.get('feature_index')))
        if base is None:
            continue
        off = (np.array([0.0, TILE_FEATURE_C_OFF, 0.6]) if role == 'C'
               else np.array([0.0, -TILE_FEATURE_I_OFF, -0.6]))
        pos[n['id']] = base + off + jit(0.08)


def generate_tiled_layout(rng, spec) -> dict[str, np.ndarray]:
    """Deterministic functional coordinates for a tiled cortical-column graph, derived
    from its ``topology`` metadata and per-node ``column_*`` / ``patch_*`` tags.

    Layout: the RGC surface is one plane (z=0) with visible patch gaps; each column
    layer rises by ``TILE_LAYER_DZ``; columns sit above their tile coordinates; inside
    every column the ordinary E form a ring with Eor toward the next layer, C on the
    feedback side and I on the inhibitory side."""
    meta = spec['topology']
    ishape, pshape = meta['input_shape'], meta['patch_shape']
    in_rows, in_cols = ishape['rows'], ishape['cols']
    p_rows, p_cols = pshape['rows'], pshape['cols']
    # layer order (RGC-adjacent first) -> z index used for the column-center height.
    layer_z = {}
    for k, layer in enumerate(meta.get('column_layers', [])):
        layer_z[layer['layer']] = (k + 1) * TILE_LAYER_DZ

    def jit(scale=_TILE_JITTER):
        return np.array([rng.uniform(-scale, scale) for _ in range(3)])

    pos: dict[str, np.ndarray] = {}
    node_by_id = {n['id']: n for n in spec['nodes']}

    # RGC plane.
    for n in spec['nodes']:
        if n['archetype'] != 'rg_source':
            continue
        gr, gc = n['input_row'], n['input_col']
        x = _tile_axis(gc, in_cols, p_cols, TILE_PX, TILE_PATCH_GAP)
        y = -_tile_axis(gr, in_rows, p_rows, TILE_PX, TILE_PATCH_GAP)
        pos[n['id']] = np.array([x, y, 0.0]) + jit(0.05)

    # Feature-gated variant: place the feature relays directly above their paired RGC, with
    # the paired C/I visibly associated but offset for edge inspection, then the competitor
    # banks/L2 via the shared column placement below.
    if meta.get('variant') == 'feature_gated':
        _place_feature_gates(pos, spec, in_rows, in_cols, p_rows, p_cols, jit)

    # column layout: gather the competitor-bank members per column (E ring + Eor + WTA I).
    # Feature relays/C/I (roles S/C/If in the feature-gated variant) are placed above by
    # ``_place_feature_gates`` and skipped here; only the classic single column C (role 'C'
    # with no feature_index) is placed as a column role below.
    columns = {c['id']: c for c in meta['columns']}
    members = {cid: dict(E=[], Eor=None, C=None, I=None) for cid in columns}
    for n in spec['nodes']:
        role = n.get('column_role')
        if role is None or role in ('S', 'If'):
            continue
        if role == 'C' and n.get('feature_index') is not None:
            continue                                  # a feature C, already placed above
        slot = members[n['column_id']]
        if role == 'E':
            slot['E'].append(n['id'])
        else:
            slot[role] = n['id']

    # column-layer tile extents, so each layer's columns are centered over the surface.
    layer_grid = {L['layer']: (L['rows'], L['cols']) for L in meta.get('column_layers', [])}
    for cid, c in columns.items():
        layer = c['layer']
        gr, gc = layer_grid.get(layer, (1, 1))
        cx = _tile_axis(c['col'], gc, max(gc, 1), TILE_COL_SPREAD, 0.0)
        cy = -_tile_axis(c['row'], gr, max(gr, 1), TILE_COL_SPREAD, 0.0)
        cz = layer_z.get(layer, TILE_LAYER_DZ)
        center = np.array([cx, cy, cz])
        e_ids = members[cid]['E']
        n_e = max(len(e_ids), 1)
        for i, eid in enumerate(e_ids):
            angle = 2.0 * math.pi * i / n_e
            pos[eid] = center + np.array([TILE_E_RING_R * math.cos(angle),
                                          TILE_E_RING_R * math.sin(angle),
                                          0.0]) + jit()
        # Eor and I both sit on the ring's central axis (the middle of the ordinary-E
        # pool) on opposite sides: Eor toward the next layer (+z), I on the far side
        # (-z). C stays off to the feedback (+y) side. All distinct and non-coincident
        # with the ring. Display only -- the I neuron has no feedforward edges, so its
        # position never enters the 1/d^2 learning-rate factor.
        pos[members[cid]['Eor']] = center + np.array([0.0, 0.0, TILE_ROLE_OFF]) + jit()
        if members[cid]['C'] is not None:            # classic column C (absent in feature-gated)
            pos[members[cid]['C']] = center + np.array(
                [TILE_ROLE_OFF, TILE_ROLE_OFF, 0.9]) + jit()
        pos[members[cid]['I']] = center + np.array([0.0, 0.0, -TILE_ROLE_OFF]) + jit()

    # Any node the metadata did not place (defensive) gets a deterministic offset rather
    # than the zero placeholder.
    for n in spec['nodes']:
        if n['id'] not in pos:
            pos[n['id']] = np.array([0.0, 0.0, -3.0]) + jit()
    return pos
