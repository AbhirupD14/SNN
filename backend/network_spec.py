"""Declarative network topology: a **NetworkSpec** (typed nodes + typed edges) the
engine builds and runs generically, the built-in presets, and validation.

A spec is JSON-serializable and is the single source of truth the editor and the
engine share::

    {
      "name": str,
      "nodes": [ {"id", "archetype", "layer", "pos":[x,y,z]|None, "pixel":int|None} ],
      "edges": [ {"id", "source", "target", "kind", "directed":bool, "sign":int|None} ],
    }

The **archetypes** (neuron types) and the **edge kinds** are the fixed rule
vocabulary; the editor composes arbitrary graphs from them, but cannot invent new
neuron behaviours or learning rules. Defaults per archetype mirror the values the
hardcoded engine used, so a preset spec rebuilds byte-identical behaviour.

Intrinsic population rules (NOT edges, not user-editable):
  * ``rg_source``   -- exogenous binary spike source. It is NOT an integrator: it
                       emits a spike on every input boundary on which its owned pixel
                       is active, and nothing in the modelled cortex can stop it.
  * ``e_sensory``   -- every threshold crosser fires (sensory sources); one fixed,
                       non-plastic external afferent.
  * ``e_encoder``   -- plastic NONCOMPETITIVE excitatory accumulator: owns plastic
                       feedforward afferents and learns with the same accumulating
                       rule as a competitor, but every threshold crosser fires (no WTA).
  * ``e_residual``  -- noncompetitive excitatory residual/error integrator. It owns
                       no plastic afferents; fixed evidence-copy edges drive it and
                       predictive inhibitory conductance shunts explained features.
  * ``e_competitor``-- deterministic winner-take-all: one winner per boundary fires
                       and learns its feedforward weights.
  * ``switch``      -- inhibitory temporal-AND relay with a local decaying eligibility
                       trace written only by its paired competitor's real spikes.
Everything else -- who inhibits whom, who relays to whom, predictive inhibition --
is expressed by edges and is fully editable.

Node ``pixel`` is the *external-input ownership* claim: exactly one cell may own a
given pixel, and only an input-sink archetype (``e_sensory``, ``rg_source``) may claim
one. Node ``grid`` is *display / receptive-field* metadata (which sensory-grid cell a
unit represents; the 9-pixel default sheet is 3x3); it is not unique and carries no
input. An ``e_encoder`` downstream of an
RG cell uses ``grid`` so the receptive-field view can still place it, while the RG cell
that actually receives the pixel owns ``pixel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Inhibitory firing threshold as a fraction of the excitatory threshold theta. This is a
# FIXED gain (a relay fires on ~a third of theta of coincident drive), not a count-derived
# quantity: it does not scale with n_pix or n_out.
I_THRESHOLD_FRAC = 1.0 / 3.0

# The tiled cortical-column preset family tag (top-level topology metadata). The engine,
# layout, and validation switch on this metadata -- NEVER on the preset name or node ids.
TILED_FAMILY = 'tiled_cortical_columns'

# Recognized structural metadata that must survive spec normalization / editor round-trips.
# Node fields the tiled builder attaches (in addition to the legacy id/archetype/layer/
# label/pixel/grid/pos). Simulation must read these instead of parsing ids.
_TILED_NODE_INT_FIELDS = (
    'column_index', 'column_row', 'column_col',
    'input_row', 'input_col', 'patch_id', 'patch_row', 'patch_col',
    'patch_local_row', 'patch_local_col', 'patch',
)
_TILED_NODE_STR_FIELDS = ('column_id', 'column_role')
# ``has_parent`` declares a dormant top C; ``multi_basal`` is the explicit per-node opt-in
# to a coincidence cell owning MORE than one learned basal afferent. Without it the
# historical exactly-one-basal invariant still applies, so no existing graph is relaxed.
_TILED_NODE_BOOL_FIELDS = ('has_parent', 'multi_basal')
# Edge projection-family field: metadata for validation / layout / dashboard filtering.
# It is NEVER a second delivery mechanism; the edge ``kind`` stays authoritative.
_EDGE_META_STR_FIELDS = ('projection',)

# --- Node archetypes ---------------------------------------------------------
# cls: 'E' excitatory (conductance LIF) | 'I' inhibitory relay | 'S' exogenous source.
# thr_frac: firing threshold as a fraction of the excitatory threshold theta.
# plastic_ff: this archetype owns plastic feedforward afferents and learns on its own spike.
# wta: this archetype competes in the LEGACY deterministic single-winner arbitration.
# event_resolved: this archetype is resolved by the analytic sub-boundary event
#   scheduler (crossing-time latency) rather than the legacy synchronous WTA/fire path.
#   Legacy archetypes are False; the three coincidence-topology archetypes are True.
# input_sink: this archetype may own an external pixel.
ARCHETYPES = {
    'rg_source':    dict(cls='S', role='rg_source',  thr_frac=1.0,
                         plastic_ff=False, wta=False, event_resolved=False, input_sink=True,
                         desc='Retinal ganglion cell: exogenous binary spike source, one '
                              'per pixel. Spikes on every input boundary its pixel is '
                              'active; owns no membrane and cannot be inhibited.'),
    'e_sensory':    dict(cls='E', role='source',     thr_frac=1.0,
                         plastic_ff=False, wta=False, event_resolved=False, input_sink=True,
                         desc='Sensory excitatory source: one fixed input afferent; '
                              'fires on every threshold crossing (no WTA).'),
    'e_encoder':    dict(cls='E', role='encoder',    thr_frac=1.0,
                         plastic_ff=True, wta=False, event_resolved=False, input_sink=False,
                         desc='Plastic noncompetitive excitatory encoder: learns its '
                              'feedforward afferents with the shared accumulating rule; '
                              'every threshold crosser fires (no WTA).'),
    'e_residual':   dict(cls='E', role='residual',   thr_frac=1.0,
                         plastic_ff=False, wta=False, event_resolved=False, input_sink=False,
                         desc='Noncompetitive excitatory residual cell: receives a fixed '
                              'evidence copy, is shunted by learned prediction, and '
                              'broadcasts unexplained feature events to switch cells.'),
    'e_competitor': dict(cls='E', role='competitor', thr_frac=1.0,
                         plastic_ff=True, wta=True, event_resolved=False, input_sink=False,
                         desc='Legacy competitor excitatory unit: plastic feedforward '
                              'weights, deterministic engine-arbitrated winner-take-all '
                              '(one winner per boundary fires + learns).'),
    'e_pretrained': dict(cls='E', role='pretrained', thr_frac=1.0,
                         plastic_ff=False, wta=False, event_resolved=True, input_sink=False,
                         desc='Fixed-input noncompetitive excitatory relay: a fixed '
                              'pretrained excitation packet makes one unobstructed source '
                              'spike fire it on the delivery boundary; owns no learned '
                              'weight; resolved by the analytic event scheduler.'),
    'e_coincidence': dict(cls='E', role='coincidence', thr_frac=1.0,
                         plastic_ff=False, wta=False, event_resolved=True, input_sink=False,
                         desc='Coincidence pyramidal cell: one learned basal afferent and '
                              'unweighted Boolean apical afferents; deposits gated basal '
                              'charge only on basal/apical coincidence; event-resolved.'),
    'e_latency_competitor': dict(cls='E', role='competitor', thr_frac=1.0,
                         plastic_ff=True, wta=False, event_resolved=True, input_sink=False,
                         desc='Latency competitor: same plastic feedforward bank and '
                              'accumulating rule as a legacy competitor, but competes by '
                              'first-spike latency + an inhibitory reset loop instead of '
                              'the deterministic WTA list.'),
    'i_relay':      dict(cls='I', role='relay',      thr_frac=I_THRESHOLD_FRAC,
                         plastic_ff=False, wta=False, event_resolved=False, input_sink=False,
                         desc='Inhibitory relay: fires the same boundary it receives any '
                              'excitatory relay event; emits a persistent conductance pulse '
                              '(legacy) or an immediate hard reset (hard_reset_inhibition).'),
    'predictor':    dict(cls='I', role='predictor',  thr_frac=I_THRESHOLD_FRAC,
                         plastic_ff=False, wta=False, event_resolved=False, input_sink=False,
                         desc='Predictive interneuron: relays its driver and owns locally '
                              'plastic inhibitory output weights onto its targets.'),
    'switch':       dict(cls='I', role='switch',     thr_frac=I_THRESHOLD_FRAC,
                         plastic_ff=False, wta=False, event_resolved=False, input_sink=False,
                         desc='Incumbent switch interneuron: strict temporal AND between '
                              'broadcast residual events and a local decaying trace from '
                              'its paired competitor; inhibits only that competitor.'),
}

# Archetypes that own plastic feedforward afferents (valid ``feedforward`` targets).
PLASTIC_FF_TARGETS = tuple(a for a, d in ARCHETYPES.items() if d['plastic_ff'])
# Archetypes whose spikes may drive a feedforward synapse.
FF_SOURCE_CLASSES = ('E', 'S')
# Archetypes that may own an external pixel.
INPUT_SINKS = tuple(a for a, d in ARCHETYPES.items() if d['input_sink'])
# Archetypes resolved by the analytic sub-boundary event scheduler.
EVENT_RESOLVED_ARCHETYPES = tuple(a for a, d in ARCHETYPES.items() if d['event_resolved'])
# Legacy excitatory archetypes that use the deterministic engine WTA / synchronous path.
LEGACY_E_ARCHETYPES = tuple(a for a, d in ARCHETYPES.items()
                            if d['cls'] == 'E' and not d['event_resolved'])

# --- Edge kinds and which archetypes they may connect -------------------------
# src/tgt requirements are an archetype name, a class letter ('E'/'I'/'S'), or a tuple
# of either. Note ``inhibition``/``predictive_inhibition`` require an 'E' target, which
# is what structurally forbids any inhibitory edge onto an 'S' (rg_source) cell.
EDGE_KINDS = {
    'feedforward':          dict(src=FF_SOURCE_CLASSES, tgt=PLASTIC_FF_TARGETS,
                                 plastic=True, sign=+1,
                                 desc='Plastic excitatory feedforward onto a competitor or '
                                      'encoder (accumulating signed-spike rule).'),
    'relay_excitation':     dict(src='E', tgt='I', plastic=False, sign=+1,
                                 desc='Structural +1 event driving an inhibitory relay / '
                                      'predictor (no learned magnitude).'),
    'fixed_excitation':     dict(src='E', tgt='e_residual', plastic=False, sign=+1,
                                 desc='Fixed excitatory evidence copy onto a residual cell; '
                                      'one presynaptic spike schedules a fixed charge pulse.'),
    'trace_excitation':     dict(src='e_competitor', tgt='switch', plastic=False, sign=+1,
                                 desc='Paired local eligibility event: a real competitor '
                                      'spike sets only its switch cell trace x_j.'),
    'inhibition':           dict(src=('i_relay', 'switch'), tgt='E', plastic=False, sign=-1,
                                 desc='Persistent inhibitory conductance pulse from a relay '
                                      'onto an excitatory target (fixed scale).'),
    'predictive_inhibition': dict(src='predictor', tgt='E', plastic=True, sign=-1,
                                 desc='Locally plastic predictive inhibitory conductance '
                                      'from a predictor onto an excitatory target.'),
    'pretrained_excitation': dict(src='S', tgt='e_pretrained', plastic=False, sign=+1,
                                 desc='Fixed (non-learned) excitatory packet from a source '
                                      'onto a pretrained relay; one spike fires the target '
                                      'on its delivery boundary.'),
    'basal_excitation':     dict(src='E', tgt='e_coincidence', plastic=True, sign=+1,
                                 desc='Single learned basal afferent onto a coincidence '
                                      'cell (the only plastic C weight).'),
    'apical_excitation':    dict(src='E', tgt='e_coincidence', plastic=False, sign=+1,
                                 desc='Unweighted structural apical afferent onto a '
                                      'coincidence cell (a Boolean permission gate).'),
    'hard_reset_inhibition': dict(src='i_relay', tgt='E', plastic=False, sign=-1,
                                 desc='Immediate zero-latency hard reset from a relay onto '
                                      'an excitatory target at the driver spike tau (wipes '
                                      'V and discards remaining drive; not a conductance).'),
}

# Edge kinds that are inherently one-way structural projections: a bidirectional
# gesture is always a modelling error and is rejected outright (not merely by the
# reverse-archetype check, which some symmetric E->E gestures could slip past).
DIRECTED_ONLY_KINDS = ('pretrained_excitation', 'basal_excitation', 'apical_excitation',
                       'relay_excitation', 'hard_reset_inhibition')

# The public built-in presets. The obsolete pi/old/rg/rg_residual graphs are no longer
# offered as built-ins (their spec builders remain in this module only as reusable
# low-level mechanics for custom/saved graphs and unit tests, NOT as public presets).
PRESETS = ('rg_coincidence', 'tiled_cc', 'tiled_cc_l1_4', 'tiled_cc_direct_identity',
           'tiled_cc_double_eor', 'rg_direct_cc4', 'two_tower_composition')

# Built-in presets that build a tiled cortical-column hierarchy on the fixed 81-pixel
# surface (as opposed to the legacy n_pix/n_out presets). Used for size resolution and
# the fixed-input guard so new tiled presets are handled without name-by-name branching.
TILED_PRESETS = ('tiled_cc', 'tiled_cc_l1_4', 'tiled_cc_direct_identity',
                 'tiled_cc_double_eor', 'two_tower_composition')

# Tiled-family topology variants. The top-level ``topology.variant`` field is the single
# source of truth construction/validation/layout branch on -- NEVER the preset name or a
# node-id prefix. ``classic`` (the whole-bank Eor/C/I column) is the DEFAULT when the field
# is absent, so saved specs that omit it stay byte-identical. ``direct_identity`` removes
# Eor: each ordinary E projects to every parent ordinary E (source-addressed identity) and
# the column C owns one learned basal afferent per local ordinary E.
TILED_VARIANT_CLASSIC = 'classic'
TILED_VARIANT_DIRECT_IDENTITY = 'direct_identity'
# DIAGNOSTIC variant. Identical to ``classic`` except a SECOND output relay is inserted in
# series (E -> Eor -> Eor2 -> parent E), which lengthens the top-down confirmation loop by
# exactly one boundary and nothing else. It exists to test the prediction that the observed
# feedback cadence is set by loop latency (period = 2 x latency), not by learning rates or
# seed. Not a research topology -- see docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md section 8.1b.
TILED_VARIANT_DOUBLE_EOR = 'double_eor'
TILED_VARIANTS = (TILED_VARIANT_CLASSIC, TILED_VARIANT_DIRECT_IDENTITY,
                  TILED_VARIANT_DOUBLE_EOR)

# Canonical construction dimensions for the tiled cortical-column preset. The input
# surface is fixed; only ``cc_e_count`` (ordinary E per column) is configurable.
TILED_CC_DEFAULTS = dict(input_rows=9, input_cols=9, patch_rows=3, patch_cols=3,
                         cc_e_count=8)
# Construction dimensions for the two-tower composition preset: two lateral 9x9 fields on
# ONE validated 9x18 sheet. See ``two_tower_composition_spec`` below.
TWO_TOWER_TOWER_COUNT = 2
TWO_TOWER_DEFAULTS = dict(input_rows=9, input_cols=18, patch_rows=3, patch_cols=3,
                          cc_e_count=8)

# Per-preset input surface for the tiled family. A tiled preset's pixel count is a
# CONSTRUCTION property of that preset, not one shared constant -- the two-tower graph is
# 162 pixels while every 9x9 variant is 81. Size resolution, the fixed-input guard and the
# preset store all read this table instead of assuming a single tiled surface.
TILED_PRESET_INPUT_SHAPE = {
    'tiled_cc': (TILED_CC_DEFAULTS['input_rows'], TILED_CC_DEFAULTS['input_cols']),
    'tiled_cc_l1_4': (TILED_CC_DEFAULTS['input_rows'], TILED_CC_DEFAULTS['input_cols']),
    'tiled_cc_direct_identity': (TILED_CC_DEFAULTS['input_rows'],
                                 TILED_CC_DEFAULTS['input_cols']),
    'tiled_cc_double_eor': (TILED_CC_DEFAULTS['input_rows'],
                            TILED_CC_DEFAULTS['input_cols']),
    'two_tower_composition': (TWO_TOWER_DEFAULTS['input_rows'],
                              TWO_TOWER_DEFAULTS['input_cols']),
}


def tiled_preset_input_shape(name: str) -> tuple[int, int]:
    """``(rows, cols)`` of one tiled preset's fixed input sheet. Raises for a name that is
    not a tiled preset, so a new tiled preset cannot silently inherit 9x9."""
    try:
        return TILED_PRESET_INPUT_SHAPE[name]
    except KeyError:
        raise KeyError(f'{name!r} is not a tiled preset; expected one of {TILED_PRESETS}')


def tiled_preset_input_size(name: str) -> int:
    """Pixel count of one tiled preset's fixed input sheet (81 for the 9x9 family, 162 for
    the two-tower 9x18 sheet)."""
    rows, cols = tiled_preset_input_shape(name)
    return rows * cols


def _e_sensory(i, pixel):
    return dict(id=f'L1E{i}', archetype='e_sensory', layer='L1', pixel=pixel,
                label=f'L1E_s{i}')


def preset_spec(name: str, n_pix: int, n_out: int, cc_e_count: int = 8) -> dict:
    """Return a built-in NetworkSpec. Positions are omitted so the engine fills them from
    the seeded functional layout (bit-exact presets).

    The public built-ins are exactly the seven names in ``PRESETS``; any other name is
    rejected. ``cc_e_count`` sizes the ordinary-E bank of every column in the configurable
    ``tiled_cc`` preset and is ignored by the fixed-shape variants. The obsolete
    pi/old/rg/rg_residual builders below are unreachable through this entry point (guarded
    by ``PRESETS``) and remain only as reusable low-level graph mechanics."""
    if name not in PRESETS:
        raise ValueError(f'unknown preset {name!r}')
    if name == 'tiled_cc':
        return tiled_cc_spec(cc_e_count=cc_e_count)
    if name == 'tiled_cc_l1_4':
        # Identical to tiled_cc but with a shallower L1 bank: four ordinary E per L1
        # column, eight in L2. Fixed-shape (does not read cc_e_count).
        return tiled_cc_spec(l1_e_count=4, l2_e_count=8, name='tiled_cc_l1_4')
    if name == 'tiled_cc_double_eor':
        # Diagnostic latency probe: classic column + one extra output relay in series.
        return tiled_cc_double_eor_spec()
    if name == 'tiled_cc_direct_identity':
        # Eor-less direct-identity variant: 8 E + multi-basal C + WTA I per column, every
        # child E addressed individually at L2. Fixed-shape (does not read cc_e_count).
        return tiled_cc_direct_identity_spec()
    if name == 'rg_direct_cc4':
        # Direct 3x3 RGC -> four ordinary E + one central WTA I (dual FE/FES experiment).
        # Fixed 3x3/4-E shape (does not read n_out); n_pix sizes the RGC surface.
        return rg_direct_cc4_spec(n_pix=n_pix)
    if name == 'two_tower_composition':
        # Two 9x9 towers on one 9x18 sheet feeding one L3 composition column. Composes the
        # same public column rules as tiled_cc; ``cc_e_count`` sizes every column bank.
        return two_tower_composition_spec(cc_e_count=cc_e_count)
    if name == 'rg':
        return _rg_spec(n_pix, n_out)
    if name == 'rg_residual':
        return _rg_residual_spec(n_pix, n_out)
    if name == 'rg_coincidence':
        return _rg_coincidence_spec(n_pix, n_out)
    # Node order is the serialization / display order and MUST match the historical
    # order [sensory, L1-inhibitory / predictors, competitors, L2I] so presets stay
    # byte-identical. (Only competitors draw RNG, so this order does not affect init.)
    nodes = [_e_sensory(i, i) for i in range(n_pix)]
    if name == 'old':
        for i in range(n_pix):
            nodes.append(dict(id=f'L1I{i}', archetype='i_relay', layer='L1', label=f'L1I{i}'))
    else:   # pi
        for j in range(n_out):
            nodes.append(dict(id=f'PI{j}', archetype='predictor', layer='L2', label=f'PI{j}'))
    for j in range(n_out):
        nodes.append(dict(id=f'L2E{j}', archetype='e_competitor', layer='L2', label=f'L2E{j}'))
    nodes.append(dict(id='L2I', archetype='i_relay', layer='L2', label='L2I'))

    edges = []

    def E(eid, src, tgt, kind, sign=None):
        e = dict(id=eid, source=src, target=tgt, kind=kind, directed=True)
        if sign is not None:
            e['sign'] = sign
        edges.append(e)

    # feedforward L1E -> L2E (dense), and each L2E -> L2I (WTA relay driver).
    for j in range(n_out):
        for i in range(n_pix):
            E(f'ff{i}->{j}', f'L1E{i}', f'L2E{j}', 'feedforward')
    for j in range(n_out):
        E(f're_l2_{j}', f'L2E{j}', 'L2I', 'relay_excitation')
    for j in range(n_out):
        E(f'inh_l2_{j}', 'L2I', f'L2E{j}', 'inhibition', sign=-1)

    if name == 'old':
        # dense L2E -> L1I (winner drives all), paired L1I[i] -> L1E[i] inhibition.
        for j in range(n_out):
            for i in range(n_pix):
                E(f're_l1i_{j}->{i}', f'L2E{j}', f'L1I{i}', 'relay_excitation')
        for i in range(n_pix):
            E(f'inh_l1_{i}', f'L1I{i}', f'L1E{i}', 'inhibition', sign=-1)
    else:   # pi
        for j in range(n_out):
            E(f're_pi_{j}', f'L2E{j}', f'PI{j}', 'relay_excitation')   # paired 1:1
        for j in range(n_out):
            for i in range(n_pix):
                E(f'pi{j}->{i}', f'PI{j}', f'L1E{i}', 'predictive_inhibition', sign=-1)

    return dict(name=name, nodes=nodes, edges=edges)


def _rg_spec(n_pix: int, n_out: int) -> dict:
    """The 'rg' preset: the `old` cortical topology with an explicit retinal-ganglion
    source layer spliced in ahead of L1.

        RG_i  --ff(plastic, 1:1)-->  L1E_i  --ff(plastic, dense)-->  L2E_j
                                       ^                               |
                                       |                        L2E_j --> L2I --> all L2E
                                       |                               |
                    L1I_i --inhibition(paired)-- L1I_i  <--relay(dense)-+

    The pixel is owned by RG (an ``rg_source``); L1E becomes a plastic noncompetitive
    ``e_encoder`` with exactly one afferent and keeps a ``grid`` tag for display. The
    external->L1E direct injection of `old` is therefore GONE in this preset: every
    L1E charge is a real, weighted, delay-1 RG spike. The cortical half (L1I, L2E,
    L2I and all their edges) is identical to `old`.

    9 RG + 9 L1E + 9 L1I + 8 L2E + 1 L2I = 36 nodes.
    9 + 72 + 8 + 8 + 72 + 9 = 178 internal edges.
    """
    nodes = [dict(id=f'RG{i}', archetype='rg_source', layer='RG', pixel=i, label=f'RG{i}')
             for i in range(n_pix)]
    # L1E keeps its id (so the seeded functional layout and the RF view still find it)
    # but is an encoder here: it owns a plastic afferent instead of a fixed external one.
    nodes += [dict(id=f'L1E{i}', archetype='e_encoder', layer='L1', grid=i, label=f'L1E{i}')
              for i in range(n_pix)]
    nodes += [dict(id=f'L1I{i}', archetype='i_relay', layer='L1', label=f'L1I{i}')
              for i in range(n_pix)]
    nodes += [dict(id=f'L2E{j}', archetype='e_competitor', layer='L2', label=f'L2E{j}')
              for j in range(n_out)]
    nodes.append(dict(id='L2I', archetype='i_relay', layer='L2', label='L2I'))

    edges = []

    def E(eid, src, tgt, kind, sign=None):
        e = dict(id=eid, source=src, target=tgt, kind=kind, directed=True)
        if sign is not None:
            e['sign'] = sign
        edges.append(e)

    for i in range(n_pix):                                   # 9 paired RG -> L1E
        E(f'rg{i}->l1e{i}', f'RG{i}', f'L1E{i}', 'feedforward')
    for j in range(n_out):                                   # 72 dense L1E -> L2E
        for i in range(n_pix):
            E(f'ff{i}->{j}', f'L1E{i}', f'L2E{j}', 'feedforward')
    for j in range(n_out):                                   # 8 L2E -> L2I
        E(f're_l2_{j}', f'L2E{j}', 'L2I', 'relay_excitation')
    for j in range(n_out):                                   # 8 L2I -> L2E (WTA)
        E(f'inh_l2_{j}', 'L2I', f'L2E{j}', 'inhibition', sign=-1)
    for j in range(n_out):                                   # 72 dense L2E -> L1I
        for i in range(n_pix):
            E(f're_l1i_{j}->{i}', f'L2E{j}', f'L1I{i}', 'relay_excitation')
    for i in range(n_pix):                                   # 9 paired L1I -> L1E
        E(f'inh_l1_{i}', f'L1I{i}', f'L1E{i}', 'inhibition', sign=-1)

    return dict(name='rg', nodes=nodes, edges=edges)


def _rg_residual_spec(n_pix: int, n_out: int) -> dict:
    """Classification-preserving RG residual/error topology.

    The main evidence path remains uninhibited::

        RG_i --ff(plastic, 1:1)--> L1E_i --ff(plastic, dense)--> L2E_j

    A parallel fixed copy drives ErrorE. Paired PI cells learn predictive inhibition
    onto ErrorE (never L1E). Unexplained ErrorE spikes broadcast to all SwitchI cells;
    only a switch carrying its paired L2E's pre-existing local trace may fire and
    inhibit that incumbent. L2I remains the shared deterministic WTA relay.

    52 cells: 9 RG + 9 L1E + 9 ErrorE + 8 L2E + 8 PI + 8 SwitchI + 1 L2I.
    274 directed projections: 9+9+72+8+72+72+8+8+8+8.
    """
    nodes = [dict(id=f'RG{i}', archetype='rg_source', layer='RG', pixel=i, label=f'RG{i}')
             for i in range(n_pix)]
    nodes += [dict(id=f'L1E{i}', archetype='e_encoder', layer='L1', grid=i, label=f'L1E{i}')
              for i in range(n_pix)]
    nodes += [dict(id=f'ErrorE{i}', archetype='e_residual', layer='ERR', grid=i,
                   label=f'ErrorE{i}') for i in range(n_pix)]
    nodes += [dict(id=f'L2E{j}', archetype='e_competitor', layer='L2', label=f'L2E{j}')
              for j in range(n_out)]
    nodes += [dict(id=f'PI{j}', archetype='predictor', layer='L2', label=f'PI{j}')
              for j in range(n_out)]
    nodes += [dict(id=f'SwitchI{j}', archetype='switch', layer='L2', label=f'SwitchI{j}')
              for j in range(n_out)]
    nodes.append(dict(id='L2I', archetype='i_relay', layer='L2', label='L2I'))

    edges = []

    def E(eid, src, tgt, kind, sign=None):
        edge = dict(id=eid, source=src, target=tgt, kind=kind, directed=True)
        if sign is not None:
            edge['sign'] = sign
        edges.append(edge)

    for i in range(n_pix):
        E(f'rg{i}->l1e{i}', f'RG{i}', f'L1E{i}', 'feedforward')
        E(f'l1e{i}->error{i}', f'L1E{i}', f'ErrorE{i}', 'fixed_excitation')
    for j in range(n_out):
        for i in range(n_pix):
            E(f'ff{i}->{j}', f'L1E{i}', f'L2E{j}', 'feedforward')
        E(f're_pi_{j}', f'L2E{j}', f'PI{j}', 'relay_excitation')
        for i in range(n_pix):
            E(f'pi{j}->error{i}', f'PI{j}', f'ErrorE{i}',
              'predictive_inhibition', sign=-1)
        for i in range(n_pix):
            E(f'error{i}->switch{j}', f'ErrorE{i}', f'SwitchI{j}', 'relay_excitation')
        E(f'trace_{j}', f'L2E{j}', f'SwitchI{j}', 'trace_excitation')
        E(f'switch_inh_{j}', f'SwitchI{j}', f'L2E{j}', 'inhibition', sign=-1)
        E(f're_l2_{j}', f'L2E{j}', 'L2I', 'relay_excitation')
        E(f'inh_l2_{j}', 'L2I', f'L2E{j}', 'inhibition', sign=-1)

    return dict(name='rg_residual', nodes=nodes, edges=edges)


def _rg_coincidence_spec(n_pix: int, n_out: int) -> dict:
    """The 'rg_coincidence' preset: coincidence pyramidal cells with an event-resolved
    latency-WTA L2, and immediate hard-reset inhibition (no conductance inhibition).

        RG_i --pretrained, paired--> L1E_i --ff, dense--> L2E_j
                                       |                     |
                             basal, paired v            apical, dense v
                                        L1C_i <----------------+
                                          | relay, paired
                                        L1I_i --hard reset, paired--> L1E_i

        L2E_j --relay--> L2I --hard reset--> every L2E_j   (emergent latency WTA)

    L1E is a fixed pretrained relay (one RG spike fires it next boundary); L1C is a
    coincidence cell with one learned basal (from its paired L1E) and eight unweighted
    apical afferents (one per L2E). Every inhibitory cell is an immediate zero-latency
    hard-reset relay.

    9 RG + 9 L1E + 9 L1C + 9 L1I + 8 L2E + 1 L2I = 45 nodes.
    9 + 72 + 9 + 72 + 9 + 9 + 8 + 8 = 196 directed edges.
    """
    nodes = [dict(id=f'RG{i}', archetype='rg_source', layer='RG', pixel=i, label=f'RG{i}')
             for i in range(n_pix)]
    nodes += [dict(id=f'L1E{i}', archetype='e_pretrained', layer='L1', grid=i,
                   label=f'L1E{i}') for i in range(n_pix)]
    nodes += [dict(id=f'L1C{i}', archetype='e_coincidence', layer='L1', grid=i,
                   label=f'L1C{i}') for i in range(n_pix)]
    nodes += [dict(id=f'L1I{i}', archetype='i_relay', layer='L1', label=f'L1I{i}')
              for i in range(n_pix)]
    nodes += [dict(id=f'L2E{j}', archetype='e_latency_competitor', layer='L2',
                   label=f'L2E{j}') for j in range(n_out)]
    nodes.append(dict(id='L2I', archetype='i_relay', layer='L2', label='L2I'))

    edges = []

    def E(eid, src, tgt, kind, sign=None):
        edge = dict(id=eid, source=src, target=tgt, kind=kind, directed=True)
        if sign is not None:
            edge['sign'] = sign
        edges.append(edge)

    for i in range(n_pix):                                   # 9 paired RG -> L1E
        E(f'rg{i}->l1e{i}', f'RG{i}', f'L1E{i}', 'pretrained_excitation')
    for j in range(n_out):                                   # 72 dense L1E -> L2E
        for i in range(n_pix):
            E(f'ff{i}->{j}', f'L1E{i}', f'L2E{j}', 'feedforward')
    for i in range(n_pix):                                   # 9 paired L1E -> L1C (basal)
        E(f'basal{i}', f'L1E{i}', f'L1C{i}', 'basal_excitation')
    for i in range(n_pix):                                   # 72 dense L2E -> L1C (apical)
        for j in range(n_out):
            E(f'apical{j}->{i}', f'L2E{j}', f'L1C{i}', 'apical_excitation')
    for i in range(n_pix):                                   # 9 paired L1C -> L1I (relay)
        E(f're_l1i_{i}', f'L1C{i}', f'L1I{i}', 'relay_excitation')
    for i in range(n_pix):                                   # 9 paired L1I -> L1E (reset)
        E(f'hr_l1_{i}', f'L1I{i}', f'L1E{i}', 'hard_reset_inhibition', sign=-1)
    for j in range(n_out):                                   # 8 L2E -> L2I (relay)
        E(f're_l2_{j}', f'L2E{j}', 'L2I', 'relay_excitation')
    for j in range(n_out):                                   # 8 L2I -> L2E (reset)
        E(f'hr_l2_{j}', 'L2I', f'L2E{j}', 'hard_reset_inhibition', sign=-1)

    return dict(name='rg_coincidence', nodes=nodes, edges=edges)


# --- Direct 3x3 cortical column: 4 ordinary E + one central WTA I (experimental) -----
# The smallest useful circuit for the dual FE/FES self-regulating experiment. A 3x3 RGC
# surface feeds four ordinary event-resolved/latency E competitors densely; each E drives
# one central WTA inhibitory relay, which hard-resets exactly those four E. There is NO
# feature relay, coincidence C, feature-specific I, Eor, apical edge, predictive/
# hierarchical feedback, L2/L3 node, or any path that drops/halves/suppresses an RGC
# feature. The single I exists only to enforce local winner-take-all among the four E.
# Metadata-driven: nodes carry column_* tags so layout/diagnostics never parse ids.
RG_DIRECT_CC4_ID = 'cc'                     # column id for the four E + WTA I
RG_DIRECT_CC4_N_E = 4                       # exactly four ordinary competitors (fixed shape)


def _rg_direct_cc4_positions(n_pix: int, n_e: int) -> dict:
    """Deterministic functional positions: a 3x3 RGC sheet on z=0 and the four E on a small
    ring around the central I one layer up, so the intended column is visually obvious and
    the per-synapse 1/d^2 learning-rate influence is well-defined (no zero placeholder)."""
    import math as _m
    width = int(_m.ceil(_m.sqrt(n_pix)))
    grid = 3.8
    z_col = 8.0
    ring_r = 1.6
    pos = {}
    for i in range(n_pix):
        row, col = divmod(i, width)
        pos[f'RGC{i}'] = [(col - (width - 1) / 2.0) * grid,
                          ((width - 1) / 2.0 - row) * grid, 0.0]
    for k in range(n_e):
        ang = 2.0 * _m.pi * k / n_e
        pos[f'{RG_DIRECT_CC4_ID}E{k}'] = [ring_r * _m.cos(ang), ring_r * _m.sin(ang), z_col]
    pos[f'{RG_DIRECT_CC4_ID}I'] = [0.0, 0.0, z_col]
    return pos


def rg_direct_cc4_spec(*, n_pix: int = 9, n_e: int = RG_DIRECT_CC4_N_E) -> dict:
    """The 'rg_direct_cc4' preset: a direct 3x3 RGC -> four-competitor cortical column.

        RGC[i] --feedforward (dense, plastic)--> ccE[k]   (9*n_e edges)
        ccE[k] --relay_excitation--> ccI                  (n_e edges)
        ccI    --hard_reset_inhibition--> ccE[k]          (n_e edges)

    At the canonical 3x3/4-E shape: 9 RGC + 4 E + 1 I = 14 nodes; 36 + 4 + 4 = 44 edges.
    Positions are attached so the four E ring the central I. Ordinary E are event-resolved
    latency competitors (compatible with hard_reset_inhibition); plasticity lives on the
    receiving E. No C/Eor/relay/feature/feedback node exists in this graph."""
    n_e = int(n_e)
    if n_e < 1:
        raise ValueError(f'n_e must be >= 1, got {n_e}')
    cid = RG_DIRECT_CC4_ID
    pos = _rg_direct_cc4_positions(n_pix, n_e)
    width = int(math.ceil(math.sqrt(n_pix)))
    nodes: list = []
    for i in range(n_pix):
        row, col = divmod(i, width)
        nodes.append(dict(id=f'RGC{i}', archetype='rg_source', layer='RGC', pixel=i,
                          label=f'RGC[{row},{col}]', input_row=row, input_col=col,
                          patch_id=0, pos=pos[f'RGC{i}']))
    e_ids = [f'{cid}E{k}' for k in range(n_e)]
    for k, eid in enumerate(e_ids):
        nodes.append(dict(id=eid, archetype='e_latency_competitor', layer='CC',
                          label=f'{cid}·E{k}', column_id=cid, column_role='E',
                          column_index=k, column_row=0, column_col=0, pos=pos[eid]))
    i_id = f'{cid}I'
    nodes.append(dict(id=i_id, archetype='i_relay', layer='CC', label=f'{cid}·I',
                      column_id=cid, column_role='I', column_row=0, column_col=0,
                      pos=pos[i_id]))

    edges: list = []
    for i in range(n_pix):                                   # dense RGC -> every E
        for k, eid in enumerate(e_ids):
            edges.append(dict(id=f'RGC{i}_{eid}', source=f'RGC{i}', target=eid,
                              kind='feedforward', projection='rg_to_column'))
    for eid in e_ids:                                        # every E drives the WTA I
        edges.append(dict(id=f'{eid}_{i_id}', source=eid, target=i_id,
                          kind='relay_excitation', projection='column_e_to_i'))
    for eid in e_ids:                                        # WTA I hard-resets every E
        edges.append(dict(id=f'{i_id}_{eid}', source=i_id, target=eid,
                          kind='hard_reset_inhibition', sign=-1, projection='column_i_to_e'))
    return dict(name='rg_direct_cc4', nodes=nodes, edges=edges)


# =====================================================================================
# Reusable tiled cortical-column construction
# =====================================================================================
# Graph construction is a set of small, pure, composable rules -- NOT a runtime
# "super-neuron". Each rule emits explicit nodes / directed edges the engine, serializer
# and dashboard all see. Simulation NEVER parses the generated ids; it reads the
# ``column_*`` / ``patch_*`` node metadata and the edge ``projection`` family.

@dataclass(frozen=True)
class ColumnHandles:
    """Opaque handles to one cortical column's node ids -- the composition surface the
    RGC-patch and column-to-column connectors bind against. Frozen so a builder cannot
    accidentally share mutable state between two columns."""
    column_id: str
    layer: str
    row: int
    col: int
    e_ids: tuple[str, ...]
    eor_id: str
    c_id: str
    i_id: str
    has_parent: bool


def build_cortical_column(column_id: str, layer: str, row: int, col: int, *,
                          n_e: int, has_parent: bool) -> tuple[ColumnHandles, list, list]:
    """Emit exactly ONE column's nodes and internal edges (no external wiring).

    Nodes: ``n_e`` ordinary E (``e_latency_competitor``), one Eor (same archetype,
    ``column_role='Eor'``), one C (``e_coincidence``), one I (``i_relay``). Internal
    edges per the fixed intra-column rule::

        E[i] -> Eor   feedforward          (column_e_to_eor)
        E[i] -> I     relay_excitation     (column_e_to_i)
        I -> E[i]     hard_reset_inhibition (column_i_to_e)
        Eor -> C      basal_excitation     (column_eor_to_c_basal)
        C -> I        relay_excitation     (column_c_to_i)

    Ordinary E and Eor are the SAME plastic event-resolved archetype and differ only by
    their edges (Eor drives/receives no local I, and feeds the parent + local C basal).
    Returns fresh lists every call (no shared mutable state)."""
    n_e = int(n_e)
    if n_e < 1:
        raise ValueError(f'n_e must be >= 1, got {n_e}')
    e_ids = tuple(f'{column_id}E{i}' for i in range(n_e))
    eor_id, c_id, i_id = f'{column_id}Eor', f'{column_id}C', f'{column_id}I'

    def _common(role):
        return dict(column_id=column_id, column_role=role,
                    column_row=int(row), column_col=int(col))

    nodes: list = []
    for i, eid in enumerate(e_ids):
        nodes.append(dict(id=eid, archetype='e_latency_competitor', layer=layer,
                          label=f'{column_id}·E{i}', column_index=i, **_common('E')))
    nodes.append(dict(id=eor_id, archetype='e_latency_competitor', layer=layer,
                      label=f'{column_id}·Eor', **_common('Eor')))
    nodes.append(dict(id=c_id, archetype='e_coincidence', layer=layer,
                      label=f'{column_id}·C', has_parent=bool(has_parent), **_common('C')))
    nodes.append(dict(id=i_id, archetype='i_relay', layer=layer,
                      label=f'{column_id}·I', **_common('I')))

    edges: list = []
    for i, eid in enumerate(e_ids):
        edges.append(dict(id=f'{column_id}_E{i}_eor', source=eid, target=eor_id,
                          kind='feedforward', projection='column_e_to_eor'))
        edges.append(dict(id=f'{column_id}_E{i}_i', source=eid, target=i_id,
                          kind='relay_excitation', projection='column_e_to_i'))
        edges.append(dict(id=f'{column_id}_i_E{i}', source=i_id, target=eid,
                          kind='hard_reset_inhibition', sign=-1, projection='column_i_to_e'))
    edges.append(dict(id=f'{column_id}_eor_c', source=eor_id, target=c_id,
                      kind='basal_excitation', projection='column_eor_to_c_basal'))
    edges.append(dict(id=f'{column_id}_c_i', source=c_id, target=i_id,
                      kind='relay_excitation', projection='column_c_to_i'))

    handles = ColumnHandles(column_id, layer, int(row), int(col), e_ids, eor_id, c_id,
                            i_id, bool(has_parent))
    return handles, nodes, edges


def connect_rgc_patch(rg_ids, column: ColumnHandles) -> list:
    """Emit the complete bipartite RGC-patch -> ordinary-E feedforward projection
    (``rg_to_column``): every RGC in the patch onto every ordinary E of the column.
    No RGC edge is generated to Eor, C or I. Plasticity belongs to the receiving E."""
    edges: list = []
    for rg in rg_ids:
        for e in column.e_ids:
            edges.append(dict(id=f'{rg}_{e}', source=rg, target=e,
                              kind='feedforward', projection='rg_to_column'))
    return edges


def connect_columns(child: ColumnHandles, parent: ColumnHandles) -> list:
    """Emit one child->parent column link (feedforward + apical feedback), generic over
    depth (never hard-codes L1/L2)::

        child.Eor -> parent.E[k]   feedforward       (column_to_column_ff)
        parent.E[k] -> child.C     apical_excitation (column_to_column_apical)

    Contributes ``2 * parent.N`` edges. Parent ordinary E -- never Eor -- feeds the
    child C apical, which makes the child C non-dormant."""
    edges: list = []
    for pe in parent.e_ids:
        edges.append(dict(id=f'{child.eor_id}_{pe}', source=child.eor_id, target=pe,
                          kind='feedforward', projection='column_to_column_ff'))
        edges.append(dict(id=f'{pe}_{child.c_id}', source=pe, target=child.c_id,
                          kind='apical_excitation', projection='column_to_column_apical'))
    return edges


def tiled_cc_spec(*, cc_e_count: int = 8, l1_e_count: int | None = None,
                  l2_e_count: int | None = None, name: str = 'tiled_cc',
                  input_rows: int = 9, input_cols: int = 9,
                  patch_rows: int = 3, patch_cols: int = 3) -> dict:
    """Compose the canonical tiled cortical-column hierarchy from the reusable rules:
    a 9x9 RGC surface tiled into nine 3x3 patches, nine L1 columns (one per patch,
    arranged 3x3), and one L2 column receiving all nine L1 outputs.

    ``cc_e_count`` sizes the ordinary-E bank of every column uniformly. ``l1_e_count`` /
    ``l2_e_count`` optionally override it per layer (each defaults to ``cc_e_count``), so a
    shallower L1 can be paired with a wider L2 without touching connectivity or metadata
    shape. At the uniform default ``cc_e_count=8`` this is exactly 191 nodes and 1052
    directed edges (10N+111 nodes, 129N+20 edges for ``N = cc_e_count``); with
    ``l1_e_count=4, l2_e_count=8`` it is 155 nodes and 620 edges. The returned spec carries
    the top-level tiling metadata that is the single source of truth for construction,
    layout, validation and dashboard grouping."""
    n_e = int(cc_e_count)
    l1_n = int(l1_e_count) if l1_e_count is not None else n_e
    l2_n = int(l2_e_count) if l2_e_count is not None else n_e
    if l1_n < 1 or l2_n < 1:
        raise ValueError(
            f'ordinary-E counts must be >= 1, got L1={l1_n}, L2={l2_n}')
    if input_rows % patch_rows != 0 or input_cols % patch_cols != 0:
        raise ValueError(
            f'patch shape ({patch_rows}x{patch_cols}) must tile the input '
            f'({input_rows}x{input_cols}) exactly')
    grid_rows, grid_cols = input_rows // patch_rows, input_cols // patch_cols

    nodes: list = []
    internal_edges: list = []
    rg_edges: list = []
    link_edges: list = []

    # RGC surface (row-major pixels), grouped into patches (patch-local row-major order).
    rg_by_patch: dict[tuple[int, int], list[str]] = {}
    for gr in range(input_rows):
        for gc in range(input_cols):
            pixel = gr * input_cols + gc
            pr, pc = gr // patch_rows, gc // patch_cols
            plr, plc = gr % patch_rows, gc % patch_cols
            patch_id = pr * grid_cols + pc
            rid = f'RGC{pixel}'
            nodes.append(dict(id=rid, archetype='rg_source', layer='RGC', pixel=pixel,
                              label=f'RGC[{gr},{gc}]', input_row=gr, input_col=gc,
                              patch_id=patch_id, patch_row=pr, patch_col=pc,
                              patch_local_row=plr, patch_local_col=plc))
            rg_by_patch.setdefault((pr, pc), []).append(rid)

    # L1 columns: one per patch, arranged as the 3x3 tile grid, each with a parent (L2).
    l1: dict[tuple[int, int], ColumnHandles] = {}
    for pr in range(grid_rows):
        for pc in range(grid_cols):
            cid = f'L1c{pr}{pc}'
            h, cn, ce = build_cortical_column(cid, 'L1', pr, pc, n_e=l1_n, has_parent=True)
            nodes += cn
            internal_edges += ce
            l1[(pr, pc)] = h
            # each L1 column's ordinary E carry their input-patch id too (display only).
            patch_id = pr * grid_cols + pc
            for node in cn:
                if node.get('column_role') == 'E':
                    node['patch'] = patch_id
            rg_edges += connect_rgc_patch(rg_by_patch[(pr, pc)], h)

    # One L2 column (top of the hierarchy); its C is intentionally dormant (no parent).
    l2, l2n, l2e = build_cortical_column('L2c00', 'L2', 0, 0, n_e=l2_n, has_parent=False)
    nodes += l2n
    internal_edges += l2e

    # Column links: every L1 (child) to the single L2 (parent).
    for pr in range(grid_rows):
        for pc in range(grid_cols):
            link_edges += connect_columns(l1[(pr, pc)], l2)

    edges = internal_edges + rg_edges + link_edges

    # Top-level tiling metadata (single source of truth). Per-column ``e_count`` reports
    # each layer's actual ordinary-E bank (L1 vs L2 may differ).
    columns_meta = [dict(id=l1[(pr, pc)].column_id, layer='L1', row=pr, col=pc,
                         e_count=l1_n, parent_ids=['L2c00'])
                    for pr in range(grid_rows) for pc in range(grid_cols)]
    columns_meta.append(dict(id='L2c00', layer='L2', row=0, col=0,
                             e_count=l2_n, parent_ids=[]))
    topology_meta = dict(
        family=TILED_FAMILY,
        input_shape=dict(rows=input_rows, cols=input_cols),
        patch_shape=dict(rows=patch_rows, cols=patch_cols),
        grid_shape=dict(rows=grid_rows, cols=grid_cols),
        column_layers=[dict(layer='L1', rows=grid_rows, cols=grid_cols),
                       dict(layer='L2', rows=1, cols=1)],
        # ``cc_e_count`` names the L1 (sensory) bank size; equal to l2_n in the uniform
        # tiled_cc default, so existing consumers/tests are unchanged there.
        cc_e_count=l1_n,
        columns=columns_meta)
    return dict(name=name, topology=topology_meta, nodes=nodes, edges=edges)


# =====================================================================================
# Two-tower composition graph
# =====================================================================================
# Two independent 9x9 towers laid out side by side on ONE validated 9x18 input sheet, each
# feeding its own L2 classic column, with both L2 columns feeding one L3 classic column::
#
#     Tower 0 (left):   9x9 RGC -> 9 L1 classic CC -> T0L2c00 -.
#                                                               >-- L3c00
#     Tower 1 (right):  9x9 RGC -> 9 L1 classic CC -> T1L2c00 -'
#
# It composes ONLY the existing public rules (``build_cortical_column`` /
# ``connect_rgc_patch`` / ``connect_columns``): no feature relay, no direct identity, no new
# archetype, no new edge kind, no gating. Because each tower L2 now declares L3 as its
# parent, its C is no longer dormant; L3 has no parent, so L3's C is the dormant one.
#
# The scientific point of the graph is exactly the single-Eor bottleneck it inherits: L3
# receives only TWO source identities in total (``T0L2c00Eor`` and ``T1L2c00Eor``), whatever
# the towers below them are representing. See docs/TWO_TOWER_COMPOSITION.md.

# --- whole-sheet stimuli -------------------------------------------------------------
# The four canonical PATTERNS are 3x3 LOCAL stimuli: they name a receptive-field feature and
# are embedded into one patch. A composition graph also needs stimuli that span the WHOLE
# input sheet and cross the tower seam, which no per-patch pattern can express. These are
# those: three deterministic one-pixel-wide glyphs on the 9x18 two-tower sheet.
#
# They are keyed by INPUT SHAPE, not by preset name -- any tiled graph whose surface is 9x18
# can be driven by them, and a 9x9 graph simply has no whole-sheet bank. ``V`` and ``A``
# deliberately share diagonal structure and differ by the crossbar; ``7`` is the requested
# straight-legged form.
SHEET_GLYPH_SHEETS = {
    # (rows, cols) -> {glyph name: {stroke name: [(row, col), ...]}}
    (9, 18): {
        'V': {
            'left_leg':  [(r, r) for r in range(9)],
            'right_leg': [(r, 17 - r) for r in range(9)],
        },
        'A': {
            'left_leg':  [(r, 8 - r) for r in range(9)],
            'right_leg': [(r, 9 + r) for r in range(9)],
            'crossbar':  [(4, c) for c in range(4, 14)],
        },
        '7': {
            'top': [(0, c) for c in range(18)],
            'leg': [(r, 17) for r in range(9)],
        },
    },
}


def sheet_glyph_names(input_rows: int, input_cols: int) -> tuple:
    """Whole-sheet stimulus names defined for this input shape (empty when none are)."""
    return tuple(SHEET_GLYPH_SHEETS.get((int(input_rows), int(input_cols)), {}))


def sheet_glyph_coordinates(input_rows: int, input_cols: int, name: str) -> list:
    """Sorted, de-duplicated ``(row, col)`` coordinates of one whole-sheet stimulus."""
    bank = SHEET_GLYPH_SHEETS.get((int(input_rows), int(input_cols)), {})
    if name not in bank:
        raise KeyError(
            f'no whole-sheet stimulus {name!r} for a {input_rows}x{input_cols} surface; '
            f'expected one of {tuple(bank)}')
    return sorted({c for stroke in bank[name].values() for c in stroke})


def sheet_glyph_bank(input_rows: int, input_cols: int) -> dict:
    """``{name: row-major binary vector}`` for every whole-sheet stimulus of this shape.

    Returns a fresh dict of fresh vectors ({} when the shape declares none), sized
    ``input_rows * input_cols`` so it can be handed straight to ``set_input``.
    """
    rows, cols = int(input_rows), int(input_cols)
    out: dict = {}
    for name in sheet_glyph_names(rows, cols):
        vec = [0] * (rows * cols)
        for r, c in sheet_glyph_coordinates(rows, cols, name):
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f'stimulus {name!r} coordinate ({r},{c}) is off-sheet')
            vec[r * cols + c] = 1
        out[name] = vec
    return out


def two_tower_composition_spec(*, cc_e_count: int = 8,
                               name: str = 'two_tower_composition') -> dict:
    """Compose the experimental two-tower / one-L3 composition hierarchy.

    ``cc_e_count`` sizes the ordinary-E bank of EVERY column uniformly (18 L1 + 2 L2 + 1
    L3 = 21 columns). At the default 8 this is exactly 393 nodes and 2162 directed edges
    (``21*(N+3)`` nodes and ``21*(3N+2) + 18*9N + 20*2N`` edges for ``N = cc_e_count``).

    Deterministic id / coordinate contract (scientific code must still select by metadata,
    never by parsing an id)::

        input:    row-major RGC0..RGC161 on one 9x18 sheet
        left L1:  T0L1c00..T0L1c22, global metadata col = local col       (0..2)
        right L1: T1L1c00..T1L1c22, global metadata col = local col + 3   (3..5)
        left L2:  T0L2c00 (layer L2, row 0, col 0)
        right L2: T1L2c00 (layer L2, row 0, col 1)
        top:      L3c00   (layer L3, row 0, col 0)

    Returns a fresh JSON-serializable spec every call. The graph is registered in
    ``PRESETS`` and can also be supplied through ``apply_topology`` like any other spec.
    """
    n_e = int(cc_e_count)
    if n_e < 1:
        raise ValueError(f'ordinary-E count must be >= 1, got {n_e}')
    d = TWO_TOWER_DEFAULTS
    input_rows, input_cols = d['input_rows'], d['input_cols']
    patch_rows, patch_cols = d['patch_rows'], d['patch_cols']
    if input_rows % patch_rows or input_cols % patch_cols:
        raise ValueError('two-tower patch shape must tile the 9x18 sheet exactly')
    grid_rows, grid_cols = input_rows // patch_rows, input_cols // patch_cols
    tower_cols = grid_cols // TWO_TOWER_TOWER_COUNT          # 3 patch columns per tower

    nodes: list = []
    internal_edges: list = []
    rg_edges: list = []
    link_edges: list = []

    # --- 1. the 162 RGC nodes, grouped by GLOBAL patch (identical arithmetic to
    # ``tiled_cc_spec``, only the sheet is wider). Pixel ownership is disjoint by
    # construction: one RGC per (row, col) of one 9x18 sheet.
    rg_by_patch: dict[tuple[int, int], list[str]] = {}
    for gr in range(input_rows):
        for gc in range(input_cols):
            pixel = gr * input_cols + gc
            pr, pc = gr // patch_rows, gc // patch_cols
            plr, plc = gr % patch_rows, gc % patch_cols
            patch_id = pr * grid_cols + pc
            rid = f'RGC{pixel}'
            nodes.append(dict(id=rid, archetype='rg_source', layer='RGC', pixel=pixel,
                              label=f'RGC[{gr},{gc}]', input_row=gr, input_col=gc,
                              patch_id=patch_id, patch_row=pr, patch_col=pc,
                              patch_local_row=plr, patch_local_col=plc))
            rg_by_patch.setdefault((pr, pc), []).append(rid)

    # --- 2/3. eighteen L1 columns (nine per tower), each wired to its OWN global patch.
    l1: dict[tuple[int, int], ColumnHandles] = {}
    for tower in range(TWO_TOWER_TOWER_COUNT):
        for lr in range(grid_rows):
            for lc in range(tower_cols):
                gc = tower * tower_cols + lc                 # global metadata column
                cid = f'T{tower}L1c{lr}{lc}'                 # id keeps the tower-LOCAL col
                h, cn, ce = build_cortical_column(cid, 'L1', lr, gc, n_e=n_e,
                                                  has_parent=True)
                nodes += cn
                internal_edges += ce
                l1[(lr, gc)] = h
                patch_id = lr * grid_cols + gc
                for node in cn:
                    if node.get('column_role') == 'E':
                        node['patch'] = patch_id             # display tag only
                rg_edges += connect_rgc_patch(rg_by_patch[(lr, gc)], h)

    # --- 4. one L2 column per tower; each declares L3 as its parent, so its C is NOT
    # dormant (the classic dormancy moves up to L3).
    l2: list[ColumnHandles] = []
    for tower in range(TWO_TOWER_TOWER_COUNT):
        h, cn, ce = build_cortical_column(f'T{tower}L2c00', 'L2', 0, tower, n_e=n_e,
                                          has_parent=True)
        nodes += cn
        internal_edges += ce
        l2.append(h)

    # --- 5. each L1 links ONLY to its own tower's L2 (no cross-tower edge is emitted).
    for tower in range(TWO_TOWER_TOWER_COUNT):
        for lr in range(grid_rows):
            for lc in range(tower_cols):
                link_edges += connect_columns(l1[(lr, tower * tower_cols + lc)], l2[tower])

    # --- 6. the single L3 composition column: no parent, so ITS C is the dormant one.
    l3, l3n, l3e = build_cortical_column('L3c00', 'L3', 0, 0, n_e=n_e, has_parent=False)
    nodes += l3n
    internal_edges += l3e

    # --- 7. both tower L2 columns feed L3 through the SAME generic child->parent rule:
    # child.L2.Eor -> every L3 ordinary E, and every L3 ordinary E -> child.L2.C apical.
    for tower in range(TWO_TOWER_TOWER_COUNT):
        link_edges += connect_columns(l2[tower], l3)

    # --- 8/9. concatenate once and return a fresh spec with full tiled metadata.
    edges = internal_edges + rg_edges + link_edges
    columns_meta = [dict(id=l1[(lr, tower * tower_cols + lc)].column_id, layer='L1',
                         row=lr, col=tower * tower_cols + lc, e_count=n_e,
                         parent_ids=[l2[tower].column_id])
                    for tower in range(TWO_TOWER_TOWER_COUNT)
                    for lr in range(grid_rows) for lc in range(tower_cols)]
    columns_meta += [dict(id=l2[t].column_id, layer='L2', row=0, col=t, e_count=n_e,
                          parent_ids=['L3c00']) for t in range(TWO_TOWER_TOWER_COUNT)]
    columns_meta.append(dict(id='L3c00', layer='L3', row=0, col=0, e_count=n_e,
                             parent_ids=[]))
    topology_meta = dict(
        family=TILED_FAMILY,
        input_shape=dict(rows=input_rows, cols=input_cols),
        patch_shape=dict(rows=patch_rows, cols=patch_cols),
        grid_shape=dict(rows=grid_rows, cols=grid_cols),
        column_layers=[dict(layer='L1', rows=grid_rows, cols=grid_cols),
                       dict(layer='L2', rows=1, cols=TWO_TOWER_TOWER_COUNT),
                       dict(layer='L3', rows=1, cols=1)],
        cc_e_count=n_e,
        # Declared tower membership: experiment code resolves a column's tower from the
        # parent chain / this map, never from the ``T0``/``T1`` id prefix.
        towers=[dict(index=t, l2_column=l2[t].column_id,
                     l1_columns=[l1[(lr, t * tower_cols + lc)].column_id
                                 for lr in range(grid_rows) for lc in range(tower_cols)],
                     input_col_range=[t * tower_cols * patch_cols,
                                      (t + 1) * tower_cols * patch_cols - 1])
                for t in range(TWO_TOWER_TOWER_COUNT)],
        columns=columns_meta)
    return dict(name=name, topology=topology_meta, nodes=nodes, edges=edges)


# =====================================================================================
# Double-Eor tiled cortical-column construction (DIAGNOSTIC latency probe)
# =====================================================================================
# Exactly the classic column with ONE extra output relay spliced into the ascending path:
#
#     classic:     E -> Eor  ------------> parent E        (loop latency 3)
#     double_eor:  E -> Eor -> Eor2 -----> parent E        (loop latency 4, predicted)
#
# Everything else -- WTA, C gate, apical feedback, C -> I delay-1 reset, caps, rates -- is
# byte-for-byte the classic rule. It changes ONE thing (loop length) so the prediction
# "period = 2 x latency" can be falsified. See docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md 8.1b.

def build_double_eor_column(column_id: str, layer: str, row: int, col: int, *,
                            n_e: int, has_parent: bool) -> tuple[ColumnHandles, list, list]:
    """Emit ONE classic column plus a second output relay ``Eor2`` in series.

    Nodes: ``n_e`` ordinary E, Eor, **Eor2**, C, I. Internal edges::

        E[i] -> Eor    feedforward           (column_e_to_eor)
        E[i] -> I      relay_excitation      (column_e_to_i)
        I -> E[i]      hard_reset_inhibition (column_i_to_e)
        Eor -> Eor2    feedforward           (column_eor_to_eor2)   <-- the extra hop
        Eor2 -> C      basal_excitation      (column_eor_to_c_basal)
        C -> I         relay_excitation      (column_c_to_i)

    ``Eor2`` -- not ``Eor`` -- is the column's output and its C's basal source, so the
    inserted relay lengthens BOTH the ascending path and the confirmation path by one
    boundary. The returned handles report ``eor_id = Eor2`` so the shared column-link
    connector wires the parent from the true output.
    """
    n_e = int(n_e)
    if n_e < 1:
        raise ValueError(f'n_e must be >= 1, got {n_e}')
    e_ids = tuple(f'{column_id}E{i}' for i in range(n_e))
    eor_id, eor2_id = f'{column_id}Eor', f'{column_id}Eor2'
    c_id, i_id = f'{column_id}C', f'{column_id}I'

    def _common(role):
        return dict(column_id=column_id, column_role=role,
                    column_row=int(row), column_col=int(col))

    nodes: list = []
    for i, eid in enumerate(e_ids):
        nodes.append(dict(id=eid, archetype='e_latency_competitor', layer=layer,
                          label=f'{column_id}·E{i}', column_index=i, **_common('E')))
    nodes.append(dict(id=eor_id, archetype='e_latency_competitor', layer=layer,
                      label=f'{column_id}·Eor', **_common('Eor')))
    nodes.append(dict(id=eor2_id, archetype='e_latency_competitor', layer=layer,
                      label=f'{column_id}·Eor2', **_common('Eor2')))
    nodes.append(dict(id=c_id, archetype='e_coincidence', layer=layer,
                      label=f'{column_id}·C', has_parent=bool(has_parent), **_common('C')))
    nodes.append(dict(id=i_id, archetype='i_relay', layer=layer,
                      label=f'{column_id}·I', **_common('I')))

    edges: list = []
    for i, eid in enumerate(e_ids):
        edges.append(dict(id=f'{column_id}_E{i}_eor', source=eid, target=eor_id,
                          kind='feedforward', projection='column_e_to_eor'))
        edges.append(dict(id=f'{column_id}_E{i}_i', source=eid, target=i_id,
                          kind='relay_excitation', projection='column_e_to_i'))
        edges.append(dict(id=f'{column_id}_i_E{i}', source=i_id, target=eid,
                          kind='hard_reset_inhibition', sign=-1, projection='column_i_to_e'))
    edges.append(dict(id=f'{column_id}_eor_eor2', source=eor_id, target=eor2_id,
                      kind='feedforward', projection='column_eor_to_eor2'))
    edges.append(dict(id=f'{column_id}_eor2_c', source=eor2_id, target=c_id,
                      kind='basal_excitation', projection='column_eor_to_c_basal'))
    edges.append(dict(id=f'{column_id}_c_i', source=c_id, target=i_id,
                      kind='relay_excitation', projection='column_c_to_i'))

    handles = ColumnHandles(column_id, layer, int(row), int(col), e_ids, eor2_id, c_id,
                            i_id, bool(has_parent))
    return handles, nodes, edges


def tiled_cc_double_eor_spec(*, cc_e_count: int = 8, name: str = 'tiled_cc_double_eor',
                             input_rows: int = 9, input_cols: int = 9,
                             patch_rows: int = 3, patch_cols: int = 3) -> dict:
    """Compose the diagnostic double-relay tiled hierarchy: the canonical 9x9 tiled
    cortical-column graph with one extra output relay per column. At ``cc_e_count=8`` this
    is exactly 201 nodes and 1062 directed edges (the classic 191/1052 plus 10 Eor2 nodes
    and 10 Eor->Eor2 edges). Carries ``topology.variant='double_eor'``."""
    n_e = int(cc_e_count)
    if n_e < 1:
        raise ValueError(f'ordinary-E count must be >= 1, got {n_e}')
    if input_rows % patch_rows != 0 or input_cols % patch_cols != 0:
        raise ValueError(
            f'patch shape ({patch_rows}x{patch_cols}) must tile the input '
            f'({input_rows}x{input_cols}) exactly')
    grid_rows, grid_cols = input_rows // patch_rows, input_cols // patch_cols

    nodes: list = []
    internal_edges: list = []
    rg_edges: list = []
    link_edges: list = []

    rg_by_patch: dict[tuple[int, int], list[str]] = {}
    for gr in range(input_rows):
        for gc in range(input_cols):
            pixel = gr * input_cols + gc
            pr, pc = gr // patch_rows, gc // patch_cols
            plr, plc = gr % patch_rows, gc % patch_cols
            patch_id = pr * grid_cols + pc
            rid = f'RGC{pixel}'
            nodes.append(dict(id=rid, archetype='rg_source', layer='RGC', pixel=pixel,
                              label=f'RGC[{gr},{gc}]', input_row=gr, input_col=gc,
                              patch_id=patch_id, patch_row=pr, patch_col=pc,
                              patch_local_row=plr, patch_local_col=plc))
            rg_by_patch.setdefault((pr, pc), []).append(rid)

    l1: dict[tuple[int, int], ColumnHandles] = {}
    for pr in range(grid_rows):
        for pc in range(grid_cols):
            cid = f'L1c{pr}{pc}'
            h, cn, ce = build_double_eor_column(cid, 'L1', pr, pc, n_e=n_e, has_parent=True)
            nodes += cn
            internal_edges += ce
            l1[(pr, pc)] = h
            patch_id = pr * grid_cols + pc
            for node in cn:
                if node.get('column_role') == 'E':
                    node['patch'] = patch_id
            rg_edges += connect_rgc_patch(rg_by_patch[(pr, pc)], h)

    l2, l2n, l2e = build_double_eor_column('L2c00', 'L2', 0, 0, n_e=n_e, has_parent=False)
    nodes += l2n
    internal_edges += l2e

    for pr in range(grid_rows):
        for pc in range(grid_cols):
            link_edges += connect_columns(l1[(pr, pc)], l2)

    edges = internal_edges + rg_edges + link_edges

    columns_meta = [dict(id=l1[(pr, pc)].column_id, layer='L1', row=pr, col=pc,
                         e_count=n_e, parent_ids=['L2c00'])
                    for pr in range(grid_rows) for pc in range(grid_cols)]
    columns_meta.append(dict(id='L2c00', layer='L2', row=0, col=0,
                             e_count=n_e, parent_ids=[]))
    topology_meta = dict(
        family=TILED_FAMILY,
        variant=TILED_VARIANT_DOUBLE_EOR,
        input_shape=dict(rows=input_rows, cols=input_cols),
        patch_shape=dict(rows=patch_rows, cols=patch_cols),
        grid_shape=dict(rows=grid_rows, cols=grid_cols),
        column_layers=[dict(layer='L1', rows=grid_rows, cols=grid_cols),
                       dict(layer='L2', rows=1, cols=1)],
        cc_e_count=n_e,
        columns=columns_meta)
    return dict(name=name, topology=topology_meta, nodes=nodes, edges=edges)


# =====================================================================================
# Direct-identity tiled cortical-column construction (no Eor)
# =====================================================================================
# The Eor relay is removed and each ordinary-E WINNER IDENTITY is transmitted directly:
# every child ordinary E projects to every parent ordinary E, so the parent owns a distinct
# plastic weight per (child column, child winner) source instead of a single pooled
# "this column was active" event. The column's C consequently owns one learned basal
# afferent PER local ordinary E (multi-basal), while apical permission stays the unweighted
# Boolean parent-E gate. See docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md.

@dataclass(frozen=True)
class DirectColumnHandles:
    """Handles to one Eor-less column: the ordinary E bank, its multi-basal C, and the
    local WTA/feedback I. There is NO output relay -- the E ids ARE the output alphabet."""
    column_id: str
    layer: str
    row: int
    col: int
    e_ids: tuple[str, ...]
    c_id: str
    i_id: str
    has_parent: bool


def build_direct_column(column_id: str, layer: str, row: int, col: int, *,
                        n_e: int, has_parent: bool) -> tuple[DirectColumnHandles, list, list]:
    """Emit exactly ONE Eor-less column's nodes and internal edges (no external wiring).

    Nodes: ``n_e`` ordinary E (``e_latency_competitor``), one multi-basal C
    (``e_coincidence``), one WTA/feedback I (``i_relay``). Internal edges::

        E[i] -> I     relay_excitation      (column_e_to_i)
        I -> E[i]     hard_reset_inhibition (column_i_to_e)
        E[i] -> C     basal_excitation      (column_e_to_c_basal)   [one per E, distinct]
        C -> I        relay_excitation      (column_c_to_i)

    The ``n_e`` basal edges are source-distinct and never collapsed: each carries its own
    learned weight, so one local owner maturing its association cannot depress another's.
    Returns fresh lists every call (no shared mutable state)."""
    n_e = int(n_e)
    if n_e < 1:
        raise ValueError(f'n_e must be >= 1, got {n_e}')
    e_ids = tuple(f'{column_id}E{i}' for i in range(n_e))
    c_id, i_id = f'{column_id}C', f'{column_id}I'

    def _common(role):
        return dict(column_id=column_id, column_role=role,
                    column_row=int(row), column_col=int(col))

    nodes: list = []
    for i, eid in enumerate(e_ids):
        nodes.append(dict(id=eid, archetype='e_latency_competitor', layer=layer,
                          label=f'{column_id}·E{i}', column_index=i, **_common('E')))
    nodes.append(dict(id=c_id, archetype='e_coincidence', layer=layer,
                      label=f'{column_id}·C', has_parent=bool(has_parent),
                      # Explicit opt-in to the multi-basal capability. Without it a C is
                      # still held to the historical exactly-one-basal invariant, so no old
                      # custom graph is silently relaxed.
                      multi_basal=True, **_common('C')))
    nodes.append(dict(id=i_id, archetype='i_relay', layer=layer,
                      label=f'{column_id}·I', **_common('I')))

    edges: list = []
    for i, eid in enumerate(e_ids):
        edges.append(dict(id=f'{column_id}_E{i}_i', source=eid, target=i_id,
                          kind='relay_excitation', projection='column_e_to_i'))
        edges.append(dict(id=f'{column_id}_i_E{i}', source=i_id, target=eid,
                          kind='hard_reset_inhibition', sign=-1, projection='column_i_to_e'))
        edges.append(dict(id=f'{column_id}_E{i}_c', source=eid, target=c_id,
                          kind='basal_excitation', projection='column_e_to_c_basal'))
    edges.append(dict(id=f'{column_id}_c_i', source=c_id, target=i_id,
                      kind='relay_excitation', projection='column_c_to_i'))

    handles = DirectColumnHandles(column_id, layer, int(row), int(col), e_ids, c_id, i_id,
                                  bool(has_parent))
    return handles, nodes, edges


def connect_direct_identity(child: DirectColumnHandles,
                            parent: DirectColumnHandles) -> list:
    """Emit one child->parent link that PRESERVES the child winner's identity::

        child.E[i] -> parent.E[k]   feedforward       (identity_child_e_to_parent_e)
        parent.E[k] -> child.C      apical_excitation (column_to_column_apical)

    Contributes ``child.N * parent.N`` feedforward edges (one per source address) plus
    ``parent.N`` apical edges. Nothing pools by column: the receiving parent E owns a
    distinct plastic weight for every ``(child column, child winner)`` pair."""
    edges: list = []
    for ce in child.e_ids:
        for pe in parent.e_ids:
            edges.append(dict(id=f'{ce}_{pe}', source=ce, target=pe,
                              kind='feedforward', projection='identity_child_e_to_parent_e'))
    for pe in parent.e_ids:
        edges.append(dict(id=f'{pe}_{child.c_id}', source=pe, target=child.c_id,
                          kind='apical_excitation', projection='column_to_column_apical'))
    return edges


def tiled_cc_direct_identity_spec(*, cc_e_count: int = 8, name: str = 'tiled_cc_direct_identity',
                                  input_rows: int = 9, input_cols: int = 9,
                                  patch_rows: int = 3, patch_cols: int = 3) -> dict:
    """Compose the Eor-less direct-identity tiled hierarchy: a 9x9 RGC surface tiled into
    nine 3x3 patches, nine L1 columns (8 ordinary E + multi-basal C + WTA I each), and one
    L2 column of the same motif whose ordinary E receive every child E identity directly.

    At the default ``cc_e_count=8`` this is exactly 181 nodes and 1546 directed edges. The
    top L2 C has no parent and is therefore dormant (zero apical), which keeps the motif
    recursively reusable if a later L3 gives L2 a parent. The returned spec carries
    ``topology.variant='direct_identity'`` -- the single source of truth that construction,
    layout and validation branch on (never the preset name or a node-id prefix)."""
    n_e = int(cc_e_count)
    if n_e < 1:
        raise ValueError(f'ordinary-E count must be >= 1, got {n_e}')
    if input_rows % patch_rows != 0 or input_cols % patch_cols != 0:
        raise ValueError(
            f'patch shape ({patch_rows}x{patch_cols}) must tile the input '
            f'({input_rows}x{input_cols}) exactly')
    grid_rows, grid_cols = input_rows // patch_rows, input_cols // patch_cols

    nodes: list = []
    internal_edges: list = []
    rg_edges: list = []
    link_edges: list = []

    rg_by_patch: dict[tuple[int, int], list[str]] = {}
    for gr in range(input_rows):
        for gc in range(input_cols):
            pixel = gr * input_cols + gc
            pr, pc = gr // patch_rows, gc // patch_cols
            plr, plc = gr % patch_rows, gc % patch_cols
            patch_id = pr * grid_cols + pc
            rid = f'RGC{pixel}'
            nodes.append(dict(id=rid, archetype='rg_source', layer='RGC', pixel=pixel,
                              label=f'RGC[{gr},{gc}]', input_row=gr, input_col=gc,
                              patch_id=patch_id, patch_row=pr, patch_col=pc,
                              patch_local_row=plr, patch_local_col=plc))
            rg_by_patch.setdefault((pr, pc), []).append(rid)

    l1: dict[tuple[int, int], DirectColumnHandles] = {}
    for pr in range(grid_rows):
        for pc in range(grid_cols):
            cid = f'L1c{pr}{pc}'
            h, cn, ce = build_direct_column(cid, 'L1', pr, pc, n_e=n_e, has_parent=True)
            nodes += cn
            internal_edges += ce
            l1[(pr, pc)] = h
            patch_id = pr * grid_cols + pc
            for node in cn:
                if node.get('column_role') == 'E':
                    node['patch'] = patch_id          # display-only patch tag (as classic)
            for rg in rg_by_patch[(pr, pc)]:
                for e in h.e_ids:
                    rg_edges.append(dict(id=f'{rg}_{e}', source=rg, target=e,
                                         kind='feedforward', projection='rg_to_column'))

    l2, l2n, l2e = build_direct_column('L2c00', 'L2', 0, 0, n_e=n_e, has_parent=False)
    nodes += l2n
    internal_edges += l2e

    for pr in range(grid_rows):
        for pc in range(grid_cols):
            link_edges += connect_direct_identity(l1[(pr, pc)], l2)

    edges = internal_edges + rg_edges + link_edges

    columns_meta = [dict(id=l1[(pr, pc)].column_id, layer='L1', row=pr, col=pc,
                         e_count=n_e, parent_ids=['L2c00'])
                    for pr in range(grid_rows) for pc in range(grid_cols)]
    columns_meta.append(dict(id='L2c00', layer='L2', row=0, col=0,
                             e_count=n_e, parent_ids=[]))
    topology_meta = dict(
        family=TILED_FAMILY,
        variant=TILED_VARIANT_DIRECT_IDENTITY,
        input_shape=dict(rows=input_rows, cols=input_cols),
        patch_shape=dict(rows=patch_rows, cols=patch_cols),
        grid_shape=dict(rows=grid_rows, cols=grid_cols),
        column_layers=[dict(layer='L1', rows=grid_rows, cols=grid_cols),
                       dict(layer='L2', rows=1, cols=1)],
        cc_e_count=n_e,
        columns=columns_meta)
    return dict(name=name, topology=topology_meta, nodes=nodes, edges=edges)


def embed_patch_pattern(input_shape, patch_shape, patch, local_vector) -> list[int]:
    """Pure helper: embed a patch-local pattern into a full row-major input vector.

    ``input_shape`` / ``patch_shape`` are ``(rows, cols)``; ``patch`` is a
    ``(patch_row, patch_col)`` tile coordinate; ``local_vector`` holds
    ``patch_rows*patch_cols`` values in patch-local row-major order. Returns a length
    ``input_rows*input_cols`` list of ints with only the selected patch's pixels set.
    Validates shapes and patch bounds; usable with no dashboard/engine present."""
    in_rows, in_cols = int(input_shape[0]), int(input_shape[1])
    p_rows, p_cols = int(patch_shape[0]), int(patch_shape[1])
    pr, pc = int(patch[0]), int(patch[1])
    if in_rows <= 0 or in_cols <= 0 or p_rows <= 0 or p_cols <= 0:
        raise ValueError('input_shape and patch_shape must be positive')
    if in_rows % p_rows != 0 or in_cols % p_cols != 0:
        raise ValueError(f'patch shape {patch_shape} must tile input shape {input_shape}')
    grid_rows, grid_cols = in_rows // p_rows, in_cols // p_cols
    if not (0 <= pr < grid_rows and 0 <= pc < grid_cols):
        raise ValueError(
            f'patch {patch} out of bounds for {grid_rows}x{grid_cols} patch grid')
    local = list(local_vector)
    if len(local) != p_rows * p_cols:
        raise ValueError(
            f'local_vector must have {p_rows * p_cols} values, got {len(local)}')
    out = [0] * (in_rows * in_cols)
    for lr in range(p_rows):
        for lc in range(p_cols):
            gr, gc = pr * p_rows + lr, pc * p_cols + lc
            out[gr * in_cols + gc] = int(1 if local[lr * p_cols + lc] > 0.5 else 0)
    return out


def tiled_input_size(spec) -> int | None:
    """Input pixel count declared by a tiled spec's top-level metadata, else None.
    Lets callers resize/validate an 81-input tiled graph at its OWN size regardless of
    the currently active engine dimensions."""
    meta = spec.get('topology') if isinstance(spec, dict) else None
    if isinstance(meta, dict) and meta.get('family') == TILED_FAMILY:
        ishape = meta.get('input_shape')
        if (isinstance(ishape, dict) and isinstance(ishape.get('rows'), int)
                and isinstance(ishape.get('cols'), int)):
            return int(ishape['rows']) * int(ishape['cols'])
    return None


class SpecError(ValueError):
    """Raised when a NetworkSpec is structurally invalid."""


def validate_spec(spec: dict, n_pix: int) -> dict:
    """Validate a NetworkSpec; return a normalized copy (edges get defaults filled).
    Raises SpecError with a human-readable message on the first problem found."""
    if not isinstance(spec, dict):
        raise SpecError('spec must be an object')
    nodes = spec.get('nodes')
    edges = spec.get('edges')
    if not isinstance(nodes, list) or not nodes:
        raise SpecError('spec.nodes must be a non-empty list')
    if not isinstance(edges, list):
        raise SpecError('spec.edges must be a list')

    seen = {}
    norm_nodes = []
    pixels_used = {}
    for n in nodes:
        nid = n.get('id')
        arch = n.get('archetype')
        if not nid or not isinstance(nid, str):
            raise SpecError(f'node missing string id: {n!r}')
        if nid in seen:
            raise SpecError(f'duplicate node id {nid!r}')
        if arch not in ARCHETYPES:
            raise SpecError(f'node {nid!r} has unknown archetype {arch!r}; '
                            f'valid: {sorted(ARCHETYPES)}')
        # ``pixel`` = external-input ownership: unique, input-sink archetypes only.
        pixel = n.get('pixel')
        if pixel is not None:
            if not ARCHETYPES[arch]['input_sink']:
                raise SpecError(f'node {nid!r} ({arch}) sets pixel but is not an external '
                                f'input sink; valid: {sorted(INPUT_SINKS)}')
            if not isinstance(pixel, int) or not (0 <= pixel < n_pix):
                raise SpecError(f'node {nid!r} pixel must be an int in [0,{n_pix})')
            if pixel in pixels_used:
                raise SpecError(f'pixel {pixel} mapped by both {pixels_used[pixel]!r} '
                                f'and {nid!r}')
            pixels_used[pixel] = nid
        # ``grid`` = display / receptive-field metadata only: never unique, no input.
        grid = n.get('grid')
        if grid is not None and (not isinstance(grid, int) or not (0 <= grid < n_pix)):
            raise SpecError(f'node {nid!r} grid must be an int in [0,{n_pix})')
        node = dict(id=nid, archetype=arch,
                    layer=n.get('layer') or _default_layer(arch),
                    label=n.get('label') or nid)
        if pixel is not None:
            node['pixel'] = pixel
        if grid is not None:
            node['grid'] = grid
        # Preserve recognized structural (tiled) node metadata verbatim so the topology
        # stays the single source of truth across validate/export/save/load. Unknown
        # keys are still dropped (the legacy normalization contract).
        for f in _TILED_NODE_STR_FIELDS:
            v = n.get(f)
            if v is not None:
                if not isinstance(v, str):
                    raise SpecError(f'node {nid!r} field {f!r} must be a string')
                node[f] = v
        role = node.get('column_role')
        # E/Eor/C/I are the cortical-column roles; Eor2 is the diagnostic double-relay
        # variant's second output relay (validated further by the tiled family).
        if role is not None and role not in ('E', 'Eor', 'Eor2', 'C', 'I'):
            raise SpecError(
                f'node {nid!r} column_role must be one of E/Eor/Eor2/C/I, got {role!r}')
        for f in _TILED_NODE_INT_FIELDS:
            v = n.get(f)
            if v is not None:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise SpecError(f'node {nid!r} field {f!r} must be an int')
                node[f] = v
        for f in _TILED_NODE_BOOL_FIELDS:
            v = n.get(f)
            if v is not None:
                node[f] = bool(v)
        if n.get('pos') is not None:
            pos = n['pos']
            if len(pos) != 3:
                raise SpecError(f'node {nid!r} pos must have 3 coordinates')
            node['pos'] = [float(x) for x in pos]
        seen[nid] = node
        norm_nodes.append(node)

    eids = set()
    norm_edges = []
    for e in edges:
        src, tgt, kind = e.get('source'), e.get('target'), e.get('kind')
        if kind not in EDGE_KINDS:
            raise SpecError(f'edge has unknown kind {kind!r}; valid: {sorted(EDGE_KINDS)}')
        if src not in seen or tgt not in seen:
            raise SpecError(f'edge {kind} references missing node ({src!r}->{tgt!r})')
        spec_kind = EDGE_KINDS[kind]
        src_arch, tgt_arch = seen[src]['archetype'], seen[tgt]['archetype']
        if not _arch_matches(src_arch, spec_kind['src']):
            raise SpecError(f'edge {kind}: source {src!r} ({src_arch}) is not a valid '
                            f'{_describe(spec_kind["src"])}')
        if not _arch_matches(tgt_arch, spec_kind['tgt']):
            raise SpecError(f'edge {kind}: target {tgt!r} ({tgt_arch}) is not a valid '
                            f'{_describe(spec_kind["tgt"])}')
        directed = bool(e.get('directed', True))
        if not directed and kind in DIRECTED_ONLY_KINDS:
            raise SpecError(
                f'edge {kind} {src!r}->{tgt!r} must be directed; {kind} is an '
                f'inherently one-way projection and cannot be bidirectional')
        if not directed:
            # A bidirectional edge delivers both ways, so the REVERSE direction must
            # also satisfy the kind's archetype rule (e.g. competitor<->competitor
            # feedforward). Most kinds are inherently one-way, so this rejects them.
            if not (_arch_matches(tgt_arch, spec_kind['src'])
                    and _arch_matches(src_arch, spec_kind['tgt'])):
                raise SpecError(
                    f'edge {kind} between {src!r} and {tgt!r} cannot be bidirectional: '
                    f'the reverse direction is not a valid {kind} (source must be '
                    f'{_describe(spec_kind["src"])}, target {_describe(spec_kind["tgt"])}). '
                    f'Use two directed edges.')
        eid = e.get('id') or f'{kind}:{src}->{tgt}'
        if eid in eids:
            raise SpecError(f'duplicate edge id {eid!r}')
        eids.add(eid)
        ne = dict(id=eid, source=src, target=tgt, kind=kind, directed=directed)
        if spec_kind['sign'] is not None:
            ne['sign'] = spec_kind['sign']
        # Preserve the projection-family metadata (validation/layout/dashboard filter);
        # it never changes simulation dispatch, which stays keyed on ``kind``.
        for f in _EDGE_META_STR_FIELDS:
            v = e.get(f)
            if v is not None:
                if not isinstance(v, str):
                    raise SpecError(f'edge {eid!r} field {f!r} must be a string')
                ne[f] = v
        norm_edges.append(ne)

    # --- structural invariants the endpoint rules alone do not spell out ---------
    for ne in norm_edges:
        # An rg_source is exogenous: it owns no membrane and no conductance state, so
        # nothing may deliver to it. Every current edge kind already rejects an 'S'
        # target via its tgt rule; this makes the invariant explicit and future-proof.
        for endpoint in ((ne['target'],) if ne['directed'] else (ne['target'], ne['source'])):
            if seen[endpoint]['archetype'] == 'rg_source':
                raise SpecError(
                    f"edge {ne['kind']} {ne['source']!r}->{ne['target']!r} targets "
                    f"{endpoint!r}, an rg_source. RG cells are exogenous spike sources: "
                    f"they cannot be inhibited or driven by any edge.")
    for n in norm_nodes:
        if n['archetype'] == 'rg_source' and n.get('pixel') is None:
            raise SpecError(f'rg_source {n["id"]!r} must own a pixel (its only drive is '
                            f'the external input for that pixel)')

    # --- top-level tiling metadata (single source of truth; validated + preserved) ---
    topo_meta = _validate_topology_metadata(spec.get('topology'))
    # A C whose column has NO parent is intentionally dormant and legal with zero apical
    # inputs. This exception is gated on validated column metadata (parent_ids) -- never
    # on a node id -- so an accidentally-unwired non-top C is still rejected below.
    dormant_c = set()
    if topo_meta is not None:
        parent_of = {c['id']: c.get('parent_ids') or [] for c in topo_meta['columns']}
        for n in norm_nodes:
            if n['archetype'] == 'e_coincidence':
                cid = n.get('column_id')
                if cid is not None and cid in parent_of and not parent_of[cid]:
                    dormant_c.add(n['id'])

    # --- coincidence / event-resolved structural invariants ----------------------
    kinds_present = {ne['kind'] for ne in norm_edges}

    # Duplicate basal/apical edges from the same source to the same target: a C cell
    # must not receive the same afferent twice on one compartment.
    for compartment_kind in ('basal_excitation', 'apical_excitation'):
        seen_pairs = set()
        for ne in norm_edges:
            if ne['kind'] != compartment_kind:
                continue
            pair = (ne['source'], ne['target'])
            if pair in seen_pairs:
                raise SpecError(
                    f'duplicate {compartment_kind} edge {ne["source"]!r}->{ne["target"]!r}; '
                    f'a coincidence cell may not receive the same afferent twice')
            seen_pairs.add(pair)

    # Every e_coincidence needs EXACTLY one incoming basal edge and >= one apical edge.
    basal_in, apical_in = {}, {}
    for ne in norm_edges:
        if ne['kind'] == 'basal_excitation':
            basal_in[ne['target']] = basal_in.get(ne['target'], 0) + 1
        elif ne['kind'] == 'apical_excitation':
            apical_in[ne['target']] = apical_in.get(ne['target'], 0) + 1
    for n in norm_nodes:
        if n['archetype'] != 'e_coincidence':
            continue
        nb = basal_in.get(n['id'], 0)
        # A cell that has NOT opted in to the multi-basal capability keeps the historical
        # exactly-one invariant. An opted-in cell needs >= 1; the duplicate check above
        # already guarantees every basal afferent is source-distinct.
        if n.get('multi_basal'):
            if nb < 1:
                raise SpecError(
                    f'multi-basal coincidence cell {n["id"]!r} must have at least one '
                    f'incoming basal_excitation edge, found 0')
        elif nb != 1:
            raise SpecError(
                f'coincidence cell {n["id"]!r} must have exactly one incoming '
                f'basal_excitation edge, found {nb}')
        napi = apical_in.get(n['id'], 0)
        if n['id'] in dormant_c:
            # A declared-dormant top C must have EXACTLY zero apicals (an apical would
            # contradict the no-parent declaration).
            if napi != 0:
                raise SpecError(
                    f'dormant top coincidence cell {n["id"]!r} (its column has no parent) '
                    f'must have zero apical_excitation edges, found {napi}')
        elif napi < 1:
            raise SpecError(
                f'coincidence cell {n["id"]!r} must have at least one incoming '
                f'apical_excitation edge, found 0')

    # A graph that uses hard_reset_inhibition commits every excitatory target to
    # event-resolved semantics, so every E-class node must be an event-resolved
    # archetype (never a legacy synchronous E cell).
    if 'hard_reset_inhibition' in kinds_present:
        for n in norm_nodes:
            a = n['archetype']
            if ARCHETYPES[a]['cls'] == 'E' and not ARCHETYPES[a]['event_resolved']:
                raise SpecError(
                    f'edge kind hard_reset_inhibition requires every excitatory target to '
                    f'be event-resolved, but {n["id"]!r} is a legacy {a!r}. Use the '
                    f'event-resolved archetypes {sorted(EVENT_RESOLVED_ARCHETYPES)}.')

    # Two conflicting within-boundary event-ordering semantics may not coexist: legacy
    # engine-arbitrated e_competitor WTA must not mix with any event-resolved archetype
    # or hard reset.
    has_legacy_wta = any(n['archetype'] == 'e_competitor' for n in norm_nodes)
    has_event = (any(ARCHETYPES[n['archetype']]['event_resolved'] for n in norm_nodes)
                 or 'hard_reset_inhibition' in kinds_present)
    if has_legacy_wta and has_event:
        raise SpecError(
            'a spec may not mix legacy engine-arbitrated e_competitor WTA with '
            'event-resolved archetypes or hard_reset_inhibition: the two define '
            'conflicting within-boundary event-ordering semantics. Use '
            'e_latency_competitor for event-resolved competition.')

    # --- tiled-family structural validation (only when the graph declares it) --------
    # Branch on the validated ``variant`` metadata, NEVER a preset name or id prefix.
    if topo_meta is not None:
        variant = topo_meta.get('variant')
        if variant == TILED_VARIANT_DIRECT_IDENTITY:
            _validate_tiled_direct_identity(topo_meta, norm_nodes, norm_edges)
        else:
            # ``classic`` and ``double_eor`` share every rule; the latter simply routes the
            # column output through one extra relay, which _validate_tiled checks inline.
            _validate_tiled(topo_meta, norm_nodes, norm_edges,
                            double_eor=(variant == TILED_VARIANT_DOUBLE_EOR))

    out = dict(name=spec.get('name') or 'custom', nodes=norm_nodes, edges=norm_edges)
    if topo_meta is not None:
        out['topology'] = topo_meta
    return out


def _validate_topology_metadata(meta):
    """Validate + normalize the optional top-level ``topology`` tiling metadata. Returns
    None for a generic (non-tiled) graph, else a normalized copy. A ``topology`` object
    with any family other than the tiled one is rejected (we can only validate what we
    understand)."""
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise SpecError('spec.topology metadata must be an object')
    family = meta.get('family')
    if family != TILED_FAMILY:
        raise SpecError(
            f'unknown spec.topology family {family!r}; expected {TILED_FAMILY!r}')

    # Tiled variant: absent means the classic whole-bank column (so saved specs that omit
    # it stay byte-identical). An explicit value must be a recognized variant.
    variant = meta.get('variant')
    if variant is not None and variant not in TILED_VARIANTS:
        raise SpecError(
            f'unknown spec.topology.variant {variant!r}; expected one of {TILED_VARIANTS}')

    def _shape(key):
        s = meta.get(key)
        if (not isinstance(s, dict) or not isinstance(s.get('rows'), int)
                or not isinstance(s.get('cols'), int)
                or isinstance(s.get('rows'), bool) or isinstance(s.get('cols'), bool)
                or s['rows'] <= 0 or s['cols'] <= 0):
            raise SpecError(f'spec.topology.{key} must have positive int rows/cols')
        return dict(rows=int(s['rows']), cols=int(s['cols']))

    input_shape = _shape('input_shape')
    patch_shape = _shape('patch_shape')
    if (input_shape['rows'] % patch_shape['rows']
            or input_shape['cols'] % patch_shape['cols']):
        raise SpecError('spec.topology.patch_shape must tile input_shape exactly')

    cols_meta = meta.get('columns')
    if not isinstance(cols_meta, list) or not cols_meta:
        raise SpecError('spec.topology.columns must be a non-empty list')
    norm_cols, ids = [], set()
    for c in cols_meta:
        cid = c.get('id')
        if not isinstance(cid, str) or not cid:
            raise SpecError('each tiled column needs a string id')
        if cid in ids:
            raise SpecError(f'duplicate tiled column id {cid!r}')
        ids.add(cid)
        ec = c.get('e_count')
        if not isinstance(ec, int) or isinstance(ec, bool) or ec < 1:
            raise SpecError(f'tiled column {cid!r} e_count must be an int >= 1')
        pids = c.get('parent_ids') or []
        if not isinstance(pids, list) or any(not isinstance(p, str) for p in pids):
            raise SpecError(f'tiled column {cid!r} parent_ids must be a list of column ids')
        norm_cols.append(dict(id=cid, layer=str(c.get('layer', '')),
                              row=int(c.get('row', 0)), col=int(c.get('col', 0)),
                              e_count=int(ec), parent_ids=list(pids)))
    for c in norm_cols:
        for p in c['parent_ids']:
            if p not in ids:
                raise SpecError(f'tiled column {c["id"]!r} names unknown parent {p!r}')

    out = dict(meta)
    out['family'] = TILED_FAMILY
    if variant is not None:
        out['variant'] = variant           # preserved verbatim; omitted when absent (classic)
    out['input_shape'] = input_shape
    out['patch_shape'] = patch_shape
    out['columns'] = norm_cols
    return out


def _validate_tiled(meta, nodes, edges, *, double_eor=False):
    """Structural validation for the tiled cortical-column family. Runs only when the
    spec declares the family, so generic graphs keep their existing rules.

    ``double_eor`` switches on the diagnostic variant: each column additionally owns an
    ``Eor2`` relay fed by its ``Eor``, and ``Eor2`` -- not ``Eor`` -- is the column output
    and the C basal source. Every other rule is shared verbatim with ``classic``."""
    out_role = 'Eor2' if double_eor else 'Eor'
    ishape, pshape = meta['input_shape'], meta['patch_shape']
    n_in = ishape['rows'] * ishape['cols']
    node_by_id = {n['id']: n for n in nodes}
    columns = {c['id']: c for c in meta['columns']}
    input_layer = (meta['column_layers'][0]['layer']
                   if meta.get('column_layers') else None)

    # --- RGC surface: exactly one unique RGC per input pixel, in exactly one patch -----
    rgc = [n for n in nodes if n['archetype'] == 'rg_source']
    pixels = sorted(int(n['pixel']) for n in rgc if n.get('pixel') is not None)
    if pixels != list(range(n_in)):
        raise SpecError(
            f'tiled input_shape {ishape["rows"]}x{ishape["cols"]} requires RGC cells '
            f'owning pixels 0..{n_in - 1}; got {len(rgc)} RGC(s)')
    rg_patch = {}
    for n in rgc:
        gr, gc = n['pixel'] // ishape['cols'], n['pixel'] % ishape['cols']
        pr, pc = gr // pshape['rows'], gc // pshape['cols']
        if n.get('patch_row') not in (None, pr) or n.get('patch_col') not in (None, pc):
            raise SpecError(f'RGC {n["id"]!r} patch metadata disagrees with its pixel')
        rg_patch[n['id']] = (pr, pc)

    # --- column membership: each E/Eor/C/I belongs to exactly one declared column ------
    _slots = ('Eor', 'Eor2', 'C', 'I') if double_eor else ('Eor', 'C', 'I')
    col_roles = {cid: dict(E=[], **{r: None for r in _slots}) for cid in columns}
    for n in nodes:
        role = n.get('column_role')
        if role is None:
            continue
        if role != 'E' and role not in _slots:
            raise SpecError(f'node {n["id"]!r} declares column_role {role!r}, which this '
                            f'tiled variant does not define')
        cid = n.get('column_id')
        if cid not in columns:
            raise SpecError(f'node {n["id"]!r} names undeclared column {cid!r}')
        slot = col_roles[cid]
        if role == 'E':
            slot['E'].append(n['id'])
        elif slot[role] is not None:
            raise SpecError(f'column {cid!r} has more than one {role} node')
        else:
            slot[role] = n['id']
    role_of = {}
    for cid, c in columns.items():
        s = col_roles[cid]
        if len(s['E']) != c['e_count']:
            raise SpecError(f'column {cid!r} must contain exactly {c["e_count"]} ordinary '
                            f'E, found {len(s["E"])}')
        for role in _slots:
            if s[role] is None:
                raise SpecError(f'column {cid!r} is missing its {role}')
        for eid in s['E']:
            role_of[eid] = (cid, 'E')
        for role in _slots:
            role_of[s[role]] = (cid, role)

    # --- classify every edge; anything touching a tiled node must match one rule -------
    rg_into_E, e_to_eor, e_to_i, i_to_e = {}, {}, {}, {}
    eor_c_basal, c_to_i, link_ff, link_apical = set(), set(), {}, {}
    eor_to_eor2 = set()                       # double_eor only
    for e in edges:
        s, t, kind = e['source'], e['target'], e['kind']
        sr, tr = role_of.get(s), role_of.get(t)
        s_is_rg = node_by_id[s]['archetype'] == 'rg_source'
        if kind == 'feedforward':
            if s_is_rg:
                if not (tr and tr[1] == 'E'):
                    raise SpecError(f'RGC feedforward {s!r}->{t!r} must target an ordinary E')
                c = columns[tr[0]]
                if rg_patch[s] != (c['row'], c['col']):
                    raise SpecError(f'RGC {s!r} feeds column {tr[0]!r} outside its patch')
                rg_into_E.setdefault(t, set()).add(s)
            elif sr and sr[1] == 'E' and tr and tr[1] == 'Eor':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-column E->Eor feedforward {s!r}->{t!r}')
                e_to_eor.setdefault(sr[0], set()).add(s)
            elif double_eor and sr and sr[1] == 'Eor' and tr and tr[1] == 'Eor2':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-column Eor->Eor2 feedforward {s!r}->{t!r}')
                eor_to_eor2.add(sr[0])
            elif sr and sr[1] == out_role and tr and tr[1] == 'E':
                child, parent = sr[0], tr[0]
                if parent not in columns[child]['parent_ids']:
                    raise SpecError(f'column link {s!r}->{t!r}: {parent!r} is not a parent '
                                    f'of {child!r}')
                link_ff.setdefault(child, set()).add(t)
            else:
                raise SpecError(f'unexpected feedforward edge {s!r}->{t!r} in tiled graph '
                                f'(no lateral E-E / same-layer projection is allowed)')
        elif kind == 'relay_excitation':
            if not (tr and tr[1] == 'I'):
                raise SpecError(f'relay_excitation {s!r}->{t!r} must target a column I')
            if sr and sr[1] == 'E':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-column E->I edge {s!r}->{t!r}')
                e_to_i.setdefault(tr[0], set()).add(s)
            elif sr and sr[1] == 'C':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-column C->I edge {s!r}->{t!r}')
                c_to_i.add(tr[0])
            else:
                raise SpecError(f'unexpected relay_excitation {s!r}->{t!r} in tiled graph')
        elif kind == 'hard_reset_inhibition':
            if not (sr and sr[1] == 'I' and tr and tr[1] == 'E'):
                raise SpecError(f'hard_reset {s!r}->{t!r} must be a column I onto an ordinary E')
            if sr[0] != tr[0]:
                raise SpecError(f'column I {s!r} resets {t!r} outside its own column')
            i_to_e.setdefault(sr[0], set()).add(t)
        elif kind == 'basal_excitation':
            if not (sr and sr[1] == out_role and tr and tr[1] == 'C' and sr[0] == tr[0]):
                raise SpecError(f'basal {s!r}->{t!r} must be a column-local {out_role}->C')
            eor_c_basal.add(sr[0])
        elif kind == 'apical_excitation':
            if not (sr and sr[1] == 'E' and tr and tr[1] == 'C'):
                raise SpecError(f'apical {s!r}->{t!r} must be a parent ordinary E -> child C '
                                f'(Eor is never an apical source)')
            child = tr[0]
            if sr[0] not in columns[child]['parent_ids']:
                raise SpecError(f'apical {s!r}->{t!r}: {sr[0]!r} is not a parent of {child!r}')
            link_apical.setdefault(child, set()).add(s)
        elif sr is not None or tr is not None or s_is_rg:
            raise SpecError(f'edge kind {kind!r} ({s!r}->{t!r}) is not part of the tiled '
                            f'cortical-column family')

    # --- per-column internal completeness (missing edge is rejected here) ---------------
    for cid in columns:
        ebank = set(col_roles[cid]['E'])
        if e_to_eor.get(cid, set()) != ebank:
            raise SpecError(f'column {cid!r}: every ordinary E must feed its Eor exactly once')
        if e_to_i.get(cid, set()) != ebank:
            raise SpecError(f'column {cid!r}: every ordinary E must drive its I')
        if i_to_e.get(cid, set()) != ebank:
            raise SpecError(f'column {cid!r}: I must hard-reset exactly its own ordinary-E bank')
        if cid not in eor_c_basal:
            raise SpecError(f'column {cid!r}: {out_role} must supply its C basal')
        if double_eor and cid not in eor_to_eor2:
            raise SpecError(f'column {cid!r}: Eor must feed its Eor2 output relay')
        if cid not in c_to_i:
            raise SpecError(f'column {cid!r}: C must drive its I')

    # --- RGC-to-column completeness (input columns get all + only their patch RGCs) -----
    for cid, c in columns.items():
        is_input = (c['layer'] == input_layer)
        expect = {rid for rid, patch in rg_patch.items() if patch == (c['row'], c['col'])}
        for eid in col_roles[cid]['E']:
            got = rg_into_E.get(eid, set())
            if is_input:
                if got != expect:
                    raise SpecError(f'ordinary E {eid!r} in input column {cid!r} must receive '
                                    f'all and only the {len(expect)} RGCs of its patch')
            elif got:
                raise SpecError(f'non-input ordinary E {eid!r} must not receive RGC feedforward')

    # --- child/parent link completeness (two-way feedforward + apical) ------------------
    for cid, c in columns.items():
        for pid in c['parent_ids']:
            parent_E = set(col_roles[pid]['E'])
            if not parent_E <= link_ff.get(cid, set()):
                raise SpecError(f'column {cid!r} {out_role} must feed every ordinary E of '
                                f'parent {pid!r}')
            if not parent_E <= link_apical.get(cid, set()):
                raise SpecError(f'column {cid!r} C must receive an apical from every ordinary '
                                f'E of parent {pid!r}')


def _validate_tiled_direct_identity(meta, nodes, edges):
    """Structural validation for the Eor-less direct-identity tiled variant. Enforces the
    exact column motif (E bank + multi-basal C + WTA/feedback I, NO Eor), the local
    source-distinct E->C basal fan-in, and the source-addressed child-E -> parent-E
    projection. Runs only when the spec declares ``variant='direct_identity'``; the classic
    validator is untouched."""
    ishape, pshape = meta['input_shape'], meta['patch_shape']
    n_in = ishape['rows'] * ishape['cols']
    node_by_id = {n['id']: n for n in nodes}
    columns = {c['id']: c for c in meta['columns']}
    input_layer = (meta['column_layers'][0]['layer']
                   if meta.get('column_layers') else None)

    # --- RGC surface: exactly one unique RGC per input pixel, in exactly one patch -----
    rgc = [n for n in nodes if n['archetype'] == 'rg_source']
    pixels = sorted(int(n['pixel']) for n in rgc if n.get('pixel') is not None)
    if pixels != list(range(n_in)):
        raise SpecError(
            f'direct-identity input_shape {ishape["rows"]}x{ishape["cols"]} requires RGC '
            f'cells owning pixels 0..{n_in - 1}; got {len(rgc)} RGC(s)')
    rg_patch = {}
    for n in rgc:
        gr, gc = n['pixel'] // ishape['cols'], n['pixel'] % ishape['cols']
        pr, pc = gr // pshape['rows'], gc // pshape['cols']
        if n.get('patch_row') not in (None, pr) or n.get('patch_col') not in (None, pc):
            raise SpecError(f'RGC {n["id"]!r} patch metadata disagrees with its pixel')
        rg_patch[n['id']] = (pr, pc)

    # --- column membership: E bank + exactly one C and one I, and NEVER an Eor ---------
    col_roles = {cid: dict(E=[], C=None, I=None) for cid in columns}
    for n in nodes:
        role = n.get('column_role')
        if role is None:
            continue
        if role == 'Eor':
            raise SpecError(
                f'node {n["id"]!r} declares column_role Eor: the direct-identity variant '
                f'has no output relay -- each ordinary E addresses the parent itself')
        cid = n.get('column_id')
        if cid not in columns:
            raise SpecError(f'node {n["id"]!r} names undeclared column {cid!r}')
        slot = col_roles[cid]
        if role == 'E':
            slot['E'].append(n['id'])
        elif slot[role] is not None:
            raise SpecError(f'column {cid!r} has more than one {role} node')
        else:
            slot[role] = n['id']
    role_of = {}
    for cid, c in columns.items():
        s = col_roles[cid]
        if len(s['E']) != c['e_count']:
            raise SpecError(f'column {cid!r} must contain exactly {c["e_count"]} ordinary '
                            f'E, found {len(s["E"])}')
        for role in ('C', 'I'):
            if s[role] is None:
                raise SpecError(f'column {cid!r} is missing its {role}')
        if not node_by_id[s['C']].get('multi_basal'):
            raise SpecError(f'column {cid!r} C {s["C"]!r} must declare multi_basal: the '
                            f'direct-identity column gives it one basal per ordinary E')
        for eid in s['E']:
            role_of[eid] = (cid, 'E')
        role_of[s['C']] = (cid, 'C')
        role_of[s['I']] = (cid, 'I')

    # --- classify every edge; anything touching a tiled node must match one rule -------
    rg_into_E, e_to_i, i_to_e = {}, {}, {}
    e_c_basal, c_to_i, link_ff, link_apical = {}, set(), {}, {}
    for e in edges:
        s, t, kind = e['source'], e['target'], e['kind']
        sr, tr = role_of.get(s), role_of.get(t)
        s_is_rg = node_by_id[s]['archetype'] == 'rg_source'
        if kind == 'feedforward':
            if s_is_rg:
                if not (tr and tr[1] == 'E'):
                    raise SpecError(f'RGC feedforward {s!r}->{t!r} must target an ordinary E')
                c = columns[tr[0]]
                if rg_patch[s] != (c['row'], c['col']):
                    raise SpecError(f'RGC {s!r} feeds column {tr[0]!r} outside its patch')
                rg_into_E.setdefault(t, set()).add(s)
            elif sr and sr[1] == 'E' and tr and tr[1] == 'E':
                child, parent = sr[0], tr[0]
                if child == parent:
                    raise SpecError(f'lateral E->E feedforward {s!r}->{t!r} inside column '
                                    f'{child!r}: columns have no lateral connections')
                if parent not in columns[child]['parent_ids']:
                    raise SpecError(f'identity link {s!r}->{t!r}: {parent!r} is not a parent '
                                    f'of {child!r}')
                link_ff.setdefault(child, {}).setdefault(s, set()).add(t)
            else:
                raise SpecError(f'unexpected feedforward edge {s!r}->{t!r} in direct-identity '
                                f'graph (only RGC->E and child-E->parent-E are allowed)')
        elif kind == 'relay_excitation':
            if not (tr and tr[1] == 'I'):
                raise SpecError(f'relay_excitation {s!r}->{t!r} must target a column I')
            if sr and sr[1] == 'E':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-column E->I edge {s!r}->{t!r}')
                e_to_i.setdefault(tr[0], set()).add(s)
            elif sr and sr[1] == 'C':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-column C->I edge {s!r}->{t!r}')
                c_to_i.add(tr[0])
            else:
                raise SpecError(f'unexpected relay_excitation {s!r}->{t!r} in tiled graph')
        elif kind == 'hard_reset_inhibition':
            if not (sr and sr[1] == 'I' and tr and tr[1] == 'E'):
                raise SpecError(f'hard_reset {s!r}->{t!r} must be a column I onto an ordinary E')
            if sr[0] != tr[0]:
                raise SpecError(f'column I {s!r} resets {t!r} outside its own column')
            i_to_e.setdefault(sr[0], set()).add(t)
        elif kind == 'basal_excitation':
            if not (sr and sr[1] == 'E' and tr and tr[1] == 'C'):
                raise SpecError(f'basal {s!r}->{t!r} must be an ordinary E -> its column C')
            if sr[0] != tr[0]:
                raise SpecError(f'cross-column basal {s!r}->{t!r}: a C receives local '
                                f'bottom-up evidence only')
            e_c_basal.setdefault(tr[0], set()).add(s)
        elif kind == 'apical_excitation':
            if not (sr and sr[1] == 'E' and tr and tr[1] == 'C'):
                raise SpecError(f'apical {s!r}->{t!r} must be a parent ordinary E -> child C')
            child = tr[0]
            if sr[0] not in columns[child]['parent_ids']:
                raise SpecError(f'apical {s!r}->{t!r}: {sr[0]!r} is not a parent of {child!r}')
            link_apical.setdefault(child, set()).add(s)
        elif sr is not None or tr is not None or s_is_rg:
            raise SpecError(f'edge kind {kind!r} ({s!r}->{t!r}) is not part of the '
                            f'direct-identity tiled cortical-column family')

    # --- per-column internal completeness (a missing edge is rejected here) ------------
    for cid in columns:
        ebank = set(col_roles[cid]['E'])
        if e_to_i.get(cid, set()) != ebank:
            raise SpecError(f'column {cid!r}: every ordinary E must drive its I')
        if i_to_e.get(cid, set()) != ebank:
            raise SpecError(f'column {cid!r}: I must hard-reset exactly its own ordinary-E bank')
        if e_c_basal.get(cid, set()) != ebank:
            raise SpecError(f'column {cid!r}: its C must receive one source-distinct basal '
                            f'from every ordinary E of the column')
        if cid not in c_to_i:
            raise SpecError(f'column {cid!r}: C must drive its I')

    # --- RGC-to-column completeness (input columns get all + only their patch RGCs) ----
    for cid, c in columns.items():
        is_input = (c['layer'] == input_layer)
        expect = {rid for rid, patch in rg_patch.items() if patch == (c['row'], c['col'])}
        for eid in col_roles[cid]['E']:
            got = rg_into_E.get(eid, set())
            if is_input:
                if got != expect:
                    raise SpecError(f'ordinary E {eid!r} in input column {cid!r} must receive '
                                    f'all and only the {len(expect)} RGCs of its patch')
            elif got:
                raise SpecError(f'non-input ordinary E {eid!r} must not receive RGC feedforward')

    # --- identity + apical link completeness ------------------------------------------
    # EVERY child ordinary E addresses EVERY parent ordinary E, so no child winner can be
    # unrepresented at the parent (the deadlock the removed Eor relay suffered).
    for cid, c in columns.items():
        for pid in c['parent_ids']:
            parent_E = set(col_roles[pid]['E'])
            reached = link_ff.get(cid, {})
            for eid in col_roles[cid]['E']:
                if reached.get(eid, set()) != parent_E:
                    raise SpecError(f'ordinary E {eid!r} must address every ordinary E of '
                                    f'parent {pid!r} (source-addressed identity projection)')
            if link_apical.get(cid, set()) != parent_E:
                raise SpecError(f'column {cid!r} C must receive an apical from every ordinary '
                                f'E of parent {pid!r}')


def _default_layer(arch: str) -> str:
    if arch == 'rg_source':
        return 'RG'
    if arch in ('e_sensory', 'e_encoder'):
        return 'L1'
    if arch == 'e_residual':
        return 'ERR'
    return 'L2'


def _describe(requirement) -> str:
    if isinstance(requirement, (tuple, list)):
        return ' or '.join(str(r) for r in requirement)
    return str(requirement)


def _arch_matches(arch: str, requirement) -> bool:
    """A requirement is an archetype name, a class letter ('E'/'I'/'S'), or a tuple of
    either -- satisfied if ANY alternative matches."""
    if isinstance(requirement, (tuple, list)):
        return any(_arch_matches(arch, r) for r in requirement)
    if requirement in ARCHETYPES:
        return arch == requirement
    return ARCHETYPES[arch]['cls'] == requirement
