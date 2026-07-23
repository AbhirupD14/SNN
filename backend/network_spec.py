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
    # feature-gated variant: the 0..(patch_size-1) index of a feature relay/C/I inside its
    # recognition module (never parsed from an id -- validation/layout/dashboard read this).
    'feature_index',
)
_TILED_NODE_STR_FIELDS = ('column_id', 'column_role')
_TILED_NODE_BOOL_FIELDS = ('has_parent',)
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
PRESETS = ('rg_coincidence', 'tiled_cc', 'tiled_cc_l1_4', 'tiled_cc_feature_gated',
           'rg_direct_cc4')

# Built-in presets that build a tiled cortical-column hierarchy on the fixed 81-pixel
# surface (as opposed to the legacy n_pix/n_out presets). Used for size resolution and
# the fixed-input guard so new tiled presets are handled without name-by-name branching.
TILED_PRESETS = ('tiled_cc', 'tiled_cc_l1_4', 'tiled_cc_feature_gated')

# Tiled-family topology variants. The top-level ``topology.variant`` field is the single
# source of truth construction/validation/layout branch on -- NEVER the preset name or a
# node-id prefix. ``classic`` (the historical whole-bank C/I column, and the DEFAULT when
# the field is absent so old saved specs stay byte-identical) vs ``feature_gated`` (the
# nine paired feature C/I gates + a separate WTA I per recognition module).
TILED_VARIANT_CLASSIC = 'classic'
TILED_VARIANT_FEATURE_GATED = 'feature_gated'
TILED_VARIANTS = (TILED_VARIANT_CLASSIC, TILED_VARIANT_FEATURE_GATED)

# Canonical construction dimensions for the tiled cortical-column preset. The input
# surface is fixed; only ``cc_e_count`` (ordinary E per column) is configurable.
TILED_CC_DEFAULTS = dict(input_rows=9, input_cols=9, patch_rows=3, patch_cols=3,
                         cc_e_count=8)


def _e_sensory(i, pixel):
    return dict(id=f'L1E{i}', archetype='e_sensory', layer='L1', pixel=pixel,
                label=f'L1E_s{i}')


def preset_spec(name: str, n_pix: int, n_out: int, cc_e_count: int = 8) -> dict:
    """Return a built-in NetworkSpec. Positions are omitted so the engine fills them from
    the seeded functional layout (bit-exact presets).

    The public built-ins are exactly ``PRESETS`` = (rg_coincidence, tiled_cc,
    tiled_cc_l1_4); any other name is rejected. ``cc_e_count`` sizes the ordinary-E bank
    of every column in the ``tiled_cc`` preset and is ignored otherwise. The obsolete
    pi/old/rg/rg_residual builders below are unreachable through this public entry point
    (guarded by ``PRESETS``) and remain only as reusable low-level graph mechanics."""
    if name not in PRESETS:
        raise ValueError(f'unknown preset {name!r}')
    if name == 'tiled_cc':
        return tiled_cc_spec(cc_e_count=cc_e_count)
    if name == 'tiled_cc_l1_4':
        # Identical to tiled_cc but with a shallower L1 bank: four ordinary E per L1
        # column, eight in L2. Fixed-shape (does not read cc_e_count).
        return tiled_cc_spec(l1_e_count=4, l2_e_count=8, name='tiled_cc_l1_4')
    if name == 'tiled_cc_feature_gated':
        # Fixed-shape eight-competitor feature-gated tiled variant (L1=8, L2=8): nine paired
        # feature C/I gates per 3x3 RF + a separate WTA I per module. Does not read cc_e_count.
        return tiled_cc_feature_gated_spec()
    if name == 'rg_direct_cc4':
        # Direct 3x3 RGC -> four ordinary E + one central WTA I (dual FE/FES experiment).
        # Fixed 3x3/4-E shape (does not read n_out); n_pix sizes the RGC surface.
        return rg_direct_cc4_spec(n_pix=n_pix)
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
# Feature-gated tiled cortical-column construction
# =====================================================================================
# The eight-competitor variant that restores the feature-specific inhibitory microcircuit
# of ``rg_coincidence``. Between each 3x3 RGC patch and its L1 competitor bank sit nine
# fixed feature relays, each with its own paired coincidence C and feature inhibitory If.
# The bank keeps a completely separate WTA-only I. This is exactly the small circuit
# (S == pretrained L1E relay, E == latency competitor, C == L1C, If == L1I) replicated
# once per feature inside every recognition module, so a mature local owner suppresses only
# the explained feature relays while novel relays stay active to recruit a rival.

@dataclass(frozen=True)
class WtaBankHandles:
    """Handles to one WTA competitor bank (used for both the L1 recognition module's
    competitor pool and the top L2 module): the ordinary E ids, the shared Eor output, and
    the WTA-only I. It carries NO C -- the feature gates own coincidence in this variant."""
    module_id: str
    layer: str
    row: int
    col: int
    e_ids: tuple[str, ...]
    eor_id: str
    wta_i_id: str


@dataclass(frozen=True)
class FeatureGateHandles:
    """Handles to one feature relay S[k] and its paired coincidence C[k] / feature
    inhibitory If[k] inside a recognition module -- the composition surface the RGC and the
    competitor bank bind against."""
    module_id: str
    feature_index: int
    rgc_id: str
    s_id: str
    c_id: str
    if_id: str


def build_wta_bank(module_id: str, layer: str, row: int, col: int, *,
                   n_e: int) -> tuple[WtaBankHandles, list, list]:
    """Emit ONE WTA competitor bank's nodes + internal edges (no external wiring).

    Nodes: ``n_e`` ordinary E (``e_latency_competitor``), one Eor (same archetype,
    ``column_role='Eor'``), one WTA-only I (``i_relay``, ``column_role='I'``). Internal
    edges (the fixed WTA rule)::

        E[i] -> Eor    feedforward            (bank_e_to_eor)
        E[i] -> Iwta   relay_excitation       (bank_e_to_wta_i)
        Iwta -> E[i]   hard_reset_inhibition  (bank_wta_i_to_e)

    There is no C and no basal here: the WTA I performs ONLY same-event winner-take-all.
    Returns fresh lists every call."""
    n_e = int(n_e)
    if n_e < 1:
        raise ValueError(f'n_e must be >= 1, got {n_e}')
    e_ids = tuple(f'{module_id}E{i}' for i in range(n_e))
    eor_id, wta_i_id = f'{module_id}Eor', f'{module_id}Iwta'

    def _common(role):
        return dict(column_id=module_id, column_role=role,
                    column_row=int(row), column_col=int(col))

    nodes: list = []
    for i, eid in enumerate(e_ids):
        nodes.append(dict(id=eid, archetype='e_latency_competitor', layer=layer,
                          label=f'{module_id}·E{i}', column_index=i, **_common('E')))
    nodes.append(dict(id=eor_id, archetype='e_latency_competitor', layer=layer,
                      label=f'{module_id}·Eor', **_common('Eor')))
    nodes.append(dict(id=wta_i_id, archetype='i_relay', layer=layer,
                      label=f'{module_id}·Iwta', **_common('I')))

    edges: list = []
    for i, eid in enumerate(e_ids):
        edges.append(dict(id=f'{module_id}_E{i}_eor', source=eid, target=eor_id,
                          kind='feedforward', projection='bank_e_to_eor'))
        edges.append(dict(id=f'{module_id}_E{i}_wtai', source=eid, target=wta_i_id,
                          kind='relay_excitation', projection='bank_e_to_wta_i'))
        edges.append(dict(id=f'{module_id}_wtai_E{i}', source=wta_i_id, target=eid,
                          kind='hard_reset_inhibition', sign=-1,
                          projection='bank_wta_i_to_e'))
    handles = WtaBankHandles(module_id, layer, int(row), int(col), e_ids, eor_id, wta_i_id)
    return handles, nodes, edges


def build_feature_gate(bank: WtaBankHandles, feature_index: int, rgc_id: str, *,
                       pixel: int, input_row: int, input_col: int, patch_id: int,
                       ) -> tuple[FeatureGateHandles, list, list]:
    """Emit ONE feature relay and its paired C/I gate for feature ``feature_index`` of a
    recognition module, wired against that module's competitor bank::

        RGC   -> S     pretrained_excitation  (rgc_to_feature_relay)
        S     -> E[j]  feedforward            (feature_relay_to_e)   [every bank E]
        S     -> C     basal_excitation       (feature_relay_to_c_basal)
        E[j]  -> C     apical_excitation       (e_to_feature_c_apical) [every bank E]
        C     -> If    relay_excitation       (feature_c_to_i)
        If    -> S     hard_reset_inhibition   (feature_i_to_relay)

    The relay S is a fixed pretrained cell (the suppressible object); C is an ordinary
    coincidence cell whose single basal source is its paired S and whose apical permission
    comes from all eight local competitors; If resets ONLY S. Returns fresh lists."""
    mid = bank.module_id
    k = int(feature_index)
    s_id, c_id, if_id = f'{mid}S{k}', f'{mid}C{k}', f'{mid}If{k}'

    def _common(role):
        return dict(column_id=mid, column_role=role, feature_index=k,
                    column_row=bank.row, column_col=bank.col)

    nodes: list = [
        dict(id=s_id, archetype='e_pretrained', layer=bank.layer,
             label=f'{mid}·S{k}', grid=int(pixel), input_row=int(input_row),
             input_col=int(input_col), patch_id=int(patch_id), **_common('S')),
        dict(id=c_id, archetype='e_coincidence', layer=bank.layer,
             label=f'{mid}·C{k}', has_parent=True, **_common('C')),
        dict(id=if_id, archetype='i_relay', layer=bank.layer,
             label=f'{mid}·If{k}', **_common('If')),
    ]

    edges: list = [dict(id=f'{rgc_id}_{s_id}', source=rgc_id, target=s_id,
                        kind='pretrained_excitation', projection='rgc_to_feature_relay')]
    for j, eid in enumerate(bank.e_ids):
        edges.append(dict(id=f'{s_id}_{eid}', source=s_id, target=eid,
                          kind='feedforward', projection='feature_relay_to_e'))
        edges.append(dict(id=f'{eid}_{c_id}', source=eid, target=c_id,
                          kind='apical_excitation', projection='e_to_feature_c_apical'))
    edges.append(dict(id=f'{s_id}_{c_id}_basal', source=s_id, target=c_id,
                      kind='basal_excitation', projection='feature_relay_to_c_basal'))
    edges.append(dict(id=f'{c_id}_{if_id}', source=c_id, target=if_id,
                      kind='relay_excitation', projection='feature_c_to_i'))
    edges.append(dict(id=f'{if_id}_{s_id}', source=if_id, target=s_id,
                      kind='hard_reset_inhibition', sign=-1, projection='feature_i_to_relay'))

    handles = FeatureGateHandles(mid, k, rgc_id, s_id, c_id, if_id)
    return handles, nodes, edges


def connect_bank_feedforward(child: WtaBankHandles, parent: WtaBankHandles) -> list:
    """Emit the child.Eor -> every parent ordinary E feedforward link (``bank_to_bank_ff``).
    Feature-gated L1 modules feed L2 with NO apical feedback: the local owner drives its own
    feature gates, so nothing descends from L2 into an L1 bank."""
    return [dict(id=f'{child.eor_id}_{pe}', source=child.eor_id, target=pe,
                 kind='feedforward', projection='bank_to_bank_ff')
            for pe in parent.e_ids]


def tiled_cc_feature_gated_spec(*, l1_e_count: int = 8, l2_e_count: int = 8,
                                name: str = 'tiled_cc_feature_gated',
                                input_rows: int = 9, input_cols: int = 9,
                                patch_rows: int = 3, patch_cols: int = 3) -> dict:
    """Compose the eight-competitor feature-gated tiled hierarchy: a 9x9 RGC surface tiled
    into nine 3x3 patches; nine L1 recognition modules (one per patch, each = eight ordinary
    competitors + Eor + a WTA-only I, wrapped by nine paired feature relay/C/I gates); and
    one top L2 WTA bank receiving all nine L1 Eor outputs.

    At the default 8/8 sizing this is exactly 424 nodes and 1932 directed edges. The
    returned spec carries ``topology.variant='feature_gated'`` -- the single source of truth
    construction/layout/validation branch on."""
    l1_n = int(l1_e_count)
    l2_n = int(l2_e_count)
    if l1_n < 1 or l2_n < 1:
        raise ValueError(f'ordinary-E counts must be >= 1, got L1={l1_n}, L2={l2_n}')
    if input_rows % patch_rows != 0 or input_cols % patch_cols != 0:
        raise ValueError(
            f'patch shape ({patch_rows}x{patch_cols}) must tile the input '
            f'({input_rows}x{input_cols}) exactly')
    grid_rows, grid_cols = input_rows // patch_rows, input_cols // patch_cols

    nodes: list = []
    internal_edges: list = []
    link_edges: list = []

    # RGC surface (row-major pixels); per patch the RGCs are kept in patch-local row-major
    # order so feature index k maps 1:1 to patch-local pixel k.
    rg_by_patch: dict[tuple[int, int], list[dict]] = {}
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
            rg_by_patch.setdefault((pr, pc), []).append(
                dict(id=rid, pixel=pixel, input_row=gr, input_col=gc, patch_id=patch_id))

    # L1 recognition modules: one per patch, each a WTA bank wrapped by nine feature gates.
    l1: dict[tuple[int, int], WtaBankHandles] = {}
    for pr in range(grid_rows):
        for pc in range(grid_cols):
            mid = f'L1m{pr}{pc}'
            patch_id = pr * grid_cols + pc
            bank, bn, be = build_wta_bank(mid, 'L1', pr, pc, n_e=l1_n)
            for node in bn:
                if node.get('column_role') == 'E':
                    node['patch'] = patch_id          # display-only patch tag (as classic)
            nodes += bn
            internal_edges += be
            l1[(pr, pc)] = bank
            for k, rg in enumerate(rg_by_patch[(pr, pc)]):
                _, gn, ge = build_feature_gate(
                    bank, k, rg['id'], pixel=rg['pixel'], input_row=rg['input_row'],
                    input_col=rg['input_col'], patch_id=rg['patch_id'])
                nodes += gn
                internal_edges += ge

    # One top L2 WTA bank (no feature gates, no C: this variant does not test composition).
    l2, l2n, l2e = build_wta_bank('L2m00', 'L2', 0, 0, n_e=l2_n)
    nodes += l2n
    internal_edges += l2e

    # Feedforward: every L1 Eor -> every L2 ordinary E (no L2 -> L1 apical feedback).
    for pr in range(grid_rows):
        for pc in range(grid_cols):
            link_edges += connect_bank_feedforward(l1[(pr, pc)], l2)

    edges = internal_edges + link_edges

    columns_meta = [dict(id=l1[(pr, pc)].module_id, layer='L1', row=pr, col=pc,
                         e_count=l1_n, parent_ids=['L2m00'])
                    for pr in range(grid_rows) for pc in range(grid_cols)]
    columns_meta.append(dict(id='L2m00', layer='L2', row=0, col=0,
                             e_count=l2_n, parent_ids=[]))
    topology_meta = dict(
        family=TILED_FAMILY,
        variant=TILED_VARIANT_FEATURE_GATED,
        input_shape=dict(rows=input_rows, cols=input_cols),
        patch_shape=dict(rows=patch_rows, cols=patch_cols),
        grid_shape=dict(rows=grid_rows, cols=grid_cols),
        column_layers=[dict(layer='L1', rows=grid_rows, cols=grid_cols),
                       dict(layer='L2', rows=1, cols=1)],
        cc_e_count=l1_n,
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
        # E/Eor/C/I are the classic column roles; S/If are the feature-gated variant's
        # feature relay and paired feature inhibitory roles (validated further per variant).
        if role is not None and role not in ('E', 'Eor', 'C', 'I', 'S', 'If'):
            raise SpecError(
                f'node {nid!r} column_role must be one of E/Eor/C/I/S/If, got {role!r}')
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
        if nb != 1:
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
        if topo_meta.get('variant') == TILED_VARIANT_FEATURE_GATED:
            _validate_tiled_feature_gated(topo_meta, norm_nodes, norm_edges)
        else:
            _validate_tiled(topo_meta, norm_nodes, norm_edges)

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

    # Tiled variant: absent means the classic whole-bank column (so old saved specs stay
    # byte-identical). An explicit value must be a recognized variant.
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


def _validate_tiled(meta, nodes, edges):
    """Structural validation for the tiled cortical-column family. Runs only when the
    spec declares the family, so generic graphs keep their existing rules."""
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
    col_roles = {cid: dict(E=[], Eor=None, C=None, I=None) for cid in columns}
    for n in nodes:
        role = n.get('column_role')
        if role is None:
            continue
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
        for role in ('Eor', 'C', 'I'):
            if s[role] is None:
                raise SpecError(f'column {cid!r} is missing its {role}')
        for eid in s['E']:
            role_of[eid] = (cid, 'E')
        role_of[s['Eor']] = (cid, 'Eor')
        role_of[s['C']] = (cid, 'C')
        role_of[s['I']] = (cid, 'I')

    # --- classify every edge; anything touching a tiled node must match one rule -------
    rg_into_E, e_to_eor, e_to_i, i_to_e = {}, {}, {}, {}
    eor_c_basal, c_to_i, link_ff, link_apical = set(), set(), {}, {}
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
            elif sr and sr[1] == 'Eor' and tr and tr[1] == 'E':
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
            if not (sr and sr[1] == 'Eor' and tr and tr[1] == 'C' and sr[0] == tr[0]):
                raise SpecError(f'basal {s!r}->{t!r} must be a column-local Eor->C')
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
            raise SpecError(f'column {cid!r}: Eor must supply its C basal')
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
                raise SpecError(f'column {cid!r} Eor must feed every ordinary E of parent {pid!r}')
            if not parent_E <= link_apical.get(cid, set()):
                raise SpecError(f'column {cid!r} C must receive an apical from every ordinary '
                                f'E of parent {pid!r}')


def _validate_tiled_feature_gated(meta, nodes, edges):
    """Structural validation for the feature-gated tiled variant. Enforces the exact local
    invariants of the nine paired feature C/I gates + separate WTA I per recognition module,
    and the L1 Eor -> L2 feedforward with NO L2 -> L1 apical feedback. Runs only when the
    spec declares ``variant='feature_gated'``; the classic validator is untouched."""
    ishape, pshape = meta['input_shape'], meta['patch_shape']
    n_in = ishape['rows'] * ishape['cols']
    n_feat = pshape['rows'] * pshape['cols']
    node_by_id = {n['id']: n for n in nodes}
    columns = {c['id']: c for c in meta['columns']}
    input_layer = (meta['column_layers'][0]['layer']
                   if meta.get('column_layers') else None)

    # --- RGC surface: exactly one unique RGC per input pixel, each in exactly one patch ---
    rgc = [n for n in nodes if n['archetype'] == 'rg_source']
    pixels = sorted(int(n['pixel']) for n in rgc if n.get('pixel') is not None)
    if pixels != list(range(n_in)):
        raise SpecError(
            f'feature-gated input_shape {ishape["rows"]}x{ishape["cols"]} requires RGC cells '
            f'owning pixels 0..{n_in - 1}; got {len(rgc)} RGC(s)')
    rg_patch, rg_feature = {}, {}
    for n in rgc:
        gr, gc = n['pixel'] // ishape['cols'], n['pixel'] % ishape['cols']
        pr, pc = gr // pshape['rows'], gc // pshape['cols']
        plr, plc = gr % pshape['rows'], gc % pshape['cols']
        rg_patch[n['id']] = (pr, pc)
        rg_feature[n['id']] = plr * pshape['cols'] + plc     # patch-local feature index

    # --- classify every tiled node by (module, role[, feature_index]) ---------------------
    mods = {cid: dict(E=[], Eor=None, I=None, S={}, C={}, If={}) for cid in columns}
    role_of = {}                     # node id -> (module_id, role, feature_index|None)
    for n in nodes:
        role = n.get('column_role')
        if role is None:
            continue
        cid = n.get('column_id')
        if cid not in columns:
            raise SpecError(f'node {n["id"]!r} names undeclared module {cid!r}')
        slot = mods[cid]
        if role == 'E':
            slot['E'].append(n['id'])
            role_of[n['id']] = (cid, 'E', None)
        elif role in ('Eor', 'I'):
            if slot[role] is not None:
                raise SpecError(f'module {cid!r} has more than one {role} node')
            slot[role] = n['id']
            role_of[n['id']] = (cid, role, None)
        else:                        # 'S' / 'C' / 'If' feature-gate roles
            fk = n.get('feature_index')
            if not isinstance(fk, int) or isinstance(fk, bool):
                raise SpecError(f'feature node {n["id"]!r} ({role}) needs an int feature_index')
            if not (0 <= fk < n_feat):
                raise SpecError(f'feature node {n["id"]!r} feature_index {fk} out of range '
                                f'[0,{n_feat})')
            if fk in slot[role]:
                raise SpecError(f'module {cid!r} has more than one {role} for feature {fk}')
            slot[role][fk] = n['id']
            role_of[n['id']] = (cid, role, fk)

    features = set(range(n_feat))
    for cid, c in columns.items():
        s = mods[cid]
        if len(s['E']) != c['e_count']:
            raise SpecError(f'module {cid!r} must contain exactly {c["e_count"]} ordinary E, '
                            f'found {len(s["E"])}')
        for role in ('Eor', 'I'):
            if s[role] is None:
                raise SpecError(f'module {cid!r} is missing its {role}')
        is_input = (c['layer'] == input_layer)
        for role in ('S', 'C', 'If'):
            got = set(s[role])
            if is_input:
                if got != features:
                    raise SpecError(f'input module {cid!r} must have exactly one {role} per '
                                    f'feature 0..{n_feat - 1}, found {sorted(got)}')
            elif got:
                raise SpecError(f'non-input module {cid!r} must have no feature {role} nodes')

    # --- classify every edge; anything touching a tiled node must match one rule ----------
    rgc_to_s, s_to_e, e_to_eor, e_to_wta = {}, {}, {}, {}
    s_to_c_basal, e_to_c_apical = {}, {}
    c_to_if, if_to_s, wta_to_e = {}, {}, {}
    link_ff = {}
    for e in edges:
        s, t, kind = e['source'], e['target'], e['kind']
        sr, tr = role_of.get(s), role_of.get(t)
        s_is_rg = node_by_id[s]['archetype'] == 'rg_source'
        if kind == 'pretrained_excitation':
            if not (s_is_rg and tr and tr[1] == 'S'):
                raise SpecError(f'pretrained_excitation {s!r}->{t!r} must be an RGC onto a '
                                f'feature relay S')
            if rg_patch[s] != (columns[tr[0]]['row'], columns[tr[0]]['col']):
                raise SpecError(f'RGC {s!r} feeds feature relay {t!r} outside its patch')
            if rg_feature[s] != tr[2]:
                raise SpecError(f'RGC {s!r} (patch-feature {rg_feature[s]}) feeds relay {t!r} '
                                f'for feature {tr[2]}: feature/pixel mismatch')
            rgc_to_s.setdefault(t, set()).add(s)
        elif kind == 'feedforward':
            if sr and sr[1] == 'S' and tr and tr[1] == 'E':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-module feature relay->E feedforward {s!r}->{t!r}')
                s_to_e.setdefault(s, set()).add(t)
            elif sr and sr[1] == 'E' and tr and tr[1] == 'Eor':
                if sr[0] != tr[0]:
                    raise SpecError(f'cross-module E->Eor feedforward {s!r}->{t!r}')
                e_to_eor.setdefault(sr[0], set()).add(s)
            elif sr and sr[1] == 'Eor' and tr and tr[1] == 'E':
                child, parent = sr[0], tr[0]
                if parent not in columns[child]['parent_ids']:
                    raise SpecError(f'bank link {s!r}->{t!r}: {parent!r} is not a parent of '
                                    f'{child!r}')
                link_ff.setdefault(child, set()).add(t)
            else:
                raise SpecError(f'unexpected feedforward edge {s!r}->{t!r} in feature-gated '
                                f'graph (only RGC-relay->E, E->Eor, Eor->parent-E allowed)')
        elif kind == 'basal_excitation':
            if not (sr and sr[1] == 'S' and tr and tr[1] == 'C'
                    and sr[0] == tr[0] and sr[2] == tr[2]):
                raise SpecError(f'basal {s!r}->{t!r} must be a feature relay S[k]->paired C[k] '
                                f'in the same module')
            s_to_c_basal.setdefault(t, set()).add(s)
        elif kind == 'apical_excitation':
            if not (sr and sr[1] == 'E' and tr and tr[1] == 'C'):
                raise SpecError(f'apical {s!r}->{t!r} must be a LOCAL ordinary E -> feature C '
                                f'(no L2->L1 feedback in this variant)')
            if sr[0] != tr[0]:
                raise SpecError(f'apical {s!r}->{t!r}: feature C receives apical only from its '
                                f'own module ordinary E, not module {sr[0]!r}')
            e_to_c_apical.setdefault(t, set()).add(s)
        elif kind == 'relay_excitation':
            if not (tr and tr[1] in ('I', 'If')):
                raise SpecError(f'relay_excitation {s!r}->{t!r} must target a WTA I or feature If')
            if tr[1] == 'I':                          # WTA relay: only ordinary E may drive it
                if not (sr and sr[1] == 'E' and sr[0] == tr[0]):
                    raise SpecError(f'WTA relay drive {s!r}->{t!r} must be a same-module '
                                    f'ordinary E (no feature C may drive the WTA I)')
                e_to_wta.setdefault(tr[0], set()).add(s)
            else:                                     # feature If: only its paired C may drive it
                if not (sr and sr[1] == 'C' and sr[0] == tr[0] and sr[2] == tr[2]):
                    raise SpecError(f'feature If drive {s!r}->{t!r} must be its paired C[k] '
                                    f'(no ordinary E may drive a feature If)')
                c_to_if.setdefault(t, set()).add(s)
        elif kind == 'hard_reset_inhibition':
            if not (sr and tr):
                raise SpecError(f'hard_reset {s!r}->{t!r} must connect two tiled nodes')
            if sr[1] == 'I' and tr[1] == 'E':         # WTA reset onto its own bank
                if sr[0] != tr[0]:
                    raise SpecError(f'WTA I {s!r} resets {t!r} outside its own module')
                wta_to_e.setdefault(sr[0], set()).add(t)
            elif sr[1] == 'If' and tr[1] == 'S':      # feature reset onto its paired relay only
                if sr[0] != tr[0] or sr[2] != tr[2]:
                    raise SpecError(f'feature If {s!r} must reset only its paired feature relay, '
                                    f'not {t!r}')
                if_to_s.setdefault(s, set()).add(t)
            else:
                raise SpecError(f'unexpected hard_reset {s!r}->{t!r}: only WTA I->own E and '
                                f'feature If[k]->paired S[k] are allowed')
        elif sr is not None or tr is not None or s_is_rg:
            raise SpecError(f'edge kind {kind!r} ({s!r}->{t!r}) is not part of the '
                            f'feature-gated tiled family')

    # --- per-module completeness (a missing edge is rejected here) ------------------------
    for cid, c in columns.items():
        s = mods[cid]
        ebank = set(s['E'])
        if e_to_eor.get(cid, set()) != ebank:
            raise SpecError(f'module {cid!r}: every ordinary E must feed its Eor exactly once')
        if e_to_wta.get(cid, set()) != ebank:
            raise SpecError(f'module {cid!r}: every ordinary E must drive its WTA I')
        if wta_to_e.get(cid, set()) != ebank:
            raise SpecError(f'module {cid!r}: WTA I must hard-reset exactly its own E bank')
        if c['layer'] != input_layer:
            continue
        expect = {rid for rid, patch in rg_patch.items() if patch == (c['row'], c['col'])}
        for fk in range(n_feat):
            s_id, c_id, if_id = s['S'][fk], s['C'][fk], s['If'][fk]
            # RGC->S: exactly the one patch RGC whose patch-feature index is fk.
            want_rg = {rid for rid in expect if rg_feature[rid] == fk}
            if rgc_to_s.get(s_id, set()) != want_rg:
                raise SpecError(f'feature relay {s_id!r} must receive exactly its paired RGC')
            # S->E: all and only the module's ordinary E.
            if s_to_e.get(s_id, set()) != ebank:
                raise SpecError(f'feature relay {s_id!r} must feed all and only its module E bank')
            # S->C basal: exactly one, the paired S.
            if s_to_c_basal.get(c_id, set()) != {s_id}:
                raise SpecError(f'feature C {c_id!r} must have exactly its paired relay {s_id!r} '
                                f'as basal source')
            # E->C apical: exactly the module's ordinary E (all eight, local only).
            if e_to_c_apical.get(c_id, set()) != ebank:
                raise SpecError(f'feature C {c_id!r} must receive apical from all and only its '
                                f'{len(ebank)} local ordinary E')
            # C->If and If->S: strict paired reset chain.
            if c_to_if.get(if_id, set()) != {c_id}:
                raise SpecError(f'feature If {if_id!r} must be driven only by its paired C {c_id!r}')
            if if_to_s.get(if_id, set()) != {s_id}:
                raise SpecError(f'feature If {if_id!r} must reset only its paired relay {s_id!r}')

    # --- WTA vs feature-I disjointness (defensive; the edge rules already guarantee it) ----
    for cid in columns:
        wta_targets = wta_to_e.get(cid, set())
        feature_reset_targets = set()
        for fk, if_id in mods[cid]['If'].items():
            feature_reset_targets |= if_to_s.get(if_id, set())
        if wta_targets & feature_reset_targets:
            raise SpecError(f'module {cid!r}: WTA I and feature If reset targets must be disjoint')

    # --- child/parent link completeness + NO L2->L1 apical feedback -----------------------
    for cid, c in columns.items():
        for pid in c['parent_ids']:
            parent_E = set(mods[pid]['E'])
            if link_ff.get(cid, set()) != parent_E:
                raise SpecError(f'module {cid!r} Eor must feed every ordinary E of parent {pid!r}')


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
