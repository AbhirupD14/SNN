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

# Inhibitory firing threshold as a fraction of the excitatory threshold theta. This is a
# FIXED gain (a relay fires on ~a third of theta of coincident drive), not a count-derived
# quantity: it does not scale with n_pix or n_out.
I_THRESHOLD_FRAC = 1.0 / 3.0

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

PRESETS = ('pi', 'old', 'rg', 'rg_residual', 'rg_coincidence')


def _e_sensory(i, pixel):
    return dict(id=f'L1E{i}', archetype='e_sensory', layer='L1', pixel=pixel,
                label=f'L1E_s{i}')


def preset_spec(name: str, n_pix: int, n_out: int) -> dict:
    """Return a built-in NetworkSpec. Positions are omitted
    so the engine fills them from the seeded functional layout (bit-exact presets)."""
    if name not in PRESETS:
        raise ValueError(f'unknown preset {name!r}')
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
        if apical_in.get(n['id'], 0) < 1:
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

    return dict(name=spec.get('name') or 'custom', nodes=norm_nodes, edges=norm_edges)


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
