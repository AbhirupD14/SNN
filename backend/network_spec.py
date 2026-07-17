"""Declarative network topology: a **NetworkSpec** (typed nodes + typed edges) the
engine builds and runs generically, the two built-in presets, and validation.

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
  * ``e_sensory``   -- every threshold crosser fires (sensory sources).
  * ``e_competitor``-- deterministic winner-take-all: one winner per boundary fires
                       and learns its feedforward weights.
Everything else -- who inhibits whom, who relays to whom, predictive inhibition --
is expressed by edges and is fully editable.
"""

from __future__ import annotations

# --- Node archetypes ---------------------------------------------------------
# cls: 'E' excitatory (conductance LIF) | 'I' inhibitory relay.
# thr_frac: firing threshold as a fraction of the excitatory threshold theta.
ARCHETYPES = {
    'e_sensory':    dict(cls='E', role='source',     thr_frac=1.0,
                         desc='Sensory excitatory source: one fixed input afferent; '
                              'fires on every threshold crossing (no WTA).'),
    'e_competitor': dict(cls='E', role='competitor', thr_frac=1.0,
                         desc='Competitor excitatory unit: plastic feedforward weights, '
                              'winner-take-all (one winner per boundary fires + learns).'),
    'i_relay':      dict(cls='I', role='relay',      thr_frac=1.0 / 3.0,
                         desc='Inhibitory relay: fires the same boundary it receives any '
                              'excitatory relay event; emits a persistent conductance pulse.'),
    'predictor':    dict(cls='I', role='predictor',  thr_frac=1.0 / 3.0,
                         desc='Predictive interneuron: relays its driver and owns locally '
                              'plastic inhibitory output weights onto its targets.'),
}

# --- Edge kinds and which archetypes they may connect -------------------------
# (source archetype class -> target archetype class) plus a human description.
EDGE_KINDS = {
    'feedforward':          dict(src='E', tgt='e_competitor', plastic=True, sign=+1,
                                 desc='Plastic excitatory feedforward onto a competitor '
                                      '(accumulating signed-spike rule).'),
    'relay_excitation':     dict(src='E', tgt='I', plastic=False, sign=+1,
                                 desc='Structural +1 event driving an inhibitory relay / '
                                      'predictor (no learned magnitude).'),
    'inhibition':           dict(src='i_relay', tgt='E', plastic=False, sign=-1,
                                 desc='Persistent inhibitory conductance pulse from a relay '
                                      'onto an excitatory target (fixed scale).'),
    'predictive_inhibition': dict(src='predictor', tgt='E', plastic=True, sign=-1,
                                 desc='Locally plastic predictive inhibitory conductance '
                                      'from a predictor onto an excitatory target.'),
}


def _e_sensory(i, pixel):
    return dict(id=f'L1E{i}', archetype='e_sensory', layer='L1', pixel=pixel,
                label=f'L1E_s{i}')


def preset_spec(name: str, n_pix: int, n_out: int) -> dict:
    """Return the built-in NetworkSpec for 'pi' or 'old'. Positions are omitted so
    the engine fills them from the seeded functional layout (bit-exact presets)."""
    if name not in ('pi', 'old'):
        raise ValueError(f'unknown preset {name!r}')
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
        pixel = n.get('pixel')
        if pixel is not None:
            if arch != 'e_sensory':
                raise SpecError(f'node {nid!r} sets pixel but is not e_sensory')
            if not isinstance(pixel, int) or not (0 <= pixel < n_pix):
                raise SpecError(f'node {nid!r} pixel must be an int in [0,{n_pix})')
            if pixel in pixels_used:
                raise SpecError(f'pixel {pixel} mapped by both {pixels_used[pixel]!r} '
                                f'and {nid!r}')
            pixels_used[pixel] = nid
        node = dict(id=nid, archetype=arch,
                    layer=n.get('layer') or ('L1' if arch == 'e_sensory' else 'L2'),
                    label=n.get('label') or nid)
        if pixel is not None:
            node['pixel'] = pixel
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
                            f'{spec_kind["src"]}')
        if not _arch_matches(tgt_arch, spec_kind['tgt']):
            raise SpecError(f'edge {kind}: target {tgt!r} ({tgt_arch}) is not a valid '
                            f'{spec_kind["tgt"]}')
        directed = bool(e.get('directed', True))
        if not directed:
            # A bidirectional edge delivers both ways, so the REVERSE direction must
            # also satisfy the kind's archetype rule (e.g. competitor<->competitor
            # feedforward). Most kinds are inherently one-way, so this rejects them.
            if not (_arch_matches(tgt_arch, spec_kind['src'])
                    and _arch_matches(src_arch, spec_kind['tgt'])):
                raise SpecError(
                    f'edge {kind} between {src!r} and {tgt!r} cannot be bidirectional: '
                    f'the reverse direction is not a valid {kind} (source must be '
                    f'{spec_kind["src"]}, target {spec_kind["tgt"]}). Use two directed edges.')
        eid = e.get('id') or f'{kind}:{src}->{tgt}'
        if eid in eids:
            raise SpecError(f'duplicate edge id {eid!r}')
        eids.add(eid)
        ne = dict(id=eid, source=src, target=tgt, kind=kind, directed=directed)
        if spec_kind['sign'] is not None:
            ne['sign'] = spec_kind['sign']
        norm_edges.append(ne)

    return dict(name=spec.get('name') or 'custom', nodes=norm_nodes, edges=norm_edges)


def _arch_matches(arch: str, requirement: str) -> bool:
    """A requirement is either an archetype name, or a class letter 'E'/'I'."""
    if requirement in ARCHETYPES:
        return arch == requirement
    return ARCHETYPES[arch]['cls'] == requirement
