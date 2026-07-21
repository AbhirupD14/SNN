"""The 'rg' topology: an explicit retinal-ganglion source layer ahead of `old`'s cortex.

Pins the four things that make `rg` a real experiment rather than decoration:

  * RG cells are exogenous and UNINHIBITABLE -- a held edge keeps emitting while its
    L1E is shunted, and no edge kind can even target an RG cell;
  * the two-hop causal chain RG(t) -> L1E(t+1) -> L2E(t+2) -> inhibition(t+3), with no
    spike ever traversing two feedforward hops in one boundary;
  * L1E learns its single RG afferent through the SAME numerical update as an
    equivalently configured L2E afferent -- there is no L1-only learning shortcut;
  * causal participation is per-target and per-boundary across both hops.
"""

import numpy as np
import pytest

from backend.network_spec import preset_spec, validate_spec, SpecError
from backend.simulation import (
    SimulationEngine, N_PIX, N_OUT, FF_INIT_MEAN, E_WEIGHT_CAP,
)
from snn.neurons import ExcitatoryNeuron, SourceNeuron


def fresh(**kw):
    e = SimulationEngine(seed=1, topology='rg', **kw)
    e.clear_input()
    return e


def by_id(d):
    return {n['id']: n for n in d['neurons']}


def fired(d):
    return {n['id'] for n in d['neurons'] if n['spiked']}


# ============================================================ structure
def test_rg_node_and_edge_counts():
    spec = preset_spec('rg', N_PIX, N_OUT)
    assert len(spec['nodes']) == 36
    assert len(spec['edges']) == 178
    validate_spec(spec, N_PIX)                     # the preset is a valid spec


def test_rg_population_ids_and_archetypes():
    e = SimulationEngine(seed=1, topology='rg')
    arch = {n['id']: n['archetype'] for n in e.topology()['neurons']}
    assert {f'RG{i}' for i in range(9)} <= set(arch)
    assert all(arch[f'RG{i}'] == 'rg_source' for i in range(9))
    assert all(arch[f'L1E{i}'] == 'e_encoder' for i in range(9))
    assert all(arch[f'L1I{i}'] == 'i_relay' for i in range(9))
    assert all(arch[f'L2E{j}'] == 'e_competitor' for j in range(8))
    assert arch['L2I'] == 'i_relay'
    assert len(arch) == 36


def test_rg_projection_shapes():
    syn = SimulationEngine(seed=1, topology='rg').topology()['synapses']
    kinds = {}
    for s in syn:
        kinds.setdefault(s['kind'], []).append((s['source'], s['target']))
    ff = kinds['feedforward']
    rg_ff = [(a, b) for a, b in ff if a.startswith('RG')]
    l1_ff = [(a, b) for a, b in ff if a.startswith('L1E')]
    # RG -> L1E is paired one-to-one: nine edges, RG_i -> L1E_i, exactly one out per RG.
    assert len(rg_ff) == 9
    assert set(rg_ff) == {(f'RG{i}', f'L1E{i}') for i in range(9)}
    # L1E -> L2E dense 9x8; L2E -> L1I dense 8x9; L1I -> L1E paired.
    assert set(l1_ff) == {(f'L1E{i}', f'L2E{j}') for i in range(9) for j in range(8)}
    assert len(l1_ff) == 72
    l1i_re = [(a, b) for a, b in kinds['relay_excitation'] if b.startswith('L1I')]
    assert set(l1i_re) == {(f'L2E{j}', f'L1I{i}') for i in range(9) for j in range(8)}
    assert len(l1i_re) == 72
    assert set(kinds['inhibition']) == (
        {(f'L1I{i}', f'L1E{i}') for i in range(9)} | {('L2I', f'L2E{j}') for j in range(8)})
    assert len([1 for a, b in kinds['relay_excitation'] if b == 'L2I']) == 8
    assert len(ff) == 81 and len(syn) == 178


def test_rg_has_no_predictive_machinery():
    e = SimulationEngine(seed=1, topology='rg')
    topo = e.topology()
    assert not [n for n in topo['neurons'] if n['archetype'] == 'predictor']
    assert not [s for s in topo['synapses'] if s['kind'] == 'predictive_inhibition']
    assert e.pi == []


def test_pi_and_old_specs_unchanged():
    # The third preset must not perturb the other two.
    for name, n_nodes, n_edges in (('pi', 26, 8 * 9 + 8 + 8 + 8 + 72),
                                   ('old', 27, 8 * 9 + 8 + 8 + 72 + 9)):
        spec = preset_spec(name, N_PIX, N_OUT)
        assert len(spec['nodes']) == n_nodes
        assert len(spec['edges']) == n_edges
        assert not [n for n in spec['nodes'] if n['archetype'] in ('rg_source', 'e_encoder')]
        # L1E in pi/old is still a fixed-afferent sensory source driven directly.
        assert all(n['archetype'] == 'e_sensory'
                   for n in spec['nodes'] if n['id'].startswith('L1E'))


# ============================================================ RG semantics
def test_active_pixel_spikes_only_its_own_rg():
    e = fresh()
    e.input_vec[4] = 1.0
    d = e.step()
    assert {n for n in fired(d) if n.startswith('RG')} == {'RG4'}


def test_inactive_pixel_never_spikes():
    e = fresh()
    e.input_vec[4] = 1.0
    for _ in range(30):
        d = e.step()
        assert 'RG0' not in fired(d)
    assert e.firing_freq('RG0') == 0.0


def test_rg_keeps_spiking_while_its_l1e_is_clamped():
    # The whole point of the source layer: cortical inhibition may silence L1E, but the
    # retina keeps delivering evidence for a held edge.
    e = fresh()
    e.input_vec[4] = 1.0
    rg_spikes = l1_spikes = 0
    for t in range(1, 41):
        # Slam a huge inhibitory conductance onto L1E4 every boundary.
        e.encoders[4].g_inh = 1e6
        d = e.step()
        f = fired(d)
        rg_spikes += ('RG4' in f)
        l1_spikes += ('L1E4' in f)
    assert rg_spikes == 40           # every boundary, undisturbed
    assert l1_spikes == 0            # L1E fully shunted the whole time
    assert e.encoders[4].acc_weights[0] == pytest.approx(  # and it never learned
        FF_INIT_MEAN * np.array([1.0]), abs=5.0)


def test_rg_respects_input_period():
    e = fresh(input_period=3)
    e.input_vec[4] = 1.0
    spikes = [1 if 'RG4' in fired(e.step()) else 0 for _ in range(9)]
    assert spikes == [0, 0, 1, 0, 0, 1, 0, 0, 1]      # only on t % 3 == 0


def test_rg_has_no_membrane_or_inhibitory_state():
    e = fresh()
    rg = e.neurons['RG0']
    assert isinstance(rg, SourceNeuron) and not isinstance(rg, ExcitatoryNeuron)
    assert not hasattr(rg, 'g_inh') and not hasattr(rg, 'acc_weights')
    assert rg.refractory_timer == 0
    assert rg.id not in e.exc                        # not on any integration path


def test_rg_is_unaffected_by_refractory_and_wta_params():
    e = fresh(refractory_steps=5)
    e.input_vec[4] = 1.0
    # A refractory period would gate any integrator; RG has none.
    assert all('RG4' in fired(e.step()) for _ in range(12))


def test_no_edge_kind_can_target_an_rg_cell():
    nodes = [{'id': 'RG0', 'archetype': 'rg_source', 'pixel': 0},
             {'id': 'EN0', 'archetype': 'e_encoder'},
             {'id': 'R', 'archetype': 'i_relay'},
             {'id': 'C0', 'archetype': 'e_competitor'},
             {'id': 'P0', 'archetype': 'predictor'}]
    for kind, src in (('inhibition', 'R'), ('predictive_inhibition', 'P0'),
                      ('feedforward', 'EN0'), ('relay_excitation', 'C0')):
        with pytest.raises(SpecError):
            validate_spec({'nodes': nodes,
                           'edges': [{'source': src, 'target': 'RG0', 'kind': kind}]}, N_PIX)


def test_rg_must_own_a_pixel_and_ownership_is_unique():
    with pytest.raises(SpecError):
        validate_spec({'nodes': [{'id': 'RG0', 'archetype': 'rg_source'}], 'edges': []}, N_PIX)
    clash = [{'id': 'RG0', 'archetype': 'rg_source', 'pixel': 0},
             {'id': 'RG1', 'archetype': 'rg_source', 'pixel': 0}]
    with pytest.raises(SpecError):
        validate_spec({'nodes': clash, 'edges': []}, N_PIX)


def test_encoder_may_not_own_a_pixel_but_may_carry_grid_metadata():
    # Input ownership belongs to the RG cell; L1E keeps only display/RF metadata, and
    # 'grid' is deliberately NOT unique-checked.
    with pytest.raises(SpecError):
        validate_spec({'nodes': [{'id': 'EN0', 'archetype': 'e_encoder', 'pixel': 0}],
                       'edges': []}, N_PIX)
    ok = validate_spec({'nodes': [{'id': 'EN0', 'archetype': 'e_encoder', 'grid': 0},
                                  {'id': 'EN1', 'archetype': 'e_encoder', 'grid': 0}],
                        'edges': []}, N_PIX)
    assert [n['grid'] for n in ok['nodes']] == [0, 0]


def test_rg_spikes_serialize_in_state_history_and_frequency():
    e = fresh()
    e.input_vec[4] = 1.0
    d = e.step()
    rec = by_id(d)['RG4']
    assert rec['spiked'] is True and rec['activation'] == 1.0
    assert 'g_inh' not in rec and 'trace' not in rec      # RG owns no conductance state
    for _ in range(9):
        e.step()
    assert e.firing_freq('RG4') == 1.0                    # spiked on every boundary
    assert e.firing_freq('RG0') == 0.0
    # RG's outgoing edge flashes like any other emitted event.
    assert 'rg4->l1e4' in e.emitted
    assert e.meta['RG4']['layer'] == 'RG' and e.meta['RG4']['type'] == 'S'


# ============================================================ delays / causal order
def test_two_hop_causal_chain_timing():
    e = fresh()
    e.input_vec[4] = 1.0
    e.encoders[4].acc_weights[0] = 1.2 * e.params['e_threshold']   # fire L1E on arrival
    e.l2e[3].acc_weights[4] = 1.2 * e.params['e_threshold']        # and L2E on the next

    d1 = by_id(e.step())                       # t=1: RG emits, nothing else
    assert d1['RG4']['spiked'] and not d1['L1E4']['spiked'] and not d1['L2E3']['spiked']

    d2 = by_id(e.step())                       # t=2: RG charge arrives -> L1E fires
    assert d2['L1E4']['spiked'] and not d2['L2E3']['spiked']       # NOT the same boundary

    d3 = e.step()                              # t=3: L1E charge arrives -> L2 winner
    b3 = by_id(d3)
    assert b3['L2E3']['spiked'] and e.winner == 'L2E3'
    assert all(b3[f'L1I{i}']['spiked'] for i in range(9))          # relays fire same boundary
    assert not d3['inhibitory_pulses']                             # conductance NOT yet delivered
    assert all(b3[f'L1E{i}']['g_inh'] == 0.0 for i in range(9))

    d4 = e.step()                              # t=4: inhibitory conductance lands
    pulses = {(p['source'], p['target']) for p in d4['inhibitory_pulses']}
    assert pulses == ({(f'L1I{i}', f'L1E{i}') for i in range(9)}
                      | {('L2I', f'L2E{j}') for j in range(8)})
    assert all(by_id(d4)[f'L1E{i}']['g_inh'] > 0 for i in range(9))


def test_no_spike_crosses_two_feedforward_hops_in_one_boundary():
    e = fresh()
    e.input_vec[4] = 1.0
    # Both hops are far supra-threshold; even so each hop costs exactly one boundary.
    e.encoders[4].acc_weights[0] = 50.0 * e.params['e_threshold']
    for c in e.l2e:
        c.acc_weights[:] = 0.0
    e.l2e[3].acc_weights[4] = 50.0 * e.params['e_threshold']
    f1, f2 = fired(e.step()), fired(e.step())
    assert 'RG4' in f1 and 'L1E4' not in f1        # RG cannot fire L1E at t
    assert 'L1E4' in f2 and 'L2E3' not in f2       # L1E cannot charge L2E at t+1
    assert 'L2E3' in fired(e.step())


def test_inhibition_and_rg_excitation_integrate_jointly_before_threshold():
    # Same boundary: RG-derived charge arrives AND an inhibitory conductance arrives.
    # The shunt must be able to veto a spike that the charge alone would have caused.
    e = fresh(leak_rate=0.0)
    e.input_vec[4] = 1.0
    e.encoders[4].acc_weights[0] = 1.05 * e.params['e_threshold']
    assert 'L1E4' not in fired(e.step())           # t=1: RG only
    e.encoders[4].g_inh = 0.0
    e._inh_next.append(dict(target='L1E4', dg=500.0, source='L1I4',
                            kind='inhibition', weight=500.0))
    d = e.step()                                   # t=2: charge + shunt together
    assert 'L1E4' not in fired(d)                  # jointly integrated -> no spike
    assert by_id(d)['L1E4']['g_inh'] > 0


def test_queued_rg_events_are_not_cancelled_by_a_cortical_winner():
    e = fresh()
    e.input_vec[4] = 1.0
    e.encoders[4].acc_weights[0] = 1.2 * e.params['e_threshold']
    e.l2e[3].acc_weights[4] = 1.2 * e.params['e_threshold']
    seen = [('RG4' in fired(e.step())) for _ in range(8)]
    assert all(seen)                               # RG never skips a boundary


# ============================================ L1 accumulation and learning
def _analytic_first_spike(w, leak, thr):
    """Boundaries of a single repeated charge w needed to reach thr under the live
    conductance equation: V_n = (w/g_L) * (1 - (1-leak)^n)."""
    import math
    g_l = -math.log(1.0 - leak)
    v_inf = w / g_l
    n = 1
    v = 0.0
    while v < thr and n < 10000:
        v = v_inf + (v - v_inf) * math.exp(-g_l)
        if v >= thr:
            return n
        n += 1
    return n


def test_one_rg_event_is_subthreshold_and_first_spike_matches_analysis():
    e = fresh(enc_init_jitter=False)               # exact FF_INIT_MEAN on every afferent
    e.input_vec[4] = 1.0
    thr, leak = e.params['e_threshold'], e.params['leak_rate']
    w0 = float(e.encoders[4].acc_weights[0])
    assert w0 == pytest.approx(FF_INIT_MEAN)
    assert w0 < thr                                # one event cannot fire an L1E

    # First L1E spike, measured. RG fires at t, charge lands at t+1, so the Nth RG
    # event is integrated on boundary N+1.
    first = None
    for t in range(1, 60):
        if 'L1E4' in fired(e.step()):
            first = t
            break
    assert first is not None
    n_events = _analytic_first_spike(w0, leak, thr)
    assert n_events == 23                          # documented audit target
    assert first == n_events + 1                   # +1 for the delay-1 hop


def test_encoder_uses_the_identical_update_as_an_equivalent_l2e_afferent():
    # Drive one L1E to a spike, then reproduce its weight change with a bare
    # ExcitatoryNeuron configured exactly like an L2E competitor.
    e = fresh()
    e.input_vec[4] = 1.0
    enc = e.encoders[4]
    w_before = float(enc.acc_weights[0])
    dfac = float(enc.acc_distance_factor[0])
    for _ in range(60):
        if 'L1E4' in fired(e.step()):
            break
    w_after = float(enc.acc_weights[0])
    assert w_after != w_before                     # it learned on its own spike

    ref = ExcitatoryNeuron(
        'REF', 'competitor', acc_weights=np.array([w_before]),
        acc_distance_factor=np.array([dfac]),
        threshold=float(e.params['e_threshold']), w_max=float(e.params['e_weight_cap']),
        leak_rate=float(e.params['leak_rate']), eta=float(e.params['eta']), learn=True)
    ref.update_acc_weights(np.array([True]))       # its one RG afferent participated
    assert float(ref.acc_weights[0]) == w_after    # bit-identical, not merely close


def test_only_the_firing_encoders_weight_changes():
    e = fresh()
    e.input_vec[4] = 1.0
    before = [float(c.acc_weights[0]) for c in e.encoders]
    for _ in range(60):
        if 'L1E4' in fired(e.step()):
            break
    after = [float(c.acc_weights[0]) for c in e.encoders]
    assert after[4] != before[4]
    assert [a for i, a in enumerate(after) if i != 4] == \
           [b for i, b in enumerate(before) if i != 4]      # nobody else moved


def test_inactive_pixel_neither_charges_nor_learns():
    e = fresh()
    e.input_vec[4] = 1.0
    for _ in range(60):
        e.step()
    assert e.encoders[0].V == pytest.approx(0.0)             # never charged
    assert float(e.encoders[0].acc_weights[0]) == pytest.approx(FF_INIT_MEAN, abs=5.0)


def test_repeated_training_grows_weight_and_shortens_l1_isis():
    e = fresh(enc_init_jitter=False)
    e.input_vec[4] = 1.0
    spikes, ws = [], []
    for t in range(1, 1200):
        if 'L1E4' in fired(e.step()):
            spikes.append(t)
            ws.append(float(e.encoders[4].acc_weights[0]))
    assert len(spikes) > 20
    isis = np.diff(spikes)
    early, late = isis[:3].mean(), isis[-3:].mean()
    assert late < early                                      # cadence accelerates
    assert ws[-1] > ws[0]                                    # via a growing weight
    assert 0.0 <= ws[-1] <= e.params['e_weight_cap']
    # Mature cadence at/near the cap: ~1 L1 spike per 3 active RG events.
    assert late == pytest.approx(3.0, abs=1.0)


def test_encoder_weight_stays_within_bounds_under_long_training():
    e = fresh()
    e.set_pattern('row 1')
    for _ in range(3000):
        e.step()
    cap = e.params['e_weight_cap']
    for c in e.encoders:
        w = float(c.acc_weights[0])
        assert 0.0 <= w <= cap


def test_frozen_encoder_never_learns():
    e = fresh(enc_plasticity_enabled=False)
    e.input_vec[4] = 1.0
    w0 = float(e.encoders[4].acc_weights[0])
    for _ in range(400):
        e.step()
    assert float(e.encoders[4].acc_weights[0]) == w0


def test_equal_init_gives_identical_weights_but_untouched_l2():
    a = SimulationEngine(seed=3, topology='rg')
    b = SimulationEngine(seed=3, topology='rg', enc_init_jitter=False)
    assert len({round(float(c.acc_weights[0]), 9) for c in b.encoders}) == 1
    assert len({round(float(c.acc_weights[0]), 9) for c in a.encoders}) == 9
    # The control must isolate RG->L1E jitter: L2E init is bit-identical.
    for x, y in zip(a.l2e, b.l2e):
        assert np.array_equal(x.acc_weights, y.acc_weights)


def test_rg_l2e_init_matches_old_at_the_same_seed():
    # So the timing experiment compares topologies, not reshuffled competitor seeds.
    old = SimulationEngine(seed=7, topology='old')
    rg = SimulationEngine(seed=7, topology='rg')
    for x, y in zip(old.l2e, rg.l2e):
        assert np.array_equal(x.acc_weights, y.acc_weights)
        assert np.array_equal(x.acc_distance_factor, y.acc_distance_factor)


# ==================================== noncompetitive L1 vs competitive L2
def test_multiple_l1e_fire_in_one_boundary_but_only_one_l2e():
    e = fresh()
    e.set_pattern('row 1')                        # pixels 3,4,5
    thr = e.params['e_threshold']
    for i in (3, 4, 5):
        e.encoders[i].acc_weights[0] = 1.2 * thr
    for c in e.l2e:                               # make several L2E crossers
        c.acc_weights[:] = 0.6 * thr
    e.step()
    d2 = e.step()
    assert {n for n in fired(d2) if n.startswith('L1E')} == {'L1E3', 'L1E4', 'L1E5'}
    d3 = e.step()
    assert len({n for n in fired(d3) if n.startswith('L2E')}) == 1     # WTA still one


def test_encoders_are_not_in_the_wta_pool():
    e = fresh()
    assert {c.id for c in e.competitors} == {f'L2E{j}' for j in range(8)}
    assert {c.id for c in e.encoders} == {f'L1E{i}' for i in range(9)}
    assert not set(e.encoders) & set(e.competitors)
    assert all(c.role == 'encoder' for c in e.encoders)
    assert all(e.meta[c.id]['role'] == 'encoder' for c in e.encoders)


def test_only_one_l2e_learns_per_boundary():
    e = fresh()
    e.set_pattern('row 1')
    thr = e.params['e_threshold']
    for i in (3, 4, 5):
        e.encoders[i].acc_weights[0] = 1.2 * thr
    for c in e.l2e:
        c.acc_weights[:] = 0.6 * thr
    for _ in range(3):
        d = e.step()
    changed = {c['id'] for c in d['changed_synapses']}
    targets = {eid.split('->')[1] for eid in changed if eid.startswith('ff')}
    assert len(targets) <= 1                      # at most one competitor's row moved


# ============================== target-specific, boundary-specific participation
def test_encoder_update_sees_only_its_own_rg_arrival():
    e = fresh()
    e.set_pattern('row 1')                        # RG3, RG4, RG5 active
    thr = e.params['e_threshold']
    for i in (3, 4, 5):
        e.encoders[i].acc_weights[0] = 1.2 * thr
    e.step()
    e.step()                                      # all three L1E fire here
    for i in (3, 4, 5):
        # Each encoder's participation is its OWN afferent, not the union of the volley.
        assert e._participation(e.encoders[i]).tolist() == [True]
    for i in (0, 1, 2):
        assert e._participation(e.encoders[i]).tolist() == [False]


def test_l2e_update_sees_only_the_l1e_spikes_delivered_to_it():
    e = fresh()
    e.set_pattern('row 1')
    thr = e.params['e_threshold']
    for i in (3, 4, 5):
        e.encoders[i].acc_weights[0] = 1.2 * thr
    e.step(); e.step()                            # L1E 3,4,5 fired at t=2
    e.step()                                      # their volley arrives at t=3
    for c in e.l2e:
        part = e._participation(c)
        assert [i for i, v in enumerate(part) if v] == [3, 4, 5]


def test_participation_cannot_leak_across_boundaries():
    e = fresh()
    e.input_vec[4] = 1.0
    e.encoders[4].acc_weights[0] = 1.2 * e.params['e_threshold']
    e.step()                                      # t=1: RG4 emits
    e.clear_input()                               # pixel released before the next boundary
    e.step()                                      # t=2: the t=1 volley arrives
    assert e._participation(e.encoders[4]).tolist() == [True]
    e.step()                                      # t=3: nothing was emitted at t=2
    assert e._participation(e.encoders[4]).tolist() == [False]   # no carry-over


def test_two_targets_with_different_source_sets_learn_from_their_own_sets():
    # L2E0 keeps all nine afferents; L2E1 is rewired to see only pixel 3. Both fire from
    # the same L1 volley, and each must read only ITS delivered set.
    e = fresh()
    spec = e.current_spec()
    spec['edges'] = [s for s in spec['edges']
                     if not (s['target'] == 'L2E1' and s['kind'] == 'feedforward'
                             and s['source'] != 'L1E3')]
    e.apply_topology(spec)
    e.set_pattern('row 1')
    thr = e.params['e_threshold']
    for i in (3, 4, 5):
        e.encoders[i].acc_weights[0] = 1.2 * thr
    e.step(); e.step(); e.step()
    c0 = next(c for c in e.l2e if c.id == 'L2E0')
    c1 = next(c for c in e.l2e if c.id == 'L2E1')
    assert c1.ff_src == ['L1E3']
    assert e._participation(c0).tolist() == [i in (3, 4, 5) for i in range(9)]
    assert e._participation(c1).tolist() == [True]      # only its own single afferent


# ============================================================ regression
def test_toggling_topologies_rebuilds_cleanly():
    e = SimulationEngine(seed=1, topology='pi')
    e.set_pattern('row 1')
    for _ in range(30):
        e.step()
    for topo, n_nodes in (('rg', 36), ('old', 27), ('rg', 36), ('pi', 26)):
        e.apply_config({'topology': topo})
        assert len(e.neurons) == n_nodes
        # No stale populations, queues, conductances or source mappings survive.
        assert (len(e.sources) == 9) == (topo == 'rg')
        assert (len(e.encoders) == 9) == (topo == 'rg')
        assert (len(e.pi) == 8) == (topo == 'pi')
        assert e._exc_next == {} and e._inh_next == []
        assert e._ff_deliv_next == {} and e._ff_deliv_now == {}
        assert e.timestep == 0 and e.winner is None
        assert all(n.g_inh == 0.0 for n in e.exc.values())
        assert {c.id for c, _ in e._input_sinks} == (
            {f'RG{i}' for i in range(9)} if topo == 'rg' else {f'L1E{i}' for i in range(9)})
        assert set(e._ff_weight_ref) and all(eid in {s['id'] for s in e.synapses}
                                             for eid in e._ff_weight_ref)
        for _ in range(20):
            e.step()


def test_rg_spec_round_trips_through_the_editor_path():
    e = SimulationEngine(seed=1, topology='rg')
    spec = e.current_spec()
    assert spec['name'] == 'rg'
    rg0 = next(n for n in spec['nodes'] if n['id'] == 'RG0')
    l1e0 = next(n for n in spec['nodes'] if n['id'] == 'L1E0')
    assert rg0['pixel'] == 0 and rg0['layer'] == 'RG' and len(rg0['pos']) == 3
    assert 'pixel' not in l1e0                          # ownership lives on RG
    # Both tags must survive: dropping 'grid' would unmap a saved rg graph from the RF view.
    assert l1e0['grid'] == 0
    e.apply_topology(spec)                              # re-applying rebuilds and runs
    assert len(e.neurons) == 36
    assert e.meta['L1E0']['pixel'] == 0                 # RF placement still resolves
    assert e.current_spec()['nodes'][9]['grid'] == 0    # and survives a second round-trip
    assert [round(x, 4) for x in e.meta['RG0']['pos']] == [round(x, 4) for x in rg0['pos']]
    e.set_pattern('row 1')
    for _ in range(50):
        e.step()


def test_rg_weights_are_settable_and_serialized_by_edge_id():
    e = SimulationEngine(seed=1, topology='rg')
    assert e.set_synapse_weight('rg4->l1e4', 250.0) == 250.0
    assert float(e.encoders[4].acc_weights[0]) == 250.0
    syn = {s['id']: s for s in e.topology()['synapses']}
    assert syn['rg4->l1e4']['weight'] == 250.0
    assert syn['rg4->l1e4']['kind'] == 'feedforward'
    # Clipped to the shared cap, like any other plastic excitatory weight.
    assert e.set_synapse_weight('rg4->l1e4', 10 * E_WEIGHT_CAP) == e.params['e_weight_cap']
