"""Structural and causal tests for the 52-cell RG residual/error preset.

These tests pin the intended mechanism without claiming that its long-horizon
classification dynamics are already successful: L1 remains complete, prediction
targets ErrorE, residual events broadcast, and only a locally traced SwitchI can
schedule paired incumbent inhibition.
"""

from collections import Counter

import pytest

from backend.network_spec import preset_spec, validate_spec
from backend.simulation import SimulationEngine, N_PIX, N_OUT
from snn.neurons import SwitchInterneuron


def fresh(**cfg):
    return SimulationEngine(seed=1, topology='rg_residual', **cfg)


def by_id(d):
    return {n['id']: n for n in d['neurons']}


def fired(d, prefix):
    return [n['id'] for n in d['neurons'] if n['spiked'] and n['id'].startswith(prefix)]


def test_exact_population_and_projection_inventory():
    spec = validate_spec(preset_spec('rg_residual', N_PIX, N_OUT), N_PIX)
    assert len(spec['nodes']) == 52
    assert len(spec['edges']) == 274
    assert Counter(n['archetype'] for n in spec['nodes']) == {
        'rg_source': 9, 'e_encoder': 9, 'e_residual': 9,
        'e_competitor': 8, 'predictor': 8, 'switch': 8, 'i_relay': 1,
    }
    assert Counter(e['kind'] for e in spec['edges']) == {
        'feedforward': 81,             # 9 RG->L1E + 72 L1E->L2E
        'fixed_excitation': 9,
        'relay_excitation': 88,        # 8 L2E->PI + 72 ErrorE->SwitchI + 8 L2E->L2I
        'predictive_inhibition': 72,
        'trace_excitation': 8,
        'inhibition': 16,              # 8 SwitchI->L2E + 8 L2I->L2E
    }


def test_prediction_never_targets_rg_or_main_l1_evidence():
    spec = preset_spec('rg_residual', N_PIX, N_OUT)
    pred = [e for e in spec['edges'] if e['kind'] == 'predictive_inhibition']
    assert len(pred) == 72
    assert all(e['source'].startswith('PI') and e['target'].startswith('ErrorE')
               for e in pred)
    inhibited = [e['target'] for e in spec['edges'] if e['kind'] in
                 ('inhibition', 'predictive_inhibition')]
    assert not any(t.startswith(('RG', 'L1E')) for t in inhibited)


def test_every_switch_is_paired_for_trace_and_output_but_receives_all_residuals():
    spec = preset_spec('rg_residual', N_PIX, N_OUT)
    edges = spec['edges']
    for j in range(N_OUT):
        assert any(e['source'] == f'L2E{j}' and e['target'] == f'SwitchI{j}'
                   and e['kind'] == 'trace_excitation' for e in edges)
        assert any(e['source'] == f'SwitchI{j}' and e['target'] == f'L2E{j}'
                   and e['kind'] == 'inhibition' for e in edges)
        residual_inputs = {e['source'] for e in edges
                           if e['target'] == f'SwitchI{j}'
                           and e['kind'] == 'relay_excitation'}
        assert residual_inputs == {f'ErrorE{i}' for i in range(N_PIX)}


def test_switch_unit_rejects_each_branch_alone_and_consumes_coincidence():
    sw = SwitchInterneuron('SwitchI3', trace_decay=0.8, trace_threshold=0.5)
    sw.begin_boundary()
    sw.receive_residual()
    assert 0.0 < sw.potential < sw.threshold
    assert sw.resolve_residual() is False               # residual alone
    sw.prime()
    sw.begin_boundary()
    assert sw.x == pytest.approx(0.8)
    assert 0.0 < sw.potential < sw.threshold
    assert sw.resolve_residual() is False               # trace alone
    sw.receive_residual()
    assert sw.resolve_residual() is True                # both
    assert sw.x == 0.0                                  # one-shot consumption


def test_switch_branch_charges_are_numeric_visible_and_individually_bounded():
    sw = SwitchInterneuron('SwitchI0')
    for _ in range(20):
        sw.receive_residual()
    assert sw.residual_events == 20
    assert sw.residual_charge == pytest.approx(0.9 * sw.threshold)
    assert sw.trace_charge == 0.0
    assert sw.potential < sw.threshold                  # repeated residual alone is safe

    sw.begin_boundary()
    sw.prime()
    sw.begin_boundary()                                 # make current trace electrically active
    assert sw.residual_charge == 0.0
    assert 0.0 < sw.trace_charge <= 0.9 * sw.threshold
    assert sw.potential < sw.threshold                  # trace alone is safe


def test_default_winner_trace_spans_observed_residual_latency():
    sw = SwitchInterneuron('SwitchI0')
    sw.prime()
    for _ in range(10):
        sw.begin_boundary()
    assert sw.x > sw.trace_threshold                    # 0.97^10 ≈ 0.737
    sw.receive_residual()
    assert sw.resolve_residual() is True


def test_error_event_charges_every_switch_even_when_none_is_eligible():
    e = fresh(leak_rate=0.0)
    e.clear_input()
    e.stimulate('ErrorE1', 1.2)
    d = e.step()
    assert not fired(d, 'SwitchI')
    records = [by_id(d)[f'SwitchI{j}'] for j in range(N_OUT)]
    assert all(r['residual_received'] and r['residual_events'] == 1 for r in records)
    assert all(0.0 < r['residual_charge'] < e.params['e_threshold'] / 3.0
               for r in records)
    assert all(0.0 < r['potential'] < e.params['e_threshold'] / 3.0
               for r in records)


def test_three_hop_evidence_timing_and_l1_remains_uninhibited():
    e = fresh(leak_rate=0.0, enc_init_jitter=False)
    e.clear_input()
    e.input_vec[4] = 1.0
    e.encoders[4].acc_weights[0] = 1.2 * e.params['e_threshold']
    for c in e.l2e:
        c.acc_weights[:] = 0.0
    e.l2e[3].acc_weights[4] = 1.2 * e.params['e_threshold']

    d1 = e.step()                                      # RG emits
    assert fired(d1, 'RG') == ['RG4']
    assert not fired(d1, 'L1E') and not fired(d1, 'ErrorE')
    d2 = e.step()                                      # L1E emits
    assert fired(d2, 'L1E') == ['L1E4']
    assert not fired(d2, 'ErrorE') and not fired(d2, 'L2E')
    d3 = e.step()                                      # ErrorE + L2E see same L1 event
    assert fired(d3, 'ErrorE') == ['ErrorE4']
    assert fired(d3, 'L2E') == ['L2E3']
    assert by_id(d3)['PI3']['spiked'] and by_id(d3)['L2I']['spiked']
    assert all(by_id(d3)[f'L1E{i}']['g_inh'] == 0.0 for i in range(N_PIX))
    assert e.pi[3].w[4] > 0.0                          # PI learned local ErrorE4 trace
    assert sum(e.pi[3].w) == pytest.approx(e.pi[3].w[4])


def test_predictive_conductance_lands_only_on_error_sheet():
    e = fresh(leak_rate=0.0)
    e.clear_input()
    e.pi[3].w[4] = 1.0
    for c in e.l2e:
        c.V = 0.0
    e.l2e[3].V = 1.2 * e.params['e_threshold']
    d1 = e.step()
    assert d1['winner'] == 'L2E3'
    assert not d1['inhibitory_pulses']
    d2 = e.step()
    pred = [p for p in d2['inhibitory_pulses'] if p['kind'] == 'predictive']
    assert {p['target'] for p in pred} == {'ErrorE4'}
    assert by_id(d2)['ErrorE4']['g_inh'] > 0.0
    assert all(by_id(d2)[f'L1E{i}']['g_inh'] == 0.0 for i in range(N_PIX))


def test_residual_broadcast_is_global_but_only_locally_traced_switch_fires():
    e = fresh(leak_rate=0.0)
    e.clear_input()
    e.switches[3].x = 1.0
    e.stimulate('ErrorE1', 1.2)
    d = e.step()
    assert fired(d, 'ErrorE') == ['ErrorE1']
    assert fired(d, 'SwitchI') == ['SwitchI3']
    assert {x for x in d['emitted'] if x.startswith('error1->switch')} == {
        f'error1->switch{j}' for j in range(N_OUT)}
    assert e.switches[3].x == 0.0
    d2 = e.step()
    pulses = [p for p in d2['inhibitory_pulses'] if p['kind'] == 'switch']
    assert [(p['source'], p['target']) for p in pulses] == [('SwitchI3', 'L2E3')]


def test_new_winner_spike_cannot_supply_same_boundary_trace_for_residual():
    e = fresh(leak_rate=0.0)
    e.clear_input()
    for c in e.l2e:
        c.V = 0.0
    e.l2e[3].V = 1.2 * e.params['e_threshold']
    e.stimulate('ErrorE1', 1.2)
    d = e.step()
    assert d['winner'] == 'L2E3' and fired(d, 'ErrorE') == ['ErrorE1']
    assert not fired(d, 'SwitchI')                    # current winner is not "recent"
    assert e.switches[3].x == 1.0                     # available only after resolution


def test_switch_conductance_ablation_preserves_visible_coincidence_without_output():
    e = fresh(leak_rate=0.0, switch_conductance_enabled=False)
    e.clear_input()
    e.switches[2].x = 1.0
    e.stimulate('ErrorE7', 1.2)
    d = e.step()
    assert fired(d, 'SwitchI') == ['SwitchI2']
    assert not e._inh_next
    assert not [p for p in e.step()['inhibitory_pulses'] if p['kind'] == 'switch']


def test_switch_state_serializes_and_reset_clears_it():
    e = fresh()
    e.switches[5].x = 0.75
    rec = by_id(e.dynamic_state())['SwitchI5']
    assert rec['winner_trace'] == 0.75 and rec['residual_received'] is False
    assert {'residual_events', 'residual_charge', 'trace_charge', 'v_pre_reset'} <= set(rec)
    e.reset()
    assert e.switches[5].x == 0.0


def _strong_natural_row1_engine():
    """A deterministic rg_residual engine whose full RG->L1E->L2E->ErrorE chain runs
    every boundary, wired ONLY through public feedforward/source weights (never by
    poking switch state). L2E3 is the sole winner for the 'row 1' pattern. This is the
    real dispatch path -- what the dashboard runs -- not a hand-set switch unit."""
    e = fresh(leak_rate=0.0, enc_init_jitter=False, e_weight_cap=2000.0)
    e.clear_input()
    e.set_pattern('row 1')                                  # pixels 3, 4, 5
    thr = e.params['e_threshold']
    for i in (3, 4, 5):
        e.set_synapse_weight(f'rg{i}->l1e{i}', 1.2 * thr)   # L1E fires on every RG event
    for j in range(N_OUT):
        for i in range(N_PIX):
            e.set_synapse_weight(f'ff{i}->{j}', 0.0)
    for i in (3, 4, 5):
        e.set_synapse_weight(f'ff{i}->3', 0.5 * thr)        # L2E3 is the only competitor
    return e


def test_natural_pathway_fires_a_switch_end_to_end_with_paired_inhibition():
    # The switch UNIT works in isolation (tests above); this proves the full NATURAL
    # dispatch supplies both operands and fires a switch. Nothing writes switch state
    # directly -- the trace comes from a real L2E spike, the residual from a real ErrorE
    # spike, both scheduled through the ordinary engine.
    e = _strong_natural_row1_engine()
    thr = e.params['e_threshold']
    i_thr = thr / 3.0
    states = {}
    switch_fires, switch_pulses = [], []
    for t in range(1, 26):
        d = e.step()
        states[t] = d
        b = by_id(d)
        sw = fired(d, 'SwitchI')
        if sw:
            switch_fires.append((t, sw))
        pulses = [(p['source'], p['target']) for p in d['inhibitory_pulses']
                  if p['kind'] == 'switch']
        if pulses:
            switch_pulses.append((t, pulses))
        # STRICT AND, both directions: any switch that did not fire this boundary must be
        # individually subthreshold -- neither the residual branch nor the trace branch
        # alone may reach the cell's own threshold.
        for j in range(N_OUT):
            rec = b[f'SwitchI{j}']
            if f'SwitchI{j}' not in sw:
                assert rec['potential'] < i_thr
                assert rec['residual_charge'] < i_thr
                assert rec['trace_charge'] < i_thr

    # The natural chain fired -- and only ever the L2E3-paired switch.
    assert len(switch_fires) >= 2
    assert all(sw == ['SwitchI3'] for _, sw in switch_fires)
    # First legitimate firing is exactly one boundary after the first L2E3 win (t=3).
    assert switch_fires[0][0] == 4
    # Every firing lands paired inhibition on the NEXT boundary, onto L2E3 and nothing
    # else (never another competitor, never L1E/RG/ErrorE).
    for tf, _ in switch_fires:
        landed = [p for tp, pulses in switch_pulses if tp == tf + 1 for p in pulses]
        assert landed == [('SwitchI3', 'L2E3')]


def test_natural_residual_only_boundary_is_visibly_charged_but_silent():
    # This pins the reported dashboard observation AS CORRECT behaviour: after the switch
    # has fired and consumed its trace, ErrorE keeps broadcasting every boundary. The
    # switch's residual branch lights up (visible charge, the edge flashes) yet the cell
    # stays subthreshold and silent because its paired L2E is WTA-suppressed and supplies
    # no new trace. A residual broadcast is one operand, and one operand never fires.
    e = _strong_natural_row1_engine()
    i_thr = e.params['e_threshold'] / 3.0
    saw_residual_only = False
    for t in range(1, 26):
        d = e.step()
        b = by_id(d)
        rec = b['SwitchI3']
        if (not rec['spiked'] and rec['residual_received']
                and rec['winner_trace'] < e.params['switch_trace_threshold']):
            saw_residual_only = True
            assert rec['residual_charge'] > 0.0        # broadcast landed (edge flashes)
            assert rec['trace_charge'] == 0.0          # no eligible trace branch
            assert 0.0 < rec['potential'] < i_thr      # visibly charged, still subthreshold
    assert saw_residual_only
