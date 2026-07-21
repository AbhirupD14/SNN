"""Phase 1 (+ regression) — linear weight-update ablation: exact algebra for every E
and C mode, the byte-compatible production baseline, absence of hidden clipping in the
cap-free modes, mode persistence across rebuilds, and unchanged WTA/hard-reset timing.
"""

import math

import numpy as np
import pytest

from snn.neurons import (ExcitatoryNeuron, CoincidencePyramidalNeuron,
                         E_UPDATE_MODES, C_UPDATE_MODES, leak_to_conductance)
from backend.simulation import SimulationEngine


# ------------------------------------------------------------------ E helpers
def make_e(w, *, dist=None, mode='quadratic_bounded', eta=0.01, w_max=500.0,
           w_floor=0.0, threshold=1000.0):
    w = np.asarray(w, dtype=float)
    dist = np.ones_like(w) if dist is None else np.asarray(dist, dtype=float)
    return ExcitatoryNeuron('E', 'test', acc_weights=w.copy(), acc_distance_factor=dist,
                            eta=eta, w_max=w_max, w_floor=w_floor, threshold=threshold,
                            learn=True, update_mode=mode)


def learn(n, participation):
    n.fire()
    n.update_acc_weights(np.asarray(participation, dtype=bool))
    return n.acc_weights.copy()


# =============================================================== E algebra
def test_positive_fe_participate_potentiates_absent_depresses_all_modes():
    for mode in ('quadratic_bounded', 'linear_bounded', 'linear_nonnegative'):
        n = make_e([100., 100., 100.], mode=mode, eta=0.01, w_max=500.)
        before = n.acc_weights.copy()
        after = learn(n, [True, True, False])
        assert after[0] > before[0] and after[1] > before[1]     # participating grow
        assert after[2] < before[2]                              # absent depresses


def test_linear_delta_is_exactly_eta_fe_signal_distance():
    n = make_e([100., 100.], dist=[1.0, 0.5], mode='linear_bounded', eta=0.01, w_max=500.)
    fe = 1.10 * 1000.0 - 200.0                                    # budget headroom = 1.10*theta
    n.fire(); n.update_acc_weights(np.array([True, False]))
    exp = np.array([100. + 0.01 * fe * (+1) * 1.0,
                    100. + 0.01 * fe * (-1) * 0.5])
    assert n.acc_weights == pytest.approx(exp)                   # no (1-q^2) factor


def test_quadratic_delta_includes_the_multiplier():
    n = make_e([490.], mode='quadratic_bounded', eta=0.01, w_max=500.)
    fe = 1.10 * 1000.0 - 490.0                                    # budget applies to both modes
    q = 1.0 - (490.0 / 500.0) ** 2
    n.fire(); n.update_acc_weights(np.array([True]))
    assert n.acc_weights[0] == pytest.approx(min(500.0, 490.0 + 0.01 * fe * q))


def test_zero_fe_produces_zero_delta_all_modes():
    for mode in E_UPDATE_MODES:
        # Zero FE now means sum(w) == the budget target (1.10*theta), not theta.
        n = make_e([550., 550.], mode=mode, eta=0.1, w_max=1000., threshold=1000.)
        before = n.acc_weights.copy()
        learn(n, [True, False])
        assert n.acc_weights == pytest.approx(before)            # FE=0 -> no change


def test_negative_fe_reverses_direction():
    for mode in E_UPDATE_MODES:
        n = make_e([600., 600.], mode=mode, eta=0.001, w_max=1000., threshold=1000.)
        after = learn(n, [True, False])                          # sum=1200 -> FE=-200
        assert after[0] < 600.0                                  # participating DECREASES
        assert after[1] > 600.0                                  # absent INCREASES


def test_linear_bounded_removes_near_cap_slowdown():
    # Near w_max the quadratic factor -> 0; linear keeps the full step.
    q = make_e([490.], mode='quadratic_bounded', eta=0.001, w_max=500.)
    l = make_e([490.], mode='linear_bounded', eta=0.001, w_max=500.)
    dq = learn(q, [True])[0] - 490.0
    dl = learn(l, [True])[0] - 490.0
    assert dl > 10 * dq                                          # linear ~25x faster here


def test_only_bounded_modes_clip_at_w_max():
    for mode, clips in (('quadratic_bounded', True), ('linear_bounded', True),
                        ('linear_nonnegative', False)):
        n = make_e([100.], mode=mode, eta=5.0, w_max=500.)       # huge step over the cap
        after = learn(n, [True])[0]
        if clips:
            assert after == pytest.approx(500.0)
        else:
            assert after > 500.0                                 # cap-free exceeds w_max


def test_every_mode_retains_the_zero_floor():
    for mode in E_UPDATE_MODES:
        n = make_e([5., 600.], mode=mode, eta=5.0, w_max=1000., threshold=1000.)
        after = learn(n, [False, True])                          # afferent 0 depressed hard
        assert after[0] == pytest.approx(0.0) and after[0] >= 0.0


def test_cap_free_weight_grows_above_historical_cap_no_hidden_clip():
    n = make_e([100.], mode='linear_nonnegative', eta=0.05, w_max=500., threshold=1e12)
    for _ in range(50):
        n.fire(); n.update_acc_weights(np.array([True]))
    assert n.acc_weights[0] > 500.0                              # no hidden ceiling
    assert math.isfinite(n.acc_weights[0])


# ------------------------------------------------------------------ C helpers
def make_c(w=505.0, *, w_max=550.0, eta_c=0.001, mode='c_quadratic_bounded',
           use_fe=True, threshold=1000.0, phi=1.0):
    return CoincidencePyramidalNeuron('L1C0', 'L1E0', 'b0', apical_sources=['L2E0'],
                                      apical_edge_ids=['a0'], basal_weight=w, w_max=w_max,
                                      eta_c=eta_c, use_fe=use_fe, update_mode=mode,
                                      threshold=threshold, basal_distance_factor=phi)


def c_fire_and_learn(c):
    c.apical_active = True
    c._deposit_signal = 1.0
    return c.update_basal_weight()


# =============================================================== C algebra
def test_c_exact_delta_all_modes():
    w, wmax, eta, theta, phi = 505.0, 550.0, 0.001, 1000.0, 1.0
    fe = 1.10 * theta - w                    # budget FE: frac*w1 - w, w1==theta at leak 0
    expected = {
        'c_quadratic_bounded': w + eta * fe * 1.0 * 1.0 * phi * (1 - (w / wmax) ** 2),
        'c_linear_bounded': w + eta * fe * 1.0 * 1.0 * phi,
        'c_linear_nonnegative': w + eta * fe * 1.0 * 1.0 * phi,
    }
    for mode, exp in expected.items():
        c = make_c(w, w_max=wmax, eta_c=eta, mode=mode, threshold=theta, phi=phi)
        assert c_fire_and_learn(c) == pytest.approx(min(exp, wmax)
                                                    if mode != 'c_linear_nonnegative' else exp)


def test_c_no_learning_on_basal_or_apical_only():
    for mode in C_UPDATE_MODES:
        c = make_c(mode=mode)
        c.apical_active = False                                  # no coincidence gate
        c._deposit_signal = 1.0
        before = c.basal_weight
        assert c.update_basal_weight() == pytest.approx(before)  # A=0 -> dw=0


def test_c_linear_bounded_clips_at_temporal_cap():
    c = make_c(w=548.0, w_max=550.0, eta_c=5.0, mode='c_linear_bounded')  # huge step
    assert c_fire_and_learn(c) == pytest.approx(550.0)           # clipped at derived cap


def test_c_linear_nonnegative_exceeds_the_cap():
    c = make_c(w=548.0, w_max=550.0, eta_c=5.0, mode='c_linear_nonnegative')
    assert c_fire_and_learn(c) > 550.0                          # genuinely uncapped


def test_c_default_mode_is_production_quadratic():
    c = make_c()
    assert c.update_mode == 'c_quadratic_bounded'


# ===================================================== integration / regression
def test_default_modes_are_production():
    # PROMOTED production defaults: linear-bounded E and linear-bounded C (multiplier
    # dropped on both; the historical quadratic rules remain as headless modes).
    e = SimulationEngine(seed=1, topology='rg_coincidence')
    assert e.latency_competitors[0].update_mode == 'linear_bounded'
    assert e.coincidence[0].update_mode == 'c_linear_bounded'


def test_engine_default_e_equals_explicit_linear_bounded():
    # The engine default must be identical to an explicit linear_bounded request, and
    # DIFFERENT from the historical quadratic mode (over a real learning trajectory).
    def sums(mode):
        kw = {} if mode is None else dict(e_weight_update_mode=mode)
        e = SimulationEngine(seed=1, topology='rg_coincidence', **kw)
        e.set_pattern('row 1')
        for _ in range(80):
            e.step()
        return [round(float(c.acc_weights.sum()), 9) for c in e.latency_competitors]
    assert sums(None) == sums('linear_bounded')          # default == explicit linear
    assert sums(None) != sums('quadratic_bounded')       # and differs from historical


def test_mode_persists_across_reset_and_reseed():
    e = SimulationEngine(seed=1, topology='rg_coincidence',
                         e_weight_update_mode='linear_bounded',
                         c_weight_update_mode='c_linear_bounded')
    e.reset()
    assert e.latency_competitors[0].update_mode == 'linear_bounded'
    assert e.coincidence[0].update_mode == 'c_linear_bounded'
    e.reseed()
    assert e.latency_competitors[0].update_mode == 'linear_bounded'


def test_mode_propagates_through_custom_topology_rebuild():
    e = SimulationEngine(seed=1, topology='rg_coincidence',
                         e_weight_update_mode='linear_nonnegative')
    spec = e.current_spec()
    e.apply_topology(spec)                                       # rebuild from a custom graph
    assert all(c.update_mode == 'linear_nonnegative' for c in e.latency_competitors)


def test_invalid_modes_rejected():
    with pytest.raises(ValueError):
        SimulationEngine(seed=1, topology='rg_coincidence', e_weight_update_mode='nope')
    with pytest.raises(ValueError):
        SimulationEngine(seed=1, topology='rg_coincidence', c_weight_update_mode='nope')


def test_instrumentation_off_is_byte_identical():
    def trace(record):
        e = SimulationEngine(seed=1, topology='rg_coincidence', leak_rate=0.0,
                             l2_init_total_frac=0.95, e_weight_cap=500.0)
        e.set_pattern('row 1')
        if record:
            for c in (*e.latency_competitors, *e.coincidence):
                c.record_updates = True
        frames = []
        for _ in range(120):
            e.step()
            frames.append(tuple(round(w, 9) for c in e.latency_competitors
                                for w in c.acc_weights))
        return frames
    assert trace(False) == trace(True)                          # recording never perturbs


def test_instrumentation_records_raw_vs_applied_delta():
    # In a bounded mode a step crossing w_max has raw != applied (clipping); the record
    # must expose both. Drive an isolated E cell to the cap.
    n = make_e([100.], mode='linear_bounded', eta=5.0, w_max=500.)
    n.record_updates = True
    n.fire(); n.update_acc_weights(np.array([True]))
    rec = n.update_log[-1]
    assert rec['raw_delta_norm'] > rec['applied_delta_norm']    # clipping shrank the step
    assert rec['n_at_max'] == 1 and rec['mode'] == 'linear_bounded'


def test_linear_e_mode_does_not_change_wta_one_spike_contract():
    e = SimulationEngine(seed=1, topology='rg_coincidence',
                         e_weight_update_mode='linear_bounded', leak_rate=0.0)
    e.set_pattern('row 1')
    for _ in range(60):
        e.step()
        assert sum(1 for c in e.latency_competitors if c.spiked) <= 1   # WTA unchanged


def test_linear_e_mode_hard_reset_shares_causal_tau():
    e = SimulationEngine(seed=1, topology='rg_coincidence',
                         e_weight_update_mode='linear_bounded', leak_rate=0.0)
    e.set_pattern('row 1')
    for _ in range(30):
        e.step()
        for c in e.latency_competitors:
            if c.spiked:
                for h in e.hard_reset_events:
                    if h['source'] == 'L2I':
                        assert h['tau'] == pytest.approx(c.spike_tau)


def test_composition_probe_freezes_learned_weights():
    from experiments.linear_ablation import _phase4_run
    r = _phase4_run('B', 1, train=300)
    # No learned E/L2E or C weight may change during either diagnostic probe.
    assert r['carried']['learned_weights_unchanged'] is True
    assert r['controlled']['learned_weights_unchanged'] is True


def test_composition_probe_captures_first_plus_driven_boundary():
    from experiments.linear_ablation import _phase4_probe

    class FakeEngine:
        """Three diagnostic frames: no drive, first drive, then a later drive."""
        def __init__(self):
            self.latency_competitors = []
            self._crossing_capture = None
            self.frames = [
                {'L2E0': {'frozen_excitation': 0.0}},
                {'L2E0': {'frozen_excitation': 2.0}},
                {'L2E0': {'frozen_excitation': 3.0}},
            ]
            self.steps = 0

        def set_input(self, _values):
            pass

        def step(self):
            self._crossing_capture.append(self.frames[self.steps])
            self.steps += 1

    eng = FakeEngine()
    snap, boundary = _phase4_probe(eng, washout=False)
    assert boundary == 1
    assert snap == eng.frames[1]
    assert eng.steps == 2                                  # later driven frame was not sampled


def test_controlled_probe_equalizes_membrane():
    # Controlled-state washout zeroes L2E V (and g_inh) before the captured boundary, so
    # the owners' boundary-start membrane is exactly rest; carried-state generally is not.
    from experiments.linear_ablation import _phase4_run
    r = _phase4_run('B', 1, train=1000)
    assert r['controlled']['row']['v_before_drive'] == 0.0
    assert r['controlled']['col']['v_before_drive'] == 0.0
    assert r['controlled']['row']['g_inh'] == 0.0


def test_crossing_capture_hook_off_is_byte_identical():
    def trace(capture):
        e = SimulationEngine(seed=1, topology='rg_coincidence', leak_rate=0.0)
        e.set_pattern('row 1')
        if capture:
            e._crossing_capture = []
        return [tuple(round(w, 9) for c in e.latency_competitors for w in c.acc_weights)
                for _ in range(80) if (e.step() or True)]
    assert trace(False) == trace(True)                          # capture never perturbs


def test_experiment_result_schema():
    from experiments.linear_ablation import (_iso_e, _iso_c, _fixed_part,
                                             _phase3_run, _phase4_run)
    e = _iso_e('linear_bounded', _fixed_part([0, 1, 2]), n_events=50)
    assert {'mode', 'reach_pct', 'neg_fe_count', 'overshoot_count', 'nan_or_inf',
            'final_max_w'} <= set(e)
    c = _iso_c('c_linear_bounded', n_coincidences=20)
    assert {'mode', 'two_event_cadence_ok_full', 'cadence_mismatch_count',
            'first_one_from_rest', 'first_reach_w1', 'min_safety_margin',
            'rejection_condition_observed', 'w1'} <= set(c)
    r3 = _phase3_run('A', 1)
    assert {'owners', 'phase_winner_counts', 'final500_winner_counts', 'init_totals',
            'final_totals', 'overshoot_max', 'floor_hits', 'min_c_safety_margin',
            'neg_fe_count', 'overshoot_count', 'weights_above_cap',
            'c_fired_from_rest', 'l1e_over_rg', 'abort'} <= set(r3)
    r4 = _phase4_run('A', 1, train=200)
    assert {'row_owner', 'col_owner', 'distinct_owners', 'carried', 'controlled'} <= set(r4)
    for probe in ('carried', 'controlled'):
        assert {'row', 'col', 'n_finite_crossings', 'tau_gap', 'earlier',
                'captured_boundary', 'learned_weights_unchanged'} <= set(r4[probe])
        assert {'v_before_drive', 'frozen_excitation', 'g_inh',
                'projected_uninhibited_end_v', 'active_input_wsum', 'refractory',
                'tau', 'finite'} <= set(r4[probe]['row'])
