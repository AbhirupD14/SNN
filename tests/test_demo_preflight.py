"""Focused coverage for the demo-recording preflight (scripts/preflight_demo_measurements.py).

The four presentation videos are only honest if their on-screen numbers come from a
deterministic measurement of the SAME configuration the recording server is driven with.
These tests pin that contract:

  * the recorded engine configuration matches the published headless experiment's
    confirmation candidate (B=5, m=100) and the dashboard construction overrides;
  * each measurement function returns the fields the recorder's captions consume;
  * the measurements are deterministic (identical across repeated runs at seed 1);
  * the two-tower measurement reproduces the DOCUMENTED NEGATIVE result rather than
    accidentally claiming L3 separates the glyphs.

Fast: dwells are cut to the smallest values that still exercise every code path. The full
scientific claims are pinned by tests/test_dual_fe_cc4_experiment.py and
tests/test_two_tower_composition.py.
"""

import pytest

from experiments.dual_fe_cc4_consolidation import REF
from scripts import preflight_demo_measurements as pf


def test_cc4_config_is_the_published_confirmation_candidate():
    # B = 5 with LR multiplier m = 100 -> eta = 0.01*100, c_eta = 0.005*100.
    assert pf.CC4_B == REF['B'] == 5.0
    assert pf.CC4_M == 100
    assert pf.CC4_CONFIG['eta'] == pytest.approx(REF['eta'] * 100)
    assert pf.CC4_CONFIG['c_eta'] == pytest.approx(REF['c_eta'] * 100)
    assert pf.CC4_CONFIG['topology'] == 'rg_direct_cc4'
    assert pf.CC4_CONFIG['dual_fe_fes'] is True
    assert pf.CC4_CONFIG['leak_rate'] == REF['leak_rate']
    assert pf.CC4_CONFIG['refractory_steps'] == REF['refractory_steps']


def test_build_applies_the_dashboard_construction_overrides():
    # The recording runs on the ORDINARY dashboard server, so every measured engine must
    # carry the construction overrides /api/config cannot undo.
    engine = pf.build(pf.CC4_CONFIG)
    assert engine.params['seed'] == pf.SEED == 1
    assert engine.params['e_weight_cap_frac'] == pf.DASHBOARD_CONSTRUCTION['e_weight_cap_frac']
    assert engine.params['topology'] == 'rg_direct_cc4'
    assert engine.params['dual_fe_B'] == 5.0


def test_build_extra_overrides_win_over_the_dashboard_defaults():
    engine = pf.build(pf.CC4_CONFIG, e_weight_cap_frac=None)
    assert engine.params['e_weight_cap_frac'] is None


def test_measure_cadence_reports_the_fields_the_caption_uses():
    r = pf.measure_cadence(total=400, early_window=100, late_window=100)
    for key in ('early_emission_ratio_mean', 'late_emission_ratio_mean',
                'cadence_lock_boundary', 'cadence_lock_criterion', 'sample_early',
                'sample_late', 'active_l1e', 'resolved_input_period'):
        assert key in r, key
    assert r['topology'] == 'rg_coincidence'
    # "row 1" is the middle row of the 3x3 sheet -> pixels 3,4,5.
    assert r['active_l1e'] == ['L1E3', 'L1E4', 'L1E5']
    assert set(r['sample_early']) <= {'0', '1'}
    assert 0.0 <= r['late_emission_ratio_mean'] <= 1.0


def test_cadence_transition_is_downward_and_locks_to_strict_alternation():
    # The scientific content of video 1: emission per RG volley FALLS toward 0.5 and every
    # active L1E ends up alternating fire/silent with no repeat.
    r = pf.measure_cadence(total=2000, early_window=400, late_window=400)
    assert r['early_emission_ratio_mean'] > r['late_emission_ratio_mean']
    assert r['late_emission_ratio_mean'] == pytest.approx(0.5, abs=1e-9)
    assert r['cadence_lock_boundary'] is not None
    assert 0 < r['cadence_lock_boundary'] < 2000
    assert r['sample_late'].startswith(('01' * 4, '10' * 4))


def test_measure_continuous_is_one_network_with_four_distinct_owners():
    r = pf.measure_continuous(dwell=500)
    assert r['order'] == list(pf.CANONICAL_ORDER)
    assert len(r['phases']) == 4
    assert r['distinct_owners'] == 4
    assert r['one_to_one'] is True
    assert r['turnover_every_switch'] is True
    assert r['recall_consistent'] is True
    assert set(r['owner_by_pattern']) == set(pf.CANONICAL_ORDER)
    # the first phase has no predecessor to turn over from
    assert r['phases'][0]['previous_owner'] is None
    assert all(p['previous_owner'] is not None for p in r['phases'][1:])


def test_measure_long_dwell_declares_a_criterion_and_a_measured_latency():
    r = pf.measure_long_dwell(train_dwell=400, long_dwell=1200, post_dwell=300)
    assert r['held_pattern'] == 'row 1' and r['switched_to'] == 'col 1'
    assert r['incumbent'] is not None
    assert r['turnover_occurred'] is True
    assert r['new_owner'] != r['incumbent']
    # The latency must be an explicit measured index under a declared criterion -- never the
    # word "instant" unless it really was the first eligible presentation.
    assert isinstance(r['turnover_eligible_presentation'], int)
    assert r['turnover_eligible_presentation'] >= 1
    assert r['first_presentation_instantaneous'] == (r['turnover_eligible_presentation'] == 1)
    assert 'unbroken run' in r['consolidation_criterion']
    assert r['turnover_boundary_after_switch'] >= r['turnover_eligible_presentation']


def test_long_dwell_incumbent_does_not_absorb_the_new_pattern():
    # The point of video 3: a hugely over-exposed incumbent still loses the new pattern.
    r = pf.measure_long_dwell(train_dwell=400, long_dwell=2000, post_dwell=400)
    counts = r['post_switch_winner_counts']
    assert counts[r['new_owner']] > counts.get(r['incumbent'], 0)


def test_measure_scaling_matches_the_documented_graph_sizes():
    s = pf.measure_scaling()
    assert s['tiled_cc']['nodes'] == 191 and s['tiled_cc']['edges'] == 1052
    assert s['two_tower_composition']['nodes'] == 393
    assert s['two_tower_composition']['edges'] == 2162
    assert s['tiled_cc']['input_shape'] == {'rows': 9, 'cols': 9}
    assert s['two_tower_composition']['input_shape'] == {'rows': 9, 'cols': 18}
    assert s['two_tower_composition']['columns_by_layer'] == {'L1': 18, 'L2': 2, 'L3': 1}
    assert 'V' in s['two_tower_composition']['patterns']


def test_two_tower_stimuli_preserve_the_documented_negative_result():
    # docs/TWO_TOWER_COMPOSITION.md: the towers separate V/A/7 internally, the single L3
    # column does NOT. The demo must never claim otherwise.
    r = pf.measure_two_tower_stimuli(dwell=400)
    assert r['glyphs'] == ['V', 'A', '7']
    assert r['l3_separates_glyphs'] is False
    assert r['distinct_l3_top_owners'] == 1
    assert r['distinct_l2_top_owners'] > r['distinct_l3_top_owners']


def test_measurements_are_deterministic_at_seed_1():
    a = pf.measure_continuous(dwell=300)
    b = pf.measure_continuous(dwell=300)
    assert a['owners'] == b['owners']
    assert a['recall'] == b['recall']
    c = pf.measure_cadence(total=300, early_window=100, late_window=100)
    d = pf.measure_cadence(total=300, early_window=100, late_window=100)
    assert c['sample_early'] == d['sample_early']
    assert c['cadence_lock_boundary'] == d['cadence_lock_boundary']


def test_theta_over_two_ceiling_does_not_change_the_cc4_scientific_result():
    """The recording server applies e_weight_cap_frac=0.5; the published headless experiment
    leaves it None. The demo is only fair if the ceiling does not change the outcome."""
    capped = pf.measure_continuous(dwell=500)
    uncapped = pf.measure_continuous(dwell=500, e_weight_cap_frac=None)
    assert capped['owners'] == uncapped['owners']
    assert capped['recall'] == uncapped['recall']
    assert capped['one_to_one'] == uncapped['one_to_one'] is True
