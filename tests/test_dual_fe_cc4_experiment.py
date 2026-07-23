"""Smoke + scientific-assertion coverage for the headless dual FE/FES CC4 experiment.

Fast, artifact-free: it drives the experiment functions directly at small dwells. It also
pins the central scientific claim -- that at the confirmation candidate (B=5, m=100) the
direct four-neuron column assigns the four overlapping patterns one-to-one -- and the honest
negatives (reference m=1 too slow, RGC frequencies never halved).
"""

import math

import pytest

from experiments.dual_fe_cc4_consolidation import (
    build_engine, run_four_pattern, run_phase, c_probe, classify_failure,
    _switch_evidence, select_confirmation_candidate, CANONICAL_ORDER, REF,
)


def test_run_four_pattern_structure_and_rgc_integrity():
    eng = build_engine(1, dual=True, B=5.0, m=100)
    res, events = run_four_pattern(eng, dwell=120)
    assert len(res['owners']) == 4
    assert len(res['phases']) == 4
    assert isinstance(res['passed'], bool)
    # No active RGC feature was frequency-halved / dropped / inhibited in any phase.
    assert res['rgc_equal_frequency_all'] is True
    for p in res['phases']:
        assert p['rgc_equal_frequency'] is True


def test_candidate_b5_m100_assigns_four_distinct_owners_one_to_one():
    # The confirmation candidate: four distinct owners, turnover at every switch, recall
    # consistent -- the passing scientific result (seed 1, deterministic).
    eng = build_engine(1, dual=True, B=5.0, m=100)
    res, _ = run_four_pattern(eng, dwell=500)
    assert res['distinct_owners'] == 4 and None not in res['owners']
    assert res['turnover_every_switch'] is True
    assert res['recall_consistent'] is True
    assert res['passed'] is True and res['failure_class'] is None


def test_reference_m1_is_honestly_too_slow_not_input_suppression():
    # Reference LR (m=1) does not reach one-to-one in a bounded dwell; the failure is slow
    # learning / incumbent absorption, NOT any suppressed input feature.
    eng = build_engine(1, dual=True, B=5.0, m=1)
    res, _ = run_four_pattern(eng, dwell=400)
    assert res['passed'] is False
    assert res['distinct_owners'] < 4
    assert res['rgc_equal_frequency_all'] is True          # inputs stayed equal-frequency


def test_incumbent_updates_are_smaller_than_free_at_a_real_turnover():
    eng = build_engine(1, dual=True, B=5.0, m=100)
    res, _ = run_four_pattern(eng, dwell=500, collect_events=True)   # FE/dw evidence on
    rows = _switch_evidence(res['phases'], CANONICAL_ORDER)
    turnovers = [r for r in rows if r['turnover']]
    assert turnovers                                       # the candidate turns over
    # At every real turnover the incumbent's total |dw| is smaller than the free competitor's
    # (its established weights + high charge shrink FE; it also loses the WTA and stops firing).
    assert all(r['incumbent_smaller_updates'] for r in turnovers)


def test_flag_off_control_runs_same_topology():
    eng = build_engine(1, dual=False, m=1)
    assert eng.latency_competitors[0].update_mode == 'linear_fe'
    res, _ = run_four_pattern(eng, dwell=120)
    assert len(res['owners']) == 4                         # same topology runs with flag off


def test_isolated_c_probe_matures_and_reports_deposit_behavior():
    cp = c_probe(5.0, 100, steps=120)
    assert math.isfinite(cp['final_basal'])
    assert cp['nonfinite'] is False
    # from rest (leak 0), a single theta/4 deposit cannot fire (theta/4 < theta).
    assert cp['one_deposit_from_rest_fires'] is False
    assert set(cp) >= {'matured_to_one_shot_at', 'valid_coincidences', 'last_fe', 'last_fes'}


def test_classify_failure_taxonomy():
    # a synthetic all-distinct/consolidated/turnover/recall case -> PASS
    phases = [dict(competitor_fires={'ccE0': 1}, consolidated=True, turnover_from_prev=False),
              dict(competitor_fires={'ccE1': 1}, consolidated=True, turnover_from_prev=True),
              dict(competitor_fires={'ccE2': 1}, consolidated=True, turnover_from_prev=True),
              dict(competitor_fires={'ccE3': 1}, consolidated=True, turnover_from_prev=True)]
    owners = ['ccE0', 'ccE1', 'ccE2', 'ccE3']
    recall = dict(zip(CANONICAL_ORDER, owners))
    passed, cls = classify_failure(phases, owners, recall, CANONICAL_ORDER)
    assert passed is True and cls is None
    # an absorbing incumbent -> the corresponding failure class
    owners2 = ['ccE0', 'ccE0', 'ccE2', 'ccE3']
    passed2, cls2 = classify_failure(phases, owners2, dict(zip(CANONICAL_ORDER, owners2)),
                                     CANONICAL_ORDER)
    assert passed2 is False and cls2 == 'one_incumbent_absorbs_multiple_patterns'


def test_confirmation_candidate_rule_prefers_smallest_m_then_b_near_5():
    grid = [dict(B=1.0, m=100, passed=True), dict(B=5.0, m=100, passed=True),
            dict(B=50.0, m=500, passed=True), dict(B=5.0, m=1, passed=False)]
    cand = select_confirmation_candidate(grid)
    assert cand == {'B': 5.0, 'm': 100}                    # smallest m, then B closest to 5
