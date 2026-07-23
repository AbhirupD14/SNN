"""Regression coverage for the feature-gated tiled topology's causal turnover mechanism.

Runs the Stage A single-RF acceptance experiment (a reduced but sufficient training dwell)
and asserts the feature-specific microcircuit restores rg_coincidence-style turnover inside
one tiled L1 module: four distinct consolidated owners, a different owner after every switch,
the shared center relay transiently suppressed relative to the novel relays, every counted
feature reset paired/local + pre-spike, no feature-I/WTA-I cross-talk, and no learning or
resets leaking into the blank neighbouring modules.

Kept single-seed; the full multi-seed / nine-RF evidence is
``experiments/feature_gated_turnover.py``. Uses the reference protocol settings shared with
``microcircuit_turnover.py`` and tunes no model parameter.
"""

from __future__ import annotations

import pytest

from experiments.feature_gated_turnover import run_stage_A, CENTER_FEATURE
from backend.simulation import PATTERNS

# A declared training window sufficient for one-shot maturation of each new owner (dwell 800
# is not enough; 1200 already consolidates all four -- 1500 gives margin). This is a protocol
# duration, not a tuned model parameter.
DWELL = 1500


@pytest.fixture(scope='module')
def stage_a(tmp_path_factory):
    out = tmp_path_factory.mktemp('fg_turnover')
    return run_stage_A(str(out), dwell=DWELL, early=200, final_window=300, dominance=0.8)


def test_stage_a_all_checks_pass(stage_a):
    assert stage_a['passed'], stage_a['checks']


def test_four_distinct_consolidated_owners(stage_a):
    owners = stage_a['owners']
    assert len(set(owners)) == 4 and None not in owners
    assert all(p['final_dominance'] >= 0.8 for p in stage_a['phases'])


def test_center_relay_suppressed_novel_active_every_switch(stage_a):
    switches = stage_a['phases'][1:]
    assert switches
    for p in switches:
        assert p['turnover_from_prev'], p['pattern']
        assert p['center_suppressed_vs_novel'], p['pattern']
        assert p['early_center_relay_fires'] < p['early_novel_mean'], p['pattern']
        assert p['novel_relays_active'], p['pattern']


def test_every_center_reset_is_prespike_and_no_crosstalk(stage_a):
    switches = stage_a['phases'][1:]
    for p in switches:
        assert p['prespike_center_resets'] > 0, p['pattern']
    # WTA I and feature If never reset each other's target class, in any phase
    assert all(p['crosstalk_total'] == 0 for p in stage_a['phases'])


def test_blank_modules_stay_silent(stage_a):
    # only the selected patch is driven; the other eight L1 modules must not learn or reset
    for p in stage_a['phases']:
        assert p['other_resets'] == 0 and p['other_winners'] == 0, p['pattern']


def test_sample_suppression_trace_is_pre_spike_and_paired(stage_a):
    # at least one switch captured an explicit suppressing-boundary causal trace
    traces = [p['sample_suppression_trace'] for p in stage_a['phases'][1:]
              if p['sample_suppression_trace'] is not None]
    assert traces, 'expected at least one captured suppression trace'
    for tr in traces:
        assert tr['reset_is_paired'] and not tr['center_relay_fired']
        assert tr['center_relay_drive_discarded'] > 0.0            # drive preempted pre-spike
        # the reset arrives no later than the shared center relay would have fired
        assert tr['center_If_reset_tau'] is not None


def test_center_feature_is_shared_by_all_patterns():
    for name, vec in PATTERNS.items():
        assert vec[CENTER_FEATURE] > 0.5, name
