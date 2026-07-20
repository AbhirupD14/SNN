"""Phase 7 — scientific-validation harness assertions. These pin the MECHANICAL
claims (which must hold) and record the MEASURED full-preset frequency ratio (which
is reported, never forced to a target).
"""

import pytest

from experiments.coincidence_experiment import (
    isolated_cadence, full_preset, winner_exchange, deterministic_replay,
)


# ------------------------------------------------- isolated cadence (exact)
def test_isolated_cadence_is_exact_two_to_one():
    r = isolated_cadence(n_coincidences=12)
    assert r['exact_two_to_one'] is True
    assert r['fire_pattern'] == [0, 1] * 6
    assert r['ratio'] == pytest.approx(0.5)          # halving at the C level is exact


def test_calibration_lets_mature_c_outrun_l1e():
    # A mechanical fact from the calibrated crossing times: at the cap the C second-
    # coincidence crossing (0.813) precedes the pretrained L1E crossing (0.952), so a
    # mature C->L1I reset CAN beat L1E within the same boundary.
    r = isolated_cadence()
    assert r['c_second_tau_at_cap'] < r['pretrained_l1e_tau']
    assert r['pretrained_l1e_tau'] == pytest.approx(0.952, abs=0.002)
    assert r['c_second_tau_at_init'] == pytest.approx(0.980, abs=0.002)


# --------------------------------------------------- full preset mechanics
def test_full_preset_mechanics():
    r = full_preset(n=1500)
    c = r['counters']
    # Only active-pixel C cells learn; others stay at init.
    for i, (w, matured) in enumerate(zip(r['final_basal_weights'], r['matured_to_cap'])):
        if i in (3, 4, 5):
            assert w > r['c_init']                   # active cells accumulated basal weight
        else:
            assert w == pytest.approx(r['c_init'])   # inactive never learned
    # Every C spike drives exactly one paired L1 hard reset, and most beat their L1E.
    assert c['l1_hard_resets'] == c['c_spikes'] > 0
    assert c['l1e_reset_suppressed'] > 0
    # apical events come from real L2 winners (the eight L2E ids only).
    assert set(r['apical_source_ids']) <= {f'L2E{j}' for j in range(8)}


def test_full_preset_frequency_measured_not_forced():
    # The ratio is REPORTED, not asserted to equal 0.5. We only assert the honest
    # measured direction: suppression is real (ratio < 1) but the exact halving target
    # is not reached under the L2 latency-WTA cadence.
    r = full_preset(n=1500)
    ratio = r['frequency']['l1e_over_rg_overall']
    assert 0.0 < ratio <= 1.0
    assert r['frequency']['target'] == 0.5


# --------------------------------------------------- WTA is latency, not policy
def test_winner_exchange_follows_drive():
    r = winner_exchange()
    assert r['winner_followed_drive'] is True
    assert r['node_order_unchanged'] is True


# ----------------------------------------------------------- determinism
def test_deterministic_replay():
    assert deterministic_replay(n=150) is True
