"""Phase 7 — scientific-validation harness assertions. These pin the MECHANICAL
claims (which must hold), including the mature full-preset frequency ratio.
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


def test_impulse_fires_at_permission_tau_independent_of_remaining_interval():
    r = isolated_cadence(impulse_tau=0.75)
    assert r['c_init_single_impulse_subthreshold'] is True
    assert r['c_max_single_impulse_suprathreshold'] is True
    assert set(r['crossing_taus']) == {0.75}
    assert r['pretrained_l1e_tau'] == pytest.approx(0.952, abs=0.002)


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
    assert r['timing']['same_tau_c_spikes'] == c['c_spikes']
    assert r['timing']['phase_mismatches'] == 0
    # apical events come from real L2 winners (the eight L2E ids only).
    assert set(r['apical_source_ids']) <= {f'L2E{j}' for j in range(8)}


def test_full_preset_mature_window_reaches_frequency_halving():
    # The overall ratio includes the immature training transient. Once the active C
    # weights mature, the established final window reaches the requested halving.
    r = full_preset(n=2500, window=500)
    ratio = r['frequency']['l1e_over_rg_overall']
    assert 0.0 < ratio <= 1.0
    assert r['frequency']['l1e_rate_last_window'] == pytest.approx(0.5)
    assert r['frequency']['target'] == 0.5


# --------------------------------------------------- WTA is latency, not policy
def test_winner_exchange_follows_drive():
    r = winner_exchange()
    assert r['winner_followed_drive'] is True
    assert r['node_order_unchanged'] is True


# ----------------------------------------------------------- determinism
def test_deterministic_replay():
    assert deterministic_replay(n=150) is True
