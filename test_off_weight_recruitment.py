"""Focused tests for the inhibitory OFF-weight redistribution/recruitment rule.

Covers the core behavior groups for the "redistribution" competitive
weight-update mode and its A/B isolation against the retained "depression" baseline
and the "none" hard-reset-only control:

  - refractory winner protection (protection is the refractory timer, not a one-hot
    winner vector or an engine winner id);
  - the structural match signal (p_match from the active EFFECTIVE-weight sum, not
    membrane charge / history);
  - conservation and bounds of the ON->OFF transfer;
  - mode selection and A/B isolation;
  - basic recruitment behaviour.

Run directly: `python3 test_off_weight_recruitment.py`.
"""

import numpy as np

from neuron_flexible import Neuron
from snn.rules import bounded_signed_update
from backend.simulation import SimulationEngine, N_OUT, N_PIX, PATTERNS


# ---------------------------------------------------------------------------
def _l2e(weights, theta=8.0, cap=8.0 / 3, floor=0.0, lr=0.02, refractory=0):
    n = Neuron(threshold=theta, refractory_period=refractory, weight_cap=cap,
               learning_rate=lr)
    n.weights = np.asarray(weights, dtype=float)
    n.min_positive_weight = floor
    n.structural_free_energy = False
    return n


def _mask(*bits):
    return np.asarray(bits, dtype=float)


# ======================================================================
# Refractory winner protection
# ======================================================================
def test_refractory_winner_no_redistribution():
    n = _l2e([2.0, 2.0, 0.2, 0.1], refractory=1)
    n.refractory_timer = 1                       # fired this step -> current winner
    n.potential = 0.8 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert rec['refractory_at_arrival'] is True
    assert rec['plasticity_applied'] is False
    assert np.array_equal(n._weights_array, before), "refractory winner redistributed"
    assert n.refractory_timer == 1, "reset must not touch the refractory timer"
    # membrane/current traces still end at exact rest/zero
    assert n.potential == n.resting_potential
    assert n.exc_trace == 0.0 and n.inh_trace == 0.0
    print("PASS refractory: current winner is protected; membrane ends at rest")


def test_nonrefractory_loser_redistributes():
    n = _l2e([2.0, 2.0, 0.2, 0.1], refractory=1)
    n.refractory_timer = 0                        # loser
    n.potential = 0.8 * n.threshold
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert rec['refractory_at_arrival'] is False
    assert rec['plasticity_applied'] is True
    assert rec['transferred'] > 0.0
    print("PASS refractory: a non-refractory loser redistributes")


def test_no_winner_vector_in_decision():
    # Two identical neurons; only the refractory timer differs. The spiked flag /
    # one-hot vector is irrelevant -- only refractory_timer gates plasticity.
    prot = _l2e([2.0, 2.0, 0.2, 0.1]); prot.refractory_timer = 1; prot.spiked = False
    loser = _l2e([2.0, 2.0, 0.2, 0.1]); loser.refractory_timer = 0; loser.spiked = True
    for n in (prot, loser):
        n.potential = 0.8 * n.threshold
    rp = prot.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    rl = loser.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert rp['plasticity_applied'] is False and rl['plasticity_applied'] is True
    print("PASS refractory: spiked/one-hot state does not drive the plasticity decision")


# ======================================================================
# Structural match signal
# ======================================================================
def test_larger_active_sum_larger_transfer():
    strong = _l2e([2.4, 2.4, 0.2, 0.1])
    weak = _l2e([1.0, 1.0, 0.2, 0.1])
    for n in (strong, weak):
        n.potential = 0.5 * n.threshold           # identical membrane
    rs = strong.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    rw = weak.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert rs['p_match'] > rw['p_match']
    assert rs['transferred'] > rw['transferred']
    print(f"PASS match: larger active sum -> larger transfer "
          f"(p_match {rw['p_match']:.3f} < {rs['p_match']:.3f})")


def test_zero_active_sum_no_redistribution():
    # All participating gates are OFF-bank (no active positive gate).
    n = _l2e([0.2, 0.1, 2.0, 2.0])
    n.potential = 0.9 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(0, 0, 0, 0), "redistribution")   # nothing active
    assert rec['p_match'] == 0.0 and rec['transferred'] == 0.0
    assert np.array_equal(n._weights_array, before)
    print("PASS match: a zero active sum causes no redistribution")


def test_membrane_charge_does_not_affect_redistribution():
    # Identical weights/input/refractory but very different membrane charge -> the
    # redistribution must be identical (signal is structural, not membrane charge).
    a = _l2e([2.0, 2.0, 0.2, 0.1]); a.potential = 0.1 * a.threshold
    b = _l2e([2.0, 2.0, 0.2, 0.1]); b.potential = 0.99 * b.threshold
    ra = a.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    rb = b.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert np.isclose(ra['transferred'], rb['transferred'])
    assert np.allclose(a._weights_array, b._weights_array)
    print("PASS match: membrane charge does not affect redistribution")


def test_distance_factor_affects_p_match_like_delivery():
    # The structural support signal must use the same effective weights that deliver
    # membrane charge, while the learning kernel itself remains unchanged.
    r_plain = _l2e([2.0, 2.0, 0.2, 0.1])
    r_plain.potential = 0.5 * r_plain.threshold
    rec_plain = r_plain.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")

    r_dist = _l2e([2.0, 2.0, 0.2, 0.1])
    r_dist.potential = 0.5 * r_dist.threshold
    r_dist.distance_weighting = True
    r_dist.distance_power = 1.0
    r_dist.distance_ref = 1.0
    r_dist.distance_min = 1.0
    r_dist._distance = np.array([2.0, 2.0, 1.0, 1.0])
    rec_dist = r_dist.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert np.isclose(rec_dist['p_match'], 0.5 * rec_plain['p_match']), \
        (rec_dist['p_match'], rec_plain['p_match'])
    print("PASS match: distance factor scales p_match exactly like charge delivery")


# ======================================================================
# Conservation and bounds
# ======================================================================
def test_conservation_active_down_off_up():
    n = _l2e([2.0, 1.8, 0.2, 0.1, 0.05])
    n.potential = 0.6 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0, 0), "redistribution")
    active = np.array(rec['active_indices']); off = np.array(rec['off_indices'])
    assert (n._weights_array[active] < before[active]).all(), "active gates must decrease"
    assert (n._weights_array[off] >= before[off]).all(), "OFF gates must not decrease"
    active_drop = float((before[active] - n._weights_array[active]).sum())
    off_gain = float((n._weights_array[off] - before[off]).sum())
    assert np.isclose(active_drop, off_gain), (active_drop, off_gain)
    assert np.isclose(before.sum(), n._weights_array.sum()), "total mass changed"
    print(f"PASS conserve: active drop {active_drop:.5f} == OFF gain {off_gain:.5f}")


def test_bounds_never_crossed():
    cap = 8.0 / 3
    n = _l2e([2.0, 1.9, cap - 0.01, cap - 0.02, 0.05], cap=cap, floor=0.0)
    for _ in range(200):
        n.potential = 0.9 * n.threshold
        n.apply_competitive_reset(_mask(1, 1, 0, 0, 0), "redistribution")
    assert (n._weights_array >= 0.0 - 1e-12).all(), n._weights_array
    assert (n._weights_array <= cap + 1e-9).all(), n._weights_array
    print("PASS bounds: no active weight crosses w_min; no OFF weight crosses w_cap")


def test_insufficient_off_capacity_scales_down_active():
    cap = 8.0 / 3
    # OFF gates almost full -> tiny capacity -> transfer < candidate release, and the
    # active decrease is scaled to only what can be transferred.
    n = _l2e([2.0, 2.0, cap - 0.001, cap - 0.001], cap=cap)
    n.potential = 0.9 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert rec['transferred'] <= rec['candidate_release'] + 1e-12
    assert rec['transferred'] < rec['candidate_release'], "expected capacity shortfall"
    assert np.isclose(before.sum(), n._weights_array.sum())
    print(f"PASS capacity: shortfall scales active decrease "
          f"(T {rec['transferred']:.5f} < R {rec['candidate_release']:.5f})")


def test_no_off_capacity_no_change():
    cap = 8.0 / 3
    n = _l2e([2.0, 2.0, cap, cap], cap=cap)     # OFF gates already at cap
    n.potential = 0.9 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    assert rec['transferred'] == 0.0
    assert np.array_equal(n._weights_array, before)
    print("PASS capacity: no OFF capacity -> no weights change")


def test_capped_allocation_redistributes_remainder():
    cap = 8.0 / 3
    # One OFF gate is near cap (small capacity) and one has plenty. The near-cap gate
    # clips and the remainder deterministically goes to the open gate; conservation
    # and the cap both hold.
    n = _l2e([2.4, 2.4, cap - 0.002, 0.05], cap=cap)
    n.potential = 0.95 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
    off = np.array(rec['off_indices'])
    assert (n._weights_array[off] <= cap + 1e-9).all()
    off_gain = float((n._weights_array[off] - before[off]).sum())
    assert np.isclose(off_gain, rec['transferred'])
    print("PASS allocate: clipping remainder is redistributed, cap respected")


def test_incremental_over_repeated_losses():
    n = _l2e([2.4, 2.4, 0.2, 0.1])
    first = None
    for i in range(5):
        n.potential = 0.8 * n.threshold
        before = n._weights_array[0]
        n.apply_competitive_reset(_mask(1, 1, 0, 0), "redistribution")
        step = before - n._weights_array[0]
        if first is None:
            first = step
        # each single event moves only a bounded increment, not the whole active mass
        assert step < 0.5 * before, step
    assert n._weights_array[0] > 0.0
    print("PASS incremental: repeated losses move capacity gradually, not at once")


# ======================================================================
# Mode selection and A/B isolation
# ======================================================================
def test_depression_leaves_off_unchanged():
    n = _l2e([2.0, 2.0, 0.2, 0.1])
    n.potential = 0.9 * n.threshold
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "depression")
    assert set(rec['depressed_indices']) == {0, 1}
    assert (n._weights_array[[0, 1]] < before[[0, 1]]).all()
    assert np.array_equal(n._weights_array[[2, 3]], before[[2, 3]]), "OFF gates changed"
    assert rec['transferred'] == 0.0
    print("PASS AB: depression decreases active gates and leaves OFF unchanged")


def test_none_changes_no_weights():
    n = _l2e([2.0, 2.0, 0.2, 0.1])
    n.potential = 0.9 * n.threshold
    n.exc_trace = 5.0
    before = n._weights_array.copy()
    rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), "none")
    assert np.array_equal(n._weights_array, before)
    assert rec['plasticity_applied'] is False
    assert n.potential == n.resting_potential and n.exc_trace == 0.0   # identical reset
    print("PASS AB: 'none' changes no weights but performs the identical hard reset")


def test_unknown_mode_rejected():
    n = _l2e([2.0, 2.0, 0.2, 0.1])
    try:
        n.apply_competitive_reset(_mask(1, 1, 0, 0), "bogus")
    except ValueError:
        print("PASS AB: an unknown mode is rejected")
        return
    raise AssertionError("unknown mode was not rejected")


def test_identical_reset_across_modes():
    # All three modes perform the SAME unconditional hard reset regardless of the
    # loser rule (spec: hard reset is unconditional in every mode).
    posts = []
    for mode in ("redistribution", "depression", "none"):
        n = _l2e([2.0, 2.0, 0.2, 0.1])
        n.potential = 0.9 * n.threshold
        n.exc_trace = 3.0; n.inh_trace = 2.0
        rec = n.apply_competitive_reset(_mask(1, 1, 0, 0), mode)
        posts.append((rec['v_post'], n.exc_trace, n.inh_trace))
    assert all(p == (0.0, 0.0, 0.0) for p in posts), posts
    print("PASS AB: every mode ends at exact rest with cleared traces")


# ======================================================================
# Recruitment behaviour (engine level)
# ======================================================================
def _canonical_engine(seed, mode):
    # Distance weighting is off here so these tests isolate redistribution itself;
    # charge attenuation is covered separately above.
    return SimulationEngine(
        seed=seed, signed_spike_learning=True, structural_free_energy=True,
        structural_fe_eta_floor=0.02, distance_weighting=False,
        l2e_init_mode='legacy_wide', competitive_weight_update=mode, refractory=1,
        l2_charge_chunks=20, l2e_weight_cap_frac=1 / 3, pos_weight_floor=1,
        l2i_threshold_frac=1 / 3, l2e_lr_frac=0.02, ei_sat_mult=4.0)


def test_engine_conservation_per_event():
    # In a full run, every applied redistribution event conserves that neuron's total
    # positive feedforward mass (winner learning is separate: the winner is refractory
    # and skips the loser update, so its total is untouched by the reset).
    eng = _canonical_engine(1, "redistribution")
    names = list(PATTERNS.keys())
    max_err = 0.0
    checked = 0
    for r in range(8):
        eng.set_pattern(names[r % len(names)])
        for _ in range(40):
            before = [n.weights.copy() for n in eng.l2.excitatory_neurons]
            eng.step()
            for nid, rec in eng._reset_events:
                if rec['mode'] == 'redistribution' and rec['plasticity_applied'] \
                        and not rec['refractory_at_arrival']:
                    j = int(nid[3:])
                    err = abs(before[j].sum() - eng.l2.excitatory_neurons[j].weights.sum())
                    max_err = max(max_err, err)
                    checked += 1
    assert checked > 0, "no applied redistribution events observed"
    assert max_err < 1e-6, max_err
    print(f"PASS engine: {checked} redistribution events, max conservation err {max_err:.2e}")


def test_engine_runs_all_modes_without_drift():
    for mode in ("redistribution", "depression", "none"):
        eng = _canonical_engine(3, mode)
        names = list(PATTERNS.keys())
        for r in range(6):
            eng.set_pattern(names[r % len(names)])
            for _ in range(50):
                eng.step()
        # no NaN / shape drift; all L2E weights finite and within cap
        cap = eng.l2.excitatory_neurons[0].weight_cap
        for n in eng.l2.excitatory_neurons:
            w = n.weights
            assert np.isfinite(w).all() and (w <= cap + 1e-6).all(), (mode, w)
        print(f"PASS engine: mode '{mode}' runs 1800 steps without drift")


if __name__ == "__main__":
    test_refractory_winner_no_redistribution()
    test_nonrefractory_loser_redistributes()
    test_no_winner_vector_in_decision()
    test_larger_active_sum_larger_transfer()
    test_zero_active_sum_no_redistribution()
    test_membrane_charge_does_not_affect_redistribution()
    test_distance_factor_affects_p_match_like_delivery()
    test_conservation_active_down_off_up()
    test_bounds_never_crossed()
    test_insufficient_off_capacity_scales_down_active()
    test_no_off_capacity_no_change()
    test_capped_allocation_redistributes_remainder()
    test_incremental_over_repeated_losses()
    test_depression_leaves_off_unchanged()
    test_none_changes_no_weights()
    test_unknown_mode_rejected()
    test_identical_reset_across_modes()
    test_engine_conservation_per_event()
    test_engine_runs_all_modes_without_drift()
    print("ALL OFF-WEIGHT RECRUITMENT TESTS PASSED")
