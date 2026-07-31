"""Phase 3 suite-level behaviour (prompt sections 9 Phase 3, 10, 11).

These assert the INVARIANTS and the direction of the characterisation results, not exact
spike times. Where a reference prediction fails to reproduce, the test pins the failure so
it cannot silently start or stop happening.
"""
from __future__ import annotations

import pytest

from nest_backend.engine import Stimulus, run_case
from nest_backend.topology import E_THRESHOLD, Timescales

CENTER = Stimulus({(1, 1): "row 1"})


@pytest.fixture(scope="module")
def center_run():
    return run_case("center", CENTER, n_presentations=8)


# --------------------------------------------------------------- determinism


def test_runs_are_deterministic_for_a_fixed_seed():
    a = run_case("det_a", CENTER, seed=3, n_presentations=6)
    b = run_case("det_b", CENTER, seed=3, n_presentations=6)
    assert a.metrics["spikes"] == b.metrics["spikes"]
    assert a.metrics["winner_multiplicity_overall"] == b.metrics["winner_multiplicity_overall"]


# ------------------------------------------------------------------ locality


def test_center_patch_isolation(center_run):
    assert set(center_run.metrics["by_column"]) <= {"L1c11", "L2c00"}


def test_two_patch_locality():
    result = run_case("two", Stimulus({(0, 0): "row 1", (2, 2): "col 1"}), n_presentations=6)
    active = set(result.metrics["by_column"])
    assert {"L1c00", "L1c22"} <= active
    assert not (active & {"L1c01", "L1c10", "L1c11", "L1c12", "L1c21"})


def test_all_nine_patches_activate_every_l1_column():
    result = run_case("all9", Stimulus({(r, c): "row 1" for r in range(3) for c in range(3)}),
                      n_presentations=6)
    active = set(result.metrics["by_column"])
    assert len([c for c in active if c.startswith("L1")]) == 9


def test_pattern_switch_does_not_rebuild_the_network():
    """The generators are rewritten; nodes, connections and state all persist."""
    result = run_case("switch", CENTER, switch_to=Stimulus({(1, 1): "col 1"}),
                      n_presentations=8)
    assert result.metrics["spike_count"] > 0
    assert result.manifest["counts"]["connected_edges"] == 1052


# ------------------------------------------------------------- the WTA result


def test_single_winner_wta_is_lost_at_the_default_separation(center_run):
    """THE HEADLINE RESULT. Recorded as a native-semantic difference, not a defect.

    The reference engine guarantees exactly one winner per column per presentation. Under
    NEST delivery with the default `S / L_wta = 10`, it does not.
    """
    overall = center_run.metrics["winner_multiplicity_overall"]
    assert overall["mean"] > 1.0, (
        "if this ever passes with mean == 1.0 the WTA finding has changed and the report "
        "must be revisited"
    )
    assert overall["max"] > 1


def test_eor_input_multiplicity_corroborates_wta_loss(center_run):
    """`Eor` is contractually a one-afferent relay; above 1 proves multiple winners."""
    eor = center_run.metrics["eor_input_multiplicity"]
    fired = [v["last_input_multiplicity"] for v in eor.values() if v["n_spikes"]]
    assert fired, "at least one Eor must have relayed"
    assert max(fired) > 1.0


def test_disabling_dispersion_maximises_winner_multiplicity():
    """Section 5.2: fast inhibition alone recovers nothing without arrival ordering."""
    flat = run_case("flat", CENTER, n_presentations=6, dispersion_enabled=False,
                    timescales=Timescales(presentation=40.0))
    assert flat.metrics["winner_multiplicity_overall"]["mean"] == pytest.approx(8.0), (
        "with a simultaneous volley every competitor in the bank should fire"
    )


def test_winner_multiplicity_falls_monotonically_with_the_separation_ratio():
    """Section 5.5 hypothesis, tested as a trend rather than at one lucky point."""
    means = []
    for spread in (0.2, 2.0, 12.0):
        ts = Timescales(h=0.1, base_ff=1.0, spread=spread, presentation=40.0)
        result = run_case(f"sweep{spread}", CENTER, timescales=ts, n_presentations=6)
        means.append(result.metrics["winner_multiplicity_overall"]["mean"])
    assert means[0] > means[1] > means[2], f"expected a falling trend, got {means}"


def test_outcomes_are_invariant_to_h_at_fixed_ratio():
    """Only the RATIO is physical. A dependence on absolute `h` would be an artifact."""
    results = []
    for h in (0.05, 0.1, 0.2):
        ts = Timescales(h=h, base_ff=10 * h, spread=20 * h, presentation=200 * h)
        result = run_case(f"res{h}", CENTER, timescales=ts, n_presentations=6)
        results.append(result.metrics["winner_multiplicity_overall"]["mean"])
    assert len(set(results)) == 1, (
        f"outcomes must not depend on absolute h at fixed ratio, got {results}"
    )


# ----------------------------------------------------------------- feedback


def test_feedback_control_is_inert_while_c_cannot_fire():
    """Honest negative: at the initial theta/4 basal weight, C never fires.

    So disabling the confirmation pathway changes nothing. This pins the reason the plain
    feedback control (case 6) is uninformative, rather than letting it read as "feedback
    does nothing".
    """
    on = run_case("fb_on", CENTER, n_presentations=6, feedback_enabled=True)
    off = run_case("fb_off", CENTER, n_presentations=6, feedback_enabled=False)
    assert on.metrics["spike_count"] == off.metrics["spike_count"]
    assert all(v["spikes"] == 0 for v in on.metrics["coincidence"].values())


def test_feedback_changes_behaviour_once_c_is_frozen_at_maturity():
    """With a mature C the confirmation pathway is live and measurably suppressive.

    This is the regression guard for a real translation defect: routing `C -> I` through
    the same rate-limited port as the lateral `E -> I` volley let the once-per-window WTA
    lockout swallow every confirmation, making feedback-on and feedback-off bit-identical.
    The reference requires the confirmation NOT to be swallowed
    (`Current_Implementation_Methodology_Equations.md` section 7).
    """
    on = run_case("fb_on_mature", CENTER, n_presentations=6, feedback_enabled=True,
                  c_basal_weight=E_THRESHOLD)
    off = run_case("fb_off_mature", CENTER, n_presentations=6, feedback_enabled=False,
                   c_basal_weight=E_THRESHOLD)
    assert any(v["spikes"] > 0 for v in on.metrics["coincidence"].values()), (
        "a C frozen at theta must fire on a single basal deposit"
    )
    assert on.metrics["spike_count"] < off.metrics["spike_count"], (
        "the top-down confirmation reset must suppress redundant re-fire"
    )


def test_confirmation_volley_is_not_swallowed_by_the_wta_lockout():
    """Directly assert the mechanism, not just its downstream effect."""
    on = run_case("confirm_mature", CENTER, n_presentations=6, feedback_enabled=True,
                  c_basal_weight=E_THRESHOLD)
    relays = on.metrics["resets"]["i_relay"]
    assert any(v["spikes"] > 0 for v in relays.values())


# ---------------------------------------------------------------- coincidence


def test_dormant_top_column_c_never_deposits_or_fires(center_run):
    dormant = center_run.metrics["coincidence"]["L2c00C"]
    assert dormant["has_parent"] is False
    assert dormant["apical_steps"] == 0, "the top column has no parent, so no apical input"
    assert dormant["deposits"] == 0
    assert dormant["spikes"] == 0


def test_the_carried_eligibility_path_is_live_in_the_full_topology(center_run):
    """A dead carry must be visible here, not inferred from a silent cell."""
    deposits = sum(v["deposits"] for v in center_run.metrics["coincidence"].values())
    basal_first = sum(v["deposit_basal_first"]
                      for v in center_run.metrics["coincidence"].values())
    assert deposits > 0, "the centre column's C must receive coincidences"
    assert basal_first > 0, "the basal-then-apical carry path must actually be exercised"


def test_c_cannot_fire_at_the_initial_basal_weight(center_run):
    """One-shot recognition needs `w_basal >= theta`; the initializer gives theta/4."""
    assert all(v["spikes"] == 0 for v in center_run.metrics["coincidence"].values())


# ------------------------------------------------------------------ cadence


def test_feedback_cadence_law_does_not_reproduce():
    """Honest negative result, pinned so it cannot change silently.

    The reference predicts strict `1010` alternation when the presentation period equals
    the graph's own loop latency `L`, and no suppression at `L + 1` hop. Neither holds
    here: the mechanism that produces exact alternation is the discarding of the NEXT
    boundary's already-frozen drive packet, and NEST has no frozen packet -- a reset at
    time T erases only what accumulated before T, so suppression is partial rather than
    total. See docs/NEST_EVENT_DRIVEN_3X3_REPORT.md.
    """
    from nest_backend.topology import NestTiledNetwork

    base = Timescales()
    probe = NestTiledNetwork(seed=1, timescales=base)
    loop = probe.feedback_loop_latency_ms()

    def pattern(period_ms):
        period = round(round(period_ms / base.h) * base.h, 10)
        ts = Timescales(h=base.h, base_ff=base.base_ff, spread=base.spread,
                        presentation=period)
        result = run_case("cadence", CENTER, timescales=ts, n_presentations=12,
                          settle_windows=1.0, c_basal_weight=E_THRESHOLD)
        return result.metrics["firing_pattern"].get("L1c11", "")

    at_l = pattern(loop)
    assert at_l, "the driven column must fire at all"
    assert "1010" not in at_l, (
        f"strict alternation unexpectedly appeared ({at_l}); the report's negative cadence "
        "result would need revisiting"
    )
