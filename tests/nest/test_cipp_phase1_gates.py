"""Regression cover for the Phase 1 continuous-competitor feasibility gates.

Source: `docs/CIPP_PHASE1_FEASIBILITY_REPORT.md`, produced by
`experiments/nest_cipp_phase1_feasibility.py`.

Why this file exists: the feasibility conclusion -- that `iaf_psc_exp_ps` recovers the
CIPP membrane-latency race under uniform delay -- is the evidence Phases 2-8 are built on.
Left in a script that nobody runs, it would rot silently while the repair proceeded on top
of it. These tests re-derive the conclusion from the same code the report was generated
from, so a change that breaks the premise fails the suite rather than the argument.

They exercise the gate functions directly rather than re-implementing them, so the tested
claim and the reported claim cannot drift apart.
"""
from __future__ import annotations

import pytest

from experiments.nest_cipp_phase1_feasibility import (
    CANDIDATE,
    CIRCUIT,
    crossing_times,
    gate_delay_scaling_preserves_winner,
    gate_drive_orders_winner,
    gate_exact_tie_is_reported,
    gate_multi_afferent_decomposition,
    gate_resolution_convergence,
    gate_single_winner_within_margin,
    wta_outcome,
)

# Coarser than the report's 0.01 so the suite stays quick. Gate 3 is the one that proves
# the choice does not matter, and it sweeps h itself.
H = 0.05


@pytest.fixture
def nest_kernel(fresh_kernel):
    """The gate functions call `nest.ResetKernel()` themselves; this just imports NEST."""
    fresh_kernel(H)
    import nest
    nest.set_verbosity("M_ERROR")
    return nest


def test_candidate_is_a_precise_model_with_continuous_current(nest_kernel):
    """The premise: the candidate must have both properties the impulse model lacks.

    A precise (`_ps`) model resolves the crossing off-grid instead of on the `h` grid, and
    a `psc_exp` synapse gives a continuous postsynaptic current instead of an
    instantaneous jump. Losing either would invalidate every gate below.
    """
    assert CANDIDATE.endswith("_ps"), f"{CANDIDATE} is not a precise-spike model"
    defaults = nest_kernel.GetDefaults(CANDIDATE)
    assert "tau_syn_ex" in defaults, "candidate must have a continuous synaptic current"
    assert CIRCUIT["neuron"]["tau_syn_ex"] > 0.0


def test_gate1_stronger_total_drive_fires_first(nest_kernel):
    """Under uniform arrival, drive alone orders the competitors."""
    result = gate_drive_orders_winner(nest_kernel, H)
    assert result["passed"], result["measurements"]
    for row in result["measurements"]:
        assert row["strong"] < row["weak"], row


def test_gate2_scaling_all_delays_together_does_not_reverse_the_winner(nest_kernel):
    """THE result: latency after arrival is a property of drive, not of conduction delay.

    If this regresses, the profile has gone back to deciding by delay order.
    """
    result = gate_delay_scaling_preserves_winner(nest_kernel, H)
    assert result["passed"], result["measurements"]

    # Raw floats, so this is a tolerance claim rather than set equality: the measured
    # spread across a 16x delay range is ~1.8e-15 ms, the last float ULPs.
    assert result["latency_spread_ms"] < 1e-12, (
        f"latency after arrival varied by {result['latency_spread_ms']} ms with "
        f"conduction delay; it must be a property of drive alone"
    )


def test_gate3_crossing_time_is_stable_under_resolution(nest_kernel):
    """Reducing `h` must not move the crossing or change the winner.

    Asserted at 1e-9 ms rather than as exact equality: the crossing is found by
    regula-falsi root finding over exactly integrated subthreshold dynamics, and the
    finest resolutions differ in the last float ULPs (~3e-15 ms measured). That is
    convergence, not disagreement, but it is not bit-identity and must not be asserted as
    such.
    """
    result = gate_resolution_convergence(nest_kernel)
    assert result["passed"], result["measurements"]
    assert result["identity_stable"], "winner identity changed with resolution"
    assert result["spike_time_spread_ms"] < 1e-9, (
        f"crossing time moved by {result['spike_time_spread_ms']} ms across h"
    )
    # And the stored figure must be the RAW difference, not a rounded stand-in. Rounding
    # to 9 decimals previously reported this 3.55e-15 ms spread as exactly 0.0, which
    # overstated convergence as bit-identity.
    strong = [row["strong"] for row in result["measurements"]]
    assert result["spike_time_spread_ms"] == max(strong) - min(strong), (
        "stored spread must equal the difference recomputed from the measurements"
    )
    assert result["spike_time_spread_ms"] > 0.0, (
        "the spread is small but nonzero; storing exactly 0.0 means it was rounded"
    )


def test_gate4_positive_margin_commits_exactly_one_winner(nest_kernel):
    """Within the measured envelope, one winner. Outside it, an honest unresolved race."""
    result = gate_single_winner_within_margin(nest_kernel, H)
    assert result["passed"], result["measurements"]
    assert result["positive_margin_cases"] >= 1, "no case cleared the loop latency"

    for row in result["measurements"]:
        if row["positive_margin"]:
            assert row["n_committed"] == 1 and row["winner"] == "strong", row
        else:
            # Must be surfaced, never quietly counted as a win.
            assert row["contract_met"] is None, row
            assert row["classification"] in ("unresolved_race", "exact_tie"), row


def test_gate5_exact_tie_names_no_winner(nest_kernel):
    """REGRESSION: an exact tie must not be resolved by node index.

    `wta_outcome` originally took `min((time, index))`, so two competitors crossing at an
    identical time silently reported competitor 0 -- a winner manufactured from creation
    order, the same class of error as manufacturing one from delay jitter.
    """
    result = gate_exact_tie_is_reported(nest_kernel, H)
    assert result["passed"], result
    assert result["tie_detected"], "equal drive must cross at an identical time"
    assert result["winner"] is None, (
        f"an exact tie must name no winner, got {result['winner']}"
    )
    assert result["classification"] == "exact_tie"


def test_gate6_decomposition_and_creation_order_do_not_change_the_decision(nest_kernel):
    """The same total drive decides identically however it is summed or ordered.

    Eight versus six synchronous 1000 pA afferents must give what one aggregate 8000/6000
    pA event gives, and building the competitors in the opposite order must change
    nothing. Compared as raw floats -- rounding here would manufacture the agreement.
    """
    result = gate_multi_afferent_decomposition(nest_kernel, H)
    assert result["passed"], result
    assert result["aggregate_equals_decomposed"], (
        f"aggregate {result['aggregate']} != decomposed {result['decomposed']}"
    )
    assert result["creation_order_invariant"], (
        f"creation order changed the outcome: {result['decomposed']} vs "
        f"{result['decomposed_reverse_creation_order']}"
    )
    # The permutation must actually permute. Creating both competitors in one
    # `Create(model, 2)` call left the strong cell on the lower GID either way, so the
    # flag reversed only connection insertion order and this gate proved less than it
    # appeared to.
    forward, reversed_ = result["decomposed"], result["decomposed_reverse_creation_order"]
    assert forward["gid_of_strong"] != reversed_["gid_of_strong"], (
        "reverse_creation_order did not change which GID holds the strong competitor"
    )
    assert forward["gid_of_strong"] == reversed_["gid_of_weak"], (
        "the two roles must actually swap GIDs"
    )


def test_winner_is_none_whenever_more_than_one_competitor_commits(nest_kernel):
    """The winner field means COMMITTED, not merely first across.

    A race the inhibitory loop failed to close has no winner, however the spike times fell.
    """
    outcome = wta_outcome(nest_kernel, 8000.0, 7900.0, h=H)
    assert outcome["n_committed"] == 2, "setup: this pair must race"
    assert outcome["winner"] is None, (
        f"two committed competitors is not a win; got winner={outcome['winner']}"
    )
    assert outcome["classification"] in ("unresolved_race", "exact_tie")


def test_spike_times_are_reported_unrounded(nest_kernel):
    """Raw floats, so a comparison cannot be flattered by rounding.

    Six-decimal rounding made two crossings differing at 1e-15 compare equal, which is how
    "identical across h" was originally overstated as bit-identity.
    """
    times = crossing_times(nest_kernel, 8000.0, 6000.0, h=H)
    assert times["strong"] != round(times["strong"], 6), (
        "crossing times must be raw; this one survived a six-decimal round trip unchanged, "
        "which suggests rounding is still being applied somewhere"
    )


def test_spike_uses_no_geometry_and_no_jitter(nest_kernel):
    """The feasibility claim is void if either was quietly reintroduced."""
    assert CIRCUIT["source_delay"] > 0.0
    # A single uniform source delay: there is no per-edge delay field to vary at all.
    assert isinstance(CIRCUIT["source_delay"], float)
    assert CIRCUIT["d_ei"] > 0.0 and CIRCUIT["d_ie"] > 0.0, (
        "the E->I->E loop must be explicitly nonzero -- NEST has no zero-delay synapse"
    )
