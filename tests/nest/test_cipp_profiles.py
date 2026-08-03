"""Phase 2: the immutable physical-time profile.

Source: `prompts/Claude_NEST_CIPP_Semantic_Repair_Prompt.md` Phase 2.

These tests pin the properties Phase 2 exists to establish -- one semantic role per
parameter, physics independent of stimulus pacing, uniform reference conduction, and
ceiling-to-grid quantization -- and the equally important property that adding them did not
disturb `impulse_characterization`, whose recorded findings the repair is measured against.

Nothing here needs NEST; the profile is pure configuration. It lives under `tests/nest/`
because it describes the NEST backend's semantics.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from nest_backend.profiles import (
    CIPP_CONTINUOUS,
    IMPULSE_CHARACTERIZATION,
    PROFILE_CIPP_CONTINUOUS,
    PROFILE_IMPULSE,
    CoincidencePolicy,
    DelayPolicy,
    EngineProfile,
    PredictionCreditPolicy,
    QuantizationPolicy,
    get_profile,
)


# ------------------------------------------------------------------ quantization policy
@pytest.mark.parametrize("requested", [0.11, 0.14, 0.15, 0.16, 0.24, 0.25, 0.26,
                                       1.01, 1.049, 1.05, 1.06, 3.6224864935184176])
def test_ceiling_quantization_never_shortens_a_requested_delay(requested):
    """The property the docstrings always claimed. A shortened delay reorders causality.

    Instrumenting real network construction showed the historical `round()` shortening
    **88 of 254** edge delays on the canonical 3x6 graph -- not an edge case.
    """
    got = QuantizationPolicy.apply(QuantizationPolicy.CEIL, requested, 0.1)
    # EXACT. A contract that says "never shorter" may not be checked with a tolerance that
    # would accept a strictly smaller float.
    assert got >= requested, f"{requested} -> {got} is shorter than requested"
    assert got >= 0.1


@pytest.mark.parametrize("h", [0.1, 0.05, 0.01, 0.001])
def test_ceiling_quantization_lands_exactly_on_the_grid(h):
    for requested in (0.0001, 0.37, 1.0, 2.71828, 9.999):
        got = QuantizationPolicy.apply(QuantizationPolicy.CEIL, requested, h)
        steps = got / h
        assert abs(steps - round(steps)) < 1e-6, f"{got} is not a multiple of {h}"


@pytest.mark.parametrize("h", [0.1, 0.01, 0.001])
def test_a_delay_already_on_the_grid_is_not_pushed_up_a_step(h):
    """Float repr must not cost a whole grid step.

    `ceil(1.0/0.1)` is 10 in exact arithmetic but the division can land at 10.000000000001,
    and a naive ceiling would return 1.1. That is why the policy carries a tolerance.
    """
    for steps in (1, 2, 7, 13, 100):
        exact = round(steps * h, 10)
        assert QuantizationPolicy.apply(QuantizationPolicy.CEIL, exact, h) == pytest.approx(exact)


@pytest.mark.parametrize("h", [0.1, 0.05, 0.01, 0.001])
def test_ceiling_is_exact_at_float_boundaries(h):
    """`nextafter(k*h, -inf)`, `k*h`, `nextafter(k*h, +inf)` across tick magnitudes.

    Exact comparisons, no tolerance. The superseded implementation subtracted a fixed
    `1e-9` in tick units and rounded to ten decimals, so it returned values strictly below
    the request -- including for `k*h` itself at k=3 and k=7, where `3*0.1` is
    `0.30000000000000004` but `round(..., 10)` gives `0.3`.
    """
    for k in (1, 2, 3, 7, 13, 40):
        exact = k * h
        for value in (math.nextafter(exact, -math.inf), exact,
                      math.nextafter(exact, math.inf)):
            got = QuantizationPolicy.apply(QuantizationPolicy.CEIL, value, h)
            assert got >= value, f"h={h} k={k}: {value!r} -> {got!r} is shorter"
            assert got >= h
            steps = got / h
            assert abs(steps - round(steps)) < 1e-6, f"{got!r} is off the {h} grid"
            # Idempotent: quantizing a quantized value must be a fixed point.
            assert QuantizationPolicy.apply(QuantizationPolicy.CEIL, got, h) == got


@pytest.mark.parametrize("h", [0.1, 0.01])
def test_ceiling_is_monotonic(h):
    """A larger request may never yield a smaller delay."""
    previous = 0.0
    for step in range(1, 400):
        got = QuantizationPolicy.apply(QuantizationPolicy.CEIL, step * h / 3.0, h)
        assert got >= previous
        previous = got


def test_quantization_rejects_non_finite_and_negative_delays():
    for bad in (float("nan"), float("inf"), -float("inf"), -0.5):
        with pytest.raises(ValueError):
            QuantizationPolicy.apply(QuantizationPolicy.CEIL, bad, 0.1)
    for bad_h in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            QuantizationPolicy.apply(QuantizationPolicy.CEIL, 1.0, bad_h)


def test_round_policy_results_are_unchanged_for_the_impulse_profile():
    """Preserved behaviour, pinned by value so a future edit cannot drift it."""
    expected = {0.15: 0.1, 0.14: 0.1, 0.24: 0.2, 0.25: 0.2, 0.26: 0.3,
                1.049: 1.0, 1.05: 1.0, 1.06: 1.1, 3.6224864935184176: 3.6}
    for requested, want in expected.items():
        assert IMPULSE_CHARACTERIZATION.quantize(requested) == pytest.approx(want)


def test_quantization_never_returns_below_one_resolution_step():
    for requested in (0.0, 1e-12, 0.0001):
        assert QuantizationPolicy.apply(QuantizationPolicy.CEIL, requested, 0.1) >= 0.1


def test_round_policy_is_retained_only_for_the_impulse_profile():
    """The historical defect is preserved deliberately, and named as a defect.

    Switching `impulse_characterization` to CEIL would lengthen 88 of 254 delays and
    invalidate the recorded negative findings the prototype exists to preserve. So the
    policy is explicit per profile rather than globally "fixed".
    """
    assert IMPULSE_CHARACTERIZATION.quantization == QuantizationPolicy.ROUND
    assert CIPP_CONTINUOUS.quantization == QuantizationPolicy.CEIL
    # And the retained behaviour really is the shortening one.
    assert IMPULSE_CHARACTERIZATION.quantize(0.15) < 0.15
    assert CIPP_CONTINUOUS.quantize(0.0149) > 0.0149


def test_unknown_quantization_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown quantization policy"):
        QuantizationPolicy.apply("nearest-ish", 1.0, 0.1)


# --------------------------------------------------------------- profile immutability
@pytest.mark.parametrize("profile", [CIPP_CONTINUOUS, IMPULSE_CHARACTERIZATION])
def test_profiles_are_frozen(profile):
    """A run's physics is fixed when the profile is chosen, not tuned mid-experiment."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.resolution_h_ms = 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.delays.jitter_ms = 3.0


def test_profiles_validate_and_describe():
    for profile in (CIPP_CONTINUOUS, IMPULSE_CHARACTERIZATION):
        profile.validate()
        described = profile.describe()
        assert described["name"] == profile.name
        assert "loop_latency_ms_floor" in described
        # Serialisable: it has to survive into a manifest and a replay header.
        assert isinstance(described["delays"], dict)


def test_get_profile_rejects_an_unknown_name():
    assert get_profile(PROFILE_CIPP_CONTINUOUS) is CIPP_CONTINUOUS
    assert get_profile(PROFILE_IMPULSE) is IMPULSE_CHARACTERIZATION
    with pytest.raises(KeyError, match="unknown engine profile"):
        get_profile("cipp_continuous_v2")


# ----------------------------------------------- reference conduction: uniform, no jitter
def test_reference_profile_uses_uniform_delay_and_no_jitter():
    """Audit P0: geometry and random jitter had come to decide content.

    Phase 1 showed neither is needed -- latency after arrival varies by 1.8e-15 ms across a
    16x range of conduction delay, so drive alone orders the crossings.
    """
    delays = CIPP_CONTINUOUS.delays
    assert delays.jitter_ms == 0.0, "the reference profile must have no delay jitter"
    assert delays.geometry_scales_delay is False, (
        "geometry is a learning-rate multiplier only; it must never set delivery time"
    )
    # Equivalent feedforward hops share one delay; there is no per-edge value to vary.
    assert delays.rg_to_column_ms == delays.column_to_column_ff_ms


def test_impulse_profile_still_declares_its_geometry_dependence():
    """Preserved, and visible. The prototype's delays DO come from layout."""
    assert IMPULSE_CHARACTERIZATION.delays.geometry_scales_delay is True


def test_loop_latency_floor_matches_the_phase_1_measurement():
    """`d_ei + relay crossing + d_ie`. Phase 1 measured 0.0242 ms at d = 0.01."""
    delays = CIPP_CONTINUOUS.delays
    assert delays.loop_latency_ms() == pytest.approx(0.02)
    assert delays.loop_latency_ms(relay_crossing_ms=0.0042) == pytest.approx(0.0242)


def test_resolution_is_chosen_for_the_delay_envelope_not_crossing_precision():
    """h must at least admit the lateral delays the profile asks for.

    Phase 1 gate 3 showed crossing time is already converged at h=0.1, so h is chosen from
    the communication-delay envelope. The profile must therefore be self-consistent: no
    declared delay may sit below the resolution, since NEST has no sub-h delay.
    """
    CIPP_CONTINUOUS.validate()
    assert CIPP_CONTINUOUS.delays.column_e_to_i_ms >= CIPP_CONTINUOUS.resolution_h_ms

    too_coarse = dataclasses.replace(CIPP_CONTINUOUS, resolution_h_ms=0.1)
    with pytest.raises(ValueError, match="below the resolution"):
        too_coarse.validate()


# ------------------------------------------- physics independent of the stimulus schedule
def test_no_neuron_parameter_is_derived_from_the_presentation_interval():
    """THE Phase 2 contract, and the direct repair of audit P0.

    The impulse profile sets `tau_basal`, `tau_apical`, `tau_deposit_lock` and the I
    relay's `t_lockout` all equal to the presentation interval D. Nothing in this profile
    may carry a stimulus schedule at all -- there is no presentation field to leak.
    """
    assert CIPP_CONTINUOUS.neuron_physics_independent_of_pacing is True
    fields = {f.name for f in dataclasses.fields(EngineProfile)}
    assert not {"presentation", "presentation_ms", "d_ms", "period_ms"} & fields, (
        "the profile must carry no stimulus schedule"
    )
    for group in (CIPP_CONTINUOUS.coincidence, CIPP_CONTINUOUS.membrane,
                  CIPP_CONTINUOUS.causal_volley):
        names = {f.name for f in dataclasses.fields(group)}
        assert not any("present" in n or "volley_period" in n for n in names), names


def test_impulse_profile_declares_the_pacing_coupling_it_still_has():
    assert IMPULSE_CHARACTERIZATION.neuron_physics_independent_of_pacing is False


def test_coincidence_windows_must_span_a_MEASURED_causal_skew():
    """The validator requires a measured skew; it no longer invents one.

    An earlier version estimated skew as
    `abs(column_to_column_apical_ms - column_eor_to_c_basal_ms)` -- a subtraction of two
    TERMINAL edges, which is not the graph's causal skew at all. Basal and apical evidence
    traverse different paths through local E->Eor, feedforward, parent integration and the
    return apical arm.
    """
    coincidence = CIPP_CONTINUOUS.coincidence
    coincidence.validate(causal_skew_ms=1.0)          # comfortably inside a 5 ms window

    with pytest.raises(ValueError, match="measured causal arrival skew"):
        coincidence.validate(causal_skew_ms=7.0)      # wider than the window

    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and non-negative"):
            coincidence.validate(causal_skew_ms=bad)


def test_coincidence_window_must_exclude_cross_volley_pairing():
    """A window at least as long as the inter-volley gap can pair across presentations.

    That is the impulse profile's inclusive-boundary failure by construction, since its
    windows ARE the presentation interval.
    """
    coincidence = CIPP_CONTINUOUS.coincidence
    coincidence.validate(causal_skew_ms=1.0, min_inter_volley_interval_ms=15.0)
    with pytest.raises(ValueError, match="pair evidence across volleys"):
        coincidence.validate(causal_skew_ms=1.0, min_inter_volley_interval_ms=4.0)


def test_deposit_dead_time_is_a_dead_time_not_a_ttl():
    """It bounds how often a deposit may repeat, and is unrelated to input pacing."""
    coincidence = CIPP_CONTINUOUS.coincidence
    assert coincidence.deposit_dead_time_ms > 0
    assert coincidence.deposit_dead_time_ms < coincidence.basal_window_ms

    with pytest.raises(ValueError, match="finite and positive"):
        CoincidencePolicy(deposit_dead_time_ms=0.0).validate(causal_skew_ms=1.0)
    with pytest.raises(ValueError, match="window would be unreachable"):
        CoincidencePolicy(basal_window_ms=2.0, apical_window_ms=2.0,
                          deposit_dead_time_ms=3.0).validate(causal_skew_ms=1.0)


def test_provisional_physical_values_are_declared_provisional():
    """These are placeholders with constraints, not measured CIPP constants.

    Phase 1's convenient model defaults must not silently become CIPP physiology, so the
    coincidence group carries an explicit flag and the binding constraints are validated
    against measured graph and schedule quantities rather than assumed.
    """
    assert CIPP_CONTINUOUS.coincidence.provisional is True


# ------------------------------------------------------- causal volley and prediction
def test_causal_volley_rule_is_scoped_to_learning_membership_only():
    """It decides the participation sign and nothing else.

    It must also carry the lower bound the impulse profile lacks -- without
    `require_arrival_before_firing`, an afferent that has been sent but not delivered can
    be scored +1 (audit P1, contract 4; measured 250.0 -> 251.78).
    """
    rule = CIPP_CONTINUOUS.causal_volley
    assert rule.require_arrival_before_firing is True
    assert rule.separation_ms > 0


def test_prediction_credit_is_counted_and_survives_silence():
    """One confirmation, one credit, spent by the next eligible evidence volley."""
    credit = CIPP_CONTINUOUS.prediction
    credit.validate()
    assert credit.credits_per_confirmation == 1
    assert credit.consumed_per_volley == 1
    assert credit.silence_consumes_credit is False, (
        "silence must not spend a credit -- that is what makes it a credit and not a timer"
    )
    assert credit.wta_reset_affects_credit is False, (
        "WTA reset traffic must neither create nor consume prediction credits"
    )


def test_prediction_credit_policy_rejects_incoherent_settings():
    with pytest.raises(ValueError, match="capacity"):
        PredictionCreditPolicy(capacity=0).validate()
    with pytest.raises(ValueError, match=">= 1 credit"):
        PredictionCreditPolicy(consumed_per_volley=0).validate()


# ----------------------------------------------------------------------- role coverage
def test_every_parameter_group_required_by_phase_2_is_present():
    """The brief's minimum list, checked as a list rather than assumed."""
    profile = CIPP_CONTINUOUS
    assert profile.resolution_h_ms > 0                              # kernel resolution
    assert profile.delays.rg_to_column_ms > 0                       # ff delay by class
    assert profile.membrane.tau_syn_ex_ms > 0                       # current shape/tau
    assert profile.membrane.tau_m_ms > 0                            # membrane tau
    assert profile.membrane.threshold_gap_mv > 0                    # threshold and reset
    assert profile.membrane.t_ref_ms >= 0                           # refractory
    assert profile.delays.column_e_to_i_ms > 0                      # E->I
    assert profile.delays.column_i_to_e_ms > 0                      # I->E
    assert profile.coincidence.basal_window_ms > 0                  # C basal window
    assert profile.coincidence.apical_window_ms > 0                 # C apical window
    assert profile.coincidence.deposit_dead_time_ms > 0             # deposit dead time
    assert profile.causal_volley.separation_ms > 0                  # causal-volley rule
    assert profile.prediction.capacity >= 1                         # credit capacity


# --------------------------------------------- new Phase 2 fields (audit corrections)
def test_relay_policy_is_its_own_group_with_its_own_validation():
    """The WTA relay lockout is not the C soma's deposit dead time."""
    relay = CIPP_CONTINUOUS.relay
    relay.validate(CIPP_CONTINUOUS.delays)
    from nest_backend.profiles import RelayPolicy
    relay_fields = {field.name for field in dataclasses.fields(RelayPolicy)}
    coincidence_fields = {field.name for field in dataclasses.fields(CoincidencePolicy)}
    assert "wta_lockout_ms" in relay_fields
    assert "wta_lockout_ms" not in coincidence_fields
    assert "deposit_dead_time_ms" in coincidence_fields
    assert "deposit_dead_time_ms" not in relay_fields
    # Non-aliasing, proved by moving one and reading the other. Equal numeric defaults are
    # allowed and establish nothing either way; what matters is that the two values live on
    # separate policy objects, so changing one cannot move the other.
    moved = dataclasses.replace(
        CIPP_CONTINUOUS, relay=RelayPolicy(wta_lockout_ms=relay.wta_lockout_ms + 3.0))
    assert moved.relay.wta_lockout_ms == relay.wta_lockout_ms + 3.0
    assert (moved.coincidence.deposit_dead_time_ms
            == CIPP_CONTINUOUS.coincidence.deposit_dead_time_ms), (
        "changing the relay lockout must not move the C deposit dead time"
    )
    # The binding constraint: a relay must not re-fire inside the loop it is closing.
    assert relay.wta_lockout_ms > CIPP_CONTINUOUS.delays.loop_latency_ms()
    assert relay.provisional is True and relay.provenance

    with pytest.raises(ValueError, match="finite and positive"):
        RelayPolicy(wta_lockout_ms=0.0).validate(CIPP_CONTINUOUS.delays)
    with pytest.raises(ValueError, match="re-fire inside its own loop"):
        RelayPolicy(wta_lockout_ms=0.001).validate(CIPP_CONTINUOUS.delays)


def test_c_refractory_is_its_own_field_with_a_stated_rationale():
    """Zero is the C contract's justified value, not a borrowed competitor refractory."""
    coincidence = CIPP_CONTINUOUS.coincidence
    assert coincidence.c_refractory_ms == 0.0
    assert coincidence.c_refractory_rationale
    assert coincidence.c_refractory_ms != CIPP_CONTINUOUS.membrane.t_ref_ms

    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="c_refractory_ms must be finite"):
            CoincidencePolicy(c_refractory_ms=bad).validate(causal_skew_ms=1.0)


def test_membrane_and_causal_volley_defaults_are_marked_provisional():
    """Not only the coincidence window. Phase 1's model defaults are not CIPP physiology."""
    for group in (CIPP_CONTINUOUS.membrane, CIPP_CONTINUOUS.causal_volley,
                  CIPP_CONTINUOUS.coincidence, CIPP_CONTINUOUS.relay):
        assert getattr(group, "provisional", None) is True, group
    assert "NOT mapped to the repository's charge units" in CIPP_CONTINUOUS.membrane.provenance
    assert CIPP_CONTINUOUS.causal_volley.provenance


def test_schedule_validation_rejects_non_finite_and_non_positive_intervals():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CIPP_CONTINUOUS.validate_schedule(within_volley_spread_ms=0.0,
                                              min_inter_volley_interval_ms=bad)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CIPP_CONTINUOUS.validate_schedule(within_volley_spread_ms=bad,
                                              min_inter_volley_interval_ms=15.0)


def test_implementation_disposition_is_serialised_and_honest():
    for profile in (CIPP_CONTINUOUS, IMPULSE_CHARACTERIZATION):
        disposition = profile.describe()["disposition"]
        assert disposition["mechanical_profile_promoted"] is False
        assert disposition["summary"]
        if disposition["scaffold"]:
            assert disposition["target_membrane_model"] != disposition["active_competitor_model"]
            assert disposition["deferred_to_phase_3_or_later"]
        else:
            assert disposition["target_membrane_model"] == disposition["active_competitor_model"]

def test_a_scaffold_may_not_claim_its_target_is_active():
    from nest_backend.profiles import ImplementationDisposition
    with pytest.raises(ValueError, match="must not report the target membrane model"):
        ImplementationDisposition(scaffold=True,
                                  target_membrane_model="event_accumulator",
                                  active_competitor_model="event_accumulator").validate()


# ------------------------------------------- Codex addendum: bare-path structural checks
def test_bare_validate_rejects_a_non_finite_c_refractory():
    """No causal skew supplied, so the deferred window check does not run -- but the
    structural fields must still be checked.

    Previously the WHOLE of `CoincidencePolicy` was skipped unless a caller passed a
    measured skew, so a profile carrying `c_refractory_ms = NaN` validated cleanly.
    """
    for bad in (float("nan"), float("inf"), -1.0):
        broken = dataclasses.replace(
            CIPP_CONTINUOUS, coincidence=CoincidencePolicy(c_refractory_ms=bad))
        with pytest.raises(ValueError, match="c_refractory_ms must be finite"):
            broken.validate()          # bare path: no causal_skew_ms


@pytest.mark.parametrize("field", ["basal_window_ms", "apical_window_ms",
                                   "deposit_dead_time_ms"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bare_validate_rejects_non_finite_or_non_positive_windows(field, bad):
    broken = dataclasses.replace(
        CIPP_CONTINUOUS, coincidence=CoincidencePolicy(**{field: bad}))
    with pytest.raises(ValueError, match="must be finite and positive"):
        broken.validate()


def test_bare_validate_rejects_a_dead_time_that_swallows_the_window():
    """The dead-time/window relation is structural and needs no graph to check.

    A dead time at least as long as the eligibility window makes the window unreachable.
    """
    broken = dataclasses.replace(
        CIPP_CONTINUOUS,
        coincidence=CoincidencePolicy(basal_window_ms=2.0, apical_window_ms=2.0,
                                      deposit_dead_time_ms=3.0))
    with pytest.raises(ValueError, match="window would be unreachable"):
        broken.validate()


def test_bare_validate_still_defers_the_causal_window_check():
    """The deferred check must stay deferred -- structural coverage is not a substitute.

    A window far narrower than a real skew still passes the BARE path, because Phase 2
    cannot measure the parent's processing latency. It fails only when a skew is supplied.
    """
    narrow = dataclasses.replace(
        CIPP_CONTINUOUS,
        coincidence=CoincidencePolicy(basal_window_ms=0.2, apical_window_ms=0.2,
                                      deposit_dead_time_ms=0.1))
    narrow.validate()                                    # bare path: structurally fine
    with pytest.raises(ValueError, match="measured causal arrival skew"):
        narrow.coincidence.validate(causal_skew_ms=1.0)  # graph-aware path rejects it


# ------------------------------- Codex addendum: impulse membrane / disposition honesty
def test_impulse_membrane_group_is_explicitly_not_applicable():
    """The prototype runs an accumulator in charge units; it has no mV/pF membrane.

    `describe()["membrane"]["model"]` previously said `iaf_psc_exp_ps` while the same
    payload's disposition said `event_accumulator` -- a contradiction in every impulse
    manifest.
    """
    membrane = IMPULSE_CHARACTERIZATION.describe()["membrane"]
    assert membrane["applicable"] is False
    assert "model" not in membrane, (
        "a non-applicable membrane group must not name a model at all"
    )
    for key in ("v_threshold_mv", "c_m_pf", "tau_m_ms"):
        assert key not in membrane, f"{key} describes nothing this profile runs"
    assert membrane["reason"]


def test_described_membrane_and_disposition_never_contradict_each_other():
    for profile in (CIPP_CONTINUOUS, IMPULSE_CHARACTERIZATION):
        described = profile.describe()
        membrane, disposition = described["membrane"], described["disposition"]
        if disposition["scaffold"]:
            assert membrane["applicable"] is True
            assert membrane["model"] == disposition["target_membrane_model"]
            assert membrane["model"] != disposition["active_competitor_model"]
        else:
            assert membrane["applicable"] is False
            assert (disposition["target_membrane_model"]
                    == disposition["active_competitor_model"])


def test_membrane_disposition_consistency_is_enforced_not_merely_arranged():
    """A profile that contradicts itself must fail to validate."""
    from nest_backend.profiles import MembranePolicy

    # A scaffold whose membrane group is switched off.
    with pytest.raises(ValueError, match="membrane group must be applicable"):
        dataclasses.replace(CIPP_CONTINUOUS,
                            membrane=MembranePolicy(applicable=False)).validate()

    # A scaffold whose disposition names a different target than its membrane group.
    with pytest.raises(ValueError, match="contradicts membrane.model"):
        dataclasses.replace(
            CIPP_CONTINUOUS,
            disposition=dataclasses.replace(CIPP_CONTINUOUS.disposition,
                                            target_membrane_model="iaf_psc_alpha_ps"),
        ).validate()

    # A non-scaffold carrying an applicable continuous membrane group.
    with pytest.raises(ValueError, match="has no continuous membrane group"):
        dataclasses.replace(IMPULSE_CHARACTERIZATION,
                            membrane=MembranePolicy(applicable=True)).validate()
