"""Validation tests for the hybrid event-resolved engine (see
``prompts/Claude_Hybrid_Engine_Validation_Prompt.md`` and
``docs/ENGINE_VALIDATION_REPORT.md``).

These are CHARACTERIZATION tests: they compare the production analytic engine against a
deliberately simple, independent RK4 / priority-queue reference (defined in
``experiments/engine_validation.py``), and they pin the current -- possibly approximate --
behaviour by its declared rule. Names reflect that: the WTA same-time result is
"order-sensitive by explicit rule", NOT "simultaneous correctness"; the relay result is
"one-emission-per-boundary coalescing", not a bug.

Nothing here tunes a learning rule, threshold, weight, delay, or tie-break.
"""

import math

import numpy as np
import pytest

from snn.neurons import ConductanceLIFNeuron, dual_fe, E_THRESHOLD
from backend.simulation import BoundaryEventScheduler
from experiments import engine_validation as ev

THETA = E_THRESHOLD


# =============================================================== Phase 2: RK4 vs analytic
def test_rk4_state_converges_toward_analytic_segment():
    """RK4 final state converges to the production analytic segment; the clean (non-fp-
    floor) cases show ~4th-order convergence, and every case is monotone-decreasing."""
    rows, conv = ev.run_phase2()
    assert conv['all_finite']
    cases = conv['cases']
    # every case is (weakly) monotone across the decreasing-dt sweep
    for name, c in cases.items():
        assert c['monotone'], f'{name} not monotone: {c}'
    # cases whose error stays above the fp floor must exhibit ~4th-order RK4 convergence
    clean = ('conductance_inhibited', 'decay_only', 'strong_inhibition_reversal',
             'subthreshold_leaky', 'subthreshold_multi', 'no_crossing_asymptote')
    for name in clean:
        order = cases[name]['observed_order']
        assert order is not None and 3.5 <= order <= 4.5, (name, order)


def test_rk4_finest_state_matches_analytic_tightly():
    rows, _ = ev.run_phase2()
    finest = {}
    for r in rows:
        if r['state_error'] is None:
            continue
        key = r['case']
        finest.setdefault(key, (r['reference_dt'], r['state_error']))
        if r['reference_dt'] < finest[key][0]:
            finest[key] = (r['reference_dt'], r['state_error'])
    # at the finest dt the RK4 state error is negligible on the theta~1000 scale
    for key, (_, err) in finest.items():
        assert err < 1e-6, (key, err)


def test_numeric_spike_time_converges():
    """Crossing-time error (RK4 + linear interpolation) converges monotonically toward the
    analytic crossing time as dt shrinks; the pure integrator is exact."""
    rows, conv = ev.run_phase2()
    cross_rows = [r for r in rows if r['spike_time_error'] is not None]
    assert cross_rows
    by_case = {}
    for r in cross_rows:
        by_case.setdefault(r['case'], []).append((r['reference_dt'], r['spike_time_error']))
    for case, seq in by_case.items():
        seq = sorted(seq, key=lambda x: -x[0])          # coarse -> fine
        errs = [e for _, e in seq]
        assert all(math.isfinite(e) for e in errs), case
        # weak monotone decrease (allow an fp floor)
        assert all(errs[i + 1] <= errs[i] + 1e-9 or errs[i] < 1e-6
                   for i in range(len(errs) - 1)), (case, errs)
        assert errs[-1] < 1e-6, (case, errs[-1])


def test_integrator_crossing_is_exact():
    # a pure-integrator crossing is a linear trajectory: linear interpolation is exact.
    rows, _ = ev.run_phase2()
    integ = [r for r in rows if r['case'] == 'integrator_no_leak_crossing']
    assert integ
    assert all((r['spike_time_error'] or 0.0) < 1e-9 for r in integ)


# ================================================= Phase 7 (weaker): partition invariance
def test_segment_partition_invariance():
    """Splitting one uninterrupted analytic segment into K pieces (no intervening event)
    reproduces the single-advance state. This is NOT network boundary refinement."""
    res = ev.run_segment_partition_invariance()
    assert res['invariant']
    assert res['max_abs_diff'] < 1e-6
    for row in res['rows']:
        assert row['abs_diff'] < 1e-6


# ================================================================ Phase 3: scheduler causality
def test_scheduler_causality_invariants():
    r = ev.run_scheduler_causality()
    assert r['earliest_wins_regardless_of_order']
    assert r['reset_invalidates_prediction']
    assert r['logical_time_monotone']


def test_causal_invalidation_after_reset_direct():
    # a hard reset that wiped V + drive invalidates the cell's previously predicted crossing
    n = ConductanceLIFNeuron('m', 't', threshold=THETA, leak_rate=0.0)
    n.V = 0.0
    n.gather_exc(3000.0)
    n.freeze_drive()
    sched = BoundaryEventScheduler([n], 1e-12)
    cell, tau = sched.next_event()
    assert cell is n and math.isfinite(tau)
    n.hard_reset(tau)
    assert sched.next_event() == (None, None)


def test_boundary_endpoint_ownership():
    """A crossing at EXACTLY tau == 1.0 is owned by (fires within) the current boundary;
    a crossing that would land just beyond 1.0 does not fire this boundary. One
    unambiguous ownership rule."""
    r = ev.run_scheduler_causality()
    assert r['endpoint_owned_by_current_boundary']
    assert r['just_beyond_endpoint_not_fired']


def test_tie_tolerance_boundary_characterized():
    # differences just inside vs just outside the tie tolerance are characterized:
    # inside -> a recorded latency_tie broken by node order; outside -> no tie.
    a = ev._make_membrane(0.0, 0.0, 0.0, 2000.0)                 # tau = 0.5
    b = ev._make_membrane(0.0, 0.0, 0.0, 2000.0 / 1.004)         # tau ~= 0.502
    within = BoundaryEventScheduler([a, b], 1e-2)
    cell, _ = within.next_event()
    assert cell is a and len(within.ties) == 1
    a2 = ev._make_membrane(0.0, 0.0, 0.0, 2000.0)
    b2 = ev._make_membrane(0.0, 0.0, 0.0, 2000.0 / 1.004)
    outside = BoundaryEventScheduler([a2, b2], 1e-6)
    cell2, _ = outside.next_event()
    assert cell2 is a2 and outside.ties == []


# ============================================================= Phase 5: event conservation
def test_event_conservation_on_tiny_graph():
    e = ev.build_chain_engine(seed=1, w0=1500.0, w1=1200.0)
    trace, ledger = ev.collect_event_trace(e, boundaries=6)
    audit = ev.audit_conservation(e, trace)
    assert audit['no_nonfinite']
    assert audit['every_reset_has_source']
    assert audit['no_duplicate_reset']
    assert ledger['boundary_monotonic']            # integer logical time never moves back
    assert ledger['monotonic']                     # every sub-boundary tau finite & in [0,1]
    assert audit['total_spikes'] > 0


def test_event_timestamp_monotonicity():
    e = ev.build_chain_engine(seed=1, w0=1500.0, w1=1200.0)
    trace, ledger = ev.collect_event_trace(e, boundaries=8)
    boundaries = [rec['boundary'] for rec in trace]
    assert boundaries == sorted(boundaries)
    assert boundaries == list(range(1, 9))


# ================================================= Phase 4: same-time concurrency / order
def test_independent_simultaneous_spikes_commute():
    """Two independent latency cells crossing at the exact same tau: the emitted-spike SET
    and reset-target SET are invariant under node/edge/id permutation (they commute)."""
    r = ev.run_permutation_independent_spikes()
    assert r['spike_set_invariant']
    assert r['reset_target_invariant']


def test_wta_same_time_is_order_sensitive_by_explicit_rule():
    """CHARACTERIZATION (not 'simultaneous correctness'): at an exact latency tie the WTA
    winner is chosen by stable node order and the tie is recorded; reversing the drive
    reverses the winner with no node reorder."""
    r = ev.run_wta_order_sensitivity()
    assert r['exact_tie_recorded']
    assert r['exact_tie_winner'] == ['L2E0']          # lowest node order wins the tie
    assert r['stronger_first_winner'] == ['L2E0']
    assert r['stronger_second_winner'] == ['L2E1']    # winner follows drive, not id


def test_relay_multiplicity_coalesces_to_one_emission():
    """CHARACTERIZATION: two same-tau excitatory drivers into one relay -> the relay emits
    exactly once per boundary; the second driver's relay input is rejected/coalesced and is
    visible in the trace only as the ABSENCE of the second relay edge in ``emitted``."""
    r = ev.run_relay_coalescing()
    assert r['both_drivers_fired']                    # both E cells DO fire (no WTA mask)
    assert r['relay_spiked']
    assert len(r['relay_emissions']) == 1             # exactly one relay emission
    assert r['second_emission_present'] is False
    assert len(r['hard_resets']) == 1                 # one reset, not two


def test_permutation_relay_emission_always_single():
    r = ev.run_permutation_independent_spikes()
    assert r['relay_emission_always_single']


# ================================================ Phase 8: dual-FE pre-reset Iaccq timing
def test_dual_fe_iaccq_is_pre_reset_frozen_charge():
    r = ev.run_phase8_learning_probe(seed=1)
    assert r['captured']
    c = r['checks']
    assert c['iaccq_positive']
    assert c['iaccq_finite']
    assert c['iaccq_not_post_reset_zero']             # NOT overwritten by post-reset zero
    assert c['fe_matches_iaccq']                      # FE evaluated from that Iaccq
    assert c['winner_membrane_reset']                 # membrane WAS reset the same boundary
    assert c['observation_does_not_learn']            # display/serialize causes no learning
    assert c['no_nonfinite_delta']
    assert r['all_pass']


def test_dual_fe_manual_fe_reconstruction():
    """Reconstruct FE by hand from the logged Iaccq and confirm it matches the recorded
    factor (bookkeeping validation; no tuning)."""
    r = ev.run_phase8_learning_probe(seed=1)
    ev_ = r['event']
    assert ev_ is not None
    fe_manual = dual_fe(ev_['iaccq'], ev_['theta'], ev_['e'], ev_['B'])
    assert fe_manual == pytest.approx(ev_['fe'], abs=1e-12)


# ============================================= no behavioural change from instrumentation
def test_passive_instrumentation_is_behaviourally_neutral():
    """Enabling the off-by-default learning-event recorder does not change the dual-FE
    dynamic trace (spikes) or the learned weights."""
    r = ev.run_instrumentation_neutrality(seed=1)
    assert r['spikes_equal']
    assert r['weights_equal']
    assert r['neutral']


# ==================================================== Phase 6: independent PQ+RK4 reference
def test_tiny_reference_matches_production_boundary_semantics():
    """The independent priority-queue + RK4 reference (boundary-delivery mode) reproduces
    production's complete (boundary, id, tau) spike trace to numeric tolerance -- same
    event multiset, matching sub-boundary crossing times."""
    r = ev.run_phase6(seed=1)
    assert r['same_event_multiset']
    assert r['max_tau_error'] < 1e-6
    assert r['matched']


def test_phase6_continuous_variant_is_diagnostic_only():
    # the diagnostic continuous-delivery variant differs from production by exactly the
    # declared one-boundary feedforward delay, not by a solver disagreement.
    r = ev.run_phase6(seed=1)
    diag = r['continuous_diag']
    assert diag['production_delay_boundaries'] == 1
    assert diag['continuous_delay_boundaries'] == 0
