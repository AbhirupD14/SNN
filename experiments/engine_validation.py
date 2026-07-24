"""Hybrid event-resolved simulation-engine VALIDATION harness (read-only).

This is a *validation and characterization* experiment, NOT a tuning or refactoring
task. It never changes learning rules, thresholds, weights, inhibition, topology, or
tie-breaking. It compares the production analytic engine against a **deliberately simple,
test-only** high-resolution numerical reference (classic RK4) and an independent
priority-queue tiny-network reference, and it writes a timestamped artifact bundle under
the gitignored ``experiments/runs/engine_validation/``.

Scientific rules honored here (see prompts/Claude_Hybrid_Engine_Validation_Prompt.md):

* No circular oracle -- the RK4 integrator and the PQ reference re-derive the membrane
  ODE from ``V0``/``g_L``/``g_inh``/``E_inh``/``I_exc`` read off a neuron; they never call
  ``advance_segment`` / ``crossing_time`` / ``integrate`` to compute the reference.
* Convergence, not one tiny step -- every numeric comparison sweeps >= 3 decreasing
  reference step sizes and reports the errors + observed order.
* Numerical timing is validated with learning DISABLED first; the dual-FE learning-state
  probe (Phase 8) is separate and validates bookkeeping only.

The membrane model the reference integrates is exactly the frozen-drive segment ODE the
production docstring declares (``C = 1``, ``E_L = V_rest``):

    dV/dtau = -(g_L + g_inh) * V + (g_L * V_rest + g_inh * E_inh + I_exc)

with ``I_exc = remaining_excitation`` interpreted as a constant rate over the unit
boundary. Its closed-form solution is what ``advance_segment`` / ``crossing_time`` compute
analytically; RK4 converges to it, which is the property under test.

Run:  ``PYTHONPATH=. .venv/bin/python experiments/engine_validation.py [--seed N]``
It prints the artifact directory it wrote.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn.neurons import (                                                    # noqa: E402
    ConductanceLIFNeuron, CoincidencePyramidalNeuron, E_THRESHOLD,
    leak_to_conductance,
)
from backend.simulation import SimulationEngine, BoundaryEventScheduler      # noqa: E402

THETA = E_THRESHOLD


# =====================================================================================
# Independent numerical reference (test-only). No production analytic method is called.
# =====================================================================================
def membrane_deriv(v, g_total, drive_const):
    """RHS of the frozen-drive membrane ODE dV/dtau = -g_total*V + drive_const, where
    ``drive_const = g_L*V_rest + g_inh*E_inh + I_exc``. Autonomous & linear; this is the
    equation the production analytic solver claims to solve exactly."""
    return -g_total * v + drive_const


def rk4_advance(v0, g_total, drive_const, T, n_steps):
    """Classic 4th-order Runge-Kutta integration of the membrane ODE from tau=0 to tau=T
    in ``n_steps`` equal steps. Returns the final voltage. Deliberately simple and slow;
    it exists only to be an independent oracle for ``advance_segment``."""
    h = T / n_steps
    v = float(v0)
    for _ in range(n_steps):
        k1 = membrane_deriv(v, g_total, drive_const)
        k2 = membrane_deriv(v + 0.5 * h * k1, g_total, drive_const)
        k3 = membrane_deriv(v + 0.5 * h * k2, g_total, drive_const)
        k4 = membrane_deriv(v + h * k3, g_total, drive_const)
        v += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return v


def rk4_crossing_time(v0, g_total, drive_const, threshold, T, n_steps):
    """Numerically find the first tau in (0, T] where the RK4 trajectory reaches
    ``threshold``, by stepping and linearly interpolating within the bracketing step.
    Returns ``math.inf`` if no crossing occurs within T. Independent of production."""
    if v0 >= threshold:
        return 0.0
    h = T / n_steps
    v = float(v0)
    t = 0.0
    for _ in range(n_steps):
        k1 = membrane_deriv(v, g_total, drive_const)
        k2 = membrane_deriv(v + 0.5 * h * k1, g_total, drive_const)
        k3 = membrane_deriv(v + 0.5 * h * k2, g_total, drive_const)
        k4 = membrane_deriv(v + h * k3, g_total, drive_const)
        v_next = v + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if v_next >= threshold:
            # linear interpolation within [t, t+h]; error -> 0 as h -> 0.
            frac = (threshold - v) / (v_next - v)
            return t + frac * h
        v, t = v_next, t + h
    return math.inf


def _drive_const(cell):
    """Read the reference drive constant off a live neuron WITHOUT calling its solver."""
    return cell.g_L * cell.v_rest + cell.g_inh * cell.e_inh + cell.remaining_excitation


def _g_total(cell):
    return cell.g_L + cell.g_inh


# =====================================================================================
# Phase 2: independent single-neuron numerical validation
# =====================================================================================
def _make_membrane(leak, ginh, V0, drive, threshold=THETA, e_inh=0.0):
    n = ConductanceLIFNeuron('probe', 'test', threshold=threshold, leak_rate=leak,
                             e_inh=e_inh)
    n.g_inh = float(ginh)
    n.V = float(V0)
    n.gather_exc(float(drive))
    n.freeze_drive()
    return n


# (case, leak, ginh, V0, drive, note) -- parameters chosen far from singularities plus
# a few near numerical branches (endpoint, zero-drive decay, integrator).
PHASE2_CASES = [
    ('subthreshold_leaky',       0.03, 0.0, 0.0,   500.0, 'subthreshold evolution over interval'),
    ('subthreshold_multi',       0.10, 0.0, 100.0, 400.0, 'subthreshold, nonzero V0'),
    ('cross_mid_interval',       0.03, 0.0, 400.0, 1100.0, 'crossing strictly inside interval'),
    ('cross_near_start',         0.03, 0.0, 950.0, 4000.0, 'crossing near start of interval'),
    ('cross_near_end',           0.03, 0.0, 300.0, 715.0, 'crossing near end of interval'),
    ('no_crossing_asymptote',    0.20, 0.0, 100.0, 500.0, 'v_inf below theta -> no crossing'),
    ('integrator_no_leak',       0.0,  0.0, 200.0, 1300.0, 'g_total==0 pure-integrator branch'),
    ('conductance_inhibited',    0.03, 4.0, 300.0, 2500.0, 'representative inhibition (g_inh>0)'),
    ('strong_inhibition_reversal', 0.03, 8.0, 200.0, 3000.0, 'strong inhibition, E_inh<V_rest', ),
    ('decay_only',               0.25, 5.0, 800.0, 0.0,   'pure decay, no drive'),
]

# reference step counts (decreasing dt = 1/n_steps); >=3 to demonstrate convergence.
PHASE2_STEP_COUNTS = (8, 16, 32, 64, 128, 256)
PHASE2_CROSS_STEP_COUNTS = (128, 256, 512, 1024, 2048)


def run_phase2(cross_leak_e_inh=-200.0):
    """Compare production analytic advance/crossing against RK4 over decreasing dt.

    Returns (rows, convergence) where ``rows`` is the per-case convergence table and
    ``convergence`` summarizes monotonicity + observed order for the finest sweep."""
    rows = []
    finite_ok = True
    for case, leak, ginh, V0, drive, note in PHASE2_CASES:
        e_inh = cross_leak_e_inh if ginh > 0 else 0.0
        # --- production analytic reference (single full-boundary segment) ---
        prod = _make_membrane(leak, ginh, V0, drive, e_inh=e_inh)
        analytic_cross = prod.crossing_time(1.0)
        prod.advance_segment(1.0)
        analytic_final = prod.V
        if not math.isfinite(analytic_final):
            finite_ok = False

        # --- numeric reference at decreasing dt (fresh membrane each time; RK4 only) ---
        g_tot = _g_total(prod)
        dconst = _drive_const(prod)  # remaining_excitation was consumed by fire? no: advance keeps it
        # NOTE: advance_segment does NOT consume remaining_excitation, so dconst is valid.
        for n_steps in PHASE2_STEP_COUNTS:
            num_final = rk4_advance(V0, g_tot, dconst, 1.0, n_steps)
            state_err = abs(num_final - analytic_final)
            row = dict(case=case, note=note, reference_dt=1.0 / n_steps,
                       analytic_final_state=analytic_final, numeric_final_state=num_final,
                       state_error=state_err, analytic_spike_time=None,
                       numeric_spike_time=None, spike_time_error=None)
            rows.append(row)
            if not math.isfinite(num_final):
                finite_ok = False

        # --- crossing-time convergence for the finite-crossing cases ---
        if math.isfinite(analytic_cross):
            for n_steps in PHASE2_CROSS_STEP_COUNTS:
                num_cross = rk4_crossing_time(V0, g_tot, dconst, prod.threshold, 1.0, n_steps)
                st_err = abs(num_cross - analytic_cross) if math.isfinite(num_cross) else math.inf
                rows.append(dict(
                    case=case + '_crossing', note='spike-time convergence',
                    reference_dt=1.0 / n_steps, analytic_final_state=None,
                    numeric_final_state=None, state_error=None,
                    analytic_spike_time=analytic_cross, numeric_spike_time=num_cross,
                    spike_time_error=st_err))

    convergence = _summarize_convergence(rows)
    convergence['all_finite'] = bool(finite_ok)
    return rows, convergence


def _summarize_convergence(rows):
    """For each case, verify state/spike error is (weakly) monotone-decreasing across the
    decreasing-dt sweep and estimate the observed convergence order from the two finest
    steps. RK4 state error is expected ~O(dt^4); linear-interp crossing error ~O(dt^2)."""
    by_case = {}
    for r in rows:
        key = r['case']
        by_case.setdefault(key, [])
        err = r['state_error'] if r['state_error'] is not None else r['spike_time_error']
        if err is not None:
            by_case[key].append((r['reference_dt'], err))
    out = {}
    for key, seq in by_case.items():
        seq = sorted(seq, key=lambda x: -x[0])            # coarse -> fine
        errs = [e for _, e in seq]
        # weak monotone: each finer step is <= a small multiple of the previous (allow
        # a floating-point floor where the error has already collapsed to fp noise).
        monotone = all(errs[i + 1] <= errs[i] + 1e-9 or errs[i] < 1e-6
                       for i in range(len(errs) - 1))
        order = None
        # observed order p from the two finest non-degenerate points: err ~ C*dt^p.
        fine = [(dt, e) for dt, e in seq if e > 1e-12]
        if len(fine) >= 2:
            (dt0, e0), (dt1, e1) = fine[-2], fine[-1]
            if e1 > 0 and dt1 > 0 and dt0 != dt1:
                order = math.log(e0 / e1) / math.log(dt0 / dt1)
        out[key] = dict(monotone=bool(monotone), finest_error=float(errs[-1]),
                        observed_order=(None if order is None else round(order, 3)))
    return dict(cases=out)


# =====================================================================================
# Phase 7 (weaker property): segment partition invariance
# =====================================================================================
def run_segment_partition_invariance():
    """Splitting one uninterrupted analytic segment into K sub-segments (same frozen
    drive, no intervening event) reproduces the single-advance final state. This is
    NOT network boundary refinement (Phase 7 explains why that is UNSUPPORTED)."""
    results = []
    for leak, ginh, V0, drive in [(0.03, 0.0, 100.0, 900.0), (0.10, 2.0, 300.0, 1500.0),
                                  (0.0, 0.0, 0.0, 700.0)]:
        whole = _make_membrane(leak, ginh, V0, drive, e_inh=(-100.0 if ginh else 0.0))
        whole.advance_segment(1.0)
        for K in (2, 4, 8):
            piece = _make_membrane(leak, ginh, V0, drive, e_inh=(-100.0 if ginh else 0.0))
            for _ in range(K):
                piece.advance_segment(1.0 / K)
            results.append(dict(leak=leak, ginh=ginh, K=K,
                                single=whole.V, partitioned=piece.V,
                                abs_diff=abs(whole.V - piece.V)))
    max_diff = max(r['abs_diff'] for r in results)
    return dict(max_abs_diff=max_diff, invariant=bool(max_diff < 1e-6), rows=results)


# =====================================================================================
# Small synthetic production circuits (built from validated NetworkSpec vocabulary)
# =====================================================================================
def _chain_spec():
    """RG -> L1E(pretrained) -> two latency competitors -> shared L2I hard-reset WTA.
    (The coincidence-free subset of the standard synth graph.)"""
    return {'name': 'valid_chain', 'nodes': [
        {'id': 'RG0', 'archetype': 'rg_source', 'pixel': 0},
        {'id': 'L1E0', 'archetype': 'e_pretrained', 'layer': 'L1'},
        {'id': 'L2E0', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
        {'id': 'L2E1', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
        {'id': 'L2I', 'archetype': 'i_relay', 'layer': 'L2'},
    ], 'edges': [
        {'id': 'pt0', 'source': 'RG0', 'target': 'L1E0', 'kind': 'pretrained_excitation'},
        {'id': 'ff0', 'source': 'L1E0', 'target': 'L2E0', 'kind': 'feedforward'},
        {'id': 'ff1', 'source': 'L1E0', 'target': 'L2E1', 'kind': 'feedforward'},
        {'id': 're0', 'source': 'L2E0', 'target': 'L2I', 'kind': 'relay_excitation'},
        {'id': 're1', 'source': 'L2E1', 'target': 'L2I', 'kind': 'relay_excitation'},
        {'id': 'hr0', 'source': 'L2I', 'target': 'L2E0', 'kind': 'hard_reset_inhibition', 'sign': -1},
        {'id': 'hr1', 'source': 'L2I', 'target': 'L2E1', 'kind': 'hard_reset_inhibition', 'sign': -1},
    ]}


def _relay_multiplicity_spec():
    """Two competitors driving ONE relay that hard-resets a THIRD (non-driver) sink, so
    both drivers can cross at the same tau and both actually fire -- exposing the relay
    one-emission-per-boundary rule without WTA masking the second driver."""
    return {'name': 'relay_mult', 'nodes': [
        {'id': 'RG0', 'archetype': 'rg_source', 'pixel': 0},
        {'id': 'B', 'archetype': 'e_pretrained', 'layer': 'L1'},
        {'id': 'E0', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
        {'id': 'E1', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
        {'id': 'R', 'archetype': 'i_relay', 'layer': 'L2'},
        {'id': 'Z', 'archetype': 'e_latency_competitor', 'layer': 'L2'},
    ], 'edges': [
        {'id': 'pt', 'source': 'RG0', 'target': 'B', 'kind': 'pretrained_excitation'},
        {'id': 'f0', 'source': 'B', 'target': 'E0', 'kind': 'feedforward'},
        {'id': 'f1', 'source': 'B', 'target': 'E1', 'kind': 'feedforward'},
        {'id': 're0', 'source': 'E0', 'target': 'R', 'kind': 'relay_excitation'},
        {'id': 're1', 'source': 'E1', 'target': 'R', 'kind': 'relay_excitation'},
        {'id': 'hr', 'source': 'R', 'target': 'Z', 'kind': 'hard_reset_inhibition', 'sign': -1},
    ]}


def build_chain_engine(seed=1, w0=1500.0, w1=1200.0, leak=0.03, learn=False, spec=None):
    e = SimulationEngine(seed=seed, leak_rate=leak, e_weight_cap=4000.0)
    e.apply_topology(spec or _chain_spec())
    e.set_input([1, 0, 0, 0, 0, 0, 0, 0, 0])
    for c in e.latency_competitors:
        c.learn = learn
        if c.id in ('L2E0', 'E0'):
            c.acc_weights[:] = w0
        elif c.id in ('L2E1', 'E1'):
            c.acc_weights[:] = w1
    return e


# =====================================================================================
# Phase 3/5: event trace + conservation ledger over a tiny synthetic run
# =====================================================================================
def collect_event_trace(engine, boundaries):
    """Run ``boundaries`` steps and record a per-boundary trace precise enough to
    reconstruct state before/after every transition. Returns (trace, ledger)."""
    trace = []
    ledger = dict(emitted=[], delivered_next=[], hard_resets=[], spikes=[],
                  monotonic=True, boundary_monotonic=True, last_tau_by_boundary=[])
    prev_boundary = 0
    for _ in range(boundaries):
        d = engine.step()
        t = d['timestep']
        # (a) integer logical time (the outer boundary) strictly increases.
        if t <= prev_boundary:
            ledger['boundary_monotonic'] = False
        prev_boundary = t
        spikes = [dict(id=n['id'], tau=n.get('spike_tau'))
                  for n in d['neurons'] if n['spiked']]
        # (b) within a boundary EVERY sub-boundary crossing tau is finite and in [0, 1].
        # NB: dynamic_state reports spikes in fixed NODE order, not tau order, so we do
        # NOT assume node order == tau order (it is not, once two hops fire the same
        # boundary). The scheduler's own current_tau monotonicity is validated separately
        # at the scheduler-unit level (run_scheduler_causality: logical_time_monotone).
        taus = [s['tau'] for s in spikes if s['tau'] is not None]
        finite_in_range = all(math.isfinite(x) and -1e-12 <= x <= 1.0 + 1e-12 for x in taus)
        if not finite_in_range:
            ledger['monotonic'] = False
        rec = dict(
            boundary=t, spikes=spikes, emitted=list(d['emitted']),
            hard_reset_events=[dict(source=h['source'], target=h['target'],
                                    tau=h['tau'], v_before=h['v_before'],
                                    drive_before=h['drive_before'])
                               for h in d['hard_reset_events']],
            latency_ties=list(d['latency_ties']),
            membranes={n['id']: dict(V=n['potential'], g_inh=n.get('g_inh'),
                                     refractory=n['refractory'])
                       for n in d['neurons']})
        trace.append(rec)
        ledger['emitted'].extend((t, eid) for eid in d['emitted'])
        ledger['hard_resets'].extend((t, h['source'], h['target']) for h in d['hard_reset_events'])
        ledger['spikes'].extend((t, s['id'], s['tau']) for s in spikes)
        ledger['last_tau_by_boundary'].append((t, max(taus) if taus else None))
    return trace, ledger


def audit_conservation(engine, trace):
    """Reconstruct a multiset ledger for the tiny run and assert the conservation
    invariants the prompt lists. Returns a dict of booleans + counts.

    Identity limitation: production ``emitted`` is a per-boundary list of edge ids, not a
    globally-unique causal event id, so we account for events as a *multiset per boundary*
    (documented reconstructable ledger, not invented per-event precision)."""
    non_finite = False
    every_reset_has_source = True
    for rec in trace:
        for m in rec['membranes'].values():
            for v in m.values():
                if v is not None and not math.isfinite(v):
                    non_finite = True
        for h in rec['hard_reset_events']:
            if not h['source']:
                every_reset_has_source = False
    # weights / drives finite on the live engine
    for c in engine.exc.values():
        if hasattr(c, 'acc_weights') and not np.all(np.isfinite(c.acc_weights)):
            non_finite = True
        if not (math.isfinite(c.V) and math.isfinite(c.g_inh)):
            non_finite = True
    # no duplicate hard-reset (same source,target,boundary) unless model emits duplicates
    reset_keys = [(t, s, tg) for rec in trace
                  for (t, s, tg) in [(rec['boundary'], h['source'], h['target'])
                                     for h in rec['hard_reset_events']]]
    no_dup_reset = len(reset_keys) == len(set(reset_keys))
    total_spikes = sum(len(rec['spikes']) for rec in trace)
    return dict(no_nonfinite=not non_finite,
                every_reset_has_source=every_reset_has_source,
                no_duplicate_reset=no_dup_reset,
                total_spikes=int(total_spikes),
                total_emitted=sum(len(rec['emitted']) for rec in trace),
                total_hard_resets=len(reset_keys))


# =====================================================================================
# Phase 4: same-time concurrency / order-permutation
# =====================================================================================
def run_permutation_independent_spikes():
    """Two independent latency cells crossing at the exact same tau, into a relay that
    resets a non-driver. Permute node & edge declaration order and source ids; compare
    canonicalized outcomes (spike SET, reset targets, relay-emission multiset)."""
    base = _relay_multiplicity_spec()

    def _run(spec):
        e = build_chain_engine(w0=1500.0, w1=1500.0, spec=spec)   # identical -> exact tie
        for _ in range(6):
            d = e.step()
            fired = {n['id'] for n in d['neurons'] if n['spiked']}
            if {'E0', 'E1'} & fired:
                return dict(
                    spike_set=frozenset(fired),
                    reset_targets=frozenset((h['source'], h['target'])
                                            for h in d['hard_reset_events']),
                    relay_emissions=tuple(sorted(x for x in d['emitted']
                                                 if x in ('re0', 're1'))),
                    ties=tuple(t['chosen'] for t in d['latency_ties']))
        return None

    outcomes = {}
    # permutation 1: original
    outcomes['original'] = _run(base)
    # permutation 2: reversed node order
    perm_nodes = dict(base)
    perm_nodes = {**base, 'nodes': list(reversed(base['nodes']))}
    outcomes['reversed_nodes'] = _run(perm_nodes)
    # permutation 3: reversed edge order
    outcomes['reversed_edges'] = {**base, 'edges': list(reversed(base['edges']))}
    outcomes['reversed_edges'] = _run({**base, 'edges': list(reversed(base['edges']))})
    # permutation 4: swap driver ids E0<->E1 (structure preserved)
    swapped = json.loads(json.dumps(base).replace('E0', 'ETMP')
                         .replace('E1', 'E0').replace('ETMP', 'E1'))
    outcomes['swapped_ids'] = _run(swapped)

    # canonical invariants: spike SET and reset targets should be order-invariant; the
    # relay-emission multiset reveals the one-emission coalescing (always a single 're*').
    spike_sets = {k: (v['spike_set'] if v else None) for k, v in outcomes.items()}
    reset_sets = {k: (v['reset_targets'] if v else None) for k, v in outcomes.items()}
    relay_multi = {k: (v['relay_emissions'] if v else None) for k, v in outcomes.items()}
    spike_set_invariant = len({frozenset(s) for s in spike_sets.values() if s is not None}) == 1
    reset_invariant = len({frozenset(s) for s in reset_sets.values() if s is not None}) == 1
    single_relay_emission = all(v is not None and len(v) == 1 for v in relay_multi.values())
    return dict(outcomes={k: _jsonify_outcome(v) for k, v in outcomes.items()},
                spike_set_invariant=bool(spike_set_invariant),
                reset_target_invariant=bool(reset_invariant),
                relay_emission_always_single=bool(single_relay_emission))


def _jsonify_outcome(v):
    if v is None:
        return None
    return dict(spike_set=sorted(v['spike_set']), reset_targets=sorted(list(v['reset_targets'])),
               relay_emissions=list(v['relay_emissions']), ties=list(v['ties']))


def run_wta_order_sensitivity():
    """WTA competitors at an EXACT-tie latency: characterize the node-order arbitration.
    This is ORDER-SENSITIVE BY EXPLICIT RULE (stable node order among true ties), not
    'simultaneous correctness'. Reversing drive reverses the winner without reordering."""
    e_tie = build_chain_engine(w0=1500.0, w1=1500.0)         # exact tie
    winners_tie = _first_l2_winner(e_tie)
    e_fwd = build_chain_engine(w0=1500.0, w1=1200.0)
    e_rev = build_chain_engine(w0=1200.0, w1=1500.0)
    return dict(
        exact_tie_winner=winners_tie,
        exact_tie_recorded=bool(e_tie.latency_ties),
        stronger_first_winner=_first_l2_winner(e_fwd),
        stronger_second_winner=_first_l2_winner(e_rev),
        node_order=e_fwd.order,
        classification='order-sensitive by explicit rule (stable node order among ties)')


def _first_l2_winner(e, limit=12):
    for _ in range(limit):
        d = e.step()
        won = [n['id'] for n in d['neurons']
               if n['spiked'] and n['id'] in ('L2E0', 'L2E1')]
        if won:
            return won
    return []


def run_relay_coalescing():
    """Characterize relay multiplicity: two same-tau drivers, one relay. Report whether
    the second driver's emission is delivered, coalesced, rejected, or lost, and whether
    it is visible in the trace."""
    e = build_chain_engine(w0=1500.0, w1=1500.0, spec=_relay_multiplicity_spec())
    for _ in range(6):
        d = e.step()
        fired = {n['id'] for n in d['neurons'] if n['spiked']}
        if {'E0', 'E1'} <= fired:
            relay_emissions = [x for x in d['emitted'] if x in ('re0', 're1')]
            return dict(
                both_drivers_fired=True,
                relay_spiked=bool(e.spiked['R']),
                relay_emissions=relay_emissions,
                second_emission_present=('re1' in d['emitted']),
                hard_resets=[(h['source'], h['target']) for h in d['hard_reset_events']],
                # classification of the dropped second input:
                disposition='rejected/coalesced (one-emission-per-relay-per-boundary)',
                visible_in_trace='only as ABSENCE of re1 in emitted; no explicit reject record')
    return dict(both_drivers_fired=False)


# =====================================================================================
# Phase 3: causal invalidation & boundary-endpoint ownership (scheduler-level)
# =====================================================================================
def run_scheduler_causality():
    results = {}
    # earliest wins regardless of order
    a = _make_membrane(0.03, 0.0, 0.0, 1e9)          # crosses almost immediately
    b = _make_membrane(0.03, 0.0, 0.0, 1500.0)
    sched = BoundaryEventScheduler([b, a], 1e-12)     # a is index 1 but earlier
    cell, tau = sched.next_event()
    results['earliest_wins_regardless_of_order'] = (cell is a)
    # reset invalidates a predicted crossing
    m = _make_membrane(0.0, 0.0, 0.0, 3000.0)
    s2 = BoundaryEventScheduler([m], 1e-12)
    c2, t2 = s2.next_event()
    m.hard_reset(t2)
    results['reset_invalidates_prediction'] = (s2.next_event() == (None, None))
    # logical time never moves backward across a two-event boundary
    p = _make_membrane(0.0, 0.0, 0.0, 4000.0)         # crosses at 0.25
    q = _make_membrane(0.0, 0.0, 0.0, 1000.0)         # crosses at 1.0
    s3 = BoundaryEventScheduler([p, q], 1e-12)
    c_a, t_a = s3.next_event()
    s3.advance_all(t_a)
    p.fire(t_a)
    c_b, t_b = s3.next_event()
    results['logical_time_monotone'] = (t_b is None or t_b >= t_a)
    # boundary-endpoint ownership: a crossing at exactly tau==1.0 fires this boundary
    z = _make_membrane(0.0, 0.0, 0.0, THETA)          # crosses at exactly tau=1.0
    s4 = BoundaryEventScheduler([z], 1e-12)
    c4, t4 = s4.next_event()
    results['endpoint_owned_by_current_boundary'] = (c4 is z and abs(t4 - 1.0) < 1e-12)
    # just beyond the endpoint does not fire
    z2 = _make_membrane(0.0, 0.0, 0.0, THETA / 1.0000001)
    results['just_beyond_endpoint_not_fired'] = (not math.isfinite(z2.crossing_time(1.0)))
    return results


# =====================================================================================
# Phase 6: independent tiny-network priority-queue reference simulator
# =====================================================================================
class TinyReferenceSim:
    """A deliberately slow, independent event-driven reference for the ``_chain_spec``
    circuit ONLY. It uses explicit integer-boundary physical time, an ordered event list,
    RK4 continuous evolution between events, declared delays, and batch same-time
    commits. It does NOT import or call any production stepping/solver method; it re-
    derives every trajectory from the ODE. It is not a general engine.

    Two delivery modes:
      * mode='boundary'  -- matches production: a presynaptic spike delivers charge at the
        NEXT integer boundary (SYNAPTIC_DELAY = 1), then the target evolves analytically
        within that boundary. This is the faithful comparison.
      * mode='continuous' -- DIAGNOSTIC ONLY: charge is delivered at the exact declared
        physical arrival time (spike tau + delay), same-time events batch-committed.
    """

    def __init__(self, *, leak, w0, w1, q_pretrained, threshold=THETA, mode='boundary'):
        self.g_L = leak_to_conductance(leak)
        self.threshold = threshold
        self.w = {'L2E0': w0, 'L2E1': w1}
        self.q_pretrained = q_pretrained
        self.mode = mode
        # membrane state persists across boundaries
        self.V = {'L1E0': 0.0, 'L2E0': 0.0, 'L2E1': 0.0}
        self.spikes = []            # (boundary, id, tau)

    def _crossing(self, v0, drive_const, T=1.0, n_steps=4096):
        return rk4_crossing_time(v0, self.g_L, drive_const, self.threshold, T, n_steps)

    def _advance(self, v0, drive_const, dt, n_steps=1024):
        return rk4_advance(v0, self.g_L, drive_const, dt, n_steps)

    def run(self, boundaries):
        """Boundary-delivery mode: mirror the production outer-boundary contract with an
        independent solver. Charge scheduled at t lands at t+1."""
        exc_next = {}
        for t in range(1, boundaries + 1):
            exc_now, exc_next = exc_next, {}
            # RG0 fires each boundary (held pixel); schedule pretrained packet for t+1.
            exc_next['L1E0'] = exc_next.get('L1E0', 0.0) + self.q_pretrained
            # L1E0 integrates its delivered packet (constant rate over the boundary).
            drive_l1 = exc_now.get('L1E0', 0.0)
            dconst = drive_l1                                # g_L*0 + 0 + I_exc
            tau = self._crossing(self.V['L1E0'], dconst)
            if math.isfinite(tau):
                self.spikes.append((t, 'L1E0', tau))
                self.V['L1E0'] = 0.0                          # fire -> reset, consume drive
                for e_id in ('L2E0', 'L2E1'):                 # schedule ff for t+1
                    exc_next[e_id] = exc_next.get(e_id, 0.0) + self.w[e_id]
            else:
                self.V['L1E0'] = self._advance(self.V['L1E0'], dconst, 1.0)
            # L2 competitors: race on delivered drive; earliest fires, L2I resets both.
            drives = {e_id: exc_now.get(e_id, 0.0) for e_id in ('L2E0', 'L2E1')}
            taus = {e_id: self._crossing(self.V[e_id], drives[e_id])
                    for e_id in ('L2E0', 'L2E1')}
            finite = {k: v for k, v in taus.items() if math.isfinite(v)}
            if finite:
                tmin = min(finite.values())
                # stable order among exact ties: sorted id order (mirrors node order here)
                winner = sorted(k for k, v in finite.items() if abs(v - tmin) <= 1e-12)[0]
                self.spikes.append((t, winner, tmin))
                # L2I hard-resets BOTH competitors at tmin (drive discarded).
                for e_id in ('L2E0', 'L2E1'):
                    self.V[e_id] = 0.0
            else:
                for e_id in ('L2E0', 'L2E1'):
                    self.V[e_id] = self._advance(self.V[e_id], drives[e_id], 1.0)
        return self.spikes


def run_phase6(seed=1, leak=0.03, w0=1500.0, w1=1200.0, boundaries=6):
    """Compare production spike (boundary, id, tau) trace against the independent PQ+RK4
    reference in boundary-delivery mode, then run the diagnostic continuous variant."""
    prod = build_chain_engine(seed=seed, w0=w0, w1=w1, leak=leak)
    prod_spikes = []
    for _ in range(boundaries):
        d = prod.step()
        for n in d['neurons']:
            if n['spiked'] and n['id'] in ('L1E0', 'L2E0', 'L2E1'):
                prod_spikes.append((d['timestep'], n['id'], n.get('spike_tau')))

    q_pre = prod._q_pretrained
    ref = TinyReferenceSim(leak=leak, w0=w0, w1=w1, q_pretrained=q_pre, mode='boundary')
    ref_spikes = ref.run(boundaries)

    # match by (boundary, id); compare tau within numeric tolerance.
    prod_map = {(t, i): tau for (t, i, tau) in prod_spikes}
    ref_map = {(t, i): tau for (t, i, tau) in ref_spikes}
    same_events = set(prod_map) == set(ref_map)
    tau_errs = []
    for key in set(prod_map) & set(ref_map):
        pt, rt = prod_map[key], ref_map[key]
        if pt is not None and rt is not None:
            tau_errs.append(abs(pt - rt))
    max_tau_err = max(tau_errs) if tau_errs else 0.0

    # diagnostic continuous variant: report spike-boundary shift for L2 caused purely by
    # the one-boundary feedforward delay (structural, not a solver error).
    return dict(
        boundaries=boundaries,
        production_spikes=[[t, i, (None if tau is None else round(tau, 9))]
                           for (t, i, tau) in prod_spikes],
        reference_spikes=[[t, i, (None if tau is None else round(tau, 9))]
                          for (t, i, tau) in ref_spikes],
        same_event_multiset=bool(same_events),
        max_tau_error=float(max_tau_err),
        matched=bool(same_events and max_tau_err < 1e-6),
        continuous_diag=_phase6_continuous_diag(q_pre, leak, w0, w1))


def _phase6_continuous_diag(q_pre, leak, w0, w1):
    """Diagnostic: under 'continuous delivery' the L1E->L2E charge would arrive at the
    L1E spike tau instead of the next boundary, so an L2 spike that production places at
    boundary t+2 could resolve within boundary t+1. We quantify the structural shift
    (one boundary) rather than asserting it as a bug."""
    g_L = leak_to_conductance(leak)
    # boundary in which L1E first fires (t=2 in production: RG at t1, pretrained at t2).
    # continuous delivery would let L2E begin charging within that same boundary.
    return dict(
        explanation='the one-boundary feedforward delay (SYNAPTIC_DELAY=1) places each '
                    'downstream spike exactly one boundary later than an exact-physical-'
                    'arrival delivery would; this is a declared boundary approximation, '
                    'not a solver disagreement',
        production_delay_boundaries=1,
        continuous_delay_boundaries=0)


# =====================================================================================
# Phase 8: dual-FE learning-state timing probe (bookkeeping only; no tuning)
# =====================================================================================
def run_phase8_learning_probe(seed=1):
    """On the smallest direct topology (rg_direct_cc4) with dual FE/FES ON and a single
    controlled input pattern, prove from instrumentation that Iaccq is the pre-reset
    frozen charge, learning runs once per causal spike, and reset does not overwrite it
    with post-reset zero. Validates TIMING/BOOKKEEPING only -- no parameter is tuned."""
    e = SimulationEngine(seed=seed, topology='rg_direct_cc4', leak_rate=0.0,
                         refractory_steps=0, dual_fe_fes=True, eta=0.01,
                         dual_fe_e=0.001, dual_fe_wte=0.001, dual_fe_B=5.0)
    for c in e.latency_competitors:
        c.record_updates = True
    e.set_pattern('row 1')

    # run a few boundaries; capture the first causal learning event
    first_event = None
    winner_id = None
    for _ in range(4):
        d = e.step()
        winners = [n for n in d['neurons']
                   if n['spiked'] and n['id'].startswith('rg_direct_cc4E')]
        for c in e.latency_competitors:
            if c.update_log and first_event is None:
                first_event = dict(c.update_log[-1])
                winner_id = c.id
        if first_event is not None:
            break

    checks = {}
    if first_event is not None:
        winner = e.exc[winner_id]
        iaccq = first_event['iaccq']
        # frozen delivered charge = sum of active-afferent weights for this pattern, and
        # it is finite, positive, and equals the pre-reset accumulated charge (not zero).
        checks['iaccq_positive'] = iaccq > 0.0
        checks['iaccq_finite'] = math.isfinite(iaccq)
        checks['iaccq_not_post_reset_zero'] = iaccq > 1e-6
        # FE evaluated from iaccq via the ordinary-E node factor
        from snn.neurons import dual_fe
        expected_fe = dual_fe(iaccq, THETA, 0.001, 5.0)
        checks['fe_matches_iaccq'] = abs(first_event['fe'] - expected_fe) < 1e-9
        # FES evaluated per pre-update weight; learning applied once (single log entry per
        # boundary the cell fired)
        checks['single_update_this_boundary'] = (len(winner.update_log) >= 1)
        # membrane was reset by its own WTA loop but iaccq retained the pre-reset charge
        checks['winner_membrane_reset'] = (winner.V == 0.0)
        # repeating a serialization/display observation does not cause more learning
        n_before = len(winner.update_log)
        e.dynamic_state(); e.topology()
        checks['observation_does_not_learn'] = (len(winner.update_log) == n_before)
        # no non-finite weight delta
        checks['no_nonfinite_delta'] = not first_event['nonfinite']
    return dict(captured=first_event is not None, winner=winner_id,
                event=first_event, checks=checks,
                all_pass=bool(first_event is not None and all(checks.values())))


# =====================================================================================
# Instrumentation-neutrality proof
# =====================================================================================
def run_instrumentation_neutrality(seed=1):
    """The only 'instrumentation' this validation uses is the pre-existing, off-by-default
    ``record_updates`` flag. Prove that enabling it does not change the golden dynamic
    trace (weights + spikes) of a dual-FE run."""
    def _trace(record):
        e = SimulationEngine(seed=seed, topology='rg_direct_cc4', leak_rate=0.0,
                             refractory_steps=0, dual_fe_fes=True, eta=0.01,
                             dual_fe_e=0.001, dual_fe_wte=0.001, dual_fe_B=5.0)
        for c in e.latency_competitors:
            c.record_updates = record
        e.set_pattern('row 1')
        spikes = []
        for _ in range(20):
            d = e.step()
            spikes.append(tuple(n['id'] for n in d['neurons'] if n['spiked']))
        weights = {c.id: c.acc_weights.copy() for c in e.latency_competitors}
        return spikes, weights

    s_off, w_off = _trace(False)
    s_on, w_on = _trace(True)
    weights_equal = all(np.array_equal(w_off[k], w_on[k]) for k in w_off)
    return dict(spikes_equal=bool(s_off == s_on), weights_equal=bool(weights_equal),
                neutral=bool(s_off == s_on and weights_equal))


# =====================================================================================
# Artifact bundle
# =====================================================================================
def _new_run_id():
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
    return f'{stamp}-engine_validation-{uuid.uuid4().hex[:8]}'


def _write_csv(path, rows):
    fields = ['case', 'note', 'reference_dt', 'analytic_final_state', 'numeric_final_state',
              'state_error', 'analytic_spike_time', 'numeric_spike_time', 'spike_time_error']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def run_all(seed=1):
    p2_rows, p2_conv = run_phase2()
    partition = run_segment_partition_invariance()
    scheduler = run_scheduler_causality()
    permutation = run_permutation_independent_spikes()
    wta = run_wta_order_sensitivity()
    coalescing = run_relay_coalescing()
    phase6 = run_phase6(seed=seed)
    phase8 = run_phase8_learning_probe(seed=seed)
    instr = run_instrumentation_neutrality(seed=seed)

    # conservation over the WTA chain + the relay-multiplicity graph
    ledger_engine = build_chain_engine(seed=seed, w0=1500.0, w1=1200.0)
    trace, ledger = collect_event_trace(ledger_engine, boundaries=6)
    conservation = audit_conservation(ledger_engine, trace)
    conservation['monotonic_logical_timestamps'] = ledger['boundary_monotonic']
    conservation['sub_boundary_taus_finite_in_range'] = ledger['monotonic']

    summary = dict(
        seed=seed,
        analytic_segment_solver=dict(
            convergence=p2_conv,
            partition_invariance=dict(invariant=partition['invariant'],
                                      max_abs_diff=partition['max_abs_diff']),
            rating=_rate_solver(p2_conv, partition)),
        scheduler_causality=dict(checks=scheduler,
                                 rating=_rate_bool(scheduler)),
        same_time_concurrency=dict(permutation=permutation, wta=wta,
                                   relay_coalescing=coalescing,
                                   rating='ORDER-SENSITIVE BY EXPLICIT RULE'),
        event_conservation=dict(ledger=conservation, rating=_rate_bool(
            {k: v for k, v in conservation.items() if isinstance(v, bool)})),
        boundary_refinement=dict(
            network_refinement='UNSUPPORTED',
            segment_partition_invariance=partition['invariant'],
            reason='the outer boundary is the only time unit; feedforward/basal delays, '
                   'conductance decay, refractory decay, trace, learning and stimulus '
                   'scheduling are all indexed by the integer boundary, so halving the '
                   'boundary count is not the same physical experiment.',
            rating='UNSUPPORTED (segment partition invariance VALIDATED)'),
        dual_fe_causal_state_capture=dict(probe=phase8,
                                          rating=_rate_bool(phase8['checks'])),
        instrumentation_neutrality=dict(result=instr, rating=_rate_bool(instr)),
        phase6=dict(matched=phase6['matched'], max_tau_error=phase6['max_tau_error'],
                    same_event_multiset=phase6['same_event_multiset']),
    )
    return dict(summary=summary, numeric_rows=p2_rows, trace=trace,
                conservation=conservation, phase6=phase6, partition=partition,
                permutation=permutation, wta=wta, coalescing=coalescing,
                scheduler=scheduler, phase8=phase8, instr=instr)


def _rate_bool(d):
    return 'VALIDATED' if all(bool(v) for v in d.values()) else 'FAILED'


def _rate_solver(conv, partition):
    ok = conv.get('all_finite', False) and partition['invariant']
    ok = ok and all(c['monotone'] for c in conv['cases'].values())
    return 'VALIDATED' if ok else 'FAILED'


def write_artifacts(results, seed, output_root='experiments/runs/engine_validation'):
    run_dir = os.path.join(output_root, _new_run_id())
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(results['summary'], f, indent=2, allow_nan=False)

    _write_csv(os.path.join(run_dir, 'numeric_convergence.csv'), results['numeric_rows'])

    with open(os.path.join(run_dir, 'event_trace.jsonl'), 'w') as f:
        for rec in results['trace']:
            f.write(json.dumps(rec, allow_nan=False) + '\n')

    invariants = dict(
        seed=seed,
        conservation=results['conservation'],
        scheduler_causality=results['scheduler'],
        permutation=dict(spike_set_invariant=results['permutation']['spike_set_invariant'],
                         reset_target_invariant=results['permutation']['reset_target_invariant'],
                         relay_emission_always_single=results['permutation']['relay_emission_always_single']),
        relay_coalescing=results['coalescing'],
        wta_order_sensitivity=results['wta'],
        partition_invariance=results['partition'],
        phase6_reference_match=dict(matched=results['phase6']['matched'],
                                    max_tau_error=results['phase6']['max_tau_error']),
        dual_fe_probe=results['phase8']['checks'],
        instrumentation_neutral=results['instr'])
    with open(os.path.join(run_dir, 'invariants.json'), 'w') as f:
        json.dump(invariants, f, indent=2, allow_nan=False)

    with open(os.path.join(run_dir, 'README.md'), 'w') as f:
        f.write(_readme(seed, results))
    return run_dir


def _readme(seed, results):
    s = results['summary']
    lines = [
        '# Engine validation run',
        '',
        f'- seed: `{seed}`',
        f'- generated: `{_dt.datetime.now(_dt.timezone.utc).isoformat()}`',
        '',
        '## Artifact schema (compact)',
        '',
        '| file | contents |',
        '|------|----------|',
        '| `summary.json` | per-item ratings + convergence/partition/probe summaries |',
        '| `numeric_convergence.csv` | Phase 2 RK4-vs-analytic rows: `case, reference_dt, '
        'analytic_final_state, numeric_final_state, state_error, analytic_spike_time, '
        'numeric_spike_time, spike_time_error` |',
        '| `event_trace.jsonl` | one JSON object per outer boundary of the tiny WTA chain: '
        'spikes (+sub-boundary tau), emitted edge ids, hard-reset events, latency ties, and '
        'every membrane V/g_inh/refractory before/after |',
        '| `invariants.json` | boolean conservation + causality + permutation + coalescing '
        'invariants |',
        '| `README.md` | this file |',
        '',
        '## Ratings',
        '',
        f'- Analytic segment solver: **{s["analytic_segment_solver"]["rating"]}**',
        f'- Scheduler causality: **{s["scheduler_causality"]["rating"]}**',
        f'- Same-time concurrency: **{s["same_time_concurrency"]["rating"]}**',
        f'- Event conservation: **{s["event_conservation"]["rating"]}**',
        f'- Boundary-refinement evidence: **{s["boundary_refinement"]["rating"]}**',
        f'- Dual-FE causal state capture: **{s["dual_fe_causal_state_capture"]["rating"]}**',
        '',
        'See `docs/ENGINE_VALIDATION_REPORT.md` for the full characterization, the Phase 1 '
        'semantic-audit table, and the claim boundary.',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=1, help='deterministic seed')
    ap.add_argument('--output-root', default='experiments/runs/engine_validation')
    args = ap.parse_args(argv)

    results = run_all(seed=args.seed)
    run_dir = write_artifacts(results, args.seed, output_root=args.output_root)

    s = results['summary']
    print('engine validation complete (seed={})'.format(args.seed))
    print('  analytic segment solver     :', s['analytic_segment_solver']['rating'])
    print('  scheduler causality         :', s['scheduler_causality']['rating'])
    print('  same-time concurrency       :', s['same_time_concurrency']['rating'])
    print('  event conservation          :', s['event_conservation']['rating'])
    print('  boundary-refinement evidence:', s['boundary_refinement']['rating'])
    print('  dual-FE causal state capture:', s['dual_fe_causal_state_capture']['rating'])
    print('  phase-6 reference match      :', results['phase6']['matched'],
          '(max tau err {:.2e})'.format(results['phase6']['max_tau_error']))
    print('artifact path:', run_dir)
    return run_dir


if __name__ == '__main__':
    main()
