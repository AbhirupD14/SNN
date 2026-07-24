# Claude Prompt: Validate the Hybrid Event-Resolved Simulation Engine

## Objective

Test whether the current simulation engine is:

1. internally correct for the equations and event semantics it actually implements; and
2. a defensible approximation of the corresponding continuous-time, discrete-event
   system in the small circuits where an independent numerical reference is practical.

This is a validation and characterization task, not a model-tuning task. Do not change
learning rules, thresholds, weights, inhibition strength, topology, delays, or tie-breaking
to improve the result. Do not rewrite the production engine around a new scheduler in this
task.

The final report must distinguish these three conclusions:

- **implementation bug**: production code disagrees with its own declared semantics;
- **declared approximation**: production behavior follows the code/documented model but
  differs from a more continuous reference because of an explicit boundary rule;
- **validated behavior**: production and the independent reference agree within a declared,
  convergent tolerance.

Do not describe a deterministic result as physically faithful merely because it is
repeatable.

## Current hypothesis to test

The current implementation appears to be best described as:

> A deterministic, boundary-synchronous hybrid simulator with analytic within-boundary
> membrane evolution and event resolution.

It is not yet established as a globally continuous-time, pure discrete-event simulator.
In particular, audit rather than assume the semantics of:

- ordinary spike delivery at the next outer boundary;
- excitation gathered and frozen at boundary start;
- analytic threshold crossings inside a unit boundary;
- zero-latency apical, relay, and hard-reset callbacks;
- one-spike-per-neuron-per-boundary behavior;
- one-emission-per-inhibitory-relay-per-boundary behavior;
- exact and tolerance-based same-time ties;
- deterministic node-order tie-breaking;
- conductance, refractory, activity, and learning finalization at boundaries;
- the relationship between the dual-FE `Iaccq` value and the frozen delivered charge.

Treat the repository as the authority for what is implemented, but do not confuse
implementation with intended physics.

## Read before editing

Inspect at least:

- `backend/simulation.py`, especially `BoundaryEventScheduler`, `_event_step`, event-buffer
  rotation, delivery callbacks, reset paths, tie logging, and finalization;
- `snn/neurons.py`, especially drive gathering/freezing, analytic advancement,
  `crossing_time`, `fire`, `hard_reset`, relay behavior, refractory state, and `iaccq`;
- `backend/network_spec.py` and the smallest synthetic topology helpers;
- `tests/test_event_scheduler.py`;
- `tests/test_lif_segments.py`;
- `tests/test_causal_step.py`;
- `tests/test_inhibitory_relay.py`;
- `tests/test_dual_fe_fes.py`;
- `tests/test_rg_direct_cc4.py`;
- `tests/test_dual_fe_cc4_experiment.py`;
- any existing engine/timing documentation and standing-problems document.

Run and record the existing focused timing tests before adding anything. Preserve all
unrelated working-tree changes. Do not commit or push unless explicitly asked after the
results are reviewed.

## Non-negotiable scientific rules

### No circular oracle

The reference calculation must not call the production methods it is validating. In
particular, an oracle for `advance_segment` or `crossing_time` may read initial state and
parameters from a neuron, but it must independently integrate the equations.

Use a deliberately simple, test-only high-resolution numerical integrator such as RK4.
Do not copy the analytic solution from production into the oracle. Avoid adding a heavy
runtime dependency solely for this experiment.

### Convergence, not one arbitrary tiny step

For numerical comparisons, run at least three decreasing reference step sizes. Demonstrate
that the numerical state and spike-time error converges toward the production analytic
result. Report the step sizes and errors. A single close result is insufficient.

### Do not fake boundary refinement

The engine currently treats an outer boundary as more than a display frame. Running twice
as many ordinary boundaries is not automatically the same physical experiment at half
`dt`: it can also double decay, learning opportunities, input events, and semantic delays.

Only claim a `dt`, `dt/2`, `dt/4` boundary-refinement result if all physical quantities are
held fixed, including:

- stimulus event times and total physical duration;
- membrane and conductance time constants;
- refractory periods;
- propagation delays;
- learning-event count and learning exposure;
- reset and relay behavior;
- observation times.

If the production engine has no explicit physical `dt` or cannot express a fixed physical
delay across multiple smaller boundaries, state that true boundary refinement is currently
unsupported. Do not modify production timing semantics merely to make this test possible.
Instead, provide the exact minimal architectural changes a future refinement test would
require.

### Separate numerical timing from learning

First validate membrane integration, crossing times, delivery, reset, and event ordering
with learning disabled. Only then run a small learning-enabled trace to verify that learning
uses the correct causal pre-reset state. A learned final classification is not an oracle for
correct event timing.

## Deliverables

Create:

```text
experiments/engine_validation.py
tests/test_engine_validation.py
docs/ENGINE_VALIDATION_REPORT.md
```

The experiment must write a timestamped, gitignored run directory under:

```text
experiments/runs/engine_validation/
```

containing:

```text
summary.json
numeric_convergence.csv
event_trace.jsonl
invariants.json
README.md
```

Keep the artifact schema compact and documented. The experiment must accept a deterministic
seed and print the artifact path.

Test-only helpers may live under `tests/` or `experiments/`. Avoid production changes. If
passive instrumentation is absolutely necessary, keep it off by default, prove it does not
alter a golden trace, and explain why existing public state was insufficient.

## Phase 1: Semantic audit

Before writing new tests, produce a table in the report with one row per timing mechanism:

```text
mechanism
production source location
logical timestamp used
same-boundary or delayed
can occur more than once per boundary?
what state it reads
what state it invalidates
declared model rule or accidental constraint?
```

Include ordinary feedforward delivery, sensory input, analytic crossings, E firing,
coincidence deposits, apical delivery, inhibitory relay firing, hard resets, learning,
conductance decay, refractory decay, and boundary finalization.

Explicitly identify which operations use continuous sub-boundary `tau` and which use only
the integer boundary index.

## Phase 2: Independent single-neuron numerical validation

For both the pure-integrator case and the conductance/leaky case, independently compare
production analytic evolution against RK4 for:

1. subthreshold evolution over several intervals;
2. a threshold crossing strictly inside an interval;
3. a crossing near the beginning of an interval;
4. a crossing near the end of an interval;
5. a no-crossing case;
6. nonzero initial voltage;
7. representative excitation and inhibition values;
8. a reset followed by continued evolution where production semantics permit it.

Use parameters far from singularities and additional cases near important numerical
branches. For each case record:

```text
case
reference_dt
analytic_final_state
numeric_final_state
state_error
analytic_spike_time
numeric_spike_time
spike_time_error
```

Assert finite values and monotonic convergence. Choose tolerances from the observed
convergence order and floating-point scale, not from a desire to force green tests.

The numerical oracle must integrate the same frozen-drive differential equation for this
phase. This establishes whether the analytic solver is correct for its own continuous
segment model; it does not by itself validate the boundary-level network approximation.

## Phase 3: Scheduler and causal invalidation

Build small synthetic cases with exactly predictable event times and test:

- earliest crossing wins regardless of node list order when times are distinguishable;
- all unaffected membranes advance to the same logical `tau`;
- a reset invalidates a previously predicted crossing;
- a changed input invalidates/recomputes the affected trajectory;
- logical time never moves backward;
- no event is delivered before its causal parent;
- one emitted event is not silently delivered twice;
- an event at exactly the boundary endpoint has one unambiguous ownership rule;
- floating-point differences just inside and outside tie tolerance are characterized.

Do not merely duplicate existing assertions. Extend them with an event trace that permits
manual reconstruction of state before and after every transition.

## Phase 4: Same-time concurrency and order sensitivity

Construct isomorphic tiny networks in which two independent neurons cross at the exact same
logical `tau`. Run permutations of:

- node declaration order;
- edge declaration order;
- source IDs while preserving graph structure;
- insertion order of simultaneous deliveries where that is controllable.

Compare canonicalized outcomes that do not depend on IDs or serialization order:

- set of emitted spikes;
- reset targets;
- post-event membrane state;
- delivered-event multiset;
- learned-weight deltas when learning is separately enabled.

Test at least:

1. independent simultaneous spikes that should commute;
2. simultaneous competitors connected to a WTA relay;
3. two simultaneous excitatory inputs to one inhibitory relay;
4. a spike and hard reset at the same timestamp;
5. simultaneous basal and apical eligibility for a coincidence cell.

If a permutation changes the scientific outcome, do not automatically call it a failing
test. Record it as order-sensitive and identify the exact production rule responsible.
Classify the outcome:

- explicitly intended arbitration;
- deterministic approximation needing documentation;
- likely implementation artifact;
- genuinely ambiguous model requirement.

For the two-input inhibitory-relay case, report whether the second event is delivered,
coalesced, rejected, or lost, and whether that choice is visible in the trace.

## Phase 5: Event conservation ledger

For each tiny-network experiment, assign or reconstruct a stable causal identity for every
emitted event and account for it as exactly one of:

- delivered;
- explicitly coalesced;
- explicitly cancelled by reset/arbitration;
- pending beyond the observation window.

Assert:

- monotonic logical timestamps;
- no orphan delivery;
- no duplicate delivery unless the model explicitly emits duplicates;
- every reset has a causal source;
- every scheduled event is represented in the final ledger;
- no non-finite membrane, conductance, crossing time, FE/FES value, or weight delta.

If current data structures cannot provide a perfect identity without changing production,
use the strongest reconstructable multiset ledger and document the limitation rather than
inventing precision.

## Phase 6: Tiny-network reference trace

Implement a deliberately slow, independent reference simulator only for a minimal,
learning-disabled graph whose equations and delays can be stated completely in the test.
It should use:

- explicit physical time;
- a priority queue or equivalent ordered event list;
- RK4 continuous evolution between scheduled events;
- declared propagation delays;
- batch collection of exact same-time events before state commit.

Keep this oracle small. It is not a replacement engine and must not grow into a general
topology implementation.

Run two comparisons:

1. A case deliberately matched to current production semantics, including its ordinary
   boundary-sized delivery delay. Compare spike times and complete causal traces.
2. A diagnostic “continuous delivery” variant in which events are delivered at their exact
   declared physical arrival time and same-time events are batch-committed.

The second comparison is diagnostic, not an automatic correctness oracle. For every
difference, identify which declared production approximation explains it.

Do not compare only final winner identity. Compare ordered physical event times, spike
multisets, resets, and final continuous state.

## Phase 7: Boundary sensitivity assessment

Determine whether the current engine can perform a scientifically valid physical
boundary-refinement test without production changes.

If yes, run `dt`, `dt/2`, and `dt/4` on:

- an isolated neuron;
- a two-neuron feedforward chain;
- a minimal WTA circuit;
- the direct `3x3 -> CC4` topology with learning disabled;
- a short, fixed-event-count dual-FE learning trace.

Compare physical spike times, event multisets, WTA ownership, resets, and weight deltas.

If no, report **UNSUPPORTED**, not PASS or FAIL. Specify exactly what prevents equivalence,
such as unit-boundary delay buffers, per-boundary spike/relay guards, boundary-based decay,
or stimulus scheduling. Give a minimal future design for explicit physical `dt` and
physical-time delay queues, but do not implement that refactor here.

You may still test the weaker property that splitting an uninterrupted analytic membrane
segment gives the same final state as advancing it once. Label this **segment partition
invariance**, not network boundary refinement.

## Phase 8: Learning-state timing probe

Using the smallest direct topology and a one-spike controlled input, enable dual FE/FES
learning and prove from the trace:

- `Iaccq` is captured from the causal pre-reset frozen charge;
- FE is evaluated from that value;
- FES is evaluated from each pre-update synaptic weight;
- learning happens exactly once for the causal spike;
- reset/finalization does not replace `Iaccq` with post-reset zero;
- repeating a display/serialization observation does not cause learning;
- no hidden boundary duplication changes the weight delta.

Manually reconstruct at least one weight update in the report. This phase validates timing
and bookkeeping only; do not tune `B`, LR, `e`, `wte`, or weights.

## Required automated tests

Add focused, deterministic tests for:

- RK4 convergence toward analytic segment evolution;
- numerical spike-time convergence;
- segment partition invariance;
- causal invalidation after reset;
- boundary-endpoint ownership;
- event timestamp monotonicity;
- event conservation on a tiny graph;
- order permutation for independent simultaneous events;
- explicit characterization of WTA same-time order sensitivity;
- explicit characterization of relay multiplicity/coalescing;
- dual-FE pre-reset `Iaccq` timing;
- no behavioral change from any passive validation instrumentation.

Scientific characterization tests may assert the current documented behavior even if that
behavior is an approximation. Name such tests accordingly; do not call a known
order-dependent result “simultaneous correctness.”

## Execution

Run:

1. the pre-change focused timing baseline;
2. the new validation tests;
3. all existing timing, neuron, inhibition, direct-CC4, and dual-FE tests;
4. the full test suite;
5. `git diff --check`;
6. the headless validation experiment with the reference seed.

Do not suppress warnings or loosen existing assertions to obtain a pass.

## Final report

`docs/ENGINE_VALIDATION_REPORT.md` and the final response must lead with a verdict containing
separate ratings:

```text
Analytic segment solver:
Scheduler causality:
Same-time concurrency:
Event conservation:
Boundary-refinement evidence:
Dual-FE causal state capture:
Overall characterization:
```

Allowed ratings are:

```text
VALIDATED
VALIDATED WITH DECLARED APPROXIMATION
ORDER-SENSITIVE BY EXPLICIT RULE
FAILED
UNSUPPORTED
```

For every non-validated item include:

- the smallest reproducing circuit;
- expected versus observed trace;
- whether it is a bug, declared approximation, or unresolved model choice;
- the narrowest plausible next action;
- whether existing scientific conclusions are affected.

End with a compact claim boundary:

- what the engine can now be claimed to simulate faithfully;
- what remains dependent on outer-boundary semantics;
- what would be required before calling it a globally continuous-time,
  pure discrete-event simulator.

Report exact commands, test counts, artifact path, and all changed files. Do not commit or
push.
