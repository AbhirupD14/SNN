# Simulation Engine Validation — Executive and Handoff Report

**Date:** July 23, 2026
**Reference seed:** 1
**Reference run:** `20260723-185239-engine_validation-10c794e6`

## Executive summary

The simulation engine is not “cheating.” Independent numerical and causal-trace tests show
that it correctly implements the mathematical and event semantics defined by the current
code.

The engine should be described as:

> A deterministic, boundary-synchronous hybrid simulator with analytic within-boundary
> membrane evolution and event resolution.

Within an outer boundary, membrane evolution and threshold-crossing times are solved
analytically and agree with an independent high-resolution RK4 reference. Across outer
boundaries, the engine intentionally uses discrete rules for propagation, decay,
refractory state, learning exposure, and stimulus scheduling.

Consequently, the engine is a validated implementation of its present hybrid model. It is
not yet established as a globally continuous-time, pure discrete-event simulator.

## Validation verdict

| Area | Result |
|---|---|
| Analytic segment solver | **VALIDATED** |
| Scheduler causality | **VALIDATED** |
| Same-time concurrency | **ORDER-SENSITIVE BY EXPLICIT RULE** |
| Event conservation | **VALIDATED**, using the strongest reconstructable multiset ledger |
| Network boundary refinement | **UNSUPPORTED** |
| Segment partition invariance | **VALIDATED** |
| Dual-FE causal state capture | **VALIDATED** |
| Overall characterization | **VALIDATED WITH DECLARED APPROXIMATION** |

The complete methods, semantic audit, numerical results, reproducing circuits, and test
commands are in
[`ENGINE_VALIDATION_REPORT.md`](ENGINE_VALIDATION_REPORT.md).

## What was independently validated

### Analytic membrane evolution

The engine's analytic segment solver was compared against a separate RK4 numerical
integrator that did not call or copy the production integration methods.

The numerical solution converged toward the analytic result at approximately fourth order
for membrane state. Numerically estimated crossing times converged at approximately second
order because of the oracle's interpolation method. The pure-integrator case matched
exactly.

This establishes that the engine correctly solves its frozen-drive differential equations
inside each boundary.

### Scheduler causality

Focused tests confirmed that:

- the earliest distinguishable threshold crossing is selected regardless of node order;
- all membranes advance to the same logical sub-boundary time;
- reset or changed input invalidates an obsolete predicted crossing;
- logical time never moves backward;
- events are not delivered before their causal parents;
- a crossing exactly at the boundary endpoint has a consistent ownership rule;
- tie-tolerance behavior is deterministic and visible in the trace.

No implementation bug was found in the audited causal scheduler.

### Independent network trace

A deliberately small priority-queue reference simulator, using RK4 integration and
explicit event scheduling, reproduced the production engine's complete causal spike trace
when configured with the production engine's declared one-boundary feedforward delay.

The maximum sub-boundary timing difference was approximately `2.3e-10`.

When the reference instead used continuous delivery, downstream events moved one boundary
earlier. This difference was fully explained by the production engine's declared
boundary-sized propagation delay rather than by an integration error.

### Dual-FE learning state

The validation confirmed that dual-FE learning reads the causal pre-reset accumulated
charge:

- `Iaccq` is captured when delivered excitation is frozen;
- reset does not replace that learning value with zero;
- FE uses the captured pre-reset value;
- FES uses each synapse's pre-update weight;
- learning occurs once for the causal spike;
- observing or serializing the simulation does not trigger an additional update.

One update was reconstructed manually. For `Iaccq = 765.478969764685`,
`theta = 1000`, `B = 5`, and `e = 0.001`, the independently calculated FE value was
`0.73969`, matching the engine log to better than `1e-12`.

This validates the timing and bookkeeping used by the dual-FE/FES rule. It does not, by
itself, prove that every learned network behavior is invariant under a future change in
global timing semantics.

## Declared approximations and remaining limitations

### Outer boundaries are part of the model

The outer boundary is not merely a display or observation frame. It currently controls:

- ordinary, pretrained, and basal delivery;
- conductance decay;
- refractory decay;
- activity and eligibility updates;
- learning opportunities;
- stimulus scheduling;
- one-spike and one-relay-emission guards.

The engine therefore combines continuous analytic evolution inside a boundary with
boundary-synchronous network propagation.

This architecture is scientifically usable as long as those boundary rules are treated as
part of the declared computational model.

### Exact WTA ties depend on stable node order

When competitors cross at distinguishable times, the earliest crossing wins independently
of declaration order. When competitors cross at exactly the same logical time within the
configured tolerance, the winner is selected by stable node order.

This is an explicit deterministic arbitration rule, not a hidden scheduler bug. It should
not be interpreted as a biological prediction about how exact physical simultaneity is
resolved.

Experiments sensitive to exact symmetry should either report this rule, introduce a
declared symmetry-breaking mechanism, or test outcome distributions across controlled
perturbations.

### Inhibitory relays emit at most once per boundary

If two drivers reach the same inhibitory relay in one boundary, the relay emits only once.
The second input is effectively rejected or coalesced because the relay is guarded after
its first emission.

The resulting absence is inferable from the event trace, but production events currently
do not include an explicit “second input rejected” record.

This is the most important remaining same-time semantic limitation. It does not invalidate
the current one-emission WTA model, but it would matter if a future scientific claim
requires:

- counting multiple same-time inhibitory inputs;
- emitting multiple inhibitory events at one timestamp;
- representing per-event inhibitory magnitude;
- distinguishing coalescing from cancellation.

The narrow future correction would be an explicit per-timestamp relay input/emission
ledger with a declared multiplicity rule. It should not be changed casually, because the
current presets rely on the one-emission behavior for WTA.

### Event conservation lacks globally unique causal IDs

Event conservation passed using a reconstructed per-boundary multiset ledger. Production
traces identify emitted edge IDs but do not assign globally unique identities to individual
causal events.

No missing or duplicated delivery was found within the information the engine exposes.
Perfect per-event auditing would require a monotonic causal-event identifier added as
passive instrumentation.

## Why physical boundary refinement is currently unsupported

A valid refinement experiment must run the same physical system at:

```text
dt
dt / 2
dt / 4
```

while preserving physical stimulus times, propagation delays, time constants, refractory
duration, learning exposure, and total simulated duration.

The current engine cannot express that experiment because it has no independent physical
`dt`. A one-boundary propagation delay, per-boundary decay, per-boundary learning, and
per-boundary stimulus schedule all change their physical meaning if the number of
boundaries is simply increased.

Running twice as many current boundaries is therefore not a legitimate half-timestep
experiment. The validation correctly reports network boundary refinement as
**UNSUPPORTED**, rather than manufacturing a pass or fail.

The weaker property of segment partition invariance was tested successfully: dividing one
uninterrupted analytic membrane interval into smaller analytic advances produces the same
final state. This validates the segment solver but does not validate the whole network
under boundary refinement.

## Minimal future design for true refinement testing

True network-level refinement would require:

1. an explicit physical-time unit and configurable `dt`;
2. membrane, conductance, refractory, trace, and eligibility constants expressed in
   physical time rather than boundary counts;
3. a delay queue keyed by absolute physical arrival time rather than fixed next-boundary
   buffers;
4. stimulus events scheduled at physical timestamps;
5. learning cadence defined by causal events or elapsed physical time;
6. tests comparing physical spike times and causal traces across progressively smaller
   observation boundaries.

This is an engine evolution project, not a parameter adjustment. It should be pursued only
if a scientific claim requires invariance to the current boundary construction.

## Impact on current scientific work

The validation supports the following claims:

- analytic voltage evolution is correct for the implemented frozen-drive equations;
- within-boundary threshold crossings are resolved at the correct analytic time;
- scheduler recomputation and reset invalidation are causally correct;
- the declared boundary-delayed network trace is reproducible by an independent reference;
- dual-FE/FES updates use the intended causal pre-reset state;
- the current engine is a coherent deterministic hybrid computational model.

The validation does **not** support the following stronger claims:

- the entire network is globally continuous-time;
- outer boundaries have no effect on network behavior;
- results are invariant under physical timestep refinement;
- exact same-time WTA outcomes are independent of the deterministic tie-break;
- one relay can represent multiple independent same-boundary inhibitory events;
- a recorded dashboard replay contains enough state for exact simulation continuation.

Current dual-FE consolidation findings remain legitimate findings about the implemented
hybrid model. They should not yet be presented as evidence that the same behavior is
invariant under a globally continuous-time implementation.

## Recommended handoff priorities

Given the remaining project time, the recommended sequence is:

1. Preserve the validation harness, tests, and detailed report in version control.
2. Use the validated hybrid-model description consistently in documentation and
   presentations.
3. Treat explicit physical `dt`, absolute-time delay queues, boundary refinement, and relay
   multiplicity as future engine work.
4. Continue current network and dual-FE experiments without changing timing semantics
   mid-study.
5. Add replay-to-live weight branching as a separate usability feature, clearly described
   as starting a fresh simulation from recorded learned weights rather than restoring
   transient state.

The timing engine should not be refactored merely to obtain a more attractive label. The
current implementation is defensible when described accurately, and the validation now
defines a clear boundary between proven behavior, explicit approximation, and future work.

## Final claim boundary

The strongest concise statement supported by the evidence is:

> The engine is a validated deterministic hybrid simulator. It analytically evolves
> continuous membrane state and resolves threshold events within each outer boundary, while
> using explicit boundary-synchronous rules for network delivery, decay, refractory state,
> learning exposure, and stimulus scheduling. Its results are correct for that declared
> model. Global continuous-time equivalence and network-level boundary-refinement
> invariance remain untested because the current architecture cannot yet express a
> physically equivalent refinement experiment.
