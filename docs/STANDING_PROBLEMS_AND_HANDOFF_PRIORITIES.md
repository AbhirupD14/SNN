# Standing Problems and Handoff Priorities

**Status:** living handoff document
**Last updated:** 2026-07-28

## Purpose

This is the single register of important problems that are known but not settled. It is
not an implementation specification and does not make candidate mechanisms part of the
scientific model. Narrow technical notes remain useful evidence; this document records
which questions still matter, what has actually been observed, and which work is worth
finishing before handoff.

Status labels:

- **Open:** the required behavior or mechanism has not been chosen.
- **Candidate identified:** there is a testable hypothesis, but it is not validated.
- **Implementation in progress:** code exists in the working tree, but integration and
  scientific acceptance are incomplete.
- **Deferred:** important, but too large or too weakly connected to the immediate
  consolidation experiment to justify implementing now.

## Recommended battle order

| Priority | Work | Value | Scope |
| --- | --- | --- | --- |
| ~~P0~~ **ADDRESSED** | ~~Resolve the Eor relay and inter-column information contract~~ | Closed two ways: `Eor` is now a **fixed** relay at `θ` in the classic columns (it can no longer strand a new local owner), and `tiled_cc_direct_identity` removes it entirely so the parent receives the **winner identity**, not mere column activity. Evidence: `docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md` | done |
| ~~P0~~ **FIXED** | ~~Resolve the `tau = 1.0` boundary-edge deadlock~~ | The event loop now drains crossings available at exactly `tau = 1.0`. Two-patch direct-identity: C spikes 0 → 500/1000, feedback resets 0 → 4000, L1 wins 2000/2000 → 1500/2000, runaway C cells 2 → 0. All goldens unchanged. `tests/test_boundary_edge_crossing.py` | done |
| P0 | Characterize frequency scaling and certainty coupling in the existing tiled topology | Auto-pacing now forces exact presentation-level alternation at the tested loop depths. The remaining question is whether alternation begins only after meaningful C confirmation and remains interpretable across patch count, pattern changes, and sparse evidence. Run the patch-count sweep 1..9 with feedback-off controls. | Medium |
| ~~P0~~ **ANSWERED** (diagnosis for the row below) | The feedback cadence measures loop latency and input pacing, not certainty | **Confirmed by construction.** `period = 2 × latency`: the diagnostic `tiled_cc_double_eor` preset (classic + one extra output relay) moved the period 6 → 8 on 8/8 seeds. The period-2 alias exists only for **odd** latency, which is why `1010` could never be secured by reseeding. Worse, the halving itself requires `latency % input_period == 0` (predicts 18/18 measured cases) — at the default `input_period=1` that holds only because 1 divides every integer latency. Slowing the input drives per-presentation emission to **1.00** in all three topologies. `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md` | Medium |
| ~~P0~~ **SOLVED** | ~~Reach exact `1010` alternation~~ | Set `input_period = loop latency` (2 / 3 / 4 for direct-identity / `tiled_cc` / double-Eor). Each volley's confirmation then lands exactly on its successor's drive packet and cancels it, so presentations alternate exactly — **strict `1010` on 12/12 seeds for all three loop depths**, depth-independent, no engine change. Condition is sharp: `ip = L+1` gives rate 1.00. `input_period` is now a runtime-only dashboard control (applies without rebuilding). `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md` | done |
| P1 | Make suppression consume the next *eligible evidence event* | No longer required for exact alternation (see above), but still the more general contract: it would make the cadence invariant at **every** pacing rather than only at `ip = L`, and robust to pauses and sibling-patch count. Contract stated in the rejected-direction report | Medium |
| P0 | Run the four-pattern, every-column consolidation experiment | Answers the internship's immediate scientific question | Medium |
| P1 | Validate the four-competitor capacity transition | Tests whether exhausting L1 capacity creates the need for another layer | Medium |
| P1 | Reconcile cap-free base learning with the intentional detector ceiling | Keeps the general learning equation distinct from the explicit θ/2 evidence-integration constraint | Small |
| P1 | Run a time-boxed NEST timing-validation prototype | Tests the circuit under a mature simulator before more scheduler-specific tuning | Small–Medium |
| P1 | Make the handoff reproducible and reconcile documentation | Keeps the next researcher from relying on obsolete topology descriptions | Small |
| Defer | Replace the hybrid scheduler with a full discrete-event simulator | Potentially valuable, but much larger than the immediate experiments | Large |
| Defer | Solve general multi-winner composition | Separate scientific problem requiring a new circuit contract | Large |

The P0 items should be treated as experiments with explicit acceptance criteria,
not as dashboard demonstrations.

---

## P0 — Frequency halving is not yet validated as a certainty signal

**Status:** Cadence reproducibility solved under auto-pacing; certainty coupling and
irregular-input semantics remain open

### Required behavior

For a sustained, predictable local pattern, an L1 cortical column should emit on half
of the eligible presentations. This should remain true when identical columns are added
spatially. It should not depend on node insertion order or on retuning a mature weight
budget for every topology.

“Half” means alternating accepted and absent evidence events:

```text
fire, silent, fire, silent, ...
```

A grouped cadence such as `fire, fire, fire, silent, silent, silent` has an average rate
of `0.5`, but it is not the required certainty signal. An irregular trace whose long-run
mean happens to approach `0.5` is likewise reduced activity, not demonstrated frequency
halving.

### Current architectural decision

Do not add another topology to address this problem, and do not remove or relax the
pattern-detector per-synapse ceiling `w_i <= theta/2`. That ceiling is intentional: one
afferent is insufficient evidence for a detector, while multiple simultaneous afferents
or repeated evidence over time may be integrated.

The existing tiled cortical-column graph is the object under test. Frequency halving is
intended as a column-level confidence and attention signal:

- high activity indicates active learning, active use, or guided attention;
- slower alternating activity indicates greater certainty that the current pattern is
  already learned;
- a feedback-suppressed presentation must produce no column winner/Eor event, so downstream
  layers receive a real absence of evidence.

### Evidence so far

> **Substantially superseded.** The `tau` race described below was one symptom of a more
> general cause, now measured and resolved: the cadence is set by the top-down loop latency
> `L` and the input pacing (`period = 2L`; suppression bites only when
> `L % input_period == 0`). Presenting one volley per resolved causal chain
> (`input_period = 0`, auto-derived) gives exact alternation on 12/12 seeds at three loop
> depths. See `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`. The paragraphs below are retained
> as the original observation.

In the original `rg_coincidence` circuit, the fixed L1 packet is approximately
`1.05 * theta`, so it crosses at `tau ~= 0.952`. The mature prediction/C path is
approximately `1.10 * theta`, crossing at `tau ~= 0.909`. Prediction therefore arrives
first on the suppressing boundaries and the circuit produces exact frequency halving.

In diagnostic tiled runs using the cap-free ordinary-E rule without the dashboard's
intentional detector ceiling, mature ordinary L1 E, Eor/L2 E, and C activity can all cross
at approximately `tau ~= 0.909`. The scheduler then sees a real numerical tie. Stable node
order selects ordinary L1 E first, after which the shared I relay has already fired and
cannot relay the later C input in that boundary. Frequency halving is lost. This diagnostic
race is separate from the sparse-evidence limitation created deliberately by the `theta/2`
detector ceiling.

A diagnostic-only intervention that rescaled mature L1 ordinary-E input from
`1.10 * theta` to `1.05 * theta`, leaving Eor/L2 at `1.10 * theta`, restored exact
halving in all nine L1 columns over the measured late window. This demonstrates the
race; it does **not** establish `1.05` as a scale-independent solution.

Spatial replication alone ought not change normalized mature charge. Changes in depth,
active evidence fraction, pipeline phase, or event multiplicity can change the race.
Therefore a fixed latency margin is a useful diagnostic and possibly a physical
parameter, but it is not yet the architectural invariant.

The current dashboard also exposes a separate sparse-evidence effect. With one active L1
patch, an L2 detector receives only one of its possible child-column afferents. Because
that afferent is intentionally bounded by `theta/2`, L2 can train slowly through temporal
accumulation but cannot mature into a one-event integrator. Its apical confirmations to
the associated L1 C cell therefore occur less frequently than accepted L1 presentations.
Maturing the C basal weight cannot create missing apical events.

The existing feedback path maps one C event to at most one later suppressed L1
presentation. If several accepted L1 events are required per L2/C event, accepted and
suppressed event counts cannot balance one-for-one. Longer training alone therefore does
not imply eventual exact one-patch alternation under the current semantics. With more
simultaneously active patches, L2 receives more evidence per presentation and may enter a
different cadence; dashboard observations have included grouped `3 fire / 3 silent`
behavior. That has the correct average but not the required temporal meaning.

### Deferred candidate: event-conserving predictive inhibition

Treat predictive inhibition as a counted local event rather than a per-boundary Boolean:

1. A `C -> I` event creates one pending predictive-suppression credit in that cortical
   column.
2. The next eligible ordinary-E competition/input event consumes exactly one credit and
   is suppressed before it can emit.
3. Two C events, including simultaneous events, create two credits; they are not
   collapsed into one relay spike.
4. Ordinary `E -> I` input still performs immediate same-event WTA and must **not**
   create a future predictive credit.
5. Credits are local, observable, and bounded by an explicit rule so a pause or topology
   fault cannot accumulate unbounded future suppression.

The intended steady-state invariant is event conservation. Let `A` be accepted L1
events, `S` suppressed eligible events, and `N = A + S` total eligible presentations.
If each accepted event eventually produces one credit and each credit suppresses one
later presentation, then away from startup/end effects `S ~= A`, hence
`A / N ~= 1/2`. This statement does not contain a mature weight fraction or column
count.

This candidate preserves the existing cells, local graph, and feedforward learning rule,
but it changes the relay/suppression semantics from lossy Boolean signaling to lossless
counted signaling. It is not the current implementation task. First characterize the
existing topology across active-patch scales; revisit event semantics only if that evidence
justifies doing so.

If counted suppression is rejected as too computational, the alternative is to model
real inhibitory synaptic delay, conductance, and decay. That is a valid continuous-time
direction, but then the suppression fraction will legitimately depend on physical time
constants and path delays. Immediate hard reset plus per-boundary Boolean relays cannot
simultaneously provide continuous-time fidelity and a parameter-free halving guarantee.

### Immediate scaling experiment

Keep the topology, `theta/2` detector ceiling, thresholds, learning rules, and feedback
path fixed. Sweep active-patch count from 1 through 9, including multiple spatial
arrangements where a count permits them, all four patterns, and a declared seed set.

For every active L1 column record:

- eligible input presentations;
- ordinary-E winner and Eor events;
- L2 winner events;
- C confirmations;
- feedback resets/suppressed presentations;
- accepted/input ratio;
- exact binary firing sequence, run-length distribution, and phase;
- owner identity and turnover after a pattern change.

Classify results as exact alternation, grouped half-rate, irregular reduced activity, or
no meaningful suppression. Report the minimum active-patch count—if any—at which exact
alternation becomes stable across seeds and arrangements. Single-patch sparse feedback is
an expected negative control for the present implementation, while still remaining an
unresolved requirement for any future general claim.

Scaling success would establish an operating regime for the existing topology; it would
not solve or erase the single-patch event-count mismatch. Scaling failure must be preserved
as a negative result rather than hidden by averaging or parameter tuning.

### Future mechanism acceptance experiment

Run the existing immediate-reset mechanism and the counted-event candidate side by side.
For each mechanism:

1. Test `rg_coincidence`, `tiled_cc`, and `tiled_cc_l1_4`.
2. Test all four 3x3 patterns, using the same pattern in every 3x3 receptive field.
3. Test multiple seeds and at least two mature-charge margins, including the exact-tie
   case.
4. Record per column: eligible inputs, accepted E events, C events, I inputs, credits
   created/consumed/dropped, and final pending-credit count.
5. Require the counted candidate's late-window L1 fraction to remain `0.5` within a
   declared finite-window tolerance without changing parameters by topology.
6. Include pauses, pattern changes, two simultaneous C inputs, and a deliberately missing
   feedback path as negative controls.

Do not promote the candidate if it achieves halving by silently building an inhibitory
backlog, suppresses novel patterns indefinitely, or relies on identifying layers/neurons
from string IDs.

### Deferred decision

The desired observable contract is now clear: exact alternating column-level absence of
evidence. What remains undecided is whether the existing circuit reaches that contract in
a sufficiently supported scaling regime, or whether a later change to event semantics
inside the same topology is required. Do not answer that question by introducing a new
topology or removing the intentional integration ceiling.

---

## P0 — Continuous timing and simultaneous-event semantics are underspecified

**Status:** Open; full rewrite deferred

### Current implementation

The event-resolved engine is a hybrid system:

- input and most synaptic delivery use integer outer-boundary buffers;
- membrane crossings within a boundary use analytic `tau` values;
- apical C input and hard-reset inhibition are zero-latency callbacks;
- exact/within-tolerance ties are resolved by stable node order;
- an I relay emits at most once per outer boundary, so later same-boundary inputs are
  discarded as causal events;
- every membrane is advanced and rescanned rather than each physical event living in a
  persistent priority queue.

These rules are deterministic, but some are numerical implementation choices rather than
declared scientific semantics. The tiled timing race exposed this distinction.

### Questions that must be settled

- Are equal-`tau` threshold crossings physically simultaneous, and if so are they batched
  before any zero-latency consequence is applied?
- Can zero-latency inhibition cancel a cell whose threshold crossing has the exact same
  timestamp, or only later crossings?
- Is event multiplicity conserved when two sources drive the same relay simultaneously?
- Which edges have physical delay, and is delay a property of an edge rather than of an
  outer-boundary implementation path?
- What state, if any, may an inhibitory event carry across an input pause or boundary?
- Which outcomes must be invariant to node list order and floating-point tolerance?

### Near-term recommendation

Do not attempt a full priority-queue simulator before the consolidation experiments.
First define and test the local simultaneous-event and multiplicity contract needed by
predictive inhibition. A later simulator can change its data structures without changing
those scientific semantics.

The larger scheduler design and multi-winner interaction are documented in
`docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`.

---

## P1 — Validate the timing circuit independently in NEST

**Status:** Required avenue; prototype only

[NEST Simulator](https://nest-simulator.readthedocs.io/) is a mature simulator for large
spiking point-neuron networks. Its standard execution combines fixed-resolution neuron
updates with spike-event communication, while selected precise-spiking models preserve
off-grid spike timestamps and can determine crossings analytically. It can therefore act
as an independent check on timing behavior currently entangled with this repository's
outer-boundary scheduler and stable-order tie handling.

NEST does not by itself settle the scientific contract. A standard interneuron receiving
two simultaneous inputs can sum both synaptic effects without necessarily emitting two
output spikes, and NEST will not automatically implement “one prediction buys one future
suppression.” The coincidence gate, fullness-error learning rule, and any counted
suppression mechanism may require a custom NEST/NESTML model.

### Required first investigation

Build a separate, headless, frozen-learning reproduction of one mature cortical column:

1. Use a precise-spiking integrate-and-fire model and explicit nonzero synaptic delays.
2. Reproduce the sustained-input frequency-halving protocol.
3. Sweep simulation resolution, pathway delays, and the equal-effective-charge case.
4. Replicate the identical column 1, 9, and 81 times without changing local parameters.
5. Record accepted L1 spikes, C and I events, suppression fraction, spike timestamps, and
   sensitivity to delay/resolution.
6. Determine whether ordinary continuous-time spiking yields robust halving or confirms
   that an explicit counted-suppression contract is required.

This prototype should be time-boxed and remain outside the dashboard. Its purpose is to
reject scheduler artifacts and inform the timing contract, not to reproduce every current
feature or learning rule.

### Possible later integration

The existing frontend could remain unchanged if a future `NestEngineAdapter` implemented
the backend's topology, control, step, and dynamic-state interfaces. This would still be
a backend port: network construction, pattern control, recordings, editable weights, and
custom plasticity would all need translation. Per-boundary extraction of every voltage
and weight could also eliminate NEST's performance advantage, since NEST is best suited to
running simulation in larger chunks.

Do not begin dashboard integration unless the isolated prototype demonstrates a scientific
advantage and profiling shows that the simulation engine—not serialization or frontend
rendering—is the important performance bottleneck. NEST is not currently installed in the
repository environment and no NEST adapter exists.

Official timing references:

- [Simulations with precise spike times](https://nest-simulator.readthedocs.io/en/latest/neurons/simulations_with_precise_spike_times.html)
- [Continuous-delay synapses](https://nest-simulator.readthedocs.io/en/latest/models/cont_delay_synapse.html)
- [Exact integration of neuron models](https://nest-simulator.readthedocs.io/en/latest/neurons/exact-integration.html)

---

## P0 — Four-pattern consolidation in every L1 column is not yet demonstrated

**Status:** Open experiment

### Scientific question

Each 3x3 receptive field should receive one of the existing four patterns. Presenting the
same pattern to all nine fields is acceptable. Determine whether each L1 cortical column
consolidates each pattern onto one ordinary E neuron, rather than merely producing a
transient winner.

Then repeat with four ordinary competitors per L1 column. The hypothesis is that all four
neurons become committed, leaving no free L1 neuron for a fifth representation and
therefore motivating recruitment in a higher layer.

### Acceptance criteria to define before running

- A quantitative maturity/consolidation threshold based on weights and repeatable firing,
  not dashboard color or a single winner event.
- Stable one-neuron assignment per pattern within each of the nine columns.
- Separation from the other three patterns and robustness across a declared seed set.
- No silent columns and no pattern monopolizing all competitors.
- A clear distinction between four-neuron capacity exhaustion and actual evidence that a
  new layer learns. Exhausting L1 does not by itself prove successful layer recruitment.
- Frozen-learning recall trials after training, including all four patterns and nearby
  distractors.

### Required comparison

Run identical protocols on `tiled_cc` (eight L1 competitors) and `tiled_cc_l1_4` (four L1
competitors). Report neuron assignments, weight totals, firing reliability, interference,
and unused-neuron count for every column. Aggregate accuracy alone is insufficient because
one failed column would be hidden by the other eight.

---

## Resolved decision — Eor relay and inter-column identity

**Status:** Addressed in both classic and identity-preserving column contracts

The original problem was real: treating `Eor` as a plastic competitor allowed a long
single-pattern dwell to drive one incoming weight to `1000` and the others to `0.001`.
After a pattern switch, the new ordinary-E owner could learn correctly while its
floor-weight packet failed to activate `Eor`, stranding otherwise valid local evidence.

The classic tiled presets now implement `Eor` as the relay its name implies. Every
ordinary-E-to-Eor afferent is initialized at `theta` and frozen, so any local winner
produces one column-active event. This removes turnover deadlock without changing the
intentional parent detector ceiling `w_i <= theta/2`. It also preserves the deliberately
compressed classic payload: the parent learns that the child column was active, not which
ordinary E won.

The `tiled_cc_direct_identity` preset supplies the other explicit contract. It removes
`Eor`, densely projects each of the eight local winner identities to the eight parent
detectors, and gives the local C eight independent basal associations. The direct-identity
acceptance sweep passed 12/12 ordered-pattern runs, retained 100% direct parent evidence,
and distinguished two compositions with the same active patch locations. Exact protocol,
results, and limitations are in `docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md`.

Neither choice is a proposed fix for certainty-coded frequency halving. Output suppression
under irregular pacing remains a separate problem: the classic circuit must suppress its
column-active event, while direct identity must suppress the outbound winner-addressed
event. Frequency is reserved for confidence/attention rather than winner identity.

---

## P1 — Sparse-evidence L2 bootstrap remains unresolved

**Status:** Open standing limitation; one-patch case is the scaling negative control

An inhibition audit found that the C-to-I hard-reset path works when exercised. In
single-patch isolation, however, L2 ordinary E receives roughly one of nine possible
afferents. Its active afferent is intentionally bounded by the pattern-detector ceiling
`theta/2`; this is not an accidental cap to remove. L2 may slowly learn and fire through
temporal accumulation, but one child event cannot make it a one-event integrator.

Consequently, apical prediction descends less frequently than the associated L1 column
accepts evidence. The L1 C basal synapse may mature, but C still cannot fire without an L2
apical event. Under the current one-C-event/one-reset mapping, this sparse confirmation
rate cannot generate exact `fire, silent` alternation.

Do not change topology, lower threshold, raise the cap, or normalize one sparse afferent to
threshold merely to make the dashboard halve. The immediate work is the fixed-parameter
active-patch scaling experiment described above. It must determine whether multiple
simultaneous child columns produce exact alternation, grouped half-rate behavior, or a
patch-count-dependent irregular cadence.

If later work returns to the single-patch requirement, it must address the mismatch between
accepted L1 evidence events and L2/C confirmation events while preserving the intentional
integration rule. More training by itself is not a proposed solution.

---

## P1 — Cap-free base learning and the intentional detector ceiling need reconciliation

**Status:** Implemented and covered by focused regression tests

Production ordinary-E learning uses a zero floor plus the neuron-wide fullness-error budget

```text
B = e_maturity_budget_frac * theta
p = B - sum(w)
delta_w proportional to eta * p * signal * distance_factor.
```

The intended fixed point is finite total incoming weight `sum(w) ~= B`; a specialist may
place most or all of that budget on one afferent and become a one-event integrator. C basal
weights and predictive-inhibitory weights retain their own role-specific bounds.

That general cap-free rule does not invalidate the explicit
`e_weight_cap_frac = 0.5` pattern-detector policy used by the dashboard/scaling protocol.
The latter is an intentional architectural constraint requiring integrated evidence, not
the historical universal cap that the base-rule work removes. Documentation, tests, and
configuration reporting must keep these two concepts separate.

Focused tests in `tests/test_ordinary_e_cap_free.py` cover convergence, non-oscillation,
weights above the historical 500 cap, and retention of C/PI bounds. The dashboard/scaling
contract deliberately adds `e_weight_cap_frac=0.5` to pattern-detector edges, while
one-afferent relay edges retain their separate `theta` ceiling. The pending scientific
work is the four-pattern consolidation protocol, not implementation of the base rule.

This change should not be credited with solving predictive-inhibition timing or sparse L2
confirmation. In particular, it must not silently bypass the declared `theta/2` detector
ceiling in the scaling experiment.

---

## Deferred — Multi-winner composition

**Status:** Deferred

The current local hard WTA admits only the first ordinary-E crossing. It cannot generally
recognize two independently learned components, such as row and column, and pass both to a
downstream composition learner. A fixed `delta_tau` admission window would merely replace
node-order sensitivity with tolerance sensitivity.

This is an important future problem but is not required to answer whether four individual
patterns consolidate in each column. Candidate mechanisms and acceptance experiments are
recorded in `docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`.

---

## Resolved maintenance — Documentation and handoff drift

**Status:** Reconciled for the 2026-07-28 handoff

The README, current methodology, dashboard boundary, and this problem register now describe
the six built-in presets, fixed classic Eor relay, direct-identity hierarchy, automatic
graph-derived pacing, `tau = 1.0` boundary drain, cap-free base rule, and the intentional
dashboard detector ceiling. Historical reports remain as evidence and are labeled or
framed by the current methodology rather than silently deleted.

Machine-readable direct-identity results live in
`experiments/direct_identity_results.json`; the remaining scaling protocol is specified in
`prompts/Claude_Existing_Topology_Frequency_Scaling_Prompt.md`. Future work should continue
to record exact commands, seeds, schedules, outputs, and implementation commits when a
standing question is resolved.

## Related evidence and design notes

- `Current_Implementation_Methodology_Equations.md`
- `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md`
- `docs/COINCIDENCE_IMPLEMENTATION_STATUS.md`
- `docs/COINCIDENCE_TURNOVER_TUNING.md`
- `docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`
- `docs/BOOLEAN_COINCIDENCE_OPEN_PROBLEM.md` (historical)
- `docs/LINEAR_WEIGHT_ABLATION_REPORT.md`
- `experiments/predictive_inhibition/FINAL_REPORT.md`

## Maintenance rule

When a problem is resolved, do not simply remove it. Add the chosen contract, the rejecting
evidence for alternatives, the acceptance command/results, and the commit that implemented
it. Then move it to a short “resolved decisions” section or a dedicated result report.
