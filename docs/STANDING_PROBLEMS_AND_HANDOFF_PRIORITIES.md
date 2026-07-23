# Standing Problems and Handoff Priorities

**Status:** living handoff document
**Last updated:** 2026-07-22

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
| P0 | Test event-conserving predictive inhibition | Replaces topology-specific latency tuning with a falsifiable invariant | Medium |
| P0 | Run the four-pattern, every-column consolidation experiment | Answers the internship's immediate scientific question | Medium |
| P1 | Validate the four-competitor capacity transition | Tests whether exhausting L1 capacity creates the need for another layer | Medium |
| P1 | Finish cap-free-learning integration and regression evidence | Prevents an artificial cap from limiting one-event integrators | Small |
| P1 | Run a time-boxed NEST timing-validation prototype | Tests the circuit under a mature simulator before more scheduler-specific tuning | Small–Medium |
| P1 | Make the handoff reproducible and reconcile documentation | Keeps the next researcher from relying on obsolete topology descriptions | Small |
| Defer | Replace the hybrid scheduler with a full discrete-event simulator | Potentially valuable, but much larger than the immediate experiments | Large |
| Defer | Solve general multi-winner composition | Separate scientific problem requiring a new circuit contract | Large |

The first three items should be treated as experiments with explicit acceptance criteria,
not as dashboard demonstrations.

---

## P0 — Predictive inhibition has no scale-independent timing invariant

**Status:** Candidate identified

### Required behavior

For a sustained, predictable local pattern, an L1 cortical column should emit on half
of the eligible presentations. This should remain true when identical columns are added
spatially. It should not depend on node insertion order or on retuning a mature weight
budget for every topology.

### Evidence so far

In the original `rg_coincidence` circuit, the fixed L1 packet is approximately
`1.05 * theta`, so it crosses at `tau ~= 0.952`. The mature prediction/C path is
approximately `1.10 * theta`, crossing at `tau ~= 0.909`. Prediction therefore arrives
first on the suppressing boundaries and the circuit produces exact frequency halving.

In the cap-free tiled circuit, mature ordinary L1 E, Eor/L2 E, and C activity can all
cross at approximately `tau ~= 0.909`. The scheduler then sees a real numerical tie.
Stable node order selects ordinary L1 E first, after which the shared I relay has already
fired and cannot relay the later C input in that boundary. Frequency halving is lost.

A diagnostic-only intervention that rescaled mature L1 ordinary-E input from
`1.10 * theta` to `1.05 * theta`, leaving Eor/L2 at `1.10 * theta`, restored exact
halving in all nine L1 columns over the measured late window. This demonstrates the
race; it does **not** establish `1.05` as a scale-independent solution.

Spatial replication alone ought not change normalized mature charge. Changes in depth,
active evidence fraction, pipeline phase, or event multiplicity can change the race.
Therefore a fixed latency margin is a useful diagnostic and possibly a physical
parameter, but it is not yet the architectural invariant.

### Leading candidate: event-conserving predictive inhibition

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
counted signaling. It should therefore be introduced as a headless ablation before any
production change.

If counted suppression is rejected as too computational, the alternative is to model
real inhibitory synaptic delay, conductance, and decay. That is a valid continuous-time
direction, but then the suppression fraction will legitimately depend on physical time
constants and path delays. Immediate hard reset plus per-boundary Boolean relays cannot
simultaneously provide continuous-time fidelity and a parameter-free halving guarantee.

### Minimal acceptance experiment

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

### Decision still required

Is the model intended to implement a computational event contract (one prediction buys
one suppression), or a continuous biophysical circuit whose frequency is an emergent
function of explicit delays and time constants? This choice should be written down before
more timing constants are tuned.

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

## P1 — Sparse-evidence L2 bootstrap remains unresolved

**Status:** Open; intended operating regime must be declared

An inhibition audit found that the C-to-I hard-reset path works when exercised. In
single-patch isolation, however, L2 ordinary E receives roughly one of nine possible
afferents. Initial weight on that afferent is far below threshold, and ordinary E learns
only when it spikes. This creates a cold-start deadlock: L2 rarely spikes, the afferent
does not mature, apical prediction rarely descends, and C/I activity is correspondingly
rare.

Removing the ordinary-E per-synapse weight cap allows a **mature** sparse afferent to
become a one-event integrator. It does not by itself make a subthreshold initial afferent
fire and therefore does not solve cold bootstrap.

Before changing L2 excitability, decide whether single-patch evidence is a required use
case. If full-field presentation is the actual protocol, single-patch silence may be a
valid negative control. If a single learned patch must drive prediction, compare only
small, explicit bootstrap candidates: normalized/concentrated initialization, a declared
priming phase, or local learning from subthreshold causal evidence. Do not choose using a
dashboard trace alone.

---

## P1 — Cap-free ordinary-E learning needs final scientific acceptance

**Status:** Implementation in progress in the current working tree

Production ordinary-E learning is being changed from a per-synapse cap to a zero floor
plus the neuron-wide fullness-error budget

```text
B = e_maturity_budget_frac * theta
p = B - sum(w)
delta_w proportional to eta * p * signal * distance_factor.
```

The intended fixed point is finite total incoming weight `sum(w) ~= B`; a specialist may
place most or all of that budget on one afferent and become a one-event integrator. C basal
weights and predictive-inhibitory weights retain their own role-specific bounds.

Focused tests in `tests/test_ordinary_e_cap_free.py` cover convergence, non-oscillation,
weights above the historical 500 cap, and retention of C/PI bounds. Remaining handoff work
is to run the complete suite and the four-pattern consolidation protocol, record the
results, and ensure no UI or serializer still describes the FE budget as a hard cap.

This change should not be credited with solving predictive-inhibition timing or L2 cold
bootstrap; those are separate problems.

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

## P1 — Documentation and handoff drift

**Status:** Open cleanup

The repository contains valuable historical reports, but several describe removed
presets, capped ordinary-E weights, conductance-era inhibition, or earlier coincidence
semantics as though they were current. The README and methodology must be reconciled with
the final three-preset dashboard and the final cap/timing decision.

Before handoff:

1. Make this file the index of unresolved work.
2. Label historical documents prominently; do not delete evidence.
3. Update the README's current preset list, test inventory, and architecture summary.
4. Record exact commands, seeds, training schedules, and machine-readable outputs for the
   chosen consolidation and inhibition experiments.
5. Make one clean handoff commit only after the current overlapping Claude changes are
   reviewed and the full test suite passes.

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
