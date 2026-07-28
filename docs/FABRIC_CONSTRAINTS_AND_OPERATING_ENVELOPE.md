# Fabric Constraints and Operating Envelope

**Status:** canonical constraint register
**Last updated:** 2026-07-28

## Purpose

This document describes the system that currently exists: where it is supported by
evidence, what assumptions define its operating envelope, and which inputs or architectural
demands make it fail.

It is intentionally different from
[`STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`](STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md).
That document prioritizes unresolved work. This one records present capability boundaries,
including constraints that may be intentional and may never need to be “fixed.”

This is also not a new experiment. It synthesizes the current implementation, validation
reports, completed experiments, and preserved negative results. A claim is included only
with one of these evidence labels:

- **Structural:** follows directly from the graph, equation, or declared state machine.
- **Validated:** checked against an independent numerical/causal reference or focused
  invariant tests.
- **Measured:** observed under the exact protocol cited; scope does not extend beyond it.
- **Open:** plausible or required, but not yet established under the current live contract.

## Scope and reference contract

The primary subject is the classic tiled cortical-column fabric:

```text
RGC patch -> ordinary-E competition -> fixed Eor
                                         |
                                         +-> parent ordinary E
                                         +-> local C basal

parent ordinary E -> child C apical -> local I -> delayed feedback reset
ordinary E -------------------------> local I -> immediate WTA reset
```

The direct-identity variant is included as an explicit contrast where it changes a
constraint. Historical conductance/predictive-inhibition graphs are not treated as current
fabric behavior.

Unless a cited experiment states otherwise, “current live contract” means:

| Parameter or rule | Current value |
| --- | --- |
| learning | dual FE/FES enabled, `B=5` |
| ordinary-E learning rate | `eta=4` |
| C basal learning rate | `c_eta=16` |
| leak / refractory | `0 / 0` |
| detector ceiling | `e_weight_cap_frac=0.5`, so each afferent is at most `theta/2` |
| one-afferent ceiling | `relay_weight_cap_frac=1.0`, so Eor/C basal can reach `theta` |
| classic Eor | initialized at `theta`, non-plastic |
| feedback | delayed `C -> I -> E` hard reset enabled |
| presentation pacing | `input_period=0`, resolved from graph loop latency |
| execution | analytic event resolution inside boundary-synchronous propagation |

The base ordinary-E learning equation is cap-free; the `theta/2` ceiling above is a
separate architectural constraint applied to detector roles in this contract.

## Architectural role: encoder, not final classifier

The current fabric is intended primarily as a hierarchical edge/feature extractor and
compressor. A separate future network would consume its distributed latent output and
diverge into a larger semantic population for symbol creation. That decoder is not
implemented and cannot be tested within the remaining project time.

This changes how higher-layer results should be interpreted. A failure to classify at the
top of the current fabric is not automatically a failure of feature extraction. In the
two-tower experiment, both lower towers learned and recalled different V/A/7 codes; only
the two-scalar Eor interface into L3 was non-identifiable.

Compression still has an information boundary. A future decoder can expand a compact code,
but it cannot recover distinctions after the complete decoder-visible event streams have
become identical. The intended hypothesis is therefore a distributed latent constellation:
spatial/population source identity carries content, while output cadence remains available
for certainty or attention.

The full interpretation, information requirement, and future claim boundary are recorded
in [`ENCODER_DECODER_ARCHITECTURE_HYPOTHESIS.md`](ENCODER_DECODER_ARCHITECTURE_HYPOTHESIS.md).

## Executive operating envelope

| Capability | Current status | Supported envelope | Known break or claim boundary |
| --- | --- | --- | --- |
| Analytic membrane evolution | **Validated** | Frozen drive within one outer boundary | Does not establish globally continuous-time network behavior |
| Scheduler causality | **Validated** | Distinguishable crossing times, reset invalidation, endpoint drain | Exact ties use stable node order |
| Classic local WTA | **Structural + tested** | One ordinary-E winner per column per boundary | Cannot emit multiple independently supported features from one column |
| Classic Eor relay | **Reliable but lossy** | Any local winner produces one pooled column-active event | Erases which ordinary E won |
| Direct-identity transmission | **Measured** | Two-pattern turnover and controlled two-patch identity discrimination | Denser graph; four-pattern capacity and broader composition remain open |
| `theta/2` detector integration | **Structural + measured** | Two sufficient afferent events, or temporal accumulation at zero leak | One child cannot be a one-event parent input |
| C coincidence | **Structural + measured** | Current/carried basal evidence plus current parent apical permission | Missing/sparse parent activity starves C regardless of basal maturity |
| Exact `1010` alternation | **Measured** | Regular input auto-paced at the graph’s loop latency | Not robustly defined for arbitrary pauses, irregular input, or mismatched pacing |
| Two-tower graph scaling | **Measured at seed 1** | 393 nodes / 2162 edges; both lower towers mature and cold-recall V/A/7 | Two tower-level Eors make these L3 inputs identical; this constrains the latent interface, not lower extraction |
| Physical timestep refinement | **Unsupported** | Segment partition invariance only | No independent physical `dt`; more boundaries change the model |
| Exact replay continuation | **Unsupported** | Cold engine reconstructed from learned weights | Replay omits complete transient/RNG/queue state |

## What is robustly supported

### 1. The engine implements its declared hybrid model

**Evidence:** Validated.

The analytic segment solver agrees with an independent RK4 reference. A small independent
priority-queue simulator reproduced the production causal spike trace when given the same
declared one-boundary propagation delay. Scheduler recomputation, causal reset invalidation,
endpoint ownership, segment partition invariance, and dual-FE pre-reset state capture are
covered by focused validation.

The supported claim is narrow but strong:

> The engine correctly implements a deterministic hybrid model with analytic evolution
> inside outer boundaries and explicit boundary-synchronous network rules.

It is not supported to call the whole network globally continuous-time or invariant to
physical timestep refinement.

### 2. The classic column relay no longer deadlocks on owner turnover

**Evidence:** Structural + regression-tested.

Every local `E -> Eor` afferent is fixed at `theta`. Any one ordinary-E winner therefore
drives Eor from reset to threshold. The earlier failure—one Eor weight saturating while
inactive-owner weights fell to the floor—cannot occur in the current classic relay bank.

This establishes relay availability, not information preservation. Eor intentionally emits
the same event for every local owner.

### 3. The reusable graph rules compose to three layers

**Evidence:** Measured.

The two-tower experiment composed:

```text
two 9x9 RGC fields on one 9x18 sheet
-> eighteen L1 columns
-> two L2 columns
-> one shared L3 column
```

The resulting 393-node / 2162-edge graph validated using the existing column and
child-to-parent rules. Both lower towers learned distinct V/A/7 owners, all active L1 and
L2 milestones matured, fixed Eor reliability was 1.0, active non-top C cells reached
one-shot readiness, and all six tower-level L2 assignments were reproduced during cold
frozen recall.

This is evidence that the fabric can be replicated, connected, trained, and replayed at
that size. It is not evidence that its compressed messages remain sufficient at the next
layer.

### 4. Winner identity can be preserved when the direct-identity contract is selected

**Evidence:** Measured.

`tiled_cc_direct_identity` removes Eor, gives the parent a source-addressed weight for every
child winner, and gives C one causal-source-specific basal weight per local winner. In the
recorded acceptance work:

- 12/12 ordered two-pattern runs showed turnover, recall, direct parent evidence, and no
  depression of the first owner’s C association;
- a controlled two-patch comparison changed one local winner while keeping patch locations
  fixed, and L2 selected a different owner for the two compositions;
- the detector ceiling remained exactly `theta/2`.

The result establishes the importance of the transmitted alphabet. It does not establish
four-pattern capacity, arbitrary-depth composition, or multi-winner behavior.

### 5. Auto-pacing makes alternation reproducible under regular presentation

**Evidence:** Measured at three loop depths.

For a feedback loop of latency `L`, `input_period=0` resolves to `L`. Each confirmation
then lands on the successor presentation’s drive packet. The tested direct-identity,
classic, and double-Eor loops (`L=2,3,4`) produced strict presentation-level `1010` on
12/12 seeds at each depth.

That is a reproducible pacing contract. It is not yet evidence that firing frequency
tracks learned certainty.

## Known constraints and smallest breaking conditions

### C1 — Classic Eor erases local winner identity by design

**Classification:** Structural; directly demonstrated.

**Trigger:** A parent must distinguish two cases that activate the same child columns but
different ordinary-E owners inside those columns.

**Mechanism:** Classic Eor is a many-to-one mapping:

```text
E0 \
E1  \
...  -> one Eor event: “column active”
E7  /
```

The relay fixes turnover silence but does not carry the winner address.

This is acceptable when the current stage is used as an edge extractor and downstream
content remains distinguishable from the distributed set of active columns. It breaks only
when the decoder is asked to distinguish cases whose complete distributed Eor streams have
already collapsed to the same code.

**Smallest decisive reproducer:** In the two-tower V/A/7 experiment, both towers learned
three distinct L2 owners. Nevertheless, L3 received only
`T0L2c00Eor` and `T1L2c00Eor`. The complete ordered L3 input traces for V, A, and 7 were
byte-identical: 166 events per source, both sources co-occurring, interval 6, and identical
charge. One L3 neuron owned and recalled all three glyphs.

**Observable symptom:** Stable lower-layer mappings with a stable but non-unique parent
owner; `representation_not_identifiable`. This is a latent-interface failure, not evidence
that the lower feature extractor failed.

**What cannot fix it:** More dwell, lower `B`, a different seed, faster learning, or a
larger parent competitor bank. These change learning on the same input; they cannot create
missing source identity.

**Known alternative:** Direct identity preserves the address but increases edge density
and is a distinct transmission contract, not a tuning change to classic Eor.

### C2 — The `theta/2` ceiling requires integrated evidence

**Classification:** Intentional structural constraint; measured consequence.

**Trigger:** A parent detector receives only one active child afferent while reset.

**Mechanism:**

```text
w_i <= theta/2
one fresh source event < theta
```

One child cannot drive a parent across threshold in one event. At zero leak, repeated
events from that same source may accumulate across boundaries, so the ceiling does not by
itself prove spatial coincidence between distinct sources.

**Measured symptom:** In the direct-identity one-patch regime, L1 won about 92% of
boundaries while L2 won about 30%. L2 could bootstrap temporally but did not become a
one-event detector for that lone source. Parent apical permission and child C confirmation
were correspondingly sparse.

**Downstream consequence:**

```text
several accepted child events
-> one parent event
-> at most one C confirmation
-> at most one later feedback reset
```

A mature C basal weight cannot generate apical permission that the parent did not emit.

**What cannot fix it:** Longer training cannot make one capped afferent a one-event
threshold packet. Removing the cap would change the intended evidence-integration
architecture.

**Open envelope:** The active-patch-count sweep from 1 through 9 is still required to map
where parent confirmation becomes reliable across arrangements, patterns, and seeds.

### C3 — Feedback suppression depends on presentation timing

**Classification:** Structural law of the current boundary model; measured exhaustively
for the declared sweep.

With a volley every boundary:

```text
output period = 2 * loop latency
suppression is effective iff loop_latency % input_period == 0
```

This produced grouped `L fire / L silent` behavior. A fragile period-2 alias occurred only
for odd loop latency and was permanently destroyed by a short input gap.

Auto-pacing resolves the regular-input case by setting `input_period=L`, but the reset is
still consumed at a wall-clock boundary. It is not stored until the next arbitrary
eligible evidence event.

**Break triggers:**

- a hand-set period that does not align with `L`;
- variable or externally irregular presentation;
- pauses that alter phase;
- interpreting raw per-boundary frequency without resolving eligible presentations.

**Measured symptom:** At every non-aligning pacing in the cadence sweep, output rate was
1.00—the feedback reset canceled no input. At historical `input_period=1`, a confirmed
classic column could show grouped `111000` rather than alternation.

**Claim boundary:** Exact auto-paced alternation is established. Certainty coding,
attention coding, and robustness to irregular input are not.

### C4 — C cannot compensate for missing hierarchy evidence

**Classification:** Structural.

C requires:

```text
basal evidence from the local output
AND
current apical permission from a parent ordinary E
```

Basal evidence may carry for one boundary. Apical input is Boolean permission. A committed
coincidence deposits at most once, and C learning occurs only on a causal C spike.

**Break triggers:**

- the column has no parent;
- the parent is silent or under-driven;
- basal and apical events miss the eligibility window;
- C matures much more slowly than the ordinary-E owner.

**Expected behavior versus failure:**

- A top C with no declared parent is intentionally dormant: zero apical edges, deposits,
  spikes, and learning.
- A non-top C with missing parent activity is evidence-starved even if its basal weight is
  mature.
- An accidentally unwired non-top C is a graph validation error.

The current `c_eta=16` was chosen so C matures shortly after the E bank in the measured
tiled regime. That timing relationship has not been proven invariant to every topology,
evidence density, or learning schedule.

### C5 — Local competition is hard single-winner

**Classification:** Structural.

The first ordinary-E crossing drives the shared local I, which hard-resets the entire
ordinary-E bank. This provides deterministic sparse competition and prevents multiple
local winners in one boundary.

**Break trigger:** A task requires two independently learned features in the same column
to co-fire—for example, row and column specialists both recognizing a plus sign and
driving a third composition learner.

**Observable symptom:** The first supported feature suppresses every later candidate,
including a genuinely supported non-redundant feature.

**What cannot fix it:** Widening an arbitrary `delta_tau` admission tolerance would make
composition depend on numerical timing and could admit untrained near-ties. A new
multi-winner admission contract is required; changing scheduler data structures alone does
not provide one.

### C6 — Exact competition ties use stable node order

**Classification:** Explicit engine rule; validated.

Distinguishable crossing times are ordered causally and independently of node declaration
order. Crossings within the configured tie tolerance are resolved by stable node order.

**Break trigger:** Symmetric inputs and weights make scientific conclusions depend on which
exactly tied competitor wins.

**Observable symptom:** A deterministic owner that changes when declaration order changes,
despite equivalent evidence.

**Required interpretation:** This is deterministic arbitration, not a biological claim.
Tie-sensitive experiments must report it and use controlled perturbations or seed
distributions rather than interpreting the chosen index.

### C7 — One inhibitory relay emits at most once per outer boundary

**Classification:** Explicit engine rule; validated as implemented.

If several drivers reach one relay in the same boundary, the first emission closes the
relay for that boundary. Later inputs are coalesced/rejected and do not create additional
inhibitory events.

**Break trigger:** A mechanism requires counting or preserving multiple simultaneous
predictions, different inhibitory magnitudes, or one suppression credit per causal input.

**Observable symptom:** Fewer relay outputs than same-boundary relay inputs; the current
trace permits inference but has no explicit “second input rejected” event.

This is compatible with current one-emission WTA. It is not compatible with an
event-conserving multi-credit interpretation without a new declared multiplicity rule.

### C8 — Outer boundaries are scientific state, not display frames

**Classification:** Validated model boundary.

Outer boundaries control:

- ordinary, pretrained, basal, and feedback-reset delivery;
- conductance, trace, eligibility, and refractory updates;
- learning exposure;
- stimulus scheduling;
- one-spike and one-relay guards.

**Break trigger:** Treating a different number of boundaries as the same physical
experiment, or assuming a boundary can be subdivided without rescaling every physical
quantity.

The engine has no independent physical `dt`. Running twice as many boundaries changes
propagation, decay, learning, and presentation semantics; it is not a half-timestep
refinement.

**Supported:** Partitioning one uninterrupted analytic membrane segment.

**Unsupported:** Whole-network invariance across `dt`, `dt/2`, and `dt/4`.

### C9 — Capacity is bounded, and current multi-pattern robustness is not closed

**Classification:** Structural upper bound plus open empirical envelope.

A column with `N` ordinary competitors can expose at most `N` simultaneous distinct owner
identities. The four-competitor preset therefore has no spare ordinary E after four unique
assignments, even if the intended one-to-one mapping succeeds.

The current robust four-pattern claim remains open. A July 22 basic-consolidation sweep
completed 32 runs across two topologies, intact/feedback-disabled conditions, and eight
seeds, but recorded 0 full Basic passes: phases often had stable firing while one incumbent
absorbed multiple patterns. Subsequent relay, C-rate, and cadence changes mean that result
is a warning and rerun requirement, not a final verdict on the current live contract.

The direct-identity acceptance result is narrower: two sequential patterns, with owner
assignment determined by seed and presentation order. It must not be cited as proof of
four-pattern capacity or pattern-intrinsic neuron labels.

### C10 — Learned-weight replay is not exact continuation

**Classification:** Tooling/state boundary.

A recorded weight snapshot can initialize a fresh compatible engine. It does not restore
the exact source simulation because replay does not contain every membrane, pending event,
eligibility, conductance, refractory, trace, RNG, and scheduler state required for exact
continuation.

Cold-state frozen recall is therefore a valid retention test. It is not the same claim as
resuming the original dynamical trajectory.

## Common failure signatures

| Symptom | First likely checks | Typical classification |
| --- | --- | --- |
| No local winner | active input ownership, delivered charge, cap/leak, refractory | `no_firing_or_bootstrap_deadlock` |
| Stable winner but new patterns reuse it | per-pattern owner matrix and weight drift | `incumbent_absorbs_multiple_patterns` |
| Child learns but parent rarely fires | number of active child sources and `theta/2` charge | `L2_evidence_collapse` or `learning_event_starvation` |
| C basal is mature but C rarely fires | parent apical events and coincidence opportunities | `C_coincidence_starvation` |
| Classic Eor does not fire after a winner | fixed-bank invariant and delivered event | implementation regression; historical `Eor_maturation_bottleneck` no longer expected |
| Different lower codes share one parent owner | compare complete parent input signatures | `representation_not_identifiable` |
| Average rate is 0.5 but grouped | resolve `L`, `input_period`, and eligible presentations | cadence/pacing mismatch, not accepted halving |
| Feedback has no observable effect | test `L % input_period`, C events, reset landing | timing misalignment or confirmation starvation |
| Winner changes with node order | inspect exact/within-tolerance latency ties | `order_sensitive_tie` |
| Two expected inhibitory effects become one | count same-boundary relay inputs/emissions | relay multiplicity constraint |
| Frozen recall passes but live continuation diverges | distinguish weights from transient state | replay continuation overclaim |
| Non-finite or runaway C voltage | inspect endpoint drain and gate state | `nonfinite_or_numerical_failure`; the known `tau=1` case is fixed |

The stable vocabulary inherited from the experiment plans remains:

```text
no_firing_or_bootstrap_deadlock
stable_but_nonunique_owner
incumbent_absorbs_multiple_patterns
transition_residual_charge
learning_event_starvation
Eor_maturation_bottleneck
C_coincidence_starvation
L2_evidence_collapse
L2_mapping_collision
recall_drift_or_forgetting
order_sensitive_tie
capacity_exhaustion
representation_not_identifiable
nonfinite_or_numerical_failure
timeout_unclassified
```

Use `Eor_maturation_bottleneck` only for historical/plastic-Eor data or a regression that
violates the current fixed-bank contract. Use `representation_not_identifiable` when fixed
Eor fires correctly but its pooled alphabet is insufficient.

## Robustness that remains unestablished

| Question | Current evidence | Needed boundary test |
| --- | --- | --- |
| Does exact alternation track C certainty rather than auto-pacing alone? | Exact cadence is reproducible; semantic coupling is not shown | Patch-count 1–9 with feedback-off controls and C maturity alignment |
| Does suppression survive irregular input? | Known timing dependence predicts fragility | Pauses, jittered presentation times, missing volleys, pattern switches |
| Do all nine classic L1 columns retain four unique patterns under the live contract? | Open; earlier sweep failed its full gate | Fresh eight-seed intact/control rerun with frozen cold recall |
| Is the four-E preset a clean capacity transition? | Structural capacity is four; successful allocation is not established | Four-pattern mapping plus fifth-pattern challenge |
| How noise-tolerant are learned mappings? | Not established | Frozen clean/noise probes with declared bit flips |
| How short can a pattern dwell be before turnover fails? | Not mapped under current contract | Equal-exposure dwell sweep with update/event counts |
| Is behavior invariant to pattern order? | Direct identity is explicitly order-dependent | Canonical/reverse order and limited declared permutations |
| Does classic composition work with more than two child columns? | Two-child Eor alphabet collapsed | Vary active-child subsets without changing hidden-only identity |
| Does direct identity scale to the two-tower L3 task? | One-level controlled discrimination succeeded | Separate authorized topology experiment and edge-cost accounting |
| Can one column recognize simultaneous component features? | Structurally blocked by hard WTA | Dedicated multi-winner circuit contract and negative controls |
| Are findings globally continuous-time? | Hybrid model validated only | Physical-time model, absolute-time delays, refinement, independent simulator |
| Are two-tower conclusions seed-robust? | Structural signature collision at seed 1 stopped sweep | No seed sweep needed for identical inputs; lower-tower learning robustness remains separate |

## Interpretation rules for future experiments

1. **Stable firing is not unique representation.** Always report the owner matrix across
   patterns and the number of distinct owners.
2. **Maturity is not identifiability.** A parent can reach threshold reliably on an input
   alphabet that contains no class distinction.
3. **Average half-rate is not alternation.** Record the exact eligible-presentation
   sequence and run lengths.
4. **Suppression means absent outbound evidence.** A lower potential or delayed event is
   not equivalent to the required missing winner/output event.
5. **A silent parent does not diagnose L1 learning.** Separate lower ownership, relay
   throughput, parent evidence density, C opportunities, and feedback.
6. **Cold recall and continuous recall are different tests.** Record which transient state
   was reset and assert frozen weights did not change.
7. **Structural collisions end parameter searches.** Once complete parent input signatures
   are identical, do not tune learning rates or dwell to manufacture separation.
8. **Dashboard traces are observations, not acceptance criteria.** Scientific metrics use
   every engine boundary or eligible presentation and preserve resumable artifacts.

## Maintenance rule

When new evidence changes this envelope:

1. record the exact graph, parameters, seed set, schedule, and commit;
2. label the result structural, validated, measured, or open;
3. add the smallest reproducing trigger and observable symptom;
4. state what the result does **not** establish;
5. distinguish a repaired implementation defect from an intentional constraint;
6. link the machine-readable artifact and regression test;
7. update the executive table rather than appending a contradictory paragraph.

Move desired future mechanisms to the standing-problems document. Keep this document about
the behavior and limits of the fabric that actually ran.

## Evidence index

- [`ENGINE_VALIDATION_REPORT.md`](ENGINE_VALIDATION_REPORT.md) — full numerical and causal
  methods/results and validated hybrid-engine claim boundary.
- [`FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`](FEEDBACK_CADENCE_AND_LOOP_LATENCY.md) — loop law,
  pacing sweep, auto-pacing resolution, and certainty limitation.
- [`TWO_TOWER_COMPOSITION.md`](TWO_TOWER_COMPOSITION.md) — lower-tower success and measured
  L3 representation collision.
- [`ENCODER_DECODER_ARCHITECTURE_HYPOTHESIS.md`](ENCODER_DECODER_ARCHITECTURE_HYPOTHESIS.md)
  — extractor/compressor role, proposed divergent semantic stage, and information boundary.
- [`DIRECT_IDENTITY_TILED_TOPOLOGY.md`](DIRECT_IDENTITY_TILED_TOPOLOGY.md) — address
  preservation, multi-basal C, controlled identity discrimination, and scope limits.
- `experiments/runs/basic_consolidation/aggregate_summary.json` — machine-readable
  historical consolidation sweep result; the current claim boundary is summarized above.
- [`EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`](EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md)
  — why hard WTA blocks simultaneous component recognition.
- [`STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`](STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md)
  — unresolved work and candidate mechanisms.
- [`../Current_Implementation_Methodology_Equations.md`](../Current_Implementation_Methodology_Equations.md)
  — authoritative current equations and architecture.
- [`../prompts/Claude_Final_Experiments_Prompt.md`](../prompts/Claude_Final_Experiments_Prompt.md)
  Task 3 — preserved failure taxonomy and the unexecuted stress-matrix design.
