# Claude Prompt: Direct-Identity Tiled Cortical Columns Without Eor

## Objective

Implement and validate a new built-in tiled cortical-column topology that removes every
Eor neuron and preserves the identity of each local ordinary-E winner across the
hierarchical boundary.

The new topology keeps the existing 9x9 RGC surface, nine 3x3 L1 receptive fields, eight
ordinary competing E neurons per column, one coincidence C, and one local WTA/feedback I.
Each of the eight L1 ordinary E neurons projects directly and densely to all eight L2
ordinary E neurons.

Use this preset name unless a repository-wide naming conflict is discovered:

```text
tiled_cc_direct_identity
```

This is a new topology, not a feature flag and not a modification hidden inside
`tiled_cc`. Preserve the existing presets and their behavior.

## Scientific motivation

The existing Eor is trained with the same competitive signed-participation rule as an
ordinary pattern detector. After a long single-pattern dwell, one incoming Eor weight can
reach `theta` while all nonparticipating weights collapse to the dual-rule floor. If a new
pattern recruits a different local ordinary-E owner, its floor-weight event cannot make
Eor fire, and Eor cannot update because its learning is postsynaptic-spike gated.

Making Eor a fixed Boolean OR would remove that relay deadlock but would still erase the
local winner identity. A parent would receive only “this child column was active,” not
which learned local pattern won.

The ordinary-E WTA winner is already a sparse one-hot latent code. This topology transmits
that source address directly:

```text
3x3 patch -> one of eight L1 ordinary-E winners
one winner identity per child column -> L2 composition learner
```

Only one ordinary E should fire per active child column per boundary, so activity remains
sparse even though structural connectivity is denser.

## Dendritic orientation: do not reverse this

Use the canonical and currently implemented cortical interpretation:

```text
bottom-up/local driving evidence -> basal dendrite
top-down/higher-layer feedback   -> apical dendrite
```

For each non-top child column:

```text
local ordinary E[0..7] -> local C basal
parent ordinary E[0..7] -> child C apical
```

The current repository already uses higher-layer L2 ordinary E as the apical source.
Do not change L2 feedback into basal input.

The current topology has:

```text
one local Eor -> C basal edge                    # learned
eight parent ordinary E -> C apical edges        # structural, unweighted
```

Those eight apical edges are not physically collapsed into one edge. Their source IDs and
delivery counts remain distinct. The C firing gate computes a Boolean `any(apical)`, so
their functional permission is OR-like, but diagnostics retain which parent source or
sources delivered the permission.

The new topology removes Eor and replaces its single basal afferent with eight
source-distinct local ordinary-E basal afferents. This requires an explicit multi-basal C
contract described below.

## Exact topology

### Input surface

- One 9x9 RGC surface: 81 source neurons.
- Tile it into nine non-overlapping 3x3 patches in a 3x3 arrangement.
- Every RGC in a patch projects to every ordinary E in that patch's L1 column.
- Preserve the existing RGC-to-L1 pattern-detector learning and distance behavior.

### Each L1 column

Exactly:

- eight ordinary event-resolved plastic E competitors;
- one multi-basal coincidence C;
- one local WTA/feedback I;
- no Eor;
- no feature relays or per-feature gates.

Internal edges:

```text
E[i] -> I       relay_excitation
I -> E[i]       hard_reset_inhibition
E[i] -> C       basal_excitation
C -> I          relay_excitation
```

The ordinary E bank retains the existing first-spike-latency WTA. The first ordinary E
winner drives I, and I hard-resets the rest of the local E pool.

### L1 to L2 identity projection

Every L1 ordinary E projects to every L2 ordinary E:

```text
for each of 9 child columns:
    for each child E[0..7]:
        for each parent L2 E[0..7]:
            child E -> parent E feedforward
```

This creates 576 source-addressed L1-to-L2 feedforward edges:

```text
9 * 8 * 8 = 576
```

Do not pool, sum by child column before delivery, replace the source with a column ID, or
otherwise erase which ordinary E emitted the event. The receiving L2 cell owns a distinct
plastic weight for every `(child column, child winner)` source.

### L2 column

Exactly:

- eight ordinary event-resolved plastic E competitors;
- one multi-basal coincidence C;
- one local WTA/feedback I;
- no Eor.

The L2 ordinary E bank uses the same local WTA rule as L1. Each L2 ordinary E receives 72
feedforward afferents:

```text
9 child columns * 8 possible child winners = 72
```

The top L2 C receives basal events from its own eight ordinary E neurons, but it has no
parent and therefore no apical inputs. It is declared dormant and must never deposit
charge or fire. Retaining the dormant C makes the column motif recursively reusable when
a later L3 experiment gives L2 a parent.

### L2 feedback to L1 C

Every L2 ordinary E projects apically to every L1 C:

```text
8 L2 E * 9 L1 C = 72 apical edges
```

Keep each edge and source identity separately observable. Apical input is structural and
unweighted. Any unique current L2 apical event opens the Boolean apical side of the C gate;
duplicates must remain diagnostic but cannot deposit a second somatic impulse.

### Expected graph size

With eight E neurons in all ten columns:

```text
nodes = 81 RGC + 9*(8 E + C + I) + (8 E + C + I) = 181
edges = 1546
```

Derive and test the edge count from the builder rather than trusting this arithmetic
blindly. The expected components are:

```text
RGC -> L1 E                         648
L1 E -> L1 I                         72
L1 I -> L1 E                         72
L1 E -> L1 C basal                   72
L1 C -> L1 I                          9
L1 E -> L2 E direct identity         576
L2 E -> L1 C apical                  72
L2 E -> L2 I                           8
L2 I -> L2 E                           8
L2 E -> dormant L2 C basal             8
L2 C -> L2 I                           1
```

## Multi-basal coincidence-cell contract

Do not weaken the coincidence gate into ordinary flat summation.

Generalize the C implementation for this new topology so one C may own multiple learned
basal afferents and multiple unweighted apical afferents:

```text
B = a current or one-boundary-carried local basal event
A = any current parent apical event
C deposits at most once iff B AND A
```

Requirements:

1. Basal source IDs, edge IDs, weights, distance factors, current deliveries, and carried
   eligibility must remain aligned and source-distinct.
2. Local WTA should normally allow only one unique basal source per column per boundary.
   Do not silently discard an unexpected second source. Record it diagnostically and
   define deterministic behavior in tests.
3. A valid coincidence uses the weight belonging to the causal basal source.
4. A current basal event is preferred over a carried basal event, matching the current
   one-basal timing contract.
5. A committed deposit consumes the participating basal eligibility and is idempotent for
   the rest of that boundary.
6. Apical source identity remains diagnostic, but the permission gate is Boolean.
7. C still cannot fire from basal-only or apical-only input.
8. C still requires an active coincidence gate even if retained membrane voltage is
   already at threshold.

### Basal learning

Do not apply ordinary-E competitive `+1/-1` participation learning to the C basal vector.
That would recreate the Eor failure inside C.

When C fires:

- update only the causal participating basal weight;
- leave every nonparticipating basal weight unchanged;
- use the existing C learning family and parameters for the active weight;
- keep the per-basal ceiling at `theta`;
- keep every weight nonnegative and finite;
- log the causal source, edge, pre-weight, update, and post-weight.

The scientific intention is pattern-specific column certainty: multiple local owners may
each mature their own C basal association over time without destroying associations that
were learned earlier.

Do not make all C basal weights positive merely because one source participated.

## Detector ceiling remains mandatory

Preserve:

```text
e_weight_cap_frac = 0.5
w_i <= theta/2
```

This cap applies to RGC-to-L1 ordinary pattern detectors and direct
L1-winner-to-L2 ordinary pattern detectors. It does not apply to C basal weights, which
retain their role-specific ceiling at `theta`.

A single child winner event must not make a reset L2 detector cross threshold
instantaneously. Two or more coordinated child events may do so. With zero leak, repeated
events from one source can still accumulate over time; report that separately rather than
claiming the cap enforces two distinct sources across an unlimited temporal window.

## Feedback inhibition

For the first implementation, preserve the existing C-to-I feedback consequence:

- ordinary `E -> I` performs immediate same-boundary WTA;
- `C -> I` schedules the existing delay-1 feedback hard reset of the ordinary E pool;
- no Eor feedback target exists in this topology.

Do not add a feature-specific gate, a new inhibitory population, an output-suppression
credit, or hidden source-ID policy in this task. The previously discussed alternative of
allowing internal competition while suppressing only the outbound identity event remains
a separate standing problem. This topology task must make its retained pool-reset behavior
explicit and must not claim that the output-target question has been solved.

## Implementation constraints

- Read `git status`, the complete working-tree diff, and recent log before editing.
- Preserve all existing uncommitted work.
- Add a new preset; do not replace or silently mutate `tiled_cc`.
- Do not reintroduce any removed feature-gated topology.
- Do not add a user-facing model flag to switch Eor behavior.
- Do not key execution behavior from node ID prefixes such as `L1` or `L2`.
- Use archetype capabilities, edge projections, and validated column metadata.
- Keep edge IDs deterministic and stable.
- Extend serialization, branch/replay validation, topology export, layout, inspector, and
  weight lookup for multi-basal C cells.
- A saved/replayed multi-basal state must preserve every basal weight and causal source.
- Existing one-basal coincidence cells and existing presets must remain behaviorally
  unchanged.
- If a separate coincidence archetype is safer than changing the existing one, use one,
  but share the tested gate primitives rather than copying the scientific equations.
- Do not tune learning rates, thresholds, delay lengths, caps, or refractory values to
  obtain a preferred result.

## Dashboard contract

Expose the new preset in the topology selector with a clear label such as:

```text
Tiled CC Direct Identity · 9x9 · 8 E/column
```

Applying it must rebuild the graph and wipe learned state, consistent with other topology
changes. The dashboard must:

- show no Eor nodes or Eor weights;
- show all direct identity edges without merging their source IDs;
- display all basal weights of a selected C with their local ordinary-E sources;
- retain distinct apical source diagnostics;
- continue to support per-patch pattern assignment on the 9x9 input.

Do not make this the default dashboard preset until its causal and scientific tests pass.

## Required tests

### Builder and validation

Test:

- exact node roles, counts, and deterministic IDs;
- no Eor node or edge anywhere in the new graph;
- 648 RGC-to-L1 edges;
- 576 direct L1-ordinary-E-to-L2-ordinary-E edges;
- 72 L2-to-L1-C apical edges;
- eight source-distinct basal edges per C;
- no RGC edge directly to C or I;
- no cross-column L1 E-to-C basal edge;
- every non-top C has at least one apical source;
- the top L2 C is dormant with zero apical sources;
- duplicate basal/apical source-target edges are rejected.

Update validation carefully: existing `e_coincidence` graphs still require exactly one
basal edge unless they explicitly use the new multi-basal capability. Do not weaken the
global invariant for every old custom graph.

### Isolated multi-basal C

Test:

- every basal source can independently participate in a valid coincidence;
- the causal source selects the matching weight;
- noncausal basal weights are byte-identical after learning;
- basal-only and apical-only streams deposit zero charge;
- one-boundary eligibility works for every source;
- eligibility expires and cannot be reused;
- multiple/duplicate delivery behavior is deterministic and observable;
- one coincidence deposits at most once;
- C firing and deposit share the permitting apical spike's `tau`;
- branch/replay round-trips all basal weights exactly.

### Direct identity transmission

Drive each L1 ordinary E individually in a controlled test and verify:

- its source identity reaches every L2 ordinary E through the correct weight index;
- a different local winner selects a different set of source-indexed afferents;
- no Eor-like pooled source appears in delivery state;
- local WTA still permits only one ordinary winner per child column.

### Learning and pattern switching

At minimum:

1. Train one patch on `row 1` long enough to establish a stable L1 owner.
2. Switch the same patch to `col 1` and require a different ordinary-E owner under the
   declared turnover criterion.
3. Verify that both owners emit direct parent evidence immediately when they win; neither
   depends on an Eor recovery process.
4. Return to `row 1` and verify recall of the earlier owner.
5. Verify that C basal learning for one owner does not depress the other owner's basal
   weight.

Repeat the causal test for all four canonical patterns and more than one seed before
making a general turnover claim.

### Composition

Use at least two active patches. Verify:

- each patch contributes its actual ordinary-E winner identity;
- an L2 cell receives distinct simultaneous evidence from the two child columns;
- the `theta/2` ceiling remains enforced per direct identity afferent;
- a composition that changes one local pattern changes the L2 input identity even when
  the same patch locations remain active;
- parent learning can distinguish the changed composition in a declared experiment;
- L2 WTA and its feedback apical source remain deterministic.

### Regression

Run the complete suite. In particular, preserve:

- `rg_coincidence` causal timing and exact-alternation positive controls;
- existing `tiled_cc` and `tiled_cc_l1_4` graph goldens unless an intentional additive
  registry/UI change requires only their public preset list expectation to change;
- one-basal C unit tests;
- replay/branch compatibility checks;
- detector-cap tests;
- feature-gated topology removal.

## Required experiment report

Write a focused report under `docs/` that records:

- the final graph contract and counts;
- why Eor was removed;
- the current basal/apical orientation;
- how multiple basal weights and Boolean apical permission work;
- exact commands, seeds, patterns, and training lengths;
- local owner turnover;
- whether L2 distinguishes compositions that differ only in local winner identity;
- one-patch sparse-evidence behavior;
- multi-patch L2 confirmation cadence;
- C and feedback cadence;
- all negative results and timing anomalies.

Do not call the topology successful merely because activity reaches L2. Success requires
preserving source identity and demonstrating that a parent can use that identity to
distinguish at least one controlled pair of compositions.

## Deliverables

- New deterministic topology builder and preset registry entry.
- Multi-basal C implementation or isolated archetype.
- Engine routing, learning, serialization, replay, and diagnostics.
- Dashboard selector/layout/inspector support.
- Focused unit and integration tests.
- Full-suite regression result.
- Headless experiment artifacts and final report.
- Documentation updates to README, methodology, dashboard guide, and standing-problems
  register.

Do not commit or push unless explicitly requested.
