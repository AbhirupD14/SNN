# Claude Prompt: Recursive Feature Gates Between Every Competition Layer

## Objective

Build and causally validate a new recursive feature-gated tiled topology. The successful
`tiled_cc_feature_gated` preset proves selective turnover only at the `RGC -> L1` boundary.
It intentionally stops there: one `Eor` per L1 module collapses local winner identity, L2 is
a plain WTA bank, and there are no feature gates between L1 and L2.

The new topology must apply the same computational motif at every competition boundary:

```text
upstream feature source
        -> fixed identity relay S[k]
        -> plastic competitors E[j]

S[k] -> C[k] basal
E[j] -> C[k] apical
C[k] -> If[k] -> S[k] hard reset
```

For the current 9x9 hierarchy this means:

1. Retain the validated nine pixel feature gates between each 3x3 RGC patch and its L1
   competitor bank.
2. Add identity-preserving feature relays and paired C/I gates between the nine L1 banks
   and the L2 competitor bank.
3. Prove that selective predicted-feature suppression and turnover occur at both levels.

Implement this as a new preset. Preserve `tiled_cc_feature_gated` as the local-only control
and preserve every older preset and artifact unchanged.

This is an implementation-and-execution task. Do not stop after proposing the graph.

## Why the current graph is insufficient

Read and accept these facts before editing:

- `tiled_cc_feature_gated` has nine `S/C/If` gates per L1 RF and successfully restores
  local turnover.
- Its L2 is deliberately a plain WTA bank with no C cells or feature gates.
- It feeds L2 through one L1 `Eor` per RF.
- A single Eor says only that a module emitted; it does not structurally preserve which
  L1 competitor owns the current pattern.
- Therefore its successful L1 result is not evidence that frequency halving, explaining
  away, turnover, or symbolic identity propagates into L2.

The new work must close those gaps. Do not merely restore the classic single column-wide
`C -> I -> entire E bank` path. That non-selective path was the failure that motivated
feature gating and must not run in parallel with the selective gates.

## Read and inspect first

Read:

- `docs/FEATURE_GATED_TILED_TOPOLOGY.md`
- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`
- `experiments/feature_gated_turnover.py`
- `experiments/microcircuit_turnover.py`
- `tests/test_tiled_cc_feature_gated.py`
- `tests/test_feature_gated_turnover.py`
- `backend/network_spec.py`, especially edge-kind validation, feature-gate builders,
  WTA-bank builders, tiled metadata, and variant validators;
- `backend/simulation.py`, especially fixed pretrained packet dispatch, delay-1 event
  output, C basal/apical delivery, the sub-boundary scheduler, relay idempotency, and hard
  resets;
- `snn/neurons.py`, especially `e_pretrained`, latency competitors, and coincidence
  dendrites;
- preset, serializer, layout, dashboard, replay-recorder, and replay-player contracts;
- the successful Stage A/B feature-gated artifacts.

Inspect the complete working-tree diff before editing. Preserve all unrelated uncommitted
work and generated scientific evidence. Do not commit unless explicitly asked.

## Exact recursive representation contract

### Boundary 1: RGC to L1

Keep the existing validated local module unchanged:

```text
RGC[pixel k] -> L1 input S[k]       fixed/pretrained event packet
L1 input S[k] -> every local L1 E   plastic feedforward owned by the target E
L1 input S[k] -> L1 input C[k]      learned basal
every local L1 E -> C[k]            Boolean apical permission
C[k] -> If[k] -> S[k]               paired selective hard reset

local L1 E -> Iwta -> local L1 E    separate same-bank WTA
```

Do not change the successful RGC-to-L1 timing or parameters.

### Boundary 2: L1 to L2

Do not collapse the eight possible L1 owners in a module onto one shared Eor before L2.
Instead, every ordinary L1 competitor is a distinct symbolic source channel.

For each L1 module `m` and each ordinary competitor `j`, create one fixed, non-plastic,
event-resolved inter-layer relay `S2[m,j]`. It is the identity-preserving output channel
for `L1[m].E[j]` and simultaneously one input feature of the L2 bank.

Required edges for every inter-layer feature `(m,j)` and L2 competitor `q`:

```text
L1[m].E[j] -> S2[m,j]    fixed relay excitation; one source spike produces one fixed
                         packet on the normal declared delay, unless S2 is suppressed
S2[m,j] -> L2.E[q]       plastic feedforward owned by L2.E[q]
S2[m,j] -> C2[m,j]       learned basal excitation
L2.E[q] -> C2[m,j]       Boolean apical permission
C2[m,j] -> If2[m,j]      structural relay excitation
If2[m,j] -> S2[m,j]      paired selective hard reset

L2.E[q] -> L2.Iwta       structural WTA drive
L2.Iwta -> L2.E[q]       same-bank hard reset
```

There are 9 L1 modules x 8 possible local owners = 72 identity-preserving L2 input
features in the eight-competitor topology. For any consolidated tiled pattern, normally
one owner per L1 module is active, so the L2 representation is a sparse nine-of-72 vector,
not nine anonymous Eor events.

The inter-layer `If2` must reset only `S2`. It must never reset an L1 competitor. This is
important: higher-level explaining-away may suppress a transmitted symbol, but it must not
erase or reopen the already-consolidated lower-level competition.

The L2 WTA I must target only L2 competitors. Feature-I and WTA-I nodes are distinct and
their reset targets are disjoint.

### Fixed relay semantics

Reuse the existing `e_pretrained` membrane behavior for `S2` if inspection confirms it is
the exact fixed one-event relay required. The dispatcher already treats fixed relay output
by source identity; however, the current `pretrained_excitation` validator describes an RGC
source only.

Generalize that structural edge contract narrowly to permit an ordinary event-resolved E
source to drive an `e_pretrained` relay, or introduce an equivalently narrow, explicitly
named fixed-relay edge kind if validation proves that safer. In either case:

- do not add a new neuron equation;
- do not make the relay plastic;
- do not make it an external input sink;
- do not change packet magnitude, delay, threshold, leak, or scheduler priority;
- one source spike must schedule exactly one relay packet;
- an inhibited relay must discard its pending packet before it fires;
- old RGC-to-relay graphs must remain behaviorally and serially unchanged.

Add a tiny isolated regression proving E-source -> fixed-relay behavior before composing
the full graph.

### Output convention and future depth

The identity relay is the hierarchy output convention. Do not use a single Eor for parent
learning in this preset.

If an Eor is retained solely for dashboard observability or compatibility, it must be
explicitly marked diagnostic, have no path into L2 learning or feature gates, and have no
effect on dynamics used by acceptance. Prefer omitting unused cells from the new graph.

Build this composition generically: a helper should accept an upstream competitor bank and
a downstream competitor bank and create one gated identity channel per upstream ordinary E.
It must not hard-code `L1`, `L2`, nine modules, or eight competitors. That helper is the
contract later used for `L2 -> L3`; do not implement L3 in this task.

## Timing and frequency-halving contract

At both boundaries a mature predicted feature must alternate between an accepted relay
event and a pre-spike suppressed relay event, apart from declared startup and transition
windows. The evidence must be causal, not inferred from a lower firing count.

For RGC-to-L1 and L1-to-L2 separately, record:

- eligible source events;
- relay events accepted and suppressed;
- owner spike tau;
- C deposit/spike tau;
- feature-I spike/reset tau;
- relay counterfactual crossing tau;
- whether reset occurred before relay emission;
- paired feature identity;
- novel-feature relay events;
- dropped, duplicated, or cross-feature reset events.

In a declared mature late window, report `accepted / eligible` for every predicted active
feature and the aggregate. The reference expectation is approximately 0.5 with exact
finite-window tolerance declared in the experiment. Do not obtain 0.5 by changing the
`1.05/1.10` margins, insertion order, or parameters.

This remains evidence under the current scheduler, not a claim that the standing
continuous-time/event-conservation problem is solved. Report equal-tau ties and relay
idempotency limitations honestly.

## L2 bootstrap and learning constraints

The identity-preserving representation expands L2 from 9 anonymous inputs to 72 possible
features, with only about 9 active for a tiled consolidated pattern. Audit the existing
normalized initialization and learning rule before running the experiment.

Do not silently raise initial weights, lower thresholds, prime L2, change FE budgets, or
special-case a seed. First run the unmodified parameters. A slow bootstrap is allowed;
declare a sufficiently long fixed timeout and measure it. If L2 never fires because the
sparse fan-in creates a genuine bootstrap deadlock, stop and report the measured deadlock
as a scientific result. Do not tune around it in this task.

## New preset and structural validation

Add one new preset with a clear stable key, preferably:

```text
tiled_cc_recursive_feature_gated
```

Suggested dashboard label:

```text
9x9 Recursive Feature-Gated CC (L1=8, L2=8)
```

Preserve the current local-only `tiled_cc_feature_gated` preset as a control. Do not
reinterpret old saved specs or artifacts under the new key.

Add a distinct validated topology variant/revision in metadata. All construction,
validation, serialization, layout, experiment discovery, and replay grouping must branch
on validated metadata, never an ID prefix or display label.

Validator invariants must prove at least:

- exactly one fixed source and one paired C/If chain per identity relay;
- one identity relay per ordinary child competitor;
- every identity relay feeds every ordinary parent competitor exactly once;
- every inter-layer C has exactly one paired basal relay and apical permission from every
  parent competitor exactly once;
- every inter-layer If resets only its paired identity relay;
- L1 and L2 WTA I targets remain local to their own competitor banks;
- no feature I resets a competitor, C, WTA I, or relay belonging to another feature;
- no single-Eor feedforward edge enters L2 in the recursive preset;
- no classic column-wide C/I reset path is present;
- the graph builder is deterministic and returns fresh objects;
- old presets retain exact graph counts and golden behavior.

Derive the exact node/edge totals from the final graph in code and document the arithmetic.
Do not copy a guessed total into tests. The structural tests must also inspect every
projection and pairing, not only totals.

Do not add the four-competitor recursive preset yet.

## Layout, dashboard, serialization, and replay

Expose the new preset without removing existing controls. Extend the metadata-driven
layout so the two feature-gate planes are distinguishable:

```text
RGC
RGC->L1 S/C/If gates
nine L1 competitor banks
L1->L2 identity S2/C2/If2 gates
L2 competitor bank
```

The dashboard must allow selecting/filtering the inter-layer feature channels without
rendering every unrelated lane when practical. Do not add new simulation controls or a
new renderer.

Round-trip the complete graph and metadata through preset storage, the serialization API,
and replay recorder/player. The replay must expose enough metadata to identify the child
module, child competitor, parent module, feature relay, paired C/If, and gate boundary.

## Required causal experiment

Create a headless experiment that reuses the existing recorder and canonical patterns.
Use the same pattern tiled into all nine RFs so each global pattern produces nine learned
L1 owner identities.

Run in strict stages.

### Stage 0: isolated inter-layer gate

With a minimal source/relay/C/If/parent-owner construction, prove:

- one child E spike produces one delayed fixed relay packet;
- without prediction the relay fires exactly once;
- with mature basal eligibility and parent apical prediction, C/If resets the relay before
  its crossing;
- a second, novel identity relay remains active;
- no event is duplicated and no other relay is reset.

Stop if this fails.

### Stage 1: preserve validated L1 behavior

Run the established seed-1 Stage A and Stage B feature-gated turnover checks against the
new topology. Require the same local claims: four distinct owners in every L1 module,
turnover on every switch, paired pre-spike feature resets, blank-module silence where
applicable, and no cross-patch inhibition.

The addition of the inter-layer gate must not change which mechanism establishes L1
turnover. Stop if the L1 result regresses.

### Stage 2: L2 acquisition and turnover

Present the canonical sequence `row1 -> col1 -> diag\\ -> diag/`, tiled across all nine
RFs, using fixed documented dwell periods. Do not freeze L1 unless the experiment declares
and reports a separate diagnostic phase; the primary run uses the intact recursive system.

For every pattern, record the nine active L1 owner identities and prove that the resulting
nine-of-72 L2 input feature vector is distinct and stable. Then require:

- L2 fires on realistic nine-module evidence without parameter changes;
- a stable L2 owner emerges for each pattern;
- all four patterns acquire distinct L2 owners;
- the incumbent turns over at every pattern switch;
- predicted inter-layer identity relays are suppressed before emission while novel
  identity relays remain available;
- L1 owner mappings remain stable while L2 feedback gates only their output relays;
- no gate resets another child/module/identity;
- WTA and feature inhibition remain causally separate.

If L2 fails to bootstrap or does not produce four-way turnover, preserve the replay and
metrics and report the negative result. Do not tune the model.

### Stage 3: frequency propagation

In mature fixed-pattern windows, measure the causal acceptance fraction at both gate
planes. Demonstrate whether halving propagates:

```text
RGC events -> L1 input relay acceptance
L1 winner events -> L2 identity relay acceptance
L2 competitor output frequency
```

Report startup transients, pipeline offsets, and exact eligible/accepted/suppressed counts.
Do not call alternating dashboard pixels proof; use event logs and causal reset traces.

Use seed 1 for this topology acceptance task. Do not launch the multi-seed recall/capacity
sweep here. That becomes the next prompt only after this causal hierarchy passes.

## Automated tests

Add focused deterministic tests covering:

- E-source -> fixed identity relay validation and runtime delivery;
- relay suppression before crossing and novel-relay survival;
- generic child-bank -> gated parent-bank composition;
- 72 unique L1-owner identity channels for the 8x8 current hierarchy;
- no Eor collapse into L2;
- exact feature-C basal/apical and If reset pairing at both boundaries;
- disjoint WTA and feature-gate reset targets;
- malformed cross-module, cross-identity, duplicate, missing, and whole-bank connections;
- metadata-driven serialization/layout/dashboard grouping;
- replay round-trip for both gate planes;
- old preset counts, golden behavior, and saved-spec compatibility;
- a short causal regression for the new topology without putting the full long experiment
  in pytest.

Run focused tests first, then the required headless stages, then the complete test suite and
`git diff --check`.

## Non-negotiable non-goals

Do not:

- alter neuron equations, learning equations, FE budgets, weight caps, thresholds, leak,
  refractory behavior, packet magnitudes, timing margins, or scheduler priority;
- add event-credit inhibition or solve the continuous-time scheduler in this task;
- restore the classic whole-bank hierarchical reset;
- reset child competitors from a parent feature gate;
- route by pattern names, expected winners, seed, or ID parsing;
- add L3, composition letters, noise tests, continuous interleaving, or four-competitor
  variants;
- run the frozen recall/capacity prompt before this topology passes;
- duplicate the recorder or dashboard renderer;
- delete negative results or unrelated work;
- commit unless explicitly asked.

## Required completion report

Report:

1. the exact graph and representation contract implemented;
2. how winner identity is preserved across L1 -> L2;
3. files changed and why;
4. derived node/edge totals and invariant-test results;
5. the isolated inter-layer gate trace;
6. L1 acquisition/turnover results;
7. L2 bootstrap, four-pattern ownership, turnover, and uniqueness results;
8. causal accepted/suppressed fractions at both feature-gate planes;
9. evidence that L2 gates never reset L1 competitors;
10. replay artifact and log paths;
11. focused and full test results plus `git diff --check`;
12. any negative result, scheduler limitation, or unresolved timing issue;
13. confirmation that no scientific parameter was tuned and no unrelated work was
    changed.

Do not commit unless explicitly asked.
