# Claude Prompt: Feature-Gated Tiled Cortical-Column Topology

## Objective

Add one new eight-competitor 9x9 topology that restores the feature-specific inhibitory
microcircuit which enabled turnover in `rg_coincidence`.

The current tiled topology connects each 3x3 RGC patch directly to an ordinary-E
competitor bank and then uses one shared C/I path to reset the whole bank. That removed
the smaller circuit's ability to suppress an explained **input feature** while leaving
novel features active.

The new topology must insert nine fixed feature relays between each 3x3 RGC patch and its
L1 competitor bank. Every feature relay has its own paired C/I gate. The L1 competitor
bank retains a completely separate WTA I. When a trained local owner predicts its active
features, the paired feature gates suppress only those feature relays; novel feature
relays remain available to recruit a different competitor.

Implement this as a new topology/preset. Preserve `rg_coincidence`, `tiled_cc`, and
`tiled_cc_l1_4` unchanged as historical controls. Do not replace or silently reinterpret
their saved artifacts.

Start with eight L1 competitors only. Do not add the four-competitor variant until this
new topology passes the single-RF causal turnover experiment.

## Read and inspect first

Read:

- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`
- `docs/BASIC_CONSOLIDATION.md`
- `experiments/microcircuit_turnover.py`
- `tests/test_microcircuit_turnover.py`
- `backend/network_spec.py`, including all archetypes/edge kinds, cortical-column
  builders, tiled metadata, validation, and preset registries;
- `backend/simulation.py`, especially event construction/dispatch, fixed pretrained
  packets, C dendrites, apical delivery, relays, hard resets, dimension resolution,
  topology serialization, and layer ordering;
- `snn/neurons.py`, especially `ConductanceLIFNeuron`, `ExcitatoryNeuron`,
  `CoincidencePyramidalNeuron`, `SourceNeuron`, and `InhibitoryNeuron`;
- `backend/layout.py`, `backend/presets.py`, and `backend/dashboard_config.py`;
- tiled topology/layout/dashboard/serialization tests;
- the implemented replay recorder if present.

Inspect the complete working-tree diff before editing. The tree contains uncommitted
recorder, consolidation, cap-free-learning, preset-cleanup, and dashboard work. Preserve
all of it. Do not delete or rewrite existing experiment artifacts.

## Scientific reference behavior

`experiments/microcircuit_turnover.py` is the local causal oracle. It demonstrates that
the smaller `rg_coincidence` circuit can learn the canonical sequence

```text
row 1 -> col 1 -> diag \ -> diag /
```

because every pattern shares center feature 4, while each switch introduces two novel
features. The mature center feature gate suppresses the shared relay, novel relays remain
active, and a different competitor takes ownership.

The new topology must reproduce that mechanism independently inside every 3x3 RF. Do not
substitute a weight cap, stronger leak, scheduler priority, centralized winner override,
label-aware routing, or pattern-specific code.

## Exact local module

For each of the nine 3x3 patches, create one L1 recognition module with:

- nine fixed event-resolved feature relays `S[k]`, one for each RGC/pixel in the patch;
- eight ordinary plastic event-resolved pattern competitors `E[j]`;
- one shared output relay `Eor`, retaining the current tiled output convention for now;
- one **WTA-only** relay `Iwta` for the eight pattern competitors;
- nine feature coincidence cells `C[k]`, one paired with each feature relay;
- nine feature inhibitory relays `If[k]`, one paired with each feature relay.

Required edges for local feature `k` and competitor `j`:

```text
RGC[k] -> S[k]       pretrained_excitation
S[k]   -> E[j]       feedforward, plastic weight owned by E[j]
S[k]   -> C[k]       basal_excitation
E[j]   -> C[k]       apical_excitation, zero-latency permission
C[k]   -> If[k]      relay_excitation
If[k]  -> S[k]       hard_reset_inhibition

E[j]   -> Iwta       relay_excitation
Iwta   -> E[j]       hard_reset_inhibition
E[j]   -> Eor        feedforward
```

Important invariants:

- `If[k]` targets exactly `S[k]`, never another feature and never the competitor bank.
- `Iwta` targets exactly the eight ordinary competitors, never feature relays, C cells, or
  Eor.
- No ordinary E drives a feature I directly.
- No feature C drives `Iwta`.
- Feature inhibition and WTA therefore cannot consume the same one-shot relay event.
- Each feature C has exactly one basal source—its paired `S[k]`—and receives apical
  permission from all eight ordinary competitors in the same recognition module.
- An L1 winner predicts local features through explicit apical edges; do not read a global
  winner variable.
- RGC sources remain stateless and uninhibited. Their paired feature relays are the
  suppressible objects.
- The nine feature relays remain fixed/pretrained; the eight pattern competitors retain
  the current cap-free FE learning rule.
- Reuse the existing `e_pretrained`, `e_latency_competitor`, `e_coincidence`, and
  `i_relay` archetypes and existing edge semantics wherever they exactly fit. Do not add a
  new neuron class merely to rename one of these roles.

The current one-C/one-I tiled L1 column is **not** to remain active in parallel. In this
new topology it is replaced by the nine feature C/I gates plus the separate WTA I. Leaving
the old column-wide C -> I -> entire-bank reset active would reintroduce the mechanism this
topology is intended to ablate.

## Timing contract

Copy the successful small-circuit causal timing; do not invent a scheduler priority:

- a feature relay receives the existing fixed pretrained packet and retains the existing
  `pretrained_exc_margin` behavior;
- an ordinary L1 pattern competitor uses the existing plastic E dynamics and maturity
  budget;
- when a mature local owner supplies apical permission and a feature C has valid paired
  basal eligibility, C/If must reset the prospective paired feature relay before that
  relay's later crossing on a suppressing boundary;
- novel feature relays without mature valid C gating must remain active;
- equal-time behavior remains governed by the existing scheduler and must be logged, not
  overridden by node ordering chosen to force the result.

Add an explicit causal test/trace comparing:

- local-owner spike tau;
- paired C deposit/spike tau;
- feature-I spike tau;
- paired feature relay's counterfactual crossing tau;
- actual hard-reset tau and whether the feature relay had already fired;
- novel feature relay spike times.

A suppressing event only counts if it preempts the paired feature relay before emission.
Resetting a relay after it already emitted is a failure.

## Higher layer for this first topology

Retain one top L2 recognition column with:

- eight ordinary plastic competitors;
- one Eor output;
- one WTA-only I;
- feedforward input from the nine L1 Eor outputs using the existing plastic feedforward
  rule.

Do not add a top C with no parent merely to satisfy an old builder invariant. Do not send
L2 apical feedback into the L1 competitor banks in this first feature-gated topology; the
local L1 owner drives its nine feature gates. This isolates and tests the input-feature
turnover mechanism.

The existing single Eor per L1 column is intentionally retained only to limit this task.
It may discard local winner identity and is not evidence of solved hierarchical
composition. Document that limitation; do not solve it here.

## Expected structural size

For the eight-competitor 9x9 topology described above, independently derive and test the
exact counts. The expected design is:

```text
81 RGC sources
81 feature relays + 81 feature C + 81 paired feature I
9 * (8 ordinary L1 E + 1 Eor + 1 WTA I)
1 * (8 ordinary L2 E + 1 Eor + 1 WTA I)
= 424 neurons
```

Expected explicit edges:

```text
81   RGC -> feature relay
648  feature relay -> local L1 ordinary E
81   feature relay -> paired feature C basal
648  local L1 ordinary E -> local feature C apical
81   feature C -> paired feature I
81   paired feature I -> paired feature relay reset
72   L1 ordinary E -> local Eor
72   L1 ordinary E -> local WTA I
72   local WTA I -> L1 ordinary E reset
72   nine L1 Eor -> eight L2 ordinary E
8    L2 ordinary E -> L2 Eor
8    L2 ordinary E -> L2 WTA I
8    L2 WTA I -> L2 ordinary E reset
= 1932 edges
```

If inspection proves a count above is inconsistent with the exact graph, stop and explain
the discrepancy before silently adding nodes or edges. Do not include display-only or
implicit connections in the scientific count.

## Topology metadata and validation

Use explicit metadata, never ID parsing, for:

- input/patch/grid shape;
- recognition-module membership;
- feature index and paired pixel/RGC;
- feature relay/C/I roles;
- ordinary-E/Eor/WTA-I roles;
- parent/child feedforward membership;
- projection family.

Add pure builder functions for the feature gate and recognition module. They must return
fresh nodes/edges and composable handles like the existing builders.

The topology must round-trip through validation, engine construction, serializer, preset
storage, and dashboard layout without losing metadata. Extend validation with exact local
invariants for this new variant. Do not weaken the existing tiled validator or allow
malformed old tiled graphs.

Prefer a declared tiled topology variant/model field over scattered preset-name checks.
Dimension/layout/serializer code should branch on validated topology metadata, not string
prefixes.

## Preset and dashboard exposure

Register one new built-in preset named clearly, preferably:

```text
tiled_cc_feature_gated
```

Expose it in the dashboard topology selector with an honest label such as:

```text
9x9 Feature-Gated CC (L1=8)
```

The intentional public preset set becomes the existing three plus this new fourth preset.
Update exact registry-contract tests and current documentation accordingly. Do not restore
any removed legacy presets or controls.

Add a readable metadata-driven layout:

- RGC at the bottom;
- feature relays directly above their paired RGC positions;
- paired feature C/I visibly associated with each relay but spatially separated enough to
  inspect edges;
- nine L1 recognition modules above the feature layer;
- L2 above L1.

Do not add winner rings or new dashboard configuration controls. Existing per-column
raster/charge filtering should remain usable; extend grouping metadata so a user can
select a local module and inspect its feature relays/gates without rendering unrelated
lanes when practical.

## Non-negotiable non-goals

Do not:

- change any existing preset's graph or behavior;
- reintroduce the ordinary-E weight cap;
- tune `1.05`, `1.10`, thresholds, leak, eta, C eta, refractory, or timing tolerance to
  make the new test pass;
- implement counted suppression, scheduler priority, conductance inhibition, or a true
  priority-queue simulator;
- add the four-competitor version;
- solve winner-specific Eor output/composition;
- add L3/two-branch symbols;
- run the old large Basic sweep against this topology before the local causal test passes;
- delete negative Basic artifacts.

## Required focused tests

Add tests proving:

1. Exact 424-node/1932-edge counts and deterministic construction.
2. Exactly 81 RGC, 81 feature relays, 81 feature C, 81 feature I, nine L1 recognition
   modules, and one L2 recognition module.
3. Every RGC maps one-to-one to its feature relay and every relay stays inside one patch.
4. Every feature relay feeds all and only the eight ordinary E in its local L1 module.
5. Every feature C has one paired basal relay and exactly eight local ordinary-E apical
   sources.
6. Every feature I is driven only by its paired C and resets only its paired relay.
7. Every WTA I is driven only by its own ordinary-E bank and resets exactly that bank.
8. Feature I and WTA I are distinct nodes with disjoint reset targets.
9. No old column-wide L1 C -> shared I -> whole-bank predictive reset exists.
10. L1 Eor-to-L2 feedforward connectivity is complete and no L2-to-L1 apical feedback is
    present in this first variant.
11. Validation rejects cross-patch feature edges, swapped C/I pairs, missing/duplicate
    basal or reset edges, WTA/feature-I role mixing, and duplicate pixel ownership.
12. Engine construction/stepping is deterministic and existing neuron equations/classes
    are reused unchanged.
13. Serialization/preset save-load preserves all new metadata and weights.
14. Layout and dashboard selector expose the new topology without changing the previous
    three.
15. Every previous preset's structural and behavioral tests remain unchanged.

## Required causal acceptance experiment

Create a headless experiment using the implemented replay recorder when available. Do not
use the browser/server.

### Stage A — one active RF

Use one selected 3x3 patch and the canonical sequence:

```text
row 1 -> col 1 -> diag \ -> diag /
```

For seed 1, train with fixed, declared dwell windows sufficient to reproduce the existing
`microcircuit_turnover.py` reference protocol. At each switch record:

- previous and new ordinary-E owner;
- early and final owner counts/dominance;
- shared-center feature relay spikes;
- both novel feature relay spikes;
- all feature C/I events and paired resets;
- tau ordering and whether resets preempted relay emissions;
- local WTA events separately;
- feature-C basal weights and ordinary-E incoming weights;
- latency ties and dropped/duplicate relay events.

Acceptance requires:

- a consolidated owner for each pattern;
- four distinct ordinary-E owners in the active L1 module;
- turnover at every switch;
- shared-center relay suppressed relative to both novel relays during every switch window;
- both novel relays remain active;
- every counted feature reset is paired/local and pre-spike;
- no feature-I/WTA-I cross-talk;
- no other L1 module learns or resets when its input patch is blank.

Compare these measurements against `microcircuit_turnover.py` semantically, not by
requiring identical arbitrary winner IDs.

### Stage B — nine independent RFs

Only after Stage A passes, present the same canonical pattern in all nine RFs. Run a
single-seed mechanical pilot demonstrating:

- all nine L1 modules receive identical local inputs;
- each independently produces stable turnover and four distinct local owners;
- no feature gate inhibits another patch;
- owner indices may differ across modules;
- L2 receives all nine Eor streams, but no L2 composition/identity claim is made.

If Stage A fails, stop after preserving its full replay/metrics/summary and diagnose the
causal failure. Do not tune parameters or proceed to Stage B.

Do not launch an eight-seed sweep in this implementation task. First report and review the
single-RF and nine-RF seed-1 evidence.

## Verification

Run:

- new builder/validation/engine/layout/dashboard tests;
- `test_microcircuit_turnover.py`;
- existing tiled and `rg_coincidence` focused suites;
- replay-recorder tests when present;
- serialization/preset registry tests;
- the complete test suite;
- `git diff --check`.

Do not regenerate unrelated goldens. Any intentional public preset-contract update must be
isolated and explained.

## Completion report

Report:

- every changed file;
- final topology name, node/edge counts, and exact graph contract;
- why passive relays alone would not provide selectivity and how paired feature C/I gates
  implement it;
- proof that WTA and feature inhibition use distinct relays;
- exact timing evidence for pre-spike center suppression;
- Stage A and, only if allowed, Stage B results/artifact paths;
- focused/full test commands and totals;
- unchanged existing-preset evidence;
- the retained single-Eor identity limitation and all unresolved timing issues.

Do not claim that this topology solves higher-level composition or continuous-time event
semantics. Its acceptance claim is limited to restoring local feature-specific turnover
inside the tiled L1 input layer.
