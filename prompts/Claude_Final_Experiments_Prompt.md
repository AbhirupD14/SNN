# Claude Prompt: Final Dual-FE Experiments and Composition Report

> **Status: superseded as the current work prompt.** Do not execute this omnibus plan as
> written. It predates the decision to preserve the intentional `theta/2`
> pattern-detector ceiling and avoid new topology work. The approved next experiment is
> `prompts/Claude_Existing_Topology_Frequency_Scaling_Prompt.md`. This file remains only as
> a record of the broader trial-and-error plan.

## Objective

Carry out the final bounded set of scientific experiments on the current classic
no-feature-relay cortical-column fabric, then consolidate every result—positive or
negative—into:

```text
docs/FINAL_EXPERIMENTS.md
```

The four tasks, in required order, are:

1. **Adaptive hold-out learning:** present one canonical pattern until the network meets a
   declared learning criterion, then move to the next pattern. There is no fixed training
   dwell, only a safety timeout.
2. **Interleaving and residual-charge ablation:** compare high/low `B` with and without a
   narrowly defined charge wipe at every pattern transition.
3. **Fabric constraint report:** identify empirically and structurally where the current
   architecture works, slows, collides, or becomes incapable of representing the task.
4. **Two-tower composition:** place two copies of the current `9x9 -> nine L1 CC -> one L2
   CC` hierarchy side by side, connect both L2 columns to one L3 column using the existing
   cortical-column connectivity rule, and test split `V`, `A`, and straight-legged `7`
   glyphs.

This is an implementation, execution, and reporting task. Do not stop after writing a
plan or creating unexecuted scripts. Keep the execution matrix bounded and resumable
because remaining project time is limited.

## Scientific scope

The primary architecture is the current classic:

```text
tiled_cc
```

with:

- `9x9` RGC input;
- nine `3x3` L1 receptive fields;
- eight ordinary E competitors per L1 column;
- one Eor, one C, and one I per classic cortical column;
- one eight-competitor L2 column;
- classic `E -> Eor`, `E -> I -> E`, `Eor -> C`, `C -> I`,
  `child.Eor -> parent.E`, and `parent.E -> child.C` connectivity;
- dual FE/FES learning enabled;
- **no feature relays** and no recursive identity-relay topology.

Do not reintroduce per-feature gating (the removed `tiled_cc_feature_gated` direction),
feature-specific inhibition, a weight cap, normalization, priming, or a new neuron
equation.

The experiments may vary only declared experimental parameters/interventions:

- dual-FE/FES `B`;
- the experimental pattern-transition charge wipe;
- input schedule/order/dwell;
- seed;
- the explicit two-tower topology in Task 4.

Keep the current dashboard learning-rate reference fixed unless a task below explicitly
labels a control:

```text
dual_fe_fes = true
dual_fe_e   = 0.001
dual_fe_wte = 0.001
eta         = 1.0
c_eta       = 0.5
leak_rate   = 0.0
refractory_steps = 0
```

These are the current dashboard confirmation parameters. Do not accidentally inherit
`SimulationEngine`'s slower general defaults (`eta=0.01`, `c_eta=0.005`). The existing
`experiments/interleaving_parallel_rf.py` currently constructs a bare engine and therefore
uses those slower defaults unless explicitly overridden; do not treat its 60,000-boundary
run as a matched dashboard-parameter result.

## Working-tree safety and dependencies

The working tree is dirty and contains important overlapping work, including engine
validation, a coincidence eligibility phase correction, dashboard fixes, and replay-to-live
weight branching. Before editing:

1. inspect `git status`, the full diff, and recent log;
2. identify which files already belong to another task;
3. preserve every unrelated modification and generated artifact;
4. run a focused baseline and record it;
5. do not regenerate or bless a golden file unless this task intentionally changes
   production behavior—which it should not.

Do not commit or push unless explicitly instructed after the results are reviewed.

Required infrastructure should already exist:

- the headless replay recorder;
- dashboard JSONL replay player;
- canonical weight reconstruction;
- ownership/consolidation analysis;
- plastic-weight snapshot/transfer/freeze helpers.

If an in-progress replay-branch implementation is incomplete, do not finish or rewrite it
as part of this task. These experiments only need the existing recorder/player.

Expected edit surface:

```text
experiments/final_experiments.py                 new
tests/test_final_experiments.py                  new
docs/FINAL_EXPERIMENTS.md                        new
docs/FINAL_EXPERIMENTS_WORKLOG.md                new only if needed
backend/network_spec.py                          only for the two-tower builder
tests/test_tiled_cc_builder.py or the new test   only for two-tower structure
```

Do not edit `snn/neurons.py`, production stepping/scheduling in
`backend/simulation.py`, dashboard frontend files, replay parser/player files, existing
experiment scripts, or golden baselines. If a blocker appears to require one of those
changes, document the blocker and continue with evaluable phases rather than broadening
the task.

## Read before editing

Read all of:

- `backend/simulation.py`;
- `snn/neurons.py`, especially ordinary and coincidence dual FE/FES implementations;
- `backend/network_spec.py`, especially `build_cortical_column`,
  `connect_rgc_patch`, `connect_columns`, `tiled_cc_spec`, metadata validation, and tiled
  family invariants;
- `backend/layout.py` and topology serialization;
- `backend/dashboard_config.py`;
- `experiments/consolidation_analysis.py`;
- `experiments/basic_consolidation.py`;
- `experiments/dual_fe_cc4_consolidation.py`;
- `experiments/interleaving_parallel_rf.py`;
- `experiments/replay_recorder.py`;
- `docs/BASIC_CONSOLIDATION.md`;
- `docs/ENGINE_VALIDATION_REPORT.md`;
- `docs/ENGINE_VALIDATION_HANDOFF_REPORT.md`;
- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`;
- `docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`;
- tests for tiled topology construction, patch input, consolidation analysis, dual FE/FES,
  replay recording, serialization, and topology validation.

Inspect the relevant existing run artifacts, especially:

- `experiments/runs/dual_fe_cc4/20260723-135744-dual_fe_cc4-62e59ae4/`;
- `experiments/runs/20260723-184817-interleaving_parallel_rf-8309d10a/`;
- the Basic-consolidation aggregate and representative per-run summaries.

Do not copy their conclusions into the new report without verifying that parameter,
topology, and schedule contracts match.

## Required implementation shape

Prefer one resumable orchestrator:

```text
experiments/final_experiments.py
```

with focused reusable helpers in a small module only if necessary. Add:

```text
tests/test_final_experiments.py
docs/FINAL_EXPERIMENTS.md
```

If the two-tower builder belongs naturally in `backend/network_spec.py`, add one narrowly
named experimental builder there with structural tests. Do not add it to the dashboard
preset registry unless doing so is necessary for engine construction. A self-contained
recorded custom topology can already be rendered by the replay player and is sufficient
for this final experiment.

The CLI must support:

```text
--phase adaptive|interleave|constraints|composition|all
--seed / --seeds
--output-root
--run-dir
--resume
--quick
```

Every phase writes an atomic completion marker and can resume without rerunning completed
cells. Never infer completion merely because a directory exists.

Use:

```text
experiments/runs/final_experiments/<run-id>/
```

for a parent run containing per-condition replay-recorder directories and aggregate
artifacts.

`--output-root` creates a new parent run and prints its exact path. `--run-dir` opens one
specific existing parent run for later phases. Reject supplying both. `--resume` may skip
only a matrix cell whose atomic marker says `status=completed` and whose recorded config
hash equals the requested config hash. A failed/interrupted cell must remain inspectable
and rerun into a new attempt directory; never truncate its replay.

## Repository-grounded execution contract

The following is the implementation map for this repository. Use these symbols directly
instead of rediscovering or replacing them:

| Need | Existing source of truth |
|---|---|
| Canonical pattern order | `experiments.consolidation_analysis.CANONICAL_PATTERN_ORDER` |
| Tiled 81-pixel input | `all_nine_patch_input(...)` and `verify_all_nine_input(...)` |
| Ordinary-E population by column | `ordinary_e_ids_by_column(...)` |
| Stable owner | `assess_ownership(...)` with `window=50`, `dominance=.95`, `stable_windows=3` |
| Plastic snapshot | `experiments.basic_consolidation.plastic_edge_weights(...)` |
| Freeze and invariance | `freeze_learning(...)` and `assert_weights_unchanged(...)` |
| Cold-state transfer | `transfer_plastic_weights(...)` |
| Replay | `experiments.replay_recorder.ReplayRecorder` |
| Classic column construction | `ColumnHandles`, `build_cortical_column(...)` |
| RGC patch wiring | `connect_rgc_patch(...)` |
| Child/parent wiring | `connect_columns(...)` |
| Tiled metadata family | `backend.network_spec.TILED_FAMILY` (`tiled_cortical_columns`) |

Do not duplicate `assess_ownership`, pattern vectors, freeze logic, or plastic-edge
enumeration. It is acceptable to add a small wrapper around them that also handles L2/L3.

### Exact reference-engine construction

Create one helper and use it for every classic single-tower condition:

```python
def make_final_engine(*, seed: int, B: float) -> SimulationEngine:
    return SimulationEngine(
        seed=seed,
        topology="tiled_cc",
        cc_e_count=8,
        dual_fe_fes=True,
        dual_fe_e=0.001,
        dual_fe_wte=0.001,
        dual_fe_B=float(B),
        eta=1.0,
        c_eta=0.5,
        leak_rate=0.0,
        refractory_steps=0,
    )
```

Assert the resulting `engine.params` values immediately. Do not call dashboard config
application or `apply_config` during a run: configuration rebuilds the graph and destroys
learned/dynamic state. A new `(B, seed, condition)` always gets a fresh engine.

Also assert every plastic ordinary/Eor cell uses `update_mode=="dual_fe_fes"` and every C
uses `update_mode=="c_dual_fe_fes"`. In these modes the learned upper cap is inactive and
the floor is the raw `dual_fe_wte=0.001`. Do not interpret retained `w_max`/configuration
edit limits as a learning cap, and fail if an update shows upper clipping.

For Task 4, construct the engine with the same parameters, call
`engine.apply_topology(two_tower_composition_spec(...))`, and assert the returned
`engine.current_spec()`/`engine.topology()` dimensions and metadata before taking an
initial weight snapshot. `apply_topology` validates and rebuilds, so enable update-log
instrumentation only afterward. Do not hand-populate engine internals.

### Build one metadata index, once per engine

Create a `FabricIndex` (dataclass or plain object) from graph metadata, not ID parsing:

- `engine.meta[nid]["layer"]`;
- `engine._role_of[nid]`;
- `engine._column_of[nid]`;
- `engine.tiled_meta["columns"]`;
- graph edge `source`, `target`, `kind`, and `projection`.

It must expose:

```text
ordinary[layer][column_id] -> ordered ordinary-E ids
eor[column_id]             -> Eor id
c[column_id]               -> C id
i[column_id]               -> I id
parents[column_id]         -> declared parent column ids
children[column_id]        -> derived child column ids
rgc_by_patch[(row,col)]     -> RGC ids ordered by patch-local position
ff_edge[(source,target)]    -> stable edge id
basal_edge[column_id]       -> Eor-to-C basal edge id
```

Use metadata roles exactly: ordinary competitor is `column_role == "E"`; Eor is
`"Eor"`; coincidence is `"C"`; relay is `"I"`. `engine.exc` contains membrane-bearing
E, Eor, and C cells. RGC sources and stateless inhibitory relays are not charge-bearing
membranes. Tests must fail if a requested layer/column/edge is missing or ambiguous.

ID parsing is allowed only in an assertion that checks builder naming; scientific
selection must still use metadata.

### Exact per-step collection order

Enable instrumentation once after construction:

```python
for cell in engine.plastic:
    cell.record_updates = True
for cell in engine.coincidence:
    cell.record_updates = True
```

For every boundary, in this order:

1. call `dyn = engine.step()`;
2. consume `dyn["column_winners"]` for ordinary winner events;
3. consume `engine.spiked[nid]`, `cell.spike_tau`, `dyn["hard_reset_events"]`,
   `dyn["latency_ties"]`, and the C diagnostics;
4. copy every new record from each cell's `update_log`, add `timestep`, `phase`,
   `pattern`, `layer`, and `column_id`, then clear that list immediately;
5. update milestone trackers from this unsampled boundary;
6. write metrics;
7. call `rec.record_frame(engine)`; replay sampling must never control analysis.

Do not retain unbounded raw event lists. Ownership needs only its trailing 150 winner
events plus counters; update logs must be aggregated and/or streamed to CSV after being
drained.

Bound replay size explicitly. For each condition calculate:

```text
planned_max_boundaries = sum of its phase timeouts or fixed schedule
record_every = max(1, ceil(planned_max_boundaries / 1500))
checkpoint_every = 50 written frames
```

Force a frame at every phase/pattern end and write semantic markers at every transition.
Thus a seed-1 condition remains loadable/scrubbable without creating hundreds of thousands
of full 191/393-neuron frames. Metrics and ownership still inspect every boundary.
Later-seed runs may be metrics-only or use a 500-frame cap. Record the calculated stride,
actual frames, bytes, and planned/actual boundaries.

### Event and delay semantics to use in causal metrics

Do not infer these again:

- external RGC activity in boundary `t` schedules feedforward delivery at `t+1`;
- a child ordinary-E spike schedules its Eor feedforward delivery at `t+1`;
- an Eor spike schedules its parent-E feedforward and local-C basal delivery at `t+1`;
- parent ordinary-E apical delivery to a child C occurs at the parent's event time in
  that boundary;
- relay and hard-reset action is zero-latency inside the current outer boundary;
- `dyn["column_winners"]` already reports only the first ordinary `E` winner per column,
  excluding Eor/C/I.

When measuring a source/target response, store the expected delivery boundary explicitly.
Do not count same-boundary correlation as causal throughput. If several eligible source
events coalesce onto one target boundary, report both source-event and delivery-boundary
denominators rather than forcing a false one-to-one match.

### Existing helpers that may be adapted but not trusted blindly

`experiments.basic_consolidation.RunContext` and `train_phase` provide the right all-nine
input and trailing-owner pattern. Extend them for L2/L3 and the full milestone gate;
do not modify `assess_ownership`. The Basic script's `train_timeout=3000` and engine
builder are not suitable here.

`experiments.interleaving_parallel_rf.py` supplies useful schedule/recording logic, but its
bare engine currently inherits `eta=.01` and `c_eta=.005`. Import pure helpers if useful;
do not modify the old run or cite it as a matched result.

`experiments.dual_fe_cc4_consolidation.py` shows how to enable and drain
`record_updates`, but its LR-multiplier sweep is not this experiment.

## Shared patterns and ownership criterion

The canonical local patterns remain exactly:

```text
row 1
col 1
diag \
diag /
```

Canonical order:

```text
row 1 -> col 1 -> diag \ -> diag /
```

For the single-tower tasks, embed the same local pattern separately into all nine declared
`3x3` patches using topology metadata and the existing tested input helper. Do not use a
single-selected-patch dashboard call.

Reuse the established event-based ownership criterion:

- most recent 50 actual ordinary-E winner events per column;
- all 50 events required;
- dominance at least `0.95`;
- the same owner for three consecutive non-overlapping assessment windows.

Do not permanently latch the first apparent owner. Continue reassessing the trailing
windows until the phase ends.

Keep these three concepts separate:

1. **stable firing owner**;
2. **one-event synaptic maturity**;
3. **full hierarchical readiness**.

A stable L1 winner does not prove that Eor, L2, or C has matured.

## Exact milestone definitions

Implement milestone evaluation as pure functions and unit-test them with synthetic
traces. Use tolerances only for floating-point comparisons, never to relax event counts.

### Stable owner

For every phase maintain the trailing ordinary winner-event sequence separately for every
column. Recompute `assess_ownership(...)` every boundary. Milestone A/D is true only when
every required column's current verdict is consolidated on that same boundary. Record:

- `first_reached`: first boundary at which the joint condition became true;
- `final_reached`: whether it remains true at phase stop;
- `owner`: current owner, not the first historical owner;
- `lost_after_first`: number and boundaries of later loss/turnover.

For L1 the required set is all nine columns (18 in Task 4). For L2 it is the one
single-tower L2 column or both tower L2 columns. For L3 it is the single L3 column.

### Structural one-event maturity from rest

With this experiment's `leak_rate=0`, zero inhibitory conductance, and a resting target,
one full delivered feedforward packet crosses if its delivered charge is at least the
cell threshold. Compute this from the target's aligned sources and live pre-update
weights:

```python
def active_charge(cell, active_source_ids):
    return sum(
        float(w)
        for src, w in zip(cell.ff_src, cell.acc_weights)
        if src in active_source_ids
    )

mature = active_charge(cell, active_source_ids) >= cell.threshold - 1e-9
```

This is a structural counterfactual from rest. Keep it separate from observed live
response, which can be affected by retained `V`, inhibition, refractory state, and event
coalescing.

Use these active source sets:

- L1 owner: the nine RGC ids active in that column's local pattern;
- L1 Eor: the current ordinary owner id only;
- L2 owner: the nine L1 Eor ids whose child columns are active for the tiled pattern;
- L2 Eor: the current L2 ordinary owner id only;
- L3 owner: the left and right L2 Eor ids that actually participate for the glyph;
- L3 Eor, if reported: the current L3 ordinary owner id only.

Record `active_charge`, `threshold`, `margin=active_charge-threshold`, active source ids,
and the component weights. Also run at least one independent cold-state empirical probe
for each claimed mature path. Do not mutate the training engine to perform the probe:
copy weights by stable edge id into a fresh identical engine and freeze learning.

### Observed throughput reliability

For each owner-to-Eor path, maintain a queue of causal source events and expected
delivery boundaries. Over the most recent 50 eligible delivery boundaries report:

```text
eligible_source_events
eligible_delivery_boundaries
target_spike_boundaries
response_reliability = target_spike_boundaries / eligible_delivery_boundaries
median/min/max target spike tau
```

Readiness requires at least 50 eligible delivery boundaries, structural one-event
maturity, and reliability `>=0.95`. If coalescing makes multiple source events share one
boundary, preserve both counts and use delivery boundaries as the reliability
denominator.

### C readiness

For each C whose column has a parent:

- a coincidence opportunity is a boundary where a valid basal-eligible signal and
  apical activation can meet under the production gate;
- a committed deposit is evidenced by
  `coincidence_deposit_count`/`deposit_committed_this_boundary`;
- a C learning event is an entry drained from that C's `update_log`;
- a C spike is `engine.spiked[c_id]`;
- the impulse one-shot condition at `leak_rate=0` is
  `cell.basal_weight >= cell.threshold - 1e-9`.

Readiness requires at least one valid committed deposit, at least one C spike/update,
and the live basal weight meeting the impulse one-shot condition. Record opportunity,
deposit, spike, and update counts separately; absence of a spike is not proof that no
opportunity occurred.

The topmost C in a hierarchy has no parent and is excluded by metadata
`has_parent == false`, not by an ID special case.

### Full phase gate and persistence

Evaluate `A+B+C+D+E+F` every boundary. To avoid stopping on a one-boundary conjunction,
require the full Boolean gate to remain true for three consecutive evaluations. Because
the individual owner verdicts already require 150 actual winner events, this confirmation
does not replace or weaken ownership.

The gate is for the currently presented pattern. It is not possible to observe all four
patterns simultaneously. After the four phases, separately check persistence of every
stored pattern/owner path from the weight snapshot and with cold recall.

## Pre-experiment algebraic audit of the B hypothesis

Before running the sweep, write a short derivation and executable assertions for both
actual learning paths. Import the production reference functions from `snn.neurons`; do
not reproduce an outdated docstring.

Ordinary E/Eor:

```text
FE  = dual_fe(Iaccq, theta, e, B)
    = e + (1-e)/(1 + B*((Iaccq/theta)-0.5)^2)
FES = dual_fes(w, theta, wte, B)
    = wte + (1-wte)/(1 + B*((2*w/theta)-0.5)^2)
dw  = eta * FE * FES * signal * distance_influence
```

Coincidence C basal:

```text
FE_C  = dual_fe_c(Iaccq, theta, e, B)
      = e + (1-e)/(1 + B*((Iaccq/theta)-1.0)^2)
FES_C = dual_fes_c(w, theta, wte, B)
      = wte + (1-wte)/(1 + B*((w/theta)-0.5)^2)
dw_C  = c_eta * FE_C * FES_C * apical_gate * basal_signal * influence
```

Because `FE <= 1`, `FES <= 1`, and `|signal| <= 1`, lowering `B` can remove suppression
away from the peaks but cannot make:

```text
|dw| > LR * influence
```

At the current dashboard rates, a weight that must move hundreds of units cannot literally
finish in two or three updates solely because `B` was lowered. Calculate the optimistic
minimum update count for:

- an ordinary/Eor weight moving from its initialization toward one-event threshold;
- the C basal weight moving from `theta/4` toward the one-shot impulse threshold.

Then measure actual update counts. Do not “rescue” the two-or-three-firing hypothesis by
silently increasing LR.

For C use its C-specific operating centers and `c_eta`; do not evaluate it with the
ordinary-E functions. Assert at representative points that the implementation functions
match the hand calculation to `1e-12`.

Calculate the lower bounds from a freshly constructed engine rather than copied constants:

```text
ordinary/Eor:
  Q0 = sum(initial weights on the active causal afferents)
  max active-charge gain/update = eta * sum(active distance influences)
  optimistic updates >= ceil(max(0, theta-Q0) / max_gain)

C basal:
  w0 = initial basal weight
  max gain/update = c_eta * basal influence
  optimistic updates >= ceil(max(0, theta-w0) / max_gain)
```

These are deliberately optimistic because they replace both FE factors and all gates by
one. Compare them with the actual drained learning-event counts, not boundaries or spikes
of a different cell.

Also state the opposing effect: lower `B` makes the tails less rigid. It may accelerate
acquisition while increasing drift/interference. `B=0` makes FE/FES flat at one and is a
useful negative control, not automatically a desirable setting.

# Task 1 — Adaptive hold-out learning and B sweep

## Question

When each pattern is held until the fabric has actually learned it rather than for a fixed
dwell, how does `B` affect:

- L1 ownership acquisition;
- one-event maturation;
- Eor throughput;
- L2 acquisition;
- C maturation;
- one-to-one mapping and frozen recall;
- actual learning-update counts?

## B values

Use this declared seed-1 sweep:

```text
B = 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0
```

Interpret:

- `5.0` as the current dashboard/reference value;
- `20.0` as a high-curvature diagnostic;
- `0.0` as the flat-plasticity negative control;
- `0.1..2.0` as the low-B hypothesis range.

Keep `eta`, `c_eta`, `e`, and `wte` fixed at the values declared above. This task isolates
`B`; do not multiply it by an LR sweep.

After the detailed seed-1 sweep, run only the reference `B=5` and the selected low-B
candidate on seeds `1..4`. Do not run a large seed matrix.

## Adaptive phase protocol

For every `(B, seed)` use one fresh `tiled_cc` engine and train the four patterns
sequentially. A pattern phase has no predetermined dwell. It ends only when:

1. the declared full-fabric learning criterion is satisfied; or
2. a finite safety timeout is reached.

Use a generous default safety timeout such as `30,000` boundaries per pattern, CLI
configurable and recorded. A timeout is a failed/unfinished phase, never evidence of
learning.

Record the first boundary at which each milestone is reached:

### Milestone A — local L1 ownership

- all nine L1 columns simultaneously meet the stable ownership criterion;
- no column is silent;
- owner identity and update count are recorded independently per column.

### Milestone B — L1 one-event maturity

For each L1 pattern owner, use the actual leak-corrected one-event condition represented by
the engine. Record active-afferent charge/weight total and demonstrate that one causal
pattern volley from rest can cross threshold.

Do not equate weight magnitude with maturity without checking the actual delivered charge
and membrane equation.

### Milestone C — Eor readiness

For every L1 column:

- identify the active owner-to-Eor edge by graph metadata;
- measure eligible owner spikes, Eor spikes, and their causal delay;
- record the owner-to-Eor weight and whether one owner event can make Eor cross from rest;
- require a declared recent-window Eor response reliability, recommended `>=0.95` over 50
  eligible owner events, accounting for the explicit one-boundary delivery delay.

Report stable L1 ownership separately even if Eor readiness is not reached.

### Milestone D — parent/L2 ownership

- determine the ordinary L2 owner using the same event-based dominance criterion;
- record active child-Eor evidence, L2 update counts, first firing, and stable acquisition;
- report sparse/silent L2 behavior rather than hiding it behind L1 success.

### Milestone E — C readiness

For each non-dormant L1 C:

- record valid basal/apical coincidences, C spikes, and basal-weight updates;
- record first C spike and first one-shot-capable basal weight;
- require the actual impulse one-shot condition from rest, not a historical cap;
- report coincidence opportunity rate and update count.

The top L2 C is intentionally dormant because it has no parent. Exclude it from the C
readiness gate and report that exclusion explicitly.

### Milestone F — top Eor readiness

Measure the L2 owner-to-L2-Eor path because Task 4 will use L2 Eor as the child output to
L3. Record whether the selected L2 owner drives L2 Eor reliably and whether the relevant
edge is one-event mature.

### Full-fabric phase stop

The primary phase stop is:

```text
A + B + C + D + E + F all satisfied simultaneously
```

If this criterion proves structurally impossible for multiple patterns—for example, Eor
learning depresses previously acquired owner channels—preserve that result. Also report
the earlier milestones so a full-fabric timeout does not erase evidence that L1 learned
quickly.

Implement the loop with this control flow:

```python
for pattern in CANONICAL_PATTERN_ORDER:
    install_verified_all_nine_input(pattern)
    reset_only_phase_counters()             # never reset network state here
    recorder.marker("pattern_start", ...)
    consecutive_full = 0
    for local_boundary in range(1, phase_timeout + 1):
        dyn = engine.step()
        collect_unsampled_boundary(dyn)
        milestones = evaluate_current_pattern()
        consecutive_full = consecutive_full + 1 if all(milestones.values()) else 0
        if consecutive_full >= 3:
            outcome = "full_fabric_ready"
            break
    else:
        outcome = "timeout"
    recorder.marker("pattern_end", data={outcome, milestone first/final states, ...})
```

A timeout does not terminate the whole B condition. Preserve the trained state, move to
the next pattern, and label all downstream interpretation as following an unfinished
phase. This is necessary to learn whether a slow Eor/C gate prevents the experiment while
L1 mappings still evolve.

Phase counters include:

```text
boundaries
winner events by column/neuron
spikes by role/layer/column
learning events and applied |dw| by role/layer/column
first spike and first update
Eor eligible/response delivery boundaries
C opportunities/deposits/spikes/updates
hard resets and latency ties
non-finite observations
```

Write one milestone row as soon as a milestone first becomes true and a final row at phase
end. Do not only write the last Boolean state.

## Mapping and recall

After all four phases:

- require four distinct L1 owners in each of the nine columns;
- require four distinct L2 owners for the four global tiled patterns;
- report Eor channel maturity for every learned owner, not just the most recent one;
- freeze every plastic feedforward and C-basal weight;
- run independent cold-state recall for each pattern;
- verify exact frozen-weight invariance;
- compare recalled L1 and L2 owners with training owners;
- measure whether C and Eor readiness survives the complete four-pattern sequence.

Record whether low `B` accelerates acquisition but damages retention. Candidate selection
must not use acquisition speed alone.

Cold recall is independent per `(trained condition, pattern)`:

1. build a fresh identical engine with the same seed/config/topology;
2. copy all plastic feedforward and C-basal weights by stable edge id using
   `transfer_plastic_weights`;
3. call `freeze_learning` and retain the returned exact snapshot;
4. install only the queried pattern;
5. run until ownership can be assessed or a declared recall timeout expires;
6. verify `assert_weights_unchanged`;
7. discard the probe engine before testing the next pattern.

Do not reuse one recall engine across patterns; otherwise residual charge makes the probe
order-dependent. The recall timeout and required event count must be reported.

## Low-B candidate rule

Choose one low-B value from `0.1, 0.25, 0.5, 1.0, 2.0` using this predeclared order:

1. all nine L1 columns have four distinct cold-recalled owners;
2. all four L2 owners are distinct and cold-recalled;
3. no non-finite values;
4. no frozen-recall drift;
5. lowest median full-fabric acquisition boundary across the four patterns;
6. tie-break toward the larger `B` to retain more tail rigidity.

Only a value satisfying items 1–4 is a validated hierarchical candidate. If no value
passes, set `low_B_candidate = null`. For Task 2, still select one diagnostic
deterministically by descending:

1. count of L1 columns with four unique cold-recalled owners;
2. count of individual L1 pattern/column cold-recall matches;
3. count of distinct cold-recalled L2 owners;
4. count of L2 pattern cold-recall matches;
5. count of pattern phases reaching full readiness;
6. lower median boundary to the last reached milestone, treating a phase with no
   milestone as infinity;
7. larger `B`.

Call this field `low_B_diagnostic`, preserve the failed gates, and label it
**diagnostic**, not validated, everywhere in Task 2/reporting.

# Task 2 — Interleaving x B x transition-charge wipe

## Exact four primary conditions

Run the required `2 x 2` ablation:

| Condition | B | Pattern-transition charge wipe |
|---|---:|---|
| high/no-wipe | `5.0` | off |
| high/wipe | `5.0` | on |
| low/no-wipe | selected low-B candidate/diagnostic | off |
| low/wipe | selected low-B candidate/diagnostic | on |

All other parameters, seeds, pattern order, initial weights, total exposure, and recording
policy must match.

## Interleaving schedule

Use:

```text
row 1 -> col 1 -> diag \ -> diag / -> repeat
```

Primary schedule:

```text
dwell = 200 boundaries/pattern
cycles = 75
total = 60,000 boundaries
```

This matches the existing long observation schedule but must use the explicit dashboard
LRs in this prompt. Run seed 1 with detailed replay/metrics for all four conditions. Run
seeds `1..4` with coarser replay stride or metrics-only summaries after the seed-1
artifacts are verified.

There is no warm-up and no wipe before the first `row 1`. A transition occurs after
exactly 200 completed boundaries of the old pattern, including `diag / -> row 1` between
cycles. Apply the wipe, if enabled, after collecting/recording the final old-pattern
boundary and immediately before `engine.set_input(new_vector)`. Emit paired
`pattern_end`/`pattern_start` markers with the transition index.

For paired fairness:

- use the same seed and topology/config except `B` and wipe;
- assert identical initial plastic weight snapshots for all four seed-matched engines;
- use the identical pattern vectors and transition timesteps;
- do not stop a condition early when it appears consolidated;
- if one condition becomes non-finite, preserve its failure and continue the other cells.

Pass/fail ownership must be computed from every engine step, never from sampled replay
frames.

## Exact charge-wipe intervention

The scientific intervention is removal of **neuron-local residual charge**, not a full
network reset.

Before implementing it, audit all charge-bearing fields and distinguish:

- somatic membrane charge/potential;
- gathered/frozen excitation owned by a neuron;
- scheduled next-boundary edge events;
- inhibitory conductance;
- activity trace;
- refractory state;
- C basal/apical eligibility;
- learned weights.

Implement one experiment-local helper selected by neuron metadata, never ID prefixes.
The primary wipe must occur after the last boundary of the old pattern and before the new
input is installed. It should mirror a simultaneous hard reset of every membrane-bearing
neuron:

- set somatic membrane potential to rest;
- discard neuron-owned gathered/frozen excitatory charge.

It must **not** change:

- any learned weight;
- inhibition/conductance;
- refractory state;
- activity trace;
- C eligibility/compartment state;
- RNG state;
- topology;
- timestep.

In the current code, implement the primary helper exactly as:

```python
def wipe_neuron_local_charge(engine):
    for cell in engine.exc.values():
        cell.hard_reset(tau=0.0, discard_drive=True)
        cell.pending_exc = 0.0
```

`hard_reset(...)` sets `V` to `v_rest` and clears `remaining_excitation`; it deliberately
preserves `g_inh`, `a`, `refractory_timer`, weights, and already-emitted spike diagnostics.
The explicit `pending_exc=0` clears gathered but not-yet-frozen neuron-owned excitation.
Take the before/after audit snapshot around this helper and assert the exact population is
`engine.exc`.

Do **not** clear these in the primary helper:

```text
cell.iaccq
cell.v_pre_reset
cell.spiked / cell.spike_tau / cell.fired_this_boundary
engine._exc_next
engine._ff_deliv_next
engine._basal_next
current/next apical or coincidence eligibility
```

`iaccq` and the spike fields are causal diagnostics from the just-finished boundary, not
charge that can drive the next one; `begin_event_boundary()` refreshes the per-boundary
diagnostics on the following step. The engine queues are network events already in flight.
Document this distinction explicitly so the audit does not falsely report a partial wipe.

RGC `SourceNeuron`s and stateless `InhibitoryNeuron` relays have no membrane charge and
must appear in the audit as excluded populations, not silently omitted.

Do not silently erase scheduled network events or C eligibility in the primary
intervention. Those are separate forms of causal state, not somatic charge. Instead:

- record their counts at every switch;
- report whether old-pattern events arrive after the wipe;
- if they materially confound interpretation, classify the limitation and specify a
  future `full-transition-flush` diagnostic in the report. Do not implement that broader
  intervention in this bounded task.

Add focused tests proving exactly which fields change and that all weights remain
byte-identical across the wipe itself.

At every transition record:

```text
population
count
sum/min/max V before and after
sum/min/max pending_exc before and after
sum/min/max remaining_excitation before and after
queued _exc_next target/event count and charge
queued _ff_deliv_next target/source count
queued _basal_next target/source count
sum/min/max g_inh
sum/min/max refractory_timer
sum/min/max activity trace a
C basal_received/basal_eligible/apical_active/coincidence state counts
plastic-edge count plus SHA-256 of stable edge-id/IEEE-754-weight pairs
```

Do not use Python's randomized `hash()`. Build checksums by sorting stable edge ids and
packing exact float64 values or using a canonical full-precision representation.

## Interleaving acceptance

For each condition and seed, assess:

- four unique L1 owners per column;
- owner consistency across late repeated visits to the same pattern;
- turnover at pattern switches;
- first-winner latency after each switch;
- collision/incumbent absorption rate;
- cold-state frozen recall after interleaved training;
- L1 Eor and L2 readiness;
- C coincidence/update rate and one-shot readiness;
- weight drift of old pattern owners;
- non-finite state;
- switch-to-switch residual charge and effect of the wipe.

“Late repeated visits” means the last 10 visits to each pattern (or all visits if a quick
run has fewer than 10). For each visit, assess ownership only from winner events emitted
during that visit. Report the modal late owner, visit-to-visit agreement, dominance, and
silent/insufficient-event visits. A transition turnover is counted only when the first
stable owner of the new visit differs from the preceding visit's final stable owner;
do not count an arbitrary first spike as turnover.

The primary question is causal:

```text
Does low B improve interleaving?
Does charge wipe improve interleaving?
Is there an interaction between them?
```

Report the four cells separately. Do not pool them into one pass rate.

# Task 3 — Identify the current fabric's breaking point and constraints

## Required report section

`docs/FINAL_EXPERIMENTS.md` must contain a section:

```text
## Where the Current Fabric Breaks
```

This must be evidence-backed, not a speculative list.

Use the Task 1/2 results plus this bounded stress matrix:

### A. Transition speed stress

For seed 1, use the selected low-B value with both wipe states and test:

```text
interleave dwell = 10, 25, 50, 100, 200
```

Keep exactly 15,000 boundaries of exposure per pattern (60,000 total) using:

| Dwell | Cycles |
|---:|---:|
| 10 | 1500 |
| 25 | 600 |
| 50 | 300 |
| 100 | 150 |
| 200 | 75 |

This locates the shortest presentation that still produces enough actual winner/update
events. Assert `dwell * cycles == 15000` before every run.

### B. Evidence-density stress

From a fresh or frozen learned state, activate:

```text
1 RF
3 RFs
9 RFs
```

Use the same local pattern selected from a successfully recalled Task-1 state and these
fixed patch coordinates:

```text
1 RF:  (1,1)
3 RFs: (0,0), (1,1), (2,2)
9 RFs: every (patch_row, patch_col)
```

Construct these vectors with topology metadata. Run both a frozen learned-state probe and,
only if cheap, a fresh-state firing diagnostic; label them separately. Measure L1
activity, Eor activity, number of L2 afferent source identities, L2 delivered charge,
L2 first firing, L2 ownership, and C opportunities. This directly characterizes the
known sparse-evidence L2 bootstrap constraint.

### C. Pattern-order stress

For the selected low-B/no-wipe condition, compare seed 1 under:

```text
canonical order
reverse order
```

Reverse means exactly:

```text
diag / -> diag \ -> col 1 -> row 1
```

Do not search permutations for a favorable result.

### D. Noise-invariance probe

Take one successfully trained state, freeze learning, and present a diagonal tiled across
three declared RFs. Add one extra active pixel in:

```text
one RF
two RFs
three RFs
```

Compare ordinary L1 and L2 recall with the clean diagonal. Use deterministic declared
noise pixels that are not part of the clean pattern. This is a frozen diagnostic; do not
allow the noisy probes to retrain the representation.

Use the three diagonal RFs `(0,0)`, `(1,1)`, `(2,2)`, place local `diag \` in each, and use
local coordinate `(0,2)` as the extra pixel because it is outside local `diag \`. Define:

```text
clean:   no extra pixels
noise-1: extra (0,2) in RF (0,0)
noise-2: extras in RFs (0,0), (1,1)
noise-3: extras in RFs (0,0), (1,1), (2,2)
```

Verify every noisy vector differs from the clean vector by exactly 1/2/3 bits and that no
extra bit was already active.

If no state passes the prerequisite clean recall, mark this probe **not evaluable** rather
than manufacturing a noise result.

## Failure taxonomy

Classify each failure using a stable vocabulary:

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

For every failure report:

- first seed/pattern/column/layer;
- smallest reproducing condition;
- milestone reached before failure;
- winner/update/event counts;
- whether it is parameter-sensitive or structural;
- whether charge wipe changes it;
- narrowest future action, without implementing it.

## Structural constraints to reconcile with measurements

Explicitly discuss:

- one Eor per column collapses ordinary-E owner identity into one output channel;
- Eor is itself a plastic ordinary accumulating neuron and learns only when it fires;
- C learns only on valid basal/apical coincidence spikes;
- the top column's C is intentionally dormant without a parent;
- a single active child among nine can under-drive L2;
- ordinary and C updates are bounded per firing by LR even when `B` is zero;
- exact WTA ties use stable node order;
- each inhibitory relay emits at most once per outer boundary;
- the engine has analytic within-boundary evolution but boundary-synchronous propagation;
- network-level physical `dt` refinement is unsupported;
- eight ordinary competitors give an upper local assignment capacity of eight, while the
  tested vocabulary uses four;
- local single-winner WTA does not by itself implement simultaneous multi-feature
  composition.

State which constraints were observed empirically, which are proven by graph/equation
structure, and which remain untested.

# Task 4 — Two 9x9 towers feeding one L3 composition column

## Purpose and gate

Run this task last. Do not allow its implementation to delay or invalidate Tasks 1–3.

The question is whether two independently learned halves can be composed by one final
classic L3 column under the same dual-FE and CC-to-CC connectivity rules.

This is a deliberately small composition probe, not a claim of image recognition.

## Required topology

Create one deterministic experimental classic tiled graph with:

```text
Tower Left:
  9x9 RGC -> 9 L1 classic CCs -> 1 L2 classic CC

Tower Right:
  9x9 RGC -> 9 L1 classic CCs -> 1 L2 classic CC

Composition:
  both L2 classic CCs -> 1 L3 classic CC
```

Every classic CC has eight ordinary E competitors plus one Eor, one C, and one I.

Use fresh namespaced IDs and metadata for the two towers. The input surface should be
represented as two lateral `9x9` fields, preferably one validated `9x18` input sheet with
two explicit tower/group identities. Do not create overlapping pixel ownership.

Use this exact deterministic naming and coordinate contract so no design work is left:

```text
input: row-major RGC0..RGC161 on a 9x18 sheet
left L1:  T0L1c00..T0L1c22, global metadata cols 0..2
right L1: T1L1c00..T1L1c22, global metadata cols 3..5
left L2:  T0L2c00, metadata (layer=L2,row=0,col=0)
right L2: T1L2c00, metadata (layer=L2,row=0,col=1)
top:      L3c00,   metadata (layer=L3,row=0,col=0)
```

For the right L1 IDs, the ID's local column is `0..2`, while `build_cortical_column` must
receive global display column `local_col + 3`. Scientific code identifies the tower from
declared parent relations/global metadata; do not parse the ID.

Construct RGC nodes exactly as `tiled_cc_spec` does:

```text
pixel = input_row * 18 + input_col
patch_row = input_row // 3
patch_col = input_col // 3
patch_id = patch_row * 6 + patch_col
patch_local_row = input_row % 3
patch_local_col = input_col % 3
```

The topology metadata must be:

```python
dict(
    family=TILED_FAMILY,
    input_shape=dict(rows=9, cols=18),
    patch_shape=dict(rows=3, cols=3),
    grid_shape=dict(rows=3, cols=6),
    column_layers=[
        dict(layer="L1", rows=3, cols=6),
        dict(layer="L2", rows=1, cols=2),
        dict(layer="L3", rows=1, cols=1),
    ],
    cc_e_count=8,
    columns=[...],
)
```

Each left L1 `parent_ids=["T0L2c00"]`; each right L1
`parent_ids=["T1L2c00"]`; both L2 columns `parent_ids=["L3c00"]`; L3
`parent_ids=[]`. Their C node `has_parent` values must agree. Preserve the standard
column metadata fields `id/layer/row/col/e_count/parent_ids`.

Implement the builder as `two_tower_composition_spec(cc_e_count=8, ...)`. Prefer keeping
it adjacent to `tiled_cc_spec` in `backend/network_spec.py` because it composes the same
public rules and needs normal validation/layout/serialization. Do not register it as a
dashboard preset. Tests may build it directly and the experiment may call the existing
custom-topology application path.

Builder algorithm:

1. emit the 162 RGC nodes and group them by global `(patch_row,patch_col)`;
2. build 18 L1 columns with `has_parent=True`;
3. call `connect_rgc_patch` once for each global patch and its own L1 handles;
4. build two L2 columns with `has_parent=True`;
5. call `connect_columns` from each L1 only to its tower's L2;
6. build L3 with `has_parent=False`;
7. call `connect_columns(T0L2, L3)` and `connect_columns(T1L2, L3)`;
8. concatenate internal, RGC, and hierarchy edges once;
9. validate and return a fresh JSON-serializable spec.

Connectivity from each tower L2 child to L3 must be emitted by the same generic
`connect_columns(child, parent)` rule:

```text
child.L2.Eor -> every L3 ordinary E       feedforward
every L3 ordinary E -> child.L2.C         apical
```

Therefore each tower L2 must declare L3 as its parent and its C is no longer dormant.
L3 has no parent, so L3 C is intentionally dormant.

Do not connect ordinary L2 winners directly to L3. Do not add identity relays or feature
gates. The single Eor bottleneck is part of what this experiment tests.

Derive node/edge totals from the built graph and assert the full connectivity. As a sanity
check—not a value to blindly encode—the expected uniform-eight arithmetic is:

```text
two existing towers: 2 * (191 nodes, 1052 edges)
one L3 classic column: 11 nodes, 26 internal edges
two L2-to-L3 links: 2 * 16 edges
total: 393 nodes, 2162 edges
```

Tests must validate structure, not just totals:

- exactly 162 unique RGC pixels across the two fields;
- 18 L1 columns, two L2 columns, one L3 column;
- every RGC feeds only its own `3x3` patch's L1 E bank;
- every L1 connects only to its own tower L2;
- both and only the two L2 columns connect to L3;
- all I resets remain column-local;
- each non-top C receives apical input from exactly its declared parent;
- L3 C has zero apical inputs and is declared dormant;
- no feature-relay archetype/variant exists;
- graph construction is deterministic and returns fresh objects.

Ensure topology metadata and layout make both towers and L3 intelligible in a recorded
dashboard replay. Do not build a new renderer.

## Exact glyph inputs

Define three deterministic `9x18` one-pixel-wide binary glyphs and store their coordinates
in artifacts/tests.

Use:

### V

- left leg: from the upper-left of the left `9x9` field toward its lower-right/center seam;
- right leg: from the upper-right of the right `9x9` field toward its lower-left/center
  seam.

One acceptable exact coordinate form is:

```text
left:  (r, r)       for r=0..8
right: (r, 17-r)    for r=0..8
```

### A

- two diagonal legs meeting near the upper center seam and spreading toward both lower
  outer corners;
- one horizontal crossbar spanning both halves near the middle.

Use:

```text
left leg:  (r, 8-r) for r=0..8
right leg: (r, 9+r) for r=0..8
crossbar:  (4, c)   for c=4..13
```

The two top pixels at columns 8 and 9 form the seam apex; the legs spread to columns 0
and 17 at the bottom. Include the exact coordinate set and ASCII rendering in the report.

### 7

- a horizontal stroke across the complete top row;
- a vertical straight leg down the far-right column.

Use:

```text
(0, c) for c=0..17
(r, 17) for r=0..8
```

This is intentionally a straight-legged `7`, matching the requested simplified form.

Before training assert that:

- all coordinates are within the `9x18` sheet;
- left/right halves are correct;
- the three vectors are distinct;
- every active pixel belongs to exactly one declared RGC and patch;
- the stored ASCII picture matches the binary vector.

## Representation-separability preflight

Before training L3, determine what L3 can actually observe.

Use a separate preflight run; do not contaminate either required composition-training
condition.

1. Build the two-tower graph with the selected condition.
2. Disable **learning only** on L3 ordinary E/Eor by metadata; leave L3 firing and all
   graph paths intact.
3. Train `V`, `A`, and `7` sequentially until both lower towers reach their L1/L2/Eor
   milestones or the declared timeout. Record failures honestly.
4. Snapshot all weights.
5. For each glyph, build a fresh identical custom-topology engine, transfer the snapshot,
   freeze all learning, and start from cold dynamic state.
6. Present the glyph for 1,000 boundaries, or until both L2 ownership verdicts are
   consolidated plus 200 further boundaries, whichever occurs first.
7. Record the exact L3 input signature as ordered
   `(relative_boundary, source_L2_Eor_id, delivered_charge)` tuples and the source spike
   `tau` that scheduled each delivery.

Do not disconnect L3 or suppress its feedback while pretraining the lower towers: those
would be different graph dynamics. Only its plastic flags are frozen. Verify those flags
and the unchanged L3 weights in a focused test.

Present `V`, `A`, and `7` independently on the cold clones. For each glyph record:
For each glyph record:

- ordinary L2 owner in each tower;
- each tower's L2 Eor spike train and sub-boundary timing;
- the two L2-Eor-to-L3 delivered event streams;
- L3's resulting two-afferent input vector/temporal trace.

The graph exposes only two L3 source identities: left L2 Eor and right L2 Eor. If the three
glyphs produce identical source participation and indistinguishable event timing at the
engine's resolution, then distinct L3 classification is structurally unidentifiable.

Classify two signatures as identical only if their complete ordered tuples match over the
common observation window. Also report the less strict summaries—event count per source,
inter-event interval distribution, and source co-occurrence by boundary—because exact
traces can differ only by startup transients. If exact traces differ but all late steady
summaries match, label `transient_only_separability`; do not call that robustly
identifiable.

In that case:

- record `representation_not_identifiable`;
- run one bounded negative L3 training probe to confirm the collision;
- do not tune timing, add relays, or connect L2 winners directly;
- preserve the result as the principal composition finding.

Do not claim that different hidden L2 owners are available to L3 when the graph transmits
only the shared Eor.

## Composition training

Run seed 1 under two declared parameter conditions:

1. reference `B=5`, no transition wipe;
2. the selected low-B/no-wipe condition from Task 1.

For each condition:

1. train `V`, `A`, and `7` sequentially using the same adaptive milestone approach;
2. record L1 ownership in all 18 patches;
3. record both L2 owners and Eor readiness;
4. record L3 owner acquisition, event/update counts, and timeout;
5. require three distinct stable L3 owners;
6. freeze all weights and run independent cold-state recall;
7. require the recalled L3 owner to match training for all three glyphs;
8. record whether `V` and `A`, which share diagonal structure, remain separable because of
   the crossbar.

For a glyph, derive `active_patch_ids` directly from its active RGC coordinates. Blank
patches cannot satisfy an ownership criterion and must not hold the adaptive stop hostage.
The composition phase gate is:

```text
all active L1 columns stably owned and one-event mature
all active L1 owner->Eor paths ready
both tower L2 columns stably owned and one-event mature
both L2 owner->Eor paths ready
all active non-top C cells (active L1 plus both L2 C) one-shot ready
L3 stably owned and structurally one-event mature from its participating L2 Eors
```

Require that inactive L1 columns remain silent or report leakage separately. Record all 18
columns even though only active ones gate the phase. Do not require different glyphs to
have different L1/L2 owners when their local/half inputs are identical; uniqueness is the
declared three-owner requirement at L3.

Use `30,000` boundaries/glyph by default, record first/final milestone states, and continue
to later glyphs after a timeout as in Task 1. L3 Eor and L3 C are reported but excluded
from the gate because L3 has no parent and no higher consumer in this experiment.

Use a finite, generous timeout. If seed 1 succeeds, confirm on seeds `2..4`. If seed 1
fails for a structural representation reason, do not spend time on a seed sweep.

Do not use charge wipe as an undeclared composition aid. A separate wipe diagnostic is
allowed only after both required no-wipe conditions are preserved.

# Artifacts

The parent final-experiment run must contain:

```text
config.json
status.json
adaptive_B_sweep.csv
adaptive_milestones.json
learning_event_counts.csv
interleave_2x2.csv
transition_charge.csv
constraint_matrix.csv
failure_catalog.json
composition_topology.json
composition_glyphs.json
composition_results.csv
aggregate_summary.json
replays/
```

Each detailed seed-1 condition must have a loadable `replay.snn.jsonl` written through the
existing recorder. Later seeds may use coarse replay sampling, but scientific metrics must
use every engine step.

Every artifact records:

- git commit and dirty flag;
- seed;
- exact topology/fingerprint;
- exact engine parameters;
- B and wipe condition;
- pattern/glyph order;
- timeouts and ownership thresholds;
- actual per-phase duration;
- recorder stride/checkpoint policy;
- completion/failure status;
- all exclusions and not-evaluable probes.

Never overwrite a previous run directory.

Define `topology_fingerprint` as SHA-256 of the validated `engine.current_spec()` after
removing only live display `pos` and serializing with sorted keys and compact JSON
separators. Define `config_hash` similarly over the complete condition config, schedule,
timeouts, criteria, and recording policy. Store the canonical payload beside each hash so
the hash is auditable.

## Minimum artifact schemas

Use stable headers and one row per declared observation. Extra columns are allowed, but do
not replace these with nested prose.

`adaptive_B_sweep.csv`:

```text
seed,B,pattern,pattern_index,outcome,start_timestep,end_timestep,boundaries,
milestone_A_at,milestone_B_at,milestone_C_at,milestone_D_at,milestone_E_at,
milestone_F_at,full_ready_at,l1_distinct_so_far,l2_owner,nonfinite_count,
preceded_by_timeout
```

`adaptive_milestones.json` stores the per-column detail omitted from the aggregate CSV:
owner verdict windows, first/final/lost states, active-source weights and charge margins,
Eor reliability denominators, C opportunity/deposit/spike/update counts, and exclusions.

`learning_event_counts.csv`:

```text
phase,condition,seed,B,pattern,timestep,layer,column_id,role,cell_id,
update_events,positive_synapses,negative_synapses,floor_hits,sum_abs_raw_dw,
sum_abs_applied_dw,mean_FE,mean_FES,min_weight,max_weight,nonfinite
```

Rows may aggregate a bounded interval, but the interval size must be recorded and all
drained update events must be represented exactly once.

`interleave_2x2.csv`:

```text
condition,seed,B,wipe,dwell,cycles,total_boundaries,column_id,pattern,
late_owner,late_dominance,unique_mapping,turnovers,incumbent_absorptions,
median_first_winner_latency,cold_recall_owner,cold_recall_match,Eor_ready,
C_ready,L2_owner,L2_ready,nonfinite_count
```

`transition_charge.csv`:

```text
condition,seed,transition_index,timestep,from_pattern,to_pattern,wipe,population,
count,V_sum_before,V_sum_after,V_max_before,V_max_after,pending_sum_before,
pending_sum_after,remaining_sum_before,remaining_sum_after,exc_next_count,
ff_next_count,basal_next_count,g_inh_sum_before,g_inh_sum_after,
refractory_sum_before,refractory_sum_after,trace_sum_before,trace_sum_after,
C_eligible_before,C_eligible_after,weight_checksum_before,weight_checksum_after
```

`constraint_matrix.csv`:

```text
stress_family,condition,seed,parameter,value,prerequisite_status,outcome,
first_failure_taxon,first_layer,first_column,first_pattern,milestone_before_failure,
winner_events,learning_events,notes
```

`composition_results.csv`:

```text
condition,seed,B,glyph,outcome,boundaries,left_L2_owner,right_L2_owner,
left_L2_Eor_ready,right_L2_Eor_ready,L3_observed_source_signature,
L3_owner,L3_dominance,L3_distinct_mapping,cold_recall_owner,cold_recall_match,
representation_identifiable,first_failure_taxon,nonfinite_count
```

`failure_catalog.json` is a list of structured records using the exact taxonomy in Task 3.
`composition_topology.json` is the exact validated spec plus counts/fingerprint.
`composition_glyphs.json` contains sorted coordinates, 162-bit vectors, per-half counts,
and ASCII renderings.

`aggregate_summary.json` must have these top-level keys even if a phase is not evaluable:

```text
schema_version
run_id
repository
config
adaptive
selected_low_B
interleave
constraints
composition
failures
artifacts
completion
```

Use `null` plus a reason for not-evaluable values. Never encode missing scientific
results as zero or `false`.

Replay subdirectories use deterministic condition slugs, for example:

```text
replays/adaptive-B0p5-seed1/
replays/interleave-low-wipe-seed1/
replays/composition-reference-seed1/
```

Construct slugs with one helper and reject collisions.

# `docs/FINAL_EXPERIMENTS.md`

Write one comprehensive, self-contained report with this structure:

```text
# Final Experiments

## Executive Summary
## Repository and Model State
## Questions and Predeclared Hypotheses
## Methods and Acceptance Criteria
## B Algebra and Update-Count Bounds
## Experiment 1 — Adaptive Hold-Out Learning
## Experiment 2 — Interleaving and Charge Wipe
## Where the Current Fabric Breaks
## Experiment 4 — Two-Tower V/A/7 Composition
## Validated Capabilities
## Negative Results and Constraints
## Handoff Recommendations
## Reproduction Commands and Artifact Index
```

The executive summary must say, without euphemism:

- whether reducing B actually reduced boundaries and learning events to acquisition;
- whether any neuron learned in two or three actual updates;
- whether low B improved interleaving;
- whether charge wipe independently helped;
- whether there was a B/wipe interaction;
- where Eor and C sit on the measured timescale;
- whether the two-tower L3 could distinguish V/A/7;
- whether any failure was structural rather than a matter of insufficient dwell.

Use tables for B comparisons, the four ablation cells, milestone latency by population,
failure conditions, and composition results. Link exact artifact paths.

Do not describe sampled dashboard appearance as evidence. Do not convert a timeout into
failure of an equation unless the measured mechanism supports that classification.

# Focused tests

Add fast deterministic tests covering:

- explicit dashboard parameter construction;
- algebraic `|dw| <= LR*influence` bound;
- B=0 and lower-B factor behavior;
- adaptive ownership does not latch an early incumbent;
- milestone calculation from synthetic event/weight traces;
- exact population selection for E, Eor, C, and L2/L3;
- charge wipe changes only declared neuron-local charge fields;
- charge wipe preserves every learned weight and excluded state;
- all-nine-patch input construction;
- equal-exposure interleave schedule calculation;
- failure taxonomy;
- glyph coordinates/ASCII rendering;
- two-tower topology counts and complete connectivity;
- tower isolation and parent metadata;
- representation-separability classification;
- recorder metadata and resume completion markers.

Do not place full 30k/60k scientific runs inside pytest.

# Execution order and gates

Before implementation, run and save this focused baseline:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_dual_fe_fes.py \
  tests/test_basic_consolidation.py \
  tests/test_tiled_cc_builder.py \
  tests/test_tiled_cc_engine.py \
  tests/test_tiled_cc_input.py \
  tests/test_tiled_cc_layout.py \
  tests/test_network_spec.py \
  tests/test_replay_recorder.py \
  tests/test_serialization_api.py
```

After implementation, run the same list plus `tests/test_final_experiments.py`. Run the
full suite only after the scientific phases have finished and artifacts have been
validated.

`--quick` is an implementation smoke test, never a scientific result. It must write under
a run id containing `quick` and use this bounded matrix:

```text
adaptive: B=[0.5,5.0], seed=1, phase_timeout=2,000
interleave: four cells, seed=1, dwell=20, cycles=2
constraints: one dwell and one evidence-density case
composition: structural/preflight plus at most 2,000 boundaries/glyph
replay: record_every=20, checkpoint_every=10 written frames
```

The report generator must refuse to cite a quick run as the final experiment.

Execute strictly:

1. focused baseline;
2. implementation tests;
3. Task 1 seed-1 B sweep;
4. inspect Task 1 artifacts and choose low B by the declared rule;
5. Task 1 reference/candidate seeds 1–4;
6. Task 2 seed-1 four-cell ablation;
7. inspect artifacts;
8. Task 2 seeds 1–4;
9. bounded Task 3 stress matrix;
10. write the preliminary constraint section;
11. implement/test the two-tower topology;
12. run representation preflight;
13. run Task 4 seed 1, then only conditionally seeds 2–4;
14. write `docs/FINAL_EXPERIMENTS.md`;
15. run focused regressions;
16. run the full suite;
17. run `git diff --check`.

Use the CLI in a resumable sequence like:

```bash
# Creates and prints RUN_DIR.
PYTHONPATH=. .venv/bin/python experiments/final_experiments.py \
  --phase adaptive --seeds 1 --output-root experiments/runs/final_experiments

# Reuse the exact printed directory; do not guess its run id.
PYTHONPATH=. .venv/bin/python experiments/final_experiments.py \
  --phase adaptive --seeds 1,2,3,4 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/final_experiments.py \
  --phase interleave --seeds 1,2,3,4 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/final_experiments.py \
  --phase constraints --seed 1 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/final_experiments.py \
  --phase composition --seeds 1,2,3,4 --run-dir "$RUN_DIR" --resume
```

The orchestrator must automatically restrict later adaptive/composition seeds according
to the declared gates; passing `1,2,3,4` authorizes eligible confirmations, not blind
execution.

At the end run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_final_experiments.py
PYTHONPATH=. .venv/bin/python -m pytest
git diff --check
```

If a long command must run detached, report the exact command, PID, log, and artifact
root. Do not claim completion while it is still running.

If context is running low, update:

```text
docs/FINAL_EXPERIMENTS_WORKLOG.md
```

with completed phases, commands, artifact paths, failures, and the exact next command.
This is a resumability aid, not a substitute for completing the task.

# Final response

Report:

1. concise scientific verdict for all four tasks;
2. selected low B and the declared selection evidence;
3. exact 2x2 interleaving results;
4. first measured breaking points;
5. V/A/7 composition result and whether L3 inputs were identifiable;
6. files changed;
7. commands and test counts;
8. artifact and report links;
9. anything still running or not evaluable;
10. confirmation that no feature-relay topology, weight cap, hidden LR change, or
    production timing change was introduced.

Do not commit or push.
