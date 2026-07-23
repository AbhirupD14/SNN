# Claude Prompt: Basic L1 Consolidation Headless Experiment

## Objective

Build and execute the **Basic consolidation experiment** for both current 9x9 cortical-
column presets:

- `tiled_cc`: eight ordinary competing E neurons in every L1 cortical column;
- `tiled_cc_l1_4`: four ordinary competing E neurons in every L1 cortical column.

The experiment asks whether each of the nine 3x3 receptive fields can consolidate the
four existing local patterns onto a stable, unique one-to-one mapping between patterns and
ordinary L1 E competitors.

For each training phase, place the same local pattern in all nine receptive fields and
hold it until every L1 cortical column has a statistically meaningful stable owner or the
phase reaches a declared timeout. Then switch to the next pattern. After all four phases,
freeze learning and present all four patterns again to determine whether every column
retains the mapping it acquired during training.

This is a headless scientific experiment. It must run without FastAPI, WebSockets, or the
browser, write quantitative metrics, and emit loadable replay artifacts through the shared
test recorder.

Do not stop after writing the script: run the required pilot matrix, validate its
artifacts, then launch the declared multi-seed sweep in the background and report exactly
what is running.

## Dependency and required execution order

This prompt depends on the recorder from:

`prompts/Claude_Headless_Test_Replay_Recorder_Prompt.md`

The recorder/schema must already be implemented, tested, and reviewed. If it is absent or
its focused tests fail, stop and report that dependency rather than inventing a second
artifact format.

The dashboard replay player is **not** required to run this experiment. Do not block the
headless work on dashboard availability.

## Read and inspect first

Read:

- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`
- the implemented replay-recorder schema/API and tests;
- `backend/simulation.py`, especially `PATTERNS`, `set_input()`, `step()`,
  `dynamic_state()`, `topology()`, and event-resolved learning;
- `backend/network_spec.py`, especially tiled metadata, `embed_patch_pattern()`,
  `build_cortical_column()`, `connect_columns()`, and projection names;
- `snn/neurons.py`, especially `ExcitatoryNeuron.update_acc_weights()` and all `learn`
  flags;
- `experiments/tiled_cc_experiment.py`;
- `experiments/linear_ablation.py::_freeze_learning`;
- `tests/test_tiled_cc_engine.py`;
- `tests/test_tiled_cc_l1_4.py`;
- `tests/test_tiled_cc_input.py`;
- `tests/test_tiled_cc_experiment.py`.

Inspect the full working-tree diff first. Preserve unrelated work and generated evidence.

## Experimental conditions: do not force a premature feedback decision

Run and report two matched conditions separately:

1. **`intact` — primary system test.** Use the complete preset unchanged, including
   parent-E apical input, C firing, `C -> I`, and I hard reset. This is the primary result
   because feedback is part of the deployed dynamical system.
2. **`feedback_disabled` — causal L1 control.** Disable only the predictive
   `C -> I -> ordinary-E` consequence for the experiment while preserving:
   - ordinary `E -> I -> ordinary-E` local WTA;
   - all RGC-to-L1 feedforward learning;
   - E-to-Eor and L1-to-L2 feedforward paths;
   - L2 dynamics;
   - C basal/apical activity and observability where possible;
   - every threshold, weight, delay, scheduler rule, and learning equation.

Do not combine these conditions into one pass rate. Report the intact primary result and
feedback-disabled control independently. A failure in intact with a pass in disabled is
scientifically important evidence that feedback interferes with consolidation; do not
hide it by calling the overall experiment a pass.

### Implementing the feedback-disabled control

Do not change a built-in preset, production default, dashboard control, or public topology
registry merely to create this control.

Use the smallest experiment-local mechanism consistent with the implemented graph. Select
the affected route from explicit node/edge metadata and
`projection='column_c_to_i'`—never by parsing IDs or layer-name prefixes. Verify and record
the exact disabled edge IDs. Do not remove/disable `column_e_to_i` or
`column_i_to_e`, because those are the local WTA path.

If an experiment-local dispatch ablation is used, encapsulate it in one well-tested helper
and annotate the replay header/topology/condition so the viewer cannot mistake the route
for active. Do not scatter mutations of private engine structures throughout the loop.
If the existing recorder cannot express the annotation, make the smallest backward-
compatible metadata addition and test it; do not revise the replay schema unnecessarily.

Prove in a focused test that:

- an ordinary E winner still drives local I and hard-resets its own ordinary-E bank;
- a C spike/deposit no longer causes an I/reset event in the disabled condition;
- intact mode is byte-identical to an unmodified engine;
- no cross-column path is changed.

## Exact input construction

The four canonical local patterns are the existing `PATTERNS` entries:

```text
row 1
col 1
diag \
diag /
```

Do not use tiled `set_pattern()` for this protocol: it embeds a pattern into one selected
patch. Construct an 81-element input that embeds the same 3x3 pattern separately into
**all nine** declared 3x3 patches using topology metadata and `embed_patch_pattern()` (or
an equally explicit metadata-driven helper).

For every generated input, assert before training that:

- its length matches the topology's input shape;
- every RGC belongs to exactly one patch;
- each of the nine patches contains exactly the selected local pattern;
- there is no cross-patch indexing error;
- all nine corresponding L1 columns are included in the analysis.

Do not hard-code a 9x9 flat-index formula when the tiled metadata already declares input,
patch, and grid shapes.

## Training protocol

Use a new deterministic engine for every `(topology, feedback_condition, seed)` run.

Default canonical pattern order:

```text
row 1 -> col 1 -> diag \ -> diag /
```

Make pattern order explicit in the CLI and artifact metadata. The initial required sweep
uses the canonical order. Support alternate orders for later robustness work, but do not
multiply the first sweep silently.

For each pattern:

1. Set the all-nine-patch input.
2. Write a replay marker declaring training phase, pattern, and start timestep.
3. Track only ordinary L1 E competitors as candidate pattern owners. Do not count RGC,
   Eor, C, I, or L2 cells as L1 owners.
4. Continue presentation until every one of the nine L1 columns independently satisfies
   the declared stabilization rule at the same current point, or the per-pattern maximum
   boundary count is reached.
5. Record the final owner/runner-up/counts and acquisition timestep for every column.
6. Switch patterns even after a timeout so later behavior remains observable, but mark the
   run failed/confounded and never reinterpret timeout as consolidation.

A column that stabilizes early remains under the same input until all columns stabilize.
Its ownership must remain stable in the current rolling window; do not permanently latch
the first apparent owner and ignore later turnover.

## Operational definition of consolidation

Implement ownership analysis as a pure, independently tested utility. Defaults must be
CLI-configurable and recorded in every artifact.

Recommended primary defaults:

- ownership window: the most recent **50 actual ordinary-E winner events per column** for
  the current pattern, not merely 50 outer boundaries;
- minimum evidence: all 50 events must exist, so a silent/rarely firing column cannot pass;
- owner dominance: at least `0.95` of events belong to one ordinary E;
- stable windows: the same owner satisfies the criterion for three consecutive
  non-overlapping assessment windows;
- strict diagnostic reported separately: whether the owner had `1.00` dominance in the
  final window;
- finite per-pattern timeout supplied by CLI and recorded in the manifest.

If pilot behavior makes these defaults impossible to evaluate in a reasonable time, do
not quietly weaken them. Report the event rates and run a labeled sensitivity analysis at
declared thresholds. Keep the original primary result.

Do not make uniqueness part of the phase stopping condition. A stable owner may be the
same neuron that owned a previous pattern. That is a mapping collision and must be exposed,
not trained around indefinitely.

## One-to-one mapping checks

After four training phases, build a pattern-by-neuron confusion/ownership table separately
for every L1 column.

Required primary checks per column:

- all four patterns reached the consolidation criterion;
- each pattern has exactly one stable owner;
- the four owner IDs are distinct within that column;
- the final dominance/minimum-event criteria are satisfied;
- no owner changed after it was declared stable and subsequently evaluated at the end of
  its phase.

Topology-specific capacity checks:

- `tiled_cc`: four distinct owners and exactly four ordinary competitors not assigned to
  the four patterns;
- `tiled_cc_l1_4`: four distinct owners and zero unassigned ordinary competitors.

Neuron indices are local arbitrary labels. Do **not** require the same competitor index to
own a pattern across different columns, seeds, topologies, or feedback conditions.

The run-level Basic gate passes only when all nine columns pass. Also report per-column
results so aggregate statistics cannot hide a failed receptive field.

## Frozen recall and mapping-retention protocol

After training, freeze **all** plastic state that can change in these presets:

- every ordinary L1 E;
- every Eor;
- every ordinary L2 E;
- every C basal learner;
- any additional plastic object discovered by inspecting the actual topology.

Snapshot every live plastic weight by stable edge ID and prove it is exactly unchanged
through recall.

Perform two clearly separated recall probes:

### A. Independent cold-state recall — primary retention result

For each pattern, construct a fresh engine with the same topology, seed, condition, and
parameters; transfer the complete trained plastic state by stable edge ID; freeze learning;
then present only that pattern from a standardized fresh dynamic state.

Do not transfer weights by array position without verifying aligned source/edge IDs. Do
not transfer membrane voltage, refractory state, pending queues, conductance, spike history,
or winner labels.

Collect enough ordinary-E winner events in every column to apply the same ownership
criterion, with a finite recall timeout. The recalled owner for `(column, pattern)` must
match the training owner and remain unique across patterns.

### B. Sequential frozen recall — dynamical stress result

On one frozen post-training engine, present all four patterns sequentially in canonical
order without resetting dynamic state between them. Use fixed, declared recall blocks or
event-count targets. Report this separately so carryover effects do not contaminate the
primary cold-state result.

Neither recall probe may update any learned weight. Assert exact pre/post equality.

## Measurements and artifacts

Use the implemented shared replay recorder. Create one artifact directory for each
`(topology, feedback_condition, seed)` run.

Use `metrics.csv` in a documented long form with enough columns to reconstruct at least:

- run ID, seed, topology, feedback condition, schedule/order;
- phase (`train`, `cold_recall`, `sequential_recall`);
- pattern;
- timestep/window boundaries;
- L1 column ID and ordinary-E neuron ID;
- winner-event count, share/dominance, owner/runner-up status;
- stable-window count and consolidation Boolean;
- acquisition time and timeout status;
- mapping-collision status;
- assigned/unassigned competitor count;
- recall owner-match Boolean;
- relevant final incoming-weight total and per-afferent weights for each pattern owner;
- all run/column acceptance checks.

Do not store a full dynamic frame in one CSV cell. Use the replay JSONL for movie state.

Replay markers must include pattern transitions, consolidation declarations, timeouts,
collisions, learning freeze, cold recall, sequential recall, and final result. Record the
feedback condition truthfully.

The summary must include:

- per-pattern acquisition duration;
- the 9-column pattern-owner matrix;
- per-column confusion matrices and dominance;
- all collisions/timeouts/silent-column failures;
- cold and sequential recall results;
- capacity usage;
- exact pass/fail checks;
- aggregate result across nine columns;
- artifact file references.

Do not make dashboard replay availability a condition for the scientific result.

## CLI, background execution, and resumability

Provide a clear CLI supporting at least:

- one or both topology names;
- one or both feedback conditions;
- explicit seed list/range;
- explicit pattern order;
- ownership window, dominance, stable-window count, training timeout, and recall timeout;
- replay recording stride and weight-checkpoint interval;
- output root;
- `--resume`/skip-completed behavior;
- worker count, defaulting conservatively to one.

Print periodic compact progress containing run, phase, pattern, timestep, active/stable
column count, and artifact directory. Do not print every neuron/frame.

Write aggregate batch-level `summary.json` and `metrics.csv` files in addition to each
run's artifacts. Aggregation must preserve intact and feedback-disabled results separately.

Provide a summarization command that can be run safely while some runs are still active;
it must label incomplete runs rather than treating them as failures or successes.

## Required automated tests

Long scientific sweeps do not belong in the default pytest suite. Add short focused tests
that prove the harness and analysis are correct:

1. All-nine-RF construction reproduces each exact 3x3 `PATTERNS` vector in all nine
   patches for both presets.
2. Candidate selection includes exactly ordinary L1 E competitors and excludes all other
   roles/layers.
3. Synthetic stable ownership passes; silence, insufficient events, alternating winners,
   owner turnover, and sub-threshold dominance fail.
4. A stable duplicate owner across two patterns is recorded as a collision and does not
   make the training loop wait forever.
5. Eight-competitor and four-competitor capacity accounting is exact.
6. Feedback-disabled mode suppresses only metadata-selected `column_c_to_i` consequences;
   local E-driven WTA remains active, while intact mode is unchanged.
7. Learning freeze covers every plastic cell/edge in each preset.
8. Cold-engine weight transfer by edge ID reproduces all trained plastic weights exactly
   and rejects missing/misaligned IDs.
9. Both recall modes leave every weight byte-identical.
10. A short smoke run produces parseable recorder artifacts, expected markers, CSV rows,
    and a truthful summary even if it does not scientifically consolidate in the shortened
    duration.
11. Timeout and interruption produce resumable, explicitly incomplete artifacts.
12. Running the harness with recording disabled versus enabled does not change dynamics.

Run focused tests, both existing tiled preset suites, recorder tests, the complete suite,
and `git diff --check` before starting the full background sweep.

## Required experiment execution

Do not stop after implementation/tests.

### Phase 1 — foreground pilot

Run seed `1` for this four-condition matrix:

```text
tiled_cc        x intact
tiled_cc        x feedback_disabled
tiled_cc_l1_4   x intact
tiled_cc_l1_4   x feedback_disabled
```

Use the primary ownership defaults and canonical pattern order. Record every boundary for
the pilot if artifact size is practical; otherwise use a declared stride after reporting a
measured size estimate. Validate every manifest, replay, CSV, summary, and aggregate file.

Report the pilot outcomes honestly. A failed consolidation hypothesis is a valid result;
do not tune the model, change thresholds, extend only favorable conditions, or implement a
fix in this task.

### Phase 2 — background multi-seed sweep

After the pilot and all tests succeed, launch seeds `1..8` for both presets and both
feedback conditions using the exact same scientific parameters. Use a conservative replay
stride chosen and recorded from the pilot's size measurement.

Run through the repository virtual environment with `PYTHONPATH=.`. Use a noninteractive
background command with stdout/stderr redirected to a named log. Record the shell PID,
exact command, start time, output root, and log path in the implementation report. The
experiment itself must checkpoint/flush artifacts and support resume.

Do not claim the sweep passed or failed while it is still running. If it completes within
the task, run the aggregate summarizer and report actual results. If it remains active,
report only verified pilot results plus process/status instructions.

Do not launch duplicate background sweeps if matching completed/running artifacts already
exist.

## Scientific reporting rules

- Distinguish stable firing ownership from mature weights; report both.
- Distinguish consolidation from one-to-one selectivity.
- Distinguish training ownership from frozen recall.
- Distinguish cold-state recall from sequential dynamic recall.
- Distinguish intact system results from feedback-disabled causal controls.
- Distinguish a timed-out/silent column from a negative mapping result.
- Do not pool nine columns and hide individual failures.
- Do not require matching arbitrary neuron indices across independent columns/runs.
- Do not call four-neuron capacity exhaustion proof that a new layer successfully learns.
- Do not modify network dynamics in response to a failed result.

## Completion report

Report:

- every changed file;
- exact ownership/consolidation/recall definitions;
- how all-nine-patch inputs are constructed and verified;
- how feedback is disabled and how local WTA is proven intact;
- how learning is frozen and weights transferred for cold recall;
- focused/full test commands and totals;
- pilot commands, runtime, artifact sizes, and all four pilot outcomes;
- the background sweep command, PID/status, output/log paths, and resume/summarize commands;
- completed multi-seed results only if actually finished;
- every timeout, collision, silent column, artifact limitation, or unresolved ambiguity.

Do not implement continuous learning, noise invariance, composition, NEST, dashboard replay,
or a model fix in this task.
