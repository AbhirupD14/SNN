# Claude Prompt: Existing-Topology Frequency Scaling Experiment

## Objective

Run a focused, headless scaling experiment on the existing classic tiled cortical-column
topology. Determine how the number and spatial arrangement of active L1 patches changes
L2 confirmation, L1 coincidence-cell feedback, and the exact temporal cadence of each
active L1 column.

This is the next approved scientific task. It is an implementation-and-execution task:
build the experiment harness, add fast tests, run the declared matrix, preserve all
positive and negative results, and write the final report.

The central question is:

> With the existing topology, intentional `theta/2` pattern-detector ceiling, and current
> graph-derived input pacing held fixed, at what active-patch scales does each L1 column
> produce exact alternating `fire, silent, fire, silent` evidence, and does that transition
> actually track mature C confirmation rather than pacing alone?

Do not report a long-run mean near `0.5` as success unless the event sequence actually
alternates.

## Approved architectural decisions

Use only:

```text
topology = tiled_cc
```

Preserve the existing graph and its current mechanics:

- 9x9 RGC surface;
- nine 3x3 L1 patches/columns;
- eight ordinary E competitors per L1 column;
- one Eor, one coincidence C, and one shared WTA/feedback I per column;
- one eight-competitor L2 column;
- ordinary local WTA;
- child Eor to parent E feedforward;
- parent E to child C apical feedback;
- C to the existing column I;
- delayed boundary-start feedback hard reset;
- dual FE/FES learning.

The pattern-detector ceiling is intentional and load-bearing:

```text
e_weight_cap_frac = 0.5
w_i <= theta/2
```

It ensures one afferent is not enough evidence for a detector. Do not remove, relax,
reinterpret, or sweep it.

Do not:

- add or build a new topology;
- add new neurons, relays, gates, edges, or event state;
- reintroduce the removed feature-gated topology;
- change thresholds, caps, learning equations, scheduler semantics, delays, WTA, C
  dendrites, or hard-reset behavior;
- tune a parameter to obtain a preferred cadence;
- implement event-conserving suppression or a confidence state machine;
- run B sweeps, charge-wipe ablations, composition towers, or unrelated final experiments;
- commit or push.

This experiment characterizes the existing fabric. It does not fix it.

## Source-of-truth documentation

Read before editing:

- `Current_Implementation_Methodology_Equations.md`, especially:
  - “Tiled cortical columns”;
  - “Intentional θ/2 integration and the sparse-evidence halving limit”;
  - “Rejected direction: per-feature gated tiled columns”;
- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`, especially:
  - “Predictive inhibition has no scale-independent timing invariant”;
  - “Immediate scaling experiment”;
  - “Sparse-evidence L2 bootstrap remains unresolved”;
- `backend/dashboard_config.py`;
- `backend/simulation.py`, especially event stepping, `column_winners`,
  `_ff_deliv_now`, C feedback scheduling, and `hard_reset_events`;
- `backend/network_spec.py`, especially tiled metadata and patch wiring;
- `experiments/consolidation_analysis.py`;
- `experiments/basic_consolidation.py`;
- `experiments/replay_recorder.py`;
- `tests/test_feedback_hard_reset.py`;
- `tests/test_patch_pattern_composition.py`;
- `tests/test_tiled_cc_engine.py`;
- `tests/test_basic_consolidation.py`.

The old `prompts/Claude_Final_Experiments_Prompt.md` is a superseded omnibus plan. Do not
execute it. Its Task 3B is useful historical context for evidence-density measurements,
but its no-cap rule, parameter values, partial 1/3/9 matrix, B/wipe tasks, and new
two-tower topology conflict with this experiment.

Inspect `git status`, the complete working-tree diff, and recent log before editing.
Feature-gated removal work and documentation changes may still be uncommitted. Preserve
them and every unrelated artifact.

## Frozen experiment configuration

Construct engines from a copy of `backend.dashboard_config.DASHBOARD_OVERRIDES`, not from
the slower general `SimulationEngine` defaults. After construction, assert and record both
the configured and graph-resolved parameters:

```text
topology            = tiled_cc
dual_fe_fes          = true
dual_fe_B            = 5.0
dual_fe_e            = 0.001
dual_fe_wte          = 0.001
eta                  = 4.0
c_eta                = 16.0
leak_rate            = 0.0
refractory_steps     = 0
input_period         = 0          # AUTO
resolved_input_period = 3         # derived tiled_cc feedback-loop latency
feedback_loop_latency = 3
e_weight_cap_frac    = 0.5
c_feedback_reset     = true        # primary conditions
eor_w_init_frac      = 1.0
eor_plasticity_enabled = false
```

If the live `DASHBOARD_OVERRIDES` or graph-derived pacing no longer resolves to these
values, stop and report the exact difference rather than silently using a stale copy.

The only model flag changed in a control is:

```text
c_feedback_reset = false
```

That matched control uses the same topology and every other resolved parameter.

## Patterns

Use all four canonical local patterns from production, in this order:

```text
row 1
col 1
diag \
diag /
```

For the primary scaling matrix, every active patch in a run receives the same local
pattern. This isolates evidence density from mixed-pattern composition.

Build every 81-pixel vector from topology metadata and `embed_patch_pattern` or a new
small generalization of the tested helpers in `experiments/consolidation_analysis.py`.
Do not use dashboard-selected-patch state, hand-authored flat pixel indices, or a union
that loses patch identity.

Assert for every vector:

- length equals the declared input size;
- exactly `3*k` pixels are active for `k` active patches;
- every active pixel belongs to exactly one selected patch;
- every selected patch contains exactly the requested local pattern;
- every unselected patch is blank.

## Patch-count and spatial-arrangement matrix

Sweep every active-patch count:

```text
k = 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Use two deterministic nested arrangements. For a given `k`, take the first `k`
coordinates in the declared order.

### Spread / center-out

```text
(1,1), (0,0), (2,2), (0,2), (2,0), (0,1), (2,1), (1,0), (1,2)
```

### Compact / row-major growth

```text
(0,0), (0,1), (1,0), (1,1), (0,2), (1,2), (2,0), (2,1), (2,2)
```

Validate all coordinates against `tiled_meta.grid_shape`. At `k=9` both arrangements are
identical; run and store that condition once, not twice.

The primary seed matrix is:

```text
spread:  seeds 1,2,3,4 x all four patterns x k=1..9
compact: seed 1         x all four patterns x k=1..8
```

Do not pool arrangements or seeds. Aggregate only after preserving every per-run and
per-column result.

## Two complementary scaling protocols

Run both protocols and label them separately.

### Protocol A — matched learned weights

This isolates evidence density from developmental differences.

For each `(seed, pattern)`:

1. Build a fresh reference engine with all nine patches active.
2. Train until the settling rule below passes or the declared timeout is reached.
3. Snapshot every plastic ordinary-E/Eor/L2 and C basal weight using the existing
   `plastic_edge_weights` helper.
4. For every requested `(arrangement, k)`, build a fresh identical engine.
5. Transfer the complete plastic snapshot by stable edge id.
6. Do not transfer membrane, queue, refractory, trace, or conductance state.
7. Freeze every plastic object and assert weights stay byte-identical.
8. Apply only the requested patch subset and run a short cold-state pipeline warm-up.
9. Measure the fixed late window.

If the all-nine reference does not settle, mark every dependent matched-weight probe
`not_evaluable` with that reason. Do not manufacture a learned snapshot.

### Protocol B — developmental scaling

This measures what each evidence density learns on its own.

For every requested `(seed, pattern, arrangement, k)`:

1. Build a fresh engine.
2. Apply exactly the requested active patches.
3. Train with learning enabled until the settling rule passes or timeout occurs.
4. Record whether it settled, timed out with activity, or failed to bootstrap.
5. Snapshot and freeze all plastic weights without resetting live membrane/queue state.
6. Measure the fixed late window.
7. Assert all frozen weights are byte-identical after measurement.

Do not transfer a dense learned state into this protocol.

## Settling and timeout rule

Do not assume that a fixed dashboard dwell implies maturity. Use a bounded adaptive rule.

Constants:

```text
minimum_training_boundaries = 6_000
diagnostic_window           = 600 eligible presentations
required_stable_windows     = 3
maximum_training_boundaries = 60_000
owner_dominance             = 0.95
rate_stability_tolerance    = 0.02
```

At non-overlapping 600-presentation checkpoints after the minimum:

1. For each active L1 column, use ordinary-E winner identities to find its dominant owner.
2. Require the same owner at dominance `>= 0.95` in all three trailing windows.
3. For each active L1 column, compute winner, C, and deduplicated feedback-reset rates in
   each trailing window.
4. Require `max(rate)-min(rate) <= 0.02` for each rate.
5. Compute the cadence classification defined below and require the same classification
   in all three windows.

The run is settled only when every active L1 column satisfies all conditions. Do not
require L2 to be a one-event integrator and do not require exact halving to declare the
trace settled; the point is to measure stable negative regimes too.

At timeout, preserve and measure the final state with:

```text
settling_status = timeout_unsettled
```

Never reinterpret a timeout as exact halving or discard it from denominators.

For Protocol A cold probes, use:

```text
pipeline_warmup = 200 boundaries
```

with learning already frozen, then measure. The warm-up is not a new training phase.

## Measurement window and event alignment

Use:

```text
measurement_eligible_presentations = 2_400
```

Measure cadence over eligible L1 evidence-arrival presentations, not raw wall-clock
boundaries.

Derive an eligible presentation from causal delivery state:

- map RGC sources to patches and ordinary E targets using validated metadata;
- for a column, count a boundary as eligible when that column's ordinary-E bank receives
  the held patch's RGC volley in `_ff_deliv_now`;
- independently assert that the expected active RGC sources continue spiking at
  `resolved_input_period=3`, while configured `input_period=0` remains in auto mode;
- do not count blank startup/pipeline boundaries;
- do not infer eligibility from whether an E winner happened to fire.

This distinction is load-bearing: a feedback-suppressed evidence event is still eligible,
but must contribute a `0` to the L1 output sequence.

For each eligible presentation and active L1 column, record:

- binary ordinary-E winner event;
- winning neuron id and `tau`, if any;
- binary Eor spike;
- binary C spike;
- binary feedback-reset volley;
- number of feedback reset target records;
- L2 ordinary-E winner id/`tau`, if any;
- active RGC event count;
- exact timestep.

`hard_reset_events` contains one record per target E. Deduplicate a feedback volley by
`(outer_boundary, source I, column)` before counting it. Also record the raw target count
and assert that a volley resets the complete local ordinary-E bank exactly once.

Inactive columns must record zero eligible presentations, winners, C spikes, and feedback
resets attributable to the active input.

## Cadence definitions

Given the binary L1 winner sequence `b[0..N-1]` over eligible presentations:

### Exact alternation

```text
all(b[t] != b[t-1] for t=1..N-1)
and abs(sum(b) - N/2) <= 1
```

Either phase (`1010...` or `0101...`) is valid. Record the phase.

### Grouped half-rate

The accepted fraction is within one event of `0.5`, but at least one adjacent pair is
equal. A representative example is:

```text
111000111000...
```

### Irregular reduced activity

The accepted fraction is strictly between `0.05` and `0.95`, but the trace is neither
exact alternation nor grouped half-rate.

### Effectively unsuppressed

```text
accepted_fraction >= 0.95
```

### Effectively silent

```text
accepted_fraction <= 0.05
```

For every sequence also report:

- accepted and silent counts;
- accepted fraction;
- transition fraction `mean(b[t] != b[t-1])`;
- separate run-length distributions for ones and zeros;
- maximum one-run and zero-run;
- lag-1 autocorrelation when mathematically defined;
- first 120 bits and a SHA-256 digest of the complete sequence.

Do not use only the 40-boundary dashboard `freq` field; collect the complete experiment
window directly.

## Feedback-off causal controls

Run matched `c_feedback_reset=false` controls for:

```text
seed        = 1
pattern     = row 1
arrangement = spread
k           = 1..9
```

Keep every other resolved parameter identical. Require:

- zero `feedback_hard_reset` events;
- no hidden change to WTA hard resets;
- identical input eligibility;
- no claim that an on/off difference is caused by feedback unless these checks hold.

The control is evidence about causality, not an alternate production configuration.

## Focused pattern-transition probe

After the primary matrix, run a bounded developmental probe:

```text
seed        = 1
arrangement = spread
k           = 1,2,3,5,9
transition  = row 1 -> col 1
```

Use learning enabled. Settle or time out on `row 1`, switch every active patch to `col 1`,
then run up to 20,000 boundaries.

Record:

- final pre-switch owner per active column;
- first post-switch winner;
- first stable post-switch owner;
- whether the owner changed;
- boundaries and eligible events to the new stable owner;
- accepted fraction and cadence classification in consecutive 600-event windows;
- C and feedback-reset rates during transition and after settling;
- whether any inactive column becomes active.

This is a diagnostic of competitive reassignment under scaling. It is not a second
four-pattern consolidation experiment, and it must not delay the primary frequency result.

## Required outputs

Implement:

```text
experiments/frequency_scaling.py
tests/test_frequency_scaling.py
docs/FREQUENCY_SCALING_RESULTS.md
```

Write runs under:

```text
experiments/runs/frequency_scaling/<run-id>/
```

The CLI must support:

```text
--phase matched|developmental|controls|transitions|report|all
--seeds
--patterns
--output-root
--run-dir
--resume
--quick
```

`--output-root` creates and prints one parent run directory. `--run-dir` resumes exactly
one existing parent. Reject supplying both. Each matrix cell must have a deterministic
condition slug, a resolved-config hash, and an atomic terminal marker. Resume may skip
only a completed cell with a matching config hash. Interrupted attempts remain inspectable
and rerun into a new attempt directory.

At minimum write:

```text
manifest.json                 parent contract, git state, complete matrix/config
cells/<slug>/summary.json     complete per-run and per-column result
cells/<slug>/metrics.csv      per-window/per-column metrics
aggregate.csv                 one row per protocol/seed/pattern/arrangement/k/column
aggregate.json                complete machine-readable aggregation
sequence_digests.json         sequence digest + first 120 bits for every measured column
docs/FREQUENCY_SCALING_RESULTS.md
```

Use `ReplayRecorder` for representative seed-1 `row 1`, spread, feedback-on conditions:

```text
k = 1,2,3,5,9
```

Use the developmental protocol for these representative replays. Record at a bounded
stride and checkpoint weights often enough for exact displayed frames. Do not write full
replays for every matrix cell.

Every artifact must record:

- complete resolved engine params;
- seed, pattern, protocol, arrangement, active coordinates, and `k`;
- training/settling status and boundaries;
- feedback condition;
- topology/spec fingerprint;
- source commit and dirty flag;
- exact classification definitions and window sizes.

## Report requirements

`docs/FREQUENCY_SCALING_RESULTS.md` must answer:

1. At which active-patch counts did every active L1 column achieve exact alternation?
2. Did the result reproduce across seeds, patterns, and spatial arrangements?
3. What was the first/minimum stable exact-alternation scale, if one exists?
4. Where did behavior remain grouped half-rate, irregular, unsuppressed, silent, or
   unsettled?
5. How did L2, C, and deduplicated feedback-reset rates scale with `k`?
6. Did matched weights and developmental runs agree?
7. Did feedback-off controls remove the suppression effect?
8. Did changed patterns recruit different owners, and how did transition cadence depend
   on `k`?
9. Does exact alternation begin only after the corresponding C association is mature and
   causally active, or does auto-pacing alternate output independently of certainty?
10. Does the existing topology have a defensible supported operating regime?
11. Which standing problems remain unresolved?

Include per-column results, not only aggregates. A single failed active column must remain
visible.

State plainly:

- `1010...` is exact halving;
- `111000...` is grouped half-rate and fails the temporal contract;
- an average near `0.5` is insufficient;
- exact auto-paced alternation is not by itself evidence of certainty; the report must
  align its onset with C maturity, C spikes, and effective feedback resets;
- sparse one-patch L2/C confirmation remains a declared negative/control dimension and is
  not evidence that the intentional cap should be removed;
- scaling success does not make the current reset robust to irregular or interrupted input;
- no model parameter or topology was changed.

Update `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md` only after results exist. Add
measured conclusions and artifact links without deleting the predeclared contract.

## Fast tests

Add deterministic tests that do not run the scientific matrix:

- patch subset construction and validation for every `k` and both arrangements;
- `k=9` arrangement deduplication;
- exact resolved dashboard-config assertion;
- eligible-presentation alignment from causal deliveries;
- full-bank feedback-volley deduplication;
- cadence classification for:
  - `1010`;
  - `0101`;
  - `111000`;
  - irregular reduced;
  - unsuppressed;
  - silent;
- run-length and transition metrics;
- owner-settling rule on synthetic windows;
- inactive-column exclusion;
- matched snapshot transfer/freeze invariance;
- feedback-off condition changes only `c_feedback_reset`;
- condition slug/config hash uniqueness;
- resume accepts only matching completed cells;
- report aggregation preserves every column.

Do not place 6,000–60,000-boundary scientific runs in pytest.

## Execution order

Run:

1. focused baseline tests;
2. implement the harness and fast tests;
3. `--quick` smoke run;
4. inspect quick artifacts manually;
5. Protocol A matched-weight matrix;
6. feedback-off controls;
7. Protocol B developmental matrix;
8. transition probes;
9. aggregate and write the report;
10. focused tests;
11. full suite;
12. `git diff --check`.

The quick run is never scientific evidence. It should use seed 1, `row 1`,
`k={1,2,9}`, one arrangement, shortened training/measurement windows, and a run id
containing `quick`.

Suggested commands:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_feedback_hard_reset.py \
  tests/test_patch_pattern_composition.py \
  tests/test_tiled_cc_engine.py \
  tests/test_basic_consolidation.py \
  tests/test_replay_recorder.py -q

PYTHONPATH=. .venv/bin/python experiments/frequency_scaling.py \
  --phase all --quick --output-root experiments/runs/frequency_scaling

# Reuse the exact printed RUN_DIR.
PYTHONPATH=. .venv/bin/python experiments/frequency_scaling.py \
  --phase matched --seeds 1,2,3,4 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/frequency_scaling.py \
  --phase controls --seeds 1 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/frequency_scaling.py \
  --phase developmental --seeds 1,2,3,4 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/frequency_scaling.py \
  --phase transitions --seeds 1 --run-dir "$RUN_DIR" --resume
PYTHONPATH=. .venv/bin/python experiments/frequency_scaling.py \
  --phase report --run-dir "$RUN_DIR"

PYTHONPATH=. .venv/bin/python -m pytest tests/test_frequency_scaling.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
git diff --check
```

## Stop conditions

Stop and report rather than changing the model if:

- the dashboard config no longer matches the frozen contract;
- the intentional cap is not active on pattern-detector afferents;
- feature-gated removal or another overlapping working-tree change is incomplete in a way
  that blocks the retained topology;
- causal eligibility cannot be observed without modifying production scheduling;
- non-finite state appears;
- a retained topology or golden changes;
- obtaining a preferred cadence would require a parameter, equation, timing, or topology
  change.

Scientific failure is not a stop condition. Preserve timeouts, sparse C events, grouped
half-rate traces, and patch-count dependence as results.

## Completion report

Report:

- exact files changed;
- exact resolved configuration;
- complete executed matrix and any not-evaluable cells;
- matched-weight versus developmental conclusions;
- per-count cadence classification summary;
- minimum exact-alternation scale, if any;
- L2/C/reset scaling;
- feedback-off causal result;
- transition-owner result;
- artifact and report paths;
- focused and full test commands/counts;
- `git diff --check`;
- confirmation that topology, `theta/2` cap, learning equations, and production timing
  were unchanged;
- anything still running.

Do not commit or push.
