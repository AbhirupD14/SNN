# Claude Prompt: Feature-Gated Recall, Robustness, and Four-Competitor Capacity

> **Deferred prerequisite:** Do not execute this prompt against the local-only
> `tiled_cc_feature_gated` topology. First complete and review
> `prompts/Claude_Recursive_Feature_Gated_Hierarchy_Prompt.md`. Then revise this recall
> prompt to target the accepted recursive preset and its identity-preserving inter-layer
> gates. The present body is retained as the intended recall methodology, not as the
> current task.

## Role and objective

Continue from the validated `tiled_cc_feature_gated` implementation. Do not redesign the
topology in this task. The immediate question is whether its successful acquisition and
turnover produce a stable learned representation that can be recalled after learning is
frozen, and whether that result is robust across seeds.

Work in this strict order:

1. Add and run frozen-recall evaluation for the existing eight-competitor feature-gated
   topology, beginning with seed 1.
2. If the seed-1 protocol is sound, run the complete acquisition-and-recall experiment for
   seeds 1 through 8.
3. Only if the eight-competitor primary acceptance gates pass for all eight seeds, add a
   four-competitor feature-gated preset and repeat the same evaluation.

This is an implementation-and-execution task. Do not stop after writing a plan or creating
an unexecuted test harness.

## Read before editing

Inspect all of the following, plus the complete working-tree diff:

- `docs/FEATURE_GATED_TILED_TOPOLOGY.md`
- `experiments/feature_gated_turnover.py`
- `tests/test_feature_gated_turnover.py`
- `tests/test_tiled_cc_feature_gated.py`
- `experiments/basic_consolidation.py`
- the replay recorder and its tests
- the preset registry, network-spec builder, serialization API, and dashboard preset list
- the artifacts from the successful seed-1 Stage A and Stage B feature-gated runs

Reuse the existing feature-gated experiment, recorder, ownership metrics, plastic-weight
snapshot/transfer helpers, and frozen-recall machinery wherever possible. Generalize
existing code instead of building a second incompatible experiment framework.

Preserve all unrelated changes and all prior experiment artifacts. The worktree is dirty.
Do not commit unless explicitly asked.

## Scientific constraints

Do not change any of the following to make the experiment pass:

- neuron equations or scheduler/event semantics;
- topology connectivity, except for the conditional competitor-count-only preset described
  below;
- initial weights, learning rates, thresholds, leak, refractory behavior, delays, timing
  margins, or update equations;
- pattern definitions or their order;
- the feature-gated C/I mechanism or the WTA mechanism;
- recorder values after a run has begun.

Do not reintroduce a weight cap. Do not add normalization, selective inhibition, warm-up
stimuli, priming, or seed-specific tuning. A negative result is valid evidence and must be
reported honestly.

The feature-gated topology has no meaningful `feedback_disabled` condition. Label its
condition clearly (for example `feature_gated`) and do not silently remove edges to imitate
the old Basic ablation.

## Canonical acquisition protocol

Use the existing, already validated feature-gated turnover protocol as the single source of
truth for training:

- Present the four canonical 3x3 patterns in the established order:
  `row1`, `col1`, `diag\\`, `diag/`.
- Tile the same active pattern across all nine RFs during the nine-module experiment.
- Use the established fixed dwell/acquisition duration and reference defaults from
  `feature_gated_turnover.py`.
- Do not replace the fixed dwell with the old Basic harness's earliest-possible
  dominance stop. The acquired state evaluated by recall must be the state produced by the
  validated reference protocol.
- Record the final owner for every `(pattern, L1 module)` pair and the existing turnover,
  dominance, feature-reset, crosstalk, and blank-module metrics.

The goal is four distinct, consistent owners per L1 module, with turnover at every pattern
switch. Keep acquisition metrics and recall metrics separate.

## Frozen recall protocol

After acquisition, capture every learned plastic weight required to reproduce the trained
network, including both ordinary feed-forward learned weights and the feature coincidence
cells' learned basal weights. Use stable edge identity rather than parsing display names.

Implement two evaluations.

### 1. Cold-state recall (primary evidence)

For each trained pattern independently:

1. Construct a fresh engine with the same topology and seed.
2. Transfer the complete learned plastic-weight snapshot by stable edge ID.
3. Do not transfer membrane charge, refractory state, event queues, spike history, winner
   state, or any other dynamic state.
4. Freeze learning on every plastic component, including latency competitors and feature
   coincidence cells.
5. Present that pattern across all nine RFs for a fixed, documented recall window.
6. Determine each module's recalled owner using the established ownership metric.
7. Compare it with the final owner learned for that pattern and module.
8. Assert that every plastic weight remains byte-for-byte unchanged throughout recall.

This is the primary acceptance test because each pattern begins from equivalent clean
dynamic state.

### 2. Sequential frozen recall (stress evidence)

Using a trained engine with learning frozen, present all four patterns sequentially without
resetting dynamic state between patterns. Report owner retention, uniqueness, dominance,
and any carryover separately from cold-state recall. Do not let this stress result obscure
or redefine the primary cold-state result.

The replay/metrics data must make clear which mode produced each observation.

## Eight-competitor execution gates

Run seed 1 first. Inspect its artifacts and confirm that the recorder, weight transfer,
freeze behavior, and ownership comparisons are correct before launching the seed sweep.

Then run seeds 1 through 8 headlessly. It is acceptable to launch the sweep as a background
process once the seed-1 pilot is verified, but do not claim completion while it is still
running. Report its command, PID/log path if applicable, final exit status, and artifact
locations.

For each seed, report at least:

- acquisition: distinct owner count per module, switch turnover, and dominance;
- cold recall: owner match per pattern/module, distinct recalled-owner count, dominance,
  and frozen-weight integrity;
- sequential recall: the same owner-match/uniqueness metrics, clearly marked as a stress
  result;
- causal feature-gate and crosstalk invariants already checked by the reference experiment;
- failure reasons and the first failing seed/module/pattern if anything fails.

The primary eight-competitor gate is all of the following for every seed and all nine
modules:

- acquisition retains four distinct owners and turnover on every switch;
- cold-state recall returns the acquisition owner for all four patterns;
- cold-state recalled owners remain four-way unique;
- no learned weight changes during either frozen-recall mode;
- existing feature-gate isolation/crosstalk invariants remain satisfied.

Keep sequential frozen recall as a separately reported stress gate. If it fails while cold
recall passes, do not reinterpret or hide the failure; identify the carryover behavior.

If any primary eight-competitor gate fails, stop before adding the four-competitor preset.
Preserve the artifacts and report the negative result. Do not tune the model in this task.

## Conditional four-competitor preset

Only after the eight-competitor primary gate passes for all seeds 1 through 8, add a new
feature-gated preset whose only intended model difference is four L1 competitors per
recognition module instead of eight. Use the existing feature-gated spec's `l1_e_count`
parameter rather than copying the builder.

Suggested stable key:

`tiled_cc_feature_gated_l1_4`

Keep the L2 competitor count and all weights, equations, delays, gates, pattern protocols,
and defaults unchanged. Add it to the same registry/API/dashboard surfaces used by the
other presets and give it a clear user-facing name. Do not change or remove an existing
preset.

With 4 L1 competitors and 8 L2 competitors, the expected structural totals are:

- 388 neurons
- 1,176 synapses

Verify those totals from the built graph and add exact connectivity/isolation regression
tests rather than trusting the arithmetic alone.

Run seed 1 first, then seeds 1 through 8 using the identical acquisition and recall
protocol. Four distinct owners consume the full L1 competitor capacity; that is the result
being tested. Apply and report the same gates as in the eight-competitor case. Do not add
the four-competitor preset if the prerequisite gate was not met.

## Artifacts and replayability

Write the normal experiment bundle under a descriptive directory such as:

`experiments/runs/feature_gated_recall/`

Each run must produce machine-readable replay data, metrics, and a summary using the
existing recorder contract. Include enough metadata to reconstruct:

- topology key and topology fingerprint;
- seed and competitor count;
- acquisition/recall phase and active pattern;
- RF/module identities;
- learned and recalled owners;
- spike/event values needed by the dashboard replay player;
- recorder stride and any deliberate reduction in sampling resolution;
- exact configuration and acceptance outcomes.

Record the full seed-1 pilot. To control artifact size, later seeds may use a coarser,
explicitly declared recorder stride, but their scientific ownership and weight-integrity
metrics must remain exact. Never downsample the internal calculation used for pass/fail.

Produce an aggregate CSV/JSON summary across seeds in addition to per-run artifacts.

## Tests

Add focused, deterministic tests that do not run the full long experiment in pytest. Cover
at least:

- plastic-state discovery, snapshot, transfer, and freeze for feature-gated topology,
  including feature-C basal weights;
- cold recall construction without transfer of dynamic state;
- owner-match and owner-mismatch evaluation;
- frozen-weight integrity checks;
- reuse of the canonical acquisition state/protocol rather than a divergent short-stop
  schedule;
- recorder schema/metadata for acquisition, cold recall, and sequential recall;
- conditional four-competitor registry, serialization, graph counts, and connectivity if
  and only if that phase is reached;
- no regressions to existing topology presets and golden tests.

Run the focused tests first. After the scientific runs, run the full test suite and
`git diff --check`. Do not modify tests merely to bless incorrect output.

## Required completion report

At completion, report:

1. files changed and why;
2. exact pilot and sweep commands;
3. acquisition, cold-recall, and sequential-recall results by competitor count and seed;
4. aggregate pass counts and the first failure, if any;
5. artifact and log paths;
6. focused and full test results;
7. whether the four-competitor preset was added, with explicit evidence that its gate was
   satisfied first;
8. any limitations or negative scientific findings;
9. confirmation that no parameter/model tuning or unrelated cleanup was performed.

Do not commit unless explicitly asked.
