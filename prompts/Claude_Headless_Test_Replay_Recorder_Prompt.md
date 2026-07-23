# Claude Prompt: Headless Test Replay Recorder and Artifact Contract

## Objective

Build a shared, versioned recording layer for long-running headless SNN experiments.
Experiment scripts must be able to run without FastAPI, WebSockets, or the browser and
write enough information to:

1. analyze the experiment quantitatively from CSV;
2. inspect its declared pass/fail result;
3. replay the simulation later in the existing dashboard as a scrub-able movie; and
4. recover useful evidence from a run that was interrupted partway through.

This task creates recording infrastructure only. Do **not** implement the Basic
consolidation, continuous-learning, noise-invariance, NEST, or composition experiments in
this change. Do not decide whether hierarchical feedback is enabled during the future
Basic experiment. The artifact metadata must be able to record that condition truthfully
when the experiment is implemented later.

## Read and inspect first

Read these files before editing:

- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`
- `backend/simulation.py`, especially `step()`, `dynamic_state()`, `topology()`,
  `changed_synapses`, `column_winners`, `hard_reset_events`, and `latency_ties`
- `backend/serializer.py`
- `experiments/tiled_cc_experiment.py`
- `tests/test_tiled_cc_experiment.py`
- `frontend/app.js`
- `frontend/renderer.js`
- `frontend/raster.js`
- `frontend/charge.js`
- `frontend/weights.js`

Inspect the complete working-tree diff before editing. The repository may contain
uncommitted work from other tasks; preserve it and avoid unrelated cleanup.

## Non-negotiable scientific constraints

- Recording must not alter neuron state, event ordering, scheduler behavior, learning,
  random-number consumption, topology, weights, thresholds, inhibition, or input timing.
- Do not add experiment-specific branches to `SimulationEngine`.
- Do not add a dashboard/server dependency to the headless experiment path.
- Do not make test success depend on dashboard winner labels or visual styling.
- Do not change any production equation, topology preset, learning parameter, or timing
  constant.
- Do not claim that replay proves scientific correctness. It is an observation artifact.
- Recorder-on and recorder-off runs with the same seed and inputs must be behaviorally
  identical.

## Artifact directory contract

Create a small reusable module in `experiments/` (choose a clear name such as
`experiments/replay_recorder.py`) that writes one directory per test run:

```text
experiments/runs/<run-id>/
├── manifest.json
├── replay.snn.jsonl
├── metrics.csv
└── summary.json
```

Do not commit generated run directories or large replay artifacts. Add only the narrow
ignore rule needed for generated `experiments/runs/` contents, while preserving a tracked
README or `.gitkeep` only if useful.

The recorder API must let an experiment supply an explicit output root so CI/tests can use
a temporary directory instead of the repository.

### `manifest.json`

The manifest is human-readable run metadata. Include at least:

- replay schema name and integer schema version;
- run ID and experiment name;
- creation/completion timestamps in UTC;
- status: `running`, `completed`, `failed`, or `interrupted`;
- seed;
- topology/preset name and complete resolved engine parameters needed to interpret it;
- the experiment schedule/configuration supplied by the caller;
- declared experimental conditions supplied by the caller, including a generic field for
  hierarchical-feedback mode (`intact`, `disabled`, `not_applicable`, or an equally clear
  documented vocabulary); do not infer this from neuron IDs;
- recording stride/checkpoint interval;
- starting/ending timestep and number of recorded frames;
- repository commit when discoverable and whether the worktree was dirty; inability to
  query Git must not make a run fail;
- paths and schema versions for the replay, metrics, and summary files.

Update status safely at completion/failure. Use an atomic replace for small JSON metadata
files so interruption cannot leave half-written JSON. Preserve the append-only replay
written before an interruption.

### `replay.snn.jsonl`

This is the self-contained dashboard replay artifact. A user must be able to load this
single file later without also selecting the manifest or CSV.

Use newline-delimited JSON with a documented, versioned record union:

1. Exactly one `header` record first. It contains:
   - schema name/version;
   - run metadata needed by the player;
   - the complete initial `engine.topology()` payload, including initial live weights;
   - stable neuron and synapse ordering/IDs;
   - experiment conditions and recording policy.
2. Zero or more `marker` records for semantic transitions such as training phase, pattern
   change, learning freeze/unfreeze, recall, timeout, or mapping collision. Marker data is
   supplied explicitly by the experiment; the recorder must not guess scientific phases.
3. `frame` records in monotonically increasing timestep/frame-index order. Each frame
   contains:
   - frame index and engine timestep;
   - current experiment annotation (`phase`, `pattern`, and optional tags/notes);
   - a canonical dynamic payload compatible with the existing dashboard's `dynamic`
     message consumer: all current neuron records plus input, winner,
     `column_winners`, `changed_synapses`, emitted events, inhibitory pulses, hard resets,
     latency ties, and statistics;
   - only the new event-log entries needed for that frame, not the same rolling log copied
     repeatedly forever.
4. Periodic `weight_checkpoint` records containing all currently weighted synapses keyed
   by stable synapse ID. Frames between checkpoints retain authoritative
   `changed_synapses`. A player must be able to reconstruct weights for an arbitrary
   frame from the nearest preceding checkpoint (or the header's initial weights).
5. An optional final `result` record containing the same high-level outcome written to
   `summary.json`.

Keep the format readable and explicit. Do not encode opaque pickles, Python class names,
NumPy binary blobs, or executable data. Reject non-finite JSON values instead of writing
invalid `NaN`/`Infinity` tokens.

The JSONL writer must append and flush at a documented bounded interval so partial runs
remain parseable. It must not keep the full run history in memory.

### Sampling policy

Support configurable state-frame recording, including `record_every=1` for exact short
diagnostics and a larger stride for long training runs. Pattern/phase markers and the
experiment's quantitative metrics remain independent of visual-frame sampling.

Do not silently invent missing boundary states in the artifact. If the recording stride
is greater than one, record the stride and actual timestep on every frame so the dashboard
can display sampled playback honestly.

The recorder must expose enough lifecycle methods for a future experiment to:

- start a run;
- attach/update current annotations;
- add a semantic marker;
- record a frame after an engine step;
- append one or more metric rows;
- write a successful or failed summary;
- close cleanly on an exception or interruption.

Use buffered file I/O. Add a small benchmark or measurement in the implementation report
comparing a short recorder-off run with recorder-on at stride 1 and at a coarser stride.
Report elapsed time and artifact size; do not impose an arbitrary performance gate.

### `metrics.csv`

CSV is the flat quantitative analysis artifact, not the source of dashboard replay.
Provide a generic, streaming CSV writer whose columns are declared once by the experiment.
It must:

- write one header;
- reject rows with missing/unknown columns unless the API explicitly documents optional
  columns;
- append rows without retaining them all in memory;
- use ordinary RFC-compatible CSV quoting;
- permit fields such as run ID, seed, topology, condition, phase, pattern, timestep,
  column ID, neuron ID, and measured values;
- never stringify an entire dynamic frame into one CSV cell.

Do not choose Basic-test ownership thresholds or metrics in this infrastructure task.

### `summary.json`

The caller supplies the scientific outcome. Store:

- completion/failure status;
- declared checks and their Boolean results;
- final result/measurements supplied by the experiment;
- failure or timeout reason when present;
- references to the manifest, replay, and metrics artifacts.

The recorder must not turn an experiment's expected hypothesis into a hard-coded pass.

## Recorder API and separation of concerns

Keep format serialization separate from experimental policy. A good design will have a
small recorder/context-manager class and, if needed, pure validation/reading helpers.

The experiment remains responsible for:

- constructing and stepping the engine;
- choosing patterns and schedules;
- deciding whether feedback/learning is enabled;
- computing consolidation or other metrics;
- deciding when to stop and whether checks passed.

The recorder is responsible only for:

- truthful metadata;
- bounded streaming writes;
- schema-valid replay records;
- checkpoints;
- lifecycle/failure handling.

Do not call private frontend code from Python. Reuse the existing backend topology and
dynamic payload vocabulary so the later dashboard player can use the existing rendering
path.

## Documentation

Add a concise schema/usage document under `docs/` and a tiny example showing how a
headless experiment wraps an engine loop. The example may use a very short run and a
temporary/output directory; it must not masquerade as a scientific Basic-test result.

Document a background invocation pattern, for example:

```bash
nohup env PYTHONPATH=. .venv/bin/python experiments/<future_experiment>.py \
  --output-root experiments/runs > experiments/runs/<run-id>.log 2>&1 &
```

Do not add a daemon manager or process supervisor in this task.

## Required tests

Add focused tests, using temporary directories, that prove:

1. A short run creates all four artifacts with the declared schema/version.
2. The JSONL header is first, self-contained, and carries a valid topology.
3. Frame indices/timesteps are monotonic and dynamic payloads contain the fields required
   by the current dashboard consumer.
4. Markers preserve caller-supplied phase/pattern annotations.
5. Initial weights plus frame changes/checkpoints reconstruct the exact live weights at
   selected frames, including backward/random access reconstruction by a pure helper.
6. `record_every` is honored and skipped timesteps are represented honestly.
7. CSV headers/rows round-trip with commas and quoted text.
8. A simulated exception/interruption leaves parseable JSONL, marks the manifest/summary
   appropriately, and does not erase completed frames.
9. Non-finite values and schema/ID mismatches fail clearly.
10. Recording is behaviorally inert: identical seeded input schedules with recording on
    and off produce identical spikes, winners, weights, and final neuron state.
11. The recorder retains bounded memory as frame count increases; do not store all frames
    internally.
12. Existing tiled experiments and the full suite remain unchanged.

Run focused recorder/serializer/experiment tests, then the full suite and
`git diff --check`.

## Completion report

At completion, report:

- every changed file;
- the exact replay schema and lifecycle;
- a sample artifact tree and reproduction command;
- recorder-off/on timing and output sizes for the short benchmark;
- focused and full-suite test commands/totals;
- any deliberately deferred compression, indexing, or dashboard work.

Do not implement the dashboard replay UI in this task. That is a separate prompt.
