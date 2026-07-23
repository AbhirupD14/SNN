# Headless Replay Recorder and Artifact Contract

`experiments/replay_recorder.py` is a shared, versioned **recording layer** for
long-running headless SNN experiments. It lets an experiment run without FastAPI,
WebSockets, or a browser while writing enough information to (1) analyze the run from CSV,
(2) inspect a declared pass/fail result, (3) replay the simulation later in the dashboard
as a scrub-able movie, and (4) recover evidence from a run interrupted partway through.

It is **recording infrastructure only**. It does not implement any experiment, does not
decide whether hierarchical feedback is enabled, and does not prove scientific
correctness — a replay is an *observation artifact*.

## Scientific inertness

Every engine observation the recorder makes is a read-only accessor (`topology()`,
`dynamic_state()`, `_live_weight`). The recorder never touches neuron state, event
ordering, the scheduler, learning, the RNG stream, weights, thresholds, inhibition, or
input timing. A recorder-on run and a recorder-off run with the same seed and inputs are
behaviorally identical (`tests/test_replay_recorder.py::test_recording_is_behaviorally_inert`).

## Run directory

One directory per run, under a caller-supplied `output_root` (tests/CI pass a temp dir):

```text
<output-root>/<run-id>/
├── manifest.json          human-readable run metadata (atomic replace)
├── replay.snn.jsonl       self-contained, append-only dashboard replay movie
├── metrics.csv            flat streaming quantitative analysis artifact
└── summary.json           caller-supplied scientific outcome (atomic replace)
```

Generated run directories are **not** committed — `experiments/runs/` is already gitignored.

## Schema versions

| Constant | Value |
| --- | --- |
| `REPLAY_SCHEMA_NAME` | `snn.replay` |
| `REPLAY_SCHEMA_VERSION` | `1` |
| `MANIFEST_SCHEMA_VERSION` | `1` |
| `METRICS_SCHEMA_VERSION` | `1` |
| `SUMMARY_SCHEMA_VERSION` | `1` |

## `replay.snn.jsonl` — versioned record union

Newline-delimited JSON. One record per line, all values finite (no `NaN`/`Infinity` are
ever written — such values raise `ValueError` instead). The self-contained artifact can be
opened alone, without the manifest or CSV.

1. **`header`** — exactly one, first. Carries the schema name/version, run metadata (run
   id, experiment, `created_utc`, seed, preset/topology name), declared `conditions`
   (including `hierarchical_feedback`), the recording policy, stable `neuron_order` /
   `synapse_order`, and the complete initial `engine.topology()` payload **including
   initial live weights**.
2. **`marker`** — zero or more caller-supplied semantic transitions (phase change, recall,
   timeout, mapping collision, freeze/unfreeze, …). The recorder never guesses phases;
   `data` is whatever the experiment passes.
3. **`frame`** — monotonically increasing `frame_index` and engine `timestep`. Carries the
   current `annotation` (`phase`, `pattern`, `tags`, `notes`), the `record_every` stride,
   and a canonical `dynamic` payload compatible with the dashboard's `dynamic` consumer:
   all neuron records plus `input`, `winner`, `column_winners`, `changed_synapses`,
   `emitted`, `inhibitory_pulses`, `hard_reset_events`, `latency_ties`, and `stats`. The
   `log` field holds **only the new event-log entries** for that frame, not the rolling
   log copied forever.
4. **`weight_checkpoint`** — periodic full snapshot of every weighted synapse keyed by
   stable synapse id. Frames between checkpoints retain authoritative `changed_synapses`.
5. **`result`** — optional final record mirroring `summary.json`.

### Weight reconstruction

`reconstruct_weights_at(records, frame_index)` (pure reader) rebuilds live weights at any
frame from the **nearest preceding checkpoint** (or the header's initial weights) plus the
intervening `changed_synapses`. It supports backward/random access.

Precision note: checkpoints and the header capture weights at 6 decimals; the production
`changed_synapses` dashboard payload is 4 decimals. Reconstruction is therefore exact at
checkpoint frames and reproduces live weights to the recorded 4-dp change precision
between checkpoints. Choose `checkpoint_every` accordingly for a long run.

### Sampling policy

`record_every=1` records every step (exact short diagnostics); a larger stride records
every N-th step for long training runs. Every written frame stores the stride and the
**actual** engine timestep, so sampled playback is honest and skipped timesteps are simply
absent — never invented. Pattern/phase markers and CSV metrics are independent of the
visual-frame stride.

## `metrics.csv`

`MetricsWriter` is a generic streaming CSV writer whose columns are declared once. It
writes a single header, appends rows without retaining them, uses RFC-4180 quoting, and
rejects rows with unknown columns or missing required columns (columns can be declared
optional). It never stringifies a whole dynamic frame into a cell.

## `summary.json`

The **caller** supplies the scientific outcome. Stored: status
(`completed`/`failed`/`interrupted`), declared `checks` and their Boolean results,
`all_checks_pass`, the final `result`/measurements, a `failure_reason` when present, and
references to the other artifacts. The recorder never turns an expected hypothesis into a
hard-coded pass.

## `manifest.json`

Human-readable run metadata: replay schema name/version, run id and experiment, UTC
create/complete timestamps, `status`, seed, preset/topology name and the complete resolved
engine parameters, the caller's schedule and declared conditions (including
`hierarchical_feedback`), the recording stride/checkpoint interval, start/end timestep and
recorded frame count, the repo commit and dirty flag when discoverable (Git failure never
fails a run), and the paths/schema-versions of every artifact. Written with an atomic
temp-file `os.replace`, so an interruption cannot leave half-written JSON.

## Status lifecycle

`running` → one of `completed`, `failed`, `interrupted`.

- `finish(STATUS_COMPLETED | STATUS_FAILED, checks=…, result=…)` writes `summary.json`, an
  optional final `result` replay record, and updates the manifest atomically.
- An unhandled exception leaving the `with` block marks the run `interrupted`
  (`KeyboardInterrupt`) or `failed` (any other), writes the summary with a
  `failure_reason`, and **preserves every frame written before the interruption** — the
  append-only JSONL stays parseable. The original exception is never suppressed.

## Usage

```python
from backend.simulation import SimulationEngine
from experiments.replay_recorder import ReplayRecorder, STATUS_COMPLETED

engine = SimulationEngine(seed=1, topology="tiled_cc", cc_e_count=8, leak_rate=0.0)
engine.set_patch(1, 1); engine.set_pattern("row 1")

with ReplayRecorder(engine, experiment="basic", output_root="experiments/runs",
                    record_every=5, checkpoint_every=100,
                    hierarchical_feedback="not_applicable",
                    metrics_columns=["timestep", "phase", "value"]) as rec:
    rec.set_annotation(phase="train", pattern="row 1")
    rec.marker("phase_start", data={"phase": "train"})
    for _ in range(steps):
        dyn = engine.step()
        rec.record_frame(engine)                 # honors record_every
        rec.metrics.append_row({"timestep": engine.timestep, "phase": "train",
                                "value": ...})   # experiment-computed metric
    rec.finish(STATUS_COMPLETED, checks={"ok": True}, result={...})
```

A runnable demonstration (short run, temp dir by default) lives in
`experiments/replay_recorder_example.py`. It is a demonstration of the contract, not a
scientific result.

## Background invocation

A future long experiment can be launched detached; the recorder streams and flushes at a
bounded interval, so the artifacts stay parseable while it runs:

```bash
nohup env PYTHONPATH=. .venv/bin/python experiments/<future_experiment>.py \
  --output-root experiments/runs > experiments/runs/<run-id>.log 2>&1 &
```

No daemon manager or process supervisor is part of this contract.

## Deliberately deferred

- Dashboard replay **player** UI — now implemented; see
  [REPLAY_PLAYER.md](REPLAY_PLAYER.md).
- Compression and indexing of `replay.snn.jsonl` (frames are full uncompressed JSON; a
  long stride keeps size manageable — see the benchmark in the implementation report).
- Any Basic-consolidation experiment, its ownership thresholds, or its metrics.
