# Read-Only Dashboard Replay Player

A minimal **Load Test** replay mode for the dashboard. Load one self-contained
`replay.snn.jsonl` produced by the headless recorder
([REPLAY_RECORDER.md](REPLAY_RECORDER.md)) and watch, pause, step, scrub, and
marker-jump the recorded test like a movie, through the dashboard's existing
topology / renderer / inspector / receptive-field / raster / charge / weights
views. It is a **playback feature, not a simulation engine**: it never reruns
learning in the browser, never infers missing scientific state, and never mutates
the recorded file or the live engine.

Supported replay schema: **`snn.replay` version 1** (see
`REPLAY_SCHEMA_NAME` / `SUPPORTED_SCHEMA_VERSION` in `frontend/replay.js`). A file
declaring any other schema name or version is rejected before replay begins.

## Using it

1. Start the dashboard and pause it.
2. Click **📼 Load Test** in the top bar and pick a `replay.snn.jsonl`.
3. The recorded topology and first frame replace the live view; a **REPLAY** strip
   appears under the top bar and the status pill reads `Replay`. Loading a replay
   issues the existing pause once so the hidden live sim stops advancing.
4. Play / pause, step ±1 frame, pick a speed, drag the timeline, or jump between
   recorded markers (ticks on the timeline + prev/next + a marker selector).
5. **✕ Exit Replay** (or `Esc`) resyncs from `/api/state` and restores the live —
   still paused — engine.

A malformed or unsupported file surfaces a visible error and does **not** enter
replay; the live view stays intact.

## How live and replay share one rendering path

`frontend/app.js` exposes two shared application functions used by **both** the
live WebSocket and the replay player — there is no forked renderer:

- `applyTopology(topo)` — rebuilds `store` (meta / weights / confidence / pattern
  vectors), the Three.js renderer, charts, controls, receptive fields, and the
  raster / charge / weights overlays. Clears any selection that no longer resolves.
- `applyDynamic(dyn)` — updates every view + inspector from one dynamic frame,
  exactly as the live path always has (including `changed_synapses`).

Live frames call these from `onMessage`. The player calls the same functions:
`applyTopology(replay.topology)` on entry, `applyDynamic` for sequential
play/step-forward, and a bounded-window rebuild (`replayBulkSeek`) for scrubbing.

While replay is active:

- Incoming live `topology`/`dynamic` WebSocket messages are ignored for display
  (`onMessage` returns early). A reconnect still updates the connection pill.
- Every UI-driven mutation POST is refused at a single choke point
  (`api.post` checks `replayActive`), so replay can never send a pattern, weight,
  topology, reset, reseed, or config mutation to the live engine — even via a
  control the visual-disable pass missed. Live mutation controls are also disabled
  and the pattern/patch/config/RF grids are neutralized via a `replay-mode` body
  class.

## Backward seeking: weights and chart history

Recorded frames only carry `changed_synapses`, and a `record_every > 1` recording
skips timesteps, so plain forward accumulation drifts between weight checkpoints.
A seek must therefore reconstruct, not carry weights over from a future frame.

`frontend/replay.js` is the pure, DOM-free core (unit-tested under `node --test`):

- `reconstructWeightsAt(replay, frameIndex)` — live weights from the nearest
  preceding `weight_checkpoint` (or the header's initial weights) plus intervening
  `changed_synapses`. Mirrors the recorder's own `reconstruct_weights_at`.
- `advanceWeights(replay, weights, pos)` — advances a weights Map one frame:
  applies that frame's `changed_synapses`, then **snaps to a checkpoint** recorded
  at that frame (authoritative). Used to rebuild a window in O(window), not O(n²).

`replayBulkSeek(replay, pos)` (in `app.js`) seeds from the canonical weights just
before the bounded window `[max(0, pos-HISTORY+1) .. pos]`, advances frame by
frame into the raster / charge / weights history views, then shows the target
frame on the 3D renderer / inspector / receptive / controls with
`renderer.setWeights(...)` so no future-frame weights survive. Each history view's
`update()` schedules a single coalesced draw, so rebuilding a 1500-frame window
redraws once, not 1500 times. Sequential play/step-forward stays on the cheap
`applyDynamic` path (append one frame), matching the live stream exactly.

Sampled recordings (`record_every > 1`) show the recorded engine timestep on every
frame; skipped timesteps are simply absent, never drawn as measured zeros.

## Deliberately deferred (non-goals for this task)

Backend replay endpoints / a run database; CSV visualization (CSV stays an
external quantitative artifact); replay editing or resume-from-frame;
export/video; interpolation between recorded frames; any scientific
Basic/continuous/noise/composition logic; NEST; a new frontend framework;
compression/indexing of `replay.snn.jsonl`; richer timeline work.

## Tests

- **JS core** — `node --test tests/replay.parser.test.mjs`: supported records
  parse; malformed JSON / wrong schema version / duplicate ids / unknown
  weight-change ids / non-finite values / non-monotonic frames / unknown record
  types are rejected before replay begins; forward, backward, and random-order
  seeks reconstruct exactly the recorded weights (cross-checked against the
  recorder's own reconstruction); the bounded-window rebuild and checkpoint-snap
  reproduce those weights for any window size.
- **Fixture guard** — `pytest tests/test_replay_player_fixture.py`: the committed
  fixture still parses with the recorder's readers and its `expected.json` matches
  the recorder's reconstruction (so a schema change that forgets to regenerate the
  fixture fails loudly).
- **Fixture generation** — `PYTHONPATH=. python tests/fixtures/make_replay_fixture.py`
  regenerates `tests/fixtures/replay_fixture.snn.jsonl` + `.expected.json` through
  the **real** recorder (never hand-maintained).

## Manual browser acceptance checklist

There is no JS DOM/browser test harness in this repo, so rendering behavior is
verified manually:

1. Start the dashboard; pause it.
2. Generate the fixture (above) or use any recorder run's `replay.snn.jsonl`.
3. **Load Test** → pick the file. Confirm: the recorded topology replaces the live
   view, the **REPLAY** strip shows run name / seed / condition
   (incl. hierarchical feedback) / frame index / timestep / phase / pattern, and
   the status pill reads `Replay` (the engine is **not** running).
4. Play, pause, step ±1, and change speed. Confirm playback follows recorded frame
   order and the shown timestep is the recorded one (sampled files skip steps).
5. Scrub the timeline forward and backward across a weight checkpoint and a pattern
   marker. Open the **Charge**, **Spike Raster**, and **Weights / time** overlays
   and confirm their histories rebuild truthfully after a backward seek (no
   carried-over future data), and that neuron spikes/charge, input pixels, column
   winners, hard resets, and inspector values agree with the recorded frame.
6. Confirm live mutation controls (start/reset/reseed, pattern/pixel/patch,
   manual firing, config apply, topology editor, RF weight editing) are disabled or
   inert, and that no `/api/*` mutation is sent (network tab stays quiet on click).
7. **Exit Replay** → confirm the live view is restored from `/api/state` and the
   engine remains paused (it is not silently resumed).
8. Load a malformed file (e.g. truncate the fixture mid-line) → confirm a visible
   error, no entry into replay, and the live state intact.
9. Load → exit → load again a few times → confirm no duplicated event-log lines,
   no runaway FPS, and no growing memory (histories stay bounded to their windows).
