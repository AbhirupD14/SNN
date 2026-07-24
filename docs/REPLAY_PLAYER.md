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

## Branch: "Continue with These Weights"

The replay player has one deliberately limited action beyond pure playback: at the selected
frame, **Continue with These Weights** reconstructs the recorded synaptic weights and installs
them into a freshly reset live engine, then exits replay and leaves the live engine paused so
you can step, change inputs, and continue learning from those weights.

This is a **new branch from learned weights, not an exact resume.** Only the complete weight
snapshot (and, optionally, the raw input vector) is restored. Everything transient is cleanly
reset and is *never* claimed to be restored:

- membrane voltage, inhibitory/excitatory conductance, refractory & activity state;
- pending feedforward / basal / apical / reset / relay events and coincidence eligibility;
- continuous stimulation, the event log, the RNG position, the recorded timestep;
- experiment-script control flow.

The fresh engine starts at its normal reset timestep; the recorded source timestep is
provenance only.

### Workflow

1. Pause the dashboard, **Load Test** a `replay.snn.jsonl`, play/scrub to a frame.
2. Press **↪ Continue with These Weights**. Playback pauses and a confirmation appears naming
   the frame, recorded timestep, and reconstruction precision, and stating that the current
   live learned state will be replaced.
3. Optional checkboxes: **Restore the displayed input pixels** (default on) and — only when the
   selected frame is *not* itself a checkpoint — **Use nearest preceding checkpoint (frame N)**.
4. Confirm. The backend independently re-validates compatibility and the snapshot, atomically
   resets the engine and installs the weights, then replay exits through the normal `/api/state`
   resynchronization path with the live runner left paused.

A validation failure shows the backend's message *without leaving replay* — the loaded replay,
selected frame, hidden live engine, and transport all stay usable.

### Compatibility contract

A branch is allowed only into a live engine whose computational graph and model-affecting
configuration match the recording. The backend builds the canonical contract from **both** the
recorded topology (sent in the request) and its own live `engine.topology()`, and compares —
it never trusts the browser's "compatible" assertion and never infers topology from neuron-id
prefixes. Compared: complete neuron identity + archetype/role/column metadata; complete synapse
identity/source/target/kind/sign; the set of weighted synapses; tiling family/variant/
dimensions/columns; the model-affecting parameters (threshold, leak, refractory, inhibition,
delay, learning rates, the active dual FE/FES rule and its FE/FES parameters, construction
dimensions); and the **seed** (required, because seed-dependent positions/distance-factors are
a non-restored computational property). On a mismatch the backend returns a precise, actionable
error (which topology/config to select) with HTTP 409 and leaves the live engine untouched; it
never applies that configuration automatically.

**Documented exclusions** from the contract: display-only state (selected patch, patch labels,
current raw input, current timestep, running flag), recorder metadata, and the headless-only
base update modes (`e_weight_update_mode` / `c_weight_update_mode`) which are not represented in
a serialized topology payload (the dashboard-switchable `dual_fe_fes` flag *is* compared).

### Canonical snapshot & precision

Weights come from the same tested reconstruction as scrubbing — nearest preceding
`weight_checkpoint` (or the header's initial weights) plus intervening `changed_synapses` up to
the chosen frame — **never** the renderer's mutable current weights, so a backward seek can
never leak a future frame's weights into a branch. The snapshot is complete for every synapse
the replay records with a non-null weight:

- **mutable** ff / predictive / coincidence-basal weights are restored (bound-checked against
  the *active* model rule — a cap-free rule accepts any finite value ≥ its floor with no
  invented historical ceiling; an out-of-range value is **rejected, never clipped**);
- **fixed** weighted edges (pretrained delivered charge) are compatibility-checked but never
  mutated into plastic weights;
- unweighted structural edges never acquire a weight.

Schema-1 precision differs by source: header/checkpoint weights are six-decimal, intervening
`changed_synapses` four-decimal. The confirmation says which applies — a checkpoint frame uses
*"the recorder's six-decimal checkpoint precision"*; a non-checkpoint frame notes *"four-decimal
recorded deltas"*. **Use nearest preceding checkpoint** re-selects the checkpoint frame (its
weights, timestep, and provenance) so an exploratory branch can opt into six-decimal precision.
Six-decimal serialization is *not* bit-exact restoration of the former in-memory float.

### Input restoration

**Restore the displayed input pixels** (default on) restores only the chosen frame's canonical
raw input vector — it does not infer experiment phase, held-pattern schedule, or patch labels.
For tiled topologies a raw-vector restore uses `set_input`, which clears any semantic
patch-pattern labels (the existing meaning of `set_input`); no labels are manufactured. With
restoration **off**, the branch starts with a blank input so pressing Play cannot unintentionally
continue the hidden replay stimulus.

### Endpoint & atomicity

`POST /api/replay/branch-weights` (schema `snn.branch` v1). The payload carries the branch/replay
schema+version, source run/frame/timestep/seed, precision classification, the recorded topology
contract, the complete reconstructed weight map, and the optional raw input. The engine loader is
two-phase and atomic: `_prepare_branch` fully validates the snapshot (and the input vector)
against the live graph *without mutating anything* and raises on the first problem
(missing/extra/unknown/non-numeric/non-finite/out-of-range/role-incompatible); only then does the
engine reset transient state, rebuild, and install every mutable weight via direct reference-map
writes (no sequential public setters, no rounding, no clipping). Because the rebuild is
deterministic from `(seed, config)`, the commit re-resolves each edge id and cannot fail after
reset. The request is JSON-only and bounded by the live graph; replay fields are treated purely
as data (never a path, never evaluated).

## Deliberately deferred (non-goals for this task)

Full engine checkpoints / exact transient-state restoration (membrane, conductance, refractory,
pending events, RNG position, timestep, experiment control flow); a server-side run database;
replay editing; browser-side simulation; continuation of an experiment script; topology
inference from id prefixes; a general save-file redesign; a second rendering path. Also still
deferred from the read-only player: CSV visualization; export/video; interpolation between
frames; compression/indexing of `replay.snn.jsonl`; richer timeline work.

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
- **Branch payload (JS core)** — `node --test tests/replay.parser.test.mjs` also covers
  `buildBranchPayload`: weights come from `reconstructWeightsAt` (not renderer state), any
  seek order builds the same snapshot, checkpoint vs delta precision is classified, the
  nearest-checkpoint choice reconstructs the checkpoint frame, the raw input comes from the
  chosen frame, malformed data cannot form a payload, and construction mutates nothing.
- **Branch engine** — `pytest tests/test_replay_branch_engine.py`: ordinary/basal/predictive
  restoration, fixed pretrained checked-not-mutated, dual cap-free weights above the historical
  cap restore unchanged, range/floor violations rejected (not clipped), missing/extra/
  non-finite/non-numeric rejected, atomic (every failure preserves the prior live state),
  transient state cleared + timestep reset, input on/off, and no learning on load.
- **Branch API** — `pytest tests/test_replay_branch_api.py`: schema/topology/config/seed/
  learning-mode validation with actionable 4xx (400 bad request / 409 incompatible), bounded
  request sizes, contract excludes weight values, a failed request never fabricates success,
  and the endpoint forces the runner paused.

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

### Branch ("Continue with These Weights")

10. Record a replay with at least two checkpoints, then alter/reset the live weights. Load the
    replay and scrub to a **checkpoint** frame. Press **Continue with These Weights**, keep
    input restoration on, confirm. Verify: replay exits, the live weights equal the checkpoint,
    transient state is clean, the input is restored, and the engine is **paused**.
11. Step once and verify the *live* simulation (not replay playback) advances from those weights.
12. Repeat at a **non-checkpoint** frame and verify the confirmation shows the four-decimal-delta
    precision warning.
13. Repeat with **Use nearest preceding checkpoint** ticked and verify the source frame shown
    changes to the checkpoint frame.
14. Turn input restoration **off**, branch, and verify the live input is blank.
15. Load a replay recorded on a *different* topology/config (or change the live config first) and
    attempt to branch; verify a visible error, that replay stays active, and the live state is
    untouched.
