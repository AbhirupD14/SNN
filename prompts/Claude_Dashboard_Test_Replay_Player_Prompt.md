# Claude Prompt: Read-Only Dashboard Test Replay Player

## Objective

Add a minimal, read-only `Load Test` replay mode to the existing dashboard. A user should
be able to select one self-contained `replay.snn.jsonl` artifact produced by the headless
recorder, then watch, pause, step, and scrub the recorded test like a movie using the
dashboard's existing topology, neuron renderer, inspector, receptive-field view, raster,
charge, and weight visualizations.

This is a playback feature, not a simulation engine. Do not port experiment logic into
JavaScript, rerun learning in the browser, infer missing scientific states, or mutate the
recorded artifact.

## Dependency and first steps

This prompt assumes the headless recorder/schema from
`prompts/Claude_Headless_Test_Replay_Recorder_Prompt.md` has been implemented and reviewed.
Do not invent an incompatible second format if it has not. If the recorder is absent,
stop after documenting the dependency instead of implementing against guesses.

Before editing, read:

- the recorder schema documentation and tests;
- `backend/simulation.py` (`topology()` and `dynamic_state()` payloads);
- `backend/serializer.py`;
- `frontend/app.js`;
- `frontend/websocket.js`;
- `frontend/controls.js`;
- `frontend/renderer.js`;
- `frontend/inspector.js`;
- `frontend/receptive.js`;
- `frontend/raster.js`;
- `frontend/charge.js`;
- `frontend/weights.js`;
- `frontend/index.html` and `frontend/style.css`;
- relevant serialization/dashboard tests.

Inspect the complete working-tree diff first. Preserve unrelated uncommitted changes.

## Scope and constraints

- Load a local replay file through the browser. Do not add server-side run storage,
  database tables, upload endpoints, directory browsing, or authentication in this task.
- The selected replay file is untrusted data: parse JSON only, never use `eval`, and render
  user-provided labels through text-safe DOM APIs rather than injecting raw HTML.
- Do not change simulation equations, scheduler behavior, topology builders, experiment
  results, or recording schema semantics.
- Do not fork or duplicate the renderer. Live and replay data must pass through one shared
  topology/dynamic application path.
- Replay must not send pattern, weight, topology, reset, reseed, or config mutations to the
  live engine.
- It is acceptable to issue the existing pause control once when entering replay so the
  hidden live simulation does not keep advancing. State this explicitly in the UI/docs.
- WebSocket messages received during replay must not overwrite the replay view.
- Leaving replay must resynchronize from the authoritative live `/api/state`; never apply
  replay weights or state back to the engine.
- Keep all chart histories and replay caches explicitly bounded where possible.

## Refactor the existing message seam first

`frontend/app.js` currently handles WebSocket `topology` and `dynamic` messages directly.
Extract small shared application functions, for example:

```text
applyTopology(topology, source)
applyDynamic(dynamic, source)
```

Both live WebSocket messages and replay frames must use these functions. Preserve current
live behavior exactly when replay mode is inactive.

The shared topology path must continue to rebuild:

- `store.meta`, weights, confidence, and pattern vectors;
- the Three.js renderer;
- charts and controls;
- receptive fields;
- raster, charge, and weights overlays.

The shared dynamic path must continue to update every existing view and inspector. Do not
create replay-only versions of these components.

## File loading and schema validation

Add a clearly labeled `Load Test` button and hidden/local file picker accepting the
recorder's replay extension/JSONL type.

On selection:

1. Parse the first nonempty line as the required header.
2. Validate schema name and supported integer version before using any data.
3. Validate the header topology has unique neuron/synapse IDs and the minimum fields the
   existing dashboard requires.
4. Parse supported marker, frame, weight-checkpoint, and result records.
5. Reject malformed JSON, non-monotonic frame indices/timesteps, unknown record types,
   invalid weight-change IDs, missing required dynamic fields, and non-finite numeric
   values with a visible, useful error.
6. Do not partially enter replay after validation fails.

Keep parsing responsive. Yield to the browser periodically or use a Web Worker if profiling
shows a realistic replay blocks the UI; do not add a worker preemptively without evidence.
Show loading progress for large files when practical.

The replay file is self-contained. Do not ask the user to select the separate manifest,
metrics CSV, or summary JSON.

## Replay-mode UI

Add a compact replay control strip containing:

- replay/run name and seed;
- topology/condition label, including the recorded hierarchical-feedback condition;
- current frame index and engine timestep;
- current experiment phase and pattern;
- play/pause;
- one-frame backward/forward buttons;
- speed selector;
- a timeline slider/scrubber;
- `Exit Replay`;
- visible parse/playback errors.

Make replay mode unmistakable in the top bar. The ordinary simulation status must say
`Replay` rather than falsely reporting that the engine is running. Disable or guard live
mutation controls while replay is active, including start/reset/reseed, input editing,
configuration application, topology editing, and manual weight editing. Restore them on
exit.

Markers should appear as lightweight labeled ticks on the timeline when feasible. At a
minimum, provide previous/next marker navigation or a marker selector for phase changes,
pattern changes, learning freeze/recall, timeouts, and mapping collisions recorded by the
experiment. Do not infer markers that are absent from the file.

## Playback semantics

- Frame order comes from recorded frame index, not wall-clock time.
- Show the recorded engine timestep exactly; sampled recordings may skip timesteps.
- Playback speed controls how quickly recorded frames are applied. It must not synthesize
  intermediate membrane states.
- Pattern/phase labels come from the record annotations/markers.
- `changed_synapses` must update the same store and renderer paths as live frames.
- Scrubbing backward must restore correct synapse weights. Reconstruct from the nearest
  preceding weight checkpoint (or the initial topology weights) plus subsequent changes;
  do not retain weights from the previously viewed future frame.
- A newly loaded topology must clear selection/history state that refers to missing IDs.
- Cancel timers/animation callbacks when paused, seeking, loading another replay, or
  exiting replay. Never allow two playback loops.

## Raster, charge, and weight history while seeking

The current raster, charge, and weights views accumulate frames sequentially. Add narrow
reset/bulk-rebuild hooks as needed so seeking backward is truthful.

When the user seeks:

1. apply the target frame to the 3D renderer, controls, inspector, and receptive-field
   view;
2. rebuild only the bounded visible/history window ending at that frame for raster and
   charge;
3. reconstruct the weights view from the appropriate checkpoint/changes and its bounded
   recent history;
4. coalesce rendering so rebuilding a history window does not redraw once per inserted
   frame.

Do not replay from boundary zero on every slider input, and do not let seeking create
unbounded chart arrays. Debounce/coalesce rapid scrubber input if needed.

If the artifact used `record_every > 1`, charts must show the recorded sampled timestamps
honestly. Do not draw skipped boundaries as measured zeros.

## Live/replay lifecycle

Entering replay:

1. request the existing live pause operation once;
2. remember that the UI is in replay mode;
3. ignore incoming live topology/dynamic messages for display purposes;
4. apply the replay header topology and first frame through the shared paths.

Exiting replay:

1. stop playback and discard replay-only caches/object URLs;
2. fetch `/api/state`;
3. apply its live topology and dynamic state through the same shared paths;
4. re-enable live controls and WebSocket display;
5. leave the live engine paused; do not silently resume it.

Handle WebSocket reconnects during replay without replacing the replay view. A reconnect
may update connection status, but its topology/dynamic payloads remain ignored until exit.

## Deliberate non-goals

Do not add:

- backend replay endpoints or a run database;
- CSV visualization in this task (CSV is for external quantitative analysis);
- replay editing or resume-from-frame;
- export/video encoding;
- interpolation between recorded frames;
- scientific Basic/continuous/noise/composition logic;
- NEST integration;
- a new frontend framework or state-management library.

## Required validation

Use a tiny deterministic replay fixture generated through the real recorder in a temporary
or test-fixture path. Do not hand-maintain a large generated replay.

Add automated tests wherever the repository's current test infrastructure supports them.
At minimum validate pure parser/reconstruction behavior. Do not introduce a large browser
test framework solely for this feature; provide a precise manual browser checklist for
rendering behavior if no JS harness exists.

Prove:

1. Supported header/frame/marker/checkpoint/result records parse successfully.
2. Malformed JSON, wrong schema version, duplicate IDs, unknown weight IDs, non-finite
   values, and non-monotonic frames are rejected before replay mode begins.
3. Random forward/backward seeks reconstruct exactly the recorded synapse weights.
4. Pattern, phase, condition, seed, timestep, and markers display from recorded data.
5. Playback, pause, step, speed, and scrub controls do not mutate the live engine.
6. Live WebSocket frames do not overwrite replay display.
7. Exit fetches and restores authoritative live topology/state and leaves it paused.
8. Existing live dashboard behavior remains unchanged outside replay mode.
9. Raster/charge/weights histories are reset and rebuilt correctly after backward seek.
10. Repeated load/exit cycles do not duplicate listeners, animation loops, renderer
    objects, or unbounded histories.

Manual acceptance checklist:

- Start the dashboard and pause it.
- Load the recorder's deterministic fixture.
- Confirm the recorded topology replaces the live view without changing the engine.
- Play, pause, step, and change speed.
- Scrub forward and backward across a weight checkpoint and pattern marker.
- Confirm neuron spikes/charge, input, column winners, hard resets, raster, charge chart,
  and weights agree with the recorded frame.
- Exit replay and confirm the live paused state is restored from `/api/state`.
- Load a malformed file and confirm the dashboard remains usable and live state intact.

Run all focused serialization/dashboard tests, the full suite, and `git diff --check`.

## Completion report

At completion, report:

- every changed file;
- the supported replay schema version;
- how live and replay share the rendering path;
- how backward seeking reconstructs weights and chart history;
- automated test commands/totals;
- manual acceptance results;
- realistic replay file size/load-time measurements;
- deliberately deferred server-side storage, compression, and richer timeline work.
