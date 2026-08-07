# Claude implementation prompt: record four presentation-ready simulation videos

## Mission

Create a deterministic, reproducible Playwright/Chromium recording workflow for the
locally hosted SNN dashboard at:

```text
http://localhost:8000
```

Produce **four separate presentation-ready videos**, not one combined montage:

1. frequency as an observable uncertainty signal;
2. continuous learning across sequential patterns;
3. resistance to training-exposure bias after a very long dwell;
4. scaling from the 9×9 tiled hierarchy to the 9×18 two-tower hierarchy.

The first three demonstrations use the repository's small 3×3 topology or topologies. The
fourth video contains two scaling chapters in one recording: 9×9 first, then the dual-tower
topology.

Implement and run the complete workflow. Inspect the resulting videos and thumbnails before
reporting completion.

## Repository and evidence you must inspect first

Do not begin by guessing UI selectors or scientific settings. Read the relevant code and
documentation first:

- `README.md`, especially **Start here**;
- `backend/api.py` and `backend/websocket.py` for deterministic REST controls and readiness;
- `backend/dashboard_config.py` for exposed settings and topology names;
- `backend/simulation.py` for reset/seed/pattern behavior;
- `frontend/index.html`, `frontend/app.js`, `frontend/controls.js`;
- `frontend/raster.js`, `frontend/charge.js`, `frontend/receptive.js`, and
  `frontend/editor.js`;
- `experiments/dual_fe_cc4_consolidation.py` and
  `tests/test_dual_fe_cc4_experiment.py` for the four-pattern and long-dwell protocols;
- `docs/SUMMER_2026_CIPP_ACCOMPLISHMENTS.md` for the defensible uncertainty language;
- `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md` and
  `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md` for the frequency limitation;
- `docs/TWO_TOWER_COMPOSITION.md` for the measured scaling result and its L3 bottleneck.

Useful repository facts that must still be verified against the current checkout:

- Ordinary dashboard server:

  ```bash
  .venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8000
  ```

- Core REST controls include:
  - `POST /api/start`
  - `POST /api/pause`
  - `POST /api/step`
  - `POST /api/reset`
  - `POST /api/speed/{steps_per_second}`
  - `POST /api/config` with `{"overrides": {...}}`
  - `POST /api/pattern` with `{"name": "..."}`
  - `POST /api/patch_pattern`
  - `POST /api/patch_patterns/clear`
  - `GET /api/state`
  - `GET /api/config`
  - `GET /api/topology`
- The browser reports WebSocket readiness through `#st-conn-label` with text `connected`.
- Live execution controls include `#g-start`, `#g-pause`, `#g-step`, `#g-reset`, and the
  mirrored overlay transports.
- Visualization entry points include the Spike Raster, Charge / time, Receptive Fields,
  and Topology Editor.
- `rg_coincidence` is the small 3×3 coincidence/feedback circuit.
- `rg_direct_cc4` is the direct 3×3 four-competitor column used by the measured four-pattern
  dual FE/FES consolidation experiment.
- `tiled_cc` is the 9×9, 191-node/1052-edge hierarchy.
- `two_tower_composition` is the 9×18, 393-node/2162-edge hierarchy.

Use REST calls for simulation mutations whenever possible; they are more deterministic than
pixel-coordinate mouse clicks. Use DOM controls when the visible interaction itself matters,
such as opening or closing a visualization, selecting a displayed column, or entering the
Topology Editor.

## Scientific claim boundaries

Presentation polish must never overstate the evidence.

### Frequency / uncertainty

The first video may present frequency/cadence as an **observable candidate uncertainty or
prediction-confirmation signal**. It must not call it calibrated confidence, a probability,
or a validated general uncertainty measure. The repository explicitly records that cadence
also depends on causal-loop latency and input pacing.

Use defensible language such as:

> The circuit exposes a frequency/cadence signal associated with confirmed bottom-up and
> top-down agreement. It is an observable substrate for uncertainty, not yet a calibrated
> confidence score.

The spike raster contains a firing-rate bar in its pinned left gutter. The charge-over-time
view displays membrane charge, threshold crossings, spikes, and inhibition. Use whichever
view actually makes the frequency transition clearest; do not claim the charge bars
themselves are a frequency estimator.

### Continuous learning

Continuous learning means that the **same live network and learned state** continue across
pattern changes. Do not reset, reseed, rebuild, reload a preset, or reapply model
configuration between pattern phases. Show several patterns sequentially and make the
owner transition visible.

### Exposure-bias resistance

The long-dwell video must demonstrate the actual measured protocol, not merely display a
new pattern after waiting. Use the dual FE/FES `rg_direct_cc4` candidate supported by
`experiments/dual_fe_cc4_consolidation.py` and its tests—currently the seed-1 candidate is
`B=5`, learning-rate multiplier `m=100`, corresponding to the experiment's explicit eta
and C-eta values. Verify those values before using them.

Train the four-pattern network, hold one selected pattern for a much longer dwell, then
switch to a different pattern without resetting. Record:

- the incumbent owner before the switch;
- the first owner after the switch;
- how many eligible presentations/boundaries consolidation actually takes;
- whether the new owner differs from the long-held incumbent;
- whether the observed behavior matches the existing headless long-dwell result.

Do not use the word “instant” unless the captured run proves consolidation on the first
eligible presentation under a declared criterion. Otherwise describe the measured latency
precisely, for example “immediate turnover in N eligible presentations” or “rapid
reconsolidation after a 6000-boundary dwell.” Do not change learning rates or neural rules
only to make the transition look faster.

### Scaling

The fourth video demonstrates structural and operational scaling:

1. `tiled_cc`: the 9×9 input sheet and nine L1 columns feeding L2;
2. `two_tower_composition`: the 9×18 sheet, eighteen L1 columns, two L2 columns, and one L3
   composition column.

Show the actual node/edge counts from the live topology. It is acceptable—and preferred—to
show the Topology Editor or a carefully framed main 3D view so the scale difference is
obvious.

Do not claim that the two-tower L3 representation distinguishes V/A/7. The repository's
measured result is that both lower towers learn and cold-recall internal codes, while the
two scalar Eor streams make the tested L3 inputs non-identifiable. This video proves graph
and lower-encoder scaling, not that the documented L3 bottleneck is solved.

## The four required demonstrations

### Video 1 — frequency / candidate uncertainty signal

Use the scientifically appropriate small 3×3 coincidence/feedback topology, expected to be
`rg_coincidence` unless inspection establishes a better existing preset.

The visual sequence must include:

1. a clean title card or unobtrusive in-page chapter label;
2. the reset 3D network before learning;
3. a short pause;
4. learning under a sustained named pattern;
5. a visible transition into the Spike Raster and/or Charge over time view;
6. enough history to show the early versus later firing cadence clearly;
7. an on-screen, evidence-based label for the observed frequency/cadence;
8. pause the simulation so the final history remains still;
9. close the history view and return to the network;
10. open the Topology Editor as requested to show the learned circuit's structure, but do
    not imply that the editor displays live learned weights if it only displays the
    `NetworkSpec`;
11. if learned receptive-field weights are needed, briefly use the Receptive Fields view,
    which is the appropriate weight visualization;
12. a short ending pause.

### Video 2 — continuous sequential learning

Use the existing small 3×3 topology that supports the measured four-pattern one-to-one
learning result, expected to be `rg_direct_cc4` with dual FE/FES enabled.

The visual sequence must:

1. reset once at the beginning;
2. show the same live network learning `row 1`, `col 1`, `diag \\`, and `diag /` one at a
   time;
3. never reset/reseed/reconfigure between those pattern phases;
4. visibly label each phase and the current/modal owner;
5. show that new patterns continue recruiting/consolidating owners while earlier learned
   weights remain in the same network;
6. use the 3D view plus Raster, Weights over time, or Receptive Fields where each makes the
   transition clearest;
7. end on a paused summary view showing the four pattern-to-owner assignments actually
   observed in the run.

Do not hard-code expected owner IDs unless they are first measured for the fixed seed. Read
them from live state or from a deterministic preflight run.

### Video 3 — long-dwell exposure-bias stress

Use the same scientifically supported 3×3 direct four-competitor configuration as Video 2.

The visual sequence must:

1. reset to the same declared fixed seed;
2. train/establish the four-pattern network under the existing protocol;
3. select one pattern and hold it for a much longer dwell (prefer the existing 6000-boundary
   stress protocol unless runtime evidence justifies another already-tested value);
4. visually compress or accelerate the uninformative middle of the long dwell rather than
   making the audience watch thousands of identical frames;
5. still show enough beginning/end evidence that the long dwell genuinely occurred;
6. switch directly to a different pattern without reset or rebuild;
7. show the incumbent and the new consolidated owner;
8. display the measured turnover latency and consolidation criterion;
9. end on a paused weight/receptive-field summary that makes the different neuron assignment
   visible.

Input sources must remain equal-frequency. Do not introduce class reweighting, balanced
sampling code, or any hidden pattern-specific learning-rate adjustment.

### Video 4 — scaling: 9×9 then two towers

This is one independent scaling video with two chapters, not a compilation of the first
three videos.

Chapter A:

1. load/reset `tiled_cc` deterministically;
2. show the 9×9 input sheet, its 3×3 patch tiling, and multiple independently driven patch
   patterns;
3. show the live network and then the Topology Editor or another full-graph view;
4. display the verified 191-node/1052-edge count if still current.

Chapter B:

1. load/reset `two_tower_composition` deterministically;
2. show the 9×18 input surface and whole-sheet glyph controls if available;
3. run a visually informative V/A/7 or repository-supported stimulus sequence;
4. show the full topology with eighteen L1 columns, two L2 columns, and one L3 column;
5. display the verified 393-node/2162-edge count if still current;
6. close with the truthful scope statement: lower encoders scale; L3 identity separation is
   not claimed by this demo.

## Deterministic orchestration requirements

Build a script-driven workflow, not a hand-recorded desktop session.

1. Use Playwright with Chromium.
2. Use a browser viewport and recorded video size of exactly 1920×1080.
3. Prefer headless Chromium so browser chrome is absent.
4. Hide the mouse cursor and any irrelevant development/debug/status clutter with
   recording-only CSS injected by Playwright. Do not edit simulation behavior.
5. Preserve scientifically relevant labels, legends, topology counts, owner IDs, chart
   axes, and transport state.
6. Start browser video recording before meaningful simulation activity.
7. Include a two-to-three-second visual pause at the start and end of each video.
8. Use deterministic REST calls for reset, configuration, patterns, speed, start, and pause.
9. Use a fixed seed—prefer seed `1`, matching the measured experiments—without calling
   `/api/reseed`.
10. Inspect how `.claude/dashboard_seed.txt` is used. If the recording script needs to set
    it before starting its own server, back up its exact prior contents and restore them in
    a `finally` block. Never destroy user runtime state.
11. The recording command should start and stop its own Uvicorn process when practical. If
    port 8000 is already occupied, either verify that it is the correct dashboard and use
    it without killing it, or fail with a clear message. Never terminate a server process
    the workflow did not start.
12. Wait for all of the following before recording actions:
    - `GET /api/state` returns HTTP 200;
    - `#st-conn-label` reads `connected`;
    - the expected topology has arrived;
    - the scene canvas and topology-sized input grid exist;
    - dynamic state has rendered at least once.
13. Wait on explicit state predicates—timestep, running flag, selected topology, owner,
    frequency, weights, or topology count—not arbitrary sleeps, except for deliberate
    presentation pauses.
14. Run a deterministic headless/preflight measurement before the final recording so the
    script knows the actual owners and phase lengths it will label.
15. Do not alter `backend/simulation.py`, `snn/neurons.py`, learning equations, thresholds,
    topology builders, or scheduler semantics merely to make the recording pass.

Temporary title cards or phase labels may be injected into the page by Playwright if they
are clearly presentation annotations and do not affect the simulation. Keep them restrained
and visually consistent with the existing dashboard.

## Recording implementation

Prefer a repository-local structure such as:

```text
scripts/record_simulation_demos.mjs
package.json
presentation_assets/
```

You may choose a small supporting module if it makes server lifecycle, API calls, video
finalization, or FFprobe validation clearer.

Install missing development dependencies locally. Prefer:

```bash
npm install --save-dev playwright
npx playwright install chromium
```

Use an existing system `ffmpeg`/`ffprobe` if available. If they are unavailable and cannot
be installed without system-wide changes, use a repository-local package such as
`ffmpeg-static` plus an appropriate local FFprobe package. Record the actual binaries used.

Add this convenient command:

```bash
npm run record-demo
```

It must create all four videos in one deterministic invocation. A useful optional interface
is `npm run record-demo -- --only uncertainty`, but the default command must produce the
complete four-video set.

Playwright recording guidance:

- create a fresh browser context per video so each WebM is genuinely independent;
- configure both `viewport: { width: 1920, height: 1080 }` and
  `recordVideo.size: { width: 1920, height: 1080 }`;
- close each page/context cleanly before moving its temporary WebM into place;
- use meaningful, stable selectors rather than screen coordinates;
- use the REST API for scientific state transitions and UI clicks only for visible view
  changes;
- preserve raw Playwright WebM recordings before FFmpeg conversion.

## Required outputs

There must be exactly four final demonstrations—no combined fifth montage. Produce a WebM,
MP4, and representative thumbnail for each:

```text
presentation_assets/uncertainty_frequency_demo.webm
presentation_assets/uncertainty_frequency_demo.mp4
presentation_assets/uncertainty_frequency_demo_thumbnail.png

presentation_assets/continuous_learning_demo.webm
presentation_assets/continuous_learning_demo.mp4
presentation_assets/continuous_learning_demo_thumbnail.png

presentation_assets/unbiased_learning_demo.webm
presentation_assets/unbiased_learning_demo.mp4
presentation_assets/unbiased_learning_demo_thumbnail.png

presentation_assets/scaling_demo.webm
presentation_assets/scaling_demo.mp4
presentation_assets/scaling_demo_thumbnail.png

presentation_assets/recording_notes.md
```

The earlier singular `simulation_demo.*` naming is superseded by this four-video
requirement. Do not create one combined `simulation_demo.mp4`.

For every MP4, use FFmpeg to produce:

- H.264 video;
- `yuv420p` pixel format;
- constant 30 FPS output;
- 1920×1080 resolution;
- `+faststart` for presentation playback compatibility.

A representative conversion shape is:

```bash
ffmpeg -y -i INPUT.webm -vf "fps=30" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart OUTPUT.mp4
```

Choose a thumbnail timestamp that actually represents the central claim of each video, not
the initial title or a blank transition.

## Verification gates

Do not declare success merely because files exist.

For each of the four videos:

1. confirm the WebM is nonempty and playable;
2. run `ffprobe` on the MP4 and assert:
   - codec is H.264;
   - pixel format is `yuv420p`;
   - width is 1920;
   - height is 1080;
   - frame rate is 30 FPS;
   - duration is positive and within the expected chapter plan;
3. decode at least one frame with FFmpeg to prove the MP4 opens;
4. verify the thumbnail is a valid 1920×1080 PNG unless a justified crop is documented;
5. inspect at least the beginning, representative middle, and end frames for blank pages,
   browser error screens, confirmation dialogs, clipped overlays, cursor artifacts, or
   misleading labels;
6. verify that the scientific state predicates recorded during the run match what the video
   claims;
7. verify the four final MP4 files are independent and no combined montage was produced.

Run focused tests for any recording-support code you add. Also run the existing behavioral
tests that pin the demonstrated protocols, at minimum:

```bash
.venv/bin/python -m pytest -q \
  tests/test_dual_fe_cc4_experiment.py \
  tests/test_coincidence_experiment.py \
  tests/test_two_tower_composition.py \
  tests/test_tiled_cc_dashboard_contract.py
```

If current test names or applicability differ, use the closest relevant suites and explain
the substitution.

## `recording_notes.md`

The notes must be sufficient for another developer to reproduce and interpret the result.
Include:

- exact setup and recording commands;
- the exact `npm run record-demo` command;
- server command and readiness checks;
- Playwright, Chromium, FFmpeg, and FFprobe versions;
- fixed seed;
- per-video topology and full simulation configuration;
- per-video pattern order and dwell lengths;
- measured owner IDs and transition/consolidation latencies;
- per-video duration, resolution, codec, pixel format, and frame rate;
- a concise shot-by-shot description of what the viewer sees;
- the uncertainty/cadence qualification;
- the two-tower L3 qualification;
- whether the long-dwell switch was literally first-presentation instantaneous or the exact
  measured number of presentations it took;
- any limitations, nondeterminism, rendering compromises, or dependencies encountered;
- a list of every generated file and its size.

## Operational constraints

- Preserve all unrelated and pre-existing worktree changes.
- Do not commit, push, publish, or upload the videos.
- Do not delete material data or overwrite existing presentation assets without first
  preserving them safely.
- Do not modify neural behavior, scientific parameters outside the declared existing
  configurations, or topology definitions to manufacture a prettier result.
- Do not replace evidence with hard-coded captions. Derive owner IDs, counts, timing, and
  frequency measurements from the actual deterministic run.
- Do not silently skip a failed video. If one scientific story does not reproduce, retain
  diagnostic artifacts, report the precise mismatch, and stop short of the unsupported
  claim.

## Final report

Return a concise implementation report containing:

1. files added or changed;
2. exact commands run;
3. the four output paths;
4. FFprobe results for all four MP4 files;
5. the fixed seed and per-video configurations;
6. the observed owner/frequency/latency evidence supporting each video;
7. test results;
8. all scientific qualifications and remaining limitations.

Do not report completion until the workflow has been run end to end and all four videos and
four thumbnails have been visually inspected.
