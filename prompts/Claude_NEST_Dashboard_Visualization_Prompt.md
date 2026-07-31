# Claude Prompt: NEST Dashboard Visualization and Replay Adapter

## Directive

Make the repository's existing dashboard capable of truthfully visualizing the completed
NEST/NESTML 3×3 characterization prototype.

Do this as an **adapter and observation project**, not as another simulation-engine rewrite.
The compiled NESTML models remain inside NEST, PyNEST remains the control and observation
boundary, and the existing frontend remains the renderer. Do not translate neural equations
into JavaScript or Python, and do not create a second dashboard.

Deliver the work in two gated stages:

1. **Required: offline NEST replay.** Convert a completed NEST run into the existing,
   versioned `snn.replay` JSONL contract so it can be opened with the dashboard's existing
   **Load Test** player.
2. **Conditional: live NEST viewing.** After offline replay is correct, add a separate,
   read-only NEST dashboard entry point that streams observation frames through the existing
   topology/dynamic message seam. Keep this stage out if it would require changing NEST's
   scientific execution or duplicating the frontend.

The offline replay is the acceptance-critical outcome. A correct replay-only implementation
is preferable to a misleading or scientifically intrusive live mode.

This prompt supersedes only the “do not build dashboard integration” exclusion in
`prompts/Claude_NEST_Event_Driven_3x3_Engine_Prompt.md`. All scientific scope, timing,
validity-envelope, and native-NEST-semantic conclusions from that prompt and
`docs/NEST_EVENT_DRIVEN_3X3_REPORT.md` remain authoritative.

---

## 1. Current truth that must be preserved

The NEST work is a characterization prototype, not a replacement for
`backend.simulation.SimulationEngine`.

Its supported scope is only:

```python
backend.network_spec.tiled_cc_spec(cc_e_count=8)
```

That is the canonical 3×3 tiled cortical-column network:

- 191 repository nodes;
- 1052 directed repository edges;
- a 9×9 RGC input surface;
- nine L1 cortical columns over disjoint 3×3 patches;
- one L2 cortical column;
- eight ordinary-E competitors per column;
- the existing `Eor`, `C`, and `I` roles.

The visualization must preserve and communicate the measured NEST results:

- structural parity is exact within this scope;
- event-node equations are exact only at `leak_rate == 0` with no persistent inhibitory
  conductance;
- default native NEST timing produces multiple ordinary-E winners in many
  column/presentation windows;
- the legacy strict `1010` feedback cadence does not reproduce;
- Phases 0–3 use frozen `static_synapse` weights;
- the dual FE/FES equation is expressible, but updates occur only when a later presynaptic
  spike visits the synapse, so learning is not integrated into the 3×3 runner;
- OpenMP thread results are deterministic in the measured prototype;
- MPI is not available in the installed conda NEST build.

These are observations to display, not defects for this task to repair.

Do not add global winner arbitration, epsilon-time callbacks, fake zero-delay resets,
browser-side learning, Python per-spike callbacks, inferred charge histories, or any other
mechanism that makes NEST appear equivalent to the legacy engine.

---

## 2. Read before editing

Inspect the complete working-tree status and diff first. Preserve unrelated work and do not
assume the NEST files have already been committed.

Read, in full:

- `prompts/Claude_NEST_Event_Driven_3x3_Engine_Prompt.md`;
- `docs/NEST_EVENT_DRIVEN_3X3_REPORT.md`;
- `nest_backend/README.md`;
- `nest_backend/topology.py`;
- `nest_backend/engine.py`;
- `nest_backend/recording.py`;
- `experiments/nest_3x3_suite.py`;
- `experiments/nest_parallel_check.py`;
- `experiments/replay_recorder.py`;
- `docs/REPLAY_RECORDER.md`;
- `docs/REPLAY_PLAYER.md`;
- `backend/serializer.py`;
- `backend/websocket.py`;
- `backend/api.py`;
- `frontend/app.js`;
- `frontend/replay.js`;
- `frontend/replay_player.js`;
- `frontend/renderer.js`;
- `frontend/inspector.js`;
- `frontend/receptive.js`;
- `frontend/raster.js`;
- `frontend/charge.js`;
- `frontend/weights.js`;
- `frontend/charts.js`;
- the existing replay, serialization, frontend-parser, and NEST tests.

Verify the actual schema and consumer behavior from code and tests. This prompt describes the
intended mapping, but it does not authorize an incompatible second replay format.

---

## 3. Architecture

Use the existing application seam:

```text
NEST / PyNEST
      │
      ├── completed run ──> NEST replay adapter ──> replay.snn.jsonl
      │                                            │
      │                                            └── existing Load Test player
      │
      └── optional live observer ──> topology/dynamic envelopes
                                                  │
                                                  └── existing dashboard renderers
```

The compiled-code boundary is not a reason to duplicate the UI. PyNEST already exposes model
construction, NEST node and connection identifiers, event recorders, simulation time, and
declared model state. The adapter's job is to serialize those observations into the
dashboard's existing vocabulary.

Keep these boundaries explicit:

- NEST owns simulation time, event delivery, delays, buffering, node execution, and threads.
- PyNEST constructs, stimulates, starts, and observes the NEST network.
- The adapter maps stable repository IDs and observed state into dashboard messages.
- The frontend renders those messages and never computes scientific state.

---

## 4. Stage A — required offline replay

### 4.1 Output contract

Produce a self-contained `replay.snn.jsonl` accepted by the existing schema-1 replay parser
and player. Reuse the recorder's constants, validation, finite-number rules, record ordering,
and checkpoint conventions wherever possible.

Do not weaken the existing replay parser merely to admit malformed NEST output.

Each replay must contain:

- one header with schema name/version and full static topology;
- stable repository node and edge IDs;
- initial frozen weights for every weighted edge;
- ordered frame records;
- optional semantic markers for presentations, pattern changes, feedback conditions, and
  experiment cases;
- a final result record carrying the NEST case summary;
- sufficient provenance to reproduce the run.

Record provenance including:

- NEST and NESTML versions;
- Python version;
- git commit and dirty flag;
- topology/spec hash;
- seed;
- resolution `h`;
- `L_wta`, `S`, `D`, and their ratios;
- thread count and MPI rank count;
- dispersion, feedback, and matured-`C` overrides;
- run/case name;
- learning mode, explicitly `frozen` for the current 3×3 runner;
- the exact adapter/replay schema version.

### 4.2 Static topology mapping

Map the canonical `NetworkSpec` and `NestTiledNetwork` manifest into the dashboard's existing
topology payload. Do not derive roles by parsing IDs.

Preserve:

- repository node IDs;
- repository edge IDs;
- node archetype, layer, column ID, column role, patch metadata, position, label, and
  `has_parent` where present;
- edge source, target, kind, projection, sign/role metadata, weight, and realized NEST
  delay;
- tiling metadata and input-grid dimensions;
- NEST global IDs only as optional provenance, never as frontend identity;
- the parameter/validity-envelope declaration.

The dashboard must continue to use repository IDs for selection, inspection, raster labels,
and edge lookup. NEST GIDs are process-local implementation details.

### 4.3 Time mapping

NEST uses millisecond timestamps on the configured resolution grid. The existing replay schema
requires integer frame/timestep ordering.

Use:

```text
nest_tick = round(nest_time_ms / h)
nest_time_ms = nest_tick * h
```

Validate that every serialized event lies on the configured grid within a strict numerical
tolerance. Store both:

- the integer NEST tick in the replay's required timestep field; and
- the physical `nest_time_ms` in the dynamic payload or frame metadata.

Update dashboard labels in replay mode so the user sees NEST time in milliseconds. Do not
present one NEST tick as one legacy engine boundary.

Multiple events at the same NEST timestamp belong to the same frame. Preserve every sender;
never collapse simultaneous ordinary-E winners into one winner.

### 4.4 Dynamic frame mapping

Build frames only from state that NEST actually records or exposes.

Required observations:

- input spikes, with timestamp and repository RGC ID;
- emitted spikes, with timestamp and repository node ID;
- all simultaneous winners;
- active/fired state for the current frame;
- column and role metadata needed by the raster and inspector;
- current run/case metadata;
- NEST time, tick, resolution, and presentation-window index;
- winner multiplicity and other metrics already computed by `nest_backend.recording`;
- reset, relay, and coincidence counters when they were actually sampled;
- current weights when available.

For the current frozen runner:

- header weights are authoritative;
- `changed_synapses` must be empty;
- do not fabricate learning animation.

The current final artifact has a complete spike record but not a historical sample of every
internal variable. Therefore:

- do not reconstruct past `q`, `q_pre`, eligibility, reset state, or membrane charge from
  spikes unless that reconstruction is mathematically exact and independently tested;
- use `null`, omit the field, or mark the panel unavailable when the historical value was
  not recorded;
- do not repeat a final-state value backward across earlier frames;
- do not label an event count as an instantaneous charge.

If additional historical state is necessary, add an explicit observation mechanism owned by
NEST or take serialized read-only snapshots during a separately validated chunked run. The
observer must not change event delivery, weights, random streams, or results.

### 4.5 Frame density

Do not automatically create one browser frame for every `h` tick when nothing happened.
Schema-1 replay already permits sampled timesteps.

At minimum, create frames at:

- every timestamp containing an input or emitted spike;
- presentation boundaries;
- stimulus/pattern switches;
- any explicitly recorded reset, coincidence, or weight event;
- the final observation point.

Merge all observations sharing a tick. Keep ordering deterministic by repository ID after
timestamp ordering.

If chart semantics require bounded regular sampling, make the visualization interval an
explicit integer multiple of `h`, record it in the manifest, and keep the default artifact
small enough for browser replay.

### 4.6 Dashboard behavior in NEST replay mode

Reuse the existing replay player and the shared `applyTopology` / `applyDynamic` path.

The following views should work from honest recorded data:

- topology/network renderer;
- node and edge selection;
- receptive-field/patch layout;
- spike raster;
- event log;
- input pattern display;
- per-column activity;
- simultaneous-winner visualization;
- static frozen weights;
- experiment/case metadata.

Add a small, unmistakable mode indicator:

```text
NEST REPLAY · native event timing · frozen learning
```

Show, without editorializing:

- NEST time in ms;
- resolution `h`;
- presentation window;
- thread count;
- winner multiplicity for the selected column/window when available.

Panels that require unavailable history must say **Not recorded for this NEST run** rather
than showing zero or stale state. In particular, a missing charge trace is unknown, not zero.

Disable or hide controls that would mutate a replay:

- topology editing;
- manual weight changes;
- learning-rule controls;
- reset/reseed;
- live pattern mutation;
- “continue with these weights” unless a future task defines and validates NEST-specific
  branching semantics.

Do not alter ordinary legacy replay behavior.

---

## 5. Stage B — conditional live NEST view

Attempt this stage only after Stage A passes.

### 5.1 Separate entry point

Do not replace the current dashboard's default `SimulationEngine` global. Add a dedicated
entry point, for example:

```bash
.nest-env/bin/uvicorn nest_backend.dashboard_api:app
```

It may serve the existing `frontend/` directory and the same WebSocket envelope types.
Avoid a general production backend selector in this task.

Add only the dependencies genuinely required to serve this entry point to the authoritative
NEST environment manifest. Do not merge `.nest-env` with the repository's ordinary `.venv`.

### 5.2 Single NEST owner

All PyNEST calls must be serialized through one owner. NEST simulation must never be entered
concurrently by two REST requests, the WebSocket loop, or background tasks.

If `nest.Simulate()` would block the asynchronous server loop, move the call to one dedicated
worker thread or process while retaining strict single-owner command serialization. Do not
allow overlapping simulations.

### 5.3 Observation chunks

Live viewing may advance NEST in finite observation chunks, but Python must not execute neural
events or scan the graph to emulate a scheduler.

Before accepting chunked execution:

1. preload an identical stimulus schedule;
2. run the same case once with one `Simulate(total_duration)` call;
3. run it again as repeated `Simulate(chunk_duration)` calls;
4. assert identical `(timestamp, repository sender ID)` spike multisets;
5. assert identical final node counters and connection weights;
6. repeat at 1, 2, and 4 threads where supported.

If chunking changes results, do not ship live mode. Offline replay remains the completed
feature.

The chunk size must be an integer multiple of `h`, recorded in the live session metadata, and
described as an **observation cadence**, not a scientific timestep.

### 5.4 Live controls

Keep live NEST mode deliberately small and read-only:

- **Start/Pause:** controls whether observation chunks are submitted;
- **Step:** advances exactly one declared observation chunk;
- **Speed:** changes wall-clock playback cadence only, never `h`, delays, `D`, or stimulus
  timestamps;
- **Reset:** stops execution and deterministically rebuilds the same NEST network/seed;
- **Pattern selection:** schedules a supported canonical 3×3 stimulus at a future legal
  NEST time, if this can be done without rewriting past generator schedules.

Unsupported controls must be visibly disabled with a short reason. Do not silently route a
NEST page's controls into the legacy Python engine.

Do not expose:

- topology editing;
- arbitrary graphs;
- weight mutation;
- learning toggles;
- MPI claims;
- exact checkpoint/resume;
- legacy branch-from-replay;
- arbitrary stimulation at a timestamp NEST has already passed.

### 5.5 Streaming semantics

Send the same topology and dynamic envelope shapes consumed by the current frontend. Live and
replay NEST frames must use the same adapter functions so they cannot drift.

Transmit only observations since the preceding frame:

- new input events;
- new spikes;
- newly sampled counters/state;
- changed weights, once plasticity is genuinely integrated in a future task.

Bound server and browser histories. Reconnecting clients should receive:

1. current static topology;
2. one current dynamic snapshot;
3. no unbounded replay of the entire live session.

---

## 6. Scientific and UI invariants

The following are hard requirements:

1. One NEST spike recorder event becomes one dashboard spike event.
2. Events at equal timestamps remain simultaneous.
3. Multiple winners remain multiple winners.
4. Repository IDs round-trip exactly.
5. No panel displays an unavailable value as zero.
6. No final state is presented as historical state.
7. Visualization never changes NEST weights, schedules, delays, RNG state, or event order.
8. Playback speed never changes scientific timing.
9. Existing Python-engine live mode and legacy replay files behave exactly as before.
10. NEST-specific limitations remain visible and documented.

Add a small source badge to inspected state:

```text
recorded       — directly present in a NEST recorder or saved snapshot
derived        — computed from recorded values by a documented exact formula
unavailable    — not present in this run
```

Do not use “derived” for a heuristic reconstruction.

---

## 7. Tests

### Adapter unit tests

Test:

- full 191-node / 1052-edge topology conversion;
- exact repository ID preservation and injective NEST-GID mapping;
- deterministic output for identical input artifacts;
- strict finite JSON values;
- correct `t_ms ↔ tick` conversion at `h = 0.05`, `0.1`, and `0.2`;
- same-timestamp event grouping without sender loss;
- stable deterministic ordering;
- frozen header weights and empty `changed_synapses`;
- unavailable state remains unavailable;
- malformed/unknown IDs fail loudly;
- provenance and validity-envelope fields are present.

### Cross-schema tests

- Generate a NEST replay through the real adapter.
- Parse it with the existing Python replay reader.
- Parse it with `frontend/replay.js` under the existing Node test harness.
- Assert topology counts, spike count, sender IDs, timestamps, and final result against the
  source NEST artifact.
- Add a small committed fixture generated by code, never maintained by hand.

### Scientific noninterference tests

For the same seed and schedule, compare dashboard recording disabled vs. enabled:

- identical spike multiset;
- identical winner multiplicity;
- identical final node counters;
- identical weights;
- identical NEST manifest parameters.

For conditional live mode, also compare one-shot vs. chunked execution as required in §5.3.

### Regression tests

Run:

```bash
NEST_TESTS_REQUIRED=1 .nest-env/bin/python -m pytest tests/nest/ -q
.venv/bin/python -m pytest -q
node --test tests/replay.parser.test.mjs
```

Do not reduce or skip existing tests to make the integration pass.

---

## 8. Manual acceptance

### Offline replay

1. Run one canonical NEST case and produce its ordinary NEST artifact.
2. Convert it into `replay.snn.jsonl`.
3. Start the existing dashboard in its normal Python environment.
4. Select **Load Test** and open the NEST replay.
5. Confirm the full 3×3 topology appears with 191 nodes and 1052 edges.
6. Confirm input and output spikes appear at their recorded NEST times.
7. Confirm simultaneous ordinary-E winners are all visible in the same frame/window.
8. Confirm the raster, event log, selected-column activity, and static weights agree with the
   source artifact.
9. Scrub backward and forward; confirm no future state leaks backward.
10. Confirm unavailable charge/history panels say they were not recorded.
11. Exit replay; confirm the ordinary live Python dashboard resynchronizes unchanged.

### Conditional live mode

1. Start the dedicated NEST dashboard entry point.
2. Confirm it identifies itself as **NEST LIVE**, not the legacy engine.
3. Start, pause, step, and change display speed.
4. Confirm speed changes wall-clock viewing only.
5. Confirm repeated steps advance by the declared observation chunk.
6. Confirm reset rebuilds the same seeded topology and frozen weights.
7. Confirm unsupported mutation controls are disabled.
8. Compare the finished live session artifact with the equivalent headless NEST run.

---

## 9. Documentation and report

Add concise documentation covering:

- how to generate and open a NEST replay;
- the exact command for the optional live server, if delivered;
- which dashboard panels are fully supported, partially supported, or unavailable;
- how NEST milliseconds map to replay ticks;
- why compiled NESTML code does not require a new frontend;
- why charge/history cannot be inferred from a final spike artifact;
- proof that observation does not change results;
- artifact size and browser performance;
- remaining limitations: frozen learning, native multi-winner behavior, failed legacy cadence,
  and unavailable MPI.

Update the NEST report only when the new measurements support the change. Do not copy stale
claims forward. In particular, rerun the Phase 4 reproducer before repeating any claim about
the `I_accq` continuous post port.

---

## 10. Stop conditions and non-goals

Stop and report rather than concealing the issue if:

- producing a replay requires changing scientific NEST execution;
- observation changes spikes, counters, weights, or timing;
- the existing replay schema cannot represent simultaneous NEST events without loss;
- live chunking changes results;
- required historical state is unavailable and the only proposed solution is inference;
- integration would require a second frontend or browser-side simulation.

Non-goals:

- fixing NEST WTA multiplicity;
- restoring strict `1010` cadence;
- completing plastic learning;
- enabling MPI;
- replacing the production Python engine;
- general topology support;
- exact transient checkpoint/resume;
- browser-side NEST or WebAssembly compilation;
- video export;
- a server-side replay database;
- a new replay schema unless a demonstrated incompatibility makes a versioned extension
  unavoidable.

---

## 11. Definition of done

The required work is complete when:

1. a canonical NEST run can be converted into a valid `snn.replay` artifact;
2. the existing dashboard opens it without a second renderer;
3. topology, IDs, weights, spike timestamps, simultaneous winners, and experiment metadata
   match the source artifact;
4. unavailable internal history is labeled honestly;
5. visualization is proven not to change simulation results;
6. legacy live and replay behavior remains green;
7. documentation gives one reproducible path from NEST run to dashboard view.

Live NEST streaming is an additional success only if one-shot/chunked equivalence is proven.
It is not required to declare the replay visualization complete.

The guiding principle is:

> Let NEST compute, let PyNEST observe, and let the existing dashboard render—without changing
> the science at any boundary.
