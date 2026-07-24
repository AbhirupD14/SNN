# Claude Prompt: Branch a Live Simulation From Replay Weights

## Objective

Extend the existing read-only replay player with one deliberately limited action:

```text
Continue with These Weights
```

At the currently selected replay frame, reconstruct the recorded synaptic weights, install
them into a freshly reset compatible live engine, optionally restore the displayed raw
input vector, exit replay, and leave the live engine paused so the user can step, run,
change inputs, and continue learning.

This feature is a **new branch from learned weights**, not an exact continuation of the
recorded simulation.

Do not claim to restore:

- membrane voltage;
- excitation/inhibition conductance;
- refractory or activity state;
- pending feedforward, basal, apical, reset, or relay events;
- coincidence eligibility windows;
- continuous-stimulation state;
- event logs;
- RNG position;
- the original timestep;
- experiment-script control flow.

All such transient state must be cleanly reset. The only persisted dynamical model state
restored in this task is the complete recorded weight snapshot, plus the raw input vector
when the user requests it.

## Product behavior

The intended workflow is:

1. Pause the dashboard and load a `snn.replay` version 1 JSONL file through **Load Test**.
2. Play, step, or scrub to an interesting frame.
3. Press **Continue with These Weights**.
4. The replay pauses and shows a confirmation that:
   - current live learned state will be replaced;
   - recorded transient neuron/event state will not be restored;
   - the selected replay frame and reconstruction precision are identified.
5. On confirmation, the backend validates compatibility and atomically resets/loads the
   complete weight snapshot.
6. Replay mode exits through the normal live-state resynchronization path.
7. The dashboard shows the restored weights in a paused live simulation. The user may now
   step or play.

Use the term **branch** or **continue with these weights** in documentation and errors.
Never label this action **resume exact state**.

## Scope

Keep this a small extension of the existing replay infrastructure. Reuse:

- `frontend/replay.js` for pure replay validation and canonical weight reconstruction;
- `frontend/replay_player.js` for replay transport and the selected frame;
- the shared live/replay rendering paths in `frontend/app.js`;
- the existing topology serializer and live `/api/state` resynchronization;
- stable synapse IDs and the engine’s existing internal weight-reference maps.

Do not add:

- full engine checkpoints;
- replay editing;
- a server-side run database;
- browser-side simulation;
- continuation of an experiment script;
- topology inference from neuron ID prefixes;
- exact transient-state restoration;
- a general save-file format redesign;
- a second rendering path;
- a new frontend framework.

Do not commit or push unless explicitly asked after review.

## Read before editing

Read:

- `experiments/replay_recorder.py`;
- `frontend/replay.js`;
- `frontend/replay_player.js`;
- the replay hooks and mutation guard in `frontend/app.js`;
- the replay controls in the dashboard HTML/CSS;
- `backend/api.py`;
- `backend/simulation.py`, especially `reset`, `apply_config`, `apply_topology`,
  `_live_weight`, `set_synapse_weight`, the feedforward/predictive/basal weight-reference
  maps, input methods, and topology serialization;
- `backend/serializer.py`;
- `docs/REPLAY_RECORDER.md`;
- `docs/REPLAY_PLAYER.md`;
- `tests/replay.parser.test.mjs`;
- `tests/test_replay_player_fixture.py`;
- `tests/test_replay_recorder.py`;
- API, serialization, topology-contract, and weight-editing tests.

Inspect the full working tree and preserve unrelated changes. Record the focused replay/API
baseline before editing.

## Compatibility policy

For this first version, branch only into a live engine whose computational graph and
model-affecting configuration are compatible with the replay.

Do not attempt to reverse-engineer a `NetworkSpec` from renderer-oriented serialized
topology. Do not silently select a “similar” preset.

Implement one strict, explicit compatibility contract based on metadata, never ID-prefix
heuristics. At minimum compare:

- complete neuron identity and archetype/role metadata needed by the computation;
- complete synapse identity, source, target, and kind;
- the set of weighted and mutable synapses;
- topology family/variant and cortical-column metadata where present;
- all model-affecting parameters represented in the replay topology;
- active learning-rule mode and its FE/FES parameters;
- threshold, leak, refractory, inhibition, delay, LR, and relevant topology dimensions.

It is acceptable to exclude display-only state, selected patch, patch labels, current raw
input, current timestep, running status, and recorder metadata from the compatibility
comparison. Document every exclusion.

Seed handling must be explicit. If the seed is required to reproduce any non-restored
computational property, require a seed match. If every seed-dependent learned quantity is
covered by the complete snapshot and the graph is otherwise identical, a different live
seed may be allowed, but the response and event log must record both source and live seed.
Do not silently imply that the RNG stream resumes from the replay.

When incompatible, return a precise error describing the first meaningful mismatch and
leave the live engine entirely unchanged. A useful error should tell the user which
topology/configuration to select; it must not apply that configuration automatically in
this task.

## Canonical weight snapshot

At the selected frame, use the already-tested canonical reconstruction:

```text
nearest preceding weight_checkpoint or header initial weights
    + intervening changed_synapses through the selected frame
```

Do not use the renderer’s mutable current `store.weights` as the source of truth. A
backward seek must never retain weights from a future frame.

The submitted snapshot must be complete for every synapse that the replay records with a
non-null weight. It may include fixed weighted edges such as pretrained excitation:

- mutable weights are restored;
- fixed weighted edges are compatibility-checked against the live engine but not mutated;
- unweighted structural edges must not acquire a weight.

Reject missing, extra, duplicate, unknown, nonnumeric, non-finite, or role-incompatible
entries. Bound validation must use the active model’s actual rules:

- ordinary cap-free feedforward weights: valid finite lower bound for the active learning
  mode, no invented historical upper cap;
- predictive weights: their declared valid range;
- coincidence basal weights: their declared valid range/mode;
- fixed delivered magnitudes: exact compatibility check, never conversion into plastic
  weights.

Do not silently clip a replay value. Reject an invalid snapshot.

## Precision labeling

Schema version 1 has different recorded precision:

- header/checkpoint weights use the recorder’s six-decimal snapshot precision;
- intermediate `changed_synapses` use their recorded four-decimal precision.

For the selected frame, expose:

```text
checkpoint frame: checkpoint precision
non-checkpoint frame: reconstructed from checkpoint plus recorded deltas
```

An arbitrary-frame branch is allowed for interactive exploration, but the confirmation and
backend provenance must say when it uses lower-precision intervening deltas.

Also provide a small **Use nearest preceding checkpoint** choice in the confirmation when
the selected frame is not itself a checkpoint. Default to the selected frame because that
matches the button’s wording, but state its precision honestly. Choosing the checkpoint
must update both the source frame/timestep shown to the user and the submitted canonical
snapshot.

Do not call six-decimal serialization bit-exact restoration of the former in-memory float.

## Backend design

Add one dedicated endpoint with a narrow request contract, for example:

```text
POST /api/replay/branch-weights
```

The payload should include:

```text
branch schema/version
replay schema/version
source run/experiment identifiers
selected source frame_index and recorded timestep
precision classification
recorded topology/config compatibility contract
complete reconstructed weight map
optional raw input vector
whether raw input restoration was requested
```

Do not upload or reparse the entire JSONL on the backend merely to reach the same already
validated frame. Conversely, do not trust the browser’s “compatible” assertion: the
backend must independently validate the supplied contract and snapshot against the live
engine.

Bound request structure and sizes using the expected live graph. JSON only; never evaluate
file content or treat replay fields as paths.

The endpoint must:

1. require/force the simulation runner to be paused;
2. validate branch schema and replay schema;
3. validate topology and model-configuration compatibility;
4. validate the complete weight set and every value without mutating the engine;
5. validate the optional input vector length and finite/binary values;
6. only after all validation succeeds, reset transient engine state;
7. install all mutable weights directly through one engine-level snapshot operation;
8. compatibility-check fixed weighted edges;
9. restore the raw input vector if requested, otherwise leave a blank input;
10. log one provenance event containing source run, frame, timestep, precision, source
    seed, and live seed;
11. broadcast the authoritative topology and dynamic state;
12. return a compact success response with restored counts and provenance.

The engine-level loader must be two-phase:

- **prepare/validate** the entire snapshot into resolved typed targets;
- **commit** only after preparation succeeds.

Do not implement this as hundreds of frontend `/api/weight` calls. Do not use sequential
public setters that can partially modify the engine before a later error. Do not round or
clip during commit.

If an unexpected commit error is possible after reset, preserve enough pre-action live
state to roll back safely or construct the new state off to the side before swapping it
in. At minimum, prove through tests that all expected validation failures occur before any
reset or assignment.

## Replay mutation guard

The existing replay player blocks all ordinary live mutation while replay is active. Keep
that choke point intact.

Add one explicit, tightly scoped `branchFromReplay` hook used only by the replay player’s
new button. Do not broadly exempt arbitrary API calls during replay and do not disable the
global guard.

The button must:

- exist only/be enabled while replay is active at a valid frame;
- pause replay playback before confirmation;
- be disabled while the request is in flight;
- reject double submission;
- show backend validation errors without exiting replay;
- exit replay only after confirmed backend success;
- resynchronize through `/api/state`;
- leave the live runner paused.

If the request fails, the loaded replay, selected frame, hidden live engine, and transport
controls remain usable.

## Input restoration

Offer one confirmation checkbox:

```text
Restore the displayed input pixels
```

Default it on.

Restore only the selected frame’s canonical raw input vector. Do not infer experiment
phase, held-pattern schedule, selected-patch ownership, or future stimuli from annotations.
For tiled topologies, a raw-vector restore may clear semantic patch-pattern labels if that
is the existing meaning of `set_input`; document this rather than manufacturing labels.

If input restoration is off, branch with a blank input so pressing Play cannot
unintentionally continue the hidden replay stimulus.

## UI copy

Use concise, explicit copy similar to:

```text
Continue with These Weights

This creates a fresh live simulation using the learned weights at replay frame 42
(recorded timestep 840). Membrane charge, inhibition, refractory state, pending events,
RNG position, and experiment control flow will not be restored.

The current live learned state will be replaced.

[x] Restore the displayed input pixels
[ ] Use nearest preceding checkpoint (frame 40)

Cancel    Continue
```

For a non-checkpoint frame, add:

```text
Weights after the checkpoint include four-decimal recorded deltas.
```

For a checkpoint frame, add:

```text
Weights use the recorder's six-decimal checkpoint precision.
```

Never use “exact resume.”

## Automated tests

### Pure replay/JavaScript tests

Extend the DOM-free replay tests to prove:

- branch weights come from `reconstructWeightsAt`, not current renderer state;
- forward, backward, and random seeks produce the same branch snapshot;
- selected checkpoint frames are classified correctly;
- non-checkpoint precision is classified correctly;
- nearest-preceding-checkpoint selection reconstructs the checkpoint frame, not the
  originally selected frame;
- source frame/timestep/provenance match the actual chosen snapshot;
- malformed/non-finite data cannot form a branch payload;
- the raw input vector comes from the chosen frame;
- replay objects and reconstructed maps are not mutated by payload construction.

### Engine tests

Test at least:

- complete ordinary feedforward restoration;
- coincidence basal restoration;
- predictive-weight restoration when a compatible graph contains it;
- fixed pretrained weights are checked but not mutated;
- dual-FE cap-free weights above the historical cap restore unchanged;
- active lower-floor/range violations are rejected rather than clipped;
- missing/extra/unknown/duplicate/non-finite values are rejected;
- topology edge-kind/source/target mismatches are rejected;
- model-parameter/learning-mode mismatch is rejected;
- all validation failures preserve the complete prior live state;
- successful branching clears voltage, conductance, refractory state, pending buffers,
  relay guards, event flags, continuous stimulation, and old input;
- successful branching preserves every submitted mutable weight at recorded precision;
- optional raw input restoration works and off means blank;
- source timestep is provenance only; the fresh live engine starts at its normal reset
  timestep;
- no learning occurs merely by loading the snapshot.

Use metadata and internal reference maps, not neuron-ID prefixes.

### API tests

Test:

- malformed and unsupported schemas;
- incompatible topology/config with actionable 4xx errors;
- runner is paused on success;
- topology and dynamic broadcasts occur once after success;
- success response contains source frame/timestep, precision, and restored counts;
- a failed request does not broadcast a fabricated success state;
- a repeated identical request is deterministic after reset;
- request sizes/entry counts are bounded by the compatible graph;
- arbitrary replay text cannot cause file access or code execution.

### Frontend/manual acceptance

If no DOM harness exists, provide a precise manual checklist:

1. Train a small topology, record a replay with at least two checkpoints, then alter/reset
   the live weights.
2. Load the replay and scrub to a checkpoint.
3. Branch and confirm live weights equal the checkpoint, transient state is clean, raw
   input is restored, and the engine is paused.
4. Step once and verify live simulation—not replay playback—advances.
5. Repeat at a non-checkpoint frame and verify the precision warning.
6. Repeat using nearest preceding checkpoint and verify the chosen frame changes.
7. Turn input restoration off and verify the live input is blank.
8. Attempt to branch into an incompatible topology/config and verify a visible error,
   replay remains active, and live state is untouched.
9. Submit a malformed/non-finite snapshot through an API test and verify atomic rejection.
10. Load/branch repeatedly and verify no duplicated handlers, requests, or growing timers.

## Documentation

Update `docs/REPLAY_PLAYER.md` with:

- the new branch workflow;
- the exact state restored and reset;
- compatibility rules;
- checkpoint versus delta precision;
- input restoration semantics;
- failure behavior;
- why it is not exact resume.

Update `docs/REPLAY_RECORDER.md` only where necessary to explain that version 1 replay
weights can now seed a fresh compatible live engine. Do not retroactively claim that the
recorder captures full resumable state.

## Execution and handoff

Run:

1. existing replay parser/player/recorder/API baselines;
2. new JS pure tests;
3. focused Python engine and API tests;
4. serialization/topology/golden regression tests;
5. the full test suite;
6. `git diff --check`;
7. the manual browser checklist to the extent the environment permits.

Report:

- exact changed files;
- endpoint and payload contract;
- compatibility fields and exclusions;
- restored versus reset state;
- atomicity strategy;
- checkpoint/non-checkpoint precision behavior;
- focused and full test commands/counts;
- manual acceptance results and any unexecuted browser steps;
- all deliberately deferred exact-resume work.

Do not commit or push.
