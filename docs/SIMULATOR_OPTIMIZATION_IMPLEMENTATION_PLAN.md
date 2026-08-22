# CIPP Simulator Optimization Implementation Plan

Status: proposed  
Scope: transition the current Python simulator from a dashboard-coupled reference
engine into a scalable reference engine with an optional faster scheduler/backend  
Primary constraint: preserve the existing neural and learning semantics

## 1. Decision and baseline

The custom simulator remains the semantic reference for CIPP. Its learning rule is not
ordinary edge-local STDP: when a postsynaptic neuron fires, it updates its complete
afferent vector using both participating (`+1`) and non-participating (`-1`) inputs.
That behavior must remain explicit and testable.

The August 2026 profile established the initial performance baseline:

- On `tiled_cc_feature_gated` with all nine patches active: about 200 numerical
  steps/second without dashboard serialization and 151 steps/second with it.
- In a 300-boundary profile, `BoundaryEventScheduler.next_event()` and
  `advance_all()` accounted for about 80% of numerical runtime.
- The complete afferent learning update accounted for less than 1%.
- Active `tiled_cc` scaling declined from about 1,874 steps/second at
  `cc_e_count=8` to 164 steps/second at `cc_e_count=128`.
- `SimulationEngine.step()` always constructs a full dynamic frame. The websocket
  runner discards that frame and constructs another; `/api/step` can construct the
  same state three times.

These measurements are development baselines, not portable performance claims. The
benchmark introduced in Phase 0 will replace them with reproducible artifacts that
record machine and runtime metadata.

## 2. Goals

1. Let the engine advance without constructing dashboard state.
2. Make headless experiments pay only for scientific state they actually consume.
3. Decouple simulation cadence from visualization cadence without losing weight
   changes required by the frontend.
4. Replace repeated whole-population crossing scans with a lazy invalidating priority
   queue while preserving event order, tie behavior, delays, resets, and learning.
5. Keep the current Python scan scheduler available as a correctness oracle.
6. Establish an interface behind which a compiled execution backend can later be
   added without changing `NetworkSpec`, presets, experiments, or serialized output.

## 3. Non-goals

- Do not alter neuron equations, thresholds, learning equations, distance factors,
  topology definitions, or plasticity defaults.
- Do not rewrite the simulator in C++, Rust, Numba, JAX, or another framework during
  the scheduler transition.
- Do not replace target-owned afferent vectors with conventional edge-owned STDP.
- Do not optimize the learning update before profiling shows it has become material.
- Do not remove diagnostic fields or change the websocket/REST payload contract.
- Do not treat dashboard frame rate as simulation throughput.

## 4. Required semantic invariants

Every phase must preserve the following for a fixed seed, topology, configuration, and
input schedule:

- spike identity and outer boundary;
- sub-boundary spike time and stable node-order tie resolution;
- winner and per-column winner identity;
- one-boundary feedforward delays;
- zero-latency apical delivery and relay behavior;
- hard-reset target, ordering, time, and pre-reset state;
- causal afferent participation for each learning event;
- learned weights and emitted weight-change values;
- membrane, conductance, trace, dendritic, and refractory state at every observed
  boundary;
- input state, firing frequencies, and reset/reseed determinism;
- topology and dynamic serialization contracts.

Diagnostics may be disabled only in an explicit headless instrumentation mode. Doing
so must not change any scientific state or later trajectory.

## 5. Target architecture

The public transition should end with these distinct responsibilities:

```text
NetworkSpec / presets
        |
        v
SimulationEngine
  advance()       mutate scientific state by one outer boundary; return None
  step()          compatibility wrapper: advance(), then snapshot()
  snapshot()      construct the current dynamic/dashboard representation
        |
        +---- ScanBoundaryEventScheduler (permanent reference oracle)
        |
        +---- HeapBoundaryEventScheduler (optimized Python scheduler)
        |
        +---- future compiled backend (same observable contract)

SimulationRunner
  simulation clock -> engine.advance()
  observation clock -> coalesce diagnostics, snapshot once, broadcast once
```

`dynamic_state()` may remain as a deprecated-compatible alias for `snapshot()` until
all internal callers have migrated. Existing callers of `step()` must continue to
receive the same dictionary throughout the transition.

## 6. Phased implementation

### Phase 0 — Reproducible benchmark and parity harness

Add `benchmarks/simulator_scaling.py` before changing execution code.

The benchmark must support:

- topologies: `rg_coincidence`, `tiled_cc`, `tiled_cc_l1_4`, and
  `tiled_cc_feature_gated`;
- `cc_e_count`: at least 8, 16, 32, 64, and 128 where supported;
- blank, one-patch, and all-patches-active stimulus loads;
- numerical-only and numerical-plus-snapshot modes;
- warmup boundaries followed by a fixed measured boundary count;
- separate construction, advance, snapshot, and total timings;
- JSON output containing Python, NumPy, platform, CPU, topology, node count, edge
  count, seed, parameters, activity load, repetitions, and raw timings.

Add a differential trace helper used by tests. For two engines receiving the same
schedule, compare boundary-by-boundary scientific state and diagnostics, and report
the first field and boundary that diverge. Do not rely only on one final digest.

Acceptance:

- The benchmark runs without importing the API server or frontend.
- Repeated runs produce comparable results and retain raw samples, not only an average.
- Current golden tests and the full test suite pass before optimization begins.
- A committed baseline JSON records the development machine results; performance
  thresholds are not enforced in CI.

### Phase 1 — Separate advancement from observation

Refactor `SimulationEngine` without changing execution order:

1. Add `advance() -> None` as the mutation-only public method.
2. Move the synchronous body of `step()` into a private advancement method that does
   not call `dynamic_state()`.
3. Make the event-resolved path mutate and return `None` as well.
4. Add `snapshot()` for frame construction; retain `dynamic_state()` as an alias or
   thin wrapper during migration.
5. Preserve `step() -> dict` as:

   ```python
   def step(self):
       self.advance()
       return self.snapshot()
   ```

6. Replace internal callers that discard `step()` results with `advance()`. Do this
   deliberately: callers that inspect a returned frame remain on `step()` until they
   are updated to call `snapshot()` explicitly.

Update dashboard paths so each operation creates at most one snapshot:

- `SimulationRunner._loop()` calls `advance()`, constructs one websocket message,
  and broadcasts that message.
- `/api/step` calls `advance()`, creates one raw snapshot, derives the websocket
  envelope from it, broadcasts it, and returns the same raw snapshot.
- websocket connect and `/api/state` each construct only their required snapshot.
- serializer helpers accept an already-created snapshot so they never silently call
  the engine a second time.

Acceptance:

- `step()` remains source-compatible and returns the same payload as before.
- `advance()` and `step()` produce identical scientific state after equal boundary
  counts.
- A spy/counter test proves one snapshot per runner iteration and one snapshot per
  `/api/step` request.
- All golden frame digests remain unchanged.
- The numerical-only benchmark no longer includes serialization cost.

### Phase 2 — Constant-time rolling firing frequencies

Replace repeated `sum(deque)` calls with an exact rolling count per neuron.

For every history append:

1. subtract the value that will be evicted when the deque is full;
2. append the current spike bit;
3. add the current spike bit to the rolling count.

Centralize this in one `_record_spike_history()` method used by both synchronous and
event-resolved advancement. `firing_freq(nid)` remains public-compatible and returns
`count / current_window_length`. `snapshot()` computes the per-neuron frequency map
once and reuses it for neuron records and aggregate statistics.

Acceptance:

- Rolling frequency equals the original `sum(history) / len(history)` at every
  boundary through empty, partially full, full, and evicting windows.
- Reset and topology rebuild clear both histories and counts.
- Golden dynamic frames remain unchanged.
- Snapshot benchmarks show no regression; record the measured improvement rather
  than imposing a machine-dependent CI threshold.

### Phase 3 — Explicit instrumentation levels for headless execution

Introduce a construction-time instrumentation policy with named levels rather than a
collection of unrelated booleans:

```text
full       all current dashboard and experiment diagnostics (default)
scientific spike history and state required for analysis, no UI-only edge animation
minimal    scientific state only; no per-boundary diagnostic object construction
```

Define the exact field matrix before implementing it. At minimum, classify:

- `changed_synapses`;
- `emitted` edge IDs;
- inhibitory pulse records;
- hard-reset records;
- latency-tie records;
- column-winner records;
- event-log messages;
- spike history and rolling frequencies.

The default must remain `full`. Experiments may opt into another level only after an
audit shows which fields they consume. Instrumentation guards belong at record
construction sites such as `_emit_ff_weight_changes()`, not around scientific updates.

Acceptance:

- Full mode is byte-compatible with current diagnostics.
- Paired full/minimal runs have identical scientific-state traces and learned weights.
- Minimal mode never skips learning, participation tracking, delay queues, resets,
  traces, or conductance updates.
- Each migrated experiment declares its instrumentation level near engine creation.
- Replay recording stays on a level that contains every field required by its schema.

### Phase 4 — Decouple simulation and observation cadence

After Phase 1 establishes a single-snapshot path, allow the runner to advance multiple
boundaries between broadcasts.

Keep two independent settings:

- simulation rate: target outer boundaries per second;
- observation rate: maximum dashboard frames per second.

Use monotonic deadlines so compute time does not accumulate as sleep drift. Preserve
the existing behavior that the dashboard runner advances only while running and a
client is connected unless a separately authorized server-headless mode is added.

Skipped frames cannot silently lose frontend state. Between broadcasts, coalesce:

- `changed_synapses`: latest value per synapse, in deterministic first-seen order;
- `emitted`: deterministic union or explicitly documented bounded event list for
  animation;
- pulses, hard resets, and ties: bounded records with a dropped-record count;
- current membrane/trace/refractory state: taken only from the final boundary;
- winner fields: final value plus any explicitly required window summary.

Do not place this aggregation into the learning mechanism. It is an observation-layer
buffer owned by the runner or serializer.

Acceptance:

- Simulation trajectories are identical at observation rates of 1, 10, 30, and one
  frame per boundary.
- The frontend receives the final value of every synapse changed since its prior frame.
- Diagnostic buffers are bounded under slow or disconnected clients.
- Pause, manual step, reset, reseed, topology changes, and new websocket connections
  force an immediate current snapshot.
- API and websocket payload shapes remain compatible.

### Phase 5 — Lazy heap event scheduler behind a feature flag

Do not replace the current scheduler in place. Rename it to
`ScanBoundaryEventScheduler` and retain it as the reference implementation. Add a
construction-time scheduler selector:

```text
event_scheduler = "scan" | "heap"
```

Initially default to `scan`.

The heap scheduler must implement:

- one initial crossing prediction per membrane;
- heap entries containing absolute crossing time, stable node order, cell ID, and a
  generation/version number;
- lazy invalidation of stale entries after a mutation;
- per-cell last-materialized time;
- materialization of a cell to the current event time immediately before any fire,
  apical delivery, or hard reset mutates it;
- recomputation only for mutated cells;
- one final materialization of all membranes at the boundary end;
- stable handling of exact and within-tolerance ties.

The engine must make scheduler-relevant mutations explicit. `_fire_event_cell()` and
its apical/relay helpers should notify the scheduler before and after mutating:

- the fired cell;
- every apical target whose state may change;
- every hard-reset target;
- any future cell type that changes another membrane during the event loop.

Tie handling requires special care. The current scheduler:

1. finds the minimum predicted time;
2. treats every candidate within tolerance of that minimum as tied;
3. chooses the lowest stable node order, even if that candidate's own predicted time
   is slightly later;
4. advances to the chosen candidate's predicted time;
5. records the minimum time in the tie diagnostic.

The heap implementation must reproduce that behavior exactly. Popped non-winners whose
prediction is now at or before the advanced time must be materialized and rescheduled
from the new current time rather than reusing a past heap entry.

Acceptance while `scan` remains the default:

- Unit tests cover stale entries, repeated invalidation, reset-before-crossing,
  apical-created same-time crossings, exact ties, tolerance ties, no-event boundaries,
  and end-of-boundary materialization.
- Differential tests run scan and heap engines across every event-resolved built-in
  topology, multiple seeds, active/blank schedules, and adversarial synthetic graphs.
- There is no divergence in the invariants from Section 4. Comparisons should use both
  exact serialized golden digests and high-precision internal state/tau checks.
- Heap mode provides a meaningful improvement on the heavily active feature-gated
  benchmark. A provisional engineering target is at least 2x numerical throughput at
  the profiled scale with no more than a 10% regression on small/quiet graphs; record
  results even if the target is missed.

### Phase 6 — Promote heap scheduler with rollback retained

Promote `heap` to the event-resolved default only after Phase 5 parity and performance
results are reviewed.

For at least one release/development cycle:

- retain `scan` as an explicit option;
- run a reduced scan-versus-heap differential matrix in CI;
- run the complete matrix before merging scheduler changes;
- document how to reproduce the first divergence;
- do not regenerate golden files merely to accommodate scheduler differences.

If a semantic mismatch appears, switch the default back to `scan`; do not patch the
expected outputs unless the scientific semantics were intentionally changed in a
separate, reviewed change.

### Phase 7 — Backend boundary and compiled-kernel feasibility

Only after the Python heap scheduler is stable, define an execution-backend protocol.
The protocol should cover:

- construction from validated `NetworkSpec` and engine parameters;
- input and delayed-event delivery;
- advancing one or many outer boundaries;
- scientific-state checkpoint/export;
- diagnostic extraction at declared instrumentation levels;
- deterministic seed and tie behavior;
- snapshot data required by the existing serializer.

Use the benchmark to decide between Numba, C++, and Rust. Prototype only the measured
hot state: membrane arrays, crossing prediction, heap/version arrays, and finalization.
Keep topology construction, API control, serialization, and the semantic oracle in
Python. A compiled backend is accepted only by differential parity against the Python
scan and heap engines.

## 7. Test strategy

### Required focused tests

- `tests/test_advancement_observation.py`: `advance()`/`step()` parity, snapshot call
  counts, no mutation during snapshot, and compatibility return values.
- `tests/test_firing_frequency.py`: rolling-window exactness and reset behavior.
- `tests/test_instrumentation_modes.py`: scientific parity across recording levels.
- `tests/test_event_scheduler.py`: retain scan tests and add heap unit/adversarial cases.
- `tests/test_scheduler_differential.py`: boundary-by-boundary scan/heap parity.
- serializer/API/websocket tests: one-snapshot behavior and unchanged envelopes.
- replay tests: aggregation and instrumentation do not corrupt recorded frames.

### Regression matrix

Run at minimum:

```text
topologies:
  rg_coincidence
  tiled_cc
  tiled_cc_l1_4
  tiled_cc_feature_gated
  rg_direct_cc4

schedulers:
  scan
  heap (event-resolved graphs)

seeds:
  1, 2, 7, 19

loads:
  blank
  one active patch/pattern
  all tiled patches active
  pattern transitions
```

Every phase runs the full `pytest` suite. Scheduler phases additionally verify the
committed golden topology traces and run the differential matrix.

## 8. Performance reporting

For every optimization phase, report:

- commit and dirty-tree state;
- machine/runtime metadata;
- node and edge counts;
- stimulus/activity load;
- construction time;
- numerical boundaries/second;
- snapshot milliseconds/frame;
- end-to-end dashboard boundaries/second and frames/second;
- peak diagnostic-buffer size where applicable;
- raw repetition values plus median and dispersion;
- semantic test and golden results.

Never compare a blank-input run to an active run or a numerical-only run to a
serialization-inclusive run without labeling the difference.

## 9. Commit and review sequence

Keep the transition bisectable. Recommended commits:

1. benchmark and differential harness;
2. `advance()`/`snapshot()` split with compatibility wrapper;
3. runner/API single-snapshot migration;
4. rolling firing-frequency counts;
5. instrumentation policy and experiment migrations;
6. observation cadence and diagnostic coalescing;
7. heap scheduler implementation behind `scan` default;
8. heap parity/adversarial tests and benchmark report;
9. heap default promotion, if accepted;
10. compiled-backend interface proposal, without implementation.

Do not combine the observation split and scheduler rewrite in one commit. Each commit
must leave the full suite passing and make any performance change independently
measurable.

## 10. Definition of done for the transition

The Python transition is complete when:

- headless advancement constructs no dashboard snapshot;
- dashboard and manual-step paths construct at most one snapshot per emitted frame;
- firing-frequency reporting is constant-time per neuron;
- headless instrumentation avoids UI-only record construction without changing
  scientific state;
- observation cadence is independent and preserves required frontend deltas;
- heap scheduling is the validated default for event-resolved graphs;
- scan scheduling remains available as a differential oracle;
- all golden, focused, full-suite, replay, API, and scheduler differential tests pass;
- benchmark artifacts demonstrate the before/after scaling curve;
- no scientific parameter or learning behavior changed as part of the optimization.

