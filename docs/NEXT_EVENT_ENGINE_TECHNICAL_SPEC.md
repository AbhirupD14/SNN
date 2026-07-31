# Custom Next-Event Engine Technical Specification

**Status:** SUPERSEDED on the NEST branches; not implemented and not to be implemented
there. `prompts/Claude_NEST_Event_Driven_3x3_Engine_Prompt.md` is the current engine
rework direction: NEST/NESTML owns the clock, delivery, delays, recording, threads and
MPI, so the custom Python next-event scheduler described below is explicitly out of
scope. This document is retained as the record of the analysis that motivated the rework
(especially §1 on why boundary rescanning does not scale) and as the semantic reference
for what the current engine does. Do not implement both.
**Scope:** replace the boundary-scanning execution kernel without introducing a new
topology or changing the current neural and learning contracts
**Reference implementation:** `backend/simulation.py`, `snn/neurons.py`,
`backend/network_spec.py`
**Reference semantics:** `Current_Implementation_Methodology_Equations.md`
**Validation baseline:** `docs/ENGINE_VALIDATION_REPORT.md`

---

## 1. Purpose

The current event-resolved engine is a deterministic hybrid simulator:

- ordinary delivery, feedback resets, stimulus pacing, eligibility expiry, conductance
  decay, refractory decay, and learning exposure are indexed by an integer outer boundary;
- membrane crossings inside a boundary are solved analytically at
  `tau ∈ [0, 1]`;
- the scheduler finds each crossing by rescanning every membrane;
- apical permission and immediate WTA reset are processed at the crossing's `tau`;
- delay-one outputs are accumulated into boundary buffers.

This is scientifically inspectable and validated for its declared model, but it scales
poorly and cannot support a meaningful physical-time refinement test. A larger graph pays
for every membrane on every event-resolution pass, even when almost every membrane is
unchanged.

The next-event engine shall instead:

1. assign every stimulus, delivery, crossing, spike, reset, expiry, and observation an
   absolute timestamp;
2. advance simulation time directly to the earliest pending timestamp;
3. update only cells and connections affected by events at that timestamp;
4. predict threshold crossings per cell and invalidate stale predictions lazily;
5. process equal timestamps as explicit concurrent batches;
6. preserve event identity and multiplicity throughout the causal chain;
7. make delay, duration, expiry, and decay physical-time properties rather than accidental
   consequences of an outer loop;
8. retain the current dashboard and experiment interfaces through an adapter;
9. provide a strict compatibility profile before any continuous-time model change.

This document defines the required scientific semantics. Queue layout, heap library, object
packing, and language are implementation choices only where this document says they are.

---

## 2. Non-goals

The first implementation shall not:

- add, remove, or redesign a topology;
- introduce feature gates, `k`-WTA, lateral inhibition, or a new composition mechanism;
- change the ordinary-E, `Eor`, or C learning equations;
- change thresholds, structural weight ceilings, initialization, or plasticity rates;
- reinterpret basal versus apical connections;
- change one-column WTA from single-winner to multi-winner;
- make the two-tower L3 representation preserve information its Eor bottleneck discards;
- claim biological wall-clock units before the parameters are calibrated to them;
- target NEST, GeNN, Loihi, SpiNNaker, or another external runtime;
- move neural computation into the frontend;
- delete the validated hybrid engine before equivalence gates pass;
- call a tick-routed implementation “pure next-event” merely because spikes are events.

The scheduler refactor and any later neuron-model refinement are separate experiments.

---

## 3. Authority and compatibility profiles

### 3.1 Authority order

Until the new engine is promoted:

1. the current production code defines existing behavior;
2. `Current_Implementation_Methodology_Equations.md` describes that behavior;
3. `docs/ENGINE_VALIDATION_REPORT.md` defines its validated claim boundary;
4. this document defines the replacement contract.

If implementation work exposes a contradiction, stop at the affected phase and add a
minimal reproducer. Do not silently select the outcome that makes the new scheduler easier.

### 3.2 Required profiles

The engine shall expose two named timing profiles.

#### `boundary_compatible`

This profile is required for migration and is implemented first.

- One legacy boundary has duration `D = 1` abstract time unit.
- Compatibility windows are identified by `floor(timestamp / D)`, but the scheduler does
  not iterate empty windows. A cell touched after several empty windows lazily applies the
  exact skipped-window recurrence.
- Existing delay-one edges have `delay = D`.
- The current frozen ordinary-E drive packet is represented explicitly as a rectangular
  drive interval with duration `D`, not converted into an impulse.
- A firing or hard reset consumes the cell's active compatible drive packet exactly as the
  current engine consumes `remaining_excitation`.
- Existing per-boundary retention factors remain discrete compatibility-window settlement
  operations. They are applied lazily or by an already-required endpoint event, never by a
  global scan.
- A basal event is usable in its arrival window and may carry into exactly one following
  window, matching the current two-window opportunity.
- Existing refractory steps map to integer multiples of `D`.
- Current presentation pacing maps to timestamped stimulus events separated by
  `resolved_input_period × D`.
- Current delay-one C feedback reset arrives at `spike_time + D`.
- Learning sees the same causal delivery group and pre-reset charge used today.
- Existing once-per-boundary membrane, C-deposit, and relay guards become explicit
  once-per-compatibility-window guards.

This profile may preserve current scientific approximations. Its purpose is to prove that
the new data structures do not change the model.

#### `continuous_physical`

This profile is a later, separately approved experiment.

- Delays, pulse shapes, decay constants, eligibility lifetimes, and refractory intervals
  are explicit physical-time parameters.
- Ordinary inputs may use impulses or declared synaptic kernels instead of the compatibility
  rectangle.
- No result from this profile may replace a compatibility result without a controlled
  comparison.
- The dashboard must label the active timing profile.

The first production promotion covers `boundary_compatible` only.

---

## 4. Time representation

### 4.1 Fixed-point timestamps

Queue keys shall not be raw floating-point values.

```text
Timestamp = signed 64-bit integer count of time_quantum
```

`time_quantum` is timestamp precision, not a simulation timestep. The engine does not visit
every quantum. It jumps from one occupied timestamp to the next.

For compatibility runs:

```text
boundary_duration_ticks = round(D / time_quantum)
```

All configured delays and lifetimes must resolve to positive integer ticks unless they are
explicitly zero-delay. Construction fails if a requested duration cannot be represented
within the declared tolerance.

### 4.2 Analytic crossing conversion

A cell may solve a crossing in floating-point elapsed time locally. The predicted absolute
crossing is converted to ticks with causal ceiling:

```text
crossing_tick = current_tick + ceil(delta_time / time_quantum - numeric_slack)
```

The conversion must never schedule a crossing earlier than the analytic solution. The event
stores both:

- `timestamp`: authoritative fixed-point queue key;
- `analytic_time`: diagnostic double-precision prediction before quantization.

Acceptance tests sweep `time_quantum` and require convergence. Compatibility defaults must
reproduce current crossing order and traces within the existing timing tolerance.

### 4.3 Monotonicity

Simulation time never decreases. No handler may enqueue an event earlier than the current
timestamp. A zero-delay child event is enqueued at the current timestamp in a later causal
generation.

Timestamp overflow, negative duration, or an event scheduled into the past is a fatal
simulation error with the originating event and edge included in the message.

### 4.4 Compatibility window identity

For the migration profile only:

```text
window_id(t) = floor(t / D)
```

The id exists to reproduce legacy guards, settlement, and causal participation. It does not
cause an update loop. When nothing is scheduled for a window, no scheduler work occurs.

At an exact multiple of `D`, an endpoint crossing predicted from the preceding window is
owned by that preceding window. New arrivals at the same absolute timestamp belong to the
new window. Section 9.5 defines the required phase order.

---

## 5. Event identity and causal ledger

Every event has an immutable envelope:

```text
Event {
    event_id: uint64
    timestamp: Timestamp
    timestamp_phase: uint16
    generation: uint32
    kind: EventKind
    source_id: StableNodeId | ControllerId
    target_id: StableNodeId | null
    edge_id: StableEdgeId | null
    parent_event_id: uint64 | null
    root_event_id: uint64
    presentation_id: uint64 | null
    target_state_version: uint64 | null
    payload: typed payload
}
```

Requirements:

- `event_id` is unique and monotonically allocated within one run.
- `root_event_id` identifies the originating stimulus or controller action.
- `parent_event_id` forms an inspectable causal tree.
- `presentation_id` identifies the externally presented evidence packet, including all
  descendants.
- `timestamp_phase` is normally zero; the compatibility profile uses its declared endpoint
  phases to preserve old-window endpoint ownership and new-window reset ordering.
- an edge emission creates a distinct event even if another edge reaches the same target at
  the same timestamp with the same numeric value;
- aggregation is a target-model operation over a multiset, never queue-level identity loss;
- processed, coalesced, rejected, invalidated, cancelled, and expired events remain
  distinguishable in the ledger.

No diagnostic may infer causality later from an edge-id list when event identity was
available at runtime.

---

## 6. Event kinds

The minimum event vocabulary is:

| Kind | Meaning |
|---|---|
| `stimulus_start` | controller presents evidence and creates a `presentation_id` |
| `stimulus_end` | optional end of a held external stimulus |
| `source_spike` | an RGC/source emits |
| `drive_start` | a rectangular compatible current begins |
| `drive_end` | that current contribution ends |
| `charge_impulse` | instantaneous somatic charge |
| `conductance_impulse` | instantaneous change in inhibitory conductance |
| `basal_arrival` | one C basal event with causal-source identity |
| `apical_arrival` | one C apical permission event |
| `eligibility_expire` | one basal eligibility token expires |
| `crossing_prediction` | versioned prediction that a membrane reaches threshold |
| `spike_commit` | a validated neuron spike |
| `relay_input` | one causal input event reaches an inhibitory relay |
| `relay_spike` | relay model emits after resolving its input multiset |
| `hard_reset` | target membrane and active compatible drive are reset |
| `refractory_end` | refractory state becomes inactive |
| `observation` | read-only sampling/replay/dashboard event |
| `controller_stop` | run-until condition reached |

Learning is a transaction attached to `spike_commit`, not a free-standing event that can be
replayed twice. A diagnostic `learning_commit` record may be emitted to the ledger.

---

## 7. Stable topology identity

Scientific behavior must not depend on list insertion order, Python dictionary order, or a
neuron id's lexical spelling.

At graph construction:

- every node receives a stable numeric `node_ordinal`;
- every edge receives a stable numeric `edge_ordinal`;
- every competition domain receives a stable `domain_id`;
- ordinals are derived from canonical validated topology serialization;
- applying the same topology produces the same ordinals;
- reordering equivalent input JSON must not change ordinals or results.

String ids remain public labels and serialization keys, not scheduler tie-break policy.

---

## 8. Global scheduler

### 8.1 Queue

The scheduler owns a min-priority queue keyed by:

```text
(timestamp, timestamp_phase, generation,
 event_class_priority, stable_target_ordinal, event_id)
```

`timestamp_phase` is part of the named timing-profile contract, not an arbitrary
implementation tie-break. Within one phase, `generation` expresses zero-delay causal
depth. The remaining fields make execution deterministic only for operations already
defined as commutative or independent. They must not decide scientific competition.

### 8.2 Advance operation

The core loop is:

```text
while queue not empty and run condition not met:
    t = queue.minimum_timestamp
    batch = queue.pop_all(timestamp == t)
    process_timestamp_closure(t, batch)
```

The engine never advances every neuron merely because time changed.

### 8.3 Timestamp closure

Events at one timestamp are processed by declared timestamp phase, then causal generation:

```text
phase 0, generation 0: events assigned to the first phase
phase 0, generation 1: zero-delay consequences of phase 0 / generation 0
phase 1, generation 0: events assigned to the next phase
```

All events in one `(phase, generation)` are batched before their effects produce the next
generation. Zero-delay children normally inherit their parent's phase and use
`generation + 1`. This prevents iteration order from turning physically simultaneous
events into a hidden sequence.

### 8.4 Per-generation phases

For each `(timestamp, timestamp_phase, generation)`:

1. **Collect** every event in the generation.
2. **Advance affected targets lazily** from their own `last_update_time` to the timestamp.
3. **Reduce arrivals by target and port** while retaining contributor event ids.
4. **Apply state discontinuities** according to the timing profile's declared endpoint
   phase: drive starts/ends, charge, conductance, dendritic arrivals, expiry, refractory
   end, and resets.
5. **Invalidate and repredict crossings** for targets whose future trajectory changed.
6. **Validate due crossing predictions** against target state versions.
7. **Resolve explicit competition domains** over the simultaneous valid candidates.
8. **Commit spikes concurrently** for accepted candidates.
9. **Commit learning once per accepted spike** from its frozen causal snapshot.
10. **Emit child events** with the same timestamp and `generation + 1` for zero-delay
    edges, or a future timestamp for delayed edges.
11. **Record outcomes** for every consumed event.

No child emitted by one event in a generation may affect whether an independent crossing
from that same generation already occurred. Competition-domain arbitration is the only
pre-commit exception.

---

## 9. Equal-time semantics

### 9.1 Independent cells

Valid equal-time crossings in different competition domains all commit. Their output order
is observational only.

Reordering nodes or edges must not change the spike multiset, weights, or descendant event
multiset for independent cells.

### 9.2 Single-winner domains

Each ordinary-E WTA bank is an explicit `single_winner` competition domain declared by
topology metadata.

If two or more cells in the same domain cross at the same timestamp and generation:

1. all are recorded as simultaneous candidates;
2. exactly one is accepted according to the domain's declared arbitration rule;
3. the initial compatibility rule is lowest stable `node_ordinal`;
4. rejected candidates are recorded as `competition_rejected`, not erased;
5. only the accepted spike learns and emits outputs.

This preserves single-winner behavior while making the tie rule a named scientific
contract rather than a side effect of scheduler iteration.

Later arbitration rules, such as seeded random ties, require a separate experiment and
must be serialized in topology metadata.

### 9.3 Zero-delay inhibition

A hard reset generated after a committed spike:

- cannot retroactively erase that spike or its learning;
- cannot erase another independent spike already committed in the same generation;
- can invalidate a crossing prediction in a later causal generation at the same timestamp;
- cancels all active compatible drive on its target;
- increments the target state version.

WTA losers are rejected by domain arbitration, not by pretending that the winner's reset
arrived earlier than a simultaneous crossing.

### 9.4 Endpoint ownership

An event exactly at a timestamp belongs to that timestamp. There is no boundary-edge case
analogous to the historical `tau == 1.0` bug because no timestamp is excluded by an open
interval.

### 9.5 Compatibility endpoint phase

At an exact compatibility boundary `t = kD`, operations occur in this order:

1. validate and commit crossings predicted at the endpoint of window `k-1`;
2. finish the zero-delay closure caused by those commits;
3. end the old drive intervals and lazily settle old-window trace/conductance/guard state;
4. deliver new-window stimulus, feedforward, pretrained, basal, and conductance arrivals;
5. assemble the new compatible drive state;
6. apply delayed feedback hard resets scheduled for this timestamp, so they discard the
   newly assembled drive exactly as the current engine does;
7. predict and process crossings due from the resulting new-window state;
8. expire basal tokens whose final carried window has completed, after they have received
   the endpoint's full zero-delay closure.

Thus a threshold reached at the old interval's endpoint is not cancelled by `drive_end`,
while a delayed feedback reset can still cancel the new interval's evidence before it
crosses. Event-class priorities implement this named contract; they are not arbitrary
tie-breaks.

---

## 10. Crossing prediction and lazy invalidation

### 10.1 Per-cell state

Every membrane cell owns:

```text
last_update_time
state_version
scheduled_crossing_event_id | null
active_drive_contributions
active_conductances
membrane state
refractory state
```

### 10.2 Local advancement

When an event touches a cell, the cell analytically advances from `last_update_time` to the
event timestamp under the piecewise-constant or exactly solvable inputs active over that
interval. Cells not touched are not advanced.

Read-only observation may materialize a projected state at the observation timestamp
without mutating the scientific state.

### 10.3 Prediction

After any trajectory-changing event, the cell:

1. increments `state_version`;
2. computes its next possible crossing from current local state;
3. enqueues one `crossing_prediction` carrying that version;
4. stores the new event id.

### 10.4 Invalidation

Heap deletion is not required. When a crossing prediction is popped:

- if its `target_state_version` differs from the cell's current version, record
  `stale_prediction` and discard it;
- if refractory, gate, or threshold conditions no longer hold, record
  `condition_invalidated`;
- otherwise recompute locally and accept only if the crossing is still due at the current
  timestamp.

A reset, new input, drive end, conductance change, gate change, or refractory change
therefore invalidates predictions in `O(1)` plus one new heap insertion.

### 10.5 Complexity requirement

Processing one event shall not scan every membrane. Runtime work is bounded by:

```text
O(log Q + affected targets + emitted fanout)
```

where `Q` is the number of queued events. Global scans are permitted only for explicit
observations, serialization, validation audits, or diagnostics.

---

## 11. Synaptic delivery

### 11.1 Delay belongs to the edge

Every edge has:

```text
delay_ticks >= 0
delivery_model
```

The engine must not infer physical delay from layer name, preset name, or a global
`SYNAPTIC_DELAY` after migration.

The compatibility graph builder initially assigns:

- ordinary feedforward, pretrained, and basal: `D`;
- C feedback reset: `D`;
- apical permission: `0`;
- relay excitation and immediate WTA hard reset: `0`.

### 11.2 Delivery values

An emitted plastic edge snapshots the weight at emission time unless the declared synapse
model says otherwise. Learning caused by the source spike affects future emissions, not an
event already in flight.

Each delivery event retains:

- edge id;
- source and target;
- emitted weight/value;
- emission and arrival timestamps;
- parent spike event;
- presentation id.

### 11.3 Target reduction

Multiple arrivals may be numerically summed for integration only after their distinct event
records are retained. Participation and learning read the contributing edge/source set, not
an already-collapsed scalar.

---

## 12. Membrane and drive semantics

### 12.1 Compatibility rectangle

For `boundary_compatible`, each ordinary excitatory delivery creates a drive contribution:

```text
DriveContribution {
    contribution_id
    source_id
    edge_id
    presentation_id
    current
    start_time
    end_time = start_time + D
}
```

The cell's current is the sum of active contributions. `drive_end` is a real queued event.
Firing or hard reset consumes the contributions selected by the compatibility policy and
invalidates their scheduled end effects without deleting their ledger entries. A membrane
may commit at most one spike per compatibility window, matching
`fired_this_boundary`; later inputs remain observable but cannot cause a second compatible
spike in that window.

This is the explicit replacement for `freeze_drive` and `remaining_excitation`.

### 12.2 Compatibility settlement

The compatibility profile preserves the current piecewise rule:

- leak/conductance are constant over the same interval in which the current engine freezes
  them;
- end-of-window conductance, trace, passive-weight, and refractory settlement occurs after
  endpoint crossings;
- skipped empty windows are applied lazily with the exact discrete recurrence;
- only cells with nontrivial state need settlement bookkeeping.

This is required for trace equivalence. Replacing a discrete end-of-window retention with a
continuous exponential during the interval would be a neuron-model change.

### 12.3 Continuous-profile conductance and leak

Only in `continuous_physical`, leak and inhibitory conductance are evolved continuously by
elapsed time. A legacy retention factor `alpha` may initialize an equivalent time constant:

```text
tau_decay = -D / ln(alpha)       for 0 < alpha < 1
g(t + dt) = g(t) * exp(-dt / tau_decay)
```

`alpha = 1` means no decay. `alpha = 0` requires an explicit expiry model and must not be
passed through the logarithm.

### 12.4 Refractory

Firing sets:

```text
refractory_until = spike_time + refractory_duration
```

Crossing prediction is disabled until `refractory_end`. Refractory is not decremented by
observation or controller calls.

---

## 13. Coincidence pyramidal cells

### 13.1 Basal tokens

Every basal arrival creates a distinct eligibility token:

```text
BasalToken {
    token_id
    source_id
    edge_id
    signal
    arrival_time
    expires_at
    presentation_id
    consumed: bool
}
```

In `boundary_compatible`, the token is valid throughout its arrival window and one carried
window. Its expiry is owned by the final endpoint phase, after endpoint crossings and their
zero-delay closure. No global boundary settle pass is used.

### 13.2 Apical permission

Apical arrivals are zero-delay events retaining their source identity. A C cell applies its
declared dendritic rule to the complete same-generation apical multiset and currently valid
basal tokens.

### 13.3 Deposit

The current rule remains:

- strict basal-and-apical coincidence;
- at most one committed deposit per compatibility window;
- causal basal weight selected by basal source in a multi-basal C;
- somatic charge impulse at the apical timestamp;
- deposit and token consumption recorded atomically;
- gate-dependent firing.

The deposit may create a valid C crossing in the next causal generation at the same
timestamp.

### 13.4 No silent token loss

Every token ends as exactly one of:

- `consumed_by_deposit`;
- `expired_unused`;
- `cancelled_by_reset` if a future model explicitly declares that behavior.

The current compatibility profile does not cancel basal eligibility on ordinary somatic
reset unless the existing code does so.

---

## 14. Inhibitory relays and multiplicity

Queue-level event multiplicity is always conserved. Neuron-level output multiplicity is
model-specific and explicit.

For each relay resolution, record:

```text
input_event_ids
input_count
accepted_input_count
coalesced_input_count
output_spike_event_ids
reset_event_ids
```

The compatibility relay model:

- receives the full simultaneous input multiset;
- may emit at most one immediate WTA spike per compatibility window;
- records additional simultaneous inputs as coalesced;
- does not pretend they never arrived.

The delayed C-feedback action remains distinct from the immediate WTA action. A C event
schedules the declared future reset even if the same relay already emitted an immediate WTA
spike at that timestamp.

A future counted-suppression model may map each accepted C event to a durable suppression
credit, but that is not introduced by this scheduler refactor.

---

## 15. Learning transaction

Learning occurs atomically with an accepted `spike_commit`.

The transaction snapshots before reset:

- membrane value and accumulated causal drive;
- active contributing deliveries by afferent;
- `presentation_id` values;
- pre-update weights;
- causal C basal token and deposit index, where applicable;
- learning parameters and distance factors.

It then:

1. computes the current production equation unchanged;
2. updates each permitted weight once;
3. records old value, delta, new value, and every factor;
4. increments the cell state version if the changed weight affects an active drive model;
5. ensures in-flight events retain emission-time values.

Serialization, observation, replay export, and dashboard polling are read-only and can never
trigger learning.

---

## 16. Stimulus and experiment controller

Stimulus pacing is a controller concern, not a hidden scheduler clock.

Required APIs:

```text
schedule_stimulus(at, input_vector, duration=None) -> presentation_id
run_until(time)
run_for(duration)
run_next_timestamp()
run_events(max_events)
run_until_idle(max_events=None, max_time=None)
```

`run_events(n)` advances by processed causal events, which is useful for hardware-oriented
event-count measurements. It must report both final physical time and event count.

`run_until(T)` is inclusive: it drains every phase and zero-delay generation at timestamp
`T` before returning. `run_next_timestamp()` likewise processes the complete closure at the
selected timestamp.

The compatibility adapter may implement the historical:

```text
step()
```

as `run_until(current_time + D)`, but scientific code must not assume that one call equals
one neural update.

Auto-pacing may derive stimulus timestamps from the graph's declared feedback latency. It
must schedule ordinary stimulus events; it must not alter queue ordering or neural state.

---

## 17. Zero-delay closure and termination

Zero-delay paths are required for apical permission, relay action, hard reset, and dependent
C firing. They also create a risk of non-terminating same-time cycles.

Termination rules:

- a compatibility membrane emits at most one spike per compatibility window; a later
  continuous-profile membrane emits at most one spike per timestamp unless it explicitly
  declares burst support;
- one basal token is consumed at most once;
- one C deposit transaction key commits at most once per compatibility window;
- a compatibility relay emits at most once per action class per compatibility window;
- every state transition must either consume a one-shot resource, increase a state version,
  or schedule into future time;
- graph validation identifies zero-delay strongly connected components and requires each
  cycle to contain a declared consumptive transition;
- `max_generations_per_timestamp` is a safety assertion, not a normal stopping rule.

Exceeding the generation limit raises `ZeroDelayCycleError` with the causal subgraph and
event chain. Events are never silently dropped to escape a loop.

---

## 18. Determinism and reproducibility

Given identical:

- canonical topology;
- parameters;
- initial weights/state;
- seed;
- stimulus schedule;
- timing profile and `time_quantum`;

the engine must reproduce:

- spike events and timestamps;
- competition decisions;
- learning updates;
- weights;
- event outcome ledger;
- final projected state.

Determinism must hold across:

- repeated runs in one process;
- equivalent topology JSON with reordered nodes/edges;
- different observation chunking (`run_until(T)` versus several smaller calls);
- dashboard polling frequency;
- enabled versus disabled passive instrumentation.

Parallel execution may not be promoted until it reproduces the serial reference ledger or
has an explicitly versioned deterministic reduction contract.

---

## 19. Observation, dashboard, and replay

### 19.1 Read-only projection

`dynamic_state(at=None)` returns the projected state at the requested timestamp. It may
analytically project a cell from `last_update_time` without committing that projection.

The payload adds:

```text
simulation_time
time_unit
time_quantum
processed_event_count
pending_event_count
last_timestamp_event_count
last_timestamp_generation_count
timing_profile
```

Legacy `timestep` may remain as a presentation/observation counter but must be labeled as
such and must not drive neural behavior.

### 19.2 Dashboard run loop

The browser refresh rate is independent of simulation time. The backend may process many
timestamps between frames or pause with no events pending.

The UI must display:

- simulation time;
- event count and event rate;
- timing profile;
- queue depth;
- whether a run stopped by time, event budget, idle state, or user action.

### 19.3 Replay

A new replay schema version records:

- engine/timing profile;
- `time_quantum`;
- absolute timestamps;
- event ids and causal parents where detailed tracing is enabled;
- observation schedule separately from scientific events.

Old boundary replays remain readable through the current player and are never silently
reinterpreted as physical-time runs.

---

## 20. Proposed module boundary

The implementation should separate policy from mechanics:

```text
backend/next_event/
    clock.py              fixed-point time conversion and validation
    events.py             typed immutable envelopes and payloads
    queue.py              timestamp batches and lazy invalidation
    scheduler.py          causal-generation closure
    ledger.py             event outcomes and causal tracing
    competition.py        explicit domain arbitration
    controller.py         stimulus and run-until APIs
    adapter.py            existing SimulationEngine/dashboard compatibility
```

Neuron-local analytic evolution remains in `snn/neurons.py` or a small shared numeric
module. Topology construction remains in `backend/network_spec.py`.

The queue must not know about layer names, C cells, learning equations, or topology presets.
The neuron model must not inspect the global queue.

---

## 21. Migration plan

### Phase 0 — freeze the semantic oracle

- Preserve the current engine unchanged.
- Capture canonical traces for `rg_direct_cc4`, `rg_coincidence`, `tiled_cc`,
  `tiled_cc_direct_identity`, and `two_tower_composition`.
- Include exact ties, `tau == 1.0`, delayed feedback reset, C eligibility carry, pattern
  change, input pause, and cold recall.
- Record emitted values, participation, pre-reset learning state, and final weights.

Gate: existing complete suite passes and trace artifacts are reproducible.

### Phase 1 — event envelope and ledger in shadow mode

- Add immutable event ids and causal ancestry to the existing engine's diagnostics.
- Do not change scheduling.
- Prove instrumentation neutrality.

Gate: spikes and weights remain byte-identical with tracing on versus off.

### Phase 2 — physical-time delay queue

- Replace delay-one dictionaries with timestamped deliveries under
  `boundary_compatible`.
- Keep the current crossing scan temporarily.
- Preserve explicit presentation ids and contributors.

Gate: canonical delivery and learning traces match.

### Phase 3 — local crossing predictions

- Add per-cell `last_update_time`, state versions, and versioned crossing events.
- Replace all-membrane rescanning with the priority queue.
- Retain compatibility rectangles and current decay/eligibility behavior mapped to time.

Gate: full compatibility trace equivalence and no production code path performs a global
crossing scan.

### Phase 4 — concurrent timestamp batches

- Implement causal generations, independent simultaneous commits, and explicit WTA domain
  arbitration.
- Add multiplicity ledgers and topology-order invariance tests.

Gate: all concurrency acceptance tests below pass. Any intended difference from stable
iteration order is reviewed as a scientific change, not hidden inside the refactor.

### Phase 5 — dashboard and replay adapter

- Implement existing configuration/topology/state interfaces.
- Add time/event controls and schema versioning.
- Keep the current engine selectable for A/B comparison.

Gate: frontend tests, replay tests, API tests, and a live dashboard smoke pass.

### Phase 6 — performance promotion

- Profile queue operations, fanout, observation, serialization, and learning separately.
- Run sparse and dense workloads at increasing topology scale.
- Publish the event-count-normalized results.

Gate: correctness gates remain green and the performance criteria below are met.

### Phase 7 — optional continuous physical profile

Requires separate approval after compatibility promotion. It may change pulse shapes,
delays, and temporal learning exposure only through explicit ablations.

---

## 22. Required acceptance tests

### 22.1 Time and queue

- fixed-point conversion never schedules early;
- earlier timestamps always process first;
- equal timestamps are popped as one batch;
- child zero-delay events use the next generation;
- past scheduling and overflow fail loudly;
- stale crossing predictions are logged and ignored;
- `run_until(T)` is invariant to controller chunking.

### 22.2 Analytic cell behavior

- pure-integrator crossing is exact before tick quantization;
- conductance-LIF state and crossing agree with the existing RK4 oracle;
- lazy advancement equals eager segment advancement;
- changing drive invalidates and replaces the crossing prediction;
- reset invalidates the old prediction;
- untouched cells perform no state update work.

### 22.3 Simultaneity

- independent equal-time crossings both fire;
- reordering independent nodes and edges changes nothing;
- a single-winner domain accepts exactly one equal-time candidate;
- arbitration is stable under canonical topology reordering;
- zero-delay reset cannot erase an already committed independent spike;
- zero-delay reset invalidates a later-generation dependent crossing;
- dependent C deposit and C spike may occur at the same timestamp in successive generations;
- timestamp closure terminates or raises a diagnosed zero-delay-cycle error.

### 22.4 Multiplicity and conservation

- two equal-time edge events remain two ledger events after numeric reduction;
- relay input/coalescing counts are explicit;
- every emitted event has exactly one terminal outcome;
- every processed child names an existing causal parent;
- no duplicate learning transaction exists for one spike;
- no event vanishes because two values shared an edge, source, target, or timestamp.

### 22.5 C cells

- basal only never deposits;
- apical only never deposits;
- basal within TTL plus apical deposits once;
- expired basal cannot deposit;
- a multi-basal C updates only the causal source's weight;
- same-time multiple apicals remain visible but do not duplicate an idempotent deposit;
- a dormant top C never fires without an apical path.

### 22.6 Learning

- ordinary-E and C deltas reconstruct exactly from recorded pre-update factors;
- participation is target-local and presentation-aware;
- reset after spike does not erase the learning snapshot;
- in-flight event magnitude is unchanged by later learning;
- observation and replay export cause no updates.

### 22.7 Compatibility topology matrix

For multiple seeds and all four 3×3 patterns:

- `rg_direct_cc4`;
- `rg_coincidence`;
- `tiled_cc`;
- `tiled_cc_l1_4`;
- `tiled_cc_direct_identity`;
- `tiled_cc_double_eor`;
- `two_tower_composition`.

Compare:

- spike and reset event multisets;
- timestamps within declared quantization tolerance;
- per-column winners;
- C deposits and causal basal indices;
- participation vectors;
- learning updates;
- final weights;
- feedback cadence and accepted/silent presentations.

### 22.8 Existing regressions

The complete Python and frontend suites remain mandatory. Golden changes require a semantic
explanation and approval; scheduler replacement alone is not permission to regenerate them.

---

## 23. Performance requirements

Correctness is the promotion gate. Performance is measured only after equivalence.

Mandatory structural requirements:

- no per-event scan of every membrane;
- no periodic event generated solely to visit an inactive neuron;
- queue and ledger memory proportional to live pending events plus configured retained
  history;
- observation cost reported separately from simulation cost;
- ability to disable detailed ancestry while retaining aggregate conservation counters.

Promotion benchmark set:

1. one active 3×3 column;
2. 9×9 `tiled_cc` with one, two, and nine active patches;
3. 9×18 `two_tower_composition`;
4. replicated tiled graphs at 10× and 100× current column count;
5. a deliberately dense all-active control.

Report:

- processed causal events per wall-clock second;
- accepted spikes per wall-clock second;
- heap operations per processed event;
- crossing recomputations per affected cell;
- peak queue depth and memory;
- serialization-disabled and dashboard-observed rates;
- scaling versus total neurons and versus active neurons.

Target, not a correctness substitution:

- at least 5× wall-clock improvement over the current engine on the sparse two-tower
  benchmark at equivalent event output;
- sparse-work growth tracks active cells and fanout rather than total membranes;
- dense worst-case behavior is no worse than 2× the current engine without a documented
  reason.

If these targets fail, retain the correct implementation as an experimental backend and
profile before redesigning scientific semantics.

---

## 24. Failure handling and resource limits

Every run accepts optional:

```text
max_time
max_events
max_pending_events
max_generations_per_timestamp
max_wall_time
```

Stopping at a resource limit returns a structured incomplete result. It is not reported as
convergence, quiescence, or successful training.

Fatal diagnostic classes include:

- `TimeRepresentationError`;
- `PastEventError`;
- `QueueCapacityError`;
- `ZeroDelayCycleError`;
- `CausalLedgerError`;
- `NonFiniteStateError`;
- `TopologyTimingContractError`.

The error bundle includes the active timestamp, causal generation, triggering event,
affected node/edge, recent causal ancestry, and queue summary.

---

## 25. Promotion and rollback

During migration:

- `hybrid_boundary` remains the default backend;
- `next_event` is explicit and experimental;
- both consume the same validated `NetworkSpec`;
- experiments record backend and timing profile;
- saved learned weights may be copied only after topology and learning-contract checks;
- transient state is not transferred unless a versioned state translator proves exact.

`next_event` becomes default only when:

1. all compatibility and full regression tests pass;
2. topology-order invariance passes;
3. event conservation is complete;
4. dashboard and replay adapters pass;
5. benchmark results are published;
6. documentation no longer describes an experimental backend as production;
7. the old engine remains available for at least one release as an oracle.

Rollback is selecting `hybrid_boundary`; it must not require reverting learned data or
rewriting topology files.

---

## 26. Completion criteria

The custom engine is complete when all of the following are true:

1. absolute fixed-point timestamps replace scientific dependence on an outer step loop;
2. simulation advances only to pending events or requested observations;
3. edge delay is explicit;
4. membrane state advances lazily and analytically;
5. crossings are versioned queue events, not products of a global scan;
6. equal timestamps are concurrent batches with explicit competition domains;
7. zero-delay causal closure is deterministic and terminating;
8. every event has identity, ancestry, and one terminal outcome;
9. aggregation never destroys multiplicity or learning provenance;
10. C basal eligibility uses timestamped tokens and expiry;
11. decay, trace, refractory, and stimulus timing use elapsed time;
12. learning commits exactly once from causal pre-reset state;
13. compatibility traces and weights match the validated hybrid engine;
14. dashboard, replay, and experiment interfaces remain usable;
15. sparse scaling improves without weakening semantics;
16. no new topology, gating mechanism, or forced frequency policy was introduced.

At that point the simulator may accurately be called an absolute-time next-event engine
under its declared profile. The `continuous_physical` profile remains a separate scientific
model until independently validated.
