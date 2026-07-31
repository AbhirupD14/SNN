# Visualizing the NEST prototype in the dashboard

Two ways to see the NEST/NESTML 3×3 prototype through the **existing** dashboard. There is
no second frontend, no browser-side simulation, and no change to NEST's scientific
execution.

> Let NEST compute, let PyNEST observe, and let the existing dashboard render.

Scientific results and their interpretation live in
[`NEST_EVENT_DRIVEN_3X3_REPORT.md`](NEST_EVENT_DRIVEN_3X3_REPORT.md). This document covers
only how to look at them.

---

## 1. Offline replay (the delivered feature)

Convert a completed NEST run into the repository's existing `snn.replay` artifact and open
it with the dashboard's normal **Load Test** control.

```bash
# 1. Produce a replay from a canonical case (runs NEST, then converts)
.nest-env/bin/python -m nest_backend.replay_adapter --case 01_center_patch_row

# 2. Start the ORDINARY dashboard, in the ordinary environment
.venv/bin/uvicorn backend.api:app

# 3. In the browser: 📼 Load Test -> pick the generated replay.snn.jsonl
```

The artifact lands under `experiments/runs/nest_3x3/replay_<case>/replay.snn.jsonl`
(gitignored). Available cases: `01_center_patch_row`, `02_center_patch_col`,
`03_two_independent_patches`, `04_all_nine_patches`, `06b_feedback_on_matured_c`.

The dashboard needs no NEST. The replay is self-contained: schema-1, one header carrying
the complete 191-node / 1052-edge topology with frozen weights, event-driven frames, and a
final result record.

### What you will see

The transport shows **NEST REPLAY** in a distinct colour, with
`native event timing · frozen learning · h=0.1 ms · threads 1 · MPI unavailable`.

Frame labels report **NEST milliseconds**, never legacy boundaries:

```
frame 12/39 · 43.1 ms (tick 431) · window 1
winners this tick: L1c11×3 · MULTI-WINNER
```

Multi-winner ticks are surfaced on the transport itself, because multi-winner behaviour is
the headline NEST observation and must not be something you have to go looking for.

---

## 2. Live view (conditional, and its condition was met)

A **separate, read-only** entry point. It does not replace the production dashboard.

```bash
.nest-env/bin/uvicorn nest_backend.dashboard_api:app --port 8100
```

`GET /api/nest/info` identifies it as NEST LIVE and enumerates what it refuses to do.

Live mode exists only because chunked execution was proven inert **first**:

```bash
.nest-env/bin/python experiments/nest_chunked_equivalence.py
```

At 1, 2 and 4 threads, 27 chunked `Simulate()` calls produce **identical** spike multisets,
node counters and connection weights versus a single `Simulate(total)` — with the chunk
(6.7 ms = 67 ticks) deliberately chosen *not* to divide the 20 ms presentation interval, so
chunk boundaries land mid-volley. `tests/nest/test_nest_live_server.py` re-proves this on
every run, so the server cannot outlive its own justification.

### Process isolation (not a stylistic choice)

NEST's kernel is **process-global and not thread-safe**: calling `ResetKernel()` from a
worker thread of a process whose main thread has already used NEST **segfaults the
interpreter**. That was observed, not theorised. So the live server runs NEST in a
dedicated **child process**, and a lock around the pipe means two requests can never be in
flight — hence never inside `Simulate()` at once.

### Controls

| Control | Effect |
|---|---|
| Start / Pause | whether observation chunks are submitted |
| Step | advances exactly one observation chunk (6.7 ms) |
| Speed | wall-clock viewing cadence **only** — never `h`, delays, `D`, or stimulus times |
| Reset | deterministically rebuilds the same seeded network |
| Pattern | rebuilds with a new stimulus (see below) |

A pattern change **rebuilds** rather than rewriting the past: NEST has already delivered the
earlier schedule, and rewriting a generator's history would change what was already
simulated. The response says so explicitly.

Refused with `409` and a stated reason: weight mutation, reseed, topology editing. Learning
toggles, MPI claims, exact checkpoint/resume and branch-from-replay are not exposed at all.

---

## 3. Panel support

| Panel | Status | Notes |
|---|---|---|
| Topology / network renderer | ✅ full | 191 nodes, 1052 edges, real positions |
| Node & edge selection | ✅ full | by repository ID |
| Receptive field / patch layout | ✅ full | from tiling metadata |
| Spike raster | ✅ full | every recorded spike, every sender |
| Event log | ✅ full | presentation and pattern-switch markers |
| Input pattern | ✅ full | the scheduled stimulus per window |
| Per-column activity | ✅ full | including simultaneous winners |
| Simultaneous winners | ✅ full | surfaced on the transport |
| Static frozen weights | ✅ full | header weights are authoritative |
| Experiment / case metadata | ✅ full | full provenance |
| **Charge over time** | ✅ with `--charge-interval` | otherwise hatched (unknown), never an empty bar |
| **Inspector charge** | ✅ with `--charge-interval` | otherwise *Not recorded for this NEST run* |
| **Charge ring in the 3D view** | ✅ with `--charge-interval` | hidden when unknown; a zero ring would mean "no charge" |
| **Inspector refractory** | ✅ with `--charge-interval` | derived from the recordable `refr_until` |
| **Coincidence charge** | ✅ with `--charge-interval` | the C cell's recordable `basal_charge` |
| **NEST model state** | ✅ with `--charge-interval` | every declared recordable, raw, under its NESTML name |
| Inspector freq | ⚠️ unavailable | these models have no firing-rate state |
| Inspector `g_inh` / trace | ⚠️ n/a | structurally absent: `leak_rate = 0`, no persistent inhibitory conductance |
| Weight-change animation | ⚠️ none | learning is frozen; `changed_synapses` is always empty |
| Branch from weights | ⛔ disabled | NEST branching semantics are undefined |

### Navigating a full-resolution artifact: **◀ event / event ▶**

At `--charge-interval = h` almost every frame is a charge sample and nothing else. The
180 ms saturation demo is 18 000 frames, and **120** of them carry a spike or an input:

```
frame      1   t=8730.01   RGC x6            ← the volley
frame    363   t=8733.63   L1c01E0           ← 362 frames later: the 3.6 ms conduction delay
frame    364   t=8733.64   L1c01I
frame    463   t=8734.63   L1c01Eor
frame    518   t=8735.18   L1c00E4
...
frame    824   t=8738.24   L2c00Eor
frame   1501   t=8745.01   RGC x6            ← the next volley, 677 frames on
```

So 99.3 % of stepping — and 99.3 % of playback time — lands on ticks where nothing
happened. At the default 8×, the gap between the RGC volley and the spike it causes is
**45 seconds** of an unchanging raster.

That dead time is real (it is the conduction delay, and the charge trace is climbing
through it), so playback keeps showing it truthfully. **◀ event / event ▶** navigates it
instead: twelve clicks walks the whole cascade above. The buttons sit on the replay bar and
in every chart overlay. Playback speed also goes to 250× for sweeping the file.

### Driving playback from inside a full-screen chart

The chart overlays (**Charge / time**, raster, weights, receptive fields) cover the top bar,
so each mirrors a transport into its own bar. That mirror drives the **live engine**, and
live mutation is disabled during replay — which used to leave an open overlay with no
working transport at all: the charge view could be opened on a NEST replay but not played,
stepped, or read, so you had to close it to move time and reopen it to look.

Each overlay now carries a **replay** transport (⏮ ▶ ⏭ plus the frame/`t_ms`/window
readout), swapped in for the live one exactly while replay owns the display. Scrubbing and
marker jumps still live on the replay bar itself.

### Full state at full resolution

Two knobs, and both go all the way.

**Every declared recordable is sampled**, not just `q`/`q_pre`. The multimeter reads the
model's own `recordables` list rather than a hard-coded pair, so nothing NEST is willing to
hand over is left behind:

```
event_accumulator   q, q_pre, refr_until
event_relay         q, q_pre, q_confirm, lock_until
event_coincidence   q, q_pre, refr_until, basal_charge,
                    basal_until, apical_until, deposit_lock_until
```

Four of these populate existing panels (`potential`, `v_pre_reset`, `refractory`,
`coincidence_charge`); all of them, raw and under their NESTML names, also ride on each
neuron as `nest_state` and render in the inspector's **NEST model state** card. A variable
added to a `.nestml` model is picked up without anyone having to widen a list.

**`--charge-interval` equal to `h` records every tick.** Frames exist for ticks that carried
an observation, and a multimeter sample *is* an observation — so sampling at the resolution
makes every tick a frame and the artifact a complete state history rather than a subsample:

```bash
.nest-env/bin/python experiments/nest_saturation_4pattern.py --h 0.01 --charge-interval 0.01
```

The player shows all of it. A live run keeps a 1500-frame rolling window because it has no
end; a **replay is a finite file and is displayed whole** — raster, charge and weights all
go unbounded on entry (`setHistoryLimit`), so nothing you loaded falls off the back.

The multimeter is attached at the **demo boundary**, not at construction
(`attach_charge_recording(interval, start_ms=...)`). Sampling every `h` across an 8.9-second
training run would mean tens of millions of samples held and then discarded, since only the
demo window is ever converted.

### Charge: recordable, and recorded on request

**Charge is not unrecoverable in NEST — it just has to be recorded.** `q` and `q_pre` are
declared NESTML **recordables** on all three committed models, and a NEST `multimeter`
samples them:

```bash
.nest-env/bin/python -m nest_backend.replay_adapter \
    --case 01_center_patch_row --charge-interval 1.0
```

That attaches one multimeter per model, sampling every cell at the given interval (which
must be a multiple of `h`). The dashboard's charge view then shows a real trace. A sample
tick becomes a frame tick, so charge is continuous rather than defined only where a spike
happened.

The trace is worth looking at on its own merits — it makes the pure-integrator design
visible. For `L1c11E4` under `row 1`:

```
t=22 ms  q=519.1     dispersed volley arriving
t=23 ms  q=772.4
t=24-41  q=772.4     PERFECTLY FLAT for 19 ms -- leak_rate = 0, no decay
t=42 ms  q=0.0       fired / hard reset
t=43 ms  q=253.3     next volley
```

A multimeter is a **passive** device: it delivers no events, consumes no RNG and alters no
connectivity. `tests/nest/test_nest_charge_recording.py` proves it at three sampling
intervals — identical spikes, per-column counts, winner multiplicity, coincidence counters
and reset counters with and without it.

What is genuinely *not* recoverable is charge **between** samples, and charge for a run
that had no multimeter. Those are reported as unknown, never interpolated: a frame tick
with no sample carries no charge field at all.

Cost: the default 8-presentation artifact grows from ~460 KB / 39 frames to ~2 MB / 174
frames at a 1.0 ms interval, because every sample tick becomes a frame. Use a coarser
interval, or omit the flag, when that matters.

Concretely, unsampled fields are **omitted** from the frame, and every frame carries a
`state_availability` map declaring why — computed per run, since whether charge is
available is a property of the run rather than of NEST:

```json
// without --charge-interval
{"spiked": "recorded", "emitted": "derived", "potential": "unavailable", ...}
// with --charge-interval
{"spiked": "recorded", "emitted": "derived", "potential": "recorded",
 "activation": "derived", ...}
```

The inspector renders those three provenances as coloured badges. `derived` is used only
for values computed from recorded ones by an exact rule: `emitted` (the outgoing edges of
the nodes that spiked) and `activation` (`q / theta`, a per-node division). Never for a
heuristic reconstruction.

**An empty bar means zero. A hatch means unknown.** Conflating them would be the one thing
this integration must not do.

---

## 4. Time mapping

NEST timestamps are milliseconds on a resolution grid; the replay schema requires integer
timesteps.

```
tick    = round(t_ms / h)
t_ms    = tick * h
```

The integer tick goes in the schema's required `timestep` field; the physical `t_ms` rides
in the frame's `nest` block and is what the UI displays. A timestamp that is **not** on the
grid raises rather than rounding — a silently rounded spike would move to a neighbouring
frame and change what you see.

All events sharing a tick share one frame. Frames exist only for ticks that carried an
observation (a spike, an input, or a presentation boundary), so a 180 ms run is ~39 frames
rather than 1800.

Every node appears in **every** frame even when idle. That is deliberate: `applyDynamic`
rebuilds state from `dynamic.neurons`, so an omitted node would keep its previous frame's
`spiked` flag and a past spike would render as current.

---

## 5. Why compiled NESTML needs no new frontend

The models are compiled C++ inside NEST, but PyNEST already exposes everything the
dashboard's vocabulary needs: node and connection identity, spike recorders, simulation
time, and declared model state. The adapter's whole job is to map those observations onto
repository IDs and the existing message shapes. The compiled boundary is not a rendering
boundary.

Live and replay frames are produced by the **same** adapter constants and frame shape, so
the two views cannot drift apart.

---

## 6. Proof that observation does not change results

| Claim | Evidence |
|---|---|
| Conversion is a pure read | `test_conversion_does_not_change_the_run` — metrics byte-identical before/after |
| Recording changes nothing | `test_recording_a_run_matches_an_unrecorded_run` — identical spikes, columns, multiplicity, coincidence counters |
| Chunking changes nothing | `test_chunked_execution_is_identical_to_one_shot` at 1/2/4 threads |
| Determinism | `test_conversion_is_deterministic_for_the_same_artifact` |
| No concurrent NEST entry | `test_only_one_thread_ever_enters_nest` — six concurrent requests yield six distinct advancing chunk times |

---

## 7. Artifact size and browser performance

| Run | Frames | Size |
|---|---|---|
| 2 presentations | 5 | 314 KB |
| 5 presentations (committed fixture) | 23 | 447 KB |
| 8 presentations | 39 | ~460 KB |
| saturation demo, spikes + charge @ 1 ms | 299 | 1.3 MB |
| saturation demo, **every tick** @ h=0.1 (180 ms) | 1 800 | 14 MB |
| saturation demo, **every tick** @ h=0.01 (180 ms) | 18 000 | 138 MB |

The 138 MB case is comfortable, which is worth stating because it looks like it should not
be: `parseReplay` takes **0.7 s** and settles at **0.34 GB heap / 0.52 GB RSS** for all
18 000 frames, and both history charts are virtualized (only the visible time window is
drawn), so the cost of the full file is paid once at load.

Roughly 250 KB is the header topology (191 nodes with full metadata + 1052 edges) and is
irreducible. Beyond that the cost is per frame and scales with `h / charge-interval`:
a spikes-only frame is ~8 KB, a fully-sampled frame ~8 KB more, and at
`--charge-interval = h` there is one frame per tick. **Full resolution is therefore
`duration_ms / h` frames** — budget for it before choosing `h`.

Two things keep that from being worse than it is: `nest_state` values are rounded to 6
decimals (NEST's float64 repr is ~18 characters per value, and unrounded it is the single
largest term in the artifact), and unsampled fields are omitted rather than sent as nulls.

---

## 8. Remaining limitations

Carried in the artifact itself (`topology.nest.known_differences`), so they travel with the
data rather than living only in this file:

- **Frozen learning.** Weights never move; `changed_synapses` is always empty.
- **Native multi-winner behaviour.** Many column/presentation windows have several
  ordinary-E winners. This is a measured native-NEST result, not a rendering artifact.
- **The strict `1010` cadence law does not reproduce.**
- **MPI unavailable** in the installed conda NEST build.
- **Charge is opt-in** via `--charge-interval`. It is sampled rather than continuous — but
  at `--charge-interval` equal to `h` the sampling grid IS the simulation grid, so nothing
  is between samples.
- **No branch-from-replay** for NEST artifacts.

---

## 9. Regenerating everything

```bash
# committed cross-schema fixture (through the real adapter, never hand-edited)
.nest-env/bin/python tests/fixtures/make_nest_replay_fixture.py

# tests
NEST_TESTS_REQUIRED=1 .nest-env/bin/python -m pytest tests/nest/ -q   # 134 passed
.venv/bin/python -m pytest -q                                        # 653 passed
node --test tests/replay.parser.test.mjs                             # legacy, 32 passed
node --test tests/nest.replay.parser.test.mjs                        # NEST, 19 passed
node --test tests/charge.history.test.mjs                            # charge path, 6 passed
```
