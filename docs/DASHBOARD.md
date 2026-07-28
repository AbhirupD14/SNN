# Dashboard boundary

The Python simulation is the source of truth. The dashboard can select inputs,
control execution, edit exposed configuration, and visualize snapshots; it does
not perform neural computation.

```text
SimulationEngine
    -> topology() / dynamic_state()
    -> serializer.py
    -> api.py + websocket.py
    -> frontend store
    -> renderer and inspectors
```

## Run

```bash
.venv/bin/uvicorn backend.api:app
```

Open <http://127.0.0.1:8000>. The backend serves the frontend and the WebSocket
from the same origin. Three.js is loaded from a CDN; all project code is local.

## Backend responsibilities

| File | Responsibility |
| --- | --- |
| `dashboard_config.py` | Dashboard preset and declarative control schema. |
| `layout.py` | Functional model coordinates and spacing constraints. |
| `api.py` | Translate HTTP/WS requests into engine methods. |
| `websocket.py` | Own the simulation run loop and client broadcasts. |
| `serializer.py` | Wrap engine state in protocol envelopes. |
| `simulation.py` | Own all model state and behavior. |

`api.py` creates the engine from `DASHBOARD_OVERRIDES`. The active browser preset is the
9x9 `tiled_cc` hierarchy under the dual FE/FES rule at fast maturation rates: zero
leak/refractory, `eta=4.0`, `c_eta=16.0`, `e_weight_cap_frac=0.5`, `input_period=0` (auto),
with two patches pre-loaded. (`SimulationEngine`'s own default stays `rg_coincidence` with
the production rule, so every golden and headless run is unaffected.) Ordinary-E learning is
cap-free
(no per-synapse weight cap on the RULE; the FE budget saturates each row total, under the
`θ/2` detector and `θ` one-afferent ceilings). The six built-in topologies are
`rg_coincidence`, `tiled_cc`, `tiled_cc_l1_4`, `tiled_cc_direct_identity`
(`Tiled CC Direct Identity · 9×9 · 8 E/column` in the selector — the Eor-less
source-addressed hierarchy, see `docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md`),
`tiled_cc_double_eor` (`Tiled CC Double-Eor · 9×9 (latency probe)` — a DIAGNOSTIC preset with
one extra output relay, see `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`), and
`rg_direct_cc4` (`3×3 Direct CC · 4 E + WTA I`).

**Input period** is the only control that applies *without* rebuilding, so a trained column
can be re-paced live. Its default `0` means auto: the engine derives the graph's top-down
loop latency and presents one RGC volley per resolved causal chain, which is what makes the
confirmed column alternate exactly. Setting it to `1` restores the historical
volley-every-boundary regime, in which the cadence reflects wiring depth instead.

Selecting a **multi-basal** coincidence cell (every C in `tiled_cc_direct_identity`) shows
its complete basal vector in the inspector — one row per local ordinary-E source with that
source's own learned weight, and the causal / current / carried source marked per boundary —
instead of the single "learned basal weight" card a one-basal C shows.
`CONFIG_SPEC` is the only list of controls shown in the browser — exactly the nine that
affect these presets (`topology`, `input_period`, `leak_rate`, `refractory_steps`, `eta`,
`c_eta`, `l2_init_total_frac`, `dual_fe_fes`, `c_feedback_reset`). Applying configuration
validates against a small allowlist (`EDITABLE_KEYS`), rejects any other key, and rebuilds
the engine, so it also clears learned state — **except** for the keys in
`RUNTIME_ONLY_KEYS` (currently just `input_period`), which change presentation pacing only
and therefore apply in place, preserving learned weights and the timestep.

To watch the validated turnover, set the dashboard to 120 steps/s and present
`row 1` for approximately 2500 steps, then `col 1` for 2500, then `row 1` again.
The top-bar winner and raster/charge views show the original owner, replacement,
and recovery respectively.

## REST API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/state` | Full `{topology, dynamic}` snapshot. |
| GET | `/api/config` | Exposed control schema and current values. |
| POST | `/api/start`, `/api/pause`, `/api/step` | Control execution. |
| POST | `/api/reset`, `/api/reseed` | Rebuild with the same or a new seed. |
| POST | `/api/speed/{sps}` | Set 0.5–120 steps per second. |
| POST | `/api/pattern` | Select a named pattern using `{"name": "row 0"}`. |
| POST | `/api/input` | Set the full topology-sized input vector. |
| POST | `/api/pixel/{i}` | Toggle one input pixel. |
| POST | `/api/clear`, `/api/random`, `/api/noise/{prob}` | Modify input. |
| POST | `/api/weight` | Edit one mutable feedforward, predictive, or C-basal weight by edge id. |
| POST | `/api/stimulate` | Stimulate one neuron. |
| POST | `/api/config` | Apply allowlisted configuration overrides and rebuild. |

## WebSocket protocol

`WS /ws` sends JSON messages shaped as `{"type": ..., "data": ...}`.

A new client first receives `topology`, then `dynamic`. A reset, reseed,
configuration change, or manual weight edit sends a new topology. While running,
the server streams dynamic state after each timestep.

Topology contains stable identities and structure:

```text
neurons           id, label, layer, type, role, threshold, functional pos
synapses          id, source, target, kind, weight (null for structural relays),
                  sign (-1 on inhibition gates)
patterns          names and input vectors
grid              input rows and columns
params            engine parameter snapshot (incl. threshold_l2, l2e_weight_cap_frac
                  used by the receptive-field / weights charts)
```

Current edge kinds are `feedforward`, `fixed_excitation`, `trace_excitation`,
`relay_excitation`, `inhibition`, `predictive_inhibition`, `pretrained_excitation`,
`basal_excitation`, `apical_excitation`, and `hard_reset_inhibition`. Structural relays
serialize with `weight: null`; inhibitory edges carry positive magnitude plus `sign: -1`.

Dynamic state contains changing values:

```text
timestep, running, speed
neurons            potential, activation, spiked, freq, refractory, assembly
changed_synapses   sparse weight deltas {id, weight}
emitted            synapse ids that carried a spike this step (edge flashes)
inhibitory_pulses  conductance pulses with source, target, kind, and before/after values
hard_reset_events delayed feedback resets with source, target, and removed charge
latency_ties       same-time event ties recorded by the event scheduler
column_winners     first ordinary-E winner and sub-boundary tau for each column
input, winner
stats, log
```

Static topology is not repeated every frame. The frontend keeps the latest
synapse values and applies sparse deltas from dynamic messages.

## Frontend map

| File | Responsibility |
| --- | --- |
| `app.js` | Shared store and component wiring; the `applyTopology`/`applyDynamic` seam shared by live frames and replay. |
| `websocket.js` | Reconnecting client. |
| `controls.js` | Inputs, execution, and generated config controls. |
| `renderer.js` | Three.js neuron and synapse view. |
| `inspector.js` | Selected-neuron details. |
| `raster.js`, `charge.js`, `weights.js`, `receptive.js` | Focused analysis panels. |
| `replay.js`, `replay_player.js` | Read-only **Load Test** replay of a recorded `replay.snn.jsonl` — see [REPLAY_PLAYER.md](REPLAY_PLAYER.md). |

The renderer stores backend coordinates as `functionalPos`. It derives a
separate `pos` by expanding within-layer and between-layer offsets. Synapse lines
use the expanded display positions; the backend's functional coordinates are used
only to set per-synapse learning rates (they never scale delivered charge).

The orthographic camera is refit after topology changes. Neuron meshes and
synapse lines are created once per topology, then mutated for each dynamic frame.

## Adding a control

1. Add the key to `DEFAULTS` and `EDITABLE_KEYS` in `backend/simulation.py` so
   `apply_config()` accepts it.
2. If it changes runtime behaviour ONLY (never graph construction or initial weights), also
   add it to `RUNTIME_ONLY_KEYS` so applying it does not rebuild and wipe learned state.
3. Add one range, toggle, or select entry to `CONFIG_SPEC` in
   `backend/dashboard_config.py`.

`frontend/controls.js` generates the element automatically. The configuration
surface is intentionally small; do not reintroduce topology sizes, per-layer
thresholds, or ablation switches.
