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

`api.py` creates the engine from `DASHBOARD_OVERRIDES`. The active browser preset is
the validated `rg_coincidence` turnover configuration: zero leak/refractory,
`eta=0.01`, `c_eta=0.005`, `l2_init_total_frac=0.95`. Ordinary-E learning is cap-free
(no per-synapse weight cap; the FE budget saturates each row total). The four built-in
topologies are `rg_coincidence`, `tiled_cc`, `tiled_cc_l1_4`, and `tiled_cc_feature_gated`
(the feature-gated tiled variant, `9×9 Feature-Gated CC (L1=8)` in the selector).
`CONFIG_SPEC` is the only list of controls shown in the browser — exactly the six that
affect these presets
(`topology`, `leak_rate`, `refractory_steps`, `eta`, `c_eta`, `l2_init_total_frac`).
Applying configuration validates against a small allowlist (`EDITABLE_KEYS`), rejects any
other key, and rebuilds the engine, so it also clears learned state.

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
| POST | `/api/input` | Set the full nine-pixel vector. |
| POST | `/api/pixel/{i}` | Toggle one input pixel. |
| POST | `/api/clear`, `/api/random`, `/api/noise/{prob}` | Modify input. |
| POST | `/api/weight` | Edit one L1E_s→L2E feedforward weight. |
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

Synapse kinds: `feedforward`, `feedback`, `coincidence_local` (paired
L1E_s[i]→L1E_new[i] sensory afferent), `relay_excitation` (structural E→I,
`weight: null`), `inhibition` (frozen I→E subtractive gate, positive magnitude +
`sign: -1`; the L1I→L1E_s wipe is delivered one step after the relay fires).

Dynamic state contains changing values:

```text
timestep, running, speed
neurons            potential, activation, spiked, freq, refractory, assembly
changed_synapses   sparse weight deltas {id, weight}
emitted            synapse ids that carried a spike this step (edge flashes)
applied_inhibition hard-wipe events {target, v_pre, charge_removed, reached_rest}
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
2. Add one range, toggle, or select entry to `CONFIG_SPEC` in
   `backend/dashboard_config.py`.

`frontend/controls.js` generates the element automatically. The configuration
surface is intentionally small; do not reintroduce topology sizes, per-layer
thresholds, or ablation switches.
