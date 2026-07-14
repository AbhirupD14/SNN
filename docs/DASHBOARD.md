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

`api.py` creates the engine from `DASHBOARD_OVERRIDES`. `CONFIG_SPEC` is the
only list of controls shown in the browser. Applying configuration rebuilds the
engine, so it also clears learned state.

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
| POST | `/api/weight` | Edit one L1E→L2E feedforward weight. |
| POST | `/api/stimulate` | Stimulate one neuron. |
| POST | `/api/config` | Apply exposed configuration overrides and rebuild. |
| POST | `/api/autocycle` | Configure automatic pattern visits. |

## WebSocket protocol

`WS /ws` sends JSON messages shaped as `{"type": ..., "data": ...}`.

A new client first receives `topology`, then `dynamic`. A reset, reseed,
configuration change, or manual weight edit sends a new topology. While running,
the server streams dynamic state after each timestep.

Topology contains stable identities and structure:

```text
neurons           id, label, layer, type, threshold, functional pos
synapses          id, source, target, kind, current weight/confidence
patterns          names and input vectors
grid              input rows and columns
params            complete engine parameter snapshot
```

Dynamic state contains changing values:

```text
timestep, running, speed
neurons           potential, activation, spike, frequency, refractory state
changed_synapses  sparse weight deltas
input, winner, episode, autocycle
stats, log
```

Static topology is not repeated every frame. The frontend keeps the latest
synapse values and applies sparse deltas from dynamic messages.

## Frontend map

| File | Responsibility |
| --- | --- |
| `app.js` | Shared store and component wiring. |
| `websocket.js` | Reconnecting client. |
| `controls.js` | Inputs, execution, and generated config controls. |
| `renderer.js` | Three.js neuron and synapse view. |
| `inspector.js` | Selected-neuron details. |
| `raster.js`, `charge.js`, `weights.js`, `receptive.js` | Focused analysis panels. |

The renderer stores backend coordinates as `functionalPos`. It derives a
separate `pos` by expanding within-layer and between-layer offsets. Synapse lines
use the expanded display positions, but distance attenuation continues to use
only the backend's functional coordinates.

The orthographic camera is refit after topology changes. Neuron meshes and
synapse lines are created once per topology, then mutated for each dynamic frame.

## Adding a control

1. Confirm the parameter is accepted by `SimulationEngine.apply_config()`.
2. Add its default experiment value to `DASHBOARD_OVERRIDES` if needed.
3. Add one range, toggle, or select entry to `CONFIG_SPEC`.

`frontend/controls.js` generates the element automatically. Keep mechanisms that
are only useful for scripted ablations out of `CONFIG_SPEC`.
