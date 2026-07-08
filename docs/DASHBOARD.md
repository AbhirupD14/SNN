# SNN Dashboard

A real-time research dashboard for the spiking neural network. The Python
simulation is the source of truth; the dashboard only visualizes network state,
inspects neurons, and drives execution. The simulation is never coupled to the
UI — the dashboard reads state through a serializer and issues control verbs
through a small HTTP/WebSocket API.

```
neuron_flexible.py / layers.py / cortical_column_flexible.py   (network core)
        │
        ▼
backend/simulation.py   SimulationEngine  (steppable wrapper, plain-Python state)
        │  serializer.py
        ▼
FastAPI  (backend/api.py + websocket.py)  ── REST control + JSON over WebSocket
        │
        ▼
frontend/ (HTML + CSS + vanilla JS + Three.js)  ── render, inspect, control
```

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.api:app          # add --reload while developing
```

Then open <http://127.0.0.1:8000>. The frontend is served by the backend, so the
WebSocket is same-origin — no CORS or file:// issues. Pick a pattern (or draw on
the 3×3 grid), press **Start**, and watch L1 drive the L2 pool.

> Three.js is loaded from a CDN via an import map, so the browser needs internet
> access. Everything else is served locally.

## Project structure

```
backend/
    simulation.py   SimulationEngine: builds the network, steps it, and produces
                    static-topology / dynamic-state snapshots. The only glue
                    between the neural code and the dashboard.
    serializer.py   Wraps engine snapshots in the {type, data} protocol envelopes.
    websocket.py    ConnectionManager (broadcast) + SimulationRunner (run loop).
    api.py          FastAPI app: REST endpoints, /ws, and static frontend mount.

frontend/
    index.html      Layout: top bar, left controls, 3D viewport, inspector, tabs.
    style.css       Dark research-tool theme.
    app.js          Orchestrator: shared store, wires WS → renderer/inspector/…
    websocket.js    Reconnecting WebSocket client.
    renderer.js     Three.js scene; objects built once, mutated per frame.
    controls.js     Left sidebar + tabs; every action is an HTTP POST.
    inspector.js    Right sidebar neuron detail cards.
    charts.js       Heat map, histogram, live line charts, stats, event log.
```

## REST API

All control endpoints are under `/api`. State changes are pushed back to every
client over the WebSocket, so responses are minimal.

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/state` | Full snapshot: `{topology, dynamic}` |
| POST | `/api/start` | Begin streaming steps |
| POST | `/api/pause` | Stop streaming |
| POST | `/api/step`  | Advance exactly one timestep |
| POST | `/api/reset` | Rebuild the network from scratch |
| POST | `/api/speed/{sps}` | Set target steps/second (0.5–120) |
| POST | `/api/pattern` | Load a named line pattern (`{"name":"row 0"}`) |
| POST | `/api/pixel/{i}` | Toggle input pixel `i` |
| POST | `/api/input` | Set the full 9-pixel input vector (`{"vector":[…]}`) |
| POST | `/api/clear` / `/api/random` | Clear / randomize the input |
| POST | `/api/noise/{prob}` | Flip each pixel with probability `prob` |
| POST | `/api/stimulate` | Manual firing: `{neuron_id, magnitude, continuous}` |

`WS /ws` — on connect the client receives a `topology` message then a `dynamic`
message; while running it receives a `dynamic` message per timestep.

## WebSocket protocol

Every message is `{"type": ..., "data": ...}`.

**`topology`** (sent once on connect and after reset — static, not re-sent per frame):

```jsonc
{
  "type": "topology",
  "data": {
    "neurons":  [{ "id":"L2E0", "label":"out 0", "layer":"L2", "type":"E",
                   "threshold":0.4, "pos":[x,y,z] }, …],
    "synapses": [{ "id":"ff0->0", "source":"L1E0", "target":"L2E0",
                   "kind":"feedforward", "weight":0.28 }, …],
    "layers": ["L1","L2"],
    "patterns": ["row 0", …],
    "pattern_vectors": { "row 0":[1,1,1,0,0,0,0,0,0], … },
    "grid": { "rows":3, "cols":3 },
    "params": { … }
  }
}
```

**`dynamic`** (per timestep):

```jsonc
{
  "type": "dynamic",
  "data": {
    "timestep": 128,
    "running": true,
    "speed": 12,
    "neurons": [{ "id":"L2E0", "potential":0.31, "activation":0.78,
                  "spiked":true, "freq":0.5, "refractory":0, "assembly":"L2E0",
                  "budget":4.0, "budget_used":3.98 }, …],  // budget* on L2E only, else null
    "changed_synapses": [{ "id":"ff0->3", "weight":0.61 }, …],  // only what changed
    "changed_confidence": [{ "id":"ff0->3", "confidence":0.74 }, …],  // L2E gate trust delta
    "input": [1,1,1,0,0,0,0,0,0],
    "winner": "L2E0",
    "stats": { "total":27, "active":4, "firing":2, "avg_activation":0.12,
               "firing_rate":0.21, "avg_weight":0.37, "winner":"L2E0" },
    "log": [{ "seq":42, "t":128, "kind":"learning", "message":"winner -> L2E0" }, …]
  }
}
```

## Serialization format

The serializer separates **static topology** (neuron identities, positions,
connectivity, initial weights) from **dynamic state** (activations, firing,
recently changed weights, stats, log). Topology is transmitted only on connect
and reset; the per-timestep payload carries only what changes, and synapse
weights are streamed as a sparse `changed_synapses` delta rather than the full
matrix. The frontend keeps a running weight map so the inspector can always show
current incoming/outgoing weights without a full retransmit.

Each L2E feedforward synapse also carries a **confidence** value: the neuron's
learned trust in that gate, distinct from the gate size (`weight`). Confidence is
sent on topology synapses and streamed as a sparse `changed_confidence` delta with
the same running-map treatment as weights. The inspector shows confidence beside
each weight, plus a per-neuron **budget usage** meter (sum of positive weights vs
the fixed budget). See `neuron_flexible.Neuron` for the confidence rule.

## Rendering pipeline

`renderer.js` builds one Three.js `Mesh` per neuron and one `Line` per synapse
**once**, from the topology message, and caches them in `Map`s keyed by id. Each
frame it only mutates existing objects:

- neuron `emissiveIntensity` and scale track activation + a decaying spike pulse;
- the winner gets a gold emissive and an orbiting halo;
- changed synapses flash white and settle to an opacity set by `|weight|`;
- display filters (only-active, hide-weak, isolate-assembly, per-layer, hide
  inhibitory) toggle `.visible` without rebuilding anything.

The scene graph is never rebuilt except on `reset`, so the render loop stays
allocation-free and scales to far more than the 27 neurons of this first
experiment. Picking is a raycast on `pointerup` (suppressed if the pointer
moved, so orbiting doesn't select).

## Extending

- **More neurons / layers:** add them in `SimulationEngine._register_neurons`
  (id, layer, type, position) and the synapse list; the frontend adapts
  automatically from the topology message.
- **New metrics:** add fields to `SimulationEngine.stats()` and a `.sbox` /
  chart in `charts.js`.
- **New experiments (plus signs, letters, hierarchy):** swap `PATTERNS` and the
  network built in `_build`; keep the serializer/protocol unchanged.
