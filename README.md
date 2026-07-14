# SNN

A small, from-scratch spiking neural network for learning four overlapping 3×3
line patterns. The model uses NumPy, local plasticity, no gradients, and no
global error signal.

This repository has two ways to run the same `SimulationEngine`:

- `backend/` + `frontend/`: interactive development dashboard.
- `experiments/`: finite headless sweeps that write durable results.

## Start here

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.api:app
```

Open <http://127.0.0.1:8000>. Add `--reload` while editing Python.

## The active code path

```text
browser action
    -> backend/api.py
    -> SimulationEngine method in backend/simulation.py
    -> neuron behavior in neuron_flexible.py and snn/rules/
    -> engine.topology() / engine.dynamic_state()
    -> backend/serializer.py and backend/websocket.py
    -> frontend rendering and inspectors
```

For a single simulation step, read these in order:

1. `SimulationEngine.step()` in `backend/simulation.py` drives L1, delivers
   feedforward charge, resolves L2 competition, applies feedback, and records
   state.
2. `Neuron.fire()` and `Neuron.apply_competitive_reset()` in
   `neuron_flexible.py` perform the neuron-local changes.
3. `snn/rules/` contains the extracted charge-delivery and learning rules.
4. `SimulationEngine.dynamic_state()` creates the data streamed to the browser.

## Small code map

| Path | Responsibility |
| --- | --- |
| `backend/dashboard_config.py` | Exact dashboard preset and the controls visible in the UI. |
| `backend/api.py` | HTTP/WebSocket adapter; it should contain no neural rules. |
| `backend/layout.py` | Seeded functional positions and collision rejection. |
| `backend/simulation.py` | Network construction, stepping, experiment state, and snapshots. |
| `neuron_flexible.py` | Neuron membrane, synapses, firing, and local updates. |
| `cortical_column_flexible.py` | L2 column wiring and propagation. |
| `snn/` | Smaller reusable neuron components and rule functions. |
| `frontend/` | Vanilla JS dashboard; visual positions never alter model distances. |
| `experiments/` | Optional headless sweep runner and read-only results viewer. |

The detailed current model is documented in
[`Current_Implementation_Methodology_Equations.md`](Current_Implementation_Methodology_Equations.md).
The browser protocol and view architecture are in
[`docs/DASHBOARD.md`](docs/DASHBOARD.md).

## Current architecture

```text
L1E pixel encoders -> L2E pattern integrators
L2E winners       -> shared L2I recruitment
L2I event         -> hard reset broadcast to the L2E pool
                     + loser weight update for non-refractory neurons
L2E feedback      -> L1I input suppression
```

The dashboard's active choices are intentionally visible in
`DASHBOARD_OVERRIDES`. General-purpose and ablation parameters remain in
`SimulationEngine`, but they are not duplicated as hundreds of browser controls.

Functional neuron coordinates are produced by the backend and are used for
distance-weighted charge delivery. `frontend/renderer.js` expands those
coordinates for legibility using separate display positions; changing that
render scale cannot change simulation behavior.

## Making a change yourself

- Change the active experiment: edit `backend/dashboard_config.py`.
- Change the order of events in a timestep: edit `SimulationEngine.step()`.
- Change charge attenuation: edit `snn/rules/delivery.py`.
- Change a local weight update: edit the matching module in `snn/rules/` and
  follow its call from `neuron_flexible.py`.
- Change only spacing, colors, or camera behavior: edit `frontend/renderer.js`.
- Add a dashboard control: add one entry to `CONFIG_SPEC`; the frontend builds
  the control from that schema.

Keep the separation between functional model state and display state. In
particular, backend coordinates and synapse distances are model data; expanded
Three.js positions are view data.

## Tests

Tests are plain executable scripts so each behavior can be read and run alone:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_layout_scatter.py
PYTHONPATH=. .venv/bin/python tests/golden/test_golden_equiv.py
```

To run every root regression script:

```bash
for test in test_*.py; do PYTHONPATH=. .venv/bin/python "$test" || break; done
```

The golden test protects exact baseline behavior while larger refactors are in
progress.

## Headless experiments

The experiment runner imports the same dashboard baseline and then applies the
overrides in its config file:

```bash
PYTHONPATH=. .venv/bin/python -m experiments.runner \
  --config experiments/config_example.json
```

See [`experiments/README.md`](experiments/README.md) for sweep and cluster usage.
