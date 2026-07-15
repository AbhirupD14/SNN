# SNN

A small, from-scratch spiking neural network for learning four overlapping 3×3
line patterns. The model uses NumPy, local plasticity, no gradients, and no
global error signal. There is one scientific model — no mode flags or ablations.

- `snn/` + `backend/` + `frontend/`: the model and its interactive dashboard.
- `experiments/`: one deterministic headless frequency experiment.

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
    -> backend/api.py                 (HTTP/WS adapter; no neural rules)
    -> SimulationEngine method in backend/simulation.py
    -> neuron behavior in snn/neurons.py
    -> engine.topology() / engine.dynamic_state()
    -> backend/serializer.py and backend/websocket.py
    -> frontend rendering and inspectors
```

For a single simulation step, read `SimulationEngine.step()` in
`backend/simulation.py` top to bottom: it delivers input to L1E_s, resolves L1E_s
firing, delivers a dense volley to L2E, runs deterministic WTA (winner + L2I hard
wipe), delivers feedback to L1E_new, relays each L1E_new fire through its paired
L1I to wipe the paired L1E_s, then applies leak/refractory and records the frame.
`ExcitatoryNeuron`/`InhibitoryNeuron` in `snn/neurons.py` own the local state
transitions and the one weight rule.

## Small code map

| Path | Responsibility |
| --- | --- |
| `snn/neurons.py` | `ExcitatoryNeuron`, `InhibitoryNeuron`, shared constants, the one weight rule. |
| `backend/simulation.py` | Topology construction, the deterministic step, and state snapshots. |
| `backend/dashboard_config.py` | The dashboard preset and the small control schema. |
| `backend/api.py` | HTTP/WebSocket adapter; contains no neural rules. |
| `backend/layout.py` | Seeded functional positions (used for learning distances only). |
| `backend/serializer.py`, `backend/websocket.py` | Protocol envelopes and the run loop. |
| `frontend/` | Vanilla JS + Three.js dashboard; display positions never alter model distances. |
| `experiments/frequency_experiment.py` | Deterministic frequency measurement (see below). |

The implemented model is documented in
[`Current_Implementation_Methodology_Equations.md`](Current_Implementation_Methodology_Equations.md).
The browser protocol and view boundary are in
[`docs/DASHBOARD.md`](docs/DASHBOARD.md).
The unresolved temporal-AND and winner-turnover limitation is isolated in
[`docs/BOOLEAN_COINCIDENCE_OPEN_PROBLEM.md`](docs/BOOLEAN_COINCIDENCE_OPEN_PROBLEM.md).
A proposed L2E intrinsic-adaptation mechanism for one-volley winner tyranny is
recorded separately in
[`docs/INTRINSIC_ADAPTATION_DESIGN.md`](docs/INTRINSIC_ADAPTATION_DESIGN.md).

## Architecture summary

```text
9 external pixels -> 9 L1E_s sensory sources -> 8 L2E competitors -> 1 L2I relay (hard wipe)
                     9 L1E_new coincidence detectors: paired L1E_s[i] + dense L2E feedback
                     9 L1I relays: L1E_new[i] -> L1I[i] -> hard-wipe L1E_s[i] (delayed one step)
```

Each L1E_new[i] receives its paired sensory input and dense L2E feedback, making
inhibition pixel-selective during a fixed pattern. The shared weight cap is
`theta/2 = 500`, which calibrates simultaneous mature inputs as a two-input
coincidence (500 + 500 = theta). At the production shared leak, repeated unmatched
inputs can still accumulate, so strict temporal AND behavior and winner turnover
remain open problems documented above.

Excitatory neurons have separate nonnegative `acc_weights` (learned) and one
frozen subtractive gate (`SUBTRACTIVE_SIGN = -1`, magnitude = target threshold,
hard wipe). Inhibitory neurons are stateless instant relays with reported
threshold `theta/3`. Functional coordinates set per-synapse learning rates only;
`frontend/renderer.js` expands separate display positions that cannot change
simulation behaviour.

## Making a change yourself

- Change the timestep order or WTA: edit `SimulationEngine.step()`.
- Change a neuron or the weight rule: edit `snn/neurons.py`.
- Add a dashboard control: add one entry to `CONFIG_SPEC` in
  `backend/dashboard_config.py` and add its key to `EDITABLE_KEYS` /`DEFAULTS` in
  `backend/simulation.py`. The frontend builds the control from the schema.
- Change only spacing, colors, or camera behavior: edit `frontend/renderer.js`.

## Tests

Behavioural `pytest` suite under `tests/`:

```bash
.venv/bin/python -m pytest tests/ -q
```

Coverage: excitatory neuron dynamics + the exact weight rule, the inhibitory
relay, exact topology and edge counts, the causal WTA step, serialization/API,
and the frequency model (`tests/test_frequency.py`).

## Frequency experiment

```bash
PYTHONPATH=. .venv/bin/python -m experiments.frequency_experiment
```

It validates the leaky periodic-integrator inequality, measures the real network's
L1E_s cadence and winner charge, searches jointly over `(leak_rate, e_weight_cap)`,
and reports whether turnover with recovery occurs. Results are written to
`experiments/frequency_results.json`. The current conclusion is a documented
negative result — see the methodology document.
