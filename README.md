# SNN

A small, from-scratch spiking neural network for learning four overlapping 3×3
line patterns. The model uses NumPy, local plasticity, no gradients, and no global
error signal. Inhibition is **persistent conductance** (never a hard wipe), every
excitatory cell carries a local activity trace, and the timestep is synchronous with
explicit unit synaptic delays.

One `enew_enabled` flag selects the topology:

- **`False` — the predictive-inhibition (PI) experiment.** Eight pattern-specific
  predictive interneurons `PI[j]`, paired 1:1 with the competitors `L2E[j]`, each
  with nine locally-plastic inhibitory synapses onto the sensory `L1E_s` cells. Tests
  temporal explaining-away / symmetry breaking on overlapping patterns.
- **`True` (default) — the retained L1E_new coincidence comparison topology.**

Directories: `snn/` + `backend/` + `frontend/` are the model and its dashboard;
`experiments/` holds the overlap symmetry-breaking experiment and a legacy frequency
analysis.

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
`backend/simulation.py` top to bottom. It runs synchronous subphases: deliver
delay-1 arrivals (inhibitory conductance, then excitatory charge) and external
input; integrate every excitatory neuron once (joint excitation/inhibition);
threshold-test and fire (L1E_s, then deterministic L2E winner-take-all); update
each cell's local activity trace; emit spikes into delay-1 queues and run the local
PI / L1I inhibitory plasticity; decay conductances and count down refractory; record
the frame. `ExcitatoryNeuron`, `InhibitoryNeuron`, and `PredictiveInterneuron` in
`snn/neurons.py` own the local state transitions, the conductance/trace dynamics, and
the two weight rules (excitatory accumulating + local predictive-inhibition).

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
| `experiments/predictive_inhibition_overlap.py` | Multi-seed row→col→row symmetry-breaking experiment + controls. |
| `experiments/frequency_experiment.py` | Legacy analytic leaky-integrator study (see below). |

The implemented model — conductance dynamics, activity trace, local PI plasticity,
timestep/delays, and the symmetry-breaking results — is documented in
[`Current_Implementation_Methodology_Equations.md`](Current_Implementation_Methodology_Equations.md).
The browser protocol and view boundary are in
[`docs/DASHBOARD.md`](docs/DASHBOARD.md). `docs/BOOLEAN_COINCIDENCE_OPEN_PROBLEM.md`
and `docs/INTRINSIC_ADAPTATION_DESIGN.md` predate the conductance/PI rewrite and are
retained as historical context only.

## Architecture summary (predictive-inhibition topology, `enew_enabled=False`)

```text
external -> 9 L1E_s ==ff==> 8 L2E --relay--> 8 PI (paired 1:1)
                ^                |                |
                |   72 locally-plastic predictive inhibitory conductance synapses
                +----------------|----------------+
                                 L2E --relay--> 1 L2I_WTA --> all L2E (WTA conductance)
```

Each `PI[j]` learns inhibitory outputs onto the sensory features that were locally
active when its paired `L2E[j]` fired (via each L1E_s cell's activity trace). On a
later overlapping pattern the incumbent's persistent conductance suppresses the
shared feature more than the novel ones, so a rival can win — measured symmetry
breaking, with the incumbent recovering its original pattern afterwards. See the
methodology document for equations, controls, and honest failure modes.

Excitatory neurons integrate `acc_weights` (learned) jointly with a persistent
inhibitory conductance `g_inh` (decaying, `E_inh = 0` shunting) and carry a local
activity trace that survives voltage reset. Inhibitory relays are stateless; the
engine turns their firing into a conductance pulse. `PredictiveInterneuron` cells own
locally-plastic inhibitory output weights. Functional coordinates set per-synapse
learning rates only; `frontend/renderer.js` expands separate display positions that
cannot change simulation behaviour.

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

Coverage: conductance/trace dynamics and local PI plasticity
(`test_conductance_neuron.py`), the excitatory weight rule, both topologies' exact
neuron/edge counts, the synchronous causal WTA step, engine-level predictive
inhibition + the symmetry-breaking causal controls (`test_predictive_inhibition.py`),
the retained coincidence branch, serialization/API, and the legacy frequency model.

## Overlap symmetry-breaking experiment

```bash
PYTHONPATH=. .venv/bin/python -m experiments.predictive_inhibition_overlap
```

Runs the deterministic `row → column → row` schedule on the PI topology across seeds
with controls (predictive conductance off, plasticity off, fast/slow association),
and measures symmetry breaking, incumbent contamination, shared-vs-novel suppression,
recovery, and sparsity. Results are written to
`experiments/predictive_inhibition_results.json` (see the methodology document).

## Legacy frequency experiment

```bash
PYTHONPATH=. .venv/bin/python -m experiments.frequency_experiment
```

An analytic study of an abstract leaky *jump* integrator (the old membrane model),
retained for reference. The live engine neuron is now conductance-based, so this
module uses a self-contained reference integrator rather than the engine neuron.
