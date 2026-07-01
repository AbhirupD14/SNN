# SNN

A from-scratch spiking neural network built up from the simplest possible
leaky integrate-and-fire (LIF) neuron, adding biological mechanisms one at a
time. Pure NumPy, neuron-local learning, no gradients or supervision.

## Core model

- **`neuron.py`** — LIF `Neuron` with fixed threshold, fixed leak, refractory
  period, and a trace-gated, sign-preserving Hebbian rule.
- **`neuron_flexible.py`** — same dynamics with arbitrary fan-in built up via
  `add_input_connection()` / `finalize_connections()`.
- **`layers.py`** — `InputLayer` (E/I pairs) and a basic `CorticalColumn`.
- **`cortical_column_flexible.py`** — `CorticalColumn` with one trainable
  feedforward synapse **per source** (per-pixel receptive fields) plus a shared
  inhibitory neuron.

### Learning rule

Each synapse carries an **eligibility trace** — the un-summed counterpart of the
membrane potential — that accumulates presynaptic activity and leaks at the same
rate. When a neuron fires:

```
weights += learning_rate * trace * sign(weights)
```

so only the synapses that actually delivered charge are credited, and each is
strengthened in the direction of its own sign (excitatory → more positive,
inhibitory → more negative). A neuron is excitatory or inhibitory purely by the
sign of the weight it lands on in its target; inhibitory neurons just fire and
train their own afferents with the identical rule.

> **Not yet added (deliberately):** a counter-force (normalization / homeostasis
> / maturation). Weights currently only ever grow toward the cap, so a single
> neuron can win every pattern — see Test B below.

## Tests

- **`test_neuron.py`** — unit tests for both neuron classes: trace gating, sign
  preservation, leak, weight cap, refractory, flexible-fan-in parity.
- **`test_8line_consolidation.py`** — 8 line patterns on a 3×3 grid through an
  L1→L2 network. Test A shows a winning neuron forming a selective receptive
  field on a pattern's active pixels; Test B characterizes the current
  single-winner collapse that awaits the counter-force.

```
python3 test_neuron.py
python3 test_8line_consolidation.py
```

## Dashboard

A real-time web dashboard (FastAPI + WebSocket backend, vanilla-JS/Three.js
frontend) visualizes the network, streams live spiking and learning, and lets
you drive patterns, inspect neurons, and control execution. The simulation stays
completely decoupled from the UI.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.api:app
```

Then open <http://127.0.0.1:8000>. See [`docs/DASHBOARD.md`](docs/DASHBOARD.md)
for the architecture, REST/WebSocket API, serialization format, and rendering
pipeline.
