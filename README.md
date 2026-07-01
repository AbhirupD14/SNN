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

### Learning rules

There are **two independent, local, event-driven, gradient-free** plasticity
systems.

**1. Excitatory plasticity — on a postsynaptic spike.** Each synapse carries an
**eligibility trace** — the un-summed counterpart of the membrane potential —
that accumulates presynaptic activity and leaks at the same rate. When a neuron
fires:

```
weights += learning_rate * trace * sign(weights)
```

so only the synapses that actually delivered charge are credited, and each is
strengthened in the direction of its own sign. A neuron is excitatory or
inhibitory purely by the sign of the weight it lands on in its target.

**2. Inhibitory plasticity — on an inhibitory discharge** (`Neuron.apply_inhibition`).
An inhibitory synapse is treated as a **finite adaptive suppression gate**. When
an inhibitory spike discharges an excitatory neuron, and *only* then, the gate
learns from how close that neuron was to firing at the moment of inhibition
(`w = |weight|`, `w_max = weight_cap`, `theta = threshold`):

```
V_pre  = V ;  V = V - w ;  V_post = V     # linear discharge
p      = V_pre / theta                    # normalized closeness to firing
delta_w = eta * p * (1 - w / w_max)       # saturating; finite synaptic resource
w      = w + delta_w                      # gate strengthens toward w_max
```

The gate strengthens most when it suppresses a **near-winner** (`p → 1`) and
saturates as `w → w_max`, so competition stays bounded **with no global
normalization**. Sign is preserved (inhibitory weights stay negative; `|w|`
grows). The two systems never touch the same weights: excitatory plasticity moves
only positive synapses, inhibitory plasticity only the negative gate it
discharged through.

> **Note on the excitatory counter-force:** a homeostatic weight *budget*
> (renormalizing positive weights to a fixed sum) exists on the `Neuron` and is
> enabled for L2 neurons in the dashboard, but is left *off* in
> `test_8line_consolidation.py`, which still shows the single-winner collapse
> (Test B). The inhibitory gate above is a *different*, local counter-force that
> bounds competition without any global sum.

## Tests

- **`test_neuron.py`** — unit tests for both neuron classes: trace gating, sign
  preservation, leak, weight cap, refractory, flexible-fan-in parity, and the
  inhibitory-discharge rule (exact dynamics, near-winner specialization,
  saturation at `w_max`, refractory gating, independence from the excitatory rule).
- **`test_8line_consolidation.py`** — 8 line patterns on a 3×3 grid through an
  L1→L2 network. Test A shows a winning neuron forming a selective receptive
  field on a pattern's active pixels; Test B characterizes the current
  single-winner collapse that awaits the counter-force.
- **`test_inhibitory_plasticity.py`** — demonstrates the inhibitory gate: gates
  onto near-threshold neurons strengthen most and saturate at `w_max`, then an
  in-network column run. Prints the per-event debug outputs (`V_pre`, `V_post`,
  `theta`, `p`, `w_before`, `delta_w`, `w_after`).

```
python3 test_neuron.py
python3 test_8line_consolidation.py
python3 test_inhibitory_plasticity.py
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
