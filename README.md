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

**1. Excitatory plasticity — on a postsynaptic spike.** The excitatory trace has
two selectable meanings (`Neuron(trace_mode=...)`), so the two can be A/B-compared
at identical seeds and hyperparameters.

*`trace_mode="activity"` (default).* Each synapse carries an **eligibility
trace** — the un-summed counterpart of the membrane potential — that accumulates
presynaptic activity and leaks at the same rate. When a neuron fires:

```
weights += learning_rate * trace * sign(weights)
```

so only the synapses that actually delivered charge are credited, and each is
strengthened in the direction of its own sign. A neuron is excitatory or
inhibitory purely by the sign of the weight it lands on in its target.

*`trace_mode="confidence"`.* Separates two biological quantities: the **weight**
is the gate size, and a new persistent per-synapse **confidence** is the neuron's
trust that opening that gate helps it fire. Confidence starts small (0.10) and
updates only on a successful spike — participating synapses grow toward 1
(`c += beta*(1-c)`), non-participants slowly forget (`c *= 1-gamma`). The fixed
learning budget is then allocated by confidence, not equally:

```
credit_i = c_i / Σ(c over active excitatory synapses)
weights += learning_rate * credit_i
```

The existing weight-budget normalization and cap run unchanged afterward, so the
finite-resource interpretation is preserved; only the *distribution* of learning
changes. Confidence is a credit-assignment rule, independent of the inhibitory
system (it only ever touches positive synapses). `Neuron.plasticity_stats()`
exposes weight/confidence entropy and concentration for diagnosing specialization.

**2. Inhibitory plasticity — on an inhibitory discharge** (`Neuron.apply_inhibition`).
An inhibitory synapse is treated as a **finite adaptive suppression gate**. When
an inhibitory spike discharges an excitatory neuron, and *only* then, the gate
learns from how close that neuron was to firing at the moment of inhibition
(`w = |weight|`, `w_max = weight_cap`, `theta = threshold`):

```
V_pre  = V ;  V = V - w ;  V_post = V     # linear discharge
p      = clamp(V_pre / theta, 0, 1)       # normalized closeness to firing
delta_w = eta * p * (1 - w^2 / w_max)     # saturating; finite synaptic resource
w      = w + delta_w                      # gate strengthens toward w_max
```

The gate strengthens most when it suppresses a **near-winner** (`p → 1`) and
saturates as `w → w_max` (`inhibitory_weight_cap`, kept separate from the
feedforward `weight_cap`), so competition stays bounded **with no global
normalization**. Because the saturation term is quadratic, the gate's *natural*
equilibrium is `w* = sqrt(w_max)`, not `w_max` itself, whenever `w_max != 1`
(growth reverses past that point) — e.g. with `L2_GATE_WMAX = 1.5` the L2
gates settle at `w* ≈ 1.22`, not 1.5. Sign is preserved (inhibitory weights
stay negative; `|w|` grows). The two systems never touch the same weights: excitatory plasticity moves
only positive synapses, inhibitory plasticity only the negative gate it
discharged through.

**3. Homeostatic synaptic scaling — on chronic silence or over-activity**
(`Neuron(homeostasis=True)`). A third local system, and the only one *not* gated
by a spike event. Each neuron keeps a slow EMA of its **own** firing rate (a
"calcium" sensor). When that average leaves a target band, the neuron applies a
**fixed** multiplicative step to its excitatory resource — `×(1+up)` when
chronically silent, `×(1-down)` when chronically over-active — with a deadband in
between. The step is a constant, *not* proportional to any error and not a
gradient; it depends only on the neuron's own rate vs its own set-point, so there
is no global signal. The scaling is multiplicative, so relative weights (the
receptive field) are preserved — it carries no pattern information. It therefore
does not violate "learning happens on fire": the pattern learning still happens
only on a spike (rules 1–2); scaling just restores a starved neuron's gain until
it can fire, after which the on-fire rule carves the field. When enabled it
*replaces* the fixed weight budget as the resource regulator. Biologically this is
Turrigiano-style synaptic scaling; it recruits silent units and tames tyrants on
the rate axis (see the caveat under the benchmark).

### L2 competition = adaptive lateral inhibition

In the dashboard network, competition among the L2 excitatory pool is produced by
this inhibitory rule, **not** by a procedural winner-take-all reset. `L2I` starts
as a **temporal integrator** of L2 population activity and, through learning,
can turn into a single-source relay for a *specific, trusted* winner. Each
`L2E→L2I` synapse is randomly initialized (independently per source), well
below `L2I`'s own threshold (`L2_EI_WEIGHT_INIT_LOW`/`_HIGH` in
`backend/simulation.py`) — early on, no single L2E winner can fire `L2I`
alone; a fast membrane leak (`L2I_LEAK_RATE`, much faster than L2E's slow
`leak_l2`) gives `L2I` a short evidence-retention window, so it only fires
once **several distinct L2E neurons** have spiked within that window (a
"round robin" phase). But these synapses are **not weight-budgeted** and each
is individually capped at `threshold_l2` itself, so as the same unmodified
Hebbian rule keeps crediting whichever L2E habitually co-occurs with an `L2I`
discharge (`L2E` fires → `L2I` fires → that `L2E` gets inhibited, repeat),
that one synapse can grow all the way to threshold and become sufficient
alone — a single spike from that now-trusted source then fires `L2I`
immediately, without needing the rest of the pool. Whenever `L2I` fires
(either regime) it discharges the **entire rest of the pool** through the
`L2I→L2E` gate — not just the neurons that also crossed threshold this step.
See `L2I_Temporal_Integration.md` for the original derivation (now partially
superseded — see the note at the top of that file) and validation
methodology. This is deliberate: the neurons that cause a *flickering*
winner are the ones sitting just **below** threshold; if only co-threshold-crossers
were inhibited, those sub-threshold rivals coasted through untouched and won the
next volley, so the winner rotated every burst. Discharging the whole pool subtracts
each rival's own learned gate magnitude, restarting the race closer to even so the
best-matched integrator can win repeatedly. The gate stays **below threshold**
(`L2_GATE_WMAX < thr_l2`), so this is a partial discharge that preserves cross-volley
evidence — *not* the old hard reset that collapsed the network to one universal
winner. Each gate's strength is *learned* per target by the inhibitory rule above,
so the gates onto the habitual runners-up strengthen and the suppression
self-organizes. See `test_l2_competition.py`.

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
- **`test_l2_competition.py`** — regression test that L2 competition is driven by
  adaptive lateral inhibition rather than a hard reset: multiple neurons fire,
  patterns map to distinct winners, `L2I` mediates the suppression, and the gates
  adapt — no collapse to a single winner (robust across seeds).

```
python3 test_neuron.py
python3 test_8line_consolidation.py
python3 test_inhibitory_plasticity.py
python3 test_l2_competition.py
```

## A/B benchmark: activity vs. confidence trace

`benchmark_trace_modes.py` drives the full engine network under both
`trace_mode` values at identical seeds and reports distinct-winner count,
receptive-field selectivity, single-volley latency, convergence, and
budget/confidence concentration — plus a weak-competition control. Writes a
comparison figure to `sweep_results/trace_mode_ab.png`.

```
.venv/bin/python benchmark_trace_modes.py
```

Finding so far: on this task the two modes are roughly equivalent (both reach
~3/8 distinct winners); confidence saturates broadly because specialization is
**competition-limited** — a neuron that wins several patterns legitimately trusts
many pixels. Confidence is a credit-assignment rule downstream of who-wins, so it
neither creates nor blocks tiling here. The intended weight/confidence divergence
(a consistently-useful gate earns more trust than an intermittent one) is shown
directly in `test_neuron.py::test_confidence_diverges_from_weight`.

Homeostasis finding: turning on homeostatic scaling clearly **recruits dead
units** — the fraction of L2 neurons that ever fire rises from ~2/8 to ~4–6/8
across seeds — and it visibly tames tyrants and grows silent units on the resource
axis (`R` spans ~1–6 after training). But it does **not** by itself produce clean
8/8 tiling (distinct winners rise only ~2→3, seed-sensitively). That is expected:
homeostasis is a *rate/magnitude* regulator, so it decides how much a neuron fires,
not *which pattern* it owns. One-to-one tiling is an assignment/symmetry-breaking
problem (the prior `sim_snn_fep` track needed physical per-step membrane noise +
deterministic first-to-fire for that); homeostasis is necessary to keep all units
in play but not sufficient to assign them.

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
