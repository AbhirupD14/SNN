# SNN

A small, from-scratch spiking neural network for learning four overlapping 3×3
line patterns. The model uses NumPy, local plasticity, no gradients, and no global
error signal. Inhibition is **persistent conductance** (never a hard wipe), every
excitatory cell carries a local activity trace, and the timestep is synchronous with
explicit unit synaptic delays.

The network is a **graph** built from a `NetworkSpec` (typed nodes + typed edges);
the engine executes whatever graph it is given. **Five** built-in presets ship,
selected by the `topology` parameter (`'pi'` default, `'old'`, `'rg'`,
`'rg_residual'`, or `'rg_coincidence'`), and you can
build arbitrary graphs live in the browser **Topology Editor** (🧬 in the top bar) and
save/load them as presets:

The general `SimulationEngine` default remains `pi` for backwards compatibility, but
the browser dashboard now opens on the validated `rg_coincidence` turnover preset:
zero leak/refractory, L2 learning rate `0.01`, normalized L2 initial afferent total
`0.95θ`, and C basal learning rate `0.001`.

- **`topology='pi'` — the predictive-inhibition (PI) experiment (26 neurons).** Eight
  pattern-specific predictive interneurons `PI[j]`, paired 1:1 with the competitors
  `L2E[j]`, each with nine locally-plastic inhibitory synapses onto the sensory
  `L1E_s` cells. Tests temporal explaining-away / symmetry breaking on overlapping
  patterns.
- **`topology='old'` — the original dense global-inhibition topology (27 neurons).**
  Nine paired `L1I` relays, fed densely by every `L2E` (every `L2E`→every `L1I`), each
  projecting a paired inhibitory conductance onto its own `L1E_s`. The single L2 winner
  drives all nine `L1I`, so every `L1E_s` is shunted — winner-gated global inhibition.
- **`topology='rg'` — the retinal-ganglion source-layer experiment (36 neurons).**
  `old`'s cortex exactly, with nine **RG** cells spliced in ahead of L1 and plastic
  paired `RG_i → L1E_i` synapses. RG cells are *exogenous spike sources*: a held edge
  makes its RG cell spike on every `input_period` boundary and **no cortical inhibition
  can stop it**, so retinal evidence persists even while L1 is shunted. The direct
  external→`L1E` injection is removed in this preset only, so `L1E` becomes a plastic
  **noncompetitive** cell that must *learn* its sensory afferent. Isolates the timing
  consequences of a persistent source layer and a plastic L1 path. It deliberately does
  **not** test contextual explaining-away — `old`'s dense `L2E→L1I` feedback erases
  winner identity, because every winner drives every `L1I`.
- **`topology='rg_residual'` — the residual/error experiment (52 neurons).** Keeps
  `RG→L1E→L2E` as a complete uninhibited evidence path and copies each L1E event into
  a separate `ErrorE` sheet. Paired PI cells learn predictive inhibition onto ErrorE;
  residual events add visible bounded charge to all eight `SwitchI` interneurons.
  A paired winner trace opens a second individually-subthreshold priming branch; only
  their charged coincidence can inhibit that incumbent, after
  which the ordinary shared-L2I WTA chooses one replacement. The exact graph has 274
  directed internal projections.
- **`topology='rg_coincidence'` — the coincidence pyramidal / event-resolved
  experiment (45 neurons, 196 edges).** The first **event-resolved** preset: membrane
  arrivals stay on integer boundaries, but crossings and inhibitory resets are ordered
  at *analytic sub-boundary times* `tau` (no micro-chunks). `RG_i` fires a fixed
  **pretrained** `L1E_i` (one spike crosses next boundary). Each `L1E_i` feeds a
  **coincidence** cell `L1C_i` on a single learned **basal** afferent, while every
  `L2E_j` feeds all `L1C` on unweighted Boolean **apical** gates; `L1C` deposits basal
  charge only when basal (current or one-boundary-carried) coincides with apical.
  Inhibition is a **zero-latency hard reset** (`L1I` resets its paired `L1E`; `L2I`
  resets every `L2E`). **L2 WTA is emergent**: the first `L2E` to reach threshold wins
  and its `L2I` reset cancels the rest — no deterministic winner phase. See the
  measured behavior below and `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md`.

The first four use the same synchronous event engine; `rg_coincidence` uses the
analytic sub-boundary scheduler (selected automatically from graph metadata, so legacy
presets stay byte-for-byte identical). The fixed editor vocabulary is eleven node
archetypes (`rg_source`, `e_sensory`, `e_encoder`, `e_residual`, `e_competitor`,
`e_pretrained`, `e_coincidence`, `e_latency_competitor`, `i_relay`, `predictor`,
`switch`) and ten edge kinds (`feedforward`, `fixed_excitation`, `trace_excitation`,
`relay_excitation`, `inhibition`, `predictive_inhibition`, `pretrained_excitation`,
`basal_excitation`, `apical_excitation`, `hard_reset_inhibition`); see
`backend/network_spec.py`.

### Measured `rg_coincidence` behavior (honest results)

The row→column→row turnover sweep is in
`experiments/coincidence_turnover_sweep.py` with complete results in
`experiments/coincidence_turnover_results.json`. At `L2 init total = 0.95θ` and
`C eta = 0.001`, all 8/8 seeds held a stable row owner, recruited a different column
owner, and recovered the original owner when the row returned; reversing L2 scheduler
order produced the same paired outcomes with zero L2 tie events. To watch that protocol
in the dashboard, run **row 1** for roughly 2500 steps, **col 1** for 2500, then return
to **row 1**. At 120 steps/s, each phase takes about 21 seconds.
The rationale, equations, rejected alternatives, and complete tuning tables are in
`docs/COINCIDENCE_TURNOVER_TUNING.md`.

Run `PYTHONPATH=. .venv/bin/python experiments/coincidence_experiment.py`
(→ `experiments/coincidence_results.json`). Mechanical correctness and the scientific
target are reported **separately**:

- **Mechanics (all hold).** The isolated C cell shows the **exact** two-coincidence
  cadence — one spike per two valid coincidences (`[0,1,0,1,…]`). Calibrated crossings
  match the spec: pretrained `L1E` τ≈0.952, C second-coincidence τ≈0.980 at init /
  ≈0.813 at the cap. Under the full held-row preset, the deliberately slower C learning
  rate moves active basal weights from ≈520.5 to ≈549–554 over 4000 steps; mean C spike
  τ≈0.895 is already **<** mean `L1E` τ≈0.952. Winner identity follows drive without
  any node reordering, and replay is bit-deterministic.
- **Scientific target (measured, not forced).** The requested `L1E`/`RG` firing ratio
  near **0.5** is **not** reached — it measures ≈**0.858** in this separate leak=0.03
  held-row validation. Suppression is real: 2442 C spikes produce 2442 paired hard
  resets, 1707 of which beat the paired L1E crossing. The exact halving remains a
  property of the *isolated* valid-coincidence cadence, not the full circuit's aggregate
  L1E/RG rate. No hidden constant is tuned to move this number.

External input does **not** always target `L1E` directly: it is delivered to whichever
cells own a `pixel` (the *input sinks*). In `pi`/`old` that is the nine `e_sensory`
`L1E` cells; in `rg`/`rg_residual` it is the nine `RG` cells, and `L1E` sees the world only through a
learned synapse. A node's `grid` field is separate: display / receptive-field metadata
with no input attached.

Directories: `snn/` + `backend/` + `frontend/` are the model and its dashboard;
`experiments/` holds the overlap symmetry-breaking experiment, the RG timing/symmetry
experiment, and a legacy frequency analysis.

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
threshold-test and fire (exogenous RG sources, then L1E_s / plastic encoders, then the
deterministic L2E winner-take-all); update each cell's local activity trace; emit
spikes into delay-1 queues and run the local PI / L1I inhibitory plasticity; decay
conductances and count down refractory; record the frame. `ExcitatoryNeuron`,
`SourceNeuron`, `InhibitoryNeuron`, `PredictiveInterneuron`, and
`SwitchInterneuron` in `snn/neurons.py` own
the local state transitions, the conductance/trace dynamics, and the two weight rules
(excitatory accumulating + local predictive-inhibition).

Feedforward dispatch is **generic over hops**: any permitted source spike (an RG cell
or any fired excitatory cell, including a competitor) schedules weighted charge onto its plastic
targets for the next boundary, and causal participation is tracked **per postsynaptic
target per arrival boundary** — never as one global source set. That is what lets `rg`
run two feedforward hops without an L1E update ever seeing an L2E's volley (or a
neighbouring boundary's).

## Small code map

| Path | Responsibility |
| --- | --- |
| `snn/neurons.py` | Excitatory/source/relay/predictor cells plus the local traced `SwitchInterneuron`. |
| `backend/network_spec.py` | The `NetworkSpec` vocabulary, five built-in presets, and `validate_spec`. |
| `backend/simulation.py` | Spec-driven construction (`_build_from_spec`), the generic edge-dispatched step, `current_spec`/`apply_topology`, state snapshots. |
| `backend/presets.py` | Server-side preset persistence (built-ins + saved-graph JSON under `.claude/presets/`). |
| `backend/dashboard_config.py` | The dashboard preset and the small control schema (topology selector + rules). |
| `backend/api.py` | HTTP/WebSocket adapter (incl. `/api/topology*` CRUD); contains no neural rules. |
| `backend/layout.py` | Seeded functional positions (used for learning distances only). |
| `backend/serializer.py`, `backend/websocket.py` | Protocol envelopes and the run loop. |
| `frontend/editor.js` | Full-screen **3D topology editor** (Three.js): drag neurons, wire edges, save/load presets. |
| `frontend/receptive.js` | Receptive-field pop-up (one feedforward grid per competitor; hand-edit any weight). |
| `frontend/` | Vanilla JS + Three.js dashboard; display positions never alter model distances. |
| `experiments/predictive_inhibition_overlap.py` | Multi-seed row→col→row symmetry-breaking experiment + controls. |
| `experiments/rg_timing_symmetry.py` | Multi-seed RG timing/symmetry experiment: `old` vs frozen/plastic/equal-init RG. |
| `experiments/frequency_experiment.py` | Legacy analytic leaky-integrator study (see below). |

The implemented model — conductance dynamics, activity trace, local PI plasticity,
timestep/delays, and the symmetry-breaking results — is documented in
[`Current_Implementation_Methodology_Equations.md`](Current_Implementation_Methodology_Equations.md).
The evolution from a GPT-5.5 architect/Claude implementer split to the current
strength-based Sol/Claude workflow is recorded in
[`docs/agent_workflow_evolution.md`](docs/agent_workflow_evolution.md).
The detailed reading of Silver's *Neuronal arithmetic*, its limits, and the staged
plan for multiplicative `SwitchI` coincidence and paired hard reset are in
[`docs/SILVER_NEURONAL_ARITHMETIC_SWITCHI.md`](docs/SILVER_NEURONAL_ARITHMETIC_SWITCHI.md).
The browser protocol and view boundary are in
[`docs/DASHBOARD.md`](docs/DASHBOARD.md). `docs/BOOLEAN_COINCIDENCE_OPEN_PROBLEM.md`
and `docs/INTRINSIC_ADAPTATION_DESIGN.md` predate the conductance/PI rewrite and are
retained as historical context only.

## Editing the topology

Open the **Topology Editor** (🧬 in the top bar). It renders the live network in 3D
at its real functional positions, so what you see is the actual topology (z is
preserved through save/load/apply — nothing is flattened). Drag a neuron to move it,
drag one node onto another to wire an edge (the kind is inferred from the two
archetypes), click an edge to toggle it directional/bidirectional or delete it, add
neurons from the palette, and **Apply** to rebuild the live network (every view
refreshes off the broadcast). Save the current graph as a named preset and load it
back later; the five built-ins (`pi`, `old`, `rg`, `rg_residual`, `rg_coincidence`) are always available. Presets
persist server-side under `.claude/presets/`.

The palette carries the two archetypes `rg` introduced. An **RG** node is an exogenous
source: it owns an input `pixel` and *cannot be the target of any edge* — the editor
will refuse the wiring and `validate_spec` rejects the graph. An **Encoder** is a
plastic noncompetitive excitatory cell: it learns feedforward afferents with the shared
accumulating rule but never joins L2's winner-take-all, and it carries a `grid` tag
(display / receptive-field only) rather than owning a pixel.

## Architecture summary (predictive-inhibition topology, `topology='pi'`)

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
(`test_conductance_neuron.py`), the excitatory weight rule, all built-ins' exact
neuron/edge counts (including `test_residual_topology.py`), the
graph-driven engine + `NetworkSpec` validation + custom-graph execution + bidirectional
edges (`test_network_spec.py`), preset persistence (`test_presets.py`), a bit-exact
behavioural regression for the four legacy presets (`test_golden_topology.py`), the
synchronous causal WTA step, engine-level predictive inhibition + the symmetry-breaking
causal controls (`test_predictive_inhibition.py`), serialization/API, and the legacy
frequency model. The event-resolved coincidence topology adds the shared LIF/segment
primitives (`test_lif_segments.py`), the isolated C cell + learning rule
(`test_coincidence_cell.py`), graph vocabulary/validation (`test_coincidence_spec.py`),
the sub-boundary scheduler + emergent latency WTA (`test_event_scheduler.py`), the full
`rg_coincidence` preset (`test_rg_coincidence.py`), its public protocol
(`test_coincidence_protocol.py`), and the scientific-validation harness
(`test_coincidence_experiment.py`).

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
