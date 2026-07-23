# SNN

A small, from-scratch spiking neural network for learning four overlapping 3×3
line patterns. The model uses NumPy, local plasticity, no gradients, and no global
error signal. Inhibition is **persistent conductance** (never a hard wipe), every
excitatory cell carries a local activity trace, and the timestep is synchronous with
explicit unit synaptic delays.

The network is a **graph** built from a `NetworkSpec` (typed nodes + typed edges);
the engine executes whatever graph it is given. **Four** built-in presets ship,
selected by the `topology` parameter (`'rg_coincidence'` default, `'tiled_cc'`,
`'tiled_cc_l1_4'`, `'tiled_cc_feature_gated'`), and you can build arbitrary graphs live in
the browser **Topology Editor** (🧬 in the top bar) and save/load them as presets:

Both the general `SimulationEngine` default and the browser dashboard open on the
validated `rg_coincidence` turnover preset: zero leak/refractory, L2 learning rate
`0.01`, normalized L2 initial afferent total `0.95θ`, and C basal learning rate `0.005`.

> **Ordinary-E learning is cap-free.** Ordinary excitatory feedforward weights (RGC→L1E,
> E→Eor, Eor→parent-E, and legacy ordinary learners) have a **zero floor and no individual
> upper bound**. Saturation is the neuron-wide free-energy term: as the incoming row total
> approaches the budget `B = e_maturity_budget_frac·θ` the update vanishes on its own, so a
> one-afferent specialist matures toward `B ≈ 1100` (θ=1000) and fires in one boundary. Only
> **C basal** and **predictive-inhibitory** weights keep their own mechanism-specific caps.

> **Removed built-ins.** The historical `pi`, `old`, `rg`, and `rg_residual` presets are no
> longer built-in topologies (they are rejected as `topology=` values). Their graph-building
> mechanics — every neuron archetype and edge kind, generic `NetworkSpec` validation, the
> synchronous engine, and custom/saved-graph execution — remain, so an equivalent graph can
> still be built and run through the **Topology Editor** or a saved `NetworkSpec`.

- **`topology='rg_coincidence'` — the coincidence pyramidal / event-resolved
  experiment (45 neurons, 196 edges).** The first **event-resolved** preset: membrane
  crossings and immediate events are ordered at *analytic sub-boundary times* `tau`
  (no micro-chunks). Ordinary excitation and basal events retain integer-boundary
  delivery; apical permission is the C-specific zero-latency exception. `RG_i` fires a fixed
  **pretrained** `L1E_i` (one spike crosses next boundary). Each `L1E_i` feeds a
  **coincidence** cell `L1C_i` on a single learned **basal** afferent, while every
  `L2E_j` feeds all `L1C` on unweighted Boolean **apical** gates; when basal (current
  or one-boundary-carried) coincides with apical, `L1C` deposits its basal charge as an
  instantaneous same-`tau` somatic impulse.
  Inhibition is a **zero-latency hard reset** (`L1I` resets its paired `L1E`; `L2I`
  resets every `L2E`). **L2 WTA is emergent**: the first `L2E` to reach threshold wins
  and its `L2I` reset cancels the rest — no deterministic winner phase. See the
  measured behavior below and `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md`.
- **`topology='tiled_cc'` — the tiled cortical-column hierarchy (191 neurons, 1052
  edges at the default `cc_e_count=8`).** Reuses the `rg_coincidence` mechanics inside
  reusable **cortical-column tiles** instead of one neuron per pixel. A `9×9` **RGC**
  input surface is tiled into nine `3×3` patches; each patch drives one **L1 column**
  (arranged `3×3`), and one **L2 column** receives all nine L1 outputs. Every column
  has `N = cc_e_count` ordinary competing **E** neurons, one output **Eor**, one
  coincidence **C**, and one immediate relay **I**. Inside a column: each `E → Eor`
  (feedforward) and `E ⇄ I` (relay + `hard_reset`) give the current immediate hard
  single-winner WTA; `Eor → C` is the one learned basal; `C → I` lets a mature C recruit
  the same relay. Between columns: a child `Eor → parent E` (feedforward) and
  `parent E → child C` (unweighted **apical**) — parent ordinary E, **never** Eor,
  supplies the child C's apical permission. **Eor is numerically an ordinary plastic E**
  (same class, threshold, leak, cap-free FE rule, budget — it is *not* a Boolean OR); it
  differs only by its edges. There are no lateral connections; columns are independent, so
  several columns may each produce one local winner in the same boundary while each stays
  hard single-winner. The **top L2 C has no parent and is intentionally dormant** — it
  keeps its single Eor basal edge and eligibility state machine but has zero apical
  inputs, so it never deposits, fires, or learns. The graph is generated from reusable
  tile/connector rules (`build_cortical_column`, `connect_rgc_patch`, `connect_columns`,
  `tiled_cc_spec` in `backend/network_spec.py`), so `N` is configurable and deeper
  hierarchies compose without copying the graph: for any `N` the counts are `10N+111`
  nodes and `129N+20` edges. Selecting it rebuilds the input surface to 81 pixels; the
  headless acceptance probe is `experiments/tiled_cc_experiment.py`. This is a
  single-winner tiled hierarchy — **row+column multi-winner composition and a pure
  discrete-event scheduler remain deferred** (`docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`).
- **`topology='tiled_cc_l1_4'` — the shallow-L1 tiled variant (155 neurons, 620 edges).**
  Identical to `tiled_cc` except each **L1** column has **four** ordinary competing E
  neurons instead of eight (the **L2** column keeps eight); every column still has one Eor,
  one C, and one I. A fixed-shape preset (does not read `cc_e_count`).
- **`topology='tiled_cc_feature_gated'` — the feature-gated tiled variant (424 neurons,
  1932 edges).** Restores `rg_coincidence`'s **feature-specific** inhibitory microcircuit
  inside the tiled L1 layer, which the `tiled_cc` whole-bank column reset had removed.
  Between each `3×3` RGC patch and its **eight** competitors sit **nine fixed feature
  relays** `S[k]` (one per pixel); each relay has a paired coincidence **C[k]** and feature
  inhibitory **If[k]**, and the competitor bank keeps a **separate WTA-only I**. Per feature:
  `RGC[k]→S[k]` (pretrained), `S[k]→E[j]` (feedforward), `S[k]→C[k]` (basal),
  `E[j]→C[k]` (local apical), `C[k]→If[k]` (relay), `If[k]→S[k]` (paired hard reset). This is
  exactly the small circuit (`S`≙`L1E` relay, `E`≙`L2E`, `C`≙`L1C`, `If`≙`L1I`) replicated
  once per feature in every recognition module, so a mature local owner suppresses **only**
  its explained relays — the shared center relay is transiently silenced while the novel
  relays stay active, handing ownership to a different competitor. `variant='feature_gated'`
  in the topology metadata is the single source of truth construction/validation/layout
  branch on. Its top L2 is a plain WTA bank (no C); this variant isolates the input-feature
  turnover mechanism and does **not** solve hierarchical composition. Headless causal
  acceptance: `experiments/feature_gated_turnover.py` (Stage A one RF, Stage B nine RFs).

All four built-in presets are **event-resolved** (the analytic sub-boundary scheduler,
selected automatically from graph metadata). A custom *non*-coincidence graph built in the
editor instead runs the synchronous event engine. The fixed editor vocabulary is eleven node
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
both `C eta = 0.005` (the production default) and `0.01`, all 8/8 seeds held a stable
row owner, recruited a different column owner, and recovered the original owner when
the row returned, with zero L2 tie events. To watch that protocol in the dashboard,
run **row 1** for roughly 2500 steps, **col 1** for 2500, then return to **row 1**. At
120 steps/s, each phase takes about 21 seconds.
The rationale, equations, rejected alternatives, and complete tuning tables are in
`docs/COINCIDENCE_TURNOVER_TUNING.md`.

Run `PYTHONPATH=. .venv/bin/python experiments/coincidence_experiment.py`
(→ `experiments/coincidence_results.json`). Mechanical correctness and the scientific
target are reported **separately**:

- **Mechanics (all hold).** The isolated immature C cell shows the **exact**
  two-coincidence cadence — one spike per two valid coincidences (`[0,1,0,1,…]`). A
  coincidence now commits an instantaneous somatic charge impulse at the permitting
  L2E spike's `tau`; every firing C has exactly the same deposit/spike `tau` as that L2E.
  Mature basal weights remain one-shot capable. Winner identity follows drive without
  node reordering, duplicate apical delivery stays observable, and replay is
  bit-deterministic.
- **Scientific target (measured).** In the 4000-boundary seed-1 held-row validation,
  active C weights mature from ≈520.5 to ≈1115–1117. All 5058 C spikes share their
  permitting L2E `tau` and drive 5058 paired hard resets, 4968 of which suppress the
  paired L1E crossing. The training-inclusive `L1E`/`RG` ratio is ≈**0.586**, while the
  final 500-boundary mature window reaches exactly **0.500**. No hidden winner policy or
  post-hoc spike flag forces that result.

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
| `backend/network_spec.py` | The `NetworkSpec` vocabulary, the four built-in presets, and `validate_spec`. |
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
back later; the four built-ins (`rg_coincidence`, `tiled_cc`, `tiled_cc_l1_4`,
`tiled_cc_feature_gated`) are always
available. Presets persist server-side under `.claude/presets/`.

The palette carries the two archetypes `rg` introduced. An **RG** node is an exogenous
source: it owns an input `pixel` and *cannot be the target of any edge* — the editor
will refuse the wiring and `validate_spec` rejects the graph. An **Encoder** is a
plastic noncompetitive excitatory cell: it learns feedforward afferents with the shared
accumulating rule but never joins L2's winner-take-all, and it carries a `grid` tag
(display / receptive-field only) rather than owning a pixel.

## Architecture summary (default topology, `topology='rg_coincidence'`)

```text
9 RG sources --pretrained--> 9 L1E --basal(learned)--> 9 L1C --relay--> L1I (hard reset -> L1E)
                              |                          ^
                              +==ff==> 8 L2E             | unweighted Boolean apical
                                        |  \-------------+  (every L2E -> every L1C)
                                        +--relay--> 1 L2I (hard reset -> all L2E; emergent WTA)
```

`RG_i` is an exogenous source that fires a fixed pretrained `L1E_i`. Each `L1E_i` feeds a
coincidence cell `L1C_i` on one learned **basal** afferent, while every `L2E_j` feeds all
`L1C` on unweighted Boolean **apical** gates; when basal and apical coincide, `L1C` deposits
its basal charge as an instantaneous same-`tau` impulse. Inhibition is a **zero-latency hard
reset** and **L2 WTA is emergent** — the first `L2E` to threshold wins and its `L2I` reset
cancels the rest. See the measured behavior above and
`docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md`.

Excitatory neurons integrate `acc_weights` (learned, **cap-free** — the FE budget saturates
the row total, see above) jointly with a persistent inhibitory conductance `g_inh` (decaying,
`E_inh = 0` shunting) and carry a local activity trace that survives voltage reset. Inhibitory
relays are stateless; the engine turns their firing into a conductance pulse or hard reset.
Functional coordinates set per-synapse
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
behavioural regression for the three built-in presets (`test_golden_topology.py`), the
synchronous causal WTA step, engine-level predictive inhibition + the symmetry-breaking
causal controls (`test_predictive_inhibition.py`), serialization/API, and the legacy
frequency model. The event-resolved coincidence topology adds the shared LIF/segment
primitives (`test_lif_segments.py`), the isolated C cell + learning rule
(`test_coincidence_cell.py`), graph vocabulary/validation (`test_coincidence_spec.py`),
the sub-boundary scheduler + emergent latency WTA (`test_event_scheduler.py`), the full
`rg_coincidence` preset (`test_rg_coincidence.py`), its public protocol
(`test_coincidence_protocol.py`), and the scientific-validation harness
(`test_coincidence_experiment.py`). The `tiled_cc` cortical-column hierarchy adds the
reusable tile builder + structural validation and exact `191`-node/`1052`-edge counts
(`test_tiled_cc_builder.py`), ordinary-E/Eor numeric parity + generic event-plastic
learning + local hard WTA + shared-tau apical/deposit timing (`test_tiled_cc_engine.py`),
the 81-pixel input surface + patch embedding + dimension config (`test_tiled_cc_input.py`),
the metadata-driven layout + serialization (`test_tiled_cc_layout.py`), the dashboard
payload/config contract (`test_tiled_cc_dashboard_contract.py`), and the headless
acceptance experiment (`test_tiled_cc_experiment.py`, driving
`experiments/tiled_cc_experiment.py`).

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
