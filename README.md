# SNN

A small, from-scratch spiking neural network for learning four overlapping 3×3
line patterns. The model uses NumPy, local plasticity, no gradients, and no global
error signal. The public presets use analytic sub-boundary event resolution, explicit
unit-delay excitation, immediate WTA hard resets, and delayed feedback hard resets.
The generic engine also retains the synchronous conductance path for compatible custom
graphs.

The network is a **graph** built from a `NetworkSpec` (typed nodes + typed edges);
the engine executes whatever graph it is given. **Six** built-in presets ship,
selected by the `topology` parameter (`'rg_coincidence'` default, `'tiled_cc'`,
`'tiled_cc_l1_4'`, `'tiled_cc_direct_identity'`, `'tiled_cc_double_eor'`,
`'rg_direct_cc4'`), and you can build arbitrary graphs live in
the browser **Topology Editor** (🧬 in the top bar) and save/load them as presets:

The general `SimulationEngine` default remains the validated `rg_coincidence` turnover
preset. The browser dashboard intentionally opens on `tiled_cc` with the fast-maturation
inspection contract (`eta=4`, `c_eta=16`, `e_weight_cap_frac=0.5`,
`input_period=0`/auto); this does not alter headless or golden defaults.

> **The learning rule is cap-free; the structural ceilings are separate.** The ordinary-E
> update itself has a **zero floor and no upper bound**: saturation comes from the
> neuron-wide free-energy term, as the incoming row total approaches the budget
> `B = e_maturity_budget_frac·θ` the update vanishes on its own. Two hard **per-synapse**
> ceilings can then be applied on top of whichever rule is active, by how many afferents the
> cell must integrate. The `θ/2` detector ceiling is explicit in the tiled dashboard and
> scaling contract; the generic engine default leaves it unset:
>
> | Ceiling | Applies to | Why |
> | --- | --- | --- |
> | `θ/2` (`e_weight_cap_frac`) | pattern-detector feedforward: RGC→L1E, Eor→parent-E, legacy ordinary learners | no single afferent may fire the cell, so it must integrate **≥ 2** evidence volleys |
> | `θ` (`relay_weight_cap_frac`) | **E→Eor** and the **C basal** | these fire on **one** afferent, so `θ/2` would silence them — `θ` is exactly their one-shot target, reached and never exceeded |
>
> **Predictive-inhibitory** weights keep their own mechanism-specific cap.

> **Input pacing is derived, not fixed.** A top-down `C → I` feedback loop of latency `L`
> takes `L` boundaries to return, so presenting a fresh RGC volley every boundary
> (`input_period=1`, the engine default) leaves ~`L` presentations overlapping in flight and
> makes the feedback cadence depend on wiring depth. `input_period=0` (the dashboard default)
> means **auto**: the engine derives `L` from the graph and presents **one volley per
> resolved causal chain**, which yields exact `fire/silent` alternation on every seed and at
> every loop depth. See `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`.

> **E→Eor is a fixed relay.** Every `E → Eor` weight starts at `θ` (`eor_w_init_frac=1.0`)
> and does **not** learn (`eor_plasticity_enabled=False`), so any single winner in a column
> drives its Eor to threshold from the first boundary and forever after. Starting at the
> ceiling leaves plasticity only one direction to move — down — and the accumulating rule
> depresses every *non-participating* afferent, so a plastic Eor would decay the afferents of
> ordinary E that have not recently won. A newly recruited owner would then win its column's
> WTA but fail to drive Eor, silencing the column's output during exactly the turnover the
> hierarchy exists to express. Only Eor's own bank is frozen: `Eor→parent-E` is owned by the
> parent (still a `θ/2` detector) and `Eor→C` basal by the C cell.

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
  measured behavior below and `Current_Implementation_Methodology_Equations.md`.
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
  supplies the child C's apical permission. **Eor uses the ordinary event-resolved
  excitatory membrane** (same class, threshold, leak, membrane and integration — it is
  not a stateless Boolean operator); it differs by its edges and by its synaptic role:
  a **fixed** relay bank held at `θ`
  rather than a plastic `θ/2` detector (see the callout above). There are no lateral connections; columns are independent, so
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
- **`topology='tiled_cc_direct_identity'` — the Eor-less direct-identity hierarchy (181
  neurons, 1546 edges).** Same tiling and same column motif *minus the `Eor` relay*: every
  ordinary **E** projects to **every** parent ordinary E, so each L2 detector owns **72**
  distinct plastic weights (9 child columns × 8 possible winners) — one per
  `(child column, child winner)` **source address** — instead of one pooled "this column
  fired" event. Because the winner identity is the output alphabet, each column's **C**
  owns one learned basal afferent **per local E** (a *multi-basal* C: the causal source
  selects the weight, and only that weight learns, so one owner's association never
  depresses another's). Apical permission stays the unweighted Boolean parent-E gate.
  Measured: turnover + recall with 100 % direct parent evidence (12/12 runs, 3 seeds), and
  L2 distinguishes two compositions that differ **only** in local winner identity.
  The former exact-two-patch `tau=1.0` C deadlock is fixed: the event loop now drains
  boundary-edge crossings, so C fires, learns, and schedules feedback without retaining
  runaway voltage. See `docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md` §8.1.
  Acceptance: `experiments/direct_identity_experiment.py`.
- **`topology='tiled_cc_double_eor'` — DIAGNOSTIC latency probe (201 neurons, 1062
  edges).** Exactly `tiled_cc` with one extra output relay spliced into the ascending path
  (`E → Eor → Eor2 → parent E`), lengthening the top-down confirmation loop by one boundary
  and changing nothing else. Built to falsify the prediction that the feedback cadence is set
  by loop latency — it is: the period moved 6 → 8 on 8/8 seeds. Not a research topology; see
  `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`.
- **`topology='rg_direct_cc4'` — the direct single-column experiment (14 neurons, 44
  edges).** A `3×3` RGC surface feeding **four** ordinary latency-E competitors densely,
  each driving one **central WTA I** that hard-resets all four. No feature relay,
  coincidence C, Eor, or hierarchical feedback — the minimal competitive column used as the
  dual FE/FES acceptance topology (`experiments/dual_fe_cc4_consolidation.py`).
- **`topology='two_tower_composition'` — the two-tower scaling experiment (393 neurons,
  2162 edges).** Two lateral `9×9` tiled towers share one `9×18` input sheet. Each tower's
  nine L1 columns feed its own L2 column, and the two L2 columns feed one L3 composition
  column through the existing column connector. Tower L2 separates the tested `V`, `A`,
  and `7` glyph halves, but the classic one-Eor output collapses each tower to the same
  source address at L3, so the three composed glyphs collide there. See
  `docs/TWO_TOWER_COMPOSITION.md`.

> **Rejected direction.** A dedicated per-feature gated tiled variant
> (`tiled_cc_feature_gated`) was implemented and then removed: it demonstrated selective
> relay suppression but at the wrong structural level. See “Rejected direction: per-feature
> gated tiled columns” in `Current_Implementation_Methodology_Equations.md`; git history is
> the archive for the deleted implementation.

All seven built-in presets are **event-resolved** (the analytic sub-boundary scheduler,
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
The current equations, rationale, and measured limitations are in
`Current_Implementation_Methodology_Equations.md` and
`docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`.

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

For a single simulation step, read `SimulationEngine.step()` and `_event_step()` in
`backend/simulation.py`. Every built-in preset is event-resolved: integer outer boundaries
deliver delay-1 feedforward/basal/reset events, while an analytic sub-boundary scheduler
orders membrane crossings by `tau`, executes zero-latency apical and WTA consequences, and
drains dependent crossings available exactly at `tau=1.0`. Custom graphs without an
event-resolved archetype or hard-reset edge retain the legacy synchronous path.
`ExcitatoryNeuron`, `CoincidencePyramidalNeuron`, `SourceNeuron`, and `InhibitoryNeuron`
in `snn/neurons.py` own the local state transitions and learning rules.

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
| `backend/network_spec.py` | The `NetworkSpec` vocabulary, the seven built-in presets, and `validate_spec`. |
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
The browser protocol and view boundary are in
[`docs/DASHBOARD.md`](docs/DASHBOARD.md). The fabric's validated operating envelope,
known failure boundaries, and unresolved work are in
[`docs/FABRIC_CONSTRAINTS_AND_OPERATING_ENVELOPE.md`](docs/FABRIC_CONSTRAINTS_AND_OPERATING_ENVELOPE.md)
and
[`docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`](docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md).
The compatibility-first absolute-time scheduler design is specified in
[`docs/NEXT_EVENT_ENGINE_TECHNICAL_SPEC.md`](docs/NEXT_EVENT_ENGINE_TECHNICAL_SPEC.md).

## Editing the topology

Open the **Topology Editor** (🧬 in the top bar). It renders the live network in 3D
at its real functional positions, so what you see is the actual topology (z is
preserved through save/load/apply — nothing is flattened). Drag a neuron to move it,
drag one node onto another to wire an edge (the kind is inferred from the two
archetypes), click an edge to toggle it directional/bidirectional or delete it, add
neurons from the palette, and **Apply** to rebuild the live network (every view
refreshes off the broadcast). Save the current graph as a named preset and load it
back later; the seven built-ins (`rg_coincidence`, `tiled_cc`, `tiled_cc_l1_4`,
`tiled_cc_direct_identity`, `tiled_cc_double_eor`, `rg_direct_cc4`,
`two_tower_composition`) are always
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
`Current_Implementation_Methodology_Equations.md`.

Excitatory neurons integrate `acc_weights` (learned by a cap-free rule under a hard
per-synapse ceiling — see the table above) jointly with a persistent inhibitory conductance `g_inh` (decaying,
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

Coverage includes the shared membrane/segment equations, causal event scheduler, boundary-edge
drain, hard-reset WTA and delayed feedback, both learning families and their structural
ceilings, graph validation, preset persistence, replay/branching, serialization/API, and
golden topology regressions. Tiled coverage includes classic 8-E and 4-E columns, fixed Eor,
the diagnostic double-Eor loop, 81-pixel patch composition, graph-derived auto-pacing, and
the Eor-less direct-identity hierarchy. The direct-identity tests pin down its 181/1546
graph, source-addressed L2 weights, multi-basal C causal learning, `tau=1.0` behavior, and
identity-discrimination experiment.

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
