# `nest_backend` — experimental NEST/NESTML event-driven backend

A **characterisation prototype** for the canonical `tiled_cc` topology. It is not a
replacement for `backend.simulation.SimulationEngine`; the validated Python engine remains
the oracle and is completely unchanged. Nothing in the production engine imports this
package.

Results and their interpretation live in
[`docs/NEST_EVENT_DRIVEN_3X3_REPORT.md`](../docs/NEST_EVENT_DRIVEN_3X3_REPORT.md).
Dashboard viewing (offline replay + the read-only live server) is documented in
[`docs/NEST_DASHBOARD.md`](../docs/NEST_DASHBOARD.md).
The complete equation, NESTML-to-C++ build, PyNEST runtime, JSONL export, and dashboard-load
pipeline is documented in
[`docs/NEST_EQUATIONS_AND_ARTIFACT_PIPELINE.md`](../docs/NEST_EQUATIONS_AND_ARTIFACT_PIPELINE.md).

## Quick start

```bash
./scripts/setup_nest_env.sh                       # creates .nest-env/, builds models, verifies
.nest-env/bin/python scripts/verify_nest_install.py
NEST_TESTS_REQUIRED=1 .nest-env/bin/python -m pytest tests/nest/ -q
.nest-env/bin/python experiments/nest_3x3_suite.py
.nest-env/bin/python experiments/nest_parallel_check.py
.nest-env/bin/python experiments/nest_chunked_equivalence.py
.nest-env/bin/python -m nest_backend.phase4_reproducer

# Dashboard: convert a run to a replay, then open it with the ORDINARY dashboard
.nest-env/bin/python -m nest_backend.replay_adapter --case 01_center_patch_row
.venv/bin/uvicorn backend.api:app                 # then: 'Load Test'

# Optional read-only live NEST view (separate entry point)
.nest-env/bin/uvicorn nest_backend.dashboard_api:app --port 8100
```

Pinned pair: **NEST 3.10.0 + NESTML 8.3.0** (conda-forge; see `environment-nest.yml`).

## Validity envelope — read this before reusing the models

The NESTML event models are an **exact** port of the reference node contracts, but only
under the scoped configuration:

* `leak_rate == 0`, and
* no persistent inhibitory conductance.

That is what degenerates the reference conductance-LIF membrane
(`Current_Implementation_Methodology_Equations.md` §2.1) to a pure integrator, making
event-only accumulation exact rather than an approximation of a trajectory.
`topology.build_reference_engine` **refuses to translate** a configuration with a non-zero
leak rather than silently producing a different model.

## Layout

| Path | Role |
|---|---|
| `models/event_accumulator.nestml` | ordinary-E competitor: accumulate, threshold, reset, refractory, `I_accq` snapshot |
| `models/event_relay.nestml` | `Eor` relay and `I` relay; separate rate-limited and confirmation ports |
| `models/event_coincidence.nestml` | `C`: strict temporal AND with declared basal/apical TTLs |
| `models/plastic_feedforward_synapse.nestml` | dual FE/FES synapse — **incomplete**, see the report |
| `build_models.py` | generate + compile + install the module (idempotent, fingerprinted) |
| `topology.py` | `NetworkSpec` → NEST, delay assignment, timescales, manifest |
| `engine.py` | stimulus construction, run driver |
| `recording.py` | metrics required by the prompt's §10 |
| `phase4_reproducer.py` | the minimal learning-rule feasibility probe |
| `replay_adapter.py` | completed run -> `snn.replay` artifact for the existing player |
| `dashboard_api.py` | read-only live NEST dashboard (separate entry point) |

## Parameter disposition

Every parameter of the reference set (`Current_Implementation_Methodology_Equations.md` §9)
plus the engine defaults it relies on, classified as **kept** / **re-expressed** /
**new** / **dropped**. Produced before any model was written, so reinterpretations are
declared rather than backfilled.

| Parameter | Reference value | Disposition | Notes |
|---|---|---|---|
| `e_threshold` (θ) | 1000 | **kept** | dimensionless charge; `θ_I = θ/3` |
| `dual_fe_fes` | `True` | **kept** | the rule the synapse model implements |
| `dual_fe_e` | 0.001 | **kept** | FE tail floor (`e_floor`) |
| `dual_fe_wte` | 0.001 | **kept** | FES tail floor and the lower weight bound |
| `dual_fe_B` | 5.0 | **kept** | bell sharpness |
| `eta` | 4.0 | **kept** | ordinary-E / `Eor` rate; inert in Phases 0–3 (frozen) |
| `c_eta` | 16.0 | **kept** | `C` basal rate; inert in Phases 0–3 |
| `e_weight_cap_frac` | 0.5 | **kept** | detector ceiling θ/2 |
| `relay_weight_cap_frac` | 1.0 | **kept** | one-afferent ceiling θ |
| `eor_w_init_frac` | 1.0 | **kept** | `Eor` bank frozen at θ |
| `eor_plasticity_enabled` | `False` | **kept** | `Eor` never learns |
| `leak_rate` | 0.0 | **kept, and enforced** | non-zero is *refused*; see the validity envelope |
| `c_feedback_reset` | `True` | **kept** | routed to `event_relay`'s `confirm` port |
| `refractory_steps` | 0 | **re-expressed** | boundaries → `t_ref` in ms. Gates *firing*, not accumulation, matching `gather_exc` + `can_fire`. Inert at the reference 0, but declared rather than left to chance |
| `c_basal_window_steps` | 1 | **re-expressed** | "carries exactly one boundary" → `tau_basal` in ms, defaulted to the presentation interval `D` |
| `input_period` | 0 (auto = L) | **re-expressed** | boundaries → `D` in ms. The graph-derived loop latency is still derived from the graph, now as a sum of assigned delays |
| `SYNAPTIC_DELAY` | 1 boundary | **re-expressed** | → `base_ff` ms per feedforward hop |
| zero-latency apical / WTA / reset | 0 | **re-expressed** | → `h`, the minimum NEST delay. NEST has no zero delay; §5 of the prompt replaces the idealisation with a declared ratio |
| `h` | — | **new** | kernel resolution; the floor on every delay |
| `base_ff` | — | **new** | flat conduction delay of one feedforward hop |
| `S` (spread) | — | **new** | within-volley arrival spread from geometry; required for latency arbitration to exist at all |
| `D` (presentation) | — | **new** | presentation interval |
| `gain` | — | **new** | ms per unit distance, solved so the median per-target spread equals `S` |
| `tau_apical` | — | **new** | apical permission window; the reference keeps permission for the remainder of its boundary |
| `tau_deposit_lock` | — | **new** | NEST-time form of `deposit_committed_this_boundary` |
| `t_lockout` | — | **new** | NEST-time form of `I` emitting at most one WTA volley per boundary |
| `alpha_inh` | 0.6 | **dropped** | inhibitory-conductance retention. The tiled column uses hard resets, not persistent conductance (§7), and this prototype scopes conductance away entirely |
| `switch_trace_threshold` | 0.50 | **dropped** | temporal-AND eligibility trace; superseded by the explicit TTL state machine in `event_coincidence` |
| activity traces / `update_trace` | — | **dropped** | consumed only by consolidation machinery outside this scope |
| `v_sat`, conductance kernels | — | **dropped** | no meaning in an event-only model at `g = 0` |
| analytic `tau ∈ [0,1]` | — | **dropped** | no sub-boundary continuum exists; crossings land on the `h` grid |

## What NEST owns

Clock, delivery order, connection delays, event buffering, spike recording, threading, and
(where compiled in) MPI. There is no Python priority queue, timestamp closure, causal
generation loop, or per-spike learning callback anywhere in this package.

## NESTML gotchas found by building

Recorded so the next person does not rediscover them:

1. **A synapse must be *named* `*synapse`.** NESTML classifies models by name. An otherwise
   valid synapse named `plastic_feedforward` is generated and registered as a **neuron**,
   and `nest.Connect(..., synapse_model=...)` fails with `UnknownSynapseType`.
2. **A synapse needs an `equations` block plus `integrate_odes()`**, even with nothing to
   integrate. Ours integrates an inert `unused_trace`; a test pins that nothing reads it.
3. **`delay_variable` must name a variable NESTML accepts as a delay**, or pairing silently
   falls back to neuron generation.
4. **Single-spike-port models publish no `receptor_types`.** Connections must then omit
   `receptor_type` entirely rather than pass a guessed index.
5. **Receptor names are uppercased** in the generated metadata.
6. **`nest.Install` resolves through the dynamic loader**, so an out-of-tree module must be
   loaded by absolute path — `sys.path` and mid-process `LD_LIBRARY_PATH` edits do nothing.
7. **`V`, `d`, `s` collide with physical unit names** and must be renamed.
8. **NEST's kernel is process-global and not thread-safe.** Calling `ResetKernel()` from a
   worker thread of a process whose main thread already used NEST **segfaults**. The live
   server therefore isolates NEST in a child process.
9. Gotcha 1 has a sharp consequence worth restating: because a misnamed synapse is
   generated as a *neuron*, its post-port wiring never runs, and the resulting
   `__I_accq was not declared` compile error looks like a missing NESTML feature. It is
   not — it is the naming rule biting one step downstream.
