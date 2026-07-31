# Claude Prompt: NEST Event-Driven 3×3 Cortical-Column Prototype

## Directive

Build a small, headless NEST/NESTML backend and validation suite for the repository's
canonical `tiled_cc` topology. NEST must own the clock, spike delivery, connection delays,
event buffering, recording, threading, and MPI execution. Put this project's node state
transitions and learning equations into NEST/NESTML models; do not implement another Python
priority queue or a custom discrete-event scheduler.

This prompt supersedes `docs/NEXT_EVENT_ENGINE_TECHNICAL_SPEC.md` as the current engine
rework direction on the dedicated NEST branches.

The result is an experimental backend, not an immediate replacement for
`backend.simulation.SimulationEngine`. Keep the validated Python engine unchanged as an
oracle while the NEST implementation is characterized.

---

## 1. Intended outcome

Deliver:

1. a reproducible, isolated NEST + NESTML development environment;
2. committed NESTML source models for the event-driven node and plastic-synapse behavior;
3. a translator from the existing canonical `NetworkSpec` to a NEST network;
4. a small Python adapter for constructing, stimulating, running, and recording that
   network through PyNEST;
5. focused node-contract tests and a small 3×3 topology experiment suite;
6. single-process, threaded, and MPI smoke/consistency checks when the installed NEST build
   exposes those capabilities;
7. a report stating exactly which current behaviors survive under native NEST semantics
   and which change.

Do not build dashboard integration, replay integration, a general topology editor bridge,
two-tower composition, or a production backend selector in this task.

---

## 2. What “event-driven” means for this task

The prototype is deliberately not an ODE port.

### 2.0 Why the no-ODE mandate is legal here

This is a scientific licence, not a style preference. Every current run uses
`leak_rate = 0.0`, hence `g_L = 0`, and the tiled family uses hard resets rather than
persistent inhibitory conductance (`Current_Implementation_Methodology_Equations.md`
§2.1, §7, §9). With no leak and no `g_inh`, the membrane is a **pure integrator**: `V` is
exactly the running sum of delivered impulses. Event-only accumulation is therefore
*exact* for the scoped configuration, not an approximation of a trajectory.

Two consequences that must appear in the report:

1. The committed NESTML models are valid **only** at `leak_rate = 0` with no persistent
   inhibitory conductance. Enabling either silently invalidates them. State this in
   `nest_backend/README.md` and assert it at network construction: refuse to translate a
   spec/parameter set with non-zero `leak_rate`.
2. The conductance machinery retained in the Python engine for historical and custom
   graphs is out of scope by §4, and the ban is safe *because* of that scoping — not
   because conductance is unrepresentable in NEST.

### Required

- Incoming spikes are handled as discrete NEST events.
- Neuron and synapse state changes are triggered by spike receipt, spike emission, or an
  explicit controller stimulus.
- Accumulation, threshold checks, reset, coincidence eligibility, refractory checks, and
  learning are algebraic event transitions.
- Elapsed-time rules use the timestamps supplied by NEST.
- NEST owns delivery order, buffering, delay enforcement, thread scheduling, MPI
  communication, and recording.
- Every connection uses a NEST-valid delay. Start with the smallest declared delay equal to
  the configured NEST resolution unless a larger pathway delay is part of the experiment.

### Forbidden

- Do not define membrane ODEs or call `integrate_odes()`.
- Do not use GSL, RK4, exact-integration propagators, continuous input ports, conductance
  kernels, or a sampled approximation of the old LIF equation.
- Do not scan all neurons each step in Python.
- Do not create a Python heap, timestamp queue, timestamp closure, causal-generation loop,
  or custom MPI exchange.
- Do not reimplement NEST scheduling to recover the old engine's zero-delay callbacks.
- Do not label NEST as a globally pure DES. NEST's nodes and communication retain NEST's
  native resolution/minimum-delay execution contract even though this model performs its
  scientific state transitions on spike events.

An empty generated `update` block may be retained only if the NESTML target requires one;
it must perform no scientific integration or periodic state transition.

If the pinned NESTML/NEST target cannot emit a neuron spike directly from the required
event-handler path, first prove that limitation with the smallest possible model. Then use
the smallest supported NEST extension mechanism that keeps the transition inside the NEST
node. Do not move per-event neural computation back into Python. Record the reason and the
generated/C++ boundary in the report.

Official semantic references:

- NEST simulation behavior:
  https://nest-simulator.readthedocs.io/en/stable/nest_behavior/running_simulations.html
- NEST precise spike timing:
  https://nest-simulator.readthedocs.io/en/stable/neurons/simulations_with_precise_spike_times.html
- NESTML language concepts and `onReceive`:
  https://nestml.readthedocs.io/en/latest/nestml_language/nestml_language_concepts.html
- NESTML NEST target:
  https://nestml.readthedocs.io/en/latest/running/running_nest.html

---

## 3. Authority and scientific posture

Use this authority order:

1. `backend/network_spec.py` defines the graph and metadata.
2. `snn/neurons.py` and `backend/simulation.py` define the current node and learning rules.
3. `Current_Implementation_Methodology_Equations.md` documents those rules.
4. Existing focused tests define established invariants.
5. NEST defines timing, delivery, minimum-delay, and parallel execution semantics for the
   new prototype.

This is not a trace-equivalence migration. The existing engine has analytic sub-boundary
crossings, zero-latency apical/reset callbacks, boundary packets, and engine-level hard
WTA. Standard NEST spike delivery has a positive minimum delay and its own same-time event
handling. Do not hide these differences with epsilon-time Python callbacks or custom
scheduler logic.

The comparison must distinguish:

- **structural parity:** the same intended nodes, edges, roles, weights, and stimuli;
- **equation parity:** the same algebraic accumulation and weight-update formulas where
  expressible in NESTML;
- **semantic difference:** behavior changed by native NEST delay, buffering, or event
  ordering;
- **implementation gap:** behavior that cannot be represented without extending NEST
  beyond this task.

Negative or changed results are valid.

---

## 4. Exact topology scope

Port only the canonical default `tiled_cc` returned by:

```python
backend.network_spec.tiled_cc_spec(cc_e_count=8)
```

This is the repository's 3×3 cortical-column topology:

- one 9×9 RGC input surface;
- nine disjoint 3×3 input patches;
- a 3×3 grid of nine L1 cortical columns, one per input patch;
- one L2 cortical column receiving the nine L1 outputs;
- eight ordinary E competitors per column at the default;
- the existing Eor, C, and I roles and the existing edge kinds;
- 191 nodes and 1052 directed edges at the default.

Do not port `tiled_cc_l1_4`, `tiled_cc_direct_identity`,
`tiled_cc_double_eor`, `two_tower_composition`, saved custom graphs, or any rejected
feature-gated topology.

Use the existing node and edge IDs as external labels. Maintain a manifest mapping each
repository ID and edge ID to its NEST node/connection representation. Tests and reports
must select components through topology metadata, not by parsing display names.

---

## 5. Native NEST timing policy: timescale separation

The prototype does not emulate zero-delay callbacks. It encodes the **physical claim those
callbacks stand in for**: local lateral inhibition closes far faster than a cell integrates
its feedforward volley.

In the Python engine that separation is infinite by construction — the WTA loop costs zero
time while a feedforward hop costs a whole boundary. NEST has no zero delay; the floor is
the kernel resolution `h` and every delay is an integer multiple of it. So express the
separation as an explicit, recorded **ratio** rather than as an idealization. This is a
gain in biological fidelity, not a concession: a real cortical WTA loop is fast, not
instantaneous.

### 5.1 The three declared timescales

| Symbol | Meaning | Required relation |
|---|---|---|
| `h` | kernel resolution; minimum delay any connection may take | — |
| `L_wta = 2h` | `E → I → E` lateral inhibition loop, one `h` per hop | minimum possible |
| `S` | afferent arrival spread within one feedforward volley (§5.2) | `S >> L_wta` |
| `D` | presentation interval, volley to volley | `D >> S` |

Start from `L_wta : S : D ≈ 1 : 10 : 100` and record all four quantities in every run.
Physiological anchor, to be stated in the report: PV+ basket-cell lateral inhibition runs
~1–3 ms while feedforward integration windows run ~10–30 ms, so a one-order separation
between loop and integration and another between integration and presentation is
measured biology, not a tuning convenience. Do not claim calibrated biological wall-clock
units — only the ratio is claimed.

### 5.2 Fast inhibition alone does not recover WTA

State this explicitly before implementing, because it is the failure mode that would
otherwise be discovered in Phase 3.

The Python engine's winner is selected by pure first-spike latency at continuous `τ`
because the drive packet is frozen as a **constant current over the boundary**, so
`V(τ) = V₀ + I_exc·τ` and the crossing time is `(θ − V₀)/I_exc`. Ordering is by drive
magnitude.

If NEST delivers a whole volley as one simultaneous impulse on delta synapses, every
competitor jumps at once and all supra-threshold competitors cross on the **same**
timestep. There is no latency ordering for inhibition to arbitrate, however fast the loop
is. Shrinking `h` alone does not fix this — it shrinks the loop and the tie together.

Recovering arbitration therefore requires a second lever: **temporally dispersed afferent
arrival**, so that competitors genuinely cross at different timesteps.

### 5.3 Delays from the existing geometry

Every node in the canonical spec already carries a 3D `pos`, and per-edge Euclidean
distances are already computed at build time (`backend/simulation.py` `_distance_factors`,
`ff_dists`, `basal_dist`). Reuse that geometry for **conduction delay**:

```text
delay(e) = base_hop(kind) + round( gain * (d(e) - d_min(kind)) / h ) * h
```

- `gain` is an explicit declared parameter chosen so that the resulting within-volley
  spread equals the target `S`. Do not derive it blindly from raw distance: the layout
  separates layers on `z`, so inter-layer distance is dominated by the layer gap and the
  informative within-patch variation is the smaller term on top of it. Record the realized
  `S` per pathway.
- This does **not** violate the standing rule that distance is a learning-rate multiplier
  only and never scales delivered charge (`Current_Implementation_Methodology_Equations.md`
  §4.3). Delay changes *when* charge arrives, never *how much*. It is nevertheless a **new
  semantic** and must be declared as one in the difference table.
- Optionally add small seeded per-synapse delay jitter to decorrelate the arrival order
  across competitors. If used, it must be drawn from the run seed and be exactly
  reproducible; record whether it was enabled.

Baseline delay assignment for canonical `tiled_cc`:

| Pathway | Python semantics | NEST delay |
|---|---|---|
| RGC → L1 ordinary E (feedforward) | delay 1 boundary | `base_ff + dispersion` |
| ordinary E → `Eor` | delay 1 boundary | `base_ff` |
| `Eor` → parent E (feedforward) | delay 1 boundary | `base_ff + dispersion` |
| `Eor` → local `C` basal | delay 1 boundary | `base_ff` |
| parent E → child `C` apical | zero latency, same `τ` | `h` |
| ordinary E → `I` (WTA recruit) | zero latency | `h` |
| `I` → ordinary E bank (hard reset) | zero latency, same `τ` | `h` |
| `C` → `I` confirmation reset | start of `t+1`, discards packet | `h` |

**CORRECTED after Phase 3 measurement.** This section previously speculated a "parity win":
that because the `C → I` confirmation reset lands in the inter-volley silence under
`D >> L_wta`, it would achieve the same functional outcome as the reference without any
packet-freezing machinery. It was flagged as something to confirm rather than assert. It was
confirmed, and it is **false**.

The reference confirmation reset lands at the *start* of the next boundary, **after that
boundary's drive packet has been frozen**, so the entire packet is discarded. NEST has no
frozen packet: charge arrives as a stream of events, and a reset at time `T` erases only what
accumulated *before* `T`. Landing in the inter-volley silence is exactly what makes it unable
to discard a drive packet — there is nothing there yet to discard. Suppression is real but
partial, and the reference's exact fire/silent alternation cannot arise. See
`docs/NEST_EVENT_DRIVEN_3X3_REPORT.md` §5.

Keep the confirmation pathway on its own non-rate-limited port regardless: the reference
requires it not be swallowed by the once-per-boundary WTA guard, and routing it through the
shared lateral port makes feedback-on and feedback-off runs bit-identical while the lockout
silently absorbs every confirmation.

### 5.4 What this recovers, and what it does not

Be precise in the report. Dispersed arrival recovers a genuine, weight-driven first-spike
race, but it arbitrates by **weighted prefix-sum over the arrival order**, not by total
frozen drive:

```text
winner = argmin_j  min{ k : Σ_{i ≤ k} w_{j, π(i)} ≥ θ }
```

for the common arrival order `π` over the patch's afferents. This coincides with the
Python engine's total-drive ordering when weight mass is roughly uniform across active
afferents — which is what `θ/4` initialization and the `θ/2` FE/FES cap produce for a
matured owner on its own pattern — and can diverge for partially matured competitors.

Because delay follows geometry, each column also acquires a fixed scan order over its 3×3
patch, so `row 1` and `col 1` present systematically different arrival profiles. Suite
cases 1 and 2 (§10) contrast exactly these, so the effect is observable by construction.

Required measurements, reported as first-class results rather than assumptions:

1. agreement rate between the NEST latency winner and the Python engine's total-drive
   winner over the suite, split by maturity (frozen init vs. trained weights);
2. winner multiplicity per presentation window (the direct WTA-loss observable);
3. sensitivity of both to `gain` / realized `S`, and to jitter on/off;
4. any row/column asymmetry attributable to scan order.

If single-winner behavior does not survive, that is a valid and publishable result. Report
it as the headline finding, not as a footnote.

### 5.5 Ratio and resolution sweep

Run two sweeps, each with a declared hypothesis, over at least three values the installed
NEST accepts:

- **`h` sweep at fixed ratio** `L_wta : S : D`. Hypothesis: outcomes are invariant, because
  only the ratio is physical. A dependence on absolute `h` at fixed ratio indicates a
  discretization artifact and must be explained.
- **`S / L_wta` ratio sweep at fixed `h`**. Hypothesis: winner multiplicity falls toward 1
  as the ratio grows, and approaches the number of supra-threshold competitors as it
  approaches 1. This is a **convergence characterization** of how a positive-min-delay
  engine approximates continuous-`τ` arbitration.

Neither sweep may be used to select the one setting that reproduces a preferred outcome.
Report the full curve, including settings where WTA fails.

### 5.6 Remaining rules

1. Select one explicit resolution `h`; record it, plus `L_wta`, realized `S`, and `D`, in
   every run.
2. All ordinary spike connections use delays accepted by NEST and no delay smaller than
   `h`.
3. One delay hop per graph edge. Do not collapse multiple repository edges into a Python
   callback.
4. Use NEST's actual received timestamps in coincidence, refractory, and learning logic.
5. Use NEST recorders as the authoritative spike record.
6. Same-time input handling must be implemented only with supported NEST/NESTML port and
   handler semantics. Local `onReceive` priorities are allowed when declared as part of a
   node model.
7. Do not add global competition arbitration. WTA must arise from the translated
   excitatory/inhibitory circuit under NEST delivery plus the declared timescale
   separation — never from a Python tie-break, a sorted scan, or a scheduler hook.
8. Do not manufacture legacy `tau`, boundary IDs, causal generations, or zero-delay
   closure.
9. Precise-spike-timing (`*_ps`) models do **not** solve this. They give off-grid spike
   times, but delays remain `≥ min_delay` and the min-delay communication contract still
   batches, so they buy back neither the zero-latency reset nor the zero-latency apical.
   Do not spend a phase discovering this.

---

## 6. Node and synapse contracts

Implement the smallest set of models needed by canonical `tiled_cc`.

### 6.1 Event accumulator / ordinary E

State should include only what is used by the event model:

- accumulated charge;
- threshold;
- reset value;
- last spike time / refractory-until time;
- spike count and optional diagnostic counters.

On an accepted feedforward spike:

1. reject or ignore it only if the declared refractory policy says so;
2. add the received synaptic contribution once;
3. test the threshold;
4. on crossing, emit one spike, snapshot the pre-reset accumulated value needed by learning,
   and reset.

There is no continuous voltage trajectory and no leak ODE. Accumulated charge changes only
on arrival, so a competitor's crossing timestep is set by *when its afferents arrive*
(§5.2–5.3), not by a ramp. If a retained model parameter has no meaning in this event-only
model, omit it and list it in the parameter disposition table (§6.6).

### 6.2 Fixed Eor relay

Represent the current non-plastic Eor role as an event relay using NEST delivery:

- one accepted upstream event produces one output spike according to its declared
  threshold/weight contract;
- it does not learn;
- it does not use a Python callback.

### 6.3 Coincidence C

Use separate basal and apical spike input ports.

State must minimally represent:

- the most recent valid basal receipt or per-source basal eligibility needed by the
  canonical graph;
- its expiry time under an explicitly declared NEST-time TTL;
- whether the current apical event finds eligible basal evidence;
- deposit and spike counters;
- the causal basal synapse required by learning.

A basal event alone does not deposit somatic charge. An apical event alone does not
deposit. Valid basal plus apical coincidence deposits once according to the declared local
idempotence rule. Use NESTML handler priorities only to make same-time basal/apical behavior
explicit; do not impose a global timestamp closure.

**Known landmine — settle the eligibility phase deliberately.** In the Python engine,
apical permission is zero-latency (same `τ`) while basal eligibility carries for exactly
one boundary and is settled *after* the event loop, precisely so a basal event arriving
before this boundary's apical still gets its chance to coincide. This repository has
already shipped a bug in exactly this place: a boundary-start eligibility resolve expired
the carry before the in-loop apical could consume it, leaving the carry path dead while
unit tests passed.

Under §5.3 apical arrives at `h` rather than at zero latency, so the ordering is now
explicit NEST time and must be handled as such:

- declare the eligibility TTL in NEST time, sized against `S` and `D` — not as "one
  boundary";
- declare, in the model, what happens when basal and apical arrive on the **same**
  timestep, using `onReceive` priorities;
- guarantee that basal-then-apical **and** apical-then-basal within the TTL both deposit
  exactly once, and that neither double-deposits;
- expose a deposit counter and an eligibility-expiry counter so a dead carry path is
  visible in recordings rather than inferred.

Ordering-specific tests are mandatory (§11); "valid vs. expired" coverage alone is what
missed this the first time.

### 6.4 Inhibitory relay and reset

The I role is an event relay. Translate the current reset pathway into a NEST-delivered
reset/inhibitory event with a valid positive delay.

On reset receipt, the target event accumulator clears the state declared by the prototype.
Document whether an already emitted same-time spike remains observable; do not attempt
rollback.

### 6.5 Plastic synapses

Move weights from the Python engine into NEST connection/synapse state.

- Implement feedforward and basal learning as NESTML plastic-synapse models where the
  required pre/post state is available.
- Preserve the current algebraic update equations, bounds, and deterministic
  initialization; do not substitute built-in STDP.
- Use NESTML neuron/synapse co-generation or supported post-state access when required.
- An in-flight event uses the value NEST assigns under its native send/delivery semantics.
- Python may initialize and inspect weights between `Simulate` calls, but it may not perform
  per-spike weight updates.

Begin all topology experiments with frozen learning. Enable plasticity only after the
frozen event path passes.

### 6.6 Parameter disposition table (Phase 0 deliverable)

Before writing any model, produce a committed table covering every parameter in the
reference set (`Current_Implementation_Methodology_Equations.md` §9) plus the engine
defaults it relies on. Each row is classified as exactly one of:

- **kept** — same meaning, same value;
- **re-expressed** — same meaning, new units (boundaries → ms, integer delay → `h`
  multiples). Give the conversion;
- **new** — introduced by this port (`h`, `gain`, `S`, `D`, delay jitter seed). Give the
  justification;
- **dropped** — no meaning in the event-only model. Give the reason.

At minimum this must resolve `leak_rate`, `refractory_steps`, `input_period`, `alpha_inh`,
`switch_trace_threshold`, `e_threshold`, `eta`, `c_eta`, `dual_fe_B`, `dual_fe_e`,
`dual_fe_wte`, `e_weight_cap_frac`, `relay_weight_cap_frac`, `eor_w_init_frac`,
`c_basal_window_steps`, and `c_feedback_reset`.

Producing this up front, rather than backfilling a semantic-difference table at the end, is
what prevents silent drift. The end-of-task difference table (§3) then reports only what
*behavior* changed, not what parameters were quietly reinterpreted.

**Do not pre-emptively declare this blocked.** Most of the dual FE/FES rule is expressible
in NESTML neuron/synapse co-generation; assume feasibility and disprove it specifically:

| Rule element | Expected mechanism | Risk |
|---|---|---|
| `I_accq` (pre-reset accumulated causal charge) | post-neuron state variable snapshotted at spike, read by co-generated synapses | low |
| `s_i = ±1` participation | per-synapse boolean set on pre-receipt, cleared at post-spike; post-spike-triggered updates reach silent synapses, so the `−1` branch is reachable | low |
| `φ_i`, per-target `d_ref` | build-time reduction in Python, baked in as a per-synapse constant (explicitly allowed above) | low |
| `w_te` floor, `w_cap` ceiling | per-synapse bounds in the update | low |
| C basal "only the causal source's weight moves" | same participation flag; non-causal basal synapses take no update | low |
| every afferent sees the **same** pre-update `FE` and its **own** pre-update `w_i` | depends on NESTML's post-spike update ordering across a target's synapses | **this is the real unknown** |

Build the minimal reproducer for the last row first — one post cell, two afferents, one
participating and one not — and verify both the shared `FE` and the independent per-synapse
`FES` numerically against the declared equation. Only if that specific guarantee cannot be
obtained does the phase stop as a documented blocked result, with a field-by-field gap
analysis. Do not approximate silently and do not write a Python learning callback.

---

## 7. Repository layout

Prefer this bounded structure:

```text
nest_backend/
    __init__.py
    README.md
    build_models.py
    engine.py
    topology.py
    recording.py
    models/
        event_accumulator.nestml
        event_coincidence.nestml
        event_relay.nestml
        event_reset_target.nestml       # only if a separate target is required
        plastic_feedforward.nestml
        plastic_basal.nestml
experiments/
    nest_3x3_suite.py
tests/
    nest/
        test_nest_environment.py
        test_nest_node_contracts.py
        test_nest_topology_translation.py
        test_nest_3x3_suite.py
scripts/
    setup_nest_env.sh
    verify_nest_install.py
environment-nest.yml
requirements-nest.txt
```

Adjust filenames if NESTML code-generation constraints require it, but keep the backend
isolated from the current production engine.

`experiments/nest_3x3_suite.py` lands in an existing tree with its own headless cluster
runner and read-only server conventions. Decide and document one of: (a) it plugs into that
runner and reuses its artifact/manifest layout, or (b) it stands alone because the NEST
environment is a separate interpreter. Option (b) is expected and acceptable — the NEST
environment is deliberately isolated per §8.1 — but say so rather than leaving it
ambiguous.

`nest_backend/topology.py` must additionally own the delay-assignment policy of §5.3 and
emit the realized `h`, `L_wta`, `S`, `D` and per-pathway delay histogram into the run
manifest.

Generated C++, shared libraries, caches, logs, and run artifacts must be gitignored.
Commit NESTML source, build orchestration, dependency manifests, tests, and reports.

---

## 8. Dependency and build setup

Dependency setup is part of the deliverable.

### 8.1 Isolation

Do not install into the repository's existing `.venv` or the system Python. Create an
isolated environment dedicated to NEST.

The required path is a local micromamba/mamba/conda environment using conda-forge for NEST
and the compiler toolchain, with NESTML installed in the same environment. This is
effectively mandatory, not merely preferred: NESTML must **compile and `nest.Install` a
generated C++ module**, which needs NEST's headers, CMake config and `nest-config` — the
released `nest-simulator` pip wheels have historically not shipped a usable set, and are
typically built without MPI, which would also make Phase 5 unreachable.

A pip-only attempt is therefore permitted only as a documented negative result: try it,
expect it to fail at model compilation or `nest.Install`, and record the exact failure in
the report. Do not present it as a supported fallback if it did not build.

Official installation references:

- NEST installation:
  https://nest-simulator.readthedocs.io/en/stable/installation/index.html
- NEST pip installation:
  https://nest-simulator.readthedocs.io/en/main/installation/pip_install.html
- NESTML installation:
  https://nestml.readthedocs.io/en/latest/installation.html

### 8.2 Version selection

Do not guess a version pair in advance.

1. Select released NEST and NESTML versions documented as mutually compatible.
2. Build and run the minimal event-model smoke test.
3. Record exact resolved versions and platform information.
4. Pin that verified pair in `environment-nest.yml` and `requirements-nest.txt`.
5. Explain which manifest is authoritative and which is the fallback.

Do not depend on moving `main` branches or unpinned Git URLs.

### 8.3 Setup script

`scripts/setup_nest_env.sh` must:

- be non-interactive and idempotent;
- never invoke `sudo`;
- detect a supported environment manager;
- create only the dedicated local environment;
- install the pinned dependency set;
- build the committed NESTML models;
- run `scripts/verify_nest_install.py`;
- print the exact command needed to activate/use the environment.

If prerequisites are unavailable, exit with a concise diagnosis and the official manual
installation alternative. Do not modify shell startup files.

### 8.4 Verification

The verifier must fail unless it can:

1. import `nest`;
2. print NEST build/version information;
3. import the NESTML toolchain;
4. generate and compile the smallest committed event model;
5. load the generated module with `nest.Install`;
6. deliver at least one spike through it;
7. record the resulting output spike;
8. report thread and MPI capabilities without assuming either exists.

The normal Python test suite may skip `tests/nest/` with an explicit reason when the NEST
environment is absent. A dedicated NEST test invocation must treat absence as failure.

---

## 9. Phased implementation and gates

Proceed in order. Do not continue past a failed gate by weakening it.

### Phase 0 — freeze scope and establish the environment

- Run the existing non-NEST test suite and record it green. This baseline is a gate, not
  only a completion criterion (§13.8) — capture it before touching anything.
- Capture the canonical `tiled_cc` topology manifest and current reference results.
- Produce the §6.6 parameter disposition table.
- Add the isolated dependency manifests, setup script, verifier, and gitignore rules.
- Record the selected NEST/NESTML versions and license/build facts.
- Smoke-test the single highest-risk unknown first: that the pinned NESTML target can
  `emit_spike()` from an `onReceive` handler. Everything downstream assumes it. If it
  cannot, §12 applies immediately and no further code is written.

Gate: the baseline suite is green and recorded; the verifier builds, loads, and executes
one event model in a fresh environment; spike emission from an event handler is
demonstrated.

### Phase 1 — event-node microcontracts

- Implement accumulator, relay, reset, and coincidence event handlers.
- Keep learning frozen.
- Test one event, multiple same-time events, below/at/above threshold, reset, refractory
  expiry, basal-only, apical-only, in-window coincidence, expired coincidence, same-time
  basal/apical receipt, **and both basal-then-apical and apical-then-basal orderings
  within the TTL** (§6.3).
- Demonstrate the two-cell WTA microcontract in isolation before touching the full
  topology: two competitors, dispersed arrival, `E→I→E` at `2h`, and show that the later
  crosser is reset before it reaches `θ` at `S/L_wta = 10` and is *not* at `S/L_wta = 1`.

Gate: every transition is caused by a recorded NEST event; no scientific ODE/update loop
exists; the eligibility carry path is provably live (deposit counter non-zero on the
basal-then-apical ordering); the isolated WTA microcontract shows the expected dependence
on `S/L_wta`.

### Phase 2 — canonical topology translation

- Translate `tiled_cc_spec(cc_e_count=8)` without copying a second hand-written topology.
- Validate node roles, edge kinds, weights, delays, and ID mappings.
- Implement the §5.3 delay-assignment policy from the spec's existing `pos` geometry, and
  record the realized per-pathway delay histogram and `S`.
- Use NEST spike generators for the 81 RGC sources and NEST recorders for model outputs.

Gate: structural counts and role/edge-kind inventories match the canonical `NetworkSpec`
(191 nodes, 1052 directed edges at the default); every delay is a valid multiple of `h`
and no smaller than `h`; the declared `L_wta : S : D` relation holds as measured, not as
intended; and one center-patch stimulus traverses the expected RGC → L1 → Eor → L2 path.

### Phase 3 — frozen-learning 3×3 suite

Run the bounded suite in Section 10 and publish artifacts.

Gate: all runs complete deterministically for the declared seed/kernel settings; all
structural invariants pass; behavioral differences are classified rather than hidden.

### Phase 4 — plastic synapses

- Implement ordinary-E feedforward learning.
- Implement C basal learning.
- Re-run node tests before enabling learning on the full topology.

Gate: isolated one-synapse updates reconstruct from the recorded inputs and declared
equation; frozen mode remains unchanged; weights update inside NEST rather than Python.

If exact learning is blocked by a NESTML interface limitation, this phase may finish as a
documented blocked feasibility result while Phases 0–3 remain a successful prototype. Do
not call the learning port complete.

### Phase 5 — threads and MPI characterization

- Compare one thread with at least two threads if supported.
- Compare one MPI rank with at least two ranks if the installed build supports MPI.
- Use identical topology, stimuli, seeds, resolution, and model parameters.
- Record spike multisets, per-column counts, final weights, runtime, and capability flags.

Gate: explain any difference. Do not promise speedup from this small graph; this phase tests
correctness and operational readiness.

---

## 10. Small 3×3 experiment suite

Keep the suite small and deterministic:

1. **Center-patch row:** `row 1` in patch `(1, 1)`.
2. **Center-patch column:** `col 1` in patch `(1, 1)`.
3. **Two independent patches:** `(0, 0)` with `row 1` and `(2, 2)` with `col 1`.
4. **All nine patches, same feature:** `row 1` in every patch.
5. **Pattern switch:** center patch `row 1`, then `col 1`, without rebuilding the network.
6. **Feedback control:** repeat the center-patch row run with the translated feedback/reset
   path disabled.
7. **Feedback cadence law:** drive the center patch at presentation interval `D` set to the
   graph-derived loop latency and check for strict fire/silent alternation.
8. **Timescale-separation sweep:** center-patch row at `S/L_wta ∈ {1, 3, 10, 30}`,
   recording winner multiplicity at each point.

Case 7 is the sharpest available parity test and should be treated as the headline
positive result. The Python engine makes a measured, falsifiable prediction
(`docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`, equations doc §8): with `h_hops` feedforward
hops from a child ordinary E to a parent ordinary E, loop latency `L = h_hops + 1`, which is
`L = 3` for `tiled_cc`; period is `2L` at a volley every boundary; suppression bites iff
`L mod input_period == 0`; and setting the presentation period equal to `L` forces strict
`1010` alternation (12/12 seeds at `L = 2, 3, 4`). This is a claim about **relative timing
only**, which is exactly what NEST owns natively, so it should survive a faithful port even
though spike times will not match. Derive `L` from the translated graph, never from a preset
name. Failure here indicates a translation error, not a native-semantic difference — treat
it as such.

For each case, record:

- NEST and NESTML versions;
- resolution `h`, `L_wta`, realized `S`, presentation interval `D`, and the realized
  `L_wta : S : D` ratio;
- minimum/maximum delay, per-pathway delay histogram, delay-jitter setting and seed;
- thread count, MPI rank count, and seeds;
- topology/model hashes;
- input spike schedule;
- spike times and sender IDs by role and column;
- per-column ordinary-E, Eor, C, and I counts;
- **winner multiplicity per presentation window** — the primary WTA-loss observable;
- **`Eor` input multiplicity per firing** — the direct corroborator: `Eor` is frozen at `θ`
  and contractually fires on **one** afferent, so a multiplicity above 1 is proof that
  single-winner WTA was lost even though `Eor` still fires;
- agreement rate with the Python engine's total-drive winner, per §5.4;
- C basal/apical receipts, valid coincidences, deposits, and eligibility expiries, split by
  basal-then-apical vs. apical-then-basal arrival order;
- reset events, split into lateral WTA and `C → I` confirmation;
- initial and final plastic weights when learning is enabled;
- wall-clock runtime.

Use timestamped, gitignored directories under:

```text
experiments/runs/nest_3x3/
```

Write a compact committed summary to:

```text
docs/NEST_EVENT_DRIVEN_3X3_REPORT.md
```

Do not require exact spike-time or boundary-trace agreement with the Python engine.
Compare invariant outcomes and explain differences caused by native NEST timing.

---

## 11. Required tests

At minimum:

- dependency/version verifier;
- generated-model build and load;
- no ODE/integration construct in committed event models;
- one incoming event is counted once;
- same-time multiplicity behavior is measured and stable;
- threshold/reset and refractory contract;
- relay propagation with NEST-valid delay;
- reset cannot erase an already recorded spike;
- basal-only/apical-only silence;
- valid and expired basal/apical coincidence;
- **basal-then-apical within TTL deposits exactly once** (the carry path is live);
- **apical-then-basal within TTL deposits exactly once** (no double deposit);
- eligibility expiry counter increments when and only when the TTL lapses unused;
- **construction refuses a non-zero `leak_rate` or a persistent inhibitory conductance**
  (§2.0);
- every assigned delay is a multiple of `h` and no smaller than `h`;
- the declared `L_wta : S : D` relation holds as measured on the built network;
- **two-competitor WTA microcontract**: later crosser is reset before threshold at
  `S/L_wta = 10`, and both fire at `S/L_wta = 1`;
- `Eor` input multiplicity is 1 whenever single-winner WTA holds;
- feedback cadence law: presentation period `= L` yields strict `1010` alternation, and
  period `= L+1` yields none;
- frozen weights never change;
- **isolated two-afferent plastic update**: participating and non-participating afferents
  of one post cell share the same `FE` and take their own pre-update `FES(w_i)`;
- isolated plastic update matches the declared equation;
- topology node count, edge count, role count, and external ID mapping;
- deterministic construction for a fixed seed;
- center-patch isolation;
- two-patch locality;
- feedback-on/off comparison;
- one-thread versus supported parallel modes.

Run the existing non-NEST regression suite as well. The prototype must not change current
engine results.

---

## 12. Stop conditions

Stop the affected phase and report instead of improvising if:

- no released compatible NEST/NESTML pair can build in the supported environment;
- the required event emission path cannot be expressed by NESTML;
- NESTML cannot guarantee that a post-spike update gives every afferent the same
  pre-update `FE` and its own pre-update `w_i` (§6.5, the one identified real risk);
- exact learning requires unavailable pre/post state or per-afferent provenance;
- the only proposed solution is a Python per-event callback or custom scheduler;
- topology translation would require changing the canonical `NetworkSpec`;
- a NEST limitation would require silently replacing a scientific equation;
- threaded or MPI execution changes results and the cause is not understood.

Explicitly **not** a stop condition: loss of single-winner WTA, or disagreement between the
prefix-sum latency winner and the Python engine's total-drive winner. Those are the
results the timescale-separation experiment exists to measure. Report them; do not tune
`gain`, `S`, or `h` until they go away.

A minimal NEST C++ extension is permissible only after a minimal NESTML reproducer proves
the need. Keep it limited to node/synapse behavior; modifying or forking the NEST scheduler
is outside scope.

---

## 13. Completion criteria

The task is complete when:

1. a fresh documented environment installs the pinned dependencies;
2. the verifier generates, builds, loads, and exercises the event models;
3. no model uses ODE integration for scientific behavior, and construction refuses the
   configurations under which that would be invalid (§2.0);
4. canonical `tiled_cc` is translated from the existing spec, with delays derived from the
   spec's own geometry rather than hand-assigned;
5. NEST owns event delivery, time, delays, recording, threading, and MPI where available;
6. the declared `L_wta : S : D` timescale separation is implemented, measured, and swept —
   not assumed;
7. the bounded 3×3 suite produces reproducible artifacts;
8. frozen node behavior and available learning equations are tested inside NEST;
9. the existing engine and test suite remain unchanged and green, against a baseline
   recorded in Phase 0;
10. the final report separates parity, native-semantic differences, and blocked features,
    and states plainly whether single-winner WTA survived;
11. documentation does not claim pure DES, exact legacy equivalence, calibrated biological
    wall-clock units, or MPI speedup that was not measured. Timescale *ratios* may be
    claimed as physiologically motivated; absolute millisecond values may not.

Do not commit or push unless explicitly requested.

---

## Claude kickoff

```text
Read and execute the complete specification in:
prompts/Claude_NEST_Event_Driven_3x3_Engine_Prompt.md

Work only on the canonical default tiled_cc topology. Build an isolated NEST/NESTML
environment (conda-forge, not pip), use event-handler-based node and synapse transitions
with no ODE integration, translate the existing NetworkSpec rather than copying the graph,
and let NEST own timing, delivery, delays, recording, threads, and MPI. Keep the current
Python engine unchanged as an oracle.

NEST has no zero delay, so do not try to recreate the zero-latency WTA reset. Instead
implement the timescale separation it stands in for (Section 5): lateral inhibition at the
minimum delay, feedforward arrival dispersed by the spec's own geometry, and a
presentation interval far longer than both. Fast inhibition alone does NOT recover
single-winner WTA -- without dispersed arrival there is no latency ordering to arbitrate.
Measure whether single-winner behavior survives; do not tune until it does.

Implement in phases and stop at a failed gate. Never add a Python event scheduler or
per-spike learning callback to emulate missing NEST behavior. Report native NEST semantic
differences honestly. Do not commit or push.
```
