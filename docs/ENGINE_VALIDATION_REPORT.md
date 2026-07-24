# Hybrid Event-Resolved Engine — Validation Report

Scope: validate whether the current simulation engine is (1) internally correct for the
equations and event semantics it actually implements, and (2) a defensible approximation
of the corresponding continuous-time, discrete-event system on the small circuits where an
independent numerical reference is practical. This was a **validation and characterization**
task: no learning rule, threshold, weight, inhibition strength, topology, delay, or
tie-break was changed. Nothing was committed or pushed.

Artifacts and code:

- `experiments/engine_validation.py` — headless harness (RK4 oracle, priority-queue tiny
  reference, all phase runners, artifact writer). `--seed` selectable; prints the run path.
- `tests/test_engine_validation.py` — 20 focused deterministic tests.
- `experiments/runs/engine_validation/<timestamp>-engine_validation-<hash>/` — per-run
  bundle (`summary.json`, `numeric_convergence.csv`, `event_trace.jsonl`, `invariants.json`,
  `README.md`); gitignored via the existing `experiments/runs/` rule.

Reference run (seed 1): `20260723-185239-engine_validation-10c794e6`.

---

## VERDICT

```
Analytic segment solver:        VALIDATED
Scheduler causality:            VALIDATED
Same-time concurrency:          ORDER-SENSITIVE BY EXPLICIT RULE
Event conservation:             VALIDATED (reconstructed multiset ledger; see limitation)
Boundary-refinement evidence:   UNSUPPORTED (segment partition invariance VALIDATED)
Dual-FE causal state capture:   VALIDATED
Overall characterization:       VALIDATED WITH DECLARED APPROXIMATION
```

The engine is best described exactly as the prompt hypothesized: **a deterministic,
boundary-synchronous hybrid simulator with analytic within-boundary membrane evolution and
event resolution.** Within one outer boundary it is a faithful continuous-time analytic
solver over frozen drive; *across* boundaries it is discrete — delivery, decay, refractory,
learning exposure and stimulus scheduling are all indexed by the integer boundary. It is
**not** a globally continuous-time, pure discrete-event simulator, and this report does not
claim it to be.

None of the findings below contradict the repository's own declared semantics, so there is
no implementation bug in the audited timing paths. The only non-VALIDATED items are declared
approximations (the outer-boundary delivery delay and the one-emission-per-relay rule) and
the explicitly unsupported network boundary-refinement test.

---

## Phase 1 — Semantic audit

Continuous sub-boundary time `tau ∈ [0, 1]` is used only for: analytic crossings, E/C
firing, coincidence deposits, apical delivery, inhibitory-relay firing, and hard resets.
Everything else uses only the **integer boundary index**: ordinary/pretrained/basal
delivery, sensory input gating, conductance decay, refractory decay, activity-trace update,
basal-eligibility settle, and all learning exposure counting.

| mechanism | production source | logical timestamp | same-boundary / delayed | >1 per boundary? | reads | invalidates | declared rule or accidental? |
|---|---|---|---|---|---|---|---|
| ordinary feedforward delivery | `simulation._emit_event_outputs` → `_sched_exc` → next-boundary `gather_exc` | integer boundary `t+1` | **delayed 1 boundary** | accumulates (additive) | pre-update `acc_weights[widx]` | — (adds to `pending_exc`) | declared: `SYNAPTIC_DELAY = 1` |
| sensory / RG input | `_event_step` input loop → `SourceNeuron.present` | integer boundary `t` | same-boundary (delay 0) | once (`input_period` gate) | `input_vec`, `input_period` | — | declared |
| pretrained packet | `_emit_event_outputs` (`_pretrained_out`) | integer boundary `t+1` | delayed 1 boundary | additive | fixed `q_pretrained` | — | declared fixed magnitude |
| basal dendritic event | `_emit_event_outputs` (`_basal_out`) → `_basal_next` | integer boundary `t+1` | delayed 1 boundary | additive | — | sets basal receipt | declared 1-boundary delay |
| analytic crossing | `BoundaryEventScheduler.next_event` → `crossing_time` | continuous `tau` | same-boundary sub-step | recomputed fresh each loop iter | live `V, g_L, g_inh, remaining_excitation, refractory, fired_this_boundary` | — (pure read) | declared analytic model |
| E firing | `_fire_event_cell` → `ConductanceLIFNeuron.fire(tau)` | continuous `tau` | same-boundary | **once/neuron/boundary** (`fired_this_boundary` + `crossing_time→inf`) | frozen drive, `V` | `V→rest`, `remaining_excitation→0`, arms refractory | declared one-spike-per-neuron-per-boundary |
| coincidence deposit | `CoincidencePyramidalNeuron.resolve_dendrites` → `apply_charge_impulse` | continuous `tau` (apical) / `0.0` (boundary-start basal) | same-boundary | **once/boundary** (`deposit_committed_this_boundary`) | basal weight, carried eligibility, current apical | consumes eligibility; impulse to `V` | declared idempotent gate |
| apical delivery | `_drive_event_apical` → `deliver_apical` | continuous `tau` (= driver's) | **zero-latency** | per apical source (gate still idempotent) | apical source set | opens gate | declared zero-latency callback |
| inhibitory relay firing | `_drive_event_relays` → `receive`/`resolve` | continuous `tau` (inherits driver) | **zero-latency** | **once/relay/boundary** (guard `relay.spiked`) | Boolean input | — | declared one-emission-per-relay-per-boundary |
| hard reset | `_drive_event_relays` → `hard_reset(tau)` | continuous `tau` (= relay) | zero-latency | once per (relay-firing, target) | target `V`, `remaining_excitation` | `V→rest`, drive discarded | declared zero-latency reset |
| feedforward learning | `_fire_event_cell` → `update_acc_weights` | continuous `tau` (at fire) | same-boundary | once per causal spike | causal per-target volley, pre-update weights, `iaccq` | writes `acc_weights` | declared causal-once |
| basal learning | `_fire_event_cell` → `update_basal_weight` | continuous `tau` (at fire) | same-boundary | once per causal spike | causal deposit signal, pre-update weight, `v_pre` | writes basal weight | declared causal-once |
| conductance decay | finalization `decay_conductance` | integer boundary | end-of-boundary | once | `g_inh, alpha_inh` | shrinks `g_inh` | declared per-boundary |
| refractory decay | finalization `advance_refractory` | integer boundary | end-of-boundary | once | `refractory_timer, spiked` | decrements timer | declared per-boundary |
| basal-eligibility settle | finalization `settle_eligibility` | integer boundary | end-of-boundary | once | `basal_received`, deposit flag | sets/clears 1-boundary carry | declared one-boundary carry |
| boundary finalization | `_event_step` tail (trace, decay, refractory, history) | integer boundary | end-of-boundary | once | end-of-boundary state | — | declared |

`Iaccq` relationship (`snn/neurons.py`): on the event path `freeze_drive` sets
`iaccq = remaining_excitation` **before** any fire/hard-reset consumes it, so it is the
frozen delivered charge (can exceed `theta`; not clamped, not overwritten by post-reset
zero). For a coincidence cell the dual rule reads `v_pre` (the pre-reset somatic membrane),
consistent with all C charge being basal deposits.

---

## Phase 2 — Independent single-neuron numerical validation

RK4 (`experiments/engine_validation.py: rk4_advance / rk4_crossing_time`) integrates the
frozen-drive ODE `dV/dtau = -(g_L+g_inh)·V + (g_L·V_rest + g_inh·E_inh + I_exc)` — the same
segment model the production docstring declares — and never calls `advance_segment` /
`crossing_time`. Ten cases cover subthreshold evolution, crossings mid/near-start/near-end,
no-crossing asymptote, the `g_total==0` integrator branch, nonzero `V0`, representative and
strong inhibition, and pure decay. Reference step counts: state `{8,16,32,64,128,256}`,
crossing `{128…2048}`.

Results (seed 1):

- **State**: monotone-decreasing error in every case; observed convergence order ≈ **4.0**
  on the cases that stay above the floating-point floor (`conductance_inhibited` 4.019,
  `decay_only` 4.025, `strong_inhibition_reversal` 4.038, `subthreshold_leaky` 3.963,
  `no_crossing_asymptote` 3.824). Cases whose error collapses to the ~1e-12 fp floor (e.g.
  `cross_mid_interval`, finest error 3.4e-12) show meaningless order estimates but remain
  monotone — the analytic value is exact and RK4 reaches it. Finest state error < 1e-6 in
  all cases; the pure integrator is exact (0.0).
- **Crossing time**: RK4 + linear interpolation converges monotonically at ≈ O(dt²)
  (interpolation-limited: `cross_mid_interval` 2.59, `cross_near_start` 1.18), finest error
  < 1e-6; the integrator crossing is exact (linear trajectory).

Tolerances were read from the observed order and fp scale, not chosen to force green.
**Conclusion: the analytic segment solver is correct for its own continuous segment model
(VALIDATED).** This validates the solver, not the boundary-level network approximation.

---

## Phase 3 — Scheduler and causal invalidation

Verified (`run_scheduler_causality` + tests): earliest finite crossing wins regardless of
node-list order when times are distinguishable; a hard reset invalidates a previously
predicted crossing (the scheduler recomputes from live state each call, so no stale event
survives); a changed/reset input recomputes the affected trajectory; logical time never
moves backward; and boundary-endpoint ownership is unambiguous — a crossing at **exactly
`tau == 1.0` fires within the current boundary**, while one that would land just beyond 1.0
does not. Tie tolerance is characterized: a difference inside tolerance is recorded as a
`latency_tie` broken by stable node order; a difference outside is not a tie. The
`event_trace.jsonl` records every spike (+`tau`), emitted edge, hard reset (with
`v_before`/`drive_before`), and every membrane `V`/`g_inh`/`refractory`, permitting manual
reconstruction of state before/after each transition. **VALIDATED.**

---

## Phase 4 — Same-time concurrency and order sensitivity

Two independent latency cells forced to cross at the **exact same `tau`** into a relay that
resets a non-driver sink. Under permutation of node order, edge order, and source ids
(`run_permutation_independent_spikes`):

- **Commuting quantities** (order-invariant): the emitted-spike **set** and the reset-target
  **set**. Both drivers fire; the outcome set does not depend on ids or serialization.
- **Order-sensitive by explicit rule**:
  - *WTA arbitration* — at an exact latency tie the winner is the lowest stable node index,
    and the tie is logged (`latency_ties`). Reversing the drive reverses the winner with no
    node reorder (`L2E0` vs `L2E1`). This is **arbitration by an explicit, documented rule**
    (stable node order among true ties), classified *explicitly intended arbitration* — it is
    named "order-sensitive by explicit rule", **not** "simultaneous correctness".
  - *Inhibitory-relay multiplicity* — a relay emits **at most once per boundary** (guard
    `relay.spiked`). With two same-`tau` drivers, the **second driver's relay input is
    rejected/coalesced**: only one `relay_excitation` edge is emitted and only one hard reset
    fires. Disposition for the two-input case: **rejected (not delivered, not accumulated —
    the relay is a stateless Boolean), and visible in the trace only as the ABSENCE of the
    second relay edge in `emitted`; there is no explicit reject record.** Classified
    *deterministic approximation needing documentation* (now documented here).

Because both permutation-invariant sets are stable and both order-sensitive results follow
an explicit production rule, this item is rated **ORDER-SENSITIVE BY EXPLICIT RULE**, not a
failure.

Smallest reproducing circuit (relay coalescing): `experiments/engine_validation.py:
_relay_multiplicity_spec` — `RG0→B(pretrained)→{E0,E1}` with equal weights, `E0,E1→R`,
`R⊣Z`. Expected vs observed: both `E0,E1` fire at `tau≈0.667` (logged tie, chosen `E0`); `R`
fires once (`re0` emitted, `re1` absent); one reset `(R,Z)`. This is a declared approximation,
not a bug. Narrowest next action if a per-relay burst is ever required: give the relay an
explicit per-`tau` emission ledger and emit one reset per accepted driver. Existing
scientific conclusions are unaffected (all presets rely on the one-emission WTA behavior).

---

## Phase 5 — Event conservation ledger

Over the tiny WTA chain (6 boundaries) the reconstructed multiset ledger asserts: no
non-finite membrane/conductance/crossing/weight; every hard reset has a causal source; no
duplicate reset `(source,target,boundary)`; integer logical timestamps strictly increasing;
every sub-boundary spike `tau` finite and in `[0,1]`. **VALIDATED.**

**Documented limitation (no invented precision):** production `emitted` is a per-boundary
list of *edge ids*, not globally-unique per-event causal ids. The ledger is therefore a
**reconstructable multiset per boundary**, not a per-event identity. Building perfect
per-event identity would require production instrumentation (a monotonic event-id counter);
it was intentionally not added, per the "avoid production changes" rule.

---

## Phase 6 — Tiny-network reference trace

`TinyReferenceSim` is an independent priority-queue + RK4 reference for the WTA-chain circuit
**only**, with explicit integer-boundary physical time, declared delays, RK4 continuous
evolution between events, and batch same-time commits. It calls no production stepping/solver
method.

- **Comparison 1 (matched to production, delay-1 boundary delivery):** the reference
  reproduces production's complete `(boundary, id, tau)` spike trace — identical event
  multiset, matching sub-boundary crossing times to **max `|Δtau|` = 2.28e-10**. This is a
  full causal-trace agreement, not a winner-identity match.
- **Comparison 2 (diagnostic "continuous delivery"):** delivering feedforward charge at the
  exact declared physical arrival (presynaptic `tau` + delay) instead of the next boundary
  would place each downstream spike **one boundary earlier**. Every such difference is
  explained by exactly one declared production approximation — the `SYNAPTIC_DELAY = 1`
  outer-boundary delivery — not by any solver disagreement. The continuous variant is
  diagnostic, not an automatic correctness oracle.

**VALIDATED (matched); the boundary-delivery delay is the sole, declared, explained
difference from continuous delivery.**

---

## Phase 7 — Boundary sensitivity assessment

**Network boundary refinement is UNSUPPORTED without production changes.** The engine has no
explicit physical `dt`: the outer boundary is the only time unit, and *every* physical
quantity that a `dt → dt/2 → dt/4` test must hold fixed is instead indexed by the integer
boundary — feedforward/pretrained/basal delivery (`SYNAPTIC_DELAY = 1`), conductance decay,
refractory decay, activity-trace and basal-eligibility updates, learning exposure, and
stimulus `input_period`. Running twice as many boundaries doubles decay, learning
opportunities, delays, and input events; it is not the same physical experiment at half
`dt`. Reporting **UNSUPPORTED** (not PASS/FAIL) is the honest result. No production timing
semantics were modified to force the test.

Minimal future design required before a valid refinement test is possible (not implemented
here):

1. Introduce an explicit physical `dt` and express all conductance/refractory/leak time
   constants and learning cadence in physical time, decoupled from the boundary count.
2. Replace the integer delay-1 double buffers with a **physical-time delay queue** keyed on
   absolute arrival time.
3. Schedule stimulus events at physical times independent of boundary granularity.
4. Make decay/refractory/trace continuous (per elapsed physical `dt`) rather than
   per-boundary.

**Weaker property tested and VALIDATED — segment partition invariance:** splitting one
uninterrupted analytic membrane segment into `K ∈ {2,4,8}` sub-advances (same frozen drive,
no intervening event) reproduces the single `advance_segment(1.0)` final state to
`max |Δ| < 1e-6`. This is labeled **segment partition invariance**, explicitly *not* network
boundary refinement.

---

## Phase 8 — Learning-state timing probe

Smallest direct topology (`rg_direct_cc4`), dual FE/FES enabled, one controlled input
pattern, using only the pre-existing off-by-default `record_updates` instrumentation. From
the trace (`run_phase8_learning_probe`):

- `Iaccq` is the causal **pre-reset frozen charge** — positive, finite, and **not** the
  post-reset zero (captured at `freeze_drive` before the WTA hard reset consumes the drive).
- FE is evaluated from that `Iaccq`: the recorded factor equals a hand-reconstructed
  `dual_fe(Iaccq, θ, e, B)` to < 1e-12.
- FES is evaluated per **pre-update** synaptic weight (vectorized in the update).
- Learning happens **once** for the causal spike; the winner's membrane was reset the same
  boundary, yet `Iaccq` retained the pre-reset value.
- Repeating a display/serialization observation (`dynamic_state()`, `topology()`) causes **no**
  additional learning.
- No non-finite weight delta; no hidden boundary duplication.

**Manual weight-update reconstruction (reference run, winner `ccE0`, first causal spike):**
`Iaccq = 765.478969764685`, `θ = 1000`, `B = 5`, `e = 0.001` ⇒
`FE = e + (1−e)/(1 + B·(Iaccq/θ − 0.5)²) = 0.001 + 0.999/(1 + 5·(0.26548)²) = 0.73969`,
matching the logged `fe` to full precision. Each `Δw_i = η·FE·FES_i·signal_i·influence_i`
with `η = 0.01`, floored at `wte = 0.001`, no cap — exactly the module equation. This phase
validates timing/bookkeeping only; `B`, LR, `e`, `wte`, and weights were not tuned.
**VALIDATED.**

---

## Instrumentation neutrality

The only instrumentation used is the pre-existing, off-by-default `record_updates` flag.
Enabling it leaves the dual-FE dynamic trace (spikes) and learned weights **byte-identical**
over a 20-boundary run (`run_instrumentation_neutrality`). No new production instrumentation
was added.

---

## Automated tests added (`tests/test_engine_validation.py`, 20 tests)

RK4 state convergence + 4th-order check; finest-state tightness; numeric spike-time
convergence; integrator-exact crossing; segment partition invariance; scheduler causality
invariants; direct causal invalidation after reset; boundary-endpoint ownership; tie-tolerance
characterization; event conservation on a tiny graph; event timestamp monotonicity;
independent simultaneous-spike commutativity; **WTA same-time order sensitivity by explicit
rule** (named as characterization); **relay multiplicity/coalescing** (named as
characterization); dual-FE pre-reset `Iaccq` timing; manual FE reconstruction; passive
instrumentation neutrality; tiny PQ+RK4 reference match; continuous-variant diagnostic.

---

## Claim boundary

**What the engine can now be claimed to simulate faithfully:**

- Within one outer boundary, the conductance-LIF membrane evolves by the **exact analytic
  solution** of its frozen-drive ODE; analytic crossing times and the `advance_segment`
  state match an independent RK4 oracle to the floating-point floor (≈4th-order convergence),
  and the pure-integrator branch is exact.
- The sub-boundary event scheduler is **causally correct**: earliest finite crossing wins,
  resets invalidate stale predictions, logical time is monotone, and boundary-endpoint
  ownership is unambiguous.
- On the tiny WTA chain, the complete causal spike trace (boundaries + sub-boundary taus)
  is reproduced by an independent priority-queue + RK4 reference to `|Δtau| ≤ 2.3e-10`.
- Dual-FE learning reads the correct causal **pre-reset** state and updates exactly once per
  causal spike.

**What remains dependent on outer-boundary semantics (declared approximations):**

- All inter-neuron delivery is quantized to the integer boundary (`SYNAPTIC_DELAY = 1`); a
  continuous-delivery model would advance downstream spikes by one boundary.
- Conductance/refractory/trace decay, learning exposure, and stimulus scheduling advance
  per boundary, not per physical `dt`.
- An inhibitory relay emits at most once per boundary; simultaneous second drivers are
  coalesced/rejected and appear in the trace only as an absence.

**What would be required before calling it a globally continuous-time, pure discrete-event
simulator:** an explicit physical `dt` decoupled from the boundary count; physical-time delay
queues replacing the integer delay-1 buffers; continuous (per-`dt`) decay/refractory/trace;
physical-time stimulus scheduling; and a per-relay per-`tau` emission model if simultaneous
multiplicity must be preserved. Until then the accurate characterization is a **deterministic,
boundary-synchronous hybrid simulator with analytic within-boundary evolution** — validated
as such, with the outer boundary as a real semantic delay, not merely a display frame.

---

## Commands, counts, artifacts

```
# pre-change focused timing baseline (before adding anything): 88 passed
.venv/bin/python -m pytest tests/test_event_scheduler.py tests/test_lif_segments.py \
    tests/test_causal_step.py tests/test_inhibitory_relay.py tests/test_dual_fe_fes.py \
    tests/test_rg_direct_cc4.py tests/test_dual_fe_cc4_experiment.py -q

# new validation tests: 20 passed
.venv/bin/python -m pytest tests/test_engine_validation.py -q

# existing timing/neuron/inhibition/direct-CC4/dual-FE/coincidence tests: 142 passed
.venv/bin/python -m pytest tests/test_event_scheduler.py tests/test_lif_segments.py \
    tests/test_causal_step.py tests/test_inhibitory_relay.py tests/test_dual_fe_fes.py \
    tests/test_rg_direct_cc4.py tests/test_dual_fe_cc4_experiment.py \
    tests/test_coincidence_cell.py tests/test_rg_coincidence.py -q

# full suite: 499 passed, 2 warnings (pre-existing FastAPI on_event deprecation; unrelated)
.venv/bin/python -m pytest -q

# whitespace/conflict check: clean
git diff --check

# headless validation experiment (reference seed):
PYTHONPATH=. .venv/bin/python experiments/engine_validation.py --seed 1
#   -> experiments/runs/engine_validation/20260723-185239-engine_validation-10c794e6/
```

**Changed / added files (working tree; nothing committed):**

- `experiments/engine_validation.py` (new)
- `tests/test_engine_validation.py` (new)
- `docs/ENGINE_VALIDATION_REPORT.md` (new)
- `experiments/runs/engine_validation/…` (new, gitignored artifacts)

Pre-existing unrelated working-tree modifications (untouched by this task):
`backend/simulation.py`, `frontend/inspector.js`, `frontend/renderer.js`, `snn/neurons.py`,
`tests/golden/rg_coincidence_baseline.json`, `tests/test_coincidence_cell.py`,
`experiments/interleaving_parallel_rf.py`, `prompts/Claude_Hybrid_Engine_Validation_Prompt.md`.

No warnings were suppressed and no existing assertion was loosened.
