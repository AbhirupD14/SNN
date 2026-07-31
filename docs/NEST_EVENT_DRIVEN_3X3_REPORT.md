# NEST event-driven 3×3 prototype — results

**Status:** Phases 0–3 complete and passing their gates. Phase 4 (learning) is a
**documented partial result**, not a completed port. Phase 5 (threads) complete; Phase 5
(MPI) **unreachable** in this environment.

**Dashboard visualization:** delivered separately — see
[`docs/NEST_DASHBOARD.md`](NEST_DASHBOARD.md). Offline replay and the conditional live
view are both shipped; chunked execution was proven inert before live mode was built.
Membrane charge **is** recordable (`q`/`q_pre` are NESTML recordables sampled by a NEST
multimeter, proven non-perturbing) and is available in replays on request.

**Scope:** `backend.network_spec.tiled_cc_spec(cc_e_count=8)` only — 191 nodes, 1052
directed edges. No other preset.

**Environment:** NEST 3.10.0 + NESTML 8.3.0, conda-forge, linux-x86_64, Python 3.12.13.
Seed 1 throughout unless stated.

**The Python engine is unchanged.** Its suite was recorded green (653 passed) before any
work began and again at the end. Nothing in `backend/` or `snn/` was modified.

---

## 1. Headline findings

1. **Single-winner WTA does not survive** native NEST delivery at the default timescale
   separation. Mean winner multiplicity is **3.6** per column-window (max 8 of 8) where the
   reference engine guarantees exactly 1.
2. **Fast inhibition alone recovers nothing.** With arrival dispersion disabled, *all eight*
   competitors fire every window (multiplicity 8.0). This is mechanical, not incidental:
   with a simultaneous volley there is no latency ordering for inhibition to arbitrate, and
   shrinking `h` shrinks the loop and the tie together.
3. **Winner multiplicity converges toward 1 as `S / L_wta` grows** — 8.0 → 1.6 over a 60×
   sweep — but does not reach it. The limit is not timing: the competitors are
   **near-degenerate**, with only a **2.17 %** spread in total afferent weight from the
   balanced initializer.
4. **Outcomes are invariant to absolute `h` at fixed ratio.** Identical results at
   `h ∈ {0.05, 0.1, 0.2}` ms. Only the ratio is physical — the hypothesis in §5.5 of the
   prompt holds exactly.
5. **The feedback cadence law does not reproduce.** No `1010` alternation at `D = L`, and no
   sharp dependence on `D`. Mechanism in §5 below.
6. **The dual FE/FES equation *is* expressible** and computes exactly — but NEST applies
   updates at the wrong *time*. See §6.
7. **Structural and locality parity is exact**: counts, roles, ID mapping, centre-patch
   isolation, two-patch locality, and the dormant top-column `C` all behave as specified.

---

## 2. Parity classification

Per prompt §3, split four ways.

### Structural parity — exact

| Property | Reference | NEST | |
|---|---|---|---|
| nodes / directed edges | 191 / 1052 | 191 / 1052 | ✅ |
| archetype inventory | 90 E-competitor, 81 RGC, 10 C, 10 I | identical | ✅ |
| edge-kind inventory | 800 ff, 90 relay, 80 reset, 72 apical, 10 basal | identical | ✅ |
| external ID mapping | — | complete + injective, round-trips | ✅ |
| initial weights | engine initializer | read from the engine itself | ✅ |
| determinism at fixed seed | yes | yes | ✅ |
| dormant top-column `C` | never fires | 0 apical, 0 deposits, 0 spikes | ✅ |
| centre-patch isolation | driven column + L2 only | driven column + L2 only | ✅ |
| two-patch locality | no crosstalk | no crosstalk | ✅ |

### Equation parity — exact where scoped

* Accumulation is a **pure sum** — verified across arrivals 100 ms apart with no decay.
  This is exact, not approximate, because `leak_rate = 0` makes the reference membrane a
  pure integrator (see `nest_backend/README.md` validity envelope).
* Threshold is `V >= θ`; at-threshold fires.
* `q_pre` retains overshoot unclamped (1.5 θ recorded as 1.5 θ).
* `C` is a strict temporal AND: basal-only and apical-only both deposit **zero** charge.
* Deposit magnitude is exactly `w_basal · s`, once per window.
* `Eor` fires on a single θ afferent.
* Dual FE/FES matches a Python oracle to **1e-6** — both the `+1` and `−1` branches, each
  synapse using its own pre-update `w_i`.

### Native-semantic differences — measured, not hidden

| Difference | Reference | NEST | Consequence |
|---|---|---|---|
| WTA arbitration | continuous-`τ` first-spike latency; winner's `I` cancels all later crossings at the same `τ` | crossings quantised to the `h` grid; reset cannot arrive sooner than `L_wta = 2h` | **multi-winner** (§3) |
| zero-latency apical / reset | 0 | `h` | requires explicit TTLs on `C` |
| same-timestep arrivals | distinct events | **summed into one handler call** per port per step | multiplicity recovered as `q_pre / w`, not from a call count |
| drive packet | frozen per boundary, discardable | no packet; charge is a stream | cadence law fails (§5) |
| eligibility expiry | settled once per boundary | detected lazily at next receipt | count exact, timestamp deferred |
| learning application time | at the instant the cell fires | at the afferent's next **pre**-spike | §6 |

### Implementation gaps

* **MPI unreachable**: the conda-forge NEST 3.10 build is not compiled with MPI support.

> **Corrected 2026-07-29.** An earlier version of this report listed `I_accq` as a NESTML
> continuous post port among the implementation gaps, on the evidence of an `__I_accq`
> undeclared compile error. That was a **downstream symptom, not the cause**: NESTML
> classifies a model as a synapse *by its name*, so `plastic_feedforward` was being
> generated as a NEURON, and the post-port wiring never ran. Renaming it to
> `plastic_feedforward_synapse` fixes it, and the reproducer now reports
> `tier_a_continuous_post_port.available: true`. The continuous post port **works**. The
> Phase 4 blocker is timing alone -- see section 6.

---

## 3. The WTA result

### Why fast inhibition is not sufficient

The reference winner is selected by pure first-spike latency because the drive packet is a
**constant current over the boundary**, so `V(τ) = V₀ + I_exc·τ` and crossing time is
`(θ − V₀)/I_exc` — ordering by drive magnitude.

Delivered as delta impulses, every competitor jumps at once and all supra-threshold
competitors cross on the **same timestep**. There is nothing to arbitrate. Proven three
ways, at increasing scale:

* **Two-cell microcontract** (`tests/nest/test_nest_node_contracts.py`): with a shared
  volley, inter-arrival gap `2.0 ms = 10 × L_wta` → one winner; gap `0` → two winners; gap
  `h = 0.5 × L_wta` → two winners. Single-winner requires the gap to **exceed `L_wta`**, not
  merely be non-zero.
* **Full topology, dispersion off**: multiplicity exactly **8.0**, `Eor` input multiplicity
  **8.0**.
* **Sweep**: monotone convergence, below.

### The `S / L_wta` sweep

Centre patch, `row 1`, 8 presentations, `h = 0.1 ms`, `L_wta = 0.2 ms`.

| `S / L_wta` | realized `S` (ms) | mean winners | max | fraction single-winner | `Eor` input mult. |
|---|---|---|---|---|---|
| 0 (dispersion off) | 0 | **8.00** | 8 | 0.00 | 8.0 |
| 1 | 0.2 | 6.33 | 8 | 0.00 | 3.0 |
| 3 | 0.6 | 6.14 | 8 | 0.00 | 5.0 |
| 10 *(default)* | 2.0 | 3.63 | 8 | 0.25 | 3.0 |
| 30 | 6.0 | 2.38 | 5 | 0.50 | 2.0 |
| 60 | 12.0 | **1.60** | 2 | 0.40 | 1.0 |

The hypothesis in prompt §5.5 holds in direction and shape. It does **not** reach 1.

### Why it stalls short of 1 — and it is not a timing problem

Measured directly: across the centre column's eight competitors the **total afferent weight
spans only 2.17 %** (2238.6 → 2287.1). That is the Sinkhorn-balanced initializer
(`l2_init_total_frac = 0.95`, doubly balanced) doing exactly what it was designed to do.

Latency arbitration can only separate competitors whose crossing times differ. With
near-identical weights, and charge persisting across volleys (pure integrator), all eight
competitors sit near θ simultaneously by the third or fourth volley and cross on the *same*
arrival. Dispersion buys separation only up to the point where weight differences stop
discriminating.

**This is the §5.4 prediction, quantified.** Arbitration is by weighted prefix-sum over
arrival order, not total drive, and the two agree only when weights discriminate.

### Agreement with the Python engine's winner

| Case | agreement rate |
|---|---|
| centre patch `row 1` | 0.50 |
| centre patch `col 1` | 0.00 |
| two independent patches | 0.11 |
| all nine patches | 0.26 |

For `row 1` the Python engine selects **one** dominant winner (`L1c11E4`, 1 distinct winner
across the run) while NEST rotates `E4, E6, E0, E4, E4, E6`. A stable reference winner
becomes an unstable NEST winner. The `row 1` vs `col 1` gap (0.50 vs 0.00) is the
**geometric scan-order asymmetry** predicted in prompt §5.4: delays follow geometry, so each
column acquires a fixed scan order over its 3×3 patch, and the two patterns present
systematically different arrival profiles.

---

## 4. Resolution invariance

At fixed ratio `L_wta : S : D = 1 : 10 : 100`, results are **bit-identical** across
`h ∈ {0.05, 0.1, 0.2}` ms: 51 spikes, mean multiplicity 3.625, fraction single-winner 0.25.

Only the ratio is physical. No discretization artifact was found, and none had to be
explained away.

---

## 5. Feedback: one parity result, one negative result

### The confirmation pathway works — after a defect was found by measurement

The reference is explicit that the `C → I` confirmation volley is **not** swallowed by the
once-per-boundary WTA guard (`Current_Implementation_Methodology_Equations.md` §7). The
first translation routed `C → I` through the same rate-limited port as the lateral `E → I`
volley, so every confirmation was absorbed by the lockout and feedback-on / feedback-off
runs were **bit-identical**. The defect was invisible in the spike record and only showed up
as `n_suppressed` rising from 3 to 5.

Fixed by giving `event_relay` a separate `confirm` port that neither consults nor extends
the lockout. With a mature `C`:

| | spikes | mean winners | `Eor` input mult. |
|---|---|---|---|
| feedback **on** | 46 | 2.75 | 2.0 |
| feedback **off** | 53 | 3.63 | 3.0 |

The top-down confirmation reset measurably suppresses redundant re-fire — its documented
purpose. `tests/nest/test_nest_3x3_suite.py` guards this.

### The plain feedback control is uninformative — and here is why

`C` initializes at `θ/4` while one-shot recognition needs `w_basal ≥ θ`. Under frozen
learning `C` **can never fire**, so the confirmation pathway never triggers and disabling it
changes nothing. Cases 6 / 7 are therefore also run with `C` frozen at its **mature** `θ`
weight — still frozen learning, but at the value that makes the pathway live. Without this,
the feedback control would have read as "feedback does nothing".

### The cadence law does not reproduce

The reference predicts strict `1010` alternation at presentation period `= L` (the
graph-derived loop latency), and no suppression at `L + 1` hop. Measured, with `L = 3.475
ms` derived from the translated graph and a mature `C`:

| period | L1c11 firing pattern |
|---|---|
| `D = L = 3.5 ms` | `011101110111` |
| `D = L + 1 hop = 4.5 ms` | `011101110111` |

Period-4, not period-2, and **identical at both periods** — no sharp `L mod input_period`
dependence.

**Mechanism.** The reference law works because the confirmation reset lands at the *start*
of the next boundary, *after that boundary's drive packet has been frozen*, so the whole
packet is discarded. NEST has no frozen packet: charge arrives as a stream of events, and a
reset at time `T` erases only what accumulated **before** `T`. Everything arriving after `T`
still accumulates. Suppression is therefore partial rather than total, and the exact
alternation cannot arise.

This **contradicts** the optimistic "parity win to verify" written into §5.3 of the driving
prompt, which speculated the reset would land harmlessly in the inter-volley silence and
reproduce the same functional outcome. It does land in the silence — and that is precisely
why it *cannot* discard a drive packet. The prompt instructed that this be confirmed
empirically rather than asserted; it was, and it failed. §5.3 of the prompt has been
corrected.

---

## 6. Phase 4 — learning: expressible, but applied at the wrong time

The prompt (§6.5) singled out one risk: *does a post-spike update give every afferent the
same pre-update `FE` and its own pre-update `w_i`?*

**That risk is resolved positively.** Minimal reproducer, one post cell, two afferents with
deliberately different starting weights (600 and 300):

| afferent | initial | at post-spike | after its own next pre-spike | oracle |
|---|---|---|---|---|
| participating (`s = +1`) | 600.0 | 600.0 *(unchanged)* | **600.3377125822307** | 600.3377125822307 |
| silent (`s = −1`) | 300.0 | 300.0 *(unchanged)* | **298.89302954037265** | 298.89302954037265 |

Both branches match the declared equation to 1e-6, and each synapse used its **own**
pre-update weight (`own_pre_update_weight_used: true`).

### Correction: what "not STDP" actually means here

An earlier draft of this section asserted the rule was "Δt-free". **That was wrong.** `s_i`
is evaluated at the postsynaptic cell when it fires, and it does consult presynaptic
arrival timing — `backend/simulation.py::_participation` scopes it to exactly one boundary,
"never … one from the preceding/following boundary".

The property that distinguishes this from STDP is not the absence of timing but **binary
membership in a structurally-derived window** versus **graded dependence on the interval**:

| | STDP | dual FE/FES |
|---|---|---|
| magnitude | `f(t_post − t_pre)`, continuous and monotone | **flat** — `s_i` is exactly ±1 |
| outside the window | decays smoothly toward 0 | snaps to **−1**, never 0 |
| window origin | a fitted time constant `τ` | the volley/boundary structure |
| shape | exponential kernel | rectangular, not tunable for effect |

A flat one-volley window is therefore legitimate and *required*. It is the same kind of
object as `tau_basal` on the coincidence cell: the NEST-time expression of "one boundary".

This surfaced a real semantic error in the synapse model, now fixed. Participation had been
a sticky flag cleared at the post-spike — i.e. "did I spike since this cell last fired".
Because an ordinary E cell needs three or four volleys to reach threshold, that flag spanned
several volleys and awarded `+1` to afferents the reference explicitly excludes. It is now a
one-volley window (`participate_until`, width `tau_volley`), consumed on use.

`tests/nest/test_nest_environment.py::test_learning_rule_is_not_stdp` enforces the
distinction mechanically: it forbids exponentials, decaying state and interval-graded
magnitudes, deliberately *permits* the flat window, and pins that the gated signal is
exactly ±1.

**The remaining blocker is timing of application.** Neither afferent is updated at the
instant the target fires. NEST visits a synapse only when a **presynaptic** spike traverses
it; post-spikes are archived on the target and replayed lazily. So every update is deferred
to the afferent's next pre-spike.

The reference rule runs the update *at the firing instant*, and subsequent updates read the
post-update weight. Deferral changes the trajectory whenever a cell fires more than once
between one afferent's spikes — which is the normal case here. An afferent that never spikes
again is never updated at all.

`I_accq` as a continuous post port **does** work, once the synapse model is named
`*_synapse` so NESTML classifies it as a synapse at all (see `nest_backend/README.md`).
The reproducer's tier A reports `available: true`.

**Phase 4 is therefore reported as a blocked feasibility result, not a completed port**, as
prompt §6.5 provides for. Phases 0–3 stand on their own. Closing the gap would need either a
NEST extension that iterates a target's afferents at post-spike time, or a redesign of the
rule to be expressible in NEST's pre-spike-triggered archive model — a scientific change,
out of scope here.

---

## 7. Phase 5 — threads and MPI

**Threads (OpenMP): correct.** Identical spike multisets at 1, 2 and 4 threads — not just
equal counts, but the same `(time, sender)` multiset.

| threads | spikes | wall clock (s) |
|---|---|---|
| 1 | 51 | 0.0059 |
| 2 | 51 | 0.0047 |
| 4 | 51 | 0.0038 |

**No speedup is claimed.** At 191 nodes and ~51 spikes these runtimes are dominated by
setup and are within noise of each other; they are reported only so that "no speedup was
measured" rests on a measurement.

**MPI: unreachable.** The conda-forge nest-simulator 3.10 package depends on `openmpi` and
ships `mpirun`, but the NEST library is **not compiled with MPI support** — launching under
a multi-rank launcher aborts with *"NEST was not compiled with MPI support"*. An earlier
version of the verifier *inferred* MPI availability from the openmpi dependency and was
wrong; it now **probes by launching two ranks** and reports absence as absence. Reaching MPI
would require building NEST from source with `-Dwith-mpi=ON`.

---

## 8. What NEST owns, and what this prototype does not do

NEST owns the clock, delivery order, connection delays, event buffering, spike recording and
threading. There is **no** Python priority queue, timestamp closure, causal-generation loop,
custom scheduler, or per-spike learning callback anywhere in `nest_backend/`.

Every committed event-node model has an `update` block containing only `dummy integer = 0`,
asserted mechanically by `tests/nest/test_nest_environment.py`, which also rejects
`integrate_odes`, `equations:`, `convolve`, kernels, continuous ports and `onCondition` in
those models. The one ODE in the package lives in the synapse, integrates an inert
`unused_trace` that no handler reads, and exists solely because NESTML will not generate a
synapse without one — also asserted by test.

**Not claimed:** pure DES (NEST retains its native resolution / min-delay execution
contract), exact legacy equivalence, calibrated biological wall-clock units (only the
*ratios* are claimed as physiologically motivated), or MPI speedup.

---

## 9. Reproducing

```bash
./scripts/setup_nest_env.sh
NEST_TESTS_REQUIRED=1 .nest-env/bin/python -m pytest tests/nest/ -q   # 70 passed
.nest-env/bin/python experiments/nest_3x3_suite.py                    # 21 cases
.nest-env/bin/python experiments/nest_parallel_check.py
.nest-env/bin/python -m nest_backend.phase4_reproducer
.venv/bin/python -m pytest -q                                         # 653 passed, unchanged
```

Per-case artifacts land in `experiments/runs/nest_3x3/<timestamp>/` (gitignored), one JSON
per case plus an `index.json`, each carrying versions, resolution, `L_wta`/`S`/`D`, delay
histogram, seeds, topology hash, full spike record and every metric above.
