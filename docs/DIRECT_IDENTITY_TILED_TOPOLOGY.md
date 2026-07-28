# Direct-Identity Tiled Cortical Columns (`tiled_cc_direct_identity`)

**Status:** implemented and causally tested. Identity preservation and composition
discrimination are demonstrated. The timing anomaly first reported here (a `tau = 1.0`
deadlock that silenced the top-down C/feedback path) has since been **resolved**; the general
cadence/loop-latency findings it led to now live in
`docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`.

Authoritative evidence: `experiments/direct_identity_experiment.py` →
`experiments/direct_identity_results.json`. Tests: `tests/test_tiled_cc_direct_identity.py`,
`tests/test_multi_basal_coincidence.py`. A dashboard screenshot is supplemental only.

---

## 1. Why Eor was removed

`Eor` was a single output relay per column trained with the **same competitive
signed-participation rule as an ordinary pattern detector**. Two consequences made it
unusable as a hierarchical output:

1. **Relay deadlock.** After a long single-pattern dwell one incoming `Eor` weight reaches
   `theta` while every non-participating weight collapses toward the dual-rule floor. If a
   new pattern recruits a *different* local owner, that owner's floor-weight event cannot
   make `Eor` fire — and `Eor` cannot recover, because its learning is gated on its own
   postsynaptic spike. Measured in the preceding task: after 8000 boundaries at dashboard
   rates, 23 of 80 `E → Eor` afferents had decayed below `theta`, the minimum reaching
   **0.00**.
2. **Identity erasure.** Even a *fixed* Boolean-OR `Eor` (the fix applied before this task:
   init at `theta`, plasticity frozen) removes the deadlock but still tells the parent only
   *"this child column was active"* — never *which* learned local pattern won.

The ordinary-E WTA winner is already a sparse one-hot latent code. This topology transmits
that source address directly instead of pooling it.

## 2. Graph contract

Fixed shape at `cc_e_count = 8`: **181 nodes, 1546 directed edges.**

| Population | Count |
| --- | --- |
| `rg_source` (9×9 RGC surface) | 81 |
| `e_latency_competitor` (ordinary E, 8 per column × 10 columns) | 80 |
| `e_coincidence` (one multi-basal C per column) | 10 |
| `i_relay` (one WTA/feedback I per column) | 10 |

| Projection | Edges |
| --- | --- |
| `rg_to_column` — RGC → local ordinary E | 648 |
| `identity_child_e_to_parent_e` — child E → **every** parent E | 576 |
| `column_to_column_apical` — L2 E → child C | 72 |
| `column_e_to_c_basal` — local E → own column C | 80 |
| `column_e_to_i` — E → local I | 80 |
| `column_i_to_e` — I → local E (hard reset) | 80 |
| `column_c_to_i` — C → local I (delay-1 feedback) | 10 |
| **total** | **1546** |

There is **no `Eor` node, edge, weight or role** anywhere; `column_role='Eor'` is rejected
outright by the variant validator. Each L2 ordinary E owns **72 distinct plastic weights**
(9 child columns × 8 possible child winners) — one per source address, never pooled by
column. The top `L2c00C` has no parent, so it is declared dormant and holds zero apical
edges: it never deposits or fires, which keeps the motif recursively reusable if a later L3
gives L2 a parent.

Construction/validation/layout branch on `topology.variant = 'direct_identity'` — never on a
preset name or a node-id prefix.

## 3. Dendritic orientation (unchanged)

```text
bottom-up / local driving evidence -> basal dendrite
top-down / higher-layer feedback   -> apical dendrite
```

The classic column had **one** learned `Eor → C` basal edge and eight structural
`parent E → child C` apical edges. This topology replaces the single basal with **eight
source-distinct local `E → C` basal edges** and keeps the apical side exactly as it was.
L2 feedback was **not** moved to basal.

## 4. Multi-basal coincidence contract

`CoincidencePyramidalNeuron` now accepts either a scalar basal source (the historical
one-basal cell, byte-identical — the `rg_coincidence` golden is unchanged) or a sequence:

```text
B = a current or one-boundary-carried basal event on ANY source
A = any current apical event
deposit at most once iff B AND A, using the CAUSAL source's own weight
```

* **Causal selection.** A *current* event is preferred over a *carried* one; within each,
  the **earliest delivered** source is causal. Source/edge/weight/distance/eligibility
  vectors stay index-aligned.
* **Unexpected second source.** Local WTA should leave only one basal source per boundary.
  A second *distinct* source is **recorded** (`basal_extra_sources`), never silently
  dropped; a repeat receipt on an already-delivered source is counted
  (`basal_duplicate_count`) and cannot deposit a second impulse.
* **Idempotence.** One coincidence commits at most one deposit; later receipts in the same
  boundary cannot re-deposit or erase the diagnostics.
* **Eligibility.** Per source, carried for exactly one boundary, consumed by a committed
  deposit, expiring otherwise. Eligibility never survives two boundaries.
* **Apical stays Boolean.** Source identity remains diagnostic; permission is `any(apical)`.
* **C still cannot fire** from basal-only or apical-only input, and a retained
  supra-threshold membrane still cannot fire without a *current* gate.

### Basal learning

**Only the causal weight is updated.** There is no `+1/-1` participation term across the
basal vector — precisely the rule that made `Eor` undeliverable. Every non-causal basal
weight is left byte-identical, so one local owner maturing its association cannot depress
another's. The per-basal ceiling stays at `theta` (`relay_weight_cap_frac`), the C's own
role-specific one-afferent bound, not the `theta/2` detector rule.

**Capability is opt-in.** A coincidence node must declare `multi_basal: true` to own more
than one basal edge; without it the historical *exactly one basal* invariant still applies,
so no existing custom or saved graph is silently relaxed.

## 5. Detector ceiling

`e_weight_cap_frac = 0.5` is preserved and applies to RGC→L1 ordinary detectors **and** to
the direct L1-winner→L2 ordinary detectors. Measured maximum L2 identity afferent after
training: **500.0 = exactly `theta/2`** — the cap binds.

One child identity event therefore cannot make a *reset* L2 detector cross instantaneously;
two coordinated child events can. **With `leak_rate = 0` this does not enforce two
*distinct* sources over an unlimited window**: repeated events from a single source
accumulate across boundaries and eventually cross. That is visible in the one-patch regime
below (L1 wins 92 % of boundaries, L2 still wins 30 %) and is reported as a property of the
zero-leak configuration, not as evidence that the cap enforces spatial coincidence.

## 6. Feedback inhibition (retained, not solved)

Unchanged from the classic column: ordinary `E → I` performs immediate same-boundary WTA,
and `C → I` schedules the existing delay-1 feedback hard reset of the **whole ordinary-E
pool**. There is no `Eor` feedback target. This task makes that retained pool-reset behavior
explicit and **does not** claim the output-target question is solved: allowing internal
competition while suppressing only the outbound identity event remains a separate standing
problem.

## 7. Experiment

```bash
PYTHONPATH=. .venv/bin/python experiments/direct_identity_experiment.py
```

Protocol constants: topology `tiled_cc_direct_identity`; rates
`dual_fe_fes=True, eta=4.0, c_eta=2.0, dual_fe_B=5.0, leak_rate=0.0, refractory_steps=0,
c_feedback_reset=True, e_weight_cap_frac=0.5`. This acceptance protocol is intentionally
frozen at the historical `c_eta=2.0`, `input_period=1`; the live dashboard now uses
`c_eta=16.0` and graph-derived auto pacing. The result artifact reports the resolved
parameters. Dwell is 2500 boundaries per phase; an owner is declared over the trailing
400 boundaries at ≥ 0.8 dominance.

### A. Turnover, recall and direct evidence (seed 1, patch (1,1))

| Phase | Pattern | Owner | Dominance | Boundaries won | Direct parent evidence | L2 winner |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `row 1` | `L1c11E6` | 1.000 | 2215/2500 | **100 %** | `L2c00E5` |
| 2 | `col 1` | `L1c11E4` | 1.000 | 2214/2500 | **100 %** | `L2c00E0` |
| 3 | `row 1` | `L1c11E6` | 1.000 | 2143/2500 | **100 %** | `L2c00E5` |

Turnover on the switch, recall of the original owner on return, and on **every** boundary
either owner won, its identity reached **all eight** L2 detectors in the same step. No
`Eor` recovery process exists or is needed.

C basal weights of `L1c11C` (init 250.0):

| after phase | `E4` | `E6` | other six |
| --- | --- | --- | --- |
| 1 (`row 1`) | 250.00 | **613.80** | 250.00 |
| 2 (`col 1`) | **612.30** | 613.80 | 250.00 |

The first owner's association changed by **exactly 0.0** while the second owner matured its
own. This is the pattern-specific column certainty the multi-basal C exists to provide.

### B. Pattern sweep — 3 seeds × 4 ordered pattern pairs

**12/12 runs** satisfied every check: stable owner per phase, turnover on switch, recall on
return, 100 % direct evidence, and no depression of the first owner's basal weight.

**Honest scope limit.** Each run presents only *two* patterns, and the owner assignment is
determined by seed and presentation order, not by pattern identity — e.g. for seed 1 the
first pattern presented always recruits `E6` and the second `E4`, whichever patterns those
are. This establishes *"a newly presented pattern recruits a different neuron than the
incumbent, and the incumbent is recalled"*. It does **not** establish four-pattern capacity
or a stable pattern→neuron assignment; that requires a separate capacity experiment.

### C. Composition — two patches (0,0)=`row 1`, (2,2)=`col 1`, seed 1

* distinct per-column owners: `L1c00E4`, `L1c22E4` (distinct cells in distinct columns);
* **1815/1815** boundaries on which both columns won also delivered **both** identities,
  distinctly, to the *same* L2 detector — no pooling, no source erasure;
* maximum L2 identity afferent **500.0 = `theta/2`**: the ceiling holds per afferent;
* L2 winner `L2c00E3`, winning 1524/2500 boundaries.

### D. Identity discrimination — the success criterion

Patch **locations held fixed**; only which local pattern (hence which local winner) one
patch emits is changed.

| Composition | `L1c00` owner | `L1c22` owner | L2 winner distribution (600-boundary probe) |
| --- | --- | --- | --- |
| A = (`row 1`, `col 1`) | `L1c00E4` | `L1c22E4` | `L2c00E3` ×247, `L2c00E4` ×2 |
| B = (`diag \`, `col 1`) | `L1c00E3` | `L1c22E4` | `L2c00E4` ×239, `L2c00E3` ×1 |

The changed patch swapped local winner (`E4 → E3`) while the unchanged patch kept `E4`, and
**L2 selected a different winner for each composition** (`E3` vs `E4`). A parent can
therefore use the child winner's identity to distinguish a controlled pair of compositions
that a pooled column-level event could not separate. **This is the criterion the topology
was built to satisfy, and it is met.**

## 8. Negative results and timing anomalies

### 8.1 `tau = 1.0` boundary-edge deadlock — **RESOLVED**

> **Fixed after the first report.** The sub-boundary event loop now *drains* events instead
> of stopping at `current_tau < 1.0`, so a crossing that only becomes available at exactly
> `tau = 1.0` is admitted. The scheduler already bounded candidates to `[current_tau, 1]`
> (a sub-threshold cell has no interval left and reports `inf`), so the extra pass admits
> exactly the cells already at or over threshold at the edge. Termination does not depend on
> `tau`: a fired cell reports `inf` for the rest of the boundary, so the drain runs at most
> once per membrane. Regression: `tests/test_boundary_edge_crossing.py`.
>
> All three goldens are **unchanged** — no existing topology ever had a crossing at exactly
> `1.0`, so the change is inert outside this case.
>
> | two-patch regime, per 1000 boundaries | before | after |
> | --- | --- | --- |
> | C spikes | 0 | **500** |
> | feedback resets | 0 | **4000** |
> | L1 winner boundaries | 2000/2000 | **1500/2000** |
> | C cells with runaway retained charge | 2 (`V ≈ 780 000`) | **0** |
>
> The `C → I` call consequently went from *absent* to *net-effective*: the confirmed columns
> now lose a quarter of their output boundaries to top-down suppression.

The original diagnosis is retained below, because the mechanism recurs wherever drive lands
exactly on `theta`.

**Symptom: the top-down C/feedback path was completely silent in the exact-two-patch
regime.**

Measured over 1000 boundaries after a 2500-boundary warm-up:

| regime | L2 spike `tau` | C deposits at `tau=1.0` | C spikes | feedback resets | C cells with runaway charge |
| --- | --- | --- | --- | --- | --- |
| one patch | `0.0` ×305 | 0 / 305 | 83 | 664 | 0 |
| two patches | **`1.0` ×1000** | **2000 / 2000** | **0** | **0** | **2** |

Mechanism, confirmed by direct trace:

1. The `theta/2` detector ceiling means two coordinated child identity events deliver
   **exactly `theta`**. With `leak_rate = 0` a reset L2 detector therefore crosses at
   exactly `dtau = theta / theta = 1.0`.
2. Apical permission reaches the child C at that same `tau = 1.0`, and the C's deposit
   commits there.
3. The sub-boundary event loop runs `while current_tau < 1.0`, so a **dependent** crossing
   at exactly `1.0` can never be scheduled in that boundary.
4. The next boundary clears `coincidence_active`, and a closed gate correctly forbids firing
   from retained charge — so the C accumulates without bound. Observed after 3500
   boundaries: `L1c00C.V = 774340.2`, `L1c22C.V = 783253.9`, against `theta = 1000`.

This is an interaction between the **mandated** `theta/2` ceiling and the half-open `[0,1)`
event window, exposed — not caused — by this topology; the classic `tiled_cc` column does
not hit it (923 C spikes, `V = 0.0` under the same protocol) because its `Eor` relay shifts
the L2 crossing off the boundary edge.

The loop bound was the correct place to fix it: adjusting the cap, leak or threshold to move
the crossing off `1.0` would have been result-driven tuning of the science, whereas the
`[0, 1)` window was simply excluding a crossing the scheduler was already willing to serve.
The original measurement is retained as `probe_e_boundary_edge_anomaly`.

### 8.1a C basal learning was 40× slower than the E pool — **FIXED (two causes)**

Even with C firing, its basal weight matured far too slowly to act as a confidence signal.
Two independent causes, measured on the two-patch surface:

1. **A display-geometry penalty.** The multi-basal C's eight basal afferents inherited the
   `1/d²` learning-rate factor from the layout's **display** ring (`TILE_E_RING_R` places the
   eight ordinary E on a circle while C sits off to one side), giving `phi` a spread of
   **0.214 – 1.000** across siblings. A column's C therefore matured at a rate decided by
   *which ring seat its owner happened to occupy* — for `L1c00`, owner `E4` meant `phi =
   0.237`, i.e. 24 % of the rate an `E1` owner would get. The one-basal column never had
   this: its single `Eor → C` edge was its own reference, so `phi = 1.0` by construction.
   Column-local basal edges now carry **no** distance penalty, restoring that invariant.
2. **The rate itself.** `c_eta` was 2.0.

| | E bank reaches its `θ/2` ceiling | C basal reaches 95 % of `θ` |
| --- | --- | --- |
| before (ring penalty, `c_eta=2`) | ~400 | **never within 8000** (20 000 to reach `θ`) |
| ring penalty removed, `c_eta=2` | ~400 | 3481 |
| ring penalty removed, `c_eta=5` | ~385 | 1593 |
| **ring penalty removed, `c_eta=16` (new default)** | ~403 | **711** |
| ring penalty removed, `c_eta=25` | ~408 | 469 |

`c_eta = 16.0` is now the dashboard default (slider max raised 5.0 → 25.0). It is chosen so
C finishes **shortly after** E rather than 40× later, and deliberately **not** faster than E:
C confirms an owner the pool has already settled on, so a C that matured first would be
confirming a winner that has not stabilized.

### 8.1b–d Cadence, loop latency and input pacing — **moved**

These findings turned out to be general to the whole tiled family and to the engine's
presentation model, not specific to this topology. The full record — including the
`period = 2 × latency` law, the `tiled_cc_double_eor` falsification probe, the
`L % input_period == 0` rule, and the resolution — now lives in
**`docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`**.

What is specific to *this* topology:

* removing `Eor` shortens the loop from latency 3 to **latency 2**, the structural floor
  (`E → L2` costs one boundary, and the confirmation lands at `tau = 1.0` after its column's
  E has already fired at `0.667` — measured 450/450 — so it cannot suppress the boundary it
  arrives in);
* at `input_period = 1` that gives period-4 `1001` on **16/16 seeds** — grouped, but unlike
  `tiled_cc` it is seed-independent, because the period-2 alias requires **odd** latency;
* at the derived pacing `input_period = 2` it gives strict `1010` on **12/12 seeds**.

An earlier draft of this report claimed classic `tiled_cc` "reaches exact `1010` alternation"
at `c_eta = 16`. That was measured on seed 1 only and was an aliasing artifact; see the
cadence document for the correction.


### 8.2 One-patch sparse evidence

With a single active patch the L1 column wins 92 % of boundaries but L2 only ~30 %. A lone
identity afferent is capped at `theta/2` and cannot cross alone; with zero leak it crosses
by *temporal* accumulation over ~3 boundaries. Sparse-evidence behavior is therefore
temporal, not spatial, in this configuration (see §5).

### 8.3 Owner assignment is order-determined, not pattern-determined

See §7B. No pattern→neuron capacity claim is made.

## 9. Regression

Full suite **598 passed** (2026-07-28). `rg_coincidence_baseline.json` is byte-unchanged
(the one-basal C path is untouched). `tiled_cc_baseline.json` /
`tiled_cc_l1_4_baseline.json` carry only the earlier Eor-relay change from the preceding
task, not this one. `git diff --check` clean;
`node --test tests/replay.parser.test.mjs` passes.

One genuine regression was caught **by the goldens** during this work and fixed: keying the
new per-`(target, source)` basal distance map in source-first order made every basal
distance-factor lookup silently fall back to `1.0`, changing `rg_coincidence` C learning.

## 10. Dashboard

Selector entry **“Tiled CC Direct Identity · 9×9 · 8 E/column”**. Applying it rebuilds the
graph and wipes learned state, like any topology change. It is **not** the startup preset.
The inspector shows a multi-basal C's full basal vector — one row per local ordinary-E
source with its own weight, and the causal / current / carried source marked per boundary —
alongside the existing distinct apical diagnostics. No Eor node or weight is shown, and
identity edges keep their own source ids.
