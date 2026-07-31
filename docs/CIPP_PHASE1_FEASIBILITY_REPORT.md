# CIPP Phase 1: continuous-competitor feasibility report

**Date:** 2026-07-31
**Scope:** `prompts/Claude_NEST_CIPP_Semantic_Repair_Prompt.md` Phase 1 — the two-competitor
feasibility spike only. The canonical topology was deliberately **not** translated.
**Reproduce:** `.nest-env/bin/python experiments/nest_cipp_phase1_feasibility.py`

## Verdict

**Feasible on candidate 1.** `iaf_psc_exp_ps`, a built-in NEST precise-spike model, recovers
the membrane-latency race under uniform conduction delay with no geometry and no jitter. All
**six** pass gates hold. Candidates 2 (NESTML continuous) and 3 (custom C++ extension) were
**not needed and were not built** — the brief's candidate order stops at the first that
passes.

The single qualification, and it is a real one: hard single-winner WTA holds only inside a
**measured operating envelope**, not absolutely. That is reported as an envelope rather than
asserted as a property, per the brief.

## The question this answers

The impulse prototype applies each arrival as an instantaneous jump and fires in the receipt
handler, so simultaneous deltas cross on the same step. The current backend therefore invents
per-edge delay dispersion to produce an order, which silently replaces the decision rule:

```
CIPP / Python oracle:  earliest crossing under total frozen drive
impulse prototype:     earliest weighted prefix sum in DELAY order
```

Phase 1 asks whether total supported drive alone can decide, with delay held uniform.

## Circuit

Two competitors on one shared synchronous source. Identical conduction delay to both; only
the total weight differs. An inhibitory relay closes an explicitly **nonzero** E→I→E loop —
NEST has no zero-delay synapse and this spike does not pretend otherwise.

| | |
|---|---|
| model | `iaf_psc_exp_ps` — exact subthreshold integration between events, off-grid crossing by regula-falsi root finding ([NEST docs](https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_ps.html)) |
| competitors | `V_th = −55`, `E_L = −70`, `C_m = 250 pF`, `tau_m = 10 ms`, `tau_syn_ex = 2 ms` |
| source delay | **uniform**, identical for every equivalent afferent |
| geometry | **disabled** |
| jitter | **0.0 ms, everywhere** |
| E→I, I→E delays | nonzero, swept 0.01–0.1 ms |

## Gate results

### Gate 1 — stronger total drive fires first (uniform arrival at 11.0 ms)

| strong (pA) | weak (pA) | strong crosses | weak crosses | ordered |
|---|---|---|---|---|
| 8000 | 6000 | 11.5521 | 11.7885 | ✅ |
| 8000 | 7000 | 11.5521 | 11.6490 | ✅ |
| 12000 | 4000 | 11.3465 | 12.4139 | ✅ |
| 20000 | 19000 | 11.1990 | 11.2102 | ✅ |

Monotonic, and it discriminates a 5 % drive difference (20000 vs 19000) at 0.0112 ms.

### Gate 2 — scaling all conduction delays together does not reverse the winner

| source delay | latency, strong | latency, weak | strong first |
|---|---|---|---|
| 0.5 ms | 0.5521 | 0.7885 | ✅ |
| 1.0 ms | 0.5521 | 0.7885 | ✅ |
| 2.0 ms | 0.5521 | 0.7885 | ✅ |
| 4.0 ms | 0.5521 | 0.7885 | ✅ |
| 8.0 ms | 0.5521 | 0.7885 | ✅ |

**The strongest result in this report.** Across a 16× range of conduction delay the raw
latency after arrival varies by **1.78 × 10⁻¹⁵ ms** — the last float ULPs. Delay
contributes nothing to the decision; drive decides entirely. This is the exact inverse of the impulse prototype, where delay order
*is* the decision rule.

### Gate 3 — resolution convergence

Raw, unrounded crossing times for the strong competitor:

| h | strong |
|---|---|
| 0.1 | 11.552059910876435 |
| 0.05 | 11.552059910876435 |
| 0.01 | 11.552059910876435 |
| 0.005 | 11.552059910876435 |
| 0.001 | 11.552059910876**432** |

Spread across h: **3.6 × 10⁻¹⁵ ms** — the last two float ULPs, not zero. `iaf_psc_exp_ps`
integrates subthreshold dynamics exactly between events and finds the off-grid crossing by
**regula-falsi root finding**, so there is essentially nothing left for a finer grid to
improve. Winner identity is stable throughout.

The times are off-grid (`11.55205991…` is not a multiple of `0.1`), which is what
distinguishes this from the impulse model's grid-quantised firing.

This also removes the incentive that produced `h = 0.001` in the current A/B defaults:
shrinking `h` buys no crossing-time accuracy here. It does still matter for a different
reason — see *Choosing h* below.

### Gate 4 — single committed winner within a positive margin

Measured E→I→E loop latency: **0.2042 ms** (`d_ei = d_ie = 0.1`).

| strong | weak | gap (ms) | loop (ms) | margin (ms) | committed | outcome |
|---|---|---|---|---|---|---|
| 8000 | 6000 | 0.2365 | 0.2042 | **+0.0323** | 1 | ✅ single winner |
| 12000 | 4000 | 1.0674 | 0.2042 | **+0.8632** | 1 | ✅ single winner |
| 8000 | 7000 | 0.0970 | 0.2042 | −0.1072 | 2 | ⚠️ unresolved race |
| 8000 | 7900 | 0.0084 | 0.2042 | −0.1958 | 2 | ⚠️ unresolved race |
| 20000 | 19000 | 0.0112 | 0.2042 | −0.1930 | 2 | ⚠️ unresolved race |

Every positive-margin case commits exactly one winner. Every nonpositive case is **surfaced
as an unresolved race**, not silently counted as a win. NEST cannot retract an emitted spike,
so this is the strongest contract available and the brief asks for exactly it.

### Gate 5 — exact ties are reported, never broken

Equal drive (8000 vs 8000) crosses at an identical `11.552059910876435` on both
competitors. Classified `exact_tie`, two committed, **`winner = None`**.

This gate caught a real bug. `wta_outcome` originally took `min((time, index))`, which
resolves an exact tie by node index — reporting competitor 0 as the winner, a result
manufactured from creation order. That is the same class of error as manufacturing one
from delay jitter. `winner` is now `None` unless exactly one competitor was permitted to
fire at all, and the first-across cell is reported separately as a diagnostic.

> [!CAUTION]
> **THIS PHASE 1 RESULT IS NOT THE ORIGINAL CUSTOM ENGINE'S TIE-BREAKING POLICY.**
>
> The built-in NEST microcircuit has no explicit CIPP tie arbiter, so this feasibility spike correctly reports an exact tie as `winner = None`. The original custom engine instead arbitrates crossings within `1e-12` normalized time by stable repository node order and records the event in `latency_ties`. A later NEST phase must either implement that declared local rule explicitly or retain `unresolved_tie` as a documented profile difference; NEST GID/creation order must not decide it implicitly.

### Gate 6 — decomposition and creation order do not change the decision

The real mechanism sums many afferents; a single aggregate event never exercises that path.
8000 vs 6000 pA was rebuilt as **eight versus six synchronous 1000 pA afferents**, then
rebuilt again with the competitors created in the opposite order.

| construction | GID strong / weak | strong | weak |
|---|---|---|---|
| one aggregate event | 1 / 2 | 11.552059910876435 | 11.788528396999347 |
| 8 × 1000 vs 6 × 1000, own generators | 1 / 2 | 11.552059910876435 | 11.788528396999347 |
| same, reverse creation order | **2 / 1** | 11.552059910876435 | 11.788528396999347 |

Raw float equality, no rounding. Each afferent is its **own** synchronous spike generator,
so the multi-source summation path is genuinely exercised, and the roles actually swap
GIDs in the reversed build — competitors are created by separate `Create` calls and
addressed by logical role throughout.

An earlier version of this gate created both competitors in one `Create(model, 2)` call,
which left the strong cell on the lower GID in both builds: it reversed connection
insertion order only, and proved less than it appeared to. The decision depends on total
drive alone — not on event insertion order, not on node creation order, not on GID.

## Operating envelope — the input Phase 2 needs

The envelope is set by loop latency, and loop latency is tunable. A **lower** weak/strong
ratio means the two drives are further apart and is therefore the *easier* case, so the
quantity of interest is the **largest sampled ratio that still resolves**, bracketed against
the adjacent sample that does not:

| d_ei = d_ie | loop latency | largest resolvable | smallest unresolved |
|---|---|---|---|
| 0.1 ms | 0.2042 ms | 0.76875 | 0.775 |
| 0.05 ms | 0.1042 ms | 0.8625 | 0.86875 |
| 0.02 ms | 0.0442 ms | 0.9375 | 0.94375 |
| 0.01 ms | 0.0242 ms | **0.9625** | 0.96875 |

Sampling step 50 pA, so each boundary is an interval rather than a point. At
`d_ei = d_ie = h = 0.01 ms`, drives as close as **0.9625** resolve to a single winner and
**0.96875** does not.

`loop latency = d_ei + relay crossing + d_ie`. The relay's own crossing time is the floor
and cannot be removed; it was ~0.0042 ms here.

### Choosing h in Phase 2

Reducing `h` no longer buys crossing-time accuracy — gate 3 settles that. It still matters,
for a different reason: `h` is the floor on `d_ei` and `d_ie`, and those set the loop
latency that sets the envelope. **Phase 2 should choose `h` from the communication-delay
envelope it needs and the compute cost it can afford, not from numerical crossing
precision.**

## What this does and does not establish

**Established.** A supported, built-in NEST model reproduces the CIPP decision rule —
earliest crossing under total drive — with uniform delay, no geometry and no jitter, and
commits it through a nonzero inhibitory loop within a measured envelope.

**Not established, and out of Phase 1 scope.**

- Whether the canonical `tiled_cc` drive differences fall inside that envelope. Phase 2/3.
- Anything about `I_accq`, coincidence windows, prediction credits or learning. Those are
  Phases 3–6 and are untouched here.
- Whether `iaf_psc_exp_ps` can carry the causal-volley state the learning rule needs. It has
  no such variable, so Phase 3 must decide between extending it via NESTML and a custom
  model. **This is the most likely place the candidate order gets revisited** — passing the
  race gates does not mean the model can carry CIPP's learning state.

## Parameter honesty

The weights here (thousands of pA) are not the repository's `theta = 1000` charge units;
they are whatever drives this model's default 15 mV threshold gap. Phase 2 defines the
physical-time profile with units and one semantic role per parameter. Nothing in this spike
should be read as a proposed parameter set — only the *ordering properties* transfer.
