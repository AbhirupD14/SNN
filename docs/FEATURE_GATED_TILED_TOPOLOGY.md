# Feature-Gated Tiled Cortical-Column Topology (`tiled_cc_feature_gated`)

**Status:** implemented; local input-feature turnover restored and causally verified.
**Acceptance claim (scope):** restores `rg_coincidence`-style feature-specific turnover
inside the tiled L1 input layer. It does **not** claim hierarchical composition, multi-winner
Eor identity, or a continuous-time event contract.

## Why this preset exists

`rg_coincidence` turns over between overlapping patterns because it has a **feature-specific**
inhibitory microcircuit: each pixel/feature `i` owns a paired chain
`L1E_i → L1C_i → L1I_i → L1E_i`. When the current owner predicts feature `i` (apical) and
feature `i` is active (basal), **only** `L1E_i` is suppressed. `tiled_cc` replaced that with
one column `C → one column I` that hard-resets the **entire** ordinary-E bank, which erases
feature identity and (as separately measured) removes turnover.

`tiled_cc_feature_gated` re-inserts the small circuit **once per feature inside every L1
recognition module**. It is structurally the `rg_coincidence` microcircuit relabelled and
relocated:

| feature-gated node | rg_coincidence analogue |
| --- | --- |
| feature relay `S[k]` (`e_pretrained`) | `L1E_i` (pretrained relay) |
| ordinary competitor `E[j]` (`e_latency_competitor`) | `L2E_j` |
| feature coincidence `C[k]` (`e_coincidence`) | `L1C_i` |
| feature inhibitory `If[k]` (`i_relay`) | `L1I_i` |

No new neuron class or edge kind was added; the engine, neuron equations, and scheduler are
unchanged. The variant is selected by validated metadata (`topology.variant='feature_gated'`),
never a preset name or id prefix.

### Why passive relays alone would not give selectivity

Fixed feature relays merely *copy* their pixel — every active relay fires every boundary, so a
bare relay layer cannot silence an *explained* feature while leaving *novel* features active;
it has no notion of "predicted". Selectivity requires the paired **coincidence gate**: `C[k]`
fires only when its paired relay `S[k]` supplies basal eligibility **and** a mature local owner
supplies apical permission (`E[j] → C[k]`). Only then does `C[k] → If[k] → S[k]` reset that one
relay. Novel-feature relays, whose `C` has no mature apical prediction, keep firing and recruit
a different competitor. The gate is what converts a passive copy into a *predicted-feature*
suppressor.

## Exact graph contract

Per L1 recognition module (nine of them, one per `3×3` patch): eight ordinary competitors
`E`, one `Eor`, one **WTA-only** `I`, and nine feature gates (`S`/`C`/`If`). The top L2 is a
plain WTA bank (eight `E`, `Eor`, WTA `I`, **no** `C`). Edges, per feature `k` / competitor
`j`:

```
RGC[k] -> S[k]     pretrained_excitation      S[k] -> Eor  (none; Eor fed by E)
S[k]   -> E[j]     feedforward (plastic, E-owned)
S[k]   -> C[k]     basal_excitation (single learned basal)
E[j]   -> C[k]     apical_excitation (Boolean permission, LOCAL owner)
C[k]   -> If[k]    relay_excitation
If[k]  -> S[k]     hard_reset_inhibition (paired; resets ONLY S[k])

E[j]   -> Eor      feedforward           E[j] -> Iwta relay_excitation
Iwta   -> E[j]     hard_reset_inhibition (WTA-only; resets exactly the E bank)
L1 Eor -> L2 E[j]  feedforward           (no L2 -> L1 apical feedback)
```

Invariants enforced by `_validate_tiled_feature_gated`: `If[k]` targets exactly `S[k]`;
`Iwta` targets exactly the eight competitors; no ordinary E drives a feature If; no feature C
drives Iwta; feature-I and WTA-I are distinct nodes with **disjoint** reset targets; each
feature C has one paired basal and exactly eight local apical sources; and there is **no** old
column-wide `C → shared I → whole-bank` reset.

### Counts (default 8/8)

```
81  RGC sources
81  feature relays S + 81 feature C + 81 feature If
9 * (8 ordinary L1 E + 1 Eor + 1 WTA I)  = 90
1 * (8 ordinary L2 E + 1 Eor + 1 WTA I)  = 10
= 424 neurons

81  RGC->S      648 S->E        81  S->C basal     648 E->C apical
81  C->If       81  If->S       72  E->Eor         72  E->Iwta   72 Iwta->E
72  L1 Eor->L2 E   8 L2 E->Eor    8 L2 E->Iwta      8 L2 Iwta->E
= 1932 edges
```

## Timing (copied from the small circuit; nothing tuned)

A feature relay is a fixed **pretrained** packet (`pretrained_exc_margin = 1.05θ`), so it would
cross at `τ ≈ 1/1.05 ≈ 0.952`. A mature ordinary competitor is a cap-free one-event integrator
(FE budget `1.10θ`) and crosses at `τ ≈ 0.909`. On a suppressing boundary the mature owner
`E[j]` fires at `0.909`, its zero-latency apical opens `C[k]`, `C[k]` deposits and fires at the
same `0.909`, and `If[k]` hard-resets `S[k]` at `0.909` — **before** the relay's `0.952`
crossing. The reset discards the relay's frozen `1.05θ = 1050` drive packet, so `S[k]` does not
fire that boundary. Novel relays (no mature `C`) keep firing. This is the identical
`1.05 vs 1.10` race that gives `rg_coincidence` its halving; equal-time behaviour remains the
scheduler's stable-node-order tie handling (logged, not overridden).

## Causal acceptance results (seed 1, reference protocol dwell 4000)

`experiments/feature_gated_turnover.py` — reference settings shared with
`microcircuit_turnover.py` (`leak_rate=0`, `refractory_steps=0`, `eta=0.01`, `c_eta=0.005`,
`l2_init_total_frac=0.95`); no model parameter tuned.

**Stage A — one active RF (module `L1m11`), canonical `row 1 → col 1 → diag \ → diag /`:**

| pattern | owner | dom | turnover | early center-S | early novel-S | pre-spike resets |
| --- | --- | --- | --- | --- | --- | --- |
| row 1 | E2 | 1.00 | — | 168 | 168 / 168 | 0 (first) |
| col 1 | E6 | 1.00 | ✓ | 146 | 173 / 173 | 54 |
| diag \ | E7 | 1.00 | ✓ | 148 | 175 / 175 | 52 |
| diag / | E1 | 1.00 | ✓ | 146 | 174 / 174 | 54 |

Four distinct consolidated owners; turnover at every switch; center relay suppressed vs both
novel relays every switch; both novel relays stay active; every counted feature reset is
paired/local and **pre-spike** (captured trace on each switch: owner τ = C-deposit τ = If-reset
τ = `0.909`, discarded drive `1050`, center relay did not fire); no feature-I/WTA-I cross-talk;
and the eight blank neighbouring modules record **zero** winners and **zero** resets.

**Stage B — same pattern in all nine RFs (seed 1):** every one of the nine L1 modules
independently produces four distinct local owners with turnover at every switch, owner indices
differing across modules, and **no** feature gate inhibits another patch
(`no_cross_patch_inhibition`). L2 receives all nine Eor streams; no L2 composition/identity
claim is made.

Durable artifacts (replay/metrics/summary, gitignored under `experiments/runs/`) are written by
the experiment; the short single-seed regression is `tests/test_feature_gated_turnover.py`.

## Retained limitations (do not over-claim)

- **Single Eor per L1 module** intentionally discards local winner identity; the L2 bank has no
  C and this variant makes no composition claim. This isolates and tests input-feature turnover
  only.
- The `1.05 / 1.10` latency margin remains a diagnostic race, not a scale-independent timing
  invariant (see `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`, P0 predictive-inhibition
  timing). The event-conserving suppression contract is still open.
- No four-competitor feature-gated variant is added yet (deferred until this eight-competitor
  variant is reviewed).
