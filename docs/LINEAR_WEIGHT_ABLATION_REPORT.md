# Linear Weight-Update Ablation — Report (promotion + validation)

Question: can the per-synapse `1 − (w/w_max)²` multiplier be removed from the E/L2E and
C accumulating rules? FE (`θ − Σw`) is deliberately preserved.

> **Production decision (implemented by this package):** the ordinary **E/L2E accumulating
> rule was intentionally changed to `linear_bounded`** — the `1 − (w/w_max)²` multiplier is
> removed from the default ordinary E update. This was done only after a 32/32 fresh-seed
> confirmation. The **E hard cap and zero floor are retained**; the **C basal rule is
> unchanged** (still quadratic-bounded, with its temporal cap). The old (`quadratic_bounded`)
> and cap-free (`linear_nonnegative`) modes remain **headless** for regression/experiment.
> The dashboard control surface did **not** change (it inherits the new engine default).

### Exact equations
```text
OLD production E (now historical `quadratic_bounded`):
    dw_i = eta * (theta - sum_j w_j) * s_i * influence_i * (1 - (w_i/w_max)^2)
    w_i <- clip(w_i + dw_i, 0, w_max)

NEW production E (default `linear_bounded`):
    dw_i = eta * (theta - sum_j w_j) * s_i * influence_i          # no (1-(w/w_max)^2)
    w_i <- clip(w_i + dw_i, 0, w_max)                             # E cap + floor 0 retained

C basal rule (UNCHANGED, default `c_quadratic_bounded`):
    dw_b = eta_C * (theta - w_b) * apical * (1 - (w_b/w_C_max)^2) * s_b * influence_b
    w_b <- clip(w_b + dw_b, 0, w_C_max)                          # C temporal cap retained
```

## Reproduction
```
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 0        # baseline fingerprints
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 2        # isolated E + C (20k)
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 3        # 17-seed factorial (explore)
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py confirm  # 32 fresh-seed A/B confirm
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 4        # composition (2 probes)
```
→ `experiments/linear_ablation/{phase0_baseline,phase2_isolated,phase3_factorial,
confirm_ab_32seeds,phase4_composition}.json`.
Fixed config: `rg_coincidence, l2_init_total_frac=0.95, eta=0.01, c_eta=0.001,
leak_rate=0, refractory_steps=0, e_weight_cap=500`. Schedule: row1 2500 → col1 2500 →
row1 2500. **17-seed exploratory** = seeds 1–16 + dashboard 4083693835.
**32-seed confirmation** = fresh seeds **2001–2032** (disjoint from the above).
Determinism verified (`replay_ok=True`) in every phase.

## Scope & dashboard
The **engine E/L2E default changed** (`quadratic_bounded` → `linear_bounded`). The
**dashboard control surface did not change**: `backend/dashboard_config.py` is diff-clean
vs HEAD (full 24-control surface), so `e_weight_floor=1.0` is not set in
`DASHBOARD_OVERRIDES` and `c_fe_enabled`/`e_weight_floor`/update-modes are **not** exposed
in the dashboard. The dashboard simply inherits the new engine default. The update-modes,
`e_weight_floor=0`, and `c_fe_enabled=True` remain **headless** engine parameters; C stays
`c_quadratic_bounded`.

## The four effects the study separates
1. **Neuron-wide FE** (`θ−Σw`) — preserved in all modes; a dynamic coupled scalar.
2. **Per-synapse quadratic slowdown** `1−(w/w_max)²` — the ablation target.
3. **Final hard clip at `w_max`** — isolated by `linear_bounded` vs `linear_nonnegative`.
4. **C's one-coincidence-subthreshold / two-coincidence-fire margin** — the C modes.

Throughout, an **observed** invariant failure (it actually happened in a run) is
reported separately from a **projected** long-horizon risk (a trend that has not yet
produced a failure within the simulated horizon).

## Evidence

### Isolated trajectories (Phase 2; 10k E events, 20k C events)
- **Linear E converges faster, never diverges.** 3-of-9 to 90/95/99% of θ: 131/168/260
  (quadratic) vs 111/133/186 (linear). Zero overshoot, zero negative FE, zero NaN in
  every E run.
- **Cap binds only for very sparse input.** 1-of-9: bounded modes stall at ½θ; cap-free
  reaches θ and self-limits there via FE (finite).
- **All three C modes keep a perfect two-event cadence over the ENTIRE 20k-event
  trajectory** (`fires[i]==i%2`, 0 mismatches). Weight behavior differs:
  `c_quadratic` self-limits at 549–550, `c_linear_bounded` clips at 550,
  **`c_linear_nonnegative` climbs to 999.98** (w₁=1000). For cap-free C the
  `min_safety_margin` (w₁ − w) reaches **0.0224**, but `first_one_from_rest`,
  `first_reach_w1`, `first_exceed_2θ`, `first_nonfinite` are all **None** →
  **no rejection condition was observed**; only a margin-→0 trend.

### Network factorial (Phase 3, 17-seed exploratory; `replay_ok=True`)
Means over 17 seeds. `col turnover` = fraction of seeds where the column owner differs
from the row owner (novelty window open); `last incumbent` = last col-phase boundary the
row owner won; return recovery = 1.00 for every condition.

| Cond | E / C mode | col turnover | last incumbent | max Cw | max E w | aborts |
|---|---|---:|---:|---:|---:|---:|
| A | quadratic / c_quadratic | 1.00 | 158 | 545 | 389 | 0 |
| B | **linear** / c_quadratic | 1.00 | 156 | 545 | 425 | 0 |
| E | linear_nonneg / c_quadratic | 1.00 | 156 | 545 | 425 | 0 |
| C | quadratic / **c_linear** | 0.65 | 1135 | 550 | 367 | 0 |
| D | linear / c_linear | 0.59 | 1216 | 550 | 380 | 0 |
| F | linear_nonneg / c_linear | 0.59 | 1216 | 550 | 380 | 0 |
| G | quadratic / **c_linear_nonneg** | 0.65 | 1135 | 849 | 367 | 0 |
| H | linear_nonneg / c_linear_nonneg | 0.59 | 1216 | 845 | 380 | 0 |

The E soft term does not affect turnover (A/B/E identical). **The C soft term controls
the novelty window**: dropping it drops column turnover 1.00→~0.6 and the row incumbent
holds to ~1200 (vs ~156) — an **observed** network degradation. Cap-free C over-matures
to ~845 (toward w₁) with the same novelty-window loss.

### 32 fresh-seed confirmation of A vs B (`confirm_ab_32seeds.json`; `replay_ok=True`)
Success gates applied per seed (row_dom≥0.90, col_dom≥0.80, ret_dom≥0.80, col≠row owner,
ret==row owner):

| Cond | gates passed | mean dom (row/col/ret) | col turnover | return recovery | neg-FE | overshoot | cap hits | aborts |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A | **32/32** | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 | 0 | 0 | 0 (0/32 seeds) | 0 |
| B | **32/32** | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 | 0 | 0 | 679 (**1/32 seeds**) | 0 |

Linear E (B) confirms cleanly: all gates pass on all 32 fresh seeds, turnover and
recovery undegraded, deterministic. **New nuance:** on 1 of 32 fresh seeds a linear-E
individual weight reaches the cap (maxW 500), whereas quadratic E always self-limits
below it (maxW ≤ 447, 0 cap hits). Linear E does not "dominate by clipping," but the hard
cap is an **occasionally-active** bound under it.

### Composition readiness (Phase 4; tightened 2-probe, 8 seeds)
`_crossing_capture` records each L2E's boundary-start V (pre-drive), frozen drive, `g_inh`,
refractory, counterfactual crossing, and finite status. **Learning is disabled during both
probes** (E/L2E + C weights frozen and asserted byte-identical afterward); both capture the
**first plus-driven boundary** (not a later one). **carried-state** = real recent membrane;
**controlled-state** = diagnostic membrane washout (each L2E `V` and `g_inh` zeroed per plus
boundary, learned weights preserved; production untouched).

| Cond | probe | row finite | col finite | both finite |
|---|---|---:|---:|---:|
| A, B | carried | **0/8** | 3–4/8 | 0/8 |
| A, B | controlled | 0/8 | **0/8** | 0/8 |
| D, F | carried | 1/8 | 4/8 | 1/8 |
| D, F | controlled | 0/8 | **0/8** | 0/8 |

**Controlled-state: no owner is ever finite in any condition** — each specialist's plus-union
weight sum is ≈θ (≈999.9 < 1000), so neither primitive fires alone from rest (a weight-budget
effect of the FE pin, Σw→θ). **The row owner is never finite even in carried-state**; only the
column owner is ever finite, and only when its residual membrane (from just winning the column
phase) is high enough at the first plus-driven boundary — which **evaporates under membrane
equalization** (controlled: 0/8). So the composition limitation is **both** learned-weight
marginality **and** residual-membrane recency; the carried-state single-finite cell does **not**
by itself establish that k-WTA is required. (The tightened probe shifts the carried col_finite
count from the earlier 8/8 to 3–4/8 by capturing the first driven boundary; the qualitative
conclusion is unchanged.)

## Decisions (per term; observed vs projected labelled)

1. **E quadratic term — REMOVED (promoted to production).** `linear_bounded` passes all
   five success gates on **32/32 fresh seeds** (2001–2032), converges materially faster,
   leaves turnover and recovery undegraded (1.0/1.0), is deterministic, and shows no
   negative-FE oscillation or overshoot. *Observed* nuance: it makes the hard cap an
   occasionally-active bound (**seed 2005** reached the cap). **Implemented:** the ordinary
   E/L2E default is now `linear_bounded`; the four legacy preset goldens were regenerated
   to the new (bit-exact) production behavior.

2. **E hard cap — NOT confirmed; keep for now.** Cap-free E (`linear_nonnegative`) was only
   run in the 17-seed exploratory pass, where it never exceeded the cap. It was **not**
   confirmed on the 32 fresh seeds. Moreover, because linear E already leans on the cap on
   some seeds, the cap is **load-bearing** under the recommended linear rule — so removing
   it is a *distinct* change, not a free rider on the E-quadratic decision. Do not remove
   the E cap without its own 32-fresh-seed confirmation (and never substitute a hidden
   ceiling). This is a *not-yet-established* result, not a failure.

3. **C quadratic term — keep.** `c_linear_bounded` preserves the isolated one-event-
   subthreshold margin and full-trajectory two-event cadence (0 mismatches, margin 450),
   **but** *observably* degrades the network column novelty window (turnover 1.00→~0.6;
   incumbent holds to ~1200 vs ~156). The quadratic slow-down near the cap is what keeps
   the novelty window open.

4. **C temporal cap — keep (projected risk, not an observed failure).** Over a 20k-event
   isolated run, cap-free C (`c_linear_nonnegative`) **never fired from rest, never reached
   w₁, kept a perfect cadence, and stayed finite** — *no rejection condition was observed.*
   However the basal weight asymptotes to 999.98 and the one-event safety margin trends to
   **0.0224 → 0**, so cap-free C has **not** been established as robust and leaves no usable
   temporal margin. Recommendation: keep the C cap because the margin trends to zero and
   cap-free behavior is unproven — **this is a projected long-horizon risk, not an observed
   invariant failure.** (This corrects the earlier report, which wrongly implied a rejection
   condition occurred.) Removing the C cap would require a temporally-derived equilibrium
   equation, not a change to threshold/leak/refractory/WTA.

5. **Composition readiness.** Learned primitives are distinct latency cells, but on the
   plus-union each delivers only ~θ (marginally sub-threshold) — a weight-budget effect of
   the FE pin (Σw→θ). The carried-state "single finite winner" is a membrane-recency
   artifact that disappears under membrane equalization. So both learned weights and
   residual membrane contribute; this diagnostic does **not** on its own prove k-WTA is
   required. It does establish that later composition work must contend with θ-pinned,
   marginally-sub-threshold primitives, and should keep the C soft term.

**Bottom line:** ordinary E/L2E is now **linear-bounded in production** (confirmed on 32
fresh seeds). The **E cap is retained** (cap-free E unconfirmed and the cap is load-bearing
under linear E). Both the **C quadratic term** (observed network novelty-window degradation
without it) and the **C temporal cap** (margin trends to zero; cap-free robustness unproven —
a projected risk, not an observed failure) are **retained**. C behavior is byte-unchanged.

## Confirmation checklist (this package)
- Old E equation and new E equation: see "Exact equations" at the top; the `1 − (w/w_max)²`
  factor no longer appears in the default ordinary E update.
- C equation unchanged (still quadratic-bounded with its temporal cap).
- **Dashboard control surface unchanged** — `backend/dashboard_config.py` diff-clean vs HEAD.
- **E cap active, C quadratic active, C cap active** — all retained and tested.
- Headless modes retained: `quadratic_bounded` (historical E), `linear_nonnegative` (cap-free
  E diagnostic), `c_linear_bounded` / `c_linear_nonnegative` (C diagnostics).
- **Test totals:** full suite **317 passed**; 4 legacy goldens regenerated (bit-exact) to the
  new production default; `git diff --check` clean.
- **Remaining uncertainty (composition):** the plus-union primitives are θ-pinned and
  marginally sub-threshold; whether k-WTA (or another mechanism) is required is *not* settled
  by this study.

## Deviations from the original plan
- The plan's Phase 2/3 C **abort** conditions were replaced by **first-occurrence recording
  over the full horizon** (FE bounds cap-free C, so there is no runaway to clamp). This is
  strictly more informative and lets observed-vs-projected be distinguished. All abort
  fields for the completed runs are `None` (no condition occurred).
- Cap-free **E** was not extended to the 32 fresh seeds (the confirmation was scoped to the
  recommended A/B change). The E-cap recommendation reflects this: not confirmed.
- Instrumentation (`record_updates`, enriched `_crossing_capture`) is opt-in and
  byte-identical when off (goldens + explicit tests). WTA, inhibition, event ordering,
  init, learning rates, thresholds, leak, refractory, durations, and topology unchanged.

## Reproduction (commands)
```
# experiments
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 0
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 2
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 3
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py confirm     # fresh seeds 2001-2032
PYTHONPATH=. .venv/bin/python experiments/linear_ablation.py 4           # tightened composition probe
# regenerate the production goldens (they encode the new linear-E default)
for t in pi old rg rg_residual; do PYTHONPATH=. .venv/bin/python tests/golden_topology.py capture ${t}_baseline $t; done
# tests
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q                        # 317 passed
git diff --check                                                         # clean
```
`experiments/linear_ablation.py 3` re-runs the 17-seed factorial with **explicit** per-
condition modes (A=historical quadratic E, B=promoted linear E), so it is independent of the
engine default. The four `tests/golden/*_baseline.json` fingerprints now encode the new
linear-E production default (bit-exact).
