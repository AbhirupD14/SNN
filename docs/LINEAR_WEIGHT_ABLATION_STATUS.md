# Linear Weight-Update Ablation — Status (authoritative)

This document supersedes all earlier layered conclusions. It records the ablation
study, the evidence-cleanup pass, and the **production promotion** of the linear-bounded
ordinary E/L2E rule.

> **Production defaults were intentionally changed by this package.** The ordinary
> accumulating E/L2E learning rule is now **linear-bounded** by default (the
> `1 − (w/w_max)²` multiplier is removed from the default ordinary E update). This was
> done only after the 32/32 fresh-seed confirmation. C learning is unchanged.

## Authoritative conclusions
- **Ordinary E/L2E → linear-bounded (new production default).** Validated on **32/32
  fresh seeds** (2001–2032) against all five success gates; selected as the default.
- **E hard cap: retained.** Under linear E the cap can become active — seed **2005**
  reached the cap (maxW 500) — so the cap remains a necessary safety bound.
- **E weight floor: retained at 0.** Unchanged.
- **C quadratic term: retained.** Removing it degraded the column novelty window
  (turnover 1.00→~0.6; incumbent held ~1200 vs ~156 boundaries).
- **C temporal cap: retained.** Cap-free C's one-event safety margin trends toward zero
  (→ 0.0224 at 20,000 events), **although no explicit invariant failure occurred** in
  that horizon (never fired from rest, never reached w₁, cadence intact, stayed finite).
  Retained on the projected-risk basis, not an observed failure.
- **FE retained** for both E and C.
- **Composition:** on the plus-sign union each learned primitive is marginally
  sub-threshold (≈θ), and the carried-state single-finite winner is a residual-membrane
  recency artifact. Both learned-weight marginality and membrane recency contribute; the
  experiment **does not by itself prove k-WTA is required.**

## New production E equation (default)
```text
FE   = theta - sum_j(w_j)
s_i  = +1 if afferent i participated, else -1
dw_i = eta * FE * s_i * influence_i
w_i <- clip(w_i + dw_i, 0, w_max)          # E cap retained, floor 0
```
Removed from the default: `1 - (w_i / w_max)^2`.

## Unchanged C equation (quadratic-bounded, default)
```text
FE_C     = theta - w_basal
dw_basal = eta_C * FE_C * apical * (1 - (w_basal / w_C_max)^2) * s_basal * influence_basal
w_basal <- clip(w_basal + dw_basal, 0, w_C_max)   # C temporal cap retained
```

## Headless modes (available for experiments/regression; not dashboard controls)
- `e_weight_update_mode`: `quadratic_bounded` (historical E rule) | **`linear_bounded`
  (new production default)** | `linear_nonnegative` (cap-free diagnostic only).
- `c_weight_update_mode`: **`c_quadratic_bounded` (production default, unchanged)** |
  `c_linear_bounded` | `c_linear_nonnegative` (cap-free diagnostic only).
- Neither is exposed in the dashboard `CONFIG_SPEC`; the dashboard inherits the engine
  production defaults. `use_fe` stays True; `e_weight_floor` stays 0.

## Factorial conditions (names fixed)
- **A** = historical quadratic-E baseline (`quadratic_bounded` / `c_quadratic_bounded`).
- **B** = linear-bounded E (`linear_bounded` / `c_quadratic_bounded`) — **now the
  promoted production rule.**

## Evidence trail (condensed)
- **Phase 0** — mode machinery + opt-in instrumentation; baseline fingerprints locked
  (seeds 1–8 + 4083693835); deterministic; instrumentation byte-identical when off.
- **Phase 1** — exact algebra for every E/C mode.
- **Phase 2 (isolated)** — linear E converges faster, never diverges; cap binds only for
  ≤1-active-pixel input; all C modes keep a perfect two-event cadence over 20k events;
  cap-free C climbs to 999.98 (margin → 0.0224) with no rejection observed.
- **Phase 3 (17-seed exploratory factorial)** — E soft term does not affect turnover;
  the C soft term controls the novelty window; cap-free C over-matures.
- **32-fresh-seed confirmation (A vs B)** — `confirm_ab_32seeds.json`; both pass all gates
  32/32; turnover/recovery 1.0; deterministic; linear-E reaches the cap on seed 2005.
- **Phase 4 (composition, 2 probes)** — controlled-state: no owner finite (each ≈θ);
  carried-state: only the recent column owner finite, evaporating under membrane washout.

## Fixed experiment config (frozen)
`topology=rg_coincidence, l2_init_total_frac=0.95, eta=0.01, c_eta=0.001, leak_rate=0,
refractory_steps=0, e_weight_cap=500`. Schedule: row1 2500 → col1 2500 → row1 2500.
Exploratory seeds 1–16 + 4083693835; confirmation seeds 2001–2032.

## Promotion pass — files changed & test totals
Default promotion (`quadratic_bounded` → `linear_bounded` for ordinary E/L2E):
- `snn/neurons.py` — `ExcitatoryNeuron` constructor default `update_mode='linear_bounded'`;
  docstrings/mode comment updated.
- `backend/simulation.py` — `DEFAULTS['e_weight_update_mode']='linear_bounded'`; enriched
  `_crossing_capture` (adds `g_inh`).
- `tests/golden/{pi,old,rg,rg_residual}_baseline.json` — **regenerated** to the new
  linear-E production default (verify **bit-exact**, frames + topology).
- `tests/test_excitatory_neuron.py` — historical-quadratic test made explicit; added
  `test_default_update_is_linear_bounded_exact`, `test_historical_quadratic_mode_exact`.
- `tests/test_linear_ablation.py` — default-mode test → linear; added
  `test_engine_default_e_equals_explicit_linear_bounded` + composition-probe tests.
- `experiments/linear_ablation.py` — tightened composition probe (learning disabled +
  weight-freeze asserted; first-driven-boundary capture; `v_after_drive` →
  `projected_uninhibited_end_v`; A/B condition labels).
- Docs: `Current_Implementation_Methodology_Equations.md`,
  `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md` — production E equation updated
  (C equation preserved).
- **Unchanged (diff-clean vs HEAD):** `backend/dashboard_config.py`,
  `tests/test_serialization_api.py`.

**Test totals:** full suite **317 passed, 2 warnings**; goldens verify **bit-exact**;
`git diff --check` clean.

## Remaining uncertainty
Composition: the plus-union primitives are θ-pinned (Σw→θ) and each marginally
sub-threshold; whether k-WTA (or another mechanism) is required is **not** settled here.
Cap-free E (E-cap removal) and cap-free C robustness remain **unconfirmed** (projected
risks, not observed failures).
