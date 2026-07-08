# Integer Fixed-Point Scale for Gates, Leaks, and Thresholds — Change Note

Branch: `feature/inhibitory-plasticity`. The default model now runs at an
**integer fixed-point scale**: thresholds are the integers `1000` / `8000`, and
every charge-like magnitude (membrane potentials, weights, gates, clips, budgets)
is scaled by `UNIT`. Leak rates stay dimensionless ratios stored as integer
numerators over `LEAK_SCALE`. The rescale is a pure change of units, verified to
reproduce the prior dynamics **exactly** (identical spike/winner/event counts;
gate magnitudes simply ×1000).

## Fixed-point scale

Defined once in `neuron_flexible.py`, imported everywhere:

```python
UNIT = 1000          # charge scale: 1 old unit == UNIT fixed-point units
LEAK_SCALE = 1000    # leak rate stored as an integer numerator over LEAK_SCALE
```

Helpers: `to_units`, `to_float`, `leak_num`, `div_round`.

## Scaling rules (by role)

The LIF + plasticity math is scale-covariant under `x → UNIT·x` for charge
quantities, provided each constant scales by the correct power:

| Role | Scale | Examples |
|---|---|---|
| Linear charge (threshold, potential, weight, gate init, clip, budget, floor) | `× UNIT` | threshold, `L2_GATE_INIT`, `ff_weights`, `weight_cap`, `homeo_budget_min` |
| Learning rate η (dw must scale with w) | `× UNIT` | `L2_GATE_ETA`; L2E/L1I/L2I η = `frac × weight_cap` auto-scale |
| Quadratic denominator `w_max` in `dw = η·p·(1−w²/w_max)` (w² scales `× UNIT²`) | `× UNIT²` | `L2_GATE_WMAX`, L2E/L1E `excitatory/inhibitory_saturation_cap` |
| Dimensionless ratios (leak, learning fractions, `ca_*`, homeo steps, threshold fracs) | unchanged | `leak_l2`, `ETA_FRAC`, `ca_target`, `ei_sat_mult` |

## Every old default → new integer/scaled equivalent

| Constant / param | Old | New | Meaning |
|---|---|---|---|
| `threshold` | `1.0` | `1 * UNIT` = **1000** | integer threshold |
| `threshold_l2` | `8.0` | `8 * UNIT` = **8000** | integer threshold |
| `weight_cap` | `1.0` | `1 * UNIT` = 1000 | linear clip |
| `leak_l1 / leak_l2` | `0.10 / 0.01` | `100 / LEAK_SCALE`, `10 / LEAK_SCALE` | ratios (unchanged value) |
| `L2I_LEAK_RATE / L1I_LEAK_RATE` | `0.07` | `70 / LEAK_SCALE` | ratios |
| `L2_GATE_INIT` | `-0.5` | `-500` | linear gate magnitude |
| `L2_GATE_WMAX` | `1.5` | `1500 * UNIT` = 1.5·UNIT² | **quadratic** w_max; gate settles at √ ≈ 1225 (< 8000) |
| `L2_GATE_ETA` | `0.1` | `100` | η, scales with charge |
| `L2_EI / L1_EI init fracs` | `0.25 / 0.5` | `1/4 / 1/2` | fractions of (scaled) threshold |
| `L2E_MIN_WEIGHT_FLOOR` | `0.01` | `10` | linear weight floor |
| `homeo_budget_min` | `0.5` | `500` | linear; `_max = 2*thr_l2` auto-scales |
| `ff_weights` init | `(0.05, 0.20)` | `(50, 200)` | linear weights |
| L1E encoder weights | `[-1.0, 1.0]` | `[-1, 1] * UNIT` | one pixel spike delivers UNIT charge |
| L1E gate `inhibitory_weight_cap` | (None→cap) | `UNIT * weight_cap` | UNIT² quadratic denom (gate stays frozen at saturation) |
| L2E `excitatory_saturation_cap` | (None→cap) | `UNIT * weight_cap` | UNIT² quadratic denom |
| `l2i/l1i_threshold_frac`, `ei_sat_mult` | `1.0` | `1` | dimensionless multipliers |
| `stimulate` default magnitude (api) | `1.0` | `1 * UNIT` | one threshold-unit nudge |

**Not scaled** (out of the gates/leaks/thresholds scope, dimensionless): base
`learning_rate`, `ETA_FRAC`, `ca_rate`/`ca_target`, `homeo_up/down`.

The E→I saturation cap `weight_cap² * ei_sat_mult` needs no edit — `weight_cap`
is `× UNIT`, so its square is already `× UNIT²`.

## Leak equation (unchanged from the first pass)

Leak is stored as the integer numerator `_leak_num` over `LEAK_SCALE`
(`leak_rate` is a derived property). The update is
`V += _leak_num*(rest−V)/LEAK_SCALE`; ratios are scale-invariant.

## Is model state integer now?

- **Thresholds and gate init/cap are integers** (`1000`, `8000`, `-500`, …), and
  `potential >= threshold` compares against an integer.
- **Potentials and weights are float-valued at the ×UNIT scale** (e.g. a live
  `L2E` potential ≈ 6717 climbing toward 8000). They are held as float, not
  `int` dtype, on purpose: the plasticity arithmetic (quadratic saturation,
  budget renormalization, homeostatic scaling) stays in float so the tuned
  dynamics are reproduced bit-for-bit rather than drifting under integer
  truncation. `div_round` remains available for a future integer-dtype pass.

## Verification — dynamics preserved exactly

`test_l2_competition.train_and_measure`, scaled vs. the pre-scale baseline, on
both resource-regulator paths:

| Path | seed | fired | winners | L2I spikes | gate discharges | gate mag |
|---|---|---|---|---|---|---|
| homeostasis=False | 1 / 2 / 3 | 3 / 4 / 3 | 2 / 3 / 3 | 350 / 277 / 341 | 2450 / 1939 / 2387 | 1.225 → **1224.7** |
| homeostasis=True  | 1–4 | 8 / 8 / 8 / 8 | 6 / 5 / 6 / 4 | — | — | — |

Every discrete count is identical; only gate magnitudes scale ×1000. Integer
event counts would diverge instantly if any scaling exponent were wrong.

## Test results

- `test_neuron.py`, `test_inhibitory_plasticity.py`, `test_refractory_gating.py`,
  `test_8line_consolidation.py` — **PASS** (bare-neuron tests use their own
  natural-scale explicit values and the unchanged formula).
- `test_l2_competition.py` — its `distinct_fired >= 4` assert **fails at 3/8**.
  This is **not** caused by the rescale (the unscaled baseline is also 3/8): it
  is due to the `homeostasis: bool = True → False` engine-default change made
  outside this task. With `homeostasis=True` (the regime the test was written
  for) the scaled engine gives 8/8 and passes. Left as-is per that change being
  intentional.

## Notes

- The bare `Neuron` / `InputLayer` / `CorticalColumn` class defaults remain
  scale-agnostic (`threshold = 1000/UNIT = 1.0`); the `SimulationEngine` chooses
  the ×UNIT scale and passes explicit values, so those defaults never reach the
  dashboard model.
- Dashboard `activation` values are `potential/threshold` ratios, so they stay in
  `[0, 1]` and the visualization is unchanged.
