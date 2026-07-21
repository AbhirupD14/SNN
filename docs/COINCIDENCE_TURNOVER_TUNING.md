# Coincidence Turnover: L2 Initialization and C-Cell Tuning

## Outcome

The `rg_coincidence` circuit now uses two calibrated values:

| Parameter | Working value | Role |
| --- | ---: | --- |
| `l2_init_total_frac` | `0.95` | Common initial L2E afferent total as a fraction of `theta` |
| `eta` | `0.01` | Ordinary E/L2E accumulating-weight learning rate |
| `c_eta` | `0.001` | C-cell basal-weight learning rate |

This combination produced stable row ownership, a different winner after switching
to an intersecting column, and recovery of the original winner when the row returned
in all 8/8 tested seeds. Reversing L2 scheduler order produced the same paired outcome
in 8/8 seeds with zero L2 latency ties.

These values solve two distinct problems. L2E needed to begin close enough to threshold
for a rival to integrate the novel pattern before the incumbent repeatedly reset it.
C cells needed to learn more slowly so the already-trained shared feature was suppressed
while the novel C branches remained immature long enough for that rival to take over.

## Original failure mode

The pre-tuning latency competitors inherited the ordinary excitatory initialization:

```text
mean weight = 0.55 * theta / 9
weight_i    = mean weight * Uniform(0.96, 1.04)
```

With `theta = 1000`, the mean individual weight was about `61.1`, and a neuron's
nine-afferent total was approximately `550`. The independently jittered totals were
also slightly unequal.

After 2500 row boundaries, the winning L2E cell had strengthened its three active row
afferents substantially. On the intersecting column, suppression of the shared center
pixel was real, but a naive rival still accumulated too slowly. The incumbent normally
won again and immediately reset every rival. Because only the L2 winner learns, the
losers could not escape that loop.

Increasing every individual weight directly to its cap was not a valid solution:

- nine weights at `500` give an afferent total of `4500`, far above `theta = 1000`;
- `FE = theta - sum(w)` becomes negative, reversing the signed update;
- the nonlinear term `1 - (w/w_max)^2` is zero at the cap, so capped weights cannot
  learn; and
- identical capped vectors replace seeded competition with stable-order symmetry.

The useful quantity was therefore the **total initial afferent weight relative to
threshold**, not the individual per-synapse cap.

## Change 1: normalized L2 initialization

For latency-WTA L2E cell `j`, the engine first draws the ordinary seeded narrow-jitter
vector `r_j`. It then rescales that vector to a common total:

```text
T = rho * theta

w_j,i(0) = r_j,i * T / sum_k(r_j,k)
```

where `rho = l2_init_total_frac` and `0 < rho < 1`.

The implementation includes a cap-aware proportional fill for custom graphs. For the
built-in coincidence topology, every L2E cell has nine afferents and reaches `T`
exactly. A custom bank too small to represent `T` cannot exceed
`n_afferents * w_max`.

At the working values:

```text
theta                         = 1000
rho                           = 0.95
sum_i w_j,i(0)                = 950 for every L2E j
initial FE_j                  = theta - 950 = 50
mean individual weight        = 950 / 9 = 105.56
individual weight cap         = 500
```

This gives every competitor the same small, positive initial free energy. It does
**not** make all synapses equal: normalization preserves the seeded within-row direction,
so the +/-4% variation can still break symmetry and produce different winners across
seeds. It only removes accidental total-magnitude advantage between competitors.

The normalized policy is scoped to `e_latency_competitor`. Ordinary PI, old, RG, and
residual competitors retain their historical per-afferent initialization.

## Change 2: slower C-cell basal learning

A C cell owns one learned basal weight. Its apical inputs remain unweighted Boolean
gates. On a causal C spike, the implemented update is:

```text
FE_C = theta - w_b

dw_b = c_eta
       * FE_C
       * A
       * (1 - (w_b / w_b,max)^2)
       * s_b
       * phi_b

w_b <- clip(w_b + dw_b, 0, w_b,max)
```

where:

- `A` is 1 only when an apical event is active on the causal boundary;
- `s_b` is the active basal signal;
- `phi_b` is the normalized inverse-square basal distance influence; and
- `w_b,max` is the C-specific basal cap derived from threshold and leak.

Originally, `c_eta` resolved to the shared excitatory rate `eta = 0.01`. That made the
C branches mature quickly. During the column phase, the novel column C cells soon
learned enough to suppress their own novel evidence, shortening the useful asymmetry
between the already-trained center and the new arms.

We separated the rates:

```text
ordinary E/L2E eta = 0.01
C basal c_eta      = 0.001
```

The slower C update preserves the sequence needed for turnover:

1. Row training matures suppression of the row features, including the shared center.
2. The pattern switches to the intersecting column.
3. The trained center is suppressed while the two novel column arms initially pass.
4. An L2 rival integrates those novel events and begins winning.
5. The novel C branches continue learning, but slowly enough that they do not erase
   the evidence advantage before ownership changes.
6. When the original row returns, its previously learned L2 owner wins again.

No threshold, leak, refractory, apical weight, inhibition burst, or pattern-change
charge-wipe mechanism was added to obtain this result.

## Sweep protocol

The headless protocol was:

```text
row 1: 2500 boundaries
col 1: 2500 boundaries
row 1: 2500 boundaries
```

Pixels `3,4,5` form `row 1`; pixels `1,4,7` form `col 1`; pixel `4` is shared.
Seeds `1..8` were used for selection. Success required:

- a stable original row owner with at least 90% final-window dominance;
- a different column owner with at least 80% final-window dominance; and
- recovery of the original row owner with at least 80% final-window dominance.

### Stage 1: initial L2 total, with `c_eta = 0.01`

| `rho` | Column replacement | Full success |
| ---: | ---: | ---: |
| `0.55` | 0/8 | 0/8 |
| `0.70` | 0/8 | 0/8 |
| `0.80` | 0/8 | 0/8 |
| `0.90` | 1/8 | 1/8 |
| `0.95` | 4/8 | 4/8 |

Raising the initial total opened a path for rivals, but initialization alone was not
sufficiently robust.

### Stage 2: C learning rate, with `rho = 0.95`

| `c_eta` | Column replacement | Row recovery | Full success | Mean last incumbent column win |
| ---: | ---: | ---: | ---: | ---: |
| `0.001` | 8/8 | 8/8 | 8/8 | 183.4 |
| `0.0025` | 8/8 | 8/8 | 8/8 | 178.3 |
| `0.005` | 7/8 | 8/8 | 7/8 | 629.9 |
| `0.01` | 4/8 | 8/8 | 4/8 | 1521.6 |

Both `0.001` and `0.0025` passed all eight seeds. The sweep selected `0.001` by its
predeclared conservative tie preference for slower C maturation; the current evidence
does not establish a meaningful performance difference between those two values.

### Confirmation

At `rho = 0.95`, `c_eta = 0.001`:

- the first rival appeared after 6--11 column boundaries;
- the old owner's final column win occurred by boundary 224 in every seed;
- the final 500-boundary column window belonged exclusively to the new owner;
- novel pixels fired 99/100 early column boundaries, while the center fired 85--86;
- the original row owner recovered in all eight seeds;
- reversing L2 scheduler order reproduced every paired result; and
- no L2 tie event occurred.

The earlier problematic dashboard seed `4083693835` also succeeded outside the
selection set:

```text
row 1 owner      L2E2
col 1 owner      L2E4
returned row     L2E2
first rival      column boundary 9
last old win     column boundary 194
```

## Production and dashboard changes

The engine now exposes:

```text
l2_init_total_frac = 0.95
c_eta              = 0.001
```

Both parameters survive configuration rebuilds, Reset, and Reseed. The public topology
payload reports their resolved values.

The browser dashboard starts with:

```text
topology           = rg_coincidence
eta                = 0.01
c_eta              = 0.001
l2_init_total_frac = 0.95
leak_rate          = 0.0
refractory_steps   = 0
e_weight_cap       = 500
```

The Model Config panel exposes both tuned controls. The general-purpose
`SimulationEngine` still defaults to the PI topology for backwards compatibility;
the coincidence topology is the explicit dashboard startup preset.

## Reproduction and evidence

- Sweep driver: `experiments/coincidence_turnover_sweep.py`
- Complete per-seed results: `experiments/coincidence_turnover_results.json`
- Engine initialization: `backend/simulation.py`
- Dashboard preset and controls: `backend/dashboard_config.py`
- Regression tests: `tests/test_rg_coincidence.py` and
  `tests/test_coincidence_turnover_sweep.py`

To watch the protocol, run the dashboard at 120 steps/s and present `row 1`, `col 1`,
then `row 1` for approximately 2500 boundaries each.

## Scope of the result

This establishes the intended turnover for one intersecting row/column transition over
eight selection seeds, plus the historical dashboard seed. It does not yet establish
all-pattern robustness, long-run multi-pattern capacity, or aggregate L1E/RG frequency
halving. Those remain separate evaluation questions. A larger confirmation should use
fresh seeds and every ordered pair of intersecting patterns without retuning these
values.
