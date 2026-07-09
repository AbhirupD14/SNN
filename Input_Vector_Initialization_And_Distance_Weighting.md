# Input Vector Initialization And Distance Weighting

## Purpose

This note captures options for managing afferent weight initialization and a proposed distance-weighted input rule.

The goal is not to externally assign labels or winners. The goal is to reduce unlucky seed dependence, improve finite-time consolidation, and keep the mechanism compatible with local/hardware-mappable computation.

## Current Direction Update

The vector-aware weight-initialization path is now de-scoped as the next lever.
Keep the empirical results below as evidence, but do not adopt
`sparse_normalized` as the standing default and do not continue optimizing
`ff_init` schemes for this phase.

Also do not use per-step membrane noise as the symmetry breaker for this phase:
`membrane_noise=0.0` should stay fixed in the distance experiments. The next
experiment should ask whether deterministic geometry and per-synapse distance
attenuation can change competition without adding random charge to the membrane.

The active plan is:

```text
baseline:
  ff_init = uniform
  membrane_noise = 0.0
  distance_weighting = off

distance ablation:
  ff_init = uniform
  membrane_noise = 0.0
  distance_weighting = on
  stored weights update normally
```

Distance should be tested as a local signal-dissipation factor:

```text
effective_weight_ji = w_ji / d_ji^2
charge_j += signal_i * effective_weight_ji
```

For inhibitory synapses, the same rule means the delivered discharge is
attenuated by distance:

```text
effective_inhibition_ji = |w_ji| / d_ji^2
V_j = max(V_j - effective_inhibition_ji, rest)
```

The stored synaptic weight remains the learned state. Distance changes what is
delivered through the synapse, not the initial stored weight distribution.

## Framing

Each neuron's afferent weights can be viewed as a vector:

```text
w_j = [w_j0, w_j1, ..., w_jn]
```

For an input vector:

```text
x = [x_0, x_1, ..., x_n]
```

the usual charge contribution is approximately:

```text
input_j = sum_i x_i * w_ji
```

If two neurons begin with very similar weight-vector directions, they may compete for the same pattern for a long time. Uniform random initialization does not guarantee useful separation between neurons. A better initialization can try to give neurons different starting directions without hard-coding which neuron owns which pattern.

## Initialization Options

### 1. Uniform Random

Current/simple baseline.

```text
w_ji ~ Uniform(low, high)
```

Pros:

- simple
- unbiased in an obvious way
- easy to reproduce with seeds

Cons:

- can produce similar afferent vectors
- can create seed luck
- total incoming weight may vary between neurons unless normalized

### 2. Random Normalized

Sample positive weights, then normalize every neuron's afferent vector to the same total incoming weight.

```text
w_j <- random_positive_vector()
w_j <- target_sum * w_j / sum(w_j)
```

Pros:

- removes total-weight luck
- competition depends more on vector direction than magnitude
- preserves positive weights
- simple first ablation

Cons:

- neurons can still start with similar directions

### 3. Random Normalized With Diversity Rejection

Sample normalized positive vectors, but reject a new vector if it is too similar to an existing neuron's vector.

Similarity can be measured by cosine similarity:

```text
cos(w_a, w_b) = dot(w_a, w_b) / (||w_a|| * ||w_b||)
```

Procedure:

```text
for each neuron:
    sample positive random vector
    normalize to target_sum
    if cosine_similarity_to_any_existing_vector > max_similarity:
        resample
```

Pros:

- reduces duplicate receptive fields
- reduces unlucky seeds
- does not assign labels or pattern ownership
- works with positive-only weights

Cons:

- adds an initialization-time global check
- must choose a similarity threshold

This was the original strongest near-term candidate, but the empirical ablation
below showed it was inert in this positive-only 9-D setting.

### 4. Sparse Random

Each neuron starts with only a few afferents above floor; the rest start near floor.

```text
choose k afferents
set chosen afferents to small positive values
set all others near floor
```

Pros:

- gives neurons more distinct initial directions
- helps avoid all neurons receiving nearly equal charge
- resembles immature sparse connectivity

Cons:

- if too sparse, some patterns may have no reachable owner
- may bias toward partial features

### 5. Sparse Normalized

Sparse random initialization plus normalization to a fixed total incoming weight.

Pros:

- combines directional diversity with equal total drive
- good ablation against dense random normalized initialization

Cons:

- choosing sparsity `k` matters

### 6. Tiled Sparse Coverage

Ensure every input dimension has at least one neuron that begins slightly sensitive to it.

Pros:

- avoids dead input dimensions
- improves initial coverage
- does not necessarily assign complete patterns

Cons:

- more engineered than random initialization
- must avoid becoming supervised pattern assignment

### 7. Orthogonal Or Near-Orthogonal Initialization

Try to initialize afferent vectors so pairwise dot products are small:

```text
dot(w_a, w_b) approx 0
```

Pros:

- maximizes separation between neurons
- directly attacks duplicate starting receptive fields

Cons:

- true orthogonality often requires negative components
- positive-only weights make exact orthogonality difficult
- may be too artificial for the biological/hardware story

For this project, a nonnegative approximate version is more appropriate than strict orthogonality.

### 8. Low-Discrepancy Initialization

Use quasi-random sequences such as Sobol, Halton, or Latin hypercube sampling to cover the afferent-vector space more evenly than ordinary random sampling.

Pros:

- reduces seed variance
- scales better than naive rejection sampling

Cons:

- more implementation complexity
- may be unnecessary for the current 8-input experiments

### 9. Developmental Broad-Then-Prune

Start with broad weak connectivity and let local plasticity specialize:

```text
active afferents potentiate
inactive afferents depress
unused weights drift toward floor
```

Pros:

- matches the signed-spike learning direction
- avoids hard-coded owners
- can become a developmental story for scaling

Cons:

- still needs sufficient initial diversity and threshold reachability

## Archived Initialization Ablations

The experimental suite should compare:

```text
uniform_random
uniform_random_normalized
sparse_random
sparse_random_normalized
diversity_rejected_normalized
nonnegative_orthogonal_approx
low_discrepancy_normalized
```

Metrics:

- time-to-stable ownership
- ownership consistency
- owner collisions
- seed variance
- dead L2E/L2I/L1I count
- final pairwise RF cosine similarity
- initial pairwise RF cosine similarity
- weight saturation percentage
- performance across blocked, interleaved, mixed, and long-dwell input regimes

The key question:

```text
Which initialization reduces seed dependence and improves finite-time consolidation without smuggling in labels?
```

These ablations have been run and are no longer the active direction. See
`Empirical Results -- Initialization Ablation` below. The useful finding was
that sparse normalized init improved tiling but did not produce stable
ownership. The next phase returns to plain uniform initialization so distance is
isolated cleanly.

## Distance-Weighted Input Rule

Proposed rule:

```text
charge_j = sum_i signal_i * w_ji * (1 / d_ji^2)
```

where:

- `w_ji` is the afferent weight from input/source `i` to neuron `j`
- `d_ji` is the distance between source `i` and target neuron `j`
- `signal_i` is the input signal, such as `+1` active and `-1` inactive, or spike/no-spike depending on layer

This turns each synapse into:

```text
effective_weight_ji = w_ji / d_ji^2
```

The neuron then receives:

```text
charge_j = sum_i signal_i * effective_weight_ji
```

## Why Distance Might Help

Distance weighting introduces a spatial prior:

- nearby inputs matter more
- far inputs matter less
- receptive fields become local by default
- neurons are encouraged to specialize in spatial neighborhoods

This is biologically and hardware plausible if physical wire distance has cost or attenuation.

For robotics and neuromorphic hardware, this is attractive because physical locality matters:

```text
short wires = stronger/faster/cheaper
long wires = weaker/slower/more expensive
```

## Important Design Choice: What Is Distance?

Distance must be defined from the geometry of the layer.

For a 2D input grid:

```text
d_ji = distance(input_pixel_position_i, neuron_position_j)
```

For the next experiment, do not use an artificial `max(d, d_min)` floor. Instead
make the functional geometry guarantee nonzero distance. L1E neurons remain on
the 3x3 input layer; L2E neurons live on a separate lateral layer/plane and get
small deterministic jitter so their distances to the L1E pixels are varied but
not chaotic.

```text
d_ji^2 = (x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2
attenuation = 1 / d_ji^2
```

Possible distance metrics:

- Euclidean distance
- Manhattan distance
- Chebyshev distance
- graph distance over local connectivity

Euclidean is the first candidate because it directly matches the physical
dissipation story. Manhattan can be tested later if Euclidean helps.

### L2E Functional Placement

Do not use a perfectly symmetric L2 grid where every L2E has nearly the same
distance profile to L1E. Also do not scatter the L2E pool across a wide random
area. Use a compact lateral layer with controlled jitter:

```text
L1E functional positions:
  fixed 3x3 grid

L2E functional positions:
  same general lateral layer
  compact centered scaffold or ring
  fixed z gap from L1E
  seed-deterministic xy jitter
  bounded xy spread
```

The purpose of the jitter is only to make the L1E-to-L2E distance profiles
different enough for competition to see geometry. It is not a noise source during
spiking and it should not encode the eight target patterns.

## Locality Implications

Distance weighting is compatible with locality if each synapse stores or derives its own attenuation:

```text
synapse stores:
    weight
    distance_attenuation
```

Then runtime integration is local:

```text
charge += signal * weight * attenuation
```

No global computation is needed during inference or learning.

Initialization-time geometry can be global, but runtime must remain local.

## Scale Concerns

The rule changes the effective scale of incoming charge.

If weights are capped at:

```text
w_cap = threshold / 3
```

then the effective contribution is:

```text
w_cap / d^2
```

This may make far afferents too weak to matter and nearby afferents dominate.

Because this phase uses direct `1 / d^2` rather than a runtime distance floor,
the functional coordinate system is part of the experiment. Choose the L1/L2
spacing and L2 jitter so the attenuation matrix is not pathological. The
nearest useful synapses should have `d^2` on the order of `1`, and the run should
print attenuation min/mean/max before interpreting any ownership result.

The implementation needs to decide whether caps apply to:

1. raw stored weights
2. distance-adjusted effective weights

Option A:

```text
stored_weight <= w_cap
effective_weight = stored_weight / d^2
```

This preserves the existing cap but reduces total drive.

Option B:

```text
effective_weight <= w_cap
stored_weight <= w_cap * d^2
```

This lets distant synapses compensate for distance but weakens the locality prior.

For the first experiment, prefer Option A because it makes physical distance meaningful.

## Learning With Distance

The signed-spike update currently acts on the stored weight:

```text
dw_i = eta * p * (1 - (w_i / w_cap)^2) * signal_i
```

With distance weighting, there are two options.

### Option 1: Learn Stored Weight Only

```text
effective_input_i = signal_i * w_i * attenuation_i
dw_i = eta * p * (1 - (w_i / w_cap)^2) * signal_i
```

Pros:

- simple
- distance affects charge, not plasticity
- weight floor/cap logic stays unchanged

Cons:

- distant active inputs learn as strongly as near active inputs even though they contribute less charge

### Option 2: Distance-Scaled Learning

```text
effective_input_i = signal_i * w_i * attenuation_i
dw_i = eta * p * attenuation_i * (1 - (w_i / w_cap)^2) * signal_i
```

Pros:

- nearby inputs both drive and learn more strongly
- stronger locality prior

Cons:

- far synapses may never learn enough
- can make receptive fields too local

For the first ablation, use Option 1. Then test Option 2 separately only if
charge-only attenuation produces useful competition changes.

## Potential Benefits

Distance weighting may:

- reduce duplicate global receptive fields
- improve local specialization
- make hardware mapping more natural
- reduce long-range interference
- create a built-in spatial bias without labels
- help scaling to larger input spaces

## Potential Risks

Distance weighting may also:

- make distant but important features too weak
- make row/column/global line patterns harder to learn
- over-bias local patches instead of extended structures
- require retuning thresholds
- interact poorly with the `threshold / 3` weight cap rule
- reduce order robustness if early local patches dominate

This is especially important for the current 8-line task, because row and column patterns are spatially extended. A strong distance penalty could make neurons learn local fragments instead of whole lines.

## Suggested Ablations

Compare:

```text
no_distance_weighting
distance_euclidean_inverse_square
distance_manhattan_inverse_square
distance_euclidean_inverse
distance_local_radius_only
```

For each, test:

```text
stored_weight_cap
effective_weight_cap
learn_stored_weight_only
distance_scaled_learning
```

Metrics:

- time-to-stable ownership
- ownership consistency
- row/column/diagonal success separately
- final RF spatial extent
- dead neuron count
- weight saturation
- pairwise RF similarity
- dependence on seed

## Recommended First Distance Experiment

Start with:

```text
initialization = uniform
membrane_noise = 0.0
distance_weighting = off
```

Then compare against:

```text
initialization = uniform
membrane_noise = 0.0
distance_weighting = euclidean_inverse_square
distance_affects_learning = false
weight_cap_mode = stored_weight_cap
```

Run the comparison in scopes:

```text
1. L1E -> L2E feedforward only
2. L1E -> L2E feedforward + L2I -> L2E inhibition
3. all E/I synapses that have defined functional positions
```

Scope 2 is the competition-critical condition: it tests whether distance-scaled
inhibitory discharge changes the round-robin rather than only changing
feedforward affinity.

Measure:

- sustained dominance
- distinct modal owners / 8
- dead L2E
- firers per sustained hold
- L2I spike rate and L2I->L2E discharge event count
- peak L2E membrane charge relative to threshold
- row, column, and diagonal results separately
- attenuation matrix min/mean/max so scale mistakes are visible

## Bottom Line

Weight initialization is no longer the active plan. The next plan is to test
distance-weighted signal dissipation as a deterministic local factor in
feedforward drive and then in L2 competition.

Distance weighting remains an ablation, not an assumed improvement. It may help
competition by giving different L2E neurons genuinely different geometric
affinities, but it may also fragment the extended row/column/diagonal patterns.

## Empirical Results — Initialization Ablation (2026-07-08)

Implemented in `weight_init.py`, selected via `SimulationEngine(ff_init=..., ff_init_kw=...)`.
Measured with the honest sustained-presentation metric (`ablation_harness.py`),
against the current default regime (signed-spike / no weight budget), 3 seeds.
Distance weighting was left OFF, per "Recommended First Experiment".

```text
scheme              init_cos  sustained_dom  distinct/8   dead   final_cos
uniform (baseline)  0.90      0.552 +-0.05   4.00 +-0.82  0.33   0.41
uniform_normalized  0.90      0.548 +-0.07   3.67 +-0.94  0.00   0.44
sparse (k=3)        0.44      0.167          1.33         6.67   0.44   (catastrophic)
sparse_normalized   0.44      0.433 +-0.02   5.67 +-0.47  0.00   0.25   <-- best tiling
diversity (<=0.8)   0.82      0.498          3.67         0.00   0.35
orthogonal          0.84      0.500          3.67 +-0.94  0.00   0.40
low_discrepancy     0.89      0.525          3.67         0.00   0.40
```

Answering the note's key question ("which initialization reduces seed dependence
and improves finite-time consolidation without smuggling in labels?"):

- **Sparsity is the only effective lever; dense direction-diversity is not.**
  Dense positive vectors are inherently ~0.9-aligned in the 9-D positive orthant,
  so diversity-rejection, near-orthogonal, and low-discrepancy can only pull the
  initial pairwise cosine down to ~0.82–0.89 — and at that level they move
  nothing (distinct, dominance, seed variance all ≈ baseline). Only sparsity
  breaks through to ~0.44, and only there does tiling improve. This overturns the
  note's prior ranking of diversity-rejection as the strongest candidate.
- **`sparse_normalized` is the winner.** It lifts distinct tiling 4.0 → 5.7 with
  0 dead, gives the most decorrelated final receptive fields (final cosine 0.25),
  and the tightest seed variance on dominance (±0.02 vs ±0.05). It trades ~0.12
  sustained dominance for that — the margin-vs-distinctness frontier, resolved
  toward distinctness.
- **Normalization is load-bearing for sparse init, inert for dense.** Raw
  `sparse` is catastrophic (6.67 dead — the note's "too sparse → no reachable
  owner"); renormalizing each afferent vector to a common total restores
  reachability. For dense init, normalization changes magnitude but not
  direction, so `uniform_normalized` ≡ `uniform`.
- **`k` sweep confirms k=3 is a real optimum,** not task-length matching: distinct
  = {k2: 4.67, k3: 5.67, k4: 4.0, k5: 3.67}, and k=2 (which does NOT match the
  3-pixel line length) still beats baseline — it is the sparsity effect, not a
  smuggled-in label.
- **Initialization improves tiling, not holding.** NO scheme reached stable
  ownership (`time_to_stable` never triggered); the round-robin under sustained
  presentation is a competition/discharge problem (see `AGENT_HANDOFF.md` §5–6),
  orthogonal to initialization. Init is a variance/tiling lever, not the fix for
  one-to-one holding.

Superseded recommendation: do not adopt `ff_init='sparse_normalized'` as the
standing default for the next phase. Use uniform initialization and
`membrane_noise=0.0`, then measure distance weighting directly. Given that the
task patterns are spatially extended lines, expect distance to help locality at
the risk of fragmenting extended-line ownership; measure before adopting.
