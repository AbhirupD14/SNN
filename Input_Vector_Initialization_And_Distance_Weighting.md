# Input Vector Initialization And Distance Weighting

## Purpose

This note captures options for managing afferent weight initialization and a proposed distance-weighted input rule.

The goal is not to externally assign labels or winners. The goal is to reduce unlucky seed dependence, improve finite-time consolidation, and keep the mechanism compatible with local/hardware-mappable computation.

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

This is the strongest near-term candidate.

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

## Recommended Initialization Ablations

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

Use a minimum distance floor to avoid division by zero:

```text
d_eff = max(d_ji, d_min)
```

Then:

```text
attenuation = 1 / d_eff^2
```

Possible distance metrics:

- Euclidean distance
- Manhattan distance
- Chebyshev distance
- graph distance over local connectivity

Euclidean or Manhattan are the simplest first candidates.

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

For a first ablation, use Option 1. Then test Option 2 separately.

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

## Recommended First Experiment

Start with:

```text
initialization = random_normalized_with_diversity_rejection
distance_weighting = off
```

Then compare against:

```text
initialization = random_normalized_with_diversity_rejection
distance_weighting = euclidean_inverse_square
distance_affects_learning = false
weight_cap_mode = stored_weight_cap
```

This isolates whether distance-weighted input helps or hurts without changing too many variables at once.

## Bottom Line

Vector-aware initialization is the safer immediate direction for reducing seed dependence.

Distance-weighted summation is promising for locality and hardware scaling, but it changes the effective input geometry and may hurt extended patterns if applied too strongly.

Treat distance weighting as an ablation, not as an assumed improvement.
