# Weight-Initialization Robustness: Options and Experimental Plan

## Decision Context

The current network can learn a held 3-pixel line, but whether it consolidates to
one L2E owner depends strongly on the initial feedforward weights. A favorable
draw gives one neuron an early lead. That neuron fires, potentiates, and pulls
away. An unfavorable draw leaves several neurons nearly tied; event-driven WTA
then rotates firing among them, each receives potentiation, and they converge to
the same capped receptive field.

Reseeding is therefore a useful diagnostic, not a learning mechanism. A robust
model should converge from a broad range of reasonable initial states, including
near-symmetric and deliberately identical feedforward weights. Biological
heterogeneity can select an initial leader, but the learning dynamics must
amplify and preserve that selection.

There are two related but distinct failures:

1. **Near-tied allocation:** several L2E neurons learn the same held pattern.
2. **Shared-pixel interference:** rows, columns, and diagonals share afferents, so
   a neuron trained on one pattern is already partially responsive to another.

No single initialization scheme resolves both. The first needs a robust
allocation/commitment mechanism; the second may require plasticity attached to
input conjunctions rather than independent pixel weights.

## Design Requirements

Any standing solution should:

- use only information locally available to neurons, synapses, or the existing
  inhibitory circuit;
- avoid labels, global assignment tables, and cross-neuron weight comparisons;
- tolerate ordinary random, adversarial near-tied, and symmetric initialization;
- concentrate learning in one owner without permanently silencing the pool;
- allow unused neurons to be recruited for later patterns;
- preserve learned ownership when a pattern is revisited;
- distinguish shared pixels by their co-active context; and
- improve honest sustained-presentation and interleaved metrics across seeds.

## Option 1: Sparse Normalized Initialization

Initialize each L2E toward a small, normalized subset of afferents, for example
three of the nine pixels.

```text
choose k candidate afferents per L2E
assign them equal or mildly jittered positive mass
set the remaining afferents near the positive floor
normalize every L2E to the same total initial drive
```

### Expected benefit

This reduces the probability that several neurons begin with the same response
to a line. It is a better developmental prior than unconstrained dense random
weights and has previously produced good participation with no dead units.

### Limitation

It only lowers the probability of failure. It does not make the dynamics
well-defined when two neurons receive equivalent subsets, and it can accidentally
seed task-shaped 3-pixel templates. It is best retained as an ablation or
developmental prior, not accepted as the primary solution.

### Verdict

Useful control; insufficient on its own.

## Option 2: Episode-Level Winner-Takes-Plasticity

The present WTA selects one firing neuron per competition step, but learning can
rotate among several winners over a sustained presentation. This option makes the
first eligible winner own the plasticity window for the episode.

```text
on presentation/episode start:
    allocation_open = true
    committed_owner = none

on the first eligible L2E spike:
    committed_owner = that L2E
    allocation_open = false

while the episode remains active:
    committed_owner may potentiate active feedforward inputs
    rivals cannot potentiate those inputs

on a genuine presentation transition or sufficient quiet interval:
    clear the commitment
```

The existing L2I discharge can carry the local "allocation closed" event. The
commitment should be represented as a decaying neuron-local eligibility state,
not as a global pattern-to-neuron table.

### Expected benefit

One early lead is amplified instead of being shared among later round-robin
winners. This directly targets the observed failure where several neurons reach
the same `[cap, cap, cap]` receptive field.

### Risks

- An arbitrary first spike can receive too much authority.
- A commitment window that is too short recreates rotating co-learning.
- A window that is too long can survive a pattern switch and corrupt the owner.
- Preventing rival potentiation without enabling later recruitment can leave
  unused units passive.

### Verdict

Highest-priority minimal experiment. It tests whether the initialization problem
is fundamentally a failure of WTA-for-learning.

## Option 3: Fast Winner Hysteresis

Give a recent winner a small, temporary excitability advantage so it continues
to win the current presentation.

```text
on L2E spike:
    fast_commit += commit_increment

effective_threshold = threshold - alpha * fast_commit
fast_commit *= fast_decay
```

Equivalent implementations could modulate gain or learning rate rather than the
threshold. The state must be bounded and must decay.

### Expected benefit

Tiny differences in weight, charge, or spike timing become decisive. After one
candidate wins, it pulls away before rivals can co-train to the cap.

### Risks

- Neuron-wide hysteresis may cause one L2E to capture multiple patterns.
- Excess hysteresis can turn an early accident into permanent ownership.
- It does not by itself distinguish a shared pixel in two different patterns.

### Verdict

Promising companion to winner-takes-plasticity, but unsafe as a standalone
assignment mechanism.

## Option 4: Slow Allocation Fatigue or Cooldown

Balance fast commitment with a slower negative eligibility trace. A recent owner
retains the current pattern over short timescales but becomes less likely to claim
unrelated patterns over longer timescales.

```text
on commitment:
    fast_commit += A
    slow_allocation_cost += B

effective_threshold = theta - fast_commit + slow_allocation_cost

fast_commit decays quickly
slow_allocation_cost decays slowly
```

### Expected benefit

This supplies recruitment pressure without scaling every synapse or using a
global owner count. Neurons that have not recently won remain more eligible for a
new input.

### Risks

- A neuron-level cooldown cannot reliably decide whether a new volley is another
  sample of the same pattern or a different pattern sharing pixels.
- Poorly separated time constants can either destroy persistence or fail to
  recruit unused neurons.
- Interleaved curricula can make a simple time-based rule order-dependent.

### Verdict

Appropriate second-stage allocation experiment after a commitment latch exists.

## Option 5: Dendritic or Conjunctive Commitment

Attach evidence and plasticity to a co-active input combination rather than to
each pixel independently. Each L2E would contain several local branches or
eligibility compartments. A branch produces a nonlinear event only when a
coherent subset of its afferents is active together.

```text
branch_drive_b = sum_i(w_bi * spike_i)
branch_event_b = branch_drive_b >= branch_threshold

if branch_event_b contributes to the somatic winner:
    consolidate only the participating branch/synapses
```

For example, the top-left pixel can participate in both a row branch and a column
branch without making the two patterns equivalent:

```text
{0, 1, 2} -> row-0 conjunction
{0, 3, 6} -> col-0 conjunction
```

Branches need not be assigned these patterns in advance. Candidate branches can
begin with overlapping sparse connectivity and organize through local
co-activity, branch plateaus, stabilization, and pruning.

### Expected benefit

This addresses the stage-4 failure where rows and columns first share pixels. It
provides a local representation of context: the meaning of a pixel depends on
which other inputs arrived with it.

### Risks

- More state and parameters: branch count, thresholds, connectivity, and branch
  plasticity timescales.
- Fixed task-shaped branches would merely hide initialization in the topology.
- Without branch competition, multiple branches can still learn the same
  conjunction.

### Verdict

Best candidate for the shared-pixel problem. It should follow, not replace, a
minimal allocation experiment, because branches also require winner selection.

## Option 6: Structural Plasticity

Allow persistent correlations to stabilize or grow connections while unused or
incoherent connections weaken and are eventually recycled.

```text
repeated local co-activity -> stabilize/grow synapse
persistent inactivity or mismatch -> prune synapse
pruned capacity -> reconnect to another reachable afferent
```

### Expected benefit

Sparse receptive fields and functional clusters emerge during learning instead
of being supplied by `sparse_normalized(k=3)` initialization. The initial graph
only needs enough redundant reachability for all pixels and neurons.

### Risks

- Slow and substantially more complex to test.
- Reconnection rules can introduce a hidden global search if not carefully local.
- Structural plasticity still needs competition to prevent duplicate clusters.

### Verdict

Biologically attractive long-term direction, but not the first diagnostic fix.

## Option 7: Microscopic Dynamic Noise

Use small membrane, release, threshold, or timing fluctuations to resolve exact
ties during allocation.

### Expected benefit

Exact symmetry becomes unstable even with identical weights. Unlike reseeding,
noise operates continuously and can represent ordinary neural variability.

### Limitation

Noise chooses a leader but does not create commitment. Without winner-takes-
plasticity or hysteresis, leadership can continue to rotate and the result remains
unstable. Large noise can also mask whether the deterministic learning rule works.

### Verdict

Valid tie stimulus for an adversarial symmetric test; not a solution by itself.

## Recommended Sequence

### Experiment A: Winner-Takes-Plasticity

Implement a configurable episode-level commitment trace, leaving the feedforward
weight equation unchanged. Compare:

1. baseline step-level WTA;
2. first-winner potentiation latch;
3. latch plus a small bounded winner-hysteresis trace.

Run each under normal random, adversarial near-tied, and identical initialization.
For the identical condition, inject one tiny transient perturbation, then remove
it after allocation. Stable ownership must not continue to depend on noise.

### Experiment B: Recruitment

If Experiment A reliably produces one held-pattern owner but interleaved patterns
collide, add slow allocation fatigue. Sweep only the ratio between fast commitment
and slow cooldown timescales; avoid broad parameter fishing.

### Experiment C: Shared-Pixel Binding

If disjoint patterns allocate cleanly but rows plus columns still collide, add a
small number of learnable conjunctive branches per L2E. Compare branches with and
without local branch competition. Do not pre-wire line templates.

### Experiment D: Structural Formation

Once branch-level learning is characterized, test whether local growth/pruning
can replace sparse candidate connectivity and reduce dependence on initial branch
wiring.

## Required Evaluation Matrix

Every candidate must be evaluated under the same initialization families:

| Initialization | Purpose |
|---|---|
| ordinary random | compatibility with the current model |
| many random seeds | estimate robustness rather than showcase a lucky run |
| adversarial near-tied | reproduce the observed co-owner failure |
| exactly identical | expose whether symmetry breaking is actually defined |
| sparse normalized | compare against the strongest initialization-only control |

Run the staged curriculum separately:

1. one held pattern;
2. two disjoint patterns;
3. all three rows;
4. rows plus columns, where shared-pixel interference begins;
5. all eight line primitives.

Report:

- sustained-presentation modal dominance;
- distinct modal owners per pattern;
- visit-to-visit owner consistency;
- winner rotation rate;
- number of co-specialists reaching near-cap receptive fields;
- dead or never-recruited L2E neurons;
- owner stability after learning/noise is disabled;
- row, column, and diagonal results separately; and
- mean and worst seed, not only the best run.

## Acceptance Criteria

An initialization-robust allocation mechanism should, at minimum:

- consolidate one held pattern to one owner for every tested seed;
- resolve adversarial near ties without multiple capped co-owners;
- resolve identical initialization after only a transient perturbation;
- retain ownership after that perturbation is removed;
- avoid increasing dead-neuron count;
- preserve distinct ownership for disjoint patterns; and
- improve rather than trade away interleaved distinctness.

A complete shared-pixel solution additionally needs clean allocation when rows
and columns are combined. Passing only the held/disjoint cases validates the
allocation mechanism, not feature binding.

## Recommendation

Do not adopt a more favorable initialization as the primary fix. Keep sparse
normalized initialization as a comparison condition. Implement episode-level
winner-takes-plasticity first, optionally followed by a small fast hysteresis
trace. Add slow allocation fatigue only if recruitment remains deficient.

If that stack succeeds for disjoint patterns but fails when patterns share
pixels, move commitment into learnable dendritic/conjunctive branches. This
separates the work into two falsifiable questions:

```text
Can one of several near-tied neurons become the sole learner?
Can learned ownership bind a combination rather than an individual shared pixel?
```

The first question is about allocation dynamics. The second is about
representation. Treating them separately avoids mistaking a lucky initialization
for a learning solution.
