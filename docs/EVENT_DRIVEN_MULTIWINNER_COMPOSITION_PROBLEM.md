# Deferred Design Problem: Event Scheduling and Multi-Winner Composition

## Status and scope

This note records two important future design problems. It is not an implementation
specification and does not authorize changes to the current scheduler, WTA behavior,
learning rules, or existing presets.

The planned cortical-column preset does not solve the second problem as part of its
initial implementation. Its local competing E population will deliberately retain the
current immediate hard single-winner WTA mechanics. Multi-winner composition remains a
separate future circuit change so the column hierarchy can first be validated without
changing competition semantics at the same time.

The problems are related but distinct:

1. The `rg_coincidence` execution path is event-resolved inside an outer boundary,
   but it is not a pure queue-driven discrete-event simulation.
2. The current WTA circuit admits one winner, while compositional recognition requires
   multiple already-learned, independently supported features to co-fire.

Changing the scheduler does not automatically solve multi-winner composition. The
scheduler determines how events are represented and ordered; the circuit and admission
rules determine which neurons are allowed to fire.

## Problem 1: How event based is the current simulation?

The current event-resolved path is more than a synchronous neuron loop. It computes
analytic within-boundary crossing times, advances membranes to the earliest event,
fires that cell, applies zero-latency consequences, and recomputes future crossings.
This makes causal timing explicit and avoids making the physical outcome depend on
ordinary Python neuron iteration order.

However, it remains a hybrid boundary/event system:

- the engine opens an outer boundary and rotates delay-one delivery buffers;
- it visits population state to initialize the boundary;
- a central scheduler searches the eligible membranes for the next crossing;
- all active membranes are advanced to each selected event time;
- inhibition or a new impulse causes candidate crossing times to be reconsidered;
- boundary-wide trace, conductance, refractory, history, and serialization work is
  performed at the end.

It is therefore **event-resolved in its causal semantics**, but not a pure
discrete-event simulation in which the next queued event is the only object inspected.
This is currently acceptable: the state scan is explicit, deterministic, testable, and
small. The concern becomes more important as cortical columns make the population much
larger.

### Possible future scheduler

A more conventional discrete-event design could use a priority queue ordered by
absolute simulation time and a deterministic secondary key. Queue entries could include:

- external and synaptic spike arrivals;
- predicted membrane threshold crossings;
- refractory release events;
- conductance or trace events only where an analytic future transition is required.

Each neuron would maintain a state version. An arrival, reset, or conductance change
would lazily advance only the affected neuron, increment its version, and schedule a new
predicted crossing. Old crossing entries would remain in the heap but be ignored when
their version no longer matched. A spike would schedule outgoing edge-arrival events.

This removes repeated whole-population crossing searches while preserving exact
invalidation after inhibition. A pragmatic intermediate step would retain outer
boundaries for input presentation and reporting while using a true priority queue inside
each boundary.

Questions to settle before such a rewrite include numerical determinism, simultaneous
event batching, exact conductance/leak prediction, stale-event invalidation, trace decay,
dashboard sampling, and compatibility with the five existing presets.

## Problem 2: Single-winner WTA blocks composition

### Required composition behavior

Suppose one neuron has fully learned a horizontal row and another has fully learned a
vertical column. When the union is presented as a plus sign:

1. the row specialist should recognize its learned component and fire;
2. the column specialist should recognize its learned component and fire;
3. their co-activity should be available as causal input to a third neuron;
4. the third neuron should learn the conjunction/composition "row + column = plus";
5. unrelated or merely near-threshold neurons should remain suppressed.

The current hard single-winner circuit prevents step 2. The first crossing recruits
inhibition that cancels every later candidate, even when a later candidate represents a
different, genuinely present component of the input.

### Why a fixed delta-tau rule is not sufficient

Admitting every neuron whose crossing lies within `delta_tau` of the first winner would
make co-firing depend on a fragile numerical tolerance rather than on represented
evidence. Closely initialized weights could admit a crowd of untrained neurons. The
result would also vary with leak, threshold, learning stage, input scale, and the units
used for time. A time-neighborhood can remain a diagnostic, but should not be the
scientific definition of composition.

### Properties a solution should have

A future multi-winner mechanism should:

- admit independently supported learned features, not simply the fastest cells;
- suppress redundant neurons explaining the same evidence;
- reject weak, untrained, or incidental near-threshold candidates;
- have a bounded and observable recruitment process;
- derive decisions from local spikes, weights, inhibition, and eligibility where
  possible rather than reading a centralized label or pattern identity;
- allow the co-winners to drive a downstream composition learner;
- preserve single-winner behavior as an option for the existing presets;
- remain stable under narrow initial-weight jitter.

### Candidate solution families

#### 1. Similarity-structured lateral inhibition

Replace uniform global inhibition with inhibition that grows when two neurons explain
the same afferent evidence. Strongly overlapping specialists suppress one another;
distinct specialists inhibit one another weakly enough to co-fire. The inhibitory
weights could be learned from co-activation or derived from normalized receptive-field
overlap.

This directly addresses redundancy, but requires a principled local learning rule and
care around the row/column center pixel: overlap at one pixel must not make the two
features mutually exclusive.

#### 2. Residual or unexplained-evidence competition

After a feature fires, it suppresses only the input evidence that it predicts. Other
candidates are then evaluated against the remaining evidence. A row winner would explain
the row while leaving the non-center column pixels available, allowing the column
specialist to fire as well.

This has a strong compositional interpretation. It must be implemented as local feedback
onto afferents or residual cells, not as a centralized routine that subtracts a known
template. The order and treatment of shared evidence also need precise rules.

#### 3. Activity-dependent inhibitory budget

Each accepted winner recruits additional inhibition, and recruitment stops only when no
remaining candidate has enough evidence to overcome the accumulated inhibitory field.
This produces an adaptive `k` rather than fixing `k = 2`.

By itself this does not distinguish distinct evidence from many similarly initialized
candidates, so it likely needs trained-evidence or redundancy gating.

#### 4. Learned admission or familiarity gate

Require a candidate to satisfy both its membrane threshold and a local measure of
learned support, such as sufficient active weight mass, confidence, or causal afferent
coverage. Global inhibition can then admit multiple familiar features while excluding
untrained near-ties.

The gate must not prevent a designated novel composition neuron from learning. Recognition
and novelty recruitment may therefore need separate cell states or separate populations.

#### 5. Sparse recurrent competition instead of hard WTA

Use recurrent excitation plus graded lateral inhibition so the circuit settles into a
sparse set of active assemblies. Multiple strongly supported, weakly redundant features
can form a stable state, while weak cells fall out.

This is biologically attractive but is a larger dynamical change. It raises convergence,
oscillation, termination, continuous-time, and dashboard-observability questions.

#### 6. Separate recognition and composition stages

Allow a feature-recognition population to be nonexclusive or sparsely competitive, then
feed its output neurons into a downstream composition population. The downstream unit
learns only when the row and column outputs co-occur. Competition can remain local to
redundant alternatives rather than global across all recognized components.

This fits naturally with hierarchical cortical columns and their `Eor` outputs. It does
not eliminate the need to define local multi-winner behavior, but it prevents a global
WTA from conflating recognition of components with selection of a single interpretation.

### Promising direction to investigate

The strongest starting hypothesis is a combination of:

1. separate feature-output and composition-learning stages;
2. evidence-specific or similarity-structured inhibition, so winners suppress redundant
   explanations rather than every neuron;
3. a local familiarity/admission condition for established features;
4. a separate novelty path through which an uncommitted neuron can learn from the
   simultaneous established outputs.

This avoids a fixed `delta_tau`, does not assume exactly two winners, and ties co-firing
to causal evidence. It remains a hypothesis until minimal circuit experiments compare it
against residual competition and sparse recurrent alternatives.

## Required future experiments

A controlled composition experiment should include at least:

1. Train one row specialist to a declared maturity criterion.
2. Train one column specialist independently to the same criterion.
3. Freeze those specialists for the diagnostic presentation.
4. Present the plus-sign union.
5. Verify that both specialists fire and that unrelated mature or untrained cells do not.
6. Verify that a third cell receives two causal feature-output events and learns the plus.
7. Present row alone, column alone, plus, and nearby distractors after learning.
8. Measure winner count, feature precision/recall, composition-cell selectivity, crossing
   times, inhibition, and dependence on seed and initial-weight spread.

Critical negative controls include all weights initialized nearly equal, many unused
candidate cells, unequal row/column strength, a shared center pixel, and sequential rather
than simultaneous component presentation.

## Open design questions

- Is multi-winner admission local to one cortical column, global within a layer, or both?
- What represents an established/familiar feature, and what locally opens learning for a
  novel composition cell?
- How many co-winners should be possible, and what circuit bounds that number?
- Should co-winners be simultaneous, merely overlap within an eligibility window, or form
  a short causal sequence?
- How should shared evidence, such as the plus sign's center pixel, be allocated?
- Does each winner recruit inhibition immediately, or only after a composition-admission
  phase?
- How do coincidence cells and feedback behave when several higher-level cells fire?
- Which behavior belongs in a new preset, and which mechanisms should remain optional so
  all current presets retain their existing semantics?
