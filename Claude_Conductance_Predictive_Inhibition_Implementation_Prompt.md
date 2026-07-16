# Implement local predictive inhibition and test symmetry breaking

## Objective

Replace subtractive hard wipes with persistent inhibitory conductance and implement
the following experimental predictive-inhibition topology:

```text
                               L2I_WTA
                                  |
                                  v
external input -> L1E_i -> L2E_j -> PI_j
                  ^                 |
                  |                 | locally plastic inhibitory synapses
                  +-----------------+
```

There are:

- 9 sensory excitatory neurons, `L1E_s[i]`;
- 8 L2 excitatory pattern detectors, `L2E[j]`;
- 8 predictive inhibitory relays, `PI[j]`, paired one-to-one with `L2E[j]`;
- 1 existing L2 winner-take-all inhibitory relay, `L2I_WTA`.

Each `PI[j]` has a potentially plastic inhibitory output synapse onto each
`L1E_s[i]`. This gives 72 candidate synapses for the experiment. This is deliberately
dense as a test scaffold, not a claim that the final circuit should remain dense.
Learning must be strictly local at each candidate synapse so that later structural
pruning or sparse synapse growth is possible.

Do not implement a centralized `8 x 9` prediction table update that reads the L2
winner index and the complete L1 spike vector. Although the candidate synapses can be
displayed mathematically as an `8 x 9` array, each synapse must own and update only
its own weight using locally available pre- and postsynaptic state.

## Answer the scientific question this implementation is testing

The intended symmetry-breaking mechanism is temporal explaining-away:

1. One L2 detector becomes associated with pattern A.
2. Its paired predictive interneuron learns inhibitory outputs onto the L1 features
   that were locally active when that detector fired.
3. On later recognition events, those inhibitory synapses establish conductance that
   persists into subsequent sensory volleys.
4. When pattern B overlaps pattern A, the shared features are suppressed more than
   B's novel features.
5. The residual novel features can drive a different L2 competitor, allowing a
   winner transition rather than permanent capture by the incumbent.

This is possible but not guaranteed. A critical failure mode must be tested:

> If the incumbent wins the overlapping pattern and its inhibitory synapses learn
> too quickly, it can associate itself with the novel features too, inhibit the whole
> new pattern, and prevent rather than promote symmetry breaking.

The implementation must therefore separate the timescale of inhibitory expression
from the slower timescale of inhibitory association, measure contamination of the
incumbent's learned outputs, and report honestly if no parameter region produces the
desired transition.

## Preserve the current working tree

Before editing, inspect:

- `git status` and `git diff`;
- `snn/neurons.py`;
- all of `backend/simulation.py`;
- `backend/serializer.py`;
- `backend/dashboard_config.py`;
- `backend/layout.py`;
- relevant frontend topology and chart consumers;
- all tests, especially `tests/test_direct_topology.py`, `tests/test_coincidence.py`,
  `tests/test_causal_step.py`, `tests/test_excitatory_neuron.py`,
  `tests/test_inhibitory_relay.py`, and `tests/test_serialization_api.py`;
- `Current_Implementation_Methodology_Equations.md`.

There are uncommitted user changes that add the `enew_enabled` toggle, implement the
old direct topology as a global L2E-to-all-L1I relay, expose the flag through config,
and add direct-topology tests. Preserve and build on those changes. Do not revert or
broadly reformat unrelated work.

## Audit of the current implementation

Verify these statements against the live files before editing.

### Current neuron behavior

`ExcitatoryNeuron` currently:

- adds excitation directly to scalar `V`;
- tests `V >= threshold` after delivery;
- resets `V` immediately when it fires;
- applies fixed leak in `advance()`;
- represents inhibition as `hard_wipe()`/`receive_subt()`, forcing `V` to rest;
- has no persistent synaptic conductance.

`InhibitoryNeuron` currently:

- is a stateless Boolean relay;
- discards input magnitude;
- has no integrating membrane or plastic weights;
- reports a visual threshold that does not participate in firing.

The predictive `PI[j]` cells may remain abstract one-to-one spike relays for this
experiment. Their output synapses, not their own membranes, carry the learned
magnitudes, and the resulting conductance persists in the L1 targets. Keep the
abstraction explicit rather than pretending the relay threshold is integrated.

### Current simulation order

`SimulationEngine.step()` currently implements an ordered same-call cascade:

1. deposit external L1 charge;
2. apply queued L1 hard wipes;
3. test/fire L1;
4. immediately deliver L1 spikes to L2 and optionally `L1E_new`;
5. choose one L2 winner;
6. immediately fire L2I and hard-wipe L2;
7. route the winner through `L1E_new` or every old-topology L1I;
8. queue L1 wipes for the next call;
9. apply leak/refractory housekeeping.

This makes scientific behavior depend heavily on Python phase order. L2I cannot
cancel the winner that caused it, and L1 inhibition only removes nonzero charge
because of a special next-step queue.

### Current topology toggle

With `enew_enabled=True`, retain the existing 36-neuron `L1E_new` coincidence
topology as a comparison condition. Do not delete it.

With `enew_enabled=False`, the live branch currently removes `L1E_new` and routes
every L2 winner to all nine L1I relays, producing global delayed hard wipes. Replace
that direct/global experimental branch with the new pattern-specific `PI[j]`
topology. Update its topology tests and counts deliberately; do not make old tests
pass by leaving hidden unused L1I neurons.

A suggested direct-topology population is:

```text
9 x L1E_s
8 x PI, paired one-to-one with L2E
8 x L2E
1 x L2I_WTA
```

The existing `enew_enabled=True` branch can retain its nine paired L1I cells because
they belong to that comparison topology.

## Local activity trace on each L1 excitatory neuron

By the time an L2 winner and its paired PI relay fire, the causal L1 neurons may have
already spiked and reset. Reading their current voltage would incorrectly make them
look inactive. Give every `L1E_s[i]` a local postsynaptic activity/calcium trace that
survives voltage reset:

```text
a_i[t+1] = alpha_a * a_i[t] + activity_increment_i[t]
```

Use only the target neuron's own state to compute `activity_increment`. A reasonable
normalized form is:

```text
depolarization_i = clip(
    (V_pre_reset_i - V_rest) / (threshold - V_rest),
    0,
    1,
)

activity_increment_i = beta_v * depolarization_i + beta_s * spike_i
```

Clamp or otherwise bound the trace. Name and document its decay constant. Preserve
`V_pre_reset` long enough to update the trace correctly when the neuron fires.

This trace means only “this postsynaptic cell was recently depolarized/firing.” It
must not identify which excitatory afferent supplied the charge.

## Strictly local inhibitory plasticity

Each candidate inhibitory synapse `PI[j] -> L1E_s[i]` owns:

- its own nonnegative weight `w_ji`;
- access to its own presynaptic PI event/trace;
- access to the local activity trace of its postsynaptic L1 target;
- its own bounds and optional slow decay/consolidation state.

It must not access other targets' traces, the full L1 vector, a pattern label, or a
central winner-row update.

On a real presynaptic `PI[j]` event, a minimal potentiation rule is:

```text
delta_w_ji = eta_i * pre_j * a_i * (w_i_max - w_ji)
w_ji <- clip(w_ji + delta_w_ji, 0, w_i_max)
```

If recovery from incorrect associations is required, add a biologically interpretable
slow passive decay or local depression term. Do not use `1 - a_i` as an aggressive
depression signal on every predictive event without checking the consequence: a PI
cell may fire from top-down context when there was intentionally no recent L1
activity, and that should not instantly erase a mature prediction.

Use the pre-update synaptic weight to generate the current inhibitory pulse. Learning
from the current event changes future predictions; it must not retroactively turn the
first observation into a mature prediction.

Initialize candidate output weights at zero or a very small deterministic seed. A
small seed must not generate meaningful global inhibition. The first observed pattern
should teach the local synapses, not be strongly suppressed as though already known.

### Required timescale separation

Expose and document separate parameters for:

- activity-trace decay;
- inhibitory synaptic learning rate;
- inhibitory conductance magnitude per unit synaptic weight;
- inhibitory conductance decay;
- optional long-term synaptic decay/consolidation.

Conductance expression should be immediate once a mature synapse is activated, while
association should be slow enough that one overlapping presentation does not make an
incumbent PI cell instantly learn every novel feature.

Do not hardcode a known pattern or pixel set to enforce this separation.

## Persistent inhibitory membrane dynamics

Replace ordinary L1 and L2 hard wipes with persistent inhibitory conductance on the
target excitatory neurons.

At minimum, each inhibited excitatory neuron needs:

```text
g_inh >= 0
tau_inh > 0, or equivalent alpha_inh in [0, 1)
E_inh <= V_rest
```

An inhibitory synaptic event adds conductance:

```text
g_inh <- g_inh + g_scale * w_synapse
```

Conductance decays rather than disappearing during a voltage reset:

```text
g_inh <- g_inh * exp(-dt / tau_inh)
```

Resetting after a spike resets `V`; it must not clear `g_inh` or the local activity
trace.

Excitation and inhibition arriving for one integration boundary must be combined
before threshold testing. Do not apply leak, add a threshold-sized excitatory jump,
test threshold, and only then apply inhibition.

Use a documented conductance/current equation, for example:

```text
C dV/dt = -g_L(V - E_L) - g_inh(V - E_inh) + I_exc
```

Treat the interval's existing charge-scale excitatory input `Q_exc` as
`I_exc = Q_exc / dt`. For constant input over one interval:

```text
g_total = g_L + g_inh
V_inf = (g_L*E_L + g_inh*E_inh + I_exc) / g_total
V_next = V_inf + (V_prev - V_inf) * exp(-g_total*dt/C)
```

Handle `g_total == 0` explicitly:

```text
V_next = V_prev + Q_exc/C
```

Map existing `leak_rate` consistently to baseline `g_L`, or replace it with a named
time constant while providing a documented migration path. Avoid unexplained units
or magic scaling constants.

The required behavioral invariant is:

> Excitation that crosses threshold in one interval without inhibition can remain
> subthreshold when sufficient `g_inh` is already present in that same interval.

## Explicit timestep semantics

Use synchronous boundaries, double-buffered arrivals, or explicit delivery queues:

1. Read state at boundary `t`.
2. Gather every excitatory and inhibitory arrival scheduled for the interval.
3. Integrate each target once using the complete gathered input.
4. Test threshold crossings after integration.
5. Resolve deterministic L2 winner-take-all from the L2 crossers.
6. Emit spikes into queues for documented future boundaries.
7. Update local traces and local synapses at their defined event boundary.
8. Retain/decay conductances exactly once.
9. Serialize the frame.

Use explicit integer synaptic delays. Do not recursively execute an entire
`L1E -> L2E -> PI -> L1E` loop inside one undifferentiated timestep. If a relay is
modeled as zero-delay, name the causal subphase and ensure all targets in that phase
are integrated symmetrically.

The causal interpretation should be:

```text
first encounter:
    L1 activity -> L2 winner -> PI event -> local inhibitory learning
    (the original L1 spike is not canceled)

later encounter/context:
    L2 winner -> PI event -> persistent g_inh
    -> later L1 sensory interval is shunted
```

## L2 winner-take-all inhibition

Keep the existing single `L2I_WTA` concept separate from the eight predictive PI
cells.

Replace its L2 hard wipe with a persistent global inhibitory-conductance pulse on all
L2E cells. Preserve deterministic single-winner arbitration. Be explicit that:

- the winning spike occurs before the feedback inhibition it caused;
- L2I cannot retroactively cancel that winner;
- the conductance suppresses other/subsequent L2 activity while it decays.

Do not describe deterministic arbitration itself as charge removal.

## Symmetry-breaking validation

Use at least two patterns with a known overlap, but derive all behavior from normal
input vectors rather than hardcoded learning cases. The existing examples are useful:

```text
row:    {3, 4, 5}
column: {1, 4, 7}
shared: {4}
novel column features: {1, 7}
```

Run a deterministic multi-phase experiment across multiple seeds:

### Phase A: establish an incumbent

Present the row until one L2 detector consistently wins. Verify that its paired PI
cell develops strong output synapses primarily onto `{3,4,5}`.

### Phase B: switch to an overlapping pattern

Switch to the column. The first column presentation may still be won by the incumbent
because of shared pixel 4; do not require impossible retroactive prevention.

On later volleys, measure whether:

- the incumbent PI conductance suppresses the shared feature 4;
- novel features 1 and 7 remain less inhibited;
- residual novel activity can drive a different L2 competitor;
- the L2 winner changes within a bounded observation window.

### Contamination check

During the transition, measure whether the incumbent PI's synapses onto novel column
features 1 and 7 grow. Successful symmetry breaking requires a useful timescale
window in which:

```text
mature inhibition of shared/old features
    > inhibition of newly encountered features
```

If the incumbent learns 1 and 7 so quickly that it suppresses the whole column before
a rival can win, record that as a failed mechanism. Adjust only named general
timescales or local rules; do not special-case overlap pixels.

### Phase C: revisit the original pattern

After inhibitory conductance has had an appropriate chance to decay, return to the
row and test whether the original detector remains associated with it. Distinguish
temporary adaptation from catastrophic erasure of the learned pattern.

### Required controls

Compare against:

- predictive conductance disabled;
- inhibitory plasticity disabled;
- fast versus slow inhibitory learning;
- at least several deterministic seeds;
- more than one pair/order of overlapping patterns.

Do not claim symmetry breaking from a single winner change. Report incumbent and
rival firing histories, residual L1 activity, synaptic weights, activity traces, and
conductances.

## Tests

### Neuron and conductance tests

1. No-inhibition dynamics match the intended LIF baseline within tolerance.
2. `g_inh` remains nonnegative and decays monotonically without new pulses.
3. A voltage reset does not clear `g_inh` or the L1 activity trace.
4. Sufficient active inhibition suppresses an otherwise instant threshold crossing.
5. Same-boundary excitation/inhibition is independent of insertion or neuron-loop
   order.
6. Zero leak/conductance cases produce no NaN or division by zero.

### Local plasticity tests

1. One inhibitory synapse updates using only its own PI event, target trace, and
   current weight.
2. A PI event potentiates a synapse onto a recently active target.
3. A synapse onto an inactive target remains weak under the chosen default.
4. Updating one synapse does not inspect or mutate another target's synapse.
5. The current pulse uses the pre-update weight.
6. Weights remain bounded and deterministic.
7. The activity trace survives spike reset and decays without activity.

### Topology tests

1. `enew_enabled=False` creates 8 pattern-specific PI cells, one per L2E.
2. Every L2E drives only its paired PI relay.
3. Each PI has candidate inhibitory outputs to the nine L1E targets.
4. No global winner-to-all-nine-Boolean-L1I shortcut remains in the direct branch.
5. `enew_enabled=True` still builds the existing `L1E_new` comparison topology.
6. Toggling rebuilds cleanly and resets learned local inhibitory weights.

### Timing and network tests

1. Synaptic effects occur only at documented delivery boundaries.
2. L2I does not cancel the winner that caused it.
3. A learned PI pulse persists until the later L1 sensory interval.
4. Predicted L1 features are suppressed more than unpredicted features.
5. Reordering internal iteration does not change a seeded trace.
6. Run the full existing test suite and update expectations only for documented
   scientific changes.

## Serialization and visualization

Keep current response keys where their meaning remains accurate. Add inspectable
state for:

- each excitatory neuron's `g_inh`;
- each L1 neuron's local activity trace;
- each live `PI[j] -> L1E_s[i]` inhibitory weight;
- PI spikes and their paired L2 source;
- inhibitory pulse source, target, synaptic weight, conductance increment,
  `g_inh_before`, `g_inh_after`, and delivery boundary;
- membrane values before and after joint integration.

Do not serialize a conductance pulse as `charge_removed`. Update API tests and
frontend consumers together when a field's meaning changes.

The dashboard should visually distinguish:

- L2 winner-take-all inhibition;
- predictive PI-to-L1 conductance;
- the legacy/comparison `L1E_new` route;
- a synaptic pulse from the conductance's continuing membrane effect.

## Sparsity follow-up, not a prerequisite

Keep the dense 72 candidate PI output synapses for this validation, but instrument
their learned distribution. Report:

- number above meaningful strength thresholds;
- effective fan-out per PI cell;
- weights that remain near zero;
- whether magnitude pruning would preserve behavior.

Do not implement destructive pruning until the dense scaffold demonstrates local
learning, inhibition, and symmetry breaking. If it succeeds, propose a separate
follow-up using pruning, local structural growth, or spatial connection constraints.

## Documentation

Update `Current_Implementation_Methodology_Equations.md` to describe the implemented
model, including:

- persistent conductance equations;
- activity-trace equation;
- strictly local inhibitory plasticity rule;
- topology-specific populations and edges;
- exact delays and boundary ordering;
- distinction between `PI`, `L2I_WTA`, and legacy `L1E_new` inhibition;
- symmetry-breaking results and failure cases;
- measured sparsity of the dense candidate projection.

Remove claims that ordinary inhibition is a frozen threshold-sized hard wipe once
that is no longer true.

## Scope constraints

- Preserve user changes in the dirty tree.
- Preserve the `enew_enabled` comparison topology.
- Do not add a centralized winner-row learning update.
- Do not give a synapse access to the full L1 activity vector.
- Do not add `L1E -> L1I` feedforward inhibition in this experiment.
- Do not use ordinary hard wipes as a conductance fallback.
- Do not clear conductance or activity trace on voltage reset.
- Do not hardcode patterns, overlap pixels, or winners.
- Do not claim that inhibition cancels the spike that caused it.
- Do not claim symmetry breaking without the overlap controls above.
- Do not tune or validate on only one seed.

## Deliverable

Implement the model, run the full suite and validation experiments, then report:

1. files changed;
2. final topology and exact edge counts in both toggle states;
3. membrane, conductance, trace, and local plasticity equations;
4. explicit timestep/delay semantics;
5. test results;
6. numerical evidence that active inhibition suppresses an otherwise instant spike;
7. overlap-pattern winner histories with and without predictive inhibition;
8. incumbent-synapse contamination measurements;
9. effective learned sparsity;
10. remaining scientific failures or parameter dependencies.
