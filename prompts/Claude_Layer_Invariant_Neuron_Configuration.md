# Task: Make Neuron Configuration Layer-Invariant and Restore Developmental Timing

## Objective

Redesign the L1 excitatory path so that neuron configuration is determined by cell
type rather than layer. L1E should no longer be a pre-trained, full-threshold binary
relay. It should accumulate repeated sensory spike events and learn with the same
excitatory-neuron configuration and update rule as L2E.

Likewise, L1I and L2I should share the same intrinsic inhibitory-neuron configuration.
Layers may differ in connectivity and circuit role, but not through arbitrary
layer-specific thresholds, leak, refractory behavior, learning configuration, or
weight scaling.

The intended invariant is:

```text
L1E neuron configuration == L2E neuron configuration
L1I neuron configuration == L2I neuron configuration

Layer differences == connections and circuit topology only
```

This is a substantial engine change. Audit the current behavior first, implement the
smallest coherent architecture that satisfies the invariant, and recalibrate tests
and frontend reporting around the resulting timescale. Do not preserve an old golden
trajectory at the expense of the new requested behavior; instead, keep an explicit
legacy/default mode only if it is scientifically useful and clearly separated from
the new mode.

## Current Problem to Confirm

The current L1E path appears to be configured approximately as follows:

- `theta_L1E = 1000`;
- fixed sensory weight `+1000`;
- fixed inhibitory input in the other afferent slot;
- excitatory learning rate forced to zero;
- one held active pixel supplies a threshold-sized event every input timestep;
- L1E therefore fires on nearly every available timestep.

L2E instead appears to use:

- `theta_L2E = 8000`;
- subthreshold feedforward weights initialized around the existing L2E initialization
  policy;
- temporal accumulation;
- active excitatory learning;
- threshold-relative caps and learning rates.

Confirm every value from the live engine and code before editing. The important
failure is not that L1E is "learning too quickly." It is that L1E is initialized as a
pre-trained, non-plastic relay whose single sensory event already equals its firing
threshold. This saturates its firing rate and removes a developmental timescale from
the hierarchy.

That compressed L1E timescale also distorts predictive inhibition: frequent L1E
activity can make paired inhibitory events and any learned `L1I→L1E` gate mature much
faster than intended.

## Required Model Invariants

### Excitatory neurons

L1E and L2E must use the same policy for:

- firing threshold;
- resting potential;
- membrane update and no-leak behavior;
- refractory behavior;
- positive-weight initialization scale/policy;
- positive-weight cap;
- positive-weight floor;
- learning rate and threshold-relative scaling;
- excitatory learning rule;
- confidence/structural learning behavior when those mechanisms are enabled;
- fixed-point units and charge accounting.

Do not copy L2's WTA or competitive reset into L1. Those are circuit connections and
population-level behavior, not intrinsic excitatory-neuron configuration.

### Inhibitory neurons

L1I and L2I must use the same policy for:

- firing threshold;
- resting potential;
- no-leak membrane behavior;
- refractory behavior;
- positive-afferent initialization scale and bounds where the connection roles are
  comparable;
- positive-afferent cap and learning-rate scaling;
- fixed-point units and charge accounting.

The actual L1I threshold may already equal L2I internally through
`l1i_threshold_frac = 1.0`. Verify this. If so, remove redundant layer-specific
threshold behavior and fix any incorrect topology metadata rather than introducing a
second threshold change.

Connection-specific rules are still allowed:

- paired `L1E_i→L1I_i` local evidence can have its own fixed synaptic role;
- predictor learning applies to `L2E_j→L1I_i` feedback afferents;
- learned paired output inhibition applies to `L1I_i→L1E_i`;
- the unweighted `L2I→L2E` competitive reset remains a structural circuit event.

Do not confuse a special connection rule with a special neuron implementation.

## Part 1: Replace Full-Threshold Sensory Injection

External input must be represented as a binary presynaptic event, not as a
threshold-sized charge injection.

The intended path is:

```text
external sensory source S_i emits a binary spike
        ↓
learnable positive S_i→L1E_i synapse
        ↓
L1E_i accumulates weight-mediated charge over repeated events
        ↓
L1E_i crosses the shared excitatory threshold
        ↓
L1E_i fires and updates its positive sensory weight using the L2E rule
```

Requirements:

1. Give each L1E exactly one positive sensory afferent associated with its pixel.
2. Preserve its paired negative `L1I→L1E` afferent as a separate connection.
3. A held pixel produces one binary input event per configured input timestep.
4. The sensory event contributes the current learned synaptic weight, not an
   artificial `theta_L1E` charge.
5. Initialize the sensory weight with the same policy and scale used for an L2E
   positive feedforward weight.
6. Enable the same positive-weight learning rule used by L2E.
7. Apply the same positive cap, floor, and learning-rate derivation as L2E.
8. Keep inactive pixels from receiving charge or changing their sensory weights.
9. Do not add L2-style WTA competition among L1E neurons.
10. Make the sensory connection visible and inspectable in topology if external
    source nodes or input-edge serialization fit the existing frontend architecture.
    If not, expose the learned sensory weight through an equally auditable state
    surface and explain the choice.

Avoid creating a second custom L1 learning rule. Route both L1E and L2E construction
through shared configuration helpers or factories so future parameter changes cannot
silently make the layers diverge again.

## Part 2: Use Shared Threshold and Initialization Policies

Use a single source of truth for excitatory thresholds and weight initialization.
Under the current dashboard scale, the likely result is that L1E uses the L2E
threshold (currently approximately `8000`) and the existing L2E feedforward
initialization policy (currently approximately `50–200`, or the balanced equivalent).
Do not hard-code those numbers in a second location; derive them from the same engine
parameters and helper functions.

Audit how `l2e_weight_cap_frac`, positive floors, balanced initialization, jitter,
signed-spike learning, and any weight budget interact with a one-afferent L1E neuron.
The rule must remain mathematically defined with a fan-in of one.

With no leak, the approximate first-spike latency is:

```text
ceil(theta_E / initial_sensory_weight)
```

For `theta_E = 8000` and an initial weight near `125`, this is roughly 64 active
sensory events. Treat that latency as a model outcome to measure, not a reason to
restore threshold-sized sensory injection.

Pattern dwell times, auto-cycle visits, tests, and experiments must be long enough to
allow the developmental path to bootstrap. Do not silently lower the threshold or
increase the initial weight merely to retain the old short test durations.

Report the measured distribution of first-spike latencies across pixels and seeds for
both legacy-wide and balanced initialization. If random per-pixel initialization
creates unacceptable task bias, address it through the shared initialization policy,
not through pixel-specific thresholds.

## Part 3: Remove Layer-Specific Leak and Trace Behavior

The target model currently uses no membrane leak. L2E does not rely on leak, and L1E
must not receive a special leak. L1I must not receive a predictive-only forced leak
if L2I is non-leaky.

Requirements:

1. Verify the live effective leak for L1E, L2E, L1I, and L2I.
2. Remove or disable the predictive-only L1I leak override.
3. Make the no-leak policy layer-invariant by neuron type.
4. Ensure frontend controls and descriptions do not claim independent layer-specific
   leak behavior if the new invariant removes it.
5. Update or remove tests that require the former predictive L1I leak equilibrium.

The target model also should not depend on a decaying local eligibility trace when L2E
learning uses the current presynaptic event vector.

Do **not** implement "no trace decay" by setting the decay multiplier to `1.0`; that
would latch a pixel's local evidence forever after one spike. Instead, determine
whether predictor learning can consume the current L1E spike vector directly:

```text
x_i(t) = 1 if L1E_i actually spiked at timestep t, else 0
```

The L2E threshold crossing is caused by an L1E volley, so current-step L1E spikes
should be the natural credit signal. Audit this timing explicitly.

If active L1E encoders become desynchronized and an actual L2E spike occurs on only
one member of a multi-pixel pattern's volley, current-step credit may omit the other
recently active pixels. Measure this behavior. Prefer a principled solution based on
the event schedule or shared initialization rather than quietly retaining an
arbitrary layer-specific decay constant. If a finite event window is scientifically
necessary, document it as a connection-level eligibility mechanism and demonstrate
why it does not violate the neuron-configuration invariant.

## Part 4: Preserve and Recalibrate Predictive Inhibition

This task must integrate with the separate predictive-inhibition audit. The desired
connection roles remain:

```text
fixed local evidence:        L1E_i→L1I_i
learned predictor feedback:  L2E_j→L1I_i
learned paired suppression:  L1I_i→L1E_i
structural L2 competition:   L2I→L2E reset fanout
```

Do not calibrate a learned `L1I→L1E` gate against the old every-timestep L1E relay.
After L1E becomes a developing accumulator, remeasure:

- L1E firing frequency during early and late sensory training;
- timing between L1E volleys and L2E threshold crossings;
- predictor feedback-update frequency;
- L1I charge and spike frequency;
- learned output-gate update frequency and saturation time;
- center-versus-novel-pixel suppression during row-to-column transfer.

The desired developmental sequence is:

```text
early:
  repeated sensory events slowly charge L1E
  L1E spikes are sparse
  sensory weights begin weak
  predictive and output inhibitory weights are immature

later:
  trained sensory afferents make L1E responses more reliable/frequent
  L2E receives increasingly reliable learned features
  predictor feedback becomes selective
  paired inhibitory output gates mature through real events
```

No connection should begin as an artificial full-threshold relay merely to make the
sequence happen quickly.

## Part 5: Correct Threshold and Weight Observability

Audit topology and dynamic-state serialization. Every neuron must report the threshold
of its live neuron object.

Known items to verify:

- L1I may internally use the same threshold as L2I while topology metadata reports
  the old L1E threshold.
- L2I metadata may report the L2E threshold instead of its actual inhibitory
  threshold.
- Charge-chart activation is derived from serialized metadata and can therefore show
  a false threshold crossing.

Requirements:

1. Serialize `threshold = neuron.threshold` for every population.
2. Normalize charge using that same live threshold.
3. Expose each L1E sensory weight so its developmental trajectory can be inspected.
4. Keep L2 competitive-reset events visually separate from paired L1 inhibition.
5. Preserve the requested applied-event marker for `L1I→L1E` inhibition at the
   timestep when it reaches L1E.
6. Make it possible to distinguish increasing L1E firing caused by sensory-weight
   learning from missing spikes caused by learned inhibition.

## Part 6: Deterministic Tests

Add focused tests for the new invariant and developmental behavior.

### Shared configuration

- L1E and L2E live thresholds are equal.
- Their intrinsic leak, refractory, cap, floor, and learning-rule settings match.
- L1I and L2I live thresholds are equal.
- Their intrinsic no-leak policy and other intended neuron settings match.
- Differences are limited to documented connection/circuit properties.

### Sensory accumulation

- One sensory event does not make an untrained L1E fire.
- Repeated events accumulate exact weight-mediated charge with no leak.
- The first spike occurs at the analytically expected timestep.
- An inactive pixel neither charges nor learns.
- Firing resets/discharges L1E using the same rule as L2E.

### Sensory learning

- A firing L1E updates its active positive sensory weight using the same numerical
  rule as an equivalently configured L2E afferent.
- Weight changes remain within the shared floor and cap.
- Repeated training changes L1E's firing cadence in the expected direction.
- No L1-specific learning shortcut runs.

### No leak and local credit

- L1E, L2E, L1I, and L2I exhibit the intended no-leak behavior.
- Predictor credit uses the intended current-event evidence or a documented
  connection-level window.
- Local evidence is never latched forever after one spike.

### Hierarchical integration

- L1E spikes drive L2E through the existing feedforward synapses.
- L2E charge builds only from actual L1E spikes.
- Training horizons are long enough for both layers to bootstrap.
- No shape/index drift occurs when L1E has a real learnable sensory afferent plus its
  inhibitory afferent.

### Predictive integration

- Row training produces selective `L2E→L1I` feedback.
- Paired L1I spikes produce delayed, weight-mediated `L1I→L1E` events.
- Output inhibition learns at the new L1E cadence rather than saturating immediately.
- In row-to-column transfer, the shared center is suppressed more strongly than the
  novel pixels once the relevant connections have matured.

### Serialization

- Every serialized threshold equals the corresponding live neuron threshold.
- Charge activation uses the live threshold.
- Sensory weights and applied paired-inhibition events are externally auditable.

## Part 7: Experiment and Timing Audit

The old dashboard and experiment schedules were designed around immediate L1E firing.
Audit all fixed-duration assumptions, including:

- pattern dwell duration;
- auto-cycle visit length;
- trained-streak logic;
- firing-frequency window;
- episode duration and quiet termination;
- predictive experiment acquisition/reversal dwell;
- tests that assume an L1E spike on the first active timestep.

Update schedules based on measured developmental latency. Keep timing parameters
explicit and explain their units in raw sensory timesteps, L1E volleys, and L2E
events. Do not hide the slower hierarchy by injecting extra charge.

Provide before/after measurements for:

- first L1E spike;
- first L2E spike;
- L1E sensory-weight growth;
- stable L1E firing cadence;
- first selective predictive-feedback change;
- first learned paired-output inhibition;
- time to stable pattern specialization.

## Preserve Existing Work

Inspect `git status` before editing. Preserve all existing uncommitted work, including:

- predictive-inhibition controls and visualization;
- charge-chart audit work;
- learned paired-output-gate work;
- neuron-renderer sizing work;
- existing user files and reports.

Do not revert unrelated changes. Reconcile this task with work already present in the
tree rather than replacing whole files from `HEAD`.

## Verification

Run all focused tests, all root tests, JavaScript syntax checks, and diff validation:

```bash
PYTHONPATH=. .venv/bin/python <new focused test files>
for t in test_*.py; do PYTHONPATH=. .venv/bin/python "$t"; done
PYTHONPATH=. .venv/bin/python tests/golden/test_golden_equiv.py
node --check frontend/charge.js
node --check frontend/renderer.js
node --check frontend/inspector.js
git diff --check
```

Run syntax checks on every other modified JavaScript file.

Because this task intentionally changes the active model, an old trajectory golden
may require an explicitly approved update or a legacy-mode comparison. Do not rewrite
golden fixtures silently. First determine whether the changed path is supposed to be
active in the golden configuration. If the golden is intended to cover legacy/default
behavior, keep that path unchanged. If it is intended to cover the redesigned model,
report the exact trajectory changes and request/record approval before updating the
fixture.

The environment has previously shown an unrelated one-ULP
`engine_dashboard__potentials` drift. Reproduce any such drift in an isolated clean
`HEAD` export before classifying it.

Launch the dashboard if practical and verify the Charge Over Time chart through the
full sensory-to-L2-to-predictive sequence.

## Deliverables

Provide:

1. A pre-change audit of live L1E/L2E and L1I/L2I configurations.
2. A table of every intrinsic parameter before and after the redesign.
3. The shared construction/configuration path used to prevent future layer drift.
4. The new sensory-synapse representation and learning semantics.
5. Analytical and measured first-spike latencies.
6. Early-versus-late L1E weight and firing-rate trajectories.
7. The revised predictor-credit timing and evidence source.
8. The effect on learned paired output inhibition.
9. Row-to-column center-versus-novel-pixel measurements.
10. All schedule changes and their justification.
11. All files changed.
12. Complete verification results and any golden implications.
