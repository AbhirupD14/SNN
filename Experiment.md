# Claude Implementation Brief: Local Predictive Inhibition

## Mission

Implement and evaluate one focused hypothesis:

> Giving each L1 inhibitory neuron access to its paired L1 excitatory activity
> will let dense L2 feedback learn context-specific feature predictions, so that
> predictable continued L1 activity is suppressed while surprising activity
> remains available to L2.

This is a local predictive-inhibition experiment, not a general rewrite of the
network. Work from the current implementation, preserve the existing L2
competition mechanism, and report honestly if the hypothesis does not improve
stable specialization.

The target circuit is:

```text
external input -> L1E_i -> all L2E
                    |
                    +------> paired L1I_i

all L2E -----------> L1I_i
                     |
                     +------| paired L1E_i on a later step
```

The intended computation is:

```text
bottom-up local evidence
+ top-down contextual evidence
-> local inhibitory response
-> less repeated predictable evidence
-> relatively more novel evidence reaches L2
```

## Non-negotiable research constraints

- No backpropagation, gradients, labels, classification correctness, or global
  error signal.
- No pattern-specific wiring, row/column masks, or one inhibitory neuron per
  pattern.
- Learning must not read `engine.winner`, `current_pattern`, pattern names,
  owner assignments, or any equivalent simulator-only result.
- An arriving L2E spike is valid presynaptic information. The global winner
  field may be used only for evaluation and visualization.
- L1E, L1I, L2E, and L2I output polarity must continue to obey Dale's principle.
- Preserve deterministic behavior for a fixed seed and input sequence.
- Keep functional model positions separate from frontend-only display spacing.
- Do not silently tune until a one-to-one mapping appears. Use declared metrics
  and report negative results.

## Current implementation: treat these as facts

Read the referenced code before editing, but do not rediscover or contradict the
following current state.

### Topology

The active network contains:

```text
9 L1E
9 L1I
8 L2E
1 shared L2I
```

Current connections are:

```text
external -> L1E
L1I_i ----| paired L1E_i
all L1E --> all L2E
all L2E --> shared L2I
shared L2I -- competitive reset --> all L2E
all L2E --> all L1I
```

There is currently no `L1E_i -> L1I_i` connection.

### L1 predictive-feedback path

- Every L1I currently has an eight-element positive afferent vector, one input
  from every L2E.
- The same task-independent initial feedback vector is copied to all nine L1I
  neurons. They therefore receive the same L2E spike vector and lack a local
  signal that could make their learned predictions feature-specific.
- `l1i_immediate_relay` is false in the active dashboard. L1I is a trainable
  threshold accumulator.
- L2E feedback is delivered to L1I after L2 competition in the same outer
  timestep.
- An L1I spike is stored in `l1i_feedback_delay` and inhibits its paired L1E on
  the next outer timestep.
- The simulator has no general synaptic event queue or reusable delay system.
  The one-step L1 inhibition register is explicit in `SimulationEngine.step()`.

### L1 inhibition is already weighted

Do not reimplement weighted L1 inhibition.

`Neuron.apply_inhibition()` already performs a bounded weighted subtraction:

```text
V_after = max(V_rest, V_before - |w_inhibitory|)
```

The active L1E inhibitory gate is initialized to a magnitude equal to the L1E
threshold, so one L1I event currently cancels one complete external input pulse.
Its learning equilibrium is arranged so that this gate is effectively fixed.

For this experiment, reuse `apply_inhibition()`. The initial predictive preset
may set the gate to a configurable fraction of the L1E threshold and must set
its inhibitory learning rate to zero unless output-gate plasticity is being
tested in a later, separate phase.

Be careful with units: the existing `inhibitory_weight_cap` is used as the
quadratic saturation denominator, whose equilibrium magnitude is its square
root. It is not simply a linear magnitude cap.

### L2 competition is a different mechanism

Current L2 competition uses:

```text
L2E spike -> L2I recruitment -> unweighted competitive-reset event to all L2E
```

`Neuron.apply_competitive_reset()` resets membrane/current traces and applies
the selected loser update. The active loser update is conservative ON-to-OFF
redistribution, with refractory state protecting the current firing neuron from
the loser weight update.

Do not replace this with a learned L2I-to-L2E gate in this experiment. L1
predictive inhibition and L2 competition are independent research variables.
Changing both would make the result uninterpretable.

### L2 evidence accumulation is already constrained

The dashboard uses:

```text
l2e_weight_cap_frac = 1/3
l2_charge_chunks = 20
distance_weighting = true
```

One L1E afferent can therefore contribute at most one third of the L2E threshold
before distance attenuation. Do not add a new “multi-evidence mode” unless a
focused test demonstrates that one ordinary afferent can still determine a
winner under the active preset.

### Configuration and dashboard

- `SimulationEngine.params` is the runtime source of truth.
- `backend/dashboard_config.py` contains the active experiment overrides and the
  small declarative browser-control schema.
- The browser currently exposes 15 active controls in one flat list.
- Advanced mechanisms remain callable from Python and the headless experiment
  runner; they do not need to be restored to the primary UI.
- There is no dashboard config-import/export system to preserve.

Keep the dashboard lean. Do not build a nested configuration framework or a
large Basic/Advanced/Diagnostics settings system for this experiment.

## Required topology repair

Add exactly nine paired excitatory projections:

```text
L1E_0 -> L1I_0
L1E_1 -> L1I_1
...
L1E_8 -> L1I_8
```

Do not add all-to-all L1E-to-L1I connectivity.

The preferred L1I afferent layout is explicit and consistent:

```text
index 0: paired local L1E input
index 1..8: feedback from L2E_0..L2E_7
```

Update every offset-dependent path together:

- construction and weight initialization;
- input delivery;
- topology serialization;
- `_all_weights()` and weight-delta tracking;
- inspector/weight labels if needed;
- tests that directly manipulate L1I weight arrays.

Serialize the local edge with a stable ID such as `local{i}` and a distinct kind
such as `local_evidence`.

### Local projection behavior

- Use one shared configurable initial magnitude for all nine local projections.
- Keep the local projection fixed in the first experiment.
- Do not initialize it from task structure.
- If it shares a `SynapseBank` with plastic feedback afferents, prevent the normal
  postsynaptic learning rule from modifying it. Prefer a per-afferent plasticity
  mask or a comparably explicit mechanism over index checks scattered through
  learning code.

## Predictive feedback learning

The feedback matrix should learn an estimate of:

```text
how reliably L2E_j predicts recent activity in L1E_i
```

Use a bounded event-local rule. A suitable first rule is defined below and
should be used unless an existing abstraction requires an algebraically
equivalent form.

For L1I `i`, let:

```text
x_i        paired L1E eligibility trace in [0, 1]
y_j        arriving spike from L2E_j (0 or 1)
G          linear feedback-weight cap, normally L1I threshold
u_ji       normalized feedback weight w_ji / G in [0, 1]
eta_up     prediction acquisition rate
eta_down   false-prediction turnover rate
```

On an arriving L2E spike (`y_j = 1`):

```text
u_ji <- clip(
    u_ji
    + eta_up   * x_i       * (1 - u_ji)
    - eta_down * (1 - x_i) * u_ji,
    0,
    1
)

w_ji <- G * u_ji
```

Properties of this rule:

- context plus recent local activity strengthens the prediction;
- context without recent local activity weakens the prediction;
- unrelated feedback afferents are unchanged;
- weights remain bounded;
- every required variable is locally available at the target L1I neuron;
- learning does not require the L1I neuron to spike, avoiding a cold-start
  deadlock.

The local trace must be generated only from paired L1E spikes and must have a
short configurable decay/window. It must not read the input vector directly.

In predictive mode, this rule must be the sole learning rule for L2E-to-L1I
feedback afferents. Do not also run the generic postsynaptic excitatory update on
those same weights. With predictive mode disabled, preserve legacy feedback
learning exactly.

## L1I activation and timing

Use the current explicit phase order rather than building a general event queue.
The minimal intended outer-timestep sequence is:

```text
1. Apply the prior step's queued L1I inhibition to L1E.
2. Deliver external input and resolve L1E spikes.
3. Deliver L1E spikes to L2 and resolve the existing L2 competition.
4. Build one L1I input vector per neuron:
       [paired L1E spike, L2E_0 spike, ..., L2E_7 spike]
5. Deliver that vector once to L1I and update the predictive feedback rule.
6. Resolve L1I spikes.
7. Queue L1I spikes for paired L1E inhibition on the next step.
```

This order is deterministic and ensures that recurrent inhibition never cancels
the initial evidence in the same timestep. It suppresses only continued or
repeated evidence.

The predictive preset should be coincidence-sensitive through ordinary
integration, threshold, leak, and trace timing:

```text
isolated local input:                 normally subthreshold
isolated untrained feedback input:    normally subthreshold
local + sufficiently learned context: more likely to cross threshold
```

Do not add a software Boolean AND gate. Use the existing L1I leak switch and
choose documented initial weights/thresholds that satisfy these constraints over
the declared coincidence window. Add tests for repeated isolated input as well
as a single isolated event so slow accumulation cannot masquerade as
coincidence.

## Implementation phases

Work in these phases and run relevant tests after each one.

### Phase 0: Preserve the baseline

1. Run all retained regression scripts and the golden equivalence test.
2. Record the active dashboard parameters and a deterministic baseline trace.
3. Add no behavior before the baseline is reproducible.

### Phase 1: Add the disabled topology path

1. Add the paired afferent and serialization support behind a default-off flag.
2. Keep its weight fixed.
3. With the flag off, require exact legacy behavior and golden equivalence.
4. Add topology, indexing, weight-delta, and paired-locality tests.

### Phase 2: Add local evidence and predictor learning

1. Add the paired L1 trace.
2. Implement the bounded predictor rule as its own focused strategy/function.
3. Ensure the fixed local afferent is excluded from plasticity.
4. Ensure generic feedback plasticity does not run simultaneously.
5. Add isolated rule tests before network-level tuning.

### Phase 3: Add a predictive preset and ablations

Add named backend/headless presets for:

```text
baseline
    local L1E -> L1I disabled
    predictive feedback rule disabled

feedback_only
    local projection disabled
    legacy feedback behavior retained

local_only
    local projection enabled
    L2E -> L1I input disabled

local_plus_feedback
    local projection enabled
    predictive feedback enabled

time_shuffled_feedback
    same local path and feedback firing-rate statistics
    feedback events decoupled from their true presentation/context in time
```

Do not use a fixed permutation of L2E identities as the main shuffled control.
L2E identities are exchangeable, so the network can simply relearn a permuted
mapping. Break the temporal context-feature relationship while preserving rates.

### Phase 4: Instrument and evaluate

Use the headless experiment runner for sweeps and durable results. The browser is
for inspection, not the primary batch-experiment controller.

Only after the fixed-output-gate experiment is understood may a separate phase
test slow `L1I -> L1E` gate plasticity. Do not change L2 hard-reset competition as
part of this brief.

## Required tests

### Regression

- Every existing retained root regression script passes when the new feature is
  disabled.
- `tests/golden/test_golden_equiv.py` remains exact in disabled/baseline mode.
- Same seed, configuration, and input sequence produce identical histories.

### Topology and indexing

- Exactly nine `L1E_i -> L1I_i` local edges exist when enabled.
- No cross-paired local edges exist.
- Every L1I retains all eight candidate L2E feedback afferents.
- Weight serialization uses the correct local/feedback offsets.

### Predictor rule

- Context spike plus active local trace increases only `w_ji`.
- The same context spike with no local trace decreases only `w_ji`.
- Weight updates clamp to `[0, G]`.
- A disabled predictor is behaviorally identical to the current feedback rule.
- The fixed local afferent never changes.

### Activation and inhibition

- Isolated local events do not cause broad repeated L1I firing under the
  predictive preset.
- Untrained feedback alone does not cause all nine L1I neurons to fire.
- Paired local evidence plus a learned context produces more L1I response than
  either input alone.
- An L1I spike affects only its paired L1E on the next outer timestep.
- `L1I -> L1E` charge removal remains graded, bounded at rest, and uses the
  existing `apply_inhibition()` path.
- L2 competition remains the existing unweighted hard-reset mechanism.

### Recovery

- After reversing a context-feature contingency, the old prediction weight
  declines and the new one grows.
- No L1E feature becomes permanently silent after the context disappears.

## Experimental design

Run every ablation with matched:

- seed;
- initial L2 feedforward weights;
- pattern order;
- dwell length;
- number of presentations;
- active L2 competition parameters.

The four center-crossing patterns alone cannot prove contextual prediction:
frequency-only local inhibition can learn that the center pixel is common.

Therefore report both:

1. performance on the existing four-pattern task; and
2. a context test where feature marginal frequencies are matched but the
   context-feature relationship is broken by time-shuffled feedback.

Compare `local_only` with `local_plus_feedback` at matched or reported total
L1I spike count and inhibitory charge. Otherwise additional inhibition could be
mistaken for contextual prediction.

## Metrics

Record per presentation or fixed reporting window:

- L1E, L1I, L2E, and L2I spike counts;
- L2 spike/winner history for evaluation only;
- L2E-to-L1I predictor-weight matrix;
- fixed local L1E-to-L1I weights;
- L1I-to-L1E gate magnitudes;
- L1I events classified as local-only, feedback-only, or coincident;
- membrane value before and after L1 inhibition;
- inhibitory charge removed;
- pattern owner, visit-level owner consistency, owner collisions, and dead L2E
  count;
- recovery time after a contingency reversal.

Define a contextual suppression contrast such as:

```text
suppression(feature i | learned context j)
- suppression(feature i | rate-matched wrong/shuffled context)
```

The primary research questions are:

1. Does local-plus-feedback suppress a feature more selectively than local-only?
2. Does that selectivity improve stable L2 specialization or merely reduce total
   firing?
3. Does the circuit recover when a learned prediction stops being true?

Stable one-to-one ownership is an outcome metric, not a software acceptance
test. Report failure rather than changing unrelated mechanisms until it passes.

## Dashboard scope

Do not rebuild the dashboard.

If browser controls are needed, add at most:

- a predictive-inhibition preset selector;
- enable/disable paired local evidence;
- enable/disable predictor learning;
- one shared local-weight or output-gate-strength control.

Keep detailed rates, trace decay, and ablation parameters in Python/headless
configuration. Reuse `backend/dashboard_config.py`; do not duplicate defaults in
frontend JavaScript.

Add read-only diagnostics only where existing inspector/chart components can
display them with a small change. Prefer the headless artifacts for matrices and
long-run comparisons.

## Likely files to inspect

- `backend/simulation.py`: construction, phase ordering, serialization, params.
- `backend/dashboard_config.py`: active preset and small browser schema.
- `neuron_flexible.py`: existing weighted inhibition and learning dispatch.
- `snn/synapses.py`: aligned afferent state and any plasticity mask.
- `snn/rules/`: focused local learning strategies.
- `layers.py`: current L1 E/I construction.
- `backend/serializer.py`: topology/dynamic envelopes.
- `frontend/inspector.js` and focused chart modules: optional diagnostics.
- `experiments/runner.py`: matched-seed ablations and metrics.
- existing L1 feedback, inhibition, competition, and golden tests.

Do not create a second simulator or a parallel configuration system.

## Deliverables

1. A short pre-edit architecture map naming exact methods and current array
   layouts.
2. Baseline test/metric results.
3. Small, phased commits or clearly separated diffs.
4. New local-path and predictor-rule tests.
5. Matched-seed ablation support.
6. A concise result report containing parameters, traces, predictor matrices,
   contextual suppression contrasts, specialization metrics, and failures.
7. A list of limitations and the next single experiment justified by the data.

## Final principle

The reusable tile is:

```text
receive initial bottom-up evidence
transmit it upward
receive contextual spikes
combine context with paired local evidence
suppress only continued predictable activity
preserve relatively surprising activity
unlearn predictions that stop being true
```

The row/column behavior must emerge from local traces, dense candidate feedback,
bounded synaptic weights, and spike timing. Never encode the answer in topology,
labels, simulator-owned winner updates, or task-specific rules.
