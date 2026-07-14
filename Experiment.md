# CIPP Technical Specification: Local Predictive Inhibition and Dashboard Simplification

## 1. Objective

Modify the existing CIPP implementation to restore and improve the predictive inhibitory mechanism between L2E, L1I, and L1E while preserving the current architecture and biological intent.

The implementation must:

1. Work from the existing codebase and current topology.
2. Preserve the current neuron, synapse, simulation, training, and visualization infrastructure wherever possible.
3. Restore inhibitory synapses as weighted gates rather than unconditional membrane resets.
4. Add local L1E information to each paired L1I neuron.
5. Allow L1I neurons to combine local activity with L2E feedback.
6. Support gradual, bounded suppression of predictable L1E activity.
7. Avoid introducing pattern-specific inhibitory neurons or manually engineered pattern circuits.
8. Simplify the dashboard so that normal experiments expose only the most important controls.
9. Keep advanced configuration available, but remove it from the primary workflow.

The broader goal is to improve the reusable cortical-column fabric, not to hard-code a solution for row and column patterns.

---

# 2. Research Context

CIPP is a biologically inspired spiking neural network architecture based on local neuronal and synaptic dynamics.

The system does not use:

* backpropagation;
* global error signals;
* differentiable layers;
* dense matrix optimization as the learning mechanism;
* task-specific learning rules.

The current experiment uses a simplified abstraction of cortical columns.

A single L1E or L1I neuron should not be interpreted as a literal biological neuron. Each current unit is an aggregate surrogate for a future excitatory or inhibitory population inside a cortical column.

The current implementation is intended to identify the correct reusable computation before replacing these abstract units with larger populations.

The desired reusable column-level computation is:

```text
bottom-up evidence
+
top-down contextual prediction
→
local predictive inhibition
→
reduced transmission of expected information
```

The system should preferentially preserve novel or surprising evidence while reducing repeated, contextually predictable evidence.

---

# 3. Current Topology

The current network contains:

```text
9 L1E neurons
9 L1I neurons
8 L2E neurons
1 L2I neuron
```

Current connectivity:

```text
Input → L1E

L1E → L2E
dense feedforward connectivity

L2E → L2I
L2I → L2E
shared L2 competition

L2E → L1I
dense feedback connectivity

L1I → paired L1E
local inhibition
```

Each L1I neuron is paired with one L1E neuron.

Conceptually:

```text
L1I_i ─| L1E_i
```

The existing dense L2E-to-L1I connectivity should remain available. It represents candidate feedback connectivity. Functional selectivity should emerge through synaptic weights and plasticity rather than manually defined topology.

---

# 4. Problem Statement

## 4.1 L1I lacks sufficient local information

At present, L1I neurons primarily receive a shared or highly similar L2E feedback stream.

This creates a locality problem.

An L1I neuron may know:

```text
an L2 hypothesis fired
```

but not reliably know:

```text
my paired L1E feature was active
```

Because the L1I neurons receive similar higher-level feedback, they can form arbitrary phase groups or synchronized behavior based on random initialization rather than meaningful feature-specific statistics.

The system needs a local input path:

```text
L1E_i → L1I_i
```

This gives every L1I neuron access to the activity of its paired excitatory feature.

---

## 4.2 Inhibition currently wipes membrane charge

The current inhibitory behavior reportedly sends the target neuron directly to resting potential.

Conceptually:

```text
on inhibitory spike:
    V_target = V_rest
```

This is too coarse.

It acts as an unconditional reset rather than a synaptic interaction with a learnable strength.

It cannot represent:

* weak prediction;
* moderate prediction;
* strong prediction;
* partial charge removal;
* uncertainty;
* gradual learning;
* different suppression strengths for different contexts.

The inhibitory gate behavior must be restored.

An inhibitory event should remove an amount of membrane charge determined by a synaptic weight.

Conceptually:

```text
V_target = max(V_rest, V_target - inhibitory_weight)
```

The exact implementation should remain consistent with the existing neuron and synapse abstractions.

If the simulator uses conductances, currents, or signed postsynaptic potentials rather than direct voltage updates, implement the equivalent bounded inhibitory effect through the existing mechanism.

Do not introduce a detached special-case operation if the current synapse model can support this behavior.

---

## 4.3 One-spike integration can make feedback causally irrelevant

Predictive inhibition can only affect evidence that has not already irreversibly determined an L2 winner.

If a single L1E spike immediately causes an L2E neuron to cross threshold, then later L2E feedback through L1I cannot retract that spike.

This specification does not require a complete redesign of L2 integration, but the implementation must preserve a regime in which inhibitory feedback can influence subsequent evidence accumulation.

At minimum, the system should support one or more of the following:

* L2E requires evidence from multiple afferents;
* L2E requires multiple spikes;
* L2E integrates across a short temporal window;
* winner selection remains provisional for a short period;
* inhibition reduces later spikes during the same presentation;
* predictive state persists briefly;
* predictable L1E output is attenuated before it contributes additional charge to L2E.

Do not solve this by hard-coding pattern-specific delays or manually selecting winners.

---

# 5. Intended Predictive Computation

Consider a learned row pattern represented by L2E neuron A.

After training:

```text
row1 activates L1E features
→
L2E_A learns the row1 combination
→
L2E_A feeds back into the L1I population
→
the L1I neurons paired with row1 features become active
→
those L1E features have their continued firing reduced
```

The inhibition is not meant to remove the first evidence required to recognize the pattern.

It is meant to reduce repeated or continued evidence once it has become predictable.

The desired sequence is approximately:

```text
1. L1E transmits initial bottom-up evidence.
2. One or more L2E hypotheses begin integrating that evidence.
3. An L2E hypothesis becomes active.
4. L2E feedback reaches L1I.
5. L1I combines feedback with paired L1E activity.
6. Predictable L1E activity is partially suppressed.
7. Unpredicted L1E activity continues contributing evidence.
8. Competing L2E hypotheses may then overtake an incorrect initial hypothesis.
```

For example:

```text
row1 trained by neuron A
col1 later presented
```

If neuron A activates initially because of a shared feature such as the center pixel, A's feedback should suppress only the local activity that A predicts and that is actually active.

The remaining col1-specific features should continue transmitting evidence, allowing neuron B to become dominant.

The circuit should therefore support iterative hypothesis correction rather than irreversible first-spike commitment.

---

# 6. Required Topology Changes

Do not replace the existing architecture.

Make the smallest topology repair necessary.

## 6.1 Add paired local excitation

Add:

```text
L1E_i → L1I_i
```

for all nine L1 pairs.

This should be one-to-one.

```text
L1E_0 → L1I_0
L1E_1 → L1I_1
...
L1E_8 → L1I_8
```

This connection provides local feature evidence.

It should not encode a pattern.

Initial recommendation:

* make this connection fixed or very slowly plastic;
* keep its weight identical across all L1 pairs by default;
* make the weight configurable through one shared parameter;
* do not expose nine individual weights in the normal dashboard.

The connection should be implemented using the existing synapse system.

---

## 6.2 Preserve dense L2E-to-L1I feedback

Keep:

```text
all L2E_j → all L1I_i
```

The dense connectivity represents a pool of possible contextual predictions.

The weights should determine which L2E neurons functionally predict which L1E columns.

Do not manually preassign:

```text
L2E_A → only row pixels
L2E_B → only column pixels
```

Instead, allow those relationships to emerge through learning.

---

## 6.3 Preserve paired L1I-to-L1E inhibition

Keep:

```text
L1I_i ─| L1E_i
```

Do not add one inhibitory neuron per pattern.

Each L1I remains the local inhibitory controller for one abstract L1 column.

The identity of the higher-level predictor should be represented in the incoming L2E-to-L1I synaptic weights.

For example, one L1I can receive predictive feedback from several L2E neurons:

```text
L2E_A ─┐
L2E_B ─┼→ L1I_center ─| L1E_center
L2E_C ─┘
```

This is intentional.

Different L2 contexts may predict the same lower-level feature.

---

# 7. Inhibitory Gate Requirements

## 7.1 Replace unconditional reset behavior

Remove or disable the behavior in which an inhibitory event always performs:

```text
V = V_rest
```

Replace it with a bounded weighted inhibitory operation.

Preferred conceptual behavior:

```text
available_charge = max(0, V - V_rest)
removed_charge = min(available_charge, w_inhibitory)
V_new = V - removed_charge
```

Equivalent simplified form:

```text
V_new = max(V_rest, V - w_inhibitory)
```

The implementation must not push the neuron below rest unless the existing biological neuron model explicitly supports inhibitory reversal potentials and conductance dynamics.

---

## 7.2 Weight bounds

The inhibitory weight must be bounded.

Default maximum:

```text
max_inhibitory_weight = V_threshold - V_rest
```

This allows a sufficiently strong inhibitory event to remove all charge accumulated above rest, while preventing arbitrary negative membrane values.

If the simulator uses normalized membrane potential, adapt the bound accordingly.

If inhibitory conductance is used, derive an equivalent cap that ensures stable behavior and document it.

Required constraints:

```text
0 <= inhibitory_weight <= inhibitory_weight_cap
```

Weights must be clamped after every update.

---

## 7.3 Preserve gate semantics

The previous inhibitory gate behavior should be restored conceptually.

The gate is a weighted local effect, not a Boolean switch.

The gate should support:

```text
weight = 0
no inhibition

small weight
minor charge reduction

medium weight
partial suppression

weight near cap
return approximately to rest
```

Avoid multiplying arbitrary activation values through a nonbiological software gate if a synaptic voltage, current, or conductance effect can express the same behavior.

---

# 8. Plasticity Responsibilities

The code should clearly distinguish the roles of the different connections.

## 8.1 L1E-to-L1I

Purpose:

```text
provide local evidence that the paired column is active
```

Initial implementation:

* fixed weight by default;
* optionally configurable as slowly plastic through an advanced flag;
* no pattern-specific initialization;
* no per-pair manual setup.

This connection is primarily an information path, not the main prediction memory.

---

## 8.2 L2E-to-L1I

Purpose:

```text
learn which higher-level contexts predict which lower-level columns
```

This is the primary location for context-specific predictive learning.

A weight:

```text
w_feedback[j, i]
```

should represent approximately:

```text
how strongly L2E_j predicts activity in L1E_i
```

Strengthening condition should rely only on locally available signals or traces, such as:

* presynaptic L2E activity;
* postsynaptic L1I activity;
* paired L1E activity exposed to L1I;
* eligibility traces;
* local timing relationships.

The rule must not inspect global pattern labels.

The rule must not be given row or column identities.

The rule must not use classification correctness.

Where supported by the existing framework, the connection should also weaken when an L2 context repeatedly occurs without the paired L1 feature.

This can be implemented using local traces and a delayed learning window.

---

## 8.3 L1I-to-L1E

Purpose:

```text
control how strongly predicted activity is suppressed locally
```

Recommended phased implementation:

### Phase 1

Keep L1I-to-L1E weights fixed but weighted and bounded.

Use L2E-to-L1I plasticity to learn prediction identity.

This makes debugging easier because only one side of the feedback path is learning.

### Phase 2

Allow L1I-to-L1E inhibitory efficacy to adapt slowly.

This weight may then encode:

```text
how much suppression this lower-level column requires
```

Do not enable both new plasticity mechanisms simultaneously without configuration flags and clear logging.

---

# 9. L1I Activation Requirements

L1I should integrate:

```text
paired local L1E input
+
L2E contextual feedback
```

The desired default behavior is coincidence-sensitive.

Conceptually:

```text
L1E alone:
weak or insufficient activation

L2E feedback alone:
weak or insufficient activation

L1E + relevant L2E feedback:
reliable L1I firing
```

This does not need to be literal multiplication.

It can arise from ordinary membrane integration and thresholds.

The implementation should provide defaults that allow local and contextual input to combine within a short temporal window.

Avoid immediate same-timestep feedback that creates ambiguous update-order behavior.

Preferred event sequence:

```text
t:
L1E spikes

t + feedforward delay:
L2E integrates or spikes

t + feedback delay:
L1I receives L2 context

t + local inhibitory delay:
L1I suppresses later L1E activity
```

All delays should use the existing synaptic delay mechanism.

---

# 10. Timing and Update Order

Claude must inspect the existing simulator update order before modifying behavior.

Document whether the current loop performs:

```text
input update
synaptic delivery
membrane integration
spike detection
reset
plasticity
```

or another order.

The new recurrent path must not depend on accidental iteration order in an array.

Requirements:

* L1E-to-L1I events use explicit delays;
* L2E-to-L1I feedback uses explicit delays;
* L1I-to-L1E inhibition uses explicit delays;
* no connection should have magical same-call effects outside the event system;
* simulation behavior must remain deterministic under a fixed random seed.

If the existing engine has a zero-delay mode, do not use zero-delay recurrence as the default.

---

# 11. L2 Competition

Preserve the current L2 topology:

```text
8 L2E
1 L2I

L2E → L2I
L2I ─| L2E
```

Do not replace the shared L2I neuron as part of this task unless required to fix a direct bug.

The L2I circuit remains responsible for competition among L2E hypotheses.

The L1I circuit is responsible for predictive suppression of lower-level evidence.

These roles should remain distinct.

```text
L2I:
which L2 hypothesis dominates?

L1I:
which lower-level evidence is expected under the current hypothesis?
```

---

# 12. One-Spike Integration Safeguard

Inspect whether an ordinary single L1E spike can currently cause an L2E neuron to cross threshold immediately.

If so, add a configurable safeguard using the smallest code change consistent with the current architecture.

Acceptable approaches include:

* reduce default L1E-to-L2E synaptic strength;
* increase the L2E threshold;
* require multiple afferent contributions;
* introduce a short competition or evidence window;
* delay winner finalization;
* make a single afferent incapable of contributing the entire threshold under normal defaults.

Do not permanently prohibit single-spike activation. It may be useful in future experiments.

Instead, expose one high-level preset or parameter controlling evidence accumulation.

Recommended high-level modes:

```text
multi_evidence
single_spike_sensitive
```

Default to:

```text
multi_evidence
```

for predictive-inhibition experiments.

---

# 13. Biological Fidelity Constraints

The implementation should preserve the following principles.

## Required

* Dale's principle remains enforced.
* L1E neurons produce excitatory outputs.
* L1I neurons produce inhibitory outputs.
* L2E neurons produce excitatory outputs.
* L2I produces inhibitory outputs.
* Plasticity uses local synaptic or neuronal information.
* Feedback is transmitted through spikes and synapses.
* Synaptic delays are explicit.
* Inhibition is graded and bounded.
* No task labels enter the learning rules.
* No pattern-specific neurons are created.
* No manually defined row or column masks are added.
* The same L1 column motif is used for all nine L1 features.

## Acceptable abstractions

Because current units are cortical-column surrogates, it is acceptable for one abstract inhibitory event to approximate a population-level inhibitory effect.

The abstraction should remain replaceable.

A future internal population should be able to replace:

```text
L1E_i
L1I_i
```

without changing the external interfaces:

```text
bottom-up input
feedforward output
top-down context
local inhibitory modulation
```

---

# 14. Tiling Requirements

The resulting motif must be reusable at higher layers.

The generic layer-to-layer pattern should be:

```text
Layer N E
    → feedforward
Layer N+1 E

Layer N+1 E
    → feedback
Layer N I

Layer N E
    → paired local evidence
Layer N I

Layer N I
    ─| Layer N E
```

The implementation should avoid names or assumptions that make the new behavior L1-specific where possible.

Prefer abstractions such as:

```text
PredictiveLayer
ColumnPair
ExcitatoryPopulation
InhibitoryPopulation
FeedbackProjection
LocalEvidenceProjection
```

over hard-coded logic such as:

```text
if layer == 1
```

It is acceptable to wire the current experiment explicitly, but the underlying components should support reuse.

---

# 15. Dashboard Simplification

The current dashboard exposes too many configuration options at once.

The dashboard should be restructured around a small number of experiment-level controls.

Do not delete advanced parameters from the underlying configuration system.

Instead, separate the interface into:

```text
Basic
Advanced
Diagnostics
```

## 15.1 Basic panel

The default view should expose only the controls required for common experiments.

Recommended controls:

### Experiment

* pattern set;
* training pattern;
* number of presentations or epochs;
* random seed;
* start;
* pause;
* reset;
* load preset.

### Learning

* learning enabled;
* excitatory learning rate;
* feedback prediction learning rate;
* inhibitory learning enabled;
* homeostasis enabled.

### Predictive inhibition

* predictive inhibition enabled;
* local L1E-to-L1I input enabled;
* inhibitory gate strength or initial weight;
* inhibitory weight cap;
* feedback delay;
* local inhibitory delay.

### Neuron dynamics

* simulation timestep;
* membrane threshold;
* membrane leak;
* refractory period;
* noise level.

Where possible, several detailed parameters should be represented by a preset rather than individual controls.

---

## 15.2 Presets

Add experiment presets.

At minimum:

### Baseline

Current or nearest reproducible behavior without the new local predictive repair.

```text
local L1E → L1I disabled
legacy inhibitory behavior selectable for comparison
```

### Predictive Inhibition

Recommended new defaults.

```text
paired L1E → L1I enabled
weighted inhibitory gates enabled
L2 feedback plasticity enabled
multi-evidence L2 integration
```

### No Feedback

Ablation.

```text
L2E → L1I disabled
```

### Local Only

Ablation.

```text
L1E → L1I enabled
L2E → L1I disabled
```

### Shuffled Feedback

Optional diagnostic ablation.

Use a fixed shuffled mapping or shuffled feedback activity while preserving firing-rate statistics.

Presets should write into the existing configuration object rather than bypassing it.

---

## 15.3 Advanced panel

Move detailed controls here.

Examples:

* individual learning-rule constants;
* STDP windows;
* eligibility trace decay;
* synaptic scaling constants;
* heterosynaptic pruning thresholds;
* per-projection delays;
* adaptation constants;
* conductance decay;
* refractory details;
* maximum and minimum weights;
* normalization frequency;
* pruning cadence;
* per-layer noise;
* initialization distributions.

Use collapsible sections.

Recommended sections:

```text
Neuron Dynamics
Synapse Dynamics
Feedforward Plasticity
Feedback Plasticity
Inhibitory Plasticity
Homeostasis
Pruning and Scaling
Timing and Delays
Initialization
```

---

## 15.4 Diagnostics panel

Provide read-only visualizations and logging controls.

Recommended diagnostics:

* L1E spike counts per feature;
* L1I spike counts per feature;
* L2E spike counts;
* L2 winner over time;
* L2I activity;
* L2E-to-L1I weight matrix;
* L1I-to-L1E inhibitory weights;
* L1E-to-L1I local weights;
* membrane traces for selected neurons;
* delivered inhibitory charge;
* predicted versus unpredicted feature activity;
* active preset;
* random seed;
* simulation step.

Do not put diagnostic plot configuration in the Basic panel.

---

# 16. Dashboard Usability Requirements

The dashboard should:

* open in a usable default state;
* fit the main controls without excessive scrolling;
* avoid exposing every neuron or synapse parameter individually;
* show units for numeric parameters;
* provide concise tooltips;
* group related options;
* hide dependent controls when a feature is disabled;
* mark modified values that differ from the selected preset;
* provide a reset-to-preset action;
* preserve configuration import and export if currently supported.

Avoid a complete dashboard rewrite unless the current structure makes incremental simplification impossible.

Use the existing UI framework and components.

---

# 17. Configuration Refactor

Create or maintain one authoritative configuration object.

The UI should not directly mutate neuron or synapse instances.

Recommended structure:

```text
config:
    experiment
    topology
    neuron
    synapse
    learning
    predictive_inhibition
    timing
    diagnostics
    dashboard
```

Example:

```yaml
predictive_inhibition:
  enabled: true
  local_e_to_i_enabled: true
  local_e_to_i_weight: 0.25
  feedback_plasticity_enabled: true
  inhibitory_gate_mode: weighted_charge_removal
  inhibitory_weight_initial: 0.2
  inhibitory_weight_min: 0.0
  inhibitory_weight_max: threshold_distance
  require_local_context_coincidence: true
  feedback_delay_steps: 1
  inhibitory_delay_steps: 1
```

Do not hard-code physical values from this example without adapting them to the existing simulator's units.

Configuration migration must preserve old saved configs where practical.

If old fields are renamed, provide compatibility aliases or a migration function.

---

# 18. Legacy Comparison Mode

For research reproducibility, retain the old reset behavior behind an explicit legacy flag.

Example:

```text
inhibitory_effect_mode:
    legacy_reset
    weighted_charge_removal
```

Default new experiments to:

```text
weighted_charge_removal
```

Use `legacy_reset` only for comparisons.

The dashboard may place this option under Advanced or inside a comparison preset.

Do not silently change old saved experiment results when loading an old configuration.

---

# 19. Required Instrumentation

Add enough instrumentation to determine whether the new circuit is learning predictability rather than frequency alone.

For every presentation or configurable reporting window, record:

```text
L1E spike count by neuron
L1I spike count by neuron
L2E spike count by neuron
L2 winner identity
L2E → L1I weights
L1I → L1E weights
L1E → L1I weights
inhibitory events delivered
inhibitory charge removed
membrane value before inhibition
membrane value after inhibition
```

Where practical, also record:

```text
L1I events with local input only
L1I events with feedback only
L1I events with both local and feedback traces
```

This will help verify whether L1I acts as a coincidence-sensitive predictive unit.

Logging should be optional and configurable to avoid excessive runtime cost.

---

# 20. Required Experiments

Implement or preserve a way to run the following ablations with the same random seed and initial conditions.

## Experiment A: Current feedback only

```text
L2E → L1I enabled
L1E → L1I disabled
weighted inhibition enabled
```

Purpose:

Measure whether arbitrary shared or phase-based L1I behavior remains.

---

## Experiment B: Local only

```text
L1E → L1I enabled
L2E → L1I disabled
```

Purpose:

Measure whether L1I learns or tracks local firing frequency without context.

---

## Experiment C: Local plus feedback

```text
L1E → L1I enabled
L2E → L1I enabled
```

Purpose:

Test the full predictive mechanism.

---

## Experiment D: Shuffled feedback

```text
L1E → L1I enabled
L2E feedback shuffled
```

Purpose:

Determine whether useful suppression depends on correct contextual relationships rather than additional excitation alone.

---

## Experiment E: Legacy reset versus weighted gate

Compare:

```text
legacy_reset
weighted_charge_removal
```

Use identical seeds and training sequences.

Measure:

* stability;
* L1E spike reduction;
* L2 winner behavior;
* recovery after context changes;
* frequency of permanent silencing;
* oscillatory behavior;
* learned feedback selectivity.

---

# 21. Behavioral Acceptance Tests

## 21.1 Weighted inhibitory gate

Given:

```text
V_rest = 0
V_threshold = 1
V = 0.8
w_inhibitory = 0.3
```

Expected:

```text
V_after = 0.5
```

Given:

```text
V = 0.2
w_inhibitory = 0.8
```

Expected:

```text
V_after = V_rest
```

The result must not go below rest under the default model.

---

## 21.2 Weight cap

An attempted update above the cap must clamp to the cap.

An attempted update below zero must clamp to zero.

---

## 21.3 Paired locality

Activity from `L1E_i` must directly affect only its paired local-evidence projection into `L1I_i`.

Do not accidentally create all-to-all L1E-to-L1I connectivity.

---

## 21.4 Feedback remains dense

Every L2E must retain a candidate connection to every L1I unless a configuration explicitly enables sparse initialization.

---

## 21.5 Context-sensitive suppression

After training a higher-level context that reliably includes feature `i`:

```text
context active + feature i active
```

should cause more suppression of `L1E_i` than:

```text
feature i active without that context
```

---

## 21.6 No suppression from inactive local feature

An L2E context alone should not cause strong repeated L1I firing across all nine L1I units under recommended defaults.

Inactive local columns should remain mostly unaffected.

---

## 21.7 Recovery

If a previously predictive context-feature relationship stops occurring, the associated predictive influence should eventually weaken or cease dominating.

Permanent silencing is a failure.

---

## 21.8 Reproducibility

Two runs with the same:

* seed;
* preset;
* pattern sequence;
* configuration;

must produce the same spike and weight histories, subject to existing deterministic guarantees.

---

# 22. Performance Requirements

The new topology adds only nine paired L1E-to-L1I connections.

The implementation should not materially change asymptotic cost.

Avoid:

* per-pattern inhibitory neuron allocation;
* pattern-by-feature circuit expansion;
* dense history scans on every timestep;
* global loops over all recorded spikes for every plasticity update.

Use existing traces or bounded rolling state.

Dashboard diagnostics should support downsampling or disabled collection.

---

# 23. Code Quality Requirements

Claude should first inspect and identify:

* neuron model classes;
* synapse model classes;
* event-delivery system;
* weight update logic;
* topology construction;
* configuration schema;
* dashboard components;
* experiment runner;
* logging and plotting paths;
* existing tests.

Then provide a brief implementation map before editing.

Changes should:

* reuse existing abstractions;
* avoid duplicate synapse logic;
* avoid special-case checks for specific neuron indices;
* include type annotations where the project uses them;
* add comments explaining computational intent rather than restating code;
* add or update tests;
* preserve current naming conventions;
* preserve backward compatibility where practical.

Do not create an entirely separate predictive-inhibition simulator.

---

# 24. Recommended Implementation Phases

## Phase 1: Inspection and baseline preservation

1. Identify current topology construction.
2. Locate the unconditional inhibitory reset.
3. Confirm current simulation update order.
4. Run the current baseline.
5. Save baseline metrics and plots.
6. Add a reproducible baseline preset.

No behavioral modifications should be made before the baseline can be reproduced.

---

## Phase 2: Weighted inhibitory gate

1. Add the weighted inhibitory mode.
2. Add weight bounds.
3. Retain legacy reset mode.
4. Add unit tests.
5. Expose one high-level dashboard control.
6. Compare legacy and weighted modes.

Do not enable new L1E-to-L1I connections yet in this phase.

---

## Phase 3: Local L1E-to-L1I path

1. Add nine paired local projections.
2. Use fixed shared initial weight.
3. Add delays through the normal event system.
4. Add topology tests.
5. Add local-only ablation.
6. Add diagnostics for local and contextual contributions.

---

## Phase 4: Predictive feedback plasticity

1. Review existing L2E-to-L1I plasticity.
2. Modify only as necessary to use local L1 activity or traces.
3. Ensure learning remains local.
4. Add bounds and decay or depression as supported.
5. Add the full local-plus-feedback experiment.
6. Add shuffled-feedback controls.

---

## Phase 5: Dashboard simplification

1. Add presets.
2. Build Basic, Advanced, and Diagnostics sections.
3. Move low-level controls out of the default view.
4. Preserve config import and export.
5. Add reset-to-preset.
6. Ensure experiments remain reproducible.

---

## Phase 6: Validation

Run all required experiments with matched seeds.

Produce a concise report containing:

* topology used;
* preset;
* relevant parameters;
* spike-count comparisons;
* weight matrices;
* membrane traces;
* L2 winner histories;
* evidence of context-sensitive suppression;
* evidence of stability or failure;
* remaining open issues.

---

# 25. Non-Goals

Do not include the following in this task unless required for compatibility:

* replacing each abstract neuron with a full neural population;
* adding multiple inhibitory interneuron classes;
* adding one inhibitory unit per pattern;
* implementing a full predictive-coding loss;
* calculating explicit prediction errors;
* adding backpropagation;
* adding supervised labels;
* redesigning the entire L2 competition circuit;
* creating a robotics controller;
* scaling beyond the current two-layer experiment;
* rewriting the entire dashboard from scratch;
* introducing pattern-specific masks or hand-coded row/column knowledge.

---

# 26. Open Design Choices

Where the current code does not already determine the answer, prefer the simplest biologically coherent implementation.

Claude may choose implementation details for:

* exact local L1E-to-L1I delay;
* exact feedback delay;
* initial inhibitory weight;
* coincidence window;
* trace decay;
* whether the first version keeps L1I-to-L1E weights fixed;
* whether conductance-based or direct charge subtraction better matches the current simulator.

For each choice:

1. explain the existing code constraint;
2. choose the smallest compatible change;
3. make it configurable;
4. provide a reasonable default;
5. add a test.

Do not introduce new complexity merely because it is more biologically detailed.

---

# 27. Deliverables

Claude should produce:

1. A short codebase architecture summary.
2. A list of files to modify.
3. The implementation changes.
4. Configuration migration or compatibility handling.
5. Dashboard simplification.
6. Unit and integration tests.
7. Presets for baseline and predictive inhibition.
8. Ablation experiment support.
9. A concise validation report.
10. A list of remaining limitations and recommended next experiments.

---

# 28. Final Design Principle

The implementation should treat the current E and I units as abstract cortical-column components.

The goal is not to make the toy network anatomically complete.

The goal is to create a minimal, reusable, biologically grounded computational tile with the following external behavior:

```text
receive bottom-up evidence
transmit initial evidence upward
receive higher-level contextual feedback
combine feedback with local activity
reduce predictable continued activity
preserve novel evidence
recover when predictions change
```

Every change should support that reusable fabric.

Do not solve the present row and column task by embedding its structure into the topology. The row and column behavior must emerge from the existing dense candidate connectivity, local activity, synaptic weights, and plasticity.
