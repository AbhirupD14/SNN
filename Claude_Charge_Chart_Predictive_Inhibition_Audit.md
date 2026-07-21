# Task: Audit and Correct Predictive Inhibition from the Charge Chart

## Objective

Audit the predictive-inhibition mechanism end to end, using the **Charge Over Time**
chart as the primary observable. Then update both the chart and the engine wherever
the implementation does not match the intended learned-inhibition behavior.

Do not assume the current behavior is correct merely because existing tests pass.
Some tests and documentation may encode the current fixed-gate implementation rather
than the behavior now required. This task supersedes earlier language that describes
the predictive `L1I→L1E` output gate as fixed and non-plastic.

The target scenario is:

1. Enable all three predictive controls **before** training:
   - Paired local evidence
   - Predictor learning
   - Deliver L2E feedback to L1I
2. Train `row 1` until an L2E neuron specializes and produces actual spikes for it.
3. Switch to `col 1`.
4. When the trained row specialist actually spikes on the new input, its learned
   feedback should activate the L1I paired with the shared center pixel.
5. That L1I spike should produce a one-step-delayed, weight-mediated inhibitory
   event on the center L1E.
6. The two novel column pixels should initially receive less effective inhibition,
   increasing their relative influence and helping a new L2E neuron acquire the
   column pattern.

## Keep the Two Inhibitory Systems Separate

There are two different mechanisms in the dashboard:

1. `L2I→L2E` competitive reset
   - Serialized as `reset_inhibition` / `reset->{j}`.
   - Structurally unweighted and shown as red ticks in the L2E lanes.
   - This mechanism is outside the requested learning change and should remain an
     unweighted competitive hard reset unless an independent defect is proven.

2. Paired `L1I_i→L1E_i` inhibition
   - Serialized as `li{i}`.
   - This is the mechanism under audit.
   - It must remain paired, delayed by one timestep, and weight-mediated.
   - Its output gate is expected to learn from real inhibitory events and mature
     toward an effective hard reset.

Do not modify the L2 competitive-reset system while attempting to fix the paired L1
predictive-inhibition path.

## Expected Causal Timeline

The Charge Over Time chart should make this sequence auditable:

```text
t:
  center L1E spikes
  a trained L2E source actually spikes
  its learned L2E→L1I feedback reaches the center-paired L1I
  center L1I crosses threshold and spikes

t+1:
  the queued L1I4→L1E4 event is actually applied
  the learned inhibitory weight removes charge from L1E4
  center L1E is reduced or reset
  novel L1E pixels remain comparatively unaffected
```

Do not treat the dashboard's persistent winner label as an emitted feedback event.
Predictive feedback must be tied to an **actual L2E spike** in the charge trace.

## Part 1: Audit the Engine

Trace the complete path through the implementation:

- fixed paired `L1E_i→L1I_i` local-evidence afferent;
- plastic `L2E_j→L1I_i` predictive feedback afferent;
- local trace and predictor update;
- L1I threshold, leak, refractory state, integration, and spike;
- one-step delayed `L1I_i→L1E_i` delivery;
- downstream inhibitory gate initialization, bounds, and learning rule;
- actual pre/post L1E membrane values when inhibition is applied;
- dynamic-state serialization consumed by the charge chart.

Answer each of these questions with code references and deterministic evidence:

1. When an actual L2E spike occurs, is it delivered to the correct L1I afferent?
2. Does predictor learning strengthen `L2E_j→L1I_i` only where recent local
   evidence supports that association?
3. Do feedback weights for inactive/uncommon pixels remain weak or decrease?
4. When local evidence and sufficiently trained feedback coincide, does the intended
   L1I cross its real threshold and spike?
5. Is that L1I spike queued and applied to exactly its paired L1E at `t+1`?
6. What is the `L1I→L1E` weight before and after each applied inhibitory event?
7. Is the output gate genuinely learned, or is it currently assigned a fixed value
   and frozen?
8. Does a mature learned gate become an effective hard reset because its weight is
   large enough to remove all available charge down to rest?
9. Do inactive pixels retain weak output gates because they did not receive relevant
   training events?
10. Do the topology and dynamic payload serialize every neuron's actual firing
    threshold, especially L1I and L2I?

Build a small deterministic diagnostic trace if necessary. Record, per timestep:

- active input pattern;
- L1E spikes;
- actual L2E spikes;
- delivered feedback vector;
- relevant local traces;
- relevant `L2E→L1I` weights;
- L1I pre-charge, post-delivery charge, threshold, refractory state, and spike;
- queued and applied `L1I→L1E` events;
- output-gate weight before/after learning;
- L1E potential before/after inhibition.

## Part 2: Implement Learned Paired Output Gates if the Audit Confirms They Are Fixed

The intended model is:

- `L1E→L1I` local evidence is fixed and non-plastic.
- `L2E→L1I` feedback is trained by the predictor.
- `L1I→L1E` output inhibition begins weak/untrained and learns only through real
  paired inhibitory events.
- Repeated successful inhibition of an active L1E strengthens its paired output gate.
- A mature learned gate can behave like a hard reset, but only because its learned
  weight has become large enough—not because the engine bypasses the synapse with an
  unconditional reset.
- Pixels that did not participate in prior training retain weaker output gates.

If the current predictive path sets all output gates to the same fixed fraction of
`theta_L1E` and disables inhibitory plasticity, replace that behavior with learned
output gates.

Implementation requirements:

1. Reuse the existing inhibitory plasticity machinery where its rule matches the
   intended behavior. If it does not, explain precisely why and make the smallest
   coherent rule change.
2. Keep inhibition subtractive and weight-mediated. Clamp the target at resting
   potential; never drive it below rest.
3. Do not implement an unconditional `L1I→L1E` reset.
4. Initialize the learned gate weakly enough that "untrained" has a meaningful
   behavioral effect, but nonzero if the learning machinery requires an inhibitory
   synapse to exist before it can update.
5. Bound the learned magnitude to a documented fraction of `theta_L1E`. The full
   predictive configuration must permit maturation to an effective reset.
6. Update only the paired gate whose L1I produced the actual inhibitory event.
7. Preserve the one-step delay and prohibit cross-pair delivery.
8. Do not make the fixed local-evidence afferent plastic.
9. Do not allow generic L1I afferent learning to run simultaneously with predictor
   learning.
10. Preserve legacy/default-OFF behavior exactly.

Audit the meaning of `predictive_output_gate_frac`. It currently appears to describe
a fixed output magnitude. Once the output gate is learned, the frontend and engine
must use truthful semantics. A reasonable minimal design is to make it the learned
maximum/target fraction, but choose the design only after tracing the existing
learning bounds. If a separate initialization fraction is necessary, keep it an
engine parameter unless another frontend control is genuinely justified.

Update all affected labels, descriptions, comments, tests, experiment configuration,
and reports. Do not leave documentation claiming that the gate is fixed or frozen.

## Part 3: Make the Charge Chart Show the Real Events

The current red ticks in L2E lanes represent `L2I→L2E` competitive resets. Preserve
them.

Add an explicitly different marker for an **applied** `L1I_i→L1E_i` inhibitory
event in the affected L1E lane. The marker must appear at the timestep when L1E
inhibition is applied, not one timestep earlier when L1I initially spikes.

Do not infer applied inhibition from `emitted: ["li4"]`. That emission represents the
source L1I spike at `t`; the actual queued inhibition occurs at `t+1`. Serialize the
authoritative applied event from `_inh_events` or an equivalent engine-side record.

The dynamic event should include at least:

- synapse ID;
- source L1I ID;
- target L1E ID;
- applied timestep;
- inhibitory weight before learning;
- inhibitory weight after learning;
- learning delta;
- target potential before inhibition;
- target potential after inhibition;
- charge removed;
- whether the event reached rest and therefore acted as an effective reset.

Chart requirements:

1. Preserve the existing L2 competitive-reset ticks.
2. Show L1I spikes as spikes in their own lanes, as today.
3. Add a visually distinct marker for applied `L1I→L1E` inhibition in L1E lanes.
4. Clearly distinguish in the legend:
   - L2 competitive reset;
   - L1I spike;
   - applied paired L1 inhibition.
5. Make weak partial inhibition distinguishable from a mature effective reset when
   practical, for example through marker height or intensity.
6. Expose exact event values through a tooltip, inspector, or event-detail panel if
   the chart architecture supports it without a large rewrite. At minimum, retain
   them in the dynamic payload and event log.
7. Keep chart history bounded and avoid introducing per-frame allocation or rendering
   work that scales with the full unbounded simulation history.

Also fix threshold normalization. The chart must normalize membrane charge using each
live neuron's real threshold. Verify L1I and L2I metadata in particular; do not use
L1E or L2E thresholds as substitutes. Add tests that compare serialized thresholds
against the corresponding live neuron objects.

## Part 4: Deterministic Behavioral Tests

Add tests that prove the causal chain rather than only checking flags or topology.

### 1. Predictor selectivity

- Train a known L2E source while row-local evidence is active.
- Its feedback weights to row-paired L1Is increase.
- Its weights to inactive pixels remain weak or decrease.
- The fixed local afferents do not move.

### 2. L1I activation

- A trained feedback event plus center-local evidence causes the center L1I to spike.
- Weak feedback to a novel-pixel L1I does not produce the same result under equivalent
  initial conditions.
- The test uses the real L1I threshold and accounts for refractory state.

### 3. Delayed paired inhibition

- L1I4 spikes at `t`.
- No `li4` inhibition is applied to L1E4 at `t`.
- The actual event is applied at `t+1`.
- No other L1E receives the event.

### 4. Output-gate plasticity

- The gate for a repeatedly active pair strengthens across real inhibitory events.
- An inactive/untrained pixel's gate remains at initialization.
- The learned gate stays within its configured bounds.
- Partial gates subtract only their learned magnitude.
- A mature gate removes the expected charge and floors the target at rest.

### 5. Dynamic serialization and chart timing

- The applied event appears in the dynamic payload at `t+1` with the correct IDs,
  weights, delta, pre/post potential, and removed charge.
- The source L1I spike remains represented at `t`.
- The L2 reset event type remains separate.

### 6. Threshold correctness

- Serialized L1I and L2I thresholds equal the live neurons' thresholds.
- Dynamic activation uses those same thresholds.

### 7. Row-to-column integration

Run a deterministic row-training phase followed by column presentation and report:

- which L2E sources actually spike in each phase;
- learned feedback weights from the row specialist to pixels 1, 4, and 7;
- learned output-gate weights for pixels 1, 4, and 7;
- L1I spike counts for pixels 1, 4, and 7;
- applied L1 inhibition counts and magnitudes;
- L1E firing rates for the shared center and the two novel pixels;
- subsequent L2 competition and recruitment.

Demonstrate that the shared center is suppressed more strongly than the novel pixels.
Report the measured novel-to-center firing-rate ratio. The intended qualitative
effect is a substantial novelty advantage, with approximately 2x as the conceptual
target when the trained predictor fires at the required cadence. If that ratio is not
reached, identify whether the limiting factor is:

- insufficient `L2E→L1I` learning;
- L2 spike timing/frequency;
- L1I threshold, leak, or refractory behavior;
- output-gate initialization or learning;
- the one-step delivery schedule.

Do not weaken the test merely to make the current implementation pass. Separate a
deterministic unit-level proof of the intended mechanism from the emergent task-level
measurement when necessary.

## Part 5: Preserve Existing Work and Verify

Inspect `git status` before editing. Preserve all existing uncommitted predictive-
inhibition and neuron-rendering changes. Do not revert unrelated user work.

Run:

```bash
PYTHONPATH=. .venv/bin/python <new focused test files>
for t in test_*.py; do PYTHONPATH=. .venv/bin/python "$t"; done
PYTHONPATH=. .venv/bin/python tests/golden/test_golden_equiv.py
node --check frontend/charge.js
node --check frontend/renderer.js
node --check frontend/inspector.js
git diff --check
```

Run syntax checks on any other modified JavaScript files as well.

The environment has previously produced a one-ULP drift in
`engine_dashboard__potentials`. If it occurs, reproduce the golden test in an isolated
clean `HEAD` export before classifying it as a regression. Do not claim it is
pre-existing without that comparison.

If practical, launch the dashboard and visually confirm the complete `t`/`t+1`
sequence in the Charge Over Time chart.

## Deliverables

Provide:

1. A concise audit report describing the pre-change behavior.
2. Code references and a timestep-level trace proving where the causal chain worked
   or failed.
3. The exact mismatch between the old implementation and the learned-output-gate
   model.
4. The engine changes and the rationale for the chosen initialization, learning rule,
   and cap.
5. The updated meaning of every affected frontend control.
6. The charge-chart event types, timing semantics, and legend.
7. Before/after predictive-feedback and output-gate weights.
8. Row-versus-column firing-rate measurements and the novel-to-center ratio.
9. All files changed.
10. Complete verification results, including any independently reproduced golden
    drift.
