# Claude Sparse Flow Rate Prompt

Please inspect the current charge accumulation path in the repo, especially
`neuron_flexible.py`, `backend/simulation.py`, config/API code, frontend config
controls, and the methodology docs.

Goal: add an optional sparse/event-driven flow-rate model for positive charge
accumulation, behind a config flag, while preserving the current instantaneous
`V += dot(weights, spikes)` behavior as a toggleable baseline.

This does not replace the discrete event winner-processing procedure. It only
changes how positive charge is accumulated before threshold checks.

## Motivation

Currently synaptic weights behave like instant charge packets:

```text
V += w
```

A more realistic interpretation is:

```text
weight = gate/current amplitude
spike opens a current trace
membrane charge integrates that current over time
current decays
```

This should be implemented in a scalable sparse/lazy way, not as a dense
per-neuron per-timestep sweep.

## Winner Processing Stays Event-Based

Keep the existing event-based winner-processing policy:

```text
1. Add/advance charge to neurons for the current outer timestep.
2. Check theta/threshold for relevant neurons and collect threshold-crossers.
3. Pick the max-charge threshold-crosser as the single winner where the current
   architecture expects winner-take-all arbitration.
4. Process that winner's fire.
5. Process inhibition for the other excitatory neurons.
6. Advance to the next outer timestep and repeat.
```

The reason is unchanged: discrete timesteps can allow several neurons to cross
threshold in the same represented instant even though a continuous-time system
would have a first crosser. The max-charge selection is an explicit
pseudo-continuous arbitration policy. The flow-rate work below only changes the
charge-addition model used before this arbitration.

## Scope Across Neuron Types

Flow-rate accumulation should apply to neurons that integrate positive synaptic
charge, not only L2E.

Apply flow-rate mode to positive charge accumulation into:

```text
L2E: from L1E spikes
L2I: from L2E spikes, unless intentionally kept as an immediate relay
L1I: from L2E feedback, unless immediate relay mode is enabled
future L3E/L3I: by default when those layers are added
```

L1E is exempt for now because it is being treated as an abstract pretrained
sensory source. Do not make L1E itself use flow-rate accumulation unless the
repo already has a clear reason to do so.

If there is already an `l1i_immediate_relay` or similar mode, preserve it:
immediate relay mode should bypass learned/trace-based L1I threshold integration.

## Desired Modes

### 1. Instantaneous mode

Current/default-compatible behavior:

```text
on input spike:
  V += dot(weights, spikes)
```

Existing behavior should be preserved when the new flag is off.

### 2. Flow-rate mode

On input spike:

```text
lazily advance this neuron's local excitatory trace to current time
add new weighted drive into an excitatory current trace
integrate current into V using closed-form skipped-time math
```

Do not update every neuron every timestep just to decay traces. Only touched
neurons or active/scheduled trace-bearing neurons should pay the update cost.

Add config such as:

```text
excitatory_flow_rate = False
exc_trace_decay = 0.8
exc_trace_normalized = True
```

Add local state per neuron:

```text
exc_trace
exc_trace_last_t
```

or equivalent names.

## Sparse/Lazy Trace Math

Assume the discrete model:

```text
each timestep:
  V += I
  I *= d
```

where:

```text
I  = excitatory current trace
d  = exc_trace_decay
dt = current_time - exc_trace_last_t
```

When a neuron is touched at time `t`, first lazily advance its trace:

```text
if dt > 0:
  V += I * (1 - d**dt) / (1 - d)
  I *= d**dt
  exc_trace_last_t = t
```

Important: with flow-rate traces, a neuron can cross threshold on a timestep
where it receives no new spike, because residual current is still flowing. The
implementation must account for threshold checks after trace advancement even on
no-new-input timesteps. For the current small repo, an active-trace set is
acceptable. For future scale, this should point toward sparse scheduled events
or predicted-crossing checks rather than dense global sweeps.

Handle edge cases:

- If `d == 0`, the trace contributes only once.
- If `d` is very close to 1, avoid numerical instability.
- Clamp/saturate `V` exactly as the existing membrane saturation logic does.
- Respect refractory behavior consistently with current semantics. If
  instantaneous mode ignores charge during refractory, flow mode should not
  accidentally accumulate hidden current unless deliberately documented.

## Injection Rule

In flow-rate mode, when input spikes arrive:

```text
drive = dot(positive/excitatory weights, spikes)
```

If `exc_trace_normalized=True`, inject:

```text
I += drive * (1 - d)
```

so the infinite total delivered charge is approximately `drive`, matching
current instantaneous total charge over time.

If `exc_trace_normalized=False`, inject:

```text
I += drive
```

Default should probably be normalized for comparability.

## Ordering

On each outer timestep, for every neuron that is touched by a new input event or
has an active trace that needs advancement:

1. Lazily advance the neuron to `t`.
2. Update `V` from residual current flow.
3. Inject any new drive from incoming spikes.
4. Decide whether newly injected current contributes to `V` immediately on the
   same outer timestep or beginning next timestep.

Pick one behavior, document it, and keep it deterministic. I recommend
same-timestep contribution by injecting then applying one local trace integration
step, because it preserves responsiveness.

Threshold checks should occur after the trace contribution to `V` for that
timestep.

This means threshold checks may occur on timesteps with no new external input
arrival. The charge flow itself can be the event that pushes a neuron across
theta.

## L1E Firing Frequency Nuance

Because L1E is abstracted/pretrained, it can remain the sensory spike source.
However, flow-rate mode changes the downstream meaning of an L1E spike: one L1E
spike opens downstream current traces rather than depositing the full charge
instantly.

Please inspect how often L1E currently fires while an input pattern is held. If
L1E currently fires only once per displayed pattern or too sparsely for the new
flow model, document the implication and, only if needed, add a clearly scoped
configurable option for sustained sensory firing while a pixel/pattern remains
active. Do not make this a large redesign; the primary change is the downstream
flow-rate accumulation.

## Interaction With Chunking

- `l2_charge_chunks` should only apply to instantaneous charge mode.
- If `excitatory_flow_rate=True`, force effective chunking to 1, even if the
  config value for `l2_charge_chunks` is larger.
- Flow-rate mode is the finer temporal representation, so do not combine it with
  artificial weight chunking.
- If `excitatory_flow_rate=False`, preserve the chunking ablation:
  - `l2_charge_chunks=1` means current instantaneous behavior.
  - `l2_charge_chunks>1` splits instantaneous L2 feedforward charge into K
    pieces.
- Make this interaction explicit in config/docs/UI labels if applicable, so
  users understand that chunking is ignored or coerced when flow-rate mode is
  enabled.

Effective config logic:

```text
if excitatory_flow_rate:
  effective_l2_charge_chunks = 1
else:
  effective_l2_charge_chunks = l2_charge_chunks
```

## Constraints

- Keep this local to neurons where possible.
- Do not CUDA/GPU-optimize this.
- Do not implement distance weighting.
- Do not add membrane noise.
- Keep weight initialization uniform/random.
- Preserve existing dashboard behavior: one rendered state per outer timestep,
  no per-trace subframes.
- Add optional diagnostics only if useful, e.g. current `exc_trace` per neuron.

## Testing / Verification

1. Add focused unit tests for the closed-form lazy integration:
   - Compare dense step-by-step trace integration against lazy closed-form
     integration for several `dt` values.
   - Verify normalized injection delivers approximately the original weight over
     a long window.
2. Add a behavior-preservation test:
   - With `excitatory_flow_rate=False`, current tests / baseline behavior should
     remain unchanged.
3. Add a small integration/harness check:
   - With `excitatory_flow_rate=True`, charge should build smoothly over
     multiple timesteps from an input volley in neurons that receive positive
     synaptic input.
   - A neuron with residual trace should be able to cross threshold on a later
     timestep even if no new spike arrives on that timestep.
   - Inactive neurons should not require per-timestep updates; their trace should
     be advanced only when touched or read for state export.
4. Add a chunking interaction check:
   - With `excitatory_flow_rate=True`, effective chunks should be 1 even if
     `l2_charge_chunks > 1`.
   - With `excitatory_flow_rate=False`, chunking should remain toggleable.
5. Update methodology docs to clearly distinguish:
   - instantaneous charge packet mode
   - sparse flow-rate trace mode
   - weight as current/gate amplitude rather than directly deposited charge
   - the unchanged event-based threshold/winner/inhibition procedure
   - L1E's current role as an abstract pretrained sensory source

6. Update the equations/methodology sheet, especially
   `Current_Implementation_Methodology_Equations.md`, with a dedicated section
   for the flow-rate math. This is more mathematically involved than most of the
   current implementation notes and should not be buried in prose. Include:
   - the instantaneous baseline equation
   - the current-trace state variables
   - the recursive dense update
   - the closed-form lazy skipped-time update
   - the normalized injection rule and why it preserves total delivered charge
   - the ordering relative to threshold checks and winner arbitration
   - the chunking interaction: flow-rate mode forces effective
     `l2_charge_chunks = 1`

Please first verify how `receive_input`, threshold checking, refractory,
leak/update, L2 chunking, L1E firing cadence, state serialization, and dashboard
config currently interact. Then implement the smallest scoped toggleable version
of sparse excitatory flow-rate accumulation.
