# Claude Kickoff Prompt: SwitchI Investigation

You are working in the SNN repository. Investigate why the `SwitchI` neurons in the `rg_residual` preset are not firing during normal dashboard simulations.

Do not implement multiplicative coincidence yet. First make the existing charged two-branch coincidence mechanism work and prove that it works. We need to separate a broken `SwitchI` implementation from an upstream network that never supplies the required temporal sequence.

## Relevant files

- `snn/neurons.py`
- `backend/simulation.py`
- `backend/network_spec.py`
- `backend/serializer.py`
- `frontend/`
- `tests/test_residual_topology.py`
- `README.md`

## Current intended mechanism

Each `SwitchI[j]` is paired with `L2E[j]`.

1. A real `L2E[j]` spike travels over the explicit `trace_excitation` edge and primes only `SwitchI[j]`.
2. This creates a decaying local eligibility/winner trace.
3. A later real `ErrorE[i]` spike broadcasts a residual event to all structurally connected switches.
4. Only the switch carrying the eligible trace should fire.
5. `SwitchI[j]` should schedule inhibition only onto its paired `L2E[j]`.
6. The current-boundary L2 winner must not supply the trace used by a same-boundary residual. The trace should become eligible only on a later boundary.
7. Neither branch may fire a switch alone.

## Dashboard observation

The connections sometimes appear to light up, but the `SwitchI` cells do not visibly accumulate the expected charge or fire during the normal `rg_residual` simulation.

Do not assume this is only a renderer problem. Audit the complete causal path:

```text
L2E spike
→ trace_excitation dispatch
→ paired switch trace
```

and:

```text
L1E spike
→ ErrorE excitation
→ actual ErrorE threshold crossing
→ residual broadcast
→ SwitchI residual charge
→ coincidence resolution
→ SwitchI spike
→ delayed paired inhibition of L2E
```

## Investigation procedure

First, understand the existing implementation and synchronous subphase order. Do not change timing semantics merely to make a test pass.

Then build or run a deterministic controlled test with no reliance on training.

### 1. Residual alone

- Stimulate one `ErrorE`.
- Verify that every connected switch receives visible residual charge.
- Verify that no switch fires.

### 2. Trace alone

- Force `L2E3` to win/fire.
- Verify that only `SwitchI3` receives a winner trace.
- Advance boundaries without residual input.
- Verify that no switch fires.

### 3. Temporal coincidence

- Make `L2E3` fire first.
- On a later boundary, while its trace remains eligible, make one `ErrorE` fire.
- Verify that exactly `SwitchI3` fires.
- Verify that the other seven switches remain silent.
- On the following boundary, verify that inhibition from `SwitchI3` lands only on `L2E3`.

### 4. Same-boundary rejection

- Cause a new `L2E3` winner and an ErrorE spike on the same boundary without a pre-existing trace.
- Verify that `SwitchI3` does not fire.
- Verify that the new L2 spike primes the trace only for a future boundary.

Record these fields for every relevant boundary:

- `winner_trace`
- `residual_received`
- `residual_events`
- `residual_charge`
- `trace_charge`
- `potential`
- `v_pre_reset`
- `spiked`
- emitted trace/residual/inhibitory edge IDs
- scheduled and delivered inhibitory pulses

After the isolated mechanism is proven, run the normal `rg_residual` pathway and identify whether it naturally produces:

```text
earlier L2E[j] spike → surviving paired trace → later ErrorE spike
```

Determine exactly where this sequence stops. In particular, check:

- Whether ErrorE connections light because a signal was emitted even though ErrorE never crosses threshold.
- Whether predictive conductance suppresses every later ErrorE event.
- Whether the residual arrives after the winner trace has fallen below its threshold.
- Whether `input_period`, synaptic delay, trace decay, and trace threshold are mutually compatible.
- Whether repeated L2 winners continually refresh the expected paired trace.
- Whether the dashboard is serializing and rendering the actual switch state rather than inferring activity from edge animation.
- Whether reset, preset switching, or parameter updates leave stale or mismatched switch objects.

## Fixing rules

If the isolated temporal-coincidence test fails, fix the smallest root cause in event dispatch, switch state transitions, timing, serialization, or rendering.

If the isolated test passes but natural activity never supplies both operands, fix or tune the smallest scientifically justified cause in the natural pathway. Do not bypass the residual mechanism, read a global winner variable, introduce direct special-case firing, or weaken the strict two-branch requirement.

Preserve these invariants:

- No switch may fire from trace alone.
- No switch may fire from residual alone.
- Only a locally traced switch may respond to a residual broadcast.
- Switch output remains paired to one L2E incumbent.
- A new same-boundary winner cannot inhibit itself.
- RG and the main L1 evidence path remain uninhibited.
- WTA still emits at most one L2 winner.
- Existing `pi`, `old`, and `rg` behavior must not regress.

Do not implement multiplicative coincidence during this task. We will replace the coincidence equation only after the existing pathway is visibly and causally working.

## Verification

Add focused regression tests for the discovered failure, not just implementation-shaped tests. Run:

- the focused SwitchI/residual tests;
- the complete test suite;
- golden topology checks;
- JavaScript syntax checks if frontend code changes;
- a deterministic normal-path simulation showing at least one legitimate SwitchI firing.

If browser automation is unavailable, verify the API/WebSocket state directly and clearly identify the remaining browser-only risk.

## Final report

Report:

1. The exact first broken link in the causal chain.
2. Why connections could appear active while the switch stayed silent.
3. The code or parameter changes made.
4. A boundary-by-boundary trace from L2 priming through residual coincidence and paired inhibition.
5. Evidence that both individual branches remain subthreshold.
6. Evidence that the natural `rg_residual` simulation now produces legitimate SwitchI activity.
7. All tests and verification results.
8. Any remaining scientific concern separately from implementation defects.

Do not stop after confirming that the isolated unit test passes. The task is complete only when you have explained the dashboard observation and demonstrated whether the full natural pathway can trigger the switch.
