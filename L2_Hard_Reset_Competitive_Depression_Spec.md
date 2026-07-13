# L2 Hard-Reset Competitive Depression

Technical implementation specification for replacing learned L2I-to-L2E
inhibitory gates with an unweighted hard-reset event and local depression of the
positive L1E-to-L2E weights that contributed to a losing response.

## 1. Decision

The active L2 competition architecture must no longer contain one learned
negative `L2I -> L2E` weight per L2E neuron.

When L2I fires:

1. The winning L2E is excluded because its own `fire()` already consumes its
   membrane state and runs winner learning.
2. Every non-winning L2E receives an unweighted inhibitory event.
3. That event always hard-resets the target's membrane and pending current
   traces.
4. Before reset, the target's charge is captured as a local competition signal.
5. Only positive feedforward weights whose L1E sources participated in the
   current response are depressed.

The L2I event therefore has two distinct effects:

- transient: remove all losing charge;
- structural: weaken the positive gates that produced a competing losing
  response.

There is no learned inhibitory magnitude in this path.

## 2. Goals

- Preserve full hard-reset inhibition for every L2E loser.
- Remove all active-model L2I-to-L2E negative weights and their plasticity.
- Use the same bounded signed-weight update family for winner and loser learning.
- Depress only participating positive feedforward weights on a loss.
- Scale depression by the losing neuron's own pre-reset charge.
- Keep the rule local, task-independent, deterministic, and seed-reproducible.
- Keep the positive trainable `L2E -> L2I` weights. They still determine when
  L2I has accumulated enough evidence to issue the reset event.
- Preserve the generic negative-synapse implementation for L1 feedback and
  isolated legacy experiments; only the active L2 competition path loses its
  learned negative gates.

## 3. Non-Goals

- Do not change L1E signed input encoding.
- Do not change L1I accumulation, learning, or one-step delayed feedback.
- Do not add pattern-specific initialization, center-pixel handling, labels, or
  global error signals.
- Do not make L2I fire procedurally whenever L2E fires. L2I must still cross its
  own learned threshold from its positive L2E inputs.
- Do not add a replacement inhibitory magnitude, scalar field, or target-specific
  reset strength. The reset event is binary and complete.
- Do not delete `Neuron.apply_inhibition()` globally. L1I-to-L1E and legacy unit
  tests still use ordinary negative synapses.

## 4. Target Topology

### L2E afferents

Each active-engine L2E must have exactly `N_PIX` afferents:

```text
[from_L1E0, from_L1E1, ..., from_L1E8]
```

There must be no index-0 local-I placeholder and no negative element in an L2E
weight array.

### L2I afferents

L2I retains its existing positive trainable afferents:

```text
[from_L2E0, from_L2E1, ..., from_L2E7]
```

### L2I efferents

L2I has an unweighted structural fanout to every L2E. This fanout conveys the
binary event `competitive_reset`; it is not a synaptic weight bank and must not
appear in weight snapshots, weight-change messages, weight statistics, or the
weights-over-time graph.

## 5. Event Ordering

Keep the current winner and L2I recruitment order:

```text
L1E input charges L2E
eligible L2E winner is selected
winner.fire() runs winner learning and resets the winner
winner spike charges L2I
if L2I crosses threshold:
    L2I.fire()
    for every non-winner L2E:
        capture V_pre and participating input mask
        apply competitive depression
        hard-reset membrane and pending current traces
        emit structural reset event
```

Requirements:

- The pre-reset charge used for learning is the charge after the current input
  delivery and before any reset.
- All non-winners reset when L2I fires, even when their charge is far below
  threshold. Depression may be zero, but reset is unconditional.
- If L2I does not fire, neither reset nor competitive depression occurs.
- A competitive reset must not be rejected merely because the target has a
  refractory timer. It must still guarantee zero membrane/current state.
- Do not modify the target's refractory timer during the reset.
- No later operation in the same outer step may restore charge to a reset L2E.

## 6. Shared Bounded Weight Kernel

Refactor the L2E signed-spike update into a reusable, direction-aware bounded
kernel. For a positive weight `w` with lower and upper bounds `w_min`, `w_cap`:

```text
q = clamp((w - w_min) / (w_cap - w_min), 0, 1)
```

For upward movement:

```text
H_up(q) = 1 - q^2
```

For downward movement, use the reflected function:

```text
H_down(q) = 1 - (1 - q)^2
```

Then:

```text
delta_w = +gain * H_up(q)     for direction +1
delta_w = -gain * H_down(q)   for direction -1
w_next  = clip(w + delta_w, w_min, w_cap)
```

This reflection is required. A literal negative copy of
`1 - (w / w_cap)^2` becomes zero at `w_cap`, which would make a capped losing
weight impossible to depress. The downward branch must instead become zero at
`w_min` and remain fully effective at `w_cap`.

Handle `w_cap <= w_min` as a no-op to avoid division by zero.

### Winner use

The existing signed-spike winner rule must use the shared kernel:

- participating/ON afferent: direction `+1`;
- non-participating/OFF afferent: direction `-1`;
- retain the winner rule's existing effective gain selection, including the
  structural free-energy gate when enabled;
- retain no-budget behavior and final clipping.

This makes the existing OFF-pixel update and the new losing-ON update use the
same mathematically valid downward branch.

## 7. Competitive Depression Rule

For losing L2E neuron `j`, capture:

```text
V_pre_j = membrane charge immediately before reset
p_loss_j = clamp(V_pre_j / theta_j, 0, 1)
```

The participating mask is the target neuron's most recently delivered L1E spike
vector:

```text
participating_ji = last_input_spikes_ji > 0.5
```

Only afferents satisfying both conditions are eligible:

```text
weight_ji > 0 and participating_ji
```

Do not apply a sign-reversed update to OFF inputs. That would potentiate absent
pixels. OFF weights remain unchanged during a competitive-loss event.

For each eligible weight, use direction `-1` in the shared bounded kernel with:

```text
structural_gate_j = neuron._structural_free_energy_gate()
                    if structural_free_energy is enabled
                    else 1

gain_loss_j = neuron.learning_rate * structural_gate_j * p_loss_j
```

Properties:

- zero-charge loser: no weight change, but still reset;
- weakly charged loser: small depression;
- near-threshold loser: strong depression;
- mature neuron: retains the existing task-independent structural plasticity
  brake rather than being erased rapidly by every other pattern;
- weight at `w_cap`: can move downward;
- weight at `w_min`: cannot move below the floor.

Do not use confidence, a positive-weight budget, winner voltage, population
averages, pattern identity, or target-specific inhibitory state in this rule.

## 8. Configuration

Use the existing `loser_depression` boolean as the ablation switch rather than
creating a second competing mechanism. Change its documented meaning to
"competitive depression on an L2I hard-reset event."

Canonical active configuration:

```text
l2i_hard_reset_losers = True
hard_reset_clear_traces = True
loser_depression = True
```

The depression rate comes from the L2E neuron's existing `learning_rate`; do not
use `eta_loss` in the canonical rule. Remove `eta_loss` from the dashboard. It may
remain temporarily accepted as a deprecated experiment parameter if old harnesses
still construct the engine with it, but it must not affect the new active path.

Remove or permanently archive all L2 gate controls from the active dashboard and
methodology:

- `l2_gate_eta`
- `l2_gate_eq_frac`
- `inhibitory_delta_rule`
- `inhibitory_rule_mode`
- `inhibitory_eta_up`
- `inhibitory_eta_down`
- `inhibitory_p_max`
- `inhibitory_margin_frac`
- `inhibitory_delta_eta`
- `inhibitory_flow_rate` and its trace controls for the L2 reset path

Backward-compatible constructor acceptance is allowed only to avoid breaking old
standalone experiments. These values must not create or train L2I-to-L2E gates in
`SimulationEngine`.

## 9. Code Changes

### `snn/rules/excitatory.py`

- Add a shared vectorized bounded signed-weight helper implementing Section 6.
- Change `SignedSpikeRule` to use it for both `+1` and `-1` directions.
- Expose the helper for competitive depression without duplicating its equation.

### `neuron_flexible.py`

- Add a dedicated method such as `apply_competitive_reset()` for an unweighted
  L2I event.
- The method must capture `V_pre`, compute/apply depression when enabled, reset
  `potential` to `resting_potential`, and clear `exc_trace`/`inh_trace` when
  configured.
- It must not call `apply_inhibition()` and must not require a negative synapse.
- Return a diagnostic record containing at least:

```text
v_pre
v_post
theta
p_loss
depressed_indices
weights_before
delta_weights
weights_after
```

- Retain `apply_inhibition()` for ordinary weighted inhibitory connections.
- Remove the old `_depress_losers()` hook from `apply_inhibition()` for the active
  L2 path. Do not allow both old and new loser depression to run for one event.

### `cortical_column_flexible.py`

- Preserve legacy behavior by default for standalone users/tests.
- Add an explicit connectivity option such as
  `include_local_inhibition: bool = True`.
- Track a `feedforward_offset` (`1` for legacy weighted local inhibition, `0` for
  the active hard-reset architecture).
- Make `set_feedforward_weights()` and `feedforward_weights()` use that offset.
- The active `SimulationEngine` must build L2 with
  `include_local_inhibition=False`.
- Calling `set_local_inhibition_weights()` when the option is false must raise a
  clear error rather than silently writing pixel weight zero.

### `backend/simulation.py`

- Change active `L2E_FANIN` from `1 + N_PIX` to `N_PIX`.
- Build L2E without the local-I placeholder.
- Delete the call to `set_local_inhibition_weights()` in `_build()`.
- Remove L2E inhibitory-gate cap/rule configuration.
- Update every active L2E index from `1 + i` to `i`, including:
  - feedforward delivery vectors;
  - distance arrays;
  - manual receptive-field weight editing;
  - confidence serialization;
  - feedforward snapshots and metrics.
- In `_resolve_l2_competition()`, replace the synthetic inhibitory spike vector
  and `apply_inhibition()` call with the new direct competitive-reset method.
- Continue applying the event to every non-winner when L2I fires.
- Replace `_check_l2_inhibition_phases()` with reset-phase diagnostics. Verify
  `v_post == resting_potential` and that end-of-step charge does not rise again.
- Remove `inh->{j}` from `_all_weights()` and changed-weight tracking.
- Emit structural reset edge IDs for visualization, for example
  `reset->{j}`.
- Keep `L2E{j} -> L2I` positive weights and their assembly-credit training.

### `backend/api.py`

- Enable canonical hard reset and competitive depression by default.
- Rename the visible `loser_depression` label/description to
  `Competitive depression` and describe the exact L2I-event semantics.
- Remove `eta_loss` and all learned L2 gate controls from the served config.
- Update comments that describe learned L2I-to-L2E gate magnitudes.

### Frontend

- Keep a visible L2I-to-L2E connection because the structural inhibitory fanout
  still exists, but serialize it as `kind="reset_inhibition"`, `weight=null`, and
  an ID such as `reset->{j}`.
- Render reset-inhibition edges with fixed inhibitory styling and opacity,
  independent of weight and the weak-weight filter.
- Pulse those edges when their IDs appear in `dynamic.emitted`.
- Update `frontend/charge.js` to mark inhibition from `reset->{j}` events.
- Remove the red learned-inhibitory-gate series from `frontend/weights.js`; that
  chart should contain only the nine positive feedforward weights.
- Remove inspector text that implies L2 reset strength is learned.

### Documentation

Update at least:

- `README.md`
- `Current_Implementation_Methodology_Equations.md`
- `AGENT_HANDOFF.md`
- `Inhibition_And_Consolidation_State.md`

Delete statements that L2I suppresses L2E through learned negative gates. State
that L2I recruitment is learned on its positive inputs, while its output is an
unweighted reset plus local competitive depression.

## 10. Tests

Add focused unit tests before regenerating golden data.

### Bounded kernel

- Upward direction is largest at `w_min` and zero at `w_cap`.
- Downward direction is zero at `w_min` and nonzero/maximal at `w_cap`.
- Both directions remain within `[w_min, w_cap]`.
- A capped weight can be depressed.
- A floored weight cannot be depressed further.
- Degenerate `w_cap <= w_min` is safe.

### Competitive reset

- A loser with positive charge ends at exact rest after an L2I reset event.
- Pending excitatory and inhibitory current traces are zeroed.
- Reset works without any negative afferent weight.
- Reset still occurs when `loser_depression=False`.
- No reset or depression occurs when L2I does not fire.
- Winner is never passed through the losing reset path.

### Competitive depression

- Participating positive weights decrease.
- Non-participating positive weights do not change.
- No negative or absent weight is created.
- A loser at zero charge resets but does not learn.
- A higher-charge loser receives a larger update than a lower-charge loser with
  otherwise identical state.
- Structural maturity scales the loss update exactly once.
- One inhibitory event cannot trigger both legacy and competitive depression.

### Topology and serialization

- Every active L2E weight/confidence/distance array has length `N_PIX`.
- Active L2E arrays contain no negative inhibitory gate.
- Topology contains no learned `inh->{j}` weight IDs.
- Weight snapshots contain exactly the nine feedforward weights per L2E.
- Structural reset edges remain visible and pulse on reset.
- Manual receptive-field editing still targets the requested pixel.

### Integration

- On an L2I firing step, every non-winner has exact zero end-of-step charge.
- Only participating weights of charged losers move downward.
- L2E-to-L2I weights still train and L2I still recruits through threshold
  accumulation.
- Four-pattern cycling runs without exceptions or shape/index drift.
- Existing L1I delayed-feedback, leak-toggle, and constant-input tests remain
  unchanged and pass.
- Regenerate the golden baseline only after the focused behavior tests pass.

## 11. Required Experimental Report

Run the current dashboard configuration over multiple fixed seeds and report,
without changing the algorithm between seeds:

- L2E neurons that fired at least once;
- dead L2E count;
- distinct sustained winners across the four patterns;
- sustained dominance per pattern;
- center-pixel weight distribution;
- number and mean magnitude of competitive-depression events;
- comparison against the pre-change hard-reset baseline.

This report is diagnostic, not permission to introduce task-specific weights or
pattern-aware exceptions. In particular, do not protect the center pixel by
index; any protection must emerge from the same task-independent plasticity rule.

## 12. Acceptance Criteria

The implementation is complete when all of the following are true:

1. The active engine has no learned negative `L2I -> L2E` weights or placeholder
   entries.
2. L2I firing hard-resets every non-winner to exact rest and clears pending
   current charge.
3. Participating positive weights of charged losers are depressed through the
   shared direction-aware bounded kernel.
4. OFF weights are not potentiated by the competitive-loss event.
5. Winner learning remains the signed-spike rule and uses the same kernel.
6. L2I recruitment remains learned through positive `L2E -> L2I` weights.
7. Backend topology, diagnostics, and frontend views contain no fictional learned
   reset magnitude.
8. Focused tests, impacted regression tests, and regenerated golden tests pass.
9. The multi-seed experimental report is recorded with failures and regressions
   reported honestly.

