# Inhibitory Gate Redistribution and Recruitment

Technical implementation specification for adding conservative redistribution of
a losing neuron's active feedforward gate capacity into its inactive gates as an
A/B-selectable alternative to the current charge-scaled one-sided competitive
depression rule.

This is the next architectural change. It must be implemented against the current
unweighted L2I hard-reset model; learned negative `L2I -> L2E` gates must not return.

## 1. Intended Dynamics

When an L2E neuron fires, it performs the existing signed winner update and enters
refractory state. Its spike may cause L2I to fire.

When L2I fires, it broadcasts the same unweighted competitive-reset event to every
L2E neuron:

- an L2E currently in refractory is the current winner: hard reset remains safe
  and idempotent, but its feedforward weights are not redistributed;
- a non-refractory L2E is a loser: estimate how strongly its active gates match
  the current input, move an incremental amount out of those active gates, and
  place exactly that amount into inactive gates with available headroom;
- every L2E membrane and pending current trace is hard-reset regardless of whether
  its weights changed.

No historical winner, epoch, one-hot winner exclusion, target weight, stored
initialization, or learned inhibitory magnitude is used.

The intended long-run behavior is recruitment:

```text
current winner
    -> winner signed learning
    -> refractory protection from redistribution

charged structural competitors
    -> current-pattern active gates shrink
    -> inactive gates grow by the same total amount
    -> displaced neurons become candidates for later patterns
```

If a recruited neuron later wins, that takeover is legitimate. The model protects
the neuron that fires now, not the neuron that historically owned the pattern.

## 2. Non-Goals and Prohibited Mechanisms

Do not introduce:

- competition epochs or a latched winner;
- use of the L2E one-hot spike vector to decide plasticity eligibility;
- target weights, target distributions, or recovery anchors;
- restoration toward sampled initial values;
- charge-history, peak-charge, or eligibility-window winner tracking;
- task labels, pattern lookup, or center-pixel special cases;
- learned `L2I -> L2E` weights;
- creation or destruction of total feedforward gate mass during redistribution;
- OFF growth independent of an actual active-gate decrease.

Keep L2I recruitment through the existing positive trainable `L2E -> L2I`
weights. Keep ordinary signed winner learning unchanged. Retain the current
one-sided competitive-depression rule as the baseline A/B mode; the baseline and
redistribution modes are mutually exclusive for each reset event.

## 3. Winner Protection Through Refractory State

Set the canonical dashboard refractory period to exactly:

```text
refractory = 1
```

The current phase ordering makes this a same-step local plasticity veto:

```text
winner.fire()
    -> winner learning runs
    -> winner membrane resets
    -> refractory_timer = 1

winner spike drives L2I
L2I fires and broadcasts competitive reset
    -> winner sees refractory_timer > 0 and skips redistribution
    -> losers see refractory_timer == 0 and redistribute

end-of-step update()
    -> winner refractory_timer decrements from 1 to 0
```

Thus the winner is available to fire again on the next outer step. Do not add a
second winner-protection signal. Use this same reset fanout and refractory-based
winner protection in both the redistribution and one-sided-depression modes so
the A/B comparison changes only the loser weight-update rule.

Required implementation changes:

- Remove `if j == winner: continue` from the L2I reset fanout.
- Deliver the competitive-reset call to every L2E.
- Inside `Neuron.apply_competitive_reset()`, determine weight-plasticity
  eligibility only from `refractory_timer == 0` at event arrival.
- Always perform the hard reset, including for a refractory winner.
- Do not change the refractory timer during competitive reset.
- Do not use `spiked`, the one-hot L2E vector, or an engine winner ID as a fallback.
- Apply the selected loser weight-update mode only when `refractory_timer == 0`;
  the hard reset remains unconditional in every mode.

With refractory `1`, only the neuron that fired in the current outer step is
protected. Longer refractory settings are an explicit ablation and will protect
older spikers as well; document that consequence.

## 4. Structural Competition Signal

The redistribution signal is derived from the active feedforward gates, not the
target's instantaneous membrane charge.

For losing L2E neuron `j`, let `A_j` be its positive feedforward synapses whose
current L1E inputs participated, and let `O_j` be the remaining positive
feedforward synapses.

The active delivered support is:

```text
c_ji = w_ji * delivery_factor_ji
S_on_j = sum(c_ji for i in A_j)
p_match_j = clamp(S_on_j / theta_j, 0, 1)
```

Use the same fixed delivery factor used by charge integration, including distance
weighting when enabled. Do not use membrane `V`, previous charge, winner charge,
or any population statistic.

Properties:

- a neuron with weak active gates receives little redistribution;
- a strong structural competitor receives a large update even if it fired earlier
  and its membrane has already reset;
- the signal is local to the target neuron and current presynaptic input;
- geometric delivery and learning agree about which gates made the neuron a
  competitor.

Do not multiply this rule by the existing structural free-energy maturity gate.
`p_match` is already the structural competition signal; applying the maturity
brake would protect the strongest mature competitors from redistribution.

This section applies only to the `redistribution` A/B mode. The retained
`depression` baseline must continue using its current membrane-charge signal:

```text
p_loss = clamp(V_pre / theta, 0, 1)
```

and must depress only participating positive gates through the existing reflected
downward kernel. It must not grow OFF gates or conserve the removed mass. Keeping
that baseline equation intact makes the experiment a comparison of the proposed
redistribution rule against the currently implemented loser-depression rule.

## 5. Candidate Active-Gate Decrease

Reuse the existing direction-aware bounded kernel from
`snn/rules/excitatory.py`. For each active gate:

```text
q_i = clamp((w_i - w_min) / (w_cap - w_min), 0, 1)
H_down(q_i) = 1 - (1 - q_i)^2

candidate_decrease_i = learning_rate * p_match * H_down(q_i)
```

Clip each candidate so it cannot move the gate below `w_min`:

```text
d_i = min(candidate_decrease_i, w_i - w_min)
R_candidate = sum(d_i)
```

This update is incremental. An inhibitory event transfers only the bounded
learning step, not the full active gate mass. Active gates therefore retain most
of their capacity on each event and decline gradually across repeated losses.

If `p_match == 0`, `A_j` is empty, or `R_candidate == 0`, no weights change.

## 6. Conservative OFF-Gate Recruitment

The OFF bank can accept at most:

```text
capacity_k = w_cap - w_k
C_off = sum(capacity_k for k in O_j)
```

The actual transfer is:

```text
T = min(R_candidate, C_off)
```

If `T < R_candidate`, scale every active decrease by the same factor
`T / R_candidate`. This leaves untransferable resource in the active gates rather
than destroying it.

Apply the resulting active decreases, then distribute exactly `T` across OFF
weights. Allocation must be deterministic and favor gates with greater bounded
upward headroom:

```text
q_k = clamp((w_k - w_min) / (w_cap - w_min), 0, 1)
H_up(q_k) = 1 - q_k^2
```

Allocate proportionally to `H_up(q_k)` while respecting each gate's remaining
capacity. If clipping one gate leaves a remainder, redistribute that remainder
among the other OFF gates using the same rule until either:

- the full `T` has been placed; or
- all OFF gates are at `w_cap`.

This is a small deterministic capped-allocation/water-filling operation over the
target neuron's own OFF gates. It is the only afferent-coupled operation in this
rule and exists solely to conserve the transferred resource.

Required invariant, within floating-point tolerance:

```text
sum(active weights before - active weights after)
==
sum(OFF weights after - OFF weights before)
```

Total positive L2E feedforward weight must be unchanged by a competitive
redistribution event.

OFF weights are allowed to reach `w_cap`. Saturation is accepted as recruitment,
not treated as an error or pulled toward a target. As OFF capacity fills, less
active resource can be transferred and the rule naturally stops.

## 7. Shared-Pixel and Cold-Start Consequences

No code may identify the center pixel, but the four-pattern statistics naturally
produce:

- the common pixel is active during every loss and tends to donate gate capacity;
- uncommon pixels are often OFF and receive that capacity;
- a current winner retains/potentiates the common pixel because refractory
  protects it from inhibitory redistribution;
- displaced neurons retain total synaptic capacity while reallocating it toward
  inputs that were absent from the pattern they lost;
- on a later pattern, those recruited uncommon gates reduce cold-start latency.

The two-timescale outcome is intentional:

```text
winner learning specializes the current owner
loser redistribution reallocates the remaining pool
```

## 8. `apply_competitive_reset()` Contract

Conceptual pseudocode:

```python
def apply_competitive_reset(current_input, competitive_weight_update):
    refractory_at_arrival = refractory_timer > 0

    if not refractory_at_arrival:
        if competitive_weight_update == "redistribution":
            active, off = feedforward_masks(current_input)
            p_match = active_effective_weight_sum(active) / threshold
            candidate_down = bounded_active_decreases(active, p_match)
            transfer = min(sum(candidate_down), off_capacity(off))
            scale_and_apply_active_decreases(candidate_down, transfer)
            allocate_transfer_to_off_gates(off, transfer)
        elif competitive_weight_update == "depression":
            apply_current_one_sided_depression(V_pre, current_input)
        # "none" is the hard-reset-only control

    potential = resting_potential
    clear pending excitatory/inhibitory current traces
    # refractory_timer is untouched
```

The input mask must be the same current accepted feedforward input used by the
L2E charge path. Do not infer active/OFF status from weight magnitude or pattern
names.

Return diagnostics containing at least:

```text
refractory_at_arrival
plasticity_applied
active_indices
off_indices
active_effective_sum
p_match
candidate_release
transferred
active_before / active_delta / active_after
off_before / off_delta / off_after
v_pre / v_post
```

Retain existing reset diagnostics where compatible. Record `p_loss` for the
`depression` mode and `p_match` for the `redistribution` mode; neither signal is
used by the other rule.

## 9. Configuration and Cleanup

Use one canonical mode flag:

```text
competitive_weight_update = "redistribution"
```

- Allowed values are `"redistribution"`, `"depression"`, and `"none"`.
- Default the dashboard to `"redistribution"` after the new rule is implemented.
- `"depression"` runs the currently implemented charge-scaled one-sided loser
  update, providing the direct A/B baseline.
- `"none"` still hard-resets the pool but performs no loser weight update.
- Expose the dashboard control as `Competitive weight update`, with all three
  choices, rather than as two independent booleans.
- Backward-compatible acceptance of `loser_depression` is allowed temporarily for
  old experiment files only when `competitive_weight_update` is absent: map `true`
  to `"depression"` and `false` to `"none"`.
- Never run depression and redistribution for the same event.
- Remove/hide `eta_loss`; the rule uses the L2E learning rate.
- Set the dashboard refractory default to `1` and explain its same-step
  winner-protection role.
- Do not add a target, recovery-rate, charge exponent, or redistribution-strength
  slider in the first implementation.

## 10. Code Areas

### `neuron_flexible.py`

- Retain the active one-sided competitive-depression calculation as the
  `"depression"` branch of `apply_competitive_reset()`.
- Add the refractory-gated conservative redistribution calculation as the
  mutually exclusive `"redistribution"` branch.
- Support `"none"` as hard-reset-only and reject unknown mode values.
- Keep unconditional hard reset and trace clearing.
- Keep generic weighted `apply_inhibition()` for L1 and legacy experiments.

### `backend/simulation.py`

- Broadcast competitive reset to every L2E when L2I fires.
- Remove explicit one-hot/current-winner exclusion from reset delivery.
- Supply the current L2E feedforward input mask needed by redistribution.
- Pass the resolved `competitive_weight_update` mode into every competitive-reset
  event.
- Ensure L2E refractory is `1` in the canonical engine.
- Update reset-phase diagnostics and statistics.

### `backend/api.py` and frontend

- Update the refractory and redistribution defaults/descriptions.
- Show structural reset events as before; do not add a learned inhibitory gate.
- Surface redistribution diagnostics without implying a target weight.
- Existing feedforward weight charts should show the resulting ON-to-OFF transfer.

### Documentation and experiments

Update the equation sheet, README, handoff, inhibition state, config examples,
experiment runners, reports, and affected harnesses. Describe redistribution as
the new default and current-charge competitive depression as the retained A/B
baseline once the new rule is implemented.

## 11. Required Tests

### Refractory winner protection

- The current winner enters refractory before L2I reset delivery.
- L2I reset is delivered to every L2E, including the winner.
- A refractory winner receives no redistribution.
- Its membrane/current traces still end at exact rest/zero.
- End-of-step update decrements refractory `1 -> 0`.
- The winner can receive input and compete on the next step.
- No one-hot winner vector or engine winner ID participates in the plasticity
  decision.

### Structural match

- Identical losers with larger active effective-weight sums have larger candidate
  transfers.
- A zero active sum causes no redistribution.
- Distance factors affect `p_match` exactly as they affect charge delivery.
- Live membrane charge and previous charge do not affect redistribution when
  weights/input/refractory are otherwise identical.

### Conservation and bounds

- Active weights decrease and OFF weights increase.
- The total active decrease equals total OFF increase.
- Total positive feedforward weight is invariant for the event.
- No active weight crosses `w_min`; no OFF weight crosses `w_cap`.
- When OFF capacity is insufficient, the active decrease is reduced to the amount
  that can be transferred.
- When no OFF capacity exists, no weights change.
- Capped allocation is deterministic and redistributes clipping remainder.
- Repeated losses move capacity incrementally rather than in one reset.

### Mode selection and A/B isolation

- `"redistribution"` decreases active gates and increases OFF gates by the same
  total amount.
- `"depression"` reproduces the current charge-scaled one-sided numerical update
  and leaves OFF gates unchanged.
- `"none"` changes no weights while preserving the identical hard reset.
- Exactly one mode is executed per competitive-reset event.
- All three modes use identical L2I firing, reset fanout, refractory winner
  protection, input sequence, initialization, and hard-reset behavior.
- Legacy `loser_depression` config maps to a mode only when the new mode field is
  absent.

### Recruitment behavior

- A strong non-refractory competitor reallocates more than a weak noncompetitor.
- A timing takeover protects the newly firing neuron and redistributes the
  displaced charged competitor.
- Repeated current-pattern wins increase separation from the displaced neuron.
- On a later input, the displaced neuron's recruited gates provide more drive than
  a hard-reset-only control.
- Synthetic common-active inputs lose weight in competitors while uncommon OFF
  inputs gain weight, without production code inspecting input indices.

### Regression

- L1I delayed feedback and half-frequency synchronization remain unchanged.
- L2I positive recruitment weights still train.
- Leak toggles, chunked WTA, distance delivery, manual receptive-field editing,
  topology serialization, and frontend charts remain functional.
- Regenerate golden data only after focused tests pass.

## 12. Experimental Report

Run the four-pattern dashboard configuration over multiple fixed seeds and compare
all three modes using paired seeds and otherwise identical configuration:

```text
competitive_weight_update = "redistribution"
competitive_weight_update = "depression"
competitive_weight_update = "none"
```

- sustained winner sequences and dominance;
- time for three-neuron timing loops to collapse;
- distinct winners across the four patterns;
- dead-neuron count;
- pattern-switch first-response latency;
- common versus uncommon weight distributions;
- per-event transferred mass and conservation error;
- fraction of OFF gates at cap;
- effect of refractory `1` versus the previous `0` baseline.

Report regressions honestly. Do not introduce task-aware exceptions in response.

## 13. Acceptance Criteria

Implementation is complete when:

1. Refractory `1`, not a one-hot winner check, is the sole redistribution veto.
2. Every L2E receives the L2I hard-reset event and ends with no residual charge.
3. Only non-refractory structural competitors redistribute weights.
4. Redistribution strength comes from the current active effective-gate sum, not
   membrane charge or history.
5. Active gate mass is moved incrementally into OFF gates with exact conservation.
6. No targets, anchors, learned inhibitory gates, historical winners, or
   pattern-specific logic exist.
7. Saturation is accepted as recruitment and naturally limits further transfer.
8. Focused tests, impacted regressions, and regenerated golden tests pass.
9. The multi-seed report records timing, recruitment, and cold-start outcomes.
10. Paired-seed A/B results directly compare redistribution with the retained
    one-sided-depression baseline, with hard-reset-only reported as a control.
