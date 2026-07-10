# Claude Prompt: Structural Free Energy Plasticity Gate

You are working in `/home/adasgup/Documents/SNN` on branch
`feature/inhibitory-plasticity`.

The current worktree may contain local edits or untracked prompt files. Do not
revert unrelated work. Make narrow, reviewable changes only.

## Context

The current signed-spike L2E feedforward rule still uses a voltage-derived
closeness term:

```text
p = clamp(theta / v_pre, 0, 1)
```

or, in related paths, a voltage/threshold ratio. That makes the learning gate
depend on the current input event, membrane carryover, flow-rate timing,
inhibition, and overshoot. This appears to be part of the pattern-switch problem:
when a neuron that already specialized for one pattern fires during a later
pattern, the local signed rule can reinterpret the new event as evidence and
reshape/depress weights that should have remained consolidated.

We want to test a more structural free-energy signal:

```text
FE_struct = theta - sum_positive_excitatory_afferent_weights
```

The intent is: once an excitatory neuron has enough learned excitatory support to
explain a threshold crossing, it should become less plastic. In words:

```text
my weights are minimizing free energy; I should stop learning so aggressively
```

This is an experiment. Keep it behind a flag and measure it against the current
default.

## Important Correction

Do not use `sum(weights) / theta` directly as the plasticity gain unless you are
explicitly testing that as a diagnostic variant.

If:

```text
maturity = clamp(sum_positive_excitatory_afferent_weights / theta, 0, 1)
```

then the plasticity gate should decrease as maturity rises:

```text
structural_fe_gate = clamp(1 - maturity, eta_floor, 1)
```

or equivalently:

```text
structural_fe_gate = clamp((theta - sum_positive_excitatory_afferent_weights) / theta,
                           eta_floor,
                           1)
```

This makes under-built excitatory neurons plastic and mature/consolidated
excitatory neurons stable.

## Scope: Excitatory Neurons Only

Apply this structural free-energy plasticity gate only to excitatory postsynaptic
neurons.

For the current architecture, the first target is:

```text
L2E feedforward plasticity on positive L1E -> L2E synapses
```

Do not apply this rule to:

```text
L2I or L1I plasticity, even though their incoming E->I afferents are positive
negative inhibitory gates such as L2I -> L2E
L1E sensory source behavior
inhibitory-gate learning / apply_inhibition
assembly_flow_credit on inhibitory neurons
```

The deciding property is the postsynaptic neuron type. This experiment is for
excitatory neurons only.

## Proposed Config

Add a default-off/default-config flag such as:

```text
structural_free_energy = False
structural_fe_eta_floor = 0.02
structural_fe_sum_scope = "positive_afferents"
```

Naming can vary if there is already a better local convention, but keep the
meaning clear.

When enabled for L2E signed-spike learning, compute:

```text
sum_pos = sum(w_i for positive feedforward afferents into this L2E)
maturity = clamp(sum_pos / theta_l2e, 0, 1)
gate = max(structural_fe_eta_floor, 1 - maturity)
```

Then use:

```text
eta_eff = eta_base * gate
dw_i = eta_eff * signed_signal_i * saturating_term_i
```

where the signed signal remains:

```text
+1 for active inputs
-1 for inactive inputs
```

and the local cap/floor behavior remains unchanged.

## Sum Scope

The primary experiment should use all positive feedforward afferents into the
excitatory neuron, not only currently active inputs:

```text
sum_pos = sum(all positive L1E -> L2E feedforward weights)
```

Reason: if the sum only includes active afferents, the signal remains tied to the
current input pattern. The point of this experiment is to make the consolidation
brake structural rather than input-state dependent.

However, report the broad-RF risk explicitly. A neuron with many medium weights
can have a high total sum while being a poor specialist. Do not hide that failure
mode. If the primary experiment fails for that reason, add a clearly separate
diagnostic variant such as:

```text
top_k_sum, k=3
or
active_sum only
```

but keep the first pass focused on the all-positive structural sum.

## Expected Effect

Hypothesis:

- Early under-trained L2E neurons learn normally.
- A neuron that has already accumulated enough excitatory support slows down.
- On pattern switch, a previously consolidated specialist should resist being
  reshaped by signed inactive depression.
- Interleaved training should retain prior pattern owners better.

Non-goals:

- This is not a global assignment rule.
- This is not a label or pattern-ID memory.
- This should not look up rival neurons.
- This should not directly decide which neuron owns which pattern.

## Measurements

Do not trust the old short-window `metrics_consolidation.py` dominance result as
the ownership metric.

Measure at least:

```text
1. Single held pattern consolidation
2. Pattern-switch retention
3. Interleaved all-8 training
4. Sustained-presentation dominance
5. distinct modal winners across 8 patterns
6. dead/ownerless neurons
7. receptive-field sharpness or active-pixel match
```

Use at least 3 seeds for comparisons. Report means and seed spread.

Specific comparisons:

```text
baseline current dashboard/default config
structural_free_energy=True, eta_floor in [0.0, 0.01, 0.02, 0.05]
optionally structural_free_energy=True with loser_depression reduced below eta_loss=10
```

The last comparison matters because strong loser depression fixes one held
pattern but over-depresses interleaved training. If structural FE protects mature
specialists, it may allow a lower loser-depression rate.

## Success Criteria

This is promising only if it improves retention without creating broad,
multi-pattern owners.

Minimum useful outcome:

```text
more distinct winners than the current interleaved failure mode
fewer ownerless patterns after pattern switches
no collapse to one/few dominant neurons
no worse sustained-presentation dominance than baseline by a large margin
```

Strong outcome:

```text
8/8 distinct modal owners
improved sustained dominance
same owner reappears after pattern A -> pattern B -> pattern A
receptive fields remain sharp rather than broad
```

## Implementation Notes

Keep this local:

- Use only the target L2E neuron's own threshold and incoming feedforward weights.
- Do not inspect labels, pattern IDs, modal-owner tables, or rival neurons.
- Do not use membrane voltage in the structural FE gate.
- Do not apply the structural FE gate to inhibitory neurons.
- Keep existing behavior exactly reproducible with the flag off.

The existing signed-spike rule returns early from `_update_weights`, bypassing
the old budget/confidence/loser-depression feedforward stack. Make the structural
FE gate live inside that signed-spike path first.

If adding helper methods, prefer names that expose the model:

```text
_positive_afferent_weight_sum(...)
_structural_free_energy_gate(...)
```

Document the equations in `Current_Implementation_Methodology_Equations.md` and
update `AGENT_HANDOFF.md` only with tested results, not speculation.

## Dashboard Config Cleanup

The frontend Model Config panel is becoming cluttered. As part of this work,
clean up the dashboard config surface so the active experiment is readable.

Do not delete engine support for old experiments unless the code is truly dead
and tests/docs agree. Instead, remove or archive failed/inert knobs from the main
dashboard config.

Recommended main dashboard controls:

```text
signed_spike_learning
structural_free_energy
structural_fe_eta_floor
loser_depression
eta_loss
assembly_flow_credit
excitatory_flow_rate
exc_trace_decay
event_driven
refractory
l2e_lr_frac
leak_l2
distance_weighting only if functional distances are actually assigned
```

Recommended archive/advanced controls, hidden from the default panel:

```text
signed_depression
eta_off
l2e_budget
confidence_consolidation
l2_charge_chunks
inhibitory_flow_rate
inh_trace_decay
inh_trace_normalized
exc_trace_normalized
subtractive_reset
v_sat_frac
distance_power
distance_ref
distance_min
inhibitory_eta_up
inhibitory_eta_down
inhibitory_p_max
l1i_immediate_relay
```

Rationale:

- `signed_depression` / `eta_off` are superseded by signed-spike learning in the
  current default path.
- `l2e_budget`, `confidence_consolidation`, and legacy signed depression are inert
  under signed-spike learning or belong to the old budget regime.
- `l2_charge_chunks` is ignored when excitatory flow-rate mode is on.
- `inhibitory_flow_rate` did not solve consolidation and should not be prominent.
- `subtractive_reset` and `v_sat_frac` are diagnostics, not active defaults.
- Low-level distance and inhibitory turnover parameters should not crowd the
  primary experiment controls unless they are being actively swept.

Acceptable cleanup approaches:

```text
1. Add a category/advanced/archive field to CONFIG_SPEC and make the frontend show
   main controls by default with an "Advanced" disclosure.
2. Split CONFIG_SPEC into MAIN_CONFIG_SPEC and ARCHIVED_CONFIG_SPEC while keeping
   apply_config able to accept all parameters.
3. Remove archived entries from CONFIG_SPEC only if they remain settable through
   code/tests and documented presets.
```

Do not make a large frontend redesign. The goal is to reduce clutter while
preserving reproducibility.

## Tests

At minimum, add or update tests/scripts that verify:

```text
flag off preserves existing signed-spike behavior
structural FE gate is applied only to L2E/excitatory postsynaptic feedforward updates
gate shrinks as positive afferent sum approaches theta
gate does not update or inspect inhibitory gates
dashboard config endpoint exposes the new main controls and hides/archive-clusters old ones
```

Then run the relevant plain scripts:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_assembly_flow_credit.py
```

If adding a new focused test, use the existing plain-script style.

