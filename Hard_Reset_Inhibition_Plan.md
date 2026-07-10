# Hard Reset Inhibition Plan

This plan defines the immediate next inhibition experiment. It deliberately moves
away from dendritic fan-out and inhibitory flow-rate dynamics for now.

The goal is to test a simpler idea:

```text
L2I inhibition should use the loser's accumulated charge for learning,
then clamp the loser's membrane charge back to rest.
```

In other words, inhibition is not a small subtractive gate and not a slow flow.
It is a hard local reset of transient charge after the local learning signals
have been computed.

## Motivation

The measured round-robin problem has a clear asymmetry:

```text
winner fires -> winner resets to rest
losers are inhibited -> losers keep most of their charge
next race -> a loser starts ahead and can fire next
```

That means the winner pays the full reset cost, while losing competitors keep
enough membrane charge to continue the round-robin. Those repeated loser firings
also let multiple neurons keep learning the same pattern and push feedforward
weights toward cap.

This experiment changes the interpretation of L2I inhibition:

```text
once L2I declares a winner, losing L2E charge is consumed by local learning
and then cleared to rest
```

Weights are not wiped. Only transient membrane charge is reset.

## Core Hypothesis

If all non-winning L2E neurons are returned to the same membrane baseline after
an L2I event, then the next competition starts from a fair charge state. The
neuron with the best learned feedforward weights for the current pattern should
then rebuild charge fastest and repeatedly win.

This directly attacks loser carryover without adding:

```text
dendritic sites
inhibitory flow traces
lasting inhibition fields
loser depression
confidence consolidation
weight budgets
global assignment logic
```

## Intended Minimal Regime

The experiment should use the following regime:

```text
signed_spike_learning = True
l2i_hard_reset_losers = True
loser_depression = False
confidence_consolidation = False
signed_depression = False  # superseded by signed-spike rule
l2e_budget = False
homeostasis = False
lasting_inhibition = False
inhibitory_flow_rate = False
dendritic_inhibition = False
membrane_noise = 0
```

Keep lateral inhibition and feedback inhibition active. The change is only how
L2I inhibition acts on losing L2E membrane charge.

## Learning Rules

### 1. L2E feedforward learning: signed spikes

Keep the L2E excitatory feedforward rule local and signed:

```text
signal_i = +1 if input i participated in the firing volley
signal_i = -1 if input i did not participate

dw_i = eta_eff * signal_i * saturating_term_i
w_i  = clamp(w_i + dw_i, w_floor, w_cap)
```

The active/inactive input sign is the learning direction. Negative inhibitory
gates are not updated by this path.

If the structural free-energy gate is included in this experiment, it should be
used only for excitatory postsynaptic neurons:

```text
sum_pos  = sum(positive L1E -> L2E feedforward weights)
maturity = clamp(sum_pos / theta_l2e, 0, 1)
eta_eff  = eta_base * max(structural_fe_eta_floor, 1 - maturity)
```

This makes under-trained excitatory neurons plastic and mature neurons less
plastic. Do not apply this structural gate to L2I, L1I, or inhibitory gates.

If this structural gate is not implemented in the first pass, the first pass
should still use the current signed-spike L2E rule unchanged. The hard-reset
inhibition can be tested independently.

### 2. I-neuron / inhibitory-gate learning: old charge-based method

For inhibitory learning, do **not** use flow-based delivery or flow-based credit.
Use the charge that was present in the target neuron before reset.

On L2I inhibitory event into a non-winning L2E:

```text
v_pre = losing L2E membrane charge before reset
theta = losing L2E threshold
p_inh = clamp(v_pre / theta, 0, 1)
```

Then update the inhibitory gate by the old charge-based method:

```text
dw = eta_inh * p_inh * saturating_term
gate <- clamp(gate + dw, 0, gate_cap)
```

Preserve the inhibitory sign in storage:

```text
stored_weight = -gate
```

This learning rule uses the charge in the target neuron as evidence that it was
a real competitor. It does not use inhibitory traces, averages, global rank, or
pattern labels.

## Critical Ordering

The order is the main point of the experiment.

On an L2E winner event:

```text
1. Winner L2E fires and performs its signed-spike feedforward update.
2. Winner spike drives L2I.
3. If L2I fires, each non-winning L2E is processed:
   a. capture v_pre from the losing L2E
   b. use v_pre to update the L2I -> L2E inhibitory gate
   c. run any other allowed local I-learning bookkeeping
   d. clamp the losing L2E membrane charge to rest
4. Continue the normal timestep bookkeeping.
```

Do not reset before learning. The charge must be read first because it is the
local signal saying how serious that loser was.

The intended interpretation:

```text
charge above rest is the energy available for local learning;
after it is used, inhibition clears it.
```

## Hard Reset Rule

For each non-winning L2E target when L2I fires:

```text
V_loser <- rest
```

Usually:

```text
rest = 0
```

If excitatory or inhibitory current traces exist, this experiment should either:

```text
clear those traces for the reset L2E
```

or:

```text
disable flow modes so there are no residual traces
```

Recommended first pass:

```text
excitatory_flow_rate = False
inhibitory_flow_rate = False
```

This keeps the experiment about hard membrane reset, not hidden residual current.

## What This Replaces

Do not combine the first pass with:

```text
loser_depression
inhibitory_flow_rate
dendritic inhibitory fan-out
lasting inhibition
subtractive reset
v_sat clamps
confidence consolidation
homeostatic scaling
```

Those can be reintroduced later only if the hard reset shows a clear partial win
and a specific remaining failure.

## Expected Behavior

If the hypothesis is right:

- loser charge carryover drops to zero after L2I events,
- immediate rival re-fires should disappear or sharply decline,
- round-robin should weaken because losers no longer start the next race ahead,
- the best-matching neuron should rebuild charge fastest from its weights,
- losing competitors should receive fewer postsynaptic fires and therefore stop
  racing their feedforward weights to cap.

This is not a procedural owner assignment. The winner still has to win the next
race by having better weights for the input.

## Main Risks

The timing is critical.

Risk 1: early lucky lock-in

```text
If hard reset starts before weights differentiate,
the first slightly lucky winner may monopolize a pattern.
```

Risk 2: dead or starved neurons

```text
If losers are reset too aggressively across all patterns,
they may not fire often enough to learn.
```

Risk 3: multi-pattern capture

```text
The strongest neuron may still grab several patterns if signed-spike learning
does not protect prior ownership or allocate unused neurons.
```

Risk 4: hidden traces

```text
If flow traces are left active, resetting V may not actually reset the race.
Residual current could refill the membrane and obscure the result.
```

For this reason, the clean first pass should disable flow-rate modes.

## Flags

Suggested minimal flags:

```text
l2i_hard_reset_losers = False
hard_reset_clear_traces = True
hard_reset_after_learning = True
```

Optional timing gate if immediate reset causes early lock-in:

```text
hard_reset_start_step
hard_reset_start_epoch
hard_reset_min_l2i_weight
hard_reset_min_winner_margin
```

Do not add timing gates in the first implementation unless the simple version
clearly collapses too early.

## Measurements

Compare:

```text
baseline current minimal signed-spike regime
signed-spike + hard L2I loser reset
signed-spike + structural FE gate + hard L2I loser reset, if implemented
```

Report:

```text
single held-pattern dominance
interleaved all-8 distinct owners
sustained-presentation dominance
dead/ownerless patterns
immediate re-fire rate after L2I event
loser V/theta immediately before reset
loser V/theta immediately after reset
time to next rival L2E fire after L2I event
feedforward weight saturation rate
winner consistency on A -> B -> A pattern switches
L2I spike count and discharge count
```

Use at least 3 seeds.

## Success Criteria

Minimum useful result:

```text
immediate loser re-fires are strongly reduced
loser charge carryover is eliminated
single held-pattern ownership improves without loser depression
no large increase in dead neurons
```

Strong result:

```text
sustained dominance improves
distinct owners improve under interleaved all-8 training
pattern A -> B -> A returns to the same A owner
feedforward weights stop saturating across many co-specialists
```

## Failure Criteria

Archive or redesign if:

```text
one neuron captures most patterns
many L2E neurons go dead before learning
interleaved distinct owners decrease
hard reset only improves firing cleanliness but not ownership
flow traces or other mechanisms make the reset ambiguous
```

## Tests

Add focused tests before broad sweeps:

```text
1. flag off preserves current inhibition behavior
2. on L2I fire, non-winning L2E V is reset to rest
3. winner L2E is not reset by L2I
4. inhibitory learning uses v_pre captured before reset
5. reset happens after inhibitory learning, not before
6. loser depression is disabled in the clean hard-reset preset
7. signed-spike L2E feedforward update remains unchanged
8. flow-rate traces are disabled or cleared under the hard-reset preset
```

Suggested script:

```bash
PYTHONPATH=. .venv/bin/python test_hard_reset_inhibition.py
```

Also run:

```bash
PYTHONPATH=. .venv/bin/python test_inhibitory_plasticity.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_neuron.py
```

## Documentation Updates After Testing

If the experiment is implemented and measured, update:

```text
Current_Implementation_Methodology_Equations.md
Inhibition_And_Consolidation_State.md
AGENT_HANDOFF.md
```

Do not describe it as solved unless sustained-presentation dominance and
interleaved distinct ownership both improve.

