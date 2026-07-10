# Claude Inhibitory Delta Rule Prompt

Please inspect the current inhibitory plasticity rule, especially
`Neuron.apply_inhibition()` in `neuron_flexible.py`, the L2I->L2E gate setup in
`backend/simulation.py`, tests, and
`Current_Implementation_Methodology_Equations.md`.

## Default Run Context

The default excitatory feedforward learning mode is `signed_spike_learning=True`.
Treat this as the canonical/default run behavior for this phase.

The default excitatory update is:

```text
p = clamp(theta / v_pre, 0, 1)

for each positive synapse:
  signal_i = +1 if input i participated in the firing volley
  signal_i = -1 if input i did not participate

  dw_i = eta_exc * p * (1 - (w_i / w_cap)^2) * signal_i
  w_i  = clamp(w_i + dw_i, w_min, w_cap)
```

In code terms:

```python
if self.signed_spike_learning:
    pos = self._weights_array > 0
    if pos.any() and self.weight_cap > 0:
        w = self._weights_array[pos]
        signal = np.where(participating[pos], 1.0, -1.0)
        dw = self.learning_rate * p * (1.0 - (w / self.weight_cap) ** 2) * signal
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        self._weights_array[pos] = np.clip(w + dw, w_min, self.weight_cap)
    return
```

Do not change this excitatory signed-spike formula while implementing the
inhibitory delta/margin rule. The inhibitory change should be evaluated against
this default feedforward learning mode.

For tests, harnesses, and docs, make sure the default run means:

```text
signed_spike_learning = True
no excitatory weight budget
inhibitory delta/margin rule enabled if we decide it is the new default
instantaneous vs flow-rate mode controlled separately
```

## Problem

The current inhibitory gate update is:

```text
p  = clamp(v_pre / theta, 0, 1)
dw = eta * p * (1 - w^2 / w_max)
```

This makes every sufficiently-used inhibitory gate converge to the same
equilibrium:

```text
w* = sqrt(w_max)
```

So the gate magnitude encodes "has this target been inhibited enough times to
saturate," not "how much inhibition does this target need." In observed runs,
all L2I->L2E gates converge to about `1224.7` with almost no spread.

## Goal

Add a local inhibitory plasticity rule whose equilibrium depends on the inhibited
target's own pre-inhibition charge, so habitual near-winners / strong losers
learn stronger incoming inhibition while weak/dead units keep smaller gates.

This should be the default run behavior for this phase, but preserve the old
rule behind a config flag or compatibility option so it remains available for
ablation.

Suggested config:

```text
inhibitory_delta_rule = True
inhibitory_margin_frac = 0.5
```

If the repo convention strongly prefers preserving old defaults, keep the old
default but update the dashboard/default run config to enable the new rule.

## Desired Local Rule

For each real inhibitory discharge into a non-refractory target:

```text
v_pre = target membrane potential before inhibition
theta = target threshold
w     = current inhibitory gate magnitude, abs(weight)
```

Choose a desired post-inhibition margin:

```text
target_post = inhibitory_margin_frac * theta
```

Compute the target gate magnitude needed to bring this neuron down to that
margin:

```text
s = clamp(v_pre - target_post, 0, w_max)
```

Then update:

```text
dw    = eta * (s - w)
w_new = clamp(w + dw, 0, w_max)
```

Keep the inhibitory sign negative:

```text
weight = -w_new
```

Interpretation:

```text
high-charge loser / near-winner -> large s -> stronger future inhibition
low-charge loser / dead unit    -> small s -> weaker future inhibition
```

The equilibrium is target-specific:

```text
w* = s
```

or, over repeated events, the target's typical required discharge.

This is local: it uses only the inhibitory synapse weight, the target neuron's
`v_pre`, the target neuron's `theta`, and the arriving inhibitory spike.

## Important Constraints

- Preserve old inhibitory behavior when the compatibility flag selects the old
  rule.
- Apply this only when a real inhibitory discharge occurs and the target is not
  refractory, matching the current `apply_inhibition()` gating.
- Do not update negative inhibitory gates from excitatory fire logic.
- Do not change the signed-spike excitatory feedforward formula.
- Do not use global rank, population averages, or winner identity in the weight
  update.
- Keep L2I broadcast inhibition behavior unchanged for now.
- Do not implement distance weighting.
- Do not add membrane noise.
- Keep weight initialization uniform/random.

## Delivery vs Learning

For this first change, keep the actual inhibitory delivery as the existing
linear subtraction:

```text
V_post = max(V_pre - w, rest)
```

Only change the learning rule. If conductance-style delivery is worth testing
later, leave it as a separate future ablation.

## Docs

Update `Current_Implementation_Methodology_Equations.md` with a dedicated
subsection comparing:

1. Current saturating inhibitory rule and why it converges uniformly to
   `sqrt(w_max)`.
2. New local delta/margin rule and why its equilibrium is target-specific.
3. Locality argument: what variables are used and why no global information is
   required.
4. Default-run context: signed-spike excitatory learning remains the canonical
   feedforward rule and is unchanged by this inhibitory update.

## Tests

Add focused tests showing:

1. With the old inhibitory rule selected, old behavior is preserved.
2. With `inhibitory_delta_rule=True`, two targets with different `v_pre` values
   learn different inhibitory gate magnitudes.
3. A low-charge target below `target_post` gives `s=0`, so the gate decays or
   weakens toward zero.
4. Refractory targets still receive no inhibitory discharge and no gate update.
5. The signed-spike excitatory update still runs unchanged in default runs.
