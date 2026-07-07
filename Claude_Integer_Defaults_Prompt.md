# Prompt for Claude: Integer Defaults for Gates, Leaks, and Thresholds

You are working in `/home/adasgup/Documents/SNN` on branch
`feature/inhibitory-plasticity`.

First read:

- `Current_Implementation_Methodology_Equations.md`
- `README.md`
- targeted code ranges in `backend/simulation.py`, `neuron.py`, and
  `neuron_flexible.py`

Do not reread unrelated docs unless a targeted lookup is insufficient.

## Goal

Change the default numeric representation for gate weights, leak controls, and
thresholds so the default model avoids floating-point numbers entirely in those
categories. Use integer or fixed-point representations. Preserve the current
relative dynamics as closely as possible.

This is not a request to solve one-to-one consolidation yet. Do not add labels,
supervision, procedural winner assignment, winner facilitation, or hard resets.

## Required Design Constraint

Do not blindly replace decimals with nearby integers.

Current values like `leak_l2 = 0.01`, `L2I_LEAK_RATE = 0.07`,
`L2_GATE_INIT = -0.5`, and `L2_GATE_WMAX = 1.5` encode ratios. Replacing them
with `1`, `7`, `0`, or `2` would change the model. Instead, introduce an
integer fixed-point convention and update the equations consistently.

Recommended convention:

```python
UNIT = 1000
LEAK_SCALE = 1000
```

Then map current defaults as:

```text
threshold 1.0        -> 1000
threshold_l2 8.0     -> 8000
L2_GATE_INIT -0.5    -> -500
L2_GATE_WMAX 1.5     -> 1500
leak_l1 0.10         -> 100 / LEAK_SCALE
leak_l2 0.01         -> 10 / LEAK_SCALE
L1I_LEAK_RATE 0.07   -> 70 / LEAK_SCALE
L2I_LEAK_RATE 0.07   -> 70 / LEAK_SCALE
```

For fractional initialization ranges, prefer numerator/denominator constants:

```text
0.25 -> 1 / 4
0.5  -> 1 / 2
```

Use integer arithmetic to derive initialized weights.

## Scope

Update all default gate weights, leak values, and thresholds in:

- `backend/simulation.py`
- `neuron.py`
- `neuron_flexible.py`
- any tests that hard-code expected values in those categories
- dashboard serialization only if needed to display fixed-point values clearly

Preserve the public simulation behavior as much as possible. If API outputs need
human-readable values, provide explicit conversion helpers rather than leaking
implicit float arithmetic into model state.

## Equation Changes

Replace leak update:

```python
V += leak_rate * (resting_potential - V)
```

with integer fixed-point arithmetic, for example:

```python
V += (leak_num * (resting_potential - V)) // LEAK_SCALE
```

or use a helper such as:

```python
def leak_toward_rest(v, rest, leak_num, leak_scale):
    return v + (leak_num * (rest - v)) // leak_scale
```

If truncation changes behavior too much, use deterministic integer rounding:

```python
def div_round(n, d):
    return (n + d // 2) // d if n >= 0 else -((-n + d // 2) // d)
```

For thresholds and gate weights, keep all stored neuron potentials and weights in
the same fixed-point unit so comparisons such as:

```python
potential >= threshold
```

remain integer comparisons.

## Audit Requirements

After the change, run targeted searches:

```bash
rg -n "threshold.*[0-9]+\\.[0-9]+|leak.*[0-9]+\\.[0-9]+|GATE.*[0-9]+\\.[0-9]+" .
rg -n "dtype=float|astype\\(float\\)|float\\(" neuron.py neuron_flexible.py backend/simulation.py
```

The first command should find no default gate/leak/threshold decimals.

The second command may reveal unrelated serialization or diagnostic conversions.
Do not ignore it. Either remove the conversion from the default model path or
document why it is strictly display-only. The goal is no floating point in the
model state path for gates, leaks, thresholds, potentials, and weight updates.

## Tests

Run:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_8line_consolidation.py
PYTHONPATH=. .venv/bin/python test_inhibitory_plasticity.py
PYTHONPATH=. .venv/bin/python test_refractory_gating.py
```

If the exact numeric assertions fail because fixed-point rounding changes a
least-significant unit, update the assertions to compare the integer fixed-point
quantity or its documented rational equivalent. Do not weaken behavioral tests
that verify refractory gating, floor-at-rest inhibition, no hard reset, and broad
L2E participation.

## Guardrails

- Do not reintroduce winner facilitation.
- Do not add supervised labels or pattern IDs to the learning rule.
- Do not add procedural one-to-one assignment logic.
- Do not use hard WTA resets.
- Do not change the qualitative L2I/L1I two-regime integrator design unless the
  fixed-point conversion forces a mechanical equivalent.
- Keep edits narrow and explain any unavoidable rounding effect.

## Deliverable

Commit-ready code plus a short note describing:

- the fixed-point scale used,
- every old default and its new integer equivalent,
- whether model state is fully integer on the default path,
- any remaining float conversions and why they are display-only or deferred,
- the test results.

