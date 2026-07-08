# Claude Prompt: Minimal Local Learning With Signed Input Signals

We are working in the SNN repo on branch `feature/inhibitory-plasticity`.

The project has accumulated too many compensating mechanisms. We need to simplify
the model back to the core local circuit:

- charge accumulates
- neurons fire
- firing updates local feedforward weights
- lateral inhibition regulates competition
- feedback inhibition regulates input/frequency
- repeat

The goal is not to optimize a complicated stack of remedies. The goal is to test
whether this minimal local loop can learn stable pattern ownership.

## Current Conceptual Change

Replace the current binary spike participation convention for feedforward
learning with a signed input signal:

```text
input present     -> signal = +1
input not present -> signal = -1
```

When a postsynaptic neuron fires, every positive feedforward synapse should get a
local signed update:

```text
dw_i = eta * p * (1 - (w_i / w_cap)^2) * signal_i
```

Where:

```text
p = clamp(theta / v_pre, 0, 1)
signal_i = +1 if that input was active in the firing input vector
signal_i = -1 if that input was inactive in the firing input vector
```

So:

- active inputs potentiate
- inactive inputs depress
- the same local equation handles both directions
- no global budget is needed to trade weights against each other

Weights must still be bounded locally:

```text
w_floor <= w_i <= w_cap
```

Use a small positive floor for feedforward excitatory weights so neurons do not
go permanently deaf to inputs. Keep inhibitory weights governed by the existing
inhibitory plasticity rule unless there is a clear local reason to change them.

## Simplification Request

Create a minimal model configuration or branch of the code that disables or
removes the current compensating mechanisms from the core learning path.

Disable by default for this experiment:

```text
confidence_consolidation = False
loser_depression = False
signed_depression = False
homeostasis = False
subtractive_reset = False
lasting_inhibition = False
event_driven = False
v_sat / membrane saturation = off
l2e_budget = False
```

Keep:

```text
charge accumulation
on-fire local feedforward plasticity
learned L2I lateral inhibition
L1I feedback inhibition
no refractory lockout for the minimal experiment (`refractory = 0`)
local weight floor and ceil
```

Important: the model currently has a single neuron implementation:

```text
neuron_flexible.py
```

Do not resurrect `neuron.py`.

## Implementation Details

Update `neuron_flexible.Neuron._update_weights` or add a clean experimental path
behind a default-off/default-config flag.

The feedforward update should use signed participation:

```text
participating = last_input_spikes > 0.5
signal = +1 for participating positive synapses
signal = -1 for non-participating positive synapses
dw = learning_rate * p * (1 - (w / weight_cap)^2) * signal
w = clip(w + dw, min_positive_weight, weight_cap)
```

Notes:

- Apply this only to positive/excitatory feedforward synapses.
- Do not update negative inhibitory gates in this method.
- `v_pre` must still be captured before discharge.
- `p = clamp(theta / v_pre, 0, 1)` should stay local.
- Remove the L2E positive-weight budget from this experimental path.
- Set the minimal experiment's refractory period to `0`. The goal is to let
  inhibition regulate firing frequency, not a hard refractory lockout.
- Preserve existing tests where possible, but update numeric expectations where
  they explicitly assume inactive positive synapses are untouched.
- If a flag is used, name it clearly, e.g. `signed_spike_learning=True`, and make
  the minimal experiment use it.

## Scale Consistency

Keep weight, charge, and threshold units consistent. This is critical.

The current fixed-point scale is suspect because many threshold/gate issues
appeared after moving from small floats to integer-scaled magnitudes. Do not mix
old small-float values with current `UNIT=1000` values casually.

Before changing dynamics, explicitly decide and document the numeric scale for
the minimal experiment:

```text
Option A: keep current UNIT-scaled values
Option B: revert the minimal experiment to the older small-float values
```

Whichever scale you choose, all these must be dimensionally consistent:

```text
membrane potential / charge
threshold
feedforward weights
inhibitory gate weights
weight floors and caps
learning-rate deltas
quadratic denominators
```

Rules of thumb:

```text
potentials, thresholds, weights, floors, caps -> same linear scale
learning deltas / eta -> same linear scale as weights
leak rates, probabilities, p, signal -> dimensionless
quadratic denominators for w^2 terms -> square of the weight scale
```

If using current fixed-point values:

```text
threshold=1 * UNIT
threshold_l2=8 * UNIT
weight_cap and floor in the same charge units
w_cap in (1 - (w / w_cap)^2) is a linear cap, not a quadratic denominator
```

Prefer writing the signed update as:

```text
dw = eta * p * (1 - (w / w_cap)^2) * signal
```

rather than:

```text
dw = eta * p * (1 - w^2 / w_max)
```

unless `w_max` is clearly documented as a quadratic denominator. This avoids
the previous ambiguity between linear caps and squared saturation denominators.

## Why Remove The Budget

The old budget was acting as an artificial resource normalizer:

```text
if one weight grows, all positive weights are renormalized downward
```

With signed input updates, inactive inputs naturally decrease on fire. That should
provide the increase/decrease pressure the budget was previously imposing. The
budget may be masking whether the local learning rule can manage receptive-field
formation on its own.

## Metrics: Correct Interpretation Of Tiling

Do not define one-to-one tiling as sustained per-cycle lockout.

Correct definition:

```text
Show pattern P1 -> neuron A wins.
Later show pattern P1 again -> neuron A wins again.
Show pattern P2 -> neuron B wins.
Later show pattern P2 again -> neuron B wins again.
Across 8 patterns, each pattern should have a stable owner.
```

Primary metric:

```text
visit-level ownership consistency
```

For each pattern:

```text
owner(pattern) = most common early winner across repeated visits
consistency(pattern) = fraction of visits where owner(pattern) won
```

Report:

```text
pattern -> owner map
distinct owners / N
per-pattern consistency
mean consistency
collisions
dead L2E count
firers per visit / hold
L2I spike count
L1I spike count
RF match score if available
```

Do not use `metrics_consolidation.py` dominance as proof of success. It has known
short-window artifacts.

## Required 5-Stage Harness

Build a staged harness instead of jumping straight to all eight patterns.

The harness should train and evaluate progressively:

```text
stage 1: one pattern, one owner, receptive field forms
stage 2: two orthogonal patterns, two owners
stage 3: rows only, three owners
stage 4: rows + columns, six owners
stage 5: all eight patterns
```

For each stage:

1. Train with repeated visits.
2. Evaluate visit-level owner consistency.
3. Run across seeds `1-4`.
4. Report where collapse first appears.
5. Keep dwell length explicit and swept where practical.

Dwell matters. A one-cycle visit can produce a misleading perfect first-winner
map. Include a dwell sweep, for example:

```text
dwell_cycles = [1, 2, 4, 8, 16, 40]
```

The harness should make it clear whether failures come from:

- feedforward receptive fields not forming
- lateral inhibition timing
- feedback inhibition
- long dwell exposure
- collisions between related patterns
- dead/silent neurons

## Locality And Hardware Constraint

All model operations must remain local and hardware-mappable.

Allowed:

- each neuron reads its own membrane potential
- each neuron reads its own threshold and refractory state
- each synapse reads its own weight and local presynaptic signal
- local L2I and L1I inhibition
- local floors, ceilings, and saturation

Not allowed:

- pattern labels inside the model
- global assignment tables used by learning
- cross-neuron rival lookup
- non-local per-target winner memory
- supervised correction signals

Measurement scripts may compute pattern-owner maps externally, but the network
cannot use that information.

## Deliverables

1. A clear patch implementing the minimal signed-spike learning experiment.
2. Removal or disabling of unnecessary compensating mechanisms for the minimal
   experiment.
3. Update `backend/simulation.py` so the minimal state can be run and viewed in
   the frontend:
   - expose the relevant config fields in `SimulationEngine.__init__`
   - apply `refractory=0` for the minimal experiment
   - apply `signed_spike_learning=True`
   - apply `l2e_budget=False`
   - keep lateral and feedback inhibition active
   - make the frontend/dashboard use this state through existing config or a
     clear constructor/default preset
4. Update `backend/api.py` / `CONFIG_SPEC` if needed so the frontend can toggle
   or display the minimal experiment settings.
5. A staged harness, e.g. `stage_learning_harness.py`.
6. A short report with:
   - stage-by-stage results
   - dwell sensitivity
   - seed sensitivity
   - whether signed input depression replaces the budget
   - first stage where ownership fails
7. Updated docs explaining the minimal experiment, the no-refractory choice, the
   scale convention, and the correct tiling metric.

Run at minimum:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_refractory_gating.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_8line_consolidation.py
PYTHONPATH=. .venv/bin/python stage_learning_harness.py
```

If any existing test must change because inactive positive synapses now depress
instead of staying untouched, update the test to reflect the new intended local
rule and explain why.
