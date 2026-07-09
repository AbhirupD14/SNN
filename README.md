# SNN

A from-scratch spiking neural network for the eight 3x3 line primitives:
three rows, three columns, and two diagonals. The project is intentionally small:
pure NumPy neurons, local plasticity, no gradients, no labels, and no global
error signal.

## Core Status

The current branch is `feature/inhibitory-plasticity`.

The network has good participation and a clean modal assignment: after
interleaved training, the eight patterns usually map to eight distinct L2E
specialists. The hard part is not solved yet: under sustained presentation, the
pool still round-robins. The honest metric is sustained dominance: hold one
pattern for about 40 intrinsic cycles and measure how often its modal specialist
wins. Current defaults are around `8/8` distinct modal winners but only
`0.34-0.36` mean sustained dominance, not stable one-to-one ownership.

Do not use `metrics_consolidation.py` as the ownership metric. It presents each
pattern in short visits and reports the reliable first-cycle-after-switch winner,
so it can print `8/8` and `1.00` dominance while sustained presentation still
rotates.

## Code Map

- `neuron_flexible.py` - the single neuron implementation. It supports both
  fixed fan-in (`Neuron(n_inputs=...)`) and staged wiring via
  `add_input_connection()` / `finalize_connections()`.
- `layers.py` - `InputLayer` and an older simple `CorticalColumn` wrapper.
- `cortical_column_flexible.py` - explicit per-source feedforward fan-in for the
  active L2 column.
- `backend/simulation.py` - the active engine: builds L1/L2, steps dynamics,
  owns defaults, live config, auto-cycle, and serialization state.
- `backend/api.py` - FastAPI REST/WebSocket app and static frontend server.
- `frontend/` - vanilla JS / Three.js dashboard.
- `AGENT_HANDOFF.md` - the most current project handoff and experimental status.

## Model Summary

Architecture:

```text
L1E pixel encoders -> L2E pattern integrators
L2E winners       -> shared L2I
L2I               -> learned inhibitory gates onto L2E pool
L2E feedback      -> L1I input suppression
```

The L2E neurons each receive one trainable feedforward synapse per L1 pixel plus
one local inhibitory gate from L2I. E/I identity is carried by synapse sign.

Default engine highlights:

- `threshold_l2 = 8 * UNIT`
- `confidence_consolidation = True`
- `loser_depression = True`
- `signed_depression = True`
- `eta_off = 0.20`
- L2E feedforward budget = `2 * threshold_l2`
- `event_driven = False`
- `lasting_inhibition = False`
- `homeostasis = False`

Excitatory plasticity runs on a postsynaptic spike and updates only positive
synapses active in the most recent input event:

```text
p  = clamp(theta / V_pre, 0, 1)
dw = eta * p * (1 - w^2 / w_max)
```

Signed depression optionally pushes inactive positive gates toward the floor on
the same spike. Confidence consolidation slows learning for mature gates and
protects them from loser depression.

Inhibitory plasticity runs only when a negative synapse actually discharges a
non-refractory target:

```text
V_pre = V
V     = max(V - |w|, rest)
p     = clamp(V_pre / theta, 0, 1)
dw    = eta * p * (1 - |w|^2 / w_max)
```

The L2I -> L2E gate saturates below the L2 threshold, so it remains a partial
learned discharge rather than a hard reset.

## Tests

The tests are plain scripts:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_refractory_gating.py
PYTHONPATH=. .venv/bin/python test_inhibitory_plasticity.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_8line_consolidation.py
```

`test_8line_consolidation.py` is an older characterization path and still
contains a no-counter-force collapse scenario. `test_l2_competition.py` verifies
the active engine no longer collapses to one winner and that L2I-mediated gates
adapt.

## Dashboard

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/uvicorn backend.api:app
```

Then open <http://127.0.0.1:8000>. See `docs/DASHBOARD.md` for the REST and
WebSocket protocol.

## Minimal signed-spike experiment

A stripped-down configuration that tests whether the **core local loop alone**
(charge → fire → local feedforward update → learned L2I lateral inhibition → L1I
feedback inhibition → repeat) can learn stable pattern ownership, with all the
accumulated compensating mechanisms turned off. See
`Claude_Minimal_Signed_Spike_Learning_Prompt.md`.

**The rule (`signed_spike_learning=True`, L2E only, default off).** On a
postsynaptic fire, every *positive* feedforward synapse gets one local **signed**
update — active inputs potentiate, inactive inputs depress — through the same
saturating term written with a **linear** cap:

```
signal_i = +1 if input i was active in the firing volley, else -1
p        = clamp(theta / v_pre, 0, 1)
dw_i     = learning_rate * p * (1 - (w_i / w_cap)^2) * signal_i
w_i      = clip(w_i + dw_i, w_floor, w_cap)
```

There is **no weight budget**: the `-1` signal on inactive inputs supplies the
downward pressure the budget used to impose. Negative inhibitory gates are never
touched here (they still learn only via `apply_inhibition`). This one equation
replaces the whole potentiation + OFF-depression + confidence + budget stack.

**Minimal config** (used by `stage_learning_harness.py`, exposed on the engine
and dashboard): `signed_spike_learning=True`; `confidence_consolidation`,
`loser_depression`, `signed_depression`, `homeostasis`, `subtractive_reset`,
`lasting_inhibition`, `event_driven`, membrane saturation (`v_sat_frac`), and
`l2e_budget` all **off**; `refractory=0`. The no-refractory choice is
deliberate: **inhibition**, not a hard lockout, is meant to regulate firing
frequency. Lateral (L2I) and feedback (L1I) inhibition stay active.

**Capacity rule** (`l2e_weight_cap_frac`, `pos_weight_floor`, and the I-threshold
fractions, set in the dashboard/harness preset):

```
per-afferent positive weight cap = E-neuron threshold / 3   (l2e_weight_cap_frac=1/3)
I-neuron threshold               = E-neuron threshold / 3   (l2i/l1i_threshold_frac=1/3)
positive-afferent floor          = 1                        (pos_weight_floor=1)
```

so **three maximally strong active afferents reach threshold** (matching the
3-pixel line patterns), one strong winner can recruit its inhibitory neuron, and
E→I init ranges (`[0.25,0.5]×thr_I`) recompute from the lowered I thresholds.
The negative L2I→L2E gate is **not** floored as a positive weight — it stays
negative and is bounded by magnitude in `apply_inhibition`. Defaults
(`l2e_weight_cap_frac=1.0`, `pos_weight_floor=None`, I-threshold fracs `=1`)
reproduce the prior behavior, so the existing tests are unchanged.

**Scale convention (Option A — current fixed-point `UNIT=1000`).** All
charge/threshold/weight magnitudes share the linear `UNIT` scale
(`threshold_l2 = 8*UNIT`, L2E `weight_cap = thr_l2/3`, floor `= 1`); `p`,
`signal`, and leaks are dimensionless. The `w_cap` in `(1 - (w/w_cap)^2)` is the
**linear** cap, not a squared denominator, so linear-vs-quadratic scaling is
unambiguous. `test_neuron.py::test_fixed_point_scale_invariance` guards this.

**Dashboard views.** The **Spike Raster** shows discrete spikes only; a separate
**Charge / time** overlay shows membrane charge `V/θ` per neuron (threshold line,
spike peaks, carryover across pattern switches); a **Weights / time** overlay
shows a selected L2E's feedforward weights and inhibitory gate evolving toward
the cap (RF formation under the signed `+1/-1` rule). All three open full-screen
from the bottom tab bar; the low-value activation-histogram, "currently firing",
statistics, and rolling-line-chart panels were removed.

**Tiling metric.** Ownership is **visit-level**, not per-cycle:
`owner(P)` = most common early (first) winner across P's repeated visits;
`consistency(P)` = fraction of visits that owner won. Do **not** use
`metrics_consolidation.py` dominance (short-window artifact).

```bash
PYTHONPATH=. .venv/bin/python stage_learning_harness.py
```

The harness grows the task in five stages (1 pattern → 2 disjoint → rows → rows +
columns → all 8), sweeps dwell length `[1,2,4,8,16,40]`, runs seeds 1–4, and
reports where ownership first collapses. **Finding:** receptive fields *do* form
under the signed rule with no budget (RF-match ≈ 1.0 for simple stages), and
distinct owners hold through stage 3, but patterns start **sharing owners at
stage 4** (rows and columns overlap on a shared pixel) and per-pattern owner
*stability* is weak from stage 1 — because with no budget nothing stops several
neurons from co-specializing on the same pattern, and with `homeostasis` off
there is **no recruitment** force, so lateral inhibition starves losing units
(they rarely fire, so their feedforward weights rarely update) and they go dead.

## Next Experiment

The next targeted experiment is deterministic distance-weighted signal
attenuation. Keep uniform feedforward initialization and `membrane_noise=0.0`,
place L2E functional positions on a compact jittered lateral layer, and deliver
synaptic events as `w / d^2`.

Measure it first on L1E→L2E feedforward drive, then on the competition-critical
case where L2I→L2E inhibitory discharge is also distance-attenuated. The question
is whether local geometry changes competition and tiling without adding random
membrane noise or relying on special weight initialization.
