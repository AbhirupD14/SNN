# SNN

A from-scratch spiking neural network for four center-crossing 3x3 line
primitives: the middle row, middle column, and two diagonals. The project is
intentionally small: pure NumPy neurons, local plasticity, no gradients, no
labels, and no global error signal.

Every retained pattern activates the center pixel. This narrows the task toward
center-on retinal stimuli, but L1 is still a binary pixel encoder; an explicit
antagonistic surround-off receptive field is not yet modeled.

## Core Status

The current branch is `feature/inhibitory-plasticity`.

The four-pattern task deliberately increases overlap because all stimuli share
the center pixel. Pool participation remains good, but stable one-to-one
ownership must be re-measured on this task rather than inferred from the earlier
eight-pattern results. The honest metric remains sustained dominance: hold one
pattern for about 40 intrinsic cycles and measure how often its modal specialist
wins.

Do not use `metrics_consolidation.py` as the ownership metric. It presents each
pattern in short visits and reports the reliable first-cycle-after-switch winner,
so it can print perfect short-visit dominance while sustained presentation still
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

> **Architecture update (L2 hard-reset competitive depression).** L2I no longer
> suppresses L2E through a learned negative gate. L2I *recruitment* is still learned
> on its positive `L2E -> L2I` inputs, but its *output* is an unweighted event: when
> L2I fires, every non-winning L2E is hard-reset to rest (membrane and pending
> current traces cleared) and its participating positive feedforward weights are
> locally depressed, scaled by its own pre-reset charge. There is no learned
> `L2I -> L2E` magnitude anywhere on the active path. See
> `L2_Hard_Reset_Competitive_Depression_Spec.md` and `Neuron.apply_competitive_reset`.
> Statements below about learned `L2I->L2E` gates describe the superseded design and
> apply only to the legacy standalone `apply_inhibition` path (still used for
> `L1I->L1E` feedback).

Architecture:

```text
L1E pixel encoders -> L2E pattern integrators
L2E winners       -> shared L2I   (learned positive E->I recruitment)
L2I fires         -> unweighted competitive reset of every non-winner L2E
                     (hard reset to rest + local depression of participating +weights)
L2E feedback      -> L1I input suppression
```

The eight L2E neurons each receive exactly one trainable feedforward synapse per L1
pixel and no negative gate. The output pool is deliberately overcomplete for the
four patterns. E/I identity is carried by synapse sign.

Default engine highlights:

- `threshold_l2 = 8 * UNIT`
- `input_period = 1`; held active pixels are driven every step
- `cycle_period = volley_period`; the intrinsic clock is independent of input
- `signed_spike_learning = True`; L2E has no feedforward weight budget
- `l1i_immediate_relay = False`; L1I is a trainable accumulator
- L2E, L2I, and L1I membrane leaks are off by default
- balanced L2E initialization is opt-in; legacy wide random init is the default
- `event_driven = True`
- `lasting_inhibition = False`
- `homeostasis = False`

Trainable L1I feedback is delivered through a one-step register: an L1I spike at
step `t` suppresses its paired L1E at `t+1` only. The L1I bank starts from one
shared random L2E feedback vector, uses temporal contributor credit, and blocks
consecutive feedback spikes. It therefore learns a phase-aligned alternating
rhythm under constant stimulation rather than splitting pixels into random phase
groups; the dashboard regression reaches a 0.5 firing rate for both active L1E
and L1I neurons.

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
PYTHONPATH=. .venv/bin/python test_constant_input_feedback.py
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
(charge → fire → local feedforward update → L2I competitive reset → L1I
feedback inhibition → repeat) can learn stable pattern ownership, with all the
accumulated compensating mechanisms turned off. See
`Claude_Minimal_Signed_Spike_Learning_Prompt.md`.

**The rule (`signed_spike_learning=True`, L2E only, default off).** On a
postsynaptic fire, every *positive* feedforward synapse gets one local **signed**
update — active inputs potentiate, inactive inputs depress — through the shared
**direction-aware bounded kernel** (`snn.rules.bounded_signed_update`), which is
also what the competitive-depression loser update uses:

```
q        = clamp((w_i - w_floor) / (w_cap - w_floor), 0, 1)
signal_i = +1 if input i was active in the firing volley, else -1
p        = clamp(theta / v_pre, 0, 1)         # or the structural gate when enabled
dw_i     = +learning_rate * p * (1 - q^2)          if signal_i = +1   (H_up)
dw_i     = -learning_rate * p * (1 - (1 - q)^2)    if signal_i = -1   (H_down, reflected)
w_i      = clip(w_i + dw_i, w_floor, w_cap)
```

The reflected downward branch is required: it goes to zero at `w_floor` (a floored
weight can't be pushed lower) and stays fully effective at `w_cap` (a capped losing
weight can still be depressed). There is **no weight budget**: the `-1` signal on
inactive inputs supplies the downward pressure the budget used to impose. This one
kernel replaces the whole potentiation + OFF-depression + confidence + budget stack,
and the same kernel drives the L2I-event competitive depression.

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
per-afferent positive weight cap = L2E threshold / 3       (l2e_weight_cap_frac=1/3)
L2I threshold                    = L2E threshold / 3       (l2i_threshold_frac=1/3)
L1I threshold                    = L2I threshold           (l1i_threshold_frac=1.0)
positive-afferent floor          = 1                       (pos_weight_floor=1)
```

so **three maximally strong active afferents reach threshold** (matching the
3-pixel line patterns), one strong winner can recruit its inhibitory neuron, and
E→I init ranges (`[0.25,0.5]×thr_I`) recompute from the lowered I thresholds.
The active L2E has no negative afferent at all — L2 competition is the unweighted
competitive reset, not a learned gate. Defaults
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
