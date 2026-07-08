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

## Next Experiment

The most targeted next experiment is reset-by-subtraction behind a flag:
`potential -= threshold` on `fire()` instead of resetting the winner to rest.
That directly attacks the measured discharge asymmetry where the winner is fully
reset but inhibited losers keep most of their accumulated charge.
