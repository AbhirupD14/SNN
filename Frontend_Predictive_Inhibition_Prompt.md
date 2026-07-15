# Frontend Brief: Surface the Local Predictive Inhibition path

The predictive-inhibition mechanism (paired `L1E_i → L1I_i` edges + per-L1I local
trace + Section 6 predictor over the L2E→L1I feedback weights) is fully implemented
in the engine and validated (`experiments/predictive_inhibition/FINAL_REPORT.md`),
but the dashboard cannot see or enable it. Add the small, brief-sanctioned browser
surface (Experiment.md §14: "at most a predictive preset selector, paired-local
enabled, predictor enabled, and one strength control").

## Non-negotiable constraints

- **Default OFF.** Do NOT add any predictive flag to `DASHBOARD_OVERRIDES`.
  `paired_local_enabled` must stay `False` by default so the current network, the
  live view, and the `engine_dashboard` golden case (`tests/golden/`) are all
  unchanged and bit-exact. Enabling the preset is an explicit user action.
- **Reuse `backend/dashboard_config.py`.** Do not duplicate defaults in JavaScript or
  create a second config system. New controls are plain `CONFIG_SPEC` entries; their
  current values come from `config_values(params)` (already reads `params[key]`).
- Keep functional positions separate from frontend-only display spacing.
- The golden gate and all 23 root tests must stay green after the change.

## What to add

### 1. `backend/simulation.py` — make the keys live-tunable
`apply_config` rejects any key not in `SimulationEngine.TUNABLE`. Add:
`paired_local_enabled`, `predictive_feedback_enabled`, `l2_to_l1i_delivery_enabled`
(booleans), and one strength control — use `predictive_output_gate_frac` (float,
the L1I→L1E suppression strength; most visually meaningful). Add the three flags to
the bool-coercion branch in `apply_config`; the float already follows the generic
float branch. No new engine params are needed — all four already exist.

Keep these as independent controls. Their valid full-predictive combination is:

```text
paired_local_enabled = true
predictive_feedback_enabled = true
l2_to_l1i_delivery_enabled = true
```

The engine already makes predictor learning contingent on the paired topology, so
do not add predictive logic or a second preset abstraction in the frontend. Explain
the dependency in the control descriptions instead.

### 2. `backend/dashboard_config.py` — expose the controls
Append to `CONFIG_SPEC` (they render automatically via `frontend/controls.js`):
- `paired_local_enabled` — toggle — "Paired local evidence (L1E→L1I)".
- `predictive_feedback_enabled` — toggle — "Predictor learning (L2E→L1I)".
- `l2_to_l1i_delivery_enabled` — toggle — "Deliver L2E feedback to L1I".
- `predictive_output_gate_frac` — range 0.0–1.0 step 0.05 — "L1I→L1E gate (×θ)".

`config_values()` already includes any `CONFIG_SPEC` key, so no change there. A note
in the control descriptions that these rebuild the network (like the other structural
toggles) is enough; do not build a nested preset system.

### 3. `frontend/renderer.js` — draw the `local{i}` edges
The topology already serializes `local{i}` synapses (kind `local_evidence`, source
`L1E{i}`, target `L1I{i}`) when paired. The renderer colors edges by
`COLORS[s.kind]`, and there is no `local_evidence` entry, so those edges currently
render with an undefined color. Add one entry to the `COLORS` map, e.g.
`local_evidence: 0x9be15d` (a distinct green-yellow, since it is an
excitatory-evidence edge, not inhibition).

Do **not** special-case it like `reset_inhibition`: `local_evidence` is a genuine
weighted synapse, while the reset fanout is structurally unweighted. Let the existing
weighted opacity and weak-filter paths handle it. Under the active predictive preset
its serialized magnitude is about 1066.67, well above the renderer's absolute
`WEAK = 0.25` threshold, so it will remain visible.

### 4. `frontend/inspector.js` — label the edge
`inspector.js` special-cases `reset_inhibition` (~line 111). Add a short branch for
`local_evidence`: describe it as the fixed, non-plastic paired local afferent
(nominally 0.40·G), display its actual serialized weight rather than hard-coding a
number, and note the feedback afferents (`fb{j}->{i}`) are the ones the predictor
moves. Do not add the local trace in this frontend pass: `l1i_trace` is backend
state but is not currently present in the dynamic websocket envelope, and expanding
that protocol is outside this focused task.

## Verify

1. `PYTHONPATH=. .venv/bin/python tests/golden/test_golden_equiv.py` — still bit-exact.
2. `for t in test_*.py; do PYTHONPATH=. .venv/bin/python "$t"; done` — all pass.
3. Add a focused config test: defaults still serialize no `local{i}` edges; applying
   all three boolean controls rebuilds with exactly nine paired `local{i}` edges and
   no cross-pairs; applying `predictive_output_gate_frac` changes the nine L1E
   inhibitory gate magnitudes after rebuild.
4. Start the dashboard. Enable paired local evidence: the nine `local{i}` edges
   appear (L1E_i→L1I_i, no cross-pairs) and the inspector labels them. Enable
   predictor learning and feedback delivery for the full predictive mechanism;
   changing gate strength changes suppression after Apply. Disable paired local
   evidence and Apply: the topology returns to the legacy view.

Scope guard: do not rebuild or prune the existing 15 controls, and do not add
predictive logic to the engine — it is done. This is display + four controls only.
