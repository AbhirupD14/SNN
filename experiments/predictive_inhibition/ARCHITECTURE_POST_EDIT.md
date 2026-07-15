# Post-edit architecture map — Local Predictive Inhibition

Final methods and array layouts after Phases 0–4. Pairs with
`ARCHITECTURE_PRE_EDIT.md` (Section 15.1). Nothing here changes L2 competition, L2
feedforward learning, the fixed-point units, or the shared `SimulationEngine`.

## New / changed symbols

### `neuron_flexible.Neuron` (two default-True strategy boundaries)
- `postsynaptic_learning_enabled` (default True) — when False, `fire()` still resets
  and delivers, but `_update_weights` returns immediately (generic excitatory
  learning disabled). Set False on predictive L1I.
- `inhibitory_plastic` (default True) — when False, `apply_inhibition` still performs
  the graded, rest-floored discharge but skips the gate-magnitude rule. Left True on
  the paired L1I→L1E gate so it LEARNS (see below); the non-paired legacy gate stays
  frozen-at-saturation instead. Defaults keep the golden baseline bit-exact.

### `snn/rules/predictive.py` (the only predictive learning rule)
- `trace_lambda(tau)` = exp(-1/tau); `decay_and_set_trace(x, spiked, lam)` — the
  per-L1I local trace x_i (Section 6.1).
- `predictor_update(w_fb, x_i, delivered, G, eta_up, eta_down)` — Section 6.2 update
  on u = w/G; only delivered feedback afferents move; clamps to [0, G]; the caller
  passes only the feedback slice so the local afferent never enters.

### `backend.simulation.SimulationEngine`
- New params (all legacy-preserving): `paired_local_enabled`,
  `predictive_feedback_enabled`, `l2_to_l1i_delivery_enabled`,
  `predictive_local_weight_frac`, `predictive_feedback_init_frac`,
  `predictive_feedback_eta_up`, `predictive_feedback_eta_down`,
  `predictive_trace_tau_steps`, `predictive_l1i_leak_rate`,
  `predictive_output_gate_frac` (learned-gate CAP ×θ_L1E),
  `predictive_output_gate_init_frac` (weak init ×θ_L1E),
  `predictive_output_gate_eta` (gate learning rate).
- New state: `_l1i_paired`, `_l1i_fb_offset` (0 legacy / 1 paired), `_l1i_G`,
  `_predictive`, `_trace_lambda`, `l1i_trace` (N_PIX), `delivered_feedback` (N_OUT),
  `actual_l2e` (N_OUT), `_feedback_override` (replay hook).
- New method `_select_delivered_feedback(l2e)` (STEP 7): override → live (if
  delivery enabled) → zeros.
- `step()` implements the exact Section 7 order: STEP 5 trace decay/set after L1E
  resolves; STEP 7–10 deliver one `[local, fb0..fb7]` vector per L1I using pre-update
  weights, then apply the predictor to the feedback slice (even while the L1I is
  refractory). The legacy delivery path now honours `l2_to_l1i_delivery_enabled`.

## Final array layouts

- **L1E afferents** `[local_gate, external]`. Legacy gate = −UNIT (frozen at
  saturation). Paired gate is LEARNED: starts weak at
  −`predictive_output_gate_init_frac`·θ_L1E, matures via real paired inhibitory
  events (margin rule, margin_frac 0, `predictive_output_gate_eta`) up to the cap
  `predictive_output_gate_frac`·θ_L1E; at cap 1.0 a mature gate floors the pixel to
  rest (effective reset). `inhibitory_plastic=True`.
- **L1I afferents**: legacy `[L2E0..L2E7]` (8); paired
  `[local L1E_i, L2E0..L2E7]` (9) — index 0 fixed at 0.40·G (never plastic),
  indices 1..8 = feedback (predictor-plastic). θ_L1I = G = 2666.67; feedback init
  0.10·G; L1I leak 0.50 (paired). `postsynaptic_learning_enabled=False` when paired.
- **L2E afferents**: unchanged — exactly N_PIX pixel weights at 0..N_PIX-1.
- **Serialization**: `local{i}` (kind `local_evidence`) added when paired;
  `fb{j}->{i}` retained; `_all_weights`/topology/emitted offset the feedback index.

## Experiment package (`experiments/predictive_inhibition/`)
`config` (five modes as flag sets + features), `schedules` (4-pattern + contextual +
reversal + warm-up), `tape` (reference tape + per-phase derangement),
`driver`/`Recorder` (protocol + per-timestep history), `metrics` (Section 12,
recomputed from history), `run_matrix` (durable, resumable 100-run orchestrator),
`report` (aggregate + Section 13 acceptance + Section 15 report). The authoritative
L2 feedforward-matrix helper `l2_feedforward_matrix` lives in `experiments/runner.py`.
