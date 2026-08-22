# Pre-edit architecture note — Local Predictive Inhibition

Verified against the repository at branch `feature/inhibitory-plasticity`, commit
`8dbf626` (working tree; only untracked slideshow files present). This note records
the exact methods and array layouts that Section 3 of `Experiment.md` says to
confirm before editing. All line references are as-read during this session.

## 1. Active network (matches Experiment.md §3.1)

`backend/simulation.py` builds the active network in `SimulationEngine._build`:

- `N_PIX = 9` L1E (`InputLayer.excitatory_neurons`), `N_PIX = 9` L1I
  (`InputLayer.inhibitory_neurons`), `N_OUT = 8` L2E
  (`CorticalColumn.excitatory_neurons`), one shared L2I
  (`CorticalColumn.inhibitory_neuron`).
- Pixel indices are row-major in the 3×3 grid (`PATTERNS` in simulation.py).
- Neuron registry ids: `L1E{0..8}`, `L1I{0..8}`, `L2E{0..7}`, `L2I`
  (`_register_neurons`). Golden neuron order = that registration order
  (indices 0–8 L1E, 9–17 L1I, 18–25 L2E, 26 L2I).

### Array layouts (verified)

- **L1E afferents**: `Neuron` with `n_inputs=2`, weights `[-1.0, 1.0] * UNIT`
  (`_build` ~line 664). Index 0 = local-inhibition gate (negative, magnitude
  `UNIT = 1000 = theta_L1E`); index 1 = external pixel drive (+UNIT).
  `inhibitory_weight_cap = UNIT * weight_cap` so the saturating rule's quadratic
  term is exactly 0 at the gate (it does not drift under the legacy saturating
  rule). `learning_rate = 0.0`.
- **L1I afferents**: `Neuron` constructed by `InputLayer` with
  `n_inputs = max(1, n_feedback_inputs) = N_OUT = 8`. In `_build` each L1I gets
  `inh.weights = l1i_feedback_init.copy()` — one 8-element positive vector
  `[L2E0..L2E7]`, shared across all nine L1I (drawn once from `rng_fb`). There is
  **no** `L1E_i -> L1I_i` afferent today. `threshold = thr_l1i`,
  `weight_cap = thr_l1i`, `refractory_period = L1I_FEEDBACK_REFRACTORY = 2`,
  `leak_rate = L1I_LEAK_RATE (0.07)` only if `l1i_leak_enabled` else 0.0,
  `assembly_flow_credit = True`, `assembly_decay_frac = 0.0` (E→I assembly credit
  is the L1I learning rule today).
- **L2E afferents**: exactly `N_PIX = 9` feedforward pixel weights at indices
  `0..8` (column built `include_local_inhibition=False`, `feedforward_offset = 0`,
  `cortical_column_flexible.py`). **No** index-0 placeholder, **no** negative
  L2I→L2E gate.
- **L2I afferents**: `N_OUT` E→I weights (`set_lateral_excitation_weights`), no
  feedback (`n_feedback_inputs = 0`).

### Connections (matches §3.1)

`external -> L1E`; `L1I_i --| L1E_i` (delayed one step); `all L1E --> all L2E`;
`all L2E --> L2I`; `L2I -- unweighted competitive reset --> all L2E`
(`Neuron.apply_competitive_reset`, `synapse kind='reset_inhibition'`, weight=null);
`all L2E --> all L1I` (feedback). Synapse ids: `ff{i}->{j}`, `reset->{j}`,
`{j}->inh`, `li{i}` (L1I_i→L1E_i), `fb{j}->{i}` (L2E_j→L1I_i).

## 2. Timing and mechanisms (matches §3.2)

`SimulationEngine.step()` is the phase coordinator. Verified order:

1. External L1E charge deposited (`e.receive_input([0, ext])`) **before** queued
   `t-1` L1I inhibition is applied (`e.apply_inhibition([1,0])`). Preserved.
2. L1E spikes resolved (`check_threshold`/`fire`).
3. L1E→L2E feedforward delivered; L2 competition resolved
   (`_resolve_l2_competition`): argmax crosser fires, drives L2I, and if L2I fires
   it broadcasts `apply_competitive_reset` to every L2E. Chunked by
   `l2_charge_chunks` (dashboard = 20). `event_driven=True`.
4. L2E winner vector `l2e` (length `N_OUT`) delivered to every L1I
   (`inh.receive_input(l2e, t=t)`).
5. L1I fires: accumulator (`check_threshold`) since dashboard
   `l1i_immediate_relay=False`.
6. Membrane update (leak + refractory countdown) for all populations.
7. `self.l1i_feedback_delay = l1i.copy()` — L1I spike at `t` targets paired L1E at
   `t+1`. `l1i_feedback_delay` is a length-`N_PIX` register replaced every step.

`Neuron.apply_inhibition` subtracts linearly, floored at rest, records `v_pre`,
`v_post`; runs the selected inhibitory-gate rule (dashboard uses turnover via
`inhibitory_delta_rule=True`). Reuse it for the L1I→L1E gate.

L2 competition is `L2E spike -> L2I -> apply_competitive_reset()`, **not** a learned
L2I→L2E gate. Do not change it.

## 3. Runtime truth / dashboard overrides

`SimulationEngine.params` is runtime truth (set in `__init__`). `apply_config`
rebuilds in place; `TUNABLE` gates which keys are live. `backend/dashboard_config.py`
`DASHBOARD_OVERRIDES` supplies the active preset; `CONFIG_SPEC` has 15 controls.
Active dashboard values confirmed: `l2e_weight_cap_frac = 1/3`,
`l2_charge_chunks = 20`, `distance_weighting = True`,
`competitive_weight_update = "redistribution"`, `l1i_immediate_relay = False`,
`refractory = 1`. `_build` pins `excitatory_flow_rate=False` (and the other flow
flags) regardless of params (the "NEUTERED" block ~line 629): delivery is
instantaneous `V += dot(w, spikes)` (`snn/rules/delivery.InstantaneousDelivery`).

The headless runner (`experiments/runner.py`) changes patterns with **no** blank
timestep (`present()` calls `set_pattern` then `dwell` steps). Normative
inter-presentation blank interval for this experiment = 0.

## 4. Known runner defect (matches §3.3) — authorized repair

Active L2E have nine feedforward afferents at indices `0..8`, no index-0 inhibitory
placeholder. `experiments/runner.py` slices `_weights_array[1:1 + N_PIX]` in
`weight_saturation` (line ~154) and `_render_live_plots` (line ~360). With
`N_PIX = 9` that is `[1:10]` → indices 1..8 (8 values), **dropping pixel 0** and
returning one column short. Fix: authoritative helper returning the full
`N_OUT x N_PIX` matrix from `0:N_PIX`; add a nine-distinct-value sentinel test.
`run_combo`'s final snapshot (line ~305) already saves the full `_weights_array`
(correct); route it through the helper for consistency. Historical artifacts that
omitted pixel 0 are labeled noncomparable, not overwritten.

## 5. Golden-gate contradiction (recorded per Experiment.md lines 3–6)

`tests/golden/test_golden_equiv.py` **fails at baseline before any edit**: 98/100
arrays bit-exact; the two `engine_dashboard__*` arrays drift
(`__potentials` first at (t=0, neuron 18 = L2E0); `__weights_final` first at index
0). Cause: `golden_cases.collect()` builds the dashboard case from
`backend.api.engine`, whose seed = `backend.api._load_seed()` reads the
**uncommitted** `.claude/dashboard_seed.txt`; absent that file it returns 1. The
committed baseline was generated under a different (environment-specific) seed, so
a fresh checkout cannot reproduce it. Determinism verified (seed=1 identical across
runs); no seed in 0..59 reproduces the committed value.

Resolution (test-only, no neural-behavior change): make the dashboard golden case
build `SimulationEngine(seed=1, **DASHBOARD_OVERRIDES)` directly (committed state,
reproducible), and regenerate `golden_baseline.npz` **once** in Phase 0 before any
behavior change — asserting every non-dashboard array is byte-identical to the old
committed baseline. The gate is then locked and must stay bit-exact for all later
phases with new features off.

## 6. Planned new files (no second simulator / config system)

- `snn/rules/predictive.py` — focused predictor rule + local trace (Experiment §6).
- `experiments/predictive_inhibition/` — experiment driver: five modes, schedules,
  reference tape/replay, metrics, artifacts (Experiment §9–12). The authoritative
  L2 feedforward-matrix helper lives in `experiments/runner.py` and is imported
  here. `SimulationEngine` is extended in place; parameters added with
  legacy-preserving defaults.
