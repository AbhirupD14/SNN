# Coincidence Pyramidal Cell — Implementation Status

Controlling docs: `prompts/Claude_Coincidence_Pyramidal_Cell_Implementation_Prompt.md`
(workflow) and `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md` (contract).

## Phase table

| Phase | Title | Status |
| --- | --- | --- |
| 0 | Repository audit and clean baseline | ✅ complete |
| 1 | Shared LIF and exact segment primitives | ✅ complete |
| 2 | Coincidence dendrite and C-cell local behavior | ✅ complete |
| 3 | Graph vocabulary, validation, construction | ✅ complete |
| 4 | Event-resolved scheduler and emergent WTA | ✅ complete |
| 5 | Full `rg_coincidence` preset | ✅ complete |
| 6 | Public protocol and frontend support | ✅ complete |
| 7 | Scientific validation and documentation | ✅ complete |

---

## Phase 0 — audit and baseline (complete)

### Baseline test result
- Command: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`
- Result: **160 passed, 2 warnings** (pre-existing FastAPI `on_event` deprecation
  warnings in `test_serialization_api.py`; not failures).
- **No pre-existing test failures.**

### Working-tree baseline (do not disturb)
- Modified (pre-existing user work, PRESERVE): `docs/agent_workflow_evolution.md`.
- `git diff --check` whitespace baseline: trailing-whitespace diagnostics **only** in
  `docs/agent_workflow_evolution.md` (lines 50–202). This is the Phase 0 whitespace
  baseline; do not edit that file to clean it.
- Untracked (preserve): the two controlling docs + this status file.

### Golden oracle note (pre-existing, benign)
- `tests/test_golden_topology.py` asserts **only** the `frames` (behavioural) digest.
  All four presets are `FRAMES OK`.
- The standalone `tests/golden_topology.py verify` script additionally prints
  `TOPO DIFFERS` for pi/old/rg. That topology-fingerprint digest includes serialized
  `params` and has drifted since the Jul-17 golden capture. It is **not** asserted by
  any test and predates this work. Not to be "fixed" or attributed to this refactor.

### Numeric verification of spec equations (θ=1000, leak=0.03)
Verified by direct substitution into the spec's exact segment/crossing equations:
- `g_L = 0.030459`, `r = 0.97`, `κ = 0.984924`.
- `w_2(1) = 515.384`, `w_1 = 1015.307`; `c_basal_weight_init = 520.538`,
  `c_basal_weight_max = 566.923`.
- **Invariant `c_basal_weight_max (566.9) < w_1 (1015.3)` holds.**
- Pretrained L1E crossing from rest: **τ = 0.9517** (spec ~0.952 ✓).
- C 2nd-coincidence crossing at init: **τ = 0.9796** (spec ~0.980 ✓).
- C 2nd-coincidence crossing at cap: **τ = 0.8131** (spec ~0.813 ✓).
- `advance_segment(Δτ=1)` == full-boundary `integrate(dt=1)`: diff **0.0** (exact) for
  sampled (V0, I, g_inh) — confirms the Phase 1 equivalence requirement is satisfiable.
- Note: default `e_weight_cap = θ/2 = 500 < c_basal_weight_max`, which is exactly why
  the spec mandates a **separate** C basal cap. No conflict.

**No unresolved contradiction blocks Phase 1.**

### Implementation map (spec name → live code)

Neurons — `snn/neurons.py`:
- `ExcitatoryNeuron` — conductance LIF; whole-boundary `integrate(dt=1.0)`, `can_fire`,
  `fire`, `update_trace`, `decay_conductance`, `advance_refractory`,
  `update_acc_weights`. **No `tau`/sub-boundary concept today.**
- `SourceNeuron` (rg_source), `InhibitoryNeuron` (stateless relay), `SwitchInterneuron`,
  `PredictiveInterneuron`.
- To ADD: `ConductanceLIFNeuron` (extract membrane base), `DendriticCompartment`,
  `CoincidencePyramidalNeuron`. `InhibitoryNeuron` gains zero-latency relay use via a
  new edge kind (behaviour lives on the edge, not the cell).
- Layout reuse: `backend/layout.py` already generates `L1Enew{i}` positions (no RNG at
  RG/Error). `L1C[i]` reuses `L1Enew[i]` positions → no new RNG draw. Confirmed present.

Graph vocabulary — `backend/network_spec.py`:
- `ARCHETYPES` dict (add `e_pretrained`, `e_coincidence`, `e_latency_competitor`).
- `EDGE_KINDS` dict (add `pretrained_excitation`, `basal_excitation`,
  `apical_excitation`, `hard_reset_inhibition`).
- `validate_spec` (add the new structural invariants + legacy/event-resolved mixing
  rejection). `preset_spec` (add `rg_coincidence`; keep `PRESETS` gated until Phase 5).

Engine — `backend/simulation.py`:
- `SimulationEngine._build_from_spec` builds flat FF banks, registries
  (`self.competitors/encoders/...`, `self.plastic`), execution adjacency
  (`_relayexc_out`, `_inh_out`, etc.), `_ff_weight_ref`. Add: `self.coincidence`,
  `self.latency_competitors`, `self._basal_weight_ref`, basal/apical dispatch maps.
- `step()` is a fixed 8-subphase synchronous loop with `max(V)` WTA (line ~670). The
  event-resolved path (`BoundaryEventScheduler`, sub-boundary `tau`, immediate
  hard-reset relays) is entirely NEW and must branch on archetype/edge metadata so
  legacy presets stay on the byte-compatible path.
- Config: `DEFAULTS`, `EDITABLE_KEYS`, `VALID_TOPOLOGIES`, `_public_params`. Add
  `c_eta`, `c_basal_weight_init/max`, `c_basal_window_steps=1`,
  `pretrained_exc_margin=1.05`, `crossing_time_tolerance=1e-12`.
- `stimulate()` (line ~879) must reject `e_coincidence` targets.
- `set_synapse_weight()` must support basal edges with the C cap.
- `dynamic_state()`/`topology()` add C dynamic fields, `hard_reset_events`,
  `spike_tau`, `latency_ties`; generalize `isinstance(n, ExcitatoryNeuron)` (lines
  ~1056) to the shared base.

Presets/config/API:
- `backend/presets.py` `BUILTINS = PRESETS`; `backend/dashboard_config.py` topology
  select + config controls; `backend/api.py` `_vocabulary()` propagates
  archetypes/edge kinds; `stimulate` endpoint surfaces the C rejection.

Frontend (Phase 6, vocabulary-driven):
- `frontend/editor.js` — `NODE_COLOR`/edge color maps + `_inferKind` (the basal-vs-apical
  ambiguity point: must request explicit kind, not pick first valid).
- `frontend/renderer.js` — edge color constants by kind; add C node + basal/apical/reset.
- `frontend/inspector.js` — per-role/kind fields; add C dendritic state + reset diags.

Tests — `tests/`: `test_conductance_neuron.py` (LIF contract to preserve),
`golden_topology.py` + `test_golden_topology.py` (frames oracle), `test_network_spec.py`
(validation), `test_presets.py`, `test_serialization_api.py`. New focused tests per phase.

Experiments — `experiments/`: add isolated cadence + full frequency-halving drivers
(Phase 7). Do not edit existing `*_results.json`.

---

## Phase 1 — shared LIF + exact segment primitives (complete)

### What changed
- `snn/neurons.py`: extracted `ConductanceLIFNeuron` base (intrinsic membrane only,
  no afferent vector / learning rule). `ExcitatoryNeuron` now subclasses it and keeps
  `acc_weights`/`update_acc_weights`. Legacy `integrate()` preserved verbatim.
- Added event-path primitives on the base: `freeze_drive()`, `advance_segment(dtau)`
  (max-tracking `v_pre_reset`), `crossing_time(remaining_interval)` (analytic, encodes
  refractory/fired validity, `0.0` if already supra, `inf` if unreachable),
  `fire(tau=None)` (records `spike_tau`, consumes `remaining_excitation`, sets
  `fired_this_boundary`), `hard_reset(tau, discard_drive=True)`.
- New fields: `remaining_excitation`, `spike_tau`, `fired_this_boundary`. Legacy path
  never touches them; `fire()` still callable with no arg (byte-compat).
- `snn/__init__.py`: export `ConductanceLIFNeuron`.
- **No `backend/` or `frontend/` change** — legacy network dispatch untouched.

### Tests / results
- Focused: `tests/test_lif_segments.py` — **19 passed** (segment==full-integrate
  equivalence incl. g=0 branch; composition additivity; crossing-time vs trajectory
  substitution; no-crossing/immediate/refractory inf; `fire(tau)`; `hard_reset` clears
  V+drive only; max `v_pre_reset` tracking).
- Legacy contract: `test_conductance_neuron.py` + `test_excitatory_neuron.py` — 28 passed.
- Full suite: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` → **179 passed, 2
  warnings** (pre-existing FastAPI deprecations).
- Golden frames (behavioural invariant): all four **FRAMES OK** — legacy E dynamics
  byte-identical.
- `git diff --check` on phase files: **0** new whitespace diagnostics.

---

## Phase 2 — coincidence dendrite + C-cell local behavior (complete)

### What changed
- `snn/neurons.py`: added `DendriticCompartment` (named basal/apical input compartment;
  plastic weight+distance only when basal) and `CoincidencePyramidalNeuron` over the
  shared base. Added base `begin_event_boundary()` helper (event-path transient reset).
- C cell: one learned basal weight + unweighted Boolean apical gate; one-boundary basal
  eligibility state machine (`resolve_dendrites`); gated additive charge (deposit =
  `w*s` iff (current OR carried basal) AND current apical, exactly once); arrival-order
  invariant; `can_fire`/`crossing_time` gated on `coincidence_active`;
  `update_basal_weight()` = exact spec equation using pre-update weight + causal state;
  **no** `update_acc_weights`/`acc_weights`, **no** apical weight vector.
- `snn/__init__.py`: export `CoincidencePyramidalNeuron`, `DendriticCompartment`.
- **No `backend/`/`frontend/` change** — C cell is isolated; `rg_coincidence` not built.

### Tests / results
- Focused: `tests/test_coincidence_cell.py` — **28 passed**: full truth table (incl.
  basal→apical(t+1) valid, basal→apical(t+2) invalid, apical→basal invalid unless new
  apical, multi-apical→one gate, current+carried→one deposit, consumed-not-reused,
  arrival-order invariance); exact learning equation numeric check + FE=θ−w; no learning
  without spike; pre-update deposit; cap saturation; E-parity of intrinsic dynamics;
  suprathreshold-without-gate can't fire; valid-event-during-refractory charges but
  can't fire/learn; one-spike-per-boundary drive consumption; **calibrated
  two-coincidence cadence** = `[F,T,F,T,F,T,F,T]` from the derived init weight; one cap
  deposit subthreshold; `c_max < w_1`.
- Full suite: **207 passed, 2 warnings** (pre-existing).
- Golden frames: all four **FRAMES OK**.
- New whitespace diagnostics: **0**.

---

## Phase 3 — graph vocabulary, validation, construction (complete)

### What changed
- `backend/network_spec.py`: added archetypes `e_pretrained`, `e_coincidence`,
  `e_latency_competitor` (all `event_resolved=True`) + an `event_resolved` flag on every
  archetype; added edge kinds `pretrained_excitation`, `basal_excitation`,
  `apical_excitation`, `hard_reset_inhibition`; `EVENT_RESOLVED_ARCHETYPES`,
  `LEGACY_E_ARCHETYPES`, `DIRECTED_ONLY_KINDS`. New validation: exactly-one-basal /
  ≥1-apical per C; duplicate basal/apical rejection; directed-only bidirectional
  rejection; hard-reset requires all-E-targets event-resolved; legacy `e_competitor` may
  not mix with event-resolved archetypes/hard-reset.
- `backend/simulation.py`: `_resolve_coincidence_params()` (derives `c_init`/`c_max`/
  `c_eta`/`q_pretrained` from θ+leak; rejects `c_max ≥ w_1`); construction branches for
  the three archetypes (registries `self.coincidence`/`self.latency_competitors`/
  `self.pretrained`; basal φ from geometry; `_basal_weight_ref`); structured adjacency
  maps (`_pretrained_out`/`_basal_out`/`_apical_out`/`_hardreset_out`);
  `self.event_resolved` metadata flag; latency-competitor ff-init as a 3rd RNG pass
  (byte-identical for preset without them); config keys `c_eta`,
  `c_basal_weight_init/max`, `c_basal_window_steps=1`, `pretrained_exc_margin=1.05`,
  `crossing_time_tolerance=1e-12`; generalized `dynamic_state` membrane check to
  `ConductanceLIFNeuron`.
- **No event loop yet** (Phase 4). Legacy `step()` unchanged; C/latency cells are inert
  under it (not in the legacy fire/WTA lists). `rg_coincidence` not added to PRESETS.
- `tests/test_network_spec.py`: deliberate additive vocabulary-contract update (the only
  existing-test edit; sanctioned by the spec's additive-vocabulary clause).

### Tests / results
- Focused: `tests/test_coincidence_spec.py` — **17 passed**: vocab shape; valid graph
  validate+build; idempotent round-trip; deterministic construction; synthetic C+WTA
  fixture dendrites/weights/adjacency/φ; all invalid-graph rejections (missing/multiple
  basal, missing/duplicate apical, bidirectional basal/reset, hard-reset-onto-legacy-E,
  legacy/event mixing); `c_max ≥ w_1` config rejection; legacy presets not event-resolved.
- Full suite: **224 passed, 2 warnings**.
- Golden frames: all four **FRAMES OK** (draw-order + construction changes are
  behaviour-preserving for legacy presets).
- New whitespace diagnostics: **0**.

---

## Phase 4 — event-resolved scheduler + emergent WTA (complete)

### What changed
- `backend/simulation.py`: `BoundaryEventScheduler` (pure latency ordering; recomputes
  candidates each iteration so a hard reset auto-invalidates stale crossings; exact/
  within-tolerance ties fall back to stable node order and are logged, distinguishable
  times are not). `step()` dispatches to `_event_step()` when `self.event_resolved`.
  `_event_step` implements the full boundary contract: rotate delay-1 buffers, deliver
  feedforward/pretrained charge + basal/apical dendritic events, RG external input +
  emission, resolve C gates, freeze drive; sub-boundary loop (earliest crossing →
  `_fire_event_cell` → immediate `_drive_event_relays` hard resets → recompute);
  finalize trace/conductance/refractory. Helpers `_emit_event_outputs`,
  `_drive_event_relays`, `_fire_event_cell`. Diagnostics `hard_reset_events`,
  `latency_ties`, relay `spike_tau`.
- `snn/neurons.py`: `InhibitoryNeuron.spike_tau` (relays inherit driver's tau).
- Legacy synchronous `step()` path unchanged.

### Tests / results
- Focused: `tests/test_event_scheduler.py` — **16 passed**: scheduler picks earliest;
  earliest wins regardless of node order; exact tie → stable order + logged; within- vs
  outside-tolerance; advance-all; hard-reset invalidates stale candidate; engine:
  higher-drive wins + resets loser; reversed drive reverses winner (same node order);
  winner spike+learning survive its own reset; ≤1 L2 spike/boundary; two same-τ resets
  logged; loser drive discarded; no legacy-WTA path; one-boundary delay preserved
  (RG→L1E, no 2-edge hop); winner apical delivered next boundary; legacy graph still
  synchronous.
- Full suite: **240 passed, 2 warnings**.
- Golden frames: all four **FRAMES OK**.
- New whitespace diagnostics: **0**.

---

## Phase 5 — full `rg_coincidence` preset (complete)

### What changed
- `backend/network_spec.py`: `_rg_coincidence_spec` (45 nodes / 196 edges exactly);
  added `rg_coincidence` to `PRESETS` and the `preset_spec` dispatch.
- `backend/simulation.py`: added `rg_coincidence` to `VALID_TOPOLOGIES` (public
  constructor + `apply_config` topology switch).
- `backend/layout.py`: `L1C[i]` placed at a COPY of the historical `L1Enew[i]` position
  — no new RNG draw, so legacy layout/weights/goldens are unshifted.

### Causal verification (public entry point, seed 1, pattern "row 1")
- Node/edge counts exact; edge kinds
  `{pretrained:9, feedforward:72, basal:9, apical:72, relay:17, hard_reset:17}`.
- Each C: 1 paired basal (`L1E{i}`) + 8 apical (all `L2E`); no `acc_weights`/apical weights.
- L1I hard-reset one-to-one; L2I hard-resets all 8 L2E.
- Chain observed: RG(t=1) → pretrained L1E(t=2) → L2E accumulates → first L2E latency
  win (t=8) → C coincidence fires (t=21, only active-pixel cells L1C3/4/5) → each C→L1I
  hard-resets its paired L1E at the C spike's τ; L2I reset τ == winning L2E τ.
- Basal learning runs only on active-pixel C cells (520.54→~522.8 over 40 steps);
  inactive-pixel C cells stay at init. `c_max < w_1` invariant holds.
- Timing sanity (derived, verified live): pretrained L1E crossing τ=0.952; C
  2nd-coincidence at init τ=0.980, at cap τ=0.813.

### Tests / results
- Focused: `tests/test_rg_coincidence.py` — **14 passed**.
- Full suite: **254 passed, 2 warnings**.
- Golden frames: all four legacy presets **FRAMES OK**.
- New whitespace diagnostics: **0**.

---

## Phase 6 — public protocol + frontend support (complete)

### What changed
- `backend/simulation.py`: `_live_weight` serializes live basal weight, `None` for
  apical/hard-reset, fixed magnitude for pretrained. `dynamic_state` adds C fields
  (`basal_weight`, `basal_received/eligible`, `apical_active/sources`,
  `coincidence_active/charge`) + `spike_tau` (guarded on `event_resolved`) + top-level
  `hard_reset_events`/`latency_ties` (empty for legacy). `set_synapse_weight` edits
  basal (clip to C cap); `stimulate` rejects C cells with a clear error.
- `backend/api.py`: `_vocabulary` propagates `event_resolved`; `/api/stimulate` maps the
  C rejection to HTTP 400.
- `backend/dashboard_config.py`: `rg_coincidence` topology option (+ description). No
  new config keys (CONFIG_SPEC key-set unchanged).
- `frontend/editor.js`: node/edge colors + layout maps for the new vocabulary; explicit
  kind selection when an E→e_coincidence gesture is ambiguous (basal vs apical) — never
  silently picks the first. `frontend/renderer.js`: colors for the four new edge kinds
  (basal violet vs apical pink distinction; bright-red hard reset). `frontend/inspector.js`:
  C dendritic-state cards (learned basal weight, Boolean coincidence gate, spike τ).

### Tests / results
- Focused: `tests/test_coincidence_protocol.py` — **14 passed** (edge-weight-by-kind
  serialization; C dynamic fields + diagnostics; hard-reset schema; legacy payloads
  byte-unchanged; basal weight clip; apical/pretrained/reset not editable; C stimulate
  rejected; full-state envelope; deterministic replay; API vocabulary; API runs preset;
  stimulate endpoint → 400).
- Full suite: **268 passed, 2 warnings**. Golden frames: all four **FRAMES OK**.
- JS `node --check`: editor/renderer/inspector all OK.
- Live HTTP smoke (real uvicorn server): config→200, 45/196 topology, stepping grows C
  basal weight, diagnostics present, C-stimulation→400, vocabulary propagates flags,
  restore-pi→200.
- New whitespace diagnostics: **0**.

---

## Phase 7 — scientific validation + documentation (complete)

### What changed
- `experiments/coincidence_experiment.py`: deterministic harness with the required
  counters (RG/L1E/C/L2 spikes, basal/apical events + source ids, valid coincidences,
  L1/L2 hard resets, reset-suppressed L1E, event-resolved spike taus, L2 winner/margin,
  latency ties, L1E frequency first-vs-last window). Drivers: `isolated_cadence`,
  `full_preset`, `winner_exchange`, `deterministic_replay`. Writes
  `experiments/coincidence_results.json`.
- `README.md`: preset list → five; new `rg_coincidence` section + a **measured-behavior**
  block reporting mechanics and the honest ratio separately; vocabulary + tests updated.

### Measured results (seed 1, "row 1" active pixels 3/4/5; NOT tuned)
- **Isolated cadence (exact):** `[0,1,0,1,…]` — one C spike per two valid coincidences.
  Calibrated crossings: pretrained L1E τ=0.9517; C 2nd-coincidence τ=0.9796 (init) /
  0.8131 (cap).
- **Full preset (4000 boundaries):** RG=12000, L1E=9693, C=2358, L2=1583,
  L1-resets=2358 (2304 beat their L1E), valid coincidences=4749, latency-ties=6460
  (perfectly-symmetric pretrained L1E3/4/5 and C3/4/5 tie exactly → resolved by stable
  node order, deterministically). Active C basal weights matured to the cap (566.92);
  inactive stayed at init. mean C spike τ=0.847 **<** mean L1E τ=0.952.
- **Frequency target:** L1E/RG = **0.808**, target 0.5 — **not reached**. First→last
  window L1E rate 0.862→0.800.
- **Winner exchange:** winner follows drive (L2E0↔L2E5) with node order unchanged.
- **Deterministic replay:** identical.

### Scientific vs mechanical (reported separately, per spec)
- **Mechanics: CORRECT.** Branch identity survives delivery; unmatched branch activity
  deposits zero; one-boundary eligibility holds; only the basal weight learns (exact
  equation) on a causal C spike; C intrinsic dynamics equal E; pretrained RG fires L1E
  in one event; analytic crossing times order spikes; immediate relays hard-reset at the
  causal τ; L2 WTA emerges from first-spike latency + the L2I loop; the isolated C cell
  shows the calibrated two-coincidence cadence.
- **Scientific target: NOT MET (honest).** The exact `L1E/RG≈0.5` halving is a property
  of the *isolated* cadence contract, not of the full preset. In the full loop C firing
  is gated by the L2 latency-WTA accumulation cadence (one L2E wins per boundary, needs
  several boundaries to charge), so C — and its paired L1E suppression — fires ~1/5
  boundaries, giving ≈0.81. No hidden constant was tuned to move this. The likely lever
  (not pursued, would change shared dynamics the spec forbids touching) is the L2E
  charging/turnover rate that sets how often apical gates open.

### Tests / results
- Focused: `tests/test_coincidence_experiment.py` — **6 passed** (isolated 2:1 exact;
  calibration lets mature C outrun L1E; full-preset mechanics; ratio measured-not-forced;
  winner follows drive; deterministic replay).
- **Full suite: 274 passed, 2 warnings** (pre-existing FastAPI deprecations).
- Golden frames: all four legacy presets **FRAMES OK** (byte-exact throughout).
- New whitespace diagnostics in changed source: **0**.

---

## Final handoff

### Files changed (and why)
| Path | Change |
| --- | --- |
| `snn/neurons.py` | `ConductanceLIFNeuron` base (segment/crossing/hard-reset/fire(τ)); `DendriticCompartment`; `CoincidencePyramidalNeuron`; `InhibitoryNeuron.spike_tau`. |
| `snn/__init__.py` | Export the new classes. |
| `backend/network_spec.py` | 3 archetypes + 4 edge kinds + `event_resolved` flag; new validation; `rg_coincidence` preset (45/196). |
| `backend/simulation.py` | Construction of C/latency/pretrained cells + adjacency/refs; `_resolve_coincidence_params`; `BoundaryEventScheduler` + `_event_step` (dispatched by `event_resolved`); config keys; serialization (C fields, `spike_tau`, `hard_reset_events`, `latency_ties`, basal weight edit, C-stimulate rejection). |
| `backend/layout.py` | `L1C[i]` = copy of `L1Enew[i]` (no new RNG). |
| `backend/dashboard_config.py` | `rg_coincidence` topology option. |
| `backend/api.py` | Vocabulary `event_resolved`; `/api/stimulate` → 400 for C cells. |
| `frontend/editor.js`,`renderer.js`,`inspector.js` | Colors/layout for new vocab; explicit basal-vs-apical kind selection; C inspector state. |
| `tests/test_network_spec.py` | Deliberate additive vocabulary-contract update. |
| new `tests/test_{lif_segments,coincidence_cell,coincidence_spec,event_scheduler,rg_coincidence,coincidence_protocol,coincidence_experiment}.py` | Per-phase focused suites. |
| new `experiments/coincidence_experiment.py` + `docs/COINCIDENCE_IMPLEMENTATION_STATUS.md`, `README.md` | Measurements + docs. |

### Test commands
- Full: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` → **274 passed**.
- Goldens: `PYTHONPATH=. .venv/bin/python tests/golden_topology.py verify <name>_baseline <topo>` → FRAMES OK (pi/old/rg/rg_residual).
- Experiment: `PYTHONPATH=. .venv/bin/python experiments/coincidence_experiment.py`.

### Deviations from the spec
- None in mechanics. The **scientific** frequency-halving target (`L1E/RG≈0.5`) is not
  achieved in the full preset (measured ≈0.81); reported honestly, mechanics preserved.
- `spike_tau` and the C dynamic fields are emitted only for event-resolved graphs, and
  `hard_reset_events`/`latency_ties` are empty for legacy graphs — additive and
  backward-compatible; legacy dynamic payloads and goldens are byte-unchanged.

### Compatibility / remaining risk
- Legacy presets: byte-identical goldens; legacy `step()` path untouched.
- Perfectly-symmetric pretrained L1E / active C cells tie exactly every boundary;
  resolved deterministically by stable node order (auditable via `latency_ties`).
- The RF panel omits C cells (spec-permitted); node color is by type (C shares E color) —
  the basal/apical **edge** distinction is the implemented visual separation.

---

### Legacy vs event-resolved branch point
Legacy presets (`pi`/`old`/`rg`/`rg_residual`) contain none of the new archetypes or
edge kinds → they take the existing synchronous `step()` path unchanged (goldens
protected). `event_resolved` execution is selected from archetype/edge metadata
(never node IDs or preset name), per spec "Engine build changes" §9.
