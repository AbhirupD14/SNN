# Agent Handoff — SNN Cortical Column, `feature/inhibitory-plasticity`

> **Architecture update — L2 hard-reset competitive depression (2026-07-13).**
> The learned negative `L2I -> L2E` gate is **gone** from the active engine. L2I
> recruitment is still learned on its positive `L2E -> L2I` inputs, but its output
> is now an **unweighted competitive-reset event**: when L2I fires, every
> non-winner L2E is unconditionally hard-reset to rest (traces cleared) and its
> participating positive feedforward weights are locally depressed via the shared
> bounded kernel, scaled by the loser's own pre-reset charge. No learned
> `L2I->L2E` magnitude exists on the active path; the `inhibitory_delta_rule` /
> turnover / `L2_GATE_*` machinery below applies only to the legacy standalone
> `apply_inhibition` path (`L1I->L1E` feedback, old experiments). Active L2E have
> exactly `N_PIX` positive afferents and no index-0 gate. Multi-seed diagnostic:
> `report_competitive_depression.py` (competitive depression roughly doubles
> sustained per-pattern dominance, 0.63 -> 0.85, but reduces distinct winners /
> raises dead-unit count — it does **not** on its own deliver clean 8/8 one-to-one
> ownership). See `L2_Hard_Reset_Competitive_Depression_Spec.md`.

Read this top-to-bottom before touching the competition code. It tells you where
the implementation is, what the real goal is, what has been tried (and rejected,
with evidence), and what to try next. It reflects the state as of this branch.

---

## 1. What this project is

An overcomplete four-pattern recognition SNN. Input is a 3x3 pixel grid
(`N_PIX=9`); the retained patterns are the center row, center column, and two
diagonals. Architecture: `L1E` pixel encoders -> `L2E` (8 competing
excitatory units) with a shared `L2I` inhibitory neuron providing lateral
inhibition; `L1I` gives feedback suppression. The whole engine is
`backend/simulation.py::SimulationEngine`; the single neuron implementation is
**`neuron_flexible.py`**. It supports both fixed fan-in (`Neuron(n_inputs=...)`)
and staged fan-in (`add_input_connection()` / `finalize_connections()`).

Fixed-point convention: potentials/thresholds/weights run at `UNIT=1000` scale.
`threshold_l2 = 8*UNIT = 8000`.

---

## 2. The goal and the ONLY honest metric

**Goal = true one-to-one tiling:** show a trained pattern → one specific L2E
neuron lights up **and holds** while the pattern is present; show another pattern
→ its own neuron; re-show the first → the same neuron again. Stable, repeatable,
per-pattern ownership.

**Honest metric = SUSTAINED-presentation dominance.** Hold each pattern for ~40
cycles and measure what fraction of cycles its modal specialist wins. A perfect
1-to-1 map = dominance ≈ 1.0 with 8/8 distinct specialists.

**DO NOT trust `metrics_consolidation.py`'s "8/8 dominance 1.0".** That is a
MEASUREMENT ARTIFACT: it presents each pattern for one short window per sweep and
takes the modal winner over sweeps, which only captures the specialist's reliable
*first-cycle-after-switch* win and never sees the round-robin that follows. Under
sustained presentation the pool ROUND-ROBINS.

**Current best: sustained dominance ≈ 0.35–0.36 with 7–8/8 distinct.** i.e. the
correct specialist is the clear plurality winner per pattern, but rivals still
take ~60% of cycles. True 1-to-1 is NOT solved.

---

## 3. Current code state (defaults + where things live)

**DEFAULT REGIME CHANGED (2026-07-08):** the L2E learning rule is now the
**minimal signed-spike rule with NO weight budget** —
`signed_spike_learning=True`, `l2e_budget=False` by default. On fire, every
positive feedforward synapse updates by `dw = eta·p·(1−(w/w_cap)²)·signal`,
`signal=+1` if its input participated else `−1`; the `−1` supplies the downward
pressure the budget used to. It `return`s early in `_update_weights`, so
**`confidence_consolidation`, `loser_depression`, `signed_depression` and the
budget are all bypassed for L2E** regardless of their own defaults (left `True`
but inert). The pre-2026-07-08 budget/charge regime below can be reconstructed
with `signed_spike_learning=False, l2e_budget=True`. Honest metric under the new
default (via `ablation_harness.py`, 3 seeds): sustained dominance ≈0.55, distinct
≈4/8 with high seed variance (±0.8), no stable ownership — the budget was masking
seed dependence. The vector-init ablation is now de-scoped as the next lever:
do **not** adopt sparse init as the standing default, and do **not** use
`membrane_noise` as the symmetry breaker. The next planned lever is deterministic
distance-weighted signal attenuation (see
`Input_Vector_Initialization_And_Distance_Weighting.md`).

Defaults in `SimulationEngine.__init__` (backend/simulation.py):

| feature | default | notes |
|---|---|---|
| `signed_spike_learning` | **True** | canonical L2E rule; takes over `_update_weights`, bypasses the four below |
| `l2e_budget` | **False** | budget off; inert under signed-spike anyway |
| `confidence_consolidation` | True | bypassed under signed-spike (was: keeps specialists distinct) |
| `loser_depression` | True | bypassed under signed-spike (was: decorrelation) |
| `signed_depression` (4a) | True | bypassed under signed-spike (was: OFF-pixel gate depression) |
| `eta_off` | 0.20 | depression rate; only relevant in the old budget regime |
| `event_driven` | **True** | canonical per-step competition: resolve one argmax winner every timestep (was False = once-per-cycle; still reachable by turning off) |
| `l2_charge_chunks` | **1** | K: deliver this step's L1→L2E drive in K chunks (w/K per synapse) inside a frozen timestep, resolving the argmax WTA after each and stopping at the first crosser (consolidation-first). K=1 = un-chunked baseline |
| `input_period` | **1** | Constant held-pattern drive; independent `cycle_period` still defaults to `volley_period`. |
| `l1i_immediate_relay` | **False** | Default is the trainable threshold accumulator. Its spikes inhibit paired L1E one step later; shared random initialization, temporal contributor credit, and an effective one-step refractory interval produce synchronized half-frequency feedback. On enables the deterministic relay. |
| `excitatory_flow_rate` | **False** | weight = current amplitude: a spike opens a decaying excitatory current trace integrated into V over time (closed-form lazy advance) for L2E/L2I/L1I (not L1E, not relay L1I). Off = instantaneous `V += dot(w,spikes)`. Forces effective `l2_charge_chunks`=1. `exc_trace_decay`=0.8 (d), `exc_trace_normalized`=True (inject `g(1-d)` so total≈g) |
| `inhibitory_delta_rule` | **True** | differentiating L2I→L2E gate rule instead of legacy saturating (all gates → sqrt(w_max), uniform). `inhibitory_rule_mode`="turnover" (default): `du=eta_up·p_t·(1−u) − eta_down·u`, `u=w/G`, `G=sqrt(w_max)`, `p_t=clamp(v_pre/θ,0,p_max)` — event-local, no target voltage/averages; high-charge rivals accumulate stronger gates, weak ones decay (spread ~260 vs 4, distinct winners preserved). "margin" mode = diagnostic (`s=clamp(v_pre−margin·θ,0,G)`). Params `inhibitory_eta_up`=0.02, `inhibitory_eta_down`=0.005, `inhibitory_p_max`=1.0 |
| `lasting_inhibition` | **False** | experimental scalar field; FAILED — leave off |
| `homeostasis` | False | Turrigiano scaling; off by default |
| `refractory` | 2 | irrelevant to the round-robin (see §5) |

Key mechanisms & locations:
- **Competition**: `SimulationEngine.step()`, section "2b/2c" + helper
  `_resolve_l2_competition` — EVERY step (default, `event_driven`),
  `winner = argmax(potential among eligible threshold-crossers)`, fires; `L2I` fires
  and one-shot partial-discharges the rest via `apply_inhibition` (all L2E except the
  winner, not just co-crossers). This step's feedforward drive arrives in
  `l2_charge_chunks`=K chunks inside the frozen timestep, resolving after each and
  stopping at the first crosser (K=1 = un-chunked). Turning `event_driven` off
  resolves the same argmax competition only at `cycle_boundary`; `lasting_inhibition`
  is a separate alternate mechanism in the same block (off).
- **Excitatory learning** (`neuron_flexible.py::_update_weights`): participation-
  gated charge rule `dw = eta·p·(1 − w²/w_max)`, p = clamp(theta/v_pre,0,1); plus
  the signed-depression OFF-gate branch; then `_apply_budget_and_cap` (renorm sum
  to weight_budget, floor at `min_positive_weight`, clip).
- **Inhibitory gate learning** (`apply_inhibition`): `dw = eta·p·(1 − w²/w_max)`,
  gate saturates at `√L2_GATE_WMAX = 1225`.
- **Confidence** (`_decay_confidence`, `_update_weights`): maturity, eta-gating,
  activity-gated decay. `confidence_init` seeds the ACTIVE `_confidence` array.

Dashboard (`backend/api.py`): FastAPI + websocket + static `frontend/`. Live
config via `/api/config` (`apply_config` rebuilds; `CONFIG_SPEC` drives the
"Model Config" slider panel). Auto-cycle via `/api/autocycle`
(`set_auto_cycle`) — visit-based curriculum that rotates patterns and marks one
"trained" when its per-visit winner is stable across rounds; self-disables at 8/8.

New this branch: signed depression, 2× budget, `(w/w_cap)²`, event_driven &
lasting_inhibition flags, live config panel, auto-cycle. Removed: dead
confidence-beta/gamma params and `benchmark_trace_modes.py`.

---

## 4. What worked / what didn't (evidence)

WORKED (keep):
- **Weight budget** — load-bearing. OFF ⇒ collapse (distinct→~4, dead units).
- **Confidence consolidation + loser depression** — keeps specialists distinct.
  Stripping to "minimal" (budget+Hebbian) raises dominance but collapses distinct
  7→3. NOT overengineering.
- **Signed depression (4a)** — the RF-margin lever. Sharpens RFs, lifts sustained
  dominance 0.23→0.39, holds 8/8 distinct.
- **2× budget** — clean 8/8 distinct, 0 dead (at eta_off≈0.5), no dominance cost.

DIDN'T WORK (don't re-try without a new idea):
- **Lasting inhibition (scalar decaying field)** — pattern-blind; collapses
  distinct to 2–4 whether applied in training or readout. Flag left off.
- **Event-driven firing** — bounds membrane (overshoot 2.79→1.62×) but collapses
  tiling 8→5–6. Flag left off.
- **Raising the L2I gate cap / stronger inhibition** — collapses tiling 8→5,
  dom 1.0→0.40 (recreates single-winner collapse).
- **Raising `leak_l2`** — cliff: ≥0.05 kills the pool (all dead).
- **`(w/w_cap)²` saturation change + wider init** — INERT for the margin; the
  budget caps the weight SUM so per-gate saturation & init magnitude wash out.
- **Stronger loser depression** — flat (~0.35) across eta_loss.

---

## 5. The core open problem (why 1-to-1 is unsolved)

Two coupled causes of the round-robin, both measured:

**(a) Discharge asymmetry.** On fire, the winner resets fully to rest, but losers
only get a partial inhibitory discharge (~1225). So losers keep most of their
charge, ratchet up over cycles to ~2× threshold (~13k–16k), and overtake the
freshly-zeroed winner next cycle. Refractory is NOT the cause (refractory=2 <
cycle_period=4, empirically flat across 0–3).

**(b) The gate is capped and scale-blind.** The L2I→L2E gate climbs 500→**1225**
then saturates (√L2_GATE_WMAX), = 0.15× threshold. It cannot bite against a
13k–16k membrane. The cap is DELIBERATE (`L2_GATE_WMAX < thr_l2` so a saturated
gate can't hard-reset → avoids single-winner collapse). Also `p =
clamp(v_pre/theta,0,1)` saturates at 1, so the gate grows no faster onto big
overshooters. Raising the cap collapses tiling (tested).

**The frontier:** margin (dominance) vs distinctness. Tight budget + strong
depression → dom 0.58 but distinct→4; loose budget → 8/8 distinct but dom 0.35.
The same concentration that sharpens one RF (margin) lets a neuron grab multiple
patterns (kills distinct) unless a decorrelation force holds it — and loser
depression, the available local decorrelator, doesn't move the margin.

**Constraint from the project owner:** learning/competition rules must stay
LOCAL. Per-target lasting inhibition (winner suppresses its *specific* rivals)
was explicitly REJECTED as non-local. No cross-neuron rival lookup.

---

## 6. Next candidate experiments (ranked, all local)

1. **Distance-weighted signal attenuation** — keep uniform feedforward
   initialization and `membrane_noise=0.0`, then deliver each synaptic event as
   `w / d²` using deterministic functional positions. L2E positions should be a
   compact lateral layer with small seed-deterministic jitter so L1E→L2E
   distances vary without becoming random spatial slop. First test feedforward
   L1E→L2E only; then test the competition-critical scope where L2I→L2E
   inhibitory discharge is also attenuated by distance. Learning updates stored
   weights normally in the first pass; distance changes delivered charge, not
   the weight update equation.
2. **Reset-by-subtraction in `fire()`** — `potential -= threshold` instead of
   `→ rest`. Standard LIF; leaves the winner its residual overshoot like the
   losers keep theirs, directly attacking the discharge asymmetry (a). Fully
   local. Build behind a flag, measure sustained dominance.
3. **Unclamp `p`** in the gate rule — let `p = v_pre/theta` exceed 1 so the gate
   grows MORE onto the worst overshooters (differentiates habitual runners-up).
   Local; does not raise the hard cap. Measure whether it differentiates gates
   without collapse.
4. **Bound loser accumulation** so the membrane can't ratchet to 2× threshold
   (makes the fixed-cap gate proportionally meaningful). Careful not to recreate
   the leak cliff.

Always A/B behind a flag (default off), measure with SUSTAINED dominance +
distinct + dead across ≥3 seeds. For distance experiments also report attenuation
matrix min/mean/max, firers per held pattern, L2I spike/discharge counts, and
row/column/diagonal breakdown. Do NOT optimize the artifact metric.

---

## 7. How to run / measure

- Dashboard: `PYTHONPATH=. .venv/bin/python -m uvicorn backend.api:app --reload`
  then open the served page (hard-refresh after frontend edits — ES modules cache).
- Tests (plain scripts, no pytest): `PYTHONPATH=. .venv/bin/python test_*.py`.
  `test_8line_consolidation.py` is a *characterization* test that expects
  single-winner collapse without a counter-force — that's intended.
- Sustained-dominance harness: not committed as a file; the pattern is: build
  engine → interleaved-train 60 epochs (each pattern `cycle_period` steps) →
  for each pattern, hold `cycle_period`×40 steps, record per-cycle argmax winner,
  take modal fraction = dominance, count distinct modal winners. (See the memory
  note `one-to-one-and-lasting-inhibition` for exact numbers.)

---

## 8. File map

- `backend/simulation.py` — engine, competition, all defaults & flags, auto-cycle,
  `apply_config`, module constants (`L2_GATE_WMAX`, `L2E_BUDGET_MULT`, etc.).
- `neuron_flexible.py` — the single neuron class and fixed-point constants.
- `backend/api.py` — REST/ws, `CONFIG_SPEC`, `/api/config`, `/api/autocycle`.
- `frontend/{index.html,controls.js,style.css}` — dashboard + Model Config +
  auto-cycle panels.
- `metrics_consolidation.py` — the ARTIFACT metric; useful for distinct-winner
  counts but NOT for dominance (see §2).
