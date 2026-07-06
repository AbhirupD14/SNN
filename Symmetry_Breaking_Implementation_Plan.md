# Symmetry Breaking Implementation Plan

**Audience:** implementation agent. Execute this plan narrowly. Do not reread the
whole repository. Do not use winner facilitation as a solution.

## STATUS — executed and evaluated (2026-07-06, facilitation OFF throughout)

All phases implemented; every experiment forces `facil_boost=0.0`. All new params
default to prior behavior (existing tests pass); new tests/harnesses added:
`test_refractory_gating.py` (Phase 4), `test_symmetry_breaking_metrics.py` (Phase 5),
`sweep_symmetry_breaking.py` + `sweep_symmetry_breaking_stage4.py` (Phases 3/6).

**Findings, per lever:**

- **Phase 2 (lower inhibitory threshold): a NO-OP by construction.** Because the
  plan (correctly) requires the E→I init range AND learning cap to scale with the
  inhibitory neuron's own threshold, the round-robin dynamics are scale-invariant:
  `l2i_threshold_frac ∈ {1.0, 2/3, 1/2, 1/3}` gave *identical* A/B/rf_cos. Lowering
  the threshold without that scaling would just overdrive inhibition (collapse) —
  so there is no useful setting. Drop this lever.
- **Phase 1 (separate E/I learning rates): marginal.** Higher `l2e_lr_frac` (0.02)
  slightly improves RF diversity and A; `l2i_lr_frac` has little effect. Not a
  symmetry-breaker on its own.
- **Phase 3 (homeostasis): the real driver — and specifically DOWN-ONLY.** With
  homeostasis OFF the network collapses (A=0.38, B=3.0, rf_cos=0.91, 5 dead units).
  Turning on **down-only** homeostasis (`homeo_up=0.0`, `homeo_down=0.05`) is the
  single biggest win: **A 0.30→0.57, rf_cos 0.80→0.56 (far more diverse fields),
  reserve≈0, no collapse** — with NO facilitation. Adding `homeo_up` (silent-unit
  recruitment) HURTS RF diversity (rf_cos 0.80), so anti-tyranny alone is better.

**Acceptance vs. the criteria (5 seeds unless noted):** distinct winners reach the
≥6/8 bar only marginally and only under aggressive settings (`homeo_up=0, down=0.08,
noise=0.1` → B=6.0, A=0.52); most down-only configs give B≈5.0. Within-pattern
dominance clearly improves without winner inertia (0.30→0.57) ✓; RF cosine diversity
clearly improves (0.80→0.56) ✓; no single-winner collapse ✓; refractory
negative-weight behavior verified correct ✓. **Held-out recruitment (Phase 3
follow-up) is a NEGATIVE result:** the pool spreads across ~all 8 units even for 6
trained patterns (reserve≈0), so introducing 2 new patterns is absorbed by
*remapping*, not spare units (held-out-taken-by-reserve=0.0; only ~44% of original
winners survive). The network does not maintain reserve capacity.

**Bottom line:** the biologically-cleaner route delivers a real *representation-quality*
win (dominance + RF diversity) via one change — **down-only homeostasis
(`homeo_up=0.0`)** — but does not, without facilitation, robustly reach 1:1 tiling
(B≥6/8) and does not yield reserve/continual-learning capacity. The
inhibitory-threshold lever is inert; separate learning rates are marginal. Defaults
were left unchanged (no lever hard-coded); adopt `homeo_up=0.0` if the goal is RF
diversity + stability rather than strict tiling.

## Goal

Improve L2 pattern assignment and stability using biologically cleaner mechanisms:

- separate learning-rate controls for E and I populations
- lower inhibitory-neuron firing thresholds
- explicit homeostatic-scaling ablations
- tests confirming inhibitory gates do not update during refractory

The previous post-fire winner-facilitation boost is diagnostic only. It must stay
disabled in every experiment in this plan.

## Hard Guardrails

- Do not use winner facilitation.
- Every new experiment and metrics script must pass `facil_boost=0.0` explicitly.
- Do not include `facil_boost` in any sweep.
- Do not change `apply_inhibition` or `_update_weights`, except to add tests around
  their current behavior.
- Do not add global labels, supervised assignment, hard-coded pattern ownership, or
  procedural winner resets.
- Keep all new parameters defaulting to current behavior.
- Preserve locality: all learning decisions must use only neuron-local or
  synapse-local state.

## Token-Efficient Workflow

Use small Git and symbol queries first:

```bash
git status --short --branch
git diff --stat
rg -n "facil_|effective_potential|ETA_FRAC|learning_rate|inhibitory_learning_rate|threshold_l2|homeostasis|apply_inhibition|refractory_timer|membrane_noise" neuron.py neuron_flexible.py backend/simulation.py test_neuron.py
```

Then inspect only targeted hunks or functions:

```bash
git diff -- neuron.py neuron_flexible.py backend/simulation.py
sed -n '<small range>' backend/simulation.py
sed -n '<small range>' neuron.py
sed -n '<small range>' neuron_flexible.py
```

Do not read broad docs or full source files unless a targeted query cannot answer
the question.

## Phase 0 - Ensure Facilitation Is Off

Audit the current facilitation implementation if present.

Required state:

- `SimulationEngine(...).params["facil_boost"] == 0.0` by default.
- L2E neurons have `facil_boost == 0.0` by default.
- New metrics/sweep scripts force `facil_boost=0.0`.
- Old facilitation scripts, if retained, must be labeled experimental and must not
  be used for acceptance.

In new metrics helpers, reject accidental facilitation:

```python
assert kwargs.get("facil_boost", 0.0) == 0.0
```

## Phase 1 - Separate E/I Learning Rates

Current suspicion: the same effective learning-rate fraction is being applied too
broadly across E and I populations.

Add engine parameters, defaulting to current behavior:

```python
l2e_lr_frac = ETA_FRAC
l2i_lr_frac = ETA_FRAC
l1i_lr_frac = ETA_FRAC
l2_gate_eta = L2_GATE_ETA
```

Interpretation:

- `l2e_lr_frac`: L2E feedforward positive-weight learning.
- `l2i_lr_frac`: L2I incoming E-to-I positive-weight learning.
- `l1i_lr_frac`: L1I incoming feedback positive-weight learning.
- `l2_gate_eta`: inhibitory gate plasticity on L2I-to-L2E negative weights.

Initial sweep:

```text
l2e_lr_frac: 0.005, 0.01, 0.02
l2i_lr_frac: 0.0025, 0.005, 0.01
l2_gate_eta: keep 0.10 first; later sweep 0.05, 0.10, 0.20 only on top candidates
```

Hypothesis: slower I-neuron maturation may prevent early global suppression
patterns from locking before E receptive fields differentiate.

## Phase 2 - Lower Inhibitory Thresholds

Add separate inhibitory threshold fractions, defaulting to current behavior:

```python
l2i_threshold_frac = 1.0
l1i_threshold_frac = 1.0
```

Compute:

```python
thr_l2i = threshold_l2 * l2i_threshold_frac
thr_l1i = threshold * l1i_threshold_frac
```

Sweep L2I first while holding L1I fixed:

```text
l2i_threshold_frac: 1.0, 2/3, 1/2, 1/3
l1i_threshold_frac: 1.0
```

Then, only if useful, test L1I:

```text
l1i_threshold_frac: 2/3, 1/2
l2i_threshold_frac: best from first sweep
```

Important implementation detail: when an inhibitory neuron threshold changes, its
incoming E-to-I initialization range and learning cap must scale with that
inhibitory neuron's own threshold. Do not keep using the E threshold for those
synapses, or lowering the I threshold will trivially overdrive inhibition.

## Phase 3 - Homeostasis Ablation

Do not assume homeostatic scaling is beneficial. Test it explicitly.

Conditions:

```text
homeostasis=False
homeostasis=True, homeo_up=0.0,  homeo_down=0.01
homeostasis=True, homeo_up=0.0,  homeo_down=0.05
homeostasis=True, homeo_up=0.01, homeo_down=0.01
```

Report reserve capacity:

- number of L2E neurons that never fired
- homeostatic budgets per L2E
- distinct winners
- whether old winners are disrupted
- whether unused neurons remain available

Optional follow-up: train on 6 patterns, hold out 2, then introduce the held-out
patterns. Measure whether unused neurons can pick them up without forced
recruitment.

## Phase 4 - Refractory-Gated Negative-Weight Tests

Add or verify tests proving negative weights on E neurons update only on real
inhibitory discharge into a non-refractory neuron.

Required assertions:

- `apply_inhibition()` returns `[]` during refractory.
- potential is unchanged during refractory.
- negative weight is unchanged during refractory.
- flexible neuron has identical behavior.
- in-engine: if an L2E is refractory when L2I fires, its L2I-to-L2E gate does not
  update.

This preserves the principle that inhibitory plasticity updates only when
inhibition actually reduces charge in an active target.

## Phase 5 - Metrics Script

Create `test_symmetry_breaking_metrics.py`.

It must force:

```python
facil_boost=0.0
```

Report:

- A: within-pattern dominance
- B: distinct dominant winners out of 8
- receptive-field diversity: pairwise cosine among L2E feedforward vectors
- receptive-field concentration: HHI or entropy
- reserve capacity: never-fired L2E count
- L2I spike count
- L2 gate update count

Do not use only winner IDs. Winner IDs can improve while representations remain
poor.

## Phase 6 - Staged Sweeps

Avoid one giant grid.

Stage 1: inhibitory threshold only.

```text
l2i_threshold_frac: 1.0, 2/3, 1/2, 1/3
membrane_noise: 0.0, 0.1
homeostasis: False
facil_boost: 0.0
```

Stage 2: learning-rate ratios on top two threshold settings.

```text
l2e_lr_frac: 0.005, 0.01, 0.02
l2i_lr_frac: 0.0025, 0.005, 0.01
facil_boost: 0.0
```

Stage 3: homeostasis ablation on top candidates.

Stage 4: small joint sweep using only top candidates.

## Acceptance Criteria

A candidate is architecturally interesting only if:

- `facil_boost == 0.0`
- distinct winners >= 6/8 averaged over seeds
- within-pattern dominance improves over baseline without winner inertia
- receptive-field cosine diversity improves
- no single-winner collapse occurs
- forced recruitment is not required unless the data clearly justifies it
- refractory negative-weight behavior remains correct

A candidate is not accepted merely because A/B metrics improve. It must improve
them without violating local computation or hiding assignment in a persistent
post-fire boost.
