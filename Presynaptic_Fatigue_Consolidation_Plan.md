# Plan: Emergent Inverse-Frequency Weighting via Source Spike-Frequency Adaptation

Status: design agreed, not yet implemented. Branch `feature/inhibitory-plasticity`.
Keep edits local and unsupervised; no labels, no match/vigilance gating, no
frequency estimator on the target gates.

## Problem being solved

The network tiles the 8 line primitives 1:1 under **interleaved** training (8/8
distinct winners, dominance 1.0), but **fails under blocked/sequential** training
(0/8 preservation, catastrophic forgetting). Root cause is geometric: any two of
the 8 lines intersect in **at most one pixel**, and today that shared pixel is
weighted as strongly as the discriminative ones. So a new pattern (e.g. `col0`)
fires a prior specialist (`row0`) through the single shared pixel and re-carves
it. See `metrics_consolidation.py` and the memory note
`confidence-consolidation-result`.

Pixel frequencies across the 8 patterns: **center (4): 4 patterns**, corners
(0,2,6,8): 3 each, edges (1,3,5,7): 2 each. Common pixels carry less information
and should end up **weighted less**; the budget should spread **unequally toward
rare/informative pixels**.

## Approaches considered and rejected

- **Match / vigilance gating** (compare current input to the neuron's whole RF,
  discount mature-but-mismatched winners): effective but rejected as too
  supervised — it inspects all gates to judge "direction."
- **Per-synapse frequency trace on the target** (`f_i`, weight `∝ 1/f_i`):
  rejected as non-local. Synapse state lives on the *postsynaptic* neuron here, so
  a per-input frequency estimate means the target "knows" previous-layer
  statistics it can't observe — a frequency budget pushed onto the target gates.
- **Presynaptic release depression (charge-per-spike)**: rejected as volatile.
  The target rule is *participation-gated* (`dw` for any line with
  `_last_input_spikes > 0.5`, independent of delivered charge), so reducing charge
  per spike consolidates *nothing* — it only attenuates transiently at inference
  and evaporates on recovery.

## Chosen mechanism: source-side firing throttle (spike-frequency adaptation)

Put a **self-fatigue** variable on each `L1E_i` encoder, driven **only by its own
firing** (same locality class as the existing `ca` calcium sensor — a neuron
reading its own recent activity). Chronic firing → an adaptation/AHP-like term
that **reduces how often the encoder fires**, recovering slowly.

Why this and not the others:
- **Local (strict sense):** reads only the source's own spikes. Nothing crosses
  layers; the target's gates stay pure weights.
- **Emergent, not an estimator:** no frequency is computed and divided. `L1E_i`
  just gets "tired" from overuse; because `L1E_i` is a dedicated pixel encoder,
  its own firing rate *is* pixel `i`'s frequency. Common pixels → busy encoders →
  throttled → transmit less; rare pixels stay potent. Inverse-frequency weighting
  falls out. (This is sensory adaptation: "ignore the predictable.")
- **Dual of the existing budget:** the postsynaptic weight budget is a finite
  resource shared across a neuron's afferents (spatial competition); source
  fatigue is a finite resource across a terminal's spikes over time (temporal
  competition).

### What carries the effect (persistence)

Fatigue itself is **volatile activity-state** on the source and decays with its
recovery time constant — it carries nothing durable alone. Persistence comes from
**the target's own weights**, sculpted indirectly:

> fatigued encoder fires **less often** → participates in fewer of the target's
> fire events → the target's *participation-gated* rule potentiates that input
> **less often** → the target's weight on the common pixel ends up **smaller**.

So fatigue is a transient teaching signal; the durable record is the L2E
receptive field. This is why the throttle must reduce **firing** (couples to the
participation-gated rule), not charge-per-spike (would not consolidate). No
frequency budget is imposed on the gates — the gate simply learns less about an
input that now shows up less.

## Honest limitations (do not oversell)

- "Informative vs common" is inherently a **cross-pattern** statistic. In a
  single-pattern block all active pixels fire equally → no signal within the
  block. Blocked-training benefit depends entirely on the fatigue state
  **persisting across pattern switches** (slow recovery). The **first pattern**
  trained is at a cold start (no prior fatigue) and remains partly hijackable.
- Therefore this is expected to **sharpen interleaved tiling and improve blocked
  training**, but is **not** claimed to deliver full order-invariance by itself.
- Mild information loss: a pattern distinguishable *only* by a common pixel would
  suffer. Not the case for the 8 lines (each has lower-frequency pixels; lines
  overlap in ≤1 pixel), so acceptable here — note we are partly exploiting task
  structure.

## Open question to resolve next: interaction with the refractory period

The refractory period is already a hard, fast "don't fire too much" limiter, and
SFA is a slow, graded version of the same idea — so they must be designed to
compose rather than fight. Seed thoughts for that discussion:

- **Timescale separation is essential.** Refractory acts within a volley
  (`refractory=2`, `volley_period=4`), so `L1E` already fires ~once per volley.
  SFA must operate on a **much slower** timescale (across volleys / across pattern
  presentations) to encode *frequency*, not the within-volley refractory rhythm.
  If SFA's timescale collapses toward the refractory timescale it will just track
  the volley clock and carry no frequency information.
- **Where SFA acts vs. where refractory acts.** Refractory hard-blocks firing for
  a fixed count after a spike; SFA should raise the *effective threshold* / add a
  slow hyperpolarizing term so a chronically-driven encoder *sometimes skips a
  volley it would otherwise have won*. Question: does SFA raise threshold, extend
  an effective refractory window, or subtract a slow current? Each composes with
  the existing refractory gate differently.
- **Does refractory already do part of the job?** Because `L1E` is refractory-
  capped at ~one spike/volley, a common pixel is *not* already firing more per
  volley than a rare one — both fire once when present. The frequency difference
  is purely in *how many patterns* they appear in, i.e. across-presentation
  firing counts. So SFA must integrate over presentations (slow `ca`-like EMA),
  and refractory does **not** already provide the frequency signal.
- **Downstream refractory (L2E) interaction.** If a throttled encoder skips
  firing, L2E integrates less charge that volley; check this does not merely delay
  L2E firing into a later volley (recovering the same charge) and thus wash out
  the effect — the intrinsic cycle clock (`cycle_period`) and L2E leak (`leak_l2`)
  matter here.
- **Consistency with fixed-point scale + existing knobs.** SFA state is
  dimensionless (a firing-rate-like quantity), so it does not need `UNIT` scaling;
  any threshold offset it induces on `L1E` *is* charge-scaled (`× UNIT`). Confirm
  it does not interfere with `homeostasis=False`, the confidence-consolidation
  flags, or the round-robin L1I/L2I integrator dynamics.

## Related refactor: reparameterize `w²/w_max` → `(w/w_cap)²`

Independent of the fatigue mechanism, but worth doing first/alongside because it
cleans up the exact `w_max` scaling the fatigue work sits next to.

**It is the same curve, just re-parameterized.** Both forms are the identical
downward parabola `1 − (w/w*)²`; only the meaning of the parameter changes:

- current `1 − w²/w_max`: zero-growth equilibrium at `w* = √w_max`, so `w_max`
  means *the square of the settle point*;
- proposed `1 − (w/w_cap)²`: equilibrium at `w* = w_cap`, so `w_cap` means *the
  settle point itself*.

**Behavior is bit-for-bit identical IF the constants are re-derived** as
`w_cap = √(old w_max) = old equilibrium`. Migration:

| synapse | current `w_max` | new `w_cap = √(old)` |
|---|---|---|
| L2I→L2E gate (`inhibitory_weight_cap`) | `1.5 · UNIT²` (1.5e6) | `UNIT·√1.5 ≈ 1225` |
| E→I (`excitatory_saturation_cap = weight_cap²·ei_sat_mult`) | `weight_cap²·mult` | `weight_cap·√mult` (= `weight_cap` at mult 1, `2·weight_cap` at mult 4) |
| L2E feedforward (`excitatory_saturation_cap = UNIT·weight_cap`) | `UNIT·weight_cap` (8e6) | `√(UNIT·weight_cap) ≈ 2828` |

**Why do it:**
- `w_cap` becomes a **linear** (`× UNIT`) quantity everywhere, so it **eliminates
  the `× UNIT²` quadratic-denominator special case** and the `weight_cap²` trick
  from the fixed-point work — every weight-like cap then scales `× UNIT`
  uniformly, simplifying the scaling story in `Integer_Defaults_Note.md`.
- `w_cap` is directly interpretable ("the weight the gate settles at") instead of
  an abstract squared quantity; `ei_sat_mult` can be redefined to act on the
  equilibrium directly (its √), keeping it a clean multiplier.

**Caveat / do-not:** a blind formula swap that keeps today's numbers sends every
equilibrium to the old `w_max` value (e.g. the gate would target 1.5e6 instead of
~1225) → weights explode. Every cap must be re-derived as `√(old)`.

**Verification:** because it is a pure reparameterization, use the same
event-count equivalence check as the fixed-point rescale — spike / winner /
gate-discharge counts must match the pre-refactor baseline exactly across seeds,
with gate magnitudes unchanged.

## Implementation sketch (when built)

- Add a per-`L1E` adaptation variable (reuse/parallel `ca`) with a slow recovery
  constant; expose it behind a flag (e.g. `sfa` / `encoder_fatigue`) defaulting
  off, like the other opt-in mechanisms.
- Make it throttle **firing** (threshold offset or slow AHP current), not release.
- Verify: (1) with **no frequency term anywhere in code**, the *target* L2E RFs
  put mass on rare pixels and **suppress the center** (persistence shows up as
  smaller center weights, not just transient quieting); (2) interleaved keeps
  8/8; (3) blocked improves, with a **recovery-timescale sweep** demonstrating the
  dependence; (4) resolve the refractory-interaction questions above empirically.
