# Local Predictive Inhibition — Final Report

Implementation and evaluation of the bounded hypothesis in `Experiment.md`:

> A paired local L1E trace combined with dense L2E spike feedback can let each L1I
> column learn context-feature contingencies through a local bounded rule,
> suppressing continued predictable L1 activity more than equally frequent but
> contextually surprising activity.

**Result: SUPPORTED.** All six preregistered Section 13 criteria pass across all ten
seeds. Full per-run numbers are in `RESULTS.md` / `results_summary.json`; the
reproducible artifacts are the 100 per-run history npz files, `metrics.jsonl`,
`shuffle.json`, and `summary.json` under the (gitignored) run directory.

---

## 1. Architecture map

Pre-edit and post-edit maps with the exact methods and final array layouts are in
`ARCHITECTURE_PRE_EDIT.md` and `ARCHITECTURE_POST_EDIT.md`. In brief: two default-True
strategy boundaries were added to `neuron_flexible.Neuron`
(`postsynaptic_learning_enabled`, `inhibitory_plastic`); the trace + predictor live in
`snn/rules/predictive.py`; `SimulationEngine` gained nine legacy-preserving parameters,
the paired 9-afferent L1I layout `[local, fb0..fb7]`, the `local{i}`/`local_evidence`
serialization, and the exact Section 7 timestep order. L2 competition, L2 feedforward
learning, weighted L1 inhibition, and the fixed-point units are unchanged.

## 2. Parameter registry (resolved, `local_plus_feedback`)

Machine-readable in `results_summary.json` (`parameter_registry`). Human-readable:

| Parameter | Value | Parameter | Resolved |
| --- | --- | --- | --- |
| paired_local_enabled | true | θ_L1I = G | 2666.67 |
| predictive_feedback_enabled | true | local afferent (0.40·G) | 1066.67 (fixed) |
| l2_to_l1i_delivery_enabled | true | feedback init (0.10·G) | 266.67 |
| predictive_local_weight_frac | 0.40 | L1I→L1E gate (0.50·θ_L1E) | −500 (frozen) |
| predictive_feedback_init_frac | 0.10 | predictive L1I leak | 0.50 |
| predictive_feedback_eta_up / down | 0.08 / 0.04 | trace λ = exp(−1/2) | 0.60653 |
| predictive_trace_tau_steps | 2.0 | L1I refractory | 2 |
| predictive_output_gate_frac | 0.50 | | |

## 3. Software verification

| Check | Result |
| --- | --- |
| Root regression scripts (`test_*.py`) | **23/23 pass** |
| Golden equivalence (`tests/golden`) | **100/100 arrays bit-exact** (features off) |
| Deterministic repeat (same seed/config/schedule) | **identical** (legacy & live) |
| Pixel-0 sentinel (`test_runner_pixel_zero.py`) | **pass** (all 9 columns, pixel 0 first) |
| Baseline trace + DASHBOARD_OVERRIDES | saved in `baseline/` |
| 100 primary runs | **complete, 0 exceptions, all artifacts present** |

Golden was found broken *before any edit* (its dashboard case depended on the
uncommitted `.claude/dashboard_seed.txt`); it was made reproducible from committed
state (explicit `seed=1`) and regenerated once in Phase 0, with every non-dashboard
array verified byte-identical. See Phase 0 commit.

## 4. Primary outcome (Section 13)

CSC_primary = last-100-acquisition contextual suppression contrast (higher = more
context-selective suppression of the predictable feature). Median over 10 seeds:

| Mode | median CSC | mean ± std | recovered |
| --- | --- | --- | --- |
| baseline (no feedback loop) | 0.0000 | 0.0000 | 0/10 |
| local_only (paired, no feedback) | 0.0000 | 0.0000 | 0/10 |
| time_shuffled_feedback (rate-matched control) | **−0.0063** | −0.0055 ± 0.0086 | 10/10 |
| legacy_feedback (historical reference) | 0.0388 | 0.0411 ± 0.0225 | 10/10 |
| **local_plus_feedback (hypothesis)** | **0.0518** | 0.0464 ± 0.0173 | 10/10 |

| Criterion | Result |
| --- | --- |
| 1. median CSC(live) > 0 | 0.0518 ✓ |
| 2. live > local_only, ≥8/10 | 10/10 ✓ |
| 3. live > shuffled, ≥8/10 | 10/10 ✓ |
| 4. M_shuffled ≤ 0.75·M_live (M_shuffled=−0.006) | ✓ |
| 5. reversal recovery, ≥7/10 | 10/10 ✓ |
| 6. no criterion met by permanent L1E silence | ✓ (no silent seeds) |

**All six pass → the first experiment is SUPPORTIVE.**

## 5. Secondary outcomes

- **Coincidence gating.** For live feedback, P(spike | local+feedback coincident) =
  0.968 but P(spike | feedback-only) = 0.005 — the predictive L1I fires almost only
  when the paired local feature and the feedback coincide. Legacy L1I fires on
  feedback regardless of the local feature (0.937 vs 0.933). This is the mechanistic
  signature of the design: the local trace gates the feedback.
- **Rate-controlled.** The shuffled control fires ~85–90 % as many L1I spikes as live
  (median 10748 vs 12337) and removes ~80 % as much charge, yet its CSC is ≈0 /
  slightly negative. Selectivity is therefore not explained by firing amount or
  charge — it is carried by the temporal contingency the derangement destroys.
- **Reversal.** Live CSC reverses sign from +0.052 (acquisition) to −0.034
  (reversal), and 10/10 live seeds meet the 3-window recovery rule: the learned
  suppression tracks the flipped contingency.
- **Predictor matrix.** The predictor weights differentiate by source, but the
  acquisition-frozen contrasts PWD_F, PWD_G are **negative** (≈−1340), not the
  naively-expected positive — a genuine emergent effect (see §6, Q1).
- **L2 specialization.** Four-pattern distinct-owner count is 3.2 for *every* mode
  (baseline through live), despite L1I firing from 0 to 33736 — the L1 feedback loop
  does not drive or reshape L2 competition.
- **Shuffle checks.** Every seed/phase used a genuine derangement (tries 1–6; no
  fallback needed), and per-source counts + all-zero-vector counts were preserved
  exactly.

## 6. Direct answers (Section 15.7)

1. **Did local information create feature-specific predictors?** Yes — feedback
   weights became source- and feature-specific. But the frozen-group weight contrast
   (PWD) is *negative*, not positive, because of a suppression↔trace feedback loop:
   when L1I successfully suppresses a predictable feature, the paired L1E fires less,
   its local trace x is low more often, and the `eta_down·(1−x)` term drives the
   predictor *down* on exactly the feature it is suppressing. The suppression (CSC),
   not the raw predictor weight, is the clean readout of the learned contingency.
2. **Was selectivity contextual rather than feature frequency?** Yes. The CSC contrast
   holds feature frequency constant (the same feature is externally active in both
   terms); live feedback yields +0.052 while the count-preserving shuffled control
   yields −0.006.
3. **Did time shuffling remove it?** Yes. M_shuffled/M_live = −0.12 (criterion ≤ 0.75);
   deranging the reference tape eliminated the effect.
4. **Did suppression and predictor weights reverse?** Suppression reverses robustly
   (acquisition +0.052 → reversal −0.034, recovery 10/10). The predictor weights move
   too, but their sign is confounded by the same suppression↔trace feedback as in Q1,
   so this is scored *partial*.
5. **Did L2 specialization change independently of total firing?** Yes — 3.2 distinct
   owners across all modes independent of L1I firing rate.
6. **What failed?** No preregistered criterion. Caveats (not failures): the L2 pool
   collapses to ~3 active neurons, so the predictor sees few context sources (X/Y
   groups of size 1); the raw PWD sign is inverted by suppression-trace feedback; and
   legacy_feedback also reaches a comparable CSC (0.041) — so the local predictor is
   not *uniquely necessary* for some selectivity, though it achieves it at ~1/3 the
   L1I firing and with genuine coincidence gating (feedback-only spike prob 0.005 vs
   legacy 0.93).

## 7. Limitations

- **Substrate-limited context sources.** Only ~3 of 8 L2E survive competition, so the
  effect is carried by 1–2 X/Y specialists per seed; PWD groups are often size 1 and
  noisy. This is a property of the preserved L2 side, not the predictor.
- **Modest absolute effect.** CSC ≈ 0.05: differential suppression removes ~5 % of the
  offered charge between contexts. Real and reproducible, but small.
- **Predictor weight is not a clean observable.** The suppression↔trace feedback makes
  the raw predictor contrast anti-correlate with successful suppression; only the
  behavioural readout (CSC) reflects the contingency straightforwardly.
- **Legacy is not a null.** The historical assembly-credit L1I develops comparable
  context selectivity; the decisive evidence is live-vs-its-own-shuffled-tape, not
  live-vs-legacy.

## 8. One next experiment

**Scale the context substrate, hold the rule fixed.** Re-run the identical matrix
while varying only the L2 side to raise the number of surviving distinct L2 owners
from ~3 toward ~6–8 (e.g., sweep L2E count / competition parameters that the brief
lets us treat as the fixed substrate), keeping the predictive-inhibition preset
byte-for-byte. Test two predictions: (a) CSC grows with the number of context-
specialized sources, and (b) once suppression is distributed across more sources the
per-source suppression↔trace feedback weakens and PWD_F/PWD_G turn *positive*,
becoming a clean readout. This isolates whether the small CSC and inverted PWD are
substrate-limited (few sources) rather than rule-limited — the single most
informative follow-up these data justify.
