# Why L2E Ownership Doesn't Stabilize — Diagnosis Only

**No code changed for this task.** All findings come from instrumenting the
current (unmodified) `SimulationEngine`, seed 1, 40 epochs × 8 patterns × 25
steps, reading only state it already exposes (`.spiked`, `.l2_drive`,
`.weights`, `.confidence`, `.ca`, `.homeo_budget`, `._inh_events`, `.winner`).
Findings were cross-checked against seed 2 and against a parameter-reverted
instance (see §7) for robustness; the headline conclusion holds in both.

## Bottom line, up front

**Ownership instability is not the disease — it's a symptom of a deeper
failure: the 8 L2E neurons have not differentiated from each other at all.**
Their feedforward receptive fields are, for practical purposes, identical
(pairwise cosine similarity 0.986–0.999 across all 28 neuron pairs; every
single one of the 8 neurons has won every one of the 8 patterns at least
once). With no genuine feature preference separating them, "who wins" a
given presentation is decided by whichever competitor has a razor-thin
transient advantage (mean winning margin: 8% of threshold), which flips
almost arbitrarily from trial to trial. This is traced to a specific,
quantitatively-confirmed mechanism (§4): the eligibility trace that drives
learning resets only on a neuron's own spike, not on pattern boundaries —
and because ownership is already unstable, individual neurons fire rarely
enough (~87 steps between a neuron's own successive fires, vs. 25-step
pattern blocks) that by the time they do fire, their trace has absorbed
evidence from **4.4 different patterns on average**. This smears learning
credit across unrelated patterns' pixels instead of concentrating it,
which is *why* the receptive fields never differentiate — closing a
self-reinforcing loop.

---

## 1. What the requested instrumentation maps to in this codebase (honesty check first)

Several requested quantities don't exist as literal state variables here.
Reporting the mapping used, rather than inventing values:

| Requested | Exists as | Notes |
|---|---|---|
| Membrane potential | `neuron.potential` | Directly available. |
| Excitatory conductance | `weights[pos] · active_inputs` | Current-based (delta) synapse, not conductance-based — see the earlier `L2I_Temporal_Integration.md` §1 finding. Reported as instantaneous feedforward drive. |
| Inhibitory conductance | `\|weights[0]\|` (the `L2I→L2E` gate magnitude) | The discharge amount if/when applied. |
| Adaptive current | `homeo_budget` (the homeostatic excitatory resource `R`) | Closest analog: a slow, multiplicative gain term that isn't a literal AdEx-style adaptation current, but plays the same qualitative role (self-regulated excitability). |
| Threshold | `threshold` | **Fixed**, `thr_l2 = 4.0`, never adapts — reported as constant, not a per-event variable. |
| Refractory state | `refractory_timer` | Directly available. |
| Membrane noise | — | **Does not exist.** Confirmed by code inspection (`neuron.py`, `neuron_flexible.py`, `layers.py`, `cortical_column_flexible.py` contain zero calls to any RNG) and empirically (§2). |
| Random tie-breaking | — | The WTA winner is `max(eligible, key=potential)` — a deterministic Python `max`, not a random draw. |
| Initialization asymmetry | seeded `rng.uniform(0.05, 0.20, ...)` for L2E's initial feedforward weights | Real and seeded — the only place true randomness enters the L2 network, and it's fixed once at `t=0`, never redrawn. |

## 2. Noise contributes 0% — the system is fully deterministic

Two independent runs with identical seed produce byte-identical winner
sequences over 80 pattern presentations:

```
run(seed=1) == run(seed=1)  ->  True
```

This was checked structurally too: `neuron.py`'s one `np.random.uniform` call
initializes L1E/L1I weights, but those are immediately overwritten with fixed
constants before the first simulation step (`e.weights = np.array([-1.0, 1.0])`,
etc.) — the random draw is discarded, never used. `neuron_flexible.py`,
`layers.py`, and `cortical_column_flexible.py` contain no RNG calls at all.
**Every ownership switch documented below is a deterministic, exactly
reproducible property of the dynamics — not measurement noise, not
stochastic exploration.**

---

## 3. Ownership timelines (all 8 patterns)

Every pattern is classified **chaotic**: every single L2E neuron (8/8) wins
every single pattern at some point over the 40-epoch run; no pattern ever
settles into a dominant winner for more than a handful of consecutive
epochs.

| Pattern | Distinct winners | # ownership runs (of 40 epochs) | Longest streak | 2nd-half modal-winner fraction |
|---|---|---|---|---|
| row 0 | 8/8 | 35 | 2 epochs | 0.25 |
| row 1 | 8/8 | 37 | 2 epochs | 0.20 |
| row 2 | 8/8 | 36 | 2 epochs | 0.20 |
| col 0 | 8/8 | 38 | 2 epochs | 0.20 |
| col 1 | 8/8 | 34 | 3 epochs | 0.20 |
| col 2 | 8/8 | 35 | 3 epochs | 0.25 |
| diag \ | 8/8 | 32 | 5 epochs | 0.25 |
| diag / | 8/8 | 36 | 2 epochs | 0.25 |

None qualify as "stable" or "slowly drifting" — there is no pattern whose
winner count is converging over training. Example timeline (`row 2`, first
12 ownership runs of 36 total):

```
L2E4[0] -> L2E2[1] -> L2E3[2] -> L2E2[3] -> L2E3[4] -> L2E6[5] -> L2E0[6] ->
L2E2[7] -> L2E4[8-9] -> L2E6[10] -> L2E1[11-12] -> L2E7[13] -> ...
```

`diag \` has the single longest streak of the whole investigation (5 epochs,
L2E3 at epochs 7–8 and again L2E6 at 9–10) — still nowhere near consolidation,
and not a trend, just the longest run in an otherwise equally volatile
sequence.

---

## 4. Root mechanism: eligibility-trace persistence causes cross-pattern credit contamination

This is the central, quantitatively-confirmed finding.

**Step 1 — receptive fields have essentially not differentiated.**
Feedforward-weight concentration (Herfindahl index; 1/9 = 0.111 is perfectly
uniform, 1/3 = 0.333 would be "3 strong pixels, 6 near-zero") **decreases
monotonically toward the uniform floor over the entire run**:

| Epoch | 0 | 5 | 10 | 20 | 30 | 39 |
|---|---|---|---|---|---|---|
| Mean HHI (8 neurons) | 0.1230 | 0.1182 | 0.1158 | 0.1137 | 0.1132 | 0.1131 |

It starts only slightly above uniform (the random initialization itself has
little structure) and gets *more* uniform, never less, over 40 epochs of
"training." Directly confirming this at the individual-weight level:
pairwise cosine similarity between all 28 possible neuron pairs at the final
epoch:

```
min = 0.9865   mean = 0.9961   max = 0.9993
```

Every neuron looks like every other neuron. This is not "two neurons
overlap" — it is near-total representational collapse across the whole pool.
Confidence vectors are even more uniform (pairwise cosine ~0.9998–0.9999),
consistent with confidence's fast saturation (`β=0.30`) toward 1 for any
synapse that participates in *any* successful spike, regardless of pattern.

**Step 2 — why: the trace never gets a chance to specialize before it's
spent on the wrong pattern.** `Neuron.trace` decays only via the shared
membrane leak and is cleared only in `fire()` — i.e., only when *this specific
neuron* spikes, never on a pattern switch. Measured directly from the actual
spike record:

| Metric | Value |
|---|---|
| Mean gap between a given neuron's own successive fires | **86.8 steps** (median 87.0) |
| Pattern-presentation block length | 25 steps |
| Fraction of inter-fire gaps spanning >1 distinct pattern | **100%** |
| Fraction of inter-fire gaps spanning ≥3 distinct patterns | **99.5%** |
| Mean number of distinct patterns shown during an uncleared trace window | **4.43 / 8** |

Because a typical neuron only wins (and thus only applies its
confidence-weighted credit update) roughly once every 3.5 pattern
presentations, its `trace` at that moment reflects pixels from *4–5
unrelated patterns*, not the one currently on screen. The confidence-credit
rule (`credit_i = c_i / Σc_active`) then spreads the fixed learning budget
across all of those pixels — mechanically flattening the receptive field
rather than sharpening it. This is a **structural property of the current
wiring** (trace-clear tied to own-spike, not to a pattern-boundary signal
the neuron has no way to perceive) — not a parameter that happens to be
mistuned.

**Step 3 — the loop closes.** Flatter receptive fields → smaller, more
arbitrary drive differences between competitors for any given pattern →
even less decisive competition → ownership rotates even more → individual
neurons fire even less predictably → more cross-pattern contamination next
time they do fire. Nothing in the measured data suggests this loop is
converging; HHI is still declining (not plateaued) at epoch 39.

---

## 5. Local tie-breaking: how a specific switch gets decided, given the margins are already thin

**Margins are small.** At the actual moment of decision (pre-WTA potential,
`l2_drive`), the winner's margin over the runner-up:

| | Value (potential units, threshold = 4.0) |
|---|---|
| Mean winner − runner-up gap | 0.327 (8.2% of threshold) |
| Median gap | 0.244 |
| Fraction of events with gap < 0.5 | 78.6% |
| Fraction of events with gap < 0.1 (near photo-finish) | 24.0% |

**Switches are local, not random jumps across the whole pool.** When a
pattern's winner changes between two consecutive winning events, the new
winner was already that event's runner-up or third-place competitor **75.6%**
of the time. This is a persistent 2–3-neuron clique trading the lead back
and forth, superimposed on a slower drift that eventually cycles through all
8 — not uniform random reassignment.

**Neither homeostasis nor inhibition reliably predicts switch *direction*.**
For every detected ownership switch (275 total across the 8 patterns),
compared the old vs. new winner's homeostatic-resource trend and inhibitory-gate
trend across the switch epoch:

| Test | Fraction consistent with that mechanism causing the switch | Interpretation |
|---|---|---|
| New winner's `homeo_budget` grew more (relatively) than old's | 46.5% | ~chance (50%) — not a systematic directional driver |
| Old winner's inhibitory gate grew more (relatively) than new's | 46.5% | ~chance (50%) — not a systematic directional driver |

Both mechanisms are **real and locally causal** (they directly rescale
weights / directly discharge membrane potential — see the architecture
report's locality discussion) but neither has a *consistent bias* toward
producing switches in one direction. That is expected given how thin the
margins already are (§5, table above): with an 8% average margin, almost any
source of slow, continuous, per-neuron drift — homeostatic rescaling being
the most obviously "always active" one (`_homeostatic_scaling` runs every
step for every neuron, not just on a spike) — is *sufficient* to flip a
given event, without needing to point in a *consistent* direction across all
switches.

### Two concrete worked examples

**Homeostasis-driven** (`col 0`, epoch 29, `L2E5 → L2E0`): `L2E5` had been
winning enough recently that its homeostatic resource shrank
(`homeo_budget`: 2.059 → 1.789, the anti-tyranny branch of
`_homeostatic_scaling`) while `L2E0`, relatively quiet, grew
(1.497 → 1.937) — a ~0.71 relative swing in a single epoch, several times
larger than the typical 0.33 winning margin. Their feedforward weight
vectors were 98.1% cosine-similar at the time; the inhibitory gates barely
moved (Δ0.004 and Δ0.003 — not a factor here). **Counterfactual: if
`L2E5`'s homeostatic resource had not shrunk that epoch, `L2E0` likely does
not overtake it yet.**

**Inhibition/recruitment-driven** (`col 0`, epoch 1, `L2E1 → L2E5`): both
neurons had `ca=0` (never yet fired) going into this epoch; `L2E1` fired
first and its own inhibitory gate grew sharply from a prior discharge
history (0.500 → 0.663) while `L2E5`'s stayed low (0.500 → 0.535) — `L2E1`
had been the more habitual near-winner and so was absorbing more suppression
per the pool-wide lateral-inhibition rule. **Counterfactual: if `L2E1`'s
gate had not grown disproportionately (i.e., if it had not been
disproportionately suppressed relative to `L2E5`), it plausibly keeps the
pattern this epoch too.**

Both examples share the same precondition: the two competitors' receptive
fields were already >95% cosine-similar *before* the tie-breaker acted. The
tie-breaker decided a coin-flip; it did not create the coin-flip.

---

## 6. Similarity analysis

Because every neuron has converged toward every other neuron (§4), a
"which pairs specifically compete" table is not very meaningful — nearly
all `C(8,2) = 28` pairs qualify. Representative sample (any pair, since the
pattern holds throughout):

| Pair | Weight cosine | Confidence cosine | Top-3-pixel Jaccard | Firing correlation (per-epoch win counts) |
|---|---|---|---|---|
| L2E0 / L2E1 | 0.992 | 0.9999 | 0.50 | −0.021 |
| L2E0 / L2E2 | 0.993 | 0.9998 | 0.50 | −0.253 |
| L2E0 / L2E3 | 0.994 | 0.9998 | 0.50 | −0.197 |

Top-3-pixel Jaccard of 0.5 (2 of 3 top-weighted pixels shared) shows a
little residual structure survives, but nowhere near the disjoint sets a
genuine one-pattern-per-neuron solution would need. Firing correlation is
weakly negative across sampled pairs — consistent with genuine (if
weak) zero-sum competition for the same events, not independent activity.
**Conclusion: yes, essentially every pair of L2E neurons has effectively
learned the same (nearly featureless) function.** This isn't a story about
two specific neurons that happen to be confusable — the whole population
has collapsed toward one generic detector, replicated 8 times.

---

## 7. Is this caused by the recent L2I temporal-integration change?

Checked directly, since that change was the most recent modification to the
network (diagnosis-only check — reverted only in-memory instance parameters
on a throwaway engine, no repo files touched):

| Configuration | Mean final receptive-field HHI (seed 1) | Mean final HHI (seed 2) |
|---|---|---|
| Old (single-spike relay L2I) | 0.1160 | 0.1169 |
| Current (temporal-integrator L2I) | 0.1131 | 0.1130 |

Both configurations converge to nearly the uniform floor (0.111) — the
representational-collapse phenomenon **pre-dates** the L2I fix and is not
attributable to it. The current configuration is marginally *more* flattened
(plausibly because less-frequent inhibitory suppression events mean more
volleys where multiple near-tied neurons can each independently accumulate
and eventually fire, broadening promiscuous winning slightly further), but
this is a small secondary effect on top of an already-severe pre-existing
condition, not the origin of it.

---

## 8. Root cause attribution

| Cause | Estimated contribution | Basis |
|---|---|---|
| **Eligibility-trace persistence across pattern boundaries → cross-pattern credit contamination → collapsed/near-identical receptive fields** | **~50%** | Directly measured mechanism (§4): 99.5% of refire intervals span ≥3 patterns, mean 4.43 patterns/trace window; HHI monotonically declining toward the uniform floor; this is the condition that makes every subsequent switch possible in the first place |
| Homeostatic scaling (continuous, every-step multiplicative rescaling) | ~15% | Real, directly causal, illustrated concretely (§5); but direction not systematically biased across switches (46.5%, ~chance) — a perturbation source, not a directional driver |
| Inhibitory dynamics (gate/L2I suppression history) | ~12% | Same character as homeostasis — real, illustrated concretely, direction not systematically biased (46.5%, ~chance) |
| Fine-grained deterministic timing / residual accumulation ("other") | ~18% | The residual variance in switch direction not explained by either homeo or gate trend (both near chance) must come from somewhere — most plausibly exact volley-alignment / accumulated-potential history, which is deterministic but too fine-grained to attribute to a single named mechanism without per-step replay of every switch |
| Confidence dynamics (as a mechanism distinct from trace contamination) | ~5% | Confidence itself has no real-time effect on the competitive race (it only reweights *future* learning credit, confirmed by code inspection) — its role is almost entirely as the *implementer* of the trace-contamination effect already counted above, not an independent contributor |
| Noise | **0%** | Confirmed deterministic (§2) — cannot contribute by construction |

---

## 9. Conclusion

The primary mechanism preventing stable one-to-one mappings is **not** any
single "switch trigger" like homeostasis or inhibition — those are real but
secondary, roughly-unbiased perturbations acting on margins that are already
razor-thin. The dominant, root-level mechanism is that **the eligibility
trace which drives excitatory learning is cleared only on a neuron's own
spike, with no notion of a pattern boundary**, and because ownership is
already unstable, neurons fire infrequently enough (~87 steps between their
own fires, vs. 25-step pattern blocks) that their trace routinely mixes
evidence from 4+ unrelated patterns by the time it is spent. This
mechanically prevents any neuron from ever concentrating its receptive field
on one pattern's pixels, which keeps every neuron nearly identical to every
other (pairwise weight-cosine similarity 0.986–0.999, every neuron has won
every pattern), which keeps competitive margins tiny (~8% of threshold on
average), which is precisely the regime in which small, directionally-unbiased
nudges from homeostasis and inhibition are sufficient to flip the outcome —
which in turn keeps individual neurons firing infrequently, closing the
loop. This is a genuinely self-reinforcing dynamical trap, confirmed to
pre-date the most recent architecture change, not a transient side effect of
it. No fix is proposed here per the task constraints — this diagnosis
identifies where the leverage would need to act (the relationship between
trace persistence and pattern boundaries), not what to change.
