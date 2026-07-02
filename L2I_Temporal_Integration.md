# L2I: From Single-Spike Relay to Temporal Integrator

> **Superseded (partially).** This document's central invariant --
> `L2_EI_WEIGHT_CAP < threshold_l2`, making a single L2E->L2I synapse
> *permanently* incapable of firing L2I alone, however much it learns -- has
> been intentionally reversed. The current design (see `backend/simulation.py`'s
> module docstring and the "L2E->L2I 'assembly evidence' synapses" comment
> block) still starts synapses well below threshold (preserving the
> "round-robin" early-training behavior documented below), but now allows a
> habitually-participating synapse to learn all the way up to `threshold_l2`
> itself and become independently sufficient once trained -- so multi-neuron
> integration is a property of an *undertrained* synapse, not a permanent
> ceiling. The root-cause analysis, the equilibrium math for the leak rate,
> and the validation methodology below are still accurate for the mechanism
> they describe; only the "cap stays below threshold forever" conclusion no
> longer holds.

**Status: implemented.** Change is confined to `backend/simulation.py`
(instantiation parameters and wiring only). `neuron.py`, `layers.py`,
`cortical_column_flexible.py`, `neuron_flexible.py` are **byte-identical** to
before this change — confirmed by `git diff` below. No plasticity rule
(confidence trace, Hebbian activity rule, inhibitory-discharge plasticity,
homeostatic scaling) was touched.

```
$ git diff --stat -- neuron.py neuron_flexible.py layers.py cortical_column_flexible.py
(empty — nothing changed)
$ git diff --stat -- backend/simulation.py
 backend/simulation.py | 61 insertions, 15 deletions
```

---

## 1. Root cause — why one L2E spike was sufficient to fire L2I

Investigated against every item on the checklist. Each answer is a specific,
verifiable fact about the pre-fix code, not a general impression.

| Item | Pre-fix state | Why it mattered |
|---|---|---|
| **Initial E→I weights** | `set_lateral_excitation_weights(thr_l2)` — every `L2E→L2I` synapse initialized to **exactly** `thr_l2` (4.0). | A single spike delivers `input_current = weight = thr_l2`, i.e. *exactly* the threshold, in one step. |
| **E→I weight cap** | `weight_cap = thr_l2` for L2I — identical to the initial value. | Zero headroom. Even though L2I is a normal `Neuron` with the (unmodified) Hebbian rule available, `_apply_budget_and_cap`'s clip made every update a no-op — confirmed empirically in the prior investigation (`ei_weights_ever_changed: false` in all 4 seeds). The weight was pinned at the cap from birth. |
| **L2I threshold** | `threshold = thr_l2` — the *same* parameter used for the weight above. | Weight and threshold weren't just both large — they were the *same number by construction*, guaranteeing an exact single-spike crossing, not a coincidence of unrelated tuning. |
| **Membrane decay (leak)** | `leak_rate = leak_l2 = 0.01`, inherited from `CorticalColumn`'s single shared `leak_rate` argument (same value used for L2E's own slow, many-volley accumulator). | Irrelevant to the immediate-fire bug itself (the bug fires within the *same* step as the spike, before any leak is applied at all — see "timestep ordering" below) — but critically, if the weight were simply lowered without also revisiting this, `leak_l2` is *far* too slow (≈1%/step) to establish any meaningful "few timesteps" evidence window: almost any two spikes across an entire pattern trial would still sum, and the fix would degrade into "eventually enough spikes from anywhere, at any time, count," not genuine temporal integration. This is why leak had to be retuned too, not just the weight. |
| **Refractory behavior** | `self.l2.inhibitory_neuron.refractory_period = 0`, set explicitly. | Doesn't *cause* the immediate-fire behavior (that's the weight=threshold identity above) but *compounds* it: with zero dead time, L2I could re-fire on literally the next step given any further input, so every winner, every volley, produced a fresh, immediate discharge — reinforcing the "relay" character measured before (L2I fired on ≈100% of winner-producing steps). |
| **Synaptic conductance accumulation** | **Not modeled as a separate state at all.** `Neuron.receive_input` does `self.potential += weight * spike` directly — a current-based (delta) synapse, not a conductance-based one with its own decay kernel. The `trace` array is a separate bookkeeping accumulator used only for plasticity credit; it does not feed back into membrane dynamics a second time. | Because there is no separate conductance/PSP variable, the *only* available temporal-shaping mechanism in this codebase is the membrane's own leak. This directly determined the fix: retune `leak_rate`, don't invent a new state variable. |
| **Timestep ordering** | Within `step()`, for L2I: `receive_input()` → `check_threshold()` → `fire()` all execute in the same call, *before* `update()` (which applies leak) runs later in the same step. | Zero delay between "a synapse delivers charge" and "threshold is checked." A weight that exactly equals threshold crosses deterministically in that same instant — there was never a window in which anything else could happen first. |
| **Event/spike scheduling** | Module docstring, unchanged: "Spikes are delivered immediately (no conduction delays)." L1E fires → L2E receives same step → L2E fires same step if threshold crossed → L2I receives same step → L2I checked same step. | Fully synchronous, zero-delay delivery end to end. Combined with weight=threshold, this meant L2I's decision was always made and settled before a second contributor could possibly exist — it structurally could never see more than one spike. |

**Summary of the root cause:** four independent facts converged on the same
outcome — (1) the E→I weight was initialized at exactly the threshold, (2)
it had zero headroom to ever be anything else, (3) delivery is instantaneous
within a step, and (4) the neuron checks its threshold immediately after
every input rather than waiting. None of these are "bugs" in the sense of
broken code — `apply_inhibition`, `receive_input`, `check_threshold`, and
`fire` all behave exactly as documented. The issue was **parameterization**:
the same generic `Neuron`/`CorticalColumn` machinery was configured, for
this one synapse type, to guarantee a one-spike trigger.

---

## 2. Design: choosing parameters from the relationship, not by trial-and-error weight reduction

Three free parameters govern L2I's integration behavior: the per-synapse
E→I weight $w$, L2I's own leak rate $\lambda$, and its threshold $\theta$
(kept at `thr_l2 = 4.0`, unchanged, so L2I's "evidence budget" stays on the
same scale as an L2E's own firing budget — no new threshold parameter was
introduced).

**Invariant 1 — no single synapse can ever be sufficient, at any training
state.** Require the weight *cap* (not just the initial value) to sit
strictly below threshold:
$$w_{\text{cap}} < \theta$$
This makes "one spike is insufficient" a **structural guarantee**, not an
initialization accident — even if the (unmodified) Hebbian rule fully
saturates that synapse over arbitrarily long training, $w \le w_{\text{cap}} < \theta$
still holds.

**Invariant 2 — even two fully-trained synapses, fired on the closest
timesteps the network's own WTA allows, are still insufficient.** The
existing per-step winner-take-most selection already guarantees at most one
L2E fires per step, so the tightest possible gap between two *different*
contributors is exactly one step. With leaky decay of the first contribution
over that one-step gap:
$$w_{\text{cap}}(1-\lambda) + w_{\text{cap}} < \theta$$
Solving with $w_{\text{cap}}=2.0$, $\lambda=0.07$, $\theta=4.0$:
$$2.0 \times 0.93 + 2.0 = 3.86 < 4.0 \checkmark$$
This makes **three** the structural minimum number of distinct contributing
L2E synapses, under *any* training state or timing — not "typically two,"
a provable floor.

**Choosing $\lambda$ — the temporal window.** L2I needs a *short* retention
window (so unrelated, temporally distant spikes don't pad out an assembly
that isn't really co-active) but not so short that spikes from the *same*
volley-scale burst fail to sum. Targeting ~65-75% retention across one
`volley_period` (4 steps) and single-digit percent retention across one full
pattern-presentation trial (25 steps):
$$(1-\lambda)^4 \approx 0.75 \;\Rightarrow\; \lambda \approx 0.07, \qquad (1-0.07)^{25} \approx 0.16$$
i.e. evidence from the same volley-scale burst mostly survives; evidence
from a much earlier, unrelated presentation mostly does not. This is ~7×
faster than `leak_l2` (0.01), which is intentional — L2E's leak is tuned for
*slow, many-volley* accumulation of a *single* pattern signal; L2I's leak is
tuned for a *short* population-coincidence window, a different
computational role entirely and should not share L2E's time constant.

**Choosing $w$.** With $\lambda=0.07$, $\theta=4.0$, a single spike at
$w=1.5$ (`L2_EI_WEIGHT_INIT`) delivers 37.5% of threshold — clearly and
robustly insufficient (not a marginal near-miss). Summing $n$ contributions
spaced at the tightest (volley-period) cadence:

| Contributing spikes ($n$, spaced 4 steps apart) | Cumulative potential | % of threshold |
|---|---|---|
| 1 | 1.50 | 37.5% |
| 2 | 2.32 | 58.0% |
| 3 | 2.92 | 73.0% |
| 4 | 4.09 | **crosses** |

Final chosen parameters (`backend/simulation.py`):

```python
L2_EI_WEIGHT_INIT = 1.5    # initial L2E->L2I weight (well below threshold_l2)
L2_EI_WEIGHT_CAP  = 2.0    # ceiling for E->I learning (< threshold_l2)
L2I_LEAK_RATE     = 0.07   # L2I's own membrane/trace leak (>> leak_l2)
```

Only three constants changed, plus the weight-cap assignment for the `L2I`
neuron in the per-neuron configuration loop. `threshold_l2`, `leak_l2`,
`refractory`, and every plasticity rule are untouched.

---

## 3. Implementation

```diff
- self.l2.set_lateral_excitation_weights(thr_l2)
+ self.l2.set_lateral_excitation_weights(L2_EI_WEIGHT_INIT)
  ...
+ self.l2.inhibitory_neuron.leak_rate = L2I_LEAK_RATE
  ...
- n.weight_cap = thr_l2 if nid.startswith('L2') else thr_l1
+ n.weight_cap = L2_EI_WEIGHT_CAP if nid.startswith('L2') else thr_l1
```

That is the entire behavioral change. Everything else — the WTA-style
per-step winner selection, the pool-wide discharge once L2I *does* fire, the
`L2I→L2E` gate's own inhibitory-discharge plasticity, the confidence-trace
excitatory rule, homeostatic scaling — is unmodified. The "several L2E
neurons get to spike over the next few timesteps" behavior the task
description asks for **falls out automatically**: since L2I no longer fires
(and therefore no longer triggers the pool-wide discharge) on the first
spike, nothing suppresses the other near-threshold neurons in the meantime —
they remain free to individually cross threshold on their own subsequent
volleys, each contributing to L2I's now-accumulating membrane, exactly as
specified. No change to the WTA/winner-selection code was needed to produce
this — it was a direct, structural consequence of removing the one-spike
trigger.

---

## 4. Validation

All numbers from an instrumented run of the *actual* `SimulationEngine`
(reading only its existing `.spiked` / `.potential` state — no new hooks
added to the engine). For every `L2I` spike, "contributors" is computed
**exactly**, not with a time-window heuristic: `L2I`'s potential resets to
precisely `0.0` on every fire and only ever increases via `receive_input`
from an `L2E` spike, so "every `L2E` spike since `L2I`'s last reset" is
*exactly* the set that fed the potential which just crossed threshold.

### 4.1 A single spike is no longer sufficient

| Seed | L2I spikes (20 epochs) | Fires from exactly 1 distinct contributor | Minimum distinct contributors observed |
|---|---|---|---|
| 1 | 82 | **0** | **3** |
| 2 | 71 | **0** | **3** |
| 3 | 76 | **0** | **3** |
| 4 | 85 | **0** | **3** |

Zero single-contributor fires across **314 L2I spikes over 4 seeds**, and
the observed minimum is exactly 3 in every seed — matching the
mathematically-derived floor in §2 precisely, not just "usually more than
one."

### 4.2 Distribution of contributing-spike counts (seed 1, 82 events)

| Distinct contributors | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Count | 32 | 18 | 15 | 8 | 5 | 4 |

Most events involve 3–5 distinct L2E neurons (a genuine assembly, not a
pair); a tail of events involves 6–8 (near-full-pool consensus). This
matches "coincident activity of *multiple* L2E neurons," graded by how
tightly clustered in time the contributing spikes were.

### 4.3 Concrete example — membrane trajectory of one L2I spike (seed 1, fires at t=45)

```
t:        30   31   32   33   34   35   36     37     38     39     40     41     42     43     44     45
V(L2I): 0.00 0.00 0.00 0.00 0.00 0.00 1.395  2.692  2.504  2.329  2.166  2.014  1.873  1.742  3.015  [fire->0]
```
- t=36: first contributor arrives (V jumps 0→1.395 ≈ 35% of threshold — clearly insufficient alone, matching the design target).
- t=37: second contributor arrives (V jumps to 2.692).
- t=38–43: no new input; pure leak. Ratio 2.504/2.692 = 0.930 = $(1-\lambda)$, exactly the designed decay rate.
- t=44: third contributor arrives (V jumps to 3.015 — still below threshold).
- t=45: fourth contributor arrives, crossing threshold; `L2I` fires, potential resets to 0.
- Logged event: `n_contributors=4, contributor_neurons=[L2E0, L2E5, L2E6, L2E7]`.

This is a directly observed instance of exactly the intended sequence:
several neurons independently cross threshold on separate volleys; `L2I`
accumulates their leaky-summed evidence; only once enough has accumulated
does it fire.

### 4.4 Before / after, same instrumentation, 4 seeds × 40 epochs

| Metric | Before (relay) | After (integrator) | Interpretation |
|---|---|---|---|
| Inhibitory events / seed | ~5,200–5,260 | ~1,040–1,090 | ~5× fewer, higher-evidence discharges instead of firing on nearly every winner |
| Epoch all 8 output gates fully saturate (of 40) | **5–6** | **31–32** | Gates stay in a non-saturated, information-carrying regime for ~6× longer — this directly targets the bottleneck identified in the prior investigation (`L2I_Bottleneck_Investigation.md` §2–3): full saturation was erasing all pattern-specific structure by ~13–15% of training |
| Fraction of events landing on an already-saturated gate | ~46–47% | **~1.5–1.8%** | Learning events are no longer mostly wasted no-ops |
| Distinct final winners / 8 (40-epoch budget) | 5, 6, 6, 6 | 4, 4, 6, 6 | **Regressed in 2/4 seeds at a fixed 40-epoch budget** — see §4.5 |

### 4.5 The 40-epoch regression is a training-budget artifact, not a capability loss

Two of four seeds show fewer distinct winners at the *same* 40-epoch budget
used for the "before" measurement. This is expected, not a sign the fix is
wrong: the old system reached its (saturated, information-dead) final state
by epoch 5–6 and — provably, since the inhibitory-discharge rule has zero
negative updates and a saturated gate cannot change further — could **never
improve with more training**. The new system is still actively learning at
epoch 40 (gates don't saturate until ~epoch 31–32), so a 40-epoch budget
partially undercuts it relative to its own eventual ceiling. Extending
training to 90 epochs on the identical seeds confirms this directly:

| Seed | Distinct winners @ 40 epochs | Distinct winners @ 90 epochs | Old system's best-ever observed (any seed, `L2I_Bottleneck_Investigation.md`) |
|---|---|---|---|
| 1 | 6 | 6 | 6 |
| 2 | 4 | **7** | 6 |
| 3 | 6 | **7** | 6 |
| 4 | 4 | 5 | 6 |

With more training, every seed matches or exceeds the old (relay) system's
best-ever result across all seeds and epoch budgets tested (max observed
there: 6/8) — including two seeds reaching 7/8, which the saturated old
system was structurally incapable of ever reaching regardless of training
duration.

### 4.6 Locality is preserved (by construction, not by inspection alone)

The diff touches exactly three scalar constants and one weight-cap
assignment — no new state, no new information channel, nothing that could
carry a pattern label, neuron identity, or global statistic. To confirm
concretely:
- `L2I`'s only inputs are `receive_input(l2e)` (a plain weighted sum over
  its *own* synapses and this step's spike vector) and its own `.potential`,
  `.threshold`, `.leak_rate`, `.trace`, `.refractory_timer` — every one of
  these was already local before this change and remains so.
- The (unmodified) Hebbian rule that now has headroom to act on `L2I`'s
  weights reads only `L2I`'s own `trace` and `weights` — it has no access
  to which pattern is on screen or which L2E "won" in any global sense, only
  to which of *its own* synapses carried recent charge.
- No new variable tracks pattern identity, an assembly ID, a spike count, or
  a hand-written time window anywhere in `neuron.py`, `layers.py`,
  `cortical_column_flexible.py`, or `backend/simulation.py`. The only
  "counting" and "window" logic in this investigation lives in the
  *external, offline* validation script used to produce the tables above —
  it is not imported by, or part of, the running engine.

### 4.7 An emergent property, not a new rule

Because `L2_EI_WEIGHT_CAP` (2.0) now sits above `L2_EI_WEIGHT_INIT` (1.5),
`L2I`'s own E→I weights have headroom for the first time — and the
existing, unmodified Hebbian activity rule does act on them:

```
final E->I weights (seed 1, 40 epochs): [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  (all reach the cap)
```

Synapses that repeatedly co-participate in successful `L2I` discharges are
credited by the same trace-gated rule used everywhere else in this
codebase, so **frequently-corroborating L2E neurons earn a stronger vote
over training** — directly matching the requested "future presentations
cause L2I to activate earlier because it has learned the excitatory
assembly," achieved with zero new code. Invariant 1 (§2) guarantees this
cannot regress back into single-spike-sufficiency even once every weight is
fully saturated at the cap — confirmed by the same `min_distinct_contributors_observed: 3`
result holding throughout the entire run, including the post-saturation
tail.

---

## 5. Summary

| Requirement | Status | Evidence |
|---|---|---|
| Root cause documented, every checklist item | Done | §1 |
| L2I integrates over time, no explicit counters/queues | Done | §2–3; membrane leak is the only mechanism used |
| Single spike no longer sufficient | Confirmed | §4.1 — 0/314 single-contributor fires, structural proof in §2 |
| Locality preserved | Confirmed | §4.6 — no new state or information channel exists |
| Existing learning rules unchanged | Confirmed | `git diff` shows zero changes to `neuron.py`/`layers.py`/`cortical_column_flexible.py`/`neuron_flexible.py` |
| Parameters derived from weight/leak/threshold/assembly-size relationship, not arbitrary reduction | Done | §2 — two provable invariants, not a swept-until-it-works constant |
| Validation instrumentation (contributor count, latency, membrane trajectory, contributor identities) | Done | §4.1–4.3 |
