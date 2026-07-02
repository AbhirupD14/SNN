# Is a Single Shared L2I Neuron the Bottleneck to One-to-One Specialization?

**Status: investigation only — no architecture or learning-rule code changed.**
All numbers below come from an instrumented run of the existing, unmodified
`SimulationEngine` (`backend/simulation.py`, `neuron.py`) using its own debug
hooks (`apply_inhibition`'s returned events, `.spiked`, `.winner`). The
instrumentation script is not part of the codebase; it only reads state the
engine already exposes.

**Protocol:** 4 seeds × 40 epochs × 8 patterns × 25 steps/pattern
(≈ 8,000 steps/seed, ≈ 5,200–5,260 inhibitory-discharge events/seed).

## Verdict up front

**Partially confirmed, but not for the assumed reason.** A single L2I neuron
is not under-capacity in the naive sense ("one neuron can't represent K
assemblies") — it already owns 8 independent output parameters (one gate per
target), which is architecturally identical to 8 separate scalars. The real,
empirically-confirmed bottleneck is structural: **every competitive event is
routed through one fixed, fully-connected, winner-agnostic broadcast**, and
the discharge rule that writes to those 8 gates is a **pure ratchet with no
forgetting term**. Together, independent of how many "assemblies" exist to
represent, this combination *guarantees* that all 8 gates converge to the
same saturated ceiling and become informationally identical — erasing
whatever pattern-specific structure might have existed early in training.
That is a real bottleneck, but it is a **connectivity/broadcast** problem,
not a **parameter-count** problem — which changes which fix actually works
(§5–7).

---

## 1. Is L2I acting as a multi-pattern detector on its input side?

**Hypothesis:** L2I integrates multiple co-active L2E neurons per spike,
forcing it to represent a blended, ambiguous "assembly."

**Test:** for every step L2I fired, count how many L2E neurons spiked that
same step (its only source of excitatory drive).

| Seed | L2I spikes measured | Presynaptic ensemble size distribution |
|---|---|---|
| 1 | 754 | **100% size 1** |
| 2 | 752 | **100% size 1** |
| 3 | 749 | **100% size 1** |
| 4 | 745 | **100% size 1** |

**Finding — disproven, and more strongly than expected.** L2I never
integrates more than one L2E spike. This is not incidental: the E→I weight
is initialized at exactly `thr_l2` (`set_lateral_excitation_weights(thr_l2)`,
`weight_cap = thr_l2`), so a single spike drives L2I to exactly threshold
and fires it immediately — and the WTA race already guarantees at most one
L2E can spike per step. **L2I is a stateless single-spike relay, not an
integrator, and structurally cannot become "a detector for multiple
patterns"** because it never sees more than one input at a time.

We also checked whether L2I's own afferent (E→I) weights adapt via the
existing Hebbian rule (they are plastic in principle — L2I is a regular
`Neuron`). They do not move in any seed:

| Seed | E→I weights, epoch 0 | E→I weights, epoch 39 | Changed? |
|---|---|---|---|
| 1–4 | all `4.0` (= `thr_l2`, the cap) | all `4.0` | **No** |

They are born at the weight cap, so the clip in `_apply_budget_and_cap`
holds them there forever. **The entire representational burden is on the
output side** — the 8 independent `L2I→L2E` gates.

---

## 2. Do the output gates converge to one, several, or unstable assemblies?

**Rule reminder** (`Neuron.apply_inhibition`, unchanged):
$$\Delta w = \eta_{\text{inh}} \cdot p \cdot \left(1 - \frac{w}{w_{\max}}\right), \qquad p = \operatorname{clip}(V_{\text{pre}}/\theta,\,0,\,1) \ge 0$$

Note $\Delta w \ge 0$ for all $p, w \le w_{\max}$ — **this rule can only grow
a gate, never shrink it.** We verified this is not merely a formula reading
but an empirical property of every logged event:

| Seed | Inhibitory events | $\Delta w < 0$ | $\Delta w = 0$ (already saturated) | $\Delta w > 0$ |
|---|---|---|---|---|
| 1 | 5,260 | **0** | 2,485 (47.2%) | 2,775 |
| 2 | 5,251 | **0** | 2,378 (45.3%) | 2,873 |
| 3 | 5,224 | **0** | 2,414 (46.2%) | 2,810 |
| 4 | 5,199 | **0** | 2,427 (46.7%) | 2,772 |

Zero negative deltas across **20,934 events over 4 seeds**. The gate is a
**monotonic ratchet with no decay term** (unlike, e.g., confidence's `γ`
forgetting for non-participating synapses). Consequence, also measured
directly from per-epoch weight snapshots:

| Seed | Epoch all 8 gates reach ≥0.99·w_max | out of | Final gate magnitudes (all 8 targets) |
|---|---|---|---|
| 1 | **5** | 40 | all `1.500` (= `L2_GATE_WMAX`) |
| 2 | **5** | 40 | all `1.500` |
| 3 | **6** | 40 | all `1.500` |
| 4 | **6** | 40 | all `1.500` |

**Finding.** In every seed, **all 8 targets fully saturate to the identical
ceiling within the first 12–15% of training**, and stay there. After that
point (~85% of the run) roughly half of all further inhibitory events land
on an already-saturated gate and change nothing (see the ≈46–47% column
above). By the time training "finishes," the 8 gates carry **zero
differentiating information** — they are numerically identical. This directly
answers "does inhibitory learning plateau" and "is there a measurable
capacity limit": yes, and it is not soft — it is a hard ceiling hit early and
held for the rest of training.

---

## 3. Do different patterns compete for the same gates?

For every (target, pattern) pair we summed $|\Delta w|$ and normalized within
each target to get a **purity** score (1.0 = one pattern owns 100% of a
gate's learning signal; 0.125 = perfectly uniform across all 8 patterns).

| Seed | Purity range across the 8 targets | Interpretation |
|---|---|---|
| 1 | 0.178 – 0.239 | far below "owned by one pattern" |
| 2 | 0.164 – 0.204 | far below "owned by one pattern" |
| 3 | 0.143 – 0.232 | far below "owned by one pattern" |
| 4 | 0.170 – 0.252 | far below "owned by one pattern" |

Every target's purity sits close to the 0.125 uniform baseline — no target's
gate is dominated by one pattern's evidence; each gate is trained by a
near-uniform blend of **most or all 8 patterns**.

We also built each pattern's "rival set" (targets receiving ≥10% of that
pattern's discharge mass) and computed pairwise Jaccard overlap between
patterns' rival sets:

| Seed | Mean pairwise rival-set Jaccard overlap |
|---|---|
| 1 | 0.608 |
| 2 | 0.687 |
| 3 | 0.600 |
| 4 | 0.695 |

0.6–0.7 mean overlap is high — most pattern pairs implicate almost the same
broad set of rival neurons. **Finding: yes, multiple unrelated patterns are
competing for the same inhibitory representation**, and the mechanism is now
identifiable rather than assumed: because the discharge amplitude for target
$j$ depends only on $j$'s own $V_{\text{pre}}$ (never on *which* pattern's
winner triggered it), the rule structurally cannot distinguish "$j$ is a
near-winner under pattern A" from "$j$ is a near-winner under pattern B" —
it can only accumulate an undifferentiated, pattern-blind average.

---

## 4. Does inhibition fire earlier as training progresses?

| Seed | Mean first-spike latency, epochs 0–4 | Mean first-spike latency, epochs 35–39 | Δ |
|---|---|---|---|
| 1 | 8.72 steps | 8.00 steps | −0.72 |
| 2 | 8.92 steps | 8.60 steps | −0.32 |
| 3 | 8.31 steps | 8.30 steps | −0.01 |
| 4 | 8.31 steps | 8.70 steps | **+0.39** |

**Finding — disconfirmed.** No consistent shortening; two seeds get slightly
faster, one is flat, one gets slower. This is expected given the fixed
excitatory weight budget (`weight_budget = thr_l2`, or the homeostatic
equivalent) — total feedforward drive per neuron is bounded by design, so
specialization reshapes *which* pixels matter, not *how much* total charge
accumulates before threshold. There is no evidence of an increasingly
confident/fast population response.

---

## 5. How does 5–6/8 distinguishable winners survive fully saturated, pattern-blind gates?

| Seed | Distinct final winners | Patterns forced to share a winner |
|---|---|---|
| 1 | 6/8 | `L2E6`←{row 0, diag "\\"}, `L2E4`←{row 1, col 1} |
| 2 | 6/8 | `L2E0`←{row 1, diag "/"}, `L2E3`←{row 2, col 0} |
| 3 | 6/8 | `L2E4`←{row 0, row 2}, `L2E5`←{row 1, col 1} |
| 4 | 5/8 | `L2E0`←{row 0, col 2}, `L2E7`←{row 2, col 0, diag "\\"} |

This is the key reframing. Because all 8 gates converge to the **same**
value, `L2I→L2E` inhibition — once training has run for more than ~15% of
its duration — functions as a **uniform "everybody who didn't win this
volley gets discharged by the same fixed amount"** signal, not a
pattern-conditioned suppression map. The differentiation that *does* survive
(5–6 distinct winners) is carried entirely by the **feedforward
confidence-weighted receptive fields and the WTA race** (which neuron
happens to accumulate charge fastest for a given input), not by the
inhibitory circuit. The inhibitory circuit's only remaining functional role
post-saturation is generic anti-flicker pressure (§3.7 of the architecture
report) — it prevents one neuron from re-winning every single volley, but it
has no means to *steer* which neuron should own which pattern. The residual
collapses (two patterns sharing a winner, in every seed) are consistent with
that: nothing in the inhibitory pathway can break a tie between two patterns
whose best-matching feedforward neuron happens to coincide.

---

## 6. Capacity analysis — direct answers

- **Does one L2I neuron effectively become a detector for multiple patterns?**
  No — it structurally cannot integrate more than one spike per event (§1);
  it is a relay, not a detector, and its own input weights never move.
- **Is there evidence of interference between learned inhibitory memories?**
  Yes, on the output side: low purity (0.14–0.25, near the 0.125 uniform
  floor) and high rival-set overlap (0.60–0.70) across all 4 seeds (§3).
- **Does learning plateau because contradictory assemblies overwrite each
  other?** Reframe: nothing is *overwritten* (0/20,934 negative deltas) —
  it is *conflated*. Every meaningfully-active gate monotonically climbs to
  the same ceiling regardless of which pattern(s) drove it, erasing
  differentiation rather than fighting over it. This is arguably a **harder**
  failure mode than classical catastrophic interference, since there is no
  dynamic tension left to observe once saturation hits — by epoch 5–6/40 the
  system has already lost the information it would need to specialize.
- **Is there a measurable capacity limit?** Yes, and it is sharp: full
  saturation of all 8 output parameters by ~13–15% of training, in every
  seed, with ~46–47% of all subsequent inhibitory events wasted as no-ops.
  In steady state, the inhibitory output carries on the order of **0 bits**
  of pattern-specific information — every target's gate is numerically
  identical to every other's.

---

## 7. Topology comparison

All five topologies below reuse the *existing* `apply_inhibition` rule,
`_update_weights` (confidence-trace) rule, and homeostatic scaling
unmodified — only wiring (and, where noted, weight-cap headroom, a parameter
not a rule) changes.

### Topology A — Current (one shared L2I)

- **Mechanism:** single relay, full broadcast to all 7 non-winners on every
  event, using one shared set of 8 output gates.
- **Advantages:** simplest possible circuit; cheap; already demonstrated to
  break the *catastrophic* 1-neuron collapse (1→5–6 distinct winners,
  `37458b1`/`8ed2a96`).
- **Disadvantages:** proven above — pattern-blind by construction once gates
  saturate (~epoch 5–6/40); no mechanism can make it pattern-selective
  without changing the plasticity rule itself, because the shared-broadcast
  structure guarantees identical treatment of every winner.
- **Biological realism:** a single shared basket-cell-like interneuron
  driving global feedback inhibition over a whole column is a documented
  simplification of cortical soft-WTA circuits, but real cortical
  microcircuits contain *populations* of interneurons (PV+, SOM+, VIP+ etc.)
  with distinct connectivity — one-cell-does-everything is the coarsest
  available abstraction.
- **Expected scalability:** poor — a fixed number of independent scalars (8,
  one per target) that all saturate under the same pattern-blind pressure
  will not scale representational richness as pattern count grows; the
  saturation-time finding (~13–15% of training) suggests it saturates *faster*,
  not slower, as more patterns contribute traffic to the same gates.

### Topology B — Small, fully-connected inhibitory population

- **Mechanism:** replace L2I with `{I0..I3}`, each still receiving from
  *all* L2E and projecting to *all* L2E, no I–I interaction.
- **Analysis:** if every `Ik` has identical fixed E→I weights (as A does
  today) and identical trigger conditions, every `Ik` fires on exactly the
  same events as every other — **there is nothing to break the symmetry
  between them**. Each would independently ratchet its own redundant copy of
  the same 8 gates toward the same ceiling. This is $k\times$ the parameter
  count for **zero** additional information, unless something (noise,
  distinct initial weights with headroom to differentiate, or mutual
  inhibition) breaks the symmetry.
- **Biological realism:** redundant/degenerate interneuron populations do
  exist, but without competitive interaction they aren't doing distinct
  computational work — this is closer to redundancy than specialization.
- **Expected outcome without further changes: no improvement over A.**

### Topology C — Column-local (one dedicated I per E)

- **Mechanism:** `E_i <-> I_i`, each `I_i` presumably driven by "the rest of
  the pool" and projecting only back to `E_i`.
- **Analysis — important, non-obvious result:** because each gate `w_j` in
  the *current* system already evolves purely as a function of $j$'s own
  $V_{\text{pre}}$ whenever *any* other neuron wins (verified: gate dynamics
  never reference another target's state), wiring a dedicated `I_j` that
  fires under the same "someone else won" trigger and discharges only `E_j`
  reproduces **numerically identical** per-gate dynamics to today's shared
  L2I. This directly answers the task's own question: **it would simply
  stabilize/relabel the existing computation, not create one-to-one
  ownership**, unless the trigger condition for `I_j` is also changed to
  something pattern-conditioned (which pushes this into Topology D/E
  territory).
- **Biological realism:** reciprocal E–I pairs with tight perisomatic
  feedback are well documented (fast feedback inhibition loops) — this is
  the **most biologically standard** of the five topologies at the
  microcircuit level.
- **Expected scalability:** same saturation ceiling problem as A, just
  distributed across more (functionally redundant) neurons — no scalability
  gain without an accompanying trigger/connectivity change.

### Topology D — Sparse, fixed-partition inhibitory groups

- **Mechanism:** disjoint groups (`E0,E1,E2→I0`; `E3,E4,E5→I1`; ...), each
  `Ik` inhibiting only its own group.
- **Analysis:** this *does* break the broadcast symmetry — within a group,
  fewer competing targets share a gate, which should reduce the purity
  problem measured in §3 for intra-group conflicts. But it introduces a new
  failure mode: two winners in **different** groups are never compared or
  suppressed by each other at all, so redundant neurons across groups (two
  groups both "claiming" visually similar patterns) are never resolved —
  trading intra-group interference for inter-group redundancy. It is also an
  externally hand-designed partition, unrelated to which patterns actually
  share pixels — not an emergent solution.
- **Biological realism:** plausible (sparse/local interneuron connectivity is
  real), but the specific fixed grouping would be an engineering choice, not
  something that emerges from the data.
- **Expected outcome:** ambiguous net effect; likely helps within-group
  purity, likely hurts cross-group one-to-one mapping, without further
  measures (e.g. sparse but *overlapping* or *learned* connectivity).

### Topology E — Competitive inhibitory population (I–I lateral inhibition)

- **Mechanism:** `{I0..I3}` all receive from L2E *and* mutually inhibit each
  other (reusing the exact local-race + `apply_inhibition` pattern the
  architecture already uses for L2E competition — no new selection
  mechanism, just the same mechanism applied one level up). No global
  controller decides which `Ik` fires; it is resolved the same way L2E
  competition already is, by local threshold-crossing dynamics.
- **Analysis:** this is the only topology that structurally creates the
  missing ingredient identified in §3 — **event routing that can depend on
  which winner triggered it**. If (a) the E→I weights for the small
  population are given headroom above their current pinned-at-ceiling value
  (a parameter change, not a rule change) so the *existing* Hebbian rule can
  actually differentiate them over training, and (b) I–I competition means
  only a subset of `{I0..I3}` fires per event, then *which* `Ik` wins that
  local race can become correlated with *which* L2E won upstream (through
  the now-differentiable E→I weights), which means different winners could
  come to route their suppression through different physical gate-sets.
  That is precisely the mechanism needed to stop the pattern-blind pooling
  measured in §3, **without touching the plasticity formula at all** — the
  same ratchet-only `apply_inhibition` rule, applied through a
  context-dependent path, stops being pattern-blind because the *path* now
  carries pattern information that a single fixed relay cannot.
- **Constraint check:** "no global winner-take-all / argmax / centralized
  scheduling" — the existing codebase already resolves L2E competition via a
  local race over threshold-crossers (`max(eligible, key=potential)`,
  interpreted in-repo as a continuous-time first-to-threshold race, not a
  centralized controller). Applying the identical, already-accepted
  mechanism to a 4-neuron I population is not a new centralized mechanism;
  it is reuse of the one local-race abstraction already in the codebase, one
  level up. Flagging this explicitly since the letter of "no argmax" and the
  code's own existing implementation are in tension — worth resolving with
  you before implementation.
- **Biological realism:** competitive/mutually-inhibiting interneuron
  populations (e.g., PV+ basket cell networks with lateral inhibition) are
  well documented; this is a reasonable abstraction, not a stretch.
- **Expected scalability:** best of the five — the number of independent,
  potentially-differentiable gate-sets scales with the size of the I
  population rather than staying fixed at 1, and unlike B/C it has an actual
  mechanism (I–I competition + E→I headroom) to make that scaling
  meaningful rather than redundant.

### Comparison table

| Topology | New params over A | Breaks broadcast symmetry? | Needs E→I headroom to work? | Biological realism | Expected effect on saturation/interference (§2–3) | Expected effect on 1:1 mapping |
|---|---|---|---|---|---|---|
| A — shared L2I | — | No | — | Coarse but standard simplification | Full saturation ~epoch 5–6/40, pattern-blind (measured) | Caps at 5–6/8 (measured) |
| B — small pop., full connectivity, no I–I | ×k gates, redundant | **No** (symmetric) | irrelevant without asymmetry | Redundancy without specialization | No change (predicted) | No change (predicted) |
| C — column-local (E↔I pairs) | ×8 gates, but numerically ≡ A | **No** (same trigger) | irrelevant | Most standard microcircuit motif | No change (proven equivalent, §Topology C) | "Stabilizes," does not create 1:1 (per the task's own framing) |
| D — sparse fixed partition | fewer gates/group | Partial (within-group only) | No | Plausible but hand-designed | Better intra-group purity; new inter-group redundancy risk | Ambiguous / unproven |
| E — competitive I population | ×k gates + I–I synapses | **Yes** | **Yes** | Well-documented (PV+ lateral inhibition) | Predicted to reduce pattern-blind pooling by giving different winners different gate-sets | Best predicted candidate — untested |

---

## 8. Recommendations

- **Most biologically plausible: Topology C** (column-local reciprocal E–I
  pairs) — it is the closest match to well-documented fast perisomatic
  feedback-inhibition loops. But per the analysis above, wired the obvious
  way (same "someone else won" trigger), it is **provably equivalent** to
  today's shared-L2I dynamics — it changes the diagram, not the computation.
  If pursued, it should be considered a *refactor for clarity/scalability*,
  not a fix for one-to-one mapping.

- **Most likely to produce stable one-to-one mappings: Topology E**
  (competitive inhibitory population with I–I lateral inhibition), on the
  condition that E→I weights are given room to differentiate. This is the
  only topology in the comparison that structurally introduces
  winner-dependent routing — the specific ingredient shown missing in §1–3 —
  without modifying any plasticity rule. It is a prediction, not a
  demonstrated result; it should be validated on a small instrumented build
  (same metrics as §1–4: purity, rival-set Jaccard, saturation epoch) before
  being adopted, and the "no global WTA/argmax" constraint should be
  explicitly re-confirmed against the existing L2E race convention before
  implementation (§Topology E constraint check).

- **Why this is a topology fix, not "just add another learning rule":** the
  diagnosis in §1–3 is not "the inhibitory rule is the wrong formula" — the
  same formula, applied through a single fixed fully-connected relay,
  necessarily produces pattern-blind, fully-saturated gates because nothing
  about the *rule* has access to "which pattern caused this event," and
  nothing about the *topology* routes different patterns' events through
  different physical parameters. Changing the rule (e.g., adding a decay
  term) would slow saturation but would not, by itself, stop different
  patterns from writing to the *same* gate — it treats the symptom
  (saturation) without touching the cause (shared broadcast). Changing the
  topology so that different winners *can* route through different gates is
  the only lever, among those investigated, that addresses the mechanism
  identified as the actual bottleneck.

---

## Appendix — raw per-seed metrics

| Metric | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|
| Inhibitory events logged | 5,260 | 5,251 | 5,224 | 5,199 |
| Negative Δw events | 0 | 0 | 0 | 0 |
| Events on already-saturated gate | 47.2% | 45.3% | 46.2% | 46.7% |
| Epoch all gates saturated (of 40) | 5 | 5 | 6 | 6 |
| Purity range (8 targets) | 0.178–0.239 | 0.164–0.204 | 0.143–0.232 | 0.170–0.252 |
| Mean rival-set Jaccard overlap | 0.608 | 0.687 | 0.600 | 0.695 |
| First-spike latency Δ (late − early epochs) | −0.72 | −0.32 | −0.01 | +0.39 |
| Distinct final winners / 8 | 6 | 6 | 6 | 5 |
| Presynaptic ensemble size at L2I spike | 100% = 1 | 100% = 1 | 100% = 1 | 100% = 1 |
