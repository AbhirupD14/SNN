# SNN Architecture & Learning-Rule Design Report

**Repository:** `SNN` — a from-scratch spiking neural network built up one biological
mechanism at a time, in pure NumPy, with no gradients, no supervision, and no
global error signal.

**Branches covered:**
- `main` — baseline: LIF neuron + a single trace-gated Hebbian rule + hard
  winner-take-all (WTA) reset. Documents its own known failure mode.
- `feature/inhibitory-plasticity` — adds two more independent local learning
  systems (inhibitory-discharge plasticity, homeostatic scaling), a
  confidence-based credit-assignment variant of the excitatory rule, and
  replaces the hard WTA reset with a *learned, self-organizing* lateral
  inhibition circuit.

This document lists every architectural decision in the order it was made,
gives the governing equation(s), the measured effect, and an assessment of
**locality** (does the rule use only information available at the synapse/
neuron?) and **biological grounding** (what mechanism it is modeling, and how
faithfully).

---

## 1. Network topology (both branches)

```
L1 (9 pixels)                              L2 (8 pattern integrators)
┌─────────────┐   feedforward (learned)    ┌───────────────┐
│  L1E_i  ───────────────────────────────▶ │   L2E_j        │
│    ▲        │                            │     │   ▲      │
│    │ inhib  │   feedback (E2→I1)         │     ▼   │      │
│  L1I_i  ◀──────────────────────────────  │   L2I (shared) │
└─────────────┘                            └───────────────┘
```

- Each L1 pixel is an excitatory/inhibitory (E/I) pair (`InputLayer`); each L2
  pattern-integrator pool shares **one** inhibitory neuron (`CorticalColumn`).
- **Per-source fan-in**: every L2E neuron has its own trainable synapse *per
  L1 pixel* (`cortical_column_flexible.py`), not one aggregated "from-below"
  input. This is the precondition for receptive-field formation — a single
  summed input destroys the pattern before it reaches a trainable weight.
- **E/I identity is carried purely by the sign of the weight** at the target,
  not by a distinct neuron class — one `Neuron` implementation is reused for
  both. This is a deliberate simplification of Dale's principle (see §6).

---

## 2. Core neuron model (LIF) — unchanged across both branches

Membrane integration, on every input spike vector $\mathbf{s}(t)$, only while
not refractory:

$$
I(t) = \sum_i w_i \, s_i(t), \qquad V(t) \mathrel{+}= I(t)
$$

Per-synapse eligibility trace (the "un-summed" membrane — tracks *which*
lines delivered charge, not just the total):

$$
\tau_i(t) \mathrel{+}= s_i(t)
$$

Leak, applied once per step when not refractory (toward $V_{rest}=0$), with
the trace decaying at the *same* rate as the membrane:

$$
V(t{+}1) = V(t) + \lambda\,(V_{rest} - V(t)), \qquad
\tau_i(t{+}1) = \tau_i(t)\,(1-\lambda)
$$

Threshold and spike:

$$
\text{spike}(t) = \mathbb{1}\!\left[\,r(t) \le 0 \ \wedge\ V(t) \ge \theta\,\right]
$$

On a spike: $V \leftarrow V_{rest}$, refractory timer $r \leftarrow r_{max}$,
trace $\boldsymbol{\tau} \leftarrow \mathbf{0}$, weights updated (§3).

**Locality:** every quantity here ($V$, $\theta$, $r$, $\tau_i$, $w_i$)
belongs to the neuron or its own synapses. No other neuron's state is read.
**Biology:** standard discrete-time leaky integrate-and-fire — a well-supported
reduction of Hodgkin–Huxley dynamics; the trace as "unsummed membrane" is an
abstraction of a fast presynaptic calcium/vesicle-release marker, not a
literal biophysical quantity.

---

## 3. Decision log

Decisions are listed in the order they were introduced. Each entry states the
problem, the rule, the measured effect, and why it is local/biological.

### 3.1 Trace-gated, sign-preserving Hebbian rule (`main`, baseline)

**Rule** — on a postsynaptic spike:

$$
\Delta w_i = \eta \cdot \tau_i \cdot \operatorname{sign}(w_i)
$$

Only synapses that actually delivered charge this window ($\tau_i>0$) are
credited, and each moves *further in the direction of its own sign* — an
excitatory synapse only ever grows more positive, inhibitory only more
negative. This is what makes E/I identity self-consistent under a
sign-encoded Dale's principle.

**Effect (documented in-repo):** on the 8-line-pattern task
(`test_8line_consolidation.py`, Test A) a winning neuron reliably carves a
receptive field onto a pattern's active pixels. But with *no* counter-force,
weights only ever grow toward the cap — Test B shows a **single neuron wins
every pattern** (representational collapse).

**Locality:** reads only this neuron's own trace and weights.
**Biology:** a spike-gated Hebbian/eligibility-trace rule — standard
computational abstraction of "cells that fire together wire together,"
consistent with eligibility-trace theories of three-factor learning (minus
the neuromodulatory third factor, which is absent here).

### 3.2 Homeostatic weight *budget* (static counter-force candidate)

Positive (excitatory) afferent weights renormalize to a fixed total $B$ after
every update:

$$
w_i \leftarrow w_i \cdot \frac{B}{\sum_{j:\,w_j>0} w_j} \quad \forall i: w_i>0,
\qquad w_i \leftarrow \operatorname{clip}(w_i,\,-w_{cap},\,w_{cap})
$$

Strengthening one synapse necessarily weakens the others — a finite-resource
constraint. Present from the "Emergent learning rebuild," but left **off** in
the collapse-characterizing test, and later shown to be insufficient alone
(§3.9) — it bounds growth but does not, by itself, stop one neuron from
monopolizing every pattern, because it says nothing about *inter-neuron*
competition.

**Locality:** sum is taken only over this neuron's own afferents.
**Biology:** a reasonable abstraction of finite postsynaptic resources
(receptor/spine budget), but the *instantaneous, exact* renormalization has no
known local circuit implementation — real synaptic scaling is a slow (hours
to days) process. Flagged as an idealization (§6).

### 3.3 Hard WTA reset (early competition mechanism, superseded)

When one L2 neuron fired, every other L2 neuron in the pool was forced to
resting potential in the same step — a procedural reset, not a learned
synapse.

**Effect (measured, `37458b1`):** this destroyed the sub-threshold charge that
losing neurons had accumulated across volleys. Only one neuron ever fired,
only it ever received the spike-triggered Hebbian update, and **all 8
patterns collapsed onto a single universal winner** (1/8 participation, 1
distinct winner). This is the central failure mode that the rest of the
`feature/inhibitory-plasticity` branch was built to fix.

**Locality/biology:** this rule is **not local** — it directly overwrites
other neurons' membrane state outside any synaptic pathway — and it has no
biological analog (no synapse, no signal propagation, no delay, no learning).
It is explicitly the thing the branch replaces.

### 3.4 Inhibitory-discharge plasticity — 2nd independent learning system

**Rule** (`Neuron.apply_inhibition`), triggered only when an inhibitory
synapse ($w<0$) carries a spike into this neuron — i.e. *event-driven on
inhibition*, not on this neuron's own spike:

$$
\begin{aligned}
V_{\text{pre}} &= V \\
V &\leftarrow V - w, \qquad w = |w_{ij}| \qquad \text{(linear discharge)}\\
V_{\text{post}} &= V \\[4pt]
p &= \operatorname{clip}\!\left(\frac{V_{\text{pre}}}{\theta},\,0,\,1\right)
   \qquad \text{(normalized closeness to firing)} \\[4pt]
\Delta w &= \eta_{\text{inh}} \cdot p \cdot \left(1 - \frac{w}{w_{\max}}\right)
   \qquad \text{(saturating)} \\
w &\leftarrow w + \Delta w \qquad \text{(sign preserved: } w_{ij}\leftarrow -w\text{)}
\end{aligned}
$$

An inhibitory synapse behaves as a **finite adaptive suppression gate**: it
strengthens most when it discharges a neuron that was *close to firing*
($p\to1$) and saturates toward $w_{\max}$ (`inhibitory_weight_cap`, kept
**separate** from the feedforward `weight_cap`) — a finite-resource ceiling
achieved without any global normalization. $p$ is clamped to $[0,1]$
(`2f6e9b8`) so a neuron already above threshold at inhibition time doesn't
drive the gate past its natural saturation.

**Locality:** uses only this neuron's own $V$, $\theta$, and the discharging
synapse's own $w$ — no information about any other neuron.
**Biology:** models inhibitory (GABAergic) synapses as *learning*, not fixed,
gates — consistent with known inhibitory plasticity (e.g. iSTDP literature)
that stabilizes excitation/inhibition balance in cortical circuits. The
"closeness-to-firing" credit signal is a computational proxy, not a literal
biophysical quantity, but it captures a plausible functional role: inhibition
that specifically dampens near-winners is exactly what's needed for
soft competition.

**Verified independence:** the excitatory rule (§3.1) only ever touches
positive weights on this neuron's own spike; this rule only ever touches the
negative gate it discharged through. The two systems provably never write the
same weight (`test_neuron.py`).

### 3.5 Episode-based competition window (interpretation layer only)

Replaces an instantaneous per-step winner readout with a resolver over a
short spiking episode:

$$
\text{episode ends when} \quad
\underbrace{t - t_{\text{last spike}} \ge K}_{\text{Condition A}}
\quad \text{or} \quad
\underbrace{t_{\text{episode}} \ge T_{\max}}_{\text{Condition B}}
$$

with $K{=}5$ silent steps or $T_{\max}{=}12$ steps, whichever comes first.
The winner is then resolved from spike history alone: latest spike time,
tie-broken by spike count — no argmax over membrane potentials, no ranking
across neurons.

**Effect:** verified **dynamics-neutral** — weights and membrane potentials
are byte-identical over 600 steps with vs. without this logic. It touches
only `self.winner` and its own bookkeeping fields; it is read-only with
respect to the network. When first added (before §3.6/3.7 existed), it
exposed the hard-reset collapse directly: the underlying WTA fired exactly
one neuron (`L2E7`) across every pattern.

**Locality/biology:** this is a *report-only* aggregation layer over already-
emitted spikes (analogous to a downstream reader integrating a spike train
over a short window) — it does not participate in learning or membrane
dynamics at all.

### 3.6 Adaptive lateral inhibition replaces the hard reset (co-threshold-crossers)

**Rule change:** when several L2E neurons cross threshold in the same volley,
one fires (highest potential among eligible) and drives the shared inhibitory
neuron `L2I` (weight $w_{E\to I}=\theta_{L2}$ guarantees `L2I` fires). `L2I`
then delivers an inhibitory-discharge event (§3.4) through its own
`L2I→L2E` gate to the **other threshold-crossers only** — sub-threshold
accumulators are left untouched.

$$
j^{*} = \operatorname*{arg\,max}_{j \,\in\, \text{Eligible}} V_j(t), \qquad
\text{Eligible} = \{\, j : V_j(t) \ge \theta_{L2} \,\}
$$

Each `L2I→L2E_j` gate is a real, learned synapse (governed by §3.4) with its
own saturation ceiling $w_{\max}=1.5 < \theta_{L2}=4.0$ — so a saturated gate
can *suppress* but can never fully reset a neuron the way §3.3 did.

**Effect (measured, seed 1, `37458b1`):** 6/8 neurons fire, 5 distinct
winners across the 8 patterns (up from 1/8 and 1). `L2I` fired 617× and drove
718 gate discharges — competition is now demonstrably inhibition-driven, not
procedural. Robust across seeds 1–4 (`test_l2_competition.py`): 5–6 fired,
5–6 distinct winners, gates measurably move away from their initial value,
none reach the sub-threshold ceiling.

**Locality:** the winner-take-most decision reads only this-step membrane
potentials of the pool (a legitimate race, not global information injected
into learning); the suppression itself is delivered through a real synapse
and governed entirely by the local rule in §3.4.
**Biology:** models lateral inhibition via a shared interneuron pool — the
canonical cortical soft-WTA circuit (basket-cell-mediated feedback
inhibition). The $\arg\max$ tie-break for *which* neuron fires first is an
abstraction of "first past threshold wins" under continuous time; the
underlying LIF race is still what actually decides it.

### 3.7 Pool-wide adaptive lateral inhibition (extends 3.6)

**Refinement:** `L2I` now discharges the **entire rest of the pool**, not
just the same-step threshold-crossers. Diagnosis: the neurons causing a
*flickering* winner (rotating every burst) were the ones sitting just
**below** threshold (e.g. 3.9 vs. 4.0) — untouched by 3.6, they coasted
through and won the very next volley.

$$
\forall\, j \neq j^{*}: \quad \text{apply\_inhibition}\big(L2E_j,\; \text{gate } L2I{\to}L2E_j\big)
$$

Subtracting each rival's own learned gate magnitude restarts the race closer
to even every volley, so the best-matched integrator can win *repeatedly* —
the precondition for consolidation. The gate stays below threshold
($w_{\max}=1.5 < \theta_{L2}=4.0$), so this remains a **partial** discharge
that preserves cross-volley evidence, not the universal-collapse reset of
§3.3.

**Locality/biology:** same as §3.6 — the discharge is delivered through the
same learned `L2I→L2E` synapse to every target; nothing new about locality is
introduced, only the *set* of targets it reaches this step (which is still
determined by who is wired to `L2I`, a fixed anatomical fact, not an
ad hoc global rule).

### 3.8 Confidence-weighted excitatory credit assignment (`trace_mode="confidence"`)

Separates two biological quantities that the original rule (§3.1) conflated:
**weight** (gate size) and **confidence** (this neuron's trust that opening
that particular gate helps it fire).

$$
\text{participated}_i = \mathbb{1}[\tau_i > \epsilon]
$$

$$
c_i \leftarrow
\begin{cases}
c_i + \beta\,(1-c_i) & \text{participated (fast, saturating} \to 1)\\
c_i\,(1-\gamma) & \text{otherwise (slow forgetting)}
\end{cases}
$$

$$
\text{credit}_i = \frac{c_i}{\displaystyle\sum_{j \,\in\, \text{active, excitatory}} c_j},
\qquad \Delta w_i = \eta \cdot \text{credit}_i \ \ (w_i > 0 \text{ only})
$$

The same budget normalization and cap from §3.2 run unchanged afterward, so
only the *distribution* of the fixed learning budget changes, not the
finite-resource interpretation. Confidence never touches inhibitory
synapses, so it stays disjoint from §3.4.

**Effect (measured, `benchmark_trace_modes.py`):** on the current task,
`activity` and `confidence` modes are roughly equivalent (~3/8 distinct
winners each) — confidence saturates broadly here because specialization is
**competition-limited**, not credit-limited: a neuron that legitimately wins
several patterns legitimately trusts many pixels. The intended divergence
(a consistently-useful gate earns more trust than an intermittent one) is
directly demonstrated in isolation by
`test_neuron.py::test_confidence_diverges_from_weight`.

**Locality:** confidence and its update are per-synapse, on this neuron only.
**Biology:** models a separate, slower "synaptic tag"/eligibility-consolidation
signal, distinct from synaptic strength — related to synaptic-tagging-and-
capture theories where a fast tag and a slower consolidation/trust signal are
biologically dissociable. The exact update form ($\beta$/$\gamma$ EMA-like
dynamics) is a computational approximation, not a literal biophysical model.

### 3.9 Homeostatic synaptic scaling — 3rd independent learning system

**The only mechanism *not* gated by a spike.** Each neuron keeps a slow EMA
("calcium sensor") of its **own** firing rate:

$$
c_a(t{+}1) = c_a(t) + \alpha_{ca}\,\big(\text{spiked}(t) - c_a(t)\big)
$$

When $c_a$ leaves a deadband around its own set-point $c_a^{\text{target}}$,
the neuron applies a **fixed multiplicative** step to its excitatory resource
$R$ (which replaces the static budget $B$ from §3.2 as the renormalization
target when homeostasis is on):

$$
R \leftarrow
\begin{cases}
R\,(1+u) & c_a < (1-\delta)\,c_a^{\text{target}} \quad \text{(chronically silent — grow)}\\[2pt]
R\,(1-d) & c_a > (1+\delta)\,c_a^{\text{target}} \quad \text{(chronically over-active — shrink)}\\[2pt]
R & \text{otherwise (within band — no change)}
\end{cases}
$$

Critically, $u$ and $d$ are **constants** — not proportional to any error and
not a gradient — and the scaling is multiplicative, so relative weights (the
shape of the receptive field) are preserved. It carries no pattern
information, so "learning happens on fire" still holds for the meaningful
(pattern) learning (§3.1/§3.8) — this rule only restores a starved neuron's
gain until it can fire again.

**Effect (measured, `8ed2a96`):** turning homeostasis on clearly **recruits
dead units** — the fraction of L2 neurons that ever fire rises from ~2/8 to
~4–6/8 across seeds, and $R$ spans roughly 1–6 after training (visibly taming
tyrants and growing silent units on the *rate* axis). It does **not** by
itself produce clean 8/8 tiling — distinct winners rise only ~2→3,
seed-sensitively — because it regulates *how much* a neuron fires, not *which
pattern* it owns; one-to-one tiling is a symmetry-breaking/assignment
problem, not a rate-regulation problem. (The set-point $c_a^{\text{target}}=0.012$
was chosen between an observed specialist's rate (~0.01) and a tyrant's
(~0.02) — a parameter fit to this task, flagged in §6.)

**Locality:** depends only on this neuron's own spike history vs. its own
set-point — no global rate signal, no comparison to other neurons.
**Biology:** directly modeled on Turrigiano-style synaptic scaling — well
documented in cortical cultures and in vivo (activity deprivation/blockade
experiments) as a slow, multiplicative, homeostatic process distinct from
Hebbian plasticity. The *timescale* here (tens of steps) is compressed far
below the biological one (hours–days) — an explicit computational
approximation for tractable simulation (§6).

---

## 4. Historical context: an earlier, abandoned approach to the same problem

Before the current design, an earlier iteration (`sim_snn_fep/`, since
removed in the "Clean slate" commit) attacked the *identical* problem —
one neuron monopolizing every input, there called the **"Tyrant State"** —
with a different toolkit: a **Recurrent Competitive Loop**,
**resonance-based selection**, and **"Weight Divorce."** That track was
abandoned and the network rebuilt from a bare LIF neuron upward
("Emergent learning rebuild"). It's worth noting to your mentor as evidence
that the tyrant/collapse failure mode is not incidental to one
implementation choice — it reappeared under a different architecture and
was independently diagnosed and re-solved by the inhibition + homeostasis
route documented in §3.6–3.9, which is the version with reproducible,
tested, seed-robust numbers behind it.

---

## 5. Locality & biological-plausibility summary

| # | Mechanism | Triggered by | Reads/writes only local state? | Biological classification |
|---|---|---|---|---|
| 3.1 | Excitatory Hebbian (activity trace) | own spike | Yes | Reasonable abstraction (eligibility-trace Hebbian) |
| 3.2 | Fixed weight budget | own weight update | Yes (sum over own afferents) | Computational approximation (exact instantaneous renorm has no known local mechanism) |
| 3.3 | Hard WTA reset | any pool spike | **No** — overwrites other neurons directly | Biologically implausible / not a synaptic mechanism (superseded) |
| 3.4 | Inhibitory-discharge plasticity | inhibitory spike into this neuron | Yes | Reasonable abstraction of inhibitory (iSTDP-like) plasticity |
| 3.5 | Episode-based winner window | spike history, read-only | Yes (downstream observer, not part of the circuit) | N/A — instrumentation layer |
| 3.6 / 3.7 | Adaptive lateral inhibition (co-threshold → pool-wide) | pool spikes routed through real synapse | Yes — delivered via the learned `L2I→L2E` synapse | Reasonable abstraction of interneuron-mediated soft-WTA |
| 3.8 | Confidence-weighted credit | own spike | Yes | Weak-to-reasonable support (synaptic tagging/consolidation analogy) |
| 3.9 | Homeostatic synaptic scaling | own long-run rate, not a spike | Yes | Supported by evidence (Turrigiano-style scaling), compressed timescale |

**No mechanism in the current (`feature/inhibitory-plasticity`) network
uses a global error signal, backpropagation, or information from a neuron
other than what arrives through an actual modeled synapse.** The one
architectural component that violated this (§3.3, the hard reset) was
identified and removed; its replacement (§3.6/3.7) is the only inter-neuron
suppression mechanism left, and it is entirely synapse-mediated and
itself plastic.

---

## 6. Known simplifications / open questions (for review, not hidden)

- **Dale's principle is soft, not structural.** E/I identity is carried by
  weight *sign* on a single shared `Neuron` class, not by distinct excitatory
  vs. inhibitory cell populations with fixed output sign. Nothing currently
  prevents a neuron's weight vector from mixing signs across its own
  afferents (only *inputs* to a neuron are typed by sign, not the neuron's
  own outgoing projections as a class).
- **Instantaneous, exact budget renormalization** (§3.2, and the target $R$
  in §3.9) has no known local biophysical implementation — real synaptic
  scaling is gradual (hours–days), and exact-sum renormalization implies a
  form of precise global bookkeeping within the cell that is a computational
  convenience, not an observed mechanism.
- **No conduction delay** in the current volley-based delivery (an earlier
  version modeled distance-based per-synapse delays and coincidence
  detection; the current engine delivers spikes immediately within a step for
  engineering simplicity) — a explicit, acknowledged step back in temporal
  realism in exchange for tractable competition analysis.
- **Tiling is not yet solved.** Even with adaptive inhibition + homeostasis,
  distinct winners plateau around 3/8, not 8/8. The repo's own conclusion:
  this is a symmetry-breaking/assignment problem, not a rate or
  credit-assignment problem — worth flagging as the open frontier rather than
  a solved result.
- **Several numeric constants are task-fit, not derived from data**
  (e.g. `L2_GATE_WMAX=1.5`, `ca_target=0.012`, `EPISODE_QUIET_K=5`) — chosen
  by parameter sweeps on this specific 8-pattern task, not from biological
  measurement — exactly the kind of assumption that should be interrogated
  per-parameter before any claim of biological fidelity.

---

## 7. One-paragraph summary for your mentor

The network started from a single, provably local, spike-gated Hebbian rule
and one procedural shortcut (a hard reset) to get competition for free; that
shortcut caused total representational collapse (1 neuron wins everything),
which is documented, reproduced, and root-caused in the repo's own tests. The
fix was to remove the only non-local, non-synaptic piece of the network and
replace it with a *second* local, spike-driven, gradient-free learning rule
(inhibitory-discharge plasticity) that lets competition emerge from a real,
adapting synapse — and to add a *third* local rule (homeostatic scaling) that
regulates neurons by their own firing rate rather than any global signal.
Every mechanism now reads and writes only quantities that belong to the
neuron or synapse enacting it; the measured effects (participation
2/8→6/8, distinct winners 1→3–6, robust across seeds) are all reported from
the project's own regression tests rather than asserted.
