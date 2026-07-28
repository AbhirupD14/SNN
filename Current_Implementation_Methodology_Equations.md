# Implementation methodology and equations

**Reference topology: `tiled_cc` — the 9×9 tiled cortical-column hierarchy with eight
competing ordinary E per column plus `Eor`.** Everything below describes what the code
actually does today. Superseded lineages (the `pi` / `old` / `rg` / `rg_residual` graphs,
predictive interneurons, the residual/switch pathway, the removed feature-gated variant,
and the historical bounded/quadratic weight rules) have been removed from this document;
their builders survive only as low-level mechanics for tests and custom graphs.

Equations render as LaTeX. Authority order when they disagree: the code
(`snn/neurons.py`, `backend/simulation.py`, `backend/network_spec.py`) is the source of
truth, then this document.

Notation used throughout: $\theta$ is the excitatory firing threshold
(`e_threshold`, 1000 in every current run), $N$ the ordinary-E count per column
(`cc_e_count`, default 8), $\tau \in [0,1]$ the analytic sub-boundary time within one outer
boundary, and $t$ the integer outer boundary.

---

## 1. The reference graph

A `9×9` RGC surface is tiled into nine `3×3` patches. Each patch drives one L1 cortical
column (arranged `3×3`); one L2 column receives all nine L1 outputs. The graph is composed
from four pure builders — `build_cortical_column`, `connect_rgc_patch`, `connect_columns`,
`tiled_cc_spec` — so no connectivity is hand-written and depth composes.

Every column, at any layer, is the same motif:

| Role | Count | Archetype | Membrane | Learns |
|---|---:|---|---|---|
| ordinary E (pattern detector, WTA competitor) | $N=8$ | `e_latency_competitor` | yes | yes |
| `Eor` (pooled column output relay) | 1 | `e_latency_competitor` | yes | **no** (frozen bank) |
| `C` (coincidence pyramidal cell) | 1 | `e_coincidence` | yes | yes (one basal weight) |
| `I` (inhibitory relay) | 1 | `i_relay` | no | no |

Intra-column edges (fixed rule, per column):

$$
E_i \to \mathrm{Eor},\qquad
E_i \to I,\qquad
I \to E_i,\qquad
\mathrm{Eor} \to C\ (\text{basal}),\qquad
C \to I
$$

Inter-column edges, emitted by the single generic rule `connect_columns(child, parent)`:

$$
\mathrm{child.Eor} \to \mathrm{parent}.E_k \ (\text{feedforward}),
\qquad
\mathrm{parent}.E_k \to \mathrm{child}.C \ (\text{apical})
$$

Counts are exact functions of $N$ — $10N + 111$ nodes and $129N + 20$ edges — giving
**191 nodes / 1052 directed edges** at $N=8$:

$$
\underbrace{81}_{\text{RGC}} + \underbrace{10(N{+}3)}_{\text{columns}} \ \text{nodes},
\qquad
\underbrace{81N}_{\text{RGC}\to E} + \underbrace{18N}_{\text{L1}\to\text{L2 links}} + \underbrace{10(3N{+}2)}_{\text{intra-column}} \ \text{edges}
$$

There are no lateral or cross-column edges, so the nine L1 columns are independent: each
may produce its own winner in the same boundary, and one column's reset never touches
another.

---

## 2. Neuron models

### 2.1 Conductance LIF membrane

Every excitatory cell (ordinary E, `Eor`, `C`) is the same conductance leaky integrator
with unit capacitance:

$$
C_m \frac{dV}{dt} = -g_L\,(V - E_L)\;-\;g_{\mathrm{inh}}\,(V - E_{\mathrm{inh}})\;+\;I_{\mathrm{exc}},
\qquad C_m = 1,\quad E_L = V_{\mathrm{rest}} = 0
$$

The per-step leak fraction maps to a baseline conductance so the no-inhibition,
no-input case reduces exactly to the historical geometric decay:

$$
g_L = -\ln(1 - \texttt{leak\_rate})
\qquad\Longrightarrow\qquad
V \leftarrow (1 - \texttt{leak\_rate})\,V
$$

**Every current run uses $\texttt{leak\_rate} = 0$, hence $g_L = 0$.** With no inhibitory
conductance either, the membrane is a pure integrator: charge persists across boundaries
until the cell fires or is hard-reset. This is what makes a sub-threshold cell able to
accumulate evidence over several boundaries, and it is why "cannot cross from one volley"
is a statement about one-event maturity, not about ever firing.

### 2.2 Analytic sub-boundary crossing

Within one outer boundary the drive packet is frozen constant, so the crossing time to
threshold is solved in closed form rather than searched:

$$
\Delta\tau \;=\;
\begin{cases}
\dfrac{\theta - V}{I_{\mathrm{exc}}}, & g = 0 \ \text{(pure integrator)}\\[2ex]
\dfrac{1}{g}\,\ln\!\left(\dfrac{V_\infty - V}{V_\infty - \theta}\right), & g > 0,\ V_\infty > \theta\\[2ex]
\infty, & \text{otherwise}
\end{cases}
\qquad
g = g_L + g_{\mathrm{inh}},
\quad
V_\infty = \frac{g_L E_L + g_{\mathrm{inh}} E_{\mathrm{inh}} + I_{\mathrm{exc}}}{g}
$$

The scheduler repeatedly takes the earliest finite $\Delta\tau$ across all membranes,
advances **every** membrane to that $\tau$ along its own exact trajectory, fires that cell,
applies same-$\tau$ consequences (apical delivery, hard resets), and recomputes. Exact ties
fall back to stable node order and are recorded as `latency_ties`. Selection is pure
first-spike latency — never a comparison of end-of-boundary voltages.

Advance over a segment:

$$
V \leftarrow
\begin{cases}
V + I_{\mathrm{exc}}\,\Delta\tau, & g = 0\\
V_\infty + (V - V_\infty)\,e^{-g\,\Delta\tau}, & g > 0
\end{cases}
$$

Firing resets $V \leftarrow V_{\mathrm{rest}}$, consumes the frozen drive packet, arms
refractory, and marks the boundary spent. A hard reset does the same to $V$ and the packet
but deliberately leaves learned weights, the activity trace, $g_{\mathrm{inh}}$, refractory
state and an already-emitted spike untouched.

### 2.3 Coincidence pyramidal cell (`C`)

`C` owns one **learned basal** weight and $\ge 1$ **unweighted apical** permission inputs.
It is a strict temporal AND, not a summing unit. With $B$ = basal availability (a basal
event this boundary, or one carried from the previous boundary) and $A$ = any apical event
this boundary:

$$
\text{deposit} \;=\;
\begin{cases}
w_{\mathrm{basal}} \cdot s, & B \wedge A \ \text{(once per boundary)}\\
0, & \text{otherwise}
\end{cases}
$$

The deposit is an **instantaneous charge impulse** at the current $\tau$ ($\Delta V = q$,
since $C_m = 1$), not a current integrated over the boundary. `C` may only fire while its
gate is open — its `crossing_time` returns $\infty$ whenever the gate is closed — so a
retained supra-threshold membrane cannot fire on a non-coincident boundary.

Basal eligibility carries for **exactly one** boundary and is settled once per boundary
*after* the event loop, so a basal event that arrives before this boundary's apical still
gets its chance to coincide.

At $\texttt{leak\_rate}=0$ the one-shot recognition condition is therefore simply

$$
w_{\mathrm{basal}} \;\ge\; \theta
$$

**The top column's `C` is intentionally dormant.** Every column contains a `C`, including
the topmost. The L2 column has no parent, so its `C` owns its `Eor` basal edge and its
eligibility state machine but has **zero apical inputs**: the gate never opens, so it never
deposits, never fires and never learns, however long it runs. This zero-apical case is legal
*only* because the column metadata declares no parent (`has_parent = False`); validation
still rejects an accidentally unwired non-top `C`. Any readiness gate must therefore exclude
it by metadata, never by an id special case.

### 2.4 Inhibitory relay (`I`) and RGC source

`I` is stateless: threshold $\theta_I = \theta/3$, no membrane, no learning, and it emits
**at most one WTA spike per outer boundary**. RGC cells are exogenous binary sources with
no membrane at all — nothing for cortical feedback to act on. Neither owns a weight.

---

## 3. Boundary schedule and delays

Every internal projection costs exactly one boundary (`SYNAPTIC_DELAY = 1`), except apical
permission and relay/hard-reset action, which are zero-latency within the boundary:

| Event at boundary $t$ | Lands |
|---|---|
| RGC active | feedforward delivery at $t{+}1$ |
| ordinary-E spike | its `Eor` feedforward at $t{+}1$ |
| `Eor` spike | parent-E feedforward **and** local `C` basal at $t{+}1$ |
| parent ordinary-E spike | child `C` apical at the **same** $\tau$ |
| `I` WTA reset | same $\tau$, immediate |
| `C → I` confirmation reset | start of $t{+}1$, after the next drive packet is frozen (so that packet is discarded) |

Per boundary the engine: rotates the delay-one buffers, delivers arrivals, resolves each
`C` gate, freezes each membrane's drive packet, applies any scheduled feedback resets, runs
the sub-boundary event loop, then settles eligibility, traces, conductance and refractory.

---

## 4. The learning rule: dual FE/FES

One flag (`dual_fe_fes`) switches **both** plastic families to the inverse-quadratic dual
node/synapse free-energy rule. It is the rule used in every current run.

### 4.1 Ordinary E and `Eor` (pattern detectors)

$$
\mathrm{FE}(I_{\mathrm{accq}}) \;=\; e + \frac{1-e}{1 + B\left(\dfrac{I_{\mathrm{accq}}}{\theta} - \tfrac12\right)^{2}}
\qquad\text{(node factor, shared by all afferents)}
$$

$$
\mathrm{FES}(w_i) \;=\; w_{te} + \frac{1-w_{te}}{1 + B\left(\dfrac{2 w_i}{\theta} - \tfrac12\right)^{2}}
\qquad\text{(per-synapse factor)}
$$

$$
\boxed{\;\Delta w_i \;=\; \eta \cdot \mathrm{FE} \cdot \mathrm{FES}(w_i) \cdot s_i \cdot \varphi_i\;}
\qquad
w_i \leftarrow \min\!\big(\max(w_i + \Delta w_i,\; w_{te}),\; w_{\mathrm{cap}}\big)
$$

- $I_{\mathrm{accq}}$ is this cell's **pre-reset accumulated causal charge** — the frozen
  drive packet captured before the event loop can zero it. It is never clamped, so an
  overshoot by a strong incumbent is retained and *reduces* its own plasticity.
- $s_i = +1$ if afferent $i$ spiked in the causal volley delivered to **this** target on
  **this** boundary, else $-1$. Absence never becomes a positive update; participation is
  read per-target, so one hop's volley can never leak into another target's update.
- The update runs only when the cell itself fires.
- $\mathrm{FES}$ is evaluated from each synapse's own **pre-update** weight, so a vector
  update still moves every synapse by its own factor.

### 4.2 Coincidence basal

`C` is a different kind of cell — one weight, an impulse deposit, and it always fires at
$I_{\mathrm{accq}} \approx \theta$ — so it gets its own operating centers rather than the
detector's:

$$
\mathrm{FE}_C(I_{\mathrm{accq}}) \;=\; e + \frac{1-e}{1 + B\left(\dfrac{I_{\mathrm{accq}}}{\theta} - 1\right)^{2}},
\qquad
\mathrm{FES}_C(w) \;=\; w_{te} + \frac{1-w_{te}}{1 + B\left(\dfrac{w}{\theta} - \tfrac12\right)^{2}}
$$

$$
\boxed{\;\Delta w \;=\; \eta_C \cdot \mathrm{FE}_C \cdot \mathrm{FES}_C(w) \cdot A \cdot s \cdot \varphi\;}
$$

with $A \in \{0,1\}$ the Boolean apical gate at the causal spike, $s$ the causal basal
signal, and $\varphi$ the basal distance influence. Here $I_{\mathrm{accq}}$ is the somatic
membrane immediately before the firing boundary's reset (all `C` charge is basal deposits).

There is no apical weight, no negative-participation term, and on a multi-basal cell only
the causal source's weight moves — so an owner that matured its own association is never
depressed by a different owner's coincidence.

**Why the centers differ.** Applying the ordinary-E references to `C` would pin it at
$I_{\mathrm{accq}} = \theta \Rightarrow \mathrm{FE} = 0.445$ and pull its single weight
toward $\theta/4$ — permanently below the $\theta$ it needs for one-shot recognition. The
`C` references instead put peak plasticity at the cell's real operating point: maximal at
the firing charge $\theta$, and maximal for a weight climbing through $\theta/2$ toward the
one-shot target.

### 4.3 Distance influence

Geometry is a **learning-rate multiplier only** — it never scales delivered charge:

$$
\varphi_i \;=\; \left(\frac{d_{\mathrm{ref}}}{\max(d_i,\, d_{\mathrm{ref}})}\right)^{2}
$$

$d_{\mathrm{ref}}$ is taken **per plastic target** in the tiled family: each target's own
closest incoming feedforward distance scores $1.0$. That keeps a short within-column edge
from rescaling a long inter-layer projection's learning rate. Column-local basal edges
(`Eor → C`) carry no distance penalty at all: $\varphi = 1$.

### 4.4 Why this rule — the bell curve

$\mathrm{FES}$ is an inverse-quadratic bell in **weight space**, and $\mathrm{FE}$ is the
same bell in **charge space**. Both peak at $1$ and decay toward a floor of
$w_{te} = e = 0.001$ in the tails. The whole design is a statement about *when a synapse
should be allowed to change*:

> **Start every plastic weight at the peak of the bell — maximum plasticity — and let
> specialization carry it out into a tail, where it becomes almost unmodifiable. A weight
> that has learned something stops being plastic.**

Concretely, for an ordinary detector at $B=5$:

| $w$ | $\mathrm{FES}$ | meaning |
|---|---:|---|
| $\theta/4$ | **1.000** | initialization — peak plasticity |
| $\theta/8$ | 0.762 | drifting down |
| $3\theta/8$ | 0.762 | drifting up |
| $\theta/2$ | 0.445 | at the detector ceiling — $2.25\times$ less plastic than at init |
| $\theta$ | 0.083 | far tail |
| $\to w_{te}$ | $\to 0.001$ | fully depressed — $1000\times$ less plastic |

and the same shape in charge space:

| $I_{\mathrm{accq}}$ | $\mathrm{FE}$ | meaning |
|---|---:|---|
| $\theta/2$ | **1.000** | half-threshold accumulation — peak |
| $\theta$ | 0.445 | firing exactly at threshold |
| $2\theta$ | 0.083 | strong incumbent overshooting — self-limiting |

Four properties follow, and they are the reason the rule was chosen:

1. **Self-regulating specialization.** A naive rule needs an external cap or normalizer to
   stop runaway growth. Here saturation is intrinsic: as a weight climbs toward its ceiling
   its own $\mathrm{FES}$ collapses, so it asymptotes rather than diverges. The structural
   ceilings in §6 are a *hard* backstop, not the mechanism.
2. **Consolidation without a separate consolidation phase.** "Less plastic after learned"
   is not a schedule, a decay term or a frozen flag — it is the position of the weight on
   its own curve. A matured owner resists being overwritten by a competitor's activity
   simply because it now sits in a tail.
3. **Symmetric protection of the depressed tail.** The bell is symmetric, so a fully
   depressed afferent ($w \to w_{te}$) is *also* nearly unmodifiable. This is deliberate: a
   pattern detector's "off" pixels stay off instead of jittering back up on every
   non-participation event, and the receptive field stays sharp.
4. **The overshoot brake.** $\mathrm{FE}$ peaking at $\theta/2$ rather than at threshold
   means a cell that is *massively* over-driven learns less, not more. An incumbent winning
   by a large margin slows its own further specialization, which leaves room for turnover.

The cost is the flip side of the same property: **a weight parked in a tail is slow to
un-learn.** Recovery from a badly-committed assignment is genuinely slow, and $B$ sets that
trade — larger $B$ narrows the peak (faster commitment, more rigid tails), $B = 0$ makes
$\mathrm{FE} = \mathrm{FES} = 1$ everywhere and is the flat-plasticity negative control.

---

## 5. Weight initialization — every cell class

**Every plastic weight is initialized by where it sits on its own $\mathrm{FES}$ bell, not
by a shared constant.** Pattern detectors start exactly *at* the peak — maximum plasticity,
free to move either way. The coincidence basal starts deliberately *below* its peak, on the
rising flank, so it accelerates through maximum plasticity on its way to the one-shot
target and decelerates as it arrives. `Eor` is the exception that proves the rule: it is not
a learner at all, so it is placed directly at its final value and frozen.

Values below are the live `tiled_cc` values at $\theta = 1000$, dual FE/FES on
(`SimulationEngine(seed=1, **DASHBOARD_OVERRIDES)`).

| Cell | Afferents | Init | Jitter | Ceiling | Learns |
|---|---:|---|---|---|---|
| **L1 ordinary E** | 9 RGC (its patch) | $\theta/4 = 250$ | $\pm4\%$ | $\theta/2 = 500$ | yes |
| **L2 ordinary E** | 9 child `Eor` | $\theta/4 = 250$ | $\pm4\%$ | $\theta/2 = 500$ | yes |
| **`Eor`** (any layer) | $N=8$ local E | $\theta = 1000$ | **none** | $\theta = 1000$ | **no — frozen** |
| **`C` basal** | 1 (`Eor`) | $\theta/4 = 250$ | none | $\theta = 1000$ | yes |
| **`C` apical** | $N$ from parent E | unweighted | — | — | structural |
| **`I`** | — | no weight; $\theta_I = \theta/3$ | — | — | no |
| **RGC** | — | exogenous source, no weight | — | — | no |

$$
w^{(0)}_i \;=\; \max\!\Big(\tfrac{\theta}{4}\cdot U(0.96,\,1.04),\ w_{te}\Big)
\qquad\text{(ordinary E, per afferent)}
$$

### 5.1 Why $\theta/4$ for pattern detectors

$\mathrm{FES}$ peaks exactly where $2w/\theta = 1/2$, i.e. at $w = \theta/4$. So the
initialization *is* the peak of the plasticity curve — every detector starts maximally
willing to learn and can move in either direction with equal ease.

It also sets a deliberate starting competence. Nine afferents at $\theta/4$ sum to
$2.25\,\theta$, but only the afferents that are actually active deliver. A three-pixel
canonical pattern delivers $3 \times \theta/4 = 0.75\,\theta$ — **sub-threshold**. A fresh
competitor therefore cannot fire on one volley; it must integrate across two boundaries.
Only as its active weights grow does it *mature into a one-volley integrator*. Maturity is
an earned property, not an initial condition, and it is directly measurable:

$$
\text{mature} \iff \sum_{i \in \text{active}} w_i \;\ge\; \theta
$$

The $\pm4\%$ jitter exists only to break exact ties between the eight identical
competitors so the first-spike race has a definite winner. It is narrow enough that it
does not impose a feature preference.

### 5.2 Why `Eor` is initialized at $\theta$ and frozen

`Eor` is **not** a pattern detector. It relays whichever ordinary E won its column's WTA,
and it fires on a *single* afferent. Two consequences:

- **Init at $\theta$** makes any single winner drive it to threshold from the very first
  boundary — it starts already mature, at the same $\theta$ its ceiling holds it to.
- **Frozen** because a plastic bank starting at its ceiling could only move *down*, and the
  signed rule depresses every non-participating afferent. Afferents of ordinary E that have
  not recently won would decay toward zero, so a newly recruited owner would win its
  column's WTA and then fail to drive `Eor` — silencing the column's output during exactly
  the turnover the hierarchy exists to express.

Only `Eor`'s own incoming bank is frozen. `Eor → parent E` is a parent-owned plastic
detector weight, and `Eor → C` is the `C`-owned learned basal.

The freeze is applied *after* the seeded row is drawn and discarded, so toggling
`eor_w_init_frac` changes only `Eor`'s values and leaves the RNG cursor — and therefore
every other cell's initialization — untouched.

### 5.3 Why `C` starts at $\theta/4$ with a $\theta$ ceiling

`C` starts at $\theta/4$, which is *not* its $\mathrm{FES}_C$ peak ($\theta/2$) — it starts
on the rising flank at $\mathrm{FES}_C = 0.762$ and climbs **through** the peak toward the
one-shot target $\theta$, where $\mathrm{FES}_C$ has fallen back to $0.445$. So the cell
accelerates into learning and then decelerates as it approaches one-shot capability,
saturating just below/at the ceiling instead of overshooting it.

The rate is set so `C` matures *shortly after* the ordinary-E pool, never before: `C`
confirms an owner the pool has already settled on, so a `C` that matured first would be
confirming an unstable winner. That is what $\eta_C = 16$ buys against $\eta = 4$.

`C` also carries a second, non-binding bound: `w_max` $= 4\theta$ under the dual rule. That
is a manual-edit clip only — the binding ceiling is the structural `w_cap` $= \theta$.

---

## 6. Structural ceilings

The dual rule itself has **no upper cap** (floor $w_{te}$ only). Two hard per-synapse
ceilings are applied on top, by role, after every update:

$$
w_i \le \tfrac{\theta}{2} \ \ \text{(pattern detectors)},
\qquad
w \le \theta \ \ \text{(one-afferent \(\mathrm{Eor}\) bank and \(C\) basal)}
$$

**$\theta/2$ for detectors** is an evidence-integration constraint, not a tuning defect. No
single afferent can drive a detector across threshold in one event, so every detector is a
true integrator needing $\ge 2$ coincident afferents. Two afferents at the ceiling sum to
exactly $\theta$.

**$\theta$ for the one-afferent cells** is the analogue: `Eor` relays one winner and a `C`
deposit is its lone basal event, so a $\theta/2$ ceiling would leave them structurally
unable to fire at all. At $\theta$ one afferent reaches threshold exactly and can never
overshoot it.

---

## 7. Inhibition — two distinct forms

The tiled column uses hard resets, not persistent conductance:

**Lateral WTA (`E → I → E`), zero latency.** The first eligible ordinary-E crossing recruits
its column's `I` at that $\tau$; `I` hard-resets the *entire* local ordinary-E bank —
including the winner, whose already-emitted spike and its learning survive its own reset —
and cancels every later prospective crossing in that column. `I` emits at most one *WTA*
volley per boundary: a second same-boundary input creates no burst and no second reset.

**Top-down confirmation (`C → I → E`), delay one.** When `C` fires — its pattern has been
confirmed at the parent level — it schedules a hard reset of its own column's E bank for
the *start* of the next boundary, applied after that boundary's drive packet is frozen so
the packet is discarded too. A same-boundary reset could not do this: a $\tau \approx 0$
crosser has already fired and reset itself. This suppresses the trained integrator's
redundant re-fire and drops a confirmed column toward frequency halving.

This confirmation volley is deliberately **not** swallowed by the once-per-boundary WTA
guard above — `I` may already have fired its immediate lateral reset this boundary, and the
top-down volley is a distinct, deferred event. With `c_feedback_reset = False` the `C → I`
trigger falls back to the guarded path and is a no-op.

Persistent inhibitory conductance still exists in the engine for historical and custom
graphs, but it is not the tiled column's mechanism.

---

## 8. Feedback cadence and input pacing

Each hop costs a boundary, and the confirmation is always downstream of the spike it would
suppress. Define loop latency $L$ = boundaries from an ordinary-E spike to the feedback
reset landing on its own bank:

$$
L \;=\; h + 1
$$

where $h$ is the number of feedforward hops from a child ordinary E to a parent ordinary E.
For `tiled_cc` that path is $E \to \mathrm{Eor} \to \mathrm{parent}\,E$, so $h = 2$ and
$L = 3$. The engine derives $L$ from the graph, never from a preset name.

Two measured laws, confirmed by construction with the diagnostic `tiled_cc_double_eor`
preset ($L: 3 \to 4$):

$$
\text{period} = 2L \quad\text{(at a volley every boundary)},
\qquad
\text{suppression bites} \iff L \bmod \texttt{input\_period} = 0
$$

Setting $\texttt{input\_period} = L$ makes each volley's confirmation land exactly on its
successor's drive packet and cancel it, forcing exact fire/silent alternation independent of
loop depth (strict `1010` on 12/12 seeds at $L = 2, 3, 4$). The condition is sharp:
$\texttt{input\_period} = L+1$ gives no suppression at all.

**`input_period = 0` (the default) derives $L$ from the graph**, so pacing re-tracks on any
topology change. Physically this is one presentation per *resolved causal chain*: a real
cortical loop settles far faster than the input changes, so the overlapping-wave regime at
`input_period = 1` is an artifact of the unit-delay discretization. Full record:
`docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`.

---

## 9. Reference parameter set

The live dashboard/experiment contract (`backend/dashboard_config.py:DASHBOARD_OVERRIDES`
plus the engine defaults it relies on):

| Parameter | Value | Role |
|---|---|---|
| `dual_fe_fes` | `True` | the learning rule above |
| `dual_fe_e`, `dual_fe_wte` | $0.001$ | FE / FES tail floors |
| `dual_fe_B` | $5.0$ | bell sharpness |
| `eta` | $4.0$ | ordinary-E / `Eor` rate |
| `c_eta` | $16.0$ | `C` basal rate (matures shortly after the E pool) |
| `leak_rate` | $0.0$ | pure integrator, $g_L = 0$ |
| `refractory_steps` | $0$ | — |
| `input_period` | $0$ | auto: one volley per resolved chain ($L=3$) |
| `e_weight_cap_frac` | $0.5$ | detector ceiling $\theta/2$ |
| `relay_weight_cap_frac` | $1.0$ | one-afferent ceiling $\theta$ |
| `eor_w_init_frac` | $1.0$ | `Eor` bank at $\theta$ |
| `eor_plasticity_enabled` | `False` | `Eor` bank frozen |
| `c_feedback_reset` | `True` | delay-one `C → I` confirmation reset |
| `e_threshold` | $1000$ | $\theta$ (and $\theta_I = \theta/3$) |
| `cc_e_count` | $8$ | competitors per column |

Optimistic per-update bound, useful for sanity-checking acquisition counts: since
$\mathrm{FE} \le 1$, $\mathrm{FES} \le 1$, $|s| = 1$ and $\varphi \le 1$,

$$
|\Delta w_i| \;\le\; \eta \cdot \varphi_i \;\le\; \eta
$$

so a detector weight moving $\theta/4 \to \theta/2$ needs at least
$\lceil 250 / 4 \rceil = 63$ firing events even in the best case. Lowering $B$ cannot make a
weight arrive in two or three updates.

---

## 10. Other built-in topologies

`tiled_cc` is the reference. The remaining built-ins share every mechanic above and differ
only as noted:

| Preset | Nodes / edges | Difference from `tiled_cc` |
|---|---:|---|
| `tiled_cc_l1_4` | 155 / 620 | four ordinary E per L1 column, eight in L2 |
| `tiled_cc_direct_identity` | 181 / 1546 | **no `Eor`**: each ordinary E addresses every parent E directly; each `C` is multi-basal (one weight per local E). See `docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md` |
| `tiled_cc_double_eor` | 201 / 1062 | diagnostic only: one extra output relay in series ($L: 3 \to 4$) |
| `two_tower_composition` | 393 / 2162 | two 9×9 towers on one 9×18 sheet → two L2 → one L3. See `docs/TWO_TOWER_COMPOSITION.md` |
| `rg_direct_cc4` | 14 / 44 | minimal 3×3 column: four competitors + one WTA `I`, no `Eor`/`C`/feedback |
| `rg_coincidence` | 45 / 196 | the original 3×3 coincidence/turnover circuit |

---

## 11. Scope limits

Stated plainly, because they bound what any result here can mean:

- **One winner per column per boundary.** Multi-winner composition, $k$-WTA, `delta_tau`
  co-winner admission and lateral inhibition are out of scope; see
  `docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`.
- **Boundary-synchronous propagation.** Causal timing is analytic *within* a boundary, but
  every projection still costs a whole boundary. Network-level `dt` refinement is
  unsupported.
- **The scheduler scans all membranes** for the next crossing rather than using a priority
  queue. Correct and deterministic, but not a pure discrete-event simulator. The specified
  compatibility-first replacement is `docs/NEXT_EVENT_ENGINE_TECHNICAL_SPEC.md`.
- **Local capacity is $N = 8$** ordinary competitors per column against a tested vocabulary
  of four patterns; single-winner WTA does not by itself implement simultaneous
  multi-feature composition.
- **One `Eor` per column collapses owner identity into one output channel.** At L1→L2 this
  is partly repaired because the parent sees *which* children fired; where a parent has only
  two children it is not repaired at all — measured and preserved in
  `docs/TWO_TOWER_COMPOSITION.md`.
