# Implementation methodology and equations

## Current implementation snapshot (2026-07-28)

The live network is defined by a validated **`NetworkSpec`** of typed nodes and edges in
`backend/network_spec.py`. Neuron and dendritic behavior lives in `snn/neurons.py`;
graph construction, event delivery, analytic crossing-time scheduling, learning dispatch,
runtime pacing, and serialization live in `backend/simulation.py`.

The supported built-in presets are:

| Preset | Nodes / edges | Current role |
| --- | ---: | --- |
| `rg_coincidence` | 45 / 196 | validated 3×3 coincidence/turnover circuit |
| `tiled_cc` | 191 / 1052 | classic 9×9 tiled hierarchy; dashboard default |
| `tiled_cc_l1_4` | 155 / 620 | classic hierarchy with four L1 competitors |
| `tiled_cc_direct_identity` | 181 / 1546 | Eor-less source-addressed hierarchy |
| `tiled_cc_double_eor` | 201 / 1062 | diagnostic feedback-latency probe only |
| `rg_direct_cc4` | 14 / 44 | minimal direct four-competitor column |
| `two_tower_composition` | 393 / 2162 | two 9×9 towers feeding one L3 composition column |

All seven built-ins are event-resolved. Integer outer boundaries still carry delay-one
feedforward, basal, and feedback-reset events, but membrane crossings are resolved at
analytic sub-boundary timestamps `tau`. The scheduler repeatedly selects the earliest
crossing, advances the membranes, applies same-`tau` apical and hard-reset consequences,
and drains dependent events available exactly at `tau=1.0`. Custom graphs without an
event-resolved archetype or hard-reset edge retain the legacy synchronous path.

The current tiled path uses two distinct forms of inhibition:

- ordinary `E→I→E` WTA is an immediate same-`tau` hard reset of the local ordinary-E bank;
- `C→I→E` confirmation schedules a delay-one boundary-start hard reset, applied after the
  next drive packet is frozen so that the packet is discarded.

Persistent inhibitory conductance remains available to historical/custom graph mechanics,
but it is not the tiled column's WTA or confirmation mechanism.

Production ordinary-E learning has a cap-free base equation with a zero floor and
neuron-wide free-energy budget. The tiled dashboard/scaling contract then applies
structural per-synapse ceilings by role: `theta/2` for pattern detectors and `theta` for
the one-afferent Eor and C-basal relays. The generic engine default leaves the optional
detector ceiling unset. Classic Eor is initialized at `theta` and frozen. Direct-identity
removes Eor, transmits the ordinary-E winner address directly to the parent, and gives C
one causal-source-specific basal weight per local winner.

The editor vocabulary currently contains eleven archetypes and ten edge kinds, as declared
by `ARCHETYPES` and `EDGE_KINDS` in `backend/network_spec.py`; those registries, rather than
the historical lists below, are authoritative.

## Historical predictive-inhibition lineage (retained evidence)

The following sections through the older RG/predictive-inhibition results document removed
or low-level experimental graphs (`pi`, `old`, `rg`, `rg_residual`). Their builders and
mechanics remain useful to tests and custom graphs, but they are not accepted built-in
`topology=` values and their old configuration lists are not the dashboard contract. The
current tiled-family methodology resumes at “Tiled cortical columns.”

## Populations and topology

| Population | Count | IDs | Type | In topology |
| --- | --- | --- | --- | --- |
| RG (retinal ganglion sources) | 9 | `RG0..8` | S | rg, rg_residual |
| L1E_s (sensory source) | 9 | `L1E0..8` | E | pi, old |
| L1E (plastic noncompetitive encoder) | 9 | `L1E0..8` | E | rg, rg_residual |
| ErrorE (residual sheet) | 9 | `ErrorE0..8` | E | rg_residual only |
| L2E (competitors) | 8 | `L2E0..7` | E | all |
| L2I_WTA (winner-take-all relay) | 1 | `L2I` | I | all |
| PI (predictive interneurons) | 8 | `PI0..7` | I | pi, rg_residual |
| L1I (paired relays) | 9 | `L1I0..8` | I | old, rg |
| SwitchI (local incumbent gates) | 8 | `SwitchI0..7` | I | rg_residual only |

`L1E0..8` is the same *id* in all four presets but not the same *archetype*: in
`pi`/`old` it is an `e_sensory` cell with one fixed, non-plastic external afferent; in
`rg`/`rg_residual` it is an `e_encoder` — a plastic noncompetitive accumulator whose only afferent is
a learned `RG_i → L1E_i` synapse.

**PI-preset edges — 168 total:** 72 feedforward `L1E_s→L2E` · 8
`relay_excitation L2E[j]→PI[j]` (paired 1:1) · 8 `relay_excitation L2E→L2I` · 72
`predictive_inhibition PI[j]→L1E_s[i]` (candidate, locally plastic) · 8
`inhibition L2I→L2E` (WTA conductance).

**Old-preset edges — 169 total:** 72 feedforward `L1E_s→L2E` · 72
`relay_excitation L2E→L1I` (DENSE, every L2E→every L1I) · 8 `relay_excitation L2E→L2I`
· 9 `inhibition L1I[i]→L1E_s[i]` (paired) · 8 `inhibition L2I→L2E` (WTA conductance).

**RG-preset — 36 nodes, 178 internal edges:** 9 RG + 9 L1E + 9 L1I + 8 L2E + 1 L2I.
Edges: 9 feedforward `RG_i→L1E_i` (plastic, paired 1:1) · 72 feedforward `L1E→L2E`
(dense) · 8 `relay_excitation L2E→L2I` · 8 `inhibition L2I→L2E` · 72
`relay_excitation L2E→L1I` (DENSE) · 9 `inhibition L1I[i]→L1E[i]` (paired). The
cortical half is byte-for-byte `old`'s. External pixel presentation drives each RG cell
and is not serialized as a tenth edge category. There are no PI cells and no
predictive-inhibition edges in `rg`.

**RG-residual preset — 52 nodes, 274 internal edges:** 9 plastic paired `RG→L1E` ·
9 fixed paired `L1E→ErrorE` · 72 plastic dense `L1E→L2E` · 8 paired `L2E→PI` ·
72 learned `PI→ErrorE` inhibitory outputs · 72 dense `ErrorE→SwitchI` broadcasts ·
8 paired `L2E→SwitchI` trace events · 8 paired `SwitchI→L2E` inhibition · 8
`L2E→L2I` · 8 `L2I→L2E`. No inhibitory edge targets RG or L1E.

Each `SwitchI_j` is a numerically charged two-branch interneuron. Every ErrorE event
adds `0.55 θ_I` to its residual branch; the branch saturates at `0.90 θ_I`, so even
repeated residual activity alone cannot spike it. Its paired local trace `x_j` decays
as `x_j ← 0.97 x_j`; once `x_j ≥ 0.5`, it opens priming charge up to `0.90 θ_I`, also
strictly subthreshold alone. The visible SwitchI potential is the sum of these two
branch charges, and firing requires both branch predicates plus `V ≥ θ_I`.

Residual broadcasts are evaluated against the trace carried into the boundary.
Coincidence fires the switch, consumes `x_j`, and schedules paired L2 inhibition for
the next boundary. Only after that resolution does a current real `L2E_j` spike set
`x_j ← 1` for future boundaries. Thus residual alone, trace alone, and a brand-new
same-boundary winner are insufficient; no global winner id is consulted. Dynamic
state exposes `residual_events`, `residual_charge`, `trace_charge`, and `winner_trace`
so arriving ErrorE events visibly charge SwitchI even when it does not fire.

The four center-crossing patterns on the 3×3 surface: `row 1`, `col 1`, `diag \`,
`diag /`.

## Membrane dynamics: conductance-based joint integration

Every excitatory neuron is a conductance LIF unit. Per boundary it **gathers** all
excitatory charge `Q_exc` and all inhibitory conductance `g_inh`, then integrates
**once**, combining leak, inhibition, and excitation *before* the threshold test
(never leak → threshold-sized jump → test → inhibit). With `C = 1`, `dt = 1`,
`E_L = V_rest = 0`:

```text
C dV/dt = -g_L (V - E_L) - g_inh (V - E_inh) + I_exc,     I_exc = Q_exc / dt
g_total = g_L + g_inh
g_total == 0:  V <- V + Q_exc                              (pure integrator)
else:          V_inf = (g_L·E_L + g_inh·E_inh + I_exc) / g_total
               V     <- V_inf + (V - V_inf)·exp(-g_total·dt)
```

* **Baseline leak conductance** `g_L = -ln(1 - leak_rate)` (`leak_to_conductance`),
  so with no inhibition and no input the update reduces *exactly* to the historical
  `V ← (1 - leak_rate)·V` — the documented migration path from the old per-step leak.
* **Inhibitory reversal** `E_inh = 0 ≤ V_rest` (shunting). A real inhibitory event
  raises `g_total`, pulling `V_inf` down and shunting the excitatory drive.
* **Persistent conductance.** An inhibitory event adds `g_inh += g_scale · w`;
  `g_inh` then **decays once per boundary** by a retention factor and **is not
  cleared by a voltage reset**. Firing resets `V` to rest but leaves `g_inh` and the
  activity trace intact.

**Behavioural invariant (measured).** Excitation of `1.5·θ` that crosses threshold
instantly with no inhibition (`V = 1477`, fires) stays sub-threshold when enough
`g_inh` is already present in the same boundary: `g_inh=2 → V=642` (no fire),
`g_inh=6 → V=248`, `g_inh=12 → V=125`.

### Inhibitory-conductance decay is split by target population

Retention is per-target so the two inhibitory roles have **independent timescales**
(a required separation — otherwise a persistent WTA pulse alone drives turnover with
no predictive inhibition at all):

* `alpha_inh` (default **0.6**) — decay on **L2E** (the `L2I_WTA` target). Fast, so
  winner-take-all is a clean single-winner suppressor that does not itself cause
  turnover.
* `alpha_inh_l1` (default **0.95**) — decay on **L1E_s / L1E_new** (the predictive PI
  / legacy L1I target). This is the symmetry-breaking lever (see results).

## Local post-synaptic activity trace

Every excitatory neuron carries a local "calcium" trace `a` that **survives voltage
reset**, so a cell that fired and reset earlier in the interval still registers as
recently active when a PI cell later reads it:

```text
depol = clip((v_pre_reset - V_rest) / (threshold - V_rest), 0, 1)
a <- clip(alpha_a · a + beta_v · depol + beta_s · spike, 0, a_max)
```

`v_pre_reset` is the post-integration membrane captured **before** any reset.
Defaults `alpha_a = 0.85`, `beta_v = 0.30`, `beta_s = 1.00`, `a_max = 1.0`. The
trace means only "this cell was recently depolarized/firing"; it carries **no**
information about which afferent supplied the charge.

## The one accumulating excitatory weight rule (production: linear FE)

Runs when a plastic excitatory neuron fires, on `acc_weights` only. The production default
is the cap-free `linear_fe` base rule: the historical per-synapse quadratic multiplier and
universal `w_max` clip are absent, while the zero floor and neuron-wide maturity budget are
retained:

```text
p        = maturity_budget_frac*threshold - sum(acc_weights)  # pre-update, signed
signal_i = +1 if afferent i spiked in the causal volley else -1
delta_i  = eta · p · signal_i · distance_factor_i
w_i      = max(w_i + delta_i, 0)                      # base rule: floor only
```

Geometry is a per-synapse learning-rate multiplier only; it never scales delivered
charge. `p` is signed, so a row self-limits as its total approaches the maturity budget.
An independent structural `w_cap`, when configured, is applied after the rule in every
mode: the dashboard uses `theta/2` for pattern detectors and `theta` for one-afferent
relays. This architectural ceiling must not be confused with the removed universal
learning-rule cap.

The historical `linear_bounded` and `quadratic_bounded` modes remain headless regression
controls. The experimental dashboard can instead select the dual FE/FES rule; the same
role-specific structural ceiling is still applied after that update.

## Strictly-local predictive-inhibition plasticity

Each candidate synapse `PI[j] → L1E_s[i]` owns a nonnegative weight `w_ji`. On a real
presynaptic PI event (its paired `L2E[j]` won), the update is **element-wise local** —
entry `i` reads and writes only `w_ji` and its own target's trace `a_i`:

```text
w_ji <- clip(w_ji + pi_eta · a_i · (pi_w_max - w_ji), 0, pi_w_max)
```

No synapse sees another target's trace, the full L1 spike vector, a pattern label, or
a central winner-row. The emitted inhibitory pulse uses the **pre-update** weight
(`g_inh += pi_g_scale · w_ji`), so learning changes only future predictions. Weights
start at **zero** (a first-seen pattern teaches the synapses; it is not pre-suppressed).
A slow passive decay `w_ji ← w_ji·(1 - pi_lt_decay)` per boundary gives recovery from
stale associations (local: each weight decays from itself).

**Required timescale separation** (all exposed and independently controllable):
activity-trace decay `alpha_a`; inhibitory **association** rate `pi_eta` (slow, so one
overlapping presentation does not let an incumbent learn every novel feature);
inhibitory **expression** magnitude `pi_g_scale` (immediate once a synapse matures);
inhibitory conductance **decay** `alpha_inh_l1`; long-term synaptic decay `pi_lt_decay`.

## L2 winner-take-all

`L2I_WTA` is a deterministic single-winner selector, kept separate from the eight PI
cells. Among the L2E threshold-crossers in a boundary, the highest membrane wins
(tie-break: lowest index); **only** the winner fires and learns. Arbitration is
**selection, not charge removal** — losers keep their membrane charge and are instead
suppressed by the WTA **conductance** pulse `g_inh += l2i_g_scale` delivered to every
L2E on the **next** boundary. The winning spike therefore precedes the feedback
inhibition it causes; `L2I_WTA` cannot retroactively cancel its own winner.

## Synchronous timestep and explicit delays

Every internal projection has an integer synaptic delay of **1**; relays
(`L2I_WTA`, `PI`, `L1I`) fire in the same boundary as their source spike and schedule
their inhibitory **conductance** output for the next boundary; external input arrives
at the current boundary. Each boundary runs these subphases in order (no behaviour
depends on Python neuron-iteration order, because every target integrates once from
double-buffered arrivals):

1. Deliver arrivals scheduled at `t-1` (inhibitory conductance first, then excitatory
   charge) and deposit external input.
2. Integrate every excitatory neuron once (joint exc/inh).
3. Threshold test + fire: exogenous RG sources assert their spike (rg); L1E_s
   crossers; plastic `e_encoder` crossers (rg — every crosser fires and learns its own
   delivered volley, no WTA); deterministic L2E WTA (one winner fires + learns
   feedforward).
4. Update every excitatory neuron's local activity trace.
5. Emit spikes into delay-1 queues: L1E_s→L2E feedforward (+ enew local sensory);
   winner→`L2I_WTA` conductance (all L2E) and, per topology, winner→paired `PI`
   (direct) or winner→dense feedback to `L1E_new` (enew); PI/L1I relays run their
   **local plasticity** now (reading the traces finalized in step 4) and schedule
   their inhibitory conductance for `t+1`.
6. Decay each `g_inh` once; count down refractory; PI passive weight decay.
7. Serialize the frame.

### The `rg` two-hop chain

`rg` has two feedforward hops, so the same delay-1 rule produces:

```text
t    : active RG_i emits (exogenous; nothing in the cortex can veto it)
t+1  : RG_i -> L1E_i charge arrives; L1E_i integrates it jointly with any L1
       inhibitory conductance; every L1E crosser fires and learns its OWN RG
       afferent; its L1E -> L2E events are queued
t+2  : L1E -> L2E charge arrives; L2 WTA selects at most one crosser; the winner
       learns only from the L1 afferents delivered to IT in this volley, and drives
       L2I plus all nine L1I relays
t+3  : L1I -> L1E and L2I -> L2E conductance arrives, before integration
```

RG keeps emitting at `t+1`, `t+2` and onward while the edge is held; queued RG events
are never cancelled by a cortical winner. Feedforward dispatch is generic: any
permitted source spike schedules weighted charge to its plastic targets for the next
boundary, and **causal participation is recorded per postsynaptic target per arrival
boundary**, so an L1E update can never see an L2E's volley (or an adjacent boundary's).
Measured on the live engine: RG at `t=1`, L1E at `t=2`, L2E winner + all nine L1I at
`t=3`, inhibitory pulses at `t=4`.

## RG semantics (`topology='rg'`)

An RG cell is a real, visible network node — it appears in dynamic state, raster,
firing-frequency, topology, renderer and emitted-edge views — but it is a
`SourceNeuron`, not a conductance LIF:

```text
RG_i_spike(t) = input_arrives(t) AND input_vec[i] > 0.5
```

* It owns no membrane, no `g_inh`, no refractory timer, and no learning rule, so
  there is literally nothing for L1I / L2I / PI / WTA to act on. Every edge kind's
  target rule already forbids an RG target, and `validate_spec` rejects such a graph
  with an explicit structural error.
* It does not learn. The plastic weight on the path is the **postsynaptic** `RG_i →
  L1E_i` afferent, owned by L1E.
* "Uninhibited" means uninhibited **by this modelled cortical feedback loop**. It is
  not a claim that the biological retina lacks inhibitory circuitry — retinal
  amacrine/horizontal inhibition is simply outside this model's scope.
* A held edge produces one RG spike per `input_period` boundary — no more, and no
  spontaneous background firing.

**Retinal evidence persisting is not the same thing as L1 continuing to spike.** In
`rg` the retina keeps delivering evidence while L1I shunts `L1E`; the *cortical* L1
cell still goes silent for the duration of the shunt. What the source layer buys is
that the evidence is still arriving when the shunt decays, rather than having been
consumed by a single external injection.

### L1E is layer-invariant with L2E

`L1E` in `rg` uses the *same* `ExcitatoryNeuron` class and the *same*
`update_acc_weights()` rule as `L2E`, with `participation = [True]` when its RG afferent
supplied the charge — and the update runs only when that L1E actually fires. It shares
L2E's excitatory threshold, resting potential/reset, baseline leak, refractory
behaviour, `eta`, positive weight cap, activity-trace equation, and accumulating-weight
implementation. A test pins this numerically: the encoder's post-spike weight is
*bit-identical* to a bare `ExcitatoryNeuron` configured as a competitor and given the
same participation.

The one thing that differs is inhibitory-conductance retention: L1E keeps
`alpha_inh_l1` (0.95) and L2E keeps `alpha_inh` (0.6). That is an **inhibitory circuit
timescale keyed to which relay population targets the cell** (L1I→L1E vs L2I→L2E), not
a second L1 excitatory learning rule, and it is exactly the split `old` already used.
It is not silently changed by adding RG.

L1E is **noncompetitive**: every threshold crosser fires in the same boundary. It is
not an `e_competitor` special-cased out of WTA by id or layer string — `e_encoder` is a
distinct archetype whose `wta` flag is false.

### RG→L1E initialization, cap, and developmental cadence

The built-in preset uses the shared seeded policy, not a hand-authored pixel-specific
pattern:

```text
FF_INIT_MEAN = 0.55 * theta / 9   = ~61.1     (the L2 per-afferent scale, literally)
jitter       = uniform(0.96, 1.04)            (the same seeded narrow jitter)
cap          = e_weight_cap = theta/2 = 500   (shared)
eta          = 0.01                           (shared)
```

With one afferent, `sum(w) <= 500 < theta`, so `p = theta - sum(w)` stays positive and
the weight rises monotonically toward the cap (under the production linear-bounded rule,
`dw = eta·p·s·influence`, clipped at `w_max`; the historical rule additionally damped
this near the cap with `(1-(w/w_max)^2)`).
**This is expected**: L1E is a temporal *accumulator* whose cadence accelerates during
training, not a one-event threshold relay. Analytic targets under `V_n = (w/g_L)·(1 -
(1-leak)^n)` with `leak=0.03`, `g_L = -ln(0.97)`:

| | analytic | measured |
| --- | --- | --- |
| first spike at `w≈61` | 23 active RG events | first L1E spike at boundary **24** (= 23 events + the delay-1 hop) |
| mature cadence at `w≈500` | ~1 L1 spike / 3 RG events | mean L1 ISI **3.5–3.7** boundaries |
| weight at cap | — | active channels reach **450–471** of 500; inactive stay at **~61** |

No recalibration was needed or applied: the built-in preset ships the shared
initialization, cap and `eta` unchanged. `enc_w_init` / `enc_init_jitter` /
`enc_plasticity_enabled` are explicit **projection-level** parameters used only by the
experiment's controls; they are never pixel-, pattern-, or winner-specific.

Causal reading — *first encounter:* `L1 activity → L2 winner → PI event → local
inhibitory learning` (the original L1 spike is **not** cancelled). *Later encounter:*
`L2 winner → PI event → persistent g_inh → a later L1 sensory interval is shunted`.

## Configuration

Editable keys (`apply_config` rebuilds; unknown keys rejected): `leak_rate`,
`refractory_steps`, `eta`, `e_weight_cap`, `input_period`, `topology`
(`'pi'`|`'old'`|`'rg'`|`'rg_residual'`), `alpha_inh`, `alpha_inh_l1`, `alpha_a`, `beta_v`, `beta_s`,
`a_max`, `e_inh`, `pi_eta`, `pi_w_max`, `pi_lt_decay`, `pi_g_scale`, `l2i_g_scale`,
`pi_conductance_enabled`, `pi_plasticity_enabled`, `enc_plasticity_enabled`,
`enc_init_jitter`, `enc_w_init`, `residual_exc_scale`, `switch_trace_decay`,
`switch_trace_threshold`, `switch_residual_charge_frac`, `switch_trace_charge_frac`,
`switch_g_scale`, `switch_conductance_enabled`. Arbitrary custom graphs are applied
via `apply_topology(spec)` / `POST /api/topology` (validated `NetworkSpec`), bypassing
the preset selector. Fixed/derived: `e_threshold=1000`,
`i_threshold=θ/3` (reported invariant), `synaptic_delay=1`, distance exponent 2.

## Symmetry-breaking results (overlap experiment)

`experiments/predictive_inhibition_overlap.py` runs a deterministic `row → column →
row` schedule (shared pixel 4; novel column pixels {1,7}) across seeds, with controls.
All behaviour is derived from ordinary input vectors; **no overlap pixel, pattern, or
winner is hardcoded** into any rule.

**Phase A** forms an incumbent whose paired PI learns output synapses **only** onto
the row's active pixels: across 5 seeds, exactly **3/72 candidate synapses are
nonzero** (the incumbent onto {3,4,5}, ~0.28 each), zero onto inactive pixels.

**Phase B** (switch to the overlapping column): the incumbent predicts and suppresses
the **shared** pixel 4 more than the **novel** pixels {1,7}
(`g_shared/g_novel ≈ 1.3–1.4`). With the shared feature shunted, the incumbent — whose
column drive was almost entirely pixel 4 — loses drive; a different competitor
accumulates the residual novel-feature activity and wins. Representative seed-1
history: incumbent `L2E0` wins ~1000 steps, then `L2E4` takes over (first rival win at
step ~76), with incumbent-PI contamination of {1,7} held low (~0.14).

**Phase C** (return to the row): the original detector reclaims the row in **100%** of
runs — temporary adaptation, not catastrophic erasure.

**Controls (5 seeds each), default parameters:**

| Condition | Symmetry break | Recover | Mechanism check |
| --- | ---: | ---: | --- |
| full (PI on, plastic) | **1.00** | 1.00 | works |
| predictive conductance OFF | **0.00** | 1.00 | expression is necessary |
| PI plasticity OFF | **0.00** | 1.00 | association is necessary |
| fast association (`pi_eta=0.10`) | 0.80 | 1.00 | more contamination (0.27) |
| slow association (`pi_eta=0.005`) | 0.00 | 1.00 | incumbent PI never matures in time |

A second overlap pair (`diag \ → diag /`, shared pixel 4, novel {2,6}) breaks in 2/3
seeds and recovers 3/3.

### The load-bearing parameter and its sensitivity

Symmetry breaking is gated by the **predictive conductance persistence
`alpha_inh_l1`**, not by conductance magnitude:

| `alpha_inh_l1` | break rate (8 seeds) |
| ---: | ---: |
| 0.60 (fast) | 0.00 |
| 0.85 | 0.00 |
| 0.92 | 0.12–0.25 |
| **0.95** | **1.00** |
| 0.97 | 1.00 |

The shared-feature shunt must persist across the rival's accumulation window
(~90 boundaries). Below ~0.92 it decays too fast and the incumbent always recovers
drive before a rival can accumulate; increasing `pi_g_scale` alone does **not** fix
this. `pi_eta` has a working band: too fast contaminates the incumbent's novel-feature
synapses and erodes reliability; too slow fails to mature the incumbent PI within the
window.

## RG timing/symmetry results (`experiments/rg_timing_symmetry.py`)

5 seeds × 3 schedules × 1500 boundaries per pattern phase. All conditions share
bit-identical L2E initialization at a given seed (the engine draws competitor jitter
before encoder jitter), so nothing below is a reshuffled L2 seed. Raw per-run counts
(boundaries, RG events, L1 spikes, L2 winner events) are in
`experiments/rg_timing_results.json`.

The two frozen controls are deliberately kept apart:

* **`rg_frozen`** freezes RG→L1E at the new ~61-unit init. It changes topology + delay
  **and** L1 cadence.
* **`rg_frozen_matched`** freezes RG→L1E at the old `SENSORY_WEIGHT` (θ/3 = 333),
  reproducing `old`'s per-event L1 charge. This isolates **only** topology + delay.

### Headline: the RG layer itself is behaviourally free; RG *plasticity* is a regression

`row 1 → col 1 → row 1`, means over 5 seeds:

| condition | first L1 | first L2 | L1 ISI early→mature | L1 sync (mature) | winner dominance | row/col owners distinct | recover | strong afferents / L2 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `old` | 4.0 | 35.4 | 3.77 → 8.14 | 1.00 | 0.67 | **1.00** | **1.00** | **0.42** | useful assembly symmetry breaking |
| `rg_frozen_matched` | 5.0 | 36.4 | 3.77 → 8.14 | 1.00 | 0.67 | **1.00** | **1.00** | **0.42** | useful assembly symmetry breaking |
| `rg_frozen` | 23.2 | — | 7.46 → 7.55 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | **developmental deadlock** |
| `rg_plastic` | 23.2 | 221.2 | 3.47 → 4.01 | 0.00 | 0.86 | 0.40 | 1.00 | 0.12 | **temporal phase breaking** |
| `rg_plastic_equal_init` | 24.0 | 219.8 | 3.49 → 3.83 | 0.00 | 0.74 | 0.20 | 0.60 | 0.10 | **temporal phase breaking** |
| `rg_plastic_symmetric` | 24.0 | 209.0 | 3.52 → 4.45 | 0.00 | 1.00 | 0.00 | 1.00 | 0.12 | **winner tyranny + temporal phase breaking** |

1. **`rg_frozen_matched` reproduces `old` exactly** on every symmetry measure
   (row/col distinctness 1.00, recovery 1.00, 0.42 strong afferents per L2, 75-boundary
   escape latency), with first-L1 at 5 vs 4 — precisely the one extra hop. So the RG
   layer's *structure and delay* cost exactly one boundary and change nothing else. This
   is the control that validates the implementation.
2. **`rg_frozen` deadlocks.** At `w≈61` frozen, L1 fires every ~7.5 boundaries, the
   three active channels desynchronize (dispersion 5.8 boundaries, sync 0.01), and L2
   **never fires at all** across every seed and schedule (0 L2 updates, 8/8 dead L2).
   Without coincident L1 arrivals there is nothing for a competitor to integrate, and
   leak drains the membrane between the staggered arrivals. This is **structural, not a
   horizon artifact**: at 20 000 boundaries on sustained `row 1` (4.4× the experiment's
   horizon) L1 has fired 2575 times and the best L2E membrane has still only reached
   **322.8 / 1000**. Lengthening training does not rescue it — the failure is that a
   frozen ~61-unit afferent cannot make L1 fast enough to produce L2 coincidence.
3. **`rg_plastic` bootstraps but degrades the science.** RG→L1E learns a genuine sensory
   selectivity — driven channels saturate to 450–471/500 by t≈3480, undriven stay at
   ~61 — and L1 cadence accelerates 23 → ~3.5, matching the analytic prediction. But
   relative to `old`: row/col owner distinctness collapses **1.00 → 0.40**, strong L2
   afferents per cell **0.42 → 0.12**, and 57% of L2 updates are driven by a **single**
   active feature. Exact-volley learning plus desynchronized L1 means the winner
   specializes to whichever channel escapes first. **`rg` is worse than `old` at the
   task `old` already does.**

### Does init jitter create artificial feature priority? No — the geometry does

`rg_plastic_equal_init` (all nine RG→L1E weights identical at init) behaves like the
jittered preset: same first-L1 (24.0), same saturation time (3480), same final weight
spread (438.8 vs 440.8), same verdict. **Weight jitter is not the source of L1 phase
splitting.**

The `rg_plastic_symmetric` condition removes the last per-synapse asymmetry — the 1/d²
geometric learning-rate factor, which still differs per channel because the L1E end of
the layout is jittered — and settles the attribution:

* Under **sustained** single-pattern drive, the three active channels become *perfectly
  locked* (sync **1.000**, identical weights 450.56/450.56/450.56, identical spike
  counts 99/99/99). The phase split under sustained drive is therefore **entirely a
  geometric artifact**: a fixed per-synapse learning-rate difference that the
  accumulating rule integrates into a weight difference and hence a cadence difference.
  It is not learned or dynamic symmetry breaking.
* Under a **changing** schedule the channels still desynchronize (sync 0.00) even when
  fully symmetric, because they accumulate different drive *histories*. That desync is
  genuinely dynamic — but it produces **winner tyranny** (dominance 1.00, dwell 45.4,
  row/col distinctness 0.00), not useful assembly formation.

### Verdict

Across all three schedules `rg` yields **temporal phase breaking, not useful assembly
symmetry breaking**, with substantial **single-feature collapse**, and it degrades the
row/column owner distinction that `old` achieves. The one condition that reproduces
`old`'s useful behaviour (`rg_frozen_matched`) is precisely the one where RG learns
nothing and delivers `old`'s charge — i.e. where the RG layer is a pure relabelling.

As predicted in the design, `rg` shows **no contextual explaining-away**, and it was
never expected to: the dense `L2E→L1I` feedback erases winner identity because every
winner drives every `L1I`.

## Tiled cortical columns (`topology='tiled_cc'`)

The tiled preset reuses every `rg_coincidence` mechanic — the conductance-LIF membrane,
the analytic sub-boundary event scheduler, the accumulating plastic-E rule, the
coincidence-cell basal/apical gate, and the immediate zero-latency hard reset — inside a
reusable **cortical-column tile** instead of one neuron per pixel. A `9×9` RGC surface is
tiled into nine `3×3` patches; each patch drives one L1 column (arranged `3×3`), and one
L2 column receives all nine L1 outputs. The graph is composed from four pure rules
(`build_cortical_column`, `connect_rgc_patch`, `connect_columns`, `tiled_cc_spec`) so the
ordinary-E count `N = cc_e_count` is configurable and deeper hierarchies compose without
copying a hard-coded graph. For any `N` the graph has exactly `10N+111` nodes and
`129N+20` edges (default `N=8` → **191 nodes, 1052 directed edges**).

**Eor is a membrane relay with a fixed afferent bank.** It still uses the ordinary
event-resolved excitatory membrane, threshold, leak, refractory, analytic crossing, trace,
and delay-one emission. Its synaptic role is now deliberately different: every local
`E→Eor` weight is initialized at `theta`, capped at `theta`, and frozen
(`eor_w_init_frac=1.0`, `eor_plasticity_enabled=False`). Any one local WTA winner therefore
drives Eor from the first boundary and forever after. This prevents the signed
participation rule from depressing inactive-owner afferents and stranding a newly recruited
winner. Only Eor's own incoming bank is frozen: `Eor→parent E` is a parent-owned plastic
pattern-detector weight, while `Eor→C` is the C-owned learned basal weight. Eor remains a
pooled “this column fired” message; use `tiled_cc_direct_identity` when the parent must
receive the local winner address.

Fan-in differs across the hierarchy (9 RGCs into an L1 E, `N` local E into Eor, 9 child
Eor into an L2 E). Ordinary detector rows retain seeded initialization and per-target
distance normalization; the fixed Eor override is applied only after consuming the same RNG
draws, so enabling the relay contract does not perturb other cells' initialization.

**The shared one-shot relay I gives immediate hard single-winner WTA.** Every ordinary E
drives its column's single I (`relay_excitation`) and is reset by it
(`hard_reset_inhibition`); the C also drives the same I. The first eligible ordinary-E
crossing recruits I at that `tau`; I hard-resets the *entire* local ordinary-E bank —
including the winner whose emitted spike (and its learning) survives its own reset — and
cancels later prospective crossings in that column. I emits **at most once per boundary**:
a later same-boundary E or C input creates no second reset. If C fires first it may recruit
the same I and suppress the local E population before it emits; once I has fired, a later C
spike cannot retroactively cancel an already-emitted E spike. There are no lateral or
cross-column edges, so columns are independent: one L1 column's winner never resets another
L1 column, and several columns may each produce one local winner in the same outer
boundary while each remains hard single-winner (`column_winners` records one ordinary-E
winner per column per boundary; the legacy single `winner` field is preserved for legacy
consumers).

**The top L2 C is explicitly valid and observably dormant.** Every column, including the
top one, contains a C. The L2 column has no parent, so its C has its single Eor basal edge
and its one-boundary basal eligibility state machine but **zero apical inputs**: with no
apical permission the coincidence gate is never open, so it never deposits charge, never
fires, and never learns its basal weight over an arbitrarily long basal-only run. This
zero-apical case is legal *only* because the column metadata declares no parent
(`has_parent=False` / empty `parent_ids`) — validation still rejects an accidentally
unwired non-top C, and `rg_coincidence` C cells still require at least one apical.

**Deferred by design.** This is a single-winner tiled hierarchy. More than one ordinary-E
winner per column per boundary, row-plus-column composition learning, `k`-WTA / `delta_tau`
co-winner admission, lateral inhibition, and a pure priority-queue discrete-event scheduler
are **out of scope** and remain recorded in
`docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`. The engine still scans all membranes
to find the next crossing rather than using a priority queue.

The deterministic headless acceptance evidence is `experiments/tiled_cc_experiment.py`
(a center-patch isolation probe demonstrating the full `RGC → L1 E WTA → Eor → L2 E WTA →
apical permission to all nine L1 C → gated deposit` chain with the top C dormant and zero
cross-column reset leakage, plus a two-patch probe confirming two independent L1 winners
under a single hard L2 winner). A dashboard screenshot is supplemental; the headless trace
is authoritative.

### Intentional θ/2 integration and the sparse-evidence halving limit

The dashboard configuration applies a deliberate per-synapse ceiling of
`e_weight_cap_frac = 0.5` to pattern-detector feedforward weights:

```text
w_i <= theta/2
```

This is an evidence-integration constraint, not a tuning defect. No single afferent can
drive a detector across threshold in one event. A detector must combine at least two
sufficient simultaneous afferents or accumulate repeated evidence over time when its leak
permits that accumulation. Eor and C basal weights retain their separate role-specific
semantics.

For one active L1 patch, L2 receives one active child-column afferent out of its possible
inputs. It can train slowly through repeated Eor events, but the intentional ceiling means
that it never becomes a one-event integrator for that lone child. L2 apical feedback to the
associated L1 C is therefore sparser than accepted L1 evidence. A mature C basal weight
only makes C responsive when apical permission arrives; it cannot increase the frequency
of L2 permission.

This creates an event-count limitation in the current feedback implementation:

```text
several accepted L1 events -> one L2 event -> at most one C event
                            -> at most one later suppressed L1 event
```

Exact alternating halving requires accepted and suppressed eligible presentations to
balance one-for-one. Longer training alone cannot manufacture missing L2/C confirmations,
and the `theta/2` constraint must not be removed to force the result.

The later cadence audit established an additional, independent timing effect. With a volley
every boundary, the confirmation loop produces `L` fires followed by `L` silences, and
suppression is effective only when the reset happens to land on a drive packet. The
dashboard now uses graph-derived auto-pacing (`input_period=0`, resolved to the loop latency)
so each confirmation lands on its successor presentation and forces presentation-level
alternation. This resolves reproducible cadence under the model's paced-input abstraction;
it does not prove that cadence encodes certainty under arbitrary or interrupted input.

The remaining scaling experiment keeps the graph, cap, thresholds, learning rules, auto
pacing, and feedback path fixed while sweeping one through nine active patches. It must
distinguish whether exact alternation begins only after C is meaningfully confirmed from a
trivial pacing artifact that alternates any mature loop.

Evaluation must distinguish:

```text
101010...   exact alternating halving; required certainty signal
111000...   average 0.5 but grouped; not accepted as halving
irregular   reduced activity only, even if its long-run mean is near 0.5
```

No new topology is approved for this issue. Any later mechanism that makes suppression
robust to irregular timing must preserve the intentional integration ceiling and existing
sparse column graph; the standing candidate is for a prediction to be consumed by the next
eligible evidence event rather than by a wall-clock boundary.

## Rejected direction: per-feature gated tiled columns

**Decision:** the dedicated `tiled_cc_feature_gated` topology was implemented and tested,
but it is not the selected architecture and is being removed from the supported preset
surface.

The experiment inserted a separate fixed relay, coincidence cell, and inhibitory relay
(`S/C/If`) for every input feature in every L1 receptive field. When an established local
owner predicted an active feature, that feature's C/I chain suppressed only its paired
relay. Novel feature relays remained active. In the tested seed-1 protocols this produced
clean local frequency alternation and allowed changed patterns to recruit different
ordinary-E owners. The experiment therefore established that selective feature suppression
can cause turnover; it was not rejected because the mechanism failed to operate.

It was rejected because it solves the problem at the wrong structural and representational
level:

* **Topology density grows too quickly.** The local-only version replicated nine `S/C/If`
  chains in each of nine receptive fields. Applying the same motif between competition
  layers would require identity relays and paired C/I gates for every child competitor,
  making inter-column connectivity substantially denser at each hierarchical boundary.
  This works against the intended sparse cortical-column design.
* **Competitive learning already allocates changed patterns.** An ordinary E that has
  specialized for one pattern becomes less competitive for a sufficiently different
  pattern under the existing weight-distribution and fullness-error dynamics. Continued
  presentation can therefore allow a different neuron to acquire the new pattern without
  reproducing a gate for every input feature.
* **The desired frequency signal is column-level certainty.** Frequency halving is intended
  as groundwork for certainty and attention guidance. A highly active region should mean
  that it is actively learning, actively being used, or being held at high gain by guided
  attention. A slower-firing region should mean that its current pattern is already learned
  with greater certainty. Per-feature explaining-away does not directly encode that
  region-level state.
* **Suppressed evidence must become absent evidence.** When feedback suppresses a confirmed
  presentation, the column should emit no winner/output evidence for that presentation.
  Downstream layers should observe a genuine missing event, not a feature-expanded alternate
  representation and not merely a lower average caused by an unrelated feedback-loop
  latency.

No replacement topology is approved. The immediate direction is to characterize scaling in
the existing tiled graph with the intentional `theta/2` detector ceiling unchanged. The
desired future semantics remain sparse and column-level: once a local owner is confirmed,
eligible presentations should alternate `fire, silent, fire, silent`, and a suppressed
presentation should emit no evidence. A changed pattern should be handled by the existing
competitive population rather than per-feature gates. This describes the target semantics
only; neither the rejected experiment nor the present feedback loop proves that exact
single-patch alternation has been implemented.

## Direct-identity tiled columns (`topology='tiled_cc_direct_identity'`)

The `Eor` output relay is removed and each ordinary-E winner's **identity** is transmitted
directly: every child ordinary E projects to every parent ordinary E, so a parent detector
owns one plastic weight per `(child column, child winner)` source address rather than a
single pooled "this column was active" event. 181 nodes, 1546 edges at 8 E per column.

Two equations change shape; none change form.

**Multi-basal coincidence gate.** A column C now owns one learned basal afferent per local
ordinary E. With `k` the causal basal source (a current event preferred over a carried one;
within each, the earliest delivered):

```text
B = current OR one-boundary-carried basal event on ANY source
A = any current apical event
q = w_k * s_k        deposited at most once per boundary iff B AND A
```

**Causal-source-only basal learning.** The C update is the unchanged learning family
applied at index `k` alone:

```text
w_k  <- clip(w_k + dw(w_k, s_k, phi_k), 0, theta)
w_j  <- w_j          for every j != k        (exactly unchanged, no participation term)
```

There is deliberately **no** `+1/-1` participation term across the basal vector. That term
is what made `Eor` undeliverable: it depresses every non-participating afferent, so an
owner that has not recently won loses the weight it needs to be heard. Measured
(seed 1, patch (1,1), 2500-boundary phases): the `row 1` owner's basal weight moved
`250.00 -> 613.80`, and while the `col 1` owner then matured `250.00 -> 612.30`, the first
owner's weight changed by **exactly 0.0**.

Apical permission is unchanged: unweighted, Boolean `any(apical)`, with source identity
retained only as a diagnostic. The dendritic orientation is unchanged — bottom-up/local
evidence is basal, top-down feedback is apical.

The full contract, protocol, results and negative findings are in
`docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md`. The topology exposed a boundary-edge scheduler
defect: with the mandated `theta/2` detector ceiling and zero leak, two coordinated child
identity events deliver exactly `theta`, so L2 and its dependent C crossing can occur at
`tau = 1.0`. The event loop now drains all crossings already available at that edge. C
therefore fires, learns, and schedules feedback; a sub-threshold cell still reports
`inf` because no interval remains. Regression coverage is
`tests/test_boundary_edge_crossing.py`.

## Feedback cadence: loop latency and input pacing

The top-down `C -> I` confirmation is not instantaneous. Each hop costs a whole boundary
(`SYNAPTIC_DELAY = 1`), and the confirmation is always downstream of the spike it would
suppress — measured 450/450 boundaries, the C fires at `tau = 1.0` while its column's E
already fired at `0.667`, because that E spike IS the evidence that travels up. Define the
loop latency `L` as boundaries from an ordinary-E spike to the feedback reset landing on its
own bank:

```text
child E --(h feedforward hops)--> parent E     fires at +h
parent E --apical (zero latency)--> child C    fires at the same boundary
C --> I --(delay-1 feedback reset)-->          lands at +h+1        =>  L = h + 1
```

Two measured laws follow, both confirmed by construction with the diagnostic
`tiled_cc_double_eor` preset (classic column + one extra output relay, `L` 3 -> 4):

```text
period      = 2 * L                        (with a volley every boundary)
suppression bites  <=>  L % input_period == 0
```

`period = 2L` because exactly `L` emissions escape before the first confirmation returns, so
`L` confirmations then arrive back-to-back. Observed 4 / 6 / 8 for `L` = 2 / 3 / 4. The
period-2 alias seen on some `tiled_cc` seeds requires ODD `L` and is not robust (a
2-boundary input gap destroys it permanently).

The divisibility rule is the important one: at the historical `input_period = 1` the halving
holds only because every integer divides 1. Setting

```text
input_period = L
```

makes each volley's confirmation land exactly on its successor's drive packet and cancel it,
forcing exact `fire, silent` alternation independent of loop depth (strict 1010 on 12/12
seeds at `L` = 2, 3 and 4). The condition is sharp: `input_period = L + 1` gives no
suppression at all. `input_period = 0` derives `L` from the graph so the pacing re-tracks on
any topology change.

Physically this is one presentation per RESOLVED causal chain: a real cortical loop settles
far faster than the input changes, so the overlapping-wave regime at `input_period = 1` is an
artifact of the unit-delay discretization rather than a property of the circuit. Full record
and measurements: `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`.

## Failure modes and honest limitations

The limitations below concern the earlier conductance-based predictive-inhibition overlap
experiment, not the rejected `tiled_cc_feature_gated` topology described immediately above.

* **Contamination is real.** While the incumbent still wins the overlapping pattern it
  *does* learn the novel features (measured: novel-weight sum rises during Phase B).
  Symmetry breaking works only because the persistent shared-feature shunt removes the
  incumbent's drive **faster** than contamination accumulates. In the fast-association
  regime contamination wins and reliability drops.
* **Suppressing the shared feature is necessary but not sufficient.** The rival must
  still accumulate enough *novel-feature* drive to cross threshold. Rivals begin with
  unlearned (~61) feedforward weights on the novel pixels, so turnover depends on a
  long enough silent window for that slow accumulation — this is why the persistence
  timescale, not the conductance magnitude, is load-bearing. This is the
  association-bootstrap limitation resurfacing in the conductance model.
* The result is robust for `row/col` (8/8 at `alpha_inh_l1 ≥ 0.95`) but weaker for
  `diag\ / diag/` (2/3), so it is not claimed as pattern-independent.
* The dense 72-candidate projection is deliberate scaffolding; learned use is sparse
  (~3 synapses/pattern). Magnitude pruning to the strong synapses would preserve
  single-pattern behaviour; destructive pruning is left as a follow-up, not done here.
