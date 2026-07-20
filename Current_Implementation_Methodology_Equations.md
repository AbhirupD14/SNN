# Current implementation: methodology and equations

This document describes **only** the model that exists in code today: a
conductance-based spiking network with **persistent inhibitory conductance**
(there are no hard wipes anywhere), a local post-synaptic activity trace, and
**local predictive inhibition** (PI). Neuron behaviour lives in `snn/neurons.py`;
topology, the synchronous timestep, and serialization live in
`backend/simulation.py`.

The network is defined by a **`NetworkSpec`** (typed nodes + typed edges, in
`backend/network_spec.py`); the engine executes whatever graph it is given via
per-edge-kind dispatch. The fixed vocabulary is eight node archetypes
(`rg_source`, `e_sensory`, `e_encoder`, `e_residual`, `e_competitor`, `i_relay`,
`predictor`, `switch`) and six edge kinds (`feedforward`, `fixed_excitation`,
`trace_excitation`, `relay_excitation`, `inhibition`, `predictive_inhibition`). Two
intrinsic population rules are NOT edges: `e_sensory` — every threshold crosser
fires; `e_competitor` — deterministic single-winner WTA (one winner fires + learns
its feedforward weights). Four built-in presets ship, selected by the `topology`
parameter, and arbitrary graphs can be built/saved/loaded live in the browser
Topology Editor:

* **`topology='pi'` — the predictive-inhibition (PI) experiment.** 26 neurons.
  Eight pattern-specific predictive interneurons `PI[j]`, paired one-to-one with the
  competitors `L2E[j]`, each owning nine locally-plastic inhibitory output synapses
  onto the sensory `L1E_s` cells. This is the topology the symmetry-breaking science
  is about.
* **`topology='old'` — the original dense global-inhibition topology.** 27 neurons.
  Nine paired `L1I` relays fed densely by every `L2E` (every `L2E`→every `L1I`), each
  projecting a paired inhibitory conductance onto its own `L1E_s`. The single L2
  winner drives all nine `L1I`, so every `L1E_s` is shunted — winner-gated global
  inhibition. Its inhibition is conductance too (no hard wipes anywhere).
* **`topology='rg'` — old cortex with explicit RG sources.** 36 neurons. Plastic
  paired RG→L1E afferents precede the unchanged old dense-feedback topology.
* **`topology='rg_residual'` — classification-preserving residual/error circuit.**
  52 neurons. L1E remains complete and uninhibited; learned prediction targets a
  separate ErrorE sheet, whose unexplained events drive locally traced incumbent
  switches before ordinary L2 WTA re-competition.

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

## The one accumulating excitatory weight rule (production: linear-bounded)

Runs when an excitatory neuron fires, on `acc_weights` only. The **production default is
linear-bounded** — the per-synapse `(1 - (w_i/w_max)^2)` multiplier was removed after a
32/32 fresh-seed confirmation (see `docs/LINEAR_WEIGHT_ABLATION_REPORT.md`). The E hard
cap and the zero floor are retained:

```text
p        = threshold - sum(acc_weights)               # pre-update, signed
signal_i = +1 if afferent i spiked in the causal volley else -1
delta_i  = eta · p · signal_i · distance_factor_i     # linear_bounded (production)
w_i      = clip(w_i + delta_i, 0, w_max)              # E cap retained, floor 0
```

Geometry is a per-synapse learning-rate multiplier only; it never scales delivered
charge. `p` (the neuron-wide FE) is signed, so a projection self-limits as its total
approaches threshold. L2E feedforward weights learn by this rule; sensory `L1E_s` weights
are frozen.

The **historical** rule multiplied `delta_i` by `(1 - (w_i/w_max)^2)`; it remains
available as the headless `e_weight_update_mode='quadratic_bounded'` for regression
comparison. Note this is the ORDINARY E/L2E rule; the **coincidence C basal rule is
unchanged and still keeps its `(1 - (w_b/w_C_max)^2)` term** (see the C section / spec).

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

## Failure modes and honest limitations

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
