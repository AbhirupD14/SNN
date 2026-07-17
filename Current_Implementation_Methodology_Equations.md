# Current implementation: methodology and equations

This document describes **only** the model that exists in code today: a
conductance-based spiking network with **persistent inhibitory conductance**
(there are no hard wipes anywhere), a local post-synaptic activity trace, and
**local predictive inhibition** (PI). Neuron behaviour lives in `snn/neurons.py`;
topology, the synchronous timestep, and serialization live in
`backend/simulation.py`.

The network is defined by a **`NetworkSpec`** (typed nodes + typed edges, in
`backend/network_spec.py`); the engine executes whatever graph it is given via
per-edge-kind dispatch. The fixed vocabulary is four node archetypes
(`e_sensory`, `e_competitor`, `i_relay`, `predictor`) and four edge kinds
(`feedforward`, `relay_excitation`, `inhibition`, `predictive_inhibition`). Two
intrinsic population rules are NOT edges: `e_sensory` — every threshold crosser
fires; `e_competitor` — deterministic single-winner WTA (one winner fires + learns
its feedforward weights). Two built-in presets ship, selected by the `topology`
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

## Populations and topology

| Population | Count | IDs | Type | In topology |
| --- | --- | --- | --- | --- |
| L1E_s (sensory source) | 9 | `L1E0..8` | E | both |
| L2E (competitors) | 8 | `L2E0..7` | E | both |
| L2I_WTA (winner-take-all relay) | 1 | `L2I` | I | both |
| PI (predictive interneurons) | 8 | `PI0..7` | I | pi only |
| L1I (paired relays) | 9 | `L1I0..8` | I | old only |

**PI-preset edges — 168 total:** 72 feedforward `L1E_s→L2E` · 8
`relay_excitation L2E[j]→PI[j]` (paired 1:1) · 8 `relay_excitation L2E→L2I` · 72
`predictive_inhibition PI[j]→L1E_s[i]` (candidate, locally plastic) · 8
`inhibition L2I→L2E` (WTA conductance).

**Old-preset edges — 169 total:** 72 feedforward `L1E_s→L2E` · 72
`relay_excitation L2E→L1I` (DENSE, every L2E→every L1I) · 8 `relay_excitation L2E→L2I`
· 9 `inhibition L1I[i]→L1E_s[i]` (paired) · 8 `inhibition L2I→L2E` (WTA conductance).

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

## The one accumulating excitatory weight rule (unchanged)

Runs when an excitatory neuron fires, on `acc_weights` only:

```text
p        = threshold - sum(acc_weights)               # pre-update, signed
signal_i = +1 if afferent i spiked in the causal volley else -1
delta_i  = eta · p · signal_i · distance_factor_i · (1 - (w_i/w_max)**2)
w_i      = clip(w_i + delta_i, 0, w_max)
```

Geometry is a per-synapse learning-rate multiplier only; it never scales delivered
charge. `p` is signed, so a projection self-limits as its total approaches
threshold. L2E feedforward and (in the comparison branch) L1E_new coincidence
weights learn by this rule; sensory `L1E_s` weights are frozen.

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
3. Threshold test + fire: L1E_s crossers; L1E_new crossers (enew, each learns);
   deterministic L2E WTA (one winner fires + learns feedforward).
4. Update every excitatory neuron's local activity trace.
5. Emit spikes into delay-1 queues: L1E_s→L2E feedforward (+ enew local sensory);
   winner→`L2I_WTA` conductance (all L2E) and, per topology, winner→paired `PI`
   (direct) or winner→dense feedback to `L1E_new` (enew); PI/L1I relays run their
   **local plasticity** now (reading the traces finalized in step 4) and schedule
   their inhibitory conductance for `t+1`.
6. Decay each `g_inh` once; count down refractory; PI passive weight decay.
7. Serialize the frame.

Causal reading — *first encounter:* `L1 activity → L2 winner → PI event → local
inhibitory learning` (the original L1 spike is **not** cancelled). *Later encounter:*
`L2 winner → PI event → persistent g_inh → a later L1 sensory interval is shunted`.

## Configuration

Editable keys (`apply_config` rebuilds; unknown keys rejected): `leak_rate`,
`refractory_steps`, `eta`, `e_weight_cap`, `input_period`, `topology` (`'pi'`|`'old'`),
`alpha_inh`, `alpha_inh_l1`, `alpha_a`, `beta_v`, `beta_s`, `a_max`, `e_inh`,
`pi_eta`, `pi_w_max`, `pi_lt_decay`, `pi_g_scale`, `l2i_g_scale`,
`pi_conductance_enabled`, `pi_plasticity_enabled`. Arbitrary custom graphs are applied
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
