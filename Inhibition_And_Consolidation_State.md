# Inhibition & Consolidation — Current State

Status snapshot of the L2 competition / inhibition work (2026-07-09). Records the
problem, what was built, what worked, what didn't, and the open question.

## UPDATE 2026-07-09 — L2I firing deadlock diagnosed and fixed (assembly flow credit)

After consolidating a held pattern to a single L2E, **L2I stopped firing** (freq 0),
even though its winner's `L2E->L2I` weight "should" have grown to threshold. Root
cause was a **bootstrap deadlock**, not weak inhibition:

- `L2I` fires two ways — (A) several distinct L2E round-robin and *sum* past its
  threshold, or (B) one habitual winner's `L2E->L2I` synapse grows to self-
  sufficiency and fires it alone. Consolidating to one L2E removes regime A, and
  regime B's growth **only happens on L2I's own fire** (postsynaptic-gated Hebbian,
  `_update_weights` runs on fire) — so once L2I is silent it can never learn its way
  back. Live symptom: single winner delivered ~582 vs L2I threshold 1142.86, and the
  `L2E->L2I` weight was stuck near its init range (never grew).

- The deeper flaw: the E->I credit rule only credited **the last input volley**
  (`_last_input_spikes`), i.e. whichever L2E happened to tip L2I over threshold. In
  the round-robin phase that spreads credit across all rotating members, so the
  eventual winner's synapse never matured before consolidation removed the round
  robin. (This had *replaced* an older trace-based credit-splitting rule.)

**Fix — flow-proportional assembly credit** (`assembly_flow_credit`, opt-in, on
L2I/L1I). On the inhibitory neuron's own fire, credit each incoming positive E->I
synapse in proportion to the flow it delivered over the retention window (the
per-synapse leaky `_trace`, which decays at the neuron's own leak so it spans the
same window the membrane integrates), **normalized by the max flow** so the dominant
driver gets the full learning rate; non-contributors decay toward the floor
(`assembly_decay_frac`). It still fires only on L2I's own spike, so it shares one
clock with loser depression (both are driven by the L2I discharge — no boolean gate,
which was explicitly rejected as un-neuron-like). It **prevents** the deadlock rather
than reviving an already-dead L2I: the winner's synapse matures during the early
round-robin, before consolidation strips the multi-winner volleys.

Result (held `row 0`, `eta_loss=10`, default `l2i_lr_frac=0.01`, seed 1): one L2E4
specialist + **L2I firing on a ~16-step rhythm**, winner's `L2E->L2I` matured to
threshold (1142.9). Legacy last-volley credit deadlocks in the same regime: L2I
nearly silent (37 fires, 0 late), weight stalled at 631. NOTE: a *faster* E->I rate
(`l2i_lr_frac=0.05`) destabilizes it (L2I over-inhibits early) — the default 0.01 is
the stable choice. Dashboard now defaults ON; see `test_assembly_flow_credit.py`.

Also fixed here: loser depression could push feedforward gates **negative** at high
`eta_loss` (slider reaches 20) because the depression step overshot the
`min_positive_weight` floor; `_depress_losers` now clamps at the floor.

Still open below (unchanged): clean **8/8 one-to-one ownership** across the whole
pattern set. `eta_loss=10` over-depresses across multiple patterns, so this L2I fix
is validated on a *single held* pattern, not interleaved training.

## The goal

Holding one input pattern (e.g. `row 0`), we want a **single** L2E to own it and
fire in a clean rhythm with L1E, L1I and L2I — one specialist per pattern. The
charge-over-time and raster views should show one L2E locked in phase, not a pool
taking turns.

## The problem observed

Holding `row 0` with the current dashboard config, **5–6 different L2E round-robin**
the pattern (dominance ~0.20, i.e. each wins ~1/6 of the time). Inhibition "does
nothing": a neuron often fires the very step after L2I fired.

## Root cause — it's a feed-forward SYMMETRY problem, not an inhibition problem

Holding `row 0`, six L2E all learn to be **identical** row-0 detectors (each ~0.9·cap
on all three row-0 pixels; only the two unused units stay near the floor):

```
L2E0/2/3/4/6/7: row0 pixels ≈ [0.90, 0.90, 0.90]   ← six co-specialists
L2E1/5:         ≈ [0.03, 0.04, 0.03]               ← empty
```

Inhibition — however large or sustained — hits these six **equally**, so it can
suppress them but can never pick one. You cannot break a symmetry by pushing evenly
on all sides. So bigger/longer inhibition changes the firing *rate*, never the number
of owners.

## Why inhibition looked weak (the numbers, for the dashboard preset)

- `threshold_l2 = 8000`; the L2I→L2E gate is capped at `G = sqrt(L2_GATE_WMAX) ≈ 1224
  ≈ 0.15·θ` by design (a gate near θ would fully reset rivals → old hard-reset
  collapse). The **turnover** rule settles it even lower (~500–800 ≈ 0.06–0.10·θ).
- So one discharge removes only **~7–8% of threshold**. With `refractory=0` and
  `v_sat` off (charge piles above θ), a rival knocked down ~8% just re-crosses and
  fires again next step.

## What was built this phase (all opt-in / toggleable; defaults preserve prior behavior unless noted)

| mechanism | flag(s) | default | note |
|---|---|---|---|
| Sparse excitatory **flow-rate** charge (weight = current amplitude; a spike opens a decaying trace integrated over time via closed-form lazy advance) | `excitatory_flow_rate`, `exc_trace_decay`, `exc_trace_normalized` | dashboard runs it ON | forces effective `l2_charge_chunks=1` |
| **Distance** attenuation of delivered drive `(d_ref/max(d,d_min))^power` (weight=gate, distance=delivery, trace=temporal) | `distance_weighting`, `distance_power/ref/min` | OFF (per-synapse distances=1) | mechanism only; functional geometry not yet assigned |
| Differentiating **inhibitory-gate rule** (turnover: `du=eta_up·p_t·(1−u)−eta_down·u`; also a `margin` diagnostic) vs legacy saturating | `inhibitory_delta_rule`, `inhibitory_rule_mode`, `inhibitory_eta_up/down`, `inhibitory_p_max` | delta ON, mode `turnover` | gates differentiate (spread ~260 vs 4) but stay sub-θ |
| **Inhibitory flow** (discharge drains out over time, symmetric to excitatory flow) | `inhibitory_flow_rate`, `inh_trace_decay`, `inh_trace_normalized` | OFF | **NEW** |
| L1I **immediate relay** (fires on any nonzero feedback, no integration) | `l1i_immediate_relay` | ON | |
| Chunked L2 charge; post-inhibition charge snapshot for the graph; per-L2E phase diagnostics + invariant warnings | `l2_charge_chunks`; `l2_charge`, `l2_inh_phases` | — | graph now shows the inhibition dip |

## What worked / didn't for CONSOLIDATION (hold row 0)

| lever | effect | consolidates? |
|---|---|---|
| **Gate ceiling** `l2_gate_eq_frac` 0.15→1.5 | discharge 7%→30% of θ; fewer immediate re-fires | **No** — distinct stays ~5.5, dominance ~0.20 |
| **Inhibitory flow** (normalized / un-normalized, various decays) | un-normalized suppresses more (fires 839→484) | **No** — distinct stays ~5.5 |
| **Loser depression** `eta_loss` 0.01→10 | at ~10 the five losers' row-0 weights decay away | **Yes (single held pattern)** — distinct=1, dominance=1.00 |

`eta_loss` sweep (hold row 0): `0.01→6 winners`, `0.5→6`, `2.0→6`, **`10.0→1 winner`**.

## The catch (still open)

Strong loser depression only fixes a **single held** pattern. Trained interleaved on
all 8 patterns, `eta_loss=10` gives **4/8 distinct owners and leaves a pattern
ownerless** (over-depression). Clean **8/8 one-to-one** ownership across the whole set
remains **unsolved** — inhibition + loser depression buy a single winner per *held*
pattern, not stable per-pattern ownership. This matches the long-standing
consolidation gap noted elsewhere in the repo.

## Takeaways / where to push next

1. Inhibition **magnitude/persistence is the wrong lever for consolidation** — it can
   only regulate rate, not break the feed-forward symmetry.
2. The symmetry-breaker is **loser depression**; its default is ~1000× too weak to
   matter. The `eta_loss` dashboard slider now reaches 20 so this is explorable live.
3. The next real problem is preventing multiple neurons from co-learning the same
   pattern in the first place (a learning/allocation problem), not making the hit
   harder. Candidates: much stronger/earlier loser depression with a floor that
   protects one winner, an allocation/novelty bias so a fresh unit claims an
   unowned pattern, or a winner-protect term.

## Tests / harnesses

`test_flow_rate.py` (+ distance), `test_inhibitory_delta_rule.py`,
`test_l1i_immediate_relay.py`, `test_l2_chunked_charge.py`,
`test_assembly_flow_credit.py` (the L2I deadlock fix: flow-proportional credit +
the integration test showing L2I comes alive where legacy stalls) — all green, plus
the legacy suite. Ablation scripts used for the tables live in the session scratchpad.
