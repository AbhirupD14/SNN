# Feedback Cadence, Loop Latency, and Input Pacing

**Status:** resolved. Exact fire/silent alternation is reproducible on every tiled topology
and every seed tested, and the pacing that produces it is now derived from the graph rather
than hand-set.

This is the primary record for the top-down `C → I` feedback cadence — why it was irregular,
what actually determined it, and why the fix is a statement about the model's timing
abstraction rather than a tuning choice. It supersedes the cadence claims in earlier
documents; `docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md` §8.1b–d now points here.

Topologies referenced (loop latency `L` in brackets):

| preset | relays in the ascending path | `L` |
| --- | --- | --- |
| `tiled_cc_direct_identity` | none (child E addresses parent E directly) | **2** |
| `tiled_cc`, `tiled_cc_l1_4` | one `Eor` | **3** |
| `tiled_cc_double_eor` | two (`Eor → Eor2`) — **diagnostic probe, built for this** | **4** |

---

## 1. The symptom

At the dashboard's rates the confirmed column was expected to halve its output into a clean
`fire, silent, fire, silent` cadence. It did not do so reliably. Measured winner sequences of
a confirmed column, two active patches, after 3000 boundaries:

```text
tiled_cc, seed 1              101010101010…   strict alternation
tiled_cc, seed 1302528033     000111000111…   period 6, grouped
```

Over 16 seeds `tiled_cc` gave strict `1010` on **10/16** and grouped period-6 on the rest.
The **rate** was exactly 0.50 on every seed of every topology; only the *cadence* varied.

A grouped cadence at rate ½ is explicitly not the target
(`docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`): *"a grouped cadence such as
`fire, fire, fire, silent, silent, silent` has an average rate of 0.5, but it is not the
required certainty signal."*

## 2. Prerequisites that had to be fixed first

Three defects were masking the real behaviour. Each is recorded in full in
`docs/DIRECT_IDENTITY_TILED_TOPOLOGY.md`; summarised here because the cadence numbers below
are only meaningful after all three.

1. **`tau = 1.0` boundary-edge deadlock.** The event loop ran `while current_tau < 1.0`, so a
   crossing that only became available *at* the edge was unschedulable. A C whose apical
   permission arrived from a parent firing at exactly 1.0 deposited, sat above threshold with
   an open gate, then had the gate cleared by the next boundary — accumulating to
   `V ≈ 780 000` against `θ = 1000` without ever firing or learning. The loop now drains
   events. Two-patch direct-identity: C spikes 0 → 500 per 1000 boundaries.
2. **A display-geometry penalty on multi-basal C learning.** The eight basal afferents
   inherited the `1/d²` factor from the layout's display ring, spreading `phi` over
   0.214–1.000, so a column's C matured at a rate decided by which ring seat its owner
   occupied. Column-local basal edges now carry no distance penalty.
3. **C maturing ~40× slower than the E pool.** `c_eta` 2.0 → **16.0**: the E bank reaches its
   `θ/2` ceiling at ~400 boundaries and the C basal now reaches 95 % of `θ` at ~711 rather
   than ~20 000. Deliberately still *after* E — C confirms an owner the pool has settled on.

## 3. What actually determines the cadence

### 3.1 The loop

Impulse-measured (inject one coordinated L1 spike pair into an otherwise blank input, trace
forward). For `tiled_cc`:

```text
b0   L1 E fires          <- the evidence
b1   Eor fires
b2   L2 E fires, C fires  <- confirmation
b3   reset lands on the L1 bank
```

Every L1 spike removes one L1 spike **`L` boundaries later**. Crucially the confirmation is
always *downstream* of the spike it would suppress: measured **450/450 boundaries**, the C
fired at `tau = 1.0` while its column's E had already fired at `0.667`. The C cannot confirm
before the E fires, because the E's spike *is* the evidence that travels up. So same-boundary
suppression is impossible in principle, not just unimplemented.

### 3.2 The law: `period = 2 × latency`

With latency `L`, exactly `L` emissions get out before the first confirmation returns, and
those `L` confirmations then arrive back-to-back producing `L` consecutive suppressions.

`tiled_cc_double_eor` was built specifically to falsify this: the classic column with one
extra output relay spliced into the ascending path (`E → Eor → Eor2 → parent E`), changing
loop length and nothing else — same WTA, C gate, apical feedback, caps and rates.

| topology | `L` | predicted period | observed (8 seeds) |
| --- | --- | --- | --- |
| `tiled_cc_direct_identity` | 2 | 4 | **4 on 8/8** |
| `tiled_cc` | 3 | 6 | 6 on 4/8, **2 on 4/8** (alias) |
| `tiled_cc_double_eor` | 4 | 8 | **8 on 8/8** |

Confirmed. Adding one relay moved the period 6 → 8 on every seed.

### 3.3 Why `1010` appeared on some seeds — and why it was worthless

Period 2 requires a spike at an even boundary to suppress an **odd** one, which needs **odd**
`L`. `tiled_cc` has `L = 3`, so an aliased 2-cycle also satisfies the loop and about half the
seeds fall into it. Both even-latency topologies give a single period on 8/8 seeds.

The alias is fragile, and this is the decisive observation: a network sitting in period-2 was
pushed into **period-6 by a 2- or 3-boundary input gap and stayed there permanently**, while
period-6 networks never left.

```text
seed 1, baseline          101010101010…   period 2
seed 1, after 2 blank b.  100011100011…   period 6   (permanent)
seed 1, after 3 blank b.  000111000111…   period 6   (permanent)
```

Period-6 is absorbing; period-2 is not reachable by reseeding and would not survive the first
pattern switch. The two driven columns were verified to be **in phase on every seed**, ruling
out cross-column desynchronisation.

### 3.4 The halving itself was an arithmetic accident

The engine presents a fresh RGC volley every `input_period` boundaries — it does **not** wait
for the causal chain, so at the historical `input_period = 1` roughly `L` presentations are
always in flight. The `C → I` reset is applied at `t+1` and discards whatever drive packet is
frozen there, so it only suppresses anything if a volley fired at `t₀ + L`:

```text
RGC fires t0 -> drive frozen t0+1 -> E fires -> ... -> C fires -> reset lands t0+1+L
suppression bites  <=>  a volley's drive is frozen at t0+1+L  <=>  L % input_period == 0
```

Measured per-presentation emission over 18 (topology, `input_period`) combinations — the rule
predicts **18/18**:

| topology | `L` | halving survives at `ip` = | no effect at `ip` = |
| --- | --- | --- | --- |
| `tiled_cc_direct_identity` | 2 | 1, 2 | 3, 4, 5 |
| `tiled_cc` | 3 | 1, 3 | **2**, 4, 5, 6 |
| `tiled_cc_double_eor` | 4 | 1, 2, 4 | 3, 5, 6, 7 |

At every failing pacing the emission rate is **1.00** — the feedback does nothing at all.
`tiled_cc` fails at `input_period = 2`; a two-boundary gap is enough.

So the halving observed at the default configuration was not the mechanism working. It was
the accident that **every integer divides 1**. The cadence measured loop latency and input
pacing; it carried no information about column certainty.

## 4. The resolution

### 4.1 One presentation per resolved chain

The same arithmetic gives the fix. Set the input period equal to the loop latency:

```text
reset lands at        t0 + 1 + L
successor volley's drive frozen at   t0 + ip + 1
these coincide  <=>  ip = L
```

Each volley's confirmation then lands exactly on its successor's drive packet and cancels it.
That volley emits nothing, so it generates no confirmation, so the volley after it is
uncancelled and fires. Alternation is **forced**, and `L` cancels out of the condition — the
cadence becomes depth-independent.

| topology | `L` | `input_period = L` | strict `1010` per presentation |
| --- | --- | --- | --- |
| `tiled_cc_direct_identity` | 2 | 2 | **12/12 seeds** |
| `tiled_cc` | 3 | 3 | **12/12 seeds** |
| `tiled_cc_double_eor` | 4 | 4 | **12/12 seeds** |

Three loop depths, three pacings, exact alternation on every seed — including the seed that
gave grouped `111000` at `input_period = 1`. One case (double-Eor, seed 2) needed a longer
warm-up (12 000 rather than 6 000 boundaries) before settling; strict thereafter.

**The condition is sharp, not a lower bound.** `ip = L+1` — "let the chain fully finish, then
present" — measures rate **1.00**: the reset lands one boundary *before* the successor's drive
arrives and cancels nothing. Presenting a single volley with nothing behind it makes
`C → I → E` a guaranteed no-op.

### 4.2 Why this is the sound abstraction, not a tuning choice

A real cortical feedback loop resolves on a far shorter timescale than the input changes.
One presentation per **resolved causal chain** is therefore the physically meaningful regime;
the overlapping-wave regime at `input_period = 1` is an artifact of the unit-delay
discretization, in which each hop costs a whole boundary (`SYNAPTIC_DELAY = 1`).

Read that way, the `period = 2L` result is evidence *for* the abstraction rather than a
property of the circuit: the cadence depended on wiring depth precisely because the
discretization let waves overlap in a way the modelled system would not.

### 4.3 Pacing is derived from the graph

Because the correct period is a structural property, `input_period = 0` (the dashboard
default) means **AUTO** and the engine derives it:

```text
child E --(h feedforward hops)--> parent E     fires at +h
parent E --apical (zero latency)--> child C    fires at the same boundary
C --> I --(delay-1 feedback reset)-->          lands at +h+1        =>  L = h + 1
```

Derived values match the impulse measurements exactly (2 / 3 / 3 / 4 for direct-identity /
`tiled_cc` / `tiled_cc_l1_4` / double-Eor) and are `None` for graphs with no top-down loop
(`rg_coincidence`, `rg_direct_cc4`), which fall back to 1. It reads the **graph** — never a
preset name — so it re-tracks on every topology change:

```text
tiled_cc (auto ip=3)  ->  direct-identity (auto ip=2)  ->  double-Eor (auto ip=4)
```

A hand-set value is honoured verbatim and deliberately does **not** re-track. That is the
failure this replaces: pinned at 3 and switched to direct-identity, `L % ip = 2` and the
halving stops silently.

`input_period` is the only dashboard control that applies **without rebuilding**, so a
trained column can be re-paced live and its cadence re-measured without retraining.

## 5. What this does and does not establish

**Established.** Exact `1010` per presentation, on 12/12 seeds at three different loop
depths, with the pacing derived structurally. The cadence is now reproducible and
seed-independent, and the earlier seed-dependence is fully explained.

**Not established.** That the cadence *means* certainty. Suppression is still consumed by a
wall-clock boundary; it works because `ip = L` guarantees a volley is always there to cancel.
Under a variable, interrupted, or externally-driven input the same fragility returns. The
more general contract — suppression consumed by the **next eligible evidence event** rather
than by boundary `t+1` — would make the cadence invariant at *every* pacing rather than only
at `ip = L`. It is no longer a prerequisite for alternation, and is tracked as P1.

Nor does exact alternation by itself demonstrate a confidence signal: a column paced at its
own loop latency alternates whether or not its C has anything meaningful to say. Establishing
that requires showing the cadence *changes* when confirmation status changes — a separate
experiment.

## 6. Reproducing

```bash
# derived latency per topology, and auto-pacing behaviour
PYTHONPATH=. .venv/bin/python -m pytest tests/test_boundary_edge_crossing.py -q
```

Live: the dashboard opens on `tiled_cc` with `input_period = 0` (auto → 3). The confirmed
column alternates exactly. Set the control to 1 to see the historical grouped cadence, or
switch topology and watch the resolved period re-track in the serialized params
(`resolved_input_period`, `feedback_loop_latency`).

Regression coverage: `tests/test_boundary_edge_crossing.py` — the boundary-edge drain, the
`ip = L` alternation parametrized over all three loop depths, latency derivation against the
impulse-measured values, auto re-tracking on topology change, the no-loop fallback, and that
re-pacing preserves learned state.
