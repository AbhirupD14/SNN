# Current implementation: methodology and equations

This document describes **only** the model that exists in code today. There is one
scientific model, no experiment-mode selectors, and no legacy ablations. Neuron
behaviour lives in `snn/neurons.py`; topology, the causal step, and serialization
live in `backend/simulation.py`.

## Populations and topology

36 neurons, in five populations:

| Population | Count | IDs | Type | Threshold |
| --- | --- | --- | --- | --- |
| L1E_s (sensory source) | 9 | `L1E0..8` | E | 1000 |
| L1E_new (supervisory) | 9 | `L1Enew0..8` | E | 1000 |
| L1I (instant relay) | 9 | `L1I0..8` | I | 333.33 |
| L2E (competitors) | 8 | `L2E0..7` | E | 1000 |
| L2I (instant relay) | 1 | `L2I` | I | 333.33 |

```text
9 external pixels
       |                        (sensory, frozen, subthreshold)
       v
  9 x L1E_s  ===============>  8 x L2E  ------>  1 x L2I
       |  \       dense acc        |   ^             |
       |   \ paired local          |   +-------------+
       |    \ coincidence          |   frozen subtractive hard wipe
       ^     v                     |
       |  9 x L1E_new  <===========+
       |     |         dense acc feedback
       |     | instant paired relay
       |  9 x L1I
       +-----+  frozen subtractive hard wipe, DELAYED one step (I->E synapse)
```

L1E_new[i] is a local **coincidence detector** with nine accumulating afferents:
index 0 is the paired local sensory afferent from L1E_s[i]; indices 1..8 are dense
L2E feedback.

Internal edges (187):

| Kind | Edges | Count |
| --- | --- | --- |
| `feedforward` | L1E_s[i] → L2E[j] (dense) | 72 |
| `feedback` | L2E[j] → L1E_new[i] (dense) | 72 |
| `coincidence_local` | L1E_s[i] → L1E_new[i] (paired) | 9 |
| `relay_excitation` | L1E_new[i] → L1I[i]; L2E[j] → L2I | 9 + 8 = 17 |
| `inhibition` | L1I[i] → L1E_s[i] (delayed); L2I → L2E[j] | 9 + 8 = 17 |

There is **no** `L2E → L1I` edge: L2 feedback targets the supervisory L1E_new
population. The 9 sensory pixel → L1E_s afferents are not serialized as edges
(their source is not a neuron); the pixel-grid UI shows the input.

The four center-crossing patterns on the 3×3, 9-pixel surface: `row 1`, `col 1`,
`diag \`, `diag /`.

## Neuron state

**Excitatory** (`ExcitatoryNeuron`): membrane `V` (rest 0), shared threshold
`theta`, leak rate, refractory steps/timer, nonnegative `acc_weights`, aligned
per-afferent `acc_distance_factor` (learning only), one frozen subtractive gate
magnitude `subt_magnitude`, and spike state (`spiked`, `v_pre`).

**Inhibitory** (`InhibitoryNeuron`): a stateless instant relay. No weight vector,
no membrane, no leak, no plasticity. It tracks `received_signal` and `spiked`, and
reports a threshold of `theta/3` as a scientific/visual invariant only.

## Delivery equations

Accumulating delivery (geometry never appears here):

```text
V <- V + sum_i(acc_weights[i] * spike_i)
```

Subtractive delivery, under one explicit negative sign (`SUBTRACTIVE_SIGN = -1`):

```text
signed = SUBTRACTIVE_SIGN * subt_magnitude
V <- max(V_rest, V + signed)          # then the hard-wipe invariant is enforced
```

Every wired inhibitory gate is pretrained to the target threshold, so a real
inhibitory spike floors any charge to rest. The postcondition `V == V_rest` is
enforced explicitly so a threshold *crosser* (`V > theta`) cannot leave residual.

Firing: a neuron fires only when not refractory and `V >= theta`; firing records
`v_pre`, resets `V` to rest, and arms the refractory timer.

Leak (once per completed step, only when not refractory), `0 <= lambda <= 1`:

```text
V <- V_rest + (1 - lambda) * (V - V_rest)
```

Refractory off-by-one convention: `refractory_steps = R` is the number of steps
**after** the firing step during which the neuron cannot fire (the firing step
itself does not count). Default `R = 0`. Leak is suppressed during refractory.

## The one accumulating-weight update

Runs only when an excitatory neuron fires, and only on `acc_weights` (never the
subtractive gate):

```text
p            = theta - sum(acc_weights)            # pre-update, signed
signal_i     = +1 if afferent i spiked in the threshold-crossing volley else -1
factor_i     = (d_ref / max(d_i, d_ref)) ** 2      # per-projection normalized, in (0,1]
delta_i      = eta * p * signal_i * factor_i * (1 - (w_i / w_max) ** 2)
acc_weights[i] <- clip(acc_weights[i] + delta_i, 0, w_max)
```

Decisions and their rationale:

- **`p` is the literal signed `theta - sum(acc_weights)`**, computed from the
  pre-update weights. It deliberately changes sign as the stored weights cross
  threshold: below threshold a participating afferent potentiates; exactly at
  threshold it is frozen; above threshold it depresses. No clamp, no absolute
  value. Because the rule is Hebbian only while `p > 0`, each learned projection is
  **initialized below threshold in total** (see below), so a selective receptive
  field forms and the rule self-limits as the total approaches `theta`.
- **`(w / w_max) ** 2`**, not the historical `w**2 / w_max`.
- **Geometry is a learning-rate multiplier only.** `d_i` is the Euclidean distance
  between source and target functional coordinates; the factor is normalized per
  projection so the closest synapse has factor 1. It never scales delivered charge,
  and the exponent (2) is fixed in the model. Frontend display coordinates affect
  neither delivery nor learning.
- **Causal participation window** is simply the spikes delivered in the
  threshold-crossing step. There are no eligibility traces.

L1E_s sensory weights are frozen (a topology-level fact), so L1E_s runs no update.

## Shared configuration and derived values

Editable (small allowlist; `apply_config` rebuilds and rejects any other key):

| Key | Default | Meaning |
| --- | --- | --- |
| `leak_rate` | 0.03 | per-step membrane decay for every E neuron (nonzero: see coincidence) |
| `refractory_steps` | 0 | steps blocked after firing |
| `eta` | 0.01 | shared accumulating-weight learning rate |
| `e_weight_cap` | 500 | the one shared per-synapse accumulating cap = theta/2 |

The shared cap is `theta/2 = 500`, which calibrates the two-input L1E_new
coincidence exactly: a mature paired-sensory weight (~500) plus a mature winning-L2E
weight (~500) sums to `theta` on a coincident step, while either branch alone (~500)
stays below `theta`. Zero leak is **not** a valid default for this circuit (two
lone-branch events could accumulate and falsely fire L1E_new); the default leak is
the value selected by the coincidence experiment.

Fixed / derived (not browser-configurable):

```text
e_threshold         = 1000                (shared across all E populations)
i_threshold         = e_threshold / 3     (~333.3333, reported invariant)
subt_gate_magnitude = e_threshold         (frozen)
population sizes    = 9 / 9 / 9 / 8 / 1
distance exponent   = 2 (fixed)
seed                = persisted dashboard seed (fallback 1)
input_period        = 1
```

### Projection-specific initialization (construction policy, not runtime options)

One shared cap does not impose the same volley charge on projections with
different active fan-in, so each learned projection is initialized around a
documented mean whose **total is below threshold** (so `p > 0` and the rule is
Hebbian). The init is a small uniform seed, not the mature target; the rule
sparsifies it.

| Projection | Afferents | ~Active | Init policy |
| --- | --- | --- | --- |
| sensory L1E_s | 1 | 1 | frozen at `theta/3` (subthreshold; ~3 steps per source spike) |
| feedforward L1E_s→L2E | 9 | ~3 | total ≈ `0.55*theta` (mean ≈ 61), learns, cap 500 |
| L1E_new (coincidence) | 9 | 1 sensory + 1 winner | index 0 (paired sensory) ≈ `0.49*theta` = 490; each of 8 feedback ≈ `0.06*theta` = 60; total ≈ 970 < theta, learns, cap 500 |

The L1E_new init is a strong-but-subthreshold paired sensory seed plus a small
nonzero seed on each feedback afferent, so `p = theta - sum(acc_weights)` starts
positive (Hebbian). The sub-threshold-total invariant (`sum < theta`) is enforced
after jitter. At maturity the paired sensory weight and the associated winning-L2E
weight approach 500 each (sum = theta) while unrelated feedback weights decay to 0.
Deterministic ±4% seeded jitter breaks symmetry. No weight budget, no charge
renormalization.

## Deterministic timestep order

Read top to bottom; no recursion, no hidden same-step side effects. Inhibition
carries a one-step delay that lives in the `I→E` synapse (not the relay).

1. Deposit external sensory charge into `L1E_s` (`input_period` gates delivery).
2. Deliver queued `L1I→L1E_s` inhibition from the previous step — **after** the new
   sensory deposit and **before** the `L1E_s` threshold check — so it removes real
   accumulating charge and lowers the source cadence.
3. Resolve `L1E_s` threshold crossings (frozen sensory → no weight update).
4. Deliver each `L1E_s` spike densely to all `L2E` **and** locally to its paired
   `L1E_new[i]` (afferent index 0).
5. Resolve L2 competition: the winner is the highest pre-fire membrane, tie-broken
   by lowest index; **only** the winner fires and updates; its `+1` fires `L2I`
   immediately; `L2I` hard-wipes every `L2E`; record the events.
6. Deliver the winner spike through dense `L2E→L1E_new` feedback (afferent
   index `1 + winner`).
7. Resolve `L1E_new` coincidence crossings; each firing detector learns from its
   nine real afferents (participation index 0 = paired `L1E_s` spiked; indices 1..8
   = which `L2E` spiked).
8. Each spiking `L1E_new[i]` triggers `L1I[i]`, which fires **now**.
9. `L1I[i]` **queues** its subtractive output for delivery to `L1E_s[i]` on the next
   timestep (the one-step delay is the `I→E` synapse; the relay itself is instant).
10. Apply leak and refractory countdown once to excitatory neurons.
11. Record history, sparse weight changes, spike frequency, log, and the snapshot.

The delay is load-bearing: without it, inhibition would reach `L1E_s[i]` after it
had already fired and reset in the same step, removing zero charge. Delivered on the
next step (phase 2, after the fresh sensory deposit), it removes real charge.

## Coincidence circuit and symmetry breaking: measured status

L1E_new was corrected from a pure feedback integrator (every winner trained every
detector, so all L1I eventually fired → **global** inhibition) into a local
coincidence detector: a firing L1E_new needs its paired sensory input **and** an
L2E winner. The intent is pixel-selective inhibition and emergent symmetry breaking
on pattern change, with no supervised activity mask.

The leaky periodic-integrator model is exact here. For charge `Q` every `T` steps
with retained fraction `r = 1 - lambda`, the post-volley peak is
`V_peak(T) = Q / (1 - r**T)`. Lone-branch rejection with coincident firing needs

```text
Q_off < theta*(1 - r**T) <= Q_on,   Q = sum(active afferent weights),
```

where `Q_off` is the largest lone-branch charge (~500) and `Q_on` the coincident
charge (~1000). This is validated analytically, against the engine's exact
irregular-interval recurrence `V_pre[k] = V_post[k-1]*r**dt[k]`, `V_post[k] =
V_pre[k] + Q[k]`, and for several active fan-in counts in
`experiments/frequency_experiment.py`.

**What works.**

- The delayed `I→E` inhibition removes **nonzero** charge (measured: ~30000 units
  over 400 steps at the default leak) with the one-step delay verified (a relay
  firing at step *t* never wipes its source at *t*; the wipe lands at *t+1*), and it
  lengthens the paired-source cadence (3.0 → ~3.5 steps at zero leak).
- At the default leak (0.03), inactive-pixel L1E_new stay **completely quiet**
  (0 false fires over 1500 steps) while L2E still bootstraps — so the reported bug
  is fixed: inhibition is **pixel-selective**, not global. After a row→column
  switch, inhibition lands only on the column-active pixels {1,4,7}.
- Neuron-level coincidence is genuine: a mature detector (sensory 500 + winner 500)
  rejects each lone 500 branch and fires only on coincidence once leak ≥ ~0.16.

**What does not (negative, honest).**

- **No single shared leak** satisfies both L2E bootstrapping and strict lone-branch
  rejection. L2E bootstraps only at `leak ≤ 0.04` (its sub-threshold ff volley must
  accumulate); a mature lone-500 branch is rejected only at `leak ≥ 0.16`. The two
  windows are **disjoint**. Per the brief, no population-specific leak, boolean
  coincidence gate, eligibility trace, or reset heuristic was introduced.
- **No symmetry breaking.** With row→column→row, the same competitor (L2E5) wins all
  three phases. At the bootstrap-compatible default leak, lone sensory (500) still
  fires L1E_new for *every* active pixel, so the "new-exclusive pixels stay
  available for a rival" mechanism never gates on winner association; and the
  incumbent's learned feedforward weights let it win the overlapping pattern.

**Production stance.** The default leak (0.03) sits in the window where the network
is alive and inhibition is pixel-selective (the bug fix). Strict two-input
coincidence and symmetry breaking would require breaking one of the two structural
facts above (e.g. a delay register that decouples L1E_new firing from the source
volley, or letting losers learn) — genuine design changes beyond this correction,
recorded here rather than hidden as a heuristic. Measurements are in
`experiments/frequency_results.json`.
