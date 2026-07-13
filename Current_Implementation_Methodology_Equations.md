# Current Implementation Methodology and Equations

This document describes the **currently implemented default dashboard configuration only** — the model
the frontend actually runs (`backend/api.py` engine construction). Mechanisms that
exist in the code but are **off by default** (flow-rate accumulation, confidence
consolidation, the legacy `signed_depression` add-on, learned inhibitory-gate plasticity,
homeostasis, weight budget, lasting inhibition, balanced init, membrane
saturation, immediate-relay L1I, subtractive reset) have been removed from this
sheet; they remain in the source as reversible experiments. Every equation below is
cross-checked against `neuron_flexible.py`, `snn/rules/`, `backend/simulation.py`,
and `backend/api.py` as of 2026-07-13. The planned refractory-protected,
active-gate redistribution rule in `Inhibitory_Off_Weight_Recruitment_Spec.md`
is intentionally not presented as current behavior here.

## Network Topology

The dashboard engine builds a two-layer network over four center-crossing 3x3 line
patterns:

- `N_PIX = 9`: one input pixel per grid cell.
- `N_OUT = 8`: an overcomplete L2E output pool (2x the four patterns), so
  recruitment/competition dynamics are visible.
- `L1E_i`: fixed pixel encoder for pixel `i` (pretrained, `learning_rate = 0`).
- `L1I_i`: paired inhibitory neuron for `L1E_i`. It is a **trainable threshold
  accumulator** that integrates the L2E winner stream and, when it crosses its own
  threshold, suppresses its paired pixel one step later (step 9).
- `L2E_j`: trainable output neuron with **exactly one feedforward synapse from each
  L1E pixel and no negative afferent**. L2 competition is an unweighted competitive
  reset (see *L2 Competition*), not a learned gate.
- `L2I`: one shared inhibitory neuron. It integrates positive `L2E -> L2I`
  recruitment weights; when it fires it issues the competitive-reset event.

The four input patterns are `row 1`, `col 1`, `diag \`, `diag /`.

## Default Parameters (dashboard)

Fixed-point scale `UNIT = 1000`: thresholds and weights are integers at this scale
(1 model unit = 1/UNIT). Values below are shown in model units.

```text
threshold (L1)        = 1.0
threshold_l2          = 8.0
l2i_threshold_frac    = 1/3          -> thr_l2i = 8/3
l1i_threshold_frac    = 1.0          -> thr_l1i = thr_l2i
weight_cap (base)     = 1.0
l2e_weight_cap_frac   = 1/3          -> L2E per-afferent cap = thr_l2/3
pos_weight_floor      = 1  (unit)    -> positive-weight floor w_min
refractory            = 0            (current L1E/L2E dashboard value)
volley_period         = 4
input_period          = 1            (held pixel drives every step)
cycle_period          = 4
event_driven          = True         (resolve competition every step)
l2_charge_chunks      = 20           (chunked WTA race)
l2e_lr_frac           = 0.02         -> L2E eta = 0.02 * cap
ei_sat_mult           = 4.0          (push E->I saturation past the clip)
leak_enabled          = False        (L2E/L2I/L1I are pure integrators)
distance_weighting    = True   (power 2, ref 7.472, min 1)
signed_spike_learning = True   (L2E feedforward rule)
structural_free_energy= True   (eta_floor 0.02; replaces the voltage term)
loser_depression      = True   (competitive depression on the reset event)
l2i_hard_reset_losers = True
hard_reset_clear_traces = True
l2e_init_mode         = legacy_wide   (balanced initialization is off)
l2e_init_jitter       = 0.05          (inert in legacy_wide mode)
l1i_immediate_relay   = False         (trainable L1I accumulator)
homeostasis           = False
l2e_budget            = False
confidence_consolidation = False
signed_depression     = False         (superseded by signed-spike learning)
```

Resolved population state:

```text
L1E refractory/leak   = 0 / 0.10
L1I refractory/leak   = 2 / 0
L2E refractory/leak   = 0 / 0
L2I refractory/leak   = 0 / 0
```

`refractory = 1` is the planned same-step winner-plasticity veto in the new
redistribution specification; it is not implemented in the current dashboard.

Learning rule per population (dashboard):

- **L2E feedforward**: signed-spike rule with the structural free-energy gate.
- **L2E -> L2I recruitment**: charge rule (below).
- **L2E -> L1I feedback**: flow-proportional assembly credit (below).
- **L1E**: frozen (no learning).

## Weight Initialization

Three deterministic, independent RNG streams are derived from the engine seed so
resizing or reordering one population does not shift the others.

The active dashboard uses legacy-wide, task-independent L2E feedforward
initialization:

$$
w^{\mathrm{L1E\to L2E}}_{ji} \sim \mathrm{Uniform}(50, 200),
\qquad
w_{\min}=1,
\qquad
w_{\mathrm{cap}}=\theta_{\mathrm{L2}}/3.
$$

L2E-to-L2I recruitment weights start independently below the resolved L2I
threshold:

$$
w^{\mathrm{L2E\to L2I}}_j
\sim \mathrm{Uniform}(0.25\,\theta_{\mathrm{L2I}},
                      0.50\,\theta_{\mathrm{L2I}}).
$$

Every L1I receives the same L2E winner stream, so the L1I bank starts from copies
of one random task-independent feedback vector:

$$
\mathbf{w}^{\mathrm{L2E\to L1I}}
\sim \mathrm{Uniform}(0.25\,\theta_{\mathrm{L1I}},
                      0.50\,\theta_{\mathrm{L1I}})^{N_{\mathrm{OUT}}},
$$

and each `L1I_i` receives a copy of that vector. This prevents arbitrary
pixel-phase classes without encoding a pattern or pixel preference.

## State Variables

For neuron `n`: membrane $V_n$, threshold $\theta_n$, refractory timer $r_n$,
resting potential $R_n = 0$, afferent weights $w_{ni}$, input spikes $s_i$, leak
rate $\lambda_n$. Positive weights are excitatory and negative weighted synapses
are inhibitory. The active L2I-to-L2E path is the exception: it is an unweighted
structural reset event, so L2 inhibition is not represented by a negative afferent.

## Charge Integration (instantaneous)

If the target is not refractory, an input volley deposits its full charge on the
spike:

$$
I_n(t) = \sum_i w_{ni}(t)\,s_i(t)\,\mathrm{factor}_{ni},
\qquad
V_n(t^+) = V_n(t) + I_n(t),
\qquad
\mathrm{last\_input}_i = s_i(t).
$$

A refractory neuron's `receive_input()` is a no-op.

### Distance attenuation (`distance_weighting = True`)

The delivered drive is scaled per afferent by a fixed geometric attenuation
(delivery only — it never changes the stored weight):

$$
\mathrm{factor}_{ni}
= \left(\frac{\mathrm{distance\_ref}}{\max(d_{ni},\,\mathrm{distance\_min})}\right)^{\mathrm{distance\_power}} .
$$

For L2E the per-afferent distances $d_{ni}$ are the euclidean L2E-to-pixel
distances from the 3D layout (`distance_ref = 7.472` is the farthest such distance,
so every factor is $\ge 1$: nearest pixel ~3.5x, farthest 1.0x). All other
populations keep $d = 1$ (no attenuation).

## Threshold and Firing

A neuron fires when $r_n \le 0 \land V_n \ge \theta_n$. On fire:

$$
v_{\mathrm{pre}} \leftarrow V_n,\quad
V_n \leftarrow R_n,\quad
r_n \leftarrow \mathrm{refractory\_period},\quad
\mathrm{spiked}_n \leftarrow \mathrm{True},
$$

then excitatory plasticity runs using $v_{\mathrm{pre}}$. Winner ranking uses raw
membrane potential; there is no membrane noise and no winner facilitation.

## Leak and Refractory Update

The slow calcium sensor updates first,
$ca_n \leftarrow ca_n + \alpha_{\mathrm{ca}}(\mathrm{spiked}_n - ca_n)$, then:

$$
(V_n, r_n) \leftarrow
\begin{cases}
(R_n,\ r_n - 1), & r_n > 0 \\
\big((1-\lambda_n)V_n,\ r_n\big), & r_n \le 0
\end{cases}
$$

With $\lambda_n = 0$ (default) the membrane is a pure integrator between resets.

## Excitatory Plasticity

Runs only on the postsynaptic neuron's own spike, on positive synapses. The
closeness signal is $p_{\mathrm{exc}} = \mathrm{clamp}(\theta / v_{\mathrm{pre}}, 0, 1)$.

### Shared bounded kernel

For a positive weight $w$ with bounds $w_{\min}, w_{\mathrm{cap}}$, let
$q = \mathrm{clamp}\big((w - w_{\min})/(w_{\mathrm{cap}} - w_{\min}), 0, 1\big)$.
A direction-aware bounded update with gain $\eta$ and signal $\sigma \in \{+1,-1\}$:

$$
\Delta w =
\begin{cases}
+\eta\,(1 - q^2), & \sigma = +1 \quad (H_{\mathrm{up}}) \\
-\eta\,\big(1 - (1-q)^2\big), & \sigma = -1 \quad (H_{\mathrm{down}})
\end{cases}
\qquad
w \leftarrow \mathrm{clip}(w + \Delta w,\; w_{\min},\; w_{\mathrm{cap}}).
$$

The downward branch is the **reflection** of the upward one: it is zero at
$w_{\min}$ (a floored weight cannot go lower) and maximal at $w_{\mathrm{cap}}$ (a
capped losing weight can still be depressed). $w_{\mathrm{cap}} \le w_{\min}$ is a
no-op. This one kernel is used by both the signed-spike winner rule and the
competitive-depression loser update.

### Signed-spike rule with structural free-energy gate (L2E feedforward)

On fire, every positive synapse takes a $\pm 1$ update: $+1$ if its input
participated in this volley, $-1$ if not. With `structural_free_energy` on, the
gain is the **structural maturity brake** (which replaces $p_{\mathrm{exc}}$):

$$
\mathrm{gate} = \max\!\Big(\mathrm{eta\_floor},\ 1 - \mathrm{clamp}\big(\tfrac{\sum_i \max(w_i,0)}{\theta}, 0, 1\big)\Big),
\qquad
\eta = \mathrm{learning\_rate}\cdot\mathrm{gate},
$$

$$
\sigma_i = \begin{cases} +1, & \mathrm{last\_input}_i > 0.5 \\ -1, & \text{otherwise,} \end{cases}
\qquad
w_i \leftarrow \text{bounded\_kernel}(w_i,\ w_{\min},\ \mathrm{weight\_cap},\ \eta,\ \sigma_i).
$$

`gate` uses only this neuron's own positive-afferent sum and its own threshold — no
voltage, no rivals, no labels. An under-built neuron ($\sum w^+ \ll \theta$) stays
fully plastic; one whose excitatory support already covers a threshold crossing
gates to the floor. There is no weight budget on this path: the $-1$ signal on OFF
pixels supplies the downward pressure.

### Charge rule (L2E -> L2I recruitment)

The shared inhibitory neuron's incoming positive weights use charge potentiation on
its own fire (participating synapses only), with the quadratic saturation ceiling
$w_{\max} = \mathrm{weight\_cap}^2\cdot\mathrm{ei\_sat\_mult}$:

$$
\mathrm{active}_i = (w_i > 0)\land(\mathrm{last\_input}_i > 0.5),
\qquad
\Delta w_i = \eta_{\mathrm{exc}}\,p_{\mathrm{exc}}\Big(1 - \frac{w_i^2}{w_{\max}}\Big),
$$

then the floor/clip tail (below). With `ei_sat_mult = 4`, $\sqrt{w_{\max}}$ lies
above the hard clip, so a habitually-participating `L2E -> L2I` synapse climbs all
the way to the cap (= thr_l2i) and becomes a **single-source relay**: one spike
from that trusted source fires L2I on its own.

### Assembly flow credit (L2E -> L1I feedback)

L1I integrates a *sequence* of L2E winners, so its E->I credit must cover the whole
window, not just the last winner. On L1I's own fire, credit is split by the flow
each synapse delivered over the retention window — the per-synapse eligibility
trace $\phi_i$ — normalized by the max so the dominant driver gets the full rate
(non-contributor decay is 0 in the dashboard):

$$
\phi_{\max} = \max_{i:\,w_i>0}\phi_i,
\qquad
\Delta w_i =
\begin{cases}
\eta_{\mathrm{exc}}\,p_{\mathrm{exc}}\,\dfrac{\phi_i}{\phi_{\max}}\Big(1 - \dfrac{w_i^2}{w_{\max}}\Big), & \phi_i > 0 \\
0, & \phi_i = 0.
\end{cases}
$$

### Floor / clip tail

After the charge and assembly rules (the signed rule returns before it), positive
weights are floored and hard-clipped (no budget renormalization is active, since no
`weight_budget` is set):

$$
w_i \leftarrow \max(w_i,\ w_{\min}) \ \ (w_i>0),
\qquad
w_i \leftarrow \mathrm{clip}(w_i,\ -\mathrm{weight\_cap},\ \mathrm{weight\_cap}).
$$

## L2 Competition (chunked WTA + competitive reset)

L2 competition resolves **every step** (`event_driven`). This step's L1E->L2E drive
is delivered in $K = 20$ equal chunks ($w_{ji}/K$ per active synapse) inside the
frozen outer timestep; after each chunk the WTA is re-attempted and the first chunk
that produces a threshold-crosser resolves it (the earliest strong responder wins
before rivals pile up charge). When the WTA is attempted:

$$
\mathcal{E}(t) = \{\,j \mid r_{\mathrm{L2E}_j} \le 0 \land V_{\mathrm{L2E}_j} \ge \theta_{\mathrm{L2}}\,\},
\qquad
j^\ast = \arg\max_{j \in \mathcal{E}(t)} V_{\mathrm{L2E}_j}.
$$

The winner `L2E_{j*}` fires (resetting itself) and drives L2I. **If L2I crosses its
own threshold**, it fires and issues an unweighted competitive-reset event to every
non-winner; if it does not, no reset or depression occurs this step.

### Competitive reset + depression (per non-winner `j`)

The event has two parts, using the pre-reset charge
$p_{\mathrm{loss}} = \mathrm{clamp}(V_{\mathrm{pre},j}/\theta, 0, 1)$:

**Structural (competitive depression, `loser_depression` on).** Only the *positive*
feedforward weights whose inputs participated in the losing response are depressed,
via the shared bounded kernel with $\sigma = -1$:

$$
\eta_{\mathrm{loss}} = \mathrm{learning\_rate}\cdot\mathrm{gate}_j\cdot p_{\mathrm{loss}},
\qquad
w_{ji} \leftarrow \text{bounded\_kernel}(w_{ji},\ w_{\min},\ \mathrm{weight\_cap},\ \eta_{\mathrm{loss}},\ -1)
$$

for each afferent with $w_{ji} > 0 \land \mathrm{last\_input}_{ji} > 0.5$, where
$\mathrm{gate}_j$ is the same structural brake as the signed rule. OFF pixels are
never touched (that would potentiate absent pixels). A zero-charge loser
($p_{\mathrm{loss}} = 0$) learns nothing.

**Transient (hard reset, unconditional).** The membrane and pending current traces
are cleared, even if the target is refractory (the refractory timer itself is left
untouched):

$$
V_j \leftarrow R_j,\qquad \text{exc\_trace}_j \leftarrow 0,\qquad \text{inh\_trace}_j \leftarrow 0.
$$

There is **no learned inhibitory magnitude** anywhere on this path; the reset is
binary and complete.

This is the currently implemented rule. It intentionally does not include
OFF-gate recruitment, active-gate-sum matching, conserved redistribution, or
refractory winner protection; those changes are specified but not yet implemented
in `Inhibitory_Off_Weight_Recruitment_Spec.md`.

## Inhibitory Delivery (L1I -> L1E feedback)

The one path that still uses a weighted inhibitory synapse. When L1I fires, its
paired L1E takes a floored subtraction through the L1E's (frozen, saturated)
negative gate of magnitude $w$:

$$
v_{\mathrm{pre}} \leftarrow V,\qquad V \leftarrow \max(V - w,\ R),\qquad v_{\mathrm{post}} \leftarrow V.
$$

Inhibition cannot push the membrane below rest, and a refractory target takes no
discharge. The L1E gate is fixed at saturation, so no inhibitory plasticity runs in
the default configuration.

## Simulation Step Order

At each engine step ($\mathrm{input\_arrives} = (t \bmod \mathrm{input\_period} = 0)$,
$\mathrm{cycle\_boundary} = (t \bmod \mathrm{cycle\_period} = 0)$):

1. L1E receives external pixel drive on input-arrival steps: `L1E_i.receive_input([0, ext_i])`.
2. The one-step L1I feedback register suppresses L1E where
   $\mathrm{input\_arrives} \land \mathrm{l1i\_feedback\_delay}_i = 1$
   via `L1E_i.apply_inhibition([1, 0])`.
3. L1E neurons over threshold fire.
4. L1E spikes are delivered to every L2E receptive field (instantaneous charge,
   distance-attenuated).
5. L2 competition resolves (chunked WTA + competitive reset; above).
6. The L2E winner spike is delivered to all L1I neurons.
7. L1I fires when its accumulated feedback crosses its own threshold (trainable
   accumulator; all L1I start from a copy of one random L2E feedback vector, so
   every L1I observes the same global winner stream and no arbitrary phase classes
   form). A refractory of 2 blocks exactly the next outer step.
8. Emitted synapses are recorded for visualization.
9. All neurons run `update()` (calcium sensor, refractory countdown, membrane leak).
10. `l1i_feedback_delay` is replaced by the current L1I spike vector, so an L1I
    spike at $t$ inhibits its paired L1E at $t+1$ only. Under trained constant drive
    this gives a synchronized `fire, suppress, fire, ...` rhythm that halves the
    active-pixel frequency.
11. Sparse changed weights/confidence and the episode winner readout update for the
    dashboard.

## Episode Winner Readout

Interpretation only — it never touches potentials, weights, learning, or firing. An
episode starts on a volley tick if none is active and records L2E spikes until
$t - t_{\mathrm{last\_L2\_spike}} \ge \mathrm{EPISODE\_QUIET\_K}$ or
$T_{\mathrm{episode}} \ge \mathrm{EPISODE\_MAX\_LEN}$. The reported winner is the
latest L2E spiker (ties broken by most spikes in the episode).

## Open Problems

### 1. Interleaved one-to-one assignment

The mechanisms give participation and competition but not stable one-to-one
pattern-to-neuron assignment (typically 2 distinct sustained owners across the four
patterns). All four patterns share the center pixel, which is what lets a single
neuron win multiple patterns once it is cleared to.

### 2. Competition timing vs. tyranny (round-robin)

On contested patterns several L2E round-robin because the competitive reset is
gated on L2I firing, which requires accumulating ~3 L2E spikes; during that window
rivals with under-trained `L2E -> L2I` weights fire "for free" with no reset. This
slow resolution is **load-bearing**: it gives rivals the free firing windows they
need to specialize on other patterns. Clearing rivals faster (resetting every
winner's non-winners immediately) resolves competition perfectly but collapses to a
single global tyrant that owns all four patterns — the recruitment-vs-consolidation
frontier. See `L2_Hard_Reset_Competitive_Depression_Report.md` and
`Inhibition_And_Consolidation_State.md`.

### 3. Cold start on pattern switch

The signed rule's $-1$ signal depresses every OFF (non-participating) gate toward
the floor while one pattern is held, so switching to a new pattern starts those
pixels cold. The rule has no memory distinguishing "never mine" from "belonged to a
pattern I haven't seen lately."
