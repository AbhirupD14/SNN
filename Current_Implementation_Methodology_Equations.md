# Current Implementation Methodology and Equations

This document describes the current implementation on branch
`feature/inhibitory-plasticity` as of 2026-07-07. It is descriptive, not a new
proposal.

## Current Status

The network currently has good L2E participation and inhibition-mediated
competition, but it does not reliably consolidate the eight 3x3 line primitives
into a one-to-one pattern-to-neuron map.

Observed from the current tests:

- `test_l2_competition.py`: all 8 L2E neurons participate, L2I fires, L2I->L2E
  gate discharges occur, and gates adapt. Pattern winners are differentiated but
  still collide, typically around 4 to 6 distinct winners across 8 patterns.
- `test_8line_consolidation.py`: the older interleaved characterization path
  still shows failure to form full assignment; in that test every pattern maps
  to one L2E winner.

So the active issue is assignment/consolidation, not dead competition.

## Network Topology

The dashboard engine builds a two-layer network around the eight 3x3 line
patterns:

- `N_PIX = 9`: one input pixel per grid cell.
- `N_OUT = 8`: one L2E candidate output neuron per primitive.
- `L1E_i`: fixed pixel encoder for pixel `i`.
- `L1I_i`: paired inhibitory neuron for `L1E_i`; receives feedback from all L2E
  neurons and can suppress its paired input. By default (`l1i_immediate_relay`) it
  is a deterministic relay -- it fires on any nonzero L2E feedback rather than on a
  learned threshold crossing (see step 9).
- `L2E_j`: trainable output neuron with one feedforward synapse from each L1E
  pixel plus one local inhibitory gate from L2I.
- `L2I`: one shared inhibitory neuron receiving from all L2E neurons and
  suppressing the L2E pool through per-target gates.

The eight input patterns are:

```text
row 0, row 1, row 2,
col 0, col 1, col 2,
diag \, diag /
```

## Default Engine Parameters

`SimulationEngine` defaults:

```text
threshold           = 1.0
threshold_l2        = 8.0
leak_l1             = 0.10
leak_l2             = 0.01
learning_rate       = 0.05
weight_cap          = 1.0
refractory          = 2
volley_period       = 4
input_period        = volley_period
cycle_period        = volley_period
homeostasis         = False
ca_rate             = 0.01
ca_target           = 0.012
homeo_up            = 0.01
homeo_down          = 0.01
l2e_lr_frac         = ETA_FRAC
l2i_lr_frac         = ETA_FRAC
l1i_lr_frac         = ETA_FRAC
l2_gate_eta         = L2_GATE_ETA
l2i_threshold_frac  = 1.0
l1i_threshold_frac  = 1.0
ei_sat_mult         = 1.0
l1i_ei_init_frac    = None
confidence_consolidation = True
loser_depression        = True
conf_cap_frac           = 1/3
eta_min                 = 0.05
eta_loss                = 0.01
signed_depression       = True
eta_off                 = 0.20
l2e_budget              = True
event_driven            = True
l2_charge_chunks        = 1
l1i_immediate_relay     = True
excitatory_flow_rate    = False
exc_trace_decay         = 0.8
exc_trace_normalized    = True
lasting_inhibition      = False
```

Module constants:

```text
L2_GATE_INIT                  = -0.5
L2_GATE_WMAX                  = 1.5
L2_GATE_ETA                   = 0.1
L2_EI_WEIGHT_INIT_LOW_FRAC    = 0.25
L2_EI_WEIGHT_INIT_HIGH_FRAC   = 0.5
L2I_LEAK_RATE                 = 0.07
L1_EI_WEIGHT_INIT_LOW_FRAC    = 0.25
L1_EI_WEIGHT_INIT_HIGH_FRAC   = 0.5
L1I_LEAK_RATE                 = 0.07
ETA_FRAC                      = 0.01
L2E_MIN_WEIGHT_FLOOR          = 0.01
```

The dashboard API overrides the engine for visualization:

```text
homeostasis        = False
l2e_lr_frac        = 0.02
ei_sat_mult        = 4.0
l1i_ei_init_frac   = None
```

## State Variables

For neuron `n`:

$$
\begin{array}{ll}
V_n & \text{membrane potential} \\
\theta_n & \text{firing threshold} \\
r_n & \text{refractory timer} \\
R_n & \text{resting potential, currently } 0.0 \\
w_{ni} & \text{weight of afferent synapse } i \\
s_i & \text{input spike on afferent } i,\ \text{usually } 0 \text{ or } 1 \\
\lambda_n & \text{leak rate}
\end{array}
$$

Positive weights are excitatory. Negative weights are inhibitory. Excitatory
and inhibitory neurons use the same neuron dynamics; inhibition or excitation is
encoded by the sign of the weight landing on the target.

## Charge Integration

There are two charge-accumulation models, selected by `excitatory_flow_rate`.
The default is the **instantaneous** model (a weight is a charge packet deposited
in full on the spike). The opt-in **flow-rate** model (a weight is a current
amplitude integrated over time) is derived in its own section below. Both feed the
SAME unchanged event-based threshold / winner / inhibition procedure — flow-rate
only changes how positive charge accumulates *before* the threshold check.

### Instantaneous model (default, `excitatory_flow_rate = False`)

If the target neuron is not refractory:

$$
\begin{aligned}
I_n(t) &= \sum_i w_{ni}(t)\,s_i(t) \\
V_n(t^+) &= V_n(t) + I_n(t) \\
\mathrm{last\_input}_i &= s_i(t)
\end{aligned}
$$

If the neuron is refractory, `receive_input()` is a no-op.

## Excitatory Flow-Rate Accumulation (`excitatory_flow_rate = True`)

An optional, sparser interpretation of a synapse: the **weight is a current /
gate amplitude**, a spike opens an excitatory **current trace**, the membrane
integrates that current over time, and the current decays. This replaces the
instantaneous charge packet above only for the positive-charge integrators —
**L2E** (from L1E spikes), **L2I** (from L2E spikes), and **L1I** (from L2E
feedback, only when it is a trainable integrator, i.e. NOT in
`l1i_immediate_relay` mode). **L1E is exempt**: it is an abstract pretrained
sensory source and keeps instantaneous dynamics; while a pattern is held it
re-fires every `input_period` steps, so it is a sustained periodic current source
for the traces downstream.

### State

Per neuron: the current trace amplitude $I$ (`exc_trace`) and the last outer
timestep it was advanced to, $t_\ell$ (`exc_trace_last_t`). Constant decay
$d = $ `exc_trace_decay` $\in [0, 1)$.

### Dense recursion (reference)

Conceptually, every timestep the membrane absorbs the current and the current
decays:

$$
V \mathrel{+}= I, \qquad I \mathrel{\leftarrow} d\,I .
$$

### Closed-form lazy (skipped-time) advance

Running that recursion densely for every neuron every step does not scale. Since
$I$ only decays between events, the effect of $\Delta t = t - t_\ell$ idle steps
is a geometric sum, applied lazily in $O(1)$ when the neuron is next touched:

$$
V \mathrel{+}= I \cdot \frac{1 - d^{\,\Delta t}}{1 - d},
\qquad
I \mathrel{\leftarrow} I\,d^{\,\Delta t},
\qquad
t_\ell \mathrel{\leftarrow} t .
$$

Edge cases: $d = 0$ makes the factor $1$ (the current contributes exactly once);
as $d \to 1$ the factor's limit is $\Delta t$, used directly to avoid the
$1/(1-d)$ blow-up. `advance_trace(t)` implements this; `receive_input(..., t)`
calls it, then injects, then integrates one step (below). Only touched neurons
(or, for future scale, scheduled trace-bearers) pay the cost — an idle neuron is
never swept. Because residual current keeps flowing, a neuron can cross threshold
on a timestep with **no new input**.

### Normalized injection

On a spike at time $t$ with drive $g = \sum_i \max(w_{ni}, 0)\,s_i(t)$, the
current is injected as (default, `exc_trace_normalized = True`):

$$
I \mathrel{+}= g\,(1 - d) .
$$

The infinite geometric total then delivered to $V$ is
$g(1-d)\sum_{k\ge 0} d^{k} = g$, so the flow-rate total charge matches the
instantaneous $g$ — the two models are magnitude-comparable. With
`exc_trace_normalized = False`, $I \mathrel{+}= g$ and the total is $g/(1-d)$.

### Ordering (deterministic, same-timestep contribution)

On each outer timestep, a touched neuron: (1) lazily advances residual current up
to $t-1$; (2) injects new drive into $I$; (3) applies one integration step at $t$
($V \mathrel{+}= I;\ I \leftarrow dI$) so a fresh spike still moves $V$ this same
timestep; (4) clamps $V$ to the membrane saturation ceiling exactly as the
instantaneous path does. **Threshold checks happen after** this contribution, so
they may fire on no-new-input timesteps. Refractory neurons neither advance nor
inject (no hidden current builds while clamped); `fire()` discharges $I$ to $0$
along with the membrane.

### Interaction with chunking

Flow-rate mode is the finer temporal representation, so it is **not** combined
with the artificial `l2_charge_chunks` weight splitting:

$$
\text{effective } l2\_charge\_chunks =
\begin{cases}
1 & \text{if } excitatory\_flow\_rate \\
l2\_charge\_chunks & \text{otherwise.}
\end{cases}
$$

### Distance attenuation (`distance_weighting`, opt-in)

Three separable roles for a synapse: **weight = learned gate strength**,
**distance = delivery attenuation**, **trace = temporal flow**. Distance weighting
scales only the *delivered drive amplitude* — it does not change the stored weight
and is **not** part of the trace decay/integration math. For each afferent $i$
into target $j$ with distance $d_{ji}$:

$$
\mathrm{factor}_{ji}
= \left(\frac{\mathrm{distance\_ref}}{\max(d_{ji},\,\mathrm{distance\_min})}\right)^{\mathrm{distance\_power}},
\qquad
g_j = \sum_i \mathrm{spike}_i \; w_{ji}\,\mathrm{factor}_{ji}.
$$

This effective drive $g_j$ replaces the raw $\sum_i \mathrm{spike}_i\,w_{ji}$ in the
same place in both accumulation modes. In flow-rate mode the normalized injection is
then $I_j \mathrel{+}= g_j(1 - d)$, so the total future charge one spike delivers is
$\approx w_{ji}\,\mathrm{factor}_{ji}$ (e.g. $w_{ji}/d_{ji}^2$ with
`distance_power=2`, `distance_ref=1`, `distance_min=1`) rather than raw $w_{ji}$. In
instantaneous mode the same $g_j$ is used and remains independent of chunking
(`l2_charge_chunks` merely splits $g_j$ into $K$ equal pieces). Distances are stored
per synapse and default to $1$ (no attenuation). `distance_weighting` is OFF by
default, and the delivery ($V \leftarrow \max(V-w, R)$ for inhibition,
etc.) is otherwise unchanged.

## Threshold and Firing

A neuron fires when:

$$
r_n \le 0 \quad \land \quad V_n \ge \theta_n
$$

On fire:

$$
\begin{aligned}
v_{\mathrm{pre}} &\leftarrow V_n \\
V_n &\leftarrow R_n \\
r_n &\leftarrow \mathrm{refractory\_period} \\
\mathrm{spiked}_n &\leftarrow \mathrm{True}
\end{aligned}
$$

Then excitatory plasticity runs using $v_{\mathrm{pre}}$, and the archived trace
is cleared.

Winner facilitation is not part of the current core model. Threshold checks and
L2 winner ranking use raw membrane potential.

## Leak and Refractory Update

The calcium/homeostasis sensor is updated first:

$$
ca_n \leftarrow ca_n + \alpha_{\mathrm{ca}}\left(\mathrm{spiked}_n - ca_n\right)
$$

where $\alpha_{\mathrm{ca}} = \mathrm{ca\_rate}$.

Then:

$$
(V_n, r_n) \leftarrow
\begin{cases}
(R_n,\ r_n - 1), & r_n > 0 \\
\left(V_n + \lambda_n(R_n - V_n),\ r_n\right), & r_n \le 0
\end{cases}
$$

Since `R_n = 0.0`, the non-refractory leak is:

$$
V_n \leftarrow (1 - \lambda_n)V_n
$$

The archived trace decays with the same factor but is no longer used for
learning.

## Excitatory Plasticity

Excitatory plasticity runs only when the postsynaptic neuron fires. It updates
only positive synapses that participated in the most recent input event:

$$
\begin{aligned}
\mathrm{active}_i
  &= (w_i > 0) \land (\mathrm{last\_input}_i > 0.5) \\
p_{\mathrm{exc}}
  &= \mathrm{clamp}\left(\frac{\theta}{v_{\mathrm{pre}}}, 0, 1\right) \\
w_{\max}
  &=
  \begin{cases}
  \mathrm{excitatory\_saturation\_cap}, & \text{if set} \\
  \mathrm{weight\_cap}, & \text{otherwise}
  \end{cases} \\
\Delta w_i
  &= \eta_{\mathrm{exc}}\,p_{\mathrm{exc}}
     \left(1 - \frac{w_i^2}{w_{\max}}\right) \\
w_i
  &\leftarrow w_i + \Delta w_i
\end{aligned}
$$

where $\eta_{\mathrm{exc}} = \mathrm{learning\_rate}$.

Then the shared budget/cap tail runs.

For L2E feedforward weights, the fixed positive-weight budget is normally:

$$
\sum_i \max(w_i, 0) = 2\theta_{\mathrm{L2}}
$$

unless homeostasis is enabled, in which case the target resource is the
homeostatic resource `R_homeo`.

The budget/cap tail is:

$$
T =
\begin{cases}
R_{\mathrm{homeo}}, & \text{if homeostasis is enabled} \\
\mathrm{weight\_budget}, & \text{otherwise}
\end{cases}
$$

If $T$ is set and $S_+ = \sum_{i:w_i>0} w_i > 0$, every positive synapse is
renormalized as:

$$
w_i \leftarrow w_i \frac{T}{S_+}
\qquad \text{for all } i \text{ where } w_i > 0
$$

If `min_positive_weight` is set, then:

$$
w_i \leftarrow \max(w_i, w_{\min})
\qquad \text{for all } i \text{ where } w_i > 0
$$

Finally:

$$
w_i \leftarrow \mathrm{clip}(w_i, -\mathrm{weight\_cap}, \mathrm{weight\_cap})
$$

## Inhibitory Plasticity

Inhibitory plasticity is an independent event-driven rule. It runs only when an
inhibitory spike is delivered to a non-refractory target through a negative
synapse. There are two rules, selected by `inhibitory_delta_rule`: the legacy
**saturating** rule (below) and the default **delta/margin** rule (further down).
Both use the SAME linear floored-at-rest delivery ($V \leftarrow \max(V-w, R)$)
and only differ in how the gate magnitude is learned.

### Saturating rule (`inhibitory_delta_rule = False`, legacy)

For each active negative synapse:

$$
\begin{aligned}
w &= |\mathrm{weight}| \\
v_{\mathrm{pre}} &\leftarrow V \\
V &\leftarrow \max(V - w, R) \\
v_{\mathrm{post}} &\leftarrow V \\
p_{\mathrm{inh}}
  &= \mathrm{clamp}\left(\frac{v_{\mathrm{pre}}}{\theta}, 0, 1\right) \\
w_{\max}
  &=
  \begin{cases}
  \mathrm{inhibitory\_weight\_cap}, & \text{if set} \\
  \mathrm{weight\_cap}, & \text{otherwise}
  \end{cases} \\
\Delta w
  &= \eta_{\mathrm{inh}}\,p_{\mathrm{inh}}
     \left(1 - \frac{w^2}{w_{\max}}\right) \\
w_{\mathrm{new}}
  &= \mathrm{clip}(w + \Delta w, 0, w_{\max}) \\
\mathrm{weight}
  &\leftarrow -w_{\mathrm{new}}
\end{aligned}
$$

where $\eta_{\mathrm{inh}} = \mathrm{inhibitory\_learning\_rate}$.

Current safety behavior:

- A refractory target gets no inhibitory discharge and no gate update.
- Inhibition is floored at resting potential, so it removes existing charge but
  cannot push a membrane negative.

The quadratic term has natural zero-growth point:

$$
w^\ast = \sqrt{w_{\max}}
$$

when no hard clip intervenes. This is why some code decouples the hard clip from
the quadratic saturation ceiling.

**Why this saturates uniformly.** The equilibrium `sqrt(w_max)` is the *same for
every target* — `p_inh` only scales the *rate* of approach, never the destination.
So any gate discharged enough times converges to the identical ceiling (observed:
all L2I→L2E gates at ≈1224.7 with spread <0.4%). The gate magnitude ends up
encoding "has this been inhibited enough to saturate," not "how much inhibition
does this target need."

### Turnover rule (`inhibitory_delta_rule = True`, `inhibitory_rule_mode = "turnover"`, default)

To make the equilibrium **target-specific without a hand-set target voltage and
without any average**, the default rule updates the *normalized* gate
$u = w / G$, where the ceiling $G = \sqrt{w_{\max}}$ is the same sub-threshold
scale the saturating rule converges to. On each real discharge into a
non-refractory target:

$$
\begin{aligned}
p_t &= \mathrm{clip}\!\left(\frac{v_{\mathrm{pre}}}{\theta},\; 0,\; p_{\max}\right) \\
\Delta u &= \eta_{\uparrow}\,p_t\,(1 - u)\;-\;\eta_{\downarrow}\,u \\
u_{\mathrm{new}} &= \mathrm{clip}(u + \Delta u,\; 0,\; 1),
\qquad w_{\mathrm{new}} = u_{\mathrm{new}}\,G,
\qquad \mathrm{weight} \leftarrow -w_{\mathrm{new}}
\end{aligned}
$$

with $\eta_{\uparrow} = \mathrm{inhibitory\_eta\_up}$,
$\eta_{\downarrow} = \mathrm{inhibitory\_eta\_down}$,
$p_{\max} = \mathrm{inhibitory\_p\_max}$. The first term **strengthens** the gate
in proportion to the target's charge ($p_t$) and its remaining headroom ($1-u$);
the second is a **size-proportional turnover** that decays every gate toward zero.
A high-charge rival that repeatedly triggers large strengthening accumulates a big
gate; a weak/dead target that rarely contributes charge is dominated by turnover
and drifts down. Gates differentiate (observed spread ≈260 vs 4 for the saturating
rule) while competition does not collapse.

The algorithm stores and computes **no average** — the update is purely
event-local. For *offline* intuition only, the per-event fixed point
($\Delta u = 0$) at a fixed $p_t$ is
$u^\ast = \eta_{\uparrow} p_t / (\eta_{\downarrow} + \eta_{\uparrow} p_t)$, which
increases with $p_t$; this is descriptive, not part of the rule.

**Locality.** The update uses only the inhibitory synapse's own weight $w$, the
target neuron's $v_{\mathrm{pre}}$ and $\theta$, and the arriving inhibitory spike.
No global rank, population average, or winner identity enters it.

### Margin rule (`inhibitory_rule_mode = "margin"`, diagnostic)

A non-default diagnostic that relaxes the gate toward the magnitude which would
bring the target to a fixed post-inhibition level
$\mathrm{target\_post} = \mathrm{inhibitory\_margin\_frac}\cdot\theta$:

$$
s = \mathrm{clip}(v_{\mathrm{pre}} - \mathrm{target\_post},\,0,\,G),
\qquad
\Delta w = \eta_{\delta}\,(s - w),
\qquad w^\ast = s,
$$

$\eta_{\delta} = \mathrm{inhibitory\_delta\_eta}$. It also differentiates, but its
equilibrium is a hand-tuned target voltage; the turnover rule is preferred because
it needs none.

**Default-run context.** This inhibitory change is orthogonal to the excitatory
path: `signed_spike_learning` remains the canonical feedforward rule and is
untouched by it. Instantaneous-vs-flow-rate accumulation is controlled separately
(`excitatory_flow_rate`). The delivery (linear, floored at rest) is unchanged.

## Homeostatic Scaling

Homeostasis is local and non-Hebbian. It regulates the neuron's positive-weight
resource from its own slow firing-rate sensor:

$$
\begin{aligned}
ca_{\mathrm{lo}} &= ca_{\mathrm{target}}(1 - ca_{\mathrm{band}}) \\
ca_{\mathrm{hi}} &= ca_{\mathrm{target}}(1 + ca_{\mathrm{band}})
\end{aligned}
$$

$$
R_{\mathrm{homeo}} \leftarrow
\begin{cases}
R_{\mathrm{homeo}}(1 + u), & ca < ca_{\mathrm{lo}} \\
R_{\mathrm{homeo}}(1 - d), & ca > ca_{\mathrm{hi}} \\
R_{\mathrm{homeo}}, & ca_{\mathrm{lo}} \le ca \le ca_{\mathrm{hi}}
\end{cases}
$$

where $u = \mathrm{homeo\_up}$ and $d = \mathrm{homeo\_down}$.

Then positive weights are multiplicatively rescaled to the new resource. This
preserves relative receptive-field shape and carries no pattern label or global
signal.

## Simulation Step Order

At each engine step:

1. Compute:

$$
\begin{aligned}
\mathrm{input\_arrives}
  &= (t \bmod \mathrm{input\_period} = 0) \\
\mathrm{cycle\_boundary}
  &= (t \bmod \mathrm{cycle\_period} = 0)
\end{aligned}
$$

2. L1E receives external pixel drive on input-arrival steps:

$$
\mathrm{ext}_i =
\begin{cases}
1, & \mathrm{input\_arrives} \land \mathrm{input\_vec}_i = 1 \\
0, & \text{otherwise}
\end{cases}
$$

Then `L1E_i.receive_input([0, ext_i])` is called.

3. The previous L1I latch suppresses L1E on input-arrival steps:

$$
\mathrm{input\_arrives} \land \mathrm{l1i\_hold}_i = 1
$$

When this condition is true, `L1E_i.apply_inhibition([1, 0])` is called.

4. L1E neurons that crossed threshold fire.

5. L1E spikes are delivered to every L2E feedforward receptive field, charging
   the membrane by the instantaneous or flow-rate model (see Charge Integration /
   Excitatory Flow-Rate Accumulation).

6. L2E integration is deterministic: there is no membrane-potential noise.

7. L2 competition resolves every step (`event_driven`, the default per-step
   single-winner flow); with `event_driven` off it resolves only on an intrinsic
   cycle boundary. In the instantaneous model this step's L1->L2E feedforward
   drive is delivered in effective `l2_charge_chunks` = $K$ equal chunks
   ($w_{ji}/K$ per active synapse) inside the frozen outer timestep; after each
   chunk the WTA below is re-attempted and the first chunk that yields a
   threshold-crosser resolves the competition (the rest are skipped). $K=1$
   (default) delivers the full drive at once. In flow-rate mode $K$ is forced to 1
   (the current trace already spreads the drive over time). $K=1$ is the
   un-chunked baseline. When the WTA is attempted:

$$
\mathcal{E}(t) =
\left\{j \mid r_{\mathrm{L2E}_j} \le 0
\land V_{\mathrm{L2E}_j} \ge \theta_{\mathrm{L2}}\right\}
$$

If $\mathcal{E}(t)$ is non-empty, the instantaneous firing winner is:

$$
j^\ast = \arg\max_{j \in \mathcal{E}(t)} V_{\mathrm{L2E}_j}
$$

Then `fire()` is called on the winning `L2E` neuron, a one-hot L2E spike vector
is delivered to L2I, and if L2I crosses threshold:

$$
\forall j \ne j^\ast,\quad \mathrm{L2E}_j.\mathrm{apply\_inhibition}([1,0,\ldots,0])
$$

The winner is not procedurally protected after firing. The non-winner L2E
neurons are suppressed through their actual L2I->L2E negative gate.

8. The L2E winner spike is delivered immediately to all L1I neurons.

9. L1I fires. With `l1i_immediate_relay` = True (default) L1I is an **immediate
   deterministic relay**: any L1I that received a nonzero L2E feedback signal fires
   this same step -- it does NOT integrate charge, does NOT require a learned
   threshold crossing, and does NOT depend on L1I feedback-weight training (the
   winner vector is delivered identically to every L1I, so a winner makes every L1I
   fire). With the flag off, the legacy behavior returns: L1I fires only when its
   accumulated feedback crosses its own threshold. Output is binary in both modes,
   and the L1I->L1E inhibition path (steps 3, 12) is unchanged.

10. Emitted synapses are recorded for visualization.

11. All neurons run `update()`.

12. On cycle boundaries, `l1i_hold` is replaced by the current L1I spikes.

13. Sparse changed weights/confidence and episode-level winner readout are
    updated for the dashboard.

## Episode Winner Readout

The episode mechanism affects interpretation only. It does not modify potentials,
weights, learning, inhibition, or firing.

An episode starts on a volley tick if no episode is active. It records L2E spikes
until either:

$$
t - t_{\mathrm{last\_L2\_spike}} \ge \mathrm{EPISODE\_QUIET\_K}
$$

or:

$$
T_{\mathrm{episode}} \ge \mathrm{EPISODE\_MAX\_LEN}
$$

The reported winner is the latest L2E spiker. If there is a same-time tie, the
winner is the neuron with the most spikes within the episode.

## Current Consolidation Gap

The implementation has several mechanisms that prevent collapse:

- L2I-mediated adaptive lateral inhibition instead of hard reset.
- Pool-wide suppression of non-winning L2E neurons on L2I discharge.
- Refractory-gated inhibitory learning.
- Homeostasis or fixed weight budgets to regulate L2E resource use.
- Optional membrane noise and E/I timing controls for experiments.

Those mechanisms produce participation and competition. They do not yet produce
stable one-to-one assignment for all eight symbols. The missing piece is a
symmetry-breaking or assignment-stabilization mechanism that makes a given
pattern consistently owned by one neuron while also discouraging two patterns
from sharing that same owner.
