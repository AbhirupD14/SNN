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
  neurons and can suppress its paired input.
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
membrane_noise      = 0.0
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
event_driven            = False
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

If the target neuron is not refractory:

$$
\begin{aligned}
I_n(t) &= \sum_i w_{ni}(t)\,s_i(t) \\
V_n(t^+) &= V_n(t) + I_n(t) \\
\mathrm{last\_input}_i &= s_i(t)
\end{aligned}
$$

If the neuron is refractory, `receive_input()` is a no-op.

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
synapse.

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

5. L1E spikes are delivered immediately to every L2E feedforward receptive
   field.

6. Optional membrane noise perturbs non-refractory L2E potentials. The default is
   zero, so the default run is deterministic.

7. On an intrinsic cycle boundary, L2 competition resolves:

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

9. L1I neurons that crossed threshold fire.

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
