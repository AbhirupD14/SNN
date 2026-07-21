# Design Direction: Intrinsic Adaptation for Winner Tyranny

> **Historical (pre-conductance).** A proposed alternative from the hard-wipe era.
> The implemented model instead uses persistent inhibitory conductance and local
> predictive inhibition; see `Current_Implementation_Methodology_Equations.md`.
> Kept for context only.


## Status

This document records a proposed mechanism. Intrinsic adaptation is **not** part
of the current production model.

The existing L2I circuit is still necessary: after one L2E neuron crosses
threshold, L2I wipes the other competitors' partial charge and enforces one
winner for that competition. That inhibition keeps non-winning, untrained
weights from accumulating indefinitely.

It does not solve winner tyranny when an established L2E neuron crosses threshold
from one input volley. Such a neuron is an instant integrator for that pattern.

## Causal limit of frequency and post-spike inhibition

If the active feedforward charge for competitor `j` is already

```text
Q_j = sum(active acc_weights) >= theta,
```

then the neuron fires during that volley. Frequency and leak describe what
happens *between* volleys, so neither has an opportunity to prevent this crossing.
Inhibition emitted after the crossing cannot retract the winner's spike. Reducing
the frequency of shared sensory sources also affects every competitor receiving
those sources and does not specifically disadvantage the incumbent.

Therefore an instant one-volley winner can be controlled only by changing at
least one quantity before its next threshold evaluation:

- its effective input charge;
- its effective threshold;
- its permission to fire;
- or the active connectivity delivering that volley.

The network currently has strong spatial negative feedback for choosing a winner,
but it lacks temporal negative feedback targeted locally to the neuron that has
been winning.

## Proposed mechanism: spike-triggered intrinsic adaptation

Give each L2E neuron one nonnegative adaptation state `a_j`. All L2E neurons use
the same baseline threshold and the same adaptation equation:

```text
fires_j(t) iff V_j(t) >= theta + a_j(t)
```

When neuron `j` fires:

```text
a_j <- a_j + delta_a
```

When it does not fire, adaptation recovers toward zero:

```text
a_j <- rho * a_j,       0 < rho < 1
```

An equivalent continuous-time interpretation is exponential recovery with time
constant `tau_a`. The implementation should choose one convention and document
its exact timestep order.

The baseline excitatory threshold remains `theta = 1000` for every E neuron.
`a_j` is transient internal state, not a different configured threshold. If it is
important that the literal comparison continue using the fixed threshold, the
same dynamics can be written as a subtractive adaptation current:

```text
fires_j(t) iff V_j(t) - a_j(t) >= theta.
```

L2I's hard membrane wipe must not erase `a_j`. Inhibition resets the current
competition; adaptation remembers which neuron recently won.

## Boolean and sequential interpretation

An E neuron is a weighted threshold gate. For an adapted L2E neuron, the logical
relation becomes

```text
winner_j = feature_match_j AND NOT recently_dominant_j.
```

The first term is the learned feedforward threshold condition. The second is a
local, decaying memory set only by the neuron's own output spikes. This turns the
fabric from purely combinational threshold logic into sequential logic.

Nothing selects the replacement winner. The incumbent's own activity temporarily
reduces its eligibility, leaving the ordinary WTA competition to choose among
the remaining neurons.

## Required operating inequalities

For the incumbent to win initially:

```text
Q_winner >= theta.
```

For adaptation to prevent an immediate repeat:

```text
Q_winner < theta + a_winner.
```

An unadapted rival remains eligible when:

```text
Q_rival >= theta + a_rival,       a_rival = 0.
```

Recovery must eventually restore the incumbent:

```text
a_winner(t + k) -> 0
```

so a returning preferred pattern can recruit it again.

A fixed `delta_a` is the smallest starting model. A later experiment may compare
an overshoot-sensitive increment,

```text
delta_a_j = a_0 + beta * max(0, V_pre_j - theta),
```

which gives stronger fatigue to a neuron that crosses threshold by a large
margin. That extension should not be promoted without evidence that the fixed
increment is inadequate.

## Relationship to inhibition and refractory behavior

The mechanisms serve different roles:

- **L2I inhibition:** selects one winner within the current volley and clears
  non-winner membrane charge.
- **Intrinsic adaptation:** makes repeated victories temporarily harder for the
  neuron that actually fired.
- **Refractory period:** the hard limiting case in which firing is completely
  forbidden for a fixed number of steps.

A refractory period could force turnover, but it imposes an exact exclusion
duration. Graded adaptation is preferred because strong input can overcome
partial fatigue and recovery is continuous rather than an engine-level removal
of the winner.

## Related alternatives

These alternatives modify the same causal quantities but add different state:

1. **Short-term synaptic depression.** Each used afferent has a temporary resource
   variable, and delivered charge is `w_ij * x_ij`. It is feature-specific but
   adds state to every synapse and complicates interpretation of learned weights.
2. **Neuron-wide input gain.** Delivered charge is multiplied by one recovering
   scalar `g_j` that falls after a spike. It is compact but less directly tied to
   the current membrane model.
3. **Adaptation current.** A decaying negative internal current is mathematically
   close to an adaptive threshold while preserving a literal shared threshold.
4. **Winner-specific recurrent inhibition.** A delayed local inhibitory loop can
   veto the next volley, but this is a hard event-based exclusion unless its
   effect is graded.

The first experiment should use one adaptation scalar per L2E neuron. Do not add
per-synapse depression, noise, or a second mechanism until that minimal model has
been measured.

## Minimal experimental plan

Keep the current topology, long-term weight rule, threshold, weight cap, and L2I
WTA behavior. Add adaptation only to L2E, initially as fixed scientific constants
rather than dashboard controls.

Measure across multiple seeds and row -> column -> row schedules:

1. The original winner forms normally from an unadapted state.
2. Its adaptation changes only after its actual spike, not from a sticky winner
   identifier or membrane charge.
3. L2I wipes membranes but leaves adaptation intact.
4. The adapted winner fails at least one otherwise sufficient one-volley input.
5. Unadapted rivals remain eligible and ordinary WTA chooses the next winner.
6. No engine code explicitly disables the incumbent or chooses the replacement.
7. Adaptation decays during silence.
8. The original winner can respond again after recovery and pattern return.
9. Long-term acc_weights are unchanged by adaptation except through the existing
   rule when a neuron genuinely fires.

The experiment must distinguish actual L2E spikes from the engine's historical
winner field. Report whether the result is stable across seeds, whether it creates
oscillation without useful specialization, and whether turnover persists for a
neuron whose feedforward charge substantially exceeds threshold.

## Design principle

Spatial negative feedback selects a winner. Temporal negative feedback prevents
that winner from becoming permanent. The current inhibitory circuit supplies the
first function; intrinsic adaptation is the smallest local candidate for the
second, including in the one-volley regime where frequency modulation cannot act
before threshold crossing.
