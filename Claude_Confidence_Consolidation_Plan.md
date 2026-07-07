# Claude Plan: Confidence-Gated Consolidation Without Homeostatic Scaling

You are working in `/home/adasgup/Documents/SNN` on branch
`feature/inhibitory-plasticity`.

The current worktree may already contain local edits, including fixed-point-style
numeric constants. Do not revert or rewrite unrelated local changes. Make narrow
edits only.

## Goal

Improve one-to-one consolidation for the eight 3x3 line primitives while keeping
learning local and avoiding global labels, backpropagation, hard resets, and
procedural assignment.

The current network has good L2E participation and inhibition-mediated
competition, but patterns still collide onto shared winners. The goal of this
plan is to make stable specialists less plastic while keeping genuinely unused
or stale neurons reusable over long timescales.

## Architectural Decision

Disable homeostatic synaptic scaling in the active/default architecture.

Rationale:

- Homeostasis is a firing-rate/resource regulator, not an assignment mechanism.
- In this task it can recruit neurons into competition merely because they have
  been quiet, even when they are not a good match for the current pattern.
- The dashboard has already been run with `homeostasis=False`, so measured
  dashboard behavior reflects the fixed-budget path.

Required default state:

```python
Neuron(..., homeostasis=False)              # already true in both neuron classes
SimulationEngine(..., homeostasis=False)    # make this the backend default
backend.api SimulationEngine(homeostasis=False)
```

Keep `_homeostatic_scaling()` as an explicit opt-in experiment. Do not delete it.
Keep the calcium/activity trace, but use it as an activity-memory signal rather
than as a default weight-scaling mechanism.

## Local Confidence Metric

Introduce a local confidence or maturity variable for each positive L2E
feedforward gate. It should depend only on local synapse/target-neuron state:

- the gate weight,
- the gate's local effective cap,
- the gate's local floor,
- whether that gate was active in the current input event,
- the target neuron's own activity trace.

No labels. No pattern IDs. No global assignment table. No source-neuron internal
state beyond the spike that arrived on this synapse.

Define local instantaneous maturity:

$$
m_i =
\mathrm{clamp}
\left(
\frac{w_i - w_{\min}}
     {w_{\mathrm{conf\_cap}} - w_{\min}},
0, 1
\right)
$$

where:

- \(w_i\) is the current positive feedforward gate,
- \(w_{\min}\) is `min_positive_weight` or 0 if no floor is configured,
- \(w_{\mathrm{conf\_cap}}\) is the effective mature value for that synapse.

Important: \(w_{\mathrm{conf\_cap}}\) should be the effective reachable mature
synapse value, not necessarily the hard clip if budget normalization makes the
hard clip unreachable in ordinary training.

Maintain persistent confidence:

$$
C_i \in [0,1]
$$

On a postsynaptic L2E fire, for active positive feedforward gates:

$$
C_i \leftarrow C_i + \beta_C x_i(m_i - C_i)
$$

where \(x_i \in \{0,1\}\) is the gate's local active-spike indicator from the
most recent input event.

Interpretation:

- Strong active gates that repeatedly participate in successful firing become
  confident.
- Weak or unused gates do not become confident just because the neuron fired.
- Confidence is local to the synapse and target neuron.

## Confidence-Gated Potentiation

Use confidence to reduce excitatory plasticity on mature gates while preserving
a small plasticity floor:

$$
\eta_i^+ =
\eta_0
\left[
\eta_{\min}
+ (1-\eta_{\min})(1-C_i)
\right]
$$

Then use \(\eta_i^+\) in the existing positive-weight update for active gates.

This means:

- New/uncertain gates learn normally.
- Mature gates still learn a little.
- No gate becomes permanently frozen.

## Loser Depression

When an L2E neuron is inhibited by a real L2I->L2E discharge, apply a local
depression rule to only the active positive feedforward gates that helped make
that neuron a loser/near-winner.

Do not depress inactive gates.

Use the inhibitory event's local closeness signal:

$$
p_{\mathrm{loss}} =
\mathrm{clamp}
\left(
\frac{V_{\mathrm{pre}}}{\theta},
0, 1
\right)
$$

For active positive gates:

$$
\Delta w_i^- =
\eta_-\,
p_{\mathrm{loss}}\,
x_i\,
(1-C_i)\,
\left(
\frac{w_i-w_{\min}}
     {w_{\mathrm{conf\_cap}}-w_{\min}}
\right)^\gamma
(w_i-w_{\min})
$$

$$
w_i \leftarrow w_i - \Delta w_i^-
$$

Then run the existing floor/budget/cap tail.

Recommended starting values:

```text
eta_loss: 0.005 to 0.02
gamma:    1 or 2
eta_min:  0.02 to 0.10 of base learning
beta_C:   small; confidence should require repeated successful use
```

This rule is intentionally nonlinear in the protect-small / punish-large
direction:

- Small gates near the floor barely move.
- Large, still-unconfident active gates take the largest hit.
- Confident mature gates are protected.
- Far-away suppressed neurons learn little because \(p_{\mathrm{loss}}\) is small.

## Activity-Dependent Confidence Decay

Reuse the existing calcium/activity trace as a tracker of whether the target
neuron has been meaningfully active. Do not use it for default weight scaling.

Existing trace:

$$
ca_j \leftarrow ca_j + \alpha_{\mathrm{ca}}(spiked_j - ca_j)
$$

Confidence should decay very slowly for active neurons and faster only after a
properly long inactivity window.

Do not let confidence decay away just because the network is training other
patterns. A specialist may only fire once per full interleaved pattern cycle.

Add a long grace counter:

```text
inactive_steps_j increments when ca_j is below a dead threshold
inactive_steps_j resets or decreases when ca_j is above that threshold
```

Then:

$$
\rho_j =
\begin{cases}
\rho_{\mathrm{active}}, & ca_j \ge ca_{\mathrm{dead}} \\
\rho_{\mathrm{dead}}, & ca_j < ca_{\mathrm{dead}}
  \land \mathrm{inactive\_steps}_j \ge T_{\mathrm{grace}} \\
0, & \text{otherwise}
\end{cases}
$$

$$
C_i \leftarrow C_i(1-\rho_j)
$$

Timescale guidance:

- \(T_{\mathrm{grace}}\) should be many full training cycles, not a few pattern
  presentations.
- If a full 8-pattern sweep is roughly a few hundred steps, start with a grace
  window in the thousands to tens of thousands of steps.
- \(\rho_{\mathrm{active}}\) should be extremely small.
- \(\rho_{\mathrm{dead}}\) can be larger than active decay, but still slow.

Conceptually, this should behave like long-term memory:

- recent active specialists keep their confidence,
- inactive specialists forget only after genuinely long disuse,
- stale neurons eventually become reusable.

## Expected Effects

This should help with:

- protecting trained instant integrators from being rewritten by low-affinity
  accidental wins,
- reducing repeated near-winner collisions through loser depression,
- keeping unused or stale neurons plastic enough for continuous learning.

This will not by itself solve pure tyrant behavior if a tyrant always wins and
is never inhibited. If that remains, inspect whether chronic winners need a
separate local anti-tyranny mechanism, but do not re-enable default homeostatic
weight scaling without measuring the assignment tradeoff.

## Verification

Run:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_8line_consolidation.py
PYTHONPATH=. .venv/bin/python test_inhibitory_plasticity.py
PYTHONPATH=. .venv/bin/python test_refractory_gating.py
```

Also add a focused metric script or test that reports:

- distinct winners across 8 patterns,
- within-pattern winner dominance,
- old-pattern preservation after training new patterns,
- number of neurons that never fired,
- confidence distribution per L2E,
- loser-depression event counts.

## Guardrails

- Do not reintroduce winner facilitation.
- Do not use labels or pattern IDs.
- Do not add a procedural one-to-one assignment table.
- Do not add hard WTA resets.
- Do not make confidence an externally supervised score.
- Keep confidence and decay local to the target neuron and its synapses.

