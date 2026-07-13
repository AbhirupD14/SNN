# Dendritic Inhibitory Fan-Out Plan

This plan formalizes a proposed experiment for improving L2 competition dynamics
by replacing each single `L2I -> L2E` inhibitory gate with multiple local
dendritic inhibitory gate sites on the target L2E neuron.

The key constraint: this should use **negative flow-rate inhibition**, not an
instant summed discharge. The goal is better timing and sustained suppression,
not simply a larger one-shot inhibitory hit.

## Motivation

The current L2 competition problem is not just that inhibition is weak. The
single `L2I -> L2E` gate has to satisfy two conflicting requirements:

```text
1. suppress near-winners quickly enough to stop round-robin competition
2. avoid wiping out every rival so hard that the model collapses to one winner
```

Because the gate is capped below threshold, it preserves local competition but
often cannot stop loser charge carryover. Losers keep enough membrane charge to
fire shortly after inhibition, which lets them continue learning the same
pattern and push their feedforward weights toward cap.

A dendritic fan-out model gives each target L2E several small inhibitory sites.
Each site stays locally capped below threshold, but the combined time-distributed
effect can suppress competitors earlier and for longer without a single hard
reset.

## Core Hypothesis

Replacing one large-ish inhibitory gate with several small dendritic inhibitory
flow sites can:

- reduce loser charge carryover,
- prevent rival L2E neurons from firing immediately after L2I discharge,
- reduce repeated co-learning of the same pattern,
- slow down runaway feedforward weights in losing competitors,
- improve sustained ownership without recreating single-winner collapse.

The mechanism should remain local. It should not know pattern labels, winner
identity beyond the arriving inhibitory event, global rank, or rival-neuron
state.

## Proposed Model

Current conceptual model:

```text
L2I -> L2E_j : one negative gate
```

Proposed model:

```text
L2I -> L2E_j : K dendritic inhibitory gate sites

site 0: small negative gate, own cap, own inhibitory trace
site 1: small negative gate, own cap, own inhibitory trace
...
site K-1: small negative gate, own cap, own inhibitory trace
```

Each target L2E owns its own dendritic inhibitory sites. Sites do not directly
communicate across L2E neurons.

## Delivery Rule: Flow, Not Instant Sum

Do **not** implement the first version as:

```text
V <- max(V - sum(site_weights), rest)
```

That mostly retests the already-failed "raise the inhibitory cap" experiment.

Instead, each site should inject into an inhibitory current trace:

```text
I_inh_site_k += site_weight_k * injection_scale
```

Then, over subsequent timesteps:

```text
V <- max(V - sum_k I_inh_site_k, rest)
I_inh_site_k <- decay_k * I_inh_site_k
```

If using normalized injection:

```text
I_inh_site_k += site_weight_k * (1 - decay_k)
```

so the total future drain from that site is approximately `site_weight_k`.

The value of the fan-out is temporal distribution:

```text
several small local drains over time > one large instant subtraction
```

## Initial Parameterization

Start with a small fan-out:

```text
K = 3 or 4 sites per L2I -> L2E target
```

Suggested site caps:

```text
site_cap = 0.03θ to 0.08θ
total_possible_cap = K * site_cap
```

The total possible cap may exceed the old single-gate cap, but it should not be
delivered as an instantaneous hard reset. Its effect should be spread over time
through inhibitory traces.

Suggested trace decays:

```text
site 0: fast    decay ~= 0.3 to 0.5
site 1: medium  decay ~= 0.6 to 0.8
site 2: slow    decay ~= 0.85 to 0.95
```

This gives a fast immediate bite plus a lingering tail that can delay the next
loser spike.

If using `K = 4`, either add another intermediate decay or duplicate the medium
site with a different learning rate/cap.

## Why Site Diversity Matters

If all sites have the same cap, decay, learning rate, and update rule, they will
likely become identical copies of one gate. That adds magnitude but little new
dynamics.

The first implementation should include at least one form of local diversity:

```text
different trace decays per site
or different caps per site
or different learning rates per site
```

Recommended first diversity:

```text
different decay constants
```

Reason: the experiment is specifically about timing. Different decays directly
test whether fast + lingering inhibitory flow can suppress round-robin carryover.

## Local Learning Rule

Each site should learn only when a real L2I inhibitory event targets the L2E and
the target is not refractory.

Use local target state:

```text
v_pre = target L2E membrane charge before inhibitory flow injection
theta = target threshold
u_k   = site_weight_k / site_cap_k
```

A simple turnover rule:

```text
p_t = clamp(v_pre / theta, 0, p_max)
du_k = eta_up_k * p_t * (1 - u_k) - eta_down_k * u_k
u_k <- clamp(u_k + du_k, 0, 1)
site_weight_k <- u_k * site_cap_k
```

This mirrors the current differentiating inhibitory turnover rule, but per
dendritic site.

Interpretation:

- high-charge rivals strengthen their inhibitory sites,
- weak/dead targets drift down,
- no site grows beyond its local cap,
- no global information is used.

## Optional Site Competition

If all sites still saturate together, add a local site-competition term later.
Do not add it in the first implementation unless the no-competition version
clearly degenerates.

Possible local competition:

```text
total_u = sum_k u_k
if total_u > target_site_mass:
    u_k <- u_k * target_site_mass / total_u
```

or:

```text
large slow sites decay faster when fast sites already handled the event
```

This should remain local to the target neuron's dendritic inhibitory sites.

## Interaction With Existing Inhibitory Flow

The repo already has an `inhibitory_flow_rate` mechanism. This experiment should
reuse that concept but extend it from one target-level trace to multiple
site-level traces.

Important distinction:

```text
current inhibitory flow:
  one inhibitory trace per affected neuron/gate path

dendritic fan-out:
  multiple inhibitory traces per L2I -> L2E target, one per local site
```

The implementation should avoid mixing the two concepts ambiguously. Either:

```text
1. make dendritic fan-out require inhibitory_flow_rate=True
```

or:

```text
2. create a separate flag whose delivery path is explicitly flow-based
```

Recommended flags:

```text
dendritic_inhibition = False
dendritic_inh_sites = 3
dendritic_inh_trace_decays = [0.4, 0.75, 0.9]
dendritic_inh_site_cap_frac = 0.05
dendritic_inh_normalized = True
```

Keep defaults off until measured.

## Implementation Sketch

### 1. Data Model

For each L2E target, replace or augment the single negative L2I gate with:

```text
site_weights[target_j, site_k]
site_traces[target_j, site_k]
site_trace_decays[site_k]
site_caps[site_k]
```

The simplest placement may be inside `Neuron` for the L2E target, since the
negative gate currently lands on the target neuron. If that becomes awkward,
represent the L2I->L2E projection explicitly in `SimulationEngine`, but avoid a
large architecture rewrite for the first experiment.

### 2. Delivery

On L2I fire, for every non-winning L2E target:

```text
for each dendritic site k:
    inject inhibitory trace from site_weight[k]
```

Do not subtract the sum immediately.

During neuron update or trace advancement:

```text
drain = sum(site_traces)
V = max(V - drain, rest)
site_traces *= site_decays
```

Make sure the drain can affect an L2E even if no new L1 input arrives on that
exact timestep. That is the point of flow-rate inhibition.

### 3. Learning

On real L2I->L2E inhibitory event:

```text
v_pre = target potential before trace injection/drain
for each site:
    update site_weight using local turnover rule
```

Do not update dendritic inhibitory sites from L2E excitatory fire logic.

### 4. Winner Exclusion

Preserve current winner behavior:

```text
the winning L2E is not inhibited by L2I in the same competition event
all non-winning L2E targets receive the dendritic inhibitory event
```

Do not add explicit per-pattern or per-rival targeting.

## Expected Consequences

Potential benefits:

- rivals carry less charge after losing,
- fewer immediate re-fires after L2I discharge,
- less repeated positive learning by losing competitors,
- slower march of losing feedforward weights toward cap,
- better single-pattern consolidation at lower `eta_loss`,
- possibly better interleaved ownership if over-depression can be reduced.

Risks:

- too much sustained inhibition can kill recruitment and create dead neurons,
- if all sites saturate together, this is just a larger gate in disguise,
- slow traces may suppress a neuron after a pattern switch and hurt responsiveness,
- more parameters can make diagnosis harder,
- excessive inhibition can recreate single-winner collapse.

## Measurements

Compare at least:

```text
baseline current dashboard/default config
inhibitory_flow_rate=True without dendritic fan-out
dendritic_inhibition=True, K=3
dendritic_inhibition=True, K=4
```

For each condition, report:

```text
single held-pattern dominance
interleaved all-8 distinct owners
sustained-presentation dominance
dead/ownerless patterns
L2E firers per held pattern
immediate re-fire rate after L2I discharge
mean/max loser charge after inhibition
time from L2I discharge to next rival L2E fire
feedforward weight saturation rate in losers
site weight distribution by decay class
site trace magnitudes over time
L2I spike count and L2I->L2E discharge count
```

Use at least 3 seeds.

## Success Criteria

Minimum useful result:

```text
fewer immediate loser re-fires
lower loser carryover charge
no increase in dead neurons
no collapse to one/few global winners
```

Strong result:

```text
single held pattern consolidates with lower eta_loss than 10
interleaved training improves distinct owners beyond current 4-6/8 failure mode
sustained dominance improves without sacrificing distinctness
previously losing competitors stop pushing all active feedforward weights to cap
```

## Failure Criteria

Archive or redesign if:

```text
all sites saturate identically
the model behaves like simply raising the old inhibitory gate cap
L2E neurons go dead before learning useful receptive fields
pattern-switch responsiveness gets worse from lingering inhibition
distinct ownership decreases even if firing rate looks cleaner
```

## Tests

Add focused tests before broad sweeps:

```text
1. flag off preserves current inhibitory behavior
2. dendritic sites inject traces rather than instantly subtracting summed weight
3. traces drain charge over later timesteps
4. different site decays produce different temporal profiles
5. site weights stay within their local caps
6. winner is not inhibited; non-winners are
7. refractory targets do not learn from inhibitory events
8. signed-spike L2E feedforward learning remains unchanged
```

Run relevant existing scripts:

```bash
PYTHONPATH=. .venv/bin/python test_inhibitory_delta_rule.py
PYTHONPATH=. .venv/bin/python test_flow_rate.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
```

Add a new focused script if needed:

```bash
PYTHONPATH=. .venv/bin/python test_dendritic_inhibition.py
```

## Dashboard / Visualization Needs

The dashboard should expose this only as an advanced experiment initially.
Useful diagnostics:

```text
dendritic site weights per L2E target
dendritic inhibitory traces over time
total inhibitory drain per L2E
post-inhibition charge dip and delayed recovery
time-to-next-rival-fire after L2I discharge
```

Do not crowd the main config panel. Add the core toggle and perhaps `K`; keep
decays/caps in advanced config or a preset.

## Open Design Questions

1. Should sites use fixed heterogeneous decays or learned/adaptive decays?
2. Should site caps be equal or distributed across fast/medium/slow sites?
3. Should slow sites receive a lower learning rate to avoid pattern-switch drag?
4. Should fan-out apply only to L2I->L2E or eventually also L1I->L1E?
5. Should loser depression read the pre-flow `v_pre`, post-flow charge, or both?

First pass recommendation:

```text
K = 3
fixed decays = [0.4, 0.75, 0.9]
equal site caps = 0.05 * theta_l2
normalized trace injection = True
current turnover learning rule copied per site
no site competition yet
```

