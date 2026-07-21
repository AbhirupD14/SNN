# Coincidence Pyramidal Cell Technical Specification

## Status

This document is the implementation specification for the next coincidence-detector
topology. It supersedes the historical `L1E_new` accumulating detector for this new
topology. It does not replace the existing `pi`, `old`, `rg`, or `rg_residual`
presets.

The scientific behavior in this document is normative. An implementation must not
weaken the coincidence rule into ordinary summation, infer branch activity from a
global winner variable, select an L2 winner with an engine-level `max(V)` operation,
or allow excitation to propagate across multiple edges within one outer boundary.

### Phase-alignment amendment

The implemented phase-alignment rule supersedes later historical statements in this
document that give `apical_excitation` a one-boundary delay or represent a committed C
deposit as full-boundary constant drive:

- ordinary feedforward, pretrained, and basal projections remain delay-1;
- an actual L2E spike delivers unweighted apical permission to its C targets at the
  same `(t, tau)`;
- when basal availability and apical permission first overlap, the C cell atomically
  consumes basal eligibility and applies `q = w_basal * s` as an instantaneous somatic
  charge impulse, `V(tau+) = V(tau-) + q` (`C = 1`);
- a suprathreshold C cell then fires through the ordinary event scheduler at that exact
  `tau`, causally after the already-processed L2E spike;
- a dedicated per-boundary commit ledger permits at most one deposit while raw and
  duplicate apical deliveries remain observable.

This is the sole excitatory same-boundary append: it does not collapse the delay on
`RG -> L1E`, `L1E -> L2E`, or `L1E -> L1C` basal projections. Where the historical
timelines, algorithms, or test lists below conflict with this amendment, this amendment
is authoritative.

## Objective

Add an excitatory coincidence pyramidal cell, `L1C`, with separate basal and apical
input semantics:

- each `L1C[i]` has one learned basal afferent from its paired `L1E[i]`;
- each `L1C[i]` has eight unweighted apical afferents, one from each `L2E[j]`;
- an apical event is a Boolean permission signal and contributes no numeric charge;
- basal activity contributes charge only when the apical gate is active in the
  coincidence window;
- basal-only and apical-only event trains must add exactly zero membrane charge;
- `L1C` uses the same threshold, leak, reset potential, refractory period,
  conductance equation, and local activity-trace equation as ordinary excitatory
  neurons;
- only the basal weight learns, and it learns only when `L1C` fires;
- every inhibitory cell in this preset is a pretrained zero-latency relay within the
  current outer boundary;
- one relay event emits one immediate hard reset at the relay's sub-boundary event
  time. Inhibitory bursting is not part of this version;
- `RG -> L1E` is fixed and pretrained: one unobstructed RG spike is sufficient to
  make its paired `L1E` fire on the following simulation boundary.

The immediate experimental goal is stable circuit-level frequency halving. At the
initial C weight, retained subthreshold impulses produce the isolated two-coincidence
cadence. A mature C cell is deliberately one-shot capable and fires at its permitting
L2E `tau`; its paired L1 reset suppresses enough otherwise-valid `L1E` events for the
mature held-pattern output window to reach the `L1E/RG = 0.5` target.

## Terminology and time model

The simulator becomes a hybrid event-resolved synchronous model. Ordinary synaptic
arrivals remain quantized to integer outer boundaries, while membrane evolution and
zero-latency apical/relay events are resolved at continuous sub-boundary times.

An **outer boundary** is one call to `SimulationEngine.step()` and represents the
interval `[t, t + 1]`. All excitation scheduled from an earlier boundary is frozen at
the start of this interval and treated as constant drive over the interval until it
is consumed by a spike or discarded by a hard reset.

A **sub-boundary time**, `tau in [0, 1]`, is the analytically calculated time within
the current outer boundary at which a membrane reaches threshold. The engine advances
all membrane-bearing cells together from one event time to the next.

Ordinary excitatory, basal, and pretrained projections retain a one-outer-boundary
delay. A spike emitted during outer boundary `t` is delivered through one such edge at
the start of boundary `t + 1`, regardless of its `tau`. Apical permission from L2E to C
is the explicit exception described above; it arrives and may commit the gated C
charge impulse at the L2E spike's current `tau`.

`relay_excitation` into a stateless inhibitory relay and that relay's
`hard_reset_inhibition` outputs are also zero latency. They resolve at the source
spike's current `(t, tau)`.

An **event** is a spike arrival on one directed edge. A **valid coincidence** is a
basal event whose one-boundary eligibility overlaps a current apical event.

The basal eligibility window is dendritic state, not membrane charge. Creating basal
eligibility must not depolarize the C-cell soma.

## New preset

Add the built-in preset name:

```text
rg_coincidence
```

The topology is:

```text
RG[i] --pretrained fixed excitation, paired--> L1E[i]

L1E[i] --learned feedforward, dense---------> L2E[j]
L1E[i] --learned basal excitation, paired---> L1C[i]

L2E[j] --unweighted apical excitation, dense--> L1C[i]

L1C[i] --relay excitation, paired-----------> L1I[i]
L1I[i] --one immediate hard reset, paired----> L1E[i]

L2E[j] --relay excitation-------------------> L2I
L2I    --one immediate hard reset------------> every L2E[j]
```

As in the existing engine, the learned `L1E -> L2E` feedforward weights are owned by
the postsynaptic `L2E` cells. `L1E` owns no learned weight in this preset.

For the current `N_PIX = 9` and `N_OUT = 8`, the preset contains:

| Population | Count | Archetype | Behavior |
| --- | ---: | --- | --- |
| `RG` | 9 | `rg_source` | Exogenous input-driven spike source |
| `L1E` | 9 | `e_pretrained` | Fixed-input, noncompetitive excitatory relay |
| `L1C` | 9 | `e_coincidence` | Basal/apical gated excitatory accumulator |
| `L1I` | 9 | `i_relay` | Immediate one-spike reset relay |
| `L2E` | 8 | `e_latency_competitor` | Learned E cell participating in inhibitory latency WTA |
| `L2I` | 1 | `i_relay` | Immediate one-spike reset relay |

Total nodes: `45`.

The preset contains these directed edges:

| Projection | Count | Edge kind |
| --- | ---: | --- |
| `RG -> L1E` | 9 | `pretrained_excitation` |
| `L1E -> L2E` | 72 | `feedforward` |
| `L1E -> L1C` | 9 | `basal_excitation` |
| `L2E -> L1C` | 72 | `apical_excitation` |
| `L1C -> L1I` | 9 | `relay_excitation` |
| `L1I -> L1E` | 9 | `hard_reset_inhibition` |
| `L2E -> L2I` | 8 | `relay_excitation` |
| `L2I -> L2E` | 8 | `hard_reset_inhibition` |

Total edges: `196`.

There is no deterministic L2 winner-selection phase in this preset. All L2E cells
follow the same membrane and firing rule. The first L2E to reach threshold within an
outer boundary fires, immediately activates L2I, and L2I immediately hard-resets all
L2E cells. Later prospective crossings are thereby cancelled. WTA emerges from
learned drive, retained membrane state, leak/conductance, spike latency, and the
explicit `L2E -> L2I -> L2E` topology.

## Equations

### Symbols

For coincidence cell `i` on boundary `t`:

- `theta` is the shared excitatory threshold;
- `w_i` is the single learned basal weight;
- `w_max` is the C-cell basal weight cap;
- `s_i(t)` is the active basal signal in `(0, 1]`; it is `1` for the current binary
  topology;
- `a_ij(t)` is the unweighted apical event from `L2E[j]`, in `{0, 1}`;
- `A_i(t) = max_j a_ij(t)` is the Boolean apical gate;
- `e_i(t)` is the basal eligibility available from the immediately preceding
  boundary, in `{0, 1}`;
- `b_i(t)` indicates a current basal event, in `{0, 1}`;
- `phi_i` is the basal synapse's normalized inverse-square distance influence;
- `eta_C` is the C-cell basal learning rate.

There are no apical weights.

### Gated additive charge

The effective basal availability is:

\[
B_i(t)=\max(b_i(t),e_i(t)).
\]

The coincidence gate is:

\[
G_i(t)=B_i(t)A_i(t).
\]

The somatic charge deposited on boundary `t` is:

\[
Q_{C_i}(t)=G_i(t)w_i s_i(t).
\]

Because there is one basal afferent, this is a gated additive sum with one term. A
future C archetype may support multiple basal afferents by summing active basal
weights, but that generalization is not part of this preset.

The required truth table is:

| Current/prior eligible basal | Current apical | Charge deposited |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | `w_i * s_i` |

Apical activity never adds a constant, a weight, or a fraction of threshold. Repeated
basal-only or apical-only activity therefore cannot charge the cell, irrespective of
leak or presentation duration.

### Event-resolved membrane integration

After dendritic gating, C and ordinary E cells use the same conductance-LIF trajectory.
The current boundary's frozen charge packet `Q` is interpreted as constant drive
`I_exc = Q` over the unit outer interval until that packet is consumed or discarded.
With `C = 1`, `E_L = V_rest = 0`, and inhibitory reversal `E_inh`:

\[
g=g_L+g_{inh},
\]

\[
V_\infty=\frac{g_LE_L+g_{inh}E_{inh}+I_{exc}}{g}
\]

for `g > 0`. Starting from voltage `V_0` at sub-boundary time `tau_0`, the exact
voltage after an elapsed segment `Delta tau` is:

\[
V(\tau_0+\Delta\tau)=
V_\infty+(V_0-V_\infty)e^{-g\Delta\tau}.
\]

For `g = 0`:

\[
V(\tau_0+\Delta\tau)=V_0+I_{exc}\Delta\tau.
\]

The baseline leak mapping remains:

\[
g_L=-\ln(1-leak\_rate).
\]

The engine must use these exact segment equations rather than approximate the boundary
with a configurable number of excitation chunks. Fixed micro-chunks would make winner
identity depend on numerical resolution.

`g_inh` and the baseline parameters are held constant during one outer boundary and
legacy conductance decay remains a once-per-boundary finalization step. Integer
`refractory_timer > 0` suppresses firing candidates for the entire outer boundary
while membrane evolution continues; the timer advances once at finalization. This
preserves the existing refractory convention.

### Threshold-crossing time

For a membrane that is eligible to fire and begins a segment below threshold, the
candidate crossing time is calculated analytically.

For `g > 0`, a finite crossing exists only when the trajectory reaches `theta` before
the outer boundary ends. When `V_inf > theta`:

\[
\Delta\tau_{cross}=
\frac{1}{g}
\ln\left(
\frac{V_\infty-V_0}{V_\infty-\theta}
\right).
\]

For `g = 0` and `I_exc > 0`:

\[
\Delta\tau_{cross}=\frac{\theta-V_0}{I_{exc}}.
\]

A candidate is valid only if:

```text
0 <= Delta_tau_cross <= remaining_outer_interval
refractory permits firing
the neuron has not already fired this outer boundary
for C only: coincidence_active is true
```

Otherwise its candidate time is infinity. If a membrane already begins an eligible
segment at or above threshold, its candidate time is the current event time.

The scheduler selects the smallest finite absolute candidate time. This minimum is a
numerical event-ordering operation, not a WTA policy: every neuron independently
produces its candidate from the same local membrane equation.

If no finite candidate remains, advance every membrane through the rest of the outer
boundary with the exact segment equation. If no valid C coincidence occurs, `Q_C = 0`;
existing C potential may leak, but no new charge is added.

### Firing

A C cell may produce a finite threshold-crossing event only when all conditions hold
within the same outer boundary:

```text
refractory_timer == 0
V >= theta
G_i(t) == 1
fired_this_boundary == false
```

The explicit gate requirement guarantees that a C cell cannot cross during refractory,
retain a suprathreshold voltage, and later fire and learn on a boundary with no causal
basal/apical event.

On firing, the C cell uses the ordinary E reset and refractory behavior:

```text
v_pre = V
V = V_rest
refractory_timer = refractory_steps
spiked = true
spike_tau = current_tau
remaining_excitation = 0
fired_this_boundary = true
```

Every ordinary E and C cell emits at most one spike per outer boundary. The current
frozen drive packet is consumed by firing, so a zero-refractory cell cannot repeatedly
fire from the same arrival packet. The basal weight used to create C drive is the
pre-update weight. Learning affects future outer boundaries only.

### Basal learning

Only the basal weight changes, and only after its C cell actually fires.

The fullness error contains only the single learned basal weight:

\[
FE_i=\theta-w_i.
\]

The update is:

\[
\Delta w_i=
\eta_C
(\theta-w_i)
A_i(t)
\left(1-\left(\frac{w_i}{w_{max}}\right)^2\right)
s_i(t)
\phi_i.
\]

Then:

\[
w_i\leftarrow clip(w_i+\Delta w_i,0,w_{max}).
\]

This is the agreed specialization of:

```text
dw = LR * FE * apical * (1 - (w/w_max)^2) * s_i * influence
```

For the present binary topology, firing implies `A_i = 1` and `s_i = 1`. There is no
negative `s_i` update. Inactive basal input cannot cause C firing, so it cannot produce
a learning event. The implementation must not synthesize `-1` participation for this
rule and must not depress the basal weight on nonparticipating boundaries.

### Distance influence

Use the repository's bounded, normalized inverse-square factor:

\[
\phi_i=\left(\frac{d_{ref}}{\max(d_i,d_{ref})}\right)^2.
\]

`d_i` is the functional distance of `L1E[i] -> L1C[i]`. `d_ref` is the smallest
positive basal-edge distance across the `e_coincidence` target population. Geometry
changes learning rate only; it never changes delivered charge.

### Basal weight scale for the two-coincidence contract

At nonzero leak, `theta/2` is not quite enough for two consecutive full-boundary drive
packets under the conductance trajectory. From rest with no inhibition, define:

\[
r=e^{-g_L}=1-leak\_rate
\]

and the one-boundary current-to-voltage factor:

\[
\kappa=
\begin{cases}
1,&g_L=0\\
\frac{1-e^{-g_L}}{g_L},&g_L>0.
\end{cases}
\]

For valid coincidences separated by `T` boundaries, the minimum weight that crosses
on the second deposit from reset is:

\[
w_2(T)=\frac{\theta}{\kappa(1+r^T)}.
\]

The minimum weight that fires on one deposit is:

\[
w_1=\frac{\theta}{\kappa}.
\]

The isolated frequency-halving regime is therefore:

\[
w_2(T)\le w_i < w_1.
\]

For the initial `T = 1` experiment, use derived defaults:

```text
c_basal_weight_init = 1.01 * w_2(1)
c_basal_weight_max  = 1.10 * w_2(1)
```

The implementation must verify that `c_basal_weight_max < w_1`; reject a resolved
configuration that violates this invariant. At the current default leak of `0.03`,
these values are slightly above `theta/2`, which is intentional. C therefore needs a
separate basal cap rather than silently reusing `e_weight_cap`.

As a non-normative numeric sanity check at `theta = 1000`, `leak_rate = 0.03`, and the
derived defaults:

```text
pretrained L1E crossing from reset       tau ~= 0.952
C second-coincidence crossing at init    tau ~= 0.980
C second-coincidence crossing at cap     tau ~= 0.813
```

Thus an immature C initially fires too late to cancel an otherwise-valid same-boundary
L1E spike but can still learn. Basal maturation advances C latency until its immediate
L1I reset can beat L1E. The implementation must obtain this ordering from the membrane
equations; it must not encode a training phase or switch the ordering explicitly.

If the measured mature coincidence cadence is not `T = 1`, that later experiment may
derive the two-event regime from the measured `T`. Do not change the leak shared with
E merely to force the cadence.

### Pretrained RG-to-L1E excitation

`RG -> L1E` has no learned weight. It emits a fixed drive packet selected so one event
from rest has a finite threshold-crossing time within its delivery boundary when it is
not reset or inhibited.

For the baseline leak-only case:

\[
Q_{pretrained,min}=\frac{\theta}{\kappa}.
\]

Use:

```text
Q_pretrained = pretrained_exc_margin * Q_pretrained_min
pretrained_exc_margin = 1.05
```

This projection is fixed, is not clipped by `e_weight_cap`, and never appears in an
accumulating-weight update. “Instant” means that one RG spike is sufficient to produce
an L1E spike during the next outer boundary; it does not mean `tau = 0`, and the
ordinary one-boundary excitatory delay remains.

## Basal eligibility state machine

Each C cell owns these transient fields:

```text
basal_received          bool    # basal event delivered this boundary
basal_signal            float   # 1.0 in the binary topology
basal_eligible          bool    # unconsumed basal event from t-1
basal_eligible_signal   float   # carried signal; 1.0 in the binary topology
apical_sources          set     # L2E source ids delivered this boundary
apical_active           bool
coincidence_active      bool
coincidence_charge      float
```

Apical state is current-boundary only and never persists.

The basal window is directional: basal may precede apical by one boundary. Apical may
not precede basal. This matches the causal path in which `L1E` supplies the evidence
that later makes `L2E` return feedback.

At each boundary:

1. Preserve `basal_eligible` carried from `t - 1` long enough to evaluate this
   boundary.
2. Clear current-boundary basal/apical receipt fields.
3. Gather current basal and apical edge events.
4. Set `B = basal_received OR basal_eligible`.
5. Set `A = bool(apical_sources)`.
6. If `B AND A`, deposit the current pre-update basal weight exactly once and consume
   all current and carried basal availability. A current basal event is not carried
   forward after participating in this gate.
7. If current basal was received but not consumed, carry it as `basal_eligible` for
   exactly the next boundary.
8. Otherwise clear eligibility. A basal event may not survive for two future
   boundaries.
9. Clear `apical_sources` at the next boundary start.

Additional invariants:

- simultaneous basal and apical events are valid;
- basal at `t`, apical at `t + 1` is valid;
- basal at `t`, apical at `t + 2` is invalid;
- apical at `t`, basal at `t + 1` is invalid unless a new apical event also arrives at
  `t + 1`;
- multiple apical sources on one boundary still open only one Boolean gate;
- current and carried basal activity together still produce only one deposit;
- a valid coincidence consumes the carried basal event even when the C cell is
  refractory. Charge may accumulate under the existing E refractory semantics, but
  the same basal event may not be reused;
- the existing scalar `stimulate()` operation must reject an `e_coincidence` target
  with a clear error. Controlled C tests must deliver explicit basal/apical events;
  there is no diagnostic shortcut that bypasses the gate.

## Class design

### `ConductanceLIFNeuron`

Extract or otherwise provide a shared membrane abstraction containing only intrinsic
excitatory dynamics:

- `V`, `v_rest`, `threshold`, `v_pre`, and `v_pre_reset`;
- leak and `g_L`;
- `pending_exc` for boundary-start gathering and `remaining_excitation` for the frozen
  packet used by the event loop;
- persistent inhibitory conductance state;
- refractory state;
- local activity trace;
- current outer-boundary drive and `fired_this_boundary`;
- `gather_exc` and `add_inhibition`;
- `advance_segment(delta_tau)` using the exact trajectory;
- `crossing_time(remaining_interval)`;
- somatic `fire(tau)`, trace update, conductance decay, and refractory advance;
- `hard_reset(tau, discard_drive=True)`.

Because one outer boundary can now contain several segment advances and an inhibitory
reset, `v_pre_reset` is the maximum depolarized segment endpoint reached during that
boundary before any spike/reset, not merely the final post-reset voltage. Update it on
every segment advance and never clear it during hard reset. The existing once-per-
boundary activity-trace equation then continues to register real depolarization that
was subsequently wiped.

This base must not own an accumulating afferent vector or a learning rule.

Refactoring ordinary `ExcitatoryNeuron` onto this base must preserve all existing
numeric behavior and golden tests.

### `ExcitatoryNeuron`

Retain the current ordinary E behavior:

- flat `acc_weights` and distance factors;
- unconditionally summed delivered excitatory charge;
- the signed accumulating-weight rule (see the production-rule note below);
- current source, encoder, residual, and competitor uses.

> **Production E rule (updated).** The ordinary/latency E accumulating rule is now
> **linear-bounded** by default: `dw_i = eta·(θ−Σw)·s_i·influence_i`, clipped to
> `[0, w_max]`. The historical `(1 − (w_i/w_max)²)` multiplier was **removed from the
> default E update** after a 32/32 fresh-seed confirmation (row1→col1→row1, seeds
> 2001–2032; all five success gates), because it converges faster with no loss of
> turnover/recovery/determinism. **The E hard cap is retained** as a safety bound — under
> the linear rule an individual weight can reach the cap (observed on seed 2005). The
> historical quadratic rule stays available as the headless `quadratic_bounded` mode. The
> **C basal rule is intentionally NOT changed** (it keeps its `(1 − (w_b/w_C_max)²)` term
> and its temporal cap). See `docs/LINEAR_WEIGHT_ABLATION_REPORT.md`.

`e_pretrained` uses this class with empty flat afferent arrays and `learn=False`.
Fixed `pretrained_excitation` events enter through `gather_exc()` and are not stored
in `acc_weights`. The cell is registered as a distinct noncompetitive population and
never learns its RG input.

`e_latency_competitor` also uses this class, with the same flat feedforward weight
bank and update equation as legacy `e_competitor`. The difference is execution policy:
it participates in analytic event scheduling and never enters the legacy deterministic
WTA list. On its own spike it learns from its own boundary-delivered causal volley,
then immediately drives its relay edges.

### `DendriticCompartment`

Add a small reusable input-compartment abstraction. It is not a general dendritic
tree. It should own:

- a compartment name (`basal` or `apical`);
- ordered source IDs and edge IDs;
- current delivered source IDs/signals;
- aligned weights and distance factors only when the compartment is plastic;
- transient clear/gather helpers.

For the first C cell:

- basal has exactly one source, one weight, and one distance factor;
- apical has one or more sources and no weight vector;
- the C class, not the compartment, owns the cross-compartment gating rule.

### `CoincidencePyramidalNeuron`

Implement `CoincidencePyramidalNeuron` as a sibling of ordinary
`ExcitatoryNeuron` over the shared membrane base. Required behavior:

- `type = 'E'` and role `coincidence`;
- one basal and one apical `DendriticCompartment`;
- one-boundary basal eligibility;
- `gather_basal(source, signal=1.0)`;
- `gather_apical(source)`;
- `resolve_dendrites()` implementing the truth table and state machine;
- `can_fire()` requiring refractory clear, threshold crossing, and a current valid
  coincidence;
- C-local `crossing_time()` returning infinity when the coincidence gate is closed;
- ordinary E fire/reset/refractory/trace dynamics;
- `update_basal_weight()` implementing the exact equation above;
- no `update_acc_weights()` fallback;
- no apical weight storage and no apical learning method.

The engine must call `update_basal_weight()` immediately after a C spike. The method
must use the causal dendritic state from the firing boundary and the pre-update basal
weight.

### `InhibitoryNeuron`

Keep `InhibitoryNeuron` stateless and event-driven:

- any relay input at `(t, tau)` makes it fire at that same `(t, tau)`;
- it emits at most one spike per outer boundary;
- it owns no membrane, learned weight, or refractory state;
- a second same-boundary input does not create a burst or a second reset event for the
  same outgoing edge.

The output effect belongs to the outgoing edge kind. In this preset,
`hard_reset_inhibition` applies one hard reset per outgoing target immediately at the
relay's current `tau`.

### `BoundaryEventScheduler`

Implement the sub-boundary race as an engine-owned scheduler/helper rather than neuron
logic. It owns no scientific state beyond the current outer boundary and `tau`. Its
responsibilities are:

- request local crossing candidates from every event-resolved membrane;
- choose the earliest finite candidate with the fixed tie tolerance;
- advance all membranes by the same elapsed segment;
- invoke one causal spike transition;
- dispatch zero-latency relays and immediate reset edges;
- invalidate/recompute candidates after every spike/reset;
- advance all membranes through the remaining interval when no candidate remains;
- collect ordered spike, relay, reset, and tie diagnostics.

The scheduler must not inspect learned weight vectors to choose a winner, read a
global input pattern, or rank end-of-boundary voltages. Local cells calculate times;
the scheduler orders those times.

## Hard-reset inhibition

Hard reset is an immediate sub-boundary state transition, not a next-boundary queue.
Each diagnostic event contains at least:

```text
source
target
edge_id
kind = "hard_reset"
outer_boundary
tau
v_before
drive_before
```

A hard reset emitted at `(t, tau)` is applied after every membrane has been advanced
to `tau` and after the causal excitatory spike has been recorded, but before the event
scheduler searches for another threshold crossing.

For an excitatory target, hard reset performs:

```text
V = V_rest
remaining_excitation = 0
```

It therefore removes retained membrane evidence and discards the remainder of the
current outer boundary's frozen drive packet. A target whose former predicted crossing
time was later than `tau` can no longer fire from that packet. Hard reset must not:

- change learned weights;
- clear the local activity trace;
- clear persistent inhibitory conductance;
- add or alter refractory state;
- erase `spiked`, `spike_tau`, `v_pre`, or `v_pre_reset` from a target that already
  fired earlier at the same `tau`;
- erase excitation already scheduled for a future outer boundary;
- modify the external RG source;
- count as a postsynaptic spike.

If multiple hard resets target the same neuron at one event time, apply one idempotent
state reset but retain every causal reset event in diagnostics. A spike already emitted
at an earlier `tau` is not retroactively cancelled. Resetting the winning L2E after its
spike is valid: its emitted spike remains, while its membrane and unused drive are
cleared.

Existing `inhibition` and `predictive_inhibition` conductance edges remain unchanged
for existing presets. `hard_reset_inhibition` is a separate edge kind so this
experiment does not silently alter legacy scientific behavior.

## Emergent latency WTA

`rg_coincidence` must not call the existing deterministic competitor arbitration.
Within one outer boundary:

1. Every latency competitor independently calculates its next threshold-crossing
   time from its local state and frozen drive.
2. The scheduler advances all membranes to the earliest finite crossing.
3. That L2E fires and learns from its own causal delivered volley.
4. Its `relay_excitation` drives L2I at the same `tau`.
5. L2I fires at the same `tau` and hard-resets every L2E, including the winner.
6. All later L2 crossing candidates are invalidated because their voltage and
   remaining drive were cleared.

Winner identity is therefore not selected by comparing end-of-boundary voltages. It
emerges from threshold latency. The all-to-all L2I reset topology is intentionally
designed to produce WTA; which L2E wins is determined by learned weights and local
dynamics.

Exact simultaneous crossings remain possible in a perfectly symmetric deterministic
network. Existing seeded weight and geometry jitter should break ordinary ties. If
candidate times are equal within a fixed numeric tolerance, use stable node order as
the final reproducibility fallback, allow that one event to fire, and let immediate
L2I reset the other tied candidates before they emit. Record a `latency_tie` diagnostic
with the tied IDs and time. Do not use node order when crossing times are merely close
but distinguishable.

## NetworkSpec changes

Add node archetypes:

```text
e_pretrained
e_coincidence
e_latency_competitor
```

Suggested metadata:

| Archetype | Class | Role | Plastic flat FF | Engine WTA | Event resolved | Input sink |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `e_pretrained` | `E` | `pretrained` | false | false | true | false |
| `e_coincidence` | `E` | `coincidence` | false | false | true | false |
| `e_latency_competitor` | `E` | `competitor` | true | false | true | false |

The legacy `e_competitor` archetype retains its current engine-arbitrated WTA meaning
for existing presets. The new archetype shares its flat feedforward weights and
learning equation but never joins deterministic arbitration.

Add edge kinds:

```text
pretrained_excitation
basal_excitation
apical_excitation
hard_reset_inhibition
```

Validation requirements:

- `pretrained_excitation`: source class `S`, target `e_pretrained`, fixed, positive;
- `basal_excitation`: source class `E`, target `e_coincidence`, plastic, positive;
- `apical_excitation`: source class `E`, target `e_coincidence`, structural/unweighted,
  positive;
- `hard_reset_inhibition`: source `i_relay`, target class `E`, fixed, negative display
  sign;
- every `e_coincidence` in an applied graph has exactly one incoming basal edge;
- every `e_coincidence` has at least one incoming apical edge;
- duplicate basal/apical edges from the same source to the same target are rejected;
- bidirectional basal, apical, pretrained, relay, and reset edges are rejected;
- no edge may target an RG source, preserving the existing invariant.
- a spec using `hard_reset_inhibition` must mark every possible excitatory target as
  event-resolved; the new three archetypes satisfy this requirement;
- `rg_coincidence` L2 nodes must use `e_latency_competitor`, never legacy
  `e_competitor`;
- a custom spec may not mix legacy `e_competitor` engine arbitration with any
  event-resolved archetype or `hard_reset_inhibition`. Such a graph has two conflicting
  event-ordering semantics and must be rejected with a clear validation error.

Both basal and apical kinds can match an `E -> e_coincidence` gesture. The topology
editor must not silently select the first valid kind. When more than one edge kind is
valid, it must request an explicit kind or compartment selection. Preset construction
already supplies the kind explicitly.

## Engine build changes

During `_build_from_spec`:

1. Build ordinary flat feedforward banks exactly as today.
2. Build a separate basal input map for each `e_coincidence` target, including its
   edge ID, source ID, initialized weight, and distance factor.
3. Build an apical source/edge list for each `e_coincidence` target with no weights.
4. Construct `CoincidencePyramidalNeuron` instances and register them in:

   ```text
   self.coincidence
   self.exc
   self.neurons
   ```

5. Construct `e_latency_competitor` with the ordinary flat feedforward weight bank and
   learning rule, register it in `self.latency_competitors`, and do not register it in
   the legacy deterministic `self.competitors` arbitration list.
6. Do not register C cells in the ordinary `self.plastic` flat-feedforward list.
7. Add a basal-weight reference map for serialization and manual editing:

   ```text
   self._basal_weight_ref[edge_id] = (c_cell, basal_index)
   ```

8. Build explicit dispatch adjacency for basal, apical, pretrained excitation,
   zero-latency relays, and hard reset.
9. Select `event_resolved` execution from archetype/edge metadata, not from node IDs or
   the preset name. Existing presets containing none of the new event-resolved
   archetypes or reset edges stay on the byte-compatible legacy synchronous path.

Place `L1C[i]` at a copy of the already-generated historical `L1Enew[i]` functional
position (renamed for this preset). This introduces no new RNG draw. Adding the new
preset must not change existing presets' initialized weights or golden snapshots.

## Boundary execution order

The new behavior fits inside the synchronous outer engine with an event loop.

### Outer-boundary setup

1. Increment the outer boundary and set `current_tau = 0`.
2. Clear current spike, relay, tie, reset, and transient reporting state.
3. Rotate only the delay-one excitatory/conductance and causal-delivery buffers. There
   is no next-boundary hard-reset buffer.
4. Deliver legacy conductance arrivals where present.
5. Gather ordinary internal excitatory arrivals as frozen drive packets.
6. Deliver basal and apical events into their named C compartments.
7. Deliver fixed pretrained-excitation arrivals and present current external input to
   RG sources.
8. Resolve every C dendritic gate and install only valid gated somatic drive.
9. Freeze each membrane's gathered `pending_exc` into `remaining_excitation` and clear
   `pending_exc`.
10. Mark every membrane-bearing cell `fired_this_boundary = false`.

### Sub-boundary event loop

Repeat until `current_tau == 1` or no finite crossing remains:

1. Ask every event-resolved membrane that may fire for its next absolute crossing
   time in `[current_tau, 1]`.
2. Select the smallest finite time. Resolve only exact/numeric-tolerance ties by the
   documented stable fallback and record them.
3. Advance **every** membrane-bearing cell from `current_tau` to that event time with
   the exact segment trajectory, then set `current_tau` to the event time.
4. Fire the selected E or C cell and consume its remaining drive packet.
5. If it is C, update its basal weight immediately from the causal gate state.
6. If it is a latency competitor, update its ordinary feedforward weights immediately
   from its own delivered causal volley.
7. Record its ordinary excitatory outputs for delivery at outer boundary `t + 1`;
   they do not arrive during the current event loop.
8. Resolve every directly connected stateless inhibitory relay at the same
   `current_tau`.
9. Apply every outgoing hard reset from those relays immediately. Reset voltage and
   discard remaining current-boundary drive on each target.
10. Recompute crossing candidates from the changed state and continue.

When no finite candidate remains, advance every membrane through `1 - current_tau`
and set `current_tau = 1`.

### Boundary finalization

1. Update local activity traces from each cell's final/pre-reset and spike state.
2. Queue ordinary feedforward, basal, apical, and pretrained outputs for outer
   boundary `t + 1`.
3. Decay legacy conductance once and advance the existing integer refractory state
   once.
4. Record spike times, hard resets, ties, weights, and dynamic state.

There is no deterministic L2 WTA pass anywhere in this sequence.

No spike may traverse both `L1E -> L2E` and `L2E -> L1C` in one boundary. For an
isolated causal input:

```text
t:       RG fires
t+1:tau  pretrained L1E crosses and fires
t+2:     L1E basal reaches C; L1E feedforward reaches L2E
t+2:tau  first eligible L2E crosses; L2I immediately resets all L2E
t+3:     winning L2E apical reaches C while t+2 basal eligibility is valid
t+3:tau  C may cross; L1I immediately resets paired L1E if it has not fired earlier
```

If `L2E` needs additional accumulation boundaries before firing, the one-boundary
basal eligibility will expire. A later L1E event must refresh basal eligibility. This
is intended: the C cell detects temporal overlap, not an arbitrarily old association.

If paired L1E crosses before C during `t + 3`, that L1E spike is legitimate and cannot
be retroactively erased. If C crosses first, immediate L1I reset cancels L1E's later
prospective crossing. L1 suppression therefore also emerges from latency rather than
an unconditional next-boundary veto.

## Serialization and dashboard behavior

Static topology serialization must report:

- `e_pretrained`, `e_coincidence`, and `e_latency_competitor` archetypes;
- the four new edge kinds;
- the live basal weight on `basal_excitation`;
- `weight = null` for unweighted apical edges;
- the fixed delivered charge or a clearly named fixed magnitude for
  `pretrained_excitation`;
- `weight = null` for hard-reset edges.

Dynamic state for each C cell must include:

```text
basal_weight
basal_received
basal_eligible
apical_active
apical_sources
coincidence_active
coincidence_charge
potential
v_pre_reset
spiked
spike_tau
refractory
trace
```

Any engine or serializer check that currently recognizes membrane-bearing cells with
`isinstance(n, ExcitatoryNeuron)` must be generalized to the shared
`ConductanceLIFNeuron` base. Otherwise a sibling C class would silently lose
conductance, trace, refractory, and pre-reset state in the protocol.

Dynamic state must expose reset events separately from conductance pulses as:

```text
hard_reset_events
```

Each record must identify source, target, causal edge, outer boundary, and `tau`. Do
not mislabel a hard reset as conductance or fabricate a conductance magnitude.

Every event-resolved E spike must serialize `spike_tau`. Relay spikes inherit their
driver's `tau`. For compatibility, the boundary's `winner` field may report the first
latency-competitor spike, but no learning or control logic may read that field.
Serialize `latency_ties` with the tied IDs, tolerance, chosen stable fallback, outer
boundary, and `tau`.

`set_synapse_weight(edge_id, weight)` must support basal edges and clip with the
C-specific basal cap. Apical, pretrained, relay, and hard-reset edges are not manually
weight-editable.

The renderer and inspector must visually distinguish basal and apical projections.
The C-cell inspector must show the learned basal weight and the current Boolean
apical gate. The receptive-field panel may omit C cells in the first implementation;
it must not pretend that apical connections are learned receptive-field weights.

## Configuration

Add resolved parameters with these conceptual names:

```text
c_eta
c_basal_weight_init
c_basal_weight_max
c_basal_window_steps = 1
pretrained_exc_margin = 1.05
crossing_time_tolerance = 1e-12
```

Requirements:

- C threshold reads `e_threshold` directly;
- C leak reads `leak_rate` directly;
- C refractory reads `refractory_steps` directly;
- The original implementation default made `c_eta` resolve to the shared excitatory
  `eta`. Post-implementation turnover calibration now sets `c_eta=0.001` independently
  while ordinary E/L2E `eta` remains `0.01`; see
  `docs/COINCIDENCE_TURNOVER_TUNING.md` for the sweep and rationale;
- `c_basal_window_steps` is fixed at `1` for this experiment and need not initially
  be dashboard-editable;
- initial weight and cap defaults are derived from the resolved threshold and leak
  using the two-coincidence equations above unless explicitly overridden by a
  headless experiment;
- the public parameter payload reports the resolved numeric values;
- `crossing_time_tolerance` is a fixed numerical reproducibility constant, not a
  biological coincidence window or dashboard control;
- unknown or invalid values are rejected rather than silently clipped into a
  scientifically different regime.

Do not add inhibitory burst controls in this implementation.

## Required tests

### Unit tests: dendritic truth table

Test an isolated C cell with leak disabled where appropriate:

1. No inputs deposits zero.
2. One basal event deposits zero without apical activity.
3. Repeated basal-only events never change `V` except ordinary leak toward rest.
4. One or repeated apical-only events deposit zero.
5. Simultaneous basal and apical deposits exactly the pre-update basal weight.
6. Basal at `t`, apical at `t + 1` deposits exactly once.
7. Basal at `t`, apical at `t + 2` deposits zero.
8. Apical at `t`, basal at `t + 1` deposits zero without a new apical event.
9. Several apical sources on one boundary still cause one basal deposit.
10. Consumed basal eligibility cannot be reused.

### Unit tests: learning

1. Verify the exact numeric update equation from known `theta`, `w`, `w_max`, `eta`,
   `s`, and distance influence.
2. Verify `FE = theta - w` and includes no apical term.
3. Verify basal-only, apical-only, and valid subthreshold events do not update.
4. Verify only a C spike updates.
5. Verify no negative participation update exists.
6. Verify the deposited event uses the pre-update weight.
7. Verify clipping and saturation at `c_basal_weight_max`.
8. Verify there is no apical weight vector.

### Unit tests: firing, leak, and refractory

1. C threshold, leak, rest/reset, trace, and refractory values equal an equivalently
   configured ordinary E cell.
2. Exact segment advancement over `Delta tau = 1` matches the existing full-boundary
   conductance result when no within-boundary event intervenes.
3. Analytic threshold time agrees with direct substitution into the voltage trajectory.
4. A suprathreshold C membrane cannot fire on a boundary without a valid gate.
5. A valid event during refractory may affect membrane state but cannot fire or learn.
6. After refractory, learning waits for a new valid coincidence.
7. Repeated unmatched events can never cause a spike.
8. A cell emits at most one spike per outer boundary and consumes its drive on firing.

### Unit tests: calibrated two-coincidence regime

For the resolved default threshold and leak with `T = 1`:

1. Verify one maximum-weight gated deposit remains below threshold from reset.
2. Verify two consecutive initialized-weight gated deposits reach threshold.
3. Verify the isolated mature C spike sequence is every second valid coincidence after
   each reset, with refractory zero.
4. Verify the resolved cap remains below the one-deposit firing weight `w_1`.

These tests must use the actual analytic event resolver, not a jump-integrator or
fixed-microchunk approximation.

### Unit tests: pretrained L1E and hard reset

1. One RG spike schedules one paired fixed L1E charge event.
2. The unobstructed L1E fires on its delivery boundary.
3. No L1E weight changes.
4. If an immediate relay reset occurs at `tau` before the L1E crossing time, it clears
   retained `V` and remaining fixed drive and prevents the L1E spike.
5. The reset does not change weights, trace, conductance, or refractory state.
6. Multiple reset events are state-idempotent but remain individually observable.
7. One relay input creates one relay spike and one reset per outgoing edge; repeated
   same-boundary input creates no burst.
8. A reset after an earlier L1E spike does not retroactively remove that spike.
9. A pre-spike reset preserves the maximum pre-reset depolarization used by the local
   activity-trace update even though final membrane voltage is at rest.

### Unit tests: emergent latency WTA

1. Configure two latency competitors with crossing times `tau_A < tau_B`; assert A
   fires, immediately drives L2I, and B is reset before firing.
2. Reverse their drive/membrane states and assert B wins without changing node order.
3. Assert winner selection never calls the legacy deterministic WTA path or compares
   end-of-boundary voltages.
4. Assert all membranes advance to the winner's exact `tau` before reset is applied.
5. Assert the winner's spike and learning survive its own subsequent L2I reset.
6. Assert reset discards each loser's remaining drive and invalidates its former
   crossing candidate.
7. Create an exact tie; assert stable fallback, one emitted L2 spike, and a serialized
   `latency_tie` diagnostic.
8. Create two close but resolvable times outside tolerance; assert the earlier time
   wins regardless of node index.
9. Assert ordinary excitatory outputs from the winner arrive only on the next outer
   boundary despite immediate relay inhibition.
10. Assert the all-to-all L2I reset permits at most one L2E spike per outer boundary.

### Topology and timing tests

1. Assert exactly 45 nodes and 196 edges with the counts above.
2. Assert every C has one paired basal source and all eight L2 apical sources.
3. Assert C has no ordinary flat feedforward or apical weights.
4. Assert L1I hard reset is one-to-one and L2I hard reset targets all L2E cells.
5. Assert the causal chain follows `RG(t) -> L1E(t+1) -> L2E(t+2) -> apical(t+3)`.
6. Assert the basal event at `t+2` remains eligible for the apical event at `t+3`.
7. Assert a source spike cannot cross two ordinary excitatory edges in one outer
   boundary.
8. Assert L1I/L2I and their reset outputs share the causal E spike's `(t, tau)`.
9. Assert all L2 nodes are `e_latency_competitor` and no legacy deterministic
   competitor participates in this preset.
10. Assert manual topology validation rejects missing/multiple basal edges and missing
   apical fan-in.
11. Assert validation rejects a graph mixing legacy engine-WTA competitors with the
   event-resolved archetypes/reset policy.

### Regression tests

All existing tests and golden snapshots for `pi`, `old`, `rg`, and `rg_residual` must
remain unchanged unless a purely additive serialization vocabulary field is
deliberately updated. No existing topology may acquire hard-reset semantics by
accident.

### Experiment-level acceptance

Add a deterministic headless experiment with explicit counters for:

- RG spikes;
- unobstructed and reset-suppressed L1E spikes;
- basal events;
- apical events and source IDs;
- valid C coincidences;
- C spikes;
- L1 hard-reset events;
- every event-resolved E spike's `tau`;
- L2 winner latency, runner-up counterfactual latency, and winning margin;
- latency ties and fallback use;
- L1E frequency before and after the loop matures.

The isolated driver must demonstrate exact two-valid-events-to-one-C-spike behavior.
The full preset should then test whether mature paired resets produce an L1E/RG firing
ratio near `0.5`. Report the measured ratio and timing; do not force it with an
engine-level alternating-step override. It must also demonstrate that exchanging two
competitors' local drive exchanges their winner identity without changing node order.

## Implementation sequence

Implement in this order so failures remain attributable:

1. Add the shared membrane base and analytic segment/crossing primitives without
   changing ordinary E behavior; run all tests.
2. Add `DendriticCompartment`, isolated `CoincidencePyramidalNeuron` behavior, the
   exact C learning rule, and calibrated-weight tests.
3. Add NetworkSpec archetypes, edge kinds, validation, and enough metadata-driven
   construction to build small synthetic scheduler fixtures. Do not advertise the
   built-in preset while its runtime semantics are incomplete.
4. Add threshold-time scheduling, immediate relays, event invalidation, and hard-reset
   behavior on synthetic networks; run all legacy regressions.
5. Add structured basal/apical dispatch, fixed pretrained excitation, and the complete
   `rg_coincidence` preset, then verify its topology and causal timing end to end.
6. Add serialization, editor selection, renderer, and inspector support.
7. Run the complete regression suite.
8. Run the isolated cadence experiment, then the full frequency-halving experiment.
9. Update README and current-methodology documentation with measured behavior only.

## Expected implementation surface

The implementation is expected to touch at least:

| Path | Required change |
| --- | --- |
| `snn/neurons.py` | Shared LIF base, dendritic compartment, C neuron, hard reset |
| `backend/network_spec.py` | New archetypes, edge kinds, validation, preset graph |
| `backend/simulation.py` | Construction, structured dispatch, event scheduler, immediate reset, config/state |
| `backend/layout.py` | Deterministic `L1C` functional positions without new RNG draws |
| `backend/dashboard_config.py` | New preset and only the approved configuration exposure |
| `backend/api.py` | Vocabulary propagation and rejection of scalar C stimulation |
| `frontend/editor.js` | C palette entry and explicit basal/apical edge-kind selection |
| `frontend/renderer.js` | C node and basal/apical/reset edge presentation |
| `frontend/inspector.js` | C dendritic state, basal weight, and reset diagnostics |
| `tests/` | Unit, topology, timing, serialization, regression, and cadence tests |
| `experiments/` | Isolated cadence and full frequency-halving measurements |

Do not edit generated experiment results or legacy golden files merely to hide a
regression. A deliberate protocol extension must be reviewed independently from a
numeric behavior change.

## Non-goals

This implementation must not add:

- a general continuous-time synaptic-arrival or variable-outer-step solver; only
  analytic event times inside a fixed unit outer boundary are added;
- raw multiplication of basal and apical magnitudes;
- apical weights or apical plasticity;
- negative basal learning events;
- inhibitory bursts or multi-boundary inhibitory spike trains;
- per-edge variable delays;
- global winner-ID reads inside C cells;
- global input-vector reads inside C learning;
- engine-level forced alternation or frequency division;
- changes to `SwitchInterneuron`;
- changes to legacy conductance/predictive-inhibition edge behavior;
- a general biological dendritic tree model.

## Completion criteria

The implementation is complete only when:

1. branch identity survives event delivery;
2. unmatched branch activity provably deposits zero charge;
3. the one-boundary basal eligibility contract passes all timing tests;
4. only the basal weight exists and updates by the specified equation on a causal C
   spike;
5. C intrinsic dynamics numerically match E intrinsic dynamics;
6. pretrained RG input fires L1E in one unobstructed event;
7. analytic crossing times, rather than fixed chunks or maximum end voltage, order
   event-resolved E spikes;
8. immediate inhibitory relays hard-reset their targets at the causal spike's `tau`;
9. L2 WTA emerges from first-spike latency and the L2I reset loop with no deterministic
   winner-selection phase;
10. the isolated C cell demonstrates the calibrated two-coincidence cadence;
11. full-preset frequency behavior is measured without a forced alternation heuristic;
12. all legacy topology tests remain green.
