# Neuronal Arithmetic and the `SwitchI` Design

## Status and purpose

This note records the conclusions drawn from a detailed reading of R. Angus
Silver's review, *Neuronal arithmetic* (2010), and relates them to the
`rg_residual` topology. It is a design record for work that will continue later;
it is not a claim that the current switch pathway has already succeeded in the
normal classification experiment.

The immediate project order is:

1. Establish why `SwitchI` is not firing during normal dashboard activity.
2. Prove that the existing two-branch pathway receives the intended signals in
   the intended temporal order.
3. Only after that mechanism works, replace the present charged additive
   coincidence rule with an explicitly multiplicative rule.
4. Evaluate paired hard reset as a separate output policy after coincidence is
   reliable.

The investigation prompt for the first step is
[`claude_switchi_investigation_prompt.md`](../prompts/claude_switchi_investigation_prompt.md).

## Source

Silver, R. A. (2010). “Neuronal arithmetic.” *Nature Reviews Neuroscience*,
11, 474–489. DOI: [10.1038/nrn2864](https://doi.org/10.1038/nrn2864).
An open full-text version is available from
[PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC4750293/).

The paper is a review of several mechanisms rather than a proposal for one
universal biological multiplier. Its central method is to define an arithmetic
operation by how a modulatory input changes a neuron's input-output relationship.

## What “multiplication” means in the paper

Silver distinguishes driving input, `d`, from modulatory input, `m`. Two
forms of multiplicative modulation are defined:

### Input multiplication

\[
R(d,m)=f(d\,m)
\]

The modulatory input scales the effective driving input before the neuron's
spiking nonlinearity.

### Output multiplication

\[
R(d,m)=f(d)\,g(m)
\]

The modulatory input scales the response produced by the driving input.

In rate-coded experiments, multiplication is identified as a change in the
slope or gain of the driving input-output curve. Ideally this gain change occurs
without a shift in rheobase, although the review repeatedly notes that biological
mechanisms often produce mixed additive and multiplicative effects.

This is different from additive input modulation:

\[
R(d,m)=f(d+m),
\]

which moves the input-output relationship along the input axis and changes how
much driving input is required to reach threshold.

The paper therefore does **not** claim that neurons universally calculate the
raw instantaneous product of two membrane voltages. “Multiplication” normally
means that one signal changes the gain with which another signal is transformed
into output.

## Two different neural coding regimes

The paper's most important distinction for this project is between sustained
rate coding and sparse temporally correlated coding.

### Sustained rate-coded signalling

The neuron integrates ongoing asynchronous input. Its input-output relationship
is commonly represented as:

\[
\text{input firing rate} \longrightarrow \text{output firing rate}.
\]

A multiplicative operation changes the gain of this rate relationship. Relevant
mechanisms include synaptic noise, balanced excitation and inhibition,
short-term synaptic depression, shunting conductance under particular conditions,
and active somatic or dendritic channels.

### Sparse temporally correlated signalling

The neuron behaves as a coincidence detector. Its input-output relationship can
instead be expressed as spike probability versus either the number of coincident
inputs or their temporal dispersion:

\[
P_{\mathrm{spike}}=f(N_{\mathrm{coincident}})
\]

or

\[
P_{\mathrm{spike}}=f(\sigma_{\mathrm{input}}).
\]

Here, gain determines the sharpness of the coincidence window: a high-gain
relationship can sharply discriminate sufficiently synchronized input from
dispersed input.

`SwitchI` belongs primarily to this second regime. It is not meant to encode a
stable firing rate. It must recognize a temporally structured condition:

```text
recent activity of incumbent L2E[j]
AND
current unexplained ErrorE activity
```

Consequently, the paper's sparse coincidence discussion and its treatment of
local dendritic sodium, NMDA, and calcium events are more directly relevant than
its sustained-rate examples.

## Mechanisms reviewed by Silver

### Shunting inhibition

A shunting inhibitory conductance raises membrane conductance and reduces both
the amplitude and duration of an excitatory postsynaptic potential. At a
subthreshold voltage this can look like multiplicative scaling of an EPSP.

However, Silver emphasizes that this does not automatically produce divisive
gain modulation of a spiking neuron's rate-coded input-output curve. Depending
on the coding regime and the background noise, shunting inhibition can be:

- subtractive;
- divisive;
- a mixture of subtractive and divisive components; or
- weak as a gain-control mechanism.

Without biologically appropriate input-dependent noise, steady shunting
inhibition often shifts the sustained firing relationship rather than changing
its gain. Under sparse temporally correlated signalling it frequently alters the
threshold and temporal integration window rather than acting as a clean
multiplier.

This matters for the project because persistent conductance should not itself be
treated as the `SwitchI` multiplication mechanism. The coincidence computation
and the inhibitory consequence of a switch spike are separate design decisions.

### Input-dependent synaptic noise

Synaptic noise broadens the distribution of membrane voltage and smooths the
mapping from mean voltage to output firing. It can turn a sharp threshold into a
power-law-like response. When conductance and noise vary with input rate, an
otherwise additive voltage shift can become an apparent multiplicative change in
output gain.

The exact outcome depends on whether noise is fixed or input-dependent and on how
excitation and inhibition are balanced. This is a general gain mechanism, not a
simple two-input Boolean gate.

### Short-term synaptic depression

Short-term depression creates a saturating relationship between presynaptic
firing rate and mean excitatory conductance. Silver reviews experiments and
models in which that saturation converts an inhibition-mediated additive shift
into a largely multiplicative output gain change.

The reported effect remains partly additive, depends on the amount of depression
and inhibition, and develops on a timescale determined by presynaptic frequency
and synaptic dynamics. In the cerebellar example reviewed, this ranged from
approximately 4 to 75 ms.

This mechanism concerns rate-dependent transformation rather than the specific
two-event temporal AND required by `SwitchI`.

## Active dendrites and coincidence

### Dendritic sodium spikes

Spatially clustered and highly synchronized excitation can trigger a local
dendritic sodium spike. In the examples reviewed by Silver:

- the effective coincidence window can be approximately 3–6 ms;
- the event can remain restricted to one branch;
- correlated inputs are amplified sharply;
- the responsible channels may inactivate for hundreds of milliseconds; and
- the mechanism is therefore better suited to sparse correlated activity than
  sustained firing.

This resembles a fast, branch-local thresholded AND, although the paper describes
it through spike probability and dendritic thresholding rather than as an exact
algebraic product.

### NMDA spikes

NMDA receptors provide a particularly important conditional mechanism. Strong
current requires both:

1. glutamate bound to the receptor; and
2. sufficient local depolarization to relieve the voltage-dependent magnesium
   block.

The current can be viewed schematically as:

\[
I_{\mathrm{NMDA}}
\propto
s_{\mathrm{glutamate}}
B(V_{\mathrm{dendrite}}),
\]

where (s_{\mathrm{glutamate}}) is transmitter-dependent receptor activation and
(B(V)) is voltage-dependent magnesium unblock. One factor controls the efficacy
of the other.

Silver describes several useful temporal and spatial properties:

- NMDA spikes are local to the activated dendritic region because they depend on
  local glutamate binding.
- They last roughly 50–100 ms in the reviewed examples.
- Glutamate can remain bound while the channel is magnesium-blocked, creating an
  approximately 40 ms memory in which a later depolarization can unlock current.
- Their coincidence window is therefore longer than that of dendritic sodium
  spikes.
- They can interact with both sparse correlated input and short bursts.

This is the closest mechanism in the review to the desired `SwitchI` operation:
one branch supplies a local state and the other becomes effective only when that
state is present.

### Calcium spikes and spatial association

Layer-5 pyramidal neurons receive spatially separated inputs on basal dendrites
and the distal apical tuft. A somatic action potential can backpropagate into the
dendrite and lower the threshold for a calcium spike generated by apical input.
The associated event can produce a burst.

This supports nonlinear association between spatially and temporally related
signals. It should not be described as proof that all such cells calculate an
exact raw product. In the reviewed experiments, distal input can change both gain
and rheobase, again demonstrating that real neuronal arithmetic can mix
multiplicative and additive components.

### Dendritic branches as computational subunits

Clustered input can activate a nonlinear sigmoidal response in an individual
dendritic branch. Multiple nonlinear branch outputs can then be combined at the
soma and passed through another spiking threshold. Silver discusses the proposal
that this gives a single pyramidal neuron computational structure analogous to a
two-layer network.

This supports maintaining genuinely separate branch states in a simplified
model. Collapsing both signals immediately into one undifferentiated accumulator
would lose the computational distinction that active dendrites provide.

## Limits explicitly acknowledged by the paper

Silver does not present these mechanisms as settled universal explanations. The
review emphasizes that:

- additive and multiplicative components are often mixed;
- interacting nonlinear conductances make causal identification difficult;
- mechanisms differ between sparse and sustained coding regimes;
- precise spatial clustering and timing requirements may be difficult to satisfy
  in vivo;
- substantial inhibition may change the robustness of NMDA spikes; and
- additional in-vivo work is needed to establish which mechanisms implement
  particular computations.

The appropriate claim for this project is therefore:

> Nonlinear dendritic and synaptic mechanisms can allow one input to
> multiplicatively modulate a neuron's response to another. In sparse regimes,
> active dendrites can operate as nonlinear spatiotemporal coincidence
> detectors.

The paper should not be cited as demonstrating that a particular SST interneuron,
our exact topology, or a raw software product (X\times R) is uniquely correct.

## Mapping onto `rg_residual`

The current residual topology has the following conceptual mapping:

| Project component | Computational role |
| --- | --- |
| `RG → L1E` | Persistent external evidence path |
| `L1E → L2E` | Bottom-up drive to the competitors |
| `L1E → ErrorE` | Separate copy of current evidence |
| `PI[j] → ErrorE` | Prediction of evidence explained by competitor `j` |
| Surviving `ErrorE` spike | Current unexplained evidence or residual |
| `L2E[j] → SwitchI[j]` | Local record that competitor `j` recently owned the prediction |
| `SwitchI[j] → L2E[j]` | Paired release or eviction of that incumbent |
| Shared `L2I` | Ordinary one-winner arbitration |

The intended question asked by `SwitchI[j]` is:

> Was `L2E[j]` recently responsible for the active prediction, and is there now
> unexplained evidence?

This is a temporal attribution problem, not literal same-boundary coincidence.
The current engine deliberately resolves residual activity against a trace carried
into the boundary before recording the current L2 winner for future boundaries.
A newly selected winner therefore cannot use its own same-boundary spike to
inhibit itself.

## Current charged coincidence rule

The existing `SwitchInterneuron` maintains two separately visible branches:

- a decaying paired winner trace (X_j), which produces bounded trace charge when
  eligible; and
- current-boundary residual events, which produce bounded residual charge.

Neither branch can individually reach threshold. The switch also explicitly
requires a residual event and an eligible trace. It is therefore logically
AND-like, even though the visible electrical combination is currently additive.

This mechanism has unit and causal tests, but normal dashboard activity has not
yet demonstrated reliable `SwitchI` firing. The first question is not yet which
coincidence equation is best. It is whether normal activity supplies both operands
in the required order:

```text
earlier L2E[j] spike
→ surviving paired trace
→ later ErrorE spike
→ SwitchI[j] response
```

The pathway must be audited boundary by boundary before its arithmetic is changed.

## Proposed multiplicative coincidence

Once the existing pathway is demonstrated, the next experimental rule should
preserve the two branch states and combine normalized nonlinear branch activations:

\[
C_j=g(X_j)\,h(R).
\]

For a strict temporal AND, each branch must have an exact inactive region. One
possible abstract definition is:

\[
g(X_j)=
\begin{cases}
0, & X_j<X_{\min} \\
\sigma(k_x(X_j-X_0)), & X_j\ge X_{\min},
\end{cases}
\]

and

\[
h(R)=
\begin{cases}
0, & \text{no residual event on the current boundary} \\
\sigma(k_r(R-R_0)), & \text{residual present}.
\end{cases}
\]

The switch fires when:

\[
C_j\ge\theta_C.
\]

This formulation has the desired absorbing-zero property:

- no eligible context gives (C_j=0);
- no current residual gives (C_j=0); and
- both branches can produce a graded suprathreshold coincidence.

For binary variables, multiplication is exactly Boolean AND. For decaying traces
and graded evidence, it is a continuous temporal AND followed by a threshold.
Explicit inactive regions are important because a decaying floating-point trace
otherwise approaches zero without becoming exactly zero.

The product should be understood as a phenomenological approximation of
conditional dendritic gain. A more biophysical NMDA-like model would multiply a
transmitter-dependent state by a voltage-dependent unblock function, but that
additional detail is not required to test the central network hypothesis.

## Hard reset as an output policy

Multiplicative coincidence determines **when** a switch fires. Hard reset or
persistent conductance determines **what the switch spike does**. These questions
must remain experimentally separate.

A hard reset was destructive in the old topology because a winner recruited
global inhibition of the sensory population. The residual topology permits a much
more local interpretation:

\[
\text{recent }L2E_j
\times
\text{residual error}
\longrightarrow
SwitchI_j
\longrightarrow
\text{evict only }L2E_j.
\]

Simply setting the target voltage to rest may be insufficient because sustained
L1 input can recharge the same competitor immediately. The clean experimental
policy would be a one-boundary paired veto:

1. `SwitchI[j]` fires on boundary (t).
2. It schedules an eviction for (t+1).
3. On (t+1), only `L2E[j]` is clamped to rest and cannot fire or learn.
4. Its learned weights and long-term representation remain intact.
5. It becomes eligible again on (t+2).

The switch's eligibility trace should be consumed by a successful coincidence so
one old event cannot repeatedly evict the same competitor.

This local veto gives the shared WTA mechanism one boundary in which to select a
replacement. `PI → ErrorE` should remain graded predictive conductance, and the
shared L2 WTA mechanism should remain distinct from switch-mediated eviction.

Potential oscillation is the primary risk. It can be controlled by requiring the
winner trace to mature, consuming it on eviction, giving new winners a grace
period, or requiring persistent residual evidence rather than a single noisy
event.

## Resume plan

### Phase 1: establish the existing causal pathway

Run controlled experiments that demonstrate:

1. Residual alone charges switches but fires none.
2. A paired L2 trace alone fires none.
3. An earlier `L2E3` spike followed by a later residual fires exactly `SwitchI3`.
4. The resulting delayed inhibition targets only `L2E3`.
5. A new same-boundary winner cannot immediately inhibit itself.

Then trace normal `rg_residual` activity and find the first missing link between
L2 priming, ErrorE threshold crossing, residual broadcast, coincidence resolution,
and paired output.

### Phase 2: multiplicative ablation

Keep the topology fixed and compare:

- current charged additive AND;
- normalized multiplicative coincidence;
- multiplicative coincidence with a facilitating or maturing context branch; and
- multiplicative coincidence that requires repeated residual evidence.

This isolates neuron dynamics from topology changes.

### Phase 3: output-policy ablation

With coincidence fixed, compare:

- persistent paired conductance;
- membrane reset only; and
- one-boundary paired eviction.

Do not change both coincidence arithmetic and output policy in the same initial
experiment.

### Required measurements

At minimum, measure:

- whether every presentation still produces at most one L2 winner;
- switch true-positive and false-positive events;
- incumbent dwell time and successful replacement;
- classification specialization and accuracy;
- continuous-learning recovery when patterns change;
- total spike and interneuron sparsity;
- sensitivity to input period, trace decay, delays, and coincidence threshold;
- oscillation or repeated eviction of new winners; and
- whether unexplained evidence disappears after a correct replacement.

## Current conclusion

Silver provides strong support for the architectural principle of separate local
states combined through nonlinear spatiotemporal coincidence. It does not establish
that the current `SwitchI` topology is correct or that a raw product is the only
appropriate model.

For this project, the scientifically clean interpretation is:

```text
paired L2 context controls the gain of current residual evidence;
their nonlinear temporal coincidence produces a local switch event;
the switch event may then evict only the responsible incumbent.
```

Work should resume by making the existing switch pathway visibly and causally
functional. Multiplication and hard reset are the next two controlled experiments,
not substitutes for that prerequisite.
