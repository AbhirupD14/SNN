# Claude implementation brief: repair the NEST backend to express CIPP semantics

## Mission

Audit findings are in
`docs/CIPP_CUSTOM_ENGINE_TO_NEST_SEMANTIC_AUDIT.md`. Implement a new continuous-time CIPP
profile in the NEST backend that addresses those findings without deleting or relabeling
the current impulse prototype.

The goal is **not byte-for-byte replay of the Python engine's boundary scheduler**. The goal
is to preserve the CIPP causal semantics for which that scheduler was a workaround:

- weight-supported detectors race through membrane latency, not invented afferent scan order;
- sender, edge, target, and causal-volley membership remain meaningful;
- a local WTA winner suppresses later candidates when the physical timing envelope permits;
- a valid basal/apical coincidence is explicit and source-causal;
- one prediction creates one suppression credit consumed by the next eligible evidence;
- learning is local, logically firing-triggered, and uses the declared causal state.

Do not alter `backend/` or `snn/` to make NEST look correct. They are the compatibility
oracle. Do not use Python per-spike neural callbacks.

## Non-negotiable separation

Retain the current implementation under an explicit profile/name such as
`impulse_characterization`. Preserve its tests and recorded negative findings.

Add a separate `cipp_continuous` profile. No default switch is authorized until all
mechanical gates below pass.

Keep these outcome classes separate in code, manifests, tests, and reports:

1. **mechanical correctness** — event delivery, timing, WTA, C, feedback, and learning obey
   their declared contracts;
2. **compatibility** — outcomes agree with the Python oracle on cases where both profiles
   claim the same semantics;
3. **scientific result** — consolidation, turnover, hierarchy, or certainty succeeds or
   fails under a mechanically correct engine;
4. **native NEST difference** — an intentional consequence of nonzero communication delay
   or another explicitly accepted NEST constraint.

An open scientific hypothesis may fail without failing the engine. A known semantic
divergence may not be converted into a passing compatibility test.

## Phase 0: freeze evidence and add failing regression probes

Before changing a model, add focused tests that reproduce every P0 finding.

Required probes:

1. **Current-volley `I_accq`:** three 400-charge presentations into `theta=1000` must fire
   with retained membrane/charge 1200 but logical ordinary-E `I_accq=400`.
2. **C accumulation:** a frozen C with basal `theta/4` and repeated valid coincidences must
   fire on cumulative deposits; separately prove that basal `theta` is one-shot.
3. **Owner identity:** the A/B artifact must report every NEST winner, distinct-winner count,
   stability, and agreement with the Python owner—not multiplicity alone.
4. **Late afferent:** an afferent emitted in the same source volley but delivered after the
   postsynaptic spike must receive `s_i=-1`, never `+1`.
5. **Multiple deferred posts:** one synapse must process two or more postsynaptic spikes
   between presynaptic spikes using each post's historical FE snapshot and sequential own
   pre-update weight.
6. **Silent-afferent flush:** logical final weights must include all post-triggered negative
   updates even when an afferent never spikes again.
7. **Pacing independence:** a prediction credit must survive silence and suppress the next
   evidence at several irregular delays.
8. **Window independence:** changing presentation interval alone must not change C or I
   physiology.
9. **Quantization:** delay quantization must never produce a value below the requested delay.

Commit a machine-readable semantic matrix with columns:

```text
contract, python_compatibility, impulse_characterization,
cipp_continuous_target, test, status, accepted_difference
```

## Phase 1: continuous-time competitor feasibility spike

Do not translate the full topology yet.

Build a two-competitor microcircuit with:

- synchronous source spikes;
- identical conduction delay for equivalent afferents;
- different total active weights;
- continuous postsynaptic current dynamics;
- nonzero NEST-valid E-to-I and I-to-E delays;
- authoritative NEST spike recording.

Evaluate, in this order:

1. a built-in precise model (`iaf_psc_exp_ps`, `iaf_psc_alpha_ps`, or the smallest justified
   alternative);
2. a NESTML continuous model if its grid behavior converges adequately;
3. a minimal custom C++ NEST extension if NESTML cannot preserve the required crossing
   semantics or expose the causal state needed by learning.

Do not reject precise models merely because they do not provide zero-delay synapses. The
question in this phase is whether they recover the **membrane-latency race** with a fast but
nonzero inhibitory loop.

Pass gates:

- with uniform arrival time, the stronger total supported drive fires first;
- scaling all equivalent conduction delays together does not reverse the winner;
- reducing `h` converges spike time and winner identity;
- if runner-up gap exceeds the measured E-to-I-to-E latency, only one committed winner is
  emitted;
- exact equal-drive cases are reported as ties or use a separately specified local tie
  mechanism;
- geometry and random jitter are both disabled.

If no supported model can pass, stop this phase with a precise feasibility report. Do not
fall back to geometric delay spread.

## Phase 2: define the physical-time profile

Add one immutable configuration object for the new profile. Every parameter must have a
unit and one semantic role.

At minimum define:

- kernel resolution;
- feedforward conduction delay by projection class;
- excitatory synaptic-current shape and time constant;
- membrane time constant, threshold, reset, and refractory duration;
- E-to-I and I-to-E delays;
- C basal and apical coincidence windows;
- a deposit dead time/refractory rule independent of input pacing;
- a causal-volley separation rule used only for compatibility learning membership;
- prediction-credit count/capacity and consumption rule.

Remove `presentation` from every neuron-model parameter dictionary. The experiment
controller may schedule presentations, but it must not set dendritic TTLs, relay lockout,
or coincidence physiology.

For the reference profile:

- equivalent edges in one projection use uniform delays;
- layout distance remains a learning multiplier only;
- delay jitter is zero;
- optional geometry/jitter experiments live under separately named ablation profiles.

Fix delay quantization with ceiling-to-grid semantics and property tests.

## Phase 3: ordinary E, Eor, and WTA

Implement the selected continuous competitor model.

Required state must distinguish:

- membrane/current state retained by the neuron;
- charge belonging to the current causal volley;
- the firing snapshot exported as `I_accq`;
- refractory state;
- lateral WTA reset state;
- prediction-credit state and whether the current volley is being suppressed.

Do not use one reset port for both WTA and prediction. Required ports or equivalent explicit
mechanisms:

```text
feedforward_excitation
wta_reset
prediction_credit
```

Eor remains a fixed one-afferent relay at `theta` in the classic profile. It must not learn.
It must not emit multiple column outputs for one accepted column winner.

Measure the operating envelope rather than asserting absolute WTA:

```text
winner crossing time
runner-up counterfactual crossing time
E->I->E reset latency
margin = runner_up - winner - reset_latency
```

Positive-margin cases must be single-winner. Nonpositive cases must be flagged as unresolved
ties/races in metrics.

## Phase 4: event-conserving prediction feedback

Replace time-of-arrival reset semantics in `cipp_continuous` with a local counted credit.

Contract:

1. a valid C spike sends one confirmation to the column I mechanism;
2. that confirmation creates exactly one pending prediction credit;
3. silence does not consume a credit;
4. the next eligible feedforward volley consumes one credit;
5. every ordinary E in the bank suppresses that same volley;
6. a suppressed volley cannot create an ordinary-E winner, Eor output, new C confirmation,
   or another credit;
7. later volleys are unaffected unless more credits exist;
8. WTA reset traffic neither creates nor consumes prediction credits.

The implementation may use a dedicated gate/relay model, a separate NESTML port and local
state, or a justified custom extension. A fixed refractory timer that merely covers an
expected volley is not equivalent.

Acceptance schedule:

```text
evidence -> confirmation -> long silence -> evidence -> evidence
expected:     credit stored              suppressed   accepted
```

Repeat with several silence durations and presentation jitters. The accepted/suppressed
sequence must be identical.

## Phase 5: coincidence cell

Preserve separate basal and apical ports and same-time handler priority.

Required semantics:

- basal alone and apical alone deposit nothing;
- either arrival order may form a valid coincidence within declared physical windows;
- the causal basal source and delivered basal charge are retained;
- one basal token may be consumed at most once;
- one valid coincidence makes one instantaneous somatic deposit;
- repeated subthreshold deposits accumulate unless leak/refractory configuration explicitly
  says otherwise;
- C may fire only as a consequence of a valid gate opening;
- top C remains dormant by parent metadata;
- window expiry is observable and not coupled to presentation interval.

For classic `tiled_cc`, one basal afferent is sufficient. Design the state shape so a later
direct-identity port can preserve multiple basal source identities; do not bake a target-ID
special case into the neuron model.

Required tests include `theta/4` cumulative firing, `theta` one-shot firing, no cross-volley
false pairing, both arrival orders, same-time ordering, duplicate events, and top-C dormancy.

## Phase 6: learning transaction

Preserve the active dual FE/FES equations exactly. Do not substitute STDP and do not make
the magnitude a graded function of pre/post interval.

Ordinary E logical transaction at a post spike:

```text
I_accq = firing causal-volley charge, not retained membrane total
FE      = one shared historical firing snapshot
for every incoming plastic afferent i:
    FES_i = function(own pre-update w_i)
    s_i   = +1 only if i's event reached this target before the firing
            and belonged to the firing causal volley; otherwise -1
    w_i'  = clip(w_i + eta * FE * FES_i * s_i * phi_i)
```

C transaction updates only the basal edge whose event caused the firing deposit.

### Allowed implementation choices

**Preferred:** a model/extension that makes the logical post transaction material at the
post event.

**Conditionally allowed:** NEST's lazy synapse materialization, but only if all of the
following are proved:

- every archived post event retains its own `I_accq`;
- sequential updates use the synapse's correct evolving weight;
- catch-up occurs before the next delivery uses that weight;
- participation has both lower and upper bounds in delay-adjusted time;
- a post earlier than this afferent's arrival is nonparticipating;
- a non-perturbing flush materializes every logical update before checkpoint, convergence
  decision, weight export, replay header, or final metric;
- a differential test matches an independent Python oracle over randomized pre/post trains.

Do not call the phase blocked merely because materialization is lazy. Do call it blocked if
logical equivalence or a correct flush cannot be demonstrated.

## Phase 7: topology translation and observability

Translate canonical `tiled_cc` from `NetworkSpec`; do not hand-copy its graph.

Every run manifest must state:

```text
engine_profile
model and module fingerprints
NEST/NESTML versions
all physical parameters with units
learning enabled/disabled
logical weights flushed/not-flushed
delay policy (uniform/geometry/jitter)
accepted semantic differences
unsupported topology families
```

Fix artifact generation:

- learning runs export live materialized weights, not construction-time `self.weights`;
- `dashboard_topology()` and its `known_differences` are mode-dependent;
- weight recorders are attached whenever a learning artifact claims weight changes;
- replay distinguishes initial, logical-current, materialized-current, and final weights;
- the A/B report includes per-window winner identity, stability, agreement, multiplicity,
  C readiness, feedback credits, and suppression consumption.

Do not infer a causal latency from a sum of projection means. Record the actual traversed
edge delays or a distribution when no single path exists.

## Phase 8: acceptance suites

### Mechanical gate

All of these must pass before profile promotion:

1. structural counts and metadata parity;
2. uniform-delay total-drive winner microcontract;
3. WTA positive-margin single-winner contract;
4. explicit exact-tie behavior;
5. C truth table and cumulative deposit behavior;
6. event-conserving prediction under irregular pacing;
7. ordinary-E `I_accq` causal-volley probe;
8. all-afferent dual FE/FES oracle, including late and silent afferents;
9. final-weight flush and replay correctness;
10. chunked versus one-shot equivalence;
11. 1/2/4-thread spike, credit, and weight equivalence;
12. resolution-convergence sweep.

### Compatibility characterization

Run matched Python/NEST cases with identical graph, initial weights, source volleys, and
learning flag. Report rather than hide:

- winner identity and rank agreement;
- multiplicity and unresolved races;
- C deposits/spikes/readiness;
- logical weight deltas after each firing;
- feedback credits and their consuming evidence;
- differences caused by nonzero NEST delays.

Do not require identical wall-clock spike timestamps across the boundary oracle and the
continuous profile.

### Scientific suite (not an engine gate)

Repair the four-pattern experiment so it reports:

- pattern-to-owner mapping per column;
- number of unique owners;
- owner collisions;
- stable-owner duration;
- frozen-learning cold recall after the entire curriculum;
- forgetting and turnover when patterns return;
- complete seed distribution.

`stable recent owner + mature column C` is not sufficient to call a pattern learned.
Classic `tiled_cc` has one pooled C basal association, so do not describe that weight as a
pattern-specific four-association memory.

If the scientific suite fails after mechanical gates pass, report a CIPP model limitation,
not a NEST implementation defect.

## Required deliverables

1. new named `cipp_continuous` implementation profile;
2. preserved named impulse-characterization profile;
3. semantic matrix and parameter disposition table;
4. failing-before/passing-after regression probes;
5. continuous-competitor feasibility report with raw measurements;
6. mechanical, compatibility, and scientific reports kept separate;
7. corrected NEST README/report/dashboard claims;
8. exact reproduction commands and artifacts;
9. no changes to the Python oracle except independently justified test instrumentation.

## Stop conditions

Stop and report a bounded feasibility blocker rather than silently changing CIPP if any of
these cannot be achieved with supported NEST/NESTML mechanisms:

- continuous weight-driven winner ordering without delay jitter;
- WTA within a declared positive-margin operating envelope;
- causal-volley `I_accq` at firing;
- event-conserving next-evidence suppression;
- logically complete all-afferent learning with correct final materialization.

For each blocker, include the smallest reproducer, the exact NEST/NESTML limitation, and
the least invasive next option (NESTML redesign, custom C++ NEST extension, or keeping that
mechanism in the custom engine). Do not replace the rule with STDP, a Python callback,
geometric scan order, or a tuned random delay.

