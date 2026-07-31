# CIPP custom-engine to NEST semantic audit

**Audit date:** 2026-07-31  
**Scope:** the event-resolved Python engine in `backend/simulation.py` and
`snn/neurons.py`, the canonical `tiled_cc` graph, and the current PyNEST/NESTML
implementation in `nest_backend/`.  
**Verdict:** the NEST implementation is a useful characterization prototype, but it is
not yet a semantically valid CIPP engine. It has exact structural parity and several good
local contracts, but the mechanisms that determine winner identity, feedback, and
learning differ at scientifically meaningful points.

The existing NEST backend should remain available as a named **impulse-prototype**
profile. It should not be promoted by incrementally tuning `h`, geometric delay spread,
or random delay jitter. A separate continuous-time CIPP profile should be built behind
explicit acceptance gates.

## 1. What the custom engine was trying to represent

The custom engine is not the scientific target in every detail. It is a deterministic
workaround for four requirements that the original timestep implementation could not
resolve faithfully:

1. **Events are causal assertions.** Sender identity, target identity, edge identity,
   arrival time, and causal participation matter. A non-event is not an event with value
   zero.
2. **Time carries meaning.** Within a presentation, stronger supported detectors should
   reach threshold earlier. Earlier events may invalidate later prospective crossings.
3. **Space carries meaning.** A source address and the path it traverses are part of a
   representation. Display geometry must not silently change signal delivery.
4. **Learning is local and firing-triggered.** When a detector fires, all of its afferents
   undergo one logical update from a common postsynaptic snapshot; causal participants
   receive `+1` and nonparticipants receive `-1`. A C cell updates only the basal source
   that caused its valid coincidence.

The Python engine approximates continuous time by freezing one boundary's received charge
as a constant drive and analytically solving each membrane's threshold-crossing time
`tau in [0,1]`. The earliest valid crossing fires; its same-`tau` WTA reset invalidates the
later crossings. Outer boundaries also carry scientific state: delay-one delivery,
participation membership, C eligibility, refractory settlement, and the feedback-reset
workaround.

That produces the following compatibility oracle:

| Contract | Python-engine meaning |
|---|---|
| ordinary-E drive | one boundary's delivered charge is a constant current for that boundary |
| WTA | first analytic threshold crossing wins; reset discards losers' remaining drive |
| feedforward delay | one outer boundary per ordinary/basal hop |
| apical and lateral WTA | zero-latency consequences at the source spike's `tau` |
| C gate | current or one-boundary-carried basal AND current apical; at most one deposit per boundary |
| C soma | valid deposits accumulate; `theta/4` is not one-shot, but repeated coincidences can still reach threshold |
| ordinary-E learning trigger | one logical vector transaction at postsynaptic firing |
| ordinary-E `I_accq` | charge in the **firing boundary's delivered causal volley**, not retained membrane voltage |
| participation | event actually delivered to this target in the firing boundary |
| C learning | only the causal basal edge updates after a gated C spike |
| confirmation feedback | current implementation discards the next frozen packet when timing aligns |

The last row is a known workaround artifact, not the desired final CIPP rule. The intended
feedback meaning is closer to **one confirmed prediction creates one local suppression
credit, consumed by the next eligible evidence volley**. That preserves the causal event
instead of making its effect depend on a presentation clock.

The broader CIPP claim boundary is also important. The current fabric is an encoder and a
research substrate. Four-pattern consolidation, identity-preserving hierarchy, and a
calibrated certainty signal remain hypotheses or open experiments. A simulator must not
turn those open outcomes into implementation acceptance criteria.

## 2. What the NEST implementation gets right

These are real accomplishments and should be retained:

- The canonical graph translates from `NetworkSpec`, rather than from a second handwritten
  topology: 191 nodes and 1052 directed edges for default `tiled_cc`.
- Repository node and edge identities round-trip to NEST objects.
- RGC input ownership, patch locality, column roles, fixed E-to-Eor relay weights, dormant
  top C metadata, and disabled-feedback edge selection are structural rather than ID-string
  special cases.
- The event accumulator is an exact nonleaky **charge sum** at `leak_rate=0`.
- The C model has separate basal and apical ports, an explicit same-time priority, a strict
  AND gate, once-per-window deposit protection, and both basal-first and apical-first
  diagnostics.
- The I relay has separate lateral and confirmation ports, avoiding the earlier bug where
  lateral lockout swallowed every confirmation.
- NEST owns the clock, delivery, recorders, threads, and connection delays; Python does not
  perform hidden per-spike neural callbacks.
- The co-generated synapse code preserves historical postsynaptic `I_accq` samples. The
  generated C++ iterates archived post spikes and reads the continuous-variable history at
  each post time, so multiple deferred post events need not all use the latest snapshot.
- Deterministic construction, charge recording, replay conversion, and 1/2/4-thread spike
  checks are useful infrastructure.

Focused verification during this audit:

```text
Python semantic tests: 79 passed
NEST node/topology tests: 42 passed
```

Those tests establish their declared microcontracts. Some NEST tests deliberately assert
known divergence, so green tests do not imply CIPP semantic parity.

## 3. Findings

### P0. The NEST node model ports the workaround's charge total, not its continuous-time race

`event_accumulator.nestml` applies each arriving spike as an instantaneous jump in `q` and
fires in the receipt handler. The Python engine instead interprets the gathered boundary
packet as a constant drive and solves when it crosses threshold.

Both implementations can end with the same accumulated charge, but they do not encode the
same timing. With simultaneous delta arrivals, all supra-threshold competitors cross on
the same NEST step. The current backend therefore invents per-edge arrival dispersion to
create an order that the Python engine obtained from total weighted drive.

This changes the decision rule:

```text
Python compatibility oracle:  earliest crossing under total frozen drive
Current NEST prototype:        earliest weighted prefix sum in delay order
```

The distinction is measured in the repository: default NEST winner multiplicity is above
one, winner identity disagrees with the Python engine, and row/column outcomes depend on
geometric scan order.

**Required direction:** use NEST's continuous subthreshold dynamics and precise-spike
facilities, or a custom C++ NEST model if NESTML cannot emit the required precise model.
Do not use input-delay disorder as a substitute for membrane latency.

### P0. Geometry and random jitter currently decide content

The custom-engine contract says geometry scales learning influence only and never delivered
charge. The NEST backend additionally converts the engine's layout coordinates into
feedforward delays. Recent A/B and saturation scripts then use up to 3 ms of independent
per-synapse jitter and `h=0.001 ms` to recover one emitted winner.

This produces one winner, but not a stable or matching owner. In a matched row-pattern run:

```text
Python owner: L1c00E4
NEST owners across seven active windows: E7, E4, E4, E7, E6, E6, E7
NEST distinct owners: E4, E6, E7
```

The existing `pair.json` reports multiplicity but omits NEST owner identity and stability,
so it labels the runs "matched" without testing the representational outcome.

**Required direction:** the reference CIPP profile must use uniform conduction delay within
an equivalent projection. Geometry-dependent delay and biological jitter may remain as
explicit ablations, never as the parity default or the mechanism that makes WTA pass.

### P0. Hard single-winner WTA is not implemented under the default native semantics

At the prototype defaults, several ordinary E cells fire before the E-to-I-to-E loop can
reset them. Eor consequently receives and sometimes relays multiple winners. Shrinking `h`
and adding jitter makes the race easier to separate, but does not establish the intended
weight-driven competition.

NEST's minimum-delayed communication means an inhibitory event cannot retract a spike that
has already been emitted. A valid continuous-time implementation must therefore establish
an operating envelope:

```text
runner-up crossing time - winner crossing time > E->I->E loop latency
```

When that inequality holds, the local circuit must yield one committed winner. Exact ties
must be reported as ties or resolved by a separately declared local arbitration mechanism;
they must not be silently converted into a biological claim.

### P0. `I_accq` is wrong for ordinary-E learning

The Python event path captures `iaccq` in `freeze_drive()` as the current boundary's
delivered packet. Retained membrane voltage may include earlier boundaries, but it is not
the FE input.

The NEST accumulator sets `q_pre=q` at firing, which is the entire retained charge since
the last reset. A direct probe gives:

```text
three successive 400-charge volleys, theta=1000
Python at firing: V_pre=1200, I_accq=400
NEST at firing:   q_pre=1200
```

This changes FE and therefore every learning step for a cell that crosses after accumulating
more than one presentation. Existing NEST tests only cover a one-event overshoot and miss
this case.

**Required fix:** store two different quantities: persistent membrane/charge state and
`q_causal_volley`. Snapshot the latter as `I_accq` when the cell fires.

### P0. Confirmation feedback is time-triggered state erasure, not event-conserving prediction

The Python feedback workaround lands after the successor boundary's packet is frozen and
deletes that packet. Current NEST reset events can clear only charge that has already
arrived; later events in the same volley still accumulate. `t_ref_reset` can suppress a
fixed-duration remainder, but that still makes one prediction's meaning depend on a timer,
arrival spread, and presentation interval.

Neither behavior expresses the stronger CIPP contract: one prediction should suppress the
next eligible local evidence event, even if input is late, interrupted, or irregular.

**Required fix:** represent prediction as a counted local credit. A C confirmation adds one
credit to its column; the next qualifying feedforward volley consumes exactly one credit
and is suppressed as a whole. Silence must not consume it. WTA resets and prediction
credits must remain distinct ports and state transitions.

### P0. Cell physiology is coupled to the experiment controller

`tau_basal`, `tau_apical`, `tau_deposit_lock`, and I's `t_lockout` are all set to the
presentation interval `D`. Changing input pacing therefore changes dendritic coincidence,
deposit admissibility, and relay physiology. A TTL equal to `D` can also pair an apical from
one presentation with a basal from an adjacent presentation at the inclusive boundary.

`tau_volley` is a single global value derived from the RGC projection and reused for L2's
different feedforward projection.

**Required fix:** define physical, local windows independently of stimulus pacing. Derive
projection-specific delivery envelopes from actual delays. Reject schedules whose volleys
overlap when the selected compatibility profile requires non-overlap.

### P0. C maturity and feedback controls are misinterpreted

The NEST report says a C initialized at `theta/4` can never fire under frozen learning.
That confuses "not one-shot" with "cannot accumulate." Both the Python and NEST C somata
retain valid deposits.

Measured during this audit for the center C with frozen `theta/4` basal weight:

```text
8 presentations:   2 deposits, 0 spikes, q=500
20 presentations:  5 deposits, 1 spike,  q=250
100 presentations: 26 deposits, 6 spikes, q=500
```

The mature-weight override is useful for isolating immediate one-shot feedback, but it is
not required to make the feedback path fire eventually. Short feedback-on/off tests are
therefore readiness-confounded, not general feedback controls.

**Required fix:** test immature accumulation, mature one-shot behavior, and feedback effect
as three separate conditions. State readiness in every control result.

### P1. The learning port is not proven wrong for the reason currently documented, but is not accepted

The report classifies lazy post updates as an automatic semantic failure because NEST
materializes them when that synapse's next presynaptic spike is processed. That conclusion
is too broad for the active dual FE/FES equation:

- each synapse's update depends on its own previous weight;
- shared FE can be read from historical post state;
- updates for one synapse do not depend on another synapse's updated weight;
- NEST catches a synapse up before it delivers its next event.

Under those conditions, deferred materialization can be behaviorally equivalent. But the
current implementation still lacks the tests needed to prove it:

- multiple postsynaptic spikes between two presynaptic spikes;
- a permanently silent afferent and an end-of-run logical-weight flush;
- a late same-volley afferent whose event arrives after the post spike;
- both lower and upper participation-window bounds;
- exact equivalence of delivered weights, not only eventual scalar arithmetic.

The current participation state stores only `participate_until`. It is armed when the
presynaptic event is sent, before target delivery, and has no lower-bound check. A late
afferent can therefore be marked participating even when its event did not arrive before
the target fired.

**Required fix:** either implement true post-triggered incoming-afferent updates in a custom
NEST extension, or retain lazy materialization only after a proof suite establishes logical
equivalence. Track an interval with both `participate_from` and `participate_until`, in the
connection's delay-adjusted time coordinate, and provide a non-perturbing final flush.

### P1. Delay quantization does not implement its documented rule

`Timescales.quantize()` says it rounds a delay upward, but uses Python `round()`. It may
round down, shortening a causal delay. Use `ceil(delay/h)` with an explicit floating-point
tolerance and test values immediately above and below half-grid points.

### P1. Feedback latency is averaged over paths

`feedback_loop_latency_ms()` sums projection mean delays. With geometry and jitter enabled,
there is no single mean causal path: a particular winner and afferent traverse particular
edges. A mean must not schedule or validate a sharp causal event.

Counted suppression removes the need for cadence scheduling by this mean. Diagnostics should
report per-path latency distributions and the actual causal path for each confirmation.

### P1. Learning artifacts and dashboard metadata are stale or contradictory

`dashboard_topology()` always fills displayed weights from `self.weights`, the construction-
time dictionary. It does not call `plastic_weights()`. A post-training demo can therefore
claim to carry post-training weights while its header contains initial weights.

This is present in the latest recorded saturation artifact: `L1c00_eor_c` is 250 in the
replay header, while the adjacent report records its final weight as 1000. The same replay
labels both `conditions.learning` and `topology.nest.learning` as `frozen`.

The same topology payload labels learning as frozen and lists "no learning occurs" even
when `learning=True`. `run_case()` also does not attach or collect weight changes for its
learning mode.

**Required fix:** take one materialized live-weight snapshot at artifact boundaries, label
logical versus materialized weight state, and make manifest/dashboard claims conditional on
the actual mode.

### P1. The four-pattern convergence experiment accepts collisions as success

`Convergence.check()` requires a stable recent owner per driven column and mature C weights.
It does not require the new pattern to recruit a different owner or require a one-to-one
pattern-to-owner map across phases.

Recorded runs show collisions, for example `row 1` and `diag \\` both assigned to
`L1c00E4`, while later phases are still marked converged. C maturity is also column-global,
not pattern-specific in classic `tiled_cc`, so it cannot prove four distinct learned
associations.

**Required fix:** separate mechanical convergence from scientific consolidation. Report
the complete pattern-to-owner confusion matrix, unique-owner count, collisions, frozen
cold recall, turnover, and forgetting. Do not fail the simulator if CIPP itself fails this
scientific experiment.

### P2. Scope is narrower than the current CIPP architecture

The NEST translator supports only classic `tiled_cc`. It does not cover direct-identity
multi-basal C, two-tower composition, or other current topology families. Classic Eor
intentionally erases local winner identity, so this backend cannot validate the richer
source-address semantics demonstrated by direct identity.

This is acceptable for a staged port, but it must remain an explicit scope limitation.

## 4. Semantic disposition

| Area | Disposition |
|---|---|
| canonical topology and ID mapping | retain |
| fixed Eor relay at `theta` | retain for classic profile |
| separate basal/apical/confirmation/reset ports | retain |
| native NEST clock, delays, recorders, threading | retain |
| event accumulator as the production competitor | replace |
| geometric delay dispersion as parity behavior | remove from reference profile; keep ablation |
| random delay jitter as WTA fix | remove from reference profile; keep ablation |
| `D`-sized neural TTLs | replace |
| reset-at-time confirmation semantics | replace with counted next-evidence suppression |
| current `q_pre` as ordinary-E `I_accq` | replace with causal-volley snapshot |
| lazy synaptic materialization | conditionally retain only after proof and flush support |
| current NEST report's C-readiness claim | correct |
| current impulse prototype | preserve under an explicit non-production name |

## 5. External NEST constraints relevant to the repair

- NEST distinguishes neuron update resolution `h` from minimum-delay communication and
  supports precise-spike models with offset timestamps. Precise timing does not create
  zero-delay synapses, but it can restore the membrane-latency race that the current delta
  accumulator discards: <https://nest-simulator.readthedocs.io/en/stable/neurons/simulations_with_precise_spike_times.html>.
- `iaf_psc_delta_ps`, `iaf_psc_alpha_ps`, and `iaf_psc_exp_ps` are available precise-time
  reference models. The alpha/exp models provide a continuous postsynaptic current rather
  than instantaneous total-charge thresholding: <https://nest-simulator.readthedocs.io/en/stable/neurons/neuron_types.html>.
- NESTML supports paired neuron/synapse generation, postsynaptic spike handlers, continuous
  post-state ports, handler priorities, and volume-transmitter ports. Generated synapses are
  normally advanced on presynaptic events, which is why lazy-materialization behavior must
  be proved rather than guessed: <https://nestml.readthedocs.io/en/latest/running/running_nest.html>
  and <https://nestml.readthedocs.io/en/latest/nestml_language/synapses_in_nestml.html>.
- NEST's own e-prop example explicitly forces presynaptic activity to update otherwise
  unvisited synapses before final weight readout. That is precedent for requiring an
  explicit flush protocol, not evidence that stale reads are harmless:
  <https://nest-simulator.readthedocs.io/en/v3.10/auto_examples/eprop_plasticity/eprop_supervised_regression_sine-waves.html>.

## 6. Recommended decision

Do not "fix" the current prototype by finding a more favorable combination of `h`, `S`,
`D`, and jitter. That would optimize a different decision rule.

Keep the current backend for reproducibility and create two named profiles:

1. `impulse_characterization`: current event-accumulator behavior, known differences
   preserved and reported.
2. `cipp_continuous`: uniform reference conduction, continuous postsynaptic dynamics,
   physical coincidence windows, causal-volley learning state, and counted prediction
   credits.

Promotion of `cipp_continuous` must require the mechanical acceptance suite in the companion
Claude specification. Scientific experiments such as four-pattern consolidation must be
reported honestly but must remain separate from engine-correctness gates.
