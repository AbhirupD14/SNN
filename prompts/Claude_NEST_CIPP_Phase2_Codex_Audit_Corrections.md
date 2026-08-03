# Phase 2 corrections from the independent Codex audit

Implement these corrections. Do not audit, approve, grade, or close the phase; Codex owns
that judgment and will independently inspect and test the result.

Do not commit, push, install dependencies, delete material data, contact external systems,
or modify unrelated files. Preserve the impulse profile and its exact fingerprint.

## Reproduced defect 1: pacing still reaches continuous synapse physiology

`tests/nest/test_cipp_continuous_live.py::test_changing_presentation_pacing...` compares
only `_model_for()` neuron dictionaries, despite claiming neuron **and synapse** physics.
The live continuous learning path computes:

```text
self.tau_volley_ms = min(spread * 1.5 + self.ts.base_ff,
                         self.ts.presentation * 0.5)
```

and `_syn_spec()` passes that value as plastic `tau_volley`. Independently reproduced:

```text
presentation 3.0 ms -> tau_volley 1.5 ms
presentation 5.0 ms -> tau_volley 2.5 ms
```

This violates the Phase 2 contract even though the chosen 10/40 ms test pair happens to
sit on the same side of the clamp.

Required:

- On the continuous branch, source learning membership exclusively from
  `profile.causal_volley.separation_ms`; never from `Timescales.presentation`, `spread`, or
  `base_ff`.
- Compare every actual node model dictionary **and every `_syn_spec()` dictionary** across
  continuous networks with different valid schedules and different legacy
  `Timescales.spread/base_ff` values. Normalize only repository identity fields if needed;
  do not omit scientific synapse parameters.
- Do not apply impulse-only `Timescales` separation constraints (`D > S`, geometric spread,
  base hop) as continuous physiology. Validate a continuous schedule against the profile's
  windows/separation without deriving or mutating those values.
- Invoke schedule validation on the real run path, including irregular scheduled times;
  do not leave it as a helper tests must remember to call.

Keep impulse construction byte-identical.

## Reproduced defect 2: parameters have multiple semantic roles

`CoincidencePolicy.deposit_dead_time_ms` is defined as the C soma's deposit dead time, but
`_model_for_continuous()` also uses it as the inhibitory relay's `t_lockout`. Likewise the
ordinary-competitor `MembranePolicy.t_ref_ms` is reused as C-cell refractory duration even
though `MembranePolicy` declares continuous competitor physiology.

Required:

- Define a separate immutable relay policy/value for WTA-relay lockout.
- Define a separate C firing refractory value, or explicitly use the C contract's justified
  zero refractory; do not borrow the competitor refractory.
- Give each field a unit, one role, validation, manifest representation, and a provisional
  marker/rationale where it is not a measured CIPP constant.
- Mark the provisional status/provenance of the membrane and causal-volley defaults, not
  only the coincidence window.

## Reproduced defect 3: the claimed path walker does not walk paths

`causal_arrival_envelope()` groups every edge by projection, takes projection-wide extrema,
and sums those extrema. It never verifies connected source/target sequences and records no
traversed edge IDs. Its test contains the vacuous assertion:

```python
assert "sum of projection means" not in envelope["method"] or True
```

The calculation also omits the parent competitor's nonzero membrane crossing latency. It
therefore measures a conduction-only projection envelope, not complete basal/apical arrival
skew. The reported 1.0 ms must not be presented as a measured end-to-end causal skew.

Required:

- Enumerate connected basal and apical paths from `NetworkSpec` identities and edge
  endpoints. Record concrete node IDs, edge IDs, per-edge delays, and path delay for each
  eligible C path.
- Derive the conduction envelope from those concrete paths, never from independently
  combinable projection extrema.
- Label it `conduction_only` and explicitly record that parent membrane/processing latency
  is unverified until the Phase 3 continuous model and drive envelope exist.
- Do not validate a full coincidence window against conduction alone. Either include a
  justified measured processing-latency envelope or report the full window validation as
  provisional/deferred to Phase 3.
- Replace the vacuous test with assertions over actual connected edge sequences and the
  explicit deferred-processing status.

Do not invent a parent latency from Phase 1's pA examples: those weights were explicitly
not mapped to the repository's charge units.

## Reproduced defect 4: target membrane profile is advertised as active physics

The manifest serializes `profile.membrane.model = iaf_psc_exp_ps` with mV/pF parameters,
while live ordinary competitors are:

```text
event_accumulator, theta = 1000 charge units
```

Only `t_ref_ms` is borrowed. Gate 2's claim that the live network consumes the profile
exclusively is therefore false. The Phase 2/3 boundary is legitimate; the claim is not.

Required without silently implementing Phase 3:

- Add an immutable, serialized implementation-status/disposition block that distinguishes
  target profile parameters from active node/synapse models.
- Record at least target model, active competitor model, active units, implemented Phase 2
  components, deferred components, and `mechanical_profile_promoted = false`.
- Make every manifest, replay provenance record, and dashboard known-differences block
  state that this is a Phase 2 scaffold using the impulse accumulator, not yet a runnable
  continuous CIPP engine.
- Keep `impulse_characterization` the default.
- Add tests that fail if the target membrane model is reported as active or if an artifact
  implies the full mechanical profile has passed.

Do not rename the target configuration away from `cipp_continuous`; make its implementation
status honest.

## Reproduced defect 5: replay does not carry the complete profile

`replay_adapter.provenance()` carries only `engine_profile`. The Phase 2 test named
`test_replay_provenance_carries_the_profile` asserts only that string.

Required:

- Carry the complete unit-bearing target profile, implementation disposition, accepted
  differences, and conduction/deferred-processing envelope into replay provenance.
- Strengthen the test to compare those payloads with the source manifest, not just the
  profile name.

## Preserve findings that did pass

Codex independently ran 40,140 adversarial quantization cases across four resolutions;
non-shortening, minimum delay, idempotence, and boundary behavior passed. Preserve that
implementation and add exact (not `- 1e-12`) non-shortening assertions where applicable.

Also preserve:

- uniform continuous delay assignment by projection;
- jitter rejection and geometry-independent delivery;
- declared unresolved exact-tie policy with implicit GID arbitration forbidden;
- impulse Contracts 8/9 as historical xfails;
- impulse 254 edges, delay sum 368.9, spec hash `ee78be7bc4aa80a1`.

## Deliverable

Implement the corrections, run focused Phase 2 suites and the full NEST suite, and report
the exact commands/results plus files changed. Do not declare Phase 2 closed; return the
work to Codex for independent audit.

## Codex audit addendum after the first correction pass

Focused tests pass, but Codex independently reproduced four remaining defects. Correct
only these items and their focused regression tests; do not broaden into Phase 3.

1. `EngineProfile.validate()` does not structurally validate `CoincidencePolicy` unless a
   causal skew is supplied. A profile with `c_refractory_ms = NaN` currently passes. Make
   every coincidence field (including `c_refractory_ms` and the dead-time/window relation)
   validate on the bare profile path, without pretending the deferred full causal-window
   check has run.
2. The impulse profile disposition says its target/active model is `event_accumulator`, but
   `profile.describe()["membrane"]["model"]` still says `iaf_psc_exp_ps`. Remove this
   contradiction honestly. Prefer representing the continuous membrane group as not
   applicable on the impulse profile rather than attaching meaningless mV/pF values to an
   accumulator. Validate consistency between a continuous scaffold's membrane target and
   its disposition.
3. `set_input_schedule()` mutates every NEST generator before it validates the schedule.
   If validation raises, the rejected spike times remain installed. Compute/snap/validate
   first, then update generators atomically. Add a regression that installs a valid
   schedule, rejects a bad one, and proves the original generator schedules remain.
4. Remove the new vacuous assertion in
   `test_relay_policy_is_its_own_group_with_its_own_validation` (`... or True`). Test the
   structural separation/non-aliasing of the two policy fields directly; equal numeric
   defaults are allowed and do not establish shared semantics.

Run only the two focused Phase 2 test files. Return a concise implementation summary and
test result. You remain implementation-only: do not audit, approve, close, commit, push,
install, delete, or touch unrelated files.
