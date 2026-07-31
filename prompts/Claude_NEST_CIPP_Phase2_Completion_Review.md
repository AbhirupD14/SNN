# Claude implementation brief: complete and harden CIPP Phase 2

## Review verdict

The profile split is approved:

- `cipp_continuous` uses causal ceiling-to-grid quantization;
- `impulse_characterization` deliberately retains historical `round()` behavior so its
  recorded negative findings and golden artifacts remain reproducible.

Do **not** change the historical profile or regenerate its goldens. Its rounding behavior
must remain named as a defect and must never be presented as CIPP-correct behavior.

The current `nest_backend/profiles.py` is a good Phase 2 core, but Phase 2 is not closed
until the live `NestTiledNetwork` consumes it and the issues below are resolved. Do not
begin Phase 3 implementation on top of two competing parameter paths.

## Required correction 1: make causal ceiling true at float boundaries

`QuantizationPolicy.apply(CEIL, ...)` currently subtracts a fixed `1e-9` tolerance in
tick units and rounds the returned milliseconds to ten decimal places. It therefore still
shortens requests immediately above a grid point. Reproduced examples at `h = 0.1`:

```text
0.10000000000001 -> 0.1
0.10000000001    -> 0.1
0.1000000001     -> 0.1
1.00000000001    -> 1.0
```

Replace the fixed quotient tolerance with an integer-tick calculation whose returned
floating value satisfies the contract for the actual input float:

```text
result >= requested_delay
result >= h
result is the smallest representable integer multiple k*h satisfying both inequalities
```

Do not round the final delay to a decimal-place count if that can move it below the
request. Reject non-finite or negative delays explicitly. It is acceptable to correct a
provisional `ceil(delay/h)` tick count by comparing `(k-1)*h` and `k*h` directly; this also
handles values such as `3*h == 0.30000000000000004` without spuriously selecting tick 4.

Add regression/property coverage for:

- `math.nextafter(k*h, -inf)`, `k*h`, and `math.nextafter(k*h, +inf)`;
- values immediately above and below half-grid points;
- several resolutions and tick magnitudes;
- exact non-shortening, grid membership, minimum-one-tick, monotonicity, and idempotence;
- unchanged `ROUND` results in `impulse_characterization`.

The contract may acknowledge ordinary floating representation, but it may not call a
strictly smaller returned float "never shorter."

## Required correction 2: make the profile the live source of physics

Add an explicit profile selection to `NestTiledNetwork`; keep
`impulse_characterization` as the default until the complete mechanical gate passes.
Avoid a module-global profile value as the source of truth: two network instances must be
able to declare different profiles without hidden shared state.

For `cipp_continuous`:

- set kernel resolution from `profile.resolution_h_ms`;
- create competitors from `profile.membrane`;
- create C-cell windows/dead time from `profile.coincidence`;
- select every connection delay by projection class from `profile.delays`, then apply the
  profile's quantizer;
- keep equivalent edges in a projection uniform;
- use geometry only for the learning multiplier `phi`;
- reject nonzero jitter unless a separately named ablation profile was selected;
- serialize the complete `profile.describe()` payload into the manifest/replay provenance.

The experiment controller may retain a presentation schedule, but no value derived from
that schedule may enter a neuron or synapse physiology dictionary. In particular, the
current live path at `nest_backend/topology.py` that assigns `window =
self.ts.presentation` must remain only in the historical impulse branch.

Acceptance tests must construct two `cipp_continuous` networks that differ only in
presentation period and compare the actual NEST model dictionaries for every neuron role.
Their membrane, coincidence, relay, refractory, and synapse-physiology values must be
identical. Keep the existing impulse Contract 8 `xfail` as historical characterization;
add a passing continuous-profile contract rather than repointing that test.

## Required correction 3: validate the causal paths, not two terminal edges

`CoincidencePolicy.validate()` currently estimates arrival skew as:

```text
abs(column_to_column_apical_ms - column_eor_to_c_basal_ms)
```

That is not the graph's causal skew. Basal and apical evidence traverse different paths,
including local E-to-Eor, feedforward, parent integration, and return-apical components.
Do not claim the current subtraction validates the graph.

At topology/profile integration, derive or measure the relevant earliest/latest causal
arrival paths from translated `NetworkSpec` edges. Pass the resulting skew/envelope into a
window validator. Record the traversed delays or distribution in the manifest; do not use
a sum of projection means. The physical windows must admit intended same-event
coincidences while excluding cross-volley pairing for the compatibility schedule.

Similarly, validate `CausalVolleyPolicy.separation_ms` at run construction using measured
within-volley arrival spread and the experiment's minimum inter-volley interval:

```text
within_volley_spread < separation_ms < minimum_inter_volley_interval
```

This runtime check consumes schedule information for validation only; it must not mutate
or derive neuron physiology from the schedule.

## Required correction 4: declare exact-tie semantics in the profile

The original custom engine deliberately arbitrates crossings within `1e-12` normalized
outer-boundary time using stable repository node order and records `latency_ties`. The
Phase 1 built-in NEST microcircuit has no such arbiter and correctly reports an exact tie
as unresolved.

Add an explicit WTA policy group to the profile/manifest. Until a local arbiter is
implemented, `cipp_continuous` must declare:

```text
exact_tie_policy = unresolved
implicit_gid_tie_break_allowed = false
```

Do not translate `1e-12` normalized time into milliseconds without a declared mapping.
If Phase 3 implements custom-engine-compatible arbitration, specify its physical
tolerance, stable ordering key, and observable tie record there. NEST GID, creation order,
connection insertion order, geometry, and jitter may never be implicit tie-breakers.

## Required correction 5: strengthen profile validation and parameter provenance

`EngineProfile.validate()` must cover every projection delay, not only E-to-I and I-to-E.
Validate finite values and NEST's `delay >= h` requirement for all delay fields, nonnegative
jitter, valid quantization policy, membrane ordering/time constants, all coincidence
windows, and causal-volley separation.

Define prediction-credit overflow behavior. If credits saturate at `capacity`, say so and
test it; otherwise reject `credits_per_confirmation` or `consumed_per_volley` settings that
cannot be represented coherently.

The current values `t_ref = 2 ms`, basal/apical windows `5 ms`, deposit dead time `1 ms`,
and causal-volley separation `0.5 ms` are provisional physical choices. Give each a
measured constraint or compatibility rationale. Do not let Phase 1's convenient model
defaults silently become CIPP physiology.

## Phase 2 completion gate

Phase 2 closes only when all of the following hold:

1. the strengthened pure-profile tests pass;
2. the live continuous network consumes the profile exclusively;
3. changing presentation pacing leaves actual continuous-profile neuron/synapse physics
   byte-equivalent;
4. continuous delays are uniform by projection, jitter-free, geometry-independent, and
   causally ceiling-quantized;
5. float-boundary quantization tests pass;
6. actual causal-path window validation passes;
7. exact-tie behavior is declared and cannot fall through to GID order;
8. manifests contain the complete unit-bearing profile and accepted differences;
9. the historical impulse graph remains unchanged: 254 edges, delay sum 368.9, identical
   `spec_hash`, existing goldens, and existing semantic-probe outcomes.

Only after this gate should Contract 8 and Contract 9 be described as repaired for
`cipp_continuous`. Their impulse-profile `xfail` probes remain intentionally present and
must continue to characterize the preserved defects.
