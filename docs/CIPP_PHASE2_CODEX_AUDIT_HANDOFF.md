# CIPP Phase 2 Codex audit handoff

**Recorded:** 2026-08-03
**Source branch:** `feature/NEST`
**Status:** **independently audited and accepted by Codex as a Phase 2 scaffold**

## Resume instruction

Use `$orchestrate-claude-opus` to resume this task. The primary Codex agent owns the audit;
Claude may implement corrections but must not audit, approve, or close the phase.

The user explicitly authorized `--dangerously-skip-permissions` for the resumed Claude run.
Treat this as a task-specific exception, not a global skill default. Before invoking it,
confirm the previous interactive Claude session remains closed, keep the repository scope
explicit, prohibit commits/pushes/installs/destructive operations in Claude's prompt, and
retain sequential worktree ownership. Invoke Opus 5 at high effort and verify the resolved
model from structured metadata.

## Audit outcome

Codex inspected the implementation and assertions, reproduced defects independently, sent
bounded corrections to Claude Opus 5/high effort, then audited the resulting diff. Phase 2
is accepted with the scope boundary below: it defines and wires an honest physical-time
profile scaffold; it does **not** claim that the Phase 3 continuous competitor exists.

The accepted nine-item gate is:

1. strengthened pure-profile suite: 40 tests;
2. live network consumes its selected profile exclusively;
3. presentation pacing leaves continuous-profile model dictionaries byte-equivalent;
4. continuous delays are uniform, jitter-free, geometry-independent, and ceiling-quantized;
5. float-boundary quantization sweep has no failures;
6. measured causal-path coincidence skew is 1.0 ms and validates;
7. exact-tie behavior is declared with no implicit GID fallthrough;
8. manifests contain the unit-bearing profile and causal envelope;
9. impulse graph remains 254 edges, delay sum 368.9, spec hash
   `ee78be7bc4aa80a1`.

Claude also reported that impulse Contracts 8 and 9 intentionally remain `xfail`, while
separate continuous-profile contracts pass and the semantic matrix records the split.

Scope boundary reported by Claude: the live continuous competitor still uses
`event_accumulator`, not `profile.membrane.model`. Claude classified that replacement as
Phase 3 because the selected continuous implementation still needs causal-volley state and
separate `wta_reset` / `prediction_credit` ports. It also reported that
`reset_suppression_ms` is ignored in the continuous profile because timer-based prediction
suppression is to be replaced by credits.

### Defects found and corrected during Codex audit

- plastic `tau_volley` still depended on legacy `Timescales.spread`, `base_ff`, and
  presentation pacing; it now comes from `CausalVolleyPolicy.separation_ms`;
- relay lockout and C refractory had borrowed parameters owned by other mechanisms; both
  now have one-role profile fields;
- the causal envelope combined projection-wide extrema and could mismatch evidence from
  different C cells; it now walks connected endpoints and pairs basal/apical paths by the
  same target C and causal prefix;
- artifacts implied the profile's `iaf_psc_exp_ps` target was active, although the Phase 2
  scaffold still runs `event_accumulator`; manifests, dashboard payloads, and replay
  provenance now carry an explicit target-versus-active disposition;
- the impulse profile inherited a continuous membrane description it does not use; that
  group is now explicitly not applicable;
- bare profile validation skipped coincidence structure, including non-finite C refractory;
- `set_input_schedule()` installed rejected spikes before validating; validation is now
  atomic and precedes generator mutation;
- one new test contained a vacuous `or True`; it was replaced with structural ownership
  assertions.

### Independent evidence

- focused Phase 2: `98 passed`;
- adversarial Codex probe: 16,000 additional quantization boundary/random cases passed;
- causal path probe: 1,152 matched pairs, zero unmatched basal paths;
- NEST semantic/integration files outside the HTTP live-server fixture: `238 passed, 11
  xfailed`; the xfails are the preserved impulse-profile defects;
- remaining node-contract/replay/topology group: `78 passed`;
- main Python suite: `653 passed`;
- JavaScript: all five test files passed;
- impulse fingerprint: 254 edges, delay sum 368.9, spec hash `ee78be7bc4aa80a1`.

Ten live-server HTTP tests could not enter their shared Starlette `TestClient` fixture in
the installed `.nest-env`. A minimal one-route FastAPI application hangs identically and
emits a warning that this deprecated `httpx`/Starlette test-client combination should use
`httpx2`; therefore this is recorded as an environment/test-harness limitation, not a
Phase 2 topology failure. The three live chunk-equivalence tests that do not use
`TestClient` pass.

## Next semantic phase

Phase 3 must replace the ordinary competitor scaffold with the continuous model and add
the missing CIPP state/ports: causal-volley charge at firing, distinct WTA reset and
prediction-credit traffic, and the parent membrane latency needed to complete the causal
window envelope. The profile's provisional mV/pF values still lack a mapping to repository
charge units and must not be promoted as CIPP constants.

## Publication state

The accepted Phase 2 commit should be pushed to private `origin/feature/NEST`, then
published through `$publish-abhi-cipp`. The existing `../cipp-learning` checkout was dirty
and on `AbhiCIPP-EngineRework` during the previous checkpoint; do not discard or overwrite
that work. Use a clean publication worktree/checkout for `AbhiCIPP`, verify ancestry and
the integrated diff, and push without force.
