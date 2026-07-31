# CIPP Phase 2 Codex audit handoff

**Recorded:** 2026-07-31  
**Source branch:** `feature/NEST`  
**Status:** implementation checkpoint only; **not yet independently audited or accepted**

## Resume instruction

Use `$orchestrate-claude-opus` to resume this task. The primary Codex agent owns the audit;
Claude may implement corrections but must not audit, approve, or close the phase.

The user explicitly authorized `--dangerously-skip-permissions` for the resumed Claude run.
Treat this as a task-specific exception, not a global skill default. Before invoking it,
confirm the previous interactive Claude session remains closed, keep the repository scope
explicit, prohibit commits/pushes/installs/destructive operations in Claude's prompt, and
retain sequential worktree ownership. Invoke Opus 5 at high effort and verify the resolved
model from structured metadata.

## Claude's latest claim (unverified)

Claude reported the Phase 2 nine-item completion gate as passing:

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

None of the claims above are Codex findings yet.

## Required independent Codex audit

Start from `prompts/Claude_NEST_CIPP_Phase2_Completion_Review.md` and personally verify each
gate against source and executable behavior. At minimum:

- inspect profile selection for per-instance state and absence of module-global leakage;
- inspect every continuous neuron and synapse parameter dictionary for pacing-derived
  values;
- adversarially test quantization at `nextafter()` boundaries, multiple resolutions,
  monotonicity, grid membership, minimum delay, and idempotence;
- confirm the causal-window calculation follows actual translated paths rather than a sum
  of projection means or two terminal edges;
- verify the declared exact-tie policy reaches manifests and cannot fall through to NEST
  GID, creation, or connection order;
- compare the actual impulse topology/artifacts against the preserved fingerprint;
- inspect the assertions in Claude's new tests rather than accepting their pass count;
- run focused suites, then the complete mechanical NEST suite;
- decide whether leaving `event_accumulator` live is a legitimate Phase 2 boundary or
  contradicts the claim that the live network consumes the membrane profile exclusively.

If defects are found, resume Claude for bounded implementation corrections, then repeat the
Codex audit. Only Codex may close Phase 2.

## Publication state

The private `SNN` checkpoint should be pushed to `origin/feature/NEST` as unaudited WIP.

Shared GitLab publication is pending the Codex audit. At handoff time,
`../cipp-learning` was dirty and on `AbhiCIPP-EngineRework`, while the publication skill
expects a clean approved target (normally `AbhiCIPP`). Do not discard those shared-clone
changes or merge around them. After audit approval, resolve the intended target branch with
the user, make the clone clean without losing its current work, integrate the reviewed SNN
commit, verify, and push to
`git@faraday.lps.umd.edu:cipp/cipp-learning.git` without force.
