# Claude resume prompt: NEST CIPP Phase 3

Resume the NEST CIPP semantic repair in:

```text
/home/adasgup/Documents/SNN
```

You are working directly for the user. Do not invoke Codex or another agent.

## Read before editing

Read these files completely:

1. `docs/CIPP_PHASE2_CODEX_AUDIT_HANDOFF.md`
2. `prompts/Claude_NEST_CIPP_Semantic_Repair_Prompt.md`
3. `docs/CIPP_CUSTOM_ENGINE_TO_NEST_SEMANTIC_AUDIT.md`
4. `docs/CIPP_SEMANTIC_MATRIX.csv`
5. `docs/CIPP_PHASE1_FEASIBILITY_REPORT.md`

Do not follow `prompts/CURRENT_WORK_PROMPT.md`; it describes unrelated custom-engine
frequency-scaling work.

## Repository checkpoint

- Branch: `feature/NEST`
- Committed checkpoint: `676fb6a` (`Complete and audit NEST CIPP Phase 2 scaffold`)
- `origin/feature/NEST` points to the same commit.
- Phase 2 was independently accepted as an honest scaffold.
- `cipp_continuous` still uses `event_accumulator`; the continuous competitor is not
  implemented.
- `impulse_characterization` remains the default and must retain its exact historical
  behavior.
- Phase 2 focused tests currently pass: `98 passed`.
- The worktree contains an uncommitted NEST observability patch that repairs owner reporting
  and partially repairs live/replay weight export.
- Its focused tests currently report `49 passed, 8 expected impulse-profile xfails`.
- Inspect and preserve that patch. Do not restart or overwrite it.

The eight expected impulse-profile xfails cover:

- causal-volley `I_accq` at firing;
- late-afferent participation bounds;
- silent-afferent logical-weight flushing;
- counted prediction credits across irregular silence (three cases);
- historical impulse-profile pacing dependence;
- historical impulse-profile shortening quantization.

The final two are intentionally preserved historical defects. Separate continuous-profile
contracts already cover the repaired Phase 2 behavior.

## Repository separation

Custom-engine dashboard/demo work has been moved to a separate worktree at:

```text
/home/adasgup/Documents/SNN-custom-engine
```

Do not touch that worktree, the legacy `../cipp-learning` clone, or either GitLab
publication repository. The current `SNN` worktree is reserved for the NEST repair.

The publication mapping is:

```text
feature/NEST                    -> nest-gitlab/main
feature/tiled-cortical-columns -> custom-gitlab/main
```

Do not merge, commit, or publish either branch during this task.

First inspect the complete Git diff and classify every existing NEST change. Verify the
partial patch before extending it. Do not use `reset`, `checkout`, `clean`, or other
destructive Git operations.

## Implement Phase 3 only

Follow the semantic-repair specification and:

- replace the `cipp_continuous` ordinary-competitor scaffold with a genuine continuous
  membrane-latency competitor;
- retain `impulse_characterization` byte-for-byte, including its default selection,
  historical xfails, graph fingerprint, delay behavior, and negative findings;
- distinguish retained membrane/current state from current causal-volley charge;
- export causal-volley `I_accq` at firing;
- provide distinct `feedforward_excitation`, `wta_reset`, and `prediction_credit`
  mechanisms or ports;
- keep classic Eor fixed and non-plastic;
- measure winner crossing, runner-up counterfactual crossing, E-to-I-to-E reset latency,
  and WTA margin;
- report nonpositive-margin cases as unresolved races, never implicit GID/order winners;
- complete the parent processing-latency contribution to the causal coincidence envelope;
- do not implement timer-based prediction as a substitute for Phase 4 credits;
- do not invent a mapping between repository charge units and the provisional mV/pF
  profile; derive and test a defensible mapping, or stop with the smallest reproducer and
  a bounded feasibility blocker;
- do not alter `backend/` or `snn/` to make NEST agree;
- do not use Python per-spike callbacks.

Add focused failing-before/passing-after tests. Keep mechanical correctness, Python
compatibility, scientific outcomes, and accepted native NEST differences separate.

## Required verification

Run at minimum:

```bash
NEST_TESTS_REQUIRED=1 .nest-env/bin/python -m pytest \
  tests/nest/test_cipp_profiles.py \
  tests/nest/test_cipp_continuous_live.py -q

NEST_TESTS_REQUIRED=1 .nest-env/bin/python -m pytest \
  tests/nest/test_cipp_semantic_probes.py \
  tests/nest/test_nest_replay_adapter.py -q -rxX

git diff --check
```

Also run every focused Phase 3 test you add.

## Restrictions

Do not commit, push, publish, install dependencies, delete files, or touch unrelated files.

Do not claim `cipp_continuous` is promoted or that the complete mechanical gate passes
unless every required acceptance gate has actually been demonstrated.

## Final response

Report:

1. exact implementation status;
2. files changed;
3. exact commands and results;
4. remaining expected xfails;
5. any bounded feasibility blocker;
6. a concise handoff for Phase 4.
