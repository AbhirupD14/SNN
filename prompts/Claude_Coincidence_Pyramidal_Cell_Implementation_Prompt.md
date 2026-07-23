# Claude Kickoff: Coincidence Pyramidal Cell Refactor

You are the primary implementation engineer for a large, timing-sensitive refactor of this SNN repository. Work deliberately and incrementally. Do not perform this as one sweeping rewrite.

## Source of truth

Read the following specification completely before changing code:

- `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md`

Treat it as the implementation contract. Also read the repository workflow and the current code paths named by the specification, including the neuron models, network specification, simulation engine, serializer, layout, API, frontend, and relevant tests.

If the specification conflicts with live code or its equations cannot satisfy a stated invariant, do not silently reinterpret it. Isolate the conflict, show the smallest concrete example, and propose the narrowest correction before continuing past the affected phase.

## Objective

Implement the coincidence pyramidal cell architecture and the event-resolved WTA semantics described in the specification while preserving all legacy presets and their observable behavior.

The final implementation must provide:

- a coincidence pyramidal neuron with one learned basal input and unweighted Boolean apical inputs;
- gated additive charge: basal contribution is admitted only when basal and apical eligibility overlap within the defined boundary;
- the specified basal-only learning rule and bounds;
- analytic intra-boundary threshold-crossing times;
- latency-based excitatory competition;
- zero-latency inhibitory relay and hard reset at the same `(t, tau)`;
- a metadata-driven `rg_coincidence` topology;
- pretrained RG-to-L1E behavior for that topology;
- serialization, API, layout, and frontend support;
- focused tests, regression protection, and scientific measurements.

Do not implement winner selection with forced alternation, preset-specific neuron IDs, global winner reads, or other policy shortcuts. WTA must result from the causal event mechanics.

## Working-tree safety

Begin with `git status --short` and inspect relevant diffs. Preserve all pre-existing user changes and untracked files. In particular, do not overwrite or discard the existing changes to:

- `docs/agent_workflow_evolution.md`
- `docs/COINCIDENCE_PYRAMIDAL_CELL_TECHNICAL_SPEC.md`

Do not use destructive Git commands. Do not commit, push, reset, discard, or broadly reformat work unless the user explicitly authorizes it. Keep unrelated files out of the diff.

## Non-negotiable semantics

Keep these rules visible throughout the implementation:

1. A simulation boundary remains the unit for ordinary synaptic delivery. Normal excitation still has one outer-boundary delay.
2. A boundary may contain analytically computed sub-boundary threshold times `tau`; do not approximate this with arbitrary micro-timesteps.
3. Only inhibitory relay and the resulting hard reset have zero latency within a boundary.
4. Events at the same boundary are processed in increasing `(tau, stable_event_order)` order.
5. An immediate hard reset clears the target's membrane state and remaining drive at that instant. It does not retroactively erase a spike already emitted at an earlier `tau`.
6. A stable-order tie fallback is permitted only inside the specified crossing-time tolerance and must be observable in diagnostics.
7. Legacy WTA behavior remains available for legacy presets. Do not mix the legacy competitor mechanism with event-resolved competition in one execution path.
8. A C cell has exactly one learned basal weight in the initial topology. Apical connections are unweighted activity gates.
9. Basal eligibility lasts only for the specified boundary. Basal and apical arrival order inside that boundary must not change the result.
10. C-cell learning occurs on C firing, updates only the basal weight, and uses:

    ```text
    FE = theta - w_basal
    dw = eta_C * FE * A * (1 - (w_basal / w_max)^2) * s * influence
    ```

    with active non-negative basal signal `s`, Boolean apical gate `A`, the specified distance influence, and post-update weight bounds.
11. C threshold, leak, and refractory behavior match E behavior unless the specification explicitly states otherwise.
12. Inhibition remains a one-spike immediate hard-reset relay for this implementation. Burst inhibition is an ablation, not part of this change.
13. Existing presets, saved-state handling, public payloads, and golden behavior must remain compatible unless the specification explicitly adds an optional field.

## Phase protocol

Work through the phases below in order. Never batch multiple phases and test them only at the end.

At the beginning of each phase:

1. State the phase and its acceptance criteria.
2. Inspect the exact live code involved; do not implement from filenames or assumptions alone.
3. Update a single durable checklist at `docs/COINCIDENCE_IMPLEMENTATION_STATUS.md` with the phase status, relevant decisions, and baseline test result. Keep this file concise.

Within each phase:

1. Add or update focused tests for the behavior being introduced.
2. Implement the smallest coherent change that satisfies those tests.
3. Run the focused tests.
4. Run the complete existing test suite.
5. Run `git diff --check`, compare it with the Phase 0 result, and inspect the phase
   diff for accidental scope expansion. Pre-existing diagnostics are not phase
   failures, but the phase must add none.
6. Update the status checklist with exact commands and results.

Proceed automatically to the next phase only when the current phase's focused tests and full regression suite are green. If a gate fails, diagnose and fix failures caused by the phase before continuing. Stop and report rather than masking a failure with weakened assertions, regenerated goldens, relaxed tolerances, or compatibility hacks.

Do not mark a phase complete merely because the code imports or a narrow test passes.

## Phase 0 — Repository audit and clean baseline

Do not change production code in this phase.

- Read the entire technical specification.
- Trace the current neuron integration, spike scheduling, gate handling, inhibitory delivery, WTA, topology construction, serializer, API, frontend, and test paths.
- Record the actual call graph and identify where legacy behavior must branch from event-resolved behavior.
- Verify every named class/file/function in the specification against live code; adapt names in the implementation plan if the repository differs, without changing semantics.
- Run the full existing test suite before making production changes.
- Record pre-existing failures separately. Do not attribute them to this work or alter unrelated code to hide them.
- Check the analytic membrane and crossing-time equations with small numerical examples, including the timing sanity values in the specification.

Phase 0 acceptance criteria:

- the baseline is documented;
- the implementation map is grounded in live code;
- no unresolved contradiction blocks Phase 1;
- no production code has changed.

## Phase 1 — Shared LIF and exact segment primitives

Introduce the shared conductance/LIF abstraction and exact event-time primitives without adding the new topology.

- Extract or introduce `ConductanceLIFNeuron` only as far as necessary to share E/C membrane mechanics.
- Preserve the legacy E integration contract exactly, including existing drive lifecycle, gate effects, refractory behavior, serialization, and tests.
- Add exact segment advancement and analytic threshold-crossing calculation required by the specification.
- Add a hard-reset primitive that can clear membrane state and remaining current drive, but do not yet change legacy network dispatch.
- Make `delta = 1` segment advancement agree with the prior full-boundary result to tight numerical tolerance.

Focused tests must cover at least:

- segment composition;
- no-crossing and immediate-crossing cases;
- crossing-time accuracy;
- reset state clearing;
- legacy E behavior and conductance/gate behavior.

Phase 1 acceptance criteria:

- the shared primitive is independently tested;
- legacy E output is unchanged;
- the full pre-existing suite remains green.

## Phase 2 — Coincidence dendrite and C-cell local behavior

Implement C-cell mechanics in isolation from the new built-in preset.

- Add the dendritic compartment/state needed to distinguish basal from apical input.
- Add the coincidence pyramidal neuron using the shared LIF implementation.
- Enforce one learned basal weight and unweighted Boolean apical gates.
- Make basal/apical arrival order invariant within a boundary.
- Expire eligibility exactly when specified.
- Implement gated additive charge and the exact basal learning equation.
- Bound weights and handle zero-distance influence safely as specified.
- Keep threshold, leak, and refractory defaults aligned with E.

Focused tests must cover at least:

- basal-only, apical-only, and coincident input;
- both within-boundary arrival orders;
- stale basal eligibility rejection;
- no negative-signal learning event;
- no apical weight mutation;
- learning on C firing only;
- FE using only the basal afferent weight;
- saturation/bounds and distance influence;
- C/E parameter parity.

Phase 2 acceptance criteria:

- C behavior is completely testable without a full topology;
- the learning equation is verified numerically;
- no built-in preset exposes an incomplete execution path;
- the full suite remains green.

## Phase 3 — Graph vocabulary, validation, and construction

Add metadata-driven representation and construction support, but keep the new built-in preset unavailable until its runtime semantics work end to end.

- Add the C archetype and explicit basal/apical edge roles.
- Add event-resolved competition metadata and immediate-reset/relay metadata.
- Add validation for malformed combinations, including missing/multiple basal inputs and illegal mixing of legacy and event-resolved WTA semantics.
- Extend builders, runtime maps, layout, and serializer internals sufficiently to construct small synthetic networks used by scheduler tests.
- Do not key behavior to names such as `L1C`, `L2I`, preset names, or numeric IDs.

Focused tests must cover valid and invalid minimal graphs, round trips, deterministic ordering, and construction of synthetic C/WTA fixtures.

Phase 3 acceptance criteria:

- graph semantics are explicit and validated;
- construction is metadata-driven;
- incomplete `rg_coincidence` functionality is not yet advertised;
- the full suite remains green.

## Phase 4 — Event-resolved scheduler and emergent WTA

This is the highest-risk phase. Implement and verify it on minimal synthetic networks before enabling the full preset.

- Add the boundary event scheduler described by the specification.
- Compute excitatory threshold latencies analytically and process them causally.
- Implement zero-latency inhibitory relay and hard reset at the same `(t, tau)`.
- Preserve ordinary one-boundary excitation delivery.
- Add event invalidation/versioning so a queued crossing cannot fire after an earlier hard reset changes its state.
- Preserve deterministic stable ordering and expose tolerance-based ties.
- Keep the legacy scheduler/competitor path unchanged for legacy networks.

Focused tests must cover at least:

- unequal-latency competitors, where the earlier spike resets the later candidate before it fires;
- exact and within-tolerance ties;
- outside-tolerance near ties;
- stale queued-event invalidation;
- multiple same-time relay/reset events;
- reset after an already emitted earlier spike;
- absence of an artificial extra outer-boundary delay;
- rejection of mixed legacy/event-resolved competition.

Phase 4 acceptance criteria:

- WTA emerges from threshold latency plus causal inhibition;
- no forced winner policy exists;
- timing diagnostics make ordering auditable;
- legacy behavior and the full suite remain green.

## Phase 5 — Full `rg_coincidence` preset and causal integration

Only now expose and run the new built-in topology.

- Implement the exact population sizes, edge counts, roles, and metadata in the specification.
- Add pretrained RG-to-L1E invocation for this topology without changing legacy RG behavior.
- Wire L1E-to-L1C basal, L2E-to-L1C apical, L1C-to-L1I, L1I-to-L1E, and event-resolved L2 WTA exactly as specified.
- Use the specified initial weights or their documented derived values.
- Verify that learned C timing can eventually outrun and suppress the intended L1E event; do not assume this from topology alone.

Focused tests must cover at least:

- exact node/edge counts and fan-in/fan-out;
- one basal and eight apical inputs per C cell;
- pretrained RG-to-L1E behavior;
- two-event C cadence and the initial/max-weight timing sanity checks;
- immediate L1 and L2 inhibitory resets;
- topology-order invariance under stable metadata-equivalent construction.

Phase 5 acceptance criteria:

- the preset runs end to end through the public simulation entry point;
- its causal traces agree with the specification;
- all legacy preset/golden tests remain green;
- the full suite remains green.

## Phase 6 — Public protocol and frontend support

- Extend serialization and API payloads with optional, backward-compatible C/timing fields.
- Preserve existing field meanings and clients.
- Add frontend controls/visual distinctions/diagnostics specified for C cells, basal/apical edges, and event-resolved timing.
- Ensure saved-state and replay behavior is deterministic.
- Avoid exposing internal implementation details that are not part of the public contract.

Focused tests must cover protocol round trips, omitted optional fields, old payload compatibility, API execution of the new preset, and deterministic replay.

Phase 6 acceptance criteria:

- the new preset is usable and inspectable through supported interfaces;
- old payloads and presets still work;
- the full suite remains green.

## Phase 7 — Scientific validation and documentation

Run measurements rather than treating implementation tests as proof of the scientific outcome.

- Run the isolated C-cell experiment first.
- Run the complete `rg_coincidence` topology second.
- Measure C spikes, inhibition, L1 suppression, L2 WTA ties, learned basal weights, timing margins, and output frequency ratio.
- Verify deterministic replay.
- Update focused user-facing documentation and the implementation status file.

Do not tune hidden constants merely to produce a preferred plot. If the exact requested frequency-halving result is not achieved, report the measured result, traces, and likely mechanism honestly while preserving correct mechanics.

Phase 7 acceptance criteria:

- all required tests pass;
- measurements and parameters are reproducible;
- implementation deviations are explicit;
- scientific success and mechanical correctness are reported separately.

## Regression and quality rules

- Run the repository's existing test command discovered in Phase 0. When the current environment matches the repository setup, the expected full-suite form is:

  ```bash
  PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
  ```

- Use focused test commands before the full suite.
- Do not broadly regenerate golden files. Any intentional golden change requires a field-by-field explanation tied to the specification.
- Treat the Phase 0 `git diff --check` output as the whitespace baseline. Do not edit
  unrelated user files merely to clean up a pre-existing diagnostic.
- Do not weaken existing tests to make the refactor pass.
- Do not introduce a new dependency unless it is truly necessary and explicitly justified.
- Keep public compatibility shims narrow, documented, and tested.
- Prefer small, named helpers for event semantics over deeply nested scheduler logic.
- Preserve deterministic ordering; never depend on set or dictionary iteration accidentally.
- Add diagnostics for `t`, `tau`, event kind, source, target, invalidation, reset, and tie fallback where the specification requires observability.

## Stop conditions

Stop at the current phase and provide a concrete blocker report if:

- the baseline is failing in a way that prevents meaningful regression testing;
- the live architecture contradicts a core specification assumption and the resolution changes scientific semantics;
- an analytic timing invariant cannot be satisfied by the stated equations;
- progress would require changing an existing preset's behavior outside the approved scope;
- a phase cannot pass without weakening tests, hiding failures, or adding preset/ID-specific hacks;
- pre-existing user work overlaps a necessary edit and cannot be preserved safely.

A blocker report must include the failing command, minimal reproduction, relevant state/equations, root-cause hypothesis, and the smallest decision needed from the user.

## Final handoff

At completion, provide:

1. a phase-by-phase status table;
2. the files changed and why;
3. exact focused and full-suite test commands with results;
4. topology count and timing-equation verification results;
5. measured scientific outputs, including negative results;
6. all deviations from the technical specification;
7. compatibility and remaining-risk notes;
8. final `git status --short`.

Begin with Phase 0. Do not begin production edits until its audit and baseline are complete.
