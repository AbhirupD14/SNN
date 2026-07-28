# Claude Prompt: Remove the Feature-Gated Topology Direction

## Objective

Remove the dedicated feature-gated tiled topology from the repository and leave the
supported preset surface, implementation, tests, experiments, and documentation internally
consistent.

The rejected built-in preset is:

```text
tiled_cc_feature_gated
```

No recursive or four-competitor feature-gated preset was implemented, but prompts proposing
those follow-on topologies also remain in the repository. Remove those obsolete directions
as part of this cleanup.

This is a deletion and cleanup task, not a redesign task. Do not replace the removed preset
with another topology and do not implement the future confidence/attention mechanism here.

## Approved methodological decision

Feature gating was a useful experiment, but it is not the selected architecture.

The experiment replicated a separate `S/C/If` chain for each input feature in every
receptive field. It demonstrated selective relay suppression and local turnover, but the
cost and semantics do not fit the intended direction:

1. Replicating feature relays and paired coincidence/inhibitory cells substantially
   increases topology density. Extending the motif recursively would add many identity
   relays and C/I gates between competition layers.
2. Competitive learning already supplies pattern allocation: a neuron that has specialized
   for one pattern becomes less competitive for a different pattern, allowing another
   neuron to acquire the changed pattern according to the learned weight distribution and
   presentation frequency.
3. Frequency reduction is intended to be a sparse, column-level confidence signal, not
   feature-by-feature explaining-away. A highly active region should indicate active
   learning, use, or guided attention; a slower region should indicate certainty that its
   current pattern is already learned.
4. Feedback-suppressed evidence should propagate as an absence of evidence. The future
   mechanism should therefore produce true column-level silence on suppressed evidence
   events without expanding every inter-column feature into a gated channel.

The historical attempt and rejection rationale are recorded in
`Current_Implementation_Methodology_Equations.md`. Preserve that report as the sole current
methodological record of this rejected direction.

## Read and inspect before editing

Read:

- `Current_Implementation_Methodology_Equations.md`, especially “Rejected direction:
  per-feature gated tiled columns”;
- `backend/network_spec.py`;
- `backend/simulation.py`;
- `backend/layout.py`;
- `backend/dashboard_config.py`;
- `backend/presets.py`;
- `README.md` and `docs/DASHBOARD.md`;
- `tests/test_preset_registry_contract.py`;
- the feature-gated tests and experiment before deleting them, so shared coverage is not
  accidentally removed.

Inspect the complete working-tree diff before editing. Preserve unrelated user work and
scientific artifacts. Do not commit or push unless explicitly requested.

## Scope boundary

Remove only the dedicated feature-gated tiled topology direction and code that exists solely
to support it.

Preserve:

- `rg_coincidence`;
- `tiled_cc`;
- `tiled_cc_l1_4`;
- `rg_direct_cc4`;
- their graph structure, equations, scheduler behavior, serialization, dashboard behavior,
  and golden trajectories;
- the general coincidence pyramidal cell;
- classic tiled-column `C -> I` whole-column feedback;
- ordinary within-column WTA;
- generic topology-editor support that is still exercised by a retained topology;
- general metadata fields or helpers if a retained topology or saved-spec contract still
  requires them.

Do not interpret the phrase “feature gating” so broadly that it removes coincidence cells,
hard-reset inhibition, or the `rg_coincidence` research control. The target is the dedicated
`tiled_cc_feature_gated` preset and its exclusive implementation surface.

## Required removals

### 1. Preset registry and engine construction

Remove `tiled_cc_feature_gated` from every built-in or valid-topology registry, including:

- `PRESETS`;
- `TILED_PRESETS`;
- `VALID_TOPOLOGIES`;
- dashboard topology options;
- preset listing/loading branches;
- engine construction/import branches;
- public descriptions and serialized preset inventories.

After cleanup, attempting to construct
`SimulationEngine(topology="tiled_cc_feature_gated")` must fail through the normal invalid
topology path. Add a focused assertion if the retained registry-contract tests do not
already prove this.

### 2. Exclusive graph builders and validation

Delete code used only by the removed topology, including, after confirming references:

- `tiled_cc_feature_gated_spec`;
- feature-gate-only handle types and builders;
- feature-gated bank-to-bank helpers that have no retained caller;
- feature-gated topology variant constants and validator branches;
- `_validate_tiled_feature_gated`;
- feature-gate-only projection names, metadata handling, and comments.

Do not remove a generic builder, edge kind, metadata field, or validator rule merely because
the removed preset used it. Prove it has no retained caller first. Keep validation strict for
every retained graph and saved custom topology.

Saved custom specs that explicitly declare the removed feature-gated variant may cease to
validate; report this compatibility consequence plainly. Do not silently reinterpret them as
classic tiled graphs.

### 3. Layout and dashboard support

Remove layout routines and branches that exist only to place the removed preset's `S/C/If`
planes. Remove its selector label and descriptive text from the dashboard.

Do not change the dashboard startup preset or its current patch composition in this cleanup.
Do not alter renderer behavior shared with custom graphs.

### 4. Dedicated experiment and tests

Delete:

- `experiments/feature_gated_turnover.py`;
- `tests/test_feature_gated_turnover.py`;
- `tests/test_tiled_cc_feature_gated.py`.

Before deleting, identify any assertions that protect generic retained behavior. Move only
genuinely shared contracts into the closest retained test file; do not preserve tests whose
sole purpose is to validate the removed graph.

Update `tests/test_preset_registry_contract.py` and every count/list assertion to describe
only the retained built-ins.

Do not delete generated run artifacts outside version control unless explicitly requested.

### 5. Obsolete feature-gated prompts

Delete the prompts that propose building or extending the rejected topology:

- `prompts/Claude_Feature_Gated_Tiled_Topology_Prompt.md`;
- `prompts/Claude_Recursive_Feature_Gated_Hierarchy_Prompt.md`;
- `prompts/Claude_Feature_Gated_Recall_and_Capacity_Prompt.md`.

Replace `prompts/CURRENT_WORK_PROMPT.md` with a neutral handoff or the current approved work
direction. It must not instruct a future agent to build recursive feature gates.

For other historical prompts that merely mention the feature-gated work as an old control,
remove or revise statements that require preserving/running that now-deleted control. Do not
rewrite unrelated historical experiment specifications.

### 6. Documentation

The standalone `docs/FEATURE_GATED_TILED_TOPOLOGY.md` has already been intentionally deleted.
Do not recreate or replace it.

Update:

- `README.md`;
- `docs/DASHBOARD.md`;
- source docstrings and comments;
- test inventory and preset counts;
- any current handoff/status document that presents feature gating as supported or planned.

The only affirmative discussion that should remain is the rejected-direction report in
`Current_Implementation_Methodology_Equations.md` and, where useful, a short pointer to it.
Historical git history is the archive for the deleted implementation details.

## Non-goals

Do not:

- implement the proposed column-confidence state machine;
- tune frequency-halving parameters;
- change the delayed feedback reset;
- change competitive learning;
- add a replacement preset;
- rename retained presets;
- regenerate unaffected golden fixtures;
- delete `rg_coincidence` or generic coincidence-cell support;
- commit or push.

## Verification

Run focused tests for:

- preset registries and listing/loading;
- retained network-spec validation;
- classic tiled builders and layouts;
- dashboard configuration and serialization;
- event scheduling, coincidence cells, WTA, feedback hard reset, and patch composition.

Then run:

```bash
.venv/bin/python -m pytest tests/ -q
git diff --check
```

Search the repository for:

```text
tiled_cc_feature_gated
TILED_VARIANT_FEATURE_GATED
tiled_cc_feature_gated_spec
build_feature_gate
feature_gated_turnover
Recursive Feature Gates
```

Any remaining match must be either:

- this removal prompt;
- the rejected-direction methodology report; or
- a clearly historical statement that does not advertise, require, or register the removed
  topology.

## Completion report

Report:

- every deleted and edited file;
- the final retained built-in topology list;
- exclusive builders, validators, layout code, tests, experiments, and prompts removed;
- any generic code deliberately retained and its live caller;
- the behavior when an old saved feature-gated spec is loaded;
- focused and full test commands/results;
- proof that retained goldens were not rewritten;
- `git diff --check` result;
- remaining feature-gating search matches and why each remains.
