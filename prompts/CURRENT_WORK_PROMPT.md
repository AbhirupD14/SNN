# Current Work Prompt

## Status

The per-feature gated tiled direction has been rejected and removed. The supported built-in
topologies are `rg_coincidence`, `tiled_cc`, `tiled_cc_l1_4`,
`tiled_cc_direct_identity`, `tiled_cc_double_eor`, and `rg_direct_cc4`.

Since the original scaling prompt was written, four timing/output changes landed:

1. classic Eor is a fixed, non-plastic relay at `theta`;
2. direct identity removes Eor and preserves the local winner address;
3. the event loop drains dependent crossings at `tau=1.0`;
4. `input_period=0` now auto-matches the graph-derived feedback-loop latency, producing
   exact presentation-level alternation without relying on a seed-specific cadence alias.

`tiled_cc_double_eor` is a diagnostic latency probe, not an approved research topology.
The cadence result and its remaining certainty limitation are recorded in
`docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`.

The rejection rationale and the intended replacement semantics are recorded in
“Rejected direction: per-feature gated tiled columns” in
`Current_Implementation_Methodology_Equations.md`. That report is the sole current
methodological record of the removed direction; git history is the archive for the deleted
implementation details.

## Current task: existing-topology frequency scaling

Paste the block below into a fresh context:

```text
Read and execute the complete experiment specification in:
prompts/Claude_Existing_Topology_Frequency_Scaling_Prompt.md

Before editing, inspect the repository, complete working-tree diff, current dashboard
configuration, and the standing-problem/methodology sections referenced by the prompt.
Preserve all feature-gated removal work and unrelated scientific artifacts.

Use only the existing tiled_cc topology. Start from the LIVE
backend.dashboard_config.DASHBOARD_OVERRIDES and require the current resolved contract:
dual FE/FES on, eta=4, c_eta=16, input_period=0 (auto, resolved to loop latency 3),
theta/2 pattern-detector ceiling, fixed Eor at theta, and feedback reset on. Keep every
production equation, timing rule, and connection unchanged.

Sweep active patch count 1 through 9. The experiment must now distinguish two questions:
(1) whether auto-paced exact alternation is present and stable, and (2) whether its onset
tracks meaningful C confirmation/maturity rather than merely the pacing schedule. Run
matched feedback-off controls, record resolved/eligible presentations rather than raw
boundaries, preserve negative results, and write the required resumable artifacts and
final scaling report.

Do not implement a new topology or confidence mechanism. Do not commit or push.
```

The old `Claude_Final_Experiments_Prompt.md` is not the current execution prompt.
