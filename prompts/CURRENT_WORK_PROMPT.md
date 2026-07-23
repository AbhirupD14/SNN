# Current Work Prompt

## Current task: recursive feature gates through L2

Paste the block below into a fresh Claude context:

```text
Build and causally validate the recursive feature-gated hierarchy only.

Read and execute the complete specification in:
prompts/Claude_Recursive_Feature_Gated_Hierarchy_Prompt.md

Before editing, inspect the repository, its relevant documentation/tests, and the complete
working-tree diff. Preserve all unrelated uncommitted work and prior experiment artifacts.
Preserve `tiled_cc_feature_gated` as the validated local-only control. Add a new topology
that keeps winner identity by giving every L1 ordinary competitor its own fixed,
suppressible inter-layer relay. Wrap those relays with paired C/I feature gates driven by
the L2 competitors. Do not feed L2 through one anonymous Eor, restore the old whole-bank
feedback reset, or let an L2 feature gate reset an L1 competitor.

This is an implementation-and-execution task, not a planning/audit task. First prove
the isolated E-source inter-layer relay/gate, then preserve the existing L1 turnover result,
then test L2 acquisition, turnover, and causal frequency-halving propagation on seed 1.
If a stage fails, preserve and report the negative result; do not tune around it. Do not
run the multi-seed recall/capacity sweep or add the four-competitor variant in this context.
Add focused tests, run the required headless stages and full suite, run `git diff --check`,
and provide the completion report required by the specification. Do not commit unless I
explicitly ask you to commit.

If the existing dirty worktree conflicts with the task, identify the exact overlapping
files and stop before overwriting someone else's changes. Otherwise continue through
implementation and verification.
```

## Work order after this task

1. Review the causal L1 and L2 gate traces before changing any parameter.
2. If the recursive seed-1 hierarchy passes, adapt and run frozen recall across seeds.
3. Add the four-competitor recursive preset only after eight-competitor robustness passes.
4. Design interleaved continuous learning in a fresh context.
5. Run noise invariance only after stable hierarchical acquisition and recall.

Do not give one Claude context all three implementation prompts.
