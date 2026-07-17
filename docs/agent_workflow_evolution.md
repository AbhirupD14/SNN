# Evolution of Our Multi-Agent Engineering Workflow

This document records how my use of AI agents evolved during the design and
implementation of this SNN project. The important change was not simply moving from
one model to another. It was learning to assign different kinds of work according to
each agent's strengths, while preserving a detailed, inspectable engineering process.

## The original division: GPT-5.5 as architect, Claude as implementer

The early workflow had a clear separation of responsibilities.

GPT-5.5 was the architect and conversational partner. We used it to explore the
scientific problem, question assumptions, compare topologies, reason about timing,
and turn vague intuitions into explicit mechanisms. Once the design was coherent, we
converted it into a detailed implementation prompt and dispatched that prompt to
Claude—usually Opus or Fable—for the actual repository work.

This separation was practical rather than ideological. In my experience, GPT-5.5
was nowhere near the coding capability of Opus or Fable. Claude was substantially
better at carrying a large implementation across many files, following an extensive
technical specification, and producing a complete first pass. Trying to use GPT-5.5
as the main coding agent usually created more supervision and correction work than it
saved.

GPT-5.5 nevertheless remained central because it was much better for conversation.
It understood what I was trying to get at even when I had not yet found the precise
technical language. It was also better at explaining difficult ideas clearly. It
could take an abstract timing or inhibition problem and give me an analogy that made
the mechanism intuitive before we reduced it to equations and edges. That ability was
especially valuable because many of our problems were not ordinary software bugs;
they were conceptual questions about learning, locality, symmetry breaking, and
causal timing.

The original model was therefore:

```text
my intuition and observations
        ↓
conversation with GPT-5.5
        ↓
architecture, analogy, and detailed plan
        ↓
technical specification or visualization prompt
        ↓
Claude Opus/Fable implementation
        ↓
GPT review, scientific audit, and next design iteration
```

## Planning became a concrete engineering artifact

Over time, “planning” stopped meaning a short list of tasks. We began producing
artifacts that could survive context loss and move cleanly between agents:

- Exact population and projection inventories.
- Explicit causal timing phases and delay semantics.
- Edge polarity, plasticity, and multiplicity tables.
- Parameter definitions and ablation controls.
- Behavioral invariants and regression requirements.
- Flowcharts showing layouts that were difficult to understand in ASCII.
- Kickoff prompts that pointed an implementation agent at a complete specification.

This repository contains several examples, including the
[RG implementation prompt](../Claude_RG_Third_Preset_Implementation_Prompt.md), the
[residual-pathway flowchart specification](../Claude_Residual_Error_Pathway_Flowchart_Prompt.md),
the resulting [graphical circuit](residual_error_pathway_flowchart.svg), and the
[current mathematical methodology](../Current_Implementation_Methodology_Equations.md).

The visualization step was not decoration. Neural topology can look reasonable in a
verbal description while hiding a global fanout, an impossible inhibitory edge, or a
timing dependency. Drawing the populations and their indexed connections made it much
easier to ask the right questions before implementation.

## What we learned from dispatching work to Claude

Claude remained very effective for large, specification-driven implementation. But
the workflow became more disciplined as we learned that a detailed completion report
was not the same thing as an independently verified result.

We began reviewing several layers separately:

1. **Structural correctness:** Were the requested cells and edges actually present?
2. **Execution semantics:** Did events traverse those edges in the intended causal
   phase?
3. **Instrumentation correctness:** Did the reported metric measure what its label
   claimed?
4. **Scientific interpretation:** Did the results support the headline conclusion,
   or only a narrower claim?
5. **Regression safety:** Did the old presets remain bit-exact?

This mattered in the RG work. Claude's implementation was broadly sound, but review
found that a changing-pattern experiment classified updates against the first phase's
pixels and called pooled population-event gaps “L1 ISI.” We also found that a valid
competitor-to-downstream feedforward edge was accepted by the graph specification but
never dispatched by the engine. These were not reasons to abandon Claude. They were
evidence that implementation and audit are different roles and often benefit from
different agents.

Repository-based specifications also made Claude API failures less damaging. If a
response stopped halfway through, another agent could inspect the current files,
reread the specification, and resume without reconstructing the design from chat
history.

## The Sol inflection point

Moving from GPT-5.5 to Sol changed the balance.

Sol retained the qualities that had made GPT useful as the architect: conversational
understanding, clear explanations, conceptual synthesis, and the ability to produce
useful analogies. But it was also capable enough in repository-scale coding to serve
as the primary implementation agent when needed.

This removed the old hard boundary between “GPT plans” and “Claude codes.” With Sol,
the same agent could:

- Hold the scientific intent developed during the conversation.
- Inspect the actual repository rather than describe changes abstractly.
- Write the technical specification and visualization.
- Implement the topology directly.
- Run focused tests and full regression suites.
- Audit live behavior when the dashboard contradicted the intended model.
- Correct the implementation without another lossy handoff.

The residual/error pathway illustrates this transition. We initially prepared a
Claude flowchart prompt, but repeated API failures made the handoff unproductive. Sol
created the SVG and companion document directly, then implemented the 52-cell,
274-projection `rg_residual` preset. When the dashboard showed that SwitchI edges were
lighting without charging the neurons, Sol audited the implementation and found that
SwitchI was still a Boolean relay. It then replaced that abstraction with a charged,
bounded two-branch coincidence interneuron and verified the result against the exact
dashboard seed.

That sequence would have required several architect–implementer round trips in the
GPT-5.5 era. Sol could carry it end to end because it combined enough coding ability
with the conversational context behind the design.

## The current model: assign roles by strength, not brand

The present workflow is not “Sol replaced Claude.” Both are capable enough to manage
substantial work. The question is which assignment produces the best result for the
available time, context, and token budget.

| Work | Preferred strength |
| --- | --- |
| Exploring an unclear idea | Conversational understanding, analogy, patient questioning |
| Architecture and scientific reasoning | Long-context synthesis and explicit causal modeling |
| Flowcharts and implementation specifications | Precision, completeness, and visual organization |
| Large mechanical repository implementation | High coding throughput and specification adherence |
| High-context or surgical implementation | Continuity with the design conversation plus strong coding |
| Reviewing metrics and scientific claims | Skepticism, independent reproduction, and careful interpretation |
| Integration and regression protection | Repository awareness, tests, and respect for existing behavior |

Claude Opus/Fable remains a strong choice for a large implementation that already has
a stable specification. Sol is often the better primary agent when implementation is
tightly coupled to an evolving conceptual discussion, when rapid audit-and-correct
loops matter, or when handing the work off would discard important context.

## Token usage and limits became part of architecture

Sol's broader role also made token management more important. Being capable of doing
everything does not mean it is economical to do everything in one uninterrupted
conversation.

The practical rules are:

- Put durable decisions in repository documents instead of repeatedly restating them.
- Use a complete specification as the handoff boundary.
- Reserve high-context agent time for choices that genuinely depend on the history.
- Dispatch bounded, implementation-heavy work when another agent can execute it from
  the written specification.
- Keep verification local and reproducible: tests, golden trajectories, exact seeds,
  and saved experiment outputs are more efficient than repeatedly debating memory.
- Treat API and context limits as expected operational constraints. A task should be
  resumable from files even if an agent disappears mid-response.

This is why detailed planning remains valuable even when Sol writes the code itself.
The plan is not merely a prompt for another model. It is the durable contract between
the scientific idea, the implementation, the tests, and any future agent.

## Our current operating loop

The process that emerged is:

1. **Discuss the phenomenon.** Start from what the simulation or dashboard is doing,
   not from a predetermined code change.
2. **Make the mechanism understandable.** Use plain language and analogies before
   committing to equations.
3. **Specify the circuit precisely.** Record populations, local state, edges, signs,
   learning, delays, and expected causal order.
4. **Visualize dense relationships.** Produce a flowchart when topology or timing is
   difficult to audit linearly.
5. **Choose the implementation agent.** Consider coding throughput, retained context,
   token limits, and handoff cost.
6. **Implement behind an isolated preset or flag.** Preserve comparison topologies and
   avoid silently changing the scientific baseline.
7. **Test mechanics before interpretation.** Pin individual branches, causal delays,
   serialization, reset behavior, and ablations.
8. **Run regression and live checks.** Protect old goldens and compare dashboard state
   with direct engine measurements.
9. **Audit the conclusion.** Separate “the connection exists,” “the neuron fires,” and
   “the topology learns the intended classification.”
10. **Write down what remains unresolved.** A functioning implementation is not proof
    that the scientific hypothesis is correct.

## The central lesson

The biggest improvement came from stopping the search for one universally best agent.
GPT-5.5 was valuable because it helped me articulate and understand the problem, even
when Claude was clearly the stronger coder. Sol changed the workflow because it
preserved that conversational strength while becoming capable enough to own serious
implementation and verification work.

The durable principle is to use each agent for what it does best. Architecture,
implementation, visualization, explanation, and audit are distinct forms of work.
They can be assigned to different agents, or combined in one capable agent when the
context and token budget make that the better choice. The quality of the overall
process comes from explicit specifications, independent verification, and careful
handoffs—not from loyalty to a particular model.
