# Evolution of My Multi-Agent Engineering Workflow

This document records how my use of AI agents evolved during the design and
implementation of this SNN project. The important change was not simply moving from
one model to another. It was learning to assign different kinds of work according to
each agent's strengths, while preserving a detailed, inspectable engineering process.

**Author note**

Much of this document itself was written by Sol. I asked it to review the history of the repo
and its memories of how I interact with it. I'm sure you'll be able to identify the sections that
were written by Sol, so you will be able to notice just how much of the workflow sentiment 
it was able to extract from our sessions. I have put some notes throughout here to add on some important 
pieces as well.

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

**Author note**

Most of my conversations with GPT start off with a larger description of
what I am looking for. This is typed up in a text file and pasted into codex or pointed 
to for codex to read. This is usually a descriptive "story" of what exactly I want to see 
happen in the network (the behavior of nodes, the excpected emergent behavior, the equations
I want to see used, edge case handling, etc). I then conversed with GPT to see if it had an 
understanding of what I requested. This usually involves some back and forth iterations and 
clarification. Because of the detail in the kick off prompt, it usually also identified pit falls
in my assumptions and we worked through them to create a truly complete picture which finally ended up
in a technical spec.

Examples of kick off descriptions would be:

```text
Read through the repo you should find a document that describes some new neuroscience about coincidence detectors. Let me tell you the story of how things should work in this project now, and then we can have a technical discussion about how to implement this.

Currently we have some differing topologies, but the most basic is that L1 has an L1E pool and an L1I pool where I -> E connections are one to one. L1E is densely connected with L2E via feedforward and L2E -> L1I is densely connected feedback. E neurons work on summing accumulation based on the active gates. This makes a coincience detector extremely difficult to create so we would like to create a new class that is a coincidence pyramid cell. It still has weights that need to be learned upon firing, and will only fire when threshold is reached. The main difference will be that charge is only accumulated when all the inputs are active at the same time. In our most basic topology it would look like this. L1E -> L1C -> L1I -> L1E in a loop and L2E -> L1C and C neurons can only accumulate charge when both a basal and apical connection are active where basal connections are feedforward from L1 and apical connections are feedback from L2E. This will help us create a coincidence detector to suppress the evidence in a way where we can actually achieve frequency halving.

Here is the new weight update equation for coincidence nodes


dw = LR * FE * apical * (1 - (w/w_max)^2) * s_i * influence


where


LR = learning rate
FE = theta - sum(afferent_weights)
s_i = signal from feedforward (between -1 and 1)
apical = signal from feedback (between 0 and 1)
influence = 1/d^2 distance term


Essentially apical will tell us whether there is a weight update or not. We need to track some things here. When L1C get's signal, it needs to make sure that it's getting active signal in apical and basal connections before it decides to add any charge. This is important because a C cell will have 1 connection from L1E but 8 from L2E (in this topology), so we need to check "is an apical and basal connection both active right now?" before adding any charge. The weight that updates in L1C should also be the apical weight and not the basal weight, we are learning the apical feedback weight.

For C cells, the threshold is the same as E cells. Leave the leak and refractory equal for C cells to E cells. Treat all inhibitory cells as immediate relays (pretrained). We will treat L1E as pretrained for this topology so RG cells instantly invoke them to fire. Let's have inhibition fire more than 1 spike per fire event because I think that actual inhib cells do fire very fast and not just once at the same rate as E cells.

Let's discuss about this information and then talk about implementation details (do we inherit the E class for C class? do we build a dendrite class that has an apical and basal end so that we can apply this connectivity rule anywhere?). Eventually we will make a technical spec and have claude implement it for us.
```

```text
Let's talk a little bit about some things. One is a huge change. The other is a smaller but still involved change. Starting with the smaller change.

Currently the simulation is sequence, event based but is it truly event based? The sim still look around for the state of everyone in some order and then decides that it's time for some event to happen based on states. A little different from a true event based system. This is something to think about for the future but not immediately pressing. The other thing to think about is that WTA currently does not allow for 2 winners to emerge. I need this for composition to occur. Again, this is not immediately pressing but super important for the future. What are some solutions to this problem? I don't want to just use a delta tau and check if there is another winner in the vicinity. All the weights are initialized fairly close so we could get a bunch of winners. The case where multiple winners should happen is really if I have composition of a row and column into a plus. Say I trained a row and I trained a column fully. When I show a cross those two neurons should both fire and train a third neuron that is a plus. This is composition.

Type up this problem into a md file so that we can revisit it later in depth.

Now that you know a little bit about the easy problem, let's talk about a huge new topology change that touches almost everything in the repo. We want this new topology to be a new preset. We want the current topologies to still be functional after the change as well. After we chat about these problems, we will go into making a technical spec for the implementation.


CORTICAL COLUMNS (CC)
---------------------
Currently, we have RGC connected to L1E. L1E is just a neuron with a coincidence detector and an inhibitory neuron but it is meant to be an abstraction of a cortical column. Now what we want is this: A set of neurons that is configurable in excitatory neuron count, has another output excitatory neuron called Eor, a coincidence neuron, and an inhibitory neuron.

First of all, the cortical column is setup in the exact same way as the class is defined now in terms of connections within the column. We have "N" neurons with dense inputs from some previous input layer and bidirectional connections with the I neuron. An addition is one more neuron called Eor. All the E neurons in the cortical column connect to Eor. Eor is the "output" neuron. Eor connects densely with the next cortical column. It serves as the input from one cortical column to another and has connections from Eor to LxE_i where x is the upstream layer label. Another thing that Eor does is have a feedforward connection to its local coincidence neuron.

Yes, there is also a coincidence neuron in the cortical column. The C neuron has an input from the local Eor neuron. This is the basal connection. The C neuron also has connections that are feedback from the upstream cortical column. These are the apical connections.

Now that I have described the CC to CC connection schema, let's move on to what the inputs look like.

INPUTS/OVERALL TOPOLOGY
-----------------------
Let's say that our input RF is actually a 9x9 grid instead of 3x3. We break this into 9 3x3 grids, each of which connect to a cortical column in the exact same way as before (so we have 9 3x3 grids and 9 cortical columns where the connections are "1 to 1" between a mini grid and CC). The nuance here is that we are connecting RGCs directly into the cortical column instead of having any L1E neurons. The RGCs densely connect with the E cells of the cortical column. However, the whole grid should still be "connected" in input. Imagine I drew a diagonal accross the whole 9x9 grid. When I decompose this, I should get 3 3x3 grids that have that diagonal through them and these are what actually go on to be active in the RGCs. However, just to start we only want to have input in one of the 3x3 grids to isolate behavior.

So to recap, we have 9 sets of 3x3 grids each feeding 9 cortical columns. These 9 cortical columns connect to one more cortical column in the next layer (which serves as a sort of "compression" step). Let's treat the RGCs as the input layer, the 9 CCs as L1, and the final CC as L2.



Chat with me about these to show me you understand the problems.
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

**Author note**

One caveat to this is repository bloat. Over time, you can compound 50 to 100 md files involving
planning, architecting, documenting, etc. Many times I had to lean out the repo to keep it maintanable
(and myself sane). However, I still stored this in my git repo so my agent was able to use git commands
to trace back through commits to track history. So not only was there history within the local repo, but 
also long term history via the remote. This gave the agent a picture on what I had tried in the past, and what
I wanted to do now. All via md files which saved tokens and context bloat since the agent read minimal code. The existing test
case output also helped the agent complete the picture on a fresh start. With this all together, it seemed like I was
able to pick up exactly where I left off despite starting new sessions each morning. 

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

**Author note**

Additional things I noticed is that OpenAI agents in general consume tokens at a larger rate
than Anthropic models. Using gpt-5.5 for coding ate up my limit in one prompt, but claude could manage
large refactors pretty much throughout my 5 hour window. With Sol, it's a little better because they temporarily removed the 5 hour 
windows and let us use Sol out of only the weekly limit. They also optimized Sol to use less tokens. But despite this, I keep Sol
as a backup coding agent. I have found that its value as an architect outweighs its coding value. Say I use all my Sol tokens, now I'm
almost stuck. I can have a mediocre conversation with Claude that will end up in the same spot as Sol, but with more turns. And now I've wasted
my Sol tokens and my Claude tokens and am stuck at 10:30 am waiting for tokens to come back at 1pm. By using these agents to their strengths,
optimized for their token usage I can extend my usage until almost 12:15pm, take lunch, and come back at 1pm. Of course if you have heavier workloads
during these hours then you will run out faster but for my means this is what worked.

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

**Author note**

One final comment is that I ended up using these agents in "yolo" mode where I gave the agent permissions automatically. It became a pain to sit and hit "1" for a 40 minute job in which it just asked
me if it could cd into a repo or do git list. These large tasks didn't invovle removing files so I felt comfortable. However, if the work is more of an IT or system admin job I would heavily advise
against using "yolo" modes no matter how convenient it may feel. Abhi out.

-- Abhirup Dasgupta
