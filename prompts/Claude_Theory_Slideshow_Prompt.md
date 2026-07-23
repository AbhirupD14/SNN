# Claude Prompt: From Spikes to Symbols Theory Slideshow

Create a polished 16:9 slideshow and export it as a PDF based on the theoretical architecture developed in this repository.

## Purpose

This is not an implementation-status presentation. Do not discuss debugging history, round-robin behavior, consolidation problems, ownership failures, ablations, or unresolved implementation details.

Use the repository to understand the architecture and scientific ideas, then present the larger theory: what this approach should make possible, why it differs from mainstream AI, and what its expected advantages are.

## Audience

The audience understands LLMs at a general level—tokens, training data, next-token prediction, and hallucination—but is unfamiliar with spiking neural networks, computational neuroscience, and this work.

## Core Thesis

Modern LLMs learn a massive statistical surface through global optimization. This work explores a different foundation for intelligence:

- local receptive fields;
- event-driven spiking neurons;
- local continuous learning;
- excitation and inhibition;
- stable specialist neurons;
- hierarchical symbol composition;
- internal simulation and imagination;
- inspectable reasoning chains;
- explicit signals for novelty and uncertainty.

The architecture is designed to grow intelligence from local circuits rather than compressing intelligence into one globally trained model.

## Tone

Be ambitious, visionary, and technically credible.

It is acceptable to explain what the architecture should enable, even when the complete hierarchy has not yet been built. Phrase these claims as:

- “the architecture is designed to…”
- “this should enable…”
- “the expected computational gain is…”
- “the long-term hypothesis is…”

Do not turn the presentation into a list of caveats. Include one concise final note distinguishing architectural predictions from measured hardware benchmarks.

## Repository Grounding

Inspect the repository before producing the deck. Use the current architecture as the technical basis, especially:

- `README.md`
- `Current_Implementation_Methodology_Equations.md`
- `Architecture_Decisions_Report.md`
- `PaulVersion.md`
- `Inhibitory_Off_Weight_Recruitment_Spec.md`
- `docs/DASHBOARD.md`
- the active simulation, neuron, and learning-rule code

Do not make the deck a code walkthrough. Translate the underlying ideas into accessible diagrams and explanations.

## Narrative

Build the presentation around this progression:

```text
biology
→ local perception
→ stable internal symbols
→ hierarchical composition
→ imagination
→ reasoning
→ uncertainty
→ continuous learning
→ scalable neuromorphic computation
```

### 1. Why Look Beyond the LLM Paradigm?

Begin by recognizing what LLMs do well. Then explain the structural constraints motivating this research:

- Training is based on a large global optimization process.
- Knowledge is learned primarily from large, static datasets.
- Updating knowledge can require fine-tuning, retraining, or external memory.
- Computation relies heavily on dense matrix operations and data movement.
- Representations are powerful but distributed and difficult to inspect.
- Fluent generation does not guarantee grounded knowledge.
- The model is generally required to produce another token even when its internal evidence is weak.

Do not argue that LLMs are incapable of reasoning or composition. Argue that this architecture seeks a more explicit, local, continuously adaptive foundation for those capabilities.

### 2. Biology Begins With Local Computation

Connect the architecture to biological information processing.

Explain that nervous systems do not begin with a global model looking at every sensory value simultaneously. They begin with many local circuits that transform nearby events into meaningful features.

Use the retina as the first example:

- Photoreceptors provide local signals.
- Retinal circuits compare nearby regions.
- Center-ON/surround-OFF ganglion cells respond strongly when the center is activated relative to its surroundings.
- The output represents local contrast and structure, not a raw copy of every light value.
- Lateral inhibition increases separation between competing interpretations.

Connect this to the 3×3 grid used in the repository:

- Treat it as a minimal receptive field.
- The center and surrounding cells form a compact local visual neighborhood.
- Overlapping line patterns demonstrate how local cells can learn relationships between a center and its surroundings.
- The same principle can produce orientation-sensitive features such as horizontal, vertical, and diagonal lines.
- These features can become the inputs to the next layer.

Present the grid as the conceptual beginning of a retinal-to-cortical hierarchy.

Explain the relevant biological motifs:

- membrane charge accumulates over time;
- neurons communicate through spikes;
- refractory periods regulate repeated firing;
- excitatory synapses reinforce compatible activity;
- inhibitory neurons create contrast and competition;
- synapses learn from information available locally;
- receptive fields emerge through repeated experience;
- stable specialists represent recurring patterns.

Describe these as biologically inspired computational principles, not a detailed simulation of every retinal or cortical cell type.

### 3. From Competition to Stable Symbols

Explain the central representational idea:

- Multiple neurons initially compete to explain an input.
- Repeated exposure causes one neuron to become especially responsive to a recurring structure.
- That neuron becomes a specialist.
- Its spike acts as a compact symbol for the learned structure.
- The raw input can now be replaced by a sparse event meaning “this feature is present.”

Use an intuitive progression:

```text
pixels
→ line specialists
→ corner and contour specialists
→ shapes
→ objects
→ relationships
→ concepts
```

Define symbol ownership:

> A symbol is owned when a specific internal assembly responds reliably and selectively to the same meaningful structure.

Explain why ownership matters:

- It creates a persistent internal identity.
- It makes representations inspectable.
- It allows the same concept to be reused in many contexts.
- It separates recognition from language generation.
- It provides a unit that higher layers can compose.

### 4. Hierarchical Symbol Composition

Make hierarchical composition the centerpiece of the deck.

A higher layer should not need to relearn every original pixel. It should learn combinations of lower-level symbols:

```text
horizontal line + vertical line
→ corner or cross

corners + contour
→ shape

shape + texture
→ object

object + motion
→ event

event + consequence
→ causal relationship
```

Explain the expected gains:

- **Reuse:** a learned primitive participates in many larger concepts.
- **Combinatorial generalization:** known pieces can form unseen combinations.
- **Abstraction:** higher levels operate on meaning rather than raw measurements.
- **Modularity:** new concepts can be added without redesigning the whole network.
- **Interpretability:** a concept can be traced back through the specialists that compose it.
- **Efficient learning:** a new object need not be learned independently at every size, position, and context.

### 5. Size-Agnostic Features

Explain how local receptive fields should make features less dependent on total input size.

A line remains a line whether it occurs:

- in a 3×3 patch;
- inside a large image;
- at another position;
- as part of a letter;
- as part of an object.

The architecture should scale by repeating local circuits across space. Each circuit detects relationships in its neighborhood and sends sparse symbols upward.

Larger input dimensions should require:

- more local receptive fields;
- more event routing;
- more higher-level composition;

but not a completely different learning rule.

Emphasize that higher layers consume recognized features rather than every raw pixel. This allows the architecture to scale through composition instead of requiring unrestricted interaction between every input element.

### 6. Composition as Imagination

Define imagination as the internal reactivation and recombination of learned symbols.

Once a concept has a stable internal representation, it should no longer require the original sensory input to become active. Higher circuits could reactivate symbols internally and combine them into candidate situations.

Use a concrete chain:

```text
object
+ motion toward barrier
→ possible collision

possible collision
+ alternate motion
→ avoided collision
```

Explain that imagination should allow the system to:

- construct configurations it has not directly observed;
- compare possible future states;
- test alternative actions internally;
- predict likely sensory consequences;
- learn from mismatches between prediction and observation.

The goal is not image generation alone. Imagination is an internal workspace for composing possibilities.

### 7. Logical Reasoning Chains

Present logical reasoning as a sequence of grounded symbol activations:

```text
recognized state
→ learned relationship
→ predicted consequence
→ candidate action
→ expected outcome
```

Each link should be represented by identifiable neural specialists or assemblies.

Expected advantages:

- Reasoning steps can be inspected.
- Each step can be grounded in learned sensory or conceptual symbols.
- The system can distinguish a known transition from a missing one.
- Failed predictions can update the responsible local relationship.
- Frequently useful reasoning chains can themselves become reusable higher-level symbols.
- Long chains can be formed by composing shorter learned chains.

Contrast this carefully with an LLM:

- LLM reasoning is expressed through sequences of tokens generated from distributed representations.
- This architecture aims to make reasoning an internal sequence of owned symbols and learned transitions.
- Language can later describe the chain, but language is not the chain itself.

### 8. Knowing What It Does Not Know

Make this a major theoretical advantage.

The model should not define knowledge as merely producing an output. Knowledge should require a stable, sufficiently strong internal representation.

Show four states:

```text
Stable specialist with a strong competitive margin
→ “I recognize this.”

Several nearly equal specialists
→ “I am uncertain which known concept applies.”

Known parts in an unfamiliar arrangement
→ “I recognize the components, but not this composition.”

No specialist reaches the required response
→ “I do not have a representation for this.”
```

Explain how uncertainty can emerge from observable dynamics:

- firing threshold;
- response strength;
- competitive margin;
- stability over time;
- consistency of symbol ownership;
- completion or failure of a reasoning chain;
- agreement between prediction and subsequent sensory input.

This should enable explicit novelty detection and epistemic humility. Instead of always generating the next plausible answer, the architecture should be capable of stopping and reporting that a symbol or relationship is missing.

### 9. Continuous Learning

Explain that learning and inference occur through the same local circuit.

When a new event arrives:

1. Only relevant local pathways activate.
2. Neurons integrate incoming spikes.
3. Specialists compete.
4. Participating synapses update locally.
5. The new experience becomes part of the system immediately.
6. Operation continues without a separate global retraining cycle.

Expected compute gains:

- no network-wide backward pass;
- no global loss function;
- no need to retain a complete activation graph for backpropagation;
- sparse updates to participating synapses;
- learning cost tied to active events;
- inference and learning on the same substrate;
- no need to replay the complete historical dataset for every update;
- less movement of data between memory and compute;
- natural compatibility with asynchronous hardware.

Emphasize the difference between periodically rebuilding a model of the world and continuously adapting while operating in the world.

### 10. How Continuous Learning Changes Datasets

Do not claim that the system needs no data. Explain that it changes the role of data.

Conventional model development often depends on a massive, centrally assembled snapshot of past information. A continuously learning system should acquire much more of its knowledge through direct, ongoing experience.

Expected effects:

- Reduced dependence on giant frozen datasets.
- Less need to anticipate every future case during initial training.
- Immediate incorporation of new information.
- Adaptation to local environments and changing conditions.
- Reuse of existing symbols when learning new compositions.
- Potentially fewer examples required for concepts built from known parts.
- More direct identification of missing knowledge.
- Representation can grow incrementally rather than being rebuilt wholesale.

### 11. Dataset Skew and Bias

Explain this carefully and constructively.

Continuous learning does not automatically eliminate bias. A skewed stream of experience can still produce skewed internal representations.

However, this architecture should make bias more observable and addressable:

- Which concepts have specialists?
- Which concepts collide or remain undifferentiated?
- Which inputs produce no representation?
- Which specialists dominate activity?
- Which experiences are overrepresented?
- Which reasoning transitions are missing?
- Does the system recognize components but fail on a particular composition?

Because representations have explicit ownership and local receptive fields, representational gaps should be easier to inspect than bias distributed diffusely across billions of parameters.

Present the goal as moving from opaque dataset bias toward measurable representational coverage.

### 12. Scaling Architecture

Explain scaling as replication and hierarchy:

- Repeat the same local circuit across a larger sensory surface.
- Allow nearby circuits to emit sparse feature symbols.
- Route those symbols into higher competitive layers.
- Reuse the same local learning rules at every level.
- Add layers for spatial, temporal, relational, and causal composition.
- Distribute independent regions across many hardware cores.
- Activate only the regions involved in the current event.

Describe the architecture as naturally parallel:

- neurons hold local state;
- synapses update locally;
- regions communicate with spikes;
- unrelated regions can operate independently;
- there is no single global gradient that every component must wait for.

Scaling should come from adding columns and hierarchical levels rather than making every unit globally connected to every other unit.

### 13. Neuromorphic Hardware

Explain why the theory maps naturally to spiking chips:

- Event-driven execution.
- Sparse communication.
- Local memory and local computation.
- Integer or fixed-point state.
- Asynchronous operation.
- Parallel neural cores.
- Learning directly on the device.
- Low activity when the environment is unchanged.
- Reduced dependence on moving large tensors through external memory.

Contrast two compute models visually:

```text
Conventional accelerator:
memory ↔ dense matrix processor ↔ memory
plus a separate backward pass

Neuromorphic system:
local state + local synapses + spike routing
with learning at the point of activity
```

State that the expected gain is especially important for always-on systems, robotics, edge devices, adaptive sensors, and autonomous agents that must learn continuously under limited power.

Do not invent numerical speed, energy, or memory improvements. Explain the sources of expected efficiency and identify physical benchmarking as future validation.

## Recommended Slide Sequence

Create approximately 16 slides:

1. **From Spikes to Symbols**
2. **The limits of globally trained intelligence**
3. **A different foundation: intelligence from local circuits**
4. **Biology computes locally**
5. **The retina: center-ON, surround-OFF**
6. **The 3×3 receptive field**
7. **Competition creates specialist neurons**
8. **A stable specialist becomes a symbol**
9. **Hierarchies make symbols reusable**
10. **Size-agnostic learning through local features**
11. **Composition becomes imagination**
12. **Symbols become inspectable reasoning chains**
13. **Knowing, uncertainty, and the unknown**
14. **Continuous learning changes the compute equation**
15. **Smaller dependence on static datasets—and more visible bias**
16. **Scaling onto neuromorphic hardware**

## Visual Requirements

The deck should be diagram-driven, with minimal text.

Include:

- a center-ON/surround-OFF retinal receptive-field diagram;
- the four overlapping line patterns on a 3×3 grid;
- excitatory and inhibitory neuron interactions;
- a specialist-neuron competition diagram;
- a hierarchy from pixels to concepts;
- a symbol-composition diagram;
- an imagination or counterfactual reasoning loop;
- a four-state “known / uncertain / novel composition / unknown” diagram;
- a static training versus continuous learning timeline;
- a dense accelerator versus event-driven neuromorphic-chip diagram;
- a modular scaling diagram showing repeated cortical columns.

## Style

Use a sophisticated dark background, restrained electric colors, large typography, and clean scientific diagrams. Avoid generic brain stock photography, robot imagery, and excessive equations.

Use approximately 20–40 words on most slides. Put detailed explanations in speaker notes.

## Deliverables

Produce:

1. An editable slideshow source.
2. A final 16:9 PDF.
3. A sources appendix.
4. A one-page architecture summary.

Inspect the exported PDF and correct clipping, overcrowding, low contrast, unreadable diagrams, and inconsistent terminology before finishing.

## Final Message

End with:

> Instead of training one enormous model to approximate everything it has seen, we can build intelligence from local systems that learn continuously, form explicit symbols, compose new possibilities, trace their own reasoning, and recognize the boundaries of their knowledge.
