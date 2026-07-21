# From Spikes to Symbols — Speaker Notes

Detailed talking points for each slide of `From_Spikes_to_Symbols.pdf`. The slides
carry ~20–40 words each; the depth lives here. Audience assumed to understand LLMs at
a general level (tokens, training data, next-token prediction, hallucination) but new
to spiking neural networks and computational neuroscience.

---

## 1 · From Spikes to Symbols (cover)
Open with the one-line thesis: we are exploring whether intelligence can be *grown*
from many small circuits that learn continuously, rather than *compressed* into one
model trained once over a giant dataset. The little diagram is the whole arc in
miniature — a train of spikes drives a neuron that becomes a specialist, whose single
spike stands for a learned feature (a symbol), which higher layers then compose. Frame
the talk as an architecture and a set of predictions, not a finished product.

## 2 · The limits of globally trained intelligence
Be generous first: LLMs are remarkable and genuinely reason and compose in practice.
The point is not that they can't — it's that their capability rests on structural
choices worth revisiting. Knowledge is acquired through one massive global optimization
over a largely static dataset; updating it typically means fine-tuning, retraining, or
bolting on external memory. Compute leans on dense matrix operations and heavy data
movement. Representations are powerful but distributed and hard to inspect. And the
generative objective generally pushes the model to produce another plausible token even
when its internal evidence is weak. These are the pressures motivating a different
foundation.

## 3 · A different foundation: intelligence from local circuits
State the alternative plainly: not one model that sees every input value at once, but
many small circuits that each learn from what is near them and emit sparse symbols
upward. The design commitments: local receptive fields; event-driven spiking neurons;
local, continuous learning with no global error signal; excitation and inhibition to
create stable specialists; and hierarchy to compose symbols into concepts. Everything
that follows is an unpacking of this diagram.

## 4 · Biology computes locally
Nervous systems do not begin with a global model reading every sensory value
simultaneously — they begin with local circuits that transform nearby events into
meaningful features. The motifs we borrow (as computational principles, not a faithful
cell-by-cell simulation): membrane charge accumulates over time; neurons communicate
through discrete spikes; refractory periods pace repeated firing; excitatory synapses
reinforce compatible activity; inhibitory neurons create contrast and competition;
synapses learn from locally available information; and stable specialists come to
represent recurring patterns. The left chips are these motifs; the right shows a neuron
integrating charge until it crosses threshold and fires, and an excitatory neuron
recruiting an inhibitory partner that suppresses competitors.

## 5 · The retina: center-ON, surround-OFF
Use the retina as the first concrete local circuit. Photoreceptors provide local
signals; retinal circuitry compares nearby regions. A center-ON/surround-OFF ganglion
cell responds strongly when its small center is lit relative to its surround — so its
output encodes *local contrast and structure*, not a raw copy of every light value. The
"Mexican-hat" response curve (peak at the center, dip in the surround) is the signature.
Lateral inhibition sharpens the boundary between competing interpretations. This is the
template every later layer reuses: compare locally, emit a feature.

## 6 · A 3×3 window learns oriented lines
Connect the biology to the actual system. The repository's minimal receptive field is a
3×3 grid — the smallest neighborhood where a cell can learn a relationship between a
center and its surroundings. The four primitives (middle row, middle column, and the two
diagonals) all cross and share the center pixel, which deliberately makes them overlap
and compete. Local cells become orientation-sensitive — horizontal, vertical, two
diagonals — and these features become the inputs to the next layer. Present the grid as
the conceptual beginning of a retinal-to-cortical hierarchy, not as the endpoint.

## 7 · Competition creates specialist neurons
Multiple neurons initially compete to explain the same input. When one fires, it
strengthens the synapses that matched the input and it recruits a shared inhibitory
neuron. That inhibitory neuron broadcasts a reset across the pool, clearing rivals so the
best-matched unit can win *again* on the next presentation — repeated wins are the
precondition for specialization. Crucially, a losing neuron's capacity is not destroyed:
it is conservatively redistributed from the gates that matched this pattern into its
other, inactive gates — which recruits it toward *different* patterns later. Emphasize
that this competition is delivered through real, adapting synapses and local rules, not a
global controller. (E = blue, I = amber throughout the deck.)

## 8 · A stable specialist becomes a symbol
Once a neuron responds reliably and selectively to the same recurring structure, its
single spike can *replace* the raw input — it now means "this feature is present." That
is symbol ownership: a specific internal assembly consistently and selectively answers to
one meaningful structure. Ownership matters because it creates a persistent internal
identity, makes the representation inspectable, lets the same concept be reused across
contexts, separates recognition from language generation, and gives higher layers a
clean unit to compose. The progression on the right — pixels → line specialists → corner
and contour specialists → shapes → objects → relationships and concepts — is the ladder
the rest of the talk climbs.

## 9 · Hierarchies make symbols reusable (centerpiece)
This is the heart of the argument. A higher layer should never relearn raw pixels; it
learns *combinations of lower-level symbols*: horizontal + vertical → corner or cross;
corners + contour → shape; shape + texture → object; object + motion → event; event +
consequence → causal relationship. The expected gains: **reuse** (one learned primitive
participates in many larger concepts); **combinatorial generalization** (known pieces
form unseen combinations); **abstraction** (higher levels operate on meaning, not raw
measurements); **modularity** (add concepts without redesigning the network);
**interpretability** (trace a concept back through the specialists that compose it); and
**efficient learning** (a new object need not be relearned at every size, position, and
context). Dwell here.

## 10 · Size-agnostic learning through local features
Because features are defined over local windows, a line is a line whether it appears in a
3×3 patch, inside a large image, at a different position, or as part of a letter or
object. The architecture scales by *repeating the same local circuit* across the sensory
surface — each detects relationships in its neighborhood and sends sparse symbols upward.
Larger inputs need more receptive fields, more event routing, and more higher-level
composition — but **not a different learning rule**. Higher layers consume recognized
features rather than every raw pixel, so the system scales through composition instead of
requiring unrestricted interaction between every pair of input elements.

## 11 · Composition becomes imagination
Define imagination precisely: the internal reactivation and recombination of learned
symbols. Once a concept has a stable internal representation, it no longer requires the
original sensory input to become active — higher circuits can reactivate symbols
internally and combine them into candidate situations. Walk the chain: object + motion
toward a barrier → possible collision; possible collision + an alternate action →
avoided collision. This lets the system construct configurations it has never directly
observed, compare possible futures, test alternative actions internally, predict likely
sensory consequences, and learn from mismatches between prediction and observation. The
goal is not image generation — it is an internal workspace for composing possibilities.

## 12 · Symbols become inspectable reasoning chains
Present reasoning as a sequence of grounded symbol activations: recognized state →
learned relationship → predicted consequence → candidate action → expected outcome. Each
link is an identifiable specialist or assembly. The advantages: steps can be inspected;
each is grounded in learned sensory/conceptual symbols; the system can tell a known
transition from a missing one; failed predictions update the specific local relationship
responsible; frequently useful chains can themselves become reusable higher-level
symbols; and long chains form by composing shorter learned ones. Contrast carefully with
an LLM: LLM reasoning is expressed through sequences of tokens generated from distributed
representations; here reasoning aims to be a sequence of *owned symbols and learned
transitions*. Language can later describe the chain — but language is not the chain
itself.

## 13 · Symbols are grown, not authored (vs. symbolic & neuro-symbolic AI)
Pre-empt the obvious objection: "spikes → symbols → reasoning chains" sounds like classic
symbolic AI, so say plainly how it differs. Three families. **Expert systems / ontologies +
knowledge bases (GOFAI):** the symbol vocabulary and the rules are *hand-authored* before any
data; an inference engine applies logic over an external rule base; the tokens are ungrounded
(their meaning lives in the designer's head — the classic symbol-grounding problem); and the
system doesn't learn (the knowledge-acquisition bottleneck, and brittleness outside what was
encoded). **Neuro-symbolic (NeSy)** — e.g. DeepProbLog, Logic Tensor Networks, ILP-plus-nets:
a real advance because it *learns the perception→predicate grounding*, but the predicate
vocabulary and the logical rules are still usually predefined, the symbolic reasoner is a
distinct module bolted onto a neural front-end, and training is typically a separate global
(gradient) phase. **This architecture:** symbols are *emergent* — a neuron becomes a symbol by
reliably winning competition for a recurring structure (nobody names it); meaning is *grounded
by construction* because the symbol simply is the activity of the circuit that detects the
structure (nothing left to ground); "reasoning" is a sequence of learned transitions between
symbol assemblies living in the *same* substrate, not an external engine; and inference and
learning happen together, continuously, locally, with no gradients. The honest caveat (repeat
slide 18's spirit): symbolic reasoners give you sound, verifiable, compositional inference
*today* with guarantees; here the perceptual symbol-formation base is what's built, and the
reasoning tower above it is the architectural hypothesis. The aim is to reach the *virtues* of
the symbolic level — inspectable, compositional, uncertainty-aware — without paying the
authoring, grounding, and brittleness costs. Lineage note if asked: this sits with *emergent*
symbol work (assembly calculus, sparse distributed representations / HTM, predictive coding,
vector-symbolic architectures), not the ontology-and-rule-engine tradition — symbolic *within*
the neural substrate rather than imported on top.

## 14 · Knowing, uncertainty, and the unknown
Make this a headline advantage. Knowledge should not be defined as merely producing an
output; it should require a stable, sufficiently strong internal representation. Four
states fall out naturally: a stable specialist with a strong competitive margin → "I
recognize this"; several nearly equal specialists → "I'm uncertain which known concept
applies"; known parts in an unfamiliar arrangement → "I recognize the components but not
this composition"; and no specialist reaching the required response → "I have no
representation for this." These distinctions emerge from observable dynamics — firing
threshold, response strength, competitive margin, stability over time, consistency of
symbol ownership, whether a reasoning chain completes, and agreement between prediction
and subsequent input. This enables explicit novelty detection and epistemic humility:
instead of always emitting the next plausible answer, the system can stop and report that
a symbol or relationship is missing.

## 15 · Continuous learning changes the compute equation
Learning and inference run through the *same* local circuit. When a new event arrives:
only relevant local pathways activate; neurons integrate incoming spikes; specialists
compete; participating synapses update locally; the new experience becomes part of the
system immediately; and operation continues without a separate global retraining cycle.
The expected compute gains follow directly: no network-wide backward pass; no global loss
function; no need to retain a full activation graph for backpropagation; sparse updates
to only the participating synapses; learning cost tied to active events; inference and
learning on the same substrate; no replay of the entire historical dataset per update;
less data movement between memory and compute; and natural compatibility with
asynchronous hardware. The contrast to hold onto: periodically *rebuilding* a model of
the world vs. continuously *adapting* while operating in it.

## 16 · Less dependence on static datasets — and more visible bias
Do not claim the system needs no data — it changes the *role* of data. Conventional
development leans on a massive, centrally assembled snapshot of the past; a continuously
learning system acquires much more of its knowledge through direct, ongoing experience.
Expected effects: reduced dependence on giant frozen datasets; less need to anticipate
every future case up front; immediate incorporation of new information; adaptation to
local, changing conditions; reuse of existing symbols when learning new compositions;
potentially fewer examples for concepts built from known parts; more direct
identification of missing knowledge; and representation that grows incrementally rather
than being rebuilt wholesale. On bias, be careful and constructive: continuous learning
does **not** automatically remove bias — a skewed stream of experience still produces
skewed representations. But it makes bias more *observable*: which concepts own
specialists, which collide or stay undifferentiated, which inputs produce no
representation, which specialists dominate, which experiences are overrepresented, which
transitions are missing. The goal is a move from opaque dataset bias toward measurable
representational coverage.

## 17 · Scaling onto neuromorphic hardware
Explain why the theory maps naturally onto spiking chips: event-driven execution; sparse
communication; local memory and local computation co-located; integer or fixed-point
state; asynchronous operation; parallel neural cores; learning directly on the device;
low activity when the environment is unchanged; and reduced dependence on moving large
tensors through external memory. Contrast the two compute models: a conventional
accelerator shuttles data between memory and a dense matrix processor and adds a separate
backward pass over everything; a neuromorphic system keeps local state and local synapses
with spike routing and learns at the point of activity, leaving unrelated regions idle
and quiet. Scaling comes from adding columns and hierarchical levels, not from making
every unit globally connected to every other. This matters most for always-on systems,
robotics, edge devices, adaptive sensors, and autonomous agents that must learn
continuously under limited power. Do **not** invent numeric speed/energy/memory figures —
name the *sources* of expected efficiency and flag physical benchmarking as future work.

## 18 · What is predicted vs. what is measured
The one honest caveat, deliberately isolated so it does not turn the talk into a list of
disclaimers. Left column — architectural predictions: the composition, imagination,
reasoning, and scaling described here are what the architecture is *designed to enable*;
they follow from its principles and describe the long-term hypothesis. Right column —
measured today: the working system is a small, from-scratch spiking network learning 3×3
line primitives with local rules and no gradients; concrete speed, energy, and memory
advantages on neuromorphic hardware are *future validation*, stated as expected sources
of gain rather than benchmarked numbers. The gap between the columns is the research
program itself.

## 19 · Final message
Close on the thesis verbatim: *Instead of training one enormous model to approximate
everything it has seen, we can build intelligence from local systems that learn
continuously, form explicit symbols, compose new possibilities, trace their own
reasoning, and recognize the boundaries of their knowledge.*

## A1 · Sources & grounding
Point to the repository documents the deck is built on and the scientific motifs it
references. Reiterate that the motifs are cited as biological inspiration, not as a claim
of faithful simulation of any specific cell type. See `SOURCES.md` for the full list.
