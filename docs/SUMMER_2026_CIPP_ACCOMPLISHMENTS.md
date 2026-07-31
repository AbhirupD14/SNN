# CIPP Summer 2026 Technical Accomplishments

## Simulator, learning algorithms, initialization, cortical columns, and coincidence cells

**Purpose.** This document captures the technical work completed during the summer CIPP effort in enough detail for the team to build executive, programmatic, or technical presentation slides. It distinguishes what was implemented and measured from what remains a hypothesis or open research problem.

**Period covered.** The repository history used for this review runs from June 25 through July 29, 2026. In roughly one month of focused work, the project progressed from a small vectorized spiking-network experiment to a composable cortical-column fabric, a validated hybrid event-resolved simulator, new local learning rules, and a NEST/NESTML portability prototype.

**Prepared:** July 30, 2026.

---

## 1. Executive summary

The summer effort produced five connected accomplishments:

1. **A purpose-built simulator for causal, event-driven neural circuits.**  
   The team moved beyond fixed time-step vector updates to an event-resolved engine that finds analytical within-boundary spike times, preserves causal ordering, supports same-time feedback and resets, records replayable state, and exposes the internal variables needed to understand learning.

2. **A family of local learning algorithms developed specifically for CIPP.**  
   The learning work evolved from explicit selective maturation and global competition toward local, state-dependent feedforward and feedback plasticity. The current FE/FES formulation uses only information available at the neuron and synapse; it does not require labels, backpropagation, or a global error signal.

3. **A progression from generic random initialization to role-aware initialization.**  
   Early networks used small or broad random weights. Ablations showed that sparse, normalized initialization improved representational tiling but could not solve allocation by itself. The current scheme initializes each cell type according to its causal role: ordinary excitatory cells begin below one-volley threshold, relay cells transmit immediately, and coincidence cells begin on a deliberately selected portion of their learning curve.

4. **A reusable cortical-column architecture that composes into hierarchies.**  
   The project formalized a column containing ordinary excitatory cells, a relay/output cell, a coincidence cell, and an inhibitory reset cell. Builders now assemble these columns into tiled and multi-level fabrics. Experiments demonstrated patch isolation, local winner formation, causal feedforward/feedback paths, and deterministic construction at larger graph sizes.

5. **The architectural discovery of coincidence cells as a required primitive.**  
   The team found that an ordinary charge-accumulating neuron could not distinguish causally aligned feedforward and feedback evidence from unrelated evidence accumulated at different times. This led to a compartmental coincidence cell that implements a temporally constrained basal-and-apical condition and converts matched context into local inhibitory feedback.

Together, these accomplishments are more important than any single toy-model result. They establish the beginnings of a research fabric: the simulation semantics, local adaptation rules, structural primitives, and experimental controls required to study continuous online learning and compositional processing.

### Executive wording for a slide

> In roughly one month, we built the foundations of a new event-driven learning fabric: a validated simulator, local learning rules, role-aware initialization, composable cortical columns, and a coincidence mechanism that links bottom-up evidence with top-down context.

---

## 2. What was built, what was demonstrated, and what is not yet claimed

| Area | Built | Demonstrated | Not yet claimed |
|---|---|---|---|
| Simulator | Hybrid event-resolved engine, graph specification, analytical spike timing, replay/state capture, dashboard, test harnesses | Causal scheduling, event conservation, deterministic replay, analytical timing agreement with an independent numerical reference | A pure global discrete-event simulator, neuromorphic hardware performance, or production-scale distributed execution |
| Learning | Selective-maturation experiments, local excitatory and coincidence plasticity, FE/FES state-dependent updates | Turnover and recall in small circuits; local adaptation without labels, backpropagation, or global error; sensitivity to learning-rate and equation choice | General-purpose lifelong learning, catastrophic-forgetting elimination, or broad task performance |
| Initialization | Random, normalized, sparse, diversity, orthogonal, low-discrepancy, balanced, and role-aware schemes | Initialization materially changes participation and representational tiling; sparse-normalized initialization outperformed the original broad-random scheme on the tested tiling metrics | Initialization alone solves ownership, allocation, stability, or compositional learning |
| Cortical columns | Reusable column builders, tiled 3×3 hierarchy, direct-identity and multi-tower variants | Local isolation, independent lower-level winners, causal paths through multiple cell roles, deterministic topology assembly | Arbitrary-depth semantic composition or demonstrated large-scale computational scalability |
| Coincidence cells | Basal/apical compartments, one-boundary availability, immediate causal deposit, local feedback/reset path | Exact coincidence cadence in isolated tests, same-time causal feedback in integrated circuits, improved turnover behavior under the retained nonlinear rule | A calibrated uncertainty measure or proof that the mechanism is biologically complete |

This distinction should remain visible in presentations. The project produced strong engineering and architectural evidence, while the larger claims motivating CIPP remain research objectives.

---

## 3. Development timeline

| Period | Milestone | Why it mattered |
|---|---|---|
| June 25 | Selective Maturation Race network | Established the initial problem: allocate different inputs to different neurons while allowing learned owners to stabilize |
| June 29–30 | Interactive visualization and real-time simulation backend | Made membrane activity, spikes, weights, and competition observable during development |
| July 1–2 | Inhibitory-discharge plasticity, adaptive inhibition, confidence traces, and homeostatic experiments | Exposed the tension between winner stability and broad participation |
| July 6–8 | Signed plasticity and initialization ablations | Showed that learning behavior depended strongly on both negative evidence and the starting weight geometry |
| July 10–14 | Modular engine refactor, golden behavior tests, timing and initialization studies | Separated membrane, synapse, and learning concerns while preserving reproducibility |
| July 15–21 | Local coincidence feedback, event-resolved scheduling, and coincidence pyramidal cells | Added a causal feedback primitive and forced the simulator to represent within-boundary timing correctly |
| July 21 | Reusable and tiled cortical columns | Converted a one-off circuit into a composable architecture |
| July 21–23 | Linear excitatory learning study and dual FE/FES implementation | Replaced uniform update magnitude with state-dependent neuronal and synaptic learning factors |
| July 24 | Independent engine validation | Quantified timing accuracy and explicitly documented the simulator’s approximation boundary |
| July 28 | Direct-identity, feedback-cadence, and two-tower experiments | Demonstrated hierarchy construction and identified representation loss at a narrow relay interface |
| July 29 | Next-event specification and NEST/NESTML prototype | Defined the path toward absolute timestamp DES semantics and tested portability to an established simulator |

---

# Part I — Simulator accomplishments

## 4. Starting point: a useful experiment, not yet an engine

The first implementation was a vectorized NumPy spiking layer. It was sufficient to explore a selective-maturation race:

- feedforward weights were initialized randomly and normalized;
- inputs drove a population of competing neurons;
- lateral inhibition suppressed alternatives;
- self-excitation reinforced an emerging winner;
- Hebbian updates strengthened active input-to-winner connections; and
- explicit maturity state froze or retired learned neurons.

That implementation answered early conceptual questions, but it could not faithfully represent the causal interactions required by the later architecture. In particular, a single synchronous update obscures whether two events occur together, before one another, or after a reset within the same logical interval.

## 5. Simulator architecture developed during the summer

The simulator grew into several cooperating layers:

### 5.1 Declarative network specification

Networks are described as typed nodes and edges instead of being hard-coded into a single update loop. This supports:

- arbitrary graph construction;
- named cell roles;
- explicit delays and edge semantics;
- topology presets and reusable builders;
- serialization and inspection; and
- experimental variants without rewriting the engine.

This abstraction was essential for cortical columns. The same engine can execute a single isolated cell, one column, a tiled 3×3 fabric, or a multi-tower hierarchy.

### 5.2 Modular execution components

Membrane dynamics, synaptic behavior, learning strategies, topology construction, and recording were separated. A golden-behavior harness was used during this refactor to ensure that structural code changes did not silently alter established simulation behavior.

This is an important engineering accomplishment: the team did not merely add features to a monolithic script. It created replaceable components and regression controls suitable for continued research.

### 5.3 Hybrid event-resolved timing

The current custom simulator is best described as a **hybrid event-resolved engine**:

- outer integer boundaries retain the network’s logical delivery cadence;
- within each boundary, spike time is represented by an analytical offset \(\tau \in [0,1]\);
- the engine solves the membrane trajectory in closed form;
- it selects the earliest valid threshold crossing;
- all membranes advance to that time;
- same-time consequences such as relay transmission, apical permission, and hard reset are processed;
- invalidated candidate events are discarded and recomputed; and
- this repeats until no valid event remains in the boundary.

This was a meaningful shift from “update all neurons once per step” to “resolve the causal sequence of events that occurs inside the step.”

### 5.4 State visibility and replay

The custom engine preserves research-facing state that is often difficult to recover from a compiled simulator:

- current membrane charge;
- pre-reset accumulated charge;
- threshold and refractory state;
- individual synaptic weights;
- FE and FES learning factors;
- pending events and their sources;
- winner ownership;
- basal and apical coincidence state;
- hard-reset causes; and
- boundary and within-boundary spike times.

Recorder and replay tooling allows a run to be inspected after execution and supports branching from saved state. This is particularly valuable for answering “why did this neuron win?” rather than only observing that it spiked.

## 6. Why custom event semantics became necessary

The learning architecture created timing questions that an ordinary synchronous loop could not answer:

1. Did basal evidence arrive before or after apical permission?
2. Did a coincidence cell fire at the time of the permitting parent spike or one boundary later?
3. Did inhibition reset a competitor before its scheduled threshold crossing?
4. If two threshold crossings share a timestamp, which deterministic ordering applies?
5. Which pre-reset charge should control the learning update?

These are not visual details. Different answers produce different learning trajectories. The engine therefore became part of the scientific method: it had to make the causal semantics explicit and testable.

## 7. Engine validation

The engine was validated against independent references and conservation checks. The principal results were:

- the analytical membrane segment solution agreed with a high-resolution RK4 numerical solution;
- a priority-queue reference reproduced the complete spike trace of the tested network;
- the maximum observed spike-time difference in that comparison was approximately \(2.28\times10^{-10}\);
- delivered, consumed, delayed, and discarded events were reconciled;
- FE/FES state was captured at the intended pre-reset and pre-update points;
- replay was deterministic under the declared event-ordering rule; and
- same-time ties were recognized as order-sensitive and documented explicitly.

The validation conclusion was deliberately bounded: **the hybrid engine is validated under its declared approximation.** It is not presented as a pure global DES because the outer boundary remains part of the model.

See [ENGINE_VALIDATION_REPORT.md](ENGINE_VALIDATION_REPORT.md) for the validation methodology and findings.

## 8. Dashboard and experimental workflow

The simulator was paired with a real-time dashboard and structured experiment outputs. This allowed the team to:

- visualize graph topology;
- observe spikes and membrane state;
- inspect weights and learning factors;
- start, pause, and reset controlled runs;
- compare presets and parameter variants;
- export run artifacts; and
- reproduce acceptance tests from saved configuration.

The visualization was not merely a demonstration layer. It shortened the cycle between conceptual change, implementation, observation, and correction.

## 9. NEST/NESTML portability investigation

Late in the effort, the team also built a small NEST/NESTML version of the 3×3 topology. The purpose was to test whether CIPP could retain its node behavior while using an established simulator’s clock, delay handling, threading, and event infrastructure.

The prototype established that:

- the topology and several node behaviors can be represented in NEST/NESTML;
- NEST can own the simulation clock and message delivery;
- event-only models can be used without adopting a conventional continuous ODE neuron as the conceptual core; and
- simulation data can be exported to an offline dashboard.

It also identified integration constraints:

- NEST’s minimum-delay semantics alter the immediate winner/feedback loop;
- the exact FE/FES update requires carefully captured pre-reset neuronal state and pre-update synaptic state;
- compiled node models expose less internal state by default than the custom research engine; and
- the local NEST build used for the prototype did not provide MPI execution.

The NEST work should be presented as a portability and infrastructure study, not as a replacement for the validated custom semantics. It clarified which parts of CIPP are architectural and which currently depend on custom execution behavior.

## 10. Slide-ready simulator takeaways

**Possible slide headline:**  
**We had to build the instrument before we could test the theory.**

**Sparse points:**

- Progressed from synchronous vector updates to causal, event-resolved execution.
- Preserved the internal state required to explain every winner, update, coincidence, and reset.
- Validated analytical spike timing against an independent numerical reference.
- Built replay, topology, dashboard, and experiment infrastructure around the engine.
- Defined a path toward absolute timestamp DES and tested portability to NEST.

**Suggested visual:** A three-stage progression:

`vectorized step model → hybrid event-resolved engine → future absolute-timestamp DES / NEST integration`

---

# Part II — Learning algorithm accomplishments

## 11. The learning problem

The central learning challenge was not simply to make a neuron spike. It was to achieve all of the following with local mechanisms:

- allow an uncommitted neuron to become responsive;
- allocate different patterns to different owners;
- strengthen causally relevant inputs;
- weaken or ignore irrelevant inputs;
- protect consolidated knowledge;
- permit a new candidate to replace an inappropriate incumbent;
- recover a previous owner when its pattern returns; and
- do this without labels, backpropagation, or a global loss.

These objectives conflict. Strong positive feedback creates stable owners but can create a “tyrant” that wins every pattern. Strong competition encourages participation but can destroy recall. Much of the summer work mapped this stability–plasticity frontier.

## 12. Learning evolution

### 12.1 Selective maturation

The first approach combined:

- homeostatic boosting for neurons that had not won;
- lateral inhibition to sharpen a race;
- self-excitation to sustain an emerging winner;
- Hebbian strengthening;
- feedforward normalization; and
- an explicit maturity mask that froze established neurons.

This demonstrated that a curriculum could allocate patterns, but it depended on explicit state transitions and broadly coordinated competition.

### 12.2 Emergent local-competition experiments

The next generation explored inhibitory-discharge plasticity, adaptive lateral inhibition, homeostatic scaling, confidence traces, and signed updates. These experiments were valuable even where a mechanism was rejected. They showed that:

- a stable winner is easy to create;
- diverse ownership is harder;
- participation and consolidation must be balanced;
- input timing can unintentionally decide the race;
- inactive inputs must sometimes provide negative evidence; and
- initialization cannot be evaluated separately from the learning rule.

### 12.3 State-dependent excitatory learning

The project compared linear and nonlinear forms for ordinary excitatory learning. In the tested convergence experiment, the linear excitatory rule reached 90%, 95%, and 99% of threshold faster than the earlier quadratic form:

| Target | Linear excitatory rule | Quadratic excitatory rule |
|---|---:|---:|
| 90% of threshold | 111 events | 131 events |
| 95% of threshold | 133 events | 168 events |
| 99% of threshold | 186 events | 260 events |

Across 32 tested seeds, both variants passed the defined turnover and recovery gates. The linear form was promoted for ordinary excitatory cells because it converged faster without failing those gates.

The same simplification was **not** promoted for coincidence cells. Removing the retained coincidence-cell nonlinearity reduced turnover in the tested ablation from 1.00 to approximately 0.60 and allowed an incumbent to persist much longer. This is an example of the team using cell role, rather than uniformity, to choose a learning law.

### 12.4 Dual FE/FES learning

The current formulation separates learning into two local state factors:

- **FE**, a neuron-side factor based on pre-reset accumulated charge; and
- **FES**, a synapse-side factor based on the synapse’s pre-update weight.

A generic update has the form

\[
\Delta w_i = \eta \cdot FE \cdot FES_i \cdot s_i \cdot \phi_i
\]

where:

- \(w_i\) is the weight of afferent \(i\);
- \(\eta\) is the cell-class learning rate;
- \(FE\) is the neuron’s state-dependent learning factor;
- \(FES_i\) is the synapse’s state-dependent learning factor;
- \(s_i\) is the signed evidence associated with the input; and
- \(\phi_i\) is the causal eligibility or activity term.

The factors are intentionally nonlinear. Their qualitative behavior is:

- very weak or irrelevant state produces a small update;
- an intermediate state produces the largest plastic change;
- a consolidated state becomes progressively protected; and
- overshoot does not create unbounded learning.

The ordinary excitatory and coincidence-cell curves use different centers because the two cell classes play different causal roles.

The complete equations and variable definitions are in [Current_Implementation_Methodology_Equations.md](../Current_Implementation_Methodology_Equations.md).

## 13. Evidence from the dual FE/FES experiments

The direct four-candidate experiment showed that the equation shape alone was not enough; the effective learning rate mattered.

- The low-rate reference configuration failed to allocate four distinct owners in all eight tested seeds, generally producing only two or three owners.
- The higher-rate candidate passed the defined allocation test in all eight tested seeds:
  - four distinct owners;
  - turnover on every pattern switch;
  - consistent owner return on recall; and
  - equal input-pattern frequency.
- A long-dwell experiment retained complete dominance on the learned pattern and turned over when a novel pattern arrived.
- Disabling the intended learning path collapsed the network to a single owner, providing a useful negative control.

These results support a bounded claim: **local FE/FES learning can produce turnover and recall in the tested small circuit when its rate is sufficient.** They do not yet establish general-purpose continual learning.

## 14. What is novel about the learning work

The project’s novelty should be described precisely. The work does not claim to have invented local plasticity, spiking neurons, dendritic compartments, or predictive feedback individually.

The project-specific contribution is the integration of:

- role-specific cells;
- explicit causal event timing;
- state-dependent neuron and synapse learning factors;
- local feedforward and feedback plasticity;
- coincidence-gated inhibitory feedback; and
- composable column topology

into one experimental fabric designed for online, unlabeled adaptation.

This is stronger and more defensible than saying that every individual mechanism is unprecedented.

## 15. Slide-ready learning takeaways

**Possible slide headline:**  
**We developed local learning rules around the state of both the neuron and the synapse.**

**Sparse points:**

- No labels, backpropagation, or global error signal are required by the update.
- FE answers: “How plastic should this neuron be in its current state?”
- FES answers: “How plastic should this particular synapse be at its current strength?”
- Cell roles use different learning curves rather than one rule for the entire network.
- Controlled ablations identified which nonlinearities helped turnover and which could be simplified.

**Suggested visual:** Plot FE and FES as conceptual bell-shaped plasticity curves with three labeled regions: `uncommitted`, `most plastic`, and `consolidated`.

---

# Part III — Initialization progress

## 16. Why initialization became a research question

At first, weights were treated mainly as random starting conditions. The experiments showed that initialization changes the geometry of the competition:

- dense random vectors begin highly similar to one another;
- a small weight advantage can be amplified into permanent ownership;
- different total incoming weight gives some neurons an unfair starting position;
- different total outgoing weight makes some input dimensions disproportionately influential; and
- overly sparse initialization can leave most neurons permanently inactive.

Initialization therefore became part of the architecture, not an implementation detail.

## 17. Initialization progression

### Stage 1 — Random weights with post-normalization

The earliest selective-maturation model sampled each fresh neuron’s weights from a broad random range and then normalized the column to a fixed total. This created competition among different random directions while equalizing gross input magnitude.

### Stage 2 — Small broad random weights

The graph-based simulator initially used fixed-point weights in an approximate range of 5%–20% of the excitatory threshold. This was simple and symmetric in expectation, but individual neurons and input dimensions could still begin with substantial accidental advantages.

### Stage 3 — Removing weight noise from timing experiments

For some early dashboard experiments, selected weights were initialized to the same constant value. This intentionally removed weight randomness so the team could determine whether input timing and delay alone were choosing the first winner.

This was an important experimental practice: initialization was controlled to isolate a different causal variable.

### Stage 4 — Initialization ablation suite

The team compared:

- uniform random;
- uniform with normalization;
- sparse random;
- sparse with normalization;
- diversity rejection;
- nonnegative orthogonal;
- low-discrepancy; and
- later, balanced row-and-column initialization.

The three-seed initialization ablation produced the following results:

| Scheme | Initial cosine similarity | Sustained dominance | Distinct owners out of 8 | Dead neurons | Final cosine similarity |
|---|---:|---:|---:|---:|---:|
| Uniform baseline | 0.90 | 0.552 ± 0.050 | 4.00 ± 0.82 | 0.33 | 0.41 |
| Uniform normalized | 0.90 | 0.548 ± 0.070 | 3.67 ± 0.94 | 0.00 | 0.44 |
| Sparse, \(k=3\) | 0.44 | 0.167 | 1.33 | 6.67 | 0.44 |
| Sparse normalized | 0.44 | 0.433 ± 0.020 | **5.67 ± 0.47** | **0.00** | **0.25** |
| Diversity rejection | 0.82 | 0.498 | 3.67 | 0.00 | 0.35 |
| Nonnegative orthogonal | 0.84 | 0.500 | 3.67 ± 0.94 | 0.00 | 0.40 |
| Low-discrepancy | 0.89 | 0.525 | 3.67 | 0.00 | 0.40 |

The major findings were:

1. **Sparsity improved diversity.**
2. **Normalization was load-bearing.** Sparse initialization without normalization was catastrophic in this test, leaving most neurons dead.
3. **Sparse-normalized initialization produced the best tested representational tiling.**
4. **Better tiling did not automatically produce stable holding or correct long-term allocation.**

### Stage 5 — Balanced initialization

The balanced experiment equalized:

- every neuron’s total incoming weight; and
- every input dimension’s total outgoing weight,

while retaining a narrow random jitter. A Sinkhorn-style balancing procedure was used to remove both neuron-side and input-side starting advantages.

The central result was negative but useful: a mathematically fair start did not solve allocation. Competition and learning dynamics still determined whether distinct ownership emerged.

### Stage 6 — Current role-aware initialization

The current system no longer gives every cell class the same generic random distribution. Initialization is derived from each cell’s intended function.

Assuming an excitatory threshold \(\theta_E=1000\):

| Cell/edge role | Current initialization | Intended behavior |
|---|---:|---|
| Ordinary feedforward excitatory afferent | \(\theta_E/4\) with approximately ±4% jitter; nominally 250 | Three active inputs provide about \(0.75\theta_E\), so a fresh cell normally integrates across more than one delivery rather than winning instantly |
| Eor relay afferent | \(\theta_E\), no jitter, frozen | One local winner can drive the relay immediately and deterministically |
| Coincidence-cell basal afferent | \(\theta_E/4\), no jitter | Begins on the rising portion of the C-specific plasticity curve and can mature toward one-shot confirmation |
| Coincidence-cell apical input | Unweighted Boolean permission | Supplies context/timing, not accumulated synaptic magnitude |
| Inhibitory/reset paths | No learned weight in the same sense | Enforce structural inhibition or hard-reset semantics |

The current caps are also role-specific:

- a structural detector capped at \(\theta_E/2\) requires at least two supporting inputs;
- a relay or mature coincidence path capped at \(\theta_E\) can eventually transmit from one causally sufficient input.

This is the key progression:

> Initialization moved from “random numbers small enough not to explode” to an explicit statement about what evidence each cell should require before it can act.

## 18. Slide-ready initialization takeaways

**Possible slide headline:**  
**Initialization evolved from random starting weights into encoded architectural priors.**

**Sparse points:**

- Early random weights unintentionally decided who could participate.
- Ablations showed that sparsity helps only when total weight is normalized.
- A perfectly fair starting matrix still could not replace effective learning dynamics.
- Current values are chosen from cell role and threshold semantics.
- Fresh detectors integrate; relays transmit; coincidence cells mature toward confirmation.

**Suggested visual:** A left-to-right progression:

`small random → normalized/sparse ablations → balanced fairness → role-aware initialization`

Below it, show the ordinary cell at `0.25θ`, the relay at `1.0θ`, and the coincidence cell at `0.25θ → 1.0θ`.

---

# Part IV — Cortical-column development

## 19. From layers to a reusable column

Early networks were described as fixed L1 and L2 layers. The cortical-column work replaced this with a reusable local motif.

A column contains:

- **E cells:** ordinary excitatory candidates that compete to represent input;
- **Eor:** an excitatory output/relay that signals that the column has a winner;
- **C:** a coincidence cell that associates local bottom-up evidence with parent/top-down context; and
- **I:** an inhibitory cell that resets or suppresses the local excitatory population.

The canonical local and inter-column paths are:

```text
Local sensory or child input
          │
          ▼
       [ E pool ] ─────► [ Eor ] ─────► parent E pool
          │                │
          │                └── basal ──► [ C ]
          │                               ▲
          ▼                               │ apical
         [ I ] ◄──────────────────────────┘
          │
          └──────── hard reset ─────────► local E pool
```

At a non-top-level column, the parent winner supplies apical permission to the child’s coincidence cell. The child’s local Eor supplies basal evidence. When those signals align causally, C can activate I and close the local feedback loop.

The top-level C has no parent apical source and is intentionally dormant.

## 20. Reusable construction APIs

The architecture was encoded in builders rather than copied graph definitions:

- a builder creates the internal cell groups and edges of one column;
- a patch connector attaches sensory/RGC inputs;
- a column connector creates feedforward and feedback relationships; and
- tiled presets assemble complete hierarchies.

This makes depth and width architectural parameters. It also creates a stable interface for future engine and NEST implementations.

## 21. The 3×3 tiled topology

The principal tiled experiment divides a 9×9 input field into nine 3×3 patches:

```text
9×9 input field
  ├─ 3×3 patch → L1 column 1 ┐
  ├─ 3×3 patch → L1 column 2 │
  ├─ 3×3 patch → L1 column 3 │
  ├─ 3×3 patch → L1 column 4 │
  ├─ 3×3 patch → L1 column 5 ├─► one L2 parent column
  ├─ 3×3 patch → L1 column 6 │
  ├─ 3×3 patch → L1 column 7 │
  ├─ 3×3 patch → L1 column 8 │
  └─ 3×3 patch → L1 column 9 ┘
```

With eight ordinary E candidates per column, the preset contains:

- 191 nodes; and
- 1,052 edges.

For that construction, the graph scales according to:

\[
N_{nodes}=10N+111
\]

\[
N_{edges}=129N+20
\]

where \(N\) is the number of ordinary E candidates per column.

These equations describe construction growth, not yet runtime scaling on arbitrary workloads.

## 22. Tiled-column results

The tiled acceptance artifact demonstrated:

- stimulation of the center patch activated only its corresponding L1 column;
- the intended L1 winner drove its Eor;
- the Eor supplied feedforward evidence to L2;
- a parent winner returned apical permission to the appropriate child C;
- coincidence activity generated local inhibitory/reset behavior;
- no unintended cross-column reset occurred;
- independent lower patches could produce simultaneous local winners; and
- the top-level C remained dormant without a parent.

All 11 defined acceptance checks passed in the recorded tiled experiment.

This established the structural fabric and its causal isolation. It did not yet establish high-level object learning across arbitrary patterns.

## 23. Direct-identity and feedback-cadence variants

### Direct identity

A classic one-cell Eor compresses all local winners into the same outgoing identity. That is compact, but a higher layer cannot tell which child candidate won.

The direct-identity variant preserves the winner address at the interface. In the tested two-pattern sequence:

- 12 of 12 ordered runs across three seeds produced a stable owner;
- pattern switches produced turnover;
- returning patterns recalled their prior owner;
- every owner was supported by direct evidence; and
- the first coincidence association was not incorrectly depressed.

This variant improved representational identifiability at the cost of greater edge density.

### Feedback cadence

The team also tested whether feedback timing followed graph latency. When the input period was set from the loop latency \(L\), the system produced the expected alternating cadence for \(L=2,3,4\), passing 12 of 12 seeds at each tested latency.

This showed that the cadence emerged from the causal graph rather than from a hard-coded special case.

It should not yet be described as a calibrated certainty signal. It is currently evidence that the loop timing is structurally meaningful and observable.

## 24. Two-tower hierarchy and an important negative result

The two-tower experiment expanded the fabric to:

- a 9×18 input;
- 18 L1 columns;
- two L2 columns; and
- one L3 column.

The graph contained 393 nodes and 2,162 edges.

The lower towers successfully learned their tested patterns:

- distinct L2 owners emerged for the lower-level patterns; and
- cold recall succeeded in six of six checks.

The L3 task failed for an informative reason. Each tower’s single Eor collapsed different lower-level winners into the same outgoing symbol. The L3 input trace was therefore identical for distinct lower-level combinations. No learning rule at L3 could recover information that the interface had erased.

The experiment was labeled **representation not identifiable**.

This was a valuable architectural discovery:

> Hierarchical composition requires interfaces that preserve the identity needed by the next level; successful lower-level learning is not enough.

The direct-identity work is one response to this bottleneck.

## 25. Slide-ready cortical-column takeaways

**Possible slide headline:**  
**We turned a one-off network into a composable cortical-column fabric.**

**Sparse points:**

- Each column combines competition, relay, coincidence, and local inhibitory reset.
- The same builders create isolated columns, a 3×3 tiled hierarchy, and multi-tower systems.
- Tiled tests demonstrated patch isolation and causal feedforward/feedback paths.
- Scaling exposed a representation bottleneck: a narrow relay can erase winner identity.
- The negative result directly motivated an identity-preserving interface.

**Suggested visual:** Use the four-cell-role column diagram, then repeat it nine times beneath a parent column. On a second build, show two lower towers converging on L3, with the single-Eor bottleneck highlighted.

---

# Part V — Discovery and development of coincidence cells

## 26. The problem ordinary neurons could not solve

An ordinary accumulating neuron answers:

> “Has enough charge arrived?”

The CIPP feedback loop needed to answer a different question:

> “Did this specific bottom-up event and its corresponding top-down context occur together within the allowed causal window?”

If basal and apical evidence simply add to one membrane, the neuron can cross threshold after receiving unrelated evidence at different times. The total charge may be the same even though the causal meaning is different.

This distinction led to the project’s coincidence-cell primitive.

## 27. Coincidence-cell semantics

The coincidence pyramidal cell has separate input roles:

- **Basal input:** learned bottom-up evidence identifying the local winner.
- **Apical input:** unweighted top-down permission or context from a parent winner.

Its key rules are:

1. Basal availability may be current or carried for exactly the declared short window.
2. Apical input does not add ordinary weighted charge.
3. A valid basal-and-apical match deposits the basal contribution once.
4. The deposit occurs at the analytical time of the permitting event.
5. The cell may fire only while the coincidence gate is valid.
6. Learning is applied only to the causal basal source.
7. Unrelated basal owners are not punished merely because another association occurred.

This is closer to a temporally constrained logical AND than to a conventional summing neuron.

## 28. Why the coincidence cell changed the simulator

The first coincidence implementation exposed a phase error: if apical permission was only recognized at the next outer boundary, the cell’s action was delayed and the feedback loop represented the wrong causal order.

The corrected implementation commits the coincidence at the permitting parent spike’s within-boundary time. That requirement drove:

- event-resolved spike scheduling;
- same-time event handling;
- recomputation after resets;
- explicit basal availability windows; and
- source-aware learning records.

The cell and the engine therefore co-evolved. The architecture demanded better timing semantics, and the improved timing semantics made the architecture testable.

## 29. Coincidence-cell experimental evidence

An isolated cadence test delivered 12 valid coincidence opportunities and produced:

- an exact alternating `[0,1]` readiness/spike pattern repeated six times;
- six coincidence spikes; and
- a deterministic 2:1 opportunity-to-spike cadence under that configuration.

In an early 4,000-boundary integrated preset, the recorded artifact contained:

- 6,195 valid coincidences;
- 5,058 coincidence-cell spikes and associated hard resets;
- 5,058 spikes at the intended same-time phase;
- zero phase mismatches; and
- deterministic reproduction.

The observed coincidence firing frequency declined from approximately 0.84 in the first analysis window to approximately 0.50 in the last, showing that the feedback behavior changed as the system state evolved.

These figures document an important development milestone. Later changes to initialization, immediate deposit semantics, and C-specific learning altered the current exact contract, so the numbers should not be presented as universal performance specifications.

## 30. Current function in the fabric

In the current column, the coincidence cell links:

```text
local winner identity
        +
parent contextual confirmation
        ↓
local inhibitory feedback / reset
```

This makes C a candidate building block for prediction confirmation and eventually for exposing uncertainty. However, a firing cadence or coincidence rate is not yet a calibrated probability or confidence score.

The defensible current statement is:

> The fabric explicitly represents whether bottom-up evidence received matching top-down context, giving future work an observable substrate from which uncertainty measures may be developed.

## 31. Why this counts as a project discovery

“Discovery” here means an architectural discovery within the CIPP development process:

- the original neuron abstraction was insufficient;
- a specific causal ambiguity was identified;
- a compartmental primitive was designed to resolve it;
- the simulator was changed to execute it faithfully; and
- experiments verified its timing and local effect.

It should not be framed as the discovery of coincidence detection in neuroscience, which is an established concept.

## 32. Slide-ready coincidence-cell takeaways

**Possible slide headline:**  
**Summed evidence is not the same as coincident evidence.**

**Sparse points:**

- Ordinary accumulation loses the distinction between “together” and “at different times.”
- The C cell separates learned basal evidence from unweighted apical context.
- A valid causal match triggers local confirmation and inhibitory feedback.
- Correct implementation required same-time, event-resolved simulator semantics.
- The mechanism creates an observable basis for future prediction and uncertainty work.

**Suggested visual:** Show two cases:

```text
Case A: basal ───── apical      → no valid match
Case B: basal + apical together → coincidence → C spike → local reset
```

---

# Part VI — Cross-cutting significance

## 33. The accomplishments reinforce one another

The five workstreams should not be presented as separate feature lists. Each answered a question created by the others:

```text
Learning rules require exact pre-reset state
                  ↓
The simulator must expose causal event timing
                  ↓
Coincidence becomes implementable and testable
                  ↓
Coincidence closes feedback inside a column
                  ↓
Columns can be composed into a hierarchy
                  ↓
Hierarchy exposes identity and timing bottlenecks
                  ↓
Initialization and interfaces become explicit architectural priors
```

This iterative loop is arguably the main summer accomplishment. The team created enough of the stack to let theory, execution, observation, and experiment correct one another.

## 34. Scientific and engineering practices established

The work also established practices needed for continued progress:

- **Ablation before promotion.** Alternative equations and initializations were compared rather than accepted from intuition.
- **Negative results were retained.** Sparse-only initialization, low-rate FE/FES allocation, and the L3 two-tower failure each revealed a design constraint.
- **Deterministic artifacts.** Seeds, run outputs, replay state, and acceptance checks were retained.
- **Explicit invariants.** Event conservation, phase, topology isolation, and role-specific constraints were tested.
- **Separation of claim levels.** Structural scalability, runtime scalability, learning success, and broad intelligence were not treated as synonyms.

---

# Part VII — Limitations and next work

## 35. Known limitations

The following limitations should appear in technical or management backup material:

1. **The custom simulator is hybrid, not yet pure timestamp DES.**  
   It resolves events analytically inside outer logical boundaries. Boundary refinement is not currently a supported convergence operation.

2. **The strongest learning results are still small-circuit results.**  
   Turnover and recall have been demonstrated under controlled sequences, not across a broad benchmark suite.

3. **Single-winner ownership remains a central design assumption.**  
   This helps make causal learning interpretable but may limit distributed representation.

4. **Relay compression can erase identity.**  
   The two-tower experiment proved that a higher layer cannot learn distinctions absent from its input interface.

5. **Feedback timing constrains input timing.**  
   Input cadence must respect causal loop latency unless the architecture is changed to tolerate overlap.

6. **Uncertainty is observable but not calibrated.**  
   Coincidence and feedback expose prediction agreement, but they are not yet validated confidence probabilities.

7. **NEST integration is incomplete.**  
   The main blockers are exact FE/FES timing, immediate feedback semantics, and access to research-facing internal state.

8. **Hardware efficiency has not been demonstrated.**  
   The implementation runs on hardware optimized for dense binary arithmetic. Claims about neuromorphic efficiency remain prospective.

## 36. High-value next steps

- Implement the specified absolute-timestamp next-event engine and compare it event-for-event with the hybrid engine.
- Define a formal event-ordering contract for simultaneous spikes, reset, relay, and learning.
- Preserve winner identity through scalable sparse interfaces rather than dense direct identity.
- Test FE/FES across larger pattern sets, longer retention intervals, skewed exposure, and interference protocols.
- Quantify continuous adaptation with explicit stability, plasticity, recall, and forgetting metrics.
- Convert coincidence/feedback traces into a candidate uncertainty measure and test calibration.
- Complete a custom NEST node/synapse implementation or a trace-replay bridge that exposes equivalent internal state.
- Map the event workload to hardware assumptions and measure sparsity, event rate, memory traffic, and energy proxies.

---

# Part VIII — Presentation-building kit

## 37. Recommended 10-slide technical brief

| Slide | Headline | Core content | Suggested visual |
|---|---|---|---|
| 1 | What we accomplished in one month | Five accomplishments and the integrated-fabric message | Five connected blocks |
| 2 | The problem required a new experimental instrument | Why causal learning could not be evaluated in a simple synchronous loop | Step update versus ordered event timeline |
| 3 | Simulator progression | Vector model → graph engine → event-resolved timing → NEST study | Development timeline |
| 4 | A validated causal simulator | Analytical timing, conservation, replay, internal state | Reference-versus-engine spike trace |
| 5 | Local learning without global error | FE, FES, signed evidence, causal eligibility | Conceptual FE/FES curves |
| 6 | Initialization became an architectural prior | Random → sparse-normalized → balanced → role-aware | Four-stage progression and threshold bars |
| 7 | The cortical column | E, Eor, C, and I roles | Column circuit diagram |
| 8 | Columns compose—and interfaces matter | 3×3 tiled success and two-tower identity bottleneck | Tiled hierarchy plus highlighted bottleneck |
| 9 | Coincidence was the missing primitive | Together versus accumulated-at-different-times | Basal/apical timing comparison |
| 10 | What the work enables next | Timestamp DES, identity-preserving hierarchy, calibrated uncertainty, broader learning tests | Roadmap with evidence gates |

## 38. Recommended five-slide executive brief

If time is limited to approximately five minutes:

| Slide | Message |
|---|---|
| 1 | In roughly one month, we built the foundations of a different kind of learning fabric rather than modifying an existing AI model. |
| 2 | The dynamics required a custom event-resolved simulator that preserves causal timing and internal state. |
| 3 | We developed local learning and role-aware initialization that support turnover and recall in controlled circuits without labels or backpropagation. |
| 4 | We composed the cell roles into cortical columns and a tiled hierarchy, then used larger experiments to discover an identity bottleneck. |
| 5 | The discovery of coincidence cells connected bottom-up evidence with top-down context and established the basis for future online adaptation and uncertainty work. |

## 39. Key quantitative callouts

Use these numbers only with their scope attached:

- **~1 month:** focused period in which the core fabric, engine, and experiments were developed.
- **\(2.28\times10^{-10}\):** maximum spike-time discrepancy in one independent engine/reference validation trace.
- **8/8 seeds:** higher-rate four-candidate FE/FES allocation experiment passed its defined gates.
- **32/32 seeds:** both tested ordinary-E equation variants passed the defined turnover/recovery gates; linear converged faster.
- **11/11 checks:** tiled cortical-column acceptance artifact.
- **191 nodes / 1,052 edges:** 3×3 tiled preset with eight E candidates per column.
- **393 nodes / 2,162 edges:** two-tower hierarchy used to expose the relay-identity bottleneck.
- **12/12 runs across three seeds:** direct-identity two-pattern experiment met its defined ownership/turnover/recall checks.

## 40. Safe language for the brief

### Prefer

- “demonstrated in the tested small circuit”
- “validated under the engine’s declared approximation”
- “composable graph construction”
- “online local weight updates”
- “a substrate for future uncertainty measures”
- “architectural discovery within the CIPP project”
- “promising direction worth continued investigation”

### Avoid unless new evidence is produced

- “solves catastrophic forgetting”
- “proves continuous learning at scale”
- “is biologically identical to the cortex”
- “provides calibrated uncertainty”
- “is infinitely scalable”
- “is the first coincidence detector”
- “outperforms modern AI”

---

# Part IX — Source and evidence map

## 41. Current documentation

- [Current_Implementation_Methodology_Equations.md](../Current_Implementation_Methodology_Equations.md) — current equations, initialization, cell roles, and variable definitions.
- [ENGINE_VALIDATION_REPORT.md](ENGINE_VALIDATION_REPORT.md) — simulator validation and declared approximation.
- [FABRIC_CONSTRAINTS_AND_OPERATING_ENVELOPE.md](FABRIC_CONSTRAINTS_AND_OPERATING_ENVELOPE.md) — known architectural constraints and supported operating envelope.
- [TWO_TOWER_COMPOSITION.md](TWO_TOWER_COMPOSITION.md) — multi-tower experiment and identifiability finding.
- [NEXT_EVENT_ENGINE_TECHNICAL_SPEC.md](NEXT_EVENT_ENGINE_TECHNICAL_SPEC.md) — proposed absolute-timestamp engine.
- [NEST_EVENT_DRIVEN_3X3_REPORT.md](NEST_EVENT_DRIVEN_3X3_REPORT.md) — measured parity, semantic differences, and current limits of the NEST prototype.
- [NEST_DASHBOARD.md](NEST_DASHBOARD.md) — offline replay and conditional live visualization for NEST runs.

## 42. Experiment artifacts

- [`experiments/coincidence_results.json`](../experiments/coincidence_results.json) — early coincidence cadence and phase artifact.
- [`experiments/tiled_cc_results.json`](../experiments/tiled_cc_results.json) — tiled cortical-column acceptance results.
- [`experiments/runs/dual_fe_cc4/`](../experiments/runs/dual_fe_cc4/) — dual FE/FES allocation and negative-control runs.
- direct-identity, feedback-cadence, and two-tower run directories — hierarchy and interface experiments.

## 43. Historical milestones

Some important reports were part of earlier commits and may no longer be present in the sparsified working tree. They remain recoverable from Git history:

- `273fb8a` — original Selective Maturation Race implementation.
- `73aecc9` — emergent learning rebuild and real-time dashboard.
- `8b35983` through `ca1051d` — initialization/distance study, ablation harness, and recorded initialization results.
- `3b93122` — balanced initialization experiment and ownership report.
- `0542ab8` — rebuild around local coincidence feedback.
- `ca01411` — event-resolved coincidence pyramidal turnover circuit.
- `f390a12` — tiled cortical-column architecture.
- `a4fff8b` — ordinary-E and C-cell equation ablation decisions.
- `24eec09` — dual FE/FES experiment.
- `49a713a` — engine validation.
- July 28 commits — direct identity, feedback cadence, and two-tower hierarchy.

For presentation preparation, every quantitative claim should be traced either to a current report, a retained experiment artifact, or one of these historical commits.

---

## 44. Closing statement

The summer effort did not produce a finished general intelligence system, nor was that a realistic one-month objective. It produced the foundations needed to investigate a different computational paradigm seriously: a causal simulator, observable internal state, local learning laws, purposeful initialization, composable cortical columns, and a coincidence mechanism that binds bottom-up evidence to top-down context.

The most important outcome is that CIPP moved from a conceptual proposal to an executable and falsifiable research program. The team can now identify where a behavior came from, test alternatives under controlled conditions, preserve negative results, and turn architectural questions into measured experiments.
