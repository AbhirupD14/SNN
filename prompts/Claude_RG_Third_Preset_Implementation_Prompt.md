# Implement an RG-driven third topology preset

## Objective

Add a third built-in topology preset, `topology='rg'`, that explicitly represents
retinal ganglion (RG) cells upstream of L1. Use the current **old dense
global-inhibition topology** as its cortical base. Preserve `pi` and `old` exactly.

The scientific change is:

```text
held edge/pixel
    -> RG_i emits an un-inhibited binary spike every input boundary
    -> plastic one-to-one RG_i -> L1E_i synapse
    -> plastic dense L1E_i -> L2E_j synapses
    -> current old L2 competition and dense L2E -> L1I feedback
    -> paired L1I_i -> L1E_i inhibitory conductance
```

The RG layer is the persistent sensory evidence. Cortical inhibition may suppress
L1E, but it must never suppress RG or prevent a held edge from continuing to emit RG
spikes. This creates a developmental L1 timescale and lets L1E learn its sensory
afferent with the same accumulating excitatory rule used by L2E.

Do not replace either existing preset. Do not fold predictive inhibition into this
preset. The purpose of using `old` as the base is to isolate the timing consequences
of the new source layer and plastic L1 path from PI's pattern-specific mechanism.

## Read and audit before editing

Read these files and follow the live code rather than stale historical assumptions:

- `README.md`
- `Current_Implementation_Methodology_Equations.md`
- `Claude_Layer_Invariant_Neuron_Configuration.md`
- `backend/network_spec.py`
- `backend/simulation.py`, especially `_build_from_spec()` and `step()`
- `snn/neurons.py`, especially `ExcitatoryNeuron.update_acc_weights()`
- `backend/presets.py`
- `backend/dashboard_config.py`
- `backend/serializer.py`
- `backend/layout.py`
- `frontend/renderer.js`, `frontend/editor.js`, `frontend/controls.js`
- `tests/test_old_topology.py`, `tests/test_network_spec.py`,
  `tests/test_golden_topology.py`, and `tests/test_causal_step.py`

Inspect `git status` before editing and preserve all existing user work. The current
`pi` and `old` golden trajectories are behavioral contracts and must remain
bit-exact. Do not regenerate their fixtures to conceal a regression.

## Why `old` is the base

The `old` preset currently contains:

```text
external -> 9 L1E
9 L1E -> 8 L2E                         dense plastic feedforward
8 L2E -> L2I -> all 8 L2E             WTA conductance
every L2E -> every one of 9 L1I        dense relay excitation
L1I_i -> L1E_i                         paired inhibitory conductance
```

One L2 winner therefore fires all nine L1I relays and produces global L1 inhibition
one boundary later. Retain that cortical topology in `rg`. The only structural
addition is nine upstream RG cells and nine plastic one-to-one RG-to-L1E synapses;
the existing direct external-to-L1E injection is removed in this preset only.

This is intentionally not expected to reproduce PI-style contextual explaining
away. The experiment asks whether continuous un-inhibited retinal evidence, learned
L1 sensory weights, explicit delays, and global feedback produce useful temporal
symmetry breaking or merely phase locking.

## Required `rg` topology

### Populations

| Population | Count | IDs | Behavior |
| --- | ---: | --- | --- |
| RG | 9 | `RG0..RG8` | Exogenous binary spike source, one per pixel; never inhibited |
| L1E | 9 | `L1E0..L1E8` | Plastic noncompetitive excitatory accumulator |
| L1I | 9 | `L1I0..L1I8` | Same stateless relays used by `old` |
| L2E | 8 | `L2E0..L2E7` | Existing plastic competitor population with deterministic WTA |
| L2I | 1 | `L2I` | Existing WTA relay |

Expected total: **36 nodes**.

### Edges

| Projection | Count | Kind/semantics |
| --- | ---: | --- |
| `RG_i -> L1E_i` | 9 | plastic excitatory feedforward, paired one-to-one |
| `L1E_i -> L2E_j` | 72 | existing plastic excitatory feedforward, dense |
| `L2E_j -> L2I` | 8 | existing relay excitation |
| `L2I -> L2E_j` | 8 | existing WTA inhibitory conductance |
| `L2E_j -> L1I_i` | 72 | existing dense relay excitation |
| `L1I_i -> L1E_i` | 9 | existing paired inhibitory conductance |

Expected total: **178 internal synapses**. External pixel presentation is the event
source for each RG cell; it need not be serialized as a tenth edge category.

There must be no PI cells or predictive-inhibition edges in `rg`.

## RG semantics

An RG cell is a real visible network node, but it is an exogenous event source rather
than an inhibited cortical integrator.

For pixel `i` at boundary `t`:

```text
RG_i_spike(t) = input_arrives(t) AND input_vec[i] > 0.5
```

Interpret "continuously fires" precisely: a held active edge produces one RG spike
on every configured `input_period` boundary. An inactive pixel does not spike. Do not
add spontaneous background firing in this task.

Required properties:

1. RG has no incoming inhibitory edge and no inhibitory conductance state.
2. RG is unaffected by L1I, L2I, PI parameters, refractory state, or cortical WTA.
3. RG does not learn. The plastic weight is the postsynaptic `RG_i -> L1E_i`
   afferent owned by L1E.
4. RG spikes appear in dynamic state, raster/history, firing frequency, topology,
   renderer, and emitted-edge visualization.
5. RG positions form a distinct layer upstream of L1. Preserve explicit 3-D
   positions through topology save/load/apply.
6. The existing input controls and four 3x3 patterns drive RG in `rg`, while they
   continue to drive L1E directly in `pi` and `old`.

Do not model RG by depositing charge directly into L1 and merely drawing decorative
nodes. A real RG spike must be the causal presynaptic event used for L1 charge,
learning participation, serialization, and the one-step delay.

## L1E semantics and layer-invariant excitatory learning

L1E in `rg` must be a plastic excitatory neuron with exactly one positive sensory
afferent, `RG_i`. It is noncompetitive: every L1E threshold crosser may fire in the
same boundary. Do not copy L2 WTA into L1.

Use the same `ExcitatoryNeuron` implementation and the same accumulating-weight
update used by L2E:

```text
p        = threshold - sum(acc_weights)
signal_k = +1 if afferent k spiked in the causal arrival volley else -1
delta_k  = eta * p * signal_k * distance_factor_k
           * (1 - (w_k / w_max)^2)
w_k      = clip(w_k + delta_k, 0, w_max)
```

For L1E there is one entry, so a genuine RG-caused L1 spike supplies participation
`[True]`. Run the update only when that L1E actually fires, exactly as for L2E.
An inactive RG channel delivers no charge; its L1E must not learn merely because
some other RG or L1E fired.

Share these intrinsic settings with L2E unless the existing engine already treats
inhibitory conductance retention as a target/circuit role:

- excitatory threshold;
- resting potential and reset;
- baseline leak;
- refractory behavior;
- `eta`;
- positive weight cap;
- activity-trace equation;
- accumulating-weight implementation.

It is acceptable and desirable to retain the existing `alpha_inh_l1` retention for
inhibitory conductance targeting L1E and `alpha_inh` for L2E. That is an inhibitory
circuit timescale, not a second L1 excitatory learning rule. Do not silently change
the old topology's inhibitory constants while adding RG.

## RG-to-L1 weight initialization and possible recalibration

The canonical first implementation must use the current L2 **per-afferent**
feedforward initialization scale:

```text
FF_INIT_MEAN = 0.55 * theta / 9    # currently about 61 when theta=1000
```

Use the same seeded narrow jitter policy initially so the meaning of "initialized
like an L2 afferent" is literal and reproducible. Also include an experiment/control
with all nine RG-to-L1 weights initialized identically to `FF_INIT_MEAN`; this tells
us whether L1 phase splitting is learned/dynamic or simply injected by initialization
jitter. The built-in `rg` preset should use the shared seeded initialization policy,
not a hand-authored pixel-specific pattern.

Keep the shared current cap (`e_weight_cap`, currently `theta/2 = 500`) and learning
rate for the canonical preset. With one afferent, `sum(w)` cannot approach `theta`
under this cap, so `p = theta - w` stays positive and the weight eventually approaches
the cap through the saturation term. This is expected and must be documented. L1E is
therefore a temporal accumulator whose cadence accelerates during training, not a
one-event threshold relay.

At current defaults, the rough expectations for an isolated continuously driven
L1E are:

- initial `w ~= 61`, leak `0.03`: first spike after roughly 23 active RG events;
- repeated L1 spikes strengthen the sensory weight;
- near `w = 500`, the uninhibited mature cadence is roughly one L1 spike per three
  active RG events.

Treat those as audit targets, not hardcoded behavior.

If the full network bootstraps too slowly, first lengthen training/pattern dwell
horizons. Do not restore threshold-sized external injection and do not silently make
RG spikes bypass the synaptic weight. If weight recalibration is still needed after
measurement:

1. Keep the learning equation identical to L2E.
2. Change at most one of initialization scale, cap, or `eta` at a time.
3. Prefer an explicit projection-level parameter (for example an RG-to-L1 initial
   scale) over a hidden L1 code branch.
4. Never use pixel-specific values, labels, pattern identity, or winner identity.
5. Report the original and revised first-spike latency, mature cadence, saturation
   time, and effect on L2 development.
6. Do not promote a recalibrated value into the built-in preset merely because old
   short tests time out. Demonstrate a scientifically useful operating window first.

If the shared half-threshold cap proves inappropriate for a one-afferent cell, test a
fan-in-aware or RG-projection-specific cap only as a documented experimental variant.
Do not call that variant "exactly the same configuration" as L2E; the learning rule
would remain shared, but its connection policy would differ.

## Required synchronous timing

Every internal excitatory projection has delay 1. In `rg`, the causal chain must be:

```text
boundary t:
  active RG_i emits

boundary t+1:
  RG_i -> L1E_i charge arrives
  L1E_i integrates jointly with any L1 inhibitory conductance
  any L1E_i crosser fires and learns its RG afferent
  its L1E_i -> L2E events are queued

boundary t+2:
  L1E -> L2E charge arrives
  L2 WTA selects at most one crosser
  winner learns only from the L1 afferents delivered in this causal volley
  winner drives L2I and all nine old-topology L1I relays
  relay conductance outputs are queued

boundary t+3:
  L1I -> L1E and L2I -> L2E conductance arrives before integration
```

RG continues emitting at `t+1`, `t+2`, and later while the edge remains held,
regardless of L1 inhibition. Queued RG events are not cancelled by a cortical winner.
Inhibitory conductance and excitatory charge arriving at the same target boundary
must still be integrated jointly before threshold testing.

Do not collapse RG and L1 into the same subphase. Do not allow an RG spike to make L1
fire at `t` or an L1 spike to charge L2 at `t+1` and also be learned as though it
arrived in another boundary.

## Generalize feedforward execution correctly

The present engine assumes one feedforward hop from sensory cells to competitors and
stores a single global causal participation set. `rg` requires two feedforward hops
and a plastic non-WTA excitatory target.

Refactor the minimum necessary execution machinery so feedforward edges are dispatched
generically:

- any permitted excitatory source spike schedules weighted charge to its feedforward
  target for the next boundary;
- each plastic target owns an afferent list, aligned weights, distance factors, and
  edge IDs;
- causal participation is tracked **per postsynaptic target and per arrival
  boundary**, not as one global source set;
- when a plastic noncompetitive L1E fires, it learns from its own delivered RG set;
- when the selected L2E fires, it learns from its own delivered L1 set;
- inactive/absent afferents receive the existing negative signal only in the target
  that actually fired;
- no propagation depends on Python node iteration order.

Preserve exact execution for the existing one-hop `pi` and `old` presets. A generic
refactor is acceptable only if their frame digests remain unchanged.

## NetworkSpec/archetype requirements

The current vocabulary cannot correctly express this topology because `e_sensory`
has a fixed external afferent and `e_competitor` is intrinsically placed in WTA.
Extend it coherently.

Add concepts equivalent to:

- an RG/exogenous-spike-source archetype;
- a plastic, noncompetitive excitatory encoder archetype for L1E.

Names may differ if a cleaner vocabulary emerges, but roles must remain distinct.
Do not implement L1E as an `e_competitor` and then special-case it out of WTA by ID or
layer string. Likewise, do not make RG an ordinary conductance LIF neuron with an
artificial threshold-sized input merely to reuse a class.

Validation must enforce:

- RG may drive plastic excitatory targets;
- RG cannot be the target of cortical inhibition;
- L1E may be the target of plastic feedforward and relay inhibition;
- a plastic noncompetitive excitatory cell is not part of deterministic WTA;
- feedforward paths remain directed;
- pixel ownership for external input is unique among RG source cells without
  preventing L1E from retaining display/receptive-field metadata;
- arbitrary custom graphs using the new vocabulary either execute correctly or are
  rejected with a clear structural error.

Avoid ID-prefix semantics in the engine. Node archetype/role and edges should define
behavior.

## Preset, API, persistence, and frontend integration

Make `rg` a full third built-in everywhere `pi` and `old` are enumerated:

- `VALID_TOPOLOGIES` and configuration validation;
- `preset_spec()`;
- built-in preset listing/loading;
- dashboard topology selector and descriptions;
- API serialization;
- topology editor palette/validation if new archetypes are editable;
- renderer colors, labels, layer placement, and connection visibility;
- charge/raster/inspector views;
- README and methodology documentation.

The UI must make the new hierarchy visually legible:

```text
RG layer -> L1E/L1I layer -> L2E/L2I layer
```

RG spikes should flash like source events, and the nine plastic RG-to-L1E weights
must be inspectable and emit ordinary sparse weight-change updates. Ensure the
receptive-field editor does not assume that every plastic feedforward weight belongs
to an L2E 3x3 field. A one-afferent L1E weight may use the standard synapse inspector
if that is clearer than forcing it into a 3x3 grid.

Saved custom presets containing the new archetypes must round-trip without losing
positions or edge semantics. Existing saved presets must remain loadable.

## Focused deterministic tests

Add a dedicated `tests/test_rg_topology.py` or equivalent.

### Structure

- `rg` has 36 nodes and 178 internal edges.
- Counts and IDs match the tables above.
- Every RG has exactly one paired outgoing plastic edge to L1E.
- L1E-to-L2E is dense 9x8.
- L2E-to-L1I is dense 8x9.
- L1I-to-L1E remains paired 1:1.
- No PI/predictive edges exist.
- `pi` and `old` counts and specs are unchanged.

### RG behavior

- An active pixel makes only its paired RG spike on each input boundary.
- An inactive pixel produces no RG spike.
- RG continues spiking while its L1E is strongly inhibited.
- No inhibitory edge can target RG.
- RG spike serialization/history/frequency is correct.

### Delays and causal order

- RG spike at `t` cannot affect L1E until `t+1`.
- L1E spike at `t+1` cannot affect L2E until `t+2`.
- An L2 winner at `t+2` produces L1 inhibition at `t+3`, not earlier.
- Same-boundary inhibition and RG-derived excitation are jointly integrated.
- No spike propagates through both feedforward layers in one boundary.

### L1 accumulation and learning

- One initial RG event is subthreshold.
- Repeated RG events accumulate according to the live conductance/leak equation.
- First-spike timing matches an isolated analytical/reference calculation.
- On an RG-caused L1 spike, `RG_i -> L1E_i` changes by the exact same numerical
  `update_acc_weights()` equation as an equivalently configured L2E afferent.
- Other RG-to-L1 weights do not change.
- An inactive pixel does not charge or learn.
- Weight remains in `[0, e_weight_cap]`.
- Repeated training shortens L1 inter-spike intervals as the weight grows.

### Noncompetitive L1 versus competitive L2

- Multiple L1E crossers may fire in the same boundary.
- At most one L2E crosser fires and learns in a boundary.
- L1E uses the shared excitatory learning rule but never enters L2 WTA.

### Target-specific participation

- An L1E update sees only its RG afferent arrival.
- An L2E update sees only L1E spikes delivered to that L2E in that boundary.
- Participation from the preceding/following boundary cannot leak into an update.
- Two targets receiving different source sets learn from their own sets.

### Regression

- Existing `pi` and `old` behavioral golden frame digests remain bit-exact.
- Existing topology, API, persistence, and frontend tests still pass.
- Toggling among `pi`, `old`, and `rg` rebuilds cleanly with no stale weights,
  queues, conductances, source mappings, or population handles.

Create an `rg` golden only after the focused causal tests are correct. Do not alter
the two existing goldens.

## Required timing and symmetry experiment

Add a headless, deterministic experiment comparing at least these conditions:

1. `old`: current frozen direct external-to-L1 drive.
2. `rg_frozen`: explicit RG layer and delay, but RG-to-L1 weights frozen.
3. `rg_plastic`: the built-in `rg` behavior.
4. `rg_plastic_equal_init`: plastic RG-to-L1 with identical initial weights rather
   than per-synapse jitter.

The frozen condition should use a documented weight choice. If matching the old
333-unit charge is used, say that it isolates only topology/delay; if using the new
~61-unit initialization, say that it also changes L1 cadence. Do not blur these two
controls.

Run multiple seeds and at least:

- sustained `row 1` acquisition;
- `row 1 -> col 1 -> row 1`;
- preferably `diag \\ -> diag / -> diag \\` as a second overlap pair.

Training horizons must be long enough for the new two-layer developmental path.
Report raw boundary counts as well as counts of RG events, L1 spikes, and L2 winner
events.

Measure:

- first RG, L1E, and L2E spikes;
- each RG-to-L1 weight trajectory;
- early and mature L1 firing cadence;
- phase dispersion among the three active L1 features;
- L2 winner identity and actual winner-spike times, not a sticky winner field;
- L2 winner dwell time and turnover rate;
- time from L2 winner to L1 inhibitory arrival;
- time from L1 inhibitory arrival to the first subsequent L1 escape spike;
- number and identity of causal L1 afferents at every L2 learning event;
- fraction of L2 updates driven by one, two, or all three active features;
- L2 receptive-field selectivity and whether cells become single-pixel/phase
  detectors;
- owner collisions, dead L2 cells, row/column owner distinction, and recovery;
- sensitivity to RG-to-L1 initialization jitter;
- weight-cap saturation times and whether mature L1 becomes effectively too fast.

Distinguish these outcomes honestly:

- **useful assembly symmetry breaking:** different L2 cells acquire coherent
  overlapping patterns and recover appropriately;
- **temporal phase breaking:** L1 channels desynchronize and select winners by phase;
- **winner tyranny:** the same mature L2 cell always triggers the next global clamp;
- **single-feature collapse:** exact-volley learning makes L2 specialize to whichever
  L1 channel escapes first;
- **developmental deadlock:** L1 or L2 never bootstraps within a justified horizon.

Do not claim contextual explaining-away from `rg`; the old dense feedback path erases
winner identity because every winner activates every L1I.

## Documentation requirements

Update the active documentation to state:

- all three presets and their intended scientific roles;
- exact RG semantics;
- node/edge counts;
- the two-hop delay chain;
- L1's shared excitatory learning equation;
- RG-to-L1 initialization, cap, and expected developmental cadence;
- why L1 retains `alpha_inh_l1` despite sharing the excitatory learning rule;
- measured results and negative outcomes from the timing experiment;
- the difference between retinal evidence persisting and cortical L1 continuing to
  spike;
- that biological "no inhibition" here means no inhibition from this modeled
  cortical feedback loop, not a claim that the biological retina contains no
  inhibitory circuitry.

Remove or revise statements claiming that the engine has only two presets or that
external input always targets L1E directly.

## Verification

Run, at minimum:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python -m experiments.<new_rg_experiment_module>
node --check frontend/app.js
node --check frontend/controls.js
node --check frontend/editor.js
node --check frontend/renderer.js
node --check frontend/inspector.js
git diff --check
```

Also run syntax checks on every other modified JavaScript file. If a command fails,
report the exact failure and distinguish a new regression from an environmental
problem.

## Acceptance criteria

The task is complete only when:

1. `pi`, `old`, and `rg` are selectable, serializable built-in presets.
2. `rg` contains 9 real RG nodes, 9 plastic non-WTA L1E cells, and the unchanged old
   cortical feedback topology.
3. Held active pixels make RG spike continuously at the configured input rate even
   while L1 is inhibited.
4. RG spikes reach L1 only after delay 1, and L1 spikes reach L2 only after another
   delay 1.
5. L1E learns its one RG afferent through the same accumulating rule as L2E, with no
   L1-only learning shortcut.
6. Causal participation is target-specific and boundary-specific across both
   feedforward layers.
7. RG-to-L1 weights and events are inspectable in the dashboard/API.
8. Existing `pi` and `old` golden dynamics remain unchanged.
9. The new deterministic experiment reports whether the result is useful assembly
   symmetry breaking, arbitrary temporal phase breaking, single-feature collapse,
   winner tyranny, or deadlock.
10. Any L1-specific weight recalibration is explicit, isolated, measured, and not
    disguised as preservation of the shared configuration.

## Final report

Return a concise but evidence-backed report containing:

1. the pre-change architecture/execution audit;
2. the new archetypes and generic feedforward execution design;
3. exact `rg` node and edge counts;
4. exact timestep semantics;
5. L1/L2 intrinsic configuration comparison;
6. RG-to-L1 initialization and learning behavior;
7. analytical and measured L1 first-spike/cadence values;
8. timing/symmetry experiment results across conditions and seeds;
9. whether jitter creates artificial feature priority;
10. all files changed;
11. full test and syntax-check results;
12. remaining limitations and the next smallest evidence-driven experiment.
