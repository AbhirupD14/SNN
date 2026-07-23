# Claude Prompt: Dual-FE Self-Regulating 3x3 Cortical Column

## Objective

Implement and causally test an experimental learning rule in the smallest useful circuit:
a direct `3x3` RGC input feeding one cortical column with four ordinary excitatory
competitors and one central WTA inhibitory cell.

This experiment tests a different symmetry-breaking hypothesis:

- consolidation and turnover should emerge from synaptic strength and weight-dependent
  plasticity;
- the input features must not be frequency-halved or selectively suppressed;
- there must be no feature relays, coincidence-feedback path, predictive inhibition, Eor,
  or hierarchy in the acceptance topology;
- the only inhibition is the ordinary within-column WTA I cell.

The four competitors should be capable of acquiring the four canonical overlapping
`3x3` patterns one-to-one even though they share pixels. A consolidated incumbent should
not absorb a new pattern merely because it has a strong shared afferent. Its established
synapses and high instantaneous charge should make its updates small, while an unused
competitor with middle-valued weights remains plastic.

This is an implementation-and-execution task. Do not stop after proposing code or writing
unit tests. Run the reference experiment and report the result honestly.

## Scope and precedence

This task supersedes the recursive feature-gated hierarchy as the current experiment.
Do not implement `prompts/Claude_Recursive_Feature_Gated_Hierarchy_Prompt.md` in the same
change. Preserve all existing feature-gated work as a control and preserve every existing
preset, replay, experiment, and golden artifact.

Inspect the entire working tree before editing. Preserve unrelated changes. Do not commit
or push unless explicitly instructed after the results are reviewed.

## Read before editing

Read:

- `snn/neurons.py`, especially ordinary accumulating E learning, coincidence basal
  learning, pre-reset spike/update ordering, weight floors, and update logging;
- `backend/simulation.py`, especially parameter validation, mutable dashboard keys,
  latency-WTA event execution, hard resets, learning-event timing, and serialization;
- `backend/network_spec.py`, especially preset registration, edge-kind validation, and the
  smallest reusable WTA-bank builder;
- `backend/dashboard_config.py`, `backend/api.py`, `frontend/controls.js`, and the topology
  layout/renderer;
- `experiments/microcircuit_turnover.py`,
  `experiments/feature_gated_turnover.py`, and
  `experiments/basic_consolidation.py`;
- tests for ordinary-E learning, coincidence learning, preset contracts, serialization,
  dashboard configuration, deterministic topology construction, and replay recording;
- `docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md`.

Record the baseline test result before changing behavior.

## Exact experimental learning rule

Add one experimental learning mode implementing these equations exactly:

```text
FE = e + ((1 - e) /
          (1 + B * ((Iaccq / theta) - 0.5)^2))

FES_i = wte + ((1 - wte) /
               (1 + B * ((2 * wt_i / theta) - 0.5)^2))

dw_i = LR * FE * FES_i * input_signal_i * influence_i

wt_i <- max(wte, wt_i + dw_i)
```

Reference values:

```text
e   = 0.001
wte = 0.001
B   = 5.0
```

Definitions and non-negotiable semantics:

- `Iaccq` is the neuron's **instantaneous accumulated excitatory charge for the causal
  learning event**. It is not total afferent weight, a running maturity budget, an
  activity trace, or post-reset voltage.
- Capture `Iaccq` before the spike/reset destroys the causal charge. Never evaluate FE
  from the post-reset value.
- Use the actual pre-reset `Iaccq` represented by the engine. Do not silently clamp its
  ratio to `[0, 1]`; if overshoot is possible, retain it and log it.
- `theta` is the target neuron's excitatory threshold.
- Node-level `FE` is shared by all plastic synapses updated in that one causal event.
- `FES_i` is evaluated independently from each synapse's pre-update weight `wt_i`.
- `input_signal_i` retains the existing causal sign convention for the neuron type.
  Ordinary E participating afferents potentiate and nonparticipating afferents depress.
  Do not turn absence into a positive update.
- `influence_i` retains the existing deterministic distance influence. Do not change or
  renormalize it merely to make this experiment pass.
- The lower weight bound for every plastic synapse in this mode is the raw value `wte`.
  A negative update must stop at `wte`, not zero.
- There is no per-synapse upper cap in this experimental mode. Do not retain a hidden
  `w_max` clip. The experiment is specifically testing whether the FE/FES slowdown and
  input statistics regulate learning.
- Implement the inverse-quadratic expressions as written. Do not substitute an
  exponential Gaussian, the old signed neuron-wide headroom, or the historical
  quadratic-cap multiplier.
- Keep all calculations in full precision. Rounding is for reports only.

The new factors are dimensionless. Do not silently reinterpret the existing numerical
learning rates. The reference run must first use the existing dashboard values
`eta=0.01` and `c_eta=0.005`. If this is too slow to measure, preserve that result and use
an explicitly reported, isolated LR sensitivity run rather than quietly promoting a new
default.

## Ordinary E and coincidence-cell coverage

The experimental flag must switch both learning families together:

1. Ordinary accumulating E cells use the equation above with `LR=eta`.
2. Coincidence-cell learned basal synapses use the same FE/FES factors with `LR=c_eta`.

For a coincidence cell:

- preserve its Boolean apical eligibility requirement;
- preserve the causal basal input signal and basal distance influence;
- use the pre-reset instantaneous accumulated **basal charge** as `Iaccq`;
- apply the `wte` floor and no upper cap in this experimental mode;
- do not convert apical input into learned charge or an additional weight term;
- do not change its membrane equation, threshold, deposit semantics, or scheduler timing.

The acceptance topology below deliberately contains no coincidence cell. Prove the C-cell
implementation separately with focused unit tests and a tiny isolated causal probe. This
is required so the single experimental flag has coherent behavior when applied to
existing coincidence topologies, but C activity must not be able to cause the small
column's result.

## One dashboard flag

Expose exactly one new dashboard control:

```text
Dual FE/FES learning (experimental)
```

A Boolean toggle is preferred. An explicit two-option learning-rule selector is
acceptable if that matches the existing configuration architecture better.

When disabled:

- every current topology, equation, initialization, weight floor, bound, serializer, and
  golden trace must remain behaviorally unchanged;
- the current production learning rule remains the default;
- existing saved/custom specs retain their current meaning.

When enabled:

- both ordinary E and coincidence basal learning switch to the dual-FE equations;
- plastic weights use `wte` as their floor and have no upper cap;
- plastic weights initialize at the maximum-plasticity middle implied by FES,
  `wt = theta/4`, with only the existing deterministic seeded jitter needed to break an
  exact tie;
- applying the toggle rebuilds the engine and clearly warns that learned state is wiped.

Keep `e`, `wte`, and `B` validated engine/headless parameters so experiments can perform
declared sensitivity analyses. Their reference defaults are `0.001`, `0.001`, and `5.0`.
Do not add three more dashboard sliders. The dashboard gets one flag only; the existing
`eta` and `c_eta` controls remain the LRs.

Validate at least:

- `0 < e <= 1`;
- `0 < wte <= 1`;
- finite `B >= 0`;
- all derived FE/FES/update values are finite;
- serialization and replay metadata identify whether the experimental rule was active
  and record its three parameter values.

Use an unambiguous internal name such as `dual_fe_fes`. Do not call it merely `gaussian`,
because the required curve is inverse-quadratic.

## New minimal topology

Add one new built-in experimental preset, preferably:

```text
rg_direct_cc4
```

Suggested dashboard label:

```text
3x3 Direct CC · 4 E + WTA I
```

Its complete graph is:

```text
9 RGC sources
    -> dense plastic feedforward
4 ordinary event-resolved/latency E competitors
    -> one central WTA I
central WTA I
    -> hard reset of exactly those 4 E competitors
```

Expected graph arithmetic:

```text
nodes = 9 RGC + 4 E + 1 I = 14
edges = 9*4 RGC->E + 4 E->I + 4 I->E = 44
```

Derive and assert these counts from the graph; do not rely only on the arithmetic above.

Strict structural exclusions:

- no `e_pretrained` or other feature relay;
- no coincidence C node;
- no feature-specific I;
- no Eor;
- no apical edge;
- no predictive or hierarchical feedback;
- no L2/L3 node;
- no inhibition targeting an RGC input;
- no path that drops, delays, halves, or selectively suppresses an active RGC feature.

Every active RGC pixel must deliver the same declared event stream directly to all four
competitors. The single I exists only to enforce local WTA among the four competitors.
Place the four E cells around the I cell in the dashboard layout so the intended column is
visually obvious. Topology construction and layout must use metadata, not ID-prefix
parsing.

The preset must be runnable with the experimental flag either off or on. This provides a
same-topology learning-rule control.

## Algebra and state-timing tests

Before the pattern experiment, add focused tests that prove:

1. FE is exactly `1` at `Iaccq/theta = 0.5`.
2. FES is exactly `1` at `wt/theta = 0.25`.
3. Reference values at zero, normalized endpoints, and at least one overshoot point match
   the written equations. Do not assert that `B=5` reaches the asymptotic floor at an
   endpoint.
4. FE is calculated from the captured pre-reset charge, not post-reset state.
5. Each synapse uses its own pre-update FES even when a vector update is applied.
6. Participating and nonparticipating ordinary-E afferents move in opposite directions.
7. A negative update clips at `wte`, never below it.
8. A large positive update is not clipped by a historical E or C upper cap.
9. C basal learning uses the same FE/FES implementation while still requiring valid
   apical coincidence.
10. With the flag off, existing E and C update traces remain bit-exact.

Log enough precision to reconstruct one update manually:

```text
boundary and sub-boundary tau
cell and synapse id
pre-reset Iaccq
theta
FE
pre-weight
FES
LR
input signal
influence
raw dw
applied dw after the wte floor
post-weight
spike/reset identity
```

Do not enlarge normal dashboard dynamic frames with this trace. Keep it behind existing
opt-in learning instrumentation or in the experiment recorder.

## Causal symmetry-breaking probes

Before the four-pattern protocol, show that winner identity follows synaptic strength
rather than declaration order:

1. Construct an otherwise symmetric column with a small, explicitly recorded advantage
   on one competitor's active-pattern afferents.
2. Verify that competitor wins even when it is not the first node in serialized order.
3. Repeat with the advantage assigned to another competitor.
4. Confirm every active RGC source emitted and every competitor received the same number
   of source events.

Also run an exact-symmetry diagnostic. If exact equality is resolved by stable scheduler
order, report it honestly. The production experiment may use deterministic seeded initial
weight jitter, but the report must show the jitter and demonstrate that the initial winner
corresponds to the strongest active-afferent total rather than an ID-based special case.

## Reference four-pattern experiment

Create a headless experiment, preferably:

```text
experiments/dual_fe_cc4_consolidation.py
```

Use the canonical three-pixel patterns:

```text
row 1
col 1
diag \
diag /
```

All four share the center pixel. Use seed 1 first with:

```text
dual FE/FES enabled
e = 0.001
wte = 0.001
B = 5.0
eta = 0.01
c_eta = 0.005
leak_rate = 0
refractory_steps = 0
```

Protocol:

1. Start from fresh middle-initialized weights.
2. Present `row 1` for a fixed, declared dwell long enough to measure consolidation.
3. Switch to `col 1`, then `diag \`, then `diag /`, using the same fixed dwell.
4. Do not stop a phase immediately when a transient dominance criterion is first met.
5. After training, replay all four patterns without resetting weights.
6. Repeat the entire sequence with the experimental flag off as the same-topology
   control.
7. After preserving the reference result, run a modest declared seed sweep. Report the
   success fraction and every failure; do not show only a favorable seed.

Primary desired outcome:

- four distinct E owners for the four patterns;
- each pattern has one stable dominant owner;
- turnover occurs at every pattern switch;
- final recall preserves the same one-to-one mapping;
- the first owner does not absorb the other three patterns through the shared center
  pixel;
- all active RGC events remain unsuppressed and equal-frequency throughout.

Use a dominance criterion such as at least `0.95` over multiple consecutive fixed windows,
but report the complete winner counts and do not let the criterion terminate training
early.

## Evidence that the mechanism is weight-based

For every pattern switch, record:

- incumbent and new owner;
- each competitor's nine pre/post weights;
- active-afferent weight total for the old and new pattern;
- pre-reset `Iaccq`, FE, per-synapse FES, and applied updates;
- participating versus nonparticipating updates;
- WTA spike/reset events;
- RGC emitted and delivered event counts per pixel and per competitor;
- competitor firing counts and first-spike tau.

A positive result must show all of the following:

1. No active input feature was frequency-halved, dropped, or inhibited.
2. No feedback or coincidence event exists in the topology.
3. The incumbent's established weights and charge produce smaller relevant updates than
   the still-plastic competitor during turnover.
4. The new owner is selected by its effective afferent strengths, not a hard-coded
   winner, node order, or missing input events.

Do not require equal output firing frequencies. The claim is narrower: **selective input
frequency suppression is not the cause of turnover**. Ordinary WTA inhibition remains
part of the column.

Include a long-dwell stress test. Train the first pattern substantially longer than the
minimum consolidation window, then present the second overlapping pattern. This checks
whether slow nonzero tail plasticity eventually creates another one-afferent incumbent
lock. A failure after long dwell is a scientific result, not permission to shorten the
test.

## Required LR x B sensitivity study without hidden tuning

Run the exact reference values first and save their artifact before changing anything.
Never use a sweep result to overwrite or relabel the reference result.

After the reference artifact exists, run this required seed-1 grid whether the reference
passes or fails:

```text
B in {1.0, 5.0, 20.0, 50.0}

LR multiplier m in {1, 10, 100, 500}
eta   = 0.01  * m
c_eta = 0.005 * m
```

This is 16 configurations, including the reference `B=5, m=1`. Use the same initialization,
pattern order, fixed dwell, thresholds, and recording protocol for every point. Do not stop
individual grid points early. The LR multiplier is an experiment convention, not another
dashboard control.

For the direct four-E topology, test the ordinary-E behavior at every grid point. Because
the acceptance topology intentionally has no C cell, also run the same 16 `(B, m)` points
through the isolated coincidence-cell causal probe. Record C basal trajectories, FE/FES,
valid coincidences, and whether one versus two deposits can fire it; do not allow those
results to influence the direct-column dynamics.

Produce a CSV and compact heatmap/table over `B x m` containing at least:

- number of distinct final owners;
- per-pattern dominance and recall consistency;
- turnover count;
- first-consolidation time;
- minimum/maximum/final weights;
- weight drift during the late window;
- update magnitudes for incumbent and free competitors;
- floor hits and any non-finite value;
- long-dwell second-pattern turnover;
- isolated C maturation time and one/two-deposit behavior;
- explicit PASS/FAIL and failure class.

If any configuration fails, classify the failure:

- no competitor fires or learns;
- learning is measurable but too slow;
- one incumbent absorbs multiple patterns;
- weights drift upward without practical stabilization;
- weights collapse to `wte`;
- stable owners exist but are not one-to-one;
- scheduler order, rather than weight strength, decides the result.

Keep `e=wte=0.001` fixed throughout this grid. Do not expand into an adaptive or
success-directed search. Save every attempted configuration, including failures. A
successful non-reference point is evidence for feasibility, not evidence that the
reference parameters passed and not permission to promote new defaults.

If at least one point passes, select a confirmation candidate using a declared rule fixed
before inspecting the outcomes: full one-to-one mapping and recall first, long-dwell
stability second, then the smallest LR multiplier and the `B` closest to `5`. Run that
candidate over the declared multi-seed sweep. Report all ties and do not cherry-pick a
visually attractive trace.

Do not tune:

- event delays or scheduler priorities;
- thresholds;
- input rates;
- inhibition timing;
- pattern dwell independently per pattern;
- weight initialization away from the FES middle;
- topology or number of competitors.

## Artifacts and replay

Write each run beneath a gitignored directory such as:

```text
experiments/runs/dual_fe_cc4/<run-id>/
```

Include:

- resolved configuration and seed;
- topology fingerprint;
- summary JSON;
- phase/window metrics in CSV;
- per-learning-event FE/FES trace in CSV or JSONL;
- final weight matrices;
- source delivery and WTA event counts;
- replay compatible with the existing dashboard replay player;
- explicit PASS/FAIL per acceptance condition.

Add phase markers so the replay can be scrubbed across every pattern transition. Do not
invent a second replay schema or renderer.

## Compatibility and regression requirements

Add tests for:

- exact new topology nodes, edges, projections, and forbidden paths;
- deterministic fresh graph construction;
- dashboard preset registration and one new experimental toggle;
- config validation and apply/rebuild behavior;
- serialization/replay round-trip of the experimental parameters;
- ordinary E and C algebra/timing/floor behavior;
- flag-off bit-exact behavior;
- headless reference experiment smoke coverage.

Run:

1. the new focused tests;
2. existing ordinary-E, coincidence, topology, dashboard, serialization, and replay tests;
3. the existing feature-gated and basic-consolidation tests;
4. the complete Python suite;
5. the replay JavaScript tests;
6. `git diff --check`.

Do not update old golden files merely because the default changed: the default must not
change. Add a new golden only for the new preset if the existing golden-test architecture
requires it.

## Stop conditions

Stop and report rather than tuning around the result if:

- the direct topology cannot execute without changing the global event contract;
- pre-reset `Iaccq` cannot be identified unambiguously;
- the flag-off path changes an existing trace;
- the only way to obtain turnover is to suppress or frequency-halve an RGC feature;
- the only way to obtain four owners is to add relays, feedback, or extra neurons;
- weights drift indefinitely under the exact reference parameters;
- the reference result depends on node insertion order.

## Final report

Report:

1. the exact files changed;
2. the exact topology and graph counts;
3. the exact `Iaccq` state used and when it is captured;
4. algebra checks for FE/FES, including the actual values at normalized endpoints for
   `B=5`;
5. reference seed-1 outcome before any tuning;
6. control outcome with the flag off;
7. seed-sweep and long-dwell results;
8. causal evidence that all RGC feature frequencies remained unchanged;
9. weight trajectories and effective update factors at every turnover;
10. C-cell focused results;
11. tests and artifact paths;
12. limitations and negative findings.

Do not claim that hierarchy, continuous-time event conservation, noise invariance, or
large-scale composition is solved by this experiment. The sole scientific question is
whether a direct four-neuron cortical column can assign four overlapping `3x3` patterns
one-to-one through dual node/synapse plasticity, without feature-specific feedback or
input-frequency halving.
