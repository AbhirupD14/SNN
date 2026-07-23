# Claude Test Plan: Linear Weight-Update Ablation

## Assignment

Test what happens when the per-synapse nonlinear multiplier

~~~text
1 - (w / w_max)^2
~~~

is removed from the accumulating-weight rules.

Work in phases and keep the repository runnable after every phase. Preserve the current
production rule as the default until the evidence is reviewed. Do not change WTA,
inhibition, event ordering, initialization, learning rates, phase durations, or
dashboard defaults.

The experiment must distinguish four effects:

1. neuron-wide free energy;
2. per-synapse quadratic slowdown;
3. final hard clipping at w_max; and
4. the C-cell requirement that one coincidence stays subthreshold while two can fire.

The deliverable is evidence for deciding whether ordinary E/L2E weights can use a
linear, cap-free rule before composition/WTA work begins.

## Current and proposed equations

Current ordinary accumulating update:

~~~text
FE       = theta - sum_j(w_j)
signal_i = +1 if afferent i participated, otherwise -1

dw_i = eta
       * FE
       * signal_i
       * distance_i
       * (1 - (w_i / w_max)^2)

w_i <- clip(w_i + dw_i, 0, w_max)
~~~

Current C basal update:

~~~text
FE_C = theta - w_b

dw_b = c_eta
       * FE_C
       * apical
       * basal_signal
       * basal_distance
       * (1 - (w_b / w_b,max)^2)

w_b <- clip(w_b + dw_b, 0, w_b,max)
~~~

The proposed linear update removes only the quadratic multiplier:

~~~text
dw_i = eta * FE * signal_i * distance_i
~~~

FE is deliberately preserved. Although it is a scalar, it is a dynamically changing
neuron-wide factor coupled to total afferent strength, not merely another fixed
learning-rate constant.

## Frozen scope

Do not change:

- BoundaryEventScheduler or latency-tie behavior;
- L2I/L1I hard resets;
- the at-most-one-L2-spike-per-boundary contract;
- synaptic delays;
- l2_init_total_frac = 0.95;
- ordinary eta = 0.01;
- c_eta = 0.001;
- turnover leak = 0 or refractory = 0;
- C threshold, initial weight, or derived temporal cap;
- row/column presentation duration;
- dashboard defaults; or
- topology/connectivity.

Do not add co-firing, k-WTA, delayed inhibition, adaptive learning rates, weight
renormalization, or hidden safety clamps. Composition is only an observational
diagnostic in this task.

## Required headless variants

Use separate E and C controls. Suggested names are e_weight_update_mode and
c_weight_update_mode. Equivalent explicit names are acceptable.

### Ordinary E/L2E modes

A. quadratic_bounded — exact baseline:

~~~text
dw = eta * FE * signal * distance * (1 - (w/w_max)^2)
w  = clip(w + dw, 0, w_max)
~~~

B. linear_bounded — isolate soft nonlinearity:

~~~text
dw = eta * FE * signal * distance
w  = clip(w + dw, 0, w_max)
~~~

C. linear_nonnegative — cap-free E probe:

~~~text
dw = eta * FE * signal * distance
w  = max(0, w + dw)
~~~

Mode B does not prove that w_max is unnecessary because it retains the hard clip.
Mode C must initially be headless-only.

### C modes

- c_quadratic_bounded: exact current rule;
- c_linear_bounded: no quadratic multiplier, derived temporal cap retained;
- c_linear_nonnegative: no multiplier and no upper cap, diagnostic only.

Never expose c_linear_nonnegative in the dashboard during this task. With one basal
weight and FE_C = theta - w_b, it may approach theta and make one coincidence sufficient
to fire. Its purpose is to detect that failure.

## Phase 0: baseline lock and instrumentation

Before modifying either equation:

1. Run the complete existing test suite.
2. Record deterministic baseline traces for seeds 1 through 8 and dashboard seed
   4083693835.
3. Add opt-in experiment instrumentation without changing dynamics.

For each learning event record:

~~~text
boundary
cell id
pre/post total afferent weight
pre/post FE
minimum and maximum weight
number of zero weights
number at w_max in bounded modes
raw delta norm
applied delta norm after clipping
whether total weight crossed theta
~~~

For C also record:

~~~text
basal weight
derived C cap
one-deposit firing weight w_1
valid coincidences
C spikes
crossing tau
whether one coincidence could fire from rest
~~~

Prefer an experiment-local observer or opt-in headless trace hook. Do not enlarge every
dashboard frame.

Phase 0 gate:

- all existing tests pass;
- instrumentation-off behavior is byte-identical;
- same-seed traces are deterministic.

## Phase 1: algebra and compatibility tests

Refactor only enough to choose a mode without duplicating unrelated neuron logic.

Ordinary E tests must cover:

1. positive FE with participating and nonparticipating afferents;
2. zero FE producing exactly zero delta;
3. negative FE and signed direction reversal;
4. unequal distance influences;
5. a weight near w_max, proving linear_bounded removes slowdown;
6. an update crossing w_max, proving only bounded modes clip;
7. an update crossing zero, proving every mode retains the excitatory floor;
8. a cap-free weight above historical w_max, proving no hidden clipping.

C tests must cover:

- exact hand-calculated delta for all C modes;
- no learning on basal-only or apical-only input;
- learning only after an actual C spike;
- c_linear_bounded clipping at the derived temporal cap;
- c_linear_nonnegative genuinely exceeding that cap;
- current quadratic modes remaining the production defaults.

All existing golden and deterministic replay tests must remain unchanged under the
baseline mode.

## Phase 2: isolated trajectories

Run non-WTA neuron fixtures first. Use identical starting weights and inputs across
modes.

Repeated ordinary volleys:

1. one active afferent out of nine;
2. three active out of nine;
3. five active out of nine;
4. alternating disjoint three-afferent patterns;
5. three-afferent patterns with 10 percent bit-flip noise;
6. unequal distance factors.

Run at least 10,000 learning events. Save checkpoints:

~~~text
0, 1, 2, 5, 10, 25, 50, 100, 250, 500,
1000, 2500, 5000, 10000
~~~

Measure:

- events to reach 90, 95, and 99 percent of theta active-weight sum;
- total, active, and inactive weight trajectories;
- FE trajectory and negative-FE count;
- overshoot above theta;
- maximum individual weight;
- floor/cap hit counts;
- NaN/Inf occurrence;
- sensitivity to sparsity and distance.

Do not assume theta is automatically a stable total. Before clipping:

~~~text
Delta sum(w) = eta * FE * sum_i(signal_i * distance_i)
~~~

Its direction depends on the active/inactive balance. Explicitly report any case where
negative FE pushes the total farther above theta.

### Isolated C cadence

Drive one valid basal/apical coincidence per boundary. For each C mode record:

- basal weight versus valid-coincidence count;
- spike sequence and crossing tau;
- first point at which one coincidence fires from rest;
- whether the 0,1,0,1 two-valid-event cadence survives.

For c_linear_nonnegative, abort that condition if one coincidence becomes sufficient,
the weight reaches w_1, the weight exceeds 2 theta, or NaN/Inf appears. The abort is an
experimental result, not a hidden clamp.

## Phase 3: factorial network ablation

Run the real rg_coincidence engine and analytic scheduler.

| Condition | E/L2E mode | C mode | Purpose |
| --- | --- | --- | --- |
| A | quadratic bounded | quadratic bounded | Production baseline |
| B | linear bounded | quadratic bounded | Isolate E soft term |
| C | quadratic bounded | linear bounded | Isolate C soft term |
| D | linear bounded | linear bounded | Remove both soft terms |
| E | linear nonnegative | quadratic bounded | Cap-free E with safe C |
| F | linear nonnegative | linear bounded | Candidate linear E plus bounded C |
| G | quadratic bounded | linear nonnegative | C safety-failure probe |
| H | linear nonnegative | linear nonnegative | Combined safety probe |

G and H use the C abort rule. Report aborted runs as safety failures rather than
excluding them silently.

Fixed configuration:

~~~text
topology           = rg_coincidence
l2_init_total_frac = 0.95
eta                = 0.01
c_eta              = 0.001
leak_rate          = 0
refractory_steps   = 0
e_weight_cap       = 500
~~~

The historical cap is ignored only by E linear_nonnegative updates.

Fixed schedule:

~~~text
row 1 = 2500 boundaries
col 1 = 2500 boundaries
row 1 = 2500 boundaries
~~~

Use seeds 1 through 16 for the first pass. Run 4083693835 separately. If a candidate
changes the conclusion and passes initial safety gates, confirm it on 32 fresh seeds
without retuning.

Required metrics:

- row owner/dominance;
- column owner/dominance;
- returned-row owner/dominance;
- first rival and last incumbent column boundary;
- phase and final-500 winner counts;
- initial/final total weight per L2E;
- maximum individual weight;
- negative-FE count and most negative FE;
- total overshoot count and maximum;
- bounded-mode cap hits;
- cap-free weights above historical w_max;
- C coincidences, spikes, resets, weights, and cadence violations;
- L1E/RG firing ratios;
- L2 tie count;
- deterministic replay.

Keep every parameter identical between conditions.

## Phase 4: composition readiness diagnostic

Do not modify WTA.

After learning row 1 and col 1, present their plus-sign union. Immediately before the
first L2 winner's reset, record frozen-state counterfactual crossing predictions for
the learned row owner, learned column owner, and all other L2E cells.

For each seed report:

~~~text
tau_row
tau_col
absolute tau gap
which owner is earlier
whether each has a finite crossing this boundary
boundary-start membrane charge
active-input weight sum for each owner
~~~

The existing earliest cell must still fire and reset the others. An exact tie is not a
composition success. This phase asks whether both primitives are independently
detectable and measures their natural latency separation before later WTA work.

## Decision gates

### Remove the ordinary E quadratic term only if

- linear_bounded is deterministic and mechanically correct;
- convergence is materially faster;
- turnover and recovery are not degraded;
- total weights do not enter persistent negative-FE oscillation;
- behavior is not dominated by hard-cap clipping.

### Remove the ordinary E upper cap only if

- linear_nonnegative remains finite in every isolated/network run;
- no seed-dependent runaway occurs;
- negative-FE excursions self-correct;
- primitive selectivity remains intact;
- turnover/recovery remains robust;
- 32 fresh seeds confirm the result without retuning.

Do not add an undocumented emergency ceiling and call the result cap-free.

### Remove the C quadratic term only if

- c_linear_bounded preserves one-event-subthreshold behavior throughout training;
- two-event cadence remains possible;
- suppression/turnover is not degraded;
- faster C maturation does not close the column novelty window.

### Reject cap-free C under the existing FE equation if

- one coincidence ever fires C from rest;
- basal weight reaches w_1;
- frequency-halving cadence fails;
- weight diverges or becomes non-finite.

If rejected, conclude that C needs a temporally derived equilibrium equation before its
cap can be removed. Do not rescue it by changing threshold, leak, refractory, or WTA.

## Required regression coverage

Add tests for:

- exact algebra in every mode;
- byte-compatible production baseline;
- absence of hidden upper clipping in linear_nonnegative;
- nonnegative excitatory weights;
- deterministic replay for non-aborted modes;
- mode persistence across Reset and Reseed;
- custom-topology rebuild propagation;
- C gating/cadence under c_linear_bounded;
- explicit C one-event failure detection in the unbounded probe;
- unchanged one-L2-spike WTA;
- unchanged hard-reset timing;
- experiment result schema.

Run the full suite after every phase. Do not rewrite baseline expectations to conceal a
behavior change.

## Deliverables

Produce:

1. headless update-mode controls with current behavior as default;
2. focused unit/integration tests;
3. a reproducible experiment driver;
4. per-seed JSON results including abort reasons;
5. a Markdown report with weight and FE trajectory tables/plots;
6. every changed file and exact command used;
7. separate recommendations for:
   - E quadratic term;
   - E hard cap;
   - C quadratic term;
   - C temporal cap;
   - implications for later composition.

Do not promote a mode or alter dashboard defaults. Stop after evidence and recommendation.

## Claude kickoff prompt

~~~text
Execute Claude_Linear_Weight_Update_Ablation_Plan.md in phases.

Lock the current production behavior as the baseline. Do not change WTA, inhibition,
event ordering, initialization, learning rates, dashboard defaults, or presentation
durations.

Add independent, opt-in E and C update modes. First prove quadratic-bounded behavior is
byte-identical. Then isolate removal of the quadratic multiplier, followed by cap-free E
and guarded cap-free C probes.

Run the isolated trajectories and fixed multi-seed row1 -> col1 -> row1 factorial
experiment. Instrument the plus-sign probe only to record counterfactual pre-reset
crossing times; do not permit co-firing.

If a cap-free mode diverges or C can fire from one coincidence, stop that condition,
record the failure, and continue safe conditions. Do not retune parameters to rescue a
variant. Finish with JSON results and a Markdown recommendation; do not change
production defaults without review.
~~~
