# Claude Implementation Brief: Local Predictive Inhibition

This document is the complete normative specification. Do not look for a second
addendum. If a repository fact discovered during implementation contradicts this
document, stop, record the exact file and symbol, and resolve the contradiction
before changing behavior.

## 1. Mission and execution contract

Implement and evaluate one bounded hypothesis:

> A paired local L1E trace combined with dense L2E spike feedback can let each
> L1I column learn context-feature contingencies through a local bounded rule,
> suppressing continued predictable L1 activity more than equally frequent but
> contextually surprising activity.

This is not a general network rewrite. Preserve the current L2 competition,
current L2 feedforward learning, weighted L1 inhibition, fixed-point units, and
the shared `SimulationEngine`. A negative result is valid.

Work in the phases below. Run the relevant tests after every phase. Do not tune
against ownership, contextual suppression, or reversal results. Complete the
finite preregistered run before any exploratory sweep.

Before editing, produce a short architecture note naming the exact methods and
array layouts found in the repository. After implementation, produce the report
specified in Section 15.

## 2. Non-negotiable research constraints

- No backpropagation, gradients, labels, classification correctness, or global
  error signal.
- No pattern-specific learning rule, row/column mask, or inhibitory neuron per
  pattern.
- Learning must not read `engine.winner`, `current_pattern`, pattern names,
  runner trial labels, owner assignments, or an equivalent simulator result.
- An arriving L2E spike is valid local presynaptic information. Pattern and
  context labels may be used only by the runner for scheduling and evaluation.
- L1E, L1I, L2E, and L2I output polarity must continue to obey Dale's principle.
- Preserve deterministic behavior for a fixed seed, configuration, stimulus
  schedule, and feedback tape.
- Keep functional positions separate from frontend-only display spacing.
- Do not change L2 hard-reset competition, its loser update, or its parameters.
- Do not encode the desired feature/context mapping in initial weights, topology,
  neuron identities, or source ordering.
- Do not silently tune until a positive result or one-to-one ownership appears.

## 3. Verified repository baseline

Read these code paths before editing and verify this map in the pre-edit note.

### 3.1 Active network

The active network contains 9 L1E, 9 L1I, 8 L2E, and one shared L2I. Pixel
indices are row-major in the 3x3 input:

```text
0 1 2
3 4 5
6 7 8
```

Current connections are:

```text
external -> L1E
L1I_i ----| paired L1E_i
all L1E --> all L2E
all L2E --> shared L2I
shared L2I -- unweighted competitive reset --> all L2E
all L2E --> all L1I
```

There is no current `L1E_i -> L1I_i` afferent. Each L1I has an eight-element
positive vector containing L2E0..L2E7 feedback. All nine L1I neurons receive the
same L2E spike vector and start from copies of one task-independent random vector.

### 3.2 Current timing and mechanisms

- `SimulationEngine.step()` is the authoritative phase coordinator.
- External L1E charge is currently deposited before queued L1 inhibition is
  applied. Preserve that order: applying bounded inhibition at rest before the
  external pulse would remove no charge.
- An L1I spike at `t` is copied to `l1i_feedback_delay` and targets paired L1E at
  `t+1`.
- `Neuron.apply_inhibition()` performs linear subtraction floored at rest and
  records `v_pre` and `v_post`. Reuse it.
- L2 competition is `L2E spike -> L2I -> apply_competitive_reset()` and is not a
  learned L2I-to-L2E gate.
- The dashboard sets `l1i_immediate_relay=False`; L1I is a threshold accumulator.
- The active dashboard sets `l2e_weight_cap_frac=1/3`, `l2_charge_chunks=20`,
  `distance_weighting=True`, and `competitive_weight_update="redistribution"`.
- `SimulationEngine.params` is runtime truth. `backend/dashboard_config.py`
  supplies the dashboard overrides and its current 15 model controls.
- The headless runner changes patterns with no blank timestep. The normative
  inter-presentation blank interval for this experiment is therefore exactly 0.

### 3.3 Known runner defect authorized for repair

Active L2E neurons have exactly nine feedforward afferents at indices `0..8` and
no index-zero inhibitory placeholder. `experiments/runner.py` currently slices
some L2E arrays as `[1:1 + N_PIX]`, omitting pixel 0 from saturation metrics and
weight plots.

Create one authoritative helper returning the full `N_OUT x N_PIX` L2
feedforward matrix from indices `0:N_PIX`. Use it for metrics, plots, and saved
snapshots. Do not add a synthetic placeholder. Add a sentinel regression test
using nine distinct values `[0.11, 0.22, ..., 0.99]` and verify all nine columns,
including pixel 0, appear in order in the helper and plot data. Fix this before
recording new baseline metrics. Historical artifacts that omitted pixel 0 must
be labeled noncomparable, not overwritten.

## 4. Topology and configuration changes

### 4.1 Default-off paired path

Add exactly these nine optional projections:

```text
L1E_0 -> L1I_0
...
L1E_8 -> L1I_8
```

Never add cross-paired or all-to-all L1E-to-L1I edges. When enabled, every L1I
afferent array is exactly:

```text
index 0: paired local L1E
index 1..8: L2E0..L2E7 feedback
```

When disabled, retain the legacy eight-element array exactly. This is required
for bit-exact legacy behavior; do not leave a zero-valued ninth placeholder in
disabled mode.

Update construction, initialization, delivery, aligned `SynapseBank` state,
serialization, `_all_weights()`, deltas, emitted-edge diagnostics, inspector
labels if required, runner snapshots, and direct-array tests together. Serialize
the local edge as `local{i}`, kind `local_evidence`. Existing feedback IDs remain
`fb{j}->{i}`.

### 4.2 Required parameters

Add backend/headless parameters with legacy-preserving defaults:

```text
paired_local_enabled = false
predictive_feedback_enabled = false
l2_to_l1i_delivery_enabled = true
predictive_local_weight_frac = 0.40
predictive_feedback_init_frac = 0.10
predictive_feedback_eta_up = 0.08
predictive_feedback_eta_down = 0.04
predictive_trace_tau_steps = 2.0
predictive_l1i_leak_rate = 0.50
predictive_output_gate_frac = 0.50
```

Names may follow existing naming conventions, but record the final mapping in
the parameter registry. The feature flag combination, not a frontend label,
must determine behavior.

### 4.3 Fixed local afferent and learning dispatch

The local afferent uses one shared magnitude and is never plastic in this
experiment. Add a per-afferent plasticity mask if it fits `SynapseBank`; an
equally explicit strategy boundary is acceptable. Do not scatter `index != 0`
checks across learning code.

In predictive and local-only modes, disable the generic postsynaptic L1I
excitatory update entirely. Predictive feedback weights are updated only by the
rule in Section 6; the local afferent never changes. In legacy mode, preserve the
existing L1I assembly-credit behavior exactly.

## 5. Exact experimental modes

Implement these named headless presets. `legacy_feedback`, not `baseline`, is the
bit-exact current system.

| Mode | Paired local | Delivered L2 feedback | L1I feedback learning | L1E gate |
| --- | --- | --- | --- | --- |
| `baseline` | off | off | off | retained, receives no L1I events |
| `legacy_feedback` | off | on | existing legacy rule | untouched current gate |
| `local_only` | on | off | off | fixed `0.50 * theta_L1E` |
| `local_plus_feedback` | on | live L2E vector | predictor only | fixed `0.50 * theta_L1E` |
| `time_shuffled_feedback` | on | replay tape from Section 9 | predictor only | fixed `0.50 * theta_L1E` |

`baseline` is the no-L1-feedback-loop scientific control. `legacy_feedback` is
the historical reference and must be golden-equivalent when all new flags use
their defaults. `local_only` is an isolated-local negative control; under the
coincidence-sensitive preset it may legitimately produce no L1I spikes. The
rate-matched contextual control is `time_shuffled_feedback`.

All modes retain the same L2 network, feedforward learning, input schedule,
initial L2 weights, competition parameters, and random seed.

## 6. Exact local trace and predictor rule

### 6.1 Local trace

Each L1I `i` owns one scalar trace `x_i in [0,1]`, initialized to zero and driven
only by paired L1E spikes:

```text
lambda_trace = exp(-1 / tau_trace_steps)
x_i <- lambda_trace * x_i
if L1E_i spiked this outer timestep: x_i <- 1
```

Use `tau_trace_steps=2.0`, hence `lambda_trace=exp(-1/2)`. Do not read the input
vector. Reset traces only on a complete engine rebuild/reset; let them decay
across presentation boundaries.

### 6.2 Predictor update

For feedback source L2E `j` into target L1I `i`:

```text
G = theta_L1I - V_rest_L1I
u_ji = w_ji / G

on a delivered L2E_j spike only:
    u_ji <- clip(
        u_ji
        + eta_up   * x_i       * (1 - u_ji)
        - eta_down * (1 - x_i) * u_ji,
        0, 1
    )
    w_ji <- G * u_ji
```

Use `eta_up=0.08`, `eta_down=0.04`. Only delivered active feedback afferents
update. Inactive sources do not update. The local afferent never enters this
rule. Put the rule in one focused function/strategy under `snn/rules/` or an
equally focused existing abstraction and unit-test it independently.

Apply this synaptic update for every delivered feedback spike even when the
target L1I is refractory. Refractoriness may block membrane delivery through the
existing `receive_input()` behavior, but it must not make an arriving local
presynaptic event invisible to the predictor rule.

### 6.3 Delivery/update causality

For one timestep, snapshot the pre-update feedback weights, deliver L1I charge
using those weights, and then apply the predictor update. The spike that teaches
a synapse therefore does not benefit from its own new weight in the same event.
The update may occur before L1I threshold resolution because it no longer changes
already-deposited membrane charge.

Predictor learning and L1I delivery in `time_shuffled_feedback` both consume the
same replayed vector. Actual L2E spikes remain available separately for L2
bookkeeping and evaluation.

## 7. Exact predictive preset and timestep order

All charge-like fractions below are relative to threshold distance
`theta - V_rest` and must be converted to fixed-point units once.

```text
local L1E -> L1I weight             = 0.40 * theta_L1I
predictor feedback cap G            = 1.00 * theta_L1I
all predictor feedback initial w    = 0.10 * G
predictive L1I leak rate             = 0.50 per update
L1I refractory period                = existing value (2)
L1I -> L1E gate magnitude            = 0.50 * theta_L1E
L1I -> L1E inhibitory learning rate  = 0
```

All 72 predictor weights start identically. Do not use task structure. Set the
actual L1E inhibitory weight to the declared magnitude and set its inhibitory
learning rate to zero. Because it is fixed, the quadratic saturation denominator
does not determine its magnitude; document that fact rather than pretending it
is a linear cap.

Predictive mode uses an explicit L1I leak rate of `0.50`. The repository's
existing optional `L1I_LEAK_RATE=0.07` is insufficient: with a `0.40 theta`
local pulse, a held active feature would accumulate past threshold. Add a numeric
L1I leak-rate parameter while preserving the legacy default and disabled behavior.

The exact outer-timestep order is:

```text
1. Determine whether external input arrives on this timestep.
2. Deposit current external charge into each L1E.
3. Apply the queued t-1 L1I event through L1E.apply_inhibition().
4. Apply any existing manual stimulation and resolve L1E spikes.
5. Decay each local trace, then set it to 1 if its paired L1E spiked.
6. Deliver L1E spikes to L2 and run the unchanged L2 competition.
7. Select the delivered feedback vector: none, live, or replayed by mode.
8. Build exactly one L1I vector per target:
       [paired L1E spike, delivered L2E0 spike, ..., delivered L2E7 spike]
   (legacy disabled topology continues to receive its eight-element vector).
9. Snapshot weights and deliver that vector exactly once to L1I.
10. Apply predictor updates from the delivered feedback vector and current x_i.
11. Resolve L1I spikes.
12. Queue L1I spikes for paired L1E inhibition on t+1.
13. Preserve the existing update/bookkeeping order otherwise.
```

Do not build a general event queue or Boolean AND gate.

### 7.1 Required synthetic activation checks

At rest, using fractions of `theta_L1I`, verify numerically:

```text
one local event                              0.40: no spike
one initial feedback event                   0.10: no spike
one local + one initial feedback event       0.50: no spike
one local + one feedback event at u=0.80     1.20: spike
20 consecutive local-only timesteps with
  leak 0.50: no spike (pre-threshold limit is below 0.80 theta)
20 consecutive initial-feedback-only steps: no spike
```

Floating-point/fixed-point rounding must not turn an asymptotic threshold
approach into a spike. No task-level calibration is permitted. If these direct
checks fail after correct unit conversion, treat it as an implementation defect,
not permission to tune parameters.

## 8. Implementation phases and software tests

### Phase 0: baseline and runner repair

1. Run:

   ```bash
   for test in test_*.py; do PYTHONPATH=. .venv/bin/python "$test" || break; done
   PYTHONPATH=. .venv/bin/python tests/golden/test_golden_equiv.py
   ```

2. Save the active `DASHBOARD_OVERRIDES`, a fixed-seed/fixed-input 40-step trace,
   and test output.
3. Fix and test the runner pixel-zero defect without changing neural behavior.

### Phase 1: disabled topology

Add the paired path behind its default-off flag, serialization, aligned state,
and tests. With the flag off, the eight-afferent L1I arrays, histories, weights,
and golden test must remain bit-exact.

### Phase 2: trace, predictor, and preset

Add the trace, focused predictor rule, fixed local afferent, numeric predictive
leak, fixed output gate, and exact timing. Add isolated rule and activation tests
before running task experiments.

### Phase 3: modes, replay, and instrumentation

Add the five presets, deterministic stimulus schedules, reference feedback tape,
metrics, event history, and reproducible artifact reconstruction.

### Phase 4: preregistered evaluation

Run exactly the finite matrix in Sections 9-11. Do not stop early or extend a
failed run. Report negative results before any exploratory work.

### Required regression assertions

- All root regression scripts and golden equivalence pass with new features off.
- Same seed/config/schedule/tape produces identical full histories.
- Exactly nine paired edges exist only when enabled; there are no cross-pairs.
- Every enabled L1I retains all eight feedback afferents with correct offsets.
- Local weights never change.
- A context event with active local trace increases only its delivered `w_ji`.
- The same event with zero trace decreases only its delivered `w_ji`.
- Predictor weights clamp to `[0,G]`.
- Generic and predictive L1I learning cannot run simultaneously.
- A queued L1I spike affects only paired L1E on the next timestep through
  `apply_inhibition()`; charge removal is graded and floored at rest.
- L2 reset behavior and winner protection remain unchanged.
- Replay preserves its reference tape's exact per-source counts and all-zero
  vector count.
- Metrics can be recomputed exactly from saved event history.

## 9. Deterministic time-shuffled feedback control

Shuffle only the vector delivered on `L2E -> L1I`. Never alter live L2 spikes,
L2 competition, L2 logs, L1 activity, feedforward transmission, stimulus order,
or owner evaluation.

### 9.1 Reference tape

Run and save `local_plus_feedback` before `time_shuffled_feedback` for the same
seed and task. Save the complete eight-dimensional vector produced by live L2 on
every measured outer timestep:

```text
f[t] = [L2E0_spike, ..., L2E7_spike]
```

The shuffled mode replays a permutation of this reference tape even if its own
live L2 trajectory diverges. Persist both `actual_l2e[t]` and
`delivered_feedback[t]`; never call the replayed vector an actual L2 spike.

For the contextual task, permute acquisition and reversal tapes separately so
events never cross the reversal boundary. The one warm-up presentation is not
part of either tape.

### 9.2 Permutation

Use a dedicated `np.random.default_rng(seed + 10_000)`. Permute complete timestep
vectors, never individual bits. This preserves exact per-source counts,
within-vector coactivity, event count, and all-zero count relative to the
reference tape.

For a tape of length greater than one, attempt up to 100 seeded permutations and
accept the first derangement. If none is found, use a deterministic cyclic shift
by `max(1, floor(T/3))`. Record the method and permutation. There is no online
buffer, empty fallback, identity permutation, or fixed permutation of L2E IDs.

## 10. Exact stimulus protocols

Every task uses seeds `0..9`, dwell `20` outer timesteps, input period `1`, blank
interval `0`, and one extra warm-up presentation. Generate the complete schedule
before constructing mode-specific runs. The warm-up is a duplicate of the first
scheduled presentation and does not consume that presentation. During warm-up,
predictor learning is off and primary metrics are excluded; all existing L2
learning and legacy-mode behavior remain active. Time-shuffled mode receives its
own live feedback normally during warm-up.

### 10.1 Four-pattern task

Use the repository `PATTERNS` unchanged and the runner's deterministic
interleaved order:

```text
row 1, col 1, diag \\, diag /, repeat
```

Run 400 measured presentations total: exactly 100 per pattern. Report every 20
presentations. Always finish 400; never stop on stable ownership.

### 10.2 Matched-marginal contextual task

Use runner-managed trial IDs only for scheduling/evaluation. Deliver raw vectors
with `SimulationEngine.set_input()`; do not add trial labels to engine learning.

Define `F=center=index 4`, `G=middle-left=index 3`, and:

```text
X_with_F     = [1,0,1, 0,1,0, 0,0,0]  # indices 0,2,4
X_without_F  = [1,0,1, 1,0,0, 0,0,0]  # indices 0,2,3
Y_with_F     = [0,0,0, 0,1,0, 1,0,1]  # indices 4,6,8
Y_without_F  = [0,0,0, 1,0,0, 1,0,1]  # indices 3,6,8
```

Each 100-presentation acquisition block contains exactly:

```text
40 X_with_F, 10 X_without_F, 10 Y_with_F, 40 Y_without_F
```

Thus `P(F)=P(G)=0.5`, `P(F|X)=0.8`, and `P(F|Y)=0.2`; G has the opposite
conditional relationship. Independently shuffle each block with a dedicated
schedule RNG `np.random.default_rng(seed + 20_000)`. Reuse the saved schedule for
all modes of that seed. Run five acquisition blocks (500 presentations), report
every 25 presentations, and do not reset afterward.

### 10.3 Reversal

Continue the same contextual network for 300 presentations. Each independently
shuffled 100-presentation block contains:

```text
10 X_with_F, 40 X_without_F, 40 Y_with_F, 10 Y_without_F
```

Continue the same schedule RNG stream. Do not reset weights, membrane, refractory
state, queued inhibition, or local traces. Report every 10 presentations. Always
complete all three blocks.

## 11. Primary run matrix and artifacts

Run five modes x ten seeds x two tasks = 100 primary runs. The contextual
acquisition and reversal are one continuous run. The already-required
`local_plus_feedback` run supplies the reference tape for the corresponding
shuffled run; it is not an extra primary run.

Save, at minimum:

- resolved config and parameter registry;
- exact stimulus schedule and warm-up identity;
- actual L2E and delivered-feedback vectors per timestep;
- L1E/L1I/L2E/L2I spikes;
- local traces and predictor matrix at reporting boundaries;
- L1 inhibitory event records including target, trigger timestep, `v_pre`,
  `v_post`, and removed charge;
- per-presentation L2 spike counts and owner;
- initial and final relevant weight matrices;
- shuffle permutation/method;
- incremental status, metric, summary, and failure artifacts.

## 12. Exact metric definitions

### 12.1 Event charge

For an L1 inhibitory event:

```text
charge_removed = max(0, V_before - V_after)
```

Measure immediately around `apply_inhibition()`. Never substitute nominal gate
magnitude. Attribute the event to the L1I trigger at `t-1` and record the target
presentation at `t`.

### 12.2 Suppression opportunity and fraction

Context selectivity is expressed by how often/strongly an active feature is
suppressed, not by charge conditional on an inhibitory event. The latter would
be nearly constant with a fixed gate.

For feature `i` and a set of presentations `C`, exclude the first outer timestep
of each presentation from this metric to remove one-step carryover from the prior
presentation. On every remaining timestep where feature `i` is externally active
and `input_period` delivers a pulse, add the fixed positive external L1E synaptic
weight to offered charge `O_i(C)`. Sum actual removed charge targeting L1E_i on
those same timesteps as `Q_i(C)`.

```text
suppression_fraction(i,C) = Q_i(C) / max(1e-12, O_i(C))
```

Also report suppressed active-step rate and L1E spike rate per active external
step. These are evaluation metrics and may read the scheduled input; learning
may not.

### 12.3 Contextual suppression contrasts

During acquisition:

```text
CSC_F = SF(F, X_with_F) - SF(F, Y_with_F)
CSC_G = SF(G, Y_without_F) - SF(G, X_without_F)
CSC_primary = (CSC_F + CSC_G) / 2
```

Both are expected positive for a learned acquisition contingency and negative
after reversal. Compute windowed values and phase summaries over the last 100
presentations of acquisition and reversal.

For the four-pattern task, report `SF(i,pattern)` for active features and the
pattern-by-feature matrix; do not claim it alone demonstrates context.

### 12.4 L1I event classes

Before L1I integration, for each target `i`, define local event from current
paired L1E spike and feedback event from the vector actually delivered to L1I.
Classify `(1,0)=local_only`, `(0,1)=feedback_only`, `(1,1)=coincident`, and
`(0,0)=none`. Record the later L1I spike and report
`P(spike|local_only)`, `P(spike|feedback_only)`, and `P(spike|coincident)`.

### 12.5 Predictor matrix and source context

Store the source-major `8 x 9` matrix `W[j,i]`, even though each L1I owns one row
of eight feedback afferents internally. For each source `j`, normalize across
features and report entropy when its row sum is nonzero.

For evaluation only, assign each L2E source an acquisition context from its spike
rates over the final 100 acquisition presentations: X if its per-presentation
rate on the two X trials exceeds its rate on the two Y trials, Y if the reverse,
otherwise unassigned. Freeze this assignment for reversal analysis. Report mean
predictor weights to F and G separately for X-assigned and Y-assigned sources;
unassigned sources remain visible and are not forced into a group.

Define acquisition-oriented predictor contrasts from those frozen groups:

```text
PWD_F = mean_j_in_X W[j,F] - mean_j_in_Y W[j,F]
PWD_G = mean_j_in_Y W[j,G] - mean_j_in_X W[j,G]
```

Both are expected positive after acquisition and negative after reversal. If
either source group is empty, mark both grouped contrasts unavailable for that
seed; do not invent a grouping or drop the seed from aggregate counts.

### 12.6 L2 ownership

Presentation owner is the argmax of total L2E spikes during that presentation;
zero total spikes gives `None`; ties resolve to the lowest L2E index and the tie
count is reported. Compute trailing-20 modal owner consistency per trial type,
owner collisions across trial types, distinct owners, and dead L2E count. Do not
use `engine.winner` for learning.

### 12.7 Recovery and silence

Let acquisition-oriented `CSC_primary` be positive before reversal. Recovery is
the first reversal reporting window with negative `CSC_primary` that remains
negative for three consecutive windows. Recovery time is presentations elapsed
to the first of those windows; otherwise `not_recovered` at 300.

Apply the same three-window negative-sign rule separately to `PWD_F` and `PWD_G`
and report predictor recovery separately. An unavailable grouped contrast is
`not_recovered`, not missing from the denominator.

An L1E is temporarily silent in a reporting window when it has at least one
active external opportunity but emits zero spikes. It is permanently silenced
for this experiment after five consecutive such windows. Report identities and
counts; do not extrapolate beyond the budget.

### 12.8 Inhibitory amount and charge matching

Per window/mode report total removed charge, L1I spikes, L1E spike reduction, and
charge per presentation. Compare `local_only` and `local_plus_feedback` raw.
Also compare corresponding windows with the same seed, task phase, and window
index when their total removed charges differ by at most 10% of their mean. Do
not pair arbitrary time windows. If fewer than five corresponding pairs qualify
for a seed, label that seed's charge-matched analysis underpowered. Do not tune
gate strength to manufacture matches.

The shuffled replay is the primary rate-controlled test because it preserves the
reference feedback tape's source event counts. Report any difference between
reference and shuffled L1I output counts/charge rather than assuming delivery
count guarantees equal inhibition.

## 13. Scientific outcome rules

Software acceptance is independent of the hypothesis. The implementation is
accepted when all tests in Section 8 pass, artifacts reproduce metrics, and all
100 primary runs produce complete artifacts. A scientific failure such as no
spikes or no reversal is a valid completed run; an exception, corrupt artifact,
or externally interrupted run must be resumed or rerun and is not completion.

Use the last-100-acquisition `CSC_primary` as the preregistered primary score.
Call the first experiment supportive only if all are true:

1. Median `CSC_primary(local_plus_feedback)` is positive.
2. At least 8 of 10 paired seeds have
   `CSC_primary(local_plus_feedback) > CSC_primary(local_only)`.
3. At least 8 of 10 paired seeds have
   `CSC_primary(local_plus_feedback) > CSC_primary(time_shuffled_feedback)`.
4. With `M_live` and `M_shuffled` denoting those medians, the already-required
   `M_live > 0` is accompanied by `M_shuffled <= 0.75 * M_live`.
5. At least 7 of 10 live-feedback seeds recover the negative contrast within the
   300-presentation reversal budget.
6. No support criterion is achieved solely by permanent L1E silence; report any
   permanently silent feature as a failure for that seed.

Report predictor-matrix reversal, inhibitory charge, and L2 specialization as
secondary outcomes. If CSC succeeds but predictor grouping or reversal does not,
label the result mixed. If shuffled performs similarly, charge explains the
effect, reversal fails, or firing collapses, label the hypothesis unsupported or
mixed as appropriate. Do not change unrelated mechanisms within this evaluation.

## 14. Dashboard scope

Do not rebuild or prune the dashboard. The existing `CONFIG_SPEC` has 15 model
controls; removing them is unrelated scope. The headless runner is the primary
experiment controller.

If a small browser addition is useful, add at most a predictive preset selector,
paired-local enabled, predictor enabled, and one strength control (choose either
local weight or output gate). Reuse `backend/dashboard_config.py`; do not
duplicate defaults in JavaScript or create nested configuration/import systems.
Read-only diagnostics may reuse existing inspectors/charts with small changes.

## 15. Deliverables and report

Provide small phased commits when the worktree permits, otherwise clearly
separated diffs. Do not modify unrelated existing user files.

The final report must contain:

1. Pre/post architecture map with exact methods and final array layouts.
2. Human- and machine-readable parameter registry.
3. Baseline trace, root regression results, golden equivalence, deterministic
   repeat result, and pixel-zero sentinel test.
4. Per-seed and aggregate mean, standard deviation, median, minimum, and maximum
   for every mode and task; never only the best seed.
5. `8 x 9` predictor matrices over time, CSC over time, inhibitory charge,
   L1E/L1I spikes, L2 ownership/collisions, reversal trajectories, and silence
   diagnostics.
6. Exact shuffle checks and whether derangement or fallback was used.
7. Direct answers:
   - Did local information create feature-specific predictors?
   - Was selectivity contextual rather than feature frequency?
   - Did time shuffling remove it?
   - Did suppression and predictor weights reverse?
   - Did L2 specialization change independently of total firing?
   - What failed?
8. Limitations and exactly one next experiment justified by these data.

## 16. Likely implementation locations

- `backend/simulation.py`: parameters, construction, timing, traces, modes,
  serialization sources, and weight snapshots.
- `backend/dashboard_config.py`: active preset and optional small UI additions.
- `neuron_flexible.py`: existing weighted inhibition and learning dispatch.
- `snn/synapses.py`: aligned per-afferent state/mask.
- `snn/rules/`: focused predictor strategy.
- `layers.py`: L1 E/I construction.
- `backend/serializer.py`: topology/dynamic envelopes.
- `frontend/inspector.js`, `frontend/charts.js`, and focused modules only if a
  small diagnostic change is justified.
- `experiments/runner.py`: schedules, reference/replay, metrics, artifacts, and
  the existing L2 array-slice repair.
- root `test_*.py` and `tests/golden/`: regression contracts.

Do not create a second simulator or parallel configuration system. The reusable
tile must emerge from local traces, dense candidate feedback, bounded synaptic
weights, spike timing, and the existing delayed inhibitory output—not labels,
simulator-owned winners, or task-specific wiring.
