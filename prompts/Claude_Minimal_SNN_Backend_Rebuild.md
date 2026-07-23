# Claude implementation brief: minimal SNN backend rebuild

## Role and outcome

Act as the primary implementation engineer for this repository. This is an intentional backend replacement, not a compatibility-preserving refactor. The present model accumulated many experimental branches, flags, helper methods, and regression scripts. Replace it with the smallest coherent implementation of the network specified below while preserving the dashboard's visual design, layout, interaction model, and overall frontend structure.

Do the work in the repository, run the relevant verification, and leave the tree in a clean, understandable state. Do not stop after proposing an architecture. Implement it. Do not retain dead code merely because an old test imports it. Old tests and old backend APIs are not requirements unless this brief explicitly preserves them.

Before editing, inspect `git status` and `git diff`. The worktree contains intentional uncommitted work. Do not discard or overwrite unrelated user changes. Read the Markdown sources for scientific context, especially:

- `docs/REFACTOR.txt` (the authoritative new direction)
- `Current_Implementation_Methodology_Equations.md` (historical equations and terminology)
- `README.md`
- `docs/DASHBOARD.md` (browser protocol and frontend boundary)
- `Experiment.md` and the existing Claude briefs (historical context only; their predictive-learning machinery is superseded where it conflicts with this brief)

The finished repository must make the active model obvious from a small number of files. Someone should be able to understand neuron behavior, topology, timestep order, and configuration without following layers of strategies or feature flags.

## Current-state diagnosis to confirm

At the time this brief was prepared:

- `backend/simulation.py` is roughly 134 KB and mixes construction, many experiment modes, learning variants, stepping, diagnostics, episode logic, and serialization.
- `neuron_flexible.py` is roughly 61 KB and exposes a large general-purpose class with many forwarded properties and inactive mechanisms.
- `snn/rules/` contains strategy selection for mutually exclusive delivery, excitatory, inhibitory, and predictive rules.
- There are roughly 27 root-level `test_*.py` scripts plus golden fixtures.
- `SimulationEngine.__init__` has a very large option surface containing legacy ablations and mutually exclusive modes.
- The frontend consumes a stable topology/dynamic-state protocol and should remain visually and structurally recognizable.

Confirm these facts, then replace rather than incrementally decorate this architecture.

## Non-negotiable design principles

1. There is one active scientific model, not a family of legacy modes.
2. Excitatory and inhibitory neurons are separate, small types with different responsibilities. Do not retain a universal neuron class with dozens of mode flags.
3. The engine owns topology and causal event order. Neuron classes own only local state transitions.
4. Configuration is small. Derived invariants are computed, not independently configurable.
5. Subtractive weights are positive magnitudes plus one explicit negative sign convention. Do not store a mixture of positive and negative values in one ambiguous vector.
6. Inhibitory neurons have no incoming weight vector and no plasticity.
7. The frontend remains a view/controller. No neural computation moves into JavaScript.
8. Model coordinates and display coordinates stay separate. Preserve the frontend's current Three.js styling and display transform.
9. Delete obsolete paths and tests. Do not leave commented-out implementations, compatibility shims, or “temporarily disabled” mechanisms.
10. Prefer direct NumPy and plain Python over frameworks or abstractions that do not earn their complexity.

## Resolve apparent specification conflicts this way

The source note contains a few phrases that can otherwise lead to incompatible implementations. Use these resolutions and document them:

- “All neurons should have the same thresholds and weight caps” means all **excitatory** populations use one shared excitatory threshold and one shared accumulating-weight cap. Inhibitory neurons are the explicit exception: their reported threshold is always `E_THRESHOLD / 3`. Inhibitory neurons have no weights, so a weight cap is not meaningful for them.
- The shared excitatory threshold is explicitly `1000` for every E population. Do
  not retain the old L2 threshold of `8000`, and do not normalize the rebuilt model's
  threshold to `1.0`. The value `1000` is intentional because it is easier to inspect
  and interpret in the dashboard. The derived inhibitory threshold is therefore
  `1000 / 3` (approximately `333.3333`).
- The note first says subtractive weights are pretrained/frozen and later says they are “yet to be trained.” Treat the more explicit instruction as authoritative: subtractive weights are pretrained and frozen; there is no subtractive/inhibitory plasticity in this rebuild.
- A “hard charge wipe” means an inhibitory event must leave the target excitatory membrane at its resting potential, never below it. Represent the subtractive gate as a magnitude capped at the target's threshold, but enforce the hard-wipe invariant explicitly so overshoot, numeric scale, or event ordering cannot leave residual charge.
- No leak and no refractory period are the baseline. Keep both configurable, but their defaults are exactly zero. Use one leak-rate value, not an enable toggle plus separate per-layer leak settings. Use one excitatory refractory value, not per-layer values.
- Preserve the existing four 3x3 patterns and the 9-pixel input surface.
- The old predictive L1E→L1I evidence path and learned L1I output gate do not survive. The new `L1E_new → L1I → L1E_s` path replaces that mechanism.

If repository evidence forces a different interpretation, record the evidence and the exact decision in the new methodology document. Do not silently retain old behavior.

## Minimal domain model

### Excitatory neuron

Implement a compact `ExcitatoryNeuron` with only state used by the active model. The exact packaging can vary, but the conceptual state should be no larger than:

- stable identifier and population/role metadata
- membrane potential and resting potential
- shared threshold
- leak rate
- refractory period and timer
- `acc_weights`: nonnegative accumulating/excitatory magnitudes
- `subt_weights`: nonnegative subtractive/inhibitory magnitudes
- one explicit subtractive sign constant, preferably `SUBTRACTIVE_SIGN = -1`
- aligned accumulating-input participation state needed by the local update
- aligned per-afferent geometric distances/factors used by learning only
- spike state for the current step

Do not expose raw mutable internals through a forest of compatibility properties. A few clear methods are sufficient, for example:

- `receive_acc(spikes)`
- `receive_subt(spikes)` or `hard_wipe()`
- `can_fire()` / `fire()`
- `advance()` for leak and refractory countdown
- `update_acc_weights(participation)`
- a small serialization method if it materially simplifies the engine

Required behavior:

1. Accumulating inputs add charge:

   ```text
   V <- V + sum_i(acc_weights[i] * spike_i)
   ```

   **Geometry must not attenuate, scale, delay, or otherwise modify delivered charge.**
   Equal stored weights receiving equal spikes deliver equal charge regardless of synapse
   length. This deliberately removes the repository's old distance-weighted delivery
   behavior.

2. Subtractive inputs use positive magnitudes under an explicit negative sign:

   ```text
   signed_subtractive_charge = SUBTRACTIVE_SIGN * sum_k(subt_weights[k] * spike_k)
   V <- max(V_rest, V + signed_subtractive_charge)
   ```

   For the active topology, every inhibitory output gate is pretrained to the target excitatory threshold. On a real inhibitory spike, assert the hard-wipe postcondition `V == V_rest` within numeric tolerance. It is acceptable and clearer to call `hard_wipe()` after recording the signed subtractive event.

3. A neuron fires only when not refractory and `V >= threshold`. Firing records `v_pre`, resets membrane to rest, marks `spiked`, and starts the configured refractory timer.

4. Leak is applied once per completed timestep when not refractory:

   ```text
   V <- V_rest + (1 - leak_rate) * (V - V_rest)
   ```

   Validate `0 <= leak_rate <= 1`. The default is `0.0`.

5. Refractory behavior must have a documented, tested off-by-one convention. The default is `0`.

### Excitatory accumulating-weight update

There is exactly one accumulating-weight rule. It runs when the postsynaptic excitatory neuron fires. It never updates `subt_weights`.

Preserve the requested nonlinear ingredients and do not fall back to the old `p = threshold / v_pre` rule:

```text
p = threshold - sum(acc_weights)
signal_i = +1 if accumulating afferent i participated in the causal input window
           -1 otherwise
distance_factor_i = (distance_ref / max(distance_i, distance_min)) ** distance_power
delta_w_i = eta * p * signal_i * distance_factor_i * (1 - (w_i / w_max) ** 2)
acc_weights[i] <- clip(acc_weights[i] + delta_w_i, 0, w_max)
```

Important implementation constraints:

- Use `(w / w_max) ** 2`, not the dimensionally different historical expression `w**2 / w_max`.
- Compute `p` from the pre-update accumulating weights.
- Use the literal signed `p = threshold - sum(acc_weights)` unless repository evidence establishes a previously agreed normalization. Keep the required E threshold at `1000`; if the equation needs numerical calibration, adjust and document `eta` or initialization values rather than normalizing the threshold to `1.0` or restoring `8000`.
- Synapse geometry is used **only here, in plasticity**. `distance_i` comes from the
  Euclidean distance between the source and target functional/model coordinates.
  `distance_factor_i` is the normalized inverse-distance concept already used by the
  repository, relocated from charge delivery into this learning equation. Compute a
  deterministic reference distance for the relevant projection so the closest synapse
  has factor `1` and longer synapses have factors in `(0, 1]`. Keep the exponent fixed
  in the model unless there is a strong scientific reason to expose it. Do not create
  distance modes or apply this factor a second time during delivery.
- Functional coordinates therefore affect how quickly a synapse learns, never the
  instantaneous charge produced by its current weight. Frontend-expanded display
  coordinates affect neither delivery nor learning.
- The rule as written has meaningful behavior when `sum(acc_weights)` crosses threshold because `p` changes sign. Add tests on both sides of that boundary. Do not clamp or take the absolute value without explicitly documenting and justifying a change.
- Define the causal participation window simply. Prefer the spikes delivered in the threshold-crossing step. Do not reintroduce eligibility traces unless the measured frequency experiment proves they are necessary.
- All accumulating weights share the same cap `E_WEIGHT_CAP` unless a narrowly documented input source requires a fixed pretrained value.

L1 sensory accumulating weights may be initialized as pretrained values, matching the scientific assumption that sensory encoding already exists. They should still use the same data representation. If they are frozen, make that a topology-level fact rather than a browser configuration mode.

### Inhibitory neuron

Implement a very small `InhibitoryNeuron` instant relay:

- no weight vector
- no membrane integration
- no learning rule
- no leak state
- no refractory state unless an unavoidable engine invariant requires a boolean only
- a reported/serialized threshold equal to one third of the shared E threshold
- a boolean/event such as `received_signal` and `spiked`

Any incoming `+1` spike causes it to fire immediately in the same causal resolution phase and emit a `+1` event to every connected excitatory target. It should not wait to accumulate to its serialized threshold. The one-third threshold is a scientific/visual invariant, not a second integration mechanism.

Do not represent E→I connection strength as a trainable weight. Connectivity is structural. A source E spike is a relay signal.

## Required topology

Use clear role names in code and user-facing labels. Preserve IDs that help the existing frontend where practical, but do not contort the model for obsolete names.

### Layer 1

Create 27 neurons in three aligned groups of nine:

- `L1E_s[0..8]`: the original/source sensory excitatory neurons, one per 3x3 pixel
- `L1E_new[0..8]`: new supervisory excitatory neurons
- `L1I[0..8]`: instant inhibitory relays

Connections:

```text
external pixel i -> L1E_s[i]
L1E_new[i]       -> L1I[i]       (structural +1 relay signal)
L1I[i]           -> L1E_s[i]     (pretrained subtractive gate; hard wipe)
```

The supervisory branch is paired one-to-one. `L1E_new[i]` integrates feedback from L2. When it fires, its paired L1I fires immediately and wipes the paired `L1E_s[i]` charge. This late-step wipe naturally changes future L1 source frequency; do not add the old learned output gate or old predictor traces.

### Layer 2

Create:

- `L2E[0..7]`: eight excitatory competitors
- `L2I`: one shared instant inhibitory relay

Connections:

```text
all 9 L1E_s -> every L2E               (dense accumulating feedforward)
every L2E   -> L2I                      (structural +1 relay signal)
L2I         -> every L2E                (pretrained subtractive hard wipe)
every L2E   -> every L1E_new            (dense accumulating feedback)
```

This is the requested replacement for the old direct `L2E -> L1I` feedback. There must be no `L2E -> L1I` edges.

The L2 bidirectional language means `L2E -> L2I` relay connectivity plus `L2I -> L2E` subtractive connectivity. It does not mean a learned inhibitory vector.

### Topology diagram

```text
9 external pixels
       |
       v
  9 x L1E_s  ==================>  8 x L2E  ----->  1 x L2I
       ^          dense acc          |   ^             |
       |                             |   |             |
       | paired frozen subtractive   |   +-------------+
       | hard wipe                   |    frozen subtractive hard wipe
  9 x L1I                            |
       ^                             |
       | instant paired relay        |
  9 x L1E_new  <=====================+
                    dense acc feedback
```

## Deterministic timestep and event order

Write the phase order as code that can be read from top to bottom. Avoid recursive propagation and hidden same-step side effects.

Use this baseline order:

1. Clear per-step spike/event flags.
2. Sample the current nine-pixel input according to the existing held-input behavior.
3. Deliver active external inputs to `L1E_s` accumulating afferents.
4. Resolve `L1E_s` threshold crossings; firing performs its one accumulating-weight update.
5. Deliver `L1E_s` spikes densely to all `L2E` accumulating inputs.
6. Resolve L2 competition deterministically:
   - determine which L2E neurons crossed threshold after the same delivered volley;
   - if more than one crossed, choose the winner by highest pre-fire membrane, then stable neuron index as tie-breaker;
   - only the chosen winner fires and applies its accumulating-weight update;
   - its structural `+1` event makes L2I relay immediately;
   - L2I hard-wipes all L2E membranes, including any non-fired threshold crossers and the already-reset winner;
   - record the winner and every emitted/reset edge for the frontend.
7. Deliver the L2 winner spike through the dense L2E→L1E_new accumulating feedback matrix.
8. Resolve any `L1E_new` threshold crossings and their accumulating-weight updates.
9. Each spiking `L1E_new[i]` immediately triggers `L1I[i]`; that relay hard-wipes `L1E_s[i]` in the same phase. Since L1E_s firing was already resolved earlier this step, this regulates subsequent firing without an extra legacy delay register.
10. Apply leak and refractory countdown exactly once to excitatory neurons.
11. Record history, sparse weight changes, spike frequency, logs, and the dynamic snapshot.

Do not retain charge chunks, event-driven versus cycle-driven branches, flow-rate traces, episode winner logic, loser-depression modes, or multiple reset modes. If substep resolution is genuinely required, implement one fixed deterministic method and document it; do not make it another mode selector.

## WTA tyranny and frequency modulation: evidence-driven engineering task

The baseline topology intentionally makes inhibition powerful. The open research question is how a frequent winner suppresses its supporting L1 source frequency enough to leave the competition, allowing other L2 neurons to emerge, without an engine command that directly disables the winner.

Treat this as a second implementation phase after the minimal baseline is correct.

### Required causal hypothesis

The intended loop is:

```text
frequent L2 winner
  -> repeated dense feedback into L1E_new
  -> selected L1E_new threshold crossings
  -> paired instant L1I relays
  -> pauses/resets in corresponding L1E_s
  -> lower L1E_s presentation frequency
  -> the old L2 winner no longer integrates quickly enough
  -> remaining L2 pool gets an opportunity to win
```

Winner removal must be emergent from local spikes, membrane state, topology, and timing. Forbidden shortcuts include winner blacklists, cooldown lists owned by the engine, forced round-robin selection, label-aware routing, pattern identity in a learning rule, or directly zeroing a winner's feedforward weights.

### Baseline limitation to acknowledge

With exactly zero leak and no finite integration window, a pure integrator is sensitive to total spike count but not asymptotic input frequency: half-frequency input generally crosses the same threshold later rather than never. Therefore do not claim that a zero-leak pure integrator solves frequency tyranny merely because firing slows.

### Primary frequency model: jointly engineer leak and the shared weight cap

Make the leaky periodic-integrator model the primary quantitative model for the
frequency experiment. Do not sweep leak in isolation and do not choose a cap from the
task-specific fact that the current line patterns happen to activate three pixels.

Let:

```text
theta       = the shared E threshold (fixed at 1000)
lambda      = leak_rate
r           = 1 - lambda, the retained membrane fraction per step
T_fast      = measured steps between volleys in the high-frequency condition
T_slow      = measured steps between volleys after frequency modulation, T_slow > T_fast
N_active    = number of accumulating afferents that actually spike in a volley
Q           = sum of the raw weights of those active afferents
w_bar       = Q / N_active, their mean active weight when N_active > 0
```

For an ideal periodic train with the same charge `Q` every `T` steps and
`0 < lambda <= 1`, the subthreshold steady-state post-volley peak (before any
fire/reset event) is:

```text
V_peak(T) = Q / (1 - r ** T)
```

The desired frequency-selective regime is:

```text
theta * (1 - r ** T_fast) <= Q < theta * (1 - r ** T_slow)
```

For approximately equal active weights, generalize this to any number of active
afferents:

```text
theta * (1 - r ** T_fast) / N_active
    <= w_bar <
theta * (1 - r ** T_slow) / N_active
```

The familiar full-rate versus half-rate special case is only `T_fast = 1` and
`T_slow = 2`:

```text
theta * (1 - r) / N_active
    <= w_bar <
theta * (1 - r ** 2) / N_active
```

Thus division by `3` is not a model constant. It is division by the observed
`N_active` for that neuron and event stream. Distinguish carefully between total
fan-in and active fan-in: L2E has nine possible L1E_s afferents but normally three
are active for this task; L1E_new has eight possible L2E afferents but a strict WTA
event may activate only one; L1E_s has one sensory afferent. The equations and code
must remain valid if patterns, fan-in, or the number of simultaneously active inputs
change.

For heterogeneous learned weights, use the exact `Q = sum(active_weights)` inequality
as authoritative; the `w_bar` form is only an interpretable summary. A shared
per-synapse `w_max` can be tested by substituting `Q = N_active * w_max` only when the
question is explicitly about all active weights reaching the cap. In normal learned
state, measure the actual weights because a cap is an upper bound, not proof that every
weight matures to it.

Real network spike trains will not be perfectly periodic. For the actual observed
inter-volley gaps `Delta_t[k]` and volley charges `Q[k]`, validate the candidate by
replaying the local recurrence:

```text
V_pre[k]  = V_post[k-1] * r ** Delta_t[k]
V_post[k] = V_pre[k] + Q[k]
```

using the engine's exact deposit/leak phase convention. This empirical recurrence and
the actual membrane trajectory take precedence over the periodic approximation.
Geometry must not appear in `Q`; geometry affects learning deltas only.

Search jointly over `leak_rate` and one globally shared E `weight_cap`. Keep the E
threshold fixed at `1000`. For every candidate pair, evaluate the fast/slow inequality,
then run the real network to test turnover and recovery. Do not optimize a separate cap
per population.

One shared per-synapse cap does not impose the same maximum volley charge on populations
with different active fan-in. Preserve the shared cap, but initialize each accumulating
projection at a fixed, documented fraction or range chosen for that projection's
expected `N_active` and spike interval:

```text
target_Q_projection from the inequality above
target_mean_weight_projection = target_Q_projection / expected_N_active_projection
initial weights = deterministic narrow jitter around target_mean_weight_projection,
                  clipped to [0, shared_weight_cap]
```

This projection-specific initialization is a construction policy, not a collection of
runtime configuration options. Do not add a total incoming-weight budget, normalize
delivered charge, or renormalize weights during simulation. The universal cap remains
the biological bound; topology, active fan-in, and fixed initialization determine each
projection's operational scale.

### Experiment before promoting a mechanism

Build a small deterministic headless experiment (not a new general experiment framework) that includes a controlled full-rate/half-rate comparison but bases the network conclusion on the high-rate and feedback-modulated spike trains actually produced by L1E_s. Record:

- L1E_s, L1E_new, L1I, L2E, and L2I spike times/rates
- L2 winner identity over time
- time to L2 threshold under controlled full/half input and under the actual measured high/modulated L1E_s frequencies
- whether the initial winner stops winning
- whether another L2E wins
- whether the original winner can recover after input conditions change
- relevant membrane trajectories

Evaluate the smallest biologically/local plausible frequency discriminator. The first candidate is the joint leak/shared-cap model above because leak directly makes integration rate-sensitive and adds no new state. Measure `N_active`, `Q`, and inter-volley intervals from actual L1E_s spikes rather than assuming ideal full/half schedules. Search for a leak/cap regime in which high-frequency input integrates while the feedback-modulated frequency does not, or integrates sufficiently slowly for competitors to emerge.

If leak cannot satisfy the behavioral criteria, evaluate at most one minimal time-local alternative, such as a fixed finite integration window or a local inter-spike-interval-dependent charge factor. Any alternative must:

- be local to an excitatory neuron or synapse;
- have a clear equation involving only locally available spike timing/state;
- add at most one production parameter;
- default to an explicitly documented value;
- avoid global winner knowledge;
- be justified by measured behavior, not intuition alone.

Do not bring the historical trace/flow-rate framework back wholesale. Keep the default dashboard leak at `0.0` as requested even if a nonzero experimental setting demonstrates the tyranny remedy. Document clearly that the zero-leak default exposes the baseline behavior and that a nonzero slider setting is needed if the data says so.

The experiment is successful only if it demonstrates both turnover and recovery without direct winner manipulation. If no candidate succeeds, leave the clean baseline intact, include the negative result and plots/data summary, and do not hide a heuristic in production.

## Configuration surface

Replace the giant constructor signature and flag dictionary with one small validated configuration object. Start with only:

```text
seed
e_threshold
e_weight_cap
eta
leak_rate             default 0.0
refractory_steps      default 0
input_period          only if the existing input controls require it
```

Use this explicit baseline unless measured behavior requires recalibration of the
learning rate, shared cap, or projection initialization. The threshold itself is not
subject to recalibration:

```text
seed = persisted dashboard seed (fallback 1)
e_threshold = 1000
e_weight_cap = 1000 as the initial shared candidate; the joint frequency experiment
               may select a different final shared value
eta = 0.01
leak_rate = 0.0
refractory_steps = 0
input_period = 1, if retained
```

Do not treat the earlier `Uniform(80, 120)` suggestion as an incidental universal
initialization. Derive one deterministic narrow initialization range for each
accumulating projection from its expected active-afferent count and actual spike
intervals using the frequency model above. The pretrained sensory weight must also be
within the same shared cap. Keep these projection initialization constants in code and
documentation, not in the dashboard configuration. Report the derivation and final
values. Do not add initialization modes or change the shared E threshold away from
`1000`.

Derive rather than configure:

```text
i_threshold = e_threshold / 3
subt_weight_cap = target_e_threshold
L1/L2 population sizes = 9 / 9 / 9 / 8 / 1 constants
distance normalization parameters for learning = fixed documented constants
```

The frontend configuration panel should be correspondingly small. At minimum, retain sliders for leak and refractory because they were explicitly requested. It is reasonable to expose shared E threshold, shared cap, and eta if the current UI makes that useful. Do not expose topology sizes, per-layer thresholds, separate leak toggles, reset choices, learning-rule choices, initialization modes, confidence, budgets, homeostasis, predictive flags, charge chunks, flow rate, or legacy ablations.

Applying a configuration that changes model structure/state may rebuild the engine and clear learned state, as the existing dashboard does. Validate overrides against an allowlist; do not accept arbitrary stale keys.

## Frontend and protocol preservation

Keep the styling and recognizable component structure under `frontend/`: Three.js scene, controls, input grid, inspector, raster/charge/weight panels, colors, typography, camera behavior, and websocket-driven store.

Backend cleanup is allowed to simplify payload fields, but preserve the main protocol shape so the frontend does not need a rewrite:

```text
topology:
  neurons: id, label, layer, type, threshold, pos
  synapses: id, source, target, kind, weight (or null for structural relay edges)
  layers, patterns, pattern_vectors, grid, params

dynamic:
  timestep, running, speed
  neurons: id, potential, activation, spiked, freq, refractory
  changed_synapses
  emitted
  input, winner
  stats, log
```

Remove fields that only support deleted mechanisms (`confidence`, budget data, episode state, predictive traces, learned inhibitory event details, charge-chunk diagnostics) and update JavaScript consumers accordingly. Do not keep fake empty fields solely for old tests.

Add the `L1E_new` population to the existing visual layout without redesigning the page. Keep `layer` values compatible with L1/L2 filters; add an explicit role/population field if needed so source and supervisory E neurons can have distinct positions/labels. Update renderer radius lookup only as necessary. Reuse the existing E visual language rather than inventing a third neuron type.

Use unambiguous synapse kinds, for example:

- `sensory`
- `feedforward`
- `feedback`
- `relay_excitation` for E→I structural event edges
- `inhibition` for I→E pretrained subtractive edges

Structural E→I relay edges may have `weight: null`. I→E edges should expose their frozen subtractive magnitude as a positive weight plus inhibitory kind/sign metadata. The UI should label accumulating versus subtractive magnitudes clearly; it must not make a positive stored subtractive magnitude look excitatory.

Keep the REST verbs that the current page actually calls: state, start/pause/step, reset/reseed, speed, pattern/input/pixel/clear/random/noise, config, and manual feedforward weight edit if the receptive-field editor remains. Remove unused endpoint/model machinery only after searching frontend call sites.

## Backend file shape

Use judgment on exact filenames, but target a compact structure similar to:

```text
snn/
  neurons.py          ExcitatoryNeuron, InhibitoryNeuron
  plasticity.py       the one accumulating-weight equation (only if separation helps)
  network.py          topology/connectivity data (only if it keeps the engine smaller)
backend/
  simulation.py       construction, deterministic step, state snapshots
  dashboard_config.py small defaults and control schema
  api.py              HTTP/WS adapter only
  layout.py           functional positions
  serializer.py
  websocket.py
```

Delete or replace obsolete model files such as the universal flexible neuron, cortical-column wrapper, generic layer wrapper, rule-selector modules, predictive rule module, and configuration-forwarding wrappers once nothing active imports them. Search the entire repository before deletion. Update README imports and commands.

Keep the headless experiment directory only to the extent that it remains useful with the new engine. Remove old predictive experiment Python code if it cannot run and is no longer scientifically relevant, but preserve Markdown reports as historical records unless they are factually rewritten as current behavior. Do not preserve broken executable code merely as an archive.

## Test cleanup and replacement

Delete the golden baseline and tests for removed mechanisms. Exact equivalence to the old engine is specifically not a goal.

Replace the large test set with a small behavioral suite, preferably normal `pytest` tests under `tests/`. Aim for roughly 6–8 focused files, not dozens of executable scripts. Required coverage:

1. **Excitatory neuron**
   - accumulating charge is exactly the raw weighted spike sum and is invariant to geometry
   - geometric distance changes the learning delta, not delivered charge
   - threshold/reset
   - the exact nonlinear weight equation for participating `+1` and absent `-1` inputs
   - behavior for `sum(acc_weights) < threshold`, equal, and greater
   - frozen subtractive weights and hard-wipe postcondition
   - leak zero/nonzero and refractory off-by-one

2. **Inhibitory relay**
   - no weight vector
   - no spike without input
   - any `+1` input produces an immediate binary spike
   - serialized threshold is one third of E threshold

3. **Topology**
   - exact counts: 18 L1 E, 9 L1 I, 8 L2 E, 1 L2 I; total 36
   - exact required internal edge counts: 9 L1E_new→L1I + 9 L1I→L1E_s + 72 L1E_s→L2E + 8 L2E→L2I + 8 L2I→L2E + 72 L2E→L1E_new = 178 internal edges
   - if external sensory afferents are serialized as edges, treat the 9 sensory edges separately (187 including sensory) and ensure their source representation is valid for the frontend
   - no `L2E -> L1I`
   - every I→E subtractive magnitude equals the target E threshold and is frozen

4. **Causal step/WTA**
   - stable tie-break
   - one winner for simultaneous L2 crossings
   - winner invokes L2I immediately
   - all L2E membranes are at rest after the inhibitory event
   - L2 feedback reaches L1E_new, which relays through paired L1I and wipes only paired L1E_s

5. **Serialization/API smoke**
   - topology/dynamic payloads contain what the frontend consumes
   - new population IDs/roles are present
   - config rejects deleted keys
   - reset/reseed and one websocket or REST state cycle work

6. **Frequency behavior experiment/regression**
   - deterministic high-rate versus feedback-modulated-rate measurement using actual emitted L1E_s spike intervals
   - parameterized checks of the generalized inequality for multiple `N_active` values, including but not limited to the current three-active-afferent pattern
   - heterogeneous active weights use `Q = sum(active_weights)`, not `N_active * cap`
   - joint leak/shared-cap candidates agree with the engine's irregular-interval membrane recurrence
   - if a mechanism succeeds, pin the chosen regime's turnover and recovery criteria without overfitting exact long floating-point histories

7. **Frontend static smoke**
   - serve the page and ensure JS references only payload fields still emitted, if practical in the existing toolchain

Tests should assert scientific invariants and causal behavior, not internal helper calls. Avoid huge fixtures and exact golden trajectories.

## Documentation rewrite

Rewrite `Current_Implementation_Methodology_Equations.md` to describe only the implemented model. It must include:

- population counts and exact topology
- state variables
- accumulating and subtractive delivery equations
- the one accumulating-weight update equation
- frozen subtractive gates
- instant inhibitory relay semantics and one-third threshold invariant
- precise timestep order
- shared configuration and derived values
- leak/refractory defaults
- the measured status of the frequency-tyranny hypothesis, including limitations of zero-leak integration

Update `README.md` and `docs/DASHBOARD.md` so their code maps, run commands, protocol fields, and test commands are accurate. Historical reports may remain historical, but no current-facing document should point readers to deleted classes or mechanisms.

## Implementation phases

### Phase 0: protect the worktree

- Inspect status/diff and identify user changes.
- Inventory frontend call sites, imports, and payload dependencies.
- Record the files planned for replacement/deletion.
- Do not use destructive git commands.

### Phase 1: write the minimal scientific core

- Implement the two neuron types and the one weight equation.
- Use the explicit shared E threshold and cap of `1000`; remove unnecessary fixed-point conversion machinery without renormalizing the scientific values.
- Add focused unit tests before integrating the dashboard.

### Phase 2: build topology and deterministic stepping

- Construct exact populations/edges.
- Implement the explicit phase order and WTA tie-break.
- Add topology and causal integration tests.

### Phase 3: reconnect the existing dashboard

- Preserve visual structure and styling.
- Add L1E_new positions and update minimal JS assumptions.
- Simplify protocol payload and config controls.
- Verify controls, streaming, selection, charts, and emitted edge flashes.

### Phase 4: remove obsolete code

- Search imports and frontend references.
- Delete legacy model/rule/config files and stale tests/fixtures.
- Remove stale parameters, endpoints, serializers, comments, and documentation.
- Ensure there is no second dormant implementation of neuron behavior.

### Phase 5: frequency experiment

- Establish deterministic zero-leak baseline.
- Measure full- versus half-frequency behavior.
- Measure actual `N_active`, active-weight sums, and inter-volley gaps from L1E_s output.
- Search jointly over the smallest useful leak range and one shared E weight cap.
- Check the generalized fast/slow inequality analytically, then validate it against the irregular measured spike recurrence and full network.
- Derive fixed projection-specific initialization from expected active fan-in; do not add runtime budgets or charge normalization.
- Promote no mechanism without turnover and recovery evidence.
- Keep production configuration minimal.

### Phase 6: verification and handoff

- Run the complete reduced test suite.
- Start the FastAPI app and exercise core REST endpoints.
- Open or otherwise smoke-test the dashboard if the environment permits.
- Inspect topology and several dynamic frames for NaN/inf, missing IDs, invalid edge endpoints, and incorrect threshold/cap values.
- Report deletions, final code map, final parameter list, test results, and the frequency experiment conclusion.

## Acceptance criteria

The work is complete only when all of the following are true:

- The backend has one obvious model path with no legacy behavior selectors.
- Excitatory neurons have separate nonnegative `acc_weights` and `subt_weights` plus explicit subtractive sign semantics.
- Only accumulating weights learn, using the requested distance/eta/p/signal/nonlinear equation.
- Subtractive gates are frozen, capped at target threshold, and real inhibitory events hard-wipe charge.
- Inhibitory neurons are stateless instant relays with no weight vectors and reported threshold `E/3`.
- The topology is exactly 9 L1E_s + 9 L1E_new + 9 L1I + 8 L2E + 1 L2I.
- L2 feedback targets L1E_new, never L1I directly.
- WTA has deterministic causal order and no artificial winner removal.
- Leak defaults to zero and remains a slider; refractory defaults to zero and remains configurable.
- Shared E thresholds/caps are invariant across E populations.
- The frontend retains its current styling and high-level structure and correctly shows the new topology.
- The configuration surface is small and rejects legacy keys.
- Obsolete tests, golden fixtures, modes, and dead class methods are deleted.
- Current documentation describes only code that exists.
- The reduced tests pass and the app starts.
- Frequency-tyranny claims are backed by deterministic measurements; failure is reported honestly rather than masked by a heuristic.
- The frequency analysis is expressed in terms of actual active-weight sums and `N_active`, with no hard-coded division by three.

## Final report format

Return a concise but evidence-rich handoff containing:

1. Summary of the implemented architecture.
2. Files added, substantially rewritten, and deleted.
3. Exact final configuration fields and defaults.
4. Exact neuron/population/edge counts.
5. The final timestep order and weight equation, including any documented decision on `p` or distance.
6. Tests and smoke checks run with results.
7. Frequency experiment setup, measurements, conclusion, and whether anything was promoted into production.
8. Any remaining limitation or unresolved scientific question.

Do not report success if the dashboard merely loads while the scientific invariants are untested, or if old backend machinery remains reachable.
