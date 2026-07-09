# Path Forward - Clean Symmetry Ablation Plan

> Superseded for current work: do not run new membrane-noise sweeps from this
> plan. The active plan is deterministic distance-weighted signal attenuation
> with uniform feedforward initialization and `membrane_noise=0.0`; see
> `Input_Vector_Initialization_And_Distance_Weighting.md` and `AGENT_HANDOFF.md`.

**Audience:** Claude / implementation agent.

This plan continues from the previous symmetry-breaking work. It explicitly
rejects post-fire winner facilitation as an architectural solution, expands the
homeostasis ablation, and retests inhibitory excitability without scaling away
the effect.

## Why We Are Doing This

The previous work produced useful findings:

- Winner facilitation can raise within-pattern dominance, but it adds persistent
  post-fire winner inertia. That strays from the CIPP principle that spikes
  should reflect integrated synaptic evidence, not an extra hidden winner memory.
- Down-only homeostasis (`homeo_up=0.0`) improved receptive-field diversity and
  within-pattern dominance without facilitation, but it did not robustly solve
  one-to-one tiling or preserve reserve capacity.
- Lowering inhibitory thresholds was inert because threshold, E-to-I init scale,
  and E-to-I cap were all scaled together. That was safe, but it normalized away
  the biological hypothesis.
- Held-out recruitment failed: the pool used almost all L2E neurons even when
  trained on only 6 patterns, so new patterns remapped existing winners rather
  than occupying spare neurons.

The next goal is not to maximize a single metric. The goal is to understand the
mapping dynamics: who owns what, what remains unused, and which local mechanisms
actually produce stable, diverse receptive fields.

## Token-Efficient Workflow

Do not reread the full repo. Use:

```bash
git status --short --branch
git diff --stat
rg -n "facil_|effective_potential|homeostasis|homeo_|l2i_threshold_frac|l1i_threshold_frac|ETA_FRAC|L2_EI_WEIGHT|weight_cap|excitatory_saturation_cap|minimum|contributor|apply_inhibition" neuron.py neuron_flexible.py backend/simulation.py test_symmetry_breaking_metrics.py sweep_symmetry_breaking*.py
```

Then inspect only changed hunks:

```bash
git diff -- neuron.py neuron_flexible.py backend/simulation.py
sed -n '<small targeted range>' backend/simulation.py
```

## Phase 0 - Revert Winner Facilitation

Remove post-fire facilitation from the core neuron model and simulation path.

Revert these concepts from `neuron.py` and `neuron_flexible.py`:

- `facil_boost`, `facil_decay`, `facil_max`
- `self.facilitation`
- `effective_potential()`
- facilitation decay in `update()`
- facilitation bump in `fire()`
- threshold checks using `effective_potential()`

Restore threshold checks to:

```python
self.refractory_timer <= 0 and self.potential >= self.threshold
```

Remove from `backend/simulation.py`:

- `facil_boost`
- `facil_decay`
- assignment of facilitation fields to L2E neurons
- winner ranking by `effective_potential()`

Restore winner ranking to raw membrane potential:

```python
winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
```

Delete or archive facilitation-only scripts if they are not needed. If retained,
rename/comment them as rejected diagnostic experiments, not active architecture.

Why: facilitation improved metrics by adding persistent post-fire inertia. We want
the next experiments to explain symmetry breaking through inhibition,
homeostasis, noise, and local synaptic dynamics, not a hidden winner boost.

Verify:

```bash
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_refractory_gating.py
```

## Phase 1 - Expand Homeostasis Ablation

Create or update a sweep script, e.g. `sweep_homeostasis_mapping.py`.

Keep facilitation absent. Sweep homeostasis as a mapping mechanism, not just as
an A/B score helper.

Conditions:

```text
homeostasis=False

down-only:
  homeo_up=0.0
  homeo_down=0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12

weak recruitment:
  homeo_up=0.001, 0.003, 0.005
  homeo_down=0.03, 0.05, 0.08

current-style balanced:
  homeo_up=0.01
  homeo_down=0.01, 0.03, 0.05

anti-recruitment control:
  homeo_up=0.0
  homeo_down=0.0
```

For each condition, report over seeds 1-5:

- A: within-pattern dominance
- B: distinct dominant winners out of 8
- per-pattern winner map for each seed
- winner preservation across late epochs
- RF cosine diversity
- RF HHI/entropy
- L2E firing counts
- number of never-fired L2E neurons
- homeo budgets per L2E
- L2I spike count and L2I-to-L2E gate event count

Acceptance is not required here. This is an ablation map.

## Phase 2 - Held-Out Capacity Mapping

Extend the held-out experiment.

Protocols:

```text
Train on 4 patterns, hold out 4.
Train on 6 patterns, hold out 2.
Train on 7 patterns, hold out 1.
```

For each homeostasis condition from Phase 1 top candidates:

1. Train only on the initial pattern subset.
2. Record dominant winners and all L2E units that ever fired.
3. Introduce held-out patterns.
4. Measure whether held-out patterns use previously unused/weakly-used neurons.
5. Measure whether original winners are preserved.

Report:

- reserve count before held-out introduction
- held-out taken by reserve fraction
- original winner preservation fraction
- remapping events
- final B across all patterns
- final RF diversity

Why: a system that uses all neurons for early patterns may score well on a fixed
8-pattern benchmark but fail continual concept learning.

## Phase 3 - Retest Inhibitory Excitability Without Scaling It Away

The previous L2I threshold sweep was a no-op because all related E-to-I scales
moved together. Split the inhibitory pathway into four independent controls:

1. **I firing threshold**
2. **E-to-I initial weight scale**
3. **E-to-I learning cap**
4. **desired minimum contributor count for L2I firing**

Add parameters, defaulting to current behavior:

```python
l2i_threshold_frac = 1.0
l2i_ei_init_frac_scale = 1.0
l2i_ei_cap_frac = 1.0
l2i_min_contributors_target = None
```

Interpretation:

- `l2i_threshold_frac`: L2I threshold relative to L2E threshold.
- `l2i_ei_init_frac_scale`: multiplies the existing `[0.25, 0.5] * threshold`
  E-to-I init range.
- `l2i_ei_cap_frac`: E-to-I learning cap relative to L2I threshold.
- `l2i_min_contributors_target`: reporting/design target only unless you can
  implement it as a parameter-derived constraint without adding procedural logic.

Do not implement procedural contributor counting in the simulation. Use it to
derive safe init/cap regimes and to report whether the observed L2I events meet
the target.

Initial sweep:

```text
l2i_threshold_frac: 1.0, 2/3, 1/2, 1/3
l2i_ei_init_frac_scale: 0.5, 0.75, 1.0
l2i_ei_cap_frac: 0.5, 0.75, 1.0
homeostasis: top 2-3 settings from Phase 1
membrane_noise: 0.0, 0.05, 0.1
```

For each condition, instrument L2I events and report:

- average distinct contributors per L2I spike
- minimum distinct contributors observed
- fraction of L2I spikes caused by one contributor
- L2I spike latency distribution
- L2I spike count
- A/B/RF metrics

Reject conditions where L2I becomes an instant one-spike relay unless the design
explicitly calls for that as a control.

Why: biologically lower I thresholds should change inhibitory excitability, but
we must separately manage synaptic scale so inhibition neither becomes inert nor
degenerates into immediate relay.

## Phase 4 - Learning-Rate Ablation Around Useful Regimes

Only after Phases 1-3 identify promising homeostasis/inhibition regimes, sweep
learning rates locally.

```text
l2e_lr_frac: 0.005, 0.01, 0.015, 0.02, 0.03
l2i_lr_frac: 0.001, 0.0025, 0.005, 0.01
l1i_lr_frac: keep default first
l2_gate_eta: 0.03, 0.05, 0.10, 0.15
```

Report the same mapping metrics. Do not run a giant full-factorial grid across
all parameters; use top candidates from earlier phases.

## Phase 5 - Update Documentation

Update the plan/status docs with:

- facilitation reverted and rejected as core architecture
- homeostasis ablation results
- held-out capacity results
- separated inhibitory-excitability results
- whether any condition improves mapping without consuming reserve capacity

Do not claim symbol consolidation is solved unless:

- A >= 0.7 without facilitation
- B >= 6/8 over seeds
- RF diversity improves
- held-out introduction does not catastrophically remap old winners
- inhibitory plasticity remains refractory-gated and local

## Final Report Format

Report:

1. Confirmation facilitation was removed/reverted.
2. Test pass/fail summary.
3. Homeostasis ablation table.
4. Held-out capacity table.
5. Inhibitory excitability table, including contributor counts.
6. Best candidate regimes and why they are or are not architecturally acceptable.
