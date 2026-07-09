# Symbol Consolidation — Implementation Plan

> Superseded for current work: do not pursue `membrane_noise` as the symmetry
> breaker. The active plan is deterministic distance-weighted signal attenuation
> with uniform feedforward initialization and `membrane_noise=0.0`; see
> `Input_Vector_Initialization_And_Distance_Weighting.md` and `AGENT_HANDOFF.md`.
> The historical noise results below are preserved only as implementation record.

**Audience:** an implementing coding model (Nemotron Super). This plan is written to
be executed literally and in order. Do exactly what each step says. Do not
improvise, do not refactor unrelated code, and do not "improve" things that are
not listed. After every phase there is a VERIFY step — run it and confirm the
expected output before moving on. If a VERIFY step fails, STOP and report; do not
continue to the next phase.

---

## STATUS — implemented and validated (2026-07-06)

This plan has been fully implemented and measured. Summary of outcomes and the one
correction to the plan as originally written:

- **CORRECTION — `facil_decay` must be SLOW (~0.02), not 0.15/0.20.** The original
  plan defaulted `facil_decay` to 0.20 and swept 0.10/0.20. That is wrong: with a
  fast decay the post-fire boost has vanished before the next intrinsic cycle, so
  facilitation has *zero* effect (within-pattern dominance stays at the ~0.29
  rotation baseline). The validated default is `facil_decay = 0.02`; the engine and
  both neuron classes now default to it. `facil_decay >= 0.05` collapses A back to
  ~0.4 in the sweep.
- **Result.** Baseline (all features off) scores A = 0.30 (winner rotates), B = 5.4/8.
  With **`facil_boost=6.0, facil_decay=0.02`** alone: **A = 0.80, B = 6.2/8** (both
  acceptance bars met). Adding small `membrane_noise` (0.1–0.2) raises tiling to
  **B = 6.6–7.0** at a small cost to A. Best combined config in the sweep:
  `facil_boost=6.0, facil_decay=0.02, membrane_noise=0.10, homeo_down=0.05` →
  **A = 0.81, B = 6.6/8**.
- **All defaults remain OFF** (`facil_boost=0.0`, `membrane_noise=0.0`), so every
  existing test still passes unchanged. The consolidation behavior is opt-in via
  constructor parameters.
- Phases 4 and 5 are realized as `test_consolidation_metrics.py` (the A/B report)
  and `sweep_consolidation.py` (the ranked grid). Run them to reproduce the above.

The phase-by-phase instructions below are preserved as the implementation record.
Apply the `facil_decay = 0.02` correction wherever the text still says 0.20/0.15.

---

## 0. Background you need (read once, fully)

This repository is a from-scratch spiking neural network (LIF neurons, pure NumPy,
neuron-local learning, no gradients). The relevant network is:

- **L1E** — 9 input "pixel" neurons (a 3×3 grid), fixed weights, no learning.
- **L2E** — 8 output neurons. Each learns a receptive field over the 9 pixels.
- **L2I** — 1 shared inhibitory neuron that produces competition among the L2E pool.

The goal is **symbol consolidation**: after training on the 8 input patterns, we
want each pattern to be owned by a stable output neuron. There are **two separate
sub-problems**, and this plan addresses them in order:

- **Problem A — within-pattern stability.** For a fixed input pattern, ONE L2E
  neuron should win *repeatedly*. Today the winner *rotates* (a different L2E wins
  almost every cycle). Root cause: when an L2E fires it resets its membrane to 0,
  but its rivals keep ~96% of the charge they banked (`leak_l2 = 0.01`), so a rival
  always overtakes the just-reset winner next cycle.

- **Problem B — across-pattern assignment (tiling).** Different patterns should map
  to *different* L2E neurons. Today only ~3–6 of the 8 patterns get distinct
  winners.

**A is upstream of B**: while the winner rotates, the Hebbian reinforcement is
smeared across many neurons and cannot lock in an assignment. So we fix A first
(Phase 1), then strengthen the force that spreads winners across the pool (Phase 2),
then add a seed that decides *who* gets *what* (Phase 3), then measure both problems
together (Phase 4) and tune (Phase 5).

### Key files and where things live

- `neuron.py` — base `Neuron` class. Methods you will touch: `__init__`,
  `check_threshold`, `fire`, `update`. Add one new method `effective_potential`.
- `neuron_flexible.py` — `Neuron` class with the SAME dynamics but flexible fan-in.
  **This is the class the L2 neurons actually use.** Every change you make to
  `neuron.py` you must make IDENTICALLY here. A unit test asserts the two classes
  behave identically, so they must stay in lock-step.
- `backend/simulation.py` — `SimulationEngine`. Holds parameters in
  `self.params` (a dict), builds the network in `_build`, and runs one time step in
  `step`. The L2E neurons are configured in a big `for` loop inside `_build`.

### Guardrails (DO / DO NOT)

- **DO** keep every new feature OFF by default so existing tests keep passing. New
  numeric parameters default to `0.0` (disabled). The only exception is decay
  constants, which are harmless when the boost is 0.
- **DO** make identical edits to both `neuron.py` and `neuron_flexible.py`.
- **DO NOT** touch the inhibitory-plasticity rule (`apply_inhibition`), the
  excitatory weight rule (`_update_weights`), or the L2I wiring.
- **DO NOT** change `leak_l2`, thresholds, or any existing default value.
- **DO NOT** rename existing parameters or functions.
- After each phase, run the VERIFY command from the repo root
  (`/home/adasgup/Documents/SNN`) using the project's virtualenv Python:
  `PYTHONPATH=. .venv/bin/python <script>`.

### VERIFY baseline before starting

Run these three commands. All three must print their PASS lines.

```
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
PYTHONPATH=. .venv/bin/python test_8line_consolidation.py
```

Expected: `test_neuron.py` ends with `ALL NEURON UNIT TESTS PASSED`;
`test_l2_competition.py` ends with `ALL L2 COMPETITION TESTS PASSED`;
`test_8line_consolidation.py` ends with `ALL 8-LINE TESTS PASSED`.

If any fails, STOP — the tree is not clean and this plan assumes a clean start.

---

## Phase 1 — Winner facilitation (fixes Problem A)

**Idea:** give a neuron that just fired a small, decaying excitability boost, so it
can win the same pattern again next cycle instead of being overtaken by a rival that
banked charge. We add a scalar `self.facilitation` per neuron. It is added to the
membrane potential to form an "effective potential" used for BOTH threshold-crossing
and winner ranking. It jumps up when the neuron fires and decays every step. When
the boost amount is 0 (default), effective potential equals the plain potential and
nothing changes.

You will edit THREE files: `neuron.py`, `neuron_flexible.py`, `backend/simulation.py`.

### 1.1 — `neuron.py`

**(a) Add parameters to `__init__`.** FIND this exact block in the `__init__`
signature (around line 50–52):

```python
                 homeostasis=False, ca_rate=0.01, ca_target=0.02, ca_band=0.5,
                 homeo_up=0.01, homeo_down=0.01,
                 homeo_budget_min=None, homeo_budget_max=None):
```

REPLACE it with:

```python
                 homeostasis=False, ca_rate=0.01, ca_target=0.02, ca_band=0.5,
                 homeo_up=0.01, homeo_down=0.01,
                 homeo_budget_min=None, homeo_budget_max=None,
                 facil_boost=0.0, facil_decay=0.20, facil_max=None):
```

**(b) Add state in the `__init__` body.** FIND this line (it is near the end of
`__init__`, around line 172):

```python
        # Optional homeostatic weight budget (uniform per-neuron): if set, positive
```

INSERT the following block IMMEDIATELY BEFORE that line (keep that line right after):

```python
        # Winner facilitation: a decaying post-fire excitability boost. Added to
        # the membrane potential via effective_potential(), so a neuron that just
        # fired is transiently easier to fire again -- this cancels the reset
        # handicap that otherwise lets a charge-banking rival overtake the winner
        # every cycle (winner rotation). facil_boost = 0.0 (default) makes this
        # inert: facilitation stays 0, effective_potential() == potential, and all
        # behavior is identical to before.
        self.facil_boost = facil_boost                       # bump added on each fire
        self.facil_decay = facil_decay                       # per-step multiplicative decay
        self.facil_max = facil_max if facil_max is not None else threshold  # cap
        self.facilitation = 0.0                              # current boost (state)

```

**(c) Add the `effective_potential` method.** FIND the start of `check_threshold`
(around line 309):

```python
    def check_threshold(self):
```

INSERT this ENTIRE new method IMMEDIATELY BEFORE it:

```python
    def effective_potential(self):
        """Membrane potential plus the current post-fire facilitation boost.
        Threshold-crossing (check_threshold) and cross-neuron winner ranking both
        use this, so a recently-fired neuron gets a transient edge. With
        facil_boost == 0 the facilitation is always 0 and this equals
        self.potential exactly."""
        return self.potential + self.facilitation

```

**(d) Use effective potential in `check_threshold`.** FIND (around line 317):

```python
        if self.refractory_timer <= 0 and self.potential >= self.threshold:
```

REPLACE with:

```python
        if self.refractory_timer <= 0 and self.effective_potential() >= self.threshold:
```

**(e) Boost facilitation on fire.** FIND this block inside `fire()` (around line
333–337):

```python
        # Discharge: reset potential after firing (the "subtract charge" step).
        self.potential = self.resting_potential

        # Start refractory period
        self.refractory_timer = self.refractory_period
```

REPLACE with:

```python
        # Discharge: reset potential after firing (the "subtract charge" step).
        self.potential = self.resting_potential

        # Post-fire facilitation: raise this neuron's excitability so it can win
        # the same pattern again next cycle instead of being overtaken by rivals
        # that banked charge while it was reset. Capped at facil_max; decayed each
        # step in update().
        self.facilitation = min(self.facilitation + self.facil_boost, self.facil_max)

        # Start refractory period
        self.refractory_timer = self.refractory_period
```

**(f) Decay facilitation every step.** FIND this block at the very start of
`update()` (around line 283–286):

```python
        # Homeostatic firing-rate sensor: a slow EMA of the neuron's OWN spiking
        # this step (self.spiked is still set from fire() and is reset at the end
        # of this method). Purely local -- reads nothing but its own output.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)
```

REPLACE with:

```python
        # Homeostatic firing-rate sensor: a slow EMA of the neuron's OWN spiking
        # this step (self.spiked is still set from fire() and is reset at the end
        # of this method). Purely local -- reads nothing but its own output.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)

        # Decay the post-fire facilitation toward 0 every step, so the boost only
        # spans a few cycles (fast). Runs regardless of refractory state.
        self.facilitation *= (1.0 - self.facil_decay)
```

### 1.2 — `neuron_flexible.py` (identical changes)

Make the SAME six edits (a)–(f) in `neuron_flexible.py`. The anchor lines are the
same text, at these approximate locations:

- (a) `__init__` signature ends with the same `homeostasis=... homeo_budget_max=None):`
  block (around line 27–29). Add the same `facil_boost=0.0, facil_decay=0.20,
  facil_max=None):` line the same way.
- (b) Add the same facilitation state block. Put it IMMEDIATELY BEFORE the line
  that reads `# Optional homeostatic weight budget` (search for that comment). If
  that exact comment does not exist in this file, instead put the block at the very
  end of `__init__`, just before the method ends (before the next `def`).
- (c) Insert the same `effective_potential` method immediately before
  `def check_threshold(self):` (around line 252).
- (d) In `check_threshold`, replace the line
  `if self.refractory_timer <= 0 and self.potential >= self.threshold:` (around line
  262) with the `effective_potential()` version.
- (e) In `fire()`, FIND (around line 279–283):

  ```python
        # Discharge: reset potential after firing.
        self.potential = self.resting_potential

        # Start refractory period
        self.refractory_timer = self.refractory_period
  ```

  REPLACE with:

  ```python
        # Discharge: reset potential after firing.
        self.potential = self.resting_potential

        # Post-fire facilitation (see neuron.Neuron.fire for the rationale).
        self.facilitation = min(self.facilitation + self.facil_boost, self.facil_max)

        # Start refractory period
        self.refractory_timer = self.refractory_period
  ```

- (f) In `update()`, FIND (around line 228–229):

  ```python
        # Homeostatic firing-rate sensor: slow EMA of this neuron's own spiking.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)
  ```

  REPLACE with:

  ```python
        # Homeostatic firing-rate sensor: slow EMA of this neuron's own spiking.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)

        # Decay the post-fire facilitation toward 0 every step (see neuron.Neuron).
        self.facilitation *= (1.0 - self.facil_decay)
  ```

### 1.3 — `backend/simulation.py`

**(a) Add engine parameters.** FIND this block in `SimulationEngine.__init__`
(around line 256–259):

```python
                 refractory: int = 2,
                 volley_period: int = 4,
                 input_period: int | None = None,
                 cycle_period: int | None = None,
```

REPLACE with:

```python
                 refractory: int = 2,
                 volley_period: int = 4,
                 input_period: int | None = None,
                 cycle_period: int | None = None,
                 facil_boost: float = 0.0,
                 facil_decay: float = 0.20,
```

**(b) Store them in `self.params`.** FIND (around line 271–272):

```python
                           refractory=refractory, volley_period=volley_period,
                           input_period=input_period, cycle_period=cycle_period,
```

REPLACE with:

```python
                           refractory=refractory, volley_period=volley_period,
                           input_period=input_period, cycle_period=cycle_period,
                           facil_boost=facil_boost, facil_decay=facil_decay,
```

**(c) Configure the L2E neurons.** In `_build`, there is a `for` loop over
`self.neurons` that configures L2E neurons inside `if self.meta[nid]['type'] == 'E'
and nid.startswith('L2'):`. FIND this line inside that branch (around line 371):

```python
                n.inhibitory_learning_rate = L2_GATE_ETA
```

INSERT immediately AFTER it:

```python
                # Winner facilitation (see neuron.Neuron.effective_potential): a
                # decaying post-fire excitability boost that stops the just-fired
                # winner from being overtaken by charge-banking rivals next cycle.
                n.facil_boost = p['facil_boost']
                n.facil_decay = p['facil_decay']
                n.facil_max = thr_l2
```

**(d) Rank the competition by effective potential.** In `step`, FIND (around line
574, inside the `if cycle_boundary:` competition block):

```python
                winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
```

REPLACE with:

```python
                winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].effective_potential())
```

### 1.4 — VERIFY Phase 1

First, backward-compatibility (facilitation off by default → nothing changes):

```
PYTHONPATH=. .venv/bin/python test_neuron.py
PYTHONPATH=. .venv/bin/python test_l2_competition.py
```

Both must still end with their `ALL ... PASSED` lines. If not, you made a
non-identical edit between the two neuron files, or changed a default — fix it.

Second, a smoke test that facilitation raises within-pattern dominance. Create the
file `scratch_facil_check.py` at the repo root with EXACTLY this content:

```python
import numpy as np
from collections import Counter
from backend.simulation import SimulationEngine, PATTERNS, N_OUT

def dominance(facil_boost):
    e = SimulationEngine(seed=1, facil_boost=facil_boost, facil_decay=0.15)
    for ep in range(30):
        for name in PATTERNS:
            e.set_pattern(name)
            for _ in range(25):
                e.step()
    doms = []
    for name in PATTERNS:
        e.set_pattern(name)
        seq = []
        for _ in range(60):
            e.step()
            fired = [j for j in range(N_OUT) if e.spiked[f'L2E{j}']]
            if fired:
                seq.append(fired[0])
        if seq:
            c = Counter(seq)
            doms.append(c.most_common(1)[0][1] / len(seq))
    return float(np.mean(doms)) if doms else 0.0

for fb in (0.0, 2.0, 4.0, 6.0):
    print(f"facil_boost={fb}: within-pattern dominance = {dominance(fb):.3f}")
```

Run it:

```
PYTHONPATH=. .venv/bin/python scratch_facil_check.py
```

**Expected:** `facil_boost=0.0` gives dominance around `0.2–0.35` (the rotation
baseline). At least one of `facil_boost=2/4/6` should give a CLEARLY higher number
(target `> 0.5`, ideally `> 0.7`). If dominance does NOT rise with facil_boost, STOP
and report — the boost is not reaching the ranking. (Most likely cause: edit (d) in
1.3 or edit (d)/(c) in the neuron files was missed.)

Delete `scratch_facil_check.py` after confirming.

---

## Phase 2 — Homeostatic anti-tyranny (helps Problem B)

**Idea:** with facilitation on, the new risk is the opposite failure — one neuron
wins *everything* (collapse). The counter-force already exists in the code:
homeostatic synaptic scaling (`neuron.Neuron._homeostatic_scaling`) shrinks the
excitatory resource of a neuron that fires too much. It is already enabled for L2E
(`homeostasis=True` by default) but tuned weak. **No new code is required in this
phase** — it is a tuning phase, exercised through existing engine parameters
`homeo_down` (how fast an over-active neuron shrinks) and `ca_target` (the firing
set-point above which shrinking kicks in).

There is nothing to edit here. The joint sweep in Phase 5 will vary `homeo_down` and
`ca_target`. Just confirm these parameters are already plumbed:

### 2.1 — VERIFY the levers exist

Run:

```
PYTHONPATH=. .venv/bin/python -c "from backend.simulation import SimulationEngine; e=SimulationEngine(seed=1, homeo_down=0.05, ca_target=0.010); print('homeo_down', e.params['homeo_down'], 'ca_target', e.params['ca_target'])"
```

**Expected:** prints `homeo_down 0.05 ca_target 0.01`. If it errors, STOP — the
parameters are not wired (they should already be; do not add them yourself, report
instead).

---

## Phase 3 — Symmetry-breaking membrane noise (seeds Problem B)

**Idea:** even with A stable and B bounded, something must decide *which* neuron owns
*which* pattern. A tiny amount of per-step, zero-mean membrane noise on the L2E pool
makes near-ties resolve, and the Hebbian rule then amplifies whichever neuron got the
early edge on each pattern. Zero-mean noise (unlike a fixed bias) does not favor any
one neuron overall, so it breaks symmetry without causing collapse.

You will edit `backend/simulation.py` only.

**(a) Add the parameter.** FIND (the block you edited in 1.3a, now including facil):

```python
                 facil_boost: float = 0.0,
                 facil_decay: float = 0.20,
```

REPLACE with:

```python
                 facil_boost: float = 0.0,
                 facil_decay: float = 0.20,
                 membrane_noise: float = 0.0,
```

**(b) Store it in `self.params`.** FIND:

```python
                           facil_boost=facil_boost, facil_decay=facil_decay,
```

REPLACE with:

```python
                           facil_boost=facil_boost, facil_decay=facil_decay,
                           membrane_noise=membrane_noise,
```

**(c) Create a dedicated noise RNG in `_build`.** In `_build`, FIND (near the top,
around line 281):

```python
        rng = np.random.default_rng(p['seed'])
```

INSERT immediately AFTER it:

```python
        # Separate RNG for symmetry-breaking L2E membrane noise (see step()), so
        # enabling noise does not perturb the weight-init RNG stream above.
        self._noise_rng = np.random.default_rng(p['seed'] + 12345)
```

**(d) Inject noise before competition in `step`.** In `step`, FIND this line
(around line 550, right after L1E spikes are delivered to L2E):

```python
        # Capture pre-WTA potential for the charge visualisation.
        self.l2_drive = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}
```

INSERT immediately AFTER it:

```python
        # Symmetry-breaking membrane noise: a small zero-mean perturbation of each
        # non-refractory L2E potential every step, so near-ties resolve and the
        # Hebbian rule can amplify a consistent first-mover per pattern. 0.0
        # (default) disables it and leaves dynamics deterministic.
        sigma = self.params['membrane_noise']
        if sigma > 0.0:
            for e in l2.excitatory_neurons:
                if e.refractory_timer <= 0:
                    e.potential += float(self._noise_rng.normal(0.0, sigma))
```

### 3.1 — VERIFY Phase 3

Backward compatibility (noise off by default):

```
PYTHONPATH=. .venv/bin/python test_l2_competition.py
```

Must still end with `ALL L2 COMPETITION TESTS PASSED`. Then confirm noise runs
without error:

```
PYTHONPATH=. .venv/bin/python -c "from backend.simulation import SimulationEngine; e=SimulationEngine(seed=1, membrane_noise=0.3); [e.step() for _ in range(50)]; print('noise ok')"
```

**Expected:** prints `noise ok`.

---

## Phase 4 — Joint measurement harness

Create a NEW file `test_consolidation_metrics.py` at the repo root with EXACTLY this
content. It reports BOTH sub-problems: within-pattern dominance (A) and the number of
distinct winners across the 8 patterns (B). It is a measurement/reporting script, not
a pass/fail regression.

```python
"""
Joint metrics for symbol consolidation.

Reports, for a given parameter set, both quantities that matter:

  A. within-pattern dominance -- averaged over the 8 patterns, the spike fraction
     captured by that pattern's single most frequent winner. 1.0 = one stable
     winner per pattern (Problem A solved); ~0.2 = full rotation.
  B. distinct winners / 8 -- how many different neurons are the dominant winner
     across the 8 patterns. 8 = perfect tiling (Problem B solved); 1 = collapse.

Run:
    PYTHONPATH=. .venv/bin/python test_consolidation_metrics.py
"""
import numpy as np
from collections import Counter
from backend.simulation import SimulationEngine, PATTERNS, N_OUT


def train_and_measure(seed=1, epochs=30, **engine_kwargs):
    e = SimulationEngine(seed=seed, **engine_kwargs)
    for _ in range(epochs):
        for name in PATTERNS:
            e.set_pattern(name)
            for _ in range(25):
                e.step()
    doms, dominant_winner = [], {}
    for name in PATTERNS:
        e.set_pattern(name)
        seq = []
        for _ in range(60):
            e.step()
            fired = [j for j in range(N_OUT) if e.spiked[f'L2E{j}']]
            if fired:
                seq.append(fired[0])
        if seq:
            c = Counter(seq)
            top, cnt = c.most_common(1)[0]
            doms.append(cnt / len(seq))
            dominant_winner[name] = top
    within = float(np.mean(doms)) if doms else 0.0
    distinct = len(set(dominant_winner.values()))
    return within, distinct, dominant_winner


def report(label, **kwargs):
    withins, distincts = [], []
    for seed in (1, 2, 3):
        w, d, _ = train_and_measure(seed=seed, **kwargs)
        withins.append(w)
        distincts.append(d)
    print(f"{label:38s} | within-pattern dominance (A) = {np.mean(withins):.3f}"
          f" | distinct winners/8 (B) = {np.mean(distincts):.2f}")


if __name__ == "__main__":
    print("=== Symbol consolidation metrics (mean over seeds 1,2,3) ===")
    report("baseline (all off)")
    report("facilitation only", facil_boost=4.0, facil_decay=0.15)
    report("facil + noise", facil_boost=4.0, facil_decay=0.15, membrane_noise=0.3)
    report("facil + noise + strong homeo",
           facil_boost=4.0, facil_decay=0.15, membrane_noise=0.3,
           homeo_down=0.05, ca_target=0.010)
```

### 4.1 — VERIFY Phase 4

```
PYTHONPATH=. .venv/bin/python test_consolidation_metrics.py
```

**Expected:** four lines print without error. The `baseline (all off)` line should
show within-pattern dominance around `0.2–0.35` and distinct winners around `3–6`.
The other three lines are the experimental conditions; record their numbers for
Phase 5. Do not assume any particular winning row yet — Phase 5 decides.

---

## Phase 5 — Tuning sweep and acceptance

**Idea:** find the parameter set that maximizes BOTH metrics at once. This is the
research payoff. Create a NEW file `sweep_consolidation.py` at the repo root with
EXACTLY this content:

```python
"""
Grid sweep over the three consolidation levers, scoring each combo by BOTH
within-pattern dominance (A) and distinct winners (B). Prints a ranked table.

Run:
    PYTHONPATH=. .venv/bin/python sweep_consolidation.py
"""
import itertools
import numpy as np
from test_consolidation_metrics import train_and_measure

FACIL_BOOST   = [2.0, 4.0, 6.0]
FACIL_DECAY   = [0.10, 0.20]
MEMBRANE_NOISE = [0.0, 0.3]
HOMEO_DOWN    = [0.01, 0.05]

rows = []
for fb, fd, mn, hd in itertools.product(FACIL_BOOST, FACIL_DECAY, MEMBRANE_NOISE, HOMEO_DOWN):
    ws, ds = [], []
    for seed in (1, 2, 3):
        w, d, _ = train_and_measure(
            seed=seed, facil_boost=fb, facil_decay=fd,
            membrane_noise=mn, homeo_down=hd)
        ws.append(w); ds.append(d)
    within = float(np.mean(ws))
    distinct = float(np.mean(ds))
    # Combined score: reward both a stable winner (A) and broad tiling (B).
    # Normalize distinct to [0,1] by dividing by 8, then take the product so a
    # combo must be good at BOTH to score well.
    score = within * (distinct / 8.0)
    rows.append((score, within, distinct, fb, fd, mn, hd))

rows.sort(reverse=True)
print(f"{'score':>6} {'A_dom':>6} {'B_dist':>6} | {'fb':>4} {'fd':>5} {'noise':>6} {'homeo':>6}")
for score, within, distinct, fb, fd, mn, hd in rows:
    print(f"{score:6.3f} {within:6.3f} {distinct:6.2f} | {fb:4.1f} {fd:5.2f} {mn:6.2f} {hd:6.2f}")
```

### 5.1 — Run and record

```
PYTHONPATH=. .venv/bin/python sweep_consolidation.py
```

Record the FULL printed table and the top row.

### 5.2 — Acceptance criteria (report these explicitly)

The goal state is a parameter combination where **both** hold, averaged over the 3
seeds:

- **A — within-pattern dominance ≥ 0.7** (winner is stable per pattern), AND
- **B — distinct winners ≥ 6 / 8** (patterns tile across the pool).

Report the best combination and whether it meets both bars. It is acceptable and
expected that **no combination reaches a perfect 8/8** — one-to-one assignment is a
hard symmetry-breaking problem and classic competitive learning only solves it
probabilistically. If A rises but B collapses (distinct winners drops toward 1) as
facil_boost grows, that is the collapse failure mode; note it, and report which
`homeo_down` / `ca_target` value best counteracts it.

### 5.3 — Do NOT hard-code the winner

Do not bake the swept "best" parameters into `SimulationEngine`'s defaults. Leave all
new parameters defaulting to OFF. Report the best combination in your final summary
and let a human decide whether to adopt it as the new default.

---

## Final report to produce

When done, write a short summary containing:

1. Confirmation that all three original tests still pass with defaults.
2. The Phase 1 smoke-test numbers (dominance vs facil_boost).
3. The Phase 4 four-condition table.
4. The Phase 5 top rows and whether the acceptance bars (A ≥ 0.7 and B ≥ 6/8) were
   met, at which parameters.
5. Any failure modes observed (especially collapse when facilitation is too strong
   or homeostasis too weak).

## Rollback

Every change is gated behind a parameter that defaults to OFF, so nothing here alters
default behavior. If a phase misbehaves, revert that phase's edits with
`git checkout -- <file>` and re-run the Phase 0 VERIFY commands to confirm a clean
baseline before retrying.
```
