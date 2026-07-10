# SNN Refactor Plan: Object Decomposition Without Behavior Change

## Purpose

`neuron_flexible.py` has grown into a ~1000-line God object: a single `Neuron`
class carries membrane dynamics, a synapse bank (parallel numpy arrays), and a
dozen mutually-exclusive learning experiments, each gated by its own boolean flag
and cluster of tuning constants (`__init__` alone is ~180 lines / ~60 attributes).

This plan reorganizes that code toward the cleaner object decomposition seen in
the `cipp-learning` (Paul) paradigm — explicit `Synapse` / `Neuron` / `Network`
types over a shared `NeuralEntity` contract — **while keeping every existing
behavior, default, and numerical result identical.**

### Hard non-negotiables

1. **No behavior change.** Every existing test (`test_*.py`), harness
   (`stage_learning_harness.py`, `ablation_harness.py`), the dashboard defaults,
   and serialization (`get_state`) must produce **byte-identical** results with
   all flags at their current defaults.
2. **Vectorization preserved.** We do **not** adopt Paul's per-synapse Python
   objects. The SNN's numerics depend on `np.dot` over weight arrays and on
   fixed-point `UNIT`/`LEAK_SCALE` scaling; a Python-object-per-synapse loop
   would change both performance and floating-point results. The synapse concept
   becomes an object that **owns the vectorized arrays**, not an object per edge.
3. **Every flag is preserved**, just relocated. `signed_spike_learning`,
   `confidence_consolidation`, `assembly_flow_credit`, `structural_free_energy`,
   `inhibitory_delta_rule`, `homeostasis`, `subtractive_reset`,
   `excitatory_flow_rate`, `inhibitory_flow_rate`, `distance_weighting`,
   `signed_depression`, `loser_depression`, `v_sat`, budget/cap, etc.

## Guiding Principle: Characterization-First, Strangler-Fig Migration

Refactor behind a frozen behavioral contract. Nothing is deleted until an
equivalent object reproduces it exactly. Each phase is independently shippable
and revertible; the public API of `Neuron` (`receive_input`, `apply_inhibition`,
`update`, `check_threshold`, `fire`, `get_state`, the property accessors) stays
stable throughout so callers (`backend/simulation.py`,
`cortical_column_flexible.py`, tests) are untouched until the very end.

---

## Phase 0 — Freeze the contract (no production code changes)

**Goal:** make "identical behavior" a machine-checkable claim before touching
anything.

1. **Golden-output harness.** Add `tests/golden/gen_golden.py` that constructs a
   representative matrix of neurons/engines and dumps full state trajectories
   (potential, weights, confidence, trace, inhibitory events) to `.npz`:
   - a bare fixed-fan-in neuron;
   - a staged-wiring neuron;
   - each learning mode ON in isolation: `signed_spike_learning`,
     `confidence_consolidation` (+`loser_depression`, `signed_depression`),
     `assembly_flow_credit`, `structural_free_energy`;
   - each inhibitory mode: saturating, `inhibitory_delta_rule` turnover, margin;
   - each delivery mode: instantaneous, `excitatory_flow_rate`,
     `inhibitory_flow_rate`, `distance_weighting`;
   - `subtractive_reset`, `v_sat`, `homeostasis`, budget/floor;
   - one full `SimulationEngine` run at current dashboard defaults, seeded.
2. **Equivalence test** `tests/golden/test_golden_equiv.py`: regenerate live and
   assert bit-exact (`np.array_equal`) against the committed `.npz`. This is the
   gate every later phase must pass.
3. Pin seeds and any `time.time()`-derived nondeterminism (there is none in the
   fixed-point path today — confirm and document it).

**Exit criteria:** golden suite green; committed as the baseline.

---

## Phase 1 — Introduce the type hierarchy (structure only, no logic moved)

Create `snn/` package skeleton mirroring Paul's decomposition, but leave the real
logic in place and delegate to it.

```
snn/
  entity.py        # NeuralEntity(ABC): id, abstract update(dt) — Paul's contract
  membrane.py      # Membrane: potential, threshold, refractory, leak, v_sat, reset
  synapses.py      # SynapseBank: the vectorized weight arrays + trace + confidence
  neuron.py        # Neuron: composes Membrane + SynapseBank + a LearningPolicy
  network.py       # thin adapter over the existing SimulationEngine
  rules/           # strategy objects (Phase 3)
```

- `NeuralEntity(ABC)` — direct analogue of Paul's base: `entity_id` + abstract
  `update(dt)`. `Neuron`, and later layer/network objects, implement it.
- `SynapseBank` — **owns** `_weights_array`, `_trace`, `_confidence`, `_distance`,
  and the construction path (`add_input_connection`, `finalize_connections`,
  `_set_weights`, `_ensure_finalized`). This is the "Synapse as a real object"
  win from Paul's design, made vectorized: it is one object holding N afferents,
  exposing `dot(spikes)`, `positive_mask()`, `negative_indices()`, `clip()`.
- `Membrane` — owns `potential`, `resting_potential`, `threshold`,
  `refractory_timer`, leak (`_leak_num`/`LEAK_SCALE`), `v_sat`, and the two reset
  policies (`fire`-time full vs subtractive). Pure mechanics, no learning.

In this phase the existing `Neuron` in `neuron_flexible.py` is refactored to
**hold** a `Membrane` and a `SynapseBank` and forward to them, moving state but
not rewriting any formula. Property accessors (`weights`, `trace`, `confidence`,
`distance`, `n_inputs`, `leak_rate`) proxy to the sub-objects so external callers
and `get_state()` see no change.

**Exit criteria:** golden suite still bit-exact. Line count of the monolith drops
but no equation moved yet.

---

## Phase 2 — Extract delivery & membrane update (still one code path each)

Move the numeric bodies, unchanged, onto the new objects:

- `receive_input` splits into `SynapseBank.effective_weights(distance)` +
  `Membrane.deposit(current, v_sat)`, with the flow-rate branch becoming a
  `DeliveryMode` (see Phase 3). Keep both branches literally as-is first; only
  relocate.
- `update`'s leak / refractory / inh-flow-drain / ca-EMA / homeostasis block
  moves onto `Membrane.step()` + policy hooks. `ca` and `_decay_confidence`
  stay wired but are invoked through the neuron's policy list.
- `fire` becomes: capture `v_pre`, `Membrane.reset()` (policy), refractory,
  `exc_trace` discharge, then `learning_policy.on_postsynaptic_spike(v_pre)`.
- `check_threshold` moves to `Membrane`.

Each move is a pure cut/paste guarded by the golden suite. No `if`-branch is
collapsed yet.

**Exit criteria:** golden suite bit-exact; `receive_input`/`update`/`fire` on
`Neuron` are now thin orchestrators.

---

## Phase 3 — Flags become Strategy objects (the core cleanup)

Replace the boolean-flag + inline-branch stack with swappable policies. Each flag
maps to a concrete strategy; **the default configuration must instantiate the
exact strategies that reproduce today's defaults.** Nothing is removed — the
`if self.signed_spike_learning:` ladder in `_update_weights` becomes polymorphic
dispatch over one selected `ExcitatoryRule`.

### 3a. Excitatory learning rules (`rules/excitatory.py`)

Interface: `ExcitatoryRule.on_fire(bank, membrane, v_pre) -> None`.

| Current flag / branch (`_update_weights`) | Strategy |
| --- | --- |
| baseline charge-based `dw = eta*p*(1 - w^2/w_max)` | `ChargeBasedPotentiation` (default) |
| `confidence_consolidation` potentiation + confidence EMA | `ConfidenceConsolidation` (wraps ChargeBased) |
| `signed_spike_learning` (+early return) | `SignedSpikeRule` |
| `structural_free_energy` gate on the signed rule | `StructuralFreeEnergyGate` (decorator on SignedSpikeRule; swaps `p` for the maturity brake) |
| `assembly_flow_credit` (+early return) | `AssemblyFlowCredit` |
| `signed_depression` OFF-gate depression | `SignedDepression` (post-step decorator) |
| `_apply_budget_and_cap`, `min_positive_weight` | `BudgetAndCap` (post-step decorator, always applied) |

The current mutual exclusivity (signed/assembly return early; baseline+depression
+budget compose) is encoded explicitly as a **composed pipeline** rather than
implicit control flow, so the "which rules can coexist" question becomes readable
in the constructor instead of buried in early returns.

### 3b. Inhibitory learning rules (`rules/inhibitory.py`)

Interface: `InhibitoryRule.on_discharge(bank, membrane, idx, v_pre) -> event`.

| Current branch (`apply_inhibition`) | Strategy |
| --- | --- |
| saturating `dw = eta*p*(1 - w^2/w_max)` | `SaturatingInhibition` (neuron default) |
| `inhibitory_delta_rule` + mode `"turnover"` | `DeltaTurnover` (engine default) |
| `inhibitory_delta_rule` + mode `"margin"` | `DeltaMargin` |

`_depress_losers` (triggered by `loser_depression` when a discharge occurred)
becomes `LoserDepression`, a hook fired by `apply_inhibition`'s orchestrator, not
an inline call.

### 3c. Delivery & reset modes

- `DeliveryMode`: `Instantaneous` (default) vs `FlowRate` (exc/inh traces).
  `DistanceAttenuation` is a decorator on either.
- `ResetPolicy`: `FullReset` (default) vs `SubtractiveReset`.
- `Homeostasis`: `HomeostaticScaling` (opt-in) vs `NullHomeostasis` (default).

### 3d. Config → policy assembly

Add `snn/config.py` with a `NeuronConfig` dataclass that carries exactly the
current keyword arguments, plus a `build_policies(config) -> LearningPolicy`
factory that maps flags to the strategy instances above. **`Neuron.__init__`
keeps its current signature** and internally calls `build_policies`, so every
caller and the dashboard's live-config path are unchanged. The 180-line `__init__`
collapses to: build `Membrane`, build `SynapseBank`, `build_policies(config)`.

**Exit criteria:** golden suite bit-exact with default policy assembly; each mode
test (`test_flow_rate.py`, `test_inhibitory_delta_rule.py`,
`test_structural_free_energy.py`, `test_l2_chunked_charge.py`, etc.) green
against its corresponding strategy.

---

## Phase 4 — Network / layer layer

`backend/simulation.py`'s `SimulationEngine` already plays the `NeuralNetwork`
role. Give it the `NeuralEntity` contract and formalize the spike-routing it does
(L1E→L2E, L2E→L2I, L2I→L2E gate, L2E→L1I) as explicit `Projection` objects,
mirroring Paul's `Synapse`/`handle_spike` routing but vectorized per projection.
This is optional polish — do it only if Phases 1–3 have not already made the
engine readable. Keep `get_state`/serialization output identical.

---

## Phase 5 — Retire the monolith & document

1. `neuron_flexible.py` becomes a thin **compatibility shim**: `from snn.neuron
   import Neuron` (+ `UNIT`, `LEAK_SCALE`) so no import path breaks.
2. Move the extensive rationale comments (currently inline in `__init__`) into
   each strategy's docstring — the "why" travels with the code it explains.
3. Update `README.md` Code Map and `AGENT_HANDOFF.md` to point at `snn/`.
4. Delete only strategies proven dead by the golden matrix + `ablation_harness.py`
   — and only after confirming no harness/doc references them. (Candidates to
   *evaluate*, not presume dead: none removed without evidence.)

---

## Risk register

| Risk | Mitigation |
| --- | --- |
| Silent numeric drift from reordering ops | Phase 0 golden suite is bit-exact (`np.array_equal`), run as the gate on every phase |
| Fixed-point `UNIT`/`LEAK_SCALE` semantics lost in a move | `Membrane` owns the integer leak numerator; `test_fixed_point_scale_invariance` stays in the golden set |
| Strategy indirection slows the hot loop | Strategies operate on whole arrays (one call per neuron per event), not per-synapse — same `np.dot`/vectorized ops as today |
| Mutual-exclusivity of learning modes broken | Pipeline composition in `build_policies` encodes the current early-return semantics explicitly; a test asserts illegal combinations raise |
| Serialization/`get_state` shape changes | `get_state` proxied through sub-objects in Phase 1; golden `.npz` includes full state |
| Live dashboard config path breaks | `Neuron.__init__` keyword signature frozen; `build_policies` is internal |

## Definition of done

- All `test_*.py`, both harnesses, and the golden suite pass bit-exact.
- `Neuron.__init__` is < 30 lines; no learning `if flag:` ladders remain in
  `receive_input`/`apply_inhibition`/`_update_weights`.
- Every prior flag is reachable via `NeuronConfig` and produces identical output.
- `neuron_flexible.py` imports from `snn/` with zero behavior change for callers.
