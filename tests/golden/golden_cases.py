"""Golden behavioral contract for the SNN refactor (REFACTOR_PLAN.md, Phase 0).

`collect()` deterministically constructs a representative matrix of neurons and
engines, drives each through a fixed input schedule, and returns a flat dict of
numpy arrays capturing full state trajectories (potential, weights, trace,
confidence, refractory, spikes, inhibitory-gate magnitudes).

Both `gen_golden.py` (writes the baseline `.npz`) and `test_golden_equiv.py`
(regenerates live and asserts bit-exact) import THIS module, so the ONLY thing
that can move a number between baseline and re-run is the underlying
`neuron_flexible` / `snn` code -- which is precisely what the refactor must not
change. Determinism: every neuron case sets its weights explicitly (no reliance on
the global RNG), and every engine is seeded (`np.random.default_rng(seed)` in
`_build`). There is no wall-clock/`time`-derived state on the fixed-point path.

Design notes:
- Per-neuron driver order each step: receive_input -> apply_inhibition ->
  (check_threshold -> fire) -> update -> record. This order is arbitrary but
  FIXED here, so gen and re-run share it; it does not need to match the engine.
- The SimulationEngine is the source of truth for the running config: we capture
  BOTH `SimulationEngine()` at its own defaults AND the live dashboard engine
  (`backend.api.engine`), so the FE gate + loser-depression archival that live in
  the engine/api layer are pinned by this suite.
"""
import numpy as np

from neuron_flexible import Neuron

T_NEURON = 30          # steps per neuron case
T_ENGINE = 40          # steps per engine case
_ENGINE_INPUT = [0, 0, 0, 1.0, 1.0, 1.0, 0, 0, 0]   # "row 1", fixed


# --------------------------------------------------------------- neuron driver
def _run_neuron(n, schedule, T=T_NEURON):
    """Drive a finalized neuron through `schedule` (list of (inp, inh) per step,
    last entry repeats) and record its full state trajectory as arrays."""
    m = len(n.weights)
    pot = np.zeros(T)
    fired = np.zeros(T)
    refr = np.zeros(T)
    wtraj = np.zeros((T, m))
    for t in range(T):
        inp, inh = schedule[t] if t < len(schedule) else schedule[-1]
        if inp is not None:
            n.receive_input(np.asarray(inp, dtype=float), t=t)
        if inh is not None:
            n.apply_inhibition(np.asarray(inh, dtype=float))
        if n.check_threshold():
            n.fire()
            fired[t] = 1.0
        n.update()
        pot[t] = n.potential
        refr[t] = n.refractory_timer
        wtraj[t] = n.weights
    return dict(potential=pot, fired=fired, refractory=refr, weights=wtraj,
                trace=n.trace, confidence=n.confidence)


def _rep(*steps):
    """A schedule that repeats the given (inp, inh) steps forever."""
    return list(steps)


# ------------------------------------------------------------------ the matrix
def _neuron_cases():
    """Yield (name, neuron, schedule). Weights are set explicitly for determinism.
    Each case isolates one mode from REFACTOR_PLAN Phase 0's list."""

    # 1. bare fixed-fan-in neuron, default (charge-based) rule.
    n = Neuron(n_inputs=4, threshold=1.0, leak_rate=0.01, learning_rate=0.1)
    n.weights = np.array([0.6, 0.6, 0.3, 0.1])
    yield "bare_fixed", n, _rep(([1, 1, 0, 0], None))

    # 2. staged-wiring neuron, same weights/drive -> must match bare_fixed.
    n = Neuron(threshold=1.0, leak_rate=0.01, learning_rate=0.1)
    for w in (0.6, 0.6, 0.3, 0.1):
        n.add_input_connection(w)
    n.finalize_connections()
    yield "staged_wiring", n, _rep(([1, 1, 0, 0], None))

    # 3. signed-spike feedforward rule.
    n = Neuron(n_inputs=6, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.signed_spike_learning = True
    n.min_positive_weight = 0.01
    n.weights = np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
    yield "signed_spike", n, _rep(([1, 1, 1, 0, 0, 0], None))

    # 4. confidence consolidation (+ loser depression + signed depression). One
    #    negative gate (idx3) so apply_inhibition drives loser depression.
    n = Neuron(n_inputs=4, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.confidence_consolidation = True
    n.loser_depression = True
    n.signed_depression = True
    n.eta_off = 0.2
    n.conf_cap = 0.8
    n.min_positive_weight = 0.01
    n.weights = np.array([0.6, 0.6, 0.3, -0.5])
    # even steps: drive+fire (potentiation); odd steps: charge a little + inhibit
    # (loser depression, since a real discharge occurs).
    yield "confidence_consol", n, _rep(([1, 1, 0, 0], None), ([1, 0, 0, 0], [0, 0, 0, 1]))

    # 5. flow-proportional assembly credit (E->I integrator).
    n = Neuron(n_inputs=4, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.assembly_flow_credit = True
    n.excitatory_saturation_cap = 1.0
    n.min_positive_weight = 0.01
    n.weights = np.array([0.6, 0.6, 0.3, 0.05])
    yield "assembly_flow", n, _rep(([1, 1, 1, 0], None))

    # 6. structural free-energy gate on the signed rule.
    n = Neuron(n_inputs=6, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.signed_spike_learning = True
    n.structural_free_energy = True
    n.structural_fe_eta_floor = 0.02
    n.min_positive_weight = 0.01
    n.weights = np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
    yield "structural_fe", n, _rep(([1, 1, 1, 0, 0, 0], None))

    # 7-9. inhibitory rules: charge (no fire, threshold high) then discharge+learn.
    for name, setup in (
        ("inh_saturating", lambda x: None),
        ("inh_delta_turnover", lambda x: (setattr(x, "inhibitory_delta_rule", True),
                                          setattr(x, "inhibitory_rule_mode", "turnover"))),
        ("inh_delta_margin", lambda x: (setattr(x, "inhibitory_delta_rule", True),
                                        setattr(x, "inhibitory_rule_mode", "margin"))),
    ):
        n = Neuron(n_inputs=2, threshold=1.0, weight_cap=1.0, inhibitory_learning_rate=0.05)
        setup(n)
        n.weights = np.array([0.9, -0.5])   # idx0 excites, idx1 is the inhibitory gate
        yield name, n, _rep(([1, 0], [0, 1]))

    # 10. excitatory flow-rate delivery.
    n = Neuron(n_inputs=4, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.excitatory_flow_rate = True
    n.exc_trace_decay = 0.8
    n.weights = np.array([0.6, 0.6, 0.3, 0.1])
    yield "flow_rate", n, _rep(([1, 1, 0, 0], None))

    # 11. inhibitory flow-rate delivery (drains over steps in update()).
    n = Neuron(n_inputs=2, threshold=1.0, weight_cap=1.0, inhibitory_learning_rate=0.05)
    n.inhibitory_flow_rate = True
    n.weights = np.array([0.9, -0.5])
    yield "inh_flow", n, _rep(([1, 0], [0, 1]), ([0, 0], None))

    # 12. distance attenuation of delivered drive.
    n = Neuron(n_inputs=4, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.distance_weighting = True
    n.distance = np.array([1.0, 2.0, 1.0, 1.0])
    n.weights = np.array([0.6, 0.9, 0.3, 0.1])
    yield "distance", n, _rep(([1, 1, 0, 0], None))

    # 13. subtractive reset on fire (keep residual overshoot).
    n = Neuron(n_inputs=3, threshold=1.0, weight_cap=2.0, learning_rate=0.1)
    n.subtractive_reset = True
    n.weights = np.array([0.8, 0.8, 0.8])
    yield "subtractive_reset", n, _rep(([1, 1, 1], None))

    # 14. membrane saturation ceiling.
    n = Neuron(n_inputs=3, threshold=1.0, weight_cap=2.0, learning_rate=0.1)
    n.v_sat = 1.5
    n.weights = np.array([0.9, 0.9, 0.9])
    yield "v_sat", n, _rep(([1, 1, 1], None))

    # 15. homeostatic synaptic scaling.
    n = Neuron(n_inputs=3, threshold=1.0, weight_cap=2.0, learning_rate=0.1,
               homeostasis=True, ca_rate=0.1, ca_target=0.05, ca_band=0.5,
               homeo_up=0.02, homeo_down=0.02)
    n.weights = np.array([0.6, 0.6, 0.3])
    yield "homeostasis", n, _rep(([1, 1, 0], None))

    # 16. fixed positive-weight budget + floor (renorm tail).
    n = Neuron(n_inputs=4, threshold=1.0, weight_cap=1.0, learning_rate=0.1)
    n.weight_budget = 1.2
    n.min_positive_weight = 0.05
    n.weights = np.array([0.6, 0.6, 0.3, 0.1])
    yield "budget_floor", n, _rep(([1, 1, 0, 0], None))


def _run_engine(engine, tag, out):
    """Drive an engine with the fixed input for T_ENGINE steps; record every
    neuron's potential trajectory and the final sorted weight vector."""
    engine.set_input(_ENGINE_INPUT)
    ids = list(engine.neurons.keys())
    pots = np.zeros((T_ENGINE, len(ids)))
    for t in range(T_ENGINE):
        engine.step()
        for k, nid in enumerate(ids):
            pots[t, k] = engine.neurons[nid].potential
    w = engine._all_weights()
    wkeys = sorted(w.keys())
    out[f"{tag}__potentials"] = pots
    out[f"{tag}__weights_final"] = np.array([w[k] for k in wkeys], dtype=float)


def collect():
    """Return the full flat dict of golden arrays. Deterministic."""
    out = {}
    for name, neuron, schedule in _neuron_cases():
        rec = _run_neuron(neuron, schedule)
        for field, arr in rec.items():
            out[f"neuron_{name}__{field}"] = np.asarray(arr, dtype=float)

    # Engine at its OWN defaults (SimulationEngine is the config source of truth).
    from backend.simulation import SimulationEngine
    _run_engine(SimulationEngine(seed=1), "engine_defaults", out)

    # Live dashboard engine (pins the FE gate + loser-depression archival that
    # live in backend/api.py). reset() rebuilds fresh, seeded weights.
    from backend.api import engine as dash_engine
    dash_engine.reset()
    _run_engine(dash_engine, "engine_dashboard", out)
    return out
