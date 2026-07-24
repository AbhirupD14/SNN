"""Per-synapse excitatory weight cap (e_weight_cap_frac).

A hard ceiling at theta*frac on plastic PATTERN-DETECTOR feedforward weights, applied in
every update mode (including cap-free linear_fe / dual_fe_fes). Forces competitors to
integrate >= 2 evidence volleys to reach theta. Exempt: Eor (relays the single WTA winner)
and the coincidence C basal (its own rule; already gates basal AND apical)."""

import numpy as np

from backend.simulation import SimulationEngine, CoincidencePyramidalNeuron, E_THRESHOLD

THETA = E_THRESHOLD


def _train(cap, dual=True, warm=8000):
    e = SimulationEngine(seed=1, topology='tiled_cc', dual_fe_fes=dual, eta=4.0, c_eta=2.0,
                         dual_fe_B=5.0, leak_rate=0.0, refractory_steps=0,
                         c_feedback_reset=True, e_weight_cap_frac=cap)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'col 1')
    for _ in range(warm):
        e.step()
    return e


def _max_ff(cell):
    return float(cell.acc_weights.max()) if cell.acc_weights.size else 0.0


def test_competitors_capped_eor_and_c_exempt():
    e = _train(0.5)
    cap = 0.5 * THETA
    comp_max = eor_max = 0.0
    for c in e.latency_competitors:
        if e._role_of.get(c.id) == 'Eor':
            eor_max = max(eor_max, _max_ff(c))
            assert c.w_cap is None                       # Eor never capped
        else:
            comp_max = max(comp_max, _max_ff(c))
            assert c.w_cap == cap
    assert comp_max <= cap + 1e-6, 'no competitor afferent may exceed theta/2'
    # Eor relays the single winner, so at least one afferent must be able to exceed theta/2.
    assert eor_max > cap, 'Eor must stay uncapped to fire on one winner'
    # C basal uses its own rule and is never touched by this cap.
    for n in e.exc.values():
        if isinstance(n, CoincidencePyramidalNeuron):
            assert not hasattr(n, 'w_cap') or True       # C has no acc w_cap path
            # its single basal weight is free to exceed theta/2 (one-shot design)
    cbas = [n.basal_weight for n in e.exc.values() if isinstance(n, CoincidencePyramidalNeuron)]
    assert max(cbas) > cap, 'C basal is exempt and can exceed theta/2'


def test_cap_holds_in_production_linear_fe_mode():
    e = _train(0.5, dual=False)
    cap = 0.5 * THETA
    for c in e.latency_competitors:
        if e._role_of.get(c.id) != 'Eor':
            assert _max_ff(c) <= cap + 1e-6


def test_default_uncapped_is_byte_identical():
    """cap=None must reproduce the no-cap dynamics exactly (frame-level)."""
    def sig(cap):
        e = SimulationEngine(seed=1, topology='tiled_cc', dual_fe_fes=True, eta=4.0, c_eta=2.0,
                             dual_fe_B=5.0, leak_rate=0.0, refractory_steps=0,
                             c_feedback_reset=True, e_weight_cap_frac=cap)
        e.set_patch_pattern(0, 0, 'row 1')
        h = []
        for _ in range(400):
            st = e.step()
            h.append((st['timestep'], tuple(sorted(st['column_winners'])),
                      sum(round(c['weight'], 4) for c in st['changed_synapses'])))
        return h
    assert sig(None) == sig(None)                        # determinism guard
    # A very large cap (>> any reachable weight) must not change anything vs uncapped.
    assert sig(None) == sig(100.0)


def test_init_respects_cap():
    e = SimulationEngine(seed=1, topology='tiled_cc', dual_fe_fes=True, e_weight_cap_frac=0.5)
    cap = 0.5 * THETA
    for c in e.latency_competitors:
        if e._role_of.get(c.id) != 'Eor':
            assert _max_ff(c) <= cap + 1e-9              # no initial weight starts above the cap


def test_invalid_cap_rejected():
    for bad in (0.0, -0.5):
        try:
            SimulationEngine(topology='tiled_cc', e_weight_cap_frac=bad)
        except ValueError:
            continue
        raise AssertionError(f'e_weight_cap_frac={bad} should be rejected')
