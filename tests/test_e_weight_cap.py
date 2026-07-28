"""Per-synapse excitatory weight caps.

Two hard ceilings, each applied in every update mode (including the cap-free linear_fe /
dual_fe_fes rules):

  * ``e_weight_cap_frac`` (theta/2) on plastic PATTERN-DETECTOR feedforward weights, so a
    competitor must integrate >= 2 evidence volleys to reach theta;
  * ``relay_weight_cap_frac`` (theta) on the ONE-AFFERENT relays -- the Eor feedforward bank
    and the coincidence C basal. These are exempt from the theta/2 rule (each fires on a
    single afferent, so theta/2 would make them unable to fire at all), but they are NOT
    unbounded: one afferent may reach exactly theta and no more."""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, CoincidencePyramidalNeuron, E_THRESHOLD

THETA = E_THRESHOLD


def _train(cap, dual=True, warm=8000, **kw):
    e = SimulationEngine(seed=1, topology='tiled_cc', dual_fe_fes=dual, eta=4.0, c_eta=2.0,
                         dual_fe_B=5.0, leak_rate=0.0, refractory_steps=0,
                         c_feedback_reset=True, e_weight_cap_frac=cap, **kw)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'col 1')
    for _ in range(warm):
        e.step()
    return e


def _max_ff(cell):
    return float(cell.acc_weights.max()) if cell.acc_weights.size else 0.0


def _c_cells(e):
    return [n for n in e.exc.values() if isinstance(n, CoincidencePyramidalNeuron)]


def test_competitors_capped_at_half_theta_relays_at_theta():
    e = _train(0.5)
    cap = 0.5 * THETA
    comp_max = eor_max = 0.0
    for c in e.latency_competitors:
        if e._role_of.get(c.id) == 'Eor':
            eor_max = max(eor_max, _max_ff(c))
            assert c.w_cap == THETA                      # one-afferent relay cap, not theta/2
        else:
            comp_max = max(comp_max, _max_ff(c))
            assert c.w_cap == cap
    assert comp_max <= cap + 1e-6, 'no competitor afferent may exceed theta/2'
    # Eor relays the single winner, so an afferent must be able to pass theta/2 -- but the
    # relay cap still bounds it at theta.
    assert cap < eor_max <= THETA + 1e-6
    for n in _c_cells(e):
        assert n.w_cap == THETA
    cbas = max(n.basal_weight for n in _c_cells(e))
    assert cap < cbas <= THETA + 1e-6, 'C basal passes theta/2 but never exceeds theta'


def test_eor_pins_at_theta_where_the_uncapped_rule_ran_away():
    # Measured with Eor plasticity explicitly ON and its historical seeded init, since the
    # production relay is frozen at theta and never learns at all (see test_tiled_cc_engine).
    plastic = dict(eor_plasticity_enabled=True, eor_w_init_frac=None)

    def peak(relay_cap):
        e = _train(0.5, relay_weight_cap_frac=relay_cap, **plastic)
        return max(_max_ff(c) for c in e.latency_competitors
                   if e._role_of.get(c.id) == 'Eor')

    assert peak(None) > THETA        # the dual rule genuinely runs past theta without the cap
    assert peak(1.0) == THETA        # and pins exactly at the ceiling with it


def test_frozen_eor_never_leaves_theta():
    e = _train(0.5)                                          # production defaults
    for c in e.latency_competitors:
        if e._role_of.get(c.id) == 'Eor':
            assert c.learn is False
            assert np.all(c.acc_weights == THETA)


def _drive_c(w_cap, mode, eta_c, *, w0=0.9 * THETA, steps=600):
    """Repeatedly fire one coincidence cell on basal+apical coincidence and return its
    final basal weight -- the causal learning path, isolated from any topology."""
    c = CoincidencePyramidalNeuron(
        nid='C', basal_source='S', basal_edge_id='b', apical_sources=['A'],
        apical_edge_ids=['a'], basal_weight=w0, w_max=8.0 * THETA, w_cap=w_cap,
        eta_c=eta_c, update_mode=mode, threshold=THETA, leak_rate=0.0)
    for _ in range(steps):
        c.begin_event_boundary()
        c.gather_basal('S')
        c.gather_apical('A')
        c.resolve_dendrites()
        c.apical_active = True
        c._deposit_signal = 1.0
        c.v_pre = THETA                       # dual FE peaks at the threshold-firing charge
        c.update_basal_weight()
    return c.basal_weight


# Each mode gets a rate in its own stable range: the budget-FE modes converge on
# frac*w1 = 1.1*theta, the dual rule climbs without a fixed point. Both settle above theta.
@pytest.mark.parametrize('mode,eta_c', [('c_linear_bounded', 0.01),
                                        ('c_quadratic_bounded', 0.01),
                                        ('c_linear_nonnegative', 0.01),
                                        ('c_dual_fe_fes', 5.0)])
def test_c_basal_cap_binds_in_every_update_mode(mode, eta_c):
    # w_max is set far above theta, so any bound observed here is the w_cap ceiling itself
    # and not the mode's own clip or the FE budget.
    assert _drive_c(None, mode, eta_c) > THETA           # unbounded without the cap
    assert _drive_c(THETA, mode, eta_c) == THETA         # pinned exactly at the cap with it


def test_capped_c_still_fires_one_shot_from_reset():
    # theta is the one-shot TARGET, not a barrier: a lone deposit of w=theta from reset
    # satisfies the inclusive ``V >= theta`` crossing test.
    c = CoincidencePyramidalNeuron(
        nid='C', basal_source='S', basal_edge_id='b', apical_sources=['A'],
        apical_edge_ids=['a'], basal_weight=THETA, w_cap=THETA, w_max=8.0 * THETA,
        threshold=THETA, leak_rate=0.0)
    assert c.basal_weight == THETA
    q = deliver_one_coincidence(c)
    assert q == THETA                                    # deposit == the capped weight
    assert c.V >= c.threshold and c.can_fire()


def deliver_one_coincidence(c):
    c.begin_event_boundary()
    c.gather_basal('S')
    c.gather_apical('A')
    return c.resolve_dendrites()


def test_relay_cap_none_is_uncapped():
    e = SimulationEngine(seed=1, topology='tiled_cc', relay_weight_cap_frac=None)
    for c in e.latency_competitors:
        if e._role_of.get(c.id) == 'Eor':
            assert c.w_cap is None
    for n in _c_cells(e):
        assert n.w_cap is None


def test_invalid_relay_cap_rejected():
    for bad in (0.0, -1.0):
        try:
            SimulationEngine(topology='tiled_cc', relay_weight_cap_frac=bad)
        except ValueError:
            continue
        raise AssertionError(f'relay_weight_cap_frac={bad} should be rejected')


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
