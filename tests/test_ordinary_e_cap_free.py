"""The ordinary-E feedforward learning rule is cap-free: a zero floor and NO individual
upper bound. The neuron-wide free-energy (FE) budget B = maturity_budget_frac*theta is the
only saturation mechanism, so a specialist's total incoming weight converges to B without a
per-synapse ceiling. C basal and predictive-inhibitory weights keep their OWN bounds, and
the dashboard no longer exposes or visually assumes the historical 500 cap.
"""

import numpy as np
import pytest

from snn.neurons import ExcitatoryNeuron, PredictiveInterneuron, E_THRESHOLD
from backend.simulation import SimulationEngine
from backend.dashboard_config import CONFIG_SPEC

B = 1.10 * E_THRESHOLD                                # FE budget at the default frac (~1100)


def _drive(n, participation, steps):
    part = np.asarray(participation, dtype=bool)
    for _ in range(steps):
        n.fire()
        n.update_acc_weights(part)


def test_one_afferent_specialist_exceeds_500_and_converges_to_budget():
    n = ExcitatoryNeuron('E', 't', acc_weights=np.zeros(1), acc_distance_factor=np.ones(1),
                         eta=0.01, leak_rate=0.0)
    assert n.update_mode == 'linear_fe'                      # cap-free production default
    _drive(n, [True], 6000)
    assert n.acc_weights[0] > 500.0                          # exceeds the old theta/2 ceiling
    assert n.acc_weights[0] == pytest.approx(B, abs=1.0)     # one-event integrator near B


def test_specialist_total_is_finite_and_does_not_overshoot_or_oscillate():
    n = ExcitatoryNeuron('E', 't', acc_weights=np.zeros(1), acc_distance_factor=np.ones(1),
                         eta=0.01, leak_rate=0.0)
    _drive(n, [True], 4000)
    sums = []
    for _ in range(200):
        n.fire(); n.update_acc_weights(np.array([True]))
        sums.append(float(n.acc_weights.sum()))
    s = np.array(sums)
    assert np.all(np.isfinite(s))
    assert s.max() <= B + 1e-6                               # never overshoots the budget
    # monotone-nondecreasing and settled (no oscillation): step-to-step change is tiny.
    assert np.all(np.diff(s) >= -1e-9)
    assert abs(s[-1] - s[0]) < 1.0


def test_three_of_nine_detector_converges_near_budget_without_pathological_growth():
    part = [True, True, True] + [False] * 6
    n = ExcitatoryNeuron('E', 't', acc_weights=np.zeros(9), acc_distance_factor=np.ones(9),
                         eta=0.01, leak_rate=0.0)
    _drive(n, part, 8000)
    total = float(n.acc_weights.sum())
    assert total == pytest.approx(B, abs=1.0)               # total budget, not runaway
    assert np.all(np.isfinite(n.acc_weights))
    active = n.acc_weights[np.asarray(part)]
    assert active.sum() == pytest.approx(B, abs=1.0)        # the three active carry the budget
    assert float(n.acc_weights[~np.asarray(part)].sum()) == pytest.approx(0.0, abs=1e-6)


def test_tiled_l1_eor_weight_can_exceed_500_and_matures():
    # An Eor cell integrates its local ordinary-E afferents; under sustained drive its row
    # total approaches the FE budget, and individual weights are free to exceed 500.
    e = SimulationEngine(seed=1, topology='tiled_cc', leak_rate=0.0)
    e.set_pattern('row 1')
    eors = [c for c in e.latency_competitors if getattr(c, 'id', '').endswith('Eor')]
    assert eors                                              # tiled graph exposes Eor competitors
    for _ in range(4000):
        e.step()
    maxima = [float(c.acc_weights.max()) for c in e.latency_competitors]
    totals = [float(c.acc_weights.sum()) for c in e.latency_competitors]
    assert max(maxima) > 500.0                               # some ordinary-E/Eor weight exceeds 500
    assert max(totals) <= B + 1.0                            # but no total overshoots the budget


def test_predictive_inhibitory_weights_remain_bounded_by_pi_w_max():
    pi = PredictiveInterneuron('PI', 'predictor', 3, w_init=0.0, w_max=0.7, eta=0.5)
    for _ in range(200):
        pi.learn(np.ones(3))                                 # maximal drive
    assert np.all(pi.w <= 0.7 + 1e-12)                       # clipped to its OWN cap (pi_w_max)
    assert np.all(pi.w >= 0.0)


def test_c_basal_weight_bounded_by_c_specific_cap():
    e = SimulationEngine(seed=1, topology='rg_coincidence')
    # C basal edits clip to the C cell's own cap, unlike cap-free ordinary feedforward.
    huge = e.set_synapse_weight('basal0', 1e9)
    assert 0.0 < huge < 1e9
    assert huge == pytest.approx(e.coincidence[0].w_max) or huge <= e.coincidence[0].w_max


def test_dashboard_no_longer_exposes_or_assumes_500_cap():
    keys = {c['key'] for c in CONFIG_SPEC}
    assert 'e_weight_cap' not in keys                        # no cap control on the dashboard
    params = SimulationEngine(topology='tiled_cc').topology()['params']
    # the display reference for ordinary-E weights is the FE budget, not a hard cap.
    assert params['e_maturity_budget'] == pytest.approx(B)
    assert params['e_maturity_budget'] > 500.0
