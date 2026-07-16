"""Frequency-behaviour regression.

Pins the leaky periodic-integrator model that the frequency experiment is built
on: the generalized fast/slow inequality for several active fan-in counts, the
exact ``Q = sum(active weights)`` treatment of heterogeneous weights, and the
agreement between the analytic irregular-interval recurrence and the engine's real
membrane trajectory. These are model invariants, not long floating-point histories.
"""

import numpy as np
import pytest

from snn.neurons import E_THRESHOLD
from experiments.frequency_experiment import (
    v_peak,
    fires_at,
    selectivity_window,
    simulate_periodic_integrator,
    recurrence_trace,
    _jump_leak_reference_trace,
)


@pytest.mark.parametrize('leak', [0.1, 0.2, 0.3])
@pytest.mark.parametrize('N_active', [1, 3, 5])
def test_generalized_fast_slow_inequality(leak, N_active):
    # A charge in the middle of the band integrates at full rate but not half rate,
    # for ANY active fan-in -- no hard-coded division by three.
    Q_lo, Q_hi, w_lo, w_hi = selectivity_window(leak, T_fast=1, T_slow=2, N_active=N_active)
    assert Q_lo < Q_hi
    assert w_lo == pytest.approx(Q_lo / N_active)     # mean-weight band scales by N_active
    Q_mid = 0.5 * (Q_lo + Q_hi)
    assert fires_at(Q_mid, leak, T=1)
    assert not fires_at(Q_mid, leak, T=2)
    # And the real neuron agrees with the analytic prediction.
    assert simulate_periodic_integrator(Q_mid, leak, T=1) is not None
    assert simulate_periodic_integrator(Q_mid, leak, T=2) is None


def test_heterogeneous_weights_use_sum_not_cap():
    # Q is the sum of the ACTUAL active weights, not N_active * cap. A cap is an
    # upper bound; measure the real weights.
    leak, T = 0.2, 1
    active_weights = np.array([120.0, 40.0, 300.0])   # heterogeneous, none at cap
    Q = float(active_weights.sum())                    # 460
    assert fires_at(Q, leak, T) == (v_peak(Q, leak, T) >= E_THRESHOLD)
    # Using N_active * cap would badly overstate the charge here.
    wrong = len(active_weights) * E_THRESHOLD
    assert wrong > 5 * Q


def test_recurrence_matches_reference_integrator_on_irregular_intervals():
    # The frequency experiment analyses an abstract leaky *jump* integrator (its
    # V_pre/V_post recurrence). The engine neuron is now conductance-based, so the
    # recurrence is validated against the module's self-contained reference
    # integrator rather than the engine neuron.
    leak = 0.15
    gaps = [1, 2, 1, 3, 2]           # irregular inter-volley steps
    charges = [180.0, 90.0, 210.0, 60.0, 150.0]
    reference_post = _jump_leak_reference_trace(charges, gaps, leak)
    analytic = recurrence_trace(charges, gaps, leak)
    assert reference_post == pytest.approx(analytic, rel=1e-9)


def test_v_peak_zero_leak_is_unbounded_integrator():
    assert v_peak(100.0, leak_rate=0.0, T=5) == float('inf')
    assert fires_at(1.0, leak_rate=0.0, T=1000)      # any charge eventually crosses
