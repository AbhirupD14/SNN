"""Phase 1 — shared ConductanceLIFNeuron base: exact sub-boundary segment
advancement, analytic threshold-crossing times, and the hard-reset primitive.

These pin the NEW event-resolved membrane primitives added without disturbing the
legacy whole-boundary ``integrate()`` path. The equivalence tests prove a single
``advance_segment(delta_tau=1)`` reproduces the historical full-boundary result to
tight numerical tolerance, so the event loop introduced in later phases cannot
change legacy E dynamics.
"""

import math

import numpy as np
import pytest

from snn.neurons import (
    ConductanceLIFNeuron,
    ExcitatoryNeuron,
    E_THRESHOLD,
    leak_to_conductance,
)


def make_base(**kw):
    return ConductanceLIFNeuron('C', 'test', **kw)


def make_e(**kw):
    n = kw.pop('n', 3)
    d = dict(acc_weights=np.full(n, 100.0), acc_distance_factor=np.ones(n), learn=False)
    d.update(kw)
    return ExcitatoryNeuron('E', 'test', **d)


# ---------------------------------------------------------- shared base identity
def test_excitatory_is_conductance_lif():
    assert issubclass(ExcitatoryNeuron, ConductanceLIFNeuron)
    assert isinstance(make_e(), ConductanceLIFNeuron)


def test_base_owns_no_learning_or_afferents():
    n = make_base()
    assert not hasattr(n, 'acc_weights')
    assert not hasattr(n, 'update_acc_weights')


# --------------------------------------------------- segment == full integration
@pytest.mark.parametrize('leak,ginh,Q,V0', [
    (0.03, 0.0, 900.0, 0.0),
    (0.03, 2.0, 1000.0, 300.0),
    (0.10, 0.0, 400.0, 500.0),
    (0.0, 0.0, 700.0, 100.0),      # g_total == 0 integrator branch
    (0.25, 5.0, 0.0, 800.0),       # decay only
])
def test_segment_delta_one_matches_full_integrate(leak, ginh, Q, V0):
    # Legacy full-boundary integrate(): drive arrives as pending_exc.
    a = make_base(leak_rate=leak)
    a.g_inh = ginh
    a.V = V0
    a.gather_exc(Q)
    a.integrate()
    # Event path: same drive frozen into remaining_excitation, one Delta_tau = 1 segment.
    b = make_base(leak_rate=leak)
    b.g_inh = ginh
    b.V = V0
    b.gather_exc(Q)
    b.freeze_drive()
    b.advance_segment(1.0)
    assert b.V == pytest.approx(a.V, abs=1e-9)


def test_segment_composition_additivity():
    # Advancing 0.3 then 0.7 (constant frozen drive) equals one 1.0 advance.
    whole = make_base(leak_rate=0.05)
    whole.g_inh = 1.0
    whole.V = 120.0
    whole.gather_exc(800.0)
    whole.freeze_drive()
    whole.advance_segment(1.0)

    pieces = make_base(leak_rate=0.05)
    pieces.g_inh = 1.0
    pieces.V = 120.0
    pieces.gather_exc(800.0)
    pieces.freeze_drive()
    pieces.advance_segment(0.3)
    pieces.advance_segment(0.7)
    assert pieces.V == pytest.approx(whole.V, abs=1e-9)


def test_freeze_drive_moves_pending_into_frozen_packet():
    n = make_base(leak_rate=0.0)
    n.gather_exc(250.0)
    assert n.pending_exc == pytest.approx(250.0)
    assert n.remaining_excitation == 0.0
    n.freeze_drive()
    assert n.pending_exc == 0.0
    assert n.remaining_excitation == pytest.approx(250.0)


# ------------------------------------------------------------- crossing-time math
def test_crossing_time_matches_trajectory_substitution():
    # A finite crossing exists (v_inf > theta). The returned Delta_tau, substituted
    # back into the exact trajectory, must land exactly on threshold.
    n = make_base(leak_rate=0.03, threshold=1000.0)
    n.V = 400.0
    n.gather_exc(1.05 * 1000.0 / ((1 - math.exp(-leak_to_conductance(0.03))) /
                                  leak_to_conductance(0.03)))
    n.freeze_drive()
    dtau = n.crossing_time(1.0)
    assert 0.0 < dtau < 1.0
    g = n.g_L + n.g_inh
    v_inf = (g * 0.0 + n.remaining_excitation) / g
    v_at = v_inf + (400.0 - v_inf) * math.exp(-g * dtau)
    assert v_at == pytest.approx(1000.0, abs=1e-6)


def test_crossing_time_zero_leak_linear():
    n = make_base(leak_rate=0.0, threshold=1000.0)
    n.V = 200.0
    n.gather_exc(400.0)          # I_exc = 400 constant; need 800 more -> 2.0 boundaries
    n.freeze_drive()
    assert n.crossing_time(10.0) == pytest.approx((1000.0 - 200.0) / 400.0)
    # But within a unit interval it does not cross:
    assert n.crossing_time(1.0) == math.inf


def test_crossing_time_no_crossing_returns_inf():
    # v_inf below threshold: leaky cell can never reach theta from this drive.
    n = make_base(leak_rate=0.03, threshold=1000.0)
    n.V = 100.0
    n.gather_exc(500.0)          # v_inf ~= 500/g_L*... well below theta after kappa
    n.freeze_drive()
    assert n.crossing_time(1.0) == math.inf


def test_crossing_time_immediate_when_already_supra():
    n = make_base(leak_rate=0.03, threshold=1000.0)
    n.V = 1200.0                 # already above threshold at segment start
    n.freeze_drive()
    assert n.crossing_time(1.0) == 0.0


def test_crossing_time_infinite_during_refractory_or_after_firing():
    n = make_base(leak_rate=0.0, threshold=1000.0)
    n.V = 5000.0                 # wildly supra-threshold
    n.freeze_drive()
    n.refractory_timer = 1
    assert n.crossing_time(1.0) == math.inf
    n.refractory_timer = 0
    n.fired_this_boundary = True
    assert n.crossing_time(1.0) == math.inf


# ------------------------------------------------------------------- fire(tau)
def test_fire_tau_records_spike_tau_and_consumes_drive():
    n = make_base(leak_rate=0.0, threshold=1000.0, refractory_steps=2)
    n.V = 1000.0
    n.gather_exc(500.0)
    n.freeze_drive()
    n.fire(0.42)
    assert n.V == 0.0
    assert n.v_pre == pytest.approx(1000.0)
    assert n.spike_tau == pytest.approx(0.42)
    assert n.remaining_excitation == 0.0
    assert n.fired_this_boundary is True
    assert n.refractory_timer == 2


def test_legacy_fire_no_arg_still_works():
    # The legacy engine calls fire() with no tau; behaviour must be preserved.
    n = make_e(leak_rate=0.03)
    n.gather_exc(2 * E_THRESHOLD)
    n.integrate()
    assert n.can_fire()
    v = n.fire()
    assert n.V == 0.0 and v > E_THRESHOLD
    assert n.spike_tau is None


# ------------------------------------------------------------------- hard_reset
def test_hard_reset_clears_v_and_drive_only():
    n = make_base(leak_rate=0.03, threshold=1000.0, refractory_steps=3)
    n.g_inh = 4.0
    n.a = 0.7
    n.V = 600.0
    n.gather_exc(900.0)
    n.freeze_drive()
    n.refractory_timer = 2
    n.spiked = True
    n.v_pre = 950.0
    n.v_pre_reset = 950.0
    n.hard_reset(0.31)
    assert n.V == 0.0                         # membrane wiped to rest
    assert n.remaining_excitation == 0.0      # frozen drive discarded
    assert n.g_inh == pytest.approx(4.0)      # conductance preserved
    assert n.a == pytest.approx(0.7)          # trace preserved
    assert n.refractory_timer == 2            # refractory untouched
    assert n.spiked is True                   # a prior spike is not erased
    assert n.v_pre == pytest.approx(950.0)
    assert n.v_pre_reset == pytest.approx(950.0)  # max depol preserved for trace


def test_hard_reset_keep_drive_option():
    n = make_base(leak_rate=0.0, threshold=1000.0)
    n.V = 300.0
    n.gather_exc(400.0)
    n.freeze_drive()
    n.hard_reset(0.5, discard_drive=False)
    assert n.V == 0.0
    assert n.remaining_excitation == pytest.approx(400.0)


def test_v_pre_reset_tracks_max_segment_endpoint():
    # v_pre_reset must be the MAX depolarized endpoint across segments, not the last.
    n = make_base(leak_rate=0.30, threshold=1e9)   # never fires; pure decay after peak
    n.V = 0.0
    n.gather_exc(900.0)
    n.freeze_drive()
    n.advance_segment(0.5)
    peak = n.V
    n.remaining_excitation = 0.0                     # drive gone -> subsequent decay
    n.advance_segment(0.5)
    assert n.V < peak                                # decayed below the peak
    assert n.v_pre_reset == pytest.approx(peak)      # but the max endpoint is retained
