"""Synthetic L1I activation checks for the predictive preset (Experiment.md 7.1).

At rest, using fractions of theta_L1I, the predictive L1I membrane (local afferent
0.40*theta, feedback afferents, leak 0.50 per update) must satisfy EXACTLY:

    one local event                     0.40 -> no spike
    one initial feedback event          0.10 -> no spike
    one local + one initial feedback    0.50 -> no spike
    one local + one feedback at u=0.80  1.20 -> spike
    20 local-only steps (leak 0.50)          -> no spike (pre-threshold -> 0.80 theta)
    20 initial-feedback-only steps           -> no spike

These pin that the preset's unit conversion is correct and that an asymptotic
threshold approach never rounds up into a spike. No task calibration is used.

Also checks the engine's `local_only` mode (paired, feedback delivery off) produces
NO L1I spikes over a long held presentation -- the intended isolated-local control.

    PYTHONPATH=. .venv/bin/python test_l1i_activation.py
"""
import numpy as np

from neuron_flexible import Neuron
from backend.simulation import SimulationEngine
from backend.dashboard_config import DASHBOARD_OVERRIDES

THETA = 1.0          # theta_L1I; V_rest=0 so G = theta. Fractions are scale-free.
LEAK = 0.50


def _l1i(weights):
    """A predictive-preset L1I building block: threshold THETA, leak 0.50,
    refractory 2, the given afferent weights. weight_cap high enough not to clip."""
    n = Neuron(n_inputs=len(weights), threshold=THETA, weight_cap=10.0,
               leak_rate=LEAK, refractory_period=2)
    n.weights = np.array(weights, dtype=float)
    return n


def _one_event(weights, spikes):
    """Deposit one event and return whether the neuron crosses threshold (pre-leak,
    exactly as step() checks after receive_input)."""
    n = _l1i(weights)
    n.receive_input(np.array(spikes, dtype=float))
    return n.check_threshold()


def _held(weights, spikes, steps=20):
    """Drive the SAME event every step for `steps` steps with leak; return whether
    it EVER crosses threshold."""
    n = _l1i(weights)
    fired = False
    for _ in range(steps):
        n.receive_input(np.array(spikes, dtype=float))
        if n.check_threshold():
            fired = True
            n.fire()
        n.update()
    return fired


def test_single_events():
    # weights: [local=0.40, fb_initial=0.10, fb_strong=0.80]. check_threshold()
    # returns a numpy bool, so assert on truthiness (not `is True`/`is False`).
    w = [0.40 * THETA, 0.10 * THETA, 0.80 * THETA]
    assert not _one_event(w, [1, 0, 0]), "one local (0.40) should not spike"
    assert not _one_event(w, [0, 1, 0]), "one initial feedback (0.10) should not spike"
    assert not _one_event(w, [1, 1, 0]), "local+initial fb (0.50) should not spike"
    assert _one_event(w, [1, 0, 1]), "local+fb@u=0.80 (1.20) MUST spike"
    print("  single-event thresholds (0.40/0.10/0.50 silent, 1.20 spikes): OK")


def test_held_events_no_spike():
    w_local = [0.40 * THETA]
    w_fb = [0.10 * THETA]
    assert not _held(w_local, [1], steps=20), \
        "20 local-only steps must not spike (pre-threshold limit 0.80 theta)"
    assert not _held(w_fb, [1], steps=20), \
        "20 initial-feedback-only steps must not spike"
    # And confirm the pre-threshold limit is indeed below theta (headroom check).
    n = _l1i(w_local)
    peak = 0.0
    for _ in range(40):
        n.receive_input(np.array([1.0]))
        peak = max(peak, n.potential)
        n.update()
    assert peak < THETA, f"local-only pre-threshold peak {peak} reached theta"
    assert 0.79 * THETA < peak < 0.80 * THETA, f"peak {peak} not approaching 0.80 theta"
    print(f"  held local-only converges to {peak:.4f} (< 0.80 theta), no spike: OK")


def test_engine_local_only_no_spikes():
    # local_only mode: paired on, predictor off, feedback delivery off.
    e = SimulationEngine(seed=2, **{**DASHBOARD_OVERRIDES,
                                    'paired_local_enabled': True,
                                    'predictive_feedback_enabled': False,
                                    'l2_to_l1i_delivery_enabled': False})
    e.set_input([0, 0, 0, 1, 1, 1, 0, 0, 0])
    l1i_spikes = 0
    for _ in range(200):
        e.step()
        l1i_spikes += sum(int(e.spiked[f'L1I{i}']) for i in range(9))
    assert l1i_spikes == 0, f"local_only produced {l1i_spikes} L1I spikes (expected 0)"
    print("  engine local_only: 0 L1I spikes over 200 steps (isolated-local control): OK")


def main():
    test_single_events()
    test_held_events_no_spike()
    test_engine_local_only_no_spikes()
    print("PASS: L1I predictive activation checks (Section 7.1)")


if __name__ == "__main__":
    main()
