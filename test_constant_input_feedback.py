"""Regression tests for continuous sensory drive and one-step L1I feedback."""

import numpy as np

from backend.simulation import L1I_FEEDBACK_REFRACTORY, N_PIX, SimulationEngine


def _isolated_pixel_engine():
    """One active L1E pixel with L2 unable to produce feedback autonomously."""
    engine = SimulationEngine(refractory=0, input_period=None,
                              l1i_immediate_relay=False,
                              l1i_leak_enabled=False)
    engine.set_input(np.array([1.0] + [0.0] * (N_PIX - 1)))
    for neuron in engine.l2.excitatory_neurons:
        neuron._weights_array[1:] = 0.0
    return engine


def test_constant_input_is_the_default():
    engine = SimulationEngine()
    assert engine.params['input_period'] == 1
    assert engine.params['cycle_period'] == engine.params['volley_period']
    print("PASS: held input is presented every step while the intrinsic cycle stays independent")


def test_l1i_bank_starts_synchronized_and_credits_its_window():
    engine = SimulationEngine(l1i_immediate_relay=False)
    reference = engine.l1.inhibitory_neurons[0].weights
    for l1i in engine.l1.inhibitory_neurons:
        assert np.array_equal(l1i.weights, reference)
        assert l1i.assembly_flow_credit is True
        assert l1i.assembly_decay_frac == 0.0
        assert l1i.refractory_period == L1I_FEEDBACK_REFRACTORY
    print("PASS: L1I bank shares phase-neutral initialization and temporal credit")


def test_constant_drive_fires_each_step_without_feedback():
    engine = _isolated_pixel_engine()
    fired = []
    for _ in range(5):
        engine.step()
        fired.append(bool(engine.spiked['L1E0']))
    assert fired == [True] * 5, fired
    print("PASS: an uninhibited held pixel fires on every constant-drive step")


def test_l1i_spike_suppresses_exactly_the_next_step():
    engine = _isolated_pixel_engine()
    l1i = engine.l1.inhibitory_neurons[0]

    # Force one real L1I threshold event at t=0. L1E is processed earlier in the
    # step, so this spike must affect t=1, not the current input sample.
    engine.stimulate('L1I0', magnitude=l1i.threshold)
    engine.step()
    at_spike = (bool(engine.spiked['L1E0']), bool(engine.spiked['L1I0']))

    engine.step()
    suppressed = (bool(engine.spiked['L1E0']), bool(engine.spiked['L1I0']))

    engine.step()
    released = (bool(engine.spiked['L1E0']), bool(engine.spiked['L1I0']))

    assert at_spike == (True, True), at_spike
    assert suppressed == (False, False), suppressed
    assert released == (True, False), released
    assert not engine.l1i_feedback_delay.any()
    print("PASS: L1I produces one-step delayed inhibition, then releases constant input")


def test_dashboard_training_halves_held_pixel_frequency_in_phase():
    # Reuse the dashboard's public configuration as the integration contract, but
    # rebuild with a fixed seed so the regression is independent of persisted UI state.
    from backend.api import engine as dashboard_engine

    engine = SimulationEngine(**{**dashboard_engine.params, 'seed': 1})
    active = np.flatnonzero(engine.input_vec > 0.5)
    history = []
    for t in range(1500):
        engine.step()
        if t >= 1300:
            history.append((
                [bool(engine.spiked[f'L1E{i}']) for i in active],
                [bool(engine.spiked[f'L1I{i}']) for i in active],
            ))

    l1e = np.asarray([e for e, _ in history], dtype=int)
    l1i = np.asarray([i for _, i in history], dtype=int)
    assert np.all(l1e == l1e[:, :1])
    assert np.all(l1i == l1i[:, :1])
    assert np.array_equal(l1e, l1i)
    assert np.all(l1e.sum(axis=0) == len(history) // 2)
    assert not np.any(l1i[1:] & l1i[:-1])
    print("PASS: trained dashboard loop synchronizes active pixels at half frequency")


if __name__ == '__main__':
    test_constant_input_is_the_default()
    test_l1i_bank_starts_synchronized_and_credits_its_window()
    test_constant_drive_fires_each_step_without_feedback()
    test_l1i_spike_suppresses_exactly_the_next_step()
    test_dashboard_training_halves_held_pixel_frequency_in_phase()
    print("\nALL CONSTANT-INPUT FEEDBACK TESTS PASSED")
