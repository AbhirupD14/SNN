"""Inhibitory relay: no weight vector, no spontaneous spike, instant binary
firing on any input, and the one-third reported threshold invariant.
"""

import pytest

from snn.neurons import InhibitoryNeuron, ExcitatoryNeuron, E_THRESHOLD, I_THRESHOLD


def test_no_weight_vector():
    relay = InhibitoryNeuron('L1I0', 'relay')
    assert not hasattr(relay, 'acc_weights')
    assert not hasattr(relay, 'subt_magnitude')


def test_no_spike_without_input():
    relay = InhibitoryNeuron('L1I0', 'relay')
    assert relay.resolve() is False
    assert relay.spiked is False


def test_any_input_produces_immediate_binary_spike():
    relay = InhibitoryNeuron('L2I', 'relay')
    relay.receive()
    assert relay.resolve() is True
    assert relay.spiked is True
    # A second signal in the same phase does not "accumulate" -- still one spike.
    relay.receive()
    assert relay.resolve() is True


def test_clear_resets_event_state():
    relay = InhibitoryNeuron('L1I3', 'relay')
    relay.receive(); relay.resolve()
    relay.clear()
    assert relay.received_signal is False and relay.spiked is False


def test_reported_threshold_is_one_third_of_excitatory():
    relay = InhibitoryNeuron('L2I', 'relay')
    assert relay.threshold == pytest.approx(E_THRESHOLD / 3.0)
    assert relay.threshold == pytest.approx(I_THRESHOLD)
    # And it is a third of an actual excitatory neuron's threshold.
    exc = ExcitatoryNeuron('L2E0', 'competitor', acc_weights=[1.0], acc_distance_factor=[1.0])
    assert relay.threshold == pytest.approx(exc.threshold / 3.0)
