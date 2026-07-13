"""Regression tests for independent L2E and L2I membrane-leak switches."""

import numpy as np

from backend.simulation import (L1_EI_WEIGHT_INIT_HIGH_FRAC,
                                L1_EI_WEIGHT_INIT_LOW_FRAC,
                                L1I_LEAK_RATE, L2I_LEAK_RATE,
                                SimulationEngine)


def test_all_trainable_population_leaks_default_off():
    engine = SimulationEngine()
    assert engine.params['leak_enabled'] is False
    assert engine.params['l2i_leak_enabled'] is False
    assert engine.params['l1i_leak_enabled'] is False
    assert all(n.leak_rate == 0.0 for n in engine.l2.excitatory_neurons)
    assert engine.l2.inhibitory_neuron.leak_rate == 0.0
    assert all(n.leak_rate == 0.0 for n in engine.l1.inhibitory_neurons)
    print("PASS: L2E, L2I, and L1I leak all default off")


def test_constructor_switches_l2e_and_l2i_independently():
    engine = SimulationEngine(leak_enabled=False, l2i_leak_enabled=True,
                              l1i_leak_enabled=True)

    assert all(n.leak_rate == 0.0 for n in engine.l2.excitatory_neurons)
    assert np.isclose(engine.l2.inhibitory_neuron.leak_rate, L2I_LEAK_RATE)
    assert all(np.isclose(n.leak_rate, L1I_LEAK_RATE)
               for n in engine.l1.inhibitory_neurons)

    engine = SimulationEngine(leak_enabled=True, leak_l2=0.023,
                              l2i_leak_enabled=False)
    assert all(np.isclose(n.leak_rate, 0.023) for n in engine.l2.excitatory_neurons)
    assert engine.l2.inhibitory_neuron.leak_rate == 0.0
    print("PASS: constructor controls L2E and L2I leak independently")


def test_l1i_accumulator_does_not_leak_by_default():
    engine = SimulationEngine(l1i_immediate_relay=False)
    assert all(n.leak_rate == 0.0 for n in engine.l1.inhibitory_neurons)

    l1i = engine.l1.inhibitory_neurons[0]
    l1i.potential = 0.5 * l1i.threshold
    before = l1i.potential
    l1i.update()
    assert l1i.potential == before, "L1I accumulator charge decayed with leak disabled"

    engine.apply_config({"l1i_leak_enabled": True})
    assert all(np.isclose(n.leak_rate, L1I_LEAK_RATE)
               for n in engine.l1.inhibitory_neurons)
    print("PASS: trainable L1I is non-leaky by default and leak remains independently togglable")


def test_l1i_uses_l2i_threshold_scale():
    engine = SimulationEngine(l2i_threshold_frac=1 / 3,
                              l1i_threshold_frac=1.0,
                              l1i_immediate_relay=False)
    l2i = engine.l2.inhibitory_neuron
    for l1i in engine.l1.inhibitory_neurons:
        assert l1i.threshold == l2i.threshold
        assert l1i.weight_cap == l2i.weight_cap
        assert np.all(l1i.weights >= L1_EI_WEIGHT_INIT_LOW_FRAC * l2i.threshold)
        assert np.all(l1i.weights <= L1_EI_WEIGHT_INIT_HIGH_FRAC * l2i.threshold)
    assert max(float(n.weights.max()) for n in engine.l1.inhibitory_neurons) > engine.params['threshold'], \
        "L2I-scale L1I initialization was clipped at the old L1E threshold"
    print("PASS: L1I threshold, cap, and initialization use the L2I scale")


def test_l2i_does_not_decay_when_its_switch_is_off():
    engine = SimulationEngine(l2i_leak_enabled=False)

    l2i = engine.l2.inhibitory_neuron
    l2i.potential = 0.5 * l2i.threshold
    before = l2i.potential
    l2i.update()
    assert l2i.potential == before, "L2I potential decayed with L2I leak disabled"
    print("PASS: L2I charge does not decay when its own switch is off")


def test_enabled_rates_and_live_rebuild():
    engine = SimulationEngine(leak_enabled=True, leak_l2=0.023,
                              l2i_leak_enabled=True)
    assert all(np.isclose(n.leak_rate, 0.023) for n in engine.l2.excitatory_neurons)
    assert np.isclose(engine.l2.inhibitory_neuron.leak_rate, L2I_LEAK_RATE)

    engine.apply_config({"leak_enabled": False})
    assert all(n.leak_rate == 0.0 for n in engine.l2.excitatory_neurons)
    assert np.isclose(engine.l2.inhibitory_neuron.leak_rate, L2I_LEAK_RATE)

    engine.apply_config({"l2i_leak_enabled": False})
    assert all(n.leak_rate == 0.0 for n in engine.l2.excitatory_neurons)
    assert engine.l2.inhibitory_neuron.leak_rate == 0.0

    engine.apply_config({"leak_enabled": True})
    assert all(np.isclose(n.leak_rate, 0.023) for n in engine.l2.excitatory_neurons)
    assert engine.l2.inhibitory_neuron.leak_rate == 0.0
    print("PASS: live rebuild preserves independent L2E/L2I leak settings")


def test_dynamic_state_uses_same_live_membrane_phase_for_l2e_and_l2i():
    engine = SimulationEngine(leak_enabled=False)
    l2e = engine.l2.excitatory_neurons[0]
    l2i = engine.l2.inhibitory_neuron
    l2e.potential = 123.0
    l2i.potential = 456.0

    # These snapshots intentionally represent earlier phases of a step. They must
    # not replace the live membrane for only one population in dynamic_state().
    engine.l2_drive["L2E0"] = 789.0
    engine.l2_charge["L2E0"] = 999.0
    state = {n["id"]: n for n in engine.dynamic_state()["neurons"]}

    assert state["L2E0"]["potential"] == 123.0
    assert state["L2I"]["potential"] == 456.0
    print("PASS: charge API samples L2E and L2I from the same live membrane phase")


if __name__ == "__main__":
    test_all_trainable_population_leaks_default_off()
    test_constructor_switches_l2e_and_l2i_independently()
    test_l1i_accumulator_does_not_leak_by_default()
    test_l1i_uses_l2i_threshold_scale()
    test_l2i_does_not_decay_when_its_switch_is_off()
    test_enabled_rates_and_live_rebuild()
    test_dynamic_state_uses_same_live_membrane_phase_for_l2e_and_l2i()
    print("\nALL L2 LEAK-TOGGLE TESTS PASSED")
