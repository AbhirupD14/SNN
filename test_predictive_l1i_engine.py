"""Engine-level integration checks for the predictive L1I path (Experiment.md 6/7).

Complements the isolated predictor test (test_predictor_rule.py) and the activation
checks (test_l1i_activation.py) by driving the whole SimulationEngine in the
local_plus_feedback preset and asserting the Section 8 regression contracts that
only appear once the pieces are wired together:

  - the predictor moves delivered feedback weights but NEVER the fixed local
    afferent, and all feedback weights stay clamped to [0, G];
  - generic and predictive L1I learning cannot run simultaneously (generic off);
  - the predictor still updates a delivered feedback synapse while its target L1I
    is refractory;
  - a queued L1I spike inhibits ONLY its paired L1E on the next timestep, via
    apply_inhibition(), with graded removal floored at rest.

    PYTHONPATH=. .venv/bin/python test_predictive_l1i_engine.py
"""
import numpy as np

from backend.simulation import SimulationEngine, N_OUT, N_PIX
from backend.dashboard_config import DASHBOARD_OVERRIDES

LPF = {**DASHBOARD_OVERRIDES, 'paired_local_enabled': True,
       'predictive_feedback_enabled': True, 'l2_to_l1i_delivery_enabled': True}


def test_predictor_moves_feedback_not_local():
    e = SimulationEngine(seed=1, **LPF)
    # generic L1I learning must be OFF everywhere (cannot coexist with predictor).
    assert all(not n.postsynaptic_learning_enabled for n in e.l1.inhibitory_neurons)
    local0 = np.array([n._weights_array[0] for n in e.l1.inhibitory_neurons])
    fb0 = np.array([n._weights_array[1:1 + N_OUT].copy() for n in e.l1.inhibitory_neurons])
    e.set_input([1, 0, 1, 0, 1, 0, 0, 0, 0])   # a few active pixels -> L1E + L2E fire
    for _ in range(120):
        e.step()
    local1 = np.array([n._weights_array[0] for n in e.l1.inhibitory_neurons])
    fb1 = np.array([n._weights_array[1:1 + N_OUT].copy() for n in e.l1.inhibitory_neurons])
    G = e._l1i_G
    assert np.array_equal(local0, local1), "local afferent moved under the predictor"
    assert np.any(np.abs(fb1 - fb0) > 1e-9), "predictor never moved any feedback weight"
    assert np.all(fb1 >= -1e-9) and np.all(fb1 <= G + 1e-9), "feedback left [0, G]"
    print("  predictor moves feedback, local fixed, weights in [0,G], generic off: OK")


def test_predictor_updates_while_refractory():
    """A delivered feedback spike must teach the predictor even if the target L1I is
    refractory. Drive one L1I into refractory, then hand it a delivered feedback
    spike with an active trace and confirm the delivered weight still moves."""
    e = SimulationEngine(seed=1, **LPF)
    i = 4
    inh = e.l1.inhibitory_neurons[i]
    # Put the L1I into refractory so receive_input() will skip its membrane deposit.
    inh.refractory_timer = 2
    e.l1i_trace[i] = 1.0                       # active local trace
    off = e._l1i_fb_offset
    w_before = inh._weights_array[off + 3]
    # Deliver a feedback spike on source 3 to every L1I via the replay override.
    fb = np.zeros(N_OUT); fb[3] = 1.0
    e._feedback_override = fb
    v_before = inh.potential
    e.step()
    e._feedback_override = None
    w_after = inh._weights_array[off + 3]
    assert w_after > w_before, "predictor did not update a delivered synapse while refractory"
    # Membrane deposit WAS blocked by refractory (no charge added from feedback).
    assert inh.potential <= v_before + 1e-9, "refractory L1I should not have integrated charge"
    print("  predictor updates a delivered synapse while the L1I is refractory: OK")


def test_queued_l1i_inhibits_only_paired_l1e():
    e = SimulationEngine(seed=1, **LPF)
    e.set_input([1] * N_PIX)          # every pixel active so every L1E carries charge
    e.step()
    # Force ONLY L1I4 to spike this step (stimulate well above its threshold).
    thr = e.meta['L1I4']['threshold']
    e.stimulate('L1I4', magnitude=thr * 3.0)
    e.step()
    assert e.spiked['L1I4'] and not any(e.spiked[f'L1I{i}'] for i in range(N_PIX) if i != 4), \
        "expected exactly L1I4 to fire"
    # Next step delivers that queued inhibition to paired L1E4 only.
    e.step()
    targets = [nid for nid, _ in e._inh_events]
    assert targets == ['L1E4'], f"queued L1I spike hit {targets}, expected only L1E4"
    ev = [ev for nid, ev in e._inh_events if nid == 'L1E4'][0]
    removed = ev['v_pre'] - ev['v_post']
    gate = e.params['predictive_output_gate_frac'] * e.params['threshold']
    rest = e.l1.excitatory_neurons[4].resting_potential
    # Graded removal: min(gate, v_pre - rest); floored at rest (v_post >= rest).
    assert ev['v_post'] >= rest - 1e-9, "inhibition drove L1E below rest (not floored)"
    assert np.isclose(removed, min(gate, ev['v_pre'] - rest), atol=1e-6), \
        f"removed {removed}, expected graded min(gate,charge)"
    print("  queued L1I spike inhibits ONLY paired L1E, graded + floored at rest: OK")


def main():
    test_predictor_moves_feedback_not_local()
    test_predictor_updates_while_refractory()
    test_queued_l1i_inhibits_only_paired_l1e()
    print("PASS: predictive L1I engine integration (Sections 6/7)")


if __name__ == "__main__":
    main()
