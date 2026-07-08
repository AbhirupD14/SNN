"""
Demonstration + characterization of the inhibitory-discharge plasticity rule
(Neuron.apply_inhibition), the second, independent learning system.

The rule treats an inhibitory synapse as a finite adaptive suppression gate.
When an inhibitory spike discharges an excitatory neuron:

    V_pre  = V ; V = V - w ; V_post = V          (linear discharge, w = |weight|)
    p      = V_pre / theta                        (how close to firing it was)
    dw     = eta * p * (1 - w / w_max)            (saturating, local, gradient-free)
    w      = w + dw

Two things are shown here:

  DEMO 1 -- Specialization + saturation (the hard assertions).
     A pool of excitatory neurons sits at different distances from threshold when
     a shared inhibitory neuron discharges them. The gate onto the near-winner
     strengthens most, the gate onto a far-from-threshold neuron barely moves,
     and under repeated near-threshold suppression every gate saturates at w_max.
     This is exactly "inhibitory synapses specialize in suppressing near-winners,
     and saturate naturally due to limited synaptic resources."

  DEMO 2 -- In-network run through the flexible CorticalColumn, delivering the
     shared inhibitory neuron's discharge to each E neuron via apply_inhibition.
     Characterization only: we report that gates grew from their init and none
     exceeded w_max, and that the excitatory rule stayed independent.

Every inhibitory event prints the REQUIRED debug outputs:
    V_pre, V_post, theta, p = V_pre/theta, w_before, delta_w, w_after
"""

import numpy as np
from neuron_flexible import Neuron
from cortical_column_flexible import CorticalColumn


def _print_event(tag, ev):
    print(f"    [{tag}] V_pre={ev['v_pre']:+.4f}  V_post={ev['v_post']:+.4f}  "
          f"theta={ev['theta']:.3f}  p=clamp(V_pre/theta)={ev['p']:.4f}  "
          f"w_before={ev['w_before']:.4f}  delta_w={ev['delta_w']:+.5f}  "
          f"w_after={ev['w_after']:.4f}")


def demo_specialization_and_saturation():
    print("=== DEMO 1: near-winner specialization + saturation ===")
    THETA, W_MAX, ETA, W0 = 1.0, 1.0, 0.2, 0.3
    # Three excitatory neurons at different distances from threshold, each with a
    # single inhibitory gate (index 0) from a shared inhibitory neuron.
    labels = ['near (V=0.9)', 'mid  (V=0.6)', 'far  (V=0.2)']
    v_levels = [0.9, 0.6, 0.2]
    pool = []
    for _ in labels:
        n = Neuron(n_inputs=1, threshold=THETA, weight_cap=W_MAX, leak_rate=0.0,
                   refractory_period=0, inhibitory_learning_rate=ETA)
        n.weights = np.array([-W0])
        pool.append(n)

    print("\n-- one shared inhibitory discharge across the pool --")
    first_dw = []
    for lab, v, n in zip(labels, v_levels, pool):
        n.potential = v
        ev = n.apply_inhibition(np.array([1]))[0]
        _print_event(lab, ev)
        first_dw.append(ev['delta_w'])

    # Hard assertion: closer to threshold -> larger gate update.
    assert first_dw[0] > first_dw[1] > first_dw[2] > 0, first_dw
    print("  -> gate update ranks by closeness to firing (near > mid > far). PASS")

    print("\n-- repeated near-threshold suppression drives saturation --")
    n = pool[0]
    for trial in range(1, 61):
        n.potential = 0.95                     # keeps it near threshold every trial
        ev = n.apply_inhibition(np.array([1]))[0]
        if trial in (1, 2, 5, 20, 60):
            _print_event(f'trial {trial:>2}', ev)
    w_final = -float(n.weights[0])
    assert w_final <= W_MAX + 1e-9 and np.isclose(w_final, W_MAX, atol=1e-2), w_final
    print(f"  -> gate converged to w_max={W_MAX} (final |w|={w_final:.4f}), never overshot. PASS\n")


def demo_in_network_column():
    print("=== DEMO 2: in-network run (flexible CorticalColumn) ===")
    rng = np.random.default_rng(0)
    N_E, N_PIX = 4, 6
    THETA, ETA = 0.6, 0.2
    col = CorticalColumn(n_neurons=N_E, threshold=THETA, refractory_period=1,
                         learning_rate=0.03, weight_cap=1.0, leak_rate=0.05)
    col.setup_connectivity(n_feedforward_inputs=N_PIX, n_feedback_inputs=0)
    col.finalize_connections()
    col.set_local_inhibition_weights(-0.25)            # sub-cap gate: room to learn
    col.set_lateral_excitation_weights(THETA)          # any E spike fires shared I
    col.set_feedforward_weights(rng.uniform(0.10, 0.30, size=(N_E, N_PIX)))
    for e in col.excitatory_neurons:
        e.inhibitory_learning_rate = ETA

    init_gates = [-float(e._weights_array[0]) for e in col.excitatory_neurons]
    init_ff = col.feedforward_weights().copy()

    patterns = [np.array([1, 1, 1, 0, 0, 0], float),
                np.array([0, 0, 0, 1, 1, 1], float)]

    printed = 0
    i_prev = 0.0                                        # shared-I spike, delayed one step
    for step in range(400):
        pat = patterns[(step // 20) % len(patterns)]
        # Excitatory feedforward drive only (index 0 slot held at 0 here).
        ff = np.concatenate(([0.0], pat))
        for e in col.excitatory_neurons:
            e.receive_input(ff)
        # Inhibitory discharge from the PREVIOUS step's shared-I spike -> plasticity.
        if i_prev > 0.5:
            for j, e in enumerate(col.excitatory_neurons):
                evs = e.apply_inhibition(np.array([1.0] + [0.0] * N_PIX))
                if evs and printed < 4:
                    _print_event(f'E{j}', evs[0]); printed += 1
        # WTA-free firing: every E that crossed threshold fires (excitatory rule).
        spk = np.array([1.0 if e.check_threshold() else 0.0 for e in col.excitatory_neurons])
        for j, e in enumerate(col.excitatory_neurons):
            if spk[j]:
                e.fire()
        # Shared inhibitory neuron integrates E spikes, may fire this step.
        col.inhibitory_neuron.receive_input(spk)
        i_now = 1.0 if col.inhibitory_neuron.check_threshold() else 0.0
        if i_now:
            col.inhibitory_neuron.fire()
        for e in col.excitatory_neurons:
            e.update()
        col.inhibitory_neuron.update()
        i_prev = i_now

    final_gates = [-float(e._weights_array[0]) for e in col.excitatory_neurons]
    print(f"\n  inhibitory gates  init  -> final : "
          f"{np.round(init_gates, 3)} -> {np.round(final_gates, 3)}")
    assert all(f <= 1.0 + 1e-9 for f in final_gates), "a gate exceeded w_max"
    assert any(f > i + 1e-6 for f, i in zip(final_gates, init_gates)), \
        "no inhibitory gate strengthened during the run"
    # Excitatory rule stayed alive and independent: some feedforward weight moved.
    assert np.any(np.abs(col.feedforward_weights() - init_ff) > 1e-6), \
        "excitatory plasticity did not run"
    print("  -> gates strengthened within [0, w_max]; excitatory plasticity ran independently. PASS\n")


if __name__ == "__main__":
    demo_specialization_and_saturation()
    demo_in_network_column()
    print("ALL INHIBITORY-PLASTICITY DEMOS PASSED")
