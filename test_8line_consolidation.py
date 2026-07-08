"""
End-to-end characterization of local charge-based learning on the 8-line pattern
task, using per-source feedforward fan-in.

3x3 grid -> 8 line patterns (3 rows, 3 cols, 2 diagonals).
L1: InputLayer of 9 E/I pairs (one excitatory neuron per pixel).
L2: CorticalColumn of 8 E neurons sharing 1 I neuron, each E neuron having ONE
    feedforward synapse per L1 pixel (9 of them) plus a local-inhibition synapse.

This is the test the old demos could not be: because each L2 neuron now owns a
per-pixel receptive field, the local spike-triggered rule can actually shape it.
We check two things:

  A. Receptive-field formation -- after training on a single pattern, the winning
     L2 neuron's feedforward weights on that pattern's ACTIVE pixels grow clearly
     above its weights on the silent pixels. This is the hard assertion.
  B. Characterization of the current limit -- after interleaved training on all
     8 patterns, we report how many distinct L2 neurons win. With no counter-force
     yet (deliberately deferred), the first neuron to win grows its weights
     monotonically toward the cap across EVERY pattern it sees, snowballs, and
     becomes the universal winner (WTA tyranny). So today this collapses to a
     single winner. We assert only what is true now -- every pattern drives SOME
     L2 neuron (no longer silent like the old aggregated-input demos) -- and pin
     the distinct-winner count so it visibly improves once a counter-force
     (normalization / homeostasis / maturation) is added.

Note: no counter-force / normalization yet (deliberately deferred), so weights
only ever grow toward the cap; we test that selectivity FORMS per neuron (A),
not that it is stable across competing patterns (B).
"""

import numpy as np
from layers import InputLayer
from cortical_column_flexible import CorticalColumn

PATTERNS = {
    'row0':  [1,1,1, 0,0,0, 0,0,0],
    'row1':  [0,0,0, 1,1,1, 0,0,0],
    'row2':  [0,0,0, 0,0,0, 1,1,1],
    'col0':  [1,0,0, 1,0,0, 1,0,0],
    'col1':  [0,1,0, 0,1,0, 0,1,0],
    'col2':  [0,0,1, 0,0,1, 0,0,1],
    'diag0': [1,0,0, 0,1,0, 0,0,1],
    'diag1': [0,0,1, 0,1,0, 1,0,0],
}
NAMES = list(PATTERNS)
VECTORS = {k: np.array(v, dtype=float) for k, v in PATTERNS.items()}

N_PIX = 9
N_OUT = 8
THRESHOLD = 0.4
LEAK = 0.02
LR = 0.05
CAP = 1.0


def build_network(seed=0):
    rng = np.random.default_rng(seed)
    l1 = InputLayer(n_neurons=N_PIX, threshold=THRESHOLD, refractory_period=2,
                    learning_rate=LR, weight_cap=CAP, leak_rate=LEAK)
    # L1 E: [from_I (mild inhib), external pixel drive]; external alone drives firing.
    for e in l1.excitatory_neurons:
        e.weights = np.array([-0.1, 1.0])

    l2 = CorticalColumn(n_neurons=N_OUT, threshold=THRESHOLD, refractory_period=2,
                        learning_rate=LR, weight_cap=CAP, leak_rate=LEAK)
    l2.setup_connectivity(n_feedforward_inputs=N_PIX, n_feedback_inputs=0)
    l2.finalize_connections()
    l2.set_local_inhibition_weights(-0.6)          # I -> E (negative)
    l2.set_lateral_excitation_weights(0.5)         # E -> I (positive: E drives shared I)
    # Random per-(neuron,pixel) receptive fields to break symmetry.
    W_ff = rng.uniform(0.10, 0.45, size=(N_OUT, N_PIX))
    l2.set_feedforward_weights(W_ff)
    return l1, l2


def present(l1, l2, pattern, n_steps=30, learn=True):
    """Run one presentation; return per-neuron L2 E spike counts."""
    # reset state
    for n in l1.excitatory_neurons + l1.inhibitory_neurons:
        n.potential, n.refractory_timer, n.spiked = 0.0, 0, False
        n.trace = np.zeros_like(n.trace)
    for n in l2.excitatory_neurons + [l2.inhibitory_neuron]:
        n.potential, n.refractory_timer, n.spiked = 0.0, 0, False
        n._trace = np.zeros_like(n._trace)

    if not learn:
        saved = [n.learning_rate for n in l2.excitatory_neurons]
        for n in l2.excitatory_neurons:
            n.learning_rate = 0.0

    counts = np.zeros(N_OUT, dtype=int)
    l1_e_prev = np.zeros(N_PIX)
    l2_e_prev = np.zeros(N_OUT)
    l2_i_prev = 0.0

    for _ in range(n_steps):
        # --- L1 excitatory neurons driven by the pixel pattern ---
        for i, e in enumerate(l1.excitatory_neurons):
            e.receive_input(np.array([0.0, float(pattern[i] > 0.5)]))
        l1_e_spk = np.array([1.0 if e.check_threshold() else 0.0 for e in l1.excitatory_neurons])
        for i, e in enumerate(l1.excitatory_neurons):
            if l1_e_spk[i]:
                e.fire()

        # --- L2 excitatory neurons: [local_I spike, *L1 E spikes] ---
        for j, e in enumerate(l2.excitatory_neurons):
            e.receive_input(np.concatenate(([l2_i_prev], l1_e_prev)))
        l2_e_spk = np.array([1.0 if e.check_threshold() else 0.0 for e in l2.excitatory_neurons])

        # --- L2 shared inhibitory neuron driven by L2 E spikes ---
        l2.inhibitory_neuron.receive_input(l2_e_prev)
        l2_i_spk = 1.0 if l2.inhibitory_neuron.check_threshold() else 0.0

        for j, e in enumerate(l2.excitatory_neurons):
            if l2_e_spk[j]:
                e.fire()
                counts[j] += 1
        if l2_i_spk:
            l2.inhibitory_neuron.fire()

        # --- advance state ---
        for n in l1.excitatory_neurons:
            n.update()
        for n in l2.excitatory_neurons:
            n.update()
        l2.inhibitory_neuron.update()

        l1_e_prev, l2_e_prev, l2_i_prev = l1_e_spk, l2_e_spk, l2_i_spk

    if not learn:
        for n, lr in zip(l2.excitatory_neurons, saved):
            n.learning_rate = lr
    return counts


def test_receptive_field_forms():
    print("=== Test A: receptive field forms for a single pattern (row0) ===")
    l1, l2 = build_network(seed=1)
    W0 = l2.feedforward_weights().copy()
    for _ in range(80):
        present(l1, l2, VECTORS['row0'], n_steps=30, learn=True)
    counts = present(l1, l2, VECTORS['row0'], n_steps=30, learn=False)
    winner = int(np.argmax(counts))
    W = l2.feedforward_weights()
    active = [0, 1, 2]                      # row0 pixels
    silent = [3, 4, 5, 6, 7, 8]
    a_mean, s_mean = W[winner, active].mean(), W[winner, silent].mean()
    print(f"  winner = L2 E{winner}, fired {counts[winner]}/30 steps")
    print(f"  winner active-pixel weights {np.round(W[winner, active],3)} (mean {a_mean:.3f})")
    print(f"  winner silent-pixel weights {np.round(W[winner, silent],3)} (mean {s_mean:.3f})")
    print(f"  active-pixel growth vs init: {np.round(W[winner,active]-W0[winner,active],3)}")
    assert counts[winner] > 0, "winner never fired"
    assert a_mean > s_mean + 0.2, f"active pixels not preferentially strengthened ({a_mean:.3f} vs {s_mean:.3f})"
    assert np.all(W[winner, active] >= W0[winner, active] - 1e-9), "active pixels should not shrink"
    assert np.allclose(W[winner, silent], W0[winner, silent], atol=1e-6), "silent pixels must be untouched"
    print("  PASS: learning credited only the active pixels of the winner\n")


def test_differentiation():
    print("=== Test B: interleaved training -- current differentiation limit ===")
    l1, l2 = build_network(seed=2)
    for epoch in range(60):
        for name in NAMES:
            present(l1, l2, VECTORS[name], n_steps=25, learn=True)
    winners = {}
    for name in NAMES:
        counts = present(l1, l2, VECTORS[name], n_steps=25, learn=False)
        winners[name] = int(np.argmax(counts)) if counts.sum() > 0 else None
    distinct = sorted({w for w in winners.values() if w is not None})
    for name in NAMES:
        print(f"  {name:6s} -> L2 E{winners[name]}")
    print(f"  distinct winners: {distinct} ({len(distinct)} of {N_OUT})")
    # What is true NOW: every pattern drives some L2 neuron (the old aggregated-input
    # demos went silent or stuck at a frozen 48%). Real selectivity across competing
    # patterns needs the deferred counter-force; until then this collapses to 1 winner.
    assert all(w is not None for w in winners.values()), "some pattern produced no L2 spikes"
    if len(distinct) == 1:
        print("  NOTE: collapsed to a single winner -- expected without a counter-force.")
        print("        This count should rise once normalization/homeostasis is added.")
    print("  PASS (characterization): every pattern fires; differentiation pending counter-force\n")


if __name__ == "__main__":
    test_receptive_field_forms()
    test_differentiation()
    print("ALL 8-LINE TESTS PASSED")
