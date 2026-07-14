"""Sentinel regression: the runner's L2 feedforward helper must include pixel 0.

The active L2E has exactly N_PIX pixel afferents at indices 0..N_PIX-1 (no index-0
inhibitory placeholder). A prior defect sliced `_weights_array[1:1 + N_PIX]`, which
for N_PIX=9 is [1:10] -> indices 1..8: it dropped pixel 0 and returned one column
short. `l2_feedforward_matrix` is the authoritative helper metrics/plots/snapshots
must use. This test stamps nine DISTINCT values [0.11, 0.22, ..., 0.99] onto one
L2E neuron and asserts all nine appear, IN ORDER, in the helper output and in the
plot-data path (imshow matrix) -- pixel 0 included.

Plain-script style (matches the repo's other test_*.py).

    PYTHONPATH=. .venv/bin/python test_runner_pixel_zero.py
"""
import numpy as np

from backend.simulation import SimulationEngine, N_OUT, N_PIX
from backend.dashboard_config import DASHBOARD_OVERRIDES
from experiments.runner import l2_feedforward_matrix


def main():
    assert N_PIX == 9, f"sentinel assumes N_PIX==9, got {N_PIX}"
    sentinel = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99])

    engine = SimulationEngine(seed=1, **DASHBOARD_OVERRIDES)
    # Stamp the sentinel directly onto every L2E afferent vector (bypass weight caps
    # / clipping -- we are testing the READ path, not learning). Each L2E has exactly
    # N_PIX afferents at indices 0..N_PIX-1.
    for j in range(N_OUT):
        n = engine.l2.excitatory_neurons[j]
        assert len(n._weights_array) == N_PIX, (
            f"L2E{j} has {len(n._weights_array)} afferents, expected {N_PIX}")
        n._weights_array = sentinel.copy()

    W = l2_feedforward_matrix(engine)
    assert W.shape == (N_OUT, N_PIX), f"helper shape {W.shape} != {(N_OUT, N_PIX)}"

    # Every row must be the nine distinct sentinel values, IN ORDER, pixel 0 first.
    for j in range(N_OUT):
        assert np.array_equal(W[j], sentinel), (
            f"L2E{j} columns {W[j].tolist()} != sentinel {sentinel.tolist()}")
    # Pixel 0 specifically present in column 0.
    assert np.all(W[:, 0] == sentinel[0]), "pixel 0 (column 0) missing from helper output"

    # Plot-data path uses the SAME helper divided by cap; confirm all nine columns
    # (including pixel 0) survive the plot normalization in order.
    cap = engine.l2.excitatory_neurons[0].weight_cap
    M = l2_feedforward_matrix(engine) / cap
    assert M.shape == (N_OUT, N_PIX)
    for j in range(N_OUT):
        assert np.allclose(M[j], sentinel / cap), f"plot row {j} lost a column"

    print(f"PASS: l2_feedforward_matrix returns all {N_PIX} columns in order "
          f"(pixel 0 included) for all {N_OUT} L2E; plot path preserves them.")


if __name__ == "__main__":
    main()
