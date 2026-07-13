"""
Tests for the task-independent, balanced L2E feedforward initialization
(balanced_feedforward_init / _sinkhorn_balance in backend.simulation).

The balanced init replaces the legacy Uniform(50,200) with a doubly-balanced
Sinkhorn-Knopp matrix: narrow jitter Z[j,i] ~ Uniform(1-eps, 1+eps), alternately
row/column normalized until every L2E has equal total incoming weight and every
input pixel has equal total outgoing weight, then scaled to the legacy mean (125).
It is a FAIR developmental start -- no neuron or pixel privileged, no task structure
(never inspects PATTERNS, never privileges the center pixel, encodes no templates).

This file pins:
  * equal row and column sums (doubly balanced);
  * positivity and the expected mean (125);
  * deterministic reproduction from the same seed;
  * different results across seeds;
  * invariance (equivariance) under neuron/input permutations;
  * no dependency on PATTERNS;
  * the five init conditions: exactly-identical, balanced-2%, balanced-5%,
    adversarial near-tied, and legacy-wide.

    PYTHONPATH=. .venv/bin/python test_balanced_init.py
"""

from __future__ import annotations

import numpy as np

import backend.simulation as sim
from backend.simulation import (
    SimulationEngine, balanced_feedforward_init, legacy_wide_feedforward_init,
    _sinkhorn_balance, L2E_INIT_MEAN, N_OUT, N_PIX,
)

M, N = N_OUT, N_PIX                 # 8 L2E neurons x 9 input pixels
COL_TARGET = M / N


def _rng(seed=0):
    return np.random.default_rng(seed)


def _ff_matrix(engine):
    """The (N_OUT, N_PIX) feedforward block: afferents 1..N_PIX of each L2E."""
    return np.array([engine.l2.excitatory_neurons[j]._weights_array[1:1 + N]
                     for j in range(M)])


# ---------------------------------------------------------------------------
# 1. Equal row and column sums (doubly balanced)
# ---------------------------------------------------------------------------
def test_equal_row_and_column_sums():
    for jitter in (0.0, 0.02, 0.05, 0.1):
        B = balanced_feedforward_init(_rng(1), M, N, jitter=jitter)
        rs, cs = B.sum(axis=1), B.sum(axis=0)
        assert np.allclose(rs, rs[0], rtol=1e-9), f"row sums not equal (jitter={jitter}): {rs}"
        assert np.allclose(cs, cs[0], rtol=1e-9), f"col sums not equal (jitter={jitter}): {cs}"
        # Balanced totals: each row = mean*N, each col = mean*M.
        assert np.allclose(rs, L2E_INIT_MEAN * N), rs[0]
        assert np.allclose(cs, L2E_INIT_MEAN * M), cs[0]
    print("PASS: every L2E has equal incoming sum and every pixel equal outgoing sum")


# ---------------------------------------------------------------------------
# 2. Positivity and expected mean
# ---------------------------------------------------------------------------
def test_positivity_and_mean():
    for jitter in (0.0, 0.02, 0.05, 0.1):
        B = balanced_feedforward_init(_rng(2), M, N, jitter=jitter)
        assert (B > 0).all(), f"non-positive entries at jitter={jitter}"
        assert abs(B.mean() - L2E_INIT_MEAN) < 1e-9, f"mean {B.mean()} != {L2E_INIT_MEAN}"
    print(f"PASS: init is strictly positive with mean == {L2E_INIT_MEAN}")


# ---------------------------------------------------------------------------
# 3. Deterministic reproduction from the same seed
# ---------------------------------------------------------------------------
def test_deterministic_same_seed():
    a = balanced_feedforward_init(_rng(123), M, N)
    b = balanced_feedforward_init(_rng(123), M, N)
    assert np.array_equal(a, b), "same seed produced different matrices"
    # Same, end-to-end through the engine's seeded feedforward stream.
    e1 = SimulationEngine(seed=5, l2e_init_mode='balanced')
    e2 = SimulationEngine(seed=5, l2e_init_mode='balanced')
    assert np.array_equal(_ff_matrix(e1), _ff_matrix(e2)), "engine ff init not reproducible"
    print("PASS: identical output for the same seed (function and engine)")


# ---------------------------------------------------------------------------
# 4. Different results across seeds
# ---------------------------------------------------------------------------
def test_differs_across_seeds():
    mats = [balanced_feedforward_init(_rng(s), M, N) for s in range(6)]
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            assert not np.allclose(mats[i], mats[j]), f"seeds {i},{j} gave identical init"
    print("PASS: different seeds produce different (but equally balanced) inits")


# ---------------------------------------------------------------------------
# 5. Invariance (equivariance) under neuron / input permutations
# ---------------------------------------------------------------------------
def test_permutation_equivariance():
    """Permuting the jitter matrix's rows (neurons) or columns (pixels) permutes the
    balanced result identically -- the algorithm privileges no neuron or pixel."""
    Z = _rng(9).uniform(0.9, 1.1, size=(M, N))
    P = _rng(11).permutation(M)     # neuron permutation
    Q = _rng(12).permutation(N)     # input permutation
    base = _sinkhorn_balance(Z, COL_TARGET)
    permd = _sinkhorn_balance(Z[np.ix_(P, Q)], COL_TARGET)
    assert np.allclose(permd, base[np.ix_(P, Q)]), "Sinkhorn balance is not permutation-equivariant"
    # Row perm only, and column perm only.
    assert np.allclose(_sinkhorn_balance(Z[P], COL_TARGET), base[P]), "row-permutation not equivariant"
    assert np.allclose(_sinkhorn_balance(Z[:, Q], COL_TARGET), base[:, Q]), "col-permutation not equivariant"
    print("PASS: balancing is equivariant under neuron and input permutations")


# ---------------------------------------------------------------------------
# 6. No dependency on PATTERNS (task structure)
# ---------------------------------------------------------------------------
def test_no_pattern_dependency():
    """Monkeypatch PATTERNS to a totally different task; the seeded feedforward init
    must be byte-identical -- it depends only on (seed, N_OUT, N_PIX, jitter)."""
    e_ref = SimulationEngine(seed=3, l2e_init_mode='balanced')
    ref = _ff_matrix(e_ref)
    original = sim.PATTERNS
    try:
        # A different 'task': shuffled/negated pixel templates, different names.
        sim.PATTERNS = {
            'blob a': [1, 1, 0, 1, 1, 0, 0, 0, 0],
            'blob b': [0, 0, 1, 0, 0, 1, 0, 1, 1],
            'edge':   [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'dot':    [0, 0, 0, 0, 1, 0, 0, 0, 0],
        }
        e_alt = SimulationEngine(seed=3, l2e_init_mode='balanced')
        alt = _ff_matrix(e_alt)
    finally:
        sim.PATTERNS = original
    assert np.array_equal(ref, alt), "feedforward init changed when PATTERNS changed -- task leakage!"
    print("PASS: feedforward init is independent of PATTERNS (no task structure)")


def test_center_pixel_not_privileged():
    """No pixel (including the center, index 4, shared by all four line patterns) has a
    systematically different outgoing weight -- all column sums are equal."""
    B = balanced_feedforward_init(_rng(4), M, N, jitter=0.05)
    cs = B.sum(axis=0)
    assert np.allclose(cs, cs[0]), f"a pixel is privileged: column sums {cs}"
    assert abs(cs[4] - cs.mean()) < 1e-9, "center pixel (4) is privileged"
    print("PASS: the center pixel is not privileged (equal outgoing weight)")


# ---------------------------------------------------------------------------
# 7. The five initialization conditions
# ---------------------------------------------------------------------------
def _conditions():
    """Return (name, matrix, kind) for each of the five required init conditions."""
    return [
        ("exactly-identical", balanced_feedforward_init(_rng(1), M, N, jitter=0.0), "exact"),
        ("balanced-2%",       balanced_feedforward_init(_rng(1), M, N, jitter=0.02), "balanced"),
        ("balanced-5%",       balanced_feedforward_init(_rng(1), M, N, jitter=0.05), "balanced"),
        # Adversarial near-tied: jitter so small the rows are nearly identical -- the
        # worst case for competition to break, but still a valid balanced init.
        ("adversarial-near-tied", balanced_feedforward_init(_rng(1), M, N, jitter=1e-4), "balanced"),
        ("legacy-wide",       legacy_wide_feedforward_init(_rng(1), M, N), "legacy"),
    ]


def test_five_conditions():
    for name, B, kind in _conditions():
        assert B.shape == (M, N), f"{name}: wrong shape {B.shape}"
        assert (B > 0).all(), f"{name}: non-positive entries"
        if kind == "exact":
            # jitter=0 -> perfectly symmetric: every entry identical == the mean.
            assert np.allclose(B, L2E_INIT_MEAN), f"{name}: not exactly uniform"
            assert np.ptp(B) < 1e-9, f"{name}: has spread {np.ptp(B)} (should be 0)"
        elif kind == "balanced":
            rs, cs = B.sum(1), B.sum(0)
            assert np.allclose(rs, rs[0]) and np.allclose(cs, cs[0]), f"{name}: not doubly balanced"
            assert abs(B.mean() - L2E_INIT_MEAN) < 1e-9, f"{name}: wrong mean"
            if name == "adversarial-near-tied":
                # Near-tied but NOT exactly tied (tiny unbiased spread remains).
                assert 0.0 < np.ptp(B) < 1.0, f"{name}: expected a tiny nonzero spread, got {np.ptp(B)}"
        elif kind == "legacy":
            assert B.min() >= 50 and B.max() <= 200, f"{name}: outside [50,200]"
            # Legacy is unbalanced: row/col sums generally NOT equal.
            assert not np.allclose(B.sum(1), B.sum(1)[0]), f"{name}: unexpectedly balanced"
    print("PASS: exact, balanced-2%, balanced-5%, adversarial near-tied, and legacy-wide "
          "conditions all valid and distinct")


def test_engine_legacy_mode_matches_function():
    """The engine in legacy mode reproduces the wide-uniform ablation from its own
    seeded feedforward stream; explicit balanced mode differs from it."""
    e_leg = SimulationEngine(seed=8)
    Mleg = _ff_matrix(e_leg)
    assert Mleg.min() >= 50 and Mleg.max() <= 200, "engine legacy init outside [50,200]"
    assert e_leg.params['l2e_init_mode'] == 'legacy_wide'
    e_bal = SimulationEngine(seed=8, l2e_init_mode='balanced')
    assert not np.allclose(_ff_matrix(e_bal), Mleg), "balanced == legacy (mode had no effect)"
    print("PASS: engine defaults to legacy_wide, distinct from opt-in balanced mode")


def main():
    test_equal_row_and_column_sums()
    test_positivity_and_mean()
    test_deterministic_same_seed()
    test_differs_across_seeds()
    test_permutation_equivariance()
    test_no_pattern_dependency()
    test_center_pixel_not_privileged()
    test_five_conditions()
    test_engine_legacy_mode_matches_function()
    print("\nALL BALANCED-INIT TESTS PASSED")


if __name__ == "__main__":
    main()
