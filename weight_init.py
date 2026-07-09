"""
Pluggable L2E feedforward weight initialization schemes.

Implements the initialization ablations proposed in
Input_Vector_Initialization_And_Distance_Weighting.md. Each scheme returns an
(n_out, n_pix) matrix of POSITIVE feedforward weights (linear/fixed-point scale,
same units as the legacy `rng.uniform(50, 200)` init). The goal is to reduce
unlucky-seed dependence and duplicate receptive fields WITHOUT assigning labels
or pattern ownership -- so every scheme is label-free and driven only by an rng.

All schemes draw from the rng passed in, so a given (seed, scheme) is
reproducible. `uniform` reproduces the exact legacy init
(`rng.uniform(INIT_LO, INIT_HI, (n_out, n_pix))`) byte-for-byte, so it remains a
clean no-change baseline.

Normalized schemes rescale each neuron's afferent vector to the same total
incoming weight (`_target_sum`, the mean row-sum of the uniform baseline) so
competition depends on vector DIRECTION rather than magnitude. Under the
signed-spike / no-budget regime (the current default) stored weights are free to
move, so unlike the old budget regime this normalization is a genuine initial
condition rather than an enforced steady state -- direction diversity is what the
diversity / orthogonal / low-discrepancy schemes buy.

Schemes: uniform, uniform_normalized, sparse, sparse_normalized, diversity,
orthogonal, low_discrepancy. Select via SimulationEngine(ff_init=..., ff_init_kw=...).
"""

from __future__ import annotations

import numpy as np

# Legacy uniform init range (linear weights; 50..200 == 0.05..0.20 * UNIT).
INIT_LO, INIT_HI = 50.0, 200.0
# Floor for the "off" afferents in the sparse schemes (an order of magnitude
# below INIT_LO, matching L2E_MIN_WEIGHT_FLOOR in simulation.py).
SPARSE_FLOOR = 10.0


def _target_sum(n_pix: int) -> float:
    """Common per-neuron total incoming weight for the normalized schemes: the
    mean row-sum of the uniform baseline, so drive is comparable across schemes."""
    return 0.5 * (INIT_LO + INIT_HI) * n_pix


def _normalize_rows(w: np.ndarray, target_sum: float) -> np.ndarray:
    s = w.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return w * (target_sum / s)


def _row_cosines(cands: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    """(len(cands), len(accepted)) matrix of cosine similarities."""
    cn = cands / np.maximum(np.linalg.norm(cands, axis=1, keepdims=True), 1e-12)
    an = accepted / np.maximum(np.linalg.norm(accepted, axis=1, keepdims=True), 1e-12)
    return cn @ an.T


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------
def uniform(rng, n_out, n_pix):
    """Legacy baseline: independent Uniform(INIT_LO, INIT_HI) per synapse."""
    return rng.uniform(INIT_LO, INIT_HI, size=(n_out, n_pix))


def uniform_normalized(rng, n_out, n_pix):
    """Uniform positive vectors renormalized to a common total incoming weight."""
    return _normalize_rows(uniform(rng, n_out, n_pix), _target_sum(n_pix))


def sparse(rng, n_out, n_pix, k=3, floor=SPARSE_FLOOR):
    """Each neuron gets `k` afferents above floor (Uniform(INIT_LO, INIT_HI)); the
    rest sit at `floor`. Gives distinct initial directions (resembles immature
    sparse connectivity). k defaults to 3 -- the active-pixel count of a line."""
    k = min(k, n_pix)
    w = np.full((n_out, n_pix), float(floor))
    for j in range(n_out):
        idx = rng.choice(n_pix, size=k, replace=False)
        w[j, idx] = rng.uniform(INIT_LO, INIT_HI, size=k)
    return w


def sparse_normalized(rng, n_out, n_pix, k=3, floor=SPARSE_FLOOR):
    """Sparse init renormalized to a common total incoming weight."""
    return _normalize_rows(sparse(rng, n_out, n_pix, k, floor), _target_sum(n_pix))


def diversity(rng, n_out, n_pix, max_similarity=0.8, max_tries=200):
    """Random-normalized with diversity rejection (the plan's strongest near-term
    candidate). For each neuron, resample a normalized positive vector until its
    cosine similarity to every already-accepted vector is <= max_similarity; if no
    sample clears the bar within max_tries, keep the least-similar one seen.
    Label-free: it reduces duplicate receptive fields without assigning ownership."""
    target = _target_sum(n_pix)
    accepted: list[np.ndarray] = []
    for _ in range(n_out):
        best = None
        best_sim = np.inf
        for _ in range(max_tries):
            cand = _normalize_rows(rng.uniform(INIT_LO, INIT_HI, size=(1, n_pix)), target)[0]
            sim = 0.0 if not accepted else float(_row_cosines(cand[None, :], np.array(accepted)).max())
            if sim < best_sim:
                best, best_sim = cand, sim
            if sim <= max_similarity:
                best = cand
                break
        accepted.append(best)
    return np.array(accepted)


def orthogonal(rng, n_out, n_pix, n_candidates=64):
    """Nonnegative near-orthogonal approximation via greedy farthest-point
    selection: each new neuron is the candidate (of `n_candidates` normalized
    positive draws) whose MAXIMUM cosine to the already-accepted set is smallest.
    Positive-only, so it approximates rather than achieves true orthogonality."""
    target = _target_sum(n_pix)
    accepted: list[np.ndarray] = []
    for _ in range(n_out):
        cands = _normalize_rows(rng.uniform(INIT_LO, INIT_HI, size=(n_candidates, n_pix)), target)
        if not accepted:
            accepted.append(cands[0])
            continue
        worst = _row_cosines(cands, np.array(accepted)).max(axis=1)
        accepted.append(cands[int(worst.argmin())])
    return np.array(accepted)


# First n_pix primes for the Halton sequence (one base per input dimension).
_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def _halton(i: int, base: int) -> float:
    """Radical-inverse (van der Corput) value of index i in the given base."""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def low_discrepancy(rng, n_out, n_pix, scramble=True):
    """Halton low-discrepancy coverage of afferent-vector space, mapped into
    [INIT_LO, INIT_HI] and normalized. Covers the space more evenly than i.i.d.
    sampling, reducing seed variance without per-pair rejection. `scramble`
    offsets the sequence by an rng-drawn amount so different seeds differ."""
    offset = int(rng.integers(1, 10_000)) if scramble else 0
    w = np.empty((n_out, n_pix))
    for j in range(n_out):
        for d in range(n_pix):
            base = _HALTON_PRIMES[d % len(_HALTON_PRIMES)]
            w[j, d] = INIT_LO + _halton(j + 1 + offset, base) * (INIT_HI - INIT_LO)
    return _normalize_rows(w, _target_sum(n_pix))


SCHEMES = {
    'uniform': uniform,
    'uniform_normalized': uniform_normalized,
    'sparse': sparse,
    'sparse_normalized': sparse_normalized,
    'diversity': diversity,
    'orthogonal': orthogonal,
    'low_discrepancy': low_discrepancy,
}


def init_feedforward(rng, n_out, n_pix, scheme='uniform', **kw):
    """Dispatch to a named scheme; returns an (n_out, n_pix) positive matrix."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown ff_init scheme {scheme!r}; options: {sorted(SCHEMES)}")
    return SCHEMES[scheme](rng, n_out, n_pix, **kw)
