"""Reference feedback tape and deterministic time-shuffle (Experiment.md Section 9).

The reference tape is the complete per-measured-timestep live L2E spike vector
f[t] = [L2E0..L2E7] produced by the local_plus_feedback run (warm-up excluded). The
time_shuffled mode replays a PERMUTATION of complete timestep vectors of that tape
-- never individual bits -- so per-source counts, within-vector coactivity, event
count, and all-zero count are preserved exactly. Acquisition and reversal segments
are permuted separately so no event crosses the reversal boundary.
"""

import numpy as np

from .config import PERM_SEED_OFFSET


class ReferenceTape:
    """Accumulates the live L2E spike vector per measured timestep, grouped by the
    schedule phase (so acquisition/reversal can be shuffled separately)."""

    def __init__(self):
        self._segments = {}          # phase -> list of (N_OUT,) vectors

    def record(self, phase, actual_l2e):
        self._segments.setdefault(phase, []).append(
            np.asarray(actual_l2e, dtype=float).copy())

    def arrays(self):
        """{phase: (T, N_OUT) float array} in first-seen phase order."""
        return {ph: np.array(v) for ph, v in self._segments.items()}


def _derange(T, rng, max_tries=100):
    """Return (perm, method). For T>1, up to 100 seeded permutations, accepting the
    first with no fixed point (perm[i] != i for all i); otherwise a deterministic
    cyclic shift by max(1, floor(T/3)) (a rotation, hence a derangement for T>1).
    T<=1 cannot be deranged and returns identity with an explicit method label."""
    idx = np.arange(T)
    if T <= 1:
        return idx, "identity_singleton"
    for attempt in range(max_tries):
        perm = rng.permutation(T)
        if np.all(perm != idx):
            return perm, f"derangement_try{attempt + 1}"
    shift = max(1, T // 3)
    return (idx + shift) % T, f"cyclic_shift_{shift}"


def permute_tape(tape_arrays, seed):
    """Derange each phase segment with a dedicated default_rng(seed + 10000).

    Returns {phase: dict(shuffled, perm, method)}. `shuffled[k] = segment[perm[k]]`,
    a permutation of complete timestep vectors, so column sums (per-source counts)
    and the all-zero-vector count are invariant."""
    rng = np.random.default_rng(seed + PERM_SEED_OFFSET)
    out = {}
    for phase, arr in tape_arrays.items():
        T = len(arr)
        perm, method = _derange(T, rng)
        out[phase] = dict(shuffled=arr[perm] if T else arr.copy(),
                          perm=perm, method=method)
    return out


def tape_signature(arr):
    """Per-source counts and all-zero-vector count of a (T, N_OUT) tape -- the
    invariants a valid vector permutation must preserve."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return dict(per_source=[0] * 0, n_all_zero=0, n_events=0, T=0)
    per_source = arr.sum(axis=0).astype(int).tolist()
    n_all_zero = int((arr.sum(axis=1) == 0).sum())
    n_events = int(arr.sum())
    return dict(per_source=per_source, n_all_zero=n_all_zero,
                n_events=n_events, T=int(len(arr)))
