"""
scale_run_8line.py
==================

Blocked-learning driver for the Selective Maturation Race on the 8-line task:
the 8 lines of a 3x3 grid -- 3 horizontal rows, 3 vertical columns, and the 2
diagonals (16-d... actually 9-d flattened inputs).

Curriculum
----------
Patterns are presented in strict, non-interleaved chronological *blocks*
(Blocked Learning).  A block runs the decentralized neural physics until either
the pool produces a maturation (the race is won -> advance) or the block budget
is exhausted.  An already-claimed primitive is dropped from the rotation -- the
matured detector owns it, and re-presenting it would only invite a duplicate
hub into the now-vacated competitive landscape.

The *only* thing this script reads back from the network is the emergent
maturation event (to advance the curriculum); it never reaches in to steer
membranes, weights, or learning rates.  All self-organization is local.

The loop runs until a perfect, deterministic 8/8 single-hub map is consolidated:
every primitive owned by exactly one unique, retired neuron.
"""

from __future__ import annotations

import numpy as np

from snn_layer import SelfOrganizingSNNLayer, SNNConfig

GRID = 3  # 3x3 -> 3 rows + 3 cols + 2 diagonals = 8 line primitives, 9-d inputs


# ----------------------------------------------------------------------- #
#  The 8 line primitives                                                    #
# ----------------------------------------------------------------------- #
def build_line_patterns(grid: int = GRID,
                        diagonals: bool = True) -> tuple[np.ndarray, list[str]]:
    """The canonical 8 lines of a 3x3 grid: 3 horizontal rows, 3 vertical
    columns, and the 2 diagonals. Any two distinct lines meet in at most one
    pixel; the centre pixel is shared by 4 lines (row 1, col 1, both diagonals),
    which is the hardest overlap the consolidation must disambiguate."""
    pats, names = [], []
    for r in range(grid):                      # horizontal rows
        g = np.zeros((grid, grid))
        g[r, :] = 1.0
        pats.append(g.flatten())
        names.append(f"ROW_{r}")
    for c in range(grid):                       # vertical columns
        g = np.zeros((grid, grid))
        g[:, c] = 1.0
        pats.append(g.flatten())
        names.append(f"COL_{c}")
    if diagonals:
        g = np.zeros((grid, grid))             # main diagonal "\"
        for k in range(grid):
            g[k, k] = 1.0
        pats.append(g.flatten())
        names.append("DIAG_\\")
        g = np.zeros((grid, grid))             # anti-diagonal "/"
        for k in range(grid):
            g[k, grid - 1 - k] = 1.0
        pats.append(g.flatten())
        names.append("DIAG_/")
    return np.asarray(pats, dtype=np.float64), names


# ----------------------------------------------------------------------- #
#  One blocked presentation of a single primitive                           #
# ----------------------------------------------------------------------- #
def run_block(layer: SelfOrganizingSNNLayer, x: np.ndarray, pid: int) -> int | None:
    layer.reset_transient()
    for _ in range(layer.cfg.max_block_steps):
        layer.step(x)
        winner = layer.try_mature(pid)
        if winner is not None:
            if layer.cfg.sparse_substrate:
                layer.reset_losers(winner)   # melt spent runners-up
            return winner            # race won -> end block early, advance
    return None


# ----------------------------------------------------------------------- #
#  Training loop -> deterministic 8/8 consolidation                         #
# ----------------------------------------------------------------------- #
def train(seed: int = 0, max_epochs: int = 40, verbose: bool = True):
    patterns, names = build_line_patterns()
    n_pat = len(patterns)
    cfg = SNNConfig(input_dim=patterns.shape[1], n_neurons=64, seed=seed)
    layer = SelfOrganizingSNNLayer(cfg)

    claimed: dict[int, int] = {}     # pattern_id -> matured neuron_id

    for epoch in range(max_epochs):
        for pid in range(n_pat):                 # strict chronological order
            if pid in claimed:
                continue                          # primitive already owned
            winner = run_block(layer, patterns[pid], pid)
            if winner is not None:
                claimed[pid] = winner
                if verbose:
                    print(f"  [epoch {epoch:02d}] block {names[pid]:>6} "
                          f"-> neuron #{winner:<2d} matured & retired "
                          f"(mature pool = {int(layer.mature_mask.sum())})")
        if len(claimed) == n_pat:
            if verbose:
                print(f"\nConsolidation reached after epoch {epoch}.")
            break

    return layer, patterns, names, claimed


# ----------------------------------------------------------------------- #
#  Verification of the 1:1 localist map                                     #
# ----------------------------------------------------------------------- #
def verify(layer, patterns, names, claimed) -> bool:
    n_pat = len(patterns)
    print("\n" + "=" * 60)
    print("FINAL SINGLE-HUB CONSOLIDATION MAP")
    print("=" * 60)

    # Response matrix: feedforward drive of every matured neuron to every pat.
    resp = np.stack([layer.response(p) for p in patterns])   # (n_pat, N)
    theta = layer.cfg.theta_rest

    ok = True
    used = set()
    for pid in range(n_pat):
        neuron = claimed.get(pid, None)
        # Which matured neurons actually fire (drive >= theta) on this pattern?
        firing = [i for i in range(layer.cfg.n_neurons)
                  if layer.mature_mask[i] and resp[pid, i] >= theta]
        tag = "OK"
        if neuron is None or neuron not in firing or len(firing) != 1:
            ok = False
            tag = "FAIL"
        if neuron in used:
            ok = False
            tag = "FAIL(dup)"
        used.add(neuron)
        drive = resp[pid, neuron] if neuron is not None else float("nan")
        print(f"  {names[pid]:>6}  ->  neuron #{str(neuron):>3}   "
              f"drive={drive:5.2f}/{theta:.1f}   single-firing={firing}   [{tag}]")

    n_mature = int(layer.mature_mask.sum())
    unique = len(set(claimed.values())) == n_pat and len(claimed) == n_pat
    print("-" * 60)
    print(f"  primitives claimed : {len(claimed)}/{n_pat}")
    print(f"  matured neurons    : {n_mature}")
    print(f"  unique 1:1 mapping : {unique}")
    perfect = ok and unique and n_mature == n_pat and len(claimed) == n_pat
    print(f"\n  RESULT: {'PERFECT 8/8 SINGLE-HUB CONSOLIDATION' if perfect else 'INCOMPLETE'}")
    print("=" * 60)
    return perfect


# ----------------------------------------------------------------------- #
def main():
    print("Selective Maturation Race  --  8-line blocked learning  (N=64)\n")
    layer, patterns, names, claimed = train(seed=0, verbose=True)
    perfect = verify(layer, patterns, names, claimed)
    raise SystemExit(0 if perfect else 1)


if __name__ == "__main__":
    main()
