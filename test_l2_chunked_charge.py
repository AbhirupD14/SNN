"""
Regression tests for the chunked L2 feedforward charge ablation (l2_charge_chunks).

The feature delivers each outer timestep's L1->L2E feedforward drive in K equal
chunks (weight_ji/K per active synapse) INSIDE a frozen timestep, re-running the
argmax WTA after each chunk and stopping at the first threshold-crosser. The
contract this file pins down:

  * K=1 (the default) is the un-chunked baseline -- byte-identical to leaving the
    parameter unset -- so the feature is genuinely off by default.
  * K>1 actually runs the inner chunk loop (a winner can fire on a later chunk).
  * K changes the competition trajectory (it is not a no-op for K>1).
  * The clock is never advanced inside chunks and the live-tunable path matches
    construction.

    PYTHONPATH=. .venv/bin/python test_l2_chunked_charge.py
"""

from __future__ import annotations

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


def _trajectory(steps=400, seed=1, chunks=None, live=False):
    """Cycle through the patterns and record, per step, the L2E winner (or None)
    and which charge-chunk resolved the competition. Chunking is an INSTANTANEOUS-mode
    feature (flow-rate forces effective K=1), so pin excitatory_flow_rate=False to
    test it regardless of the engine's default accumulation mode."""
    if live:
        e = SimulationEngine(seed=seed, excitatory_flow_rate=False)
        e.apply_config({'l2_charge_chunks': chunks})
    elif chunks is None:
        e = SimulationEngine(seed=seed, excitatory_flow_rate=False)   # param unset -> default K
    else:
        e = SimulationEngine(seed=seed, excitatory_flow_rate=False, l2_charge_chunks=chunks)

    names = list(PATTERNS.keys())
    winners, winner_chunks, timesteps = [], [], []
    for s in range(steps):
        e.set_pattern(names[(s // 8) % len(names)])
        before = e.timestep
        e.step()
        # The outer clock advances by exactly one per step -- never per chunk.
        assert e.timestep == before + 1, (e.timestep, before)
        w = next((j for j in range(N_OUT) if e.spiked[f'L2E{j}']), None)
        winners.append(w)
        winner_chunks.append(e.l2_winner_chunk)
        timesteps.append(e.timestep)
    return dict(engine=e, winners=winners, chunks=winner_chunks, timesteps=timesteps)


def test_k1_is_the_unchunked_default():
    """Explicit K=1 and an engine with the parameter UNSET must be identical, for
    several seeds -- i.e. the feature is off by default and K=1 preserves baseline."""
    for seed in (1, 2, 3, 4):
        default = _trajectory(seed=seed, chunks=None)
        k1 = _trajectory(seed=seed, chunks=1)
        assert default['winners'] == k1['winners'], f"seed {seed}: K=1 != default winners"
        # K=1 collapses to a single chunk, so a winner (when one fires) fires on
        # chunk 0; silent steps record None.
        assert set(c for c in k1['chunks'] if c is not None) <= {0}, \
            f"seed {seed}: K=1 winner_chunk must be 0 or None, got {set(k1['chunks'])}"
    print("PASS: K=1 == parameter-unset default (baseline preserved), winner_chunk in {0, None}")


def test_k1_is_deterministic():
    a = _trajectory(seed=1, chunks=1)
    b = _trajectory(seed=1, chunks=1)
    assert a['winners'] == b['winners'], "K=1 not reproducible across runs"
    print("PASS: K=1 trajectory is deterministic")


def test_kgt1_runs_the_inner_loop():
    """With K>1 a threshold-crosser can emerge before the full drive has arrived,
    so the resolving chunk index is sometimes > 0 -- proof the inner loop runs."""
    traj = _trajectory(seed=1, chunks=8)
    seen = sorted(set(c for c in traj['chunks'] if c is not None))
    assert any(c > 0 for c in seen), f"K=8 never fired on a later chunk (seen chunks {seen})"
    assert max(seen) <= 7, f"winner_chunk out of range for K=8: {seen}"
    print(f"PASS: K=8 runs the inner chunk loop (resolving-chunk indices seen: {seen})")


def test_chunking_changes_competition():
    """Chunking is a real ablation, not a no-op: at least one seed's winner
    sequence differs between K=1 and K=8."""
    differ = False
    for seed in (1, 2, 3, 4):
        if _trajectory(seed=seed, chunks=1)['winners'] != _trajectory(seed=seed, chunks=8)['winners']:
            differ = True
            break
    assert differ, "K=1 and K=8 produced identical winners for every seed"
    print("PASS: K>1 changes the competition trajectory vs K=1")


def test_live_tunable_matches_construction():
    ctor = _trajectory(seed=2, chunks=4)
    live = _trajectory(seed=2, chunks=4, live=True)
    assert ctor['winners'] == live['winners'], "apply_config(K) != constructor(K)"
    assert live['engine'].l2_charge_chunks == 4
    print("PASS: apply_config({'l2_charge_chunks': K}) matches constructor(l2_charge_chunks=K)")


def test_invalid_chunks_clamped():
    """K is clamped to >= 1 (a 0 or negative value must not divide-by-zero)."""
    e = SimulationEngine(seed=1, l2_charge_chunks=0)
    assert e.l2_charge_chunks == 1, e.l2_charge_chunks
    e.set_pattern('row 1')
    for _ in range(20):
        e.step()
    print("PASS: l2_charge_chunks <= 0 is clamped to 1 (no divide-by-zero)")


def main():
    test_k1_is_the_unchunked_default()
    test_k1_is_deterministic()
    test_kgt1_runs_the_inner_loop()
    test_chunking_changes_competition()
    test_live_tunable_matches_construction()
    test_invalid_chunks_clamped()
    print("\nALL L2 CHUNKED-CHARGE TESTS PASSED")


if __name__ == "__main__":
    main()
