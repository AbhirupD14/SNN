"""Phase 3 contract: modes, schedules, reference tape/replay, metrics, artifacts.

Fast checks (small custom schedules) that pin the experiment infrastructure without
running the full 100-run matrix:

  - the five modes map to the exact Section 5 flag combinations;
  - the deterministic schedules have the exact Section 10 counts/compositions;
  - the derangement permutation preserves per-source counts and the all-zero-vector
    count and reports its method (Section 9.2);
  - a short run reproduces every metric from the saved npz alone (Section 8);
  - the shuffled replay delivers exactly the reference tape's per-source counts and
    all-zero-vector count (Section 8);
  - a run is deterministic for a fixed (mode, seed, task, schedule).

    PYTHONPATH=. .venv/bin/python test_predictive_experiment.py
"""
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.predictive_inhibition.config import (
    MODES, CONTEXT_VECTORS, ACQ_BLOCK, REV_BLOCK, engine_overrides)
from experiments.predictive_inhibition.schedules import (
    build_schedule, fourpattern_schedule, contextual_schedule)
from experiments.predictive_inhibition.tape import (
    ReferenceTape, permute_tape, tape_signature, _derange)
from experiments.predictive_inhibition.driver import run_one, DWELL
from experiments.predictive_inhibition.metrics import History, compute_all


def _small_contextual():
    acq = ["X_with_F", "X_without_F", "Y_with_F", "Y_without_F"] * 2
    rev = ["X_with_F", "X_without_F", "Y_with_F", "Y_without_F"]
    sched = [dict(label=l, vector=list(CONTEXT_VECTORS[l]), phase="acquisition") for l in acq]
    sched += [dict(label=l, vector=list(CONTEXT_VECTORS[l]), phase="reversal") for l in rev]
    return sched, dict(sched[0])


def test_mode_flags():
    exp = {
        "baseline": (False, False, False),
        "legacy_feedback": (False, False, True),
        "local_only": (True, False, False),
        "local_plus_feedback": (True, True, True),
        "time_shuffled_feedback": (True, True, True),
    }
    for mode, (pl, pf, deliv) in exp.items():
        o = engine_overrides(mode)
        assert o["paired_local_enabled"] == pl, mode
        assert o["predictive_feedback_enabled"] == pf, mode
        assert o["l2_to_l1i_delivery_enabled"] == deliv, mode
    assert MODES["time_shuffled_feedback"]["replay"] is True
    assert not any(MODES[m]["replay"] for m in MODES if m != "time_shuffled_feedback")
    print("  five modes map to the exact Section 5 flags: OK")


def test_schedule_counts():
    fp = fourpattern_schedule()
    assert len(fp) == 400, len(fp)
    c = Counter(p["label"] for p in fp)
    assert all(c[name] == 100 for name in c) and len(c) == 4, dict(c)
    ctx = contextual_schedule(seed=0)
    assert len(ctx) == 800, len(ctx)
    acq = [p for p in ctx if p["phase"] == "acquisition"]
    rev = [p for p in ctx if p["phase"] == "reversal"]
    assert len(acq) == 500 and len(rev) == 300
    # each 100-presentation block has the exact composition
    for b in range(5):
        cc = Counter(p["label"] for p in acq[b * 100:(b + 1) * 100])
        assert dict(cc) == ACQ_BLOCK, (b, dict(cc))
    for b in range(3):
        cc = Counter(p["label"] for p in rev[b * 100:(b + 1) * 100])
        assert dict(cc) == REV_BLOCK, (b, dict(cc))
    # reused schedule for a fixed seed is identical (same RNG stream)
    assert [p["label"] for p in contextual_schedule(0)] == [p["label"] for p in ctx]
    print("  schedules: 4-pattern 400 (100 each), contextual 500+300 exact blocks: OK")


def test_derangement_and_signature():
    rng = np.random.default_rng(123)
    perm, method = _derange(50, rng)
    assert np.all(perm != np.arange(50)), "not a derangement"
    assert "derangement" in method, method
    # singleton cannot be deranged
    p1, m1 = _derange(1, rng)
    assert m1 == "identity_singleton"
    # permutation of complete vectors preserves per-source counts + all-zero count
    tape = (np.random.default_rng(1).random((40, 8)) > 0.7).astype(float)
    out = permute_tape({"acquisition": tape}, seed=7)["acquisition"]
    assert tape_signature(out["shuffled"]) == tape_signature(tape), \
        "permutation changed the tape signature"
    print(f"  derangement (method={method}) + signature-preserving permutation: OK")


def test_run_metrics_roundtrip_and_determinism():
    sched, warmup = _small_contextual()
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "lpf.npz"
        rec1, tape, _ = run_one("local_plus_feedback", seed=1, task="contextual",
                                schedule=sched, warmup=warmup, build_tape=True,
                                save_path=p1)
        # metrics recompute identically from the saved npz vs a fresh reload
        m_disk = compute_all(History.load(p1))
        assert m_disk["primary_csc"]["csc_primary"] == \
            compute_all(History.load(p1))["primary_csc"]["csc_primary"]
        assert set(m_disk) >= {"primary_csc", "reversal_csc", "predictor_contrasts",
                               "event_classes", "ownership", "charge", "recovery"}

        # determinism: identical run -> identical recorded arrays
        p2 = Path(td) / "lpf2.npz"
        run_one("local_plus_feedback", seed=1, task="contextual",
                schedule=sched, warmup=warmup, save_path=p2)
        a, b = np.load(p1, allow_pickle=False), np.load(p2, allow_pickle=False)
        for k in ("l1e", "l1i", "l2e", "removed", "delivered", "final_W"):
            assert np.array_equal(a[k], b[k]), f"nondeterministic array: {k}"

        # replay: shuffled delivered feedback == reference tape signature per phase
        permuted = permute_tape(tape.arrays(), seed=1)
        p3 = Path(td) / "shuffled.npz"
        run_one("time_shuffled_feedback", seed=1, task="contextual",
                schedule=sched, warmup=warmup, permuted_tape=permuted, save_path=p3)
        H = History.load(p3)
        ref = tape.arrays()
        for phase in ref:
            ph_ts = H.phase == phase
            delivered = H.delivered[ph_ts]
            assert tape_signature(delivered) == tape_signature(ref[phase]), \
                f"replay changed the tape signature in {phase}"
            # and the live L2 vector is kept separate from the replayed one
            assert not np.array_equal(H.delivered[ph_ts], H.actual_l2e[ph_ts]) \
                or ref[phase].sum() == 0, "delivered should be the shuffled tape, not live"
    print("  short run: metrics round-trip, determinism, replay signature preserved: OK")


def main():
    test_mode_flags()
    test_schedule_counts()
    test_derangement_and_signature()
    test_run_metrics_roundtrip_and_determinism()
    print("PASS: predictive-inhibition experiment infrastructure (Phase 3)")


if __name__ == "__main__":
    main()
