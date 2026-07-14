"""Bit-exact equivalence gate for low-level neuron behavior.

    PYTHONPATH=. .venv/bin/python tests/golden/test_golden_equiv.py

Regenerates every golden case live and asserts it is BIT-EXACT (`np.array_equal`)
against the committed `golden_baseline.npz`. This is the single gate every refactor
phase must pass: if any equation, default, or op-order drifts, some array here
moves and the test fails with the offending case named.

Plain-script style (matches the repo's other `test_*.py`).
"""
import os
import numpy as np

from golden_cases import collect

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "golden_baseline.npz")


def main():
    if not os.path.exists(BASELINE):
        raise SystemExit(f"missing baseline {BASELINE} -- run gen_golden.py first")
    gold = np.load(BASELINE, allow_pickle=False)
    live = collect()

    gold_keys, live_keys = set(gold.files), set(live.keys())
    if gold_keys != live_keys:
        missing = gold_keys - live_keys
        extra = live_keys - gold_keys
        raise SystemExit(f"key mismatch -- missing: {sorted(missing)} extra: {sorted(extra)}")

    mismatches = []
    for k in sorted(gold_keys):
        g, l = gold[k], np.asarray(live[k])
        if g.shape != l.shape or not np.array_equal(g, l):
            # locate the first differing element for a useful message
            where = ""
            if g.shape == l.shape:
                diff = np.argwhere(g != l)
                if len(diff):
                    idx = tuple(diff[0])
                    where = f" first@{idx}: {g[idx]!r} -> {l[idx]!r}"
            mismatches.append(f"  {k}: shape {g.shape} vs {l.shape}{where}")

    n_cases = len({k.split('__')[0] for k in gold_keys})
    if mismatches:
        print(f"FAIL: {len(mismatches)}/{len(gold_keys)} arrays drifted:")
        print("\n".join(mismatches[:40]))
        raise SystemExit(1)

    print(f"PASS: all {len(gold_keys)} golden arrays across {n_cases} cases are "
          f"bit-exact against the baseline")


if __name__ == "__main__":
    main()
