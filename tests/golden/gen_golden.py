"""Generate the golden behavioral baseline `.npz`.

    PYTHONPATH=. .venv/bin/python tests/golden/gen_golden.py

Run this ONCE on known-good code to freeze the behavioral contract, commit the
resulting `golden_baseline.npz`, and thereafter let `test_golden_equiv.py` gate
every refactor phase against it. Re-run (and re-commit) only on an INTENTIONAL,
reviewed behavior change -- never to "make the test pass".
"""
import os
import numpy as np

from golden_cases import collect

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "golden_baseline.npz")

if __name__ == "__main__":
    data = collect()
    np.savez(BASELINE, **data)
    print(f"wrote {BASELINE}")
    print(f"  {len(data)} arrays across "
          f"{len({k.split('__')[0] for k in data})} cases")
