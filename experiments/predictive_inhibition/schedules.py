"""Deterministic stimulus schedules (Experiment.md Section 10).

A schedule is a list of presentation dicts {label, vector, phase}. Vectors are
delivered raw with SimulationEngine.set_input(); labels/phases are used only by the
driver for scheduling and by the metrics for grouping -- never by engine learning.
The one warm-up presentation is a duplicate of the first scheduled presentation and
does not consume it.
"""

import numpy as np

from backend.simulation import PATTERNS
from .config import (CONTEXT_VECTORS, ACQ_BLOCK, REV_BLOCK, N_ACQ_BLOCKS,
                     N_REV_BLOCKS, FOURPATTERN_ORDER, FOURPATTERN_PER_PATTERN,
                     SCHED_SEED_OFFSET)


def fourpattern_schedule():
    """Section 10.1: interleaved row1, col1, diag\\, diag/, repeat -- 100 each,
    400 total. Deterministic (no shuffle); independent of seed."""
    sched = []
    for _ in range(FOURPATTERN_PER_PATTERN):
        for name in FOURPATTERN_ORDER:
            sched.append(dict(label=name, vector=list(PATTERNS[name]),
                              phase="fourpattern"))
    return sched


def _shuffled_block(composition, phase, rng):
    """One 100-presentation block with the exact composition, shuffled by `rng`
    (a continued stream). Uses rng.permutation over indices so the draw is
    deterministic and reproducible."""
    labels = []
    for label, count in composition.items():
        labels.extend([label] * count)
    order = rng.permutation(len(labels))
    return [dict(label=labels[k], vector=list(CONTEXT_VECTORS[labels[k]]), phase=phase)
            for k in order]


def contextual_schedule(seed):
    """Sections 10.2/10.3: five acquisition blocks then three reversal blocks, each
    independently shuffled from ONE continued RNG stream default_rng(seed + 20000).
    Acquisition and reversal share the stream (reversal continues it), and the
    schedule is reused across all modes of that seed."""
    rng = np.random.default_rng(seed + SCHED_SEED_OFFSET)
    sched = []
    for _ in range(N_ACQ_BLOCKS):
        sched.extend(_shuffled_block(ACQ_BLOCK, "acquisition", rng))
    for _ in range(N_REV_BLOCKS):
        sched.extend(_shuffled_block(REV_BLOCK, "reversal", rng))
    return sched


def build_schedule(task, seed):
    """Return (schedule, warmup) for `task` in {"fourpattern", "contextual"}.
    warmup is a duplicate of schedule[0] (does not consume it)."""
    if task == "fourpattern":
        sched = fourpattern_schedule()
    elif task == "contextual":
        sched = contextual_schedule(seed)
    else:
        raise KeyError(f"unknown task {task!r}")
    warmup = dict(sched[0])   # duplicate; not part of the measured schedule or tape
    return sched, warmup
