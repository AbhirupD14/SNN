"""Preregistered run matrix + durable artifacts (Experiment.md Sections 11, 13).

Five modes x ten seeds x two tasks = 100 primary runs. For each (seed, task) the
local_plus_feedback run is executed first and its live L2E tape captured; the
time_shuffled_feedback run then replays a per-phase derangement of that tape. Every
run writes a complete history npz and a per-run metrics record; the process is
resumable (an existing, loadable npz is skipped) so an interrupted run can continue.

CLI:
    PYTHONPATH=. .venv/bin/python -m experiments.predictive_inhibition.run_matrix \
        [--seeds 0..9] [--tasks contextual fourpattern] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import MODE_NAMES, MODES, SEEDS
from .schedules import build_schedule
from .tape import permute_tape, tape_signature
from .driver import run_one
from .metrics import History, compute_all


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


class MatrixRun:
    def __init__(self, root: Path, seeds, tasks):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"pi_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seeds, self.tasks = seeds, tasks
        self.metrics_fp = open(self.dir / "metrics.jsonl", "a", buffering=1)
        self.events_fp = open(self.dir / "events.jsonl", "a", buffering=1)
        self.total = len(seeds) * len(tasks) * len(MODE_NAMES)
        self.done = 0
        _atomic_write(self.dir / "config.json", json.dumps(
            dict(seeds=seeds, tasks=tasks, modes=MODE_NAMES,
                 total_runs=self.total, started=time.time()), indent=2))
        self._link(root)
        self.status(state="starting")

    def _link(self, root):
        link = root / "pi_current"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.dir.name)
        except OSError:
            pass

    def status(self, **kw):
        rec = dict(dir=self.dir.name, total=self.total, done=self.done,
                   updated=time.time(), **kw)
        _atomic_write(self.dir / "status.json", json.dumps(rec, indent=2))

    def event(self, kind, **data):
        self.events_fp.write(json.dumps(dict(t=time.time(), kind=kind, **data)) + "\n")

    def metric(self, rec):
        self.metrics_fp.write(json.dumps(rec, default=_json_default) + "\n")

    def close(self):
        self.metrics_fp.close(); self.events_fp.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _run_dir(base: Path, task, seed):
    d = base / "runs" / task / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_or_run(mode, seed, task, sched, warmup, npz_path, **kw):
    """Skip if a loadable npz already exists (resume); otherwise run and save."""
    if npz_path.exists():
        try:
            History.load(npz_path)      # validity check
            return None, None, True     # (rec, tape, reused)
        except Exception:
            npz_path.unlink()           # corrupt -> rerun
    rec, tape, _ = run_one(mode, seed, task, sched, warmup,
                           save_path=npz_path, **kw)
    return rec, tape, False


def run_seed_task(run: MatrixRun, seed, task):
    sched, warmup = build_schedule(task, seed)
    base = run.dir
    rd = _run_dir(base, task, seed)
    tape_arrays = None

    # Non-replay modes first (captures the reference tape from local_plus_feedback).
    for mode in [m for m in MODE_NAMES if not MODES[m]["replay"]]:
        npz = rd / f"{mode}.npz"
        build_tape = (mode == "local_plus_feedback")
        try:
            rec, tape, reused = _load_or_run(mode, seed, task, sched, warmup, npz,
                                             build_tape=build_tape)
            if build_tape and not reused:
                tape_arrays = tape.arrays()
            m = compute_all(History.load(npz))   # already carries mode/seed/task
            run.metric({**m, "reused": reused})
            run.done += 1
            run.event("run_done", seed=seed, task=task, mode=mode, reused=reused)
        except Exception:
            run.event("run_failed", seed=seed, task=task, mode=mode,
                      error=traceback.format_exc())
            raise
        run.status(state="running", current=dict(seed=seed, task=task, mode=mode))

    # Reference tape may need regeneration if local_plus_feedback was reused.
    if tape_arrays is None:
        _, tape, _ = run_one("local_plus_feedback", seed, task, sched, warmup,
                             build_tape=True, save_path=None)
        tape_arrays = tape.arrays()
    permuted = permute_tape(tape_arrays, seed)
    _atomic_write(rd / "shuffle.json", json.dumps(
        {ph: dict(method=d["method"], perm=d["perm"].tolist(),
                  reference=tape_signature(tape_arrays[ph]),
                  shuffled=tape_signature(d["shuffled"]))
         for ph, d in permuted.items()}, indent=2, default=_json_default))

    # Replay modes.
    for mode in [m for m in MODE_NAMES if MODES[m]["replay"]]:
        npz = rd / f"{mode}.npz"
        try:
            _load_or_run(mode, seed, task, sched, warmup, npz, permuted_tape=permuted)
            m = compute_all(History.load(npz))   # already carries mode/seed/task
            run.metric(m)
            run.done += 1
            run.event("run_done", seed=seed, task=task, mode=mode)
        except Exception:
            run.event("run_failed", seed=seed, task=task, mode=mode,
                      error=traceback.format_exc())
            raise
        run.status(state="running", current=dict(seed=seed, task=task, mode=mode))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local Predictive Inhibition run matrix")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--tasks", nargs="*", default=["contextual", "fourpattern"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs"))
    args = ap.parse_args(argv)

    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    run = MatrixRun(root, args.seeds, args.tasks)

    def _sig(signum, frame):
        run.status(state="interrupted", signal=signum)
        run.event("interrupted", signal=signum)
        run.close(); sys.exit(1)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    run.event("matrix_start", seeds=args.seeds, tasks=args.tasks)
    for task in args.tasks:
        for seed in args.seeds:
            run_seed_task(run, seed, task)
    run.status(state="completed")
    run.event("matrix_end", done=run.done)
    run.close()
    print(f"[run_matrix] done: {run.done}/{run.total} runs in {run.dir}")
    return str(run.dir)


if __name__ == "__main__":
    main()
