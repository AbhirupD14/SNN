"""Headless replay recorder and versioned artifact contract for SNN experiments.

This module is *recording infrastructure only*. It imports no FastAPI, WebSocket, or DOM
code and it never touches the engine's neuron state, event ordering, scheduler, learning,
random-number stream, topology, weights, thresholds, inhibition, or input timing. Every
recorder method that observes the engine calls read-only accessors (``topology()``,
``dynamic_state()``, ``_live_weight``); a recorder-on run and a recorder-off run with the
same seed and inputs are therefore behaviorally identical.

It writes one directory per run::

    <output-root>/<run-id>/
    ├── manifest.json          human-readable run metadata (atomic replace)
    ├── replay.snn.jsonl       self-contained, append-only dashboard replay movie
    ├── metrics.csv            flat streaming quantitative analysis artifact
    └── summary.json           caller-supplied scientific outcome (atomic replace)

The JSONL replay is a documented, versioned union of records:

    header            exactly one, first: schema, run metadata, full initial topology
                      (including initial live weights) and stable neuron/synapse ordering.
    marker            zero or more caller-supplied semantic transitions (phase change,
                      recall, timeout, collision, ...). The recorder never guesses phases.
    frame             monotonically increasing (frame_index, timestep). Carries the
                      current annotation and a canonical ``dynamic`` payload compatible
                      with the dashboard consumer, with only the NEW event-log entries.
    weight_checkpoint periodic full snapshot of all weighted synapses keyed by synapse id.
                      Frames between checkpoints carry authoritative ``changed_synapses``.
    result            optional final record mirroring summary.json.

The recorder streams and flushes at a bounded interval and keeps no per-frame history in
memory: reconstruction of weights at an arbitrary frame is done by the pure
:func:`reconstruct_weights_at` reader over a re-read file, not from recorder state.

Scientific note: a replay is an *observation artifact*. It does not, and is not intended
to, prove scientific correctness of any experiment.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
import os
import subprocess
import tempfile
import uuid
from typing import Any, Iterable, Mapping, Optional, Sequence

# ------------------------------------------------------------------ schema ids
REPLAY_SCHEMA_NAME = "snn.replay"
REPLAY_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1

# Recorder record kinds (the JSONL union tags).
REC_HEADER = "header"
REC_MARKER = "marker"
REC_FRAME = "frame"
REC_CHECKPOINT = "weight_checkpoint"
REC_RESULT = "result"

# Documented statuses. ``running`` -> terminal one of the rest.
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
_TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, STATUS_INTERRUPTED)

# Documented vocabulary for the hierarchical-feedback condition. This is *declared* by the
# caller and never inferred from neuron IDs.
FEEDBACK_INTACT = "intact"
FEEDBACK_DISABLED = "disabled"
FEEDBACK_NOT_APPLICABLE = "not_applicable"
_FEEDBACK_VOCAB = (FEEDBACK_INTACT, FEEDBACK_DISABLED, FEEDBACK_NOT_APPLICABLE)


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_run_id(experiment: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if (c.isalnum() or c in "-_") else "-" for c in (experiment or "run"))
    return f"{stamp}-{slug}-{uuid.uuid4().hex[:8]}"


def _reject_nonfinite(obj: Any, _path: str = "$") -> None:
    """Raise ``ValueError`` if *obj* contains NaN/Infinity anywhere.

    ``json.dumps(allow_nan=False)`` already rejects these, but walking first yields a
    precise path in the error and lets callers validate a record before any bytes reach a
    partially written file.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite JSON value at {_path}: {obj!r}")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _reject_nonfinite(v, f"{_path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _reject_nonfinite(v, f"{_path}[{i}]")


def _dumps(obj: Any) -> str:
    """Strict JSON: finite-only, no NaN/Infinity tokens ever written."""
    _reject_nonfinite(obj)
    return json.dumps(obj, allow_nan=False, separators=(",", ":"))


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write pretty JSON via a temp file + ``os.replace`` so an interruption can never
    leave a half-written manifest/summary."""
    _reject_nonfinite(obj)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, allow_nan=False, indent=2, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _git_info(repo_dir: str) -> dict:
    """Best-effort repo commit + dirty flag. Inability to query Git must not fail a run."""
    info: dict = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir,
            capture_output=True, text=True, timeout=5)
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


# ============================================================ metrics CSV writer
class MetricsWriter:
    """Generic streaming CSV writer whose columns are declared once.

    - one header row, written lazily on the first data row;
    - rejects rows with unknown columns, and missing columns that are not declared
      optional (optional columns are written empty);
    - appends without retaining rows in memory;
    - ordinary RFC-4180 quoting via :mod:`csv`.
    """

    def __init__(self, path: str, columns: Sequence[str],
                 optional_columns: Sequence[str] = ()):
        cols = list(columns)
        if not cols:
            raise ValueError("metrics columns must be a non-empty sequence")
        if len(set(cols)) != len(cols):
            raise ValueError(f"duplicate metrics columns: {cols}")
        optional = set(optional_columns)
        unknown_opt = optional - set(cols)
        if unknown_opt:
            raise ValueError(f"optional columns not in declared columns: {sorted(unknown_opt)}")
        self.path = path
        self.columns = cols
        self._required = [c for c in cols if c not in optional]
        self._optional = optional
        self._rows_written = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=cols, extrasaction="raise")
        self._writer.writeheader()
        self._fh.flush()

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def append_row(self, row: Mapping[str, Any]) -> None:
        keys = set(row)
        unknown = keys - set(self.columns)
        if unknown:
            raise ValueError(f"unknown metrics columns: {sorted(unknown)}")
        missing = [c for c in self._required if c not in keys]
        if missing:
            raise ValueError(f"missing required metrics columns: {missing}")
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"non-finite metric value for column {k!r}: {v!r}")
        full = {c: row.get(c, "") for c in self.columns}
        self._writer.writerow(full)
        self._rows_written += 1
        self._fh.flush()

    def append_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.append_row(r)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


# Sentinel for "argument not supplied" so ``None`` can be an explicit annotation value.
_UNSET: Any = object()


# ============================================================ the recorder
class ReplayRecorder:
    """Streaming recorder + lifecycle for one headless experiment run.

    Use as a context manager::

        with ReplayRecorder(engine, experiment="basic", output_root=root,
                            metrics_columns=["timestep", "value"]) as rec:
            rec.set_annotation(phase="train", pattern="row 1")
            rec.marker("phase", data={"phase": "train"})
            for _ in range(n):
                engine.step()
                rec.record_frame(engine)
                rec.metrics.append_row({"timestep": engine.timestep, "value": ...})
            rec.finish(STATUS_COMPLETED, checks={"ok": True}, result={...})

    On an unhandled exception leaving the ``with`` block, the run is marked
    ``interrupted``; all frames written before the interruption remain parseable and are
    never erased.
    """

    def __init__(
        self,
        engine: Any,
        *,
        experiment: str,
        output_root: str,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
        record_every: int = 1,
        checkpoint_every: int = 100,
        flush_every: int = 1,
        conditions: Optional[Mapping[str, Any]] = None,
        schedule: Optional[Any] = None,
        hierarchical_feedback: str = FEEDBACK_NOT_APPLICABLE,
        metrics_columns: Optional[Sequence[str]] = None,
        optional_columns: Sequence[str] = (),
        repo_dir: Optional[str] = None,
    ):
        if record_every < 1:
            raise ValueError("record_every must be >= 1")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if hierarchical_feedback not in _FEEDBACK_VOCAB:
            raise ValueError(
                f"hierarchical_feedback must be one of {_FEEDBACK_VOCAB}, "
                f"got {hierarchical_feedback!r}")

        self.engine = engine
        self.experiment = experiment
        self.run_id = run_id or _new_run_id(experiment)
        self.record_every = int(record_every)
        self.checkpoint_every = int(checkpoint_every)
        self.flush_every = max(1, int(flush_every))
        self.conditions = dict(conditions or {})
        self.schedule = schedule
        self.hierarchical_feedback = hierarchical_feedback
        self.repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        params = getattr(engine, "params", {}) or {}
        self.seed = int(seed if seed is not None else params.get("seed", 0))

        self.run_dir = os.path.join(output_root, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.replay_path = os.path.join(self.run_dir, "replay.snn.jsonl")
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.summary_path = os.path.join(self.run_dir, "summary.json")

        # ---- bounded streaming state only (no per-frame history is retained) ----
        self.status = STATUS_RUNNING
        self._frame_index = -1            # last WRITTEN frame index
        self._observed = 0                # record_frame() calls seen (incl. skipped)
        self._frames_written = 0
        self._markers_written = 0
        self._checkpoints_written = 0
        self._frames_since_checkpoint = 0
        self._last_log_seq = 0            # highest event-log seq already emitted
        self._writes_since_flush = 0
        self._first_timestep: Optional[int] = None
        self._last_timestep: Optional[int] = None
        self._finalized = False
        self._created_utc = _utcnow()

        self._annotation: dict = {"phase": None, "pattern": None, "tags": [], "notes": None}

        # Metrics writer (columns optional; a run may record no metrics).
        self.metrics: Optional[MetricsWriter] = None
        if metrics_columns is not None:
            self.metrics = MetricsWriter(self.metrics_path, metrics_columns, optional_columns)

        # Buffered append-only JSONL writer.
        self._replay_fh = open(self.replay_path, "w", buffering=io.DEFAULT_BUFFER_SIZE,
                               encoding="utf-8")
        self._write_header()
        self._write_manifest()

    # ---------------------------------------------------------------- context
    def __enter__(self) -> "ReplayRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self._finalized:
            reason = f"{exc_type.__name__}: {exc}"
            status = STATUS_INTERRUPTED if issubclass(exc_type, KeyboardInterrupt) \
                else STATUS_FAILED
            try:
                self.finish(status, checks={}, result={}, failure_reason=reason)
            except Exception:
                # Never mask the original exception with a bookkeeping error.
                self._safe_close()
        else:
            self._safe_close()
        return False   # never suppress the caller's exception

    # ---------------------------------------------------------------- writes
    def _emit(self, record: Mapping[str, Any]) -> None:
        self._replay_fh.write(_dumps(record))
        self._replay_fh.write("\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.flush_every:
            self._replay_fh.flush()
            self._writes_since_flush = 0

    def _write_header(self) -> None:
        topo = self.engine.topology()
        header = {
            "record": REC_HEADER,
            "schema": REPLAY_SCHEMA_NAME,
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "created_utc": self._created_utc,
            "seed": self.seed,
            "topology_name": topo.get("params", {}).get("topology_name"),
            "preset": topo.get("params", {}).get("topology"),
            "conditions": {
                "hierarchical_feedback": self.hierarchical_feedback,
                **self.conditions,
            },
            "schedule": self.schedule,
            "recording": {
                "record_every": self.record_every,
                "checkpoint_every": self.checkpoint_every,
            },
            # Stable ordering / IDs for the player.
            "neuron_order": [n["id"] for n in topo["neurons"]],
            "synapse_order": [s["id"] for s in topo["synapses"]],
            # Complete initial topology INCLUDING initial live weights.
            "topology": topo,
        }
        self._emit(header)
        self._replay_fh.flush()

    def _build_manifest(self) -> dict:
        topo_params = self.engine.topology().get("params", {})
        completed = self.status in _TERMINAL_STATUSES
        manifest = {
            "schema": "snn.manifest",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "replay_schema": REPLAY_SCHEMA_NAME,
            "replay_schema_version": REPLAY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "created_utc": self._created_utc,
            "completed_utc": _utcnow() if completed else None,
            "status": self.status,
            "seed": self.seed,
            "topology_name": topo_params.get("topology_name"),
            "preset": topo_params.get("topology"),
            "engine_params": topo_params,
            "schedule": self.schedule,
            "conditions": {
                "hierarchical_feedback": self.hierarchical_feedback,
                **self.conditions,
            },
            "recording": {
                "record_every": self.record_every,
                "checkpoint_every": self.checkpoint_every,
                "flush_every": self.flush_every,
            },
            "start_timestep": self._first_timestep,
            "end_timestep": self._last_timestep,
            "frames_recorded": self._frames_written,
            "markers_recorded": self._markers_written,
            "checkpoints_recorded": self._checkpoints_written,
            "git": _git_info(self.repo_dir),
            "artifacts": {
                "manifest": {"path": os.path.basename(self.manifest_path),
                             "schema_version": MANIFEST_SCHEMA_VERSION},
                "replay": {"path": os.path.basename(self.replay_path),
                           "schema": REPLAY_SCHEMA_NAME,
                           "schema_version": REPLAY_SCHEMA_VERSION},
                "metrics": {"path": os.path.basename(self.metrics_path),
                            "schema_version": METRICS_SCHEMA_VERSION,
                            "present": self.metrics is not None},
                "summary": {"path": os.path.basename(self.summary_path),
                            "schema_version": SUMMARY_SCHEMA_VERSION,
                            "present": self._finalized},
            },
        }
        return manifest

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self._build_manifest())

    # ---------------------------------------------------------------- annotations
    def set_annotation(self, *, phase: Any = _UNSET, pattern: Any = _UNSET,
                       tags: Any = _UNSET, notes: Any = _UNSET) -> None:
        """Update the current experiment annotation carried on subsequent frames.

        Only the fields you pass are changed; the rest are retained.
        """
        if phase is not _UNSET:
            self._annotation["phase"] = phase
        if pattern is not _UNSET:
            self._annotation["pattern"] = pattern
        if tags is not _UNSET:
            self._annotation["tags"] = list(tags) if tags is not None else []
        if notes is not _UNSET:
            self._annotation["notes"] = notes

    def _annotation_snapshot(self) -> dict:
        a = self._annotation
        return {"phase": a["phase"], "pattern": a["pattern"],
                "tags": list(a["tags"]), "notes": a["notes"]}

    # ---------------------------------------------------------------- markers
    def marker(self, kind: str, *, data: Optional[Mapping[str, Any]] = None,
               timestep: Optional[int] = None) -> None:
        """Record a caller-supplied semantic transition. The recorder never invents one."""
        self._require_open()
        rec = {
            "record": REC_MARKER,
            "kind": str(kind),
            "frame_index": self._frame_index,     # marker attaches after the last frame
            "timestep": (int(timestep) if timestep is not None
                         else int(getattr(self.engine, "timestep", 0))),
            "annotation": self._annotation_snapshot(),
            "data": dict(data or {}),
        }
        self._emit(rec)
        self._markers_written += 1

    # ---------------------------------------------------------------- frames
    def record_frame(self, engine: Optional[Any] = None, *, force: bool = False) -> bool:
        """Record a state frame after an engine step. Honors ``record_every``.

        Returns ``True`` if a frame was written, ``False`` if this step was sampled out.
        Every written frame carries the stride and the *actual* engine timestep so sampled
        playback is honest; skipped timesteps are simply absent (never invented).
        """
        self._require_open()
        engine = engine if engine is not None else self.engine
        self._observed += 1
        if not force and (self._observed - 1) % self.record_every != 0:
            return False

        dyn = engine.dynamic_state()
        timestep = int(dyn.get("timestep", getattr(engine, "timestep", 0)))
        # Replace the rolling log with ONLY the new entries since the last written frame,
        # read from the authoritative event_log (dynamic_state truncates to the tail).
        dyn["log"] = self._new_log_entries(engine)

        self._frame_index += 1
        frame = {
            "record": REC_FRAME,
            "frame_index": self._frame_index,
            "timestep": timestep,
            "record_every": self.record_every,
            "annotation": self._annotation_snapshot(),
            "dynamic": dyn,
        }
        self._emit(frame)
        self._frames_written += 1
        self._frames_since_checkpoint += 1
        if self._first_timestep is None:
            self._first_timestep = timestep
        self._last_timestep = timestep

        if self._frames_since_checkpoint >= self.checkpoint_every:
            self._write_checkpoint(engine, timestep)
        return True

    def _new_log_entries(self, engine: Any) -> list:
        log = getattr(engine, "event_log", None)
        if not log:
            return []
        fresh = [dict(e) for e in log if e.get("seq", 0) > self._last_log_seq]
        if fresh:
            self._last_log_seq = max(e.get("seq", 0) for e in log)
        return fresh

    # ---------------------------------------------------------------- checkpoints
    def checkpoint(self, engine: Optional[Any] = None) -> None:
        """Force a full weight checkpoint at the current frame."""
        self._require_open()
        engine = engine if engine is not None else self.engine
        self._write_checkpoint(engine, int(getattr(engine, "timestep", 0)))

    def _write_checkpoint(self, engine: Any, timestep: int) -> None:
        weights = _live_weights(engine)
        rec = {
            "record": REC_CHECKPOINT,
            "frame_index": self._frame_index,
            "timestep": timestep,
            "weights": weights,
        }
        self._emit(rec)
        self._checkpoints_written += 1
        self._frames_since_checkpoint = 0

    # ---------------------------------------------------------------- finish
    def finish(self, status: str, *, checks: Optional[Mapping[str, bool]] = None,
               result: Optional[Mapping[str, Any]] = None,
               failure_reason: Optional[str] = None,
               write_result_record: bool = True) -> dict:
        """Write summary.json, an optional final ``result`` replay record, and update the
        manifest atomically. Idempotent-safe: only the first call finalizes.
        """
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"finish status must be one of {_TERMINAL_STATUSES}, got {status!r}")
        if self._finalized:
            return self._last_summary
        summary = {
            "schema": "snn.summary",
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "status": status,
            "completed_utc": _utcnow(),
            "checks": dict(checks or {}),
            "all_checks_pass": (all(bool(v) for v in checks.values()) if checks else None),
            "result": dict(result or {}),
            "failure_reason": failure_reason,
            "frames_recorded": self._frames_written,
            "artifacts": {
                "manifest": os.path.basename(self.manifest_path),
                "replay": os.path.basename(self.replay_path),
                "metrics": os.path.basename(self.metrics_path),
            },
        }
        # Optional final replay record mirrors the summary so a single .jsonl is complete.
        if write_result_record and not self._replay_fh.closed:
            self._emit({"record": REC_RESULT, **{k: summary[k] for k in
                        ("status", "checks", "all_checks_pass", "result", "failure_reason")}})

        _atomic_write_json(self.summary_path, summary)
        self.status = status
        self._finalized = True
        self._last_summary = summary
        self._write_manifest()
        self._safe_close()
        return summary

    # ---------------------------------------------------------------- helpers
    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("recorder already finalized; no further records may be written")

    def _safe_close(self) -> None:
        try:
            if not self._replay_fh.closed:
                self._replay_fh.flush()
                self._replay_fh.close()
        finally:
            if self.metrics is not None:
                self.metrics.close()

    @property
    def frames_written(self) -> int:
        return self._frames_written


# ============================================================ weight helpers
def _live_weights(engine: Any) -> dict:
    """All currently weighted synapses keyed by stable synapse id (unweighted -> omitted)."""
    out: dict = {}
    for edge in engine.synapses:
        w = engine._live_weight(edge)
        if w is None:
            continue
        wf = float(w)
        if not math.isfinite(wf):
            raise ValueError(f"non-finite live weight for synapse {edge['id']!r}: {wf!r}")
        out[edge["id"]] = round(wf, 6)
    return out


# ============================================================ pure replay reader
def iter_records(path_or_lines: Any) -> Iterable[dict]:
    """Yield parsed JSONL records from a path or an iterable of lines."""
    if isinstance(path_or_lines, (str, os.PathLike)):
        with open(path_or_lines, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        for line in path_or_lines:
            line = line.strip() if isinstance(line, str) else line
            if line:
                yield json.loads(line)


def read_records(path_or_lines: Any) -> list:
    return list(iter_records(path_or_lines))


def header_initial_weights(records: Sequence[Mapping[str, Any]]) -> dict:
    """Initial live weights from the header's topology (weight is not None)."""
    for rec in records:
        if rec.get("record") == REC_HEADER:
            out = {}
            for s in rec["topology"]["synapses"]:
                if s.get("weight") is not None:
                    out[s["id"]] = float(s["weight"])
            return out
    raise ValueError("no header record found")


def reconstruct_weights_at(records: Sequence[Mapping[str, Any]], frame_index: int) -> dict:
    """Reconstruct live weights as of ``frame_index`` using the nearest preceding
    ``weight_checkpoint`` (or the header's initial weights) plus intervening
    ``changed_synapses``. Pure: reads only the supplied records, in any accessible order.

    Supports backward / random access: call repeatedly with different targets.
    """
    records = list(records)
    if not any(r.get("record") == REC_FRAME and r.get("frame_index") == frame_index
               for r in records):
        raise ValueError(f"no frame with index {frame_index}")

    # Base = latest checkpoint at or before the target frame, else header initial weights.
    base_frame = -1
    weights = header_initial_weights(records)
    for rec in records:
        if rec.get("record") == REC_CHECKPOINT and rec.get("frame_index", -1) <= frame_index:
            if rec["frame_index"] >= base_frame:
                base_frame = rec["frame_index"]
                weights = dict(rec["weights"])

    # Apply changed_synapses for frames strictly after the base up to and incl. target.
    weights = {k: float(v) for k, v in weights.items()}
    for rec in records:
        if rec.get("record") != REC_FRAME:
            continue
        idx = rec.get("frame_index", -1)
        if base_frame < idx <= frame_index:
            for c in rec.get("dynamic", {}).get("changed_synapses", []):
                weights[c["id"]] = float(c["weight"])
    return weights
