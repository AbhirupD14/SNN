"""
Read-only experiment dashboard service.

A browser window INTO the experiments -- it never controls the simulation. There
are no pause/step/reset/config endpoints; it only reads the durable artifacts the
runner writes (status.json, events.jsonl, metrics.jsonl, summary.csv, plots/*.png).
The runner process is completely independent, so the browser (or this service) can
disconnect/restart without affecting a run, and the whole view reconstructs from
the files on disk.

    PYTHONPATH=. .venv/bin/uvicorn experiments.server:app --host 0.0.0.0 --port 8010

Set EXPERIMENTS_RUNS_DIR to point at a runs directory (default experiments/runs).
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

RUNS = Path(os.environ.get("EXPERIMENTS_RUNS_DIR",
                           Path(__file__).parent / "runs")).resolve()
VIEWER = Path(__file__).parent / "viewer" / "index.html"

app = FastAPI(title="SNN Experiments (read-only)")


def _resolve(run_id: str) -> Path:
    """Map a run id (or 'current') to a run directory, guarding against escapes."""
    if run_id in ("current", "latest"):
        link = RUNS / "current"
        if link.exists():
            return link.resolve()
        dirs = sorted([d for d in RUNS.glob("*") if d.is_dir() and d.name != "current"])
        if not dirs:
            raise HTTPException(404, "no runs yet")
        return dirs[-1]
    d = (RUNS / run_id).resolve()
    if RUNS not in d.parents or not d.is_dir():
        raise HTTPException(404, f"unknown run {run_id}")
    return d


def _tail_jsonl(path: Path, n: int):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return VIEWER.read_text() if VIEWER.exists() else "<h1>viewer missing</h1>"


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted([d for d in RUNS.glob("*") if d.is_dir() and d.name != "current"], reverse=True):
        st = d / "status.json"
        info = dict(id=d.name)
        if st.exists():
            try:
                s = json.loads(st.read_text())
                info.update(name=s.get("name", ""), state=s.get("state", ""),
                            done=s.get("done_combos"), total=s.get("total_combos"),
                            updated=s.get("updated"))
            except json.JSONDecodeError:
                pass
        out.append(info)
    return out


@app.get("/api/run/{run_id}/status")
def status(run_id: str):
    p = _resolve(run_id) / "status.json"
    if not p.exists():
        raise HTTPException(404, "no status yet")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/run/{run_id}/description", response_class=PlainTextResponse)
def description(run_id: str):
    p = _resolve(run_id) / "description.md"
    return p.read_text() if p.exists() else ""


@app.get("/api/run/{run_id}/events")
def events(run_id: str, tail: int = 100):
    return _tail_jsonl(_resolve(run_id) / "events.jsonl", tail)


@app.get("/api/run/{run_id}/metrics")
def metrics(run_id: str, tail: int = 200):
    return _tail_jsonl(_resolve(run_id) / "metrics.jsonl", tail)


@app.get("/api/run/{run_id}/summary")
def summary(run_id: str):
    p = _resolve(run_id) / "summary.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(io.StringIO(p.read_text())))


@app.get("/api/run/{run_id}/plot/{name}")
def plot(run_id: str, name: str):
    if not name.endswith(".png") or "/" in name or ".." in name:
        raise HTTPException(400, "bad plot name")
    p = _resolve(run_id) / "plots" / name
    if not p.exists():
        raise HTTPException(404, "no such plot")
    return FileResponse(p, media_type="image/png")
