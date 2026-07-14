"""
FastAPI application: REST control endpoints, the /ws websocket stream, and the
static frontend. The neural computation lives entirely in SimulationEngine; this
module only translates HTTP/WS requests into engine verbs and serializes the
results.

Run with:
    uvicorn backend.api:app          (add --reload while developing)
then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dashboard_config import CONFIG_SPEC, DASHBOARD_OVERRIDES, config_values
from .simulation import SimulationEngine
from neuron_flexible import UNIT   # fixed-point scale (potentials/thresholds run at * UNIT)
from .serializer import topology_message, full_state
from .websocket import ConnectionManager, SimulationRunner

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

# Seed persistence: the developmental seed SURVIVES server restarts and changes ONLY
# on an explicit Reseed (not Reset, not a restart). Stored next to the runtime logs
# under .claude/ so a restart reproduces the exact network the user was looking at.
_STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".claude")
_SEED_FILE = os.path.join(_STATE_DIR, "dashboard_seed.txt")


def _load_seed(default=1):
    """Read the persisted seed (default if the file is missing/unreadable)."""
    try:
        with open(_SEED_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return default


def _save_seed(seed):
    """Persist the seed so the next restart rebuilds the same network."""
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        with open(_SEED_FILE, "w") as f:
            f.write(str(int(seed)))
    except OSError:
        pass


app = FastAPI(title="SNN Dashboard")

# The active experiment is intentionally visible in one small dictionary.
engine = SimulationEngine(seed=_load_seed(), **DASHBOARD_OVERRIDES)
manager = ConnectionManager()
runner = SimulationRunner(engine, manager)


@app.on_event("startup")
async def _startup():
    runner.start_loop()


# --------------------------------------------------------------------- models
class StimulateBody(BaseModel):
    neuron_id: str
    magnitude: float = 1 * UNIT   # 1 threshold-unit of charge at the fixed-point scale
    continuous: bool = False


class InputBody(BaseModel):
    vector: list[int]


class PatternBody(BaseModel):
    name: str


class WeightBody(BaseModel):
    j: int          # L2E neuron index
    i: int          # source pixel index (0..N_PIX-1)
    weight: float   # new feedforward weight (absolute; clipped to [0, cap])


# ----------------------------------------------------------------- REST: core
@app.get("/api/state")
async def get_state():
    return full_state(engine, runner.running, runner.speed)


@app.post("/api/start")
async def start():
    runner.running = True
    engine._log("control", "simulation started")
    # Broadcast the new running flag: clients gate UI state (RF-panel edit mode,
    # play/pause highlight) on dyn.running from dynamic frames, and the run loop
    # only streams while running -- without this, a pause is never seen.
    await runner.broadcast_dynamic()
    return {"running": True}


@app.post("/api/pause")
async def pause():
    runner.running = False
    engine._log("control", "simulation paused")
    await runner.broadcast_dynamic()
    return {"running": False}


@app.post("/api/step")
async def step():
    engine.step()
    await runner.broadcast_dynamic()
    return engine.dynamic_state()


@app.post("/api/reset")
async def reset():
    runner.running = False
    engine.reset()
    await manager.broadcast(topology_message(engine))
    await runner.broadcast_dynamic()
    return {"reset": True}


@app.post("/api/reseed")
async def reseed():
    # Randomized reset: new random seed -> fresh initial weights under the SAME
    # config (works for any plasticity combination). Wipes learned state like reset.
    runner.running = False
    seed = engine.reseed()
    _save_seed(seed)   # persist so the new network survives a restart
    await manager.broadcast(topology_message(engine))
    await runner.broadcast_dynamic()
    return {"reseed": True, "seed": seed}


@app.post("/api/speed/{sps}")
async def set_speed(sps: float):
    runner.speed = max(0.5, min(120.0, sps))
    return {"speed": runner.speed}


# ----------------------------------------------------------- REST: input/control
@app.post("/api/pattern")
async def set_pattern(body: PatternBody):
    # Name is sent in the body (not the URL path) so patterns containing '/' or '\'
    # -- like "diag /" -- don't break path routing.
    try:
        engine.set_pattern(body.name)
    except KeyError:
        return JSONResponse({"error": f"unknown pattern '{body.name}'"}, status_code=404)
    await runner.broadcast_dynamic()
    return {"pattern": body.name, "input": engine.input_vec.astype(int).tolist()}


@app.post("/api/input")
async def set_input(body: InputBody):
    engine.set_input(body.vector)
    await runner.broadcast_dynamic()
    return {"input": engine.input_vec.astype(int).tolist()}


@app.post("/api/pixel/{index}")
async def toggle_pixel(index: int):
    engine.toggle_pixel(index)
    await runner.broadcast_dynamic()
    return {"input": engine.input_vec.astype(int).tolist()}


@app.post("/api/weight")
async def set_weight(body: WeightBody):
    """Manually set an L2E feedforward weight (RF panel hand-edit). Re-broadcasts the
    topology so every client's weight map updates. Best used while paused."""
    try:
        w = engine.set_feedforward_weight(body.j, body.i, body.weight)
    except IndexError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    await manager.broadcast(topology_message(engine))
    await runner.broadcast_dynamic()
    return {"j": body.j, "i": body.i, "weight": w, "synapse": f"ff{body.i}->{body.j}"}


@app.post("/api/clear")
async def clear():
    engine.clear_input()
    await runner.broadcast_dynamic()
    return {"input": engine.input_vec.astype(int).tolist()}


@app.post("/api/random")
async def random_pattern():
    engine.random_pattern()
    await runner.broadcast_dynamic()
    return {"input": engine.input_vec.astype(int).tolist()}


@app.post("/api/noise/{prob}")
async def noise(prob: float):
    engine.inject_noise(prob)
    await runner.broadcast_dynamic()
    return {"input": engine.input_vec.astype(int).tolist()}


@app.post("/api/stimulate")
async def stimulate(body: StimulateBody):
    try:
        engine.stimulate(body.neuron_id, body.magnitude, body.continuous)
    except KeyError:
        return JSONResponse({"error": f"unknown neuron '{body.neuron_id}'"}, status_code=404)
    await runner.broadcast_dynamic()
    return {"ok": True}


# ------------------------------------------------------------------- config
class ConfigBody(BaseModel):
    overrides: dict


def _current_config():
    return {"spec": CONFIG_SPEC, "values": config_values(engine.params)}


@app.get("/api/config")
async def get_config():
    return _current_config()


class AutoCycleBody(BaseModel):
    enabled: bool
    streak: int | None = None
    visit_steps: int | None = None


@app.post("/api/autocycle")
async def set_autocycle(body: AutoCycleBody):
    state = engine.set_auto_cycle(body.enabled, body.streak, body.visit_steps)
    await runner.broadcast_dynamic()
    return state


@app.post("/api/config")
async def set_config(body: ConfigBody):
    runner.running = False
    applied = engine.apply_config(body.overrides)
    await manager.broadcast(topology_message(engine))
    await runner.broadcast_dynamic()
    return {"applied": applied, **_current_config()}


# ----------------------------------------------------------------- websocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Hand the new client the static topology, then the current dynamic frame.
    await ws.send_json(topology_message(engine))
    await ws.send_json({"type": "dynamic",
                        "data": {**engine.dynamic_state(),
                                 "running": runner.running, "speed": runner.speed}})
    try:
        while True:
            await ws.receive_text()      # clients are not required to send anything
    except WebSocketDisconnect:
        manager.disconnect(ws)


# --------------------------------------------------------------- static assets
# Mounted last so /api/* and /ws take precedence; serves index.html at "/".
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
