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

from .simulation import SimulationEngine
from .serializer import topology_message, full_state
from .websocket import ConnectionManager, SimulationRunner

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(title="SNN Dashboard")
engine = SimulationEngine()
manager = ConnectionManager()
runner = SimulationRunner(engine, manager)


@app.on_event("startup")
async def _startup():
    runner.start_loop()


# --------------------------------------------------------------------- models
class StimulateBody(BaseModel):
    neuron_id: str
    magnitude: float = 1.0
    continuous: bool = False


class InputBody(BaseModel):
    vector: list[int]


class PatternBody(BaseModel):
    name: str


# ----------------------------------------------------------------- REST: core
@app.get("/api/state")
async def get_state():
    return full_state(engine, runner.running, runner.speed)


@app.post("/api/start")
async def start():
    runner.running = True
    engine._log("control", "simulation started")
    return {"running": True}


@app.post("/api/pause")
async def pause():
    runner.running = False
    engine._log("control", "simulation paused")
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
