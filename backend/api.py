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
from neuron_flexible import UNIT   # fixed-point scale (potentials/thresholds run at * UNIT)
from .serializer import topology_message, full_state
from .websocket import ConnectionManager, SimulationRunner

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(title="SNN Dashboard")
# Dashboard config: homeostasis OFF so the fixed weight_budget (= threshold_l2)
# governs each L2E's total feedforward weight -- receptive fields concentrate to
# large, visible values instead of being shrunk by homeostatic scaling. A faster
# L2E learning rate sharpens them. Edit these lines to change what you observe.
engine = SimulationEngine(
    homeostasis=False,   # use the fixed weight_budget=threshold_l2 (visible, strong RFs)
    l2e_lr_frac=0.02,    # faster L2E feedforward learning (sharper receptive fields)
    ei_sat_mult=4.0,     # push E->I saturation equilibrium above the clip so L2E->L2I /
                         # L2E->L1I weights REACH the cap (single-spike relay), instead of
                         # asymptoting just under it. Higher = reaches cap faster / more
                         # linear; 1.0 = old asymptote-at-cap behavior. Tune here.
    # l1i_ei_init_frac left at the default (None -> [0.25,0.5]*thr round-robin
    # init). Setting it to 1.0 fires ALL L1I from any L2E winner (synchronous
    # global input suppression), but without homeostasis to recruit units that
    # starves participation: it alone collapses confidence-gated consolidation to
    # ~2/8 winners with ~6 dead neurons, whereas the round-robin init reaches the
    # full 8/8 one-to-one tiling the consolidation mechanism produces. Restore
    # l1i_ei_init_frac=1.0 here to get synchronous suppression back at that cost.
)
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


# ------------------------------------------------------------------- config
# Tunable-parameter spec the frontend renders as sliders/toggles. Each entry
# drives one control and its help text; "kind" is "range" or "toggle".
CONFIG_SPEC = [
    {"key": "signed_depression", "label": "Signed depression (4a)", "kind": "toggle",
     "desc": "On fire, OFF pixels (absent inputs) push their positive gates DOWN. "
             "Sharpens receptive fields; needs eta_off > 0 to have any effect."},
    {"key": "eta_off", "label": "OFF-gate depression rate (eta_off)", "kind": "range",
     "min": 0.0, "max": 0.4, "step": 0.01,
     "desc": "How hard absent inputs are depressed. ~0.05 sharpens RFs and lifts "
             "old-pattern retention at little cost; higher over-specializes and can "
             "destabilize the tiling."},
    {"key": "event_driven", "label": "Event-driven firing", "kind": "toggle",
     "desc": "Fire an L2E the instant it crosses threshold (every step) instead of "
             "one argmax winner per cycle. Bounds the membrane near threshold (no "
             "charge pile-up) but re-couples winner timing to input rate."},
    {"key": "subtractive_reset", "label": "Reset by subtraction", "kind": "toggle",
     "desc": "On L2E fire, subtract threshold from the membrane (floored at rest) "
             "instead of a full reset to rest. Leaves the winner its residual "
             "overshoot like partially-inhibited losers keep theirs — attacks the "
             "discharge asymmetry behind the sustained round-robin. (Inert at "
             "refractory>0; hurts ownership at refractory=0. Leave off.)"},
    {"key": "refractory", "label": "Refractory period", "kind": "range",
     "min": 0, "max": 3, "step": 1,
     "desc": "Steps a neuron is locked out (membrane clamped to rest) after firing. "
             "0 = no lockout: inhibition alone regulates frequency. Ownership is "
             "identical at 0 vs 2 under the visit-consistency metric."},
    {"key": "v_sat_frac", "label": "L2E membrane saturation (×thr)", "kind": "range",
     "min": 0.0, "max": 3.0, "step": 0.25,
     "desc": "Ceiling on accumulated L2E charge as a multiple of threshold "
             "(0 = unbounded). Keeps the membrane near threshold so the small "
             "capped inhibitory gate can actually regulate firing. Local finite "
             "driving force / reversal potential."},
    {"key": "l2e_budget", "label": "L2E weight budget", "kind": "toggle",
     "desc": "Sum-renormalization competition on each L2E's feedforward weights. "
             "Required for clean 8/8 tiling — turning it off collapses competition "
             "(dead neurons, no clear winners). Kept ON."},
    {"key": "l2e_lr_frac", "label": "L2E learning rate", "kind": "range",
     "min": 0.005, "max": 0.1, "step": 0.005,
     "desc": "Feedforward potentiation speed for L2E. Higher = faster, sharper RFs "
             "but noisier competition."},
    {"key": "confidence_consolidation", "label": "Confidence consolidation", "kind": "toggle",
     "desc": "Mature gates learn slower and resist depression (protects specialists). "
             "Also gates signed depression via (1 - C)."},
    {"key": "loser_depression", "label": "Loser depression", "kind": "toggle",
     "desc": "Depress the active gates of neurons that were suppressed by lateral "
             "inhibition — pushes losers away from the winner's pattern."},
    {"key": "eta_loss", "label": "Loser-depression rate (eta_loss)", "kind": "range",
     "min": 0.0, "max": 0.05, "step": 0.005,
     "desc": "Strength of loser depression. 0 disables it even if the toggle is on."},
    {"key": "leak_l2", "label": "L2 leak", "kind": "range",
     "min": 0.001, "max": 0.05, "step": 0.001,
     "desc": "Fraction of L2 potential that decays per step. The main lever on winner "
             "rotation/stability — lower holds charge longer."},
]


class ConfigBody(BaseModel):
    overrides: dict


def _current_config():
    p = engine.params
    values = {s["key"]: p.get(s["key"]) for s in CONFIG_SPEC}
    return {"spec": CONFIG_SPEC, "values": values}


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
