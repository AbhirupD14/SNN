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
# The dashboard runs the MINIMAL SIGNED-SPIKE experiment by default (see
# Claude_Minimal_Signed_Spike_Learning_Prompt.md and the README section). The
# feedforward rule is the signed one -- on fire, active inputs (+1) potentiate and
# inactive inputs (-1) depress, dw = eta*p*(1-(w/w_cap)^2)*signal, no weight budget
# -- and every compensating mechanism is off so the core local loop is what you
# see: charge -> fire -> local signed update -> learned L2I lateral inhibition ->
# L1I feedback inhibition -> repeat. refractory=0 (inhibition, not a hard lockout,
# regulates frequency). Toggle any of these live in the "Model Config" panel.
engine = SimulationEngine(
    signed_spike_learning=True,   # signed +1/-1 feedforward learning (the algorithm)
    l2e_budget=False,             # no positive-weight budget; -1 signal supplies down-pressure
    confidence_consolidation=False,
    # Consolidation stack: loser depression breaks the feed-forward symmetry so one
    # L2E owns a held pattern; assembly-flow credit then lets that habitual winner's
    # L2E->L2I synapse climb to self-sufficiency so L2I fires in rhythm (it removes
    # the last-volley-only credit that stalled the E->I synapse below threshold --
    # the L2I firing deadlock). Both run off the SAME event (L2I's discharge), so
    # they share one clock: no boolean gate. Validated on held 'row 0' -- one L2E
    # specialist + L2I firing on a ~16-step rhythm with the winner's E->I synapse
    # matured to threshold; legacy last-volley credit stalls it at ~0.55*thr and L2I
    # stays silent. Keep the default l2i_lr_frac (0.01): a faster E->I rate makes L2I
    # over-inhibit early and destabilizes it. See Inhibition_And_Consolidation_State.md.
    loser_depression=True,
    eta_loss=10.0,                # symmetry-breaker strength (0.01 default is far too weak)
    assembly_flow_credit=True,    # flow-proportional E->I credit on L2I/L1I fire
    signed_depression=False,      # superseded by the unified signed rule
    homeostasis=False,
    refractory=0,                 # inhibition regulates frequency, not a hard lockout
    # Capacity rule: per-afferent cap = thr/3 so three strong active afferents reach
    # threshold (3-pixel lines); positive floor = 1; each I threshold = its E's / 3.
    l2e_weight_cap_frac=1 / 3,
    pos_weight_floor=1,
    l2i_threshold_frac=1 / 7,     # L2I threshold = threshold_l2 / 3
    l1i_threshold_frac=1 / 3,     # L1I threshold = threshold / 3
    l2e_lr_frac=0.02,             # L2E feedforward learning rate (fraction of the cap)
    ei_sat_mult=4.0,              # push E->I saturation above the clip so L2E->L2I reaches
                                  # the cap and L2I can sharpen into a single-source relay.
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
    {"key": "signed_spike_learning", "label": "Signed-spike learning (minimal)", "kind": "toggle",
     "desc": "Minimal local feedforward rule: on fire, active inputs (+1) potentiate "
             "and inactive inputs (-1) depress via dw=eta*p*(1-(w/w_cap)^2)*signal, "
             "no weight budget. Replaces the confidence/OFF-depression/budget stack. "
             "Run with those OFF, l2e_budget OFF, and refractory=0 for the minimal "
             "experiment."},
    {"key": "signed_depression", "label": "Signed depression (4a)", "kind": "toggle",
     "desc": "On fire, OFF pixels (absent inputs) push their positive gates DOWN. "
             "Sharpens receptive fields; needs eta_off > 0 to have any effect. "
             "(Superseded by signed-spike learning; leave off when that is on.)"},
    {"key": "eta_off", "label": "OFF-gate depression rate (eta_off)", "kind": "range",
     "min": 0.0, "max": 0.4, "step": 0.01,
     "desc": "How hard absent inputs are depressed. ~0.05 sharpens RFs and lifts "
             "old-pattern retention at little cost; higher over-specializes and can "
             "destabilize the tiling."},
    {"key": "event_driven", "label": "Event-driven firing", "kind": "toggle",
     "desc": "Resolve L2 competition every step -- one argmax winner per timestep, "
             "inhibiting the rest (DEFAULT ON, the canonical flow). Turn OFF to "
             "resolve the same argmax competition only once per cycle, which "
             "decouples winner timing from the input rate."},
    {"key": "l2_charge_chunks", "label": "L2 charge chunks (K)", "kind": "range",
     "min": 1, "max": 16, "step": 1,
     "desc": "Deliver each step's L1->L2E feedforward drive in K equal chunks "
             "within one frozen timestep, re-running the argmax WTA after each "
             "chunk and stopping at the first threshold-crosser. K=1 (default) is "
             "the un-chunked baseline; larger K lets the earliest strong responder "
             "win before rivals pile up charge. IGNORED (forced to 1) when "
             "excitatory flow-rate mode is on."},
    {"key": "excitatory_flow_rate", "label": "Excitatory flow-rate", "kind": "toggle",
     "desc": "Treat each weight as a current amplitude, not an instant charge "
             "packet: an input spike opens a decaying excitatory current trace that "
             "integrates into V over several timesteps (L2E/L2I/L1I; not L1E, not a "
             "relay L1I). OFF (default) = instantaneous V += dot(w, spikes). Forces "
             "L2 charge chunks to 1 while on."},
    {"key": "exc_trace_decay", "label": "Exc. trace decay (d)", "kind": "range",
     "min": 0.0, "max": 0.99, "step": 0.01,
     "desc": "Per-timestep decay of the excitatory current trace in flow-rate mode. "
             "Higher = current lingers and charge spreads over more timesteps; 0 = "
             "delivers in a single step (≈ instantaneous). Only used when flow-rate "
             "mode is on."},
    {"key": "assembly_flow_credit", "label": "Assembly flow credit (E→I)", "kind": "toggle",
     "desc": "On an inhibitory neuron's (L2I/L1I) OWN fire, credit its incoming "
             "positive E→I synapses in proportion to the flow each delivered over "
             "the retention window (per-synapse leaky trace), normalized so the "
             "DOMINANT driver gets the full learning rate; non-contributors decay "
             "toward the floor. Replaces the last-volley-only credit that stalled a "
             "habitual winner's E→I synapse below threshold (the L2I firing deadlock), "
             "so one specialist can grow enough to fire L2I in rhythm by itself."},
    {"key": "assembly_decay_frac", "label": "Assembly non-contributor decay", "kind": "range",
     "min": 0.0, "max": 2.0, "step": 0.05,
     "desc": "Down-pressure on E→I synapses that delivered no flow this window, as a "
             "fraction of the learning rate: dw = -eta*p*frac*(w-w_min). 0 = grow "
             "contributors only. Only used when assembly flow credit is on."},
    {"key": "inhibitory_flow_rate", "label": "Inhibitory flow-rate", "kind": "toggle",
     "desc": "Model the L2I->L2E discharge as a decaying current that drains charge "
             "over several steps (sustained suppression), symmetric to the excitatory "
             "flow, instead of a one-shot subtraction. OFF (default) = instant hit. "
             "NOTE: it suppresses more but does NOT break the round-robin on a held "
             "pattern (that needs loser depression) -- see the state doc."},
    {"key": "inh_trace_decay", "label": "Inh. trace decay (d)", "kind": "range",
     "min": 0.0, "max": 0.99, "step": 0.01,
     "desc": "Per-step decay of the inhibitory current in flow mode. Higher = the "
             "discharge lingers over more steps. Only used when inhibitory flow is on."},
    {"key": "inh_trace_normalized", "label": "Inh. trace normalized", "kind": "toggle",
     "desc": "Inject w*(1-d) so the total charge drained over time ~= the one-shot "
             "gate w. OFF injects w (total w/(1-d), a stronger sustained bite). Only "
             "used when inhibitory flow is on."},
    {"key": "exc_trace_normalized", "label": "Exc. trace normalized", "kind": "toggle",
     "desc": "Inject drive*(1-d) so the current trace's total delivered charge "
             "approximates the instantaneous dot(w, spikes) over time (comparable "
             "magnitudes). OFF injects the full drive (larger total). Only used when "
             "flow-rate mode is on."},
    {"key": "inhibitory_delta_rule", "label": "Inhibitory differentiating gate", "kind": "toggle",
     "desc": "ON (default) = event-local TURNOVER rule on each L2I->L2E gate: "
             "du = eta_up*p_t*(1-u) - eta_down*u (u=w/G, p_t=clamp(v_pre/theta,0,p_max)). "
             "High-charge rivals accumulate stronger gates; weak/dead targets drift "
             "down -- gates DIFFERENTIATE, no target voltage or averages. OFF = legacy "
             "saturating rule (every gate converges to the same sqrt(w_max), uniform)."},
    {"key": "inhibitory_eta_up", "label": "Inhibitory eta_up (strengthen)", "kind": "range",
     "min": 0.0, "max": 0.2, "step": 0.005,
     "desc": "Turnover strengthening rate: how fast a discharged high-charge target's "
             "incoming gate grows (scaled by p_t and remaining headroom 1-u). Only used "
             "when the differentiating gate is on."},
    {"key": "inhibitory_eta_down", "label": "Inhibitory eta_down (turnover)", "kind": "range",
     "min": 0.0, "max": 0.1, "step": 0.001,
     "desc": "Turnover decay rate: every gate shrinks proportional to its size each "
             "discharge, so gates that stop being reinforced drift toward zero. Only "
             "used when the differentiating gate is on."},
    {"key": "inhibitory_p_max", "label": "Inhibitory p_max", "kind": "range",
     "min": 0.5, "max": 3.0, "step": 0.1,
     "desc": "Cap on the charge signal p_t = clamp(v_pre/theta, 0, p_max) in the "
             "turnover rule. >1 lets an over-threshold target push its gate harder. "
             "Only used when the differentiating gate is on."},
    {"key": "distance_weighting", "label": "Distance attenuation", "kind": "toggle",
     "desc": "Attenuate DELIVERED excitatory drive by synapse distance: each "
             "afferent's amplitude is scaled by (d_ref/max(d,d_min))^power. Weight = "
             "learned gate, distance = delivery attenuation, trace = temporal flow. "
             "Does NOT change stored weights or trace math. OFF by default; per-synapse "
             "distances are 1.0 (no effect) until functional positions are assigned."},
    {"key": "distance_power", "label": "Distance power", "kind": "range",
     "min": 0.0, "max": 4.0, "step": 0.5,
     "desc": "Exponent in the distance factor (2 = inverse-square). Only used when "
             "distance attenuation is on."},
    {"key": "distance_ref", "label": "Distance ref (d_ref)", "kind": "range",
     "min": 0.5, "max": 8.0, "step": 0.5,
     "desc": "Reference distance: factor = (d_ref/max(d,d_min))^power, so d = d_ref "
             "delivers the full weight. Only used when distance attenuation is on."},
    {"key": "distance_min", "label": "Distance min (d_min)", "kind": "range",
     "min": 0.1, "max": 4.0, "step": 0.1,
     "desc": "Floor on distance to avoid a divide-by-zero / over-boost for very close "
             "synapses. Only used when distance attenuation is on."},
    {"key": "l1i_immediate_relay", "label": "L1I immediate relay", "kind": "toggle",
     "desc": "L1I fires immediately on ANY nonzero L2E feedback -- a deterministic "
             "relay, no learned-threshold crossing or feedback-weight training "
             "(DEFAULT ON). Turn OFF to restore the trainable threshold-integrating "
             "L1I that fires only when accumulated feedback crosses its threshold."},
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
     "min": 0.0, "max": 20.0, "step": 0.01,
     "desc": "Strength of loser depression -- the symmetry-breaker that turns a held "
             "pattern's round-robin into a single owner. 0 disables it. The default "
             "0.01 is far too weak to consolidate; ~10 collapses a held pattern to one "
             "winner (but over-depresses across multiple patterns -- see "
             "Inhibition_And_Consolidation_State.md)."},
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
