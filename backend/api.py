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
# Dashboard config: homeostasis and the L2E weight budget are both OFF. Signed
# ON/OFF updates provide the feedforward pressure directly; a faster L2E learning
# rate sharpens receptive fields. Edit these lines to change what you observe.
# The dashboard runs the MINIMAL SIGNED-SPIKE experiment by default (see
# Claude_Minimal_Signed_Spike_Learning_Prompt.md and the README section). The
# feedforward rule is the signed one -- on fire, active inputs (+1) potentiate and
# inactive inputs (-1) depress through the direction-aware bounded kernel, with no
# weight budget. The active loop is charge -> fire -> local signed update ->
# unweighted L2I competitive reset/depression -> L1I feedback inhibition -> repeat.
# The currently implemented dashboard uses refractory=0; refractory=1 is reserved
# for the next gate-redistribution architecture and is not active yet.
engine = SimulationEngine(
    seed=_load_seed(),            # persisted developmental seed (changes only on Reseed)
    leak_enabled=False,
    l2i_leak_enabled=False,
    l1i_leak_enabled=False,
    l1i_immediate_relay=False,
    l2e_init_mode='legacy_wide',  # balanced initialization is opt-in
    signed_spike_learning=True,   # signed +1/-1 feedforward learning (the algorithm)
    # Plasticity gate p = (theta - sum(positive afferent weights)) / theta, clamped to
    # [eta_floor, 1] -- the structural free-energy gate. It REPLACES the closeness term
    # (theta/v_pre) in the signed rule, so the basic nonlinear update becomes
    #   dw = eta * p * (1 - (w/w_cap)^2) * signal,   p = max(eta_floor, 1 - sum_w+/theta).
    # An under-built neuron (sum_w+ << theta) is fully plastic (p~1); one whose positive
    # support already covers a threshold crossing (sum_w+ >= theta) gates to the floor
    # (consolidated). Normalized by theta so p stays in [0,1] (raw theta - sum_w+ would
    # be ~1000 in fixed point and blow the weights up). Floor keeps mature neurons from
    # freezing completely; set structural_fe_eta_floor=0 for the pure theta-sum form.
    structural_free_energy=True,
    structural_fe_eta_floor=0.02,
    # Distance factor (toggleable): each L2E feedforward synapse delivers
    # weight * (d_ref/d)^2, where d = euclidean distance between the L2E neuron and
    # its source pixel (set from the 3D layout in _build). The 1/d^2 SHAPE is fixed
    # (power=2): a farther pixel always contributes LESS charge than a nearer one
    # (ratio = 1/d^2, "dissipating along the axon"). d_ref is the reference distance
    # that sets the absolute scale. d_ref = 7.472 is the FARTHEST L2E<->pixel distance
    # in this layout (N_OUT=8 ring), so EVERY factor is >= 1 (nearest ~3.5x .. farthest
    # 1.0x) -- the "floating-point factors greater than one" requirement. This also
    # keeps the network alive: the un-attenuated net barely fires (bootstrap is
    # marginal), so any factor < 1 (e.g. literal 1/d^2 from d_ref=1 -> 0.02..0.06)
    # silences L2E entirely. NB: d_ref tracks the geometry's max distance -- if N_OUT
    # or the layout changes, recompute (grep _apply_l2e_distances). Stored weights are
    # untouched; this scales DELIVERY only.
    distance_weighting=True,
    distance_power=2.0,
    distance_ref=7.472,
    distance_min=1.0,
    l2e_budget=False,             # no positive-weight budget; -1 signal supplies down-pressure
    confidence_consolidation=False,
    # Competitive depression (canonical, default ON): on an L2I hard-reset event
    # each losing L2E depresses only the POSITIVE feedforward weights whose L1E
    # sources participated in its (losing) response, scaled by its own pre-reset
    # charge (p_loss) through the shared bounded kernel -- the structural half of
    # the reset event (see L2_Hard_Reset_Competitive_Depression_Spec.md and
    # Neuron.apply_competitive_reset). No learned inhibitory magnitude; the rate is
    # the L2E's own learning_rate (eta_loss is not used and is removed here).
    loser_depression=True,
    # Assembly-flow credit lets a habitual winner's L2E->L2I synapse climb to
    # self-sufficiency so L2I fires in rhythm -- it removes the last-volley-only
    # credit that stalled the E->I synapse below threshold (the L2I firing deadlock).
    # Runs off L2I's own discharge. See Flow_Credit_Dynamics_Explained.md and
    # Inhibition_And_Consolidation_State.md. ARCHIVED (default OFF): in practice the
    # minimal substrate -- excitatory flow-rate + hard-reset inhibition -- is all
    # that's needed, so flow credit is off by default (still togglable in Advanced).
    assembly_flow_credit=False,   # flow-proportional E->I credit on L2I/L1I fire
    # No down-weighting of the E->I "assembly evidence" synapses: keep the credit
    # (contributors still climb to self-sufficiency) but zero the decay term so a
    # non-contributing L2E->L2I synapse is NOT pushed toward the floor (1). With the
    # old 0.5 decay, training one pattern sank every OTHER pattern's L2E->L2I to the
    # floor, so on a pattern switch the new winner's L2E->L2I was too weak to fire
    # L2I -> no lateral inhibition and competition stalled. (Applies to L1I too.)
    assembly_decay_frac=0.0,
    signed_depression=False,      # superseded by the unified signed rule
    homeostasis=False,
    # L2I hard-reset losers (canonical, default ON;
    # L2_Hard_Reset_Competitive_Depression_Spec.md): once L2I declares a winner,
    # every non-winning L2E receives an UNWEIGHTED competitive-reset event -- its
    # pre-reset charge is captured for competitive depression, then its membrane is
    # clamped back to rest so losers no longer start the next race ahead. There is
    # no learned L2I->L2E gate magnitude; the reset is binary and complete.
    # hard_reset_clear_traces also zeroes the current traces so no residual flow
    # refills the membrane.
    l2i_hard_reset_losers=True,
    hard_reset_clear_traces=True,
    # ARCHIVED: the inhibitory DIFFERENTIATING (turnover) rule is retained only for
    # generic weighted inhibitory synapses and old experiments. Active L2 competition
    # has no L2I->L2E gate, so this setting is inert for the dashboard L2E pool.
    inhibitory_delta_rule=False,
    # SWAP + NEUTER: the trace-based flow-rate delivery is GONE from the active model
    # and chunked charge (K=20) is the new timing mechanism. Instead of rate-limiting
    # charge ARRIVAL through a decaying current trace, deliver each step's L1->L2E
    # feedforward drive in 20 equal chunks (weight_ji/20 per active synapse) WITHIN one
    # frozen outer timestep, re-running the argmax WTA after each chunk and stopping at
    # the first threshold-crosser -- "who would have won as the charge trickles in?".
    # excitatory_flow_rate is passed False here for clarity, but the engine ALSO pins
    # it (and the other trace/flow flags) off in _build, so it can't be re-enabled via
    # config; its dashboard toggle is hidden. The flow-rate/current-trace CODE remains
    # in place (reversible) but is dead on the active path. See SimulationEngine._build.
    excitatory_flow_rate=False,
    l2_charge_chunks=20,
    # L2I delivers charge INSTANTLY (flow off for L2I only): a trained L2E->L2I
    # synapse (weight == L2I threshold) then fires L2I from a SINGLE spike -- the
    # single-source relay. Under flow, one spike's charge spreads over decaying
    # steps while L2I leaks, peaking at only ~0.6x the weight, so no single spike
    # could ever cross (the weight is capped at threshold). L2E keeps flow for its
    # own overshoot control; this override is L2I-only.
    l2i_excitatory_flow_rate=False,
    # (The learned L2I->L2E gate is gone -- L2 competition is the unweighted hard
    # reset + competitive depression above. l2_gate_eq_frac / l2_gate_eta are no
    # longer set here; the constructor still accepts them for old experiments but
    # they build no gate in SimulationEngine.)
    refractory=0,                 # inhibition regulates frequency, not a hard lockout
    # Capacity rule: per-afferent cap = thr/3 so three strong active afferents reach
    # threshold (3-pixel lines); positive floor = 1; each I threshold = its E's / 3.
    l2e_weight_cap_frac=1 / 3,
    pos_weight_floor=1,
    l2i_threshold_frac=1 / 3,     # L2I threshold = threshold_l2 / 3
    l1i_threshold_frac=1.0,       # L1I threshold = L2I threshold
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
# Tunable-parameter spec the frontend renders as sliders/toggles. Each entry
# drives one control and its help text; "kind" is "range" or "toggle".
CONFIG_SPEC = [
    {"key": "signed_spike_learning", "label": "Signed-spike learning (minimal)", "kind": "toggle",
     "desc": "Minimal local feedforward rule: on fire, active inputs (+1) potentiate "
             "and inactive inputs (-1) depress via dw=eta*p*(1-(w/w_cap)^2)*signal, "
             "no weight budget. Replaces the confidence/OFF-depression/budget stack. "
             "Run with those OFF, l2e_budget OFF, and refractory=0 for the minimal "
             "experiment."},
    {"key": "structural_free_energy", "label": "Structural free-energy gate", "kind": "toggle",
     "desc": "L2E only. Scale the signed-spike learning rate by a STRUCTURAL maturity "
             "brake instead of the voltage term p: gate = max(eta_floor, 1 - "
             "clamp(sum_positive_afferents/theta, 0, 1)). Under-built neurons stay "
             "plastic; a specialist whose excitatory support already covers a "
             "threshold crossing slows down and resists being reshaped on later "
             "patterns. Input/voltage/rival-independent. OFF = signed rule uses p."},
    {"key": "structural_fe_eta_floor", "label": "Structural FE eta_floor", "kind": "range",
     "min": 0.0, "max": 0.2, "step": 0.01,
     "desc": "Plasticity floor for a fully mature L2E neuron (sum>=theta): its eta is "
             "never scaled below this fraction of the base rate, so no gate freezes "
             "hard. 0 = full freeze at maturity. Only used when the structural "
             "free-energy gate is on."},
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
     "min": 1, "max": 32, "step": 1,
     "desc": "The temporal-race model: deliver each step's L1->L2E feedforward drive "
             "in K equal chunks within one frozen timestep, re-running the argmax WTA "
             "after each chunk and stopping at the first threshold-crosser. This asks "
             "'who would have won as the charge trickles in?' -- the earliest strong "
             "responder wins before rivals pile up charge. K=1 is the un-chunked "
             "baseline (whole volley lands at once); the default is K=20."},
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
             "when enabled. OFF (default) uses the trainable threshold-integrating "
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
             "Supports clean one-owner-per-pattern tiling; turning it off can collapse competition "
             "(dead neurons, no clear winners). Kept ON."},
    {"key": "l2e_lr_frac", "label": "L2E learning rate", "kind": "range",
     "min": 0.005, "max": 0.1, "step": 0.005,
     "desc": "Feedforward potentiation speed for L2E. Higher = faster, sharper RFs "
             "but noisier competition."},
    {"key": "l2e_init_mode", "label": "Balanced L2E init", "kind": "toggle",
     "desc": "ON = task-independent balanced feedforward init: narrow "
             "jitter, then row/column-normalized (Sinkhorn) so every L2E starts with "
             "equal total incoming weight and every pixel equal total outgoing weight "
             "(mean 125). A FAIR developmental start -- no neuron or pixel privileged, "
             "no task structure. OFF (default) = unconstrained legacy-wide "
             "Uniform(50,200) initialization."},
    {"key": "l2e_init_jitter", "label": "Balanced-init jitter (eps)", "kind": "range",
     "min": 0.0, "max": 0.2, "step": 0.005,
     "desc": "Jitter for the balanced init: Z[j,i] ~ Uniform(1-eps, 1+eps) before "
             "balancing. eps=0 -> exactly uniform (perfect symmetry -- competition "
             "cannot break it without a perturbation); eps>0 -> small UNBIASED "
             "differences the learning/competition rules must amplify. Only used when "
             "balanced init is on."},
    {"key": "confidence_consolidation", "label": "Confidence consolidation", "kind": "toggle",
     "desc": "Mature gates learn slower and resist depression (protects specialists). "
             "Also gates signed depression via (1 - C)."},
    {"key": "loser_depression", "label": "Competitive depression", "kind": "toggle",
     "desc": "On an L2I hard-reset event (default ON), each losing L2E depresses only "
             "the POSITIVE feedforward weights whose L1E sources participated in its "
             "losing response (weight>0 AND input spiked), via the shared bounded "
             "kernel with direction -1 and gain scaled by the loser's own pre-reset "
             "charge p_loss = clamp(V_pre/theta, 0, 1). OFF pixels are never touched. "
             "The rate is the L2E's learning_rate; there is no learned inhibitory "
             "magnitude. OFF still hard-resets losers, it just skips the depression."},
    {"key": "leak_enabled", "label": "L2E leak enabled", "kind": "toggle",
     "desc": "Controls membrane leak for the L2 excitatory population only. OFF "
             "(default) "
             "makes every L2E neuron a pure integrator; it does not change L2I."},
    {"key": "leak_l2", "label": "L2 leak", "kind": "range",
     "min": 0.001, "max": 0.05, "step": 0.001,
     "desc": "Fraction of L2E potential that decays per step when L2E leak is enabled."},
    {"key": "l2i_leak_enabled", "label": "L2I leak enabled", "kind": "toggle",
     "desc": "Controls the shared L2 inhibitory neuron's dedicated fast membrane "
             "leak independently of L2E. OFF (default) makes L2I a pure integrator between "
             "its own spike resets."},
    {"key": "l1i_leak_enabled", "label": "L1I leak enabled", "kind": "toggle",
     "desc": "Controls membrane leak for the trainable L1 inhibitory accumulators. "
             "OFF (default) preserves accumulated L2E feedback charge until each "
             "L1I fires; immediate-relay mode does not use this accumulation."},
]

# Dashboard clutter control: the panel exposes every tunable, but most are inert
# under the current default path (signed-spike + chunked-charge race) or belong to
# parked experiments. Keep the ACTIVE experiment controls on the main panel;
# everything else renders under a collapsed "Advanced" disclosure in the frontend.
# All keys stay fully settable (apply/reset send every control), so reproducibility
# is preserved -- this only reorganizes visibility. See the structural-FE prompt's
# "Dashboard Config Cleanup" section for the rationale behind the split.
_MAIN_CONFIG_KEYS = {
    "signed_spike_learning", "l2_charge_chunks", "distance_weighting",
    "l2e_init_mode", "l2e_init_jitter",
    "event_driven", "refractory", "l2e_lr_frac", "l1i_immediate_relay", "leak_enabled",
    "l2i_leak_enabled", "l1i_leak_enabled", "leak_l2",
    # Competitive depression is the canonical ablation switch for the L2 hard-reset
    # event (spec Section 8), so it lives on the main panel.
    "loser_depression",
}
# l2_charge_chunks (K) is the MAIN timing knob: the flow-rate current-trace path it
# replaced is neutered/hidden (see _HIDDEN_CONFIG_KEYS and _build). event_driven stays
# on the main panel (default ON -- one argmax winner per step). The model is the
# minimal loop -- accumulate -> fire -> learn -> inhibit -> chunk.
# loser_depression / eta_loss were archived to the Advanced panel (default OFF) --
# an imposed "punish the loser" rule that doesn't fit the local free-energy model.
# ARCHIVED as hidden (below): the structural free-energy gate is now the BAKED-IN
# learning rule (p = 1 - sum_w+/theta, engine default structural_free_energy=True), so
# it is no longer a user toggle; the inhibitory DIFFERENTIATING gate is off (legacy
# uniform saturating gate) and its turnover knobs go with it.
# NEUTERED (see SimulationEngine._build): trace-based flow-rate delivery is pinned
# permanently OFF, so its toggles no longer control anything. Hide them from the
# config panel entirely rather than showing dead switches. The spec entries stay
# defined (reversible) but are filtered out of what the dashboard is served.
_HIDDEN_CONFIG_KEYS = {
    "excitatory_flow_rate", "exc_trace_decay", "exc_trace_normalized",
    "inhibitory_flow_rate", "inh_trace_decay", "inh_trace_normalized",
    "assembly_flow_credit", "assembly_decay_frac",
    # Structural FE gate archived: baked in as the learning rule, not a toggle.
    "structural_free_energy", "structural_fe_eta_floor",
    # Learned L2I->L2E gate removed (L2_Hard_Reset spec): the inhibitory
    # differentiating (turnover) gate and its params no longer configure anything.
    "inhibitory_delta_rule", "inhibitory_eta_up", "inhibitory_eta_down",
    "inhibitory_p_max",
    # eta_loss is not used by the canonical competitive-depression rule (the rate is
    # the L2E's own learning_rate). Removed from the served config; still accepted by
    # apply_config for old harnesses, but inert on the active path.
    "eta_loss",
}
CONFIG_SPEC = [s for s in CONFIG_SPEC if s["key"] not in _HIDDEN_CONFIG_KEYS]

for _spec in CONFIG_SPEC:
    # advanced := not a primary control (archived/inert/diagnostic). Main entries
    # are explicitly advanced=False so the frontend can rely on the key existing.
    _spec["advanced"] = _spec["key"] not in _MAIN_CONFIG_KEYS


class ConfigBody(BaseModel):
    overrides: dict


def _current_config():
    p = engine.params
    values = {s["key"]: p.get(s["key"]) for s in CONFIG_SPEC}
    # l2e_init_mode is a string param ('balanced'|'legacy_wide') surfaced as a bool
    # TOGGLE (on == balanced); apply_config maps the bool back. Translate here so the
    # toggle reflects the real mode.
    if "l2e_init_mode" in values:
        values["l2e_init_mode"] = (values["l2e_init_mode"] == "balanced")
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
