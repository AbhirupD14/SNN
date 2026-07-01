"""
Serialization layer: converts SimulationEngine state into the JSON envelopes
used by the REST API and the websocket protocol.

Protocol envelopes (all messages are {"type": ..., "data": ...}):

    {"type": "topology", "data": <static topology>}   -- sent once on connect / reset
    {"type": "dynamic",  "data": <per-timestep state>} -- streamed every frame

Static topology (neurons, synapses, layout, params) is sent only when it
changes; the per-timestep "dynamic" payload carries activations, firing state,
recently changed synapse weights, statistics and the rolling event log.
"""

from __future__ import annotations

from .simulation import SimulationEngine


def topology_message(engine: SimulationEngine) -> dict:
    return {"type": "topology", "data": engine.topology()}


def dynamic_message(engine: SimulationEngine, running: bool, speed: float) -> dict:
    data = engine.dynamic_state()
    data["running"] = running
    data["speed"] = speed
    return {"type": "dynamic", "data": data}


def full_state(engine: SimulationEngine, running: bool, speed: float) -> dict:
    """Snapshot used by GET /state: both static and dynamic in one response."""
    dynamic = engine.dynamic_state()
    dynamic["running"] = running
    dynamic["speed"] = speed
    return {"topology": engine.topology(), "dynamic": dynamic}
