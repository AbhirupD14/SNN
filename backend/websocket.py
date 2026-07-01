"""
Websocket connection management and the simulation streaming loop.

ConnectionManager tracks connected clients and broadcasts JSON messages.
SimulationRunner owns the run/pause flag and speed, and -- while running --
advances the engine one timestep at a time and broadcasts the dynamic state to
every connected client. Everything runs in the single uvicorn event loop, so
engine.step() is never re-entered concurrently.
"""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from .simulation import SimulationEngine
from .serializer import dynamic_message


class ConnectionManager:
    def __init__(self):
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class SimulationRunner:
    def __init__(self, engine: SimulationEngine, manager: ConnectionManager):
        self.engine = engine
        self.manager = manager
        self.running = False
        self.speed = 12.0          # target timesteps per second while running
        self._task: asyncio.Task | None = None

    def start_loop(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def _loop(self):
        while True:
            if self.running and self.manager.active:
                self.engine.step()
                await self.manager.broadcast(
                    dynamic_message(self.engine, self.running, self.speed))
                await asyncio.sleep(1.0 / max(self.speed, 0.5))
            else:
                await asyncio.sleep(0.05)

    async def broadcast_dynamic(self):
        await self.manager.broadcast(
            dynamic_message(self.engine, self.running, self.speed))
