// Application orchestrator. Owns the shared store, wires the websocket to the
// renderer / inspector / charts / controls, and keeps the top bar in sync.

import { WSClient } from './websocket.js';
import { NeuronRenderer } from './renderer.js';
import { Inspector } from './inspector.js';
import { Charts } from './charts.js';
import { Controls } from './controls.js';

const api = {
  async post(path, body) {
    try {
      await fetch(path, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (e) { console.warn('POST failed', path, e); }
  },
};

const store = {
  topology: null,
  meta: new Map(),          // id -> static meta
  weights: new Map(),       // synapse id -> current weight
  stateById: new Map(),     // id -> latest dynamic neuron state
  patternVectors: {},
};

const renderer = new NeuronRenderer(document.getElementById('scene'), { onSelect: select });
const inspector = new Inspector(store);
const charts = new Charts(store);
const controls = new Controls(store, renderer, api);

function select(id) { inspector.select(id); renderer.select(id); }

// ---- FPS (simulation frames received per second) --------------------------
let frameStamps = [];
function tickFps() {
  const now = performance.now();
  frameStamps.push(now);
  while (frameStamps.length && now - frameStamps[0] > 1000) frameStamps.shift();
  return frameStamps.length;
}

// ---- websocket message handling -------------------------------------------
function onMessage(msg) {
  if (msg.type === 'topology') {
    const topo = msg.data;
    store.topology = topo;
    store.meta = new Map(topo.neurons.map(n => [n.id, n]));
    store.weights = new Map(topo.synapses.map(s => [s.id, s.weight ?? 0]));
    store.patternVectors = topo.pattern_vectors || {};
    renderer.build(topo);
    charts.buildStatic(topo);
    controls.onTopology(topo);
  } else if (msg.type === 'dynamic') {
    const dyn = msg.data;
    store.dynamic = dyn;
    store.stateById = new Map(dyn.neurons.map(n => [n.id, n]));
    for (const c of dyn.changed_synapses || []) store.weights.set(c.id, c.weight);
    const fps = tickFps();
    renderer.update(dyn);
    charts.update(dyn, fps);
    inspector.refresh();   // re-renders if a neuron was already selected
    controls.onDynamic(dyn);
    updateTopbar(dyn, fps);
  }
}

// ---- top bar --------------------------------------------------------------
const el = id => document.getElementById(id);
function updateTopbar(dyn, fps) {
  el('st-status').textContent = dyn.running ? 'Running' : 'Paused';
  el('st-status').style.color = dyn.running ? 'var(--ok)' : 'var(--txt-1)';
  el('st-timestep').textContent = dyn.timestep;
  el('st-speed').textContent = (dyn.speed ?? 0).toFixed(0) + ' /s';
  el('st-winner').textContent = dyn.winner || '—';
  el('st-fps').textContent = fps;
}

function onStatus(state) {
  const c = el('st-conn');
  c.className = 'conn ' + (state === 'online' ? 'online' : state === 'offline' ? 'offline' : '');
  el('st-conn-label').textContent =
    state === 'online' ? 'connected' : state === 'offline' ? 'disconnected' : 'connecting…';
}

// ---- go -------------------------------------------------------------------
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WSClient(`${proto}://${location.host}/ws`, { onMessage, onStatus });
ws.connect();
