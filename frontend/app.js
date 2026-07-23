// Application orchestrator. Owns the shared store, wires the websocket to the
// renderer / inspector / charts / controls, and keeps the top bar in sync.
//
// Live WebSocket frames and recorded replay frames pass through ONE shared pair
// of application functions -- applyTopology() and applyDynamic() -- so there is
// no forked renderer. The read-only replay player (replay_player.js) drives those
// same functions from a loaded replay.snn.jsonl file; see docs/REPLAY_PLAYER.md.

import { WSClient } from './websocket.js';
import { NeuronRenderer } from './renderer.js';
import { Inspector } from './inspector.js';
import { Charts } from './charts.js';
import { Controls } from './controls.js';
import { ReceptiveFields } from './receptive.js';
import { Raster } from './raster.js';
import { ChargeChart } from './charge.js';
import { WeightsChart } from './weights.js';
import { Editor } from './editor.js';
import { ReplayPlayer } from './replay_player.js';
import { weightsBeforePos, advanceWeights } from './replay.js';

// History window shared with the raster/charge/weights views (their own HISTORY
// caps). A replay backward-seek rebuilds at most this many frames, never from zero.
const HISTORY = 1500;

// True while the read-only replay player owns the display. Guards two things:
// (1) live WebSocket topology/dynamic messages are ignored for display, and
// (2) every UI-driven mutation POST is refused, so replay can never mutate the
// live engine even via a control that was missed by the visual disable pass.
let replayActive = false;
function setReplayActive(b) { replayActive = b; api.replayActive = b; }

const api = {
  replayActive: false,
  async post(path, body) {
    if (this.replayActive) { console.warn('replay mode: blocked live mutation', path); return; }
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
  confidence: new Map(),    // synapse id -> current confidence (L2E gates)
  stateById: new Map(),     // id -> latest dynamic neuron state
  patternVectors: {},
};

const renderer = new NeuronRenderer(document.getElementById('scene'), { onSelect: select });
const inspector = new Inspector(store);
const charts = new Charts(store);
const controls = new Controls(store, renderer, api);
const receptive = new ReceptiveFields(store, api);
const raster = new Raster(store);
const chargeChart = new ChargeChart(store);
const weightsChart = new WeightsChart(store);
const editor = new Editor(store);

function select(id) { inspector.select(id); renderer.select(id); weightsChart.setTarget(id); }

// ---- FPS (simulation frames received per second) --------------------------
let frameStamps = [];
function tickFps() {
  const now = performance.now();
  frameStamps.push(now);
  while (frameStamps.length && now - frameStamps[0] > 1000) frameStamps.shift();
  return frameStamps.length;
}

// ---- shared application seam (live + replay) ------------------------------
// Rebuild every view from a topology payload. Live topology broadcasts and the
// replay header topology both flow through here; a replay exit re-applies the
// authoritative live topology the same way.
function applyTopology(topo) {
  store.topology = topo;
  store.meta = new Map(topo.neurons.map(n => [n.id, n]));
  store.weights = new Map(topo.synapses.map(s => [s.id, s.weight ?? 0]));
  store.confidence = new Map(topo.synapses.filter(s => s.confidence != null).map(s => [s.id, s.confidence]));
  store.patternVectors = topo.pattern_vectors || {};
  // A new topology can have different ids -- drop any selection that no longer resolves.
  if (inspector.id && !store.meta.has(inspector.id)) { inspector.reset(); renderer.select(null); }
  renderer.build(topo);
  charts.buildStatic(topo);
  controls.onTopology(topo);
  receptive.build();
  raster.build(topo);
  chargeChart.build(topo);
  weightsChart.build();
}

// Apply one dynamic frame forward, updating every view + inspector exactly as the
// live path always has. Used by the live WebSocket and by replay play/step-forward.
// The top bar is updated by the caller (fps for live, frame/phase for replay).
function applyDynamic(dyn) {
  store.dynamic = dyn;
  store.stateById = new Map(dyn.neurons.map(n => [n.id, n]));
  for (const c of dyn.changed_synapses || []) store.weights.set(c.id, c.weight);
  renderer.update(dyn);
  charts.update(dyn, 0);
  receptive.update(dyn);
  raster.update(dyn);
  chargeChart.update(dyn);
  weightsChart.update(dyn);
  inspector.refresh();
  controls.onDynamic(dyn);
}

// Truthful backward/random seek: rebuild only the bounded history window ending
// at frame position `pos`, then show that frame on the non-history views. Weights
// are reconstructed from the nearest checkpoint -- never carried over from a
// previously viewed future frame. Rendering is coalesced: each history view's
// update() schedules a single animation-frame draw regardless of window size.
function replayBulkSeek(replay, pos) {
  const frames = replay.frames;
  const w0 = Math.max(0, pos - HISTORY + 1);

  // Seed with the canonical weights just before the window, then advance frame by
  // frame -- applying changed_synapses and snapping to any recorded checkpoint --
  // so each history point holds the authoritative reconstructed weights, never a
  // future frame's. store.weights ends at reconstructWeightsAt(pos).
  const weights = weightsBeforePos(replay, w0);
  raster.reset(); chargeChart.reset(); weightsChart.reset();
  for (let p = w0; p <= pos; p++) {
    const dyn = frames[p].dynamic;
    advanceWeights(replay, weights, p);
    store.weights = weights;
    raster.update(dyn);
    chargeChart.update(dyn);
    weightsChart.update(dyn);
  }
  const dyn = frames[pos].dynamic;
  store.dynamic = dyn;
  store.stateById = new Map(dyn.neurons.map(n => [n.id, n]));
  renderer.setWeights(store.weights);
  renderer.update(dyn);
  charts.update(dyn, 0);
  receptive.update(dyn);
  inspector.refresh();
  controls.onDynamic(dyn);
}

// ---- websocket message handling -------------------------------------------
function onMessage(msg) {
  if (replayActive) return;   // replay owns the display; live frames are ignored
  if (msg.type === 'topology') {
    applyTopology(msg.data);
  } else if (msg.type === 'dynamic') {
    const dyn = msg.data;
    applyDynamic(dyn);
    updateTopbar(dyn, tickFps());
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

// ---- read-only replay player ----------------------------------------------
const player = new ReplayPlayer({
  applyTopology,
  applyDynamic,
  bulkSeek: replayBulkSeek,
  updateTopbar,
  setReplayActive,
  // Engine calls the player makes directly (bypassing the mutation guard): pause
  // the hidden live sim once on entry; fetch authoritative live state on exit.
  pauseLive: () => fetch('/api/pause', { method: 'POST' }).catch(() => {}),
  fetchLiveState: async () => (await fetch('/api/state')).json(),
});
