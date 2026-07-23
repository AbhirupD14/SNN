// Pure, DOM-free parser + weight-reconstruction for the headless recorder's
// self-contained `replay.snn.jsonl` artifact (schema `snn.replay`, version 1;
// see experiments/replay_recorder.py and docs/REPLAY_RECORDER.md).
//
// This module never touches the network, the live engine, or the DOM. It only
// validates untrusted replay text and answers "what were the live weights at
// frame N?". It is unit-tested under `node --test` (tests/replay.parser.test.mjs);
// the interactive player (replay_player.js) is a thin DOM layer on top of it.
//
// SECURITY: the replay file is untrusted. We parse JSON only (never eval), reject
// anything malformed before the caller enters replay mode, and hand back plain
// data the player renders through text-safe DOM APIs.

export const REPLAY_SCHEMA_NAME = 'snn.replay';
export const SUPPORTED_SCHEMA_VERSION = 1;

const REC_HEADER = 'header';
const REC_MARKER = 'marker';
const REC_FRAME = 'frame';
const REC_CHECKPOINT = 'weight_checkpoint';
const REC_RESULT = 'result';
const KNOWN_RECORDS = new Set([REC_HEADER, REC_MARKER, REC_FRAME, REC_CHECKPOINT, REC_RESULT]);

// A parse failure the caller can surface verbatim. The player shows `.message`.
export class ReplayError extends Error {
  constructor(message) { super(message); this.name = 'ReplayError'; }
}

function isFiniteNumber(v) { return typeof v === 'number' && Number.isFinite(v); }

// JSON.parse turns an out-of-range literal like 1e999 into Infinity, so a finite
// check on every value we actually consume is meaningful, not redundant.
function requireFinite(v, where) {
  if (!isFiniteNumber(v)) throw new ReplayError(`non-finite or non-numeric value at ${where}: ${JSON.stringify(v)}`);
  return v;
}

// ------------------------------------------------------------------- top-level
// Parse and fully validate a replay file's text. Throws ReplayError on ANY
// problem so the caller can decline to enter replay without partial state.
export function parseReplay(text) {
  if (typeof text !== 'string' || text.trim() === '')
    throw new ReplayError('replay file is empty');

  const lines = text.split(/\r?\n/);
  const records = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    let rec;
    try { rec = JSON.parse(line); }
    catch (e) { throw new ReplayError(`malformed JSON on line ${i + 1}: ${e.message}`); }
    if (rec === null || typeof rec !== 'object' || Array.isArray(rec))
      throw new ReplayError(`line ${i + 1} is not a JSON object`);
    records.push(rec);
  }
  if (!records.length) throw new ReplayError('replay file has no records');

  // ---- header: exactly one, first ----
  const header = records[0];
  if (header.record !== REC_HEADER)
    throw new ReplayError(`first record must be a "${REC_HEADER}", got "${header.record}"`);
  if (header.schema !== REPLAY_SCHEMA_NAME)
    throw new ReplayError(`unsupported schema "${header.schema}" (expected "${REPLAY_SCHEMA_NAME}")`);
  if (!Number.isInteger(header.schema_version))
    throw new ReplayError(`schema_version must be an integer, got ${JSON.stringify(header.schema_version)}`);
  if (header.schema_version !== SUPPORTED_SCHEMA_VERSION)
    throw new ReplayError(
      `unsupported schema version ${header.schema_version} (this player supports ${SUPPORTED_SCHEMA_VERSION})`);

  const topology = header.topology;
  if (!topology || typeof topology !== 'object' || !Array.isArray(topology.neurons) || !Array.isArray(topology.synapses))
    throw new ReplayError('header topology is missing neurons/synapses arrays');

  const { neuronIds, synapseIds } = validateTopology(topology);
  const initialWeights = {};
  for (const s of topology.synapses)
    if (s.weight != null) initialWeights[s.id] = requireFinite(s.weight, `header weight ${s.id}`);

  // ---- remaining records ----
  const frames = [];
  const markers = [];
  const checkpoints = [];
  let result = null;
  let lastFrameIndex = -Infinity;
  let lastTimestep = -Infinity;

  for (let i = 1; i < records.length; i++) {
    const rec = records[i];
    const kind = rec.record;
    if (!KNOWN_RECORDS.has(kind)) throw new ReplayError(`unknown record type "${kind}" (record ${i + 1})`);
    if (kind === REC_HEADER) throw new ReplayError(`duplicate header at record ${i + 1}`);

    if (kind === REC_FRAME) {
      const fi = rec.frame_index, ts = rec.timestep;
      if (!Number.isInteger(fi)) throw new ReplayError(`frame ${i + 1} has non-integer frame_index`);
      if (!Number.isInteger(ts)) throw new ReplayError(`frame ${i + 1} has non-integer timestep`);
      if (fi <= lastFrameIndex)
        throw new ReplayError(`non-monotonic frame_index at record ${i + 1}: ${fi} after ${lastFrameIndex}`);
      if (ts <= lastTimestep)
        throw new ReplayError(`non-monotonic timestep at record ${i + 1}: ${ts} after ${lastTimestep}`);
      lastFrameIndex = fi; lastTimestep = ts;
      frames.push(validateFrame(rec, neuronIds, synapseIds, i + 1));
    } else if (kind === REC_CHECKPOINT) {
      checkpoints.push(validateCheckpoint(rec, synapseIds, i + 1));
    } else if (kind === REC_MARKER) {
      markers.push({
        kind: String(rec.kind ?? ''),
        frame_index: Number.isInteger(rec.frame_index) ? rec.frame_index : -1,
        timestep: Number.isInteger(rec.timestep) ? rec.timestep : 0,
        annotation: rec.annotation || null,
        data: rec.data && typeof rec.data === 'object' ? rec.data : {},
      });
    } else if (kind === REC_RESULT) {
      result = rec;
    }
  }

  if (!frames.length) throw new ReplayError('replay file contains no frames');
  checkpoints.sort((a, b) => a.frameIndex - b.frameIndex);

  const frameIndexToPos = new Map();
  frames.forEach((f, pos) => frameIndexToPos.set(f.frameIndex, pos));
  const checkpointByFrameIndex = new Map();
  for (const cp of checkpoints) checkpointByFrameIndex.set(cp.frameIndex, cp.weights);

  return {
    schemaVersion: header.schema_version,
    header,
    topology,
    neuronIds,
    synapseIds,
    initialWeights,
    frames,
    frameIndexToPos,
    checkpoints,
    checkpointByFrameIndex,
    markers,
    result,
    meta: {
      runId: header.run_id ?? null,
      experiment: header.experiment ?? null,
      createdUtc: header.created_utc ?? null,
      seed: header.seed ?? null,
      topologyName: header.topology_name ?? null,
      preset: header.preset ?? null,
      conditions: header.conditions || {},
      recording: header.recording || {},
    },
  };
}

// ------------------------------------------------------------------ validators
function validateTopology(topology) {
  const neuronIds = new Set();
  for (const n of topology.neurons) {
    if (!n || typeof n.id !== 'string') throw new ReplayError('a neuron is missing a string id');
    if (neuronIds.has(n.id)) throw new ReplayError(`duplicate neuron id "${n.id}"`);
    neuronIds.add(n.id);
    if (!Array.isArray(n.pos) || n.pos.length !== 3)
      throw new ReplayError(`neuron "${n.id}" needs a pos [x,y,z]`);
    n.pos.forEach((c, k) => requireFinite(c, `neuron ${n.id} pos[${k}]`));
    if (typeof n.layer !== 'string' || typeof n.type !== 'string')
      throw new ReplayError(`neuron "${n.id}" needs string layer and type`);
  }
  const synapseIds = new Set();
  for (const s of topology.synapses) {
    if (!s || typeof s.id !== 'string') throw new ReplayError('a synapse is missing a string id');
    if (synapseIds.has(s.id)) throw new ReplayError(`duplicate synapse id "${s.id}"`);
    synapseIds.add(s.id);
    if (typeof s.source !== 'string' || typeof s.target !== 'string')
      throw new ReplayError(`synapse "${s.id}" needs string source and target`);
    if (!neuronIds.has(s.source) || !neuronIds.has(s.target))
      throw new ReplayError(`synapse "${s.id}" references an unknown neuron`);
    if (s.weight != null) requireFinite(s.weight, `synapse ${s.id} weight`);
  }
  return { neuronIds, synapseIds };
}

function validateFrame(rec, neuronIds, synapseIds, lineNo) {
  const dyn = rec.dynamic;
  if (!dyn || typeof dyn !== 'object')
    throw new ReplayError(`frame at record ${lineNo} is missing its "dynamic" payload`);
  if (!Array.isArray(dyn.neurons))
    throw new ReplayError(`frame at record ${lineNo} is missing dynamic.neurons`);
  if (!Number.isInteger(dyn.timestep) && !Number.isInteger(rec.timestep))
    throw new ReplayError(`frame at record ${lineNo} is missing dynamic.timestep`);
  for (const n of dyn.neurons) {
    if (!n || typeof n.id !== 'string' || !neuronIds.has(n.id))
      throw new ReplayError(`frame at record ${lineNo} has a neuron with an unknown id`);
    if (n.potential != null) requireFinite(n.potential, `frame ${lineNo} neuron ${n.id} potential`);
    if (n.activation != null) requireFinite(n.activation, `frame ${lineNo} neuron ${n.id} activation`);
  }
  const changed = Array.isArray(dyn.changed_synapses) ? dyn.changed_synapses : [];
  for (const c of changed) {
    if (!c || typeof c.id !== 'string' || !synapseIds.has(c.id))
      throw new ReplayError(`frame at record ${lineNo} changed an unknown synapse id "${c && c.id}"`);
    requireFinite(c.weight, `frame ${lineNo} changed_synapses ${c.id} weight`);
  }
  return {
    frameIndex: rec.frame_index,
    timestep: rec.timestep,
    recordEvery: Number.isInteger(rec.record_every) ? rec.record_every : 1,
    annotation: rec.annotation || { phase: null, pattern: null, tags: [], notes: null },
    dynamic: dyn,
    changed,
  };
}

function validateCheckpoint(rec, synapseIds, lineNo) {
  const weights = rec.weights;
  if (!weights || typeof weights !== 'object' || Array.isArray(weights))
    throw new ReplayError(`weight_checkpoint at record ${lineNo} is missing its weights map`);
  const out = {};
  for (const [id, w] of Object.entries(weights)) {
    if (!synapseIds.has(id))
      throw new ReplayError(`weight_checkpoint at record ${lineNo} references unknown synapse "${id}"`);
    out[id] = requireFinite(w, `checkpoint ${lineNo} weight ${id}`);
  }
  return { frameIndex: Number.isInteger(rec.frame_index) ? rec.frame_index : -1, timestep: rec.timestep, weights: out };
}

// -------------------------------------------------------------- reconstruction
// Live weights as of `frameIndex`, from the nearest preceding checkpoint (or the
// header's initial weights) plus every intervening frame's changed_synapses.
// Mirrors experiments/replay_recorder.py::reconstruct_weights_at. Pure and
// order-independent, so it supports arbitrary backward / random seeking without
// retaining weights from a previously viewed future frame.
export function reconstructWeightsAt(replay, frameIndex) {
  let baseFrame = -1;
  let base = replay.initialWeights;
  for (const cp of replay.checkpoints) {
    if (cp.frameIndex <= frameIndex && cp.frameIndex >= baseFrame) { baseFrame = cp.frameIndex; base = cp.weights; }
  }
  const out = new Map();
  for (const [id, w] of Object.entries(base)) out.set(id, Number(w));
  for (const f of replay.frames) {
    if (f.frameIndex > baseFrame && f.frameIndex <= frameIndex)
      for (const c of f.changed) out.set(c.id, Number(c.weight));
  }
  return out;
}

// Live weights immediately BEFORE the frame at position `pos` (i.e. the canonical
// base the frame at `pos` builds on). Used to seed a bounded history-window
// rebuild that starts mid-run.
export function weightsBeforePos(replay, pos) {
  if (pos <= 0) return new Map(Object.entries(replay.initialWeights).map(([k, v]) => [k, Number(v)]));
  return reconstructWeightsAt(replay, replay.frames[pos - 1].frameIndex);
}

// Advance a live-weights Map from the canonical state at position pos-1 to the
// canonical state at position `pos`: apply that frame's changed_synapses, then --
// if a full weight_checkpoint was recorded at that frame -- snap to it (authoritative;
// this corrects the drift that changed_synapses accumulate when record_every > 1).
// The result equals reconstructWeightsAt(replay, frames[pos].frameIndex). Mutates
// and returns `weights`, so a window rebuild is O(window) rather than O(window^2).
export function advanceWeights(replay, weights, pos) {
  const f = replay.frames[pos];
  for (const c of f.changed) weights.set(c.id, Number(c.weight));
  const cp = replay.checkpointByFrameIndex.get(f.frameIndex);
  if (cp) { weights.clear(); for (const [id, w] of Object.entries(cp)) weights.set(id, Number(w)); }
  return weights;
}
