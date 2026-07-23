// Pure-parser / weight-reconstruction tests for the dashboard replay player.
// Run: node --test tests/replay.parser.test.mjs
//
// These cover the browser-independent core of frontend/replay.js: schema
// validation, rejection of malformed artifacts before replay mode begins, and
// exact forward/backward/random weight reconstruction. The fixture is generated
// by the REAL recorder (tests/fixtures/make_replay_fixture.py); expected.json is
// the recorder's own reconstruction, so JS and Python must agree off one file.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  parseReplay, reconstructWeightsAt, weightsBeforePos, advanceWeights,
  ReplayError, REPLAY_SCHEMA_NAME, SUPPORTED_SCHEMA_VERSION,
} from '../frontend/replay.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = readFileSync(join(HERE, 'fixtures', 'replay_fixture.snn.jsonl'), 'utf-8');
const EXPECTED = JSON.parse(readFileSync(join(HERE, 'fixtures', 'replay_fixture.expected.json'), 'utf-8'));

// ---------------------------------------------------------------- happy path
test('parses supported header/frame/marker/checkpoint/result records', () => {
  const r = parseReplay(FIXTURE);
  assert.equal(r.schemaVersion, SUPPORTED_SCHEMA_VERSION);
  assert.equal(r.header.schema, REPLAY_SCHEMA_NAME);
  assert.equal(r.frames.length, Object.keys(EXPECTED).length);
  assert.ok(r.frames.length > 0);
  assert.ok(r.markers.length >= 1);
  assert.ok(r.checkpoints.length >= 1);
  assert.ok(r.result, 'final result record parsed');
});

test('exposes recorded metadata (seed, condition, phase/pattern, markers)', () => {
  const r = parseReplay(FIXTURE);
  assert.equal(r.meta.seed, 7);
  assert.equal(r.meta.experiment, 'replay_fixture');
  assert.equal(r.meta.conditions.hierarchical_feedback, 'intact');
  // annotations carried on frames
  assert.ok(r.frames[0].annotation);
  assert.ok(['present-A', 'present-B'].includes(r.frames[0].annotation.phase));
  // marker kinds are the caller's; the recorder never invents them
  const kinds = r.markers.map(m => m.kind);
  assert.ok(kinds.includes('phase_start'));
});

// ------------------------------------------------------ weight reconstruction
test('forward reconstruction matches the recorder at every frame', () => {
  const r = parseReplay(FIXTURE);
  for (const f of r.frames) {
    const got = reconstructWeightsAt(r, f.frameIndex);
    const exp = EXPECTED[String(f.frameIndex)];
    assertWeightsEqual(got, exp, `frame ${f.frameIndex}`);
  }
});

test('backward / random-order seeks reconstruct the exact recorded weights', () => {
  const r = parseReplay(FIXTURE);
  const order = r.frames.map(f => f.frameIndex);
  // reverse
  for (const fi of [...order].reverse())
    assertWeightsEqual(reconstructWeightsAt(r, fi), EXPECTED[String(fi)], `rev ${fi}`);
  // shuffled (deterministic)
  const shuffled = [...order].sort((a, b) => ((a * 7 + 3) % 11) - ((b * 7 + 3) % 11));
  for (const fi of shuffled)
    assertWeightsEqual(reconstructWeightsAt(r, fi), EXPECTED[String(fi)], `shuf ${fi}`);
});

test('weightsBeforePos is the base a forward apply of that frame builds on', () => {
  const r = parseReplay(FIXTURE);
  // pos 0 -> header initial weights
  const before0 = weightsBeforePos(r, 0);
  assertWeightsEqual(before0, r.initialWeights, 'before pos 0');
  // pos k -> reconstruction at frame k-1
  for (let k = 1; k < r.frames.length; k++) {
    const before = weightsBeforePos(r, k);
    const prev = reconstructWeightsAt(r, r.frames[k - 1].frameIndex);
    assertWeightsEqual(before, mapToObj(prev), `before pos ${k}`);
  }
});

test('bounded-window rebuild reproduces the reconstructed weights (seek path)', () => {
  // Mirrors app.js replayBulkSeek(): seed with weightsBeforePos(w0), then advance
  // frame by frame (changed_synapses + checkpoint snap) across the window
  // [w0..pos]. The result must equal reconstructWeightsAt(pos) for ANY window
  // size -- including windows that start before an intervening checkpoint, where a
  // record_every>1 recording makes plain changed_synapses accumulation lossy.
  const r = parseReplay(FIXTURE);
  const n = r.frames.length;
  for (const H of [1, 2, 4, n]) {
    for (let pos = 0; pos < n; pos++) {
      const w0 = Math.max(0, pos - H + 1);
      const w = weightsBeforePos(r, w0);              // Map
      for (let p = w0; p <= pos; p++) advanceWeights(r, w, p);
      assertWeightsEqual(w, EXPECTED[String(r.frames[pos].frameIndex)], `H=${H} pos=${pos}`);
    }
  }
});

test('advanceWeights across a checkpoint snaps to the authoritative snapshot', () => {
  // Frame 5 in the fixture carries a weight_checkpoint. Plain changed_synapses
  // accumulation (no snap) drifts there under record_every>1; advanceWeights must
  // land exactly on reconstructWeightsAt.
  const r = parseReplay(FIXTURE);
  assert.ok(r.checkpoints.length >= 1, 'fixture has checkpoints');
  const cpFi = r.checkpoints[0].frameIndex;
  const pos = r.frameIndexToPos.get(cpFi);
  const w = weightsBeforePos(r, pos);
  advanceWeights(r, w, pos);
  assertWeightsEqual(w, EXPECTED[String(cpFi)], `checkpoint frame ${cpFi}`);
});

// ------------------------------------------------------------------ rejects
const HEADER = {
  record: 'header', schema: REPLAY_SCHEMA_NAME, schema_version: SUPPORTED_SCHEMA_VERSION,
  run_id: 'r', experiment: 'e', seed: 1, topology_name: 't', preset: 'p',
  conditions: { hierarchical_feedback: 'intact' }, recording: { record_every: 1, checkpoint_every: 10 },
  neuron_order: ['n0', 'n1'], synapse_order: ['s0'],
  topology: {
    neurons: [
      { id: 'n0', pos: [0, 0, 0], layer: 'L1', type: 'E', role: 'source', threshold: 1 },
      { id: 'n1', pos: [1, 0, 0], layer: 'L2', type: 'E', role: 'competitor', threshold: 1 },
    ],
    synapses: [{ id: 's0', source: 'n0', target: 'n1', kind: 'feedforward', weight: 10 }],
  },
};
const frame = (fi, ts, extra = {}) => ({
  record: 'frame', frame_index: fi, timestep: ts, record_every: 1,
  annotation: { phase: 'p', pattern: 'x', tags: [], notes: null },
  dynamic: {
    timestep: ts, running: false,
    neurons: [{ id: 'n0', potential: 0, activation: 0, spiked: false, freq: 0 },
              { id: 'n1', potential: 0, activation: 0, spiked: false, freq: 0 }],
    changed_synapses: [], ...extra,
  },
});
const lines = (...recs) => recs.map(r => JSON.stringify(r)).join('\n');

function rejects(label, text, matcher) {
  test(`rejects: ${label}`, () => {
    assert.throws(() => parseReplay(text), (e) => {
      assert.ok(e instanceof ReplayError, `${label} should be a ReplayError, got ${e}`);
      if (matcher) assert.match(e.message, matcher);
      return true;
    });
  });
}

rejects('empty file', '');
rejects('malformed JSON line', lines(HEADER) + '\n{not json', /malformed JSON/);
rejects('first record not a header',
  lines(frame(0, 1), HEADER), /first record must be a "header"/);
rejects('wrong schema name',
  lines({ ...HEADER, schema: 'other.thing' }), /unsupported schema/);
rejects('unsupported schema version',
  lines({ ...HEADER, schema_version: SUPPORTED_SCHEMA_VERSION + 1 }, frame(0, 1)),
  /unsupported schema version/);
rejects('non-integer schema version',
  lines({ ...HEADER, schema_version: 1.5 }), /schema_version must be an integer/);
rejects('duplicate neuron id', lines({
  ...HEADER, topology: { ...HEADER.topology, neurons: [
    HEADER.topology.neurons[0], { ...HEADER.topology.neurons[1], id: 'n0' }] } }),
  /duplicate neuron id/);
rejects('duplicate synapse id', lines({
  ...HEADER, topology: { ...HEADER.topology,
    synapses: [HEADER.topology.synapses[0], { ...HEADER.topology.synapses[0] }] } }),
  /duplicate synapse id/);
rejects('synapse references unknown neuron', lines({
  ...HEADER, topology: { ...HEADER.topology,
    synapses: [{ id: 's9', source: 'n0', target: 'ZZ', kind: 'feedforward', weight: 1 }] } }),
  /unknown neuron/);
rejects('unknown record type',
  lines(HEADER, frame(0, 1), { record: 'bogus' }), /unknown record type/);
rejects('changed_synapses unknown id',
  lines(HEADER, frame(0, 1, { changed_synapses: [{ id: 'NOPE', weight: 1 }] })),
  /unknown synapse id/);
rejects('checkpoint unknown synapse id',
  lines(HEADER, frame(0, 1), { record: 'weight_checkpoint', frame_index: 0, timestep: 1,
    weights: { NOPE: 1 } }), /unknown synapse/);
rejects('non-monotonic frame_index',
  lines(HEADER, frame(1, 1), frame(1, 2)), /non-monotonic frame_index/);
rejects('non-monotonic timestep',
  lines(HEADER, frame(0, 5), frame(1, 5)), /non-monotonic timestep/);
// JSON cannot encode Infinity (JSON.stringify makes it null), so a genuinely
// non-finite value only arises from a raw file token like 1e999, which
// JSON.parse turns into Infinity. Craft that literal in the text.
rejects('non-finite weight (1e999 -> Infinity)',
  JSON.stringify(HEADER).replace('"weight":10', '"weight":1e999'),
  /non-finite/);
rejects('missing dynamic.neurons',
  lines(HEADER, { record: 'frame', frame_index: 0, timestep: 1,
    dynamic: { timestep: 1, changed_synapses: [] } }), /dynamic\.neurons/);
rejects('no frames', lines(HEADER), /no frames/);

// ---------------------------------------------------------------- helpers
function mapToObj(m) { const o = {}; for (const [k, v] of m) o[k] = v; return o; }
function assertWeightsEqual(gotMap, expObj, where) {
  const got = gotMap instanceof Map ? mapToObj(gotMap) : gotMap;
  const gk = Object.keys(got).sort(), ek = Object.keys(expObj).sort();
  assert.deepEqual(gk, ek, `${where}: same synapse id set`);
  for (const k of ek)
    assert.ok(Math.abs(got[k] - expObj[k]) < 1e-9, `${where}: weight ${k} ${got[k]} != ${expObj[k]}`);
}
