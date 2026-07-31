// Regression tests for the node-brightness computation.
//
// Run: node --test tests/renderer.glow.test.mjs
//
// Why this file exists: a NEST replay legitimately omits `activation` and `freq` (they were
// never sampled). The renderer multiplied them directly, so `undefined * 0.9` produced NaN,
// `Math.max` propagated it, and `emissiveIntensity = NaN` rendered EVERY node black. The
// topology and spikes were all correct -- the scene was just unlit.
//
// `glowFor` is imported from a tiny pure export rather than by loading the renderer, so
// this needs no DOM, no WebGL and no three.js runtime.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));

// Extract and evaluate just the pure helper: importing renderer.js would pull in three.js
// and its addons, which are browser-resolved bare specifiers.
const SRC = readFileSync(join(HERE, '..', 'frontend', 'renderer.js'), 'utf-8');
const START = SRC.indexOf('export function glowFor');
assert.ok(START > 0, 'glowFor must remain an exported pure function');
const BODY = SRC.slice(START, SRC.indexOf('\n}', START) + 2).replace('export ', '');
const glowFor = new Function(`${BODY}; return glowFor;`)();

test('a fully unknown state yields a finite, dark glow rather than NaN', () => {
  const g = glowFor(undefined, undefined, 0);
  assert.ok(Number.isFinite(g), `glow must be finite, got ${g}`);
  assert.equal(g, 0);
});

test('null and NaN inputs are also handled', () => {
  for (const bad of [null, NaN, undefined]) {
    assert.ok(Number.isFinite(glowFor(bad, bad, bad)), `failed for ${String(bad)}`);
  }
});

test('a recorded spike still lights the node when charge is unknown', () => {
  // This is the NEST-replay case: no activation, no freq, but the spike IS recorded.
  const g = glowFor(undefined, undefined, 1.0);
  assert.ok(g > 1.0, `a spiking node must be bright, got ${g}`);
  assert.ok(Number.isFinite(g));
});

test('known charge drives brightness when it is available', () => {
  assert.ok(glowFor(1.0, undefined, 0) > glowFor(0.1, undefined, 0));
  assert.ok(glowFor(undefined, 1.0, 0) > glowFor(undefined, 0.1, 0));
});

test('charge and rate combine by max, and pulse adds on top', () => {
  assert.equal(glowFor(1.0, 0, 0), 0.5);
  assert.equal(glowFor(0, 1.0, 0), 0.9);
  assert.equal(glowFor(0, 0, 1.0), 1.6);
  assert.equal(glowFor(1.0, 1.0, 1.0), 0.9 + 1.6);
});

test('every neuron of the committed NEST fixture produces a finite glow', () => {
  const lines = readFileSync(
    join(HERE, 'fixtures', 'nest_replay_fixture.snn.jsonl'), 'utf-8')
    .split('\n').filter(Boolean).map(l => JSON.parse(l));
  const frames = lines.filter(r => r.record === 'frame');
  assert.ok(frames.length > 0);
  let checked = 0;
  for (const f of frames) {
    for (const n of f.dynamic.neurons) {
      const g = glowFor(n.activation, n.freq, n.spiked ? 1 : 0);
      assert.ok(Number.isFinite(g),
        `node ${n.id} at tick ${f.timestep} produced a non-finite glow (${g})`);
      checked++;
    }
  }
  assert.ok(checked > 1000, `expected many nodes checked, got ${checked}`);
});
