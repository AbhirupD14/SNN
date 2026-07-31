// Tests for the charge-over-time chart's data path: what a replay frame puts into the
// chart's history, and how much of that history survives.
//
// Run: node --test tests/charge.history.test.mjs
//
// Why this file exists: a NEST replay is the only source that omits `activation` on some
// ticks and supplies it on others, and it is the only source that arrives as a FINITE file
// rather than an open-ended live stream. Both properties are handled in `update()`, which
// touches no DOM -- so it is tested directly, with `Object.create` standing in for the
// constructor's element lookups.

import test from 'node:test';
import assert from 'node:assert/strict';
import { ChargeChart } from '../frontend/charge.js';

const TOPO = {
  neurons: [
    { id: 'L1c00E0', type: 'E', layer: 'L1', column_id: 'L1c00', column_role: 'E' },
    { id: 'L1c00E1', type: 'E', layer: 'L1', column_id: 'L1c00', column_role: 'E' },
    { id: 'L1c00C', type: 'E', layer: 'L1', column_id: 'L1c00', column_role: 'C' },
    { id: 'RGC0', type: 'S', layer: 'RGC', column_id: null, column_role: null },
  ],
};

// `update()` and `build()` touch only instance fields; `_schedule()` bails on a missing
// overlay. So the chart can be exercised without a DOM at all.
function chart(limit = 1500) {
  const c = Object.create(ChargeChart.prototype);
  c.limit = limit;
  c.overlay = null;
  c._raf = 0;
  c.build(TOPO);
  return c;
}

// One frame in the exact shape the NEST adapter emits: `{id, spiked}` always, and the
// sampled fields present only on ticks the multimeter actually sampled.
function frame(timestep, sampled) {
  return {
    timestep,
    inhibitory_pulses: [],
    neurons: TOPO.neurons.map(n => (
      sampled && n.id in sampled
        ? { id: n.id, spiked: false, potential: sampled[n.id] * 1000, activation: sampled[n.id] }
        : { id: n.id, spiked: false }
    )),
  };
}

test('sampled charge reaches the chart and is marked available', () => {
  const c = chart();
  c.update(frame(1, { L1c00E0: 0.42, L1c00C: 0.9 }));

  assert.equal(c.charge.length, 1);
  const [chg, have] = [c.charge[0], c.available[0]];
  assert.equal(chg[c.index.get('L1c00E0')], new Float32Array([0.42])[0]);
  assert.equal(have[c.index.get('L1c00E0')], 1);
  assert.equal(have[c.index.get('L1c00C')], 1);
});

test('an unsampled node is UNKNOWN, not zero', () => {
  const c = chart();
  c.update(frame(1, { L1c00E0: 0.42 }));

  // RGC sources carry no model state at all -- no multimeter is attached to a generator.
  // The lane must be hatched, which is `available = 0`, not a zero-height bar.
  assert.equal(c.available[0][c.index.get('RGC0')], 0);
  assert.equal(c.available[0][c.index.get('L1c00E1')], 0);
});

test('a tick with no samples at all still produces a history column', () => {
  // Between multimeter samples a frame carries only `spiked`. It must still advance the
  // timeline, or charge would render against the wrong ticks.
  const c = chart();
  c.update(frame(1, { L1c00E0: 0.5 }));
  c.update(frame(2, null));
  assert.equal(c.charge.length, 2);
  assert.equal(c.available[1][c.index.get('L1c00E0')], 0);
  assert.deepEqual(c.times, [1, 2]);
});

test('the parallel history arrays never desync while trimming', () => {
  // `available` is a separate array from `charge`; if trimming dropped one and not the
  // other, every lane would render the wrong tick's availability.
  const c = chart(5);
  for (let t = 0; t < 40; t++) c.update(frame(t, { L1c00E0: t / 40 }));

  for (const name of ['charge', 'available', 'spike', 'inhibited', 'l1inh', 'l1rest', 'times']) {
    assert.equal(c[name].length, 5, `${name} trimmed to a different length`);
  }
  assert.deepEqual(c.times, [35, 36, 37, 38, 39]);
});

test('an unbounded limit retains every frame of a finite replay', () => {
  // THE replay regression: a loaded artifact is finite and is shown whole. With the live
  // rolling cap still in force, everything before the last 1500 frames vanished from the
  // chart even though the file held it.
  const c = chart(Infinity);
  const n = 5000;
  for (let t = 0; t < n; t++) c.update(frame(t, { L1c00E0: 0.1 }));

  assert.equal(c.charge.length, n);
  assert.equal(c.available.length, n);
  assert.equal(c.times[0], 0, 'the first frame must still be there');
  assert.equal(c.times[n - 1], n - 1);
});

test('setHistoryLimit switches between the live window and the whole file', () => {
  const c = chart(1500);
  c.setHistoryLimit(Infinity);
  for (let t = 0; t < 2000; t++) c.update(frame(t, { L1c00E0: 0.1 }));
  assert.equal(c.charge.length, 2000);

  // Back to live: the cap reapplies on the next update rather than truncating on the spot.
  c.setHistoryLimit(3);
  c.update(frame(2000, { L1c00E0: 0.1 }));
  assert.equal(c.charge.length, 3);
  assert.equal(c.available.length, 3);
});
