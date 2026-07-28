// Pure-model tests for the dashboard receptive-field panel.
// Run: node --test tests/receptive.model.test.mjs
//
// These cover the browser-independent core of frontend/receptive.js: how one topology
// message becomes cards, how each card's grid shape is derived, which targets get a real
// retinal map vs an afferent list, the per-role display reference, and the structural
// sub-threshold flag. The fixtures are REAL topology payloads emitted by the engine
// (tests/fixtures/make_rf_topologies.py), so the panel is exercised against every current
// preset shape -- 3x3, 9x9 and the 9x18 two-tower sheet -- not a hand-written stub.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  buildReceptiveFieldModel, displayReference, ownsInputPixel, subThresholdFlag,
} from '../frontend/receptive.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const TOPOS = JSON.parse(readFileSync(join(HERE, 'fixtures', 'rf_topologies.json'), 'utf-8'));

const cardsById = (m) => new Map(m.cards.map(c => [c.id, c]));

test('an empty or malformed topology yields an empty model, never a throw', () => {
  for (const bad of [null, undefined, {}, { neurons: null }]) {
    const m = buildReceptiveFieldModel(bad);
    assert.deepEqual(m.cards, []);
    assert.deepEqual(m.layers, []);
  }
});

test('input-pixel ownership ignores a downstream cell display tag', () => {
  assert.equal(ownsInputPixel({ pixel: 3, owns_input: true }), true);
  assert.equal(ownsInputPixel({ pixel: 3, type: 'S' }), true);
  assert.equal(ownsInputPixel({ pixel: 3, role: 'source' }), true);
  // an encoder carrying only a DISPLAY grid tag is not a retinal afferent
  assert.equal(ownsInputPixel({ pixel: 3, role: 'encoder' }), false);
  assert.equal(ownsInputPixel({ pixel: null, owns_input: true }), false);
  assert.equal(ownsInputPixel(null), false);
});

test('display reference follows the ceiling that actually binds for the role', () => {
  const p = { threshold_l2: 1000, e_weight_cap_frac: 0.5, relay_weight_cap_frac: 1.0 };
  assert.equal(displayReference(p, 'E').ref, 500);          // theta/2 pattern detector
  assert.equal(displayReference(p, 'competitor').ref, 500);
  assert.equal(displayReference(p, 'Eor').ref, 1000);       // one-afferent relay
  assert.equal(displayReference(p, 'C').ref, 1000);         // C basal
  // with no configured ceiling it falls back to the maturity budget, never to 1
  const q = { threshold_l2: 1000, e_maturity_budget: 1100 };
  assert.equal(displayReference(q, 'E').ref, 1100);
  assert.equal(displayReference({}, 'E').ref, 1.1);
});

test('3x3 preset: every competitor is a retinal map on the whole sheet', () => {
  const m = buildReceptiveFieldModel(TOPOS.rg_direct_cc4);
  assert.equal(m.tiled, false);
  assert.deepEqual(m.inputShape, { rows: 3, cols: 3 });
  assert.ok(m.cards.length > 0);
  for (const c of m.cards) {
    assert.equal(c.kind, 'pixels');
    assert.deepEqual(c.shape, { rows: 3, cols: 3 });
    assert.equal(c.cells.length, 9);
    assert.equal(c.cells.filter(Boolean).length, 9);        // dense 3x3 receptive field
  }
});

test('tiled 9x9: L1 E are patch maps, L2 E / Eor / C are afferent lists', () => {
  const m = buildReceptiveFieldModel(TOPOS.tiled_cc);
  assert.equal(m.tiled, true);
  assert.deepEqual(m.inputShape, { rows: 9, cols: 9 });
  const by = cardsById(m);

  const l1e = by.get('L1c00E0');
  assert.equal(l1e.kind, 'pixels');
  assert.deepEqual(l1e.shape, { rows: 3, cols: 3 });        // its OWN patch, not the 9x9 sheet
  assert.equal(l1e.cells.filter(Boolean).length, 9);

  // an L2 ordinary E sees nine child-column Eor outputs -- not pixels
  const l2e = by.get('L2c00E0');
  assert.equal(l2e.kind, 'afferents');
  assert.equal(l2e.nAfferents, 9);

  // an Eor sees its own local ordinary-E bank
  const eor = by.get('L1c00Eor');
  assert.equal(eor.kind, 'afferents');
  assert.equal(eor.nAfferents, 8);
  assert.equal(eor.role, 'Eor');

  // a coincidence C owns exactly one learned basal; its apicals are unweighted
  const c = by.get('L1c00C');
  assert.equal(c.kind, 'afferents');
  assert.equal(c.nAfferents, 1);
  assert.equal(c.cells[0].label, 'basal');
});

test('two-tower 9x18: patch maps stay 3x3 and both towers are represented', () => {
  const m = buildReceptiveFieldModel(TOPOS.two_tower_composition);
  assert.deepEqual(m.inputShape, { rows: 9, cols: 18 });
  assert.deepEqual(m.layers, ['L1', 'L2', 'L3']);
  const by = cardsById(m);

  // an L1 detector maps its own 3x3 patch even though the sheet is 9x18
  for (const id of ['T0L1c00E0', 'T1L1c22E7']) {
    const c = by.get(id);
    assert.equal(c.kind, 'pixels');
    assert.deepEqual(c.shape, { rows: 3, cols: 3 });
    assert.equal(c.cells.filter(Boolean).length, 9);
  }
  // the two patches are DISJOINT pixels of the one sheet (left field vs right field)
  const left = by.get('T0L1c00E0').cells.filter(Boolean).map(c => c.pixel);
  const right = by.get('T1L1c00E0').cells.filter(Boolean).map(c => c.pixel);
  assert.equal(new Set([...left, ...right]).size, left.length + right.length);
  assert.ok(Math.max(...left) % 18 < 9);
  assert.ok(Math.min(...right) % 18 >= 9);

  // the L3 composition target sees exactly the two tower Eor outputs
  const l3 = by.get('L3c00E0');
  assert.equal(l3.kind, 'afferents');
  assert.equal(l3.nAfferents, 2);
  assert.deepEqual(l3.cells.filter(Boolean).map(c => c.sourceId).sort(),
    ['T0L2c00Eor', 'T1L2c00Eor']);
});

test('every learned edge in the graph appears exactly once across the cards', () => {
  for (const [name, topo] of Object.entries(TOPOS)) {
    const m = buildReceptiveFieldModel(topo);
    const seen = new Set();
    for (const c of m.cards) {
      for (const cell of c.cells) {
        if (!cell) continue;
        assert.equal(seen.has(cell.edgeId), false, `${name}: ${cell.edgeId} shown twice`);
        seen.add(cell.edgeId);
      }
    }
    const learned = topo.synapses
      .filter(s => s.kind === 'feedforward' || s.kind === 'basal_excitation')
      .map(s => s.id);
    assert.deepEqual([...seen].sort(), [...learned].sort(),
      `${name}: cards must cover exactly the learned edges`);
  }
});

test('a pixels card never drops an afferent off its grid', () => {
  for (const [name, topo] of Object.entries(TOPOS)) {
    const m = buildReceptiveFieldModel(topo);
    const afferentCount = new Map();
    for (const s of topo.synapses) {
      if (s.kind !== 'feedforward' && s.kind !== 'basal_excitation') continue;
      afferentCount.set(s.target, (afferentCount.get(s.target) || 0) + 1);
    }
    for (const c of m.cards) {
      assert.equal(c.cells.filter(Boolean).length, afferentCount.get(c.id),
        `${name}: ${c.id} lost afferents in layout`);
      assert.ok(c.cells.length >= c.cells.filter(Boolean).length);
      assert.equal(c.cells.length, c.shape.rows * c.shape.cols);
    }
  }
});

test('cards are grouped by column and ordered E -> Eor -> C', () => {
  const m = buildReceptiveFieldModel(TOPOS.tiled_cc);
  const g = m.groups.find(x => x.key === 'L1c00');
  assert.ok(g, 'expected one group per column');
  const roles = g.cards.map(c => c.role);
  assert.equal(roles.at(-2), 'Eor');
  assert.equal(roles.at(-1), 'C');
  assert.ok(roles.slice(0, -2).every(r => r === 'E'));
  assert.equal(g.label, 'L1 · L1c00');
});

test('sub-threshold flag is a one-volley test, not a fixed top-3 proxy', () => {
  const card = {
    threshold: 1000,
    cells: [{ edgeId: 'a' }, { edgeId: 'b' }, null, { edgeId: 'c' }],
  };
  // two theta/2 afferents reach threshold exactly -> not flagged
  assert.equal(subThresholdFlag(card, (e) => ({ a: 500, b: 500, c: 0 })[e]), false);
  // everything together still short -> flagged
  assert.equal(subThresholdFlag(card, (e) => ({ a: 200, b: 200, c: 200 })[e]), true);
  // a four-afferent cell that only clears threshold using all four is NOT flagged
  // (the old "three strongest" proxy would have called this dead)
  assert.equal(subThresholdFlag(card, (e) => ({ a: 340, b: 340, c: 340 })[e]), false);
  assert.equal(subThresholdFlag({ threshold: 1, cells: [null] }, () => 0), null);
  // a missing weight counts as zero rather than NaN-poisoning the sum
  assert.equal(subThresholdFlag(card, () => undefined), true);
});

test('the model is pure: repeated builds are deep-equal and share no cell objects', () => {
  const a = buildReceptiveFieldModel(TOPOS.two_tower_composition);
  const b = buildReceptiveFieldModel(TOPOS.two_tower_composition);
  assert.deepEqual(a.cards.length, b.cards.length);
  assert.deepEqual(a.cards[0], b.cards[0]);
  assert.notEqual(a.cards[0], b.cards[0]);
  assert.notEqual(a.cards[0].cells, b.cards[0].cells);
});
