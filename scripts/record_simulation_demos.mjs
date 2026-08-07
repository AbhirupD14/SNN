#!/usr/bin/env node
/**
 * Deterministic Playwright/Chromium recording workflow for FOUR presentation-ready SNN
 * dashboard demonstration videos (no combined montage):
 *
 *   1. uncertainty_frequency_demo  -- frequency/cadence as an OBSERVABLE candidate
 *                                     uncertainty signal (rg_coincidence)
 *   2. continuous_learning_demo    -- one live network learns four patterns in sequence
 *                                     with no reset between phases (rg_direct_cc4)
 *   3. unbiased_learning_demo      -- resistance to training-exposure bias after a
 *                                     6000-boundary dwell (rg_direct_cc4)
 *   4. scaling_demo                -- 9x9 tiled hierarchy, then the 9x18 two-tower
 *                                     hierarchy (two chapters in one recording)
 *
 * Usage:
 *   npm run record-demo
 *   npm run record-demo -- --only uncertainty
 *   npm run record-demo -- --only continuous,unbiased --port 8010
 *
 * Every scientific number shown on screen is read either from the LIVE run (topology counts,
 * owner ids, timesteps, winner tallies) or from the deterministic headless preflight in
 * scripts/preflight_demo_measurements.py. Nothing is hard-coded into a caption.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

import {
  DashboardApi, SeedFile, UvicornServer, portInUse, findFreePort, sleep,
  toolVersion, convertToMp4, probe, decodeCheck, extractFrame, pngSize, brightSpots,
} from './demo_support.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);
const ASSETS = path.join(ROOT, 'presentation_assets');
const RAW = path.join(ASSETS, 'raw_webm');
const WORK = path.join(ASSETS, '.playwright_video');
const PREFLIGHT_JSON = path.join(ASSETS, 'preflight_measurements.json');
const PYTHON = path.join(ROOT, '.venv', 'bin', 'python');

const SEED = 1;
const VIDEO_SIZE = { width: 1920, height: 1080 };
const FPS = 30;
const RUN_SPEED = 120;               // steps/s cap accepted by POST /api/speed
// Per-graph run-loop speeds the browser can actually render, measured on this machine
// (see recording_notes.md). Driving faster than these only queues frames.
const SPEED_TILED = 25;              // 191 nodes / 1052 edges
const SPEED_TWO_TOWER = 6;           // 393 nodes / 2162 edges, ~72 KiB per dynamic frame
const HOLD = { title: 3000, beat: 2500, end: 3000 };

// ---------------------------------------------------------------- engine configs
// Videos 2 and 3 use the dual FE/FES confirmation candidate measured for rg_direct_cc4 in
// experiments/dual_fe_cc4_consolidation.py: B = 5 (the engine default dual_fe_B) with the
// LR multiplier m = 100, i.e. eta = 0.01*100 = 1.0 and c_eta = 0.005*100 = 0.5.
const CC4_CONFIG = {
  topology: 'rg_direct_cc4', dual_fe_fes: true,
  eta: 1.0, c_eta: 0.5, leak_rate: 0.0, refractory_steps: 0, input_period: 0,
};
// Video 1 uses the small 3x3 coincidence/feedback circuit at the dashboard's fast-maturation
// rates so the cadence transition completes inside a recordable number of boundaries.
const COINCIDENCE_CONFIG = {
  topology: 'rg_coincidence', dual_fe_fes: true,
  eta: 4.0, c_eta: 16.0, leak_rate: 0.0, refractory_steps: 0, input_period: 0,
};
const SCALING_BASE = {
  dual_fe_fes: true, eta: 4.0, c_eta: 16.0, leak_rate: 0.0, refractory_steps: 0,
  input_period: 0,
};

const UNCERTAINTY_QUALIFICATION =
  'The circuit exposes a frequency/cadence signal associated with confirmed bottom-up and ' +
  'top-down agreement. It is an observable substrate for uncertainty — not a calibrated ' +
  'confidence score, not a probability. Cadence also depends on causal-loop latency and ' +
  'input pacing.';
const TWO_TOWER_QUALIFICATION =
  'Scope: this demonstrates graph and lower-encoder scaling. It does NOT claim the L3 ' +
  'composition column distinguishes V / A / 7 — the two scalar Eor streams make the tested ' +
  'L3 inputs non-identifiable.';

// ============================================================ page-side overlay
// Injected by Playwright as recording-only presentation chrome. It touches nothing the
// simulation reads: no engine parameter, no control, no view state.
const OVERLAY_CSS = `
  * { cursor: none !important; }
  ::-webkit-scrollbar { width: 0 !important; height: 0 !important; }
  #demo-layer {
    position: fixed; inset: 0; z-index: 2147483000; pointer-events: none;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  }
  #demo-title {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 18px;
    background: radial-gradient(ellipse at 50% 45%, #131a27 0%, #0b0e14 70%);
    opacity: 0; transition: opacity .45s ease;
  }
  #demo-title.on { opacity: 1; }
  #demo-title .eyebrow {
    font-size: 15px; letter-spacing: .30em; text-transform: uppercase;
    color: #5eead4; font-weight: 600;
  }
  #demo-title .main {
    font-size: 52px; font-weight: 650; color: #e7ecf5; letter-spacing: -.015em;
    max-width: 1280px; text-align: center; line-height: 1.15;
  }
  #demo-title .sub {
    font-size: 20px; color: #9aa6bd; max-width: 1100px; text-align: center;
    line-height: 1.5;
  }
  #demo-title .rule { width: 132px; height: 2px; background: #5eead4; opacity: .55; }
  #demo-chapter {
    position: absolute; top: 66px; left: 50%; transform: translateX(-50%); display: none;
    padding: 7px 14px; border-radius: 8px;
    background: rgba(17,21,31,.90); border: 1px solid #232a3a;
    color: #e7ecf5; font-size: 15px; font-weight: 600; letter-spacing: .01em;
    box-shadow: 0 6px 24px rgba(0,0,0,.35);
  }
  #demo-chapter .idx { color: #5eead4; margin-right: 9px; font-variant-numeric: tabular-nums; }
  #demo-caption {
    position: absolute; left: 18px; right: 18px; bottom: 16px; display: none;
    padding: 12px 16px; border-radius: 10px;
    background: rgba(11,14,20,.93); border: 1px solid #232a3a;
    color: #e7ecf5; font-size: 17px; line-height: 1.45;
    box-shadow: 0 6px 24px rgba(0,0,0,.45);
  }
  #demo-caption .lead { color: #5eead4; font-weight: 650; margin-right: 8px; }
  #demo-caption .dim { color: #9aa6bd; }
  #demo-caption .num { color: #ffce5c; font-variant-numeric: tabular-nums; font-weight: 600; }
  #demo-caption .warn { color: #fbbf24; }
  #demo-badge {
    position: absolute; top: 68px; right: 18px; display: none;
    padding: 7px 13px; border-radius: 8px;
    background: rgba(29,36,51,.94); border: 1px solid #2f3a4f;
    color: #ffce5c; font-size: 15px; font-weight: 650;
    font-variant-numeric: tabular-nums;
  }
`;

const OVERLAY_JS = `
(() => {
  const build = () => {
    if (document.getElementById('demo-layer')) return;
    const layer = document.createElement('div');
    layer.id = 'demo-layer';
    layer.innerHTML =
      '<div id="demo-title"><div class="eyebrow"></div><div class="main"></div>' +
      '<div class="rule"></div><div class="sub"></div></div>' +
      '<div id="demo-chapter"></div><div id="demo-badge"></div><div id="demo-caption"></div>';
    document.body.appendChild(layer);
  };
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', build);
  else build();

  const el = (id) => document.getElementById(id);
  window.__demo = {
    title(eyebrow, main, sub) {
      build();
      const t = el('demo-title');
      t.querySelector('.eyebrow').textContent = eyebrow || '';
      t.querySelector('.main').textContent = main || '';
      t.querySelector('.sub').textContent = sub || '';
      t.style.display = 'flex';
      requestAnimationFrame(() => t.classList.add('on'));
    },
    hideTitle() {
      const t = el('demo-title');
      if (!t) return;
      t.classList.remove('on');
      setTimeout(() => { t.style.display = 'none'; }, 500);
    },
    chapter(idx, text) {
      build();
      const c = el('demo-chapter');
      c.innerHTML = (idx ? '<span class="idx">' + idx + '</span>' : '') + (text || '');
      c.style.display = text ? 'block' : 'none';
    },
    caption(html) {
      build();
      const c = el('demo-caption');
      c.innerHTML = html || '';
      c.style.display = html ? 'block' : 'none';
    },
    badge(text) {
      build();
      const b = el('demo-badge');
      b.textContent = text || '';
      b.style.display = text ? 'block' : 'none';
    },
  };

  // ---- frame tap -----------------------------------------------------------
  // Records what the BROWSER actually rendered, frame by frame, so post-run evidence
  // (winner tallies, eligible presentations, turnover latency) comes from the recorded
  // run itself rather than from a separate headless replay. Read-only: it observes
  // WebSocket messages and never sends anything.
  // One entry per DISTINCT engine timestep. The dashboard also broadcasts a dynamic frame
  // on every control call (pattern change, start, pause), which repeats the timestep it was
  // already at; counting those would inflate the boundary tallies the captions quote.
  window.__demoTap = { frames: [], on: false, label: null, lastT: null, duplicates: 0 };
  const Native = window.WebSocket;
  window.WebSocket = function (...args) {
    const ws = new Native(...args);
    ws.addEventListener('message', (ev) => {
      const tap = window.__demoTap;
      if (!tap.on) return;
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (!msg || msg.type !== 'dynamic') return;
      const d = msg.data || {};
      if (d.timestep === tap.lastT) { tap.duplicates++; return; }
      tap.lastT = d.timestep;
      tap.frames.push({ t: d.timestep, w: d.winner ?? null, label: tap.label });
    });
    return ws;
  };
  window.WebSocket.prototype = Native.prototype;
  Object.assign(window.WebSocket, Native);
})();
`;

// ================================================================== utilities
function parseArgs(argv) {
  const out = { only: null, port: 8000, strictPort: false, keepWork: false, preflight: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--only') out.only = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (a === '--port') out.port = Number(argv[++i]);
    else if (a === '--strict-port') out.strictPort = true;
    else if (a === '--keep-work') out.keepWork = true;
    else if (a === '--preflight') out.preflight = true;
    else if (a === '--help' || a === '-h') { printHelp(); process.exit(0); }
    else throw new Error(`unknown argument ${a}`);
  }
  return out;
}

function printHelp() {
  console.log(`record_simulation_demos.mjs

  --only a,b,c    record a subset: uncertainty | continuous | unbiased | scaling
  --port N        preferred dashboard port (default 8000)
  --strict-port   fail instead of falling back to a free port when N is occupied
  --preflight     run scripts/preflight_demo_measurements.py first
  --keep-work     keep the intermediate Playwright video work directory
`);
}

const fmt = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : String(n));
const num = (v) => `<span class="num">${v}</span>`;

class Recorder {
  constructor(page, api, pre) {
    this.page = page;
    this.api = api;
    this.pre = pre;
    this.notes = [];
    this.evidence = {};
  }

  note(text) { this.notes.push(text); console.log(`      · ${text}`); }

  title(eyebrow, main, sub, ms = HOLD.title) {
    return this.page.evaluate(
      ([e, m, s]) => window.__demo.title(e, m, s), [eyebrow, main, sub],
    ).then(() => sleep(ms));
  }

  hideTitle() {
    return this.page.evaluate(() => window.__demo.hideTitle()).then(() => sleep(550));
  }

  chapter(idx, text) {
    return this.page.evaluate(([i, t]) => window.__demo.chapter(i, t), [idx, text]);
  }

  caption(html) { return this.page.evaluate((h) => window.__demo.caption(h), html); }

  badge(text) { return this.page.evaluate((t) => window.__demo.badge(t), text); }

  tap(on, label = null) {
    return this.page.evaluate(([o, l]) => {
      window.__demoTap.on = o;
      window.__demoTap.label = l;
    }, [on, label]);
  }

  tapLabel(label) {
    return this.page.evaluate((l) => { window.__demoTap.label = l; }, label);
  }

  frames() { return this.page.evaluate(() => window.__demoTap.frames); }

  /** Wait until the browser has rendered a dynamic frame at or past `timestep`. This is the
   *  readiness predicate the recording waits on -- never a bare sleep. */
  async waitRendered(timestep, timeout = 600000) {
    await this.page.waitForFunction(
      (t) => Number(document.getElementById('st-timestep')?.textContent || 0) >= t,
      timestep, { timeout, polling: 100 },
    );
  }

  /** Run the live engine forward `n` boundaries at 1x and wait for the browser to show it.
   *
   *  `speed` must be set to something the BROWSER can actually render for the active graph.
   *  The 393-node two-tower payload is ~72 KiB per frame and Chromium sustains only ~5-6
   *  frames/s of it; requesting 120 there just queues frames the viewer sees as a stutter
   *  followed by a long catch-up. Measured sustainable rates are in recording_notes.md. */
  async runVisible(n, speed = RUN_SPEED) {
    const before = await this.api.state();
    const target = before.dynamic.timestep + n;
    await this.api.speed(speed);
    await this.api.start();
    await this.api.waitFor((s) => s.dynamic.timestep >= target,
      { what: `timestep >= ${target}`, timeout: 1800000 });
    await this.api.pause();
    await this.waitRendered(target, 1800000);
    return target;
  }

  /** Compress an uninformative stretch: the boundaries genuinely execute (paused runner,
   *  POST /api/step), they are simply not shown at 1x. The on-screen badge and the live
   *  timestep readout make the compression explicit rather than hidden. */
  async fastForward(n, badgePrefix) {
    await this.api.pause();
    const before = (await this.api.state()).dynamic.timestep;
    const secs = await this.api.fastForward(n, async (done) => {
      if (done % 500 === 0) {
        await this.badge(`${badgePrefix} ▸▸ ${fmt(before + done)} / ${fmt(before + n)} boundaries`);
      }
    });
    const after = (await this.api.state()).dynamic.timestep;
    await this.waitRendered(after);
    await this.badge('');
    this.note(`fast-forward ${fmt(n)} boundaries in ${secs.toFixed(1)}s ` +
      `(${Math.round(n / Math.max(secs, 0.001))} boundaries/s), timestep ${before} -> ${after}`);
    return after;
  }

  /** These overlays are toggled by the `hidden` ATTRIBUTE, so gate on the attribute rather
   *  than on Playwright visibility (a `[hidden]` selector can never become "visible"). */
  waitOverlay(overlayId, shown, timeout = 30000) {
    return this.page.waitForFunction(
      ([id, want]) => {
        const el = document.getElementById(id);
        return !!el && el.hasAttribute('hidden') === !want;
      }, [overlayId, shown], { timeout, polling: 100 });
  }

  async openView(tab, overlayId) {
    await this.page.click(`.tab[data-tab="${tab}"]`);
    await this.waitOverlay(overlayId, true);
    await sleep(800);
  }

  /** Click a view's zoom control n times. The raster follows the newest boundary while it is
   *  scrolled to the right edge, so zooming in magnifies the most recent cadence. */
  async zoomView(buttonId, times, pause = 700) {
    for (let i = 0; i < times; i++) {
      await this.page.click(`#${buttonId}`);
      await sleep(pause);
    }
  }

  /** Set the 3D scene filter checkboxes to explicit states, e.g.
   *  `{ 'f-weak': false, 'f-l2': true }`. Clicks the real control so the renderer's own
   *  change handler runs; asserts the resulting checked state.
   *
   *  `f-weak` ("Hide weak synapses") defaults to CHECKED and hides every weighted edge below
   *  25% of the weight cap. The recordings turn it OFF so the full connectivity is drawn --
   *  notably the complete 9 L1E x 8 L2E feedforward fan, including the losing competitors'
   *  afferents, which decay under that threshold as one owner consolidates. */
  async setSceneFilters(states) {
    for (const [id, want] of Object.entries(states)) {
      const is = await this.page.$eval(`#${id}`, (el) => el.checked);
      if (is !== want) await this.page.click(`#${id}`);
      const now = await this.page.$eval(`#${id}`, (el) => el.checked);
      if (now !== want) throw new Error(`could not set #${id} to ${want}`);
    }
    // Playwright scrolls a control into view before clicking it, which pushes the pattern
    // grid off the top of the sidebar. Put it back.
    await this.scrollSidebarTop();
    await sleep(450);
  }

  /** Keep the left sidebar pinned to the top so the input pattern grid and the pattern
   *  buttons -- what is actually being presented -- stay on screen. */
  scrollSidebarTop() {
    return this.page.evaluate(() => {
      const s = document.getElementById('sidebar-left');
      if (s) s.scrollTop = 0;
    });
  }

  /** Screenshot the scene and cache the candidate neuron positions.
   *
   *  Deliberately split from `selectNeuron`: the screenshot must be taken while the canvas
   *  is UNOBSTRUCTED, but the clicks derived from it can be dispatched much later, from
   *  behind the title card (the overlay is `pointer-events: none`, so clicks reach the
   *  canvas). Neuron positions do not move between the two, so the cached points stay valid
   *  -- enabling weak synapses only adds edges, and the engine is paused throughout. */
  async captureSelectionSpots() {
    this._sceneBox = await this.page.locator('#scene canvas').boundingBox();
    if (!this._sceneBox) throw new Error('no scene canvas to screenshot');
    const shot = path.join(WORK, 'select-probe.png');
    await this.page.screenshot({ path: shot, clip: this._sceneBox });
    this._spots = await brightSpots(shot);
    await fs.rm(shot, { force: true });
    return this._spots.length;
  }

  /** Select a neuron so the right-hand inspector panel stays populated for the whole video.
   *
   *  There is no DOM control for this: `NeuronRenderer._handleClick` raycasts from the
   *  pointer onto the visible meshes, so a real pointerdown/pointerup on the canvas over a
   *  neuron body is the only way in. Clicks that hit nothing are no-ops in the renderer, so
   *  a miss is harmless.
   *
   *  `prefer` is a PRIORITY order of id prefixes; the scan ends early only on a rank-0 hit,
   *  and otherwise settles on the best id seen within the click/time budget. */
  async selectNeuron(prefer = [], { maxClicks = 60, maxSeconds = 10 } = {}) {
    if (!this._spots) await this.captureSelectionSpots();
    const box = this._sceneBox;
    const spots = this._spots;

    // One probe = one round trip. `_handleClick` reads nothing but clientX/clientY off the
    // event and `inspector.select()` renders synchronously, so dispatching the pointer pair
    // and reading the resulting id inside a single evaluate exercises exactly the same
    // handler path as page.mouse.* -- at a quarter of the latency. That matters: the probe
    // runs behind the title card, and four round trips per candidate held that card for
    // ~16 s on the coincidence graph.
    const probe = (x, y) => this.page.evaluate(([cx, cy]) => {
      const canvas = document.querySelector('#scene canvas');
      if (!canvas) return null;
      const opts = { clientX: cx, clientY: cy, bubbles: true, cancelable: true };
      canvas.dispatchEvent(new PointerEvent('pointerdown', opts));
      canvas.dispatchEvent(new PointerEvent('pointerup', opts));
      const b = document.getElementById('insp-body');
      if (!b || b.hidden) return null;
      return b.querySelector('.insp-id')?.textContent?.trim() || null;
    }, [x, y]);

    // `prefer` is a PRIORITY order, so a lower-priority hit must not end the scan. Gather
    // what the bright points actually resolve to, stopping early only on a top-priority hit,
    // then settle on the best match found.
    const rank = (id) => {
      const i = prefer.findIndex((p) => id.startsWith(p));
      return i === -1 ? prefer.length : i;
    };
    // Time budget: on the 393-node two-tower graph the render loop still makes each probe
    // slow, so settle for the best neuron found so far once the budget is spent -- any
    // populated inspector satisfies the requirement.
    const found = new Map();                       // id -> {x, y}
    let best = null;
    let clicks = 0;
    const deadline = Date.now() + maxSeconds * 1000;
    for (const s of spots) {
      if (clicks >= maxClicks || (best && Date.now() > deadline)) break;
      clicks++;
      const x = box.x + s.x, y = box.y + s.y;
      const id = await probe(x, y);
      if (!id || found.has(id)) continue;
      found.set(id, { x, y });
      if (!best || rank(id) < rank(best)) best = id;
      if (rank(best) === 0) break;                 // cannot do better
    }
    if (!best) {
      throw new Error(`could not select any neuron: probed ${clicks} of ${spots.length} ` +
        'bright points and none resolved to a neuron mesh');
    }
    const at = found.get(best);
    const confirmed = await probe(at.x, at.y);
    if (confirmed !== best) throw new Error(`re-selecting ${best} landed on ${confirmed}`);
    this.note(`inspector selected ${best}` +
      (prefer.length && rank(best) > 0
        ? ` (preferred ${prefer.join(' > ')}; saw ${[...found.keys()].join(', ')})` : ''));
    await this.scrollSidebarTop();
    return best;
  }

  async closeView(closeId, overlayId) {
    await this.page.click(`#${closeId}`);
    await this.waitOverlay(overlayId, false);
    await sleep(600);
  }

  /** Apply an engine configuration and wait for the rebuilt topology to reach the browser.
   *  The browser-side predicate is the topology-sized input grid, which `applyTopology`
   *  rebuilds from every topology broadcast. */
  async configure(overrides, expectNodes) {
    await this.api.pause();
    await this.api.config(overrides);
    const s = await this.api.waitFor((st) => st.topology.neurons.length === expectNodes,
      { what: `topology with ${expectNodes} nodes` });
    const cells = s.topology.tiling
      ? s.topology.tiling.input_shape.rows * s.topology.tiling.input_shape.cols
      : s.topology.grid.rows * s.topology.grid.cols;
    await this.page.waitForFunction(
      (n) => document.querySelectorAll('#pixel-grid .pixel').length === n,
      cells, { timeout: 60000, polling: 100 });
    await sleep(900);
    await this.scrollSidebarTop();
    return s;
  }
}

// ------------------------------------------------------- shared readiness gate
async function waitDashboardReady(page, api, { expectNodes, expectInputCells }) {
  await page.waitForSelector('#st-conn-label', { timeout: 60000 });
  // 1) websocket connected
  await page.waitForFunction(
    () => document.getElementById('st-conn-label')?.textContent.trim() === 'connected',
    null, { timeout: 60000, polling: 100 });
  // 2) GET /api/state serves the expected topology
  await api.waitFor((s) => s.topology.neurons.length === expectNodes,
    { what: `GET /api/state topology with ${expectNodes} nodes` });
  // 3) the scene canvas exists and 4) the topology-sized input grid rendered
  await page.waitForSelector('#scene canvas', { timeout: 60000 });
  await page.waitForFunction(
    (n) => document.querySelectorAll('#pixel-grid .pixel').length === n,
    expectInputCells, { timeout: 60000, polling: 100 });
  // 5) dynamic state has rendered at least once
  await page.waitForFunction(
    () => (document.getElementById('st-status')?.textContent || '—') !== '—',
    null, { timeout: 60000, polling: 100 });
}

// ================================================================== video 1
async function recordUncertainty(rec) {
  const { page, api, pre } = rec;
  const c = pre.cadence;

  // The title card is already up (raised by the driver before setup); configure behind it.
  const st = await rec.configure(COINCIDENCE_CONFIG, 45);
  await api.reset();
  await api.waitFor((s) => s.dynamic.timestep === 0, { what: 'timestep back to 0' });
  await rec.hideTitle();
  await rec.chapter('01', 'Frequency / candidate uncertainty signal');
  await sleep(600);
  const nodes = st.topology.neurons.length;
  const edges = st.topology.synapses.length;
  rec.evidence.topology = { nodes, edges };
  // Counts read from the live graph, so the anatomy captions cannot drift from the topology.
  const kinds = {};
  for (const s of st.topology.synapses) kinds[s.kind] = (kinds[s.kind] || 0) + 1;
  const byLayerRole = (layer, role) =>
    st.topology.neurons.filter((n) => n.layer === layer && n.role === role).length;
  const nL2E = byLayerRole('L2', 'competitor');
  const nL1C = byLayerRole('L1', 'coincidence');
  rec.evidence.anatomy = { kinds, l2_competitors: nL2E, l1_coincidence: nL1C };

  await rec.caption(
    `<span class="lead">rg_coincidence</span> ${num(nodes)} neurons · ${num(edges)} edges · ` +
    `seed ${num(SEED)} · <span class="dim">reset, before learning. “Hide weak synapses” is ` +
    `OFF for this whole recording, so the full connectivity is drawn: ` +
    `${num(kinds.pretrained_excitation)} pretrained RG→L1E · ` +
    `${num(kinds.feedforward)} learned L1E→L2E · ` +
    `${num(kinds.basal_excitation)} learned L1E→L1C basal · ` +
    `${num(kinds.apical_excitation)} unweighted L2E→L1C apical gates · ` +
    `${num(kinds.relay_excitation)} relay · ` +
    `${num(kinds.hard_reset_inhibition)} hard-reset.</span>`);
  await sleep(HOLD.beat + 1800);

  // --- anatomy walk: isolate the two structures that are hard to read in the full graph ---
  await rec.setSceneFilters({ 'f-rg': false, 'f-l1': false });
  await rec.caption(
    `<span class="lead">Layer 2 alone — the ${num(nL2E)}-way winner-take-all</span> ` +
    `<span class="dim">${num(nL2E)} competing ordinary-E cells (role “competitor”) and the ` +
    `ONE shared L2I relay. The WTA is emergent, not a policy: the first L2E to reach ` +
    `threshold fires, its relay drives L2I, and L2I hard-resets all ${num(nL2E)} — cancelling ` +
    `the rest in the same boundary. Every competitor is present and eligible on every ` +
    `boundary; only the winner survives it.</span>`);
  await sleep(6200);

  await rec.setSceneFilters({ 'f-l1': true, 'f-inh': false });
  await rec.caption(
    `<span class="lead">The apical gates</span> ` +
    `<span class="dim">pink: ${num(kinds.apical_excitation)} = ${num(nL2E)} L2E × ` +
    `${num(nL1C)} L1C. Every L2E projects an UNWEIGHTED Boolean apical gate to every ` +
    `coincidence cell — it carries no charge and never learns; it only grants permission. ` +
    `A L1C deposits its learned basal charge exactly when basal evidence and this top-down ` +
    `permission coincide. Inhibitory cells hidden here so the gate fan is legible.</span>`);
  await sleep(6800);

  await rec.setSceneFilters({ 'f-rg': true, 'f-inh': true });
  await rec.caption(
    `<span class="lead">Full circuit restored</span> ` +
    `<span class="dim">all ${num(nodes)} neurons and ${num(edges)} edges, weak synapses ` +
    `shown. Note: this preset has no per-feature gating — that was a separate, removed ` +
    `variant. The gates here are the ${num(kinds.apical_excitation)} apical permissions above ` +
    `plus the ${num(kinds.hard_reset_inhibition)} hard-reset inhibitory edges.</span>`);
  await sleep(HOLD.beat + 1200);

  // --- learning under a sustained named pattern ---
  await api.pattern('row 1');
  await sleep(400);
  await rec.tap(true, 'learning');
  await rec.caption(
    `<span class="lead">Sustained pattern “row 1”</span> ` +
    `<span class="dim">the circuit learns while the stimulus never changes. Watch how often ` +
    `L1E emits per RG volley.</span>`);
  await rec.runVisible(220);
  await api.start();

  // --- move into the Spike Raster: the view that makes the cadence visible ---
  await rec.openView('raster', 'raster-overlay');
  await rec.caption(
    `<span class="lead">Spike Raster</span> <span class="dim">discrete spikes; the pinned ` +
    `left gutter carries a per-neuron firing-rate bar. Early cadence is a grouped burst — ` +
    `L1E emits on ≈${num(c.early_emission_ratio_mean)} of boundaries.</span>`);

  // Run out to the full live history window so ONE raster frame holds the early grouped
  // cadence AND the later locked alternation.
  const target = Math.max(1500, (c.cadence_lock_boundary ?? 900) + 560);
  await api.speed(RUN_SPEED);
  await api.start();
  await api.waitFor((s) => s.dynamic.timestep >= Math.floor(target * 0.55),
    { what: `timestep >= ${Math.floor(target * 0.55)}`, timeout: 900000 });
  await rec.caption(
    `<span class="lead">Cadence tightening</span> <span class="dim">as the coincidence cells ` +
    `mature, confirmed presentations begin to suppress the redundant re-fire.</span>`);
  await api.waitFor((s) => s.dynamic.timestep >= target,
    { what: `timestep >= ${target}`, timeout: 900000 });
  await api.pause();
  await rec.waitRendered(target);
  await rec.tap(false);

  // --- measured evidence, then hold the still history ---
  const frames = await rec.frames();
  rec.evidence.rendered_frames = frames.length;
  await rec.caption(
    `<span class="lead">Measured (seed ${SEED}, headless preflight + this run)</span> ` +
    `L1E emission per RG volley: ${num(c.early_emission_ratio_mean)} over the first ` +
    `${num(c.early_window)} boundaries → ${num(c.late_emission_ratio_mean)} over the last ` +
    `${num(c.late_window)}. Every active L1E alternates fire/silent with no repeat from ` +
    `boundary ${num(c.cadence_lock_boundary)} onward — early <code>${c.sample_early.slice(0, 28)}</code> ` +
    `→ late <code>${c.sample_late.slice(0, 28)}</code>.<br>` +
    `<span class="warn">${UNCERTAINTY_QUALIFICATION}</span>`);
  await sleep(6500);

  // --- zoom the still history so the LATE cadence is unambiguous per boundary ---
  await rec.caption(
    `<span class="lead">Zooming the newest boundaries</span> <span class="dim">the same paused ` +
    `history, magnified: each active L1E now resolves into one spike, one gap, one spike — ` +
    `exactly ${num(c.late_emission_ratio_mean)} emissions per RG volley.</span>`);
  await rec.zoomView('raster-zoom-in', 4);
  await sleep(5000);
  await rec.zoomView('raster-zoom-out', 4, 350);

  // --- back to the network ---
  await rec.closeView('raster-close', 'raster-overlay');
  await rec.caption(
    `<span class="lead">Paused</span> <span class="dim">the learned circuit, held still. ` +
    `Charge / time shows membrane charge, threshold crossings and inhibition; it is not a ` +
    `frequency estimator, so the cadence claim above rests on the raster.</span>`);
  await sleep(HOLD.beat);

  // --- learned weights live in the Receptive Fields view ---
  await rec.openView('rf', 'rf-overlay');
  await rec.caption(
    `<span class="lead">Receptive Fields</span> <span class="dim">the weight visualization: ` +
    `actual learned feedforward weights per cell, read from live engine state.</span>`);
  await sleep(4200);
  await rec.closeView('rf-close', 'rf-overlay');

  // --- back on the live network, structure and learned state together ---
  await rec.scrollSidebarTop();
  await rec.caption(
    `<span class="lead">The learned circuit</span> <span class="dim">back on the live 3D ` +
    `network, paused: ${num(nodes)} neurons and ${num(edges)} edges with weak synapses shown. ` +
    `Edge brightness tracks the learned weight, so the consolidated L1E→L2E afferents stand ` +
    `out against the competitors that lost. The inspector on the right reports the selected ` +
    `cell's live state.</span>`);
  await sleep(5200);

  await rec.caption(
    `<span class="lead">Demonstration 1 — summary</span> a reproducible cadence transition ` +
    `from ${num(c.early_emission_ratio_mean)} to exactly ${num(c.late_emission_ratio_mean)} ` +
    `emissions per volley. <span class="warn">An observable substrate for uncertainty; not a ` +
    `calibrated confidence score.</span>`);
  await sleep(HOLD.end);
  return rec.evidence;
}

// ================================================================== video 2
async function recordContinuous(rec) {
  const { api, pre } = rec;
  const p = pre.continuous;
  const order = p.order;
  const dwell = p.dwell;

  const st = await rec.configure(CC4_CONFIG, 14);
  await api.reset();                       // the ONE reset, at the beginning
  await api.waitFor((s) => s.dynamic.timestep === 0, { what: 'timestep back to 0' });
  await rec.hideTitle();
  await rec.chapter('02', 'Continuous sequential learning');
  await sleep(600);
  const nodes = st.topology.neurons.length;
  const edges = st.topology.synapses.length;
  rec.evidence.topology = { nodes, edges };
  await rec.caption(
    `<span class="lead">rg_direct_cc4</span> ${num(nodes)} neurons · ${num(edges)} edges · ` +
    `seed ${num(SEED)} · dual FE/FES, B=${num(p.B)}, m=${num(p.m)} ` +
    `(η=${num(p.eta)}, C-η=${num(p.c_eta)}) · <span class="dim">four latency-E competitors on ` +
    `one central WTA I. This is the single reset of the run.</span>`);
  await sleep(HOLD.beat);

  await rec.tap(true, null);
  const observed = {};
  const phaseTally = [];
  for (let i = 0; i < order.length; i++) {
    const pattern = order[i];
    await rec.tapLabel(pattern);          // label before the change, so no boundary is misfiled
    await api.pattern(pattern);
    await sleep(300);
    const prior = Object.entries(observed)
      .map(([k, v]) => `${k} → ${v}`).join(' · ');
    await rec.caption(
      `<span class="lead">Phase ${i + 1} / ${order.length} — “${pattern}”</span> ` +
      `${num(dwell)} boundaries. <span class="dim">No reset, no reseed, no config change; ` +
      `the weights learned in earlier phases are still in this network.` +
      (prior ? ` Already owned: ${prior}.` : '') + `</span>`);

    // Phases 3 and 4 run inside the Receptive Fields view so the viewer can watch a NEW
    // owner's weights form while the earlier owners' weights visibly persist.
    if (i === 2) {
      await rec.openView('rf', 'rf-overlay');
      await rec.caption(
        `<span class="lead">Phase ${i + 1} / ${order.length} — “${pattern}”</span> ` +
        `<span class="dim">Receptive Fields: one 3×3 weight grid per competitor, live. Watch a ` +
        `fresh competitor grow this pattern while the earlier owners keep theirs.</span>`);
    }

    await rec.runVisible(dwell);

    const frames = await rec.frames();
    const mine = frames.filter((f) => f.label === pattern && f.w);
    const tally = {};
    for (const f of mine) tally[f.w] = (tally[f.w] || 0) + 1;
    const owner = Object.entries(tally).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
    observed[pattern] = owner;
    phaseTally.push({ pattern, owner, tally, rendered_frames: mine.length });
    rec.note(`phase "${pattern}": observed owner ${owner} (${JSON.stringify(tally)})`);

    await rec.caption(
      `<span class="lead">Phase ${i + 1} complete — “${pattern}”</span> ` +
      `owner <span class="num">${owner}</span> ` +
      `<span class="dim">(modal winner over the ${fmt(mine.length)} dynamic frames this ` +
      `browser rendered during the phase). Assignments so far: ` +
      Object.entries(observed).map(([k, v]) => `${k} → <span class="num">${v}</span>`).join(' · ') +
      `</span>`);
    await sleep(2200);
  }
  await rec.tap(false);

  // End on the paused weight view (phases 3-4 already opened it; this is the guard).
  const rfHidden = await rec.page.$eval('#rf-overlay', (e) => e.hasAttribute('hidden'));
  if (rfHidden) await rec.openView('rf', 'rf-overlay');

  const distinct = new Set(Object.values(observed).filter(Boolean)).size;
  rec.evidence.observed_owners = observed;
  rec.evidence.phase_tally = phaseTally;
  rec.evidence.distinct_owners = distinct;
  rec.evidence.matches_preflight =
    JSON.stringify(observed) === JSON.stringify(p.owner_by_pattern);

  const assignments = Object.entries(observed)
    .map(([k, v]) => `“${k}” → <span class="num">${v}</span>`).join(' &nbsp;·&nbsp; ');
  await rec.caption(
    `<span class="lead">Demonstration 2 — summary (paused)</span> ${assignments}<br>` +
    `<span class="dim">${num(distinct)} distinct owners across ${num(order.length)} patterns, ` +
    `learned sequentially in one network with no reset between phases. Owners were read from ` +
    `the recorded run; the headless preflight at the same seed gives ` +
    `${Object.entries(p.owner_by_pattern).map(([k, v]) => `${k} → ${v}`).join(' · ')} ` +
    `(match: ${rec.evidence.matches_preflight ? 'yes' : 'NO'}), with cold recall ` +
    `${p.recall_consistent ? 'reproducing every assignment' : 'NOT reproducing the assignments'}.</span>`);
  await sleep(HOLD.end + 3500);
  return rec.evidence;
}

// ================================================================== video 3
async function recordUnbiased(rec) {
  const { api, pre } = rec;
  const p = pre.long_dwell;
  const order = p.order;

  const st = await rec.configure(CC4_CONFIG, 14);
  await api.reset();
  await api.waitFor((s) => s.dynamic.timestep === 0, { what: 'timestep back to 0' });
  await rec.hideTitle();
  await rec.chapter('03', 'Long-dwell exposure-bias stress');
  await sleep(600);
  rec.evidence.topology = {
    nodes: st.topology.neurons.length, edges: st.topology.synapses.length,
  };
  await rec.caption(
    `<span class="lead">rg_direct_cc4 · seed ${num(SEED)}</span> dual FE/FES, B=${num(p.B)}, ` +
    `m=${num(p.m)} (η=${num(p.eta)}, C-η=${num(p.c_eta)}) · ` +
    `<span class="dim">the same measured four-competitor configuration as demonstration 2. ` +
    `Inputs stay equal-frequency throughout: no class reweighting, no balanced sampling, no ` +
    `pattern-specific learning rate.</span>`);
  await sleep(HOLD.beat);

  // ---- establish the four-pattern network -------------------------------------
  await rec.tap(true, null);
  await rec.caption(
    `<span class="lead">Step 1 — establish the four-pattern network</span> ` +
    `<span class="dim">${order.map((o) => `“${o}”`).join(', ')} at ${num(p.train_dwell)} ` +
    `boundaries each, exactly as demonstration 2.</span>`);
  for (let i = 0; i < order.length; i++) {
    await rec.tapLabel(`train:${order[i]}`);
    await api.pattern(order[i]);
    await sleep(250);
    await rec.badge(`training “${order[i]}” · ${i + 1}/${order.length}`);
    if (i === 0) {
      await rec.runVisible(200);                                     // seen at 1x
      await rec.fastForward(p.train_dwell - 200, `training “${order[i]}”`);
    } else {
      await rec.fastForward(p.train_dwell, `training “${order[i]}”`);
    }
  }
  await rec.badge('');
  let frames = await rec.frames();
  const trainOwners = {};
  for (const o of order) {
    const t = {};
    for (const f of frames.filter((f) => f.label === `train:${o}` && f.w)) t[f.w] = (t[f.w] || 0) + 1;
    trainOwners[o] = Object.entries(t).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  }
  rec.evidence.trained_owners_rendered = trainOwners;
  rec.note(`trained owners (rendered frames): ${JSON.stringify(trainOwners)}`);
  await rec.caption(
    `<span class="lead">Four-pattern network established</span> ` +
    `<span class="dim">preflight assignment at this seed: ` +
    Object.entries(p.trained_owner_by_pattern).map(([k, v]) => `“${k}” → ${v}`).join(' · ') +
    `. The middle of each training phase was fast-forwarded — the timestep readout in the ` +
    `top bar shows every boundary actually executed.</span>`);
  await sleep(HOLD.beat);

  // ---- the long dwell ----------------------------------------------------------
  await rec.tapLabel('dwell');
  await api.pattern(p.held_pattern);
  await sleep(250);
  const dwellStart = (await api.state()).dynamic.timestep;
  await rec.caption(
    `<span class="lead">Step 2 — long dwell on “${p.held_pattern}”</span> ` +
    `${num(fmt(p.long_dwell))} boundaries of the SAME pattern — ` +
    `${num((p.long_dwell / p.train_dwell).toFixed(1))}× the exposure any other pattern got. ` +
    `<span class="dim">This is the exposure bias the switch must survive.</span>`);
  await rec.runVisible(450);                                          // beginning, at 1x
  await rec.caption(
    `<span class="lead">Long dwell — compressing the uninformative middle</span> ` +
    `<span class="dim">the remaining boundaries execute at full speed with the display ` +
    `paused. Watch the Timestep readout climb: nothing is skipped.</span>`);
  await rec.fastForward(p.long_dwell - 900, `dwell “${p.held_pattern}”`);
  await rec.caption(
    `<span class="lead">Long dwell — final stretch at 1×</span> ` +
    `<span class="dim">back to real time for the end of the dwell.</span>`);
  await rec.runVisible(450);                                          // end, at 1x
  const dwellEnd = (await api.state()).dynamic.timestep;
  rec.evidence.dwell = { start: dwellStart, end: dwellEnd, boundaries: dwellEnd - dwellStart };

  frames = await rec.frames();
  const dwellFrames = frames.filter((f) => f.label === 'dwell' && f.w);
  const dTally = {};
  for (const f of dwellFrames) dTally[f.w] = (dTally[f.w] || 0) + 1;
  const incumbent = Object.entries(dTally).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  rec.evidence.incumbent_rendered = incumbent;
  rec.note(`incumbent after the long dwell: ${incumbent} (${JSON.stringify(dTally)})`);
  await rec.caption(
    `<span class="lead">Incumbent after ${num(fmt(dwellEnd - dwellStart))} boundaries on ` +
    `“${p.held_pattern}”</span> <span class="num">${incumbent}</span> ` +
    `<span class="dim">— dominance ${num(p.long_dwell_phase.tail_200_dominance)} over the ` +
    `final 200 eligible presentations of the dwell (headless preflight, same seed).</span>`);
  await sleep(3200);

  // ---- the switch, with no reset ------------------------------------------------
  await rec.tapLabel('switch');
  await api.pattern(p.switched_to);
  await sleep(200);
  await rec.caption(
    `<span class="lead">Step 3 — switch to “${p.switched_to}”</span> ` +
    `<span class="dim">pattern changed through POST /api/pattern only. No reset, no reseed, ` +
    `no rebuild, no learning-rate change. The incumbent keeps every weight it just spent ` +
    `${fmt(p.long_dwell)} boundaries accumulating.</span>`);
  await rec.runVisible(p.post_dwell);

  // ---- measured turnover latency, from the recorded run --------------------------
  frames = await rec.frames();
  const post = frames.filter((f) => f.label === 'switch' && f.w).map((f) => f.w);
  let turnoverIdx = null; let newOwner = null;
  for (let k = 0; k < post.length; k++) {
    if (post[k] !== incumbent && post.slice(k).every((w) => w === post[k])) {
      turnoverIdx = k + 1; newOwner = post[k]; break;
    }
  }
  const pTally = {};
  for (const w of post) pTally[w] = (pTally[w] || 0) + 1;
  rec.evidence.post_switch = {
    rendered_eligible: post.length, tally: pTally,
    turnover_rendered_presentation: turnoverIdx, new_owner: newOwner,
    head: post.slice(0, 16),
  };
  rec.note(`post-switch: new owner ${newOwner} from rendered presentation ${turnoverIdx} ` +
    `of ${post.length}; tally ${JSON.stringify(pTally)}`);

  // The claim on screen is the latency measured in THIS recording. The headless preflight at
  // the same seed is quoted alongside it as an independent cross-check, never as a substitute.
  const carry = turnoverIdx == null ? null : turnoverIdx - 1;
  const latencyPhrase = turnoverIdx === 1
    ? `on the FIRST eligible presentation after the switch`
    : turnoverIdx == null
      ? `— NO unbroken turnover was observed in this run`
      : `on eligible presentation <span class="num">${turnoverIdx}</span> of ` +
        `<span class="num">${fmt(post.length)}</span> after the switch: the incumbent won ` +
        `${num(carry)} carry-over presentation${carry === 1 ? '' : 's'}, then never again`;

  await rec.caption(
    `<span class="lead">Turnover measured in this recording</span> ` +
    `incumbent <span class="num">${incumbent}</span> → new owner ` +
    `<span class="num">${newOwner}</span>, ${latencyPhrase}. ` +
    `Post-switch winner counts: ${Object.entries(pTally)
      .map(([k, v]) => `${k} ${fmt(v)}`).join(' · ')}.<br>` +
    `<span class="dim">Criterion: ${p.consolidation_criterion}. Not instantaneous — the ` +
    `incumbent does win the first presentation${carry > 1 ? 's' : ''} after the switch. ` +
    `Independent headless cross-checks at seed ${SEED}: this same four-pattern-then-dwell ` +
    `protocol turns over at eligible presentation ` +
    `${p.turnover_eligible_presentation} (${p.incumbent} → ${p.new_owner}), and the published ` +
    `long-dwell protocol in experiments/dual_fe_cc4_consolidation.py turns over ` +
    `${pre.headless_long_dwell_crosscheck.pattern1_owner} → ` +
    `${pre.headless_long_dwell_crosscheck.pattern2_owner}.</span>`);
  await sleep(7000);

  // ---- paused weight summary ------------------------------------------------------
  await rec.openView('rf', 'rf-overlay');
  await rec.caption(
    `<span class="lead">Demonstration 3 — summary (paused)</span> ` +
    `Receptive Fields: <span class="num">${incumbent}</span> still holds the ` +
    `“${p.held_pattern}” weights it built over ${fmt(rec.evidence.dwell.boundaries)} ` +
    `boundaries, while <span class="num">${newOwner}</span> — a different neuron — now owns ` +
    `“${p.switched_to}”. <span class="dim">Equal-frequency inputs throughout; no class ` +
    `reweighting, no balanced sampling, no pattern-specific learning rate.</span>`);
  await sleep(HOLD.end + 4500);
  return rec.evidence;
}

// ================================================================== video 4
async function recordScaling(rec) {
  const { api, pre } = rec;
  const tiled = pre.scaling.tiled_cc;
  const tower = pre.scaling.two_tower_composition;
  const stim = pre.two_tower_stimuli;

  // ------------------------------- chapter A: tiled_cc ---------------------------
  const stA = await rec.configure({ ...SCALING_BASE, topology: 'tiled_cc' }, tiled.nodes);
  await api.clearPatchPatterns();
  await api.reset();
  await api.waitFor((s) => s.dynamic.timestep === 0, { what: 'timestep back to 0' });
  await rec.hideTitle();
  await rec.chapter('04 · A', 'Scaling chapter A — the 9×9 tiled hierarchy');
  await sleep(700);
  const nA = stA.topology.neurons.length;
  const eA = stA.topology.synapses.length;
  const tilingA = stA.topology.tiling;
  rec.evidence.tiled_cc = { nodes: nA, edges: eA };
  await rec.caption(
    `<span class="lead">tiled_cc — live topology</span> ${num(nA)} nodes · ${num(eA)} edges · ` +
    `${num(tilingA.input_shape.rows)}×${num(tilingA.input_shape.cols)} RGC sheet tiled into ` +
    `${num(tilingA.grid_shape.rows * tilingA.grid_shape.cols)} ` +
    `${tilingA.patch_shape.rows}×${tilingA.patch_shape.cols} patches · ` +
    `${num(tilingA.column_layers.map((l) => `${l.layer} ${l.rows}×${l.cols}`).join(', '))} ` +
    `<span class="dim">— one L1 column per patch, one L2 column receiving all nine L1 ` +
    `outputs. Counts read from GET /api/state.</span>`);
  await sleep(HOLD.beat);

  // multiple independently driven patch patterns
  const patchPlan = [[0, 0, 'row 1'], [1, 1, 'diag \\'], [2, 2, 'col 1'], [0, 2, 'diag /']];
  for (const [r, c, name] of patchPlan) {
    await api.patchPattern(r, c, name);
    await sleep(550);
    await rec.caption(
      `<span class="lead">Independently driven patches</span> ` +
      `patch (${r},${c}) ← “${name}” <span class="dim">— each 3×3 patch drives its own L1 ` +
      `column and can be changed independently while every column keeps learning. The input ` +
      `sheet is the pixel-wise union of the assigned patches.</span>`);
  }
  await sleep(1200);
  await rec.caption(
    `<span class="lead">Four patches, four different local patterns</span> ` +
    `<span class="dim">${patchPlan.map(([r, c, n]) => `(${r},${c}) “${n}”`).join(' · ')} — ` +
    `driving ${num(4)} of the ${num(9)} L1 columns at once.</span>`);
  await rec.runVisible(260, SPEED_TILED);

  await rec.caption(
    `<span class="lead">The whole 9×9 graph</span> ` +
    `<span class="dim">${num(nA)} nodes and ${num(eA)} edges at their real functional ` +
    `positions, weak synapses shown — nine L1 columns fanning into one L2 column. Counts read ` +
    `from GET /api/state.</span>`);
  await rec.runVisible(160, SPEED_TILED);
  await sleep(2600);

  // ------------------------------- chapter B: two towers -------------------------
  // The new graph is installed and its inspected neuron chosen BEFORE the chapter card:
  // the picker screenshots the scene, so a card covering the canvas would scan the card.
  await rec.caption('');
  const stB = await rec.configure(
    { ...SCALING_BASE, topology: 'two_tower_composition' }, tower.nodes);
  await api.reset();
  await api.waitFor((s) => s.dynamic.timestep === 0, { what: 'timestep back to 0' });
  rec.evidence.inspected_two_tower = await rec.selectNeuron(
    ["T0L2c00E", "T1L2c00E", "L3c00E"], { maxSeconds: 8 });
  await rec.title('Chapter B', 'Two 9×9 towers on one 9×18 sheet',
    'Eighteen L1 columns · two L2 columns · one L3 composition column, built from the same ' +
    'reusable column and connector rules.', HOLD.beat + 500);
  await rec.hideTitle();
  await rec.chapter('04 · B', 'Scaling chapter B — the 9×18 two-tower hierarchy');
  await sleep(700);
  const nB = stB.topology.neurons.length;
  const eB = stB.topology.synapses.length;
  const tilingB = stB.topology.tiling;
  const layersB = tilingB.column_layers.map((l) => `${l.layer} ${l.rows}×${l.cols}`).join(' · ');
  rec.evidence.two_tower = { nodes: nB, edges: eB, layers: layersB };
  await rec.caption(
    `<span class="lead">two_tower_composition — live topology</span> ${num(nB)} nodes · ` +
    `${num(eB)} edges · ${num(tilingB.input_shape.rows)}×${num(tilingB.input_shape.cols)} ` +
    `input surface · column layers ${num(layersB)} ` +
    `<span class="dim">— ${num(tower.columns_by_layer.L1)} L1 columns (nine per tower), ` +
    `${num(tower.columns_by_layer.L2)} L2 columns, ${num(tower.columns_by_layer.L3)} L3 ` +
    `composition column. Growth from chapter A: ${num(nA)} → ${num(nB)} nodes, ` +
    `${num(eA)} → ${num(eB)} edges.</span>`);
  await sleep(HOLD.beat + 800);

  // whole-sheet glyph stimuli
  await rec.tap(true, null);
  for (const g of stim.glyphs) {
    await api.pattern(g);
    await rec.tapLabel(`glyph:${g}`);
    await sleep(400);
    await rec.caption(
      `<span class="lead">Whole-sheet stimulus “${g}”</span> ` +
      `<span class="dim">a 9×18 glyph that crosses the tower seam — no per-patch pattern can ` +
      `express it, so it is driven through the whole-input endpoint. Tower L2 top owner at ` +
      `this seed (headless preflight): <span class="num">${stim.l2_top_owner_per_glyph[g]}</span>.</span>`);
    await rec.runVisible(78, SPEED_TWO_TOWER);
  }
  await rec.tap(false);

  await rec.caption(
    `<span class="lead">The whole two-tower graph</span> ` +
    `<span class="dim">${num(nB)} nodes and ${num(eB)} edges — eighteen L1 columns feeding ` +
    `two L2 columns, both feeding one L3 composition column through the same generic ` +
    `connector used at 9×9. Counts read from GET /api/state.</span>`);
  await rec.runVisible(48, SPEED_TWO_TOWER);
  await sleep(3200);

  // truthful scope statement
  const l3 = stim.l3_top_owner_per_glyph;
  await rec.caption(
    `<span class="lead">Demonstration 4 — scope statement</span> ` +
    `The graph scales: ${num(nA)}/${num(eA)} → ${num(nB)}/${num(eB)} nodes/edges, same ` +
    `column motif, same connector. Tower L2 recruits a distinct owner per glyph ` +
    `(${stim.glyphs.map((g) => `${g} → ${stim.l2_top_owner_per_glyph[g]}`).join(' · ')}; ` +
    `${num(stim.distinct_l2_top_owners)} distinct).<br>` +
    `<span class="warn">${TWO_TOWER_QUALIFICATION}</span> ` +
    `<span class="dim">Measured here: all three glyphs resolve to the same L3 owner ` +
    `(${stim.glyphs.map((g) => `${g} → ${l3[g]}`).join(' · ')}; ` +
    `${num(stim.distinct_l3_top_owners)} distinct). Lower encoders scale; L3 identity ` +
    `separation is not claimed by this demo.</span>`);
  await sleep(HOLD.end + 5000);
  return rec.evidence;
}

// ================================================================== driver
const VIDEOS = [
  { key: 'uncertainty', slug: 'uncertainty_frequency_demo', run: recordUncertainty,
    ready: { expectNodes: 45, expectInputCells: 9 }, thumbFraction: 0.52,
    inspect: ['L2E', 'L1C', 'L1E'],
    title: { eyebrow: 'Demonstration 1 of 4',
      main: 'Frequency as an observable uncertainty signal',
      sub: 'A 3×3 coincidence / feedback circuit exposes a cadence tied to confirmed '
        + 'bottom-up ↔ top-down agreement.' } },
  { key: 'continuous', slug: 'continuous_learning_demo', run: recordContinuous,
    ready: { expectNodes: 14, expectInputCells: 9 }, thumbFraction: 0.97,
    inspect: ['ccE'],
    title: { eyebrow: 'Demonstration 2 of 4',
      main: 'Continuous learning across sequential patterns',
      sub: 'One live network, four overlapping 3×3 patterns, presented one at a time — with '
        + 'no reset, reseed, rebuild or reconfiguration between phases.' } },
  { key: 'unbiased', slug: 'unbiased_learning_demo', run: recordUnbiased,
    ready: { expectNodes: 14, expectInputCells: 9 }, thumbFraction: 0.90,
    inspect: ['ccE'],
    title: { eyebrow: 'Demonstration 3 of 4',
      main: 'Resistance to training-exposure bias',
      sub: 'After one pattern is held for thousands of boundaries, a different pattern still '
        + 'takes ownership — with no reset and no rebuild.' } },
  { key: 'scaling', slug: 'scaling_demo', run: recordScaling,
    ready: { expectNodes: 191, expectInputCells: 81 }, thumbFraction: 0.97,
    inspect: ['L2c00E', 'L1c'],
    title: { eyebrow: 'Demonstration 4 of 4',
      main: 'Scaling: 9×9 tiled hierarchy → 9×18 two towers',
      sub: 'The same column motif and the same generic child → parent connector, composed '
        + 'into progressively larger graphs.' } },
];

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const selected = args.only
    ? VIDEOS.filter((v) => args.only.includes(v.key))
    : VIDEOS;
  if (!selected.length) throw new Error(`--only matched no video (have: ${VIDEOS.map((v) => v.key).join(', ')})`);

  await fs.mkdir(ASSETS, { recursive: true });
  await fs.mkdir(RAW, { recursive: true });
  await fs.rm(WORK, { recursive: true, force: true });
  await fs.mkdir(WORK, { recursive: true });

  if (args.preflight) {
    console.log('== running headless preflight ==');
    const { execFile } = await import('node:child_process');
    const { promisify } = await import('node:util');
    await promisify(execFile)(PYTHON, ['scripts/preflight_demo_measurements.py',
      '--compare-uncapped'],
    { cwd: ROOT, env: { ...process.env, PYTHONPATH: ROOT }, maxBuffer: 32 * 1024 * 1024 });
  }

  let pre;
  try {
    pre = JSON.parse(await fs.readFile(PREFLIGHT_JSON, 'utf8'));
  } catch {
    throw new Error(`missing ${PREFLIGHT_JSON}. Run:\n  ` +
      `PYTHONPATH=. .venv/bin/python scripts/preflight_demo_measurements.py --compare-uncapped\n` +
      `or pass --preflight.`);
  }
  if (pre.seed !== SEED) throw new Error(`preflight seed ${pre.seed} != required ${SEED}`);

  const ffmpegVersion = await toolVersion('ffmpeg');
  const ffprobeVersion = await toolVersion('ffprobe');
  const pkg = JSON.parse(
    await fs.readFile(path.join(ROOT, 'node_modules', 'playwright', 'package.json'), 'utf8')
      .catch(() => '{"version":"unknown"}'));

  // ---------------------------------------------------- port + server lifecycle
  let port = args.port;
  let ownServer = true;
  if (await portInUse(port)) {
    const probeApi = new DashboardApi(port);
    let foreign = null;
    try { foreign = await probeApi.state(); } catch { /* not our dashboard */ }
    const isDashboard = !!(foreign && foreign.topology && foreign.dynamic);
    const foreignSeed = foreign?.topology?.params?.seed;
    if (isDashboard && Number(foreignSeed) === SEED) {
      console.log(`== port ${port} already serves an SNN dashboard at seed ${SEED}; reusing it ` +
        `(will NOT be stopped by this workflow)`);
      ownServer = false;
    } else if (args.strictPort) {
      throw new Error(
        `port ${port} is occupied by ${isDashboard
          ? `an SNN dashboard at seed ${foreignSeed} (this recording requires seed ${SEED})`
          : 'a non-dashboard process'}. ` +
        `This workflow never terminates a server it did not start. ` +
        `Re-run without --strict-port to record on an automatically chosen free port, ` +
        `or free port ${port} yourself.`);
    } else {
      const alt = await findFreePort(port + 10);
      console.log(`== port ${port} is occupied by ${isDashboard
        ? `an SNN dashboard at seed ${foreignSeed} (this recording requires seed ${SEED})`
        : 'a non-dashboard process'}.`);
      console.log(`== that process is left running untouched; recording on free port ${alt} instead.`);
      port = alt;
    }
  }

  const seedFile = new SeedFile(ROOT);
  const server = new UvicornServer({
    repoRoot: ROOT, port, python: PYTHON,
    logPath: path.join(ASSETS, 'recording_server.log'),
  });
  const api = new DashboardApi(port);
  const manifest = {
    generated_at: new Date().toISOString(),
    seed: SEED, port, own_server: ownServer,
    versions: {
      node: process.version, playwright: pkg.version,
      chromium: null, ffmpeg: ffmpegVersion, ffprobe: ffprobeVersion,
    },
    seed_file: null, videos: [],
  };

  let browser = null;
  try {
    if (ownServer) {
      const captured = await seedFile.capture();
      manifest.seed_file = {
        path: path.relative(ROOT, seedFile.file),
        existed_before: captured.existed,
        previous_contents: captured.previous,
      };
      console.log(`== .claude/dashboard_seed.txt backed up ` +
        `(${captured.existed ? `was "${String(captured.previous).trim()}"` : 'did not exist'})`);
      await seedFile.write(SEED);
      const pid = await server.start();
      console.log(`== started uvicorn pid ${pid} on port ${port}`);
      await server.waitReady(api);
    }
    const live = await api.state();
    if (Number(live.topology.params.seed) !== SEED) {
      throw new Error(`server engine seed is ${live.topology.params.seed}, required ${SEED}`);
    }
    console.log(`== dashboard ready at ${api.base()} (engine seed ${live.topology.params.seed})`);

    browser = await chromium.launch({ headless: true, args: ['--hide-scrollbars', '--mute-audio'] });
    manifest.versions.chromium = browser.version();
    console.log(`== chromium ${browser.version()}, playwright ${pkg.version}`);

    for (const video of selected) {
      console.log(`\n== recording ${video.slug} ==`);
      const t0 = Date.now();
      const context = await browser.newContext({
        viewport: VIDEO_SIZE,
        deviceScaleFactor: 1,
        recordVideo: { dir: path.join(WORK, video.key), size: VIDEO_SIZE },
        reducedMotion: 'no-preference',
      });
      await context.addInitScript(OVERLAY_JS);
      const page = await context.newPage();
      page.on('pageerror', (e) => console.warn(`      ! page error: ${e.message}`));
      const tSetup = Date.now();

      // Make sure the engine is in the right topology BEFORE the page loads so the readiness
      // predicate can gate on the expected node/pixel counts.
      const bootTopo = video.key === 'uncertainty' ? COINCIDENCE_CONFIG
        : video.key === 'scaling' ? { ...SCALING_BASE, topology: 'tiled_cc' }
          : CC4_CONFIG;
      await api.pause();
      await api.config(bootTopo);
      await api.reset();

      await page.goto(api.base(), { waitUntil: 'domcontentloaded' });
      await page.addStyleTag({ content: OVERLAY_CSS });
      await waitDashboardReady(page, api, video.ready);
      await sleep(700);

      const rec = new Recorder(page, api, pre);
      // 1. Screenshot the scene while nothing covers it, to locate clickable neurons.
      await rec.captureSelectionSpots();
      // 2. Raise this video's title card. EVERY remaining setup action happens behind it, so
      //    the recording opens on the title rather than on filter toggles and click probing.
      await rec.title(video.title.eyebrow, video.title.main, video.title.sub, 0);
      await sleep(600);
      // 3. Full connectivity: "Hide weak synapses" is ON by default and suppresses any
      //    weighted edge below 25% of the weight cap, which progressively hides the losing
      //    competitors' feedforward afferents as an owner consolidates. Every other
      //    population filter stays ON so nothing is omitted.
      await rec.setSceneFilters({
        'f-weak': false, 'f-active': false, 'f-assembly': false,
        'f-rg': true, 'f-l1': true, 'f-l2': true, 'f-inh': true,
      });
      // 4. Populate the right-hand inspector (clicks pass through the pointer-events:none card).
      rec.evidence.inspected = await rec.selectNeuron(video.inspect, { maxSeconds: 8 });
      await sleep(Math.max(0, HOLD.title - (Date.now() - tSetup)));
      let evidence = {};
      let error = null;
      try {
        evidence = await video.run(rec);
      } catch (e) {
        error = e;
        console.error(`      !! ${video.slug} failed: ${e.message}`);
        await page.screenshot({ path: path.join(ASSETS, `${video.slug}_FAILURE.png`) })
          .catch(() => {});
      }

      await api.pause().catch(() => {});
      const videoHandle = page.video();
      await page.close();
      await context.close();
      const rawTmp = await videoHandle.path();
      const webm = path.join(RAW, `${video.slug}.webm`);
      await fs.rename(rawTmp, webm).catch(async () => {
        await fs.copyFile(rawTmp, webm); await fs.rm(rawTmp, { force: true });
      });
      const webmStat = await fs.stat(webm);
      console.log(`      raw webm ${path.relative(ROOT, webm)} (${(webmStat.size / 1e6).toFixed(1)} MB)`);

      manifest.videos.push({
        key: video.key, slug: video.slug, webm: path.relative(ROOT, webm),
        webm_bytes: webmStat.size,
        wall_seconds: Number(((Date.now() - t0) / 1000).toFixed(1)),
        notes: rec.notes, evidence,
        error: error ? `${error.message}` : null,
      });
      if (error) throw error;
    }
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (ownServer) {
      const stopped = await server.stop().catch(() => false);
      if (stopped) console.log(`\n== stopped the uvicorn process this workflow started`);
      const restored = await seedFile.restore().catch((e) => ({ restored: false, error: String(e) }));
      console.log(`== .claude/dashboard_seed.txt restored: ${JSON.stringify(restored)}`);
      manifest.seed_file = { ...manifest.seed_file, restore: restored };
    }
  }

  // ------------------------------------------------- convert + thumbnail + verify
  console.log('\n== converting to MP4 and extracting thumbnails ==');
  for (const entry of manifest.videos) {
    const webm = path.join(ROOT, entry.webm);
    const mp4 = path.join(ASSETS, `${entry.slug}.mp4`);
    const thumb = path.join(ASSETS, `${entry.slug}_thumbnail.png`);
    const dest = path.join(ASSETS, `${entry.slug}.webm`);
    await fs.copyFile(webm, dest);                        // raw recording preserved in raw_webm/

    await convertToMp4(webm, mp4, { fps: FPS, ...VIDEO_SIZE });
    const info = await probe(mp4);
    const packets = await decodeCheck(mp4);
    const ts = Math.max(0.5, info.duration * (VIDEOS.find((v) => v.key === entry.key).thumbFraction));
    await extractFrame(mp4, ts, thumb, VIDEO_SIZE);
    const tsize = await pngSize(thumb);

    const checks = {
      codec_h264: info.codec === 'h264',
      pix_fmt_yuv420p: info.pix_fmt === 'yuv420p',
      width_1920: info.width === 1920,
      height_1080: info.height === 1080,
      fps_30: Math.abs(info.fps - FPS) < 0.01,
      duration_positive: info.duration > 0,
      decodes: packets > 0,
      thumbnail_1920x1080: tsize.width === 1920 && tsize.height === 1080,
    };
    const failed = Object.entries(checks).filter(([, v]) => !v).map(([k]) => k);
    entry.mp4 = path.relative(ROOT, mp4);
    entry.webm_copy = path.relative(ROOT, dest);
    entry.thumbnail = path.relative(ROOT, thumb);
    entry.thumbnail_timestamp_s = Number(ts.toFixed(2));
    entry.ffprobe = info;
    entry.decoded_packets = packets;
    entry.checks = checks;
    console.log(`   ${entry.slug}: ${info.codec}/${info.pix_fmt} ${info.width}x${info.height} ` +
      `@${info.fps}fps ${info.duration.toFixed(1)}s ${(info.size_bytes / 1e6).toFixed(1)}MB ` +
      `${failed.length ? `FAILED: ${failed.join(', ')}` : 'all checks pass'}`);
    if (failed.length) throw new Error(`${entry.slug} failed verification: ${failed.join(', ')}`);
  }

  // independence: four distinct MP4s, no combined montage
  const mp4s = manifest.videos.map((v) => v.mp4);
  manifest.independent_outputs = new Set(mp4s).size === mp4s.length;

  if (!args.keepWork) await fs.rm(WORK, { recursive: true, force: true });
  const manifestPath = path.join(ASSETS, 'recording_manifest.json');
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\n== wrote ${path.relative(ROOT, manifestPath)}`);
  console.log('== done ==');
}

main().catch((e) => {
  console.error(`\nRECORDING FAILED: ${e.stack || e.message}`);
  process.exit(1);
});
