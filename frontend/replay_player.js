// Read-only replay player: load one self-contained replay.snn.jsonl, then watch,
// pause, step, scrub, and marker-jump the recorded test through the SAME shared
// renderer/inspector/chart paths the live dashboard uses (see app.js). It never
// simulates, never reruns learning, never mutates the recorded file, and never
// sends a mutation to the live engine.
//
// All engine interaction is deliberately minimal: one pause on entry so the hidden
// live sim stops advancing, and one GET /api/state on exit to resync. Everything
// else is pure playback of recorded frames. Untrusted labels (run name, phase,
// pattern, marker kinds) are rendered via textContent, never innerHTML.

import { parseReplay, ReplayError } from './replay.js';

const SPEED_OPTIONS = [1, 2, 4, 8, 16, 30];   // recorded frames applied per second
const DEFAULT_SPEED = 8;

// Live mutation controls disabled while replay owns the display (the api guard in
// app.js is the real safety net; this is the visual affordance). Dynamically-built
// controls (pattern/pixel/patch/config/RF grids) are neutralized via a body class.
const GUARDED_IDS = [
  'g-start', 'g-pause', 'g-step', 'g-reset', 'g-reseed', 'g-editor',
  'x-start', 'x-pause', 'x-resume', 'x-step', 'x-reset', 'x-reseed',
  'p-random', 'p-clear', 'p-noise', 'mf-pulse', 'mf-hold', 'mf-neuron', 'mf-mag',
  'config-apply', 'config-reset-defaults', 'speed',
  'raster-play', 'raster-pause', 'raster-stop', 'charge-play', 'charge-pause', 'charge-stop',
  'weights-play', 'weights-pause', 'weights-stop', 'rf-play', 'rf-pause',
];

const $ = id => document.getElementById(id);

export class ReplayPlayer {
  constructor(hooks) {
    this.hooks = hooks;            // {applyTopology, applyDynamic, bulkSeek, updateTopbar, setReplayActive, pauseLive, fetchLiveState}
    this.active = false;
    this.replay = null;
    this.pos = -1;
    this.playing = false;
    this._timer = null;            // playback setTimeout id (only ever one)
    this._seekRaf = 0;             // coalesced scrubber rAF
    this._pendingSeek = null;
    this.speed = DEFAULT_SPEED;

    this._wire();
  }

  _wire() {
    this.fileInput = $('replay-file');
    $('g-load-test')?.addEventListener('click', () => this.fileInput?.click());
    this.fileInput?.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) this._loadFile(f);
      this.fileInput.value = '';   // allow re-selecting the same file
    });

    $('rp-play')?.addEventListener('click', () => this._togglePlay());
    $('rp-step-back')?.addEventListener('click', () => this._step(-1));
    $('rp-step-fwd')?.addEventListener('click', () => this._step(1));
    $('rp-exit')?.addEventListener('click', () => this.exit());
    $('rp-marker-prev')?.addEventListener('click', () => this._markerJump(-1));
    $('rp-marker-next')?.addEventListener('click', () => this._markerJump(1));

    const sel = $('rp-speed');
    if (sel) {
      sel.innerHTML = '';
      for (const s of SPEED_OPTIONS) {
        const o = document.createElement('option');
        o.value = String(s); o.textContent = `${s}×`;
        if (s === DEFAULT_SPEED) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => { this.speed = +sel.value || DEFAULT_SPEED; });
    }

    this.slider = $('rp-slider');
    // Scrubbing pauses and coalesces to one seek per animation frame, so dragging
    // never seeks from zero per input event nor grows any chart array unboundedly.
    this.slider?.addEventListener('input', () => {
      this._pause();
      this._pendingSeek = +this.slider.value;
      if (!this._seekRaf) this._seekRaf = requestAnimationFrame(() => {
        this._seekRaf = 0;
        if (this._pendingSeek != null) { this._seekTo(this._pendingSeek, false); this._pendingSeek = null; }
      });
    });

    this.markerSelect = $('rp-marker-select');
    this.markerSelect?.addEventListener('change', () => {
      const v = +this.markerSelect.value;
      if (Number.isInteger(v)) this._seekTo(v, false);
    });

    window.addEventListener('keydown', (e) => {
      if (!this.active) return;
      if (e.key === 'Escape') this.exit();
    });
  }

  // -------------------------------------------------------------- file loading
  async _loadFile(file) {
    this._clearError();
    let text;
    try { text = await file.text(); }
    catch (e) { this._toast(`could not read file: ${e.message}`); return; }

    let replay;
    try { replay = parseReplay(text); }
    catch (e) {
      // Do NOT enter replay on validation failure; the live view stays intact.
      this._toast(e instanceof ReplayError ? e.message : `parse failed: ${e.message}`);
      return;
    }
    this._enter(replay, file);
  }

  // --------------------------------------------------------------- enter/exit
  async _enter(replay, file) {
    this._stopTimer();
    this.replay = replay;
    this.pos = -1;
    this.playing = false;

    try { await this.hooks.pauseLive(); } catch { /* engine pause is best-effort */ }
    this.active = true;
    this.hooks.setReplayActive(true);
    this._guard(true);

    // Header/run info (all text-safe).
    const m = replay.meta;
    $('rp-name').textContent =
      `${file?.name ?? 'replay'} · ${m.experiment ?? 'run'} · seed ${m.seed ?? '—'}`;
    const fb = m.conditions?.hierarchical_feedback ?? 'n/a';
    $('rp-condition').textContent =
      `${m.preset ?? m.topologyName ?? 'topology'} · feedback: ${fb}`;

    this._buildTimeline();
    this._buildMarkers();

    const status = $('st-status');
    if (status) { status.textContent = 'Replay'; status.style.color = 'var(--rg, #fbbf24)'; }
    $('replay-bar').hidden = false;

    // Shared paths: recorded topology, then the first frame.
    this.hooks.applyTopology(replay.topology);
    this._seekTo(0, false);
    this._setPlayIcon();
  }

  async exit() {
    if (!this.active) return;
    this._pause();
    this._stopTimer();
    this.active = false;
    this.replay = null;
    this.pos = -1;
    if (this.fileInput) this.fileInput.value = '';

    $('replay-bar').hidden = true;
    this._guard(false);
    this._clearError();

    // Resync from the authoritative live engine and leave it paused.
    try {
      const state = await this.hooks.fetchLiveState();
      this.hooks.setReplayActive(false);   // re-enable display of live frames first
      this.hooks.applyTopology(state.topology);
      this.hooks.applyDynamic(state.dynamic);
      this.hooks.updateTopbar(state.dynamic, 0);
    } catch (e) {
      this.hooks.setReplayActive(false);
      console.warn('replay exit: could not fetch live state', e);
    }
  }

  // ------------------------------------------------------------- transport
  _togglePlay() { this.playing ? this._pause() : this._play(); }

  _play() {
    if (!this.active || this.playing) return;
    if (this.pos >= this.replay.frames.length - 1) return;   // already at the end
    this.playing = true;
    this._setPlayIcon();
    this._tick();
  }

  _pause() {
    this.playing = false;
    this._stopTimer();
    this._setPlayIcon();
  }

  _tick() {
    this._stopTimer();
    if (!this.playing) return;
    if (this.pos >= this.replay.frames.length - 1) { this._pause(); return; }
    this._seekTo(this.pos + 1, true);   // sequential: cheap forward apply
    this._timer = setTimeout(() => this._tick(), 1000 / this.speed);
  }

  _stopTimer() { if (this._timer) { clearTimeout(this._timer); this._timer = null; } }

  _step(dir) {
    this._pause();
    this._seekTo(this.pos + dir, dir > 0);   // forward step may reuse the sequential path
  }

  // Go to frame position `pos`. A sequential +1 reuses the live forward path
  // (append to history); anything else rebuilds the bounded window truthfully.
  _seekTo(pos, sequential) {
    if (!this.active || !this.replay) return;
    const n = this.replay.frames.length;
    pos = Math.max(0, Math.min(n - 1, pos));
    if (sequential && pos === this.pos + 1) this.hooks.applyDynamic(this.replay.frames[pos].dynamic);
    else this.hooks.bulkSeek(this.replay, pos);
    this.pos = pos;
    this._reflect();
  }

  // ------------------------------------------------------------- markers
  _markerJump(dir) {
    const markers = this._markerFrameIndices();
    if (!markers.length) return;
    const curFi = this.replay.frames[this.pos].frameIndex;
    let targetFi = null;
    if (dir > 0) { for (const fi of markers) if (fi > curFi) { targetFi = fi; break; } }
    else { for (let i = markers.length - 1; i >= 0; i--) if (markers[i] < curFi) { targetFi = markers[i]; break; } }
    if (targetFi == null) return;
    const pos = this._posForFrameIndex(targetFi);
    if (pos != null) this._seekTo(pos, false);
  }

  // Distinct, sorted marker frame indices that map onto a real frame position.
  _markerFrameIndices() {
    if (!this._markerFis) {
      const set = new Set();
      for (const mk of this.replay.markers) {
        const pos = this._posForFrameIndex(mk.frame_index);
        if (pos != null) set.add(this.replay.frames[pos].frameIndex);
      }
      this._markerFis = [...set].sort((a, b) => a - b);
    }
    return this._markerFis;
  }

  // Frame position at (or the last one before) a marker's frame_index; markers
  // recorded before the first frame (index -1) clamp to position 0.
  _posForFrameIndex(fi) {
    const frames = this.replay.frames;
    if (fi < frames[0].frameIndex) return 0;
    if (this.replay.frameIndexToPos.has(fi)) return this.replay.frameIndexToPos.get(fi);
    let best = null;
    for (let i = 0; i < frames.length; i++) { if (frames[i].frameIndex <= fi) best = i; else break; }
    return best;
  }

  // -------------------------------------------------------------- UI helpers
  _buildTimeline() {
    const n = this.replay.frames.length;
    if (this.slider) { this.slider.min = '0'; this.slider.max = String(n - 1); this.slider.value = '0'; }
    // Marker ticks positioned by frame fraction along the timeline.
    const ticks = $('rp-ticks');
    if (ticks) {
      ticks.innerHTML = '';
      const span = Math.max(1, n - 1);
      for (const fi of this._markerFrameIndices()) {
        const pos = this._posForFrameIndex(fi);
        const t = document.createElement('span');
        t.className = 'rp-tick';
        t.style.left = `${(pos / span) * 100}%`;
        const kinds = this.replay.markers.filter(mk => this._posForFrameIndex(mk.frame_index) === pos)
          .map(mk => mk.kind);
        t.title = kinds.join(', ');
        t.addEventListener('click', () => this._seekTo(pos, false));
        ticks.appendChild(t);
      }
    }
  }

  _buildMarkers() {
    this._markerFis = null;   // recomputed lazily against the new replay
    const sel = this.markerSelect;
    if (!sel) return;
    sel.innerHTML = '';
    const head = document.createElement('option');
    head.value = ''; head.textContent = 'Jump to marker…';
    sel.appendChild(head);
    for (const mk of this.replay.markers) {
      const pos = this._posForFrameIndex(mk.frame_index);
      if (pos == null) continue;
      const o = document.createElement('option');
      o.value = String(pos);
      const ts = this.replay.frames[pos].timestep;
      o.textContent = `t=${ts} · ${mk.kind}`;   // textContent: untrusted kind is safe
      sel.appendChild(o);
    }
  }

  // Sync the bar + top bar to the current frame.
  _reflect() {
    const f = this.replay.frames[this.pos];
    const n = this.replay.frames.length;
    if (this.slider && +this.slider.value !== this.pos) this.slider.value = String(this.pos);
    $('rp-frame').textContent = `frame ${this.pos + 1}/${n} · t=${f.timestep}`;
    const a = f.annotation || {};
    $('rp-phase').textContent = `phase: ${a.phase ?? '—'} · pattern: ${a.pattern ?? '—'}`;
    if (this.markerSelect) this.markerSelect.value = '';   // reset the picker label

    const ts = $('st-timestep'); if (ts) ts.textContent = f.timestep;
    const win = $('st-winner'); if (win) win.textContent = f.dynamic.winner || '—';
    const sp = $('st-speed'); if (sp) sp.textContent = `${this.speed}× replay`;
  }

  _setPlayIcon() {
    const b = $('rp-play');
    if (b) { b.textContent = this.playing ? '⏸' : '▶'; b.title = this.playing ? 'Pause playback' : 'Play'; }
  }

  _guard(on) {
    document.body.classList.toggle('replay-mode', on);
    for (const id of GUARDED_IDS) { const e = $(id); if (e) e.disabled = on; }
  }

  // -------------------------------------------------------------- errors
  _toast(msg) {
    const t = $('replay-toast');
    if (!t) { console.warn('replay:', msg); return; }
    t.textContent = `Replay: ${msg}`;   // textContent: untrusted parse text is safe
    t.hidden = false;
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => { t.hidden = true; }, 6000);
  }

  _clearError() {
    const t = $('replay-toast'); if (t) { t.hidden = true; t.textContent = ''; }
    const e = $('rp-error'); if (e) { e.hidden = true; e.textContent = ''; }
  }
}
