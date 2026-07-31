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

import {
  parseReplay, ReplayError, buildBranchPayload,
  isCheckpointFrame, nearestCheckpointFrameIndexAt,
} from './replay.js';

// Recorded frames applied per second. The top of the range exists for full-resolution NEST
// artifacts: at `--charge-interval = h` a 180 ms run is 18 000 frames, which at 30x is ten
// minutes of watching. If a speed outpaces what `applyDynamic` can render the timer simply
// falls behind -- it degrades, it does not break.
const SPEED_OPTIONS = [1, 2, 4, 8, 16, 30, 60, 120, 250];
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

// The full-screen overlays cover the top bar, so each mirrors a transport into its own bar.
// The mirrored LIVE transport is guarded above (it drives the engine), which on its own
// would leave an open overlay with no working transport at all -- the chart could be opened
// during a replay but not played, stepped or read, so charge/time was effectively
// inaccessible without closing it. Each overlay therefore also carries a REPLAY transport,
// swapped in for the live one exactly while replay owns the display.
const MIRRORED_OVERLAYS = ['raster', 'charge', 'weights', 'rf'];

const $ = id => document.getElementById(id);

export class ReplayPlayer {
  constructor(hooks) {
    this.hooks = hooks;            // {applyTopology, applyDynamic, bulkSeek, updateTopbar,
                                   //  setReplayActive, pauseLive, fetchLiveState, branchFromReplay}
    this.active = false;
    this.replay = null;
    this.pos = -1;
    this.playing = false;
    this._timer = null;            // playback setTimeout id (only ever one)
    this._seekRaf = 0;             // coalesced scrubber rAF
    this._pendingSeek = null;
    this.speed = DEFAULT_SPEED;
    this._branchInFlight = false;  // guards double submission of a branch request

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
    $('rp-event-prev')?.addEventListener('click', () => this._eventJump(-1));
    $('rp-event-next')?.addEventListener('click', () => this._eventJump(1));
    for (const v of MIRRORED_OVERLAYS) {
      $(`${v}-rp-play`)?.addEventListener('click', () => this._togglePlay());
      $(`${v}-rp-back`)?.addEventListener('click', () => this._step(-1));
      $(`${v}-rp-fwd`)?.addEventListener('click', () => this._step(1));
      $(`${v}-rp-evt-prev`)?.addEventListener('click', () => this._eventJump(-1));
      $(`${v}-rp-evt-next`)?.addEventListener('click', () => this._eventJump(1));
    }
    $('rp-exit')?.addEventListener('click', () => this.exit());
    $('rp-marker-prev')?.addEventListener('click', () => this._markerJump(-1));
    $('rp-marker-next')?.addEventListener('click', () => this._markerJump(1));

    // Branch-from-weights: the ONE deliberately-allowed live mutation during replay. It
    // pauses playback and opens a confirmation before anything is sent.
    $('rp-branch')?.addEventListener('click', () => this._openBranchDialog());
    $('branch-cancel')?.addEventListener('click', () => this._closeBranchDialog());
    $('branch-confirm')?.addEventListener('click', () => this._submitBranch());
    $('branch-use-checkpoint')?.addEventListener('change', () => this._renderBranchDialog());

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
      if (e.key !== 'Escape') return;
      // Escape closes the branch dialog first (if open), else exits replay.
      if (this._branchOpen) this._closeBranchDialog();
      else this.exit();
    });
  }

  // ---------------------------------------------------- branch from weights
  // Open the confirmation for "Continue with These Weights". Pauses playback first so the
  // dialog describes a stable frame. The button lives in the replay bar, so it is inherently
  // available only while replay is active.
  _openBranchDialog() {
    if (!this.active || !this.replay || this._branchInFlight) return;
    this._pause();
    const cpCheck = $('branch-use-checkpoint');
    if (cpCheck) cpCheck.checked = false;             // default to the selected frame
    const restore = $('branch-restore-input');
    if (restore) restore.checked = true;              // default input restoration on
    this._branchError('');
    this._branchOpen = true;
    this._renderBranchDialog();
    const modal = $('branch-modal');
    if (modal) modal.hidden = false;
    $('branch-confirm')?.focus();
  }

  _closeBranchDialog() {
    this._branchOpen = false;
    const modal = $('branch-modal');
    if (modal) modal.hidden = true;
    this._branchError('');
  }

  // Effective frame the branch will use, honoring the nearest-checkpoint choice.
  _branchTarget(useCheckpoint) {
    const replay = this.replay;
    const selFi = replay.frames[this.pos].frameIndex;
    const selIsCp = isCheckpointFrame(replay, selFi);
    const cpFi = nearestCheckpointFrameIndexAt(replay, selFi);
    let fi = selFi;
    if (useCheckpoint && cpFi != null) fi = cpFi;
    const pos = replay.frameIndexToPos.get(fi);
    return {
      fi, pos, timestep: replay.frames[pos].timestep,
      isCheckpoint: isCheckpointFrame(replay, fi),
      showCheckpointOption: !selIsCp && cpFi != null,
      checkpointPos: cpFi != null ? replay.frameIndexToPos.get(cpFi) : null,
    };
  }

  // Sync the dialog copy to the current checkbox state (text-safe; all textContent).
  _renderBranchDialog() {
    if (!this.active || !this.replay) return;
    const useCp = !!$('branch-use-checkpoint')?.checked;
    const t = this._branchTarget(useCp);

    $('branch-desc').textContent =
      `This creates a fresh live simulation using the learned weights at replay frame `
      + `${t.pos + 1} (recorded timestep ${t.timestep}). Membrane charge, inhibition, `
      + `refractory state, pending events, RNG position, and experiment control flow will `
      + `not be restored.`;
    $('branch-precision').textContent = t.isCheckpoint
      ? "Weights use the recorder's six-decimal checkpoint precision."
      : 'Weights after the checkpoint include four-decimal recorded deltas.';

    const wrap = $('branch-checkpoint-wrap');
    if (wrap) {
      wrap.hidden = !t.showCheckpointOption;
      if (t.showCheckpointOption) {
        $('branch-checkpoint-label').textContent =
          `Use nearest preceding checkpoint (frame ${t.checkpointPos + 1})`;
      }
    }
  }

  async _submitBranch() {
    if (!this.active || !this.replay || this._branchInFlight) return;
    const useCheckpoint = !!$('branch-use-checkpoint')?.checked;
    const restoreInput = !!$('branch-restore-input')?.checked;

    let payload;
    try {
      payload = buildBranchPayload(this.replay, this.pos, { useCheckpoint, restoreInput });
    } catch (e) {
      this._branchError(e instanceof ReplayError ? e.message : `could not build branch: ${e.message}`);
      return;
    }

    this._branchInFlight = true;
    this._setBranchBusy(true);
    this._branchError('');
    try {
      const res = await this.hooks.branchFromReplay(payload);
      if (res && res.error) throw new Error(res.error);
      // Success: leave replay through the normal live-state resync path (engine stays paused).
      this._branchInFlight = false;
      this._setBranchBusy(false);
      this._closeBranchDialog();
      await this.exit();
    } catch (e) {
      // Failure: keep the replay, frame, and transport usable; show the error in the dialog.
      this._branchInFlight = false;
      this._setBranchBusy(false);
      this._branchError(e.message || 'branch failed');
    }
  }

  _setBranchBusy(busy) {
    for (const id of ['branch-confirm', 'branch-cancel', 'branch-restore-input',
                      'branch-use-checkpoint', 'rp-branch']) {
      const e = $(id); if (e) e.disabled = busy;
    }
    const b = $('branch-confirm');
    if (b) b.textContent = busy ? 'Branching…' : 'Continue';
  }

  _branchError(msg) {
    const e = $('branch-error');
    if (!e) return;
    e.textContent = msg || '';        // textContent: backend message is rendered safely
    e.hidden = !msg;
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

    // Is this a NEST artifact? Decided from the header's declared provenance, never by
    // sniffing a filename. A NEST replay has different TIME semantics (milliseconds on a
    // resolution grid, not engine boundaries) and different STATE availability, so the
    // transport must label it as such rather than let it read as a legacy run.
    this.isNest = replay.header?.nest?.engine === 'nest'
      || replay.meta.conditions?.engine === 'nest';
    this.nestProv = replay.header?.nest ?? null;

    // Header/run info (all text-safe).
    const m = replay.meta;
    $('rp-name').textContent =
      `${file?.name ?? 'replay'} · ${m.experiment ?? 'run'} · seed ${m.seed ?? '—'}`;
    const fb = m.conditions?.hierarchical_feedback ?? 'n/a';
    $('rp-condition').textContent =
      `${m.preset ?? m.topologyName ?? 'topology'} · feedback: ${fb}`;

    const badge = document.querySelector('#replay-bar .rp-badge');
    const nestInfo = $('rp-nest');
    if (this.isNest) {
      if (badge) {
        badge.textContent = 'NEST REPLAY';
        badge.classList.add('nest');
        badge.title = 'Native NEST event timing, frozen learning -- not the legacy engine';
      }
      if (nestInfo) {
        const p = this.nestProv ?? {};
        // Whether charge exists is a property of the RUN (was a multimeter attached?), not
        // of NEST, so it is read from the artifact rather than asserted. Claiming charge is
        // unavailable on a run that recorded it would send the reader looking for a missing
        // panel that is in fact populated.
        const rec = replay.header?.recording ?? {};
        const interval = rec.charge_interval_ms;
        nestInfo.hidden = false;
        nestInfo.textContent =
          `native event timing · frozen learning · h=${p.resolution_h_ms ?? '?'} ms`
          + ` · threads ${p.threads ?? '?'}`
          + ` · MPI ${p.mpi_available ? 'yes' : 'unavailable'}`
          + ` · charge ${rec.charge_recorded ? `every ${interval} ms` : 'not sampled'}`;
        nestInfo.title = rec.charge_recorded
          ? 'NEST owns the clock, delivery order and delays. Membrane charge was sampled by '
            + `a multimeter every ${interval} ms; ticks between samples carry no charge field `
            + 'and are hatched as unknown rather than drawn as zero.'
          : 'NEST owns the clock, delivery order and delays. Membrane charge was not '
            + 'sampled in this run, so charge panels report it as not recorded.';
      }
      // Branching from a NEST replay has no defined semantics yet (the prompt defers it),
      // so the control is disabled with the reason visible rather than silently routing
      // NEST weights into the legacy Python engine.
      const branch = $('rp-branch');
      if (branch) {
        branch.disabled = true;
        branch.title = 'Not available for a NEST replay: NEST-specific branching semantics '
          + 'are not defined or validated.';
      }
    } else {
      if (badge) { badge.textContent = 'REPLAY'; badge.classList.remove('nest'); badge.title = ''; }
      if (nestInfo) { nestInfo.hidden = true; nestInfo.textContent = ''; }
      const branch = $('rp-branch');
      if (branch) { branch.disabled = false; branch.title =
        'Create a fresh live simulation using the learned weights at this frame (not an exact resume)'; }
    }

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
    this._closeBranchDialog();
    this._setBranchBusy(false);
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

  // --------------------------------------------------------------- events
  // Jump to the next/previous frame that actually CARRIES an event.
  //
  // At full resolution most frames are charge samples and nothing else: an 18 000-frame
  // NEST artifact holds 120 frames with a spike or an input on them, so 99.3% of stepping
  // -- and 99.3% of playback time -- lands on ticks where nothing happened. The RGC volley
  // and the first cortical spike it causes are 362 frames apart, which at the default 8x
  // is 45 seconds of watching an unchanging raster. The dead time is real (it is a 3.6 ms
  // conduction delay) and playback keeps showing it truthfully; this navigates it.
  _eventJump(dir) {
    const positions = this._eventPositions();
    if (!positions.length) return;
    let target = null;
    if (dir > 0) { for (const p of positions) if (p > this.pos) { target = p; break; } }
    else { for (let i = positions.length - 1; i >= 0; i--) if (positions[i] < this.pos) { target = positions[i]; break; } }
    if (target == null) return;
    this._pause();
    this._seekTo(target, false);
  }

  // Frame positions carrying a spike or an input, computed once per loaded replay.
  // NEST frames declare them explicitly; a legacy frame is scanned for a spiked neuron.
  _eventPositions() {
    if (!this._eventPos) {
      this._eventPos = [];
      this.replay.frames.forEach((f, i) => {
        const nest = f.dynamic?.nest;
        const has = nest
          ? (nest.spike_events?.length > 0 || nest.input_events?.length > 0)
          : (f.dynamic?.neurons || []).some(n => n.spiked);
        if (has) this._eventPos.push(i);
      });
    }
    return this._eventPos;
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
    this._eventPos = null;    // ditto -- both are indexes INTO the replay being replaced
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
    // A NEST tick is NOT a legacy engine boundary, so it is never labelled `t=<int>`.
    // The physical millisecond time carried on the frame is shown instead, with the tick
    // alongside it for cross-referencing the raw artifact.
    const nestFrame = f.dynamic?.nest ?? null;
    const frameLabel = (this.isNest && nestFrame)
      ? `frame ${this.pos + 1}/${n} · ${nestFrame.t_ms} ms (tick ${nestFrame.tick})`
        + (nestFrame.window != null ? ` · window ${nestFrame.window}` : '')
      : `frame ${this.pos + 1}/${n} · t=${f.timestep}`;
    $('rp-frame').textContent = frameLabel;
    // The overlays hide the bar above, so they carry the same readout -- without it a
    // mirrored transport would advance time with no way to see where it landed. It is the
    // SHORT form: the overlay bars are already crowded, and the raw tick is redundant next
    // to a chart whose axis is time. The full label stays on the replay bar itself.
    const short = (this.isNest && nestFrame)
      ? `${this.pos + 1}/${n} · ${nestFrame.t_ms} ms`
        + (nestFrame.window != null ? ` · w${nestFrame.window}` : '')
      : `${this.pos + 1}/${n} · t=${f.timestep}`;
    for (const v of MIRRORED_OVERLAYS) {
      const e = $(`${v}-rp-frame`); if (e) e.textContent = short;
    }
    const a = f.annotation || {};
    if (this.isNest && nestFrame) {
      // Winner multiplicity is THE headline NEST observation; surfacing it on the
      // transport keeps multi-winner behaviour visible rather than buried in a panel.
      const mult = nestFrame.winner_multiplicity ?? {};
      const cols = Object.keys(mult);
      const worst = cols.length ? Math.max(...cols.map(c => mult[c])) : 0;
      $('rp-phase').textContent = cols.length
        ? `winners this tick: ${cols.map(c => `${c}×${mult[c]}`).join(' ')}`
          + (worst > 1 ? ' · MULTI-WINNER' : '')
        : `pattern: ${a.pattern ?? '—'}`;
    } else {
      $('rp-phase').textContent = `phase: ${a.phase ?? '—'} · pattern: ${a.pattern ?? '—'}`;
    }
    if (this.markerSelect) this.markerSelect.value = '';   // reset the picker label

    const ts = $('st-timestep'); if (ts) ts.textContent = f.timestep;
    const win = $('st-winner'); if (win) win.textContent = f.dynamic.winner || '—';
    const sp = $('st-speed'); if (sp) sp.textContent = `${this.speed}× replay`;
  }

  _setPlayIcon() {
    for (const id of ['rp-play', ...MIRRORED_OVERLAYS.map(v => `${v}-rp-play`)]) {
      const b = $(id);
      if (b) { b.textContent = this.playing ? '⏸' : '▶'; b.title = this.playing ? 'Pause playback' : 'Play'; }
    }
  }

  _guard(on) {
    document.body.classList.toggle('replay-mode', on);
    for (const id of GUARDED_IDS) { const e = $(id); if (e) e.disabled = on; }
    // Swap live transport for replay transport inside every overlay, so exactly one of
    // them is present and none of them is a dead control.
    for (const v of MIRRORED_OVERLAYS) {
      const live = $(`${v}-transport`), rp = $(`${v}-rp-transport`);
      if (live) live.hidden = on;
      if (rp) rp.hidden = !on;
    }
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
