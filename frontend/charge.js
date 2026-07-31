// Full-screen charge-over-time view: per neuron, the vertical axis within its
// lane is membrane CHARGE (potential / threshold). A per-timestep bar rises with
// charge, a dashed line marks threshold (V/theta = 1), and a spike is a full-height
// bright peak. Inhibitory discharges remain separate event markers.
//
// Same virtualized/coalesced rendering as the spike raster (viewport-sized
// canvas, only the visible time window drawn).

const MARGIN = 66;
const AXIS = 16;
// Retained-timestep cap. A LIVE run has no end, so it needs a rolling window; a REPLAY is
// a finite recorded file and is shown in full (see `setHistoryLimit`), because hiding
// everything older than the last 1500 frames of a file the user deliberately loaded would
// be a silent truncation of their own data.
const LIVE_HISTORY = 1500;
const CHARGE_CAP = 2.0;   // V/θ at the top of a lane's charge zone (overshoot visible up to 2x)
const INH_REF = 6.0;      // conductance increment mapped to the tallest violet inhibition marker
// The violet L1 inhibition marker is an ANNOTATION on the lane, not the lane's content:
// it must stay readable without hiding the charge bar or spike drawn at the same
// timestep. These cap its top-anchored tick to a fraction of the lane height and hold it
// translucent, so a weak pulse is still distinguishable from a strong one (both scale
// between the MIN and MAX of each pair) without any of them dominating the column.
const INH_MARK_H_MIN = 0.10, INH_MARK_H_MAX = 0.32;   // fraction of lane height
const INH_MARK_A_MIN = 0.35, INH_MARK_A_MAX = 0.70;   // alpha

export class ChargeChart {
  constructor(store) {
    this.store = store;
    this.overlay = document.getElementById('charge-overlay');
    this.scroll = document.getElementById('charge-scroll');
    this.spacer = document.getElementById('charge-spacer');
    this.canvas = document.getElementById('charge-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.order = [];
    this.charge = [];      // Float32Array per timestep: activation (V/θ)
    this.available = [];   // Uint8Array per timestep: 1 = charge was actually sampled
    this.spike = [];       // Uint8Array per timestep
    this.inhibited = [];   // Uint8Array: L2I competitive reset reached this L2E lane
    // Applied paired L1I->L1E inhibition (distinct from the L2 reset above): the
    // fraction of the L1E's pre-charge removed by the learned gate this step, and a
    // flag for a full reset (floored at rest). Sourced from dyn.applied_inhibition,
    // NOT from emitted:[li{i}] -- so the marker lands at t+1 when inhibition applies,
    // not at t when the source L1I spikes.
    this.l1inh = [];       // Float32Array: removed / v_pre in [0,1]
    this.l1rest = [];      // Uint8Array: 1 if the gate floored the pixel to rest
    this.times = [];
    this.limit = LIVE_HISTORY;
    this.showL1 = true;
    this.column = 'all';   // 'all' or a tiled column_id; filters rendered lanes only
    this.colW = 8;
    this.follow = true;
    this.built = false;
    this._raf = 0;
    this._cw = this._ch = 0;

    document.querySelector('.tab[data-tab="charge"]')?.addEventListener('click', () => this.open());
    document.getElementById('charge-close')?.addEventListener('click', () => this.close());
    document.getElementById('charge-zoom-in')?.addEventListener('click', () => this._zoom(1.5));
    document.getElementById('charge-zoom-out')?.addEventListener('click', () => this._zoom(1 / 1.5));
    const l1 = document.getElementById('charge-l1');
    l1?.addEventListener('change', () => { this.showL1 = l1.checked; this._schedule(); });
    this.columnSel = document.getElementById('charge-column');
    this.columnWrap = document.getElementById('charge-column-wrap');
    this.columnSel?.addEventListener('change', () => {
      this.column = this.columnSel.value;
      this._syncL1Disabled();
      this._schedule();
    });
    this.scroll?.addEventListener('scroll', () => {
      const s = this.scroll;
      this.follow = s.scrollLeft + s.clientWidth >= s.scrollWidth - 6;
      this._schedule();
    });
    this.scroll?.addEventListener('wheel', (e) => this._wheelZoom(e), { passive: false });
    window.addEventListener('resize', () => this._schedule());
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this._open()) this.close(); });
  }

  // A replay is finite and is shown whole; a live run keeps the rolling window. Called by
  // app.js on replay enter/exit. Shrinking takes effect on the next update, and any
  // backward seek rebuilds the window from scratch anyway.
  setHistoryLimit(n) { this.limit = n; }

  _open() { return this.overlay && !this.overlay.hidden; }
  _schedule() {
    if (this._raf || !this._open()) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }

  build(topo) {
    this.order = (topo?.neurons ?? []).map(n => ({
      id: n.id, type: n.type, group: n.layer + n.type, layer: n.layer,
      column_id: n.column_id ?? null, column_role: n.column_role ?? null,
      column_index: n.column_index ?? null,
    }));
    this.index = new Map(this.order.map((n, i) => [n.id, i]));
    this.charge = []; this.spike = []; this.inhibited = [];
    this.l1inh = []; this.l1rest = []; this.times = [];
    this.available = [];
    this.built = true;
    this._buildColumnSelector(topo);
  }

  // Rebuild the tiled cortical-column selector from topology metadata. Legacy
  // topologies (no tiling.columns) hide the selector and keep 'all' behaviour.
  _buildColumnSelector(topo) {
    const sel = this.columnSel, wrap = this.columnWrap;
    if (!sel || !wrap) return;
    const tiling = topo?.tiling, columns = tiling?.columns;
    if (!columns || !columns.length) {
      this.column = 'all';
      sel.innerHTML = '';
      wrap.hidden = true;
      this._syncL1Disabled();
      return;
    }
    // "All neurons" first, then every column in topology-provided order.
    sel.innerHTML = '';
    const all = document.createElement('option');
    all.value = 'all'; all.textContent = 'All neurons';
    sel.appendChild(all);
    for (const c of columns) {
      const o = document.createElement('option');
      o.value = c.id; o.textContent = c.id;
      sel.appendChild(o);
    }
    wrap.hidden = false;
    // Default to the L1 column matching selected_patch [row, col]; else the first
    // L1 column; else "All neurons".
    const patch = tiling.selected_patch;
    const l1cols = columns.filter(c => c.layer === 'L1');
    let def = 'all';
    if (l1cols.length) {
      def = l1cols[0].id;
      if (Array.isArray(patch)) {
        const m = l1cols.find(c => c.row === patch[0] && c.col === patch[1]);
        if (m) def = m.id;
      }
    }
    this.column = def;
    sel.value = def;
    this._syncL1Disabled();
  }

  // The Layer 1 checkbox only applies in "All neurons" mode; a specific column is
  // shown in full regardless, so disable the checkbox while one is selected.
  _syncL1Disabled() {
    const l1 = document.getElementById('charge-l1');
    if (l1) l1.disabled = this.column !== 'all';
  }

  update(dyn) {
    if (!this.built || !dyn || !dyn.neurons) return;
    const nN = this.order.length;
    const chg = new Float32Array(nN), spk = new Uint8Array(nN), inh = new Uint8Array(nN);
    const linh = new Float32Array(nN), lrest = new Uint8Array(nN);
    // Charge availability, per neuron per timestep. A source that never sampled membrane
    // state (the NEST prototype records spikes only) sends `activation: null`, which is
    // UNKNOWN and must not be drawn as an empty bar -- an empty bar reads as "no charge",
    // which is a different and false claim. `have[i] = 0` suppresses the bar entirely and
    // the lane is hatched instead.
    const have = new Uint8Array(nN);
    for (const n of dyn.neurons) {
      const i = this.index.get(n.id);
      if (i == null) continue;
      if (n.activation != null && Number.isFinite(n.activation)) {
        chg[i] = n.activation;
        have[i] = 1;
      }
      if (n.spiked) spk[i] = 1;
    }
    this.available.push(have);
    // Each inhibitory pulse this step is a persistent-conductance increment (NOT a
    // charge removal). An L2E target is the L2I_WTA global pulse (red tick); an L1E
    // target is a predictive PI (or legacy L1I) conductance pulse (violet marker,
    // height/opacity scale with the conductance increment).
    for (const ev of dyn.inhibitory_pulses || []) {
      const i = this.index.get(ev.target);
      if (i == null) continue;
      const frac = Math.min(1, (ev.conductance_increment || 0) / INH_REF);
      if (ev.target.startsWith('L2E')) {
        inh[i] = 1;
      } else {
        linh[i] = Math.max(linh[i], frac);
        lrest[i] = frac >= 0.8 ? 1 : lrest[i];   // 1 = strong (near-shunting) pulse
      }
    }
    // A HARD RESET is a different mechanism from the conductance pulse above: the target's
    // accumulated charge is cleared outright, with nothing persisting. It is drawn at full
    // strength because that is what it is -- there is no partial hard reset. NEST's models
    // have only this form of inhibition (no `g_inh` exists to increment), and the live
    // engine reports its own immediate resets in the same field, so both now render.
    for (const ev of dyn.hard_reset_events || []) {
      const i = this.index.get(ev.target);
      if (i == null) continue;
      if (ev.target.startsWith('L2E')) inh[i] = 1;
      else { linh[i] = 1; lrest[i] = 1; }
    }
    this.charge.push(chg); this.spike.push(spk); this.inhibited.push(inh);
    this.l1inh.push(linh); this.l1rest.push(lrest); this.times.push(dyn.timestep);
    while (this.charge.length > this.limit) {
      this.charge.shift(); this.spike.shift(); this.inhibited.shift();
      this.l1inh.shift(); this.l1rest.shift(); this.times.shift();
      this.available.shift();
    }
    this._schedule();
  }

  // Discard accumulated charge/spike/inhibition history so a replay backward-seek
  // rebuilds only the bounded window ending at the target frame (truthful).
  reset() {
    this.charge = []; this.spike = []; this.inhibited = [];
    this.l1inh = []; this.l1rest = []; this.times = [];
    this.available = [];
    this._schedule();
  }

  open() { this.overlay.hidden = false; this.follow = true; this._cw = this._ch = 0; this._draw(); }
  close() { this.overlay.hidden = true; }

  _zoom(f) {
    this._anchor = this.follow ? null : (this.scroll.scrollLeft + this.scroll.clientWidth / 2 - MARGIN) / this.colW;
    this.colW = Math.max(3, Math.min(48, this.colW * f));
    this._draw();
  }

  // Cursor-anchored zoom on a vertical wheel; shift+wheel or a horizontal wheel
  // (trackpad) falls through to normal timeline panning.
  _wheelZoom(e) {
    if (!this._open()) return;
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    const mx = e.clientX - this.scroll.getBoundingClientRect().left;   // cursor x in viewport
    const col = (this.scroll.scrollLeft + Math.max(mx, MARGIN) - MARGIN) / this.colW;
    const next = Math.max(3, Math.min(48, this.colW * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    if (next === this.colW) return;
    this.colW = next;
    this.follow = false;                    // anchored to the cursor, not the live edge
    this._anchor = col; this._anchorPx = mx;
    this._draw();
  }

  _lanes() {
    // A specific column overrides the Layer 1 filter and shows that column in full.
    if (this.column && this.column !== 'all')
      return this.order.filter(n => n.column_id === this.column);
    return this.showL1 ? this.order
      : this.order.filter(n => !n.group.startsWith('L1') && !n.group.startsWith('ERR'));
  }

  _draw() {
    if (!this._open() || !this.built) return;
    const lanes = this._lanes();
    const n = lanes.length;
    if (!n) return;
    const vw = this.scroll.clientWidth, vh = this.scroll.clientHeight;
    if (vw < 2 || vh < 2) return;
    const cols = this.charge.length;
    this.spacer.style.width = (MARGIN + cols * this.colW) + 'px';

    if (this._anchor != null) {
      const px = this._anchorPx != null ? this._anchorPx : this.scroll.clientWidth / 2;
      const S = this._anchor * this.colW + MARGIN - px;
      this.scroll.scrollLeft = Math.max(0, Math.min(S, this.scroll.scrollWidth - vw));
      this._anchor = this._anchorPx = null;
    } else if (this.follow) {
      this.scroll.scrollLeft = this.scroll.scrollWidth;
    }

    const dpr = window.devicePixelRatio || 1;
    if (this._cw !== vw || this._ch !== vh) {
      this.canvas.style.width = vw + 'px'; this.canvas.style.height = vh + 'px';
      this.canvas.width = Math.round(vw * dpr); this.canvas.height = Math.round(vh * dpr);
      this._cw = vw; this._ch = vh;
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, vw, vh);

    const css = getComputedStyle(document.documentElement);
    const cExc = css.getPropertyValue('--exc').trim() || '#5eead4';
    const cInh = css.getPropertyValue('--inh').trim() || '#f0788c';
    // Applied paired L1I->L1E inhibition marker: a distinct violet, tying the
    // feedback->inhibition story together and separating it from the red L2 reset.
    const cL1inh = css.getPropertyValue('--fb').trim() || '#c084fc';
    const cLine = css.getPropertyValue('--line').trim() || '#242b3a';
    const cTxt = css.getPropertyValue('--txt-1').trim() || '#c7d0e0';
    const cMut = css.getPropertyValue('--txt-2').trim() || '#5f6b82';

    const laneH = (vh - AXIS) / n;
    const y0 = AXIS;
    const barW = Math.max(1, this.colW - 1);
    const zone = laneH * 0.82;
    const thrFrac = 1 / CHARGE_CAP;
    const scrollX = this.scroll.scrollLeft;
    const cFrom = Math.max(0, Math.floor((scrollX - MARGIN) / this.colW) - 1);
    const cTo = Math.min(cols, Math.ceil((scrollX - MARGIN + vw) / this.colW) + 1);
    const xOf = (c) => MARGIN + (c * this.colW - scrollX);

    // Stripes + threshold guide lines.
    let gi = 0;
    for (let i = 0; i < n; i++) {
      const g = lanes[i].group, prev = i ? lanes[i - 1].group : null;
      if (g !== prev) {
        gi++;
        let j = i; while (j < n && lanes[j].group === g) j++;
        ctx.fillStyle = (gi % 2) ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.05)';
        ctx.fillRect(0, y0 + i * laneH, vw, (j - i) * laneH);
        ctx.strokeStyle = cLine; ctx.beginPath();
        ctx.moveTo(0, y0 + i * laneH + .5); ctx.lineTo(vw, y0 + i * laneH + .5); ctx.stroke();
      }
      const baseY = y0 + (i + 1) * laneH - 1;
      const ty = Math.round(baseY - thrFrac * zone) + .5;
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.setLineDash([3, 3]); ctx.beginPath();
      ctx.moveTo(MARGIN, ty); ctx.lineTo(vw, ty); ctx.stroke(); ctx.setLineDash([]);
    }

    // Charge bars (dim), then full-height spike peaks and inhibition markers.
    //
    // A timestep whose charge was never SAMPLED is drawn as a faint hatch rather than as
    // an empty lane. An empty lane means "charge was zero"; a hatch means "charge is
    // unknown for this run". Conflating them is exactly the failure mode the NEST replay
    // must not have -- its artifact records spikes only, so every charge value is unknown.
    ctx.globalAlpha = 0.30;
    for (let i = 0; i < n; i++) {
      const idx = this.index.get(lanes[i].id);
      ctx.fillStyle = lanes[i].type === 'E' ? cExc : cInh;
      const baseY = y0 + (i + 1) * laneH - 1;
      for (let c = cFrom; c < cTo; c++) {
        if (this.spike[c][idx]) continue;
        const sampled = !this.available[c] || this.available[c][idx];
        if (!sampled) continue;
        const a = this.charge[c][idx];
        if (a <= 0.02) continue;
        ctx.fillRect(xOf(c), baseY - Math.min(a / CHARGE_CAP, 1) * zone, barW, Math.min(a / CHARGE_CAP, 1) * zone);
      }
    }
    // Unsampled hatch: one thin mid-lane dash per unknown timestep.
    ctx.globalAlpha = 0.16;
    // Canvas does not resolve CSS custom properties -- assigning `var(--txt-2)` to
    // fillStyle is silently ignored and the previous lane colour is kept, so the hatch
    // rendered as a faint copy of whatever was drawn last. `cMut` is the resolved value.
    ctx.fillStyle = cMut;
    for (let i = 0; i < n; i++) {
      const idx = this.index.get(lanes[i].id);
      const midY = y0 + i * laneH + laneH * 0.62;
      for (let c = cFrom; c < cTo; c++) {
        if (this.spike[c][idx]) continue;
        if (this.available[c] && !this.available[c][idx] && ((c + i) & 1) === 0)
          ctx.fillRect(xOf(c), midY, barW, 1);
      }
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < n; i++) {
      const idx = this.index.get(lanes[i].id);
      const laneTop = y0 + i * laneH + 1;
      for (let c = cFrom; c < cTo; c++) {
        if (this.spike[c][idx]) {
          ctx.fillStyle = lanes[i].type === 'E' ? cExc : cInh;
          ctx.fillRect(xOf(c), laneTop, barW, laneH - 2);
        }
        if (this.inhibited[c][idx]) {
          ctx.fillStyle = cInh;
          ctx.fillRect(xOf(c), laneTop + Math.max(5, laneH * 0.22), barW,
                       Math.max(2, Math.min(4, laneH * 0.18)));
        }
        // Predictive/legacy inhibitory conductance pulse (L1E lanes): a top-anchored
        // violet marker whose height/opacity grow with the conductance increment;
        // a strong (near-shunting) pulse draws taller and more opaque, so weak
        // partial inhibition is visually distinct from strong predictive inhibition.
        const li = this.l1inh[c][idx];
        if (li > 0.001) {
          const rest = this.l1rest[c][idx];
          const s = rest ? 1 : li;   // strong pulses sit at the top of both ramps
          const h = (INH_MARK_H_MIN + (INH_MARK_H_MAX - INH_MARK_H_MIN) * s) * (laneH - 2);
          ctx.globalAlpha = INH_MARK_A_MIN + (INH_MARK_A_MAX - INH_MARK_A_MIN) * s;
          ctx.fillStyle = cL1inh;
          ctx.fillRect(xOf(c), laneTop, barW, Math.max(2, h));
          ctx.globalAlpha = 1;
        }
      }
    }

    // Pinned gutter + axis.
    ctx.clearRect(0, AXIS, MARGIN, vh - AXIS);
    const labelPx = Math.max(8, Math.min(11, laneH - 4));
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      ctx.font = `${labelPx}px ui-monospace, monospace`;
      ctx.fillStyle = cTxt; ctx.fillText(lanes[i].id, 5, y0 + (i + 0.5) * laneH);
    }
    ctx.strokeStyle = cLine; ctx.beginPath(); ctx.moveTo(MARGIN + .5, 0); ctx.lineTo(MARGIN + .5, vh); ctx.stroke();
    ctx.clearRect(MARGIN, 0, vw - MARGIN, AXIS);
    ctx.fillStyle = cMut; ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'top';
    if (cols) {
      // Legend: three separately-colored event types + the charge/threshold guides.
      const parts = [
        ['charge V/θ (bar) · dashed = θ', cMut],
        ['peak = spike (incl. PI / L1I)', cExc],
        ['red tick = L2I_WTA conductance', cInh],
        ['violet = predictive/L1I conductance (tall = strong)', cL1inh],
      ];
      let lx = MARGIN + 6;
      for (const [txt, col] of parts) {
        ctx.fillStyle = col; ctx.fillText(txt, lx, 3);
        lx += ctx.measureText(txt).width + 14;
      }
      ctx.fillStyle = cMut;
      ctx.fillText(`· ${cols} steps · ${this.colW.toFixed(0)} px/step · newest ->`, lx, 3);
    }
  }
}
