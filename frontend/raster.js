// Full-screen spike raster: every neuron gets its own labeled lane, stacked on a
// shared time axis. Marks are DISCRETE SPIKES ONLY -- no charge/membrane trace
// (that lives in the separate Charge-over-time view). If a neuron did not spike
// at a timestep, its lane shows nothing there.
//
// PERFORMANCE: the canvas is sized to the VIEWPORT, not the whole history. A
// spacer gives the scroll container its virtual width, and only the visible time
// window is drawn (virtualized), so the backing store stays small. Draws are
// coalesced to one per animation frame.

const MARGIN = 66;        // pinned left gutter: id label + rate bar
const AXIS = 16;          // top axis strip
const HISTORY = 1500;     // timesteps retained

// Lane colour by neuron class: 'E' cortical excitatory, 'S' exogenous RG source,
// anything else inhibitory.
function laneColor(type, cExc, cInh, cRg) {
  return type === 'E' ? cExc : type === 'S' ? cRg : cInh;
}

export class Raster {
  constructor(store) {
    this.store = store;
    this.overlay = document.getElementById('raster-overlay');
    this.scroll = document.getElementById('raster-scroll');
    this.spacer = document.getElementById('raster-spacer');
    this.canvas = document.getElementById('raster-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.order = [];
    this.spike = [];       // Uint8Array per timestep: 1 = spiked
    this.times = [];
    this.rate = new Map();
    this.showL1 = true;
    this.column = 'all';   // 'all' or a tiled column_id; filters rendered lanes only
    this.colW = 6;
    this.follow = true;
    this.built = false;
    this._raf = 0;
    this._cw = this._ch = 0;

    document.querySelector('.tab[data-tab="raster"]')?.addEventListener('click', () => this.open());
    document.getElementById('raster-close')?.addEventListener('click', () => this.close());
    document.getElementById('raster-zoom-in')?.addEventListener('click', () => this._zoom(1.5));
    document.getElementById('raster-zoom-out')?.addEventListener('click', () => this._zoom(1 / 1.5));
    const l1 = document.getElementById('raster-l1');
    l1?.addEventListener('change', () => { this.showL1 = l1.checked; this._schedule(); });
    this.columnSel = document.getElementById('raster-column');
    this.columnWrap = document.getElementById('raster-column-wrap');
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
    this.spike = []; this.times = [];
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
    const l1 = document.getElementById('raster-l1');
    if (l1) l1.disabled = this.column !== 'all';
  }

  update(dyn) {
    if (!this.built || !dyn || !dyn.neurons) return;
    const spk = new Uint8Array(this.order.length);
    for (const n of dyn.neurons) {
      const i = this.index.get(n.id);
      if (i != null && n.spiked) spk[i] = 1;
    }
    this.spike.push(spk); this.times.push(dyn.timestep);
    while (this.spike.length > HISTORY) { this.spike.shift(); this.times.shift(); }
    this.rate = new Map(dyn.neurons.map(n => [n.id, n.freq ?? 0]));
    this._schedule();
  }

  // Discard accumulated spike history so a replay backward-seek can rebuild only
  // the bounded window ending at the target frame (truthful, not carried over).
  reset() { this.spike = []; this.times = []; this.rate = new Map(); this._schedule(); }

  open() { this.overlay.hidden = false; this.follow = true; this._cw = this._ch = 0; this._draw(); }
  close() { this.overlay.hidden = true; }

  _zoom(f) {
    this._anchor = this.follow ? null : (this.scroll.scrollLeft + this.scroll.clientWidth / 2 - MARGIN) / this.colW;
    this.colW = Math.max(2, Math.min(40, this.colW * f));
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
    const next = Math.max(2, Math.min(40, this.colW * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
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
    const cols = this.spike.length;
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
    const cRg = css.getPropertyValue('--rg').trim() || '#fbbf24';
    const cLine = css.getPropertyValue('--line').trim() || '#242b3a';
    const cTxt = css.getPropertyValue('--txt-1').trim() || '#c7d0e0';
    const cMut = css.getPropertyValue('--txt-2').trim() || '#5f6b82';

    const laneH = (vh - AXIS) / n;
    const y0 = AXIS;
    const barW = Math.max(1.5, this.colW - 1);
    const scrollX = this.scroll.scrollLeft;
    const cFrom = Math.max(0, Math.floor((scrollX - MARGIN) / this.colW) - 1);
    const cTo = Math.min(cols, Math.ceil((scrollX - MARGIN + vw) / this.colW) + 1);
    const xOf = (c) => MARGIN + (c * this.colW - scrollX);
    const tickH = Math.min(laneH - 2, Math.max(3, laneH * 0.6));

    // Group stripes + separators.
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
    }

    // Spikes (discrete marks), lane-outer so fillStyle is set once per lane.
    for (let i = 0; i < n; i++) {
      const idx = this.index.get(lanes[i].id);
      ctx.fillStyle = laneColor(lanes[i].type, cExc, cInh, cRg);
      const yc = y0 + i * laneH + (laneH - tickH) / 2;
      for (let c = cFrom; c < cTo; c++) if (this.spike[c][idx]) ctx.fillRect(xOf(c), yc, barW, tickH);
    }

    // Pinned gutter: id labels + firing-rate bars + divider.
    ctx.clearRect(0, AXIS, MARGIN, vh - AXIS);
    const labelPx = Math.max(8, Math.min(11, laneH - 4));
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      const cy = y0 + (i + 0.5) * laneH;
      ctx.font = `${labelPx}px ui-monospace, monospace`;
      ctx.fillStyle = cTxt; ctx.fillText(lanes[i].id, 5, cy);
      const f = Math.max(0, Math.min(1, this.rate.get(lanes[i].id) ?? 0));
      ctx.fillStyle = laneColor(lanes[i].type, cExc, cInh, cRg); ctx.globalAlpha = 0.5;
      ctx.fillRect(MARGIN - 15, cy - 1.5, 12 * f, 3); ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = cLine; ctx.beginPath(); ctx.moveTo(MARGIN + .5, 0); ctx.lineTo(MARGIN + .5, vh); ctx.stroke();
    ctx.clearRect(MARGIN, 0, vw - MARGIN, AXIS);
    ctx.fillStyle = cMut; ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'top';
    if (cols) ctx.fillText(`spikes only · ${cols} steps · ${this.colW.toFixed(0)} px/step · newest →`, MARGIN + 6, 3);
  }
}
