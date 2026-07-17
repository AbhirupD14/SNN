// Full-screen charge-over-time view: per neuron, the vertical axis within its
// lane is membrane CHARGE (potential / threshold). A per-timestep bar rises with
// charge, a dashed line marks threshold (V/theta = 1), and a spike is a full-height
// bright peak. Inhibitory discharges remain separate event markers.
//
// Same virtualized/coalesced rendering as the spike raster (viewport-sized
// canvas, only the visible time window drawn).

const MARGIN = 66;
const AXIS = 16;
const HISTORY = 1500;
const CHARGE_CAP = 2.0;   // V/θ at the top of a lane's charge zone (overshoot visible up to 2x)
const INH_REF = 6.0;      // conductance increment mapped to a full-height violet inhibition marker

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
    this.showL1 = true;
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
    this.order = (topo?.neurons ?? []).map(n => ({ id: n.id, type: n.type, group: n.layer + n.type }));
    this.index = new Map(this.order.map((n, i) => [n.id, i]));
    this.charge = []; this.spike = []; this.inhibited = [];
    this.l1inh = []; this.l1rest = []; this.times = [];
    this.built = true;
  }

  update(dyn) {
    if (!this.built || !dyn || !dyn.neurons) return;
    const nN = this.order.length;
    const chg = new Float32Array(nN), spk = new Uint8Array(nN), inh = new Uint8Array(nN);
    const linh = new Float32Array(nN), lrest = new Uint8Array(nN);
    for (const n of dyn.neurons) {
      const i = this.index.get(n.id);
      if (i == null) continue;
      chg[i] = n.activation ?? 0;
      if (n.spiked) spk[i] = 1;
    }
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
    this.charge.push(chg); this.spike.push(spk); this.inhibited.push(inh);
    this.l1inh.push(linh); this.l1rest.push(lrest); this.times.push(dyn.timestep);
    while (this.charge.length > HISTORY) {
      this.charge.shift(); this.spike.shift(); this.inhibited.shift();
      this.l1inh.shift(); this.l1rest.shift(); this.times.shift();
    }
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

  _lanes() { return this.showL1 ? this.order
    : this.order.filter(n => !n.group.startsWith('L1') && !n.group.startsWith('ERR')); }

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
    ctx.globalAlpha = 0.30;
    for (let i = 0; i < n; i++) {
      const idx = this.index.get(lanes[i].id);
      ctx.fillStyle = lanes[i].type === 'E' ? cExc : cInh;
      const baseY = y0 + (i + 1) * laneH - 1;
      for (let c = cFrom; c < cTo; c++) {
        if (this.spike[c][idx]) continue;
        const a = this.charge[c][idx];
        if (a <= 0.02) continue;
        ctx.fillRect(xOf(c), baseY - Math.min(a / CHARGE_CAP, 1) * zone, barW, Math.min(a / CHARGE_CAP, 1) * zone);
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
        // a strong (near-shunting) pulse draws taller and fully opaque, so weak
        // partial inhibition is visually distinct from strong predictive inhibition.
        const li = this.l1inh[c][idx];
        if (li > 0.001) {
          const rest = this.l1rest[c][idx];
          const h = (0.30 + 0.55 * li) * (laneH - 2);
          ctx.globalAlpha = rest ? 1 : 0.45 + 0.5 * li;
          ctx.fillStyle = cL1inh;
          ctx.fillRect(xOf(c), laneTop, barW, h);
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
