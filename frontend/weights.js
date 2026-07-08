// Full-screen weights-over-time view: the selected L2E neuron's 9 incoming
// feedforward weights (L1E -> L2E, one per pixel) plus its L2I -> L2E inhibitory
// gate magnitude, all as a fraction of the weight cap, over the full (bounded)
// training history. This is the view for debugging whether a receptive field is
// forming and whether the signed +1/-1 rule replaces the old budget: the three
// pattern pixels should climb toward the cap while the rest are pushed to the
// floor. Pick a neuron by clicking an L2E in the 3D view.
//
// Fit-to-width (no horizontal scroll): the whole history is compressed into the
// viewport so the learning curve is visible at a glance. History is bounded.

const HISTORY = 1500;
const N_PIX = 9;
const PAD = { l: 44, r: 66, t: 22, b: 22 };

export class WeightsChart {
  constructor(store) {
    this.store = store;
    this.overlay = document.getElementById('weights-overlay');
    this.scroll = document.getElementById('weights-scroll');
    this.canvas = document.getElementById('weights-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.targetEl = document.getElementById('weights-target');
    this.target = 'L2E0';
    this.ff = [];          // Float32Array(9) per sample
    this.gate = [];        // number per sample
    this._raf = 0;
    this._cw = this._ch = 0;

    document.querySelector('.tab[data-tab="weights"]')?.addEventListener('click', () => this.open());
    document.getElementById('weights-close')?.addEventListener('click', () => this.close());
    window.addEventListener('resize', () => this._schedule());
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this._open()) this.close(); });
  }

  _open() { return this.overlay && !this.overlay.hidden; }
  _schedule() {
    if (this._raf || !this._open()) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }

  build() { this._reset(); }
  _reset() { this.ff = []; this.gate = []; }

  setTarget(id) {
    if (!id || !id.startsWith('L2E') || id === this.target) return;
    this.target = id;
    if (this.targetEl) this.targetEl.textContent = id;
    this._reset();
    this._schedule();
  }

  update(dyn) {
    const j = parseInt(this.target.slice(3), 10);
    if (Number.isNaN(j)) return;
    const w = this.store.weights;
    const ff = new Float32Array(N_PIX);
    for (let i = 0; i < N_PIX; i++) ff[i] = w.get(`ff${i}->${j}`) ?? 0;
    this.ff.push(ff);
    this.gate.push(Math.abs(w.get(`inh->${j}`) ?? 0));
    while (this.ff.length > HISTORY) { this.ff.shift(); this.gate.shift(); }
    this._schedule();
  }

  open() { this.overlay.hidden = false; this._cw = this._ch = 0; if (this.targetEl) this.targetEl.textContent = this.target; this._draw(); }
  close() { this.overlay.hidden = true; }

  _cap() {
    const p = this.store.topology?.params || {};
    const frac = p.l2e_weight_cap_frac ?? 1;
    return (frac * (p.threshold_l2 ?? 1)) || 1;
  }

  _draw() {
    if (!this._open()) return;
    const vw = this.scroll.clientWidth, vh = this.scroll.clientHeight;
    if (vw < 2 || vh < 2) return;
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
    const cInh = css.getPropertyValue('--inh').trim() || '#f0788c';
    const cLine = css.getPropertyValue('--line').trim() || '#242b3a';
    const cMut = css.getPropertyValue('--txt-2').trim() || '#5f6b82';

    const x0 = PAD.l, x1 = vw - PAD.r, y0 = PAD.t, y1 = vh - PAD.b;
    const YMAX = 1.1;                          // fraction of cap
    const cap = this._cap();
    const N = this.ff.length;
    const xOf = (i) => N <= 1 ? x0 : x0 + (i / (N - 1)) * (x1 - x0);
    const yOf = (v) => y1 - Math.max(0, Math.min(YMAX, v)) / YMAX * (y1 - y0);

    // Axes + reference lines (0, cap).
    ctx.strokeStyle = cLine; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = cMut; ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    ctx.fillText('0', x0 - 5, y1);
    const yCap = yOf(1.0);
    ctx.strokeStyle = 'rgba(255,255,255,0.16)'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x0, yCap); ctx.lineTo(x1, yCap); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText('cap', x0 - 5, yCap);
    ctx.textAlign = 'left';
    ctx.fillText(`${this.target} · weight ÷ cap (${cap.toFixed(0)}) · ${N} samples · newest →`, x0 + 4, y0 - 8 < 6 ? 8 : y0 - 8);

    if (N < 2) return;

    // 9 feedforward series (teal, lightness by pixel index) + endpoint labels.
    for (let i = 0; i < N_PIX; i++) {
      ctx.strokeStyle = `hsl(168, 70%, ${38 + i * 4}%)`;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let k = 0; k < N; k++) {
        const v = this.ff[k][i] / cap;
        k ? ctx.lineTo(xOf(k), yOf(v)) : ctx.moveTo(xOf(k), yOf(v));
      }
      ctx.stroke();
      const last = this.ff[N - 1][i] / cap;
      ctx.fillStyle = `hsl(168, 70%, ${48 + i * 4}%)`;
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillText(`p${i} ${last.toFixed(2)}`, x1 + 4, yOf(last));
    }

    // Inhibitory gate magnitude (red, thicker).
    ctx.strokeStyle = cInh; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = 0; k < N; k++) {
      const v = this.gate[k] / cap;
      k ? ctx.lineTo(xOf(k), yOf(v)) : ctx.moveTo(xOf(k), yOf(v));
    }
    ctx.stroke();
  }
}
