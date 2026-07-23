// Full-screen weights-over-time view: the selected cell's incoming feedforward
// weights (one series per afferent) as a fraction of the neuron-wide maturity budget
// B = e_maturity_budget_frac*theta (ordinary-E weights are cap-free; the FE budget, not
// a per-synapse ceiling, is the natural reference), over the full (bounded) training
// history. Topology-generic: the afferent set is read from the live topology (feedforward
// edges into the target), so it works for any plastic cell with any fan-in -- an L2E's
// child-Eor afferents, an Eor's local ordinary-E afferents, etc. Pick a target by
// clicking a competitor/Eor in the 3D view.
//
// Fit-to-width (no horizontal scroll): the whole history compresses into the viewport.

const HISTORY = 1500;
const PAD = { l: 44, r: 84, t: 22, b: 22 };
// Roles that own plastic feedforward weights (mirrors ARCHETYPES[*]['plastic_ff']).
const PLASTIC_ROLES = new Set(['competitor', 'encoder']);

export class WeightsChart {
  constructor(store) {
    this.store = store;
    this.overlay = document.getElementById('weights-overlay');
    this.scroll = document.getElementById('weights-scroll');
    this.canvas = document.getElementById('weights-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.targetEl = document.getElementById('weights-target');
    this.target = null;
    this.edges = [];       // [{id, label}] incoming feedforward edges of the target
    this.hist = [];        // Float32Array(edges.length) per sample
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

  // Rebuild on topology change: keep the current target if it still exists, else pick
  // the first plastic cell. Recompute its incoming feedforward afferent set.
  // Any cell that OWNS plastic feedforward weights is a valid target -- competitors and,
  // in the 'rg' topology, the L1E encoders (whose single RG afferent is charted here).
  build() {
    const comps = (this.store.topology?.neurons || []).filter(n => PLASTIC_ROLES.has(n.role));
    if (!comps.length) { this.target = null; this.edges = []; this._reset(); return; }
    if (!this.target || !comps.some(n => n.id === this.target)) this.target = comps[0].id;
    this._recomputeEdges();
    this._reset();
  }

  _recomputeEdges() {
    const syn = this.store.topology?.synapses || [];
    const pixelByNode = new Map((this.store.topology?.neurons || [])
      .filter(n => n.pixel != null).map(n => [n.id, n.pixel]));
    this.edges = syn.filter(s => s.kind === 'feedforward' && s.target === this.target)
      .map(s => ({ id: s.id, label: pixelByNode.has(s.source) ? 'p' + pixelByNode.get(s.source) : s.source }));
  }

  _reset() { this.hist = []; }

  // Public reset used by the replay player before rebuilding a bounded history
  // window ending at a seek target (the live path uses _reset on rebuild).
  reset() { this._reset(); this._schedule(); }

  setTarget(id) {
    if (!id || id === this.target) return;
    const meta = this.store.meta?.get(id);
    if (!meta || !PLASTIC_ROLES.has(meta.role)) return;   // only plastic cells own a feedforward RF
    this.target = id;
    if (this.targetEl) this.targetEl.textContent = id;
    this._recomputeEdges();
    this._reset();
    this._schedule();
  }

  update(dyn) {
    if (!this.target || !this.edges.length) return;
    const w = this.store.weights;
    const row = new Float32Array(this.edges.length);
    for (let i = 0; i < this.edges.length; i++) row[i] = w.get(this.edges[i].id) ?? 0;
    this.hist.push(row);
    while (this.hist.length > HISTORY) this.hist.shift();
    this._schedule();
  }

  open() { this.overlay.hidden = false; this._cw = this._ch = 0; if (!this.edges.length) this.build();
           if (this.targetEl) this.targetEl.textContent = this.target || '—'; this._draw(); }
  close() { this.overlay.hidden = true; }

  // Display reference for ordinary-E weights: the neuron-wide maturity budget
  // B = e_maturity_budget_frac * theta. There is NO hard per-synapse cap; a matured
  // one-afferent specialist approaches B, so series are normalized against it.
  _ref() {
    const p = this.store.topology?.params || {};
    const thr = p.threshold_l2 ?? 1;
    return (p.e_maturity_budget ?? (thr * (p.e_maturity_budget_frac ?? 1.1))) || 1;
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
    const cLine = css.getPropertyValue('--line').trim() || '#242b3a';
    const cMut = css.getPropertyValue('--txt-2').trim() || '#5f6b82';

    const x0 = PAD.l, x1 = vw - PAD.r, y0 = PAD.t, y1 = vh - PAD.b;
    const YMAX = 1.1, ref = this._ref();
    const N = this.hist.length, M = this.edges.length;
    const xOf = (i) => N <= 1 ? x0 : x0 + (i / (N - 1)) * (x1 - x0);
    const yOf = (v) => y1 - Math.max(0, Math.min(YMAX, v)) / YMAX * (y1 - y0);

    ctx.strokeStyle = cLine; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = cMut; ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    ctx.fillText('0', x0 - 5, y1);
    const yBudget = yOf(1.0);
    ctx.strokeStyle = 'rgba(255,255,255,0.16)'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x0, yBudget); ctx.lineTo(x1, yBudget); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText('budget', x0 - 5, yBudget);
    ctx.textAlign = 'left';
    ctx.fillText(`${this.target || '—'} · ${M} feedforward afferents ÷ FE budget (${ref.toFixed(0)}) · ${N} samples · newest →`,
                 x0 + 4, y0 - 8 < 6 ? 8 : y0 - 8);

    if (N < 2 || !M) {
      if (!M) { ctx.fillStyle = cMut; ctx.textAlign = 'center';
        ctx.fillText('selected neuron has no feedforward afferents — click a competitor or encoder in the 3D view',
                     (x0 + x1) / 2, (y0 + y1) / 2); }
      return;
    }

    // One series per afferent; hue spread across the fan-in, endpoint labels at right.
    for (let i = 0; i < M; i++) {
      const hue = 168 + (i / Math.max(M, 1)) * 150;
      ctx.strokeStyle = `hsl(${hue}, 70%, 55%)`; ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let k = 0; k < N; k++) {
        const v = this.hist[k][i] / ref;
        k ? ctx.lineTo(xOf(k), yOf(v)) : ctx.moveTo(xOf(k), yOf(v));
      }
      ctx.stroke();
      const last = this.hist[N - 1][i] / ref;
      ctx.fillStyle = `hsl(${hue}, 70%, 62%)`;
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillText(`${this.edges[i].label} ${last.toFixed(2)}`, x1 + 4, yOf(last));
    }
  }
}
