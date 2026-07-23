// Receptive-field view (full-screen pop-up). One 3x3 feedforward grid per COMPETITOR
// neuron, generalized to any topology: each grid cell is the feedforward weight from
// the sensory afferent mapped to that input pixel (topology.synapses of kind
// 'feedforward', keyed by the source node's pixel). Cells with no such edge are blank.
//
// Cells show the ACTUAL weight ('weight' mode) or weight / maturity-budget reference
// ('ratio', B = e_maturity_budget_frac*theta). Ordinary-E weights are cap-free.
// When PAUSED each mapped cell is editable: type a value + Enter to set that exact
// synapse via /api/weight {synapse: edgeId}. A unit whose three strongest afferents
// can no longer sum to threshold is flagged "dead".

const N_PIX = 9;   // fixed 3x3 sensory input surface

export class ReceptiveFields {
  constructor(store, api) {
    this.store = store;
    this.api = api;
    this.overlay = document.getElementById('rf-overlay');
    this.grid = document.getElementById('rf-grid');
    this.inputEl = document.getElementById('rf-input');
    this.cards = [];
    this.inputCells = [];
    this.mode = 'weight';
    this.running = true;
    this.ffMap = new Map();   // compId -> {pixel -> edgeId}
    this._wireModeToggle();
    this._wireOverlay();
  }

  _wireOverlay() {
    document.querySelector('.tab[data-tab="rf"]')?.addEventListener('click', () => this.open());
    document.getElementById('rf-close')?.addEventListener('click', () => this.close());
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this._open()) this.close(); });
  }

  _open() { return this.overlay && !this.overlay.hidden; }
  open() { this.overlay.hidden = false; if (!this.built) this.build(); this.update(this.store.dynamic); }
  close() { this.overlay.hidden = true; }

  _wireModeToggle() {
    const toggle = document.getElementById('rf-mode-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', (e) => {
      const btn = e.target.closest('.rf-toggle-btn');
      if (!btn) return;
      this.mode = btn.dataset.mode;
      for (const b of toggle.querySelectorAll('.rf-toggle-btn'))
        b.classList.toggle('is-on', b.dataset.mode === this.mode);
      this.update(this.store.dynamic);
    });
  }

  _caps() {
    const p = this.store.topology?.params || {};
    const thr = p.threshold_l2 || 1;
    // Ordinary-E weights are cap-free; scale displays against the neuron-wide maturity
    // budget B = e_maturity_budget_frac * theta (a matured one-afferent cell approaches B),
    // NOT a hard per-synapse ceiling.
    const ref = p.e_maturity_budget ?? (thr * (p.e_maturity_budget_frac ?? 1.1));
    return { thr, ref: ref || 1 };
  }

  // Rebuild from the CURRENT topology (called on every topology broadcast, so it
  // follows the editor: competitors, sensory pixels, and feedforward edges can all
  // change). Maps each competitor's 3x3 grid cells to feedforward edge ids by pixel.
  build() {
    const topo = this.store.topology;
    if (!topo) return;
    // Tiled topology: an L1 E has a 3x3 patch RF, but L2 E (nine child-column inputs),
    // Eor (N local afferents) and C (one basal + unweighted apicals) do NOT fit a single
    // fixed 9-pixel retinal map, and the top3<theta "dead" heuristic is not valid for
    // arbitrary N/roles. Per the spec, route per-target weights to the topology-generic
    // Weights-over-time + Inspector views instead of pretending every target is a 3x3
    // pixel map. The legacy 3x3 RF below is unchanged for the five 9-pixel presets.
    if (topo.tiling) { this._buildTiledNotice(topo); this.built = true; return; }
    this._tiled = false;
    const pixelByNode = new Map();
    for (const n of topo.neurons) if (n.pixel != null) pixelByNode.set(n.id, n.pixel);
    this.ffMap = new Map();
    for (const s of topo.synapses) {
      if (s.kind !== 'feedforward') continue;
      const px = pixelByNode.get(s.source);
      if (px == null) continue;                 // afferent not tied to an input pixel
      if (!this.ffMap.has(s.target)) this.ffMap.set(s.target, {});
      this.ffMap.get(s.target)[px] = s.id;
    }
    this.compIds = topo.neurons.filter(n => n.role === 'competitor').map(n => n.id);

    // Signed-input reference grid.
    this.inputEl.innerHTML = '';
    this.inputCells = [];
    for (let i = 0; i < N_PIX; i++) {
      const c = document.createElement('div');
      c.className = 'rf-cell';
      this.inputEl.appendChild(c);
      this.inputCells.push(c);
    }
    // One card per competitor.
    this.grid.innerHTML = '';
    this.cards = [];
    for (const id of this.compIds) {
      const root = document.createElement('div');
      root.className = 'rf-card';
      const title = document.createElement('div');
      title.className = 'rf-title';
      title.innerHTML = `<span>${id}</span><span class="rf-badge" hidden></span>`;
      const cellsEl = document.createElement('div');
      cellsEl.className = 'rf-cells';
      const cells = [];
      for (let px = 0; px < N_PIX; px++) {
        const c = document.createElement('div');
        c.className = 'rf-cell rf-num';
        c.dataset.comp = id;
        c.dataset.pixel = String(px);
        this._wireCellEditing(c);
        cellsEl.appendChild(c);
        cells.push(c);
      }
      root.appendChild(title);
      root.appendChild(cellsEl);
      this.grid.appendChild(root);
      this.cards.push({ id, root, cells, badge: title.querySelector('.rf-badge') });
    }
    this.built = true;
  }

  _edgeFor(compId, pixel) { return this.ffMap.get(compId)?.[pixel]; }

  _wireCellEditing(cell) {
    cell.addEventListener('focus', () => {
      if (!cell.isContentEditable) return;
      cell.classList.add('rf-editing');
      const r = document.createRange(); r.selectNodeContents(cell);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
    });
    cell.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); cell.blur(); }
      else if (e.key === 'Escape') { e.preventDefault(); cell._cancel = true; cell.blur(); }
    });
    cell.addEventListener('blur', () => this._commitCell(cell));
  }

  _commitCell(cell) {
    cell.classList.remove('rf-editing');
    if (cell._cancel) { cell._cancel = false; this.update(this.store.dynamic); return; }
    const edgeId = this._edgeFor(cell.dataset.comp, Number(cell.dataset.pixel));
    if (!edgeId) { this.update(this.store.dynamic); return; }   // no synapse to set
    const val = parseFloat((cell.textContent || '').trim());
    if (!Number.isFinite(val)) { this.update(this.store.dynamic); return; }
    const { ref } = this._caps();
    const weight = this.mode === 'ratio' ? val * ref : val;
    this.api.post('/api/weight', { synapse: edgeId, weight });
    this.store.weights.set(edgeId, Math.max(0, weight));   // optimistic echo (cap-free, floor 0)
  }

  _buildTiledNotice(topo) {
    this._tiled = true;
    this.compIds = [];
    this.cards = [];
    this.inputCells = [];
    if (this.inputEl) this.inputEl.innerHTML = '';
    const t = topo.tiling;
    const cols = (t.columns || []).length;
    this.grid.innerHTML = `
      <div class="rf-card" style="grid-column:1/-1;max-width:640px">
        <div class="rf-title"><span>Tiled cortical columns · ${cols} columns · N=${t.cc_e_count} ordinary E/column</span></div>
        <div style="padding:10px 4px;color:var(--txt-2);font-size:12px;line-height:1.5">
          Tiled targets have heterogeneous receptive fields — an L1 ordinary E sees a 3x3
          retinal patch, an L2 ordinary E sees nine child-column Eor inputs, an Eor sees its
          N local ordinary-E afferents, and a coincidence C has one learned basal weight plus
          unweighted Boolean apical permissions. They are inspected in the topology-generic
          <b>Weights-over-time</b> view (select any ordinary E or Eor in the 3D scene) and the
          <b>Inspector</b> (per-cell incoming weights, C basal weight, coincidence gate,
          dormant top-C marker). The single fixed 9-pixel map is intentionally not shown here.
        </div>
      </div>`;
  }

  update(dyn) {
    if (!this._open()) return;
    if (!this.built) this.build();
    if (this._tiled) return;              // tiled notice is static; nothing per-frame
    const s = this.store;
    const { thr, ref } = this._caps();

    this.running = !!(dyn && dyn.running);
    const editable = !this.running;
    this.grid.classList.toggle('rf-can-edit', editable);
    const hint = document.getElementById('rf-edit-hint');
    if (hint) hint.classList.toggle('is-active', editable);

    const input = (dyn && dyn.input) || [];
    for (let i = 0; i < N_PIX; i++) {
      const on = input[i] > 0;
      const c = this.inputCells[i];
      c.classList.toggle('sig-plus', on);
      c.classList.toggle('sig-minus', !on);
      c.textContent = on ? '+' : '−';
    }

    const winner = dyn && dyn.winner;
    for (const card of this.cards) {
      const fireRatios = [];
      for (let px = 0; px < N_PIX; px++) {
        const cell = card.cells[px];
        const edgeId = this._edgeFor(card.id, px);
        if (!edgeId) {                          // no feedforward edge for this pixel
          cell.contentEditable = 'false';
          cell.classList.remove('rf-editable');
          cell.classList.add('rf-blank');
          cell.style.background = 'transparent';
          if (document.activeElement !== cell) cell.textContent = '';
          fireRatios.push(0);
          continue;
        }
        cell.classList.remove('rf-blank');
        const w = s.weights.get(edgeId) ?? 0;
        fireRatios.push(w / thr);
        cell.contentEditable = editable ? 'true' : 'false';
        cell.classList.toggle('rf-editable', editable);
        const norm = Math.max(0, Math.min(1, w / ref));
        cell.style.background = norm > 0.001
          ? `rgba(94,234,212,${(0.08 + 0.92 * norm).toFixed(3)})` : 'transparent';
        if (document.activeElement === cell) continue;
        cell.textContent = this.mode === 'ratio'
          ? (w / ref).toFixed(3)
          : (w >= 0.05 ? w.toFixed(1) : '0');
      }
      const top3 = [...fireRatios].sort((a, b) => b - a).slice(0, 3).reduce((a, b) => a + b, 0);
      const dead = top3 < 1.0;
      card.badge.hidden = !dead;
      if (dead) card.badge.textContent = 'dead';
      card.root.classList.toggle('rf-dead', dead);

      const st = s.stateById.get(card.id);
      card.root.classList.toggle('rf-spike', !!(st && st.spiked));
      card.root.classList.toggle('rf-winner', winner === card.id);
    }
  }
}
