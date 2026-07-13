// Receptive-field view for the minimal SIGNED-SPIKE experiment.
//
// Renders, live, one 3x3 feedforward receptive field per L2E neuron. Cells show the
// ACTUAL weight by default (a "ratio" mode shows weight / per-afferent cap in [0,1]
// with more precision, since near-balanced weights collapse to the same 2-decimal
// ratio). When the simulation is PAUSED, each cell is editable: type a value and press
// Enter to set that synapse (weight, or ratio*cap), which the backend applies (clipped
// to the cap) so you can hand-push a neuron toward winning or losing, then step/resume
// to watch the effect. A unit is flagged "dead" once its three strongest pixels can no
// longer sum to threshold (it can never fire again).

const N_PIX = 9;   // 3x3 grid

export class ReceptiveFields {
  constructor(store, api) {
    this.store = store;
    this.api = api;
    this.grid = document.getElementById('rf-grid');
    this.inputEl = document.getElementById('rf-input');
    this.cards = [];        // per-L2E { root, cells[9], badge }
    this.inputCells = [];
    this.built = false;
    this.mode = 'weight';   // 'weight' | 'ratio'
    this.running = true;    // paused => editable
    this._wireModeToggle();
  }

  _wireModeToggle() {
    const toggle = document.getElementById('rf-mode-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', (e) => {
      const btn = e.target.closest('.rf-toggle-btn');
      if (!btn) return;
      this.mode = btn.dataset.mode;
      for (const b of toggle.querySelectorAll('.rf-toggle-btn'))
        b.classList.toggle('is-on', b.dataset.mode === this.mode);
      this.update(this.store.dynamic);   // repaint immediately
    });
  }

  // Effective per-afferent cap (weight_cap_frac * threshold_l2) and firing threshold.
  _caps() {
    const p = this.store.topology?.params || {};
    const thr = p.threshold_l2 || 1;
    const cap = thr * (p.l2e_weight_cap_frac ?? 1);
    return { thr, cap: cap || 1 };
  }

  build() {
    if (this.built) return;
    this.l2Ids = (this.store.topology?.neurons || [])
      .filter(n => n.layer === 'L2' && n.type === 'E')
      .map(n => n.id);
    // Signed-input reference grid.
    this.inputEl.innerHTML = '';
    this.inputCells = [];
    for (let i = 0; i < N_PIX; i++) {
      const c = document.createElement('div');
      c.className = 'rf-cell';
      this.inputEl.appendChild(c);
      this.inputCells.push(c);
    }
    // One card per L2E neuron.
    this.grid.innerHTML = '';
    this.cards = [];
    for (const id of this.l2Ids) {
      const j = Number(id.slice(3));
      const root = document.createElement('div');
      root.className = 'rf-card';
      const title = document.createElement('div');
      title.className = 'rf-title';
      title.innerHTML = `<span>L2E${j}</span><span class="rf-badge" hidden></span>`;
      const cellsEl = document.createElement('div');
      cellsEl.className = 'rf-cells';
      const cells = [];
      for (let i = 0; i < N_PIX; i++) {
        const c = document.createElement('div');
        c.className = 'rf-cell rf-num';
        c.dataset.j = String(j);
        c.dataset.i = String(i);
        this._wireCellEditing(c);
        cellsEl.appendChild(c);
        cells.push(c);
      }
      root.appendChild(title);
      root.appendChild(cellsEl);
      this.grid.appendChild(root);
      this.cards.push({ root, cells, badge: title.querySelector('.rf-badge') });
    }
    this.built = true;
  }

  _wireCellEditing(cell) {
    cell.addEventListener('focus', () => {
      if (!cell.isContentEditable) return;
      cell.classList.add('rf-editing');
      // Select all so typing replaces the shown value.
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
    const val = parseFloat((cell.textContent || '').trim());
    if (!Number.isFinite(val)) { this.update(this.store.dynamic); return; }
    const { cap } = this._caps();
    const weight = this.mode === 'ratio' ? val * cap : val;
    const j = Number(cell.dataset.j), i = Number(cell.dataset.i);
    this.api.post('/api/weight', { j, i, weight });
    // The backend re-broadcasts topology; a temporary local echo avoids a flicker.
    this.store.weights.set(`ff${i}->${j}`, Math.max(0, Math.min(cap, weight)));
  }

  update(dyn) {
    if (!this.built) this.build();
    const s = this.store;
    const { thr, cap } = this._caps();

    // Paused => editable. Reflect on the container + hint.
    this.running = !!(dyn && dyn.running);
    const editable = !this.running;
    this.grid.classList.toggle('rf-can-edit', editable);
    const hint = document.getElementById('rf-edit-hint');
    if (hint) hint.classList.toggle('is-active', editable);

    // Signed input: +1 (active) vs -1 (inactive).
    const input = (dyn && dyn.input) || [];
    for (let i = 0; i < N_PIX; i++) {
      const on = input[i] > 0;
      const c = this.inputCells[i];
      c.classList.toggle('sig-plus', on);
      c.classList.toggle('sig-minus', !on);
      c.textContent = on ? '+' : '−';
    }

    const winner = dyn && dyn.winner;   // e.g. "L2E3"
    for (let cardIndex = 0; cardIndex < this.l2Ids.length; cardIndex++) {
      const id = this.l2Ids[cardIndex];
      const j = Number(id.slice(3));
      const card = this.cards[cardIndex];
      const fireRatios = [];   // w/threshold, for dead detection (independent of view mode)
      for (let i = 0; i < N_PIX; i++) {
        const w = s.weights.get(`ff${i}->${j}`) ?? 0;
        fireRatios.push(w / thr);
        const cell = card.cells[i];
        cell.contentEditable = editable ? 'true' : 'false';
        cell.classList.toggle('rf-editable', editable);
        // Color intensity: fraction of the per-afferent cap.
        const norm = Math.max(0, Math.min(1, w / cap));
        cell.style.background = norm > 0.001
          ? `rgba(94,234,212,${(0.08 + 0.92 * norm).toFixed(3)})` : 'transparent';
        // Don't clobber a cell the user is actively editing.
        if (document.activeElement === cell) continue;
        cell.textContent = this.mode === 'ratio'
          ? (w / cap).toFixed(3)                 // [0,1], precise enough to separate
          : (w >= 0.05 ? w.toFixed(1) : '0');    // actual weight
      }
      // Dead = the three strongest pixels cannot sum to threshold, so the neuron can
      // never accumulate to fire again.
      const top3 = [...fireRatios].sort((a, b) => b - a).slice(0, 3).reduce((a, b) => a + b, 0);
      const dead = top3 < 1.0;
      card.badge.hidden = !dead;
      if (dead) card.badge.textContent = 'dead';
      card.root.classList.toggle('rf-dead', dead);

      const st = s.stateById.get(id);
      card.root.classList.toggle('rf-spike', !!(st && st.spiked));
      card.root.classList.toggle('rf-winner', winner === id);
    }
  }
}
