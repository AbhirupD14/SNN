// Receptive-field view (full-screen pop-up). One card per LEARNING TARGET, generalized to
// any topology and any scale.
//
// A target's card is one of two kinds, decided by the target's own afferents — never by a
// preset name or a fixed 9-pixel assumption:
//
//   'pixels'     every afferent comes from an input-owning source, so the card is a real
//                retinal map. Its shape is the target's own patch (tiled: patch_shape,
//                placed by the source's patch-local row/col) or the whole input sheet
//                (non-tiled: topology.grid). This covers 3x3, 9x9 and 9x18 surfaces alike.
//   'afferents'  the afferents are not pixels (an L2/L3 ordinary E sees child-column Eor
//                outputs, an Eor sees its local ordinary-E bank, a coincidence C owns one
//                learned basal). The card lists source -> weight instead of pretending the
//                inputs form a pixel grid.
//
// Cells show the ACTUAL weight ('weight' mode) or weight / display reference ('ratio').
// The reference is per role, matching the structural ceiling that actually binds: theta/2
// for pattern detectors, theta for the one-afferent Eor bank and C basal, and the
// maturity budget when no ceiling is configured. When PAUSED each mapped cell is editable:
// type a value + Enter to set that exact synapse via /api/weight {synapse: edgeId}.
//
// The model builder below is pure (no DOM) and unit-tested by
// tests/receptive.model.test.mjs.

// Roles whose learned weights this panel shows, and the kinds of edge that carry them.
const LEARNED_EDGE_KINDS = new Set(['feedforward', 'basal_excitation']);
// Cards per row for an 'afferents' list card (purely a layout choice).
const AFFERENT_COLS = 4;

/** True when this node owns an external input pixel (RGC / sensory source). Mirrors the
 *  ownership rule the input grid uses, so a downstream cell's DISPLAY 'pixel' tag can
 *  never be mistaken for a real retinal afferent. */
export function ownsInputPixel(n) {
  return !!(n && n.pixel != null
            && (n.owns_input || n.type === 'S' || n.role === 'source'));
}

/** Per-role display reference: the structural per-synapse ceiling that actually binds, so
 *  a matured weight reads ~1.0 in 'ratio' mode instead of an arbitrary fraction. */
export function displayReference(params = {}, role) {
  const thr = params.threshold_l2 || params.e_threshold || 1;
  const detector = params.e_weight_cap_frac;
  const relay = params.relay_weight_cap_frac;
  if (role === 'Eor' || role === 'C') {
    if (relay != null) return { ref: relay * thr, label: `theta x ${relay}` };
  } else if (detector != null) {
    return { ref: detector * thr, label: `theta x ${detector}` };
  }
  const budget = params.e_maturity_budget ?? thr * (params.e_maturity_budget_frac ?? 1.1);
  return { ref: budget || 1, label: 'maturity budget' };
}

/**
 * Build the complete, DOM-free description of the panel from one topology message.
 * Pure: same input -> same output, no globals touched.
 */
export function buildReceptiveFieldModel(topology) {
  const empty = { inputShape: { rows: 0, cols: 0 }, tiled: false, cards: [], layers: [], groups: [] };
  if (!topology || !Array.isArray(topology.neurons)) return empty;

  const params = topology.params || {};
  const thr = params.threshold_l2 || params.e_threshold || 1;
  const tiling = topology.tiling || null;
  const inputShape = tiling?.input_shape
    || (topology.grid ? { rows: topology.grid.rows, cols: topology.grid.cols } : { rows: 0, cols: 0 });
  const patchShape = tiling?.patch_shape || null;

  const byId = new Map();
  for (const n of topology.neurons) byId.set(n.id, n);

  // Afferents per target, in spec edge order, restricted to learned edge kinds.
  const afferents = new Map();          // targetId -> [{edgeId, sourceId, kind}]
  for (const s of topology.synapses || []) {
    if (!LEARNED_EDGE_KINDS.has(s.kind)) continue;
    if (!byId.has(s.source) || !byId.has(s.target)) continue;
    if (!afferents.has(s.target)) afferents.set(s.target, []);
    afferents.get(s.target).push({ edgeId: s.id, sourceId: s.source, kind: s.kind });
  }

  const cards = [];
  for (const n of topology.neurons) {
    const rows = afferents.get(n.id);
    if (!rows || !rows.length) continue;                 // nothing learned here
    const role = n.column_role || n.role || '';
    const layer = n.layer || '';
    const allPixels = rows.every(r => ownsInputPixel(byId.get(r.sourceId)));
    const { ref, label: refLabel } = displayReference(params, role);

    let kind, shape, cells;
    if (allPixels && inputShape.rows > 0) {
      kind = 'pixels';
      // Tiled input columns place their afferents on their OWN patch; everything else
      // lays out on the whole sheet. Both are derived from metadata, never assumed.
      const usePatch = !!(patchShape && rows.every(r => {
        const src = byId.get(r.sourceId);
        return src.patch_local_row != null && src.patch_local_col != null;
      }));
      shape = usePatch
        ? { rows: patchShape.rows, cols: patchShape.cols }
        : { rows: inputShape.rows, cols: inputShape.cols };
      cells = new Array(shape.rows * shape.cols).fill(null);
      for (const r of rows) {
        const src = byId.get(r.sourceId);
        const rr = usePatch ? src.patch_local_row : Math.floor(src.pixel / inputShape.cols);
        const cc = usePatch ? src.patch_local_col : src.pixel % inputShape.cols;
        if (rr < 0 || rr >= shape.rows || cc < 0 || cc >= shape.cols) continue;
        cells[rr * shape.cols + cc] = {
          edgeId: r.edgeId, sourceId: r.sourceId, pixel: src.pixel, label: null,
        };
      }
    } else {
      kind = 'afferents';
      const cols = Math.min(AFFERENT_COLS, rows.length);
      shape = { rows: Math.ceil(rows.length / cols), cols };
      cells = rows.map(r => ({
        edgeId: r.edgeId, sourceId: r.sourceId, pixel: null,
        // Trim the shared column prefix so a long tiled id stays readable.
        label: r.kind === 'basal_excitation' ? 'basal' : shortSourceLabel(r.sourceId, n),
      }));
      while (cells.length < shape.rows * shape.cols) cells.push(null);
    }

    cards.push({
      id: n.id,
      label: n.label || n.id,
      layer,
      role,
      columnId: n.column_id || null,
      kind,
      shape,
      cells,
      nAfferents: rows.length,
      // Unweighted apical permissions are structural, not learned: reported, never shown
      // as a weight cell.
      threshold: thr,
      reference: ref,
      referenceLabel: refLabel,
    });
  }

  // Stable grouping for display + filtering: by layer, then by column, then by id.
  const layerOrder = new Map();
  for (const L of ['RGC', 'RG', 'L1', 'ERR', 'L2', 'L3']) layerOrder.set(L, layerOrder.size);
  const layers = [...new Set(cards.map(c => c.layer))].sort((a, b) => {
    const ai = layerOrder.has(a) ? layerOrder.get(a) : 99;
    const bi = layerOrder.has(b) ? layerOrder.get(b) : 99;
    return ai - bi || String(a).localeCompare(String(b));
  });
  const roleRank = { E: 0, competitor: 0, encoder: 0, Eor: 1, C: 2 };
  cards.sort((a, b) => {
    const al = layers.indexOf(a.layer), bl = layers.indexOf(b.layer);
    if (al !== bl) return al - bl;
    const ac = String(a.columnId || ''), bc = String(b.columnId || '');
    if (ac !== bc) return ac.localeCompare(bc);
    const ar = roleRank[a.role] ?? 9, br = roleRank[b.role] ?? 9;
    return ar - br || String(a.id).localeCompare(String(b.id));
  });

  const groups = [];
  for (const c of cards) {
    const key = c.columnId || c.layer || 'graph';
    const last = groups[groups.length - 1];
    if (last && last.key === key) last.cards.push(c);
    else groups.push({ key, layer: c.layer, label: c.columnId ? `${c.layer} · ${c.columnId}` : (c.layer || 'graph'), cards: [c] });
  }

  return { inputShape, tiled: !!tiling, patchShape, cards, layers, groups };
}

function shortSourceLabel(sourceId, target) {
  const col = target.column_id;
  if (col && sourceId.startsWith(col)) return sourceId.slice(col.length) || sourceId;
  return sourceId;
}

/** One-volley maturity: flagged when even ALL of this target's learned afferents firing
 *  together cannot reach threshold, i.e. one full delivered packet from rest does not
 *  cross. It is NOT a claim that the cell can never fire -- at leak_rate=0 a sub-threshold
 *  cell still integrates across boundaries -- only that it is not yet a one-event
 *  integrator. This replaces the old fixed "three strongest" proxy, which assumed a
 *  3-pixel stimulus and a 9-afferent detector. Returns null when the card has no cells. */
export function subThresholdFlag(card, weightOf) {
  let total = 0;
  let n = 0;
  for (const cell of card.cells) {
    if (!cell) continue;
    total += weightOf(cell.edgeId) ?? 0;
    n++;
  }
  if (!n) return null;
  return total < card.threshold;
}

export class ReceptiveFields {
  constructor(store, api) {
    this.store = store;
    this.api = api;
    this.overlay = document.getElementById('rf-overlay');
    this.grid = document.getElementById('rf-grid');
    this.inputEl = document.getElementById('rf-input');
    this.filterEl = document.getElementById('rf-filter');
    this.summaryEl = document.getElementById('rf-summary');
    this.cards = [];
    this.inputCells = [];
    this.mode = 'weight';
    this.layerFilter = 'all';
    this.running = true;
    this.model = null;
    this._wireModeToggle();
    this._wireFilter();
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

  _wireFilter() {
    this.filterEl?.addEventListener('change', () => {
      this.layerFilter = this.filterEl.value;
      this.built = false;                  // re-render only the selected layer's cards
      this.build();
      this.update(this.store.dynamic);
    });
  }

  // Rebuild from the CURRENT topology (called on every topology broadcast, so it follows
  // the editor and every preset change).
  build() {
    const topo = this.store.topology;
    if (!topo) return;
    this.model = buildReceptiveFieldModel(topo);
    this._renderFilter();
    this._renderInput();
    this._renderCards();
    this.built = true;
  }

  _renderFilter() {
    if (!this.filterEl) return;
    const layers = this.model.layers;
    const want = layers.includes(this.layerFilter) ? this.layerFilter : 'all';
    this.layerFilter = want;
    this.filterEl.innerHTML = '';
    const mk = (value, text) => {
      const o = document.createElement('option');
      o.value = value; o.textContent = text; o.selected = value === want;
      this.filterEl.appendChild(o);
    };
    mk('all', `all layers (${this.model.cards.length} targets)`);
    for (const L of layers) mk(L, `${L} (${this.model.cards.filter(c => c.layer === L).length})`);
    this.filterEl.hidden = layers.length < 2;
  }

  _renderInput() {
    const { rows, cols } = this.model.inputShape;
    this.inputEl.innerHTML = '';
    this.inputCells = [];
    if (!rows || !cols) return;
    // Drive the CSS grid from the real sheet shape (3x3, 9x9, 9x18, ...).
    this.inputEl.style.setProperty('--rf-cols', String(cols));
    this.inputEl.classList.toggle('rf-input-wide', cols > 6);
    for (let i = 0; i < rows * cols; i++) {
      const c = document.createElement('div');
      c.className = 'rf-cell';
      this.inputEl.appendChild(c);
      this.inputCells.push(c);
    }
  }

  _visibleGroups() {
    if (this.layerFilter === 'all') return this.model.groups;
    return this.model.groups.filter(g => g.layer === this.layerFilter);
  }

  _renderCards() {
    this.grid.innerHTML = '';
    this.cards = [];
    for (const group of this._visibleGroups()) {
      const head = document.createElement('div');
      head.className = 'rf-group';
      head.textContent = group.label;
      this.grid.appendChild(head);
      for (const card of group.cards) this.grid.appendChild(this._renderCard(card));
    }
    if (this.summaryEl) {
      const shown = this._visibleGroups().reduce((a, g) => a + g.cards.length, 0);
      this.summaryEl.textContent =
        `${shown} of ${this.model.cards.length} learning targets`
        + (this.model.tiled ? ` · ${this.model.inputShape.rows}x${this.model.inputShape.cols} sheet` : '');
    }
  }

  _renderCard(card) {
    const root = document.createElement('div');
    root.className = `rf-card rf-card-${card.kind}`;
    const title = document.createElement('div');
    title.className = 'rf-title';
    const roleTag = card.role ? `<span class="rf-role">${card.role}</span>` : '';
    title.innerHTML = `<span>${card.label}</span>${roleTag}<span class="rf-badge" hidden></span>`;
    const cellsEl = document.createElement('div');
    cellsEl.className = 'rf-cells';
    cellsEl.style.setProperty('--rf-cols', String(card.shape.cols));
    const cells = [];
    for (const cell of card.cells) {
      const el = document.createElement('div');
      if (!cell) {
        el.className = 'rf-cell rf-num rf-blank';
        cellsEl.appendChild(el);
        cells.push(null);
        continue;
      }
      el.className = 'rf-cell rf-num';
      el.dataset.edge = cell.edgeId;
      el.title = `${cell.sourceId} -> ${card.id}`;
      if (cell.label) {
        const tag = document.createElement('span');
        tag.className = 'rf-cell-tag';
        tag.textContent = cell.label;
        el.appendChild(tag);
      }
      this._wireCellEditing(el);
      cellsEl.appendChild(el);
      cells.push({ el, ...cell });
    }
    root.appendChild(title);
    root.appendChild(cellsEl);
    this.cards.push({ card, root, cells, badge: title.querySelector('.rf-badge') });
    return root;
  }

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
    const edgeId = cell.dataset.edge;
    if (!edgeId) { this.update(this.store.dynamic); return; }
    const val = parseFloat((cell.textContent || '').trim());
    if (!Number.isFinite(val)) { this.update(this.store.dynamic); return; }
    const ref = Number(cell.dataset.ref) || 1;
    const weight = this.mode === 'ratio' ? val * ref : val;
    this.api.post('/api/weight', { synapse: edgeId, weight });
    this.store.weights.set(edgeId, Math.max(0, weight));   // optimistic echo (floor 0)
  }

  update(dyn) {
    if (!this._open()) return;
    if (!this.built) this.build();
    if (!this.model) return;
    const s = this.store;

    this.running = !!(dyn && dyn.running);
    const editable = !this.running;
    this.grid.classList.toggle('rf-can-edit', editable);
    const hint = document.getElementById('rf-edit-hint');
    if (hint) hint.classList.toggle('is-active', editable);

    const input = (dyn && dyn.input) || [];
    for (let i = 0; i < this.inputCells.length; i++) {
      const on = input[i] > 0;
      const c = this.inputCells[i];
      c.classList.toggle('sig-plus', on);
      c.classList.toggle('sig-minus', !on);
      const txt = on ? '+' : '−';
      if (c.textContent !== txt) c.textContent = txt;
    }

    const winner = dyn && dyn.winner;
    const weightOf = (eid) => s.weights.get(eid);
    for (const entry of this.cards) {
      const { card, cells, badge, root } = entry;
      const ref = card.reference || 1;
      for (const cell of cells) {
        if (!cell) continue;
        const el = cell.el;
        const w = s.weights.get(cell.edgeId) ?? 0;
        el.dataset.ref = String(ref);
        el.contentEditable = editable ? 'true' : 'false';
        el.classList.toggle('rf-editable', editable);
        const norm = Math.max(0, Math.min(1, w / ref));
        const bg = norm > 0.001
          ? `rgba(94,234,212,${(0.08 + 0.92 * norm).toFixed(3)})` : 'transparent';
        if (el.style.background !== bg) el.style.background = bg;
        if (document.activeElement === el) continue;
        const txt = this.mode === 'ratio'
          ? (w / ref).toFixed(3)
          : (w >= 0.05 ? w.toFixed(1) : '0');
        // Preserve the source tag on an 'afferents' cell; only the number changes.
        const tag = cell.label ? el.querySelector('.rf-cell-tag') : null;
        if (tag) {
          if (el.lastChild === tag || el.childNodes.length === 1) el.appendChild(document.createTextNode(txt));
          else if (el.lastChild.nodeValue !== txt) el.lastChild.nodeValue = txt;
        } else if (el.textContent !== txt) {
          el.textContent = txt;
        }
      }
      const sub = subThresholdFlag(card, weightOf);
      badge.hidden = !sub;
      if (sub) badge.textContent = 'sub-θ';
      root.classList.toggle('rf-dead', !!sub);

      const st = s.stateById.get(card.id);
      root.classList.toggle('rf-spike', !!(st && st.spiked));
      root.classList.toggle('rf-winner', winner === card.id);
    }
  }
}
