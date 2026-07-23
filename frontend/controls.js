// Left-sidebar controls: execution buttons, pattern input, manual firing, display
// filters, collapsible sections, and bottom-panel tabs. Every action is an HTTP
// POST to the backend; the resulting state arrives back over the websocket.

export class Controls {
  constructor(store, renderer, api) {
    this.store = store;
    this.renderer = renderer;
    this.api = api;
    this.activePattern = null;
    this._wireCollapse();
    this._wireExecution();
    this._wirePattern();
    this._wireManualFiring();
    this._wireFilters();
    this._wireTabs();
    this._wireResize();
    this._wireConfig();
  }

  // -------------------------------------------------------------- model config
  async _wireConfig() {
    const box = document.getElementById('config-controls');
    if (!box) return;
    let cfg;
    try {
      cfg = await (await fetch('/api/config')).json();
    } catch (e) { console.warn('config fetch failed', e); return; }
    this.configDefaults = { ...cfg.values };   // startup values == defaults
    this.configInputs = {};

    box.innerHTML = '';
    // Build one control element (and register its value getter in configInputs).
    const makeItem = (s) => {
      const item = document.createElement('div');
      item.className = 'config-item';
      const val = cfg.values[s.key];
      if (s.kind === 'toggle') {
        const lab = document.createElement('label');
        lab.className = 'check';
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.checked = !!val;
        lab.appendChild(cb);
        lab.appendChild(document.createTextNode(' ' + s.label));
        item.appendChild(lab);
        this.configInputs[s.key] = () => cb.checked;
      } else if (s.kind === 'select') {
        // Multi-choice string control (e.g. competitive weight-update mode).
        const lab = document.createElement('label');
        lab.className = 'field';
        const span = document.createElement('span');
        span.textContent = s.label;
        const sel = document.createElement('select');
        for (const o of (s.options || [])) {
          const opt = document.createElement('option');
          opt.value = o.value;
          opt.textContent = o.label || o.value;
          if (o.value === val) opt.selected = true;
          sel.appendChild(opt);
        }
        lab.append(span, sel);
        item.appendChild(lab);
        this.configInputs[s.key] = () => sel.value;
      } else {
        const lab = document.createElement('label');
        lab.className = 'field';
        const span = document.createElement('span');
        const b = document.createElement('b');
        const decimals = s.decimals ?? 3;
        b.textContent = (+val).toFixed(decimals);
        span.append(s.label + ' ', b);
        const range = document.createElement('input');
        range.type = 'range';
        range.min = s.min; range.max = s.max; range.step = s.step; range.value = val;
        range.addEventListener('input', () => {
          b.textContent = (+range.value).toFixed(decimals);
        });
        lab.append(span, range);
        item.appendChild(lab);
        this.configInputs[s.key] = () => +range.value;
      }
      const desc = document.createElement('p');
      desc.className = 'config-desc';
      desc.textContent = s.desc;
      item.appendChild(desc);
      return item;
    };

    for (const spec of cfg.spec) box.appendChild(makeItem(spec));

    document.getElementById('config-apply')?.addEventListener('click', () => {
      const overrides = {};
      for (const [k, get] of Object.entries(this.configInputs)) overrides[k] = get();
      this.api.post('/api/config', { overrides });
    });
    document.getElementById('config-reset-defaults')?.addEventListener('click', () => {
      this.api.post('/api/config', { overrides: this.configDefaults });
      // rebuild the panel so the controls snap back to the applied defaults
      setTimeout(() => this._wireConfig(), 200);
    });
  }

  // --------------------------------------------------------- resizable bottom
  _wireResize() {
    const handle = document.getElementById('bottom-resize');
    const footer = document.querySelector('.bottom');
    if (!handle || !footer) return;
    let dragging = false, startY = 0, startH = 0;
    handle.addEventListener('mousedown', (e) => {
      dragging = true; startY = e.clientY; startH = footer.getBoundingClientRect().height;
      document.body.style.userSelect = 'none'; e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const h = Math.max(120, Math.min(window.innerHeight * 0.8, startH + (startY - e.clientY)));
      document.body.style.gridTemplateRows = `56px 1fr ${h}px`;
      window.dispatchEvent(new Event('resize'));    // keep the 3D canvas sized
    });
    window.addEventListener('mouseup', () => {
      if (dragging) { dragging = false; document.body.style.userSelect = ''; }
    });
  }

  // ------------------------------------------------------------- collapsible
  _wireCollapse() {
    document.querySelectorAll('.panel-head').forEach(h =>
      h.addEventListener('click', () => h.parentElement.classList.toggle('collapsed')));
  }

  // ------------------------------------------------------------- execution
  _wireExecution() {
    const bind = (ids, fn) => ids.forEach(id => document.getElementById(id)?.addEventListener('click', fn));
    // Sidebar controls plus the transport mirrored inside the full-screen chart
    // overlays (raster/charge/weights), which cover the top bar's run indicator.
    bind(['g-start', 'x-start', 'x-resume', 'raster-play', 'charge-play', 'weights-play', 'rf-play'],
         () => this.api.post('/api/start'));
    bind(['g-pause', 'x-pause', 'raster-pause', 'charge-pause', 'weights-pause', 'rf-pause'],
         () => this.api.post('/api/pause'));
    bind(['g-step', 'x-step'], () => this.api.post('/api/step'));
    bind(['g-reset', 'x-reset'], () => this.api.post('/api/reset'));
    // Reseed = randomized reset: fresh random initial weights under the same
    // config. Wipes learned state (like Reset), so confirm before firing.
    bind(['g-reseed', 'x-reseed'], () => {
      if (window.confirm('Reseed draws new random initial weights and wipes all learned state. Continue?'))
        this.api.post('/api/reseed');
    });
    // Overlay Stop = reset: halts AND rebuilds the network from fresh weights, so
    // it wipes all learned state -- confirm before firing.
    bind(['raster-stop', 'charge-stop', 'weights-stop'], () => {
      if (window.confirm('Stop resets the simulation and wipes all learned weights. Continue?'))
        this.api.post('/api/reset');
    });

    const speed = document.getElementById('speed'), val = document.getElementById('speed-val');
    speed.addEventListener('input', () => { val.textContent = speed.value; });
    speed.addEventListener('change', () => this.api.post(`/api/speed/${speed.value}`));
  }

  // --------------------------------------------------------------- pattern
  _wirePattern() {
    // Only the input-vector buttons are wired here (grid size is topology-driven and
    // built in onTopology once the topology metadata arrives). Defaults to a 3x3 grid
    // so the panel is populated before the first topology message.
    this.pixels = [];
    this.pixelOwner = new Map();          // global pixel index -> owning input-sink node id
    this.inputShape = { rows: 3, cols: 3 };
    this.patchShape = null;
    document.getElementById('p-random').addEventListener('click', () => { this.activePattern = null; this.api.post('/api/random'); });
    document.getElementById('p-clear').addEventListener('click', () => { this.activePattern = null; this.api.post('/api/clear'); });
    document.getElementById('p-noise').addEventListener('click', () => { this.activePattern = null; this.api.post('/api/noise/0.15'); });
    this._buildInputGrid();
  }

  // Build the input grid from topology metadata: `topology.grid` gives rows/cols (3x3
  // for legacy 9-pixel presets, 9x9 for tiled_cc), and the tiling block adds the 3x3
  // patch boundaries and the local-pattern patch selector. A cell toggles its GLOBAL
  // row-major pixel index; RGC flashing is resolved through pixel-ownership metadata,
  // never an assumed L1E${i} id.
  _buildInputGrid(topology) {
    const grid = document.getElementById('pixel-grid');
    if (!grid) return;
    const tiling = topology?.tiling || null;
    this.inputShape = tiling?.input_shape
      || topology?.grid && { rows: topology.grid.rows, cols: topology.grid.cols }
      || { rows: 3, cols: 3 };
    this.patchShape = tiling?.patch_shape || null;
    this.isTiled = !!tiling;               // tiled -> pattern buttons compose per selected patch
    const { rows, cols } = this.inputShape;
    const pr = this.patchShape?.rows, pc = this.patchShape?.cols;

    // pixel-ownership map (works for any preset: RGC in tiled/rg, L1E_s in pi/old).
    this.pixelOwner = new Map();
    for (const n of topology?.neurons || []) {
      if (n.pixel != null && (n.owns_input || n.type === 'S' || n.role === 'source'))
        this.pixelOwner.set(n.pixel, n.id);
    }

    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    grid.classList.toggle('patched', !!this.patchShape);
    grid.innerHTML = '';
    this.pixels = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const cell = document.createElement('div');
        cell.className = 'pixel';
        cell.dataset.pixel = String(idx);
        if (pr && pc) {                    // strong 3x3 patch boundaries
          if ((c + 1) % pc === 0 && c + 1 < cols) cell.classList.add('patch-r');
          if ((r + 1) % pr === 0 && r + 1 < rows) cell.classList.add('patch-b');
        }
        cell.addEventListener('click', () => { this.activePattern = null; this.api.post(`/api/pixel/${idx}`); });
        grid.appendChild(cell);
        this.pixels.push(cell);
      }
    }
    this._buildPatchSelector(topology);
    this._markSelectedPatch();
  }

  _buildPatchSelector(topology) {
    const box = document.getElementById('patch-select');
    const pg = document.getElementById('patch-grid');
    if (!box || !pg) return;
    const tiling = topology?.tiling;
    const gs = tiling?.grid_shape;
    if (!tiling || !gs) { box.hidden = true; this.patchGrid = null; return; }
    box.hidden = false;
    this.selectedPatch = tiling.selected_patch || null;
    // Current per-patch assignments: "r,c" -> pattern name (drives labels + composition view).
    this.patchPatterns = new Map();
    for (const p of tiling.patch_patterns || [])
      this.patchPatterns.set(`${p.row},${p.col}`, p.name);
    pg.style.gridTemplateColumns = `repeat(${gs.cols}, 1fr)`;
    pg.innerHTML = '';
    this.patchGrid = [];
    for (let r = 0; r < gs.rows; r++) {
      for (let c = 0; c < gs.cols; c++) {
        const b = document.createElement('button');
        b.className = 'patch-btn';
        const assigned = this.patchPatterns.get(`${r},${c}`);
        b.textContent = assigned || `${r},${c}`;
        b.classList.toggle('assigned', !!assigned);
        b.title = assigned
          ? `patch ${r},${c}: "${assigned}" — click to select, right-click to clear`
          : `patch ${r},${c} — click to select, then click a pattern to drive it here`;
        // left-click selects this patch (the pattern buttons then compose into it)
        b.addEventListener('click', () => this.api.post('/api/patch', { row: r, col: c }));
        // right-click clears just this patch (keeps the other patches' patterns)
        b.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          this.api.post('/api/patch_pattern', { row: r, col: c, name: null });
        });
        pg.appendChild(b);
        this.patchGrid.push({ r, c, el: b });
      }
    }
  }

  _markSelectedPatch() {
    if (!this.patchShape || !this.selectedPatch) return;
    const [pr, pc] = this.selectedPatch;
    const { cols } = this.inputShape;
    const { rows: prows, cols: pcols } = this.patchShape;
    const inSel = new Set();
    for (let lr = 0; lr < prows; lr++)
      for (let lc = 0; lc < pcols; lc++)
        inSel.add((pr * prows + lr) * cols + (pc * pcols + lc));
    this.pixels.forEach((cell, i) => cell.classList.toggle('in-patch', inSel.has(i)));
    for (const p of this.patchGrid || [])
      p.el.classList.toggle('active', p.r === pr && p.c === pc);
  }

  buildPatternButtons(patterns) {
    const box = document.getElementById('pattern-buttons');
    box.innerHTML = '';
    this.patBtns = {};
    for (const name of patterns) {
      const b = document.createElement('button');
      b.className = 'pat-btn';
      b.textContent = name;
      b.addEventListener('click', () => {
        this.activePattern = name;
        if (this.isTiled && this.selectedPatch) {
          // Tiled: drive this pattern into the SELECTED 3x3 patch and compose it with the
          // other patches' patterns (different patterns per patch, changed independently).
          const [row, col] = this.selectedPatch;
          this.api.post('/api/patch_pattern', { row, col, name });
        } else {
          this.api.post('/api/pattern', { name });   // whole input (non-tiled presets)
        }
      });
      box.appendChild(b);
      this.patBtns[name] = b;
    }
  }

  // ----------------------------------------------------------- manual firing
  _wireManualFiring() {
    this.mfSelect = document.getElementById('mf-neuron');
    const mag = document.getElementById('mf-mag'), magVal = document.getElementById('mf-mag-val');
    mag.addEventListener('input', () => { magVal.textContent = (+mag.value).toFixed(1); });

    document.getElementById('mf-pulse').addEventListener('click', () =>
      this.api.post('/api/stimulate', { neuron_id: this.mfSelect.value, magnitude: +mag.value, continuous: false }));

    const hold = document.getElementById('mf-hold');
    hold.addEventListener('click', () => {
      const on = hold.dataset.on === '1';
      hold.dataset.on = on ? '0' : '1';
      hold.classList.toggle('active-toggle', !on);
      hold.textContent = on ? 'Continuous' : 'Stop Holding';
      this.api.post('/api/stimulate', { neuron_id: this.mfSelect.value, magnitude: on ? 0 : +mag.value, continuous: true });
    });
  }

  populateNeurons(neurons) {
    this.mfSelect.innerHTML = neurons.map(n =>
      `<option value="${n.id}">${n.id} — ${n.layer} ${n.type}</option>`).join('');
  }

  // ---------------------------------------------------------------- filters
  _wireFilters() {
    const map = { 'f-active': 'active', 'f-weak': 'weak', 'f-assembly': 'assembly',
                  'f-rg': 'rg', 'f-l1': 'l1', 'f-l2': 'l2', 'f-inh': 'inh' };
    for (const [elId, key] of Object.entries(map)) {
      const el = document.getElementById(elId);
      el.addEventListener('change', () => this.renderer.setFilters({ [key]: el.checked }));
    }
  }

  // ------------------------------------------------------------------- tabs
  _wireTabs() {
    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
      // Raster / Charge / Weights are not bottom panels -- they open full-screen
      // overlays (handled in their own modules), so they don't switch the drawer.
      if (['raster', 'charge', 'weights', 'rf'].includes(tab.dataset.tab)) return;
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.querySelector(`.tab-panel[data-panel="${tab.dataset.tab}"]`).classList.add('active');
    }));
  }

  // ------------------------------------------------------------ live updates
  onTopology(topology) {
    this._buildInputGrid(topology);         // topology-sized grid + patch controls
    this.buildPatternButtons(topology.patterns);
    this.populateNeurons(topology.neurons);
    this.selectedPatch = topology.tiling?.selected_patch || null;
    this._markSelectedPatch();
  }

  onDynamic(dyn) {
    // Reflect run state on the overlay transport (the full-screen chart covers the
    // top bar, so this is the only run/pause indicator visible while it is open):
    // highlight Play while running, Pause while halted.
    const running = !!dyn.running;
    for (const id of ['raster-play', 'charge-play', 'weights-play'])
      document.getElementById(id)?.classList.toggle('active-toggle', running);
    for (const id of ['raster-pause', 'charge-pause', 'weights-pause'])
      document.getElementById(id)?.classList.toggle('active-toggle', !running);

    const input = dyn.input || [];
    this.pixels.forEach((cell, i) => {
      const on = input[i] > 0;
      cell.classList.toggle('on', on);
      // Flash on the OWNING input sink's spike (RGC in tiled/rg, L1E_s in pi/old),
      // resolved through pixel-ownership metadata rather than an assumed id scheme.
      const ownerId = this.pixelOwner.get(i);
      const n = ownerId && this.store.stateById.get(ownerId);
      if (n?.spiked) { cell.classList.remove('fire'); void cell.offsetWidth; cell.classList.add('fire'); }
    });
    // detect which named pattern matches the current input (if any)
    if (this.patBtns) {
      let match = this.activePattern;
      const vecs = this.store.patternVectors || {};
      for (const [name, v] of Object.entries(vecs)) {
        if (v.length === input.length && v.every((x, i) => x === input[i])) { match = name; break; }
      }
      for (const [name, b] of Object.entries(this.patBtns)) b.classList.toggle('active', name === match);
    }
  }
}
