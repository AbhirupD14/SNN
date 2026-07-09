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
    for (const s of cfg.spec) {
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
      } else {
        const lab = document.createElement('label');
        lab.className = 'field';
        const span = document.createElement('span');
        const b = document.createElement('b');
        b.textContent = (+val).toFixed(3);
        span.append(s.label + ' ', b);
        const range = document.createElement('input');
        range.type = 'range';
        range.min = s.min; range.max = s.max; range.step = s.step; range.value = val;
        range.addEventListener('input', () => { b.textContent = (+range.value).toFixed(3); });
        lab.append(span, range);
        item.appendChild(lab);
        this.configInputs[s.key] = () => +range.value;
      }
      const desc = document.createElement('p');
      desc.className = 'config-desc';
      desc.textContent = s.desc;
      item.appendChild(desc);
      box.appendChild(item);
    }

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
    bind(['g-start', 'x-start', 'x-resume', 'raster-play', 'charge-play', 'weights-play'],
         () => this.api.post('/api/start'));
    bind(['g-pause', 'x-pause', 'raster-pause', 'charge-pause', 'weights-pause'],
         () => this.api.post('/api/pause'));
    bind(['g-step', 'x-step'], () => this.api.post('/api/step'));
    bind(['g-reset', 'x-reset'], () => this.api.post('/api/reset'));
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
    const grid = document.getElementById('pixel-grid');
    grid.innerHTML = '';
    this.pixels = [];
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement('div');
      cell.className = 'pixel';
      cell.addEventListener('click', () => { this.activePattern = null; this.api.post(`/api/pixel/${i}`); });
      grid.appendChild(cell);
      this.pixels.push(cell);
    }
    document.getElementById('p-random').addEventListener('click', () => { this.activePattern = null; this.api.post('/api/random'); });
    document.getElementById('p-clear').addEventListener('click', () => { this.activePattern = null; this.api.post('/api/clear'); });
    document.getElementById('p-noise').addEventListener('click', () => { this.activePattern = null; this.api.post('/api/noise/0.15'); });
    this._wireAutoCycle();
  }

  _wireAutoCycle() {
    const enable = document.getElementById('ac-enable');
    const streak = document.getElementById('ac-streak');
    const streakVal = document.getElementById('ac-streak-val');
    if (!enable || !streak) return;
    streak.addEventListener('input', () => { streakVal.textContent = streak.value; });
    const send = () => this.api.post('/api/autocycle',
      { enabled: enable.checked, streak: +streak.value });
    enable.addEventListener('change', send);
    streak.addEventListener('change', () => { if (enable.checked) send(); });
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
        this.api.post('/api/pattern', { name });   // name in body: handles '/' and '\'
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
                  'f-l1': 'l1', 'f-l2': 'l2', 'f-inh': 'inh' };
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
      if (['raster', 'charge', 'weights'].includes(tab.dataset.tab)) return;
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.querySelector(`.tab-panel[data-panel="${tab.dataset.tab}"]`).classList.add('active');
    }));
  }

  // ------------------------------------------------------------ live updates
  onTopology(topology) {
    this.buildPatternButtons(topology.patterns);
    this.populateNeurons(topology.neurons);
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
      const n = this.store.stateById.get(`L1E${i}`);
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
    this._updateAutoCycleStatus(dyn.autocycle);
  }

  _updateAutoCycleStatus(ac) {
    const el = document.getElementById('ac-status');
    const enable = document.getElementById('ac-enable');
    if (!el || !ac) return;
    // keep the checkbox in sync (e.g. after the cycle auto-disables on completion)
    if (enable && enable.checked !== ac.enabled) enable.checked = ac.enabled;
    if (!ac.enabled) {
      el.textContent = ac.trained >= ac.total && ac.total > 0
        ? `complete — ${ac.trained}/${ac.total} patterns trained`
        : 'idle';
      el.classList.toggle('done', ac.trained >= ac.total && ac.total > 0);
      return;
    }
    el.classList.remove('done');
    const w = ac.last_winner == null ? '—' : `L2E${ac.last_winner}`;
    el.textContent =
      `${ac.trained}/${ac.total} trained · "${ac.pattern}" ${ac.streak}/${ac.target} stable (→ ${w})`;
  }
}
