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
    bind(['g-start', 'x-start', 'x-resume'], () => this.api.post('/api/start'));
    bind(['g-pause', 'x-pause'], () => this.api.post('/api/pause'));
    bind(['g-step', 'x-step'], () => this.api.post('/api/step'));
    bind(['g-reset', 'x-reset'], () => this.api.post('/api/reset'));

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
  }
}
