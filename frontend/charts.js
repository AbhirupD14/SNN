// Bottom-panel visualizations: Layer-2 heat map, activation histogram, firing
// list, live statistics, rolling line charts, and the event log. DOM scaffolding
// is created once; per-frame updates only mutate existing nodes / redraw canvases.

const HISTORY = 160;   // samples kept per line chart

export class Charts {
  constructor(store) {
    this.store = store;
    this.series = { rate: [], active: [], act: [], weight: [] };
    this.logSeen = new Set();

    this.heatmap = document.getElementById('heatmap');
    this.hmWinner = document.getElementById('hm-winner');
    this.firingList = document.getElementById('firing-list');
    this.statsGrid = document.getElementById('stats-grid');
    this.eventLog = document.getElementById('event-log');
    this.hist = document.getElementById('histogram');
    this.charts = {
      rate: document.getElementById('chart-rate'),
      active: document.getElementById('chart-active'),
      act: document.getElementById('chart-act'),
      weight: document.getElementById('chart-weight'),
    };
    this._statBoxes = null;
  }

  buildStatic(topology) {
    // heat map cells for the 8 L2 excitatory neurons
    this.l2 = topology.neurons.filter(n => n.layer === 'L2' && n.type === 'E').map(n => n.id);
    this.heatmap.innerHTML = this.l2.map(id =>
      `<div class="hm-cell" data-id="${id}"><b>0</b><span>${id}</span></div>`).join('');

    const defs = [
      ['total', 'Total neurons'], ['active', 'Active'], ['firing', 'Firing now'],
      ['avg_activation', 'Avg activation'], ['firing_rate', 'Firing rate'],
      ['avg_weight', 'Avg weight'], ['winner', 'Winner'], ['fps', 'Sim FPS'],
    ];
    this.statsGrid.innerHTML = defs.map(([k, l]) =>
      `<div class="sbox"><div class="lbl">${l}</div><div class="num ${['firing_rate','avg_weight'].includes(k)?'accent':''}" data-stat="${k}">—</div></div>`).join('');
    this._statBoxes = {};
    this.statsGrid.querySelectorAll('[data-stat]').forEach(el => this._statBoxes[el.dataset.stat] = el);
  }

  update(dyn, fps) {
    const st = dyn.stats;
    // --- rolling series ---
    push(this.series.rate, st.firing_rate);
    push(this.series.active, st.active);
    push(this.series.act, st.avg_activation);
    push(this.series.weight, st.avg_weight);
    drawLine(this.charts.rate, this.series.rate, '#5eead4');
    drawLine(this.charts.active, this.series.active, '#7c9cff');
    drawLine(this.charts.act, this.series.act, '#4cc38a');
    drawLine(this.charts.weight, this.series.weight, '#ffce5c');

    // --- heat map ---
    const byId = new Map(dyn.neurons.map(n => [n.id, n]));
    for (const cell of this.heatmap.children) {
      const n = byId.get(cell.dataset.id);
      const f = n ? n.freq : 0;
      cell.querySelector('b').textContent = (f * 100).toFixed(0);
      cell.style.background = heat(f);
      cell.classList.toggle('win', dyn.winner === cell.dataset.id);
    }
    this.hmWinner.textContent = dyn.winner || '—';

    // --- histogram of activations ---
    drawHistogram(this.hist, dyn.neurons.map(n => n.activation));

    // --- currently firing chips ---
    const firing = dyn.neurons.filter(n => n.spiked);
    this.firingList.innerHTML = firing.length
      ? firing.map(n => `<span class="chip ${this.store.meta.get(n.id)?.type || 'E'}">${n.id}</span>`).join('')
      : `<span style="color:var(--txt-2);font-size:11px">— none this step —</span>`;

    // --- stats ---
    if (this._statBoxes) {
      this._statBoxes.total.textContent = st.total;
      this._statBoxes.active.textContent = st.active;
      this._statBoxes.firing.textContent = st.firing;
      this._statBoxes.avg_activation.textContent = st.avg_activation.toFixed(3);
      this._statBoxes.firing_rate.textContent = (st.firing_rate * 100).toFixed(0) + '%';
      this._statBoxes.avg_weight.textContent = st.avg_weight.toFixed(3);
      this._statBoxes.winner.textContent = st.winner || '—';
      this._statBoxes.fps.textContent = fps.toFixed(0);
    }

    // --- event log ---
    for (const e of dyn.log || []) {
      if (this.logSeen.has(e.seq)) continue;
      this.logSeen.add(e.seq);
      const div = document.createElement('div');
      div.className = `log-line ${e.kind}`;
      div.innerHTML = `<span class="t">t=${e.t}</span><span class="kind">${e.kind}</span><span class="msg">${e.message}</span>`;
      this.eventLog.appendChild(div);
    }
    while (this.eventLog.childElementCount > 200) this.eventLog.removeChild(this.eventLog.firstChild);
    this.eventLog.scrollTop = this.eventLog.scrollHeight;
  }
}

function push(arr, v) { arr.push(v); if (arr.length > HISTORY) arr.shift(); }

function fit(canvas) {
  const w = canvas.clientWidth || 300, h = canvas.height;
  if (canvas.width !== w) canvas.width = w;
  return canvas.getContext('2d');
}
function drawLine(canvas, data, color) {
  const ctx = fit(canvas), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  const max = Math.max(1e-6, ...data), min = Math.min(0, ...data);
  const x = i => (i / (HISTORY - 1)) * w;
  const y = v => h - 6 - ((v - min) / (max - min || 1)) * (h - 12);
  // area fill
  ctx.beginPath(); ctx.moveTo(x(0), h);
  data.forEach((v, i) => ctx.lineTo(x(i), y(v)));
  ctx.lineTo(x(data.length - 1), h); ctx.closePath();
  ctx.fillStyle = color + '18'; ctx.fill();
  // line
  ctx.beginPath();
  data.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
  // last value marker
  const lx = x(data.length - 1), ly = y(data[data.length - 1]);
  ctx.beginPath(); ctx.arc(lx, ly, 2.5, 0, 7); ctx.fillStyle = color; ctx.fill();
}
function drawHistogram(canvas, values) {
  const ctx = fit(canvas), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const bins = 10, counts = new Array(bins).fill(0);
  for (const v of values) {
    const b = Math.max(0, Math.min(bins - 1, Math.floor(Math.max(0, Math.min(1, v)) * bins)));
    counts[b]++;
  }
  const max = Math.max(1, ...counts), bw = w / bins;
  counts.forEach((c, i) => {
    const bh = (c / max) * (h - 10);
    ctx.fillStyle = `hsl(${170 - i * 10}, 70%, ${45 + (c / max) * 15}%)`;
    ctx.fillRect(i * bw + 2, h - bh, bw - 4, bh);
  });
}
function heat(f) {
  if (f <= 0) return 'var(--bg-3)';
  const l = 20 + f * 40;
  return `hsl(${175 - f * 40}, 75%, ${l}%)`;
}
