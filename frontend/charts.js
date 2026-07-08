// Bottom-panel visualizations kept for debugging the minimal experiment:
// the Layer-2 heat map (who is firing / who owns the current pattern) and the
// event log. Higher-value time-series views (spike raster, charge-over-time,
// weights-over-time) live in their own full-screen overlays. The old activation
// histogram, "currently firing" chips, generic statistics cards, and rolling
// line charts were removed -- they did not help debug ownership/RF formation.

export class Charts {
  constructor(store) {
    this.store = store;
    this.logSeen = new Set();
    this.heatmap = document.getElementById('heatmap');
    this.hmWinner = document.getElementById('hm-winner');
    this.eventLog = document.getElementById('event-log');
  }

  buildStatic(topology) {
    this.l2 = topology.neurons.filter(n => n.layer === 'L2' && n.type === 'E').map(n => n.id);
    if (this.heatmap) {
      this.heatmap.innerHTML = this.l2.map(id =>
        `<div class="hm-cell" data-id="${id}"><b>0</b><span>${id}</span></div>`).join('');
    }
  }

  update(dyn, fps) {
    // --- Layer-2 heat map (firing rate per L2E, winner highlighted) ---
    if (this.heatmap) {
      const byId = new Map(dyn.neurons.map(n => [n.id, n]));
      for (const cell of this.heatmap.children) {
        const n = byId.get(cell.dataset.id);
        const f = n ? n.freq : 0;
        cell.querySelector('b').textContent = (f * 100).toFixed(0);
        cell.style.background = heat(f);
        cell.classList.toggle('win', dyn.winner === cell.dataset.id);
      }
    }
    if (this.hmWinner) this.hmWinner.textContent = dyn.winner || '—';

    // --- event log ---
    if (this.eventLog) {
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
}

function heat(f) {
  if (f <= 0) return 'var(--bg-3)';
  return `hsl(${175 - f * 40}, 75%, ${20 + f * 40}%)`;
}
