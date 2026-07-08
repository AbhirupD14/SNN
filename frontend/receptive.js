// Receptive-field view for the minimal SIGNED-SPIKE experiment.
//
// Renders, live, one 3x3 feedforward receptive field per L2E neuron (weight /
// weight_cap, so cells are in [0,1]), plus the current signed input (+1 active,
// -1 inactive). This makes the signed rule visible: on each L2E fire its active
// pixels potentiate and its inactive pixels depress, with no weight budget. A
// unit is flagged "dead" once its three strongest pixels can no longer sum to
// threshold (it can never fire again), which is the characteristic failure mode
// of the minimal loop (lateral inhibition starves losers; nothing recruits them).

const N_OUT = 8;
const N_PIX = 9;   // 3x3 grid

export class ReceptiveFields {
  constructor(store) {
    this.store = store;
    this.grid = document.getElementById('rf-grid');
    this.inputEl = document.getElementById('rf-input');
    this.cards = [];        // per-L2E { root, cells[9], badge }
    this.inputCells = [];
    this.built = false;
  }

  build() {
    if (this.built) return;
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
    for (let j = 0; j < N_OUT; j++) {
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

  update(dyn) {
    if (!this.built) this.build();
    const s = this.store;
    const cap = (s.topology?.params?.threshold_l2) || 1;

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
    for (let j = 0; j < N_OUT; j++) {
      const card = this.cards[j];
      const vals = [];
      for (let i = 0; i < N_PIX; i++) {
        const w = s.weights.get(`ff${i}->${j}`) ?? 0;
        const v = Math.max(0, Math.min(1, w / cap));   // normalized to the cap
        vals.push(v);
        const cell = card.cells[i];
        cell.style.background = v > 0.01
          ? `rgba(94,234,212,${(0.08 + 0.92 * v).toFixed(3)})` : 'transparent';
        // Overlay the (normalized, weight/cap) value on top of the color.
        cell.textContent = v >= 0.005 ? v.toFixed(2) : '';
      }
      // Dead = the three strongest pixels cannot sum to threshold (cap), so the
      // neuron can never accumulate to fire again.
      const top3 = [...vals].sort((a, b) => b - a).slice(0, 3).reduce((a, b) => a + b, 0);
      const dead = top3 < 1.0;
      card.badge.hidden = !dead;
      if (dead) card.badge.textContent = 'dead';
      card.root.classList.toggle('rf-dead', dead);

      const st = s.stateById.get(`L2E${j}`);
      card.root.classList.toggle('rf-spike', !!(st && st.spiked));
      card.root.classList.toggle('rf-winner', winner === `L2E${j}`);
    }
  }
}
