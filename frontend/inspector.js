// Right-sidebar neuron inspector. Reads the shared store (static meta + current
// weights + latest dynamic state) and renders detail cards for the selected neuron.

export class Inspector {
  constructor(store) {
    this.store = store;
    this.id = null;
    this.empty = document.getElementById('insp-empty');
    this.body = document.getElementById('insp-body');
  }

  select(id) {
    this.id = id;
    this.empty.hidden = true;
    this.body.hidden = false;
    this.render();
  }

  refresh() { if (this.id) this.render(); }

  render() {
    const s = this.store;
    const meta = s.meta.get(this.id);
    const state = s.stateById.get(this.id);
    if (!meta || !state) return;

    const incoming = [], outgoing = [];
    for (const syn of s.topology.synapses) {
      const w = s.weights.get(syn.id) ?? syn.weight ?? 0;
      if (syn.target === this.id) incoming.push({ ...syn, w, other: syn.source });
      if (syn.source === this.id) outgoing.push({ ...syn, w, other: syn.target });
    }
    const strongest = [...incoming, ...outgoing].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 4);
    const col = meta.type === 'E' ? 'var(--exc)' : 'var(--inh)';

    this.body.innerHTML = `
      <div class="insp-head">
        <div class="insp-orb" style="background:${col};color:${col}"></div>
        <div>
          <div class="insp-id">${this.id}</div>
          <div class="insp-tags">
            <span class="tag">${meta.layer}</span>
            <span class="tag ${meta.type}">${meta.type === 'E' ? 'excitatory' : 'inhibitory'}</span>
            ${state.assembly ? `<span class="tag" style="color:var(--win)">assembly</span>` : ''}
          </div>
        </div>
      </div>
      <div class="insp-cards">
        ${card('Activation', state.activation.toFixed(3), bar(state.activation))}
        ${card('Membrane V', state.potential.toFixed(3))}
        ${card('Threshold', meta.threshold.toFixed(2))}
        ${card('Firing', `<span class="firing-badge ${state.spiked ? 'yes' : 'no'}">${state.spiked ? 'SPIKE' : 'idle'}</span>`, '', true)}
        ${card('Firing freq', (state.freq * 100).toFixed(0) + '%', bar(state.freq))}
        ${card('Refractory', state.refractory)}
        ${synCard('Strongest connections', strongest, this.id)}
        ${synCard(`Incoming synapses (${incoming.length})`, incoming, this.id)}
        ${synCard(`Outgoing synapses (${outgoing.length})`, outgoing, this.id)}
      </div>`;
  }
}

function card(lbl, val, extra = '', small = false) {
  return `<div class="icard">
    <div class="lbl">${lbl}</div>
    <div class="val ${small ? 'sm' : ''}">${val}</div>${extra}</div>`;
}
function bar(x) {
  const pct = Math.max(0, Math.min(1, x)) * 100;
  return `<div class="bar"><i style="width:${pct}%"></i></div>`;
}
function synCard(title, list, self) {
  if (!list.length) return `<div class="icard full"><div class="lbl">${title}</div><div class="val sm" style="color:var(--txt-2)">none</div></div>`;
  const rows = list.slice(0, 8).map(sy => {
    const mag = Math.min(1, Math.abs(sy.w));
    const pos = sy.w >= 0;
    const width = (mag * 50).toFixed(0);
    const color = pos ? 'var(--ff)' : 'var(--in)';
    const style = pos ? `left:50%;width:${width}%;background:${color}` : `right:50%;width:${width}%;background:${color}`;
    return `<div class="syn-row">
      <span class="name">${sy.other}</span>
      <span class="wbar"><i style="${style}"></i></span>
      <span class="wv">${sy.w >= 0 ? '+' : ''}${sy.w.toFixed(3)}</span></div>`;
  }).join('');
  return `<div class="icard full"><div class="lbl">${title}</div><div class="syn-list">${rows}</div></div>`;
}
