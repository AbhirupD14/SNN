// Right-sidebar neuron inspector.  Reads the shared store (static meta +
// current weights + latest dynamic state) and renders detail cards for the
// selected neuron.
//
// Bug fix: render() used to return silently when stateById was empty (race
// between click and first dynamic message, or paused simulation).  Now it
// shows a loading placeholder and retries via requestAnimationFrame so the
// panel always self-populates as soon as data arrives.

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
    // If data wasn't available yet, schedule a single retry once the browser
    // paints the next frame (handles first-load race and paused simulations).
    if (!this.store.stateById.get(id)) {
      requestAnimationFrame(() => { if (this.id === id) this.render(); });
    }
  }

  refresh() { if (this.id) this.render(); }

  render() {
    const s = this.store;
    const meta = s.meta.get(this.id);
    if (!meta) return;   // topology not received yet

    const state = s.stateById.get(this.id);
    if (!state) {
      this.body.innerHTML = '<div style="padding:1rem;color:var(--txt-2);font-size:12px">Waiting for simulation data…</div>';
      return;
    }

    const incoming = [], outgoing = [];
    for (const syn of (s.topology?.synapses ?? [])) {
      const w = s.weights.get(syn.id) ?? syn.weight ?? 0;
      const conf = s.confidence.get(syn.id) ?? syn.confidence ?? null;
      if (syn.target === this.id) incoming.push({ ...syn, w, conf, other: syn.source });
      if (syn.source === this.id) outgoing.push({ ...syn, w, conf, other: syn.target });
    }
    const strongest = [...incoming, ...outgoing].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 4);
    // 'S' is the exogenous RG source: neither excitatory-integrator nor inhibitory.
    const col = meta.type === 'E' ? 'var(--exc)' : meta.type === 'S' ? 'var(--rg)' : 'var(--inh)';
    const typeLabel = meta.type === 'E' ? 'excitatory'
      : meta.type === 'S' ? 'retinal source' : 'inhibitory';
    const chargeBarPct = Math.max(0, Math.min(1, state.activation)) * 100;

    this.body.innerHTML = `
      <div class="insp-head">
        <div class="insp-orb" style="background:${col};color:${col}"></div>
        <div>
          <div class="insp-id">${this.id}</div>
          <div class="insp-tags">
            <span class="tag">${meta.layer}</span>
            <span class="tag">${typeLabel}</span>
            <span class="tag">${meta.role}</span>
            ${state.assembly ? `<span class="tag" style="color:var(--win)">assembly</span>` : ''}
          </div>
        </div>
      </div>
      <div class="insp-cards">
        ${card('Threshold', meta.threshold.toFixed(2))}
        ${state.sensory_weight != null ? card('Sensory weight', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.sensory_weight.toFixed(1)}</span>
            <span style="color:var(--txt-2);font-size:11px">learned S→L1E (grows with training)</span>
          </div>`) : ''}
        ${state.budget != null ? card('Budget usage', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.budget_used.toFixed(3)} / ${state.budget.toFixed(2)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${Math.max(0, Math.min(1, state.budget_used / state.budget)) * 100}%;background:var(--ff);border-radius:3px;transition:width .1s"></div>
            </div>
          </div>`) : ''}
        ${card('Charge', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.potential.toFixed(3)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${chargeBarPct}%;background:${col};border-radius:3px;transition:width .1s"></div>
            </div>
            <span style="color:var(--txt-2);font-size:11px">${(state.activation * 100).toFixed(0)}%</span>
          </div>`)}
        ${card('Spike', `<span class="firing-badge ${state.spiked ? 'yes' : 'no'}">${state.spiked ? 'SPIKE' : 'idle'}</span>`, '', true)}
        ${card('Firing freq', (state.freq * 100).toFixed(1) + '%', bar(state.freq))}
        ${card('Refractory', state.refractory + ' steps')}
        ${state.basal_weight != null ? card('Learned basal weight', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.basal_weight.toFixed(1)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${Math.max(0, Math.min(1, state.basal_weight / meta.threshold)) * 100}%;background:#c084fc;border-radius:3px"></div>
            </div>
            <span style="font-size:11px;color:var(--txt-2)">the only plastic C weight</span>
          </div>`) : ''}
        ${state.coincidence_active != null ? card('Coincidence gate', `
          <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
            <span class="tag" style="color:${state.basal_received || state.basal_eligible ? '#c084fc' : 'var(--txt-2)'}">
              basal ${state.basal_received ? 'now' : (state.basal_eligible ? 'eligible' : '—')}</span>
            <span class="tag" style="color:${state.apical_active ? '#f472b6' : 'var(--txt-2)'}">
              apical ${state.apical_active ? `on (${(state.apical_sources || []).length})` : 'off'}</span>
            <span class="firing-badge ${state.coincidence_active ? 'yes' : 'no'}">
              ${state.coincidence_active ? 'COINCIDENCE' : 'no gate'}</span>
            <span style="font-size:11px;color:var(--txt-2)">charge ${(state.coincidence_charge ?? 0).toFixed(1)}</span>
          </div>`) : ''}
        ${state.spike_tau != null ? card('Spike sub-boundary τ',
          `<span style="font-variant-numeric:tabular-nums">${state.spike_tau.toFixed(4)}</span>
           <span style="font-size:11px;color:var(--txt-2);margin-left:8px">analytic within-boundary crossing time</span>`) : ''}
        ${state.winner_trace != null ? card('Local winner trace x_j', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.winner_trace.toFixed(3)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${Math.max(0, Math.min(1, state.winner_trace)) * 100}%;background:#f59e0b;border-radius:3px"></div>
            </div>
            <span style="font-size:11px;color:var(--txt-2)">${state.residual_received ? 'residual now' : 'no residual'}</span>
          </div>`) : ''}
        ${state.residual_charge != null ? card('Residual branch charge', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.residual_charge.toFixed(2)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${Math.max(0, Math.min(1, state.residual_charge / meta.threshold)) * 100}%;background:#22c55e;border-radius:3px"></div>
            </div>
            <span style="font-size:11px;color:var(--txt-2)">${state.residual_events ?? 0} ErrorE event(s)</span>
          </div>`) : ''}
        ${state.trace_charge != null ? card('Winner-priming charge', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.trace_charge.toFixed(2)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${Math.max(0, Math.min(1, state.trace_charge / meta.threshold)) * 100}%;background:#f59e0b;border-radius:3px"></div>
            </div>
          </div>`) : ''}
        ${synCard('Strongest connections', strongest, this.id)}
        ${synCard(`Incoming (${incoming.length})`, incoming, this.id)}
        ${synCard(`Outgoing (${outgoing.length})`, outgoing, this.id)}
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
  // Render every synapse in the list. Callers that want a summary (e.g. the
  // "Strongest connections" card) pre-slice their list; the Incoming/Outgoing
  // cards pass the full set and their titles show the true count, so the rows
  // shown must match that count rather than being capped here.
  const rows = list.map(sy => {
    // Structural E->I relay-excitation edge: a +1 event with no learned weight.
    if (sy.kind === 'relay_excitation') {
      return `<div class="syn-row">
        <span class="name">${sy.other}</span>
        <span class="wbar"></span>
        <span class="wv" style="color:var(--ex)" title="structural +1 relay event (no learned weight)">relay</span></div>`;
    }
    if (sy.kind === 'fixed_excitation') {
      return `<div class="syn-row">
        <span class="name">${sy.other}</span><span class="wbar"></span>
        <span class="wv" style="color:#22c55e" title="fixed evidence-copy charge">fixed +</span></div>`;
    }
    if (sy.kind === 'trace_excitation') {
      return `<div class="syn-row">
        <span class="name">${sy.other}</span><span class="wbar"></span>
        <span class="wv" style="color:#f59e0b" title="paired local winner eligibility event">trace x_j</span></div>`;
    }
    // Paired local sensory afferent L1E_s->L1E_new (coincidence input): a learned
    // excitatory weight; label it so it reads distinctly from dense L2 feedback.
    if (sy.kind === 'coincidence_local') {
      return `<div class="syn-row">
        <span class="name">${sy.other}</span>
        <span class="wbar"><i style="left:50%;width:${(Math.min(1, Math.abs(sy.w) / 500) * 50).toFixed(0)}%;background:#9be15d"></i></span>
        <span class="wv" title="paired local sensory afferent (coincidence input)">local ${sy.w.toFixed(0)}</span></div>`;
    }
    // L2I_WTA / legacy L1I inhibition (I->E): a persistent inhibitory CONDUCTANCE
    // pulse (no learned per-synapse magnitude). Render it as inhibitory.
    if (sy.kind === 'inhibition') {
      return `<div class="syn-row">
        <span class="name">${sy.other}</span>
        <span class="wbar"></span>
        <span class="wv" style="color:var(--in)" title="inhibitory conductance pulse (g_inh), decays over time; not a hard wipe">g-pulse</span></div>`;
    }
    // Predictive inhibitory output PI[j] -> L1E_s[i]: a locally-plastic weight that
    // sets the emitted inhibitory conductance (g_scale * w). Bounded to [0, w_max].
    if (sy.kind === 'predictive_inhibition') {
      const frac = Math.min(1, Math.abs(sy.w));
      return `<div class="syn-row">
        <span class="name">${sy.other}</span>
        <span class="wbar"><i style="right:50%;width:${(frac * 50).toFixed(0)}%;background:#e066c0"></i></span>
        <span class="wv" style="color:#e066c0" title="locally-learned predictive inhibitory weight; emits g_scale*w conductance">PI ${sy.w.toFixed(3)}</span></div>`;
    }
    const mag = Math.min(1, Math.abs(sy.w));
    const pos = sy.w >= 0;
    const width = (mag * 50).toFixed(0);
    const color = pos ? 'var(--ff)' : 'var(--in)';
    const style = pos ? `left:50%;width:${width}%;background:${color}` : `right:50%;width:${width}%;background:${color}`;
    // Confidence (trust in the gate) shown alongside the weight (gate size) when
    // the synapse carries one -- these are separate quantities in confidence mode.
    const conf = (sy.conf != null)
      ? `<span class="wv" title="confidence" style="color:var(--txt-2)">c ${sy.conf.toFixed(2)}</span>` : '';
    return `<div class="syn-row">
      <span class="name">${sy.other}</span>
      <span class="wbar"><i style="${style}"></i></span>
      <span class="wv">${sy.w >= 0 ? '+' : ''}${sy.w.toFixed(3)}</span>${conf}</div>`;
  }).join('');
  return `<div class="icard full"><div class="lbl">${title}</div><div class="syn-list">${rows}</div></div>`;
}
