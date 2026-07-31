// Right-sidebar neuron inspector.  Reads the shared store (static meta +
// current weights + latest dynamic state) and renders detail cards for the
// selected neuron.
//
// Bug fix: render() used to return silently when stateById was empty (race
// between click and first dynamic message, or paused simulation).  Now it
// shows a loading placeholder and retries via requestAnimationFrame so the
// panel always self-populates as soon as data arrives.
//
// Layout stability: the panel is rebuilt every frame, so which cards exist must depend
// only on the SELECTED NEURON, never on this boundary's values.  A card gated on a value
// that turns on and off each boundary (spike_tau) is added/removed from the two-column grid
// and reflows every card below it into the other column.  Gate such cards on the presence of
// the KEY and render a placeholder for the idle state instead.

export class Inspector {
  constructor(store) {
    this.store = store;
    this.id = null;
    this.empty = document.getElementById('insp-empty');
    this.body = document.getElementById('insp-body');
    this.scroller = document.getElementById('inspector');   // the overflow-y:auto aside
    this._raf = 0;
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

  // Coalesce per-frame refreshes to at most one render per animation frame. The live
  // path calls this on every websocket message, which can arrive faster than the
  // browser paints; rebuilding innerHTML on each one flickers the panel and thrashes
  // its scroll position. One rAF-batched render per paint removes the flicker.
  refresh() {
    if (!this.id || this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); });
  }

  // Clear the current selection and show the empty placeholder. Called when a new
  // topology is applied (live editor apply, or entering/leaving replay) so the
  // panel never references a neuron id that no longer exists.
  reset() {
    this.id = null;
    if (this.empty) this.empty.hidden = false;
    if (this.body) { this.body.hidden = true; this.body.innerHTML = ''; }
  }

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
    // Cap-free ordinary-E display reference: the neuron-wide maturity budget
    // B = e_maturity_budget_frac * theta (no hard per-synapse cap).
    const wref = (s.topology?.params?.e_maturity_budget)
      ?? ((meta.threshold || 1000) * (s.topology?.params?.e_maturity_budget_frac ?? 1.1));
    // 'S' is the exogenous RG source: neither excitatory-integrator nor inhibitory.
    const col = meta.type === 'E' ? 'var(--exc)' : meta.type === 'S' ? 'var(--rg)' : 'var(--inh)';
    const typeLabel = meta.type === 'E' ? 'excitatory'
      : meta.type === 'S' ? 'retinal source' : 'inhibitory';
    // A value the source never sampled arrives as null. It is UNKNOWN, not zero, and must
    // never be rendered as a number, a 0% bar, or a stale carry-over. `avail()` reports the
    // declared provenance of a field so the card can say so plainly.
    const availability = (s.dynamic && s.dynamic.nest && s.dynamic.nest.state_availability) || null;
    const known = (v) => v != null && Number.isFinite(v);
    // Variable names come from a loaded artifact, so they reach innerHTML escaped -- the
    // rest of this panel interpolates ids the topology validated, these are free text.
    const esc = (s) => String(s).replace(/[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const unknownCard = (why) =>
      `<span style="color:var(--txt-2);font-size:11px">Not recorded for this NEST run</span>` +
      (why ? `<span class="tag" style="margin-left:6px;color:var(--txt-2)">${why}</span>` : '');
    const sourceBadge = (field) => {
      if (!availability) return '';
      const kind = availability[field];
      if (!kind) return '';
      const colour = kind === 'recorded' ? 'var(--exc)'
        : kind === 'derived' ? 'var(--ff)' : 'var(--txt-2)';
      return `<span class="tag" style="color:${colour}" title="provenance of this value">${kind}</span>`;
    };
    const chargeBarPct = known(state.activation)
      ? Math.max(0, Math.min(1, state.activation)) * 100 : 0;

    // Preserve scroll position across the full innerHTML rebuild so a scrolled-down
    // inspector doesn't snap to the top (and visibly flicker) on every frame.
    const scrollTop = this.scroller ? this.scroller.scrollTop : 0;
    this.body.innerHTML = `
      <div class="insp-head">
        <div class="insp-orb" style="background:${col};color:${col}"></div>
        <div>
          <div class="insp-id">${this.id}</div>
          <div class="insp-tags">
            <span class="tag">${meta.layer}</span>
            <span class="tag">${typeLabel}</span>
            <span class="tag">${meta.role}</span>
            ${meta.column_id ? `<span class="tag" title="cortical column">${meta.column_id}</span>` : ''}
            ${meta.column_role ? `<span class="tag" title="role within its column">${meta.column_role}${meta.column_index != null ? ' ' + meta.column_index : ''}</span>` : ''}
            ${meta.column_row != null ? `<span class="tag" title="tile position (row,col)">tile ${meta.column_row},${meta.column_col}</span>` : ''}
            ${meta.column_role === 'C' && meta.has_parent === false ? `<span class="tag" style="color:var(--txt-2)" title="top column has no parent: this C receives no apical permission and stays dormant">dormant top C</span>` : ''}
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
        ${card('Charge', known(state.potential) ? `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.potential.toFixed(3)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${chargeBarPct}%;background:${col};border-radius:3px;transition:width .1s"></div>
            </div>
            <span style="color:var(--txt-2);font-size:11px">${known(state.activation) ? (state.activation * 100).toFixed(0) + '%' : ''}</span>
          </div>` : unknownCard(sourceBadge('potential')))}
        ${card('Spike', `<span class="firing-badge ${state.spiked ? 'yes' : 'no'}">${state.spiked ? 'SPIKE' : 'idle'}</span> ${sourceBadge('spiked')}`, '', true)}
        ${card('Firing freq', known(state.freq) ? (state.freq * 100).toFixed(1) + '%' : unknownCard(sourceBadge('freq')), known(state.freq) ? bar(state.freq) : '')}
        ${card('Refractory', state.refractory != null ? state.refractory + ' steps' : unknownCard(sourceBadge('refractory')))}
        ${state.basal_weights ? card(`Learned basal weights (${state.basal_weights.length})`, `
          <div class="basal-list">
            ${state.basal_weights.map((w, i) => {
              const src = (state.basal_sources || [])[i] ?? `#${i}`;
              const causal = state.deposit_source === src;
              const carrying = (state.basal_eligible_sources || []).includes(src);
              const now = (state.basal_active_sources || []).includes(src);
              const tag = causal ? 'causal' : now ? 'now' : carrying ? 'carried' : '';
              return `<div class="syn-row"${causal ? ' style="color:#c084fc"' : ''}>
                <span class="name" title="${src}">${src}</span>
                <span class="wbar"><i style="left:0;width:${Math.max(0, Math.min(1, w / meta.threshold)) * 100}%;background:#c084fc"></i></span>
                <span class="wv">${w.toFixed(1)}</span>
                <span class="wv" style="color:var(--txt-2);width:52px">${tag}</span></div>`;
            }).join('')}
          </div>
          <div style="font-size:11px;color:var(--txt-2);margin-top:6px">one learned weight per local ordinary E; only the causal source learns</div>`,
          '', true, true) : (state.basal_weight != null ? card('Learned basal weight', `
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-variant-numeric:tabular-nums">${state.basal_weight.toFixed(1)}</span>
            <div style="flex:1;height:6px;background:var(--bg-3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${Math.max(0, Math.min(1, state.basal_weight / meta.threshold)) * 100}%;background:#c084fc;border-radius:3px"></div>
            </div>
            <span style="font-size:11px;color:var(--txt-2)">the only plastic C weight</span>
          </div>`) : '')}
        ${state.coincidence_active != null ? card('Coincidence gate', `
          <div class="insp-chips">
            <span class="tag" style="color:${state.basal_received || state.basal_eligible ? '#c084fc' : 'var(--txt-2)'}">basal ${state.basal_received ? 'now' : (state.basal_eligible ? 'eligible' : '—')}</span>
            <span class="tag" style="color:${state.apical_active ? '#f472b6' : 'var(--txt-2)'}">apical ${state.apical_active ? `on (${(state.apical_sources || []).length})` : 'off'}</span>
            <span class="firing-badge ${state.coincidence_active ? 'yes' : 'no'}">${state.coincidence_active ? 'COINCIDENCE' : 'no gate'}</span>
            <span style="font-size:11px;color:var(--txt-2);font-variant-numeric:tabular-nums">charge ${(state.coincidence_charge ?? 0).toFixed(1)}</span>
          </div>`, '', false, true) : ''}
        ${state.nest_state ? card(`NEST model state (${Object.keys(state.nest_state).length})`, `
          <div class="basal-list">
            ${Object.entries(state.nest_state).map(([k, v]) => `<div class="syn-row">
                <span class="name" title="${esc(k)}">${esc(k)}</span>
                <span class="wv" style="width:auto">${Number.isFinite(v) ? v.toFixed(4) : '—'}</span></div>`).join('')}
          </div>
          <div style="font-size:11px;color:var(--txt-2);margin-top:6px">every declared NESTML recordable, sampled by the multimeter under its model name</div>`,
          '', true, true) : ''}
        ${'spike_tau' in state ? card('Spike sub-boundary τ',
          `<span style="font-variant-numeric:tabular-nums">${state.spike_tau != null ? state.spike_tau.toFixed(4) : '—'}</span>
           <div style="font-size:11px;color:var(--txt-2);margin-top:2px">analytic within-boundary crossing time</div>`) : ''}
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
        ${synCard('Strongest connections', strongest, this.id, wref)}
        ${synCard(`Incoming (${incoming.length})`, incoming, this.id, wref)}
        ${synCard(`Outgoing (${outgoing.length})`, outgoing, this.id, wref)}
      </div>`;
    if (this.scroller) this.scroller.scrollTop = scrollTop;
  }
}

// ``full`` spans both grid columns -- use it for cards whose content would otherwise be
// squeezed into a half-width cell and re-wrap as its live values change.
function card(lbl, val, extra = '', small = false, full = false) {
  return `<div class="icard${full ? ' full' : ''}">
    <div class="lbl">${lbl}</div>
    <div class="val ${small ? 'sm' : ''}">${val}</div>${extra}</div>`;
}
function bar(x) {
  const pct = Math.max(0, Math.min(1, x)) * 100;
  return `<div class="bar"><i style="width:${pct}%"></i></div>`;
}
function synCard(title, list, self, wref = 1000) {
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
        <span class="wbar"><i style="left:50%;width:${(Math.min(1, Math.abs(sy.w) / wref) * 50).toFixed(0)}%;background:#9be15d"></i></span>
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
