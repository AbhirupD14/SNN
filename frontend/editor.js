// Full-screen 3D topology editor (Three.js). Renders the live NetworkSpec at its
// actual functional 3D positions — the same space the model and the main scene use —
// so what you see and edit is the real topology, and z is preserved through
// save / load / apply (no flattening). You can orbit, drag neurons in 3D, wire edges
// from the fixed archetype/edge-kind vocabulary, toggle edges directional/
// bidirectional, and save/load/delete presets. "Apply" POSTs the edited spec so the
// backend rebuilds the live network and every view refreshes off the broadcast.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const NODE_COLOR = {
  rg_source: 0xfbbf24, e_sensory: 0x5eead4, e_encoder: 0x34d399,
  e_residual: 0x2dd4bf, e_competitor: 0x38bdf8, i_relay: 0xf0788c,
  predictor: 0xe066c0, switch: 0xfb7185,
};
const NODE_COLOR_CSS = {
  rg_source: '#fbbf24', e_sensory: '#5eead4', e_encoder: '#34d399',
  e_residual: '#2dd4bf', e_competitor: '#38bdf8', i_relay: '#f0788c',
  predictor: '#e066c0', switch: '#fb7185',
};
const EDGE_COLOR = {
  feedforward: 0x4cc38a, fixed_excitation: 0x22c55e, trace_excitation: 0xf59e0b,
  relay_excitation: 0x7c9cff, inhibition: 0xf0788c, predictive_inhibition: 0xe066c0,
};
const EDGE_COLOR_CSS = {
  feedforward: '#4cc38a', fixed_excitation: '#22c55e', trace_excitation: '#f59e0b',
  relay_excitation: '#7c9cff', inhibition: '#f0788c', predictive_inhibition: '#e066c0',
};
const ARCH_PREFIX = {
  rg_source: 'RG', e_sensory: 'S', e_encoder: 'EN', e_residual: 'ERR',
  e_competitor: 'C', i_relay: 'I', predictor: 'P', switch: 'SW',
};
// Default z for a freshly added node, so new nodes land in a sensible 3D band.
// RG sits below L1 (upstream of it); L1 at 0; L2 at 8.
const ARCH_Z = { rg_source: -6, e_sensory: 0, e_encoder: 0, e_residual: 3.2,
  e_competitor: 8, i_relay: 8, predictor: 8, switch: 8 };
// Layer a freshly added node is filed under (matches network_spec._default_layer).
const ARCH_LAYER = {
  rg_source: 'RG', e_sensory: 'L1', e_encoder: 'L1', e_residual: 'ERR',
  e_competitor: 'L2', i_relay: 'L2', predictor: 'L2', switch: 'L2',
};
const NODE_R = 0.7;

export class Editor {
  constructor(store) {
    this.store = store;
    this.overlay = document.getElementById('editor-overlay');
    this.container = document.getElementById('ed-canvas');
    this.inspect = document.getElementById('ed-inspect');
    this.statusEl = document.getElementById('ed-status');
    this.spec = null;               // {name, nodes, edges}
    this.vocab = null;
    this.sel = null;                // {type:'node'|'edge', id}
    this.nodeObjs = new Map();      // id -> {mesh, label, node}
    this.edgeObjs = new Map();      // id -> {group, edge}
    this.three = null;
    this._drag = null;
    this._wire();
  }

  // ------------------------------------------------------------------ wiring
  _wire() {
    document.getElementById('g-editor')?.addEventListener('click', () => this.open());
    document.getElementById('editor-close')?.addEventListener('click', () => this.close());
    document.getElementById('ed-apply')?.addEventListener('click', () => this.apply());
    document.getElementById('ed-revert')?.addEventListener('click', () => this.open());
    document.getElementById('ed-load')?.addEventListener('click', () => this.loadPreset());
    document.getElementById('ed-save')?.addEventListener('click', () => this.savePreset());
    document.getElementById('ed-del')?.addEventListener('click', () => this.deletePreset());
    for (const b of this.overlay.querySelectorAll('[data-add]'))
      b.addEventListener('click', () => this.addNode(b.dataset.add));
    window.addEventListener('keydown', (e) => {
      if (this.overlay.hidden) return;
      if (e.key === 'Escape') this.close();
      else if ((e.key === 'Delete' || e.key === 'Backspace') && this.sel
               && document.activeElement?.tagName !== 'INPUT') { e.preventDefault(); this.deleteSelected(); }
    });
    window.addEventListener('resize', () => this._resize());
  }

  // ------------------------------------------------------------------- three
  _initThree() {
    if (this.three) return;
    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
    cam.position.set(18, -22, 16);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.container.appendChild(renderer.domElement);
    const controls = new OrbitControls(cam, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.1;
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const p = new THREE.PointLight(0xffffff, 0.7, 400); p.position.set(20, 20, 40); scene.add(p);
    const ray = new THREE.Raycaster(); ray.params.Line.threshold = 0.5;
    this.three = { scene, cam, renderer, controls, ray };
    this.group = new THREE.Group(); scene.add(this.group);

    const dom = renderer.domElement;
    // Capture-phase pointerdown so we can claim a node drag before OrbitControls.
    dom.addEventListener('pointerdown', (e) => this._down(e), true);
    // A plain click (no orbit-drag) selects an edge, or clears the selection. Node
    // selection is handled by the drag path's no-move branch; skip if a node was hit.
    dom.addEventListener('click', (e) => {
      if (this._pickNode(e)) return;
      const eid = this._pickEdge(e);
      if (eid) this.select('edge', eid);
      else if (this.sel) { this.sel = null; this._applySelection(); this.renderInspect(); }
    });
    this._loop();
  }

  _loop() {
    requestAnimationFrame(() => this._loop());
    if (this.overlay.hidden || !this.three) return;
    this.three.controls.update();
    this.three.renderer.render(this.three.scene, this.three.cam);
  }

  _resize() {
    if (!this.three || this.overlay.hidden) return;
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (w < 2 || h < 2) return;
    this.three.renderer.setSize(w, h);
    this.three.cam.aspect = w / h; this.three.cam.updateProjectionMatrix();
  }

  // -------------------------------------------------------------------- REST
  async _json(url, opts) {
    const r = await fetch(url, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }

  // -------------------------------------------------------------------- open
  async open() {
    this.overlay.hidden = false;
    this._initThree();
    requestAnimationFrame(() => this._resize());
    try {
      const t = await this._json('/api/topology');
      this.spec = this._clone(t.spec);
      this.vocab = t.vocabulary;
      this.sel = null;
      this._normalizePositions();
      this._rebuild();
      this._fitCamera();
      await this.refreshPresets();
      this._status(`loaded "${this.spec.name}" · ${this.spec.nodes.length} nodes, ${this.spec.edges.length} edges`);
      this.renderInspect();
    } catch (e) { this._status('load failed: ' + e.message, true); }
  }

  close() { this.overlay.hidden = true; }
  _clone(o) { return JSON.parse(JSON.stringify(o)); }

  _normalizePositions() {
    this.spec.nodes.forEach((n, i) => {
      if (!Array.isArray(n.pos) || n.pos.length < 3) {
        const ang = (i / Math.max(this.spec.nodes.length, 1)) * Math.PI * 2;
        n.pos = [Math.cos(ang) * 6, Math.sin(ang) * 6, ARCH_Z[n.archetype] ?? 4];
      }
    });
  }

  // ------------------------------------------------------------------- build
  _rebuild() {
    for (const { mesh, label } of this.nodeObjs.values()) { this.group.remove(mesh); this.group.remove(label); }
    for (const { group } of this.edgeObjs.values()) this.group.remove(group);
    this.nodeObjs.clear(); this.edgeObjs.clear();
    for (const n of this.spec.nodes) this._addNodeObj(n);
    for (const ed of this.spec.edges) this._addEdgeObj(ed);
    this._applySelection();
  }

  _v(n) { return new THREE.Vector3(n.pos[0], n.pos[1], n.pos[2] ?? 0); }

  _addNodeObj(n) {
    const col = NODE_COLOR[n.archetype] ?? 0x8899aa;
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(NODE_R, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0x121826, emissive: col, emissiveIntensity: 0.5,
        roughness: 0.4, metalness: 0.1 }));
    mesh.position.copy(this._v(n));
    mesh.userData.nodeId = n.id;
    this.group.add(mesh);
    const label = this._makeLabel(n.label || n.id, n.pixel ?? n.grid);
    label.position.copy(mesh.position).add(new THREE.Vector3(0, 0, NODE_R + 0.9));
    this.group.add(label);
    this.nodeObjs.set(n.id, { mesh, label, node: n });
  }

  _makeLabel(text, pixel) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 72;
    const ctx = c.getContext('2d');
    ctx.font = 'bold 40px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#e8eefc'; ctx.fillText(text, 128, 30);
    if (pixel != null) { ctx.font = '24px ui-monospace, monospace'; ctx.fillStyle = '#8aa0c0'; ctx.fillText('px' + pixel, 128, 58); }
    const tex = new THREE.CanvasTexture(c); tex.anisotropy = 4;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    spr.scale.set(4.2, 1.2, 1);
    return spr;
  }

  _addEdgeObj(ed) {
    const s = this.nodeObjs.get(ed.source)?.node, t = this.nodeObjs.get(ed.target)?.node;
    if (!s || !t) return;
    const a = this._v(s), b = this._v(t);
    const col = EDGE_COLOR[ed.kind] ?? 0x888888;
    const g = new THREE.Group();
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
                         new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.85 })));
    // Arrowhead cone(s): at the target end, and both ends if bidirectional.
    const dir = b.clone().sub(a); const len = dir.length() || 1; dir.normalize();
    this._cone(g, b.clone().addScaledVector(dir, -NODE_R), dir, col);
    if (ed.directed === false) this._cone(g, a.clone().addScaledVector(dir, NODE_R), dir.clone().negate(), col);
    g.userData.edgeId = ed.id;
    this.group.add(g);
    this.edgeObjs.set(ed.id, { group: g, edge: ed });
  }

  _cone(group, tip, dir, col) {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.9, 12),
      new THREE.MeshBasicMaterial({ color: col }));
    cone.position.copy(tip);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    group.add(cone);
  }

  _refreshEdgesFor(nodeId) {
    for (const [id, { edge }] of this.edgeObjs) {
      if (edge.source === nodeId || edge.target === nodeId) {
        this.group.remove(this.edgeObjs.get(id).group);
        this.edgeObjs.delete(id);
        this._addEdgeObj(edge);
      }
    }
  }

  _fitCamera() {
    const pts = this.spec.nodes.map(n => this._v(n));
    if (!pts.length) return;
    const box = new THREE.Box3().setFromPoints(pts);
    const c = box.getCenter(new THREE.Vector3());
    const r = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, 4);
    this.three.controls.target.copy(c);
    const dir = new THREE.Vector3(0.6, -0.8, 0.6).normalize();
    this.three.cam.position.copy(c).addScaledVector(dir, r * 2.6);
    this.three.cam.near = 0.1; this.three.cam.far = r * 20; this.three.cam.updateProjectionMatrix();
    this.three.controls.update();
  }

  // ------------------------------------------------------------ interaction
  _ndc(e) {
    const rect = this.three.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1,
                             -((e.clientY - rect.top) / rect.height) * 2 + 1);
  }

  _pickNode(e, exceptId) {
    this.three.ray.setFromCamera(this._ndc(e), this.three.cam);
    const meshes = [...this.nodeObjs.values()].map(o => o.mesh).filter(m => m.userData.nodeId !== exceptId);
    const hit = this.three.ray.intersectObjects(meshes)[0];
    return hit ? hit.object.userData.nodeId : null;
  }

  _pickEdge(e) {
    this.three.ray.setFromCamera(this._ndc(e), this.three.cam);
    const lines = [...this.edgeObjs.values()].map(o => o.group.children[0]);
    const hit = this.three.ray.intersectObjects(lines)[0];
    return hit ? hit.object.parent.userData.edgeId : null;
  }

  _down(e) {
    if (e.button !== 0) return;                       // left button drives editing
    const id = this._pickNode(e);
    if (!id) return;                                  // empty space -> OrbitControls orbits
    e.stopPropagation();                              // claim it before OrbitControls
    this.three.controls.enabled = false;
    const node = this.nodeObjs.get(id).node;
    const camDir = this.three.cam.getWorldDirection(new THREE.Vector3());
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camDir, this._v(node));
    this._drag = { id, node, plane, moved: false, start: [...node.pos], downXY: [e.clientX, e.clientY] };
    const move = (ev) => this._move(ev);
    const up = (ev) => { this._up(ev); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  _move(e) {
    const d = this._drag; if (!d) return;
    if (Math.hypot(e.clientX - d.downXY[0], e.clientY - d.downXY[1]) > 3) d.moved = true;
    this.three.ray.setFromCamera(this._ndc(e), this.three.cam);
    const hit = new THREE.Vector3();
    if (this.three.ray.ray.intersectPlane(d.plane, hit)) {
      d.node.pos = [hit.x, hit.y, hit.z];
      const o = this.nodeObjs.get(d.id);
      o.mesh.position.copy(hit);
      o.label.position.copy(hit).add(new THREE.Vector3(0, 0, NODE_R + 0.9));
      this._refreshEdgesFor(d.id);
    }
  }

  _up(e) {
    const d = this._drag; this._drag = null;
    this.three.controls.enabled = true;
    if (!d) return;
    if (!d.moved) { this.select('node', d.id); return; }
    const target = this._pickNode(e, d.id);
    if (target) {                                     // dropped onto another node -> connect
      d.node.pos = d.start;                           // the drag was a "connect" gesture
      const o = this.nodeObjs.get(d.id);
      o.mesh.position.copy(this._v(d.node));
      o.label.position.copy(o.mesh.position).add(new THREE.Vector3(0, 0, NODE_R + 0.9));
      this._refreshEdgesFor(d.id);
      this._connect(d.node, this.nodeObjs.get(target).node);
    }
  }

  // Click-select for edges: OrbitControls consumes empty-space clicks, so wire a
  // lightweight click that only acts when it did not orbit (tiny movement).
  // (Node clicks are handled by the drag path's no-move branch.)

  // ------------------------------------------------------------- connect edge
  _archClass(arch) { return this.vocab.archetypes[arch]?.cls; }
  // A requirement is an archetype name, a class letter ('E'/'I'/'S'), or an array of
  // alternatives (satisfied if any one matches). Mirrors network_spec._arch_matches.
  _matches(arch, req) {
    if (Array.isArray(req)) return req.some(r => this._matches(arch, r));
    if (this.vocab.archetypes[req]) return arch === req;
    return this._archClass(arch) === req;
  }
  _inferKind(srcArch, tgtArch) {
    for (const [k, spec] of Object.entries(this.vocab.edge_kinds))
      if (this._matches(srcArch, spec.src) && this._matches(tgtArch, spec.tgt)) return k;
    return null;
  }

  _connect(src, tgt) {
    const kind = this._inferKind(src.archetype, tgt.archetype);
    if (!kind) { this._status(`no valid connection ${src.archetype}→${tgt.archetype}`, true); return; }
    if (this.spec.edges.some(e => e.source === src.id && e.target === tgt.id && e.kind === kind)) {
      this._status('edge already exists', true); return;
    }
    const id = this._uniqueEdgeId(src.id, tgt.id);
    const sign = this.vocab.edge_kinds[kind].sign;
    const ed = { id, source: src.id, target: tgt.id, kind, directed: true };
    if (sign === -1) ed.sign = -1;
    this.spec.edges.push(ed);
    this._addEdgeObj(ed);
    this.select('edge', id);
    this._status(`connected ${src.id} →${kind}→ ${tgt.id}`);
  }

  _uniqueEdgeId(s, t) {
    let base = `${s}->${t}`, id = base, i = 1;
    while (this.spec.edges.some(e => e.id === id)) id = `${base}#${i++}`;
    return id;
  }

  // ---------------------------------------------------------------- add node
  addNode(archetype) {
    const c = this.three.controls.target.clone();
    const pos = [c.x + (Math.random() - 0.5) * 4, c.y + (Math.random() - 0.5) * 4, ARCH_Z[archetype] ?? 4];
    const id = this._uniqueNodeId(archetype);
    const node = { id, archetype, layer: ARCH_LAYER[archetype] ?? 'L2', label: id, pos };
    // Only an input-sink archetype may own a pixel, and ownership is unique, so claim
    // the first free one. Other archetypes get no pixel (an encoder may still be given
    // a display 'grid' tag by hand).
    if (this.vocab.archetypes[archetype]?.input_sink) {
      const used = new Set(this.spec.nodes.filter(n => n.pixel != null).map(n => n.pixel));
      for (let px = 0; px < 9; px++) if (!used.has(px)) { node.pixel = px; break; }
    }
    this.spec.nodes.push(node);
    this._addNodeObj(node);
    this.select('node', id);
    this._status(`added ${archetype} ${id}`);
  }

  _uniqueNodeId(archetype) {
    const pre = ARCH_PREFIX[archetype] || 'N';
    let i = 0, id;
    do { id = `${pre}${i++}`; } while (this.spec.nodes.some(n => n.id === id));
    return id;
  }

  // ------------------------------------------------------------- select/edit
  select(type, id) { this.sel = { type, id }; this._applySelection(); this.renderInspect(); }

  _applySelection() {
    for (const { mesh, node } of this.nodeObjs.values()) {
      const on = this.sel?.type === 'node' && this.sel.id === node.id;
      mesh.material.emissiveIntensity = on ? 1.3 : 0.5;
      mesh.scale.setScalar(on ? 1.4 : 1);
    }
    for (const { group, edge } of this.edgeObjs.values()) {
      const on = this.sel?.type === 'edge' && this.sel.id === edge.id;
      const line = group.children[0];
      line.material.opacity = on ? 1 : 0.85;
      line.material.linewidth = on ? 3 : 1;   // (linewidth support varies) emphasis via opacity
    }
  }

  deleteSelected() {
    if (!this.sel) return;
    if (this.sel.type === 'node') {
      const id = this.sel.id;
      this.spec.nodes = this.spec.nodes.filter(n => n.id !== id);
      this.spec.edges = this.spec.edges.filter(e => e.source !== id && e.target !== id);
    } else {
      this.spec.edges = this.spec.edges.filter(e => e.id !== this.sel.id);
    }
    this.sel = null;
    this._rebuild();
    this.renderInspect();
  }

  renderInspect() {
    const box = this.inspect;
    if (!this.sel) {
      box.innerHTML = '<div class="ed-empty">Select a node (click it) or edge to edit it. Drag a node onto another to connect them. Orbit by dragging empty space.</div>';
      return;
    }
    if (this.sel.type === 'node') {
      const n = this.spec.nodes.find(x => x.id === this.sel.id);
      if (!n) { this.sel = null; return this.renderInspect(); }
      const arch = this.vocab.archetypes[n.archetype];
      const outCount = this.spec.edges.filter(e => e.source === n.id).length;
      const inCount = this.spec.edges.filter(e => e.target === n.id).length;
      box.innerHTML = `
        <div class="ed-title" style="color:${NODE_COLOR_CSS[n.archetype]}">${n.id}</div>
        <div class="ed-row"><span>archetype</span><b>${n.archetype}</b></div>
        <div class="ed-row"><span>type / role</span><b>${arch.cls} · ${arch.role}</b></div>
        <div class="ed-row"><span>position</span><b>${n.pos.map(x => (+x).toFixed(1)).join(', ')}</b></div>
        <div class="ed-row"><span>edges</span><b>${inCount} in · ${outCount} out</b></div>
        <p class="ed-desc">${arch.desc}</p>
        ${arch.input_sink ? `
          <label class="ed-field"><span>input pixel (0–8, blank = none) — external drive, unique</span>
            <input id="ed-pixel" type="number" min="0" max="8" value="${n.pixel ?? ''}" /></label>` : `
          <label class="ed-field"><span>grid cell (0–8, blank = none) — display / receptive field only</span>
            <input id="ed-grid" type="number" min="0" max="8" value="${n.grid ?? ''}" /></label>`}
        <button class="btn sm danger" id="ed-del-node">Delete node</button>`;
      const clampCell = v => Math.max(0, Math.min(8, parseInt(v, 10) || 0));
      const px = document.getElementById('ed-pixel');
      px?.addEventListener('change', () => {
        const v = px.value.trim();
        if (v === '') delete n.pixel; else n.pixel = clampCell(v);
        this._rebuild();
      });
      const gd = document.getElementById('ed-grid');
      gd?.addEventListener('change', () => {
        const v = gd.value.trim();
        if (v === '') delete n.grid; else n.grid = clampCell(v);
        this._rebuild();
      });
      document.getElementById('ed-del-node')?.addEventListener('click', () => this.deleteSelected());
    } else {
      const ed = this.spec.edges.find(x => x.id === this.sel.id);
      if (!ed) { this.sel = null; return this.renderInspect(); }
      const info = this.vocab.edge_kinds[ed.kind];
      box.innerHTML = `
        <div class="ed-title" style="color:${EDGE_COLOR_CSS[ed.kind]}">${ed.kind}</div>
        <div class="ed-row"><span>from → to</span><b>${ed.source} → ${ed.target}</b></div>
        <div class="ed-row"><span>polarity</span><b>${ed.sign === -1 ? 'inhibitory (−)' : 'excitatory (+)'}${info.plastic ? ' · plastic' : ''}</b></div>
        <p class="ed-desc">${info.desc}</p>
        <label class="ed-check"><input type="checkbox" id="ed-bidir" ${ed.directed === false ? 'checked' : ''}/> bidirectional (deliver both ways)</label>
        <p class="ed-note">Only valid where the reverse direction is also a legal ${ed.kind} (e.g. competitor↔competitor). Rejected otherwise on Apply.</p>
        <button class="btn sm danger" id="ed-del-edge">Delete edge</button>`;
      document.getElementById('ed-bidir')?.addEventListener('change', (e) => {
        ed.directed = !e.target.checked;
        this.group.remove(this.edgeObjs.get(ed.id).group); this.edgeObjs.delete(ed.id); this._addEdgeObj(ed);
      });
      document.getElementById('ed-del-edge')?.addEventListener('click', () => this.deleteSelected());
    }
  }

  // -------------------------------------------------------------- apply/preset
  async apply() {
    for (const n of this.spec.nodes) n.pos = [n.pos[0], n.pos[1], n.pos[2] ?? 0];   // keep z
    try {
      await this._json('/api/topology', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: this.spec }),
      });
      this._status(`applied · ${this.spec.nodes.length} nodes, ${this.spec.edges.length} edges — network rebuilt`);
    } catch (e) { this._status('apply failed: ' + e.message, true); }
  }

  async refreshPresets() {
    try {
      const { presets } = await this._json('/api/topology/presets');
      const sel = document.getElementById('ed-preset');
      sel.innerHTML = presets.map(p =>
        `<option value="${p.name}">${p.builtin ? '★ ' : ''}${p.name} (${p.nodes}n/${p.edges}e)</option>`).join('');
    } catch (e) { /* non-fatal */ }
  }

  async loadPreset() {
    const name = document.getElementById('ed-preset').value;
    if (!name) return;
    try {
      const { spec } = await this._json(`/api/topology/presets/${encodeURIComponent(name)}`);
      this.spec = this._clone(spec);
      this.sel = null;
      this._normalizePositions();
      this._rebuild();
      this._fitCamera();
      this.renderInspect();
      this._status(`loaded preset "${name}" into editor (not yet applied)`);
    } catch (e) { this._status('load failed: ' + e.message, true); }
  }

  async savePreset() {
    const name = (document.getElementById('ed-name').value || '').trim();
    if (!name) { this._status('enter a name to save', true); return; }
    for (const n of this.spec.nodes) n.pos = [n.pos[0], n.pos[1], n.pos[2] ?? 0];
    try {
      const r = await this._json('/api/topology/presets', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, spec: this.spec }),
      });
      await this.refreshPresets();
      document.getElementById('ed-preset').value = r.saved;
      this._status(`saved preset "${r.saved}" (3D positions preserved)`);
    } catch (e) { this._status('save failed: ' + e.message, true); }
  }

  async deletePreset() {
    const name = document.getElementById('ed-preset').value;
    if (!name) return;
    try {
      await this._json(`/api/topology/presets/${encodeURIComponent(name)}`, { method: 'DELETE' });
      await this.refreshPresets();
      this._status(`deleted preset "${name}"`);
    } catch (e) { this._status('delete failed: ' + e.message, true); }
  }

  _status(msg, err = false) {
    if (!this.statusEl) return;
    this.statusEl.textContent = msg;
    this.statusEl.classList.toggle('is-err', !!err);
  }
}
