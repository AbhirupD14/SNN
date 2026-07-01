// Interactive 3D neuron/synapse renderer (Three.js). Objects are built once from
// the topology and then only their materials/scales are mutated per frame.
//
// Charge rings: each neuron has a TorusGeometry ring that scales from 0→1 as
// the membrane potential climbs from 0→threshold.  The ring billboards to the
// camera every frame so it always reads as a circular halo.  On spike the ring
// briefly blooms white before snapping back to zero.
//
// Spike visualisation: traveling orbs are replaced by an instantaneous edge
// flash.  Any synapse in the `emitted` list gets pulse=1.0, turning the edge
// white for ~2 frames before fading back to its colour.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  E: 0x5eead4, I: 0xf0788c, winner: 0xffce5c,
  feedforward: 0x4cc38a, inhibition: 0xf0788c, excitation: 0x7c9cff, feedback: 0xc084fc,
};
const WEAK = 0.25;

export class NeuronRenderer {
  constructor(container, { onSelect }) {
    this.container = container;
    this.onSelect = onSelect;
    // id -> {mesh, ring, meta, pulse, spiked, act, freq, assembly}
    this.neurons = new Map();
    this.edges = new Map();       // id -> {line, mat, syn, weight, pulse}
    this.filters = { active: false, weak: true, assembly: false, l1: true, l2: true, inh: true };
    this._last = null;
    this._selected = null;

    const scene = new THREE.Scene();
    this.scene = scene;

    const cam = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
    cam.position.set(11, -9, 16);
    this.camera = cam;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    this.renderer = renderer;

    const controls = new OrbitControls(cam, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 3);
    this.controls = controls;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const p1 = new THREE.PointLight(0x9fc4ff, 0.8, 120); p1.position.set(14, 10, 24); scene.add(p1);
    const p2 = new THREE.PointLight(0x5eead4, 0.5, 120); p2.position.set(-16, -12, 6); scene.add(p2);

    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(1.05, 0.06, 12, 40),
      new THREE.MeshBasicMaterial({ color: COLORS.winner, transparent: true, opacity: 0.9 }));
    halo.visible = false;
    scene.add(halo);
    this.halo = halo;

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this._down = null;
    renderer.domElement.addEventListener('pointerdown', (e) => { this._down = [e.clientX, e.clientY]; });
    renderer.domElement.addEventListener('pointerup', (e) => this._handleClick(e));

    this._onResize();
    window.addEventListener('resize', () => this._onResize());
    this._loop();
  }

  // ------------------------------------------------------------- build
  build(topology) {
    for (const { mesh, ring } of this.neurons.values()) {
      this.scene.remove(mesh);
      if (ring) this.scene.remove(ring);
    }
    for (const { line } of this.edges.values()) this.scene.remove(line);
    this.neurons.clear(); this.edges.clear();
    this.pos = new Map();

    for (const m of topology.neurons) {
      this.pos.set(m.id, new THREE.Vector3(m.pos[0], m.pos[1], m.pos[2]));
    }

    for (const m of topology.neurons) {
      // Neuron sphere
      const r = m.layer === 'L2' ? (m.type === 'I' ? 0.62 : 0.55) : (m.type === 'I' ? 0.34 : 0.44);
      const geo = new THREE.SphereGeometry(r, 24, 18);
      const mat = new THREE.MeshStandardMaterial({
        color: 0x2b3446, emissive: COLORS[m.type], emissiveIntensity: 0.08,
        roughness: 0.45, metalness: 0.1 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(this.pos.get(m.id));
      mesh.userData.id = m.id;
      this.scene.add(mesh);

      // Charge ring: TorusGeometry around the sphere, starts at scale 0.
      // Inner radius slightly larger than the sphere; thin tube.
      const ringR = r + 0.18;
      const tubeR = m.layer === 'L2' ? 0.055 : 0.042;
      const ringGeo = new THREE.TorusGeometry(ringR, tubeR, 8, 40);
      const ringMat = new THREE.MeshBasicMaterial({
        color: COLORS[m.type], transparent: true, opacity: 0,
        depthWrite: false });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.copy(this.pos.get(m.id));
      ring.scale.setScalar(0);
      this.scene.add(ring);

      this.neurons.set(m.id, { mesh, ring, meta: m, pulse: 0, spiked: false, act: 0, freq: 0, assembly: null });
    }

    for (const s of topology.synapses) {
      const a = this.pos.get(s.source), b = this.pos.get(s.target);
      if (!a || !b) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
      const mat = new THREE.LineBasicMaterial({ color: COLORS[s.kind], transparent: true, opacity: 0.12 });
      const line = new THREE.Line(geo, mat);
      this.scene.add(line);
      this.edges.set(s.id, { line, mat, syn: s, weight: s.weight ?? 0, pulse: 0 });
    }
    this._applyEdgeWeights();
  }

  // ------------------------------------------------------------- update
  update(dynamic) {
    this._last = dynamic;
    for (const n of dynamic.neurons) {
      const e = this.neurons.get(n.id);
      if (!e) continue;
      e.act = n.activation; e.freq = n.freq; e.assembly = n.assembly;
      if (n.spiked && !e.spiked) e.pulse = 1.0;
      e.spiked = n.spiked;
    }
    for (const c of dynamic.changed_synapses || []) {
      const e = this.edges.get(c.id);
      if (e) { e.weight = c.weight; e.pulse = 1.0; }
    }
    if (dynamic.changed_synapses?.length) this._applyEdgeWeights();
    this._winner = dynamic.winner;
    if (dynamic.speed) this._simSpeed = dynamic.speed;
    this._applyFilters();

    // Flash each edge that carried a spike this step.
    for (const synId of dynamic.emitted || []) {
      const edge = this.edges.get(synId);
      if (edge) edge.pulse = 1.0;
    }
  }

  _applyEdgeWeights() {
    for (const e of this.edges.values()) {
      const mag = Math.min(1, Math.abs(e.weight));
      e.baseOpacity = 0.04 + 0.32 * mag;
    }
  }

  setFilters(f) { Object.assign(this.filters, f); this._applyFilters(); }

  _applyFilters() {
    const F = this.filters, win = this._winner;
    const assemblyNeurons = new Set();
    if (win) {
      assemblyNeurons.add(win);
      for (const e of this.edges.values())
        if (e.syn.target === win && e.syn.kind === 'feedforward' && Math.abs(e.weight) > WEAK)
          assemblyNeurons.add(e.syn.source);
    }
    for (const [id, e] of this.neurons) {
      let vis = true;
      const t = e.meta.type, layer = e.meta.layer;
      if (t === 'I' && !F.inh) vis = false;
      if (layer === 'L1' && !F.l1) vis = false;
      if (layer === 'L2' && !F.l2) vis = false;
      if (F.active && e.act < 0.05 && !e.spiked) vis = false;
      if (F.assembly && win && !assemblyNeurons.has(id)) vis = false;
      e.mesh.visible = vis;
      if (e.ring) e.ring.visible = vis;
    }
    for (const e of this.edges.values()) {
      let vis = true;
      if (F.weak && Math.abs(e.weight) < WEAK) vis = false;
      const sN = this.neurons.get(e.syn.source), tN = this.neurons.get(e.syn.target);
      if (sN && !sN.mesh.visible) vis = false;
      if (tN && !tN.mesh.visible) vis = false;
      if (F.assembly && win && e.syn.target !== win && e.syn.source !== win) vis = false;
      e.line.visible = vis;
    }
  }

  select(id) {
    if (this._selected) {
      const prev = this.neurons.get(this._selected);
      if (prev) prev.mesh.scale.setScalar(prev._selScale || 1);
    }
    this._selected = id;
  }

  _handleClick(e) {
    if (!this._down) return;
    const moved = Math.hypot(e.clientX - this._down[0], e.clientY - this._down[1]);
    this._down = null;
    if (moved > 5) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._pointer, this.camera);
    const meshes = [...this.neurons.values()].filter(n => n.mesh.visible).map(n => n.mesh);
    const hit = this._raycaster.intersectObjects(meshes)[0];
    if (hit) { this.select(hit.object.userData.id); this.onSelect?.(hit.object.userData.id); }
  }

  _onResize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    requestAnimationFrame(() => this._loop());

    for (const [id, e] of this.neurons) {
      // Sphere glow
      e.pulse *= 0.86;
      const glow = Math.max(e.act * 0.5, e.freq * 0.9) + e.pulse * 1.6;
      const isWin = id === this._winner;
      e.mesh.material.emissive.setHex(isWin ? COLORS.winner : COLORS[e.meta.type]);
      e.mesh.material.emissiveIntensity = THREE.MathUtils.clamp(0.08 + glow, 0.06, 2.2);
      const sel = id === this._selected ? 1.35 : 1;
      e.mesh.scale.setScalar((1 + e.pulse * 0.5) * sel);

      // Charge ring: scale = activation (0 → 1), blooms on spike via pulse.
      if (e.ring && e.mesh.visible) {
        const charge = THREE.MathUtils.clamp(e.act, 0, 1.5);
        const ringScale = charge + e.pulse * 0.5;
        e.ring.scale.setScalar(ringScale);
        // Opacity: invisible at zero charge, fully opaque at threshold.
        e.ring.material.opacity = charge < 0.02 ? 0 : THREE.MathUtils.clamp(0.3 + 0.6 * charge + e.pulse * 0.2, 0, 1);
        // Colour: white flash on spike, winner gold, else type colour.
        if (e.pulse > 0.6) {
          e.ring.material.color.setHex(0xffffff);
        } else {
          e.ring.material.color.setHex(isWin ? COLORS.winner : COLORS[e.meta.type]);
        }
        // Billboard to camera so the ring always reads as a flat halo.
        e.ring.quaternion.copy(this.camera.quaternion);
      }
    }

    for (const e of this.edges.values()) {
      e.pulse *= 0.9;
      e.mat.opacity = Math.min(1, (e.baseOpacity ?? 0.1) + e.pulse * 0.5);
      if (e.pulse > 0.05) e.mat.color.setHex(0xffffff);
      else e.mat.color.setHex(COLORS[e.syn.kind]);
    }

    if (this._winner && this.pos?.get(this._winner)) {
      const w = this.neurons.get(this._winner);
      this.halo.visible = !!w && w.mesh.visible;
      this.halo.position.lerp(this.pos.get(this._winner), 0.16);
      this.halo.quaternion.copy(this.camera.quaternion);
      this.halo.rotation.z += 0.01;
    } else this.halo.visible = false;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
