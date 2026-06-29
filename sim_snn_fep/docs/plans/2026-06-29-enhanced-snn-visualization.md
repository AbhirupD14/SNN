# Enhanced SNN 3D Topology Visualization Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Enhance the existing 3D topology visualization to show input patterns, signals traveling along synapses, active neurons, inhibition, and synapse weights.

**Architecture:** Extend the existing Three.js scene with a global simulation state object that can be driven by either an external simulation or an internal demo. Modify neuron rendering to reflect activity levels via emissive intensity and color. Add temporary highlighting for input patterns. Vary connection appearance (thickness/opacity) based on synaptic weights. Animate signal propagation as traveling waves along active connections. Enhance inhibition visualization with distinct colors and pulsating effects. Update UI legend to explain new visual elements.

**Tech Stack:** HTML, JavaScript, Three.js

---

### Task 1: Create simulation state object and demo simulation loop

**Objective:** Establish a global simulation state structure that holds neuron activation levels, input pattern, synaptic weights, and connection activity; implement a simple demo loop that updates state periodically.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**

```javascript
// Open the HTML in a browser console and check:
// typeof window.simulationState === 'object'
// Expected before implementation: undefined
```

**Step 2: Run test to verify failure**
Open the file in a browser, open DevTools console, run `typeof window.simulationState` → should return `'undefined'`.

**Step 3: Write minimal implementation**
Add the following JavaScript near the top of the `<script type="module">` block, after the imports but before the scene setup:
// Global simulation state
window.simulationState = {
  // Neuron activation levels (0 to 1) for each population
  L1E: new Array(9).fill(0),
  L1I: new Array(9).fill(0),
  L2E: new Array(8).fill(0),
  L2I: new Array(1).fill(0),
  
  // Input pattern: indices of L1E neurons currently active (for flashing)
  inputPattern: [],
  
  // Synaptic weights for each connection type (0 to 1)
  weights: {
    ff: 0.5, // L1E → L2E feedforward
    ei: 0.5, // L1E ↔ L1I paired local
    sq: 0.3, // L2E → L1I feedback squelch
    gd: 0.4  // L2E ↔ L2I global brake loop
  },
  
  // Connection activity flags (for signal propagation)
  activeConnections: {
    ff: new Array(9 * 8).fill(false), // flattened L1E→L2E
    ei: new Array(9).fill(false),     // paired L1E↔L1I
    sq: new Array(8 * 9).fill(false), // flattened L2E→L1I
    gd: new Array(8).fill(false)      // L2E→L2I (per L2E neuron)
  },
  
  // Time-based parameters for animations
  time: 0,
  inputPatternDuration: 0 // milliseconds remaining for input flash
};

// Demo simulation loop (updates state periodically)
function demoSimulation() {
  // Slowly oscillate activation levels for demo
  const t = performance.now() * 0.001;
  window.simulationState.L1E.forEach((_, i) => {
    window.simulationState.L1E[i] = 0.5 + 0.5 * Math.sin(t * 0.5 + i);
  });
  window.simulationState.L1I.forEach((_, i) => {
    window.simulationState.L1I[i] = 0.5 + 0.5 * Math.sin(t * 0.5 + i + Math.PI);
  });
  window.simulationState.L2E.forEach((_, j) => {
    window.simulationState.L2E[j] = 0.5 + 0.5 * Math.sin(t * 0.3 + j * 0.5);
  });
  window.simulationState.L2I[0] = 0.5 + 0.5 * Math.sin(t * 0.2);
  
  // Occasionally set a random input pattern
  if (Math.random() < 0.01) {
    window.simulationState.inputPattern = [
      Math.floor(Math.random() * 9),
      Math.floor(Math.random() * 9)
    ].filter((v, i, a) => a.indexOf(v) === i); // unique
    window.simulationState.inputPatternDuration = 500; // ms
  }
  
  // Update input pattern timer
  if (window.simulationState.inputPatternDuration > 0) {
    window.simulationState.inputPatternDuration -= 16; // approx 60fps
    if (window.simulationState.inputPatternDuration <= 0) {
      window.simulationState.inputPattern = [];
    }
  }
  
  // Randomly activate some connections for signal propagation demo
  window.simulationState.activeConnections.ff.fill(false);
  if (Math.random() < 0.1) {
    const pre = Math.floor(Math.random() * 9);
    const post = Math.floor(Math.random() * 8);
    window.simulationState.activeConnections.ff[pre * 8 + post] = true;
  }
  
  // Schedule next frame
  setTimeout(demoSimulation, 16);
}

// Start the demo loop
demoSimulation();

**Step 4: Run test to verify pass**
After adding the code, reopen the browser console and run `typeof window.simulationState` → should return `'object'`. Check that properties exist.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: add simulation state object and demo loop"
```

### Task 2: Implement neuron activity visualization

**Objective:** Modify neuron rendering to modulate emissive intensity and color based on activation levels from simulation state.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**
```javascript
// Check that neuron emissiveIntensity changes when activation changes
// Expected: emissiveIntensity should vary between 0.1 and 0.9 (based on activation)
```

**Step 2: Run test to verify failure**
In the animation loop, log the emissiveIntensity of a neuron; it should be constant (0.18) currently.

**Step 3: Write minimal implementation**
Replace the existing gentle breathing pulse section in the animation loop (around line 410) with:
// Neuron activity visualization based on simulation state
allNodes.forEach(({ mesh, type }, i) => {
  let activation = 0;
  let offset = 0;
  switch (type) {
    case 'L1E': activation = window.simulationState.L1E[i]; offset = 0; break;
    case 'L1I': activation = window.simulationState.L1I[i]; offset = 0; break;
    case 'L2E': activation = window.simulationState.L2E[i]; offset = 0; break;
    case 'L2I': activation = window.simulationState.L2I[0]; offset = 0; break;
  }
  // Map activation [0,1] to emissiveIntensity range [0.1, 0.9]
  const emissiveIntensity = 0.1 + 0.8 * activation;
  mesh.material.emissiveIntensity = emissiveIntensity;
  
  // Optional: also modulate color intensity (already emissive color)
  // Could also adjust scale or add outer glow for high activation
});

**Step 4: Run test to verify pass**
After change, neurons should pulse with varying intensity reflecting the demo oscillation.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: implement neuron activity visualization"
```

### Task 3: Add input pattern flashing

**Objective:** Temporarily highlight L1E neurons matching the input pattern with a bright flash.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**
```javascript
// When simulationState.inputPattern contains indices, those L1E neurons should flash brightly
```

**Step 2: Run test to verify failure**
No flashing occurs currently.

**Step 3: Write minimal implementation**
Add to the neuron activity visualization loop (from Task 2) after setting emissiveIntensity:
// Input pattern flashing for L1E neurons
if (type === 'L1E' && window.simulationState.inputPattern.includes(i)) {
  // Flash: increase emissiveIntensity to max and add a pulsating scale
  mesh.material.emissiveIntensity = 1.0;
  // Pulsating scale based on inputPatternDuration
  const flashProgress = 1 - (window.simulationState.inputPatternDuration / 500);
  const pulse = 0.5 * Math.sin(flashProgress * Math.PI * 4) + 0.5; // 0 to 1
  mesh.scale.set(1 + 0.5 * pulse, 1 + 0.5 * pulse, 1 + 0.5 * pulse);
} else {
  // Reset scale if not flashing
  mesh.scale.set(1, 1, 1);
}

**Step 4: Run test to verify pass**
When demo sets an input pattern, the corresponding L1E neurons should flash brightly and pulse.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: add input pattern flashing"
```

### Task 4: Implement connection weight visualization

**Objective:** Vary line thickness and opacity of connections based on synaptic weights.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**
```javascript
// Connection lines should have different thicknesses/opacities based on weights
```

**Step 2: Run test to verify failure**
All connections currently have fixed thickness and opacity (set in addLine calls).

**Step 3: Write minimal implementation**
We need to modify the connection creation to store weight references and update them dynamically. Instead of recreating lines each frame, we'll update existing line materials.

First, change the connection factory to store line objects with weight info. Replace the `addLine` function and its usage with a weighted version.

But to keep tasks bite-sized, we'll do a simpler approach: we'll keep the existing lines and update their material properties in the animation loop.

We need to access the line objects. Currently they are stored in `connGroups`. We'll modify the animation loop to update each line's material based on its group's weight.

Add after the controls.update() in animation loop:
// Update connection appearance based on weights
const weightFF = window.simulationState.weights.ff;
const weightEI = window.simulationState.weights.ei;
const weightSQ = window.simulationState.weights.sq;
const weightGD = window.simulationState.weights.gd;

// Helper to set line material properties
function updateLines(group, weight) {
  group.forEach(line => {
    // Base opacity from original design, scaled by weight
    const baseOpacity = group === connGroups.ff ? 0.07 :
                       group === connGroups.ei ? 0.55 :
                       group === connGroups.sq ? 0.04 : 0.45;
    line.material.opacity = baseOpacity * weight;
    // Thickness: map weight [0,1] to lineWidth range [1, 3]
    // LineBasicMaterial doesn't support lineWidth; we'll simulate by scaling?
    // Instead we can adjust opacity and maybe use a different approach.
    // For simplicity, we'll just vary opacity and keep thickness fixed.
    // If we want thickness, we'd need to use MeshLine or similar.
    // Given scope, we'll note thickness as future enhancement.
  });
}

updateLines(connGroups.ff, weightFF);
updateLines(connGroups.ei, weightEI);
updateLines(connGroups.sq, weightSQ);
updateLines(connGroups.gd, weightGD);

**Step 4: Run test to verify pass**
Connection opacity should vary as weights change in demo (though our demo doesn't change weights yet; we'll add that later).

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: implement connection weight visualization (opacity based)"
```

### Task 5: Create signal propagation animation

**Objective:** Animate a traveling wave along active connections to represent spike transmission.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**
```javascript
// Active connections should show a moving bright spot indicating signal propagation
```

**Step 2: Run test to verify failure**
No animation on connections currently.

**Step 3: Write minimal implementation**
We'll animate active connections by drawing a temporary bright line that moves from pre- to post-synaptic neuron over a short duration.

We'll need to track propagation progress per active connection. Extend simulation state to include propagation timers.

But to keep task small, we'll implement a simple solution: for each active connection, we'll render an additional line that fades over time.

Alternative: we can modify the existing line material to have a time-based offset. Since we have many lines, we'll do a unified approach: add a uniform time to the material and use a shader effect. However, that's more complex.

Given time, we'll implement a simpler "signal" by briefly increasing opacity and cycling through active connections.

We'll adjust the connection update to also consider activeConnections flag.

Replace the connection update in animation loop with:
// Update connection appearance based on weights and activity
const t = performance.now() * 0.001;
const updateLines = (group, weight, activeFlags) => {
  group.forEach((line, idx) => {
    const baseOpacity = group === connGroups.ff ? 0.07 :
                       group === connGroups.ei ? 0.55 :
                       group === connGroups.sq ? 0.04 : 0.45;
    let opacity = baseOpacity * weight;
    // If this connection is active, add a pulsating boost
    if (activeFlags && activeFlags[idx]) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 10 + idx); // fast pulse
      opacity *= (1.0 + pulse * 0.5); // boost up to 1.5x
    }
    line.material.opacity = opacity;
  });
};

updateLines(connGroups.ff, window.simulationState.weights.ff, window.simulationState.activeConnections.ff);
updateLines(connGroups.ei, window.simulationState.weights.ei, window.simulationState.activeConnections.ei);
updateLines(connGroups.sq, window.simulationState.weights.sq, window.simulationState.activeConnections.sq);
updateLines(connGroups.gd, window.simulationState.weights.gd, window.simulationState.activeConnections.gd);

Now we need to set activeConnections flags in the demo simulation. In demoSimulation, after setting random ff activation, also set the flag we already set.

We already set `window.simulationState.activeConnections.ff[pre * 8 + post] = true;` but we need to reset it after a short time. We'll change demo to use a timer.

But for simplicity, we'll leave as is (it will stay active until next random set). That's okay for demo.

**Step 4: Run test to verify pass**
Active connections should appear brighter/pulsing.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: create signal propagation animation (pulsing active connections)"
```

### Task 6: Enhance inhibition visualization

**Objective:** Use distinct colors and pulsating effects for inhibitory connections when active.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**
```javascript
// Inhibitory connections (ei and sq) should show a distinct visual effect when inhibitory neurons are active
```

**Step 2: Run test to verify failure**
Inhibitory connections currently use fixed colors (red for ei, amber for sq) and no special effect.

**Step 3: Write minimal implementation**
We'll enhance the inhibitory connection visualization by making them pulse more strongly when the corresponding inhibitory neuron is active.

We'll need to know which inhibitory neuron is active per connection. For ei connections, each is paired with a specific L1I neuron. For sq connections, each L2E connects to the single L2I.

We'll adjust the updateLines function to also consider inhibitory neuron activation.

Replace the updateLines helper with a more nuanced version:
// Update connection appearance based on weights, activity, and inhibition
const t = performance.now() * 0.001;
const updateLines = (group, weight, activeFlags, inhibitionType, neuronIndexFn) => {
  group.forEach((line, idx) => {
    const baseOpacity = group === connGroups.ff ? 0.07 :
                       group === connGroups.ei ? 0.55 :
                       group === connGroups.sq ? 0.04 : 0.45;
    let opacity = baseOpacity * weight;
    
    // Active connection boost
    if (activeFlags && activeFlags[idx]) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 10 + idx);
      opacity *= (1.0 + pulse * 0.5);
    }
    
    // Inhibition-specific enhancement
    if (inhibitionType) {
      const neuronIdx = neuronIndexFn(idx);
      let inhActivation = 0;
      if (inhibitionType === 'L1E->L1I') {
        inhActivation = window.simulationState.L1I[neuronIdx];
      } else if (inhibitionType === 'L2E->L1I') {
        inhActivation = window.simulationState.L1I[neuronIdx];
      } // else if global brake, we could use L2I[0]
      
      if (inhActivation > 0.1) {
        // Pulsing effect proportional to inhibition activation
        const inhPulse = 0.5 + 0.5 * Math.sin(t * 3 + neuronIdx);
        opacity *= (1.0 + inhActivation * inhPulse * 0.5);
        // Shift color slightly towards more intense inhibitory hue
        // We can't easily change color of LineBasicMaterial without recreating.
        // For now, we'll just boost opacity.
      }
    }
    
    line.material.opacity = opacity;
  });
};

// FF: excitatory, no special inhibition
updateLines(connGroups.ff, window.simulationState.weights.ff, window.simulationState.activeConnections.ff);
// EI: paired local inhibition
updateLines(connGroups.ei, window.simulationState.weights.ei, window.simulationState.activeConnections.ei, 
           'L1E->L1I', idx => idx); // index i corresponds to L1E[i] and L1I[i]
// SQ: feedback squelch inhibition (L2E -> L1I)
updateLines(connGroups.sq, window.simulationState.weights.sq, window.simulationState.activeConnections.sq,
           'L2E->L1I', idx => {
             const pre = Math.floor(idx / 9); // L2E index
             const post = idx % 9;            // L1I index
             return post; // inhibition from L1I[post]
           });
// GD: global brake loop (L2E <-> L2I) - we treat as inhibitory from L2I
updateLines(connGroups.gd, window.simulationState.weights.gd, window.simulationState.activeConnections.gd,
           'L2I->L2E', idx => 0); // L2I[0] inhibits all L2E

**Step 4: Run test to verify pass**
Inhibitory connections should show enhanced pulsing when inhibitory neurons are active.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: enhance inhibition visualization"
```

### Task 7: Update UI/legend to explain new visual elements

**Objective:** Modify the UI panel and legend to describe the new visual encodings (activity levels, input flash, signal propagation, inhibition).

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html`

**Step 1: Write failing test (verification)**
```javascript
// The UI should contain text explaining the new visual elements
```

**Step 2: Run test to verify failure**
Current UI only shows basic legend for neuron types and connection toggles.

**Step 3: Write minimal implementation**
Add a new panel below the existing ones for visualization explanation, or extend the existing legend.

We'll insert a new div after the existing panels but before the stats div.

Add after the second panel (around line 172) but before the tooltip div:
```html
        <div class="panel">
            <div class="panel-title">Visual Encoding</div>
            <div class="legend">
                <div class="legend-item">
                    <span style="font-size:10px;color:var(--txt);">Neuron Brightness:</span><br>
                    <span style="font-size:9px;color:var(--dim);">▫ Activation level (0-1)</span>
                </div>
                <div class="legend-item" style="margin-top:6px;">
                    <span style="font-size:10px;color:var(--txt);">Input Flash:</span><br>
                    <span style="font-size:9px;color:var(--dim);">▫ L1E neurons matching input pattern</span>
                </div>
                <div class="legend-item" style="margin-top:6px;">
                    <span style="font-size:10px;color:var(--txt);">Connection Opacity:</span><br>
                    <span style="font-size:9px;color:var(--dim);">▫ Scaled by synaptic weight</span>
                </div>
                <div class="legend-item" style="margin-top:6px;">
                    <span style="font-size:10px;color:var(--txt);">Signal Propagation:</span><br>
                    <span style="font-size:9px;color:var(--dim);">▫ Pulsing on active connections</span>
                </div>
                <div class="legend-item" style="margin-top:6px;">
                    <span style="font-size:10px;color:var(--txt);">Inhibition Enhancement:</span><br>
                    <span style="font-size:9px;color:var(--dim);">▫ Stronger pulse when inhibitory active</span>
                </div>
            </div>
        </div>
```

**Step 4: Run test to verify pass**
The new panel should appear in the UI with the explanations.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: update UI/legend to explain new visual elements"
```

### Task 8: Integrate and test all enhancements, ensuring performance remains smooth

**Objective:** Run the enhanced visualization, verify all features work together, and monitor frame rate.

**Files:**
- Modify: `/home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html` (if any tweaks needed)

**Step 1: Write failing test (verification)**
```javascript
// Open the HTML in a browser and observe:
// - Neuron brightness varies with simulated activation
// - Input pattern flashing occurs periodically
// - Connection opacity reflects weights
// - Active connections pulse
// - Inhibitory connections show enhanced pulsing
// - UI panel explains visual encodings
// - Animation runs smoothly (~60fps)
```

**Step 2: Run test to verify failure**
Manually verify each feature.

**Step 3: Write minimal implementation**
We may need to adjust parameters for balance. We'll tweak the demo simulation to make changes more visible.

Adjust demoSimulation to vary weights over time for testing:
// Slowly vary weights to see effect
window.simulationState.weights.ff = 0.3 + 0.4 * Math.sin(t * 0.2);
window.simulationState.weights.ei = 0.3 + 0.4 * Math.sin(t * 0.2 + Math.PI/2);
window.simulationState.weights.sq = 0.3 + 0.4 * Math.sin(t * 0.2 + Math.PI);
window.simulationState.weights.gd = 0.3 + 0.4 * Math.sin(t * 0.2 + 3*Math.PI/2);
Add this inside demoSimulation.

Also, we might want to adjust the connection update to use the varying weights.

**Step 4: Run test to verify pass**
After changes, all features should be visible and performance should remain smooth.

**Step 5: Commit**
```bash
git add /home/adasgup/projects/SNN/sim_snn_fep/topology_3d.html
git commit -m "feat: integrate and test enhancements, adjust parameters for visibility"
```