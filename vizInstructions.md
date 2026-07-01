# Task: Design and Implement a Professional Real-Time Dashboard for the Neural Simulation

## Objective

Design and implement a modern web dashboard for a biologically inspired neural simulation.

The dashboard is **not responsible for running the simulation**. The Python simulation remains the source of truth. The dashboard exists to visualize the network, inspect neuron state, control execution, and assist with debugging and experimentation.

The first experiment is intentionally small:

* 9 binary input neurons arranged as a 3×3 grid.
* The network learns to recognize the **8 possible straight-line patterns** (3 horizontal, 3 vertical, 2 diagonal).
* This experiment serves as the foundation for progressively larger experiments (plus signs, crosses, letters, hierarchical assemblies, symbolic reasoning, etc.).

Although the initial network is small, architect the dashboard so it can comfortably scale to thousands of neurons and many layers without requiring a redesign.

---

# Architecture

Use the following architecture:

```text
Python Neural Simulation
        │
        │ JSON over WebSocket
        ▼
FastAPI Backend
        │
        ▼
HTML + CSS + JavaScript Frontend
```

The simulation should never directly manipulate the frontend.

The backend is responsible for exposing simulation data.

The frontend is responsible for visualization and user interaction.

Keep the simulation completely decoupled from the UI.

---

# Backend

Use **FastAPI**.

Provide REST endpoints for simulation control.

Required endpoints:

```
GET    /state
POST   /start
POST   /pause
POST   /step
POST   /reset
```

Provide a WebSocket endpoint:

```
WS /ws
```

The websocket should continuously stream simulation updates while the simulation is running.

---

# Serialization

Create a serializer that converts the internal simulation state into JSON.

Separate static data from dynamic data whenever possible.

Dynamic state should include information such as:

* timestep
* neuron activations
* membrane potentials
* firing state
* assembly membership
* recently changed synapses
* statistics

Static topology should not be retransmitted every frame unless it changes.

---

# Frontend

Implement using:

* HTML
* CSS
* Vanilla JavaScript

Avoid frontend frameworks unless absolutely necessary.

The frontend should automatically connect to the websocket and update continuously.

Rendering should be event-driven rather than rebuilding the page every frame.

---

# Dashboard Layout

Design a polished research dashboard similar to professional engineering and scientific software.

Examples of design inspiration:

* TensorBoard
* Grafana
* Unreal Engine Editor
* Blender
* Visual Studio Code
* NVIDIA Omniverse

The interface should be clean, modular, responsive, and information-dense without feeling cluttered.

Use a modern dark theme.

---

## Overall Layout

Use a layout similar to:

```
+------------------------------------------------------------+
| Top Navigation / Simulation Status                         |
+-------------+---------------------------+------------------+
|             |                           |                  |
| Left Panel  |     Main Visualization    |  Inspector       |
| Controls    |       Workspace           |  Properties      |
|             |                           |                  |
+-------------+---------------------------+------------------+
| Bottom Statistics / Charts / Logs                          |
+------------------------------------------------------------+
```

The neuron visualization should occupy the majority of the available screen space.

---

# Top Navigation

Include:

* project title
* simulation status
* current timestep
* simulation speed
* websocket connection status

Global controls:

* Start
* Pause
* Resume
* Step
* Reset

---

# Left Sidebar

Organize controls into collapsible sections.

Include:

### Experiment Controls

* Start
* Pause
* Resume
* Step
* Reset

### Pattern Selection

Display a clickable 3×3 grid.

Allow manual activation of any input neuron.

Provide one-click buttons for all eight predefined line patterns.

Include:

* horizontal patterns
* vertical patterns
* diagonal patterns

Provide:

* Random Pattern
* Clear Pattern
* Noise Injection

### Manual Firing Mode

Allow manual stimulation of any neuron.

Allow selecting:

* neuron
* activation magnitude
* single pulse
* continuous stimulation

---

# Main Visualization

The center of the application should contain an interactive neuron visualization.

Implement:

* interactive 3D neuron rendering
* synapse rendering
* firing animations
* activation coloring
* assembly coloring
* camera controls
* zoom
* pan
* rotate

Clicking a neuron should immediately populate the inspector.

Provide optional display filters:

* show only active neurons
* hide weak synapses
* isolate assemblies
* filter by layer

---

# Right Sidebar

Display detailed information about the selected neuron.

Include:

* neuron ID
* layer
* neuron type
* activation
* threshold
* membrane potential
* firing state
* firing frequency
* assembly membership
* incoming synapses
* outgoing synapses
* strongest connections
* recent activity

Display data using clean cards rather than raw text.

---

# Bottom Panel

Provide multiple tabs.

## Activity

Display:

* heat map of Layer 2 neurons
* heat map of additional layers (when available)
* currently active neurons
* winning neuron
* activation histogram

## Statistics

Display live metrics:

* total neurons
* active neurons
* firing neurons
* average activation
* firing rate
* firing frequency
* simulation FPS
* timestep duration
* average synaptic weight

## Charts

Include continuously updating graphs for:

* firing rate
* active neuron count
* average activation
* assembly activity
* learning progress

## Event Log

Display:

* simulation events
* neuron firing events
* learning events
* warnings
* backend messages

---

# Visual Design

The application should resemble a polished desktop application rather than a simple webpage.

Design goals:

* professional
* modern
* clean
* responsive
* elegant
* minimal
* high information density

Use:

* subtle gradients
* rounded panels
* smooth animations
* consistent spacing
* restrained accent colors
* readable typography
* hover effects
* polished icons
* visually distinct cards

Avoid excessive visual effects.

The visualization should always remain the primary focal point.

---

# Performance

The dashboard should be designed for future scalability.

Requirements:

* support thousands of neurons
* avoid rebuilding the DOM
* reuse rendering objects
* update only changed state
* maintain smooth rendering
* separate simulation updates from rendering
* minimize allocations
* keep UI responsive during heavy simulation

---

# Code Organization

Suggested project structure:

```
backend/
    api.py
    websocket.py
    serializer.py

frontend/
    index.html
    style.css
    app.js
    websocket.js
    renderer.js
    controls.js
    inspector.js
    charts.js
```

Keep responsibilities clearly separated.

---

# Deliverables

Produce:

1. FastAPI backend
2. REST API
3. WebSocket implementation
4. JSON serializer
5. Responsive frontend
6. Interactive neuron visualization
7. Simulation control interface
8. Charts and statistics
9. Neuron inspector
10. Pattern input interface
11. Manual firing interface
12. Professional styling
13. Documentation explaining:

    * architecture
    * API
    * websocket protocol
    * serialization format
    * rendering pipeline
    * project structure

---

# Constraints

* Do not modify the neural computation unless absolutely necessary.
* Keep the visualization layer completely independent of the simulation.
* Favor modular, maintainable, production-quality code.
* Prioritize readability and extensibility.
* The finished product should feel like a professional neuroscience research environment suitable for long-term development of biologically inspired neural architectures.
