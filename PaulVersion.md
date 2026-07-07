# Technical Specification: Cognative Paradigm (v0.8)

## 1. System Overview
The Cognative Paradigm is a neuromorphic AI framework based on the "1 vs Z" principle: a binary register state where a neuron is either in state **Z** (resting/silent) or **1** (firing/active). The architecture simulates a hierarchical learning process to recognize line patterns on a 3x3 grid.

### High-Level Data Flow
`Input Grid (3x3)` $\rightarrow$ `Layer 1 (E/I Relay Pairs)` $\rightarrow$ `Nucleus Ring (WTA Competition)` $\rightarrow$ `Binding & Symbol Recognition`.

### Research Foundations
This framework is an original synthesis of several neuroscience and computational primitives. While it references existing literature, it is governed by the normative authority of `tenants.txt` and `paradigm_spec.md`.

*   **LIF (Leaky Integrate-and-Fire):** Based on standard biophysical models (e.g., Gerstner & Kistler) to bridge continuous potential with discrete event output.
*   **WTA (Winner-Take-All):** Implements the canonical cortical microcircuit motif (citing Amari, Douglas & Martin) for competitive pattern selection.
*   **STDP (Spike-Timing Dependent Plasticity):** Utilizes causal timing windows for local weight updates (citing Bi & Poo).
*   **Sparse Selective Coding:** Inspired by the concept of highly selective neurons representing specific patterns (citing Quiroga et al.).

**Explicit Non-Goals:** The paradigm rejects Backpropagation, Batch Training, and Binary Activation Vectors (0/1) in favor of local, continuous, event-based learning.

---

## 2. Mathematical Foundation

All calculations are **local to the neuron**. There are no global loss functions or centralized gradients.

### 2.1 LIF Membrane Dynamics (The Core Engine)
Every neuron in the system follows a discrete-time **Leaky Integrate-and-Fire (LIF)** model.

**1. The Leak (Passive Decay):**
At every timestep, the membrane potential $V_m$ decays toward the resting potential:
$$V_m(t+1) = V_m(t) \cdot \left(1 - \frac{dt}{\tau_m}\right) + V_{rest} \cdot \left(\frac{dt}{\tau_m}\right)$$
*   $\tau_m$: Membrane time constant (defines how fast the "memory" of a spike fades).
*   $dt$: Timestep resolution.

**2. Integration (Summation):**
When a presynaptic neuron fires a `1`, the postsynaptic neuron integrates the synaptic weight $w$:
$$V_m = V_m + w$$
*   For Excitatory edges: $w > 0$
*   For Inhibitory edges: $w < 0$

**3. Spiking & Reset:**
A neuron registers a `1` (spikes) if:
$$V_m \ge V_{threshold} \quad \text{and} \quad t > t_{last\_spike} + R$$
Immediately after spiking, the potential is reset:
$$V_m \rightarrow V_{reset}$$
And the neuron enters a **Refractory Period** $R$, during which it cannot fire regardless of $V_m$.

---

### 2.2 Layer 1: Local E/I Balance
Layer 1 uses "Pair Dynamics" to prevent runaway excitation.

**Feedforward Inhibition:**
The Inhibitory (I) neuron is driven by the Excitatory (E) neuron's current potential:
$$\text{Drive}_I = \max(0, (V_{m,E} - \text{offset}) \cdot \text{gain})$$
If the I-neuron spikes, it applies a local penalty to the E-neuron:
$$V_{m,E} = \max(V_{rest}, V_{m,E} - \text{strength})$$

**Collateral Recruitment:**
To ensure the I-neuron resets the E-neuron quickly after a spike:
$$\text{If } E \text{ spikes} \rightarrow \text{Drive}_I = \text{Drive}_I + w_{collateral}$$

---

### 2.3 Layer 2: Nucleus & Global WTA
The Nucleus implements a **Winner-Take-All (WTA)** competition.

**WTA Arbitration:**
If multiple competitors cross the threshold, the one with the maximum $V_m$ is selected as the winner. The system then triggers the **Central Inhibitor**:
1.  **Central Drive:** $\text{Drive}_{central} = f(\text{count of candidates})$
2.  **Central Spike:** If $V_{m,central} \ge \theta_{central} \rightarrow \text{Register} = 1$.
3.  **Local Suppression:** Every candidate *except the winner* receives an inhibitory spike:
    $$V_{m,competitor} = \max(0, V_{m,competitor} - w_{inhib})$$

---

### 2.4 Learning & Plasticity (The Memory)

#### A. STDP (Spike-Timing Dependent Plasticity)
Weights are updated based on the relative timing of spikes ($\Delta t = t_{post} - t_{pre}$).

*   **LTP (Potentiation):** If $0 < \Delta t \le \text{Window}_{pot}$
    $$w_{new} = \min(w_{max}, w_{old} + \Delta_{LTP})$$
*   **LTD (Depression):** If $0 < -\Delta t \le \text{Window}_{dep}$
    $$w_{new} = \max(w_{min}, w_{old} - \Delta_{LTD})$$

#### B. Inhibitory Homeostasis
To keep neurons from becoming permanently silent or hyperactive, the system adjusts inhibitory strength based on the **Exponential Moving Average (EMA)** of the firing rate $R$:

**EMA Rate Update:**
$$R_{ema}(t+1) = (1 - \alpha) \cdot R_{ema}(t) + \alpha \cdot \text{spike}$$
*   $\alpha$: Smoothing factor ($1/\text{window}$).

**Strength Adjustment:**
$$\text{Strength}_{new} = \text{Strength}_{old} + \eta \cdot (R_{ema} - R_{target})$$
*   $\eta$: Learning rate for homeostasis.
*   If $R_{ema} > R_{target} \rightarrow$ Inhibition increases (dampening the neuron).
*   If $R_{ema} < R_{target} \rightarrow$ Inhibition decreases (sensitizing the neuron).

---

## 3. Summary of Tiers
| Tier | Component | Primary Math/Logic | Goal |
| :--- | :--- | :--- | :--- |
| **L0** | Input | Polarized Weights | Signal Generation |
| **L1** | Relay Pairs | LIF + Local I-suppression | Feature Filtering |
| **L2** | Nucleus Ring | WTA + Binding Gate | Pattern Integration |
| **Meta** | Learning | STDP + Homeostasis | Stability & Memory |
