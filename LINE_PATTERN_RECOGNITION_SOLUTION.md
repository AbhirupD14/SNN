# Line Pattern Recognition Problem - 2-Layer Network Solution

## Problem Definition
Recognize 8 possible straight line patterns in a 3x3 pixel grid:
- **3 Rows**: Top, middle, bottom rows
- **3 Columns**: Left, middle, right columns  
- **2 Diagonals**: Top-left to bottom-right, top-right to bottom-left

Each pattern is represented as a 9-element binary vector (row-major order).

## Network Architecture (2-Layer)

### **Layer 1: Input Layer**
- **Size**: 9 E/I pairs = 18 neurons total
- **Structure**: Each pixel has a dedicated excitatory-inhibitory neuron pair
- **Connections**:
  - Each E neuron: `[from_its_I (inhibitory), from_external_pixel_input]`
  - Each I neuron: `[from_L2_E_feedback]` (feedback from cortical column)
- **Function**: Represents raw pixel inputs with local competition

### **Layer 2: Cortical Column** 
- **Size**: 8 E neurons + 1 I neuron = 9 neurons total
- **Structure**: 8 output neurons (one per line pattern) sharing 1 inhibitory neuron
- **Connections**:
  - Each E neuron: `[from_local_I (inhibitory), from_L1_aggregated_input]` 
  - I neuron: `[from_all_8_L2_E (lateral inhibition), from_feedback (none in 2-layer)]`
- **Function**: Learns to map pixel patterns to specific line categories

### **Connectivity (as specified in requirements)**
- **L1 → L2 (Feedforward)**:
  - E1 → L2E (excitatory): Each L1 E pixel connects to all L2 E neurons
  - I1 → L2E (inhibitory): Each L1 I neuron connects to all L2 E neurons
  - *Note: L1 sends to L2 E neurons only*
  
- **L2 → L1 (Feedback, Unidirectional)**:
  - **E2 → I1 (excitatory)**: Each L2 E neuron connects to all L1 I neurons
  - **I1 -/-> E₂ⱼ (NO connection)**: L1 inhibitory neurons do NOT send to L2 E neurons
  - *This satisfies the unidirectional constraint E2ⱼ → I1ᵢ (no I1ᵢ → E₂ⱼ)*

### **Learning Mechanism**
1. **Spike-dependent plasticity**: Weights increase ONLY when neuron fires
2. **Competitive learning**: Lateral inhibition in L2 creates winner-take-all dynamics
3. **Consolidation**: Repeated presentation strengthens connections from active pixels to winning L2 E neuron
4. **Specificity**: Each line pattern eventually activates a distinct L2 E neuron

## Implementation Files
- **`demo_line_patterns.py`**: Complete training and testing simulation
- **`demo_line_patterns_simple.py`**: Simplified demonstration of network response
- **`neuron.py`/`neuron_flexible.py`**: Leaky Integrate and Fire Neuron with plasticity
- **`layers.py`**: InputLayer class
- **`cortical_column_flexible.py`**: CorticalColumn class supporting hierarchical extensions

## How Learning Works
For each line pattern (presented repeatedly):
1. **Activation**: Active pixels drive corresponding L1 E neurons to fire
2. **Feedforward**: L1 E activity drives L2 E neurons via weighted connections
3. **Competition**: L2 lateral inhibition (E2 → I2 → E2) suppresses all but strongest L2 E neuron
4. **Plasticity**: Winning L2 E neuron strengthens connections from its active L1 E pixels
5. **Consolidation**: After repeated presentations, one L2 E neuron becomes selective for that pattern
6. **Progression**: Move to next pattern once current pattern shows stable winner

## Expected Outcome
After training:
- Each of the 8 line patterns maps to a distinct L2 E neuron (may use <8 if some patterns share winners)
- The network exhibits invariant recognition: same line pattern → same winning neuron
- Weight matrices show strengthened connections from pattern pixels to winning neuron
- L2 lateral inhibition ensures sparse, distributed representations

## Verification
The implementation has been tested and verified to:
- Correctly implement all neuron properties (leaky integration, spike-only updates, etc.)
- Establish proper layered connectivity with specified constraints
- Support competitive learning through lateral inhibition
- Enable the requested hierarchical extension (L3→L2→L1) via `cortical_column_flexible.py`

This 2-layer network provides a biologically plausible solution to visual pattern recognition using competitive learning in a hierarchical architecture with feedback inhibition.