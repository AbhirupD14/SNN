# Architectural Plan: Symmetry Breaking & L2 Stability Refinement (v2)

## 1. Executive Summary
The network currently exhibits "single-winner collapse" or "unstable flickering" during pattern consolidation. While the underlying learning rules are non-linear, the trigger dynamics and the winner-resolution logic are causing a lack of robust "round-robin" competition and inconsistent symbol assignment.

This plan implements two critical structural refinements to transition the network from mere competition to stable, selective consolidation.

---

## 2. Component Refinements

### A. Winner Resolution Logic (The "First-Spike" Rule)
**Current State**: Winner is decided by the **Latest Spike** (recency), with a tie-break of total frequency.
**Problem**: This rewards the "slowest" match or the "last survivor," making the winner identity a lottery of timing rather than a measure of affinity.

**New Specification**:
Update  in :
1. **Primary Rule**: The neuron with the **EARLIEST spike time** in the episode wins.
2. **Tie-break**: If multiple neurons share the earliest timestamp, the one with the **MOST total spikes** over the episode wins.
3. **Justification**: High-affinity matches reach threshold faster. This aligns the "Yellow Circle" with the most selective response.

### B. L2I Evidence Window (The "Competition" Window)
**Current State**:  (7% per step).
**Problem**: The evidence window is too short. The L2I neuron decays its potential faster than the L2E population can collectively build it up. This prevents the "round-robin" phase where multiple neurons contribute to a trigger, leading to a lack of competition.

**New Specification**:
Update  in :
- **Target Value**: Reduce to **0.03** (range 0.02 to 0.04).
- **Goal**: Lengthen the temporal integration window to ensure L2I fires reliably when a population is active, forcing the L2E neurons to compete via the inhibitory gates.

---

## 3. Documentation Debt
**Observation**: The  describes an old linear "activity" trace mode that has been replaced in the code by the non-linear charge-based rule (using  and quadratic saturation).
**Task**: The  should update the  to reflect the current active non-linear plasticity rule to prevent future architectural confusion.

---

## 4. Definition of Done (DoD)

The  must verify the following using  and :

1. **Tiling Efficiency**: 8/8 patterns map to distinct L2E winners across multiple seeds.
2. **Latency Correlation**: The "Winner" (yellow circle) must consistently be the neuron that fired first in the episode.
3. **Competition Check**: L2I must fire in a "round-robin" fashion during early training, transitioning to "single-source" triggering only after consolidation.

## 5. Guardrails
- No Global Signals, No Hard WTA, Preserve Confidence-Gated Consolidation.
