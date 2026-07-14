"""Vectorized afferent storage and synapse operations.

The "synapse as a real object" win from Paul's design, made VECTORIZED: one object
owns all N afferents as parallel numpy arrays (weights, eligibility trace,
confidence, per-afferent delivery distance, last-input mask) plus the staged
construction path. It is NOT one object per edge -- the SNN's numerics depend on
`np.dot` over weight arrays and fixed-point scaling, so per-synapse Python objects
would change both performance and floating-point results.

Phase 1 only relocates state: the arrays and construction/clip/alloc logic move
here unchanged, and `Neuron` forwards to this bank via properties so every existing
read/write site (`self._weights_array`, external `n._weights_array = ...`, etc.)
keeps working bit-for-bit. Learning formulas that mutate the arrays still live on
`Neuron` in Phase 1; they operate on the same array objects this bank exposes.

`weight_cap` and `confidence_init` are read from the owning neuron at set-time (not
cached), so a caller that changes `n.weight_cap` before re-assigning weights gets
the same clip it did when this logic lived on the neuron.
"""

import numpy as np


class SynapseBank:
    def __init__(self, owner):
        # Back-reference for set-time reads of weight_cap / confidence_init. The
        # cycle (neuron <-> bank) is intentional and cheap.
        self._owner = owner
        self.weights_list = []          # accumulates during staged construction
        self.weights_array = None       # numpy array after finalization
        self.trace = None               # per-synapse eligibility trace
        self.confidence = None          # per-synapse confidence
        self.distance = None            # per-afferent delivery distance d_i
        self.last_input_spikes = None   # binary participation mask of last volley
        self.connections_finalized = False

    def add_input_connection(self, weight):
        if self.connections_finalized:
            raise RuntimeError("Cannot add connections after finalization. "
                               "Call finalize_connections() to lock connections.")
        self.weights_list.append(float(weight))

    def finalize(self):
        if not self.connections_finalized:
            weights = self.weights_list if self.weights_list else []
            self.set_weights(weights, finalized=True)

    def set_weights(self, weights, finalized=None):
        """Replace the afferent weight vector and resize aligned local state.
        Confidence and distance are preserved across a same-length re-assignment."""
        arr = np.asarray(weights, dtype=float)
        cap = self._owner.weight_cap
        self.weights_array = np.clip(arr, -cap, cap)
        n = len(self.weights_array)
        self.trace = np.zeros(n)
        if self.confidence is None or len(self.confidence) != n:
            self.confidence = np.full(n, self._owner.confidence_init)
        # Per-afferent delivery distance; default 1.0 (no attenuation). Preserved
        # across weight re-assignments of the same fan-in.
        if self.distance is None or len(self.distance) != n:
            self.distance = np.ones(n)
        self.last_input_spikes = np.zeros(n)
        if finalized is not None:
            self.connections_finalized = finalized

    def ensure_finalized(self):
        if not self.connections_finalized:
            raise RuntimeError("Connections not finalized. "
                               "Call finalize_connections() before simulation.")
