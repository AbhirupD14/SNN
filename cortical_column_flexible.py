"""
CorticalColumn with explicit per-source feedforward fan-in.

In the legacy weighted-inhibition layout, each excitatory neuron's afferents are:

    [ from_local_I , from_below_0 , from_below_1 , ... , from_below_{n_ff-1} ]

so every feedforward source (e.g. each L1 pixel) has its OWN trainable synapse.
That is what lets the charge-based local plasticity rule carve a selective
receptive field per neuron -- a single aggregated "from_below" input cannot,
because the pattern is summed away before it reaches a trainable weight.

The shared inhibitory neuron's afferent weights are:

    [ from_local_E_0 , ... , from_local_E_{n-1} , feedback_0 , ... ]

E->I weights are excitatory (positive). A legacy column can return inhibition
through negative I->E weights. The active SimulationEngine instead constructs the
column with `include_local_inhibition=False`: E afferents are only the feedforward
sources, and the shared I neuron returns an unweighted competitive-reset event.
"""

import numpy as np
from neuron_flexible import LEAK_SCALE, UNIT, Neuron


class CorticalColumn:
    """Pool of excitatory neurons sharing one inhibitory neuron, with per-source
    feedforward fan-in and optional feedback fan-in from a higher layer."""

    def __init__(self, n_neurons, threshold=1000 / UNIT, refractory_period=2,
                 learning_rate=0.05, weight_cap=1000 / UNIT, leak_rate=10 / LEAK_SCALE,
                 include_local_inhibition=True):
        self.n_neurons = n_neurons
        # Feedforward afferent layout offset. Legacy columns prepend one weighted
        # local-I gate at index 0 (offset 1); the active hard-reset architecture has
        # no learned L2I->L2E gate, so the feedforward pixels start at index 0
        # (offset 0). set_feedforward_weights()/feedforward_weights() use this offset.
        self.include_local_inhibition = bool(include_local_inhibition)
        self.feedforward_offset = 1 if self.include_local_inhibition else 0
        # Excitatory neurons: afferents are [from_local_I, *from_below] (legacy) or
        # [*from_below] when include_local_inhibition is False.
        self.excitatory_neurons = [
            Neuron(threshold=threshold, refractory_period=refractory_period,
                   learning_rate=learning_rate, weight_cap=weight_cap, leak_rate=leak_rate)
            for _ in range(n_neurons)
        ]
        # Shared inhibitory neuron: afferents are [*from_local_E, *feedback].
        self.inhibitory_neuron = Neuron(
            threshold=threshold, refractory_period=refractory_period,
            learning_rate=learning_rate, weight_cap=weight_cap, leak_rate=leak_rate)

        self._connections_finalized = False
        self.n_feedforward_inputs = 0
        self.n_feedback_inputs = 0

    def setup_connectivity(self, n_feedforward_inputs=0, n_feedback_inputs=0):
        """
        Allocate afferent synapses (placeholders at 0.0, filled by the setters).

        Legacy E neuron: index 0 = from_local_I; indices 1..n_ff = feedforward.
        Hard-reset E neuron: indices 0..n_ff-1 are feedforward (no local-I gate).
        I neuron: indices 0..n_neurons-1 = from each local E; then n_feedback feedback.
        """
        self.n_feedforward_inputs = n_feedforward_inputs
        self.n_feedback_inputs = n_feedback_inputs

        for e in self.excitatory_neurons:
            if self.include_local_inhibition:
                e.add_input_connection(0.0)             # 0: from the shared inhibitory neuron
            for _ in range(n_feedforward_inputs):
                e.add_input_connection(0.0)             # one synapse per feedforward source

        for _ in range(self.n_neurons):
            self.inhibitory_neuron.add_input_connection(0.0)   # from each local E
        for _ in range(n_feedback_inputs):
            self.inhibitory_neuron.add_input_connection(0.0)   # feedback from the layer above

    def finalize_connections(self):
        """Lock all connections; required before simulation."""
        for e in self.excitatory_neurons:
            e.finalize_connections()
        self.inhibitory_neuron.finalize_connections()
        self._connections_finalized = True

    def _ensure_finalized(self):
        if not self._connections_finalized:
            raise RuntimeError("Connections not finalized. Call finalize_connections() first.")

    # ---- excitatory afferents -------------------------------------------------
    def set_local_inhibition_weights(self, weight):
        """I -> E (index 0 on every E neuron). Should be negative. Only valid on a
        column built WITH the local-I gate; otherwise index 0 is a feedforward pixel
        and writing it would silently corrupt a receptive field, so raise instead."""
        self._ensure_finalized()
        if not self.include_local_inhibition:
            raise RuntimeError(
                "set_local_inhibition_weights() is invalid on a column built with "
                "include_local_inhibition=False (no learned local-I gate; the active "
                "L2 competition uses Neuron.apply_competitive_reset instead).")
        for e in self.excitatory_neurons:
            w = e._weights_array.copy()
            w[0] = weight
            e._weights_array = w

    def set_feedforward_weights(self, weights):
        """
        below -> E (starting at `feedforward_offset`) on every E neuron. Positive.

        `weights` may be:
          - scalar:                 same value on every feedforward synapse
          - 1D array length n_ff:   same receptive field for every E neuron
          - 2D array (n_neurons, n_ff): a distinct receptive field per E neuron
        """
        self._ensure_finalized()
        start = self.feedforward_offset
        stop = start + self.n_feedforward_inputs
        scalar = np.isscalar(weights)
        if not scalar:
            weights = np.asarray(weights, dtype=float)
        for j, e in enumerate(self.excitatory_neurons):
            w = e._weights_array.copy()
            if scalar:
                w[start:stop] = weights
            elif weights.ndim == 2:
                w[start:stop] = weights[j]
            else:
                w[start:stop] = weights
            e._weights_array = w

    # ---- inhibitory afferents -------------------------------------------------
    def set_lateral_excitation_weights(self, weight):
        """
        local E -> I (indices 0..n_neurons-1 on the shared I neuron). Should be
        positive: an E spike drives the shared inhibitory neuron. A legacy column
        returns through weighted local inhibition; the active engine uses an
        unweighted competitive-reset event instead.
        """
        self._ensure_finalized()
        w = self.inhibitory_neuron._weights_array.copy()
        w[0:self.n_neurons] = weight
        self.inhibitory_neuron._weights_array = w

    def set_feedback_weights(self, weight):
        """above E -> I (indices n_neurons..n_neurons+n_feedback-1). Sign per source."""
        self._ensure_finalized()
        start, stop = self.n_neurons, self.n_neurons + self.n_feedback_inputs
        w = self.inhibitory_neuron._weights_array.copy()
        w[start:stop] = weight
        self.inhibitory_neuron._weights_array = w

    # ---- introspection --------------------------------------------------------
    def feedforward_weights(self):
        """Return the (n_neurons, n_ff) matrix of feedforward receptive fields."""
        self._ensure_finalized()
        start = self.feedforward_offset
        stop = start + self.n_feedforward_inputs
        return np.array([e._weights_array[start:stop]
                         for e in self.excitatory_neurons])

    def get_state(self):
        self._ensure_finalized()
        return {
            'excitatory_potentials': [n.potential for n in self.excitatory_neurons],
            'excitatory_spiked': [n.spiked for n in self.excitatory_neurons],
            'excitatory_weights': [n._weights_array.copy() for n in self.excitatory_neurons],
            'inhibitory_potential': self.inhibitory_neuron.potential,
            'inhibitory_spiked': self.inhibitory_neuron.spiked,
            'inhibitory_weights': self.inhibitory_neuron._weights_array.copy(),
            'refractory_timers_E': [n.refractory_timer for n in self.excitatory_neurons],
            'refractory_timer_I': self.inhibitory_neuron.refractory_timer,
        }
