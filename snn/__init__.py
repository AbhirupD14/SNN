"""The conductance-based predictive-inhibition SNN model.

``ExcitatoryNeuron`` (conductance LIF + local activity trace), ``InhibitoryNeuron``
(stateless WTA/relay), and ``PredictiveInterneuron`` (pattern-specific relay with
strictly-local inhibitory output plasticity) are the neuron implementations.
Topology, causal event order, and serialization live in ``backend.simulation``.
"""

from snn.neurons import (
    ExcitatoryNeuron,
    InhibitoryNeuron,
    PredictiveInterneuron,
    E_THRESHOLD,
    I_THRESHOLD,
    E_WEIGHT_CAP,
    INHIBITORY_SIGN,
    DEFAULT_ETA,
    DEFAULT_LEAK,
    DEFAULT_REFRACTORY,
    leak_to_conductance,
)

__all__ = [
    "ExcitatoryNeuron",
    "InhibitoryNeuron",
    "PredictiveInterneuron",
    "E_THRESHOLD",
    "I_THRESHOLD",
    "E_WEIGHT_CAP",
    "INHIBITORY_SIGN",
    "DEFAULT_ETA",
    "DEFAULT_LEAK",
    "DEFAULT_REFRACTORY",
    "leak_to_conductance",
]
