"""The minimal SNN model: two neuron types and their shared constants.

``ExcitatoryNeuron`` and ``InhibitoryNeuron`` are the only neuron implementations.
Topology, causal event order, and serialization live in ``backend.simulation``.
"""

from snn.neurons import (
    ExcitatoryNeuron,
    InhibitoryNeuron,
    E_THRESHOLD,
    I_THRESHOLD,
    E_WEIGHT_CAP,
    SUBTRACTIVE_SIGN,
    DEFAULT_ETA,
    DEFAULT_LEAK,
    DEFAULT_REFRACTORY,
)

__all__ = [
    "ExcitatoryNeuron",
    "InhibitoryNeuron",
    "E_THRESHOLD",
    "I_THRESHOLD",
    "E_WEIGHT_CAP",
    "SUBTRACTIVE_SIGN",
    "DEFAULT_ETA",
    "DEFAULT_LEAK",
    "DEFAULT_REFRACTORY",
]
