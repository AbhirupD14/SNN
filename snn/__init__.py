"""Reusable neuron components extracted from the original neuron implementation.

Strangler-fig migration: these types incrementally take ownership of state and
behavior that currently lives in the ~1000-line `neuron_flexible.Neuron` monolith,
under the bit-exact golden contract in tests/golden/. Nothing here changes any
equation or default; it relocates state so the God object can be decomposed.

Phase 1 introduces:
  - NeuralEntity : Paul-style base contract (entity_id + update()).
  - SynapseBank  : owns the vectorized afferent arrays + construction path.
  - Membrane     : owns the membrane scalars (potential, threshold, refractory,
                   leak, v_sat).
"""

from snn.entity import NeuralEntity
from snn.membrane import Membrane
from snn.synapses import SynapseBank
from snn.config import NeuronConfig

__all__ = ["NeuralEntity", "Membrane", "SynapseBank", "NeuronConfig"]
