"""NeuralEntity -- the shared base contract (REFACTOR_PLAN.md, Phase 1).

A direct analogue of the `cipp-learning` (Paul) base type: anything that lives on
the simulation timeline has a stable identity and advances one step via `update`.
`Neuron` implements it now; layer/network objects may in a later phase.

This is deliberately minimal -- it does NOT impose membrane or learning semantics,
which belong to the composed `Membrane` / `SynapseBank` / rule objects.
"""

from abc import ABC, abstractmethod


class NeuralEntity(ABC):
    """Base for entities that advance on the simulation clock."""

    def __init__(self, entity_id=None):
        self.entity_id = entity_id

    @abstractmethod
    def update(self, dt=1):
        """Advance internal state by one time step. Concrete subclasses define the
        actual dynamics (leak, refractory countdown, homeostasis, ...)."""
        raise NotImplementedError
