"""Charge-delivery and local learning strategies.

The boolean-flag + inline-branch stacks in `Neuron._update_weights` (excitatory)
and `Neuron.apply_inhibition` (inhibitory) become swappable Strategy objects here.
The formulas are relocated VERBATIM under the Phase 0 golden gate; the flag ladder
becomes polymorphic dispatch over one selected rule. Strategies are stateless and
operate on the neuron passed in (they read/write its forwarded Membrane/SynapseBank
state), so they can be shared singletons.
"""

from snn.rules.excitatory import select_excitatory_rule, bounded_signed_update
from snn.rules.inhibitory import select_inhibitory_rule
from snn.rules.delivery import select_delivery, effective_weights

__all__ = ["select_excitatory_rule", "bounded_signed_update",
           "select_inhibitory_rule", "select_delivery", "effective_weights"]
