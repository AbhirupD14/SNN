"""Membrane -- the neuron's membrane scalars (REFACTOR_PLAN.md, Phase 1).

Owns potential, resting potential, threshold, the refractory timer/period, the
integer leak numerator, the saturation ceiling v_sat, and the spike bookkeeping.
Pure state in Phase 1 (the leak/refractory/reset FORMULAS still live in
`Neuron.update`/`fire` and are moved here in Phase 2). Fixed-point leak control is
preserved exactly: the leak amount is the integer numerator `_leak_num` over the
integer `LEAK_SCALE`, never a float constant.
"""

from neuron_flexible import LEAK_SCALE, leak_num


class Membrane:
    def __init__(self, threshold, resting_potential, refractory_period, leak_rate,
                 v_sat=None):
        self.threshold = threshold
        self.resting_potential = resting_potential
        self.refractory_period = refractory_period
        self.refractory_timer = 0
        self.potential = resting_potential
        self.v_sat = v_sat
        self.spiked = False
        self.last_spike_time = float("-inf")
        self.leak_rate = leak_rate            # property backed by _leak_num

    @property
    def leak_rate(self):
        """Leak fraction per step, backed by the integer numerator `_leak_num`
        over `LEAK_SCALE` (fixed-point leak control)."""
        return self._leak_num / LEAK_SCALE

    @leak_rate.setter
    def leak_rate(self, value):
        self._leak_num = leak_num(value)
