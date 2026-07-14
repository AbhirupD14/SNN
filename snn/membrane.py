"""The neuron's membrane scalar state and operations.

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

    # --- mechanics (Phase 2: formulas relocated here verbatim) ----------------
    def deposit(self, amount):
        """Add `amount` of charge, then bound at the saturation ceiling v_sat
        (None = unbounded). This is the `potential += X; clamp to v_sat` pattern
        shared by the instantaneous, flow-rate, and skipped-time delivery paths."""
        self.potential += amount
        if self.v_sat is not None and self.potential > self.v_sat:
            self.potential = self.v_sat

    def check_threshold(self):
        """Fire iff out of the refractory window and at/above threshold."""
        return self.refractory_timer <= 0 and self.potential >= self.threshold

    def fire_reset(self, subtractive_reset):
        """Discharge on spike: full reset to rest, or (subtractive) keep the
        residual overshoot floored at rest; then arm refractory and mark spiked."""
        self.last_spike_time = 0
        if subtractive_reset:
            self.potential = max(self.potential - self.threshold, self.resting_potential)
        else:
            self.potential = self.resting_potential
        self.refractory_timer = self.refractory_period
        self.spiked = True

    def leak_and_countdown(self):
        """Advance the membrane one step: during refractory, clamp to rest and
        count down; otherwise apply the fixed-point leak toward rest. Returns True
        iff this was a non-refractory (leak) step, so the caller can run the
        trace-decay / inhibitory-drain that belong to that branch."""
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.potential = self.resting_potential
            return False
        self.potential += self._leak_num * (self.resting_potential - self.potential) / LEAK_SCALE
        return True
