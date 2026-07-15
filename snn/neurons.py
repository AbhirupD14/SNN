"""The two neuron types of the active SNN model.

There is exactly one scientific model here -- no mode flags, no legacy ablation
switches. An ``ExcitatoryNeuron`` integrates accumulating charge, learns its
accumulating weights with one nonlinear rule when it fires, and is silenced by a
frozen subtractive gate that hard-wipes its membrane to rest. An
``InhibitoryNeuron`` is a stateless instant relay: any +1 input makes it fire in
the same causal phase; it owns no weights and no membrane.

Numeric scale is intentionally human-readable: the shared excitatory threshold is
``1000`` so potentials and weights are easy to inspect in the dashboard. The
inhibitory threshold is a reported invariant equal to one third of that.
"""

from __future__ import annotations

import numpy as np

# --- Shared scientific constants -------------------------------------------
# Every excitatory population shares one threshold and one accumulating-weight
# cap. Inhibitory neurons are the explicit exception: their reported threshold is
# E_THRESHOLD / 3. Inhibitory neurons have no weights, so a cap is meaningless for
# them.
E_THRESHOLD = 1000.0
I_THRESHOLD = E_THRESHOLD / 3.0          # ~333.3333, a reported/visual invariant only
E_WEIGHT_CAP = 1000.0                    # shared accumulating-weight cap (freq experiment may retune)
SUBTRACTIVE_SIGN = -1                    # the one explicit negative-sign convention

DEFAULT_ETA = 0.01
DEFAULT_LEAK = 0.0
DEFAULT_REFRACTORY = 0


class ExcitatoryNeuron:
    """A leaky integrate-and-fire excitatory unit with one learning rule.

    State is deliberately small. ``acc_weights`` are nonnegative accumulating
    magnitudes; the subtractive gate is a single frozen magnitude applied under
    ``SUBTRACTIVE_SIGN`` as a hard wipe. Geometry (``acc_distance_factor``) is a
    per-afferent *learning-rate* multiplier only -- it never scales delivered
    charge.
    """

    def __init__(self, nid, role, *, acc_weights, acc_distance_factor,
                 threshold=E_THRESHOLD, w_max=E_WEIGHT_CAP,
                 leak_rate=DEFAULT_LEAK, refractory_steps=DEFAULT_REFRACTORY,
                 eta=DEFAULT_ETA, learn=True, subt_magnitude=0.0):
        self.id = nid
        self.role = role
        self.type = 'E'

        self.v_rest = 0.0
        self.V = 0.0
        self.threshold = float(threshold)
        self.w_max = float(w_max)

        if not 0.0 <= leak_rate <= 1.0:
            raise ValueError(f'leak_rate must be in [0, 1], got {leak_rate}')
        self.leak_rate = float(leak_rate)
        self.refractory_steps = int(refractory_steps)
        self.refractory_timer = 0

        self.acc_weights = np.asarray(acc_weights, dtype=float)
        self.acc_distance_factor = np.asarray(acc_distance_factor, dtype=float)
        if self.acc_weights.shape != self.acc_distance_factor.shape:
            raise ValueError('acc_weights and acc_distance_factor must align')
        self.eta = float(eta)
        self.learn = bool(learn)

        # A pretrained, frozen subtractive gate. For every wired inhibitory input
        # this equals the target threshold, so a real inhibitory spike wipes any
        # subthreshold or threshold-crossing charge to exactly rest.
        self.subt_magnitude = float(subt_magnitude)

        self.spiked = False
        self.v_pre = 0.0

    # ------------------------------------------------------------------ views
    @property
    def potential(self):
        return self.V

    @property
    def activation(self):
        return self.V / self.threshold if self.threshold else 0.0

    # --------------------------------------------------------------- dynamics
    def receive_acc(self, charge):
        """Accumulating delivery: V <- V + charge. Geometry never appears here."""
        self.V += float(charge)

    def can_fire(self):
        return self.refractory_timer == 0 and self.V >= self.threshold

    def fire(self):
        """Threshold crossing: record v_pre, reset to rest, arm refractory."""
        self.v_pre = self.V
        self.spiked = True
        self.V = self.v_rest
        self.refractory_timer = self.refractory_steps
        return self.v_pre

    def hard_wipe(self):
        """Leave the membrane at rest, never below. Returns the charge removed."""
        removed = self.V - self.v_rest
        self.V = self.v_rest
        return removed

    def receive_subt(self, active=True):
        """Apply the frozen subtractive gate as a signed event, then enforce the
        hard-wipe postcondition.

            signed = SUBTRACTIVE_SIGN * subt_magnitude
            V <- max(v_rest, V + signed)

        Because the gate magnitude equals the target threshold, this floors any
        subthreshold charge to rest. A threshold *crosser* (V > threshold) could
        leave positive residual under the raw subtraction, so the invariant is
        enforced explicitly: after a real inhibitory event ``V == v_rest``.
        """
        if not active:
            return 0.0
        pre = self.V
        signed = SUBTRACTIVE_SIGN * self.subt_magnitude
        self.V = max(self.v_rest, self.V + signed)
        self.V = self.v_rest                      # hard-wipe invariant
        return pre - self.V

    def advance(self):
        """One post-step housekeeping tick: refractory countdown, then leak.

        Off-by-one convention: ``refractory_steps`` is the number of steps AFTER
        the firing step during which the neuron cannot fire. The step in which the
        neuron fired does not count (``spiked`` is still set here), so a value of
        ``R`` blocks exactly ``R`` subsequent steps. Leak is applied once per
        completed step and only when not refractory.
        """
        if self.spiked:
            return                                 # fired this step: already at rest
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            return                                 # no leak during refractory
        if self.leak_rate:
            self.V = self.v_rest + (1.0 - self.leak_rate) * (self.V - self.v_rest)

    # -------------------------------------------------------------- learning
    def update_acc_weights(self, participation):
        """The one accumulating-weight rule. Runs when this neuron fires.

            p          = threshold - sum(acc_weights)          # pre-update, signed
            signal_i   = +1 if afferent i spiked in the causal volley else -1
            delta_i    = eta * p * signal_i * distance_factor_i * (1 - (w_i/w_max)**2)
            w_i        = clip(w_i + delta_i, 0, w_max)

        ``p`` is deliberately signed: once the stored weights sum past threshold it
        changes sign, so a saturated neuron depresses rather than potentiates. Only
        accumulating weights ever change; the subtractive gate is frozen.
        """
        if not self.learn:
            return
        w = self.acc_weights
        participation = np.asarray(participation, dtype=bool)
        p = self.threshold - float(w.sum())
        signal = np.where(participation, 1.0, -1.0)
        delta = self.eta * p * signal * self.acc_distance_factor * (1.0 - (w / self.w_max) ** 2)
        np.clip(w + delta, 0.0, self.w_max, out=w)


class InhibitoryNeuron:
    """A stateless instant relay. No weights, no membrane, no plasticity.

    Any +1 input in the current causal phase makes it fire immediately and emit a
    +1 to every connected excitatory target. The one-third threshold is a reported
    scientific/visual invariant, not a second integration mechanism.
    """

    def __init__(self, nid, role, *, threshold=I_THRESHOLD):
        self.id = nid
        self.role = role
        self.type = 'I'
        self.threshold = float(threshold)          # reported only; there is no integrator
        self.received_signal = False
        self.spiked = False

    def receive(self):
        self.received_signal = True

    def resolve(self):
        """Fire iff a signal arrived this phase. No accumulation to threshold."""
        if self.received_signal:
            self.spiked = True
        return self.spiked

    def clear(self):
        self.received_signal = False
        self.spiked = False

    # Reporting shims so the dashboard can treat every neuron uniformly. A relay
    # has no membrane, so it reads as full charge on the step it fires and zero
    # otherwise -- purely for visualization.
    @property
    def potential(self):
        return self.threshold if self.spiked else 0.0

    @property
    def activation(self):
        return 1.0 if self.spiked else 0.0

    @property
    def refractory_timer(self):
        return 0
