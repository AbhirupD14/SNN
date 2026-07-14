"""Charge-delivery modes used by neurons.

The two branches of `Neuron.receive_input` -- how a volley's charge reaches the
membrane -- become strategy objects selected by `select_delivery`:

    excitatory_flow_rate and t is not None -> FlowRateDelivery (open a decaying
        current trace, integrate into V over time)
    otherwise                              -> InstantaneousDelivery (V += dot(w, s))

Distance attenuation is a shared pre-step (`effective_weights`): it scales delivered
amplitude in both modes and never changes the stored weight. Learning rules remain
unchanged. Bodies operate on the neuron passed in (its Membrane, exc_trace,
eligibility trace, participation mask).
"""

import numpy as np


def effective_weights(n):
    """Return delivery-effective weights. With distance weighting enabled, each
    afferent is multiplied by

        (distance_ref / max(distance, distance_min)) ** distance_power.

    The returned array may be scaled, but the neuron's stored weights are untouched.
    OFF returns the stored array itself so the baseline path stays allocation-free.
    """
    if n.distance_weighting and n._distance is not None:
        factor = (n.distance_ref / np.maximum(n._distance, n.distance_min)) ** n.distance_power
        return n._weights_array * factor
    return n._weights_array


class DeliveryMode:
    def deliver(self, n, input_spikes, charge_scale, t, w_eff):
        raise NotImplementedError


class InstantaneousDelivery(DeliveryMode):
    """V += dot(w_eff, spikes) * charge_scale, bounded at v_sat; refresh the
    eligibility trace and the binary participation mask."""

    def deliver(self, n, input_spikes, charge_scale, t, w_eff):
        input_current = np.dot(w_eff, input_spikes) * charge_scale
        # Saturating membrane: bound accumulated charge at a finite ceiling.
        n._membrane.deposit(input_current)
        # ARCHIVED trace bookkeeping (kept so inspectors of ._trace keep working).
        n._trace += input_spikes * charge_scale
        # Instantaneous participation signal (binary, unscaled) for the rules.
        n._last_input_spikes = input_spikes


class FlowRateDelivery(DeliveryMode):
    """A spike opens a decaying excitatory current trace integrated into V over
    time. Only a REAL input volley refreshes the participation mask, so a neuron
    that crosses threshold on a later no-input step still learns the driving volley."""

    def deliver(self, n, input_spikes, charge_scale, t, w_eff):
        d = n.exc_trace_decay
        # 1. Advance residual current through the gap timesteps up to t-1.
        n.advance_trace(t - 1)
        n._dbg_v_after_advance = n.potential   # phase diagnostic (see step())
        # 2. Inject new excitatory (positive-weight) drive as current.
        drive = float(np.dot(np.maximum(w_eff, 0.0), input_spikes)) * charge_scale
        n.exc_trace += drive * (1.0 - d) if n.exc_trace_normalized else drive
        # 3. Same-timestep contribution: one integration step at t.
        n._membrane.deposit(n.exc_trace)
        n.exc_trace *= d
        n.exc_trace_last_t = t
        if input_spikes.any():
            n._trace += input_spikes * charge_scale
            n._last_input_spikes = input_spikes


_INSTANTANEOUS = InstantaneousDelivery()
_FLOW_RATE = FlowRateDelivery()


def select_delivery(n, t):
    """Flow-rate when enabled AND a timestep is supplied; else the instantaneous
    baseline (matches the original `if excitatory_flow_rate and t is not None`)."""
    if n.excitatory_flow_rate and t is not None:
        return _FLOW_RATE
    return _INSTANTANEOUS
