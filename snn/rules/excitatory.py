"""Excitatory local learning rules.

The three mutually-exclusive branches of the old `Neuron._update_weights`, moved
verbatim into strategy objects and selected by `select_excitatory_rule`:

    signed_spike_learning -> SignedSpikeRule   (+ optional structural-FE gate)
    assembly_flow_credit  -> AssemblyFlowCredit
    otherwise             -> ChargeBasedRule    (charge potentiation, optional
                             confidence consolidation, optional signed OFF
                             depression, then the budget/cap tail)

Each rule fires on the postsynaptic neuron's OWN spike, mutating only positive
feedforward synapses; negative inhibitory gates learn via apply_inhibition. Bodies
are byte-for-byte the originals -- they call back into the neuron's helpers
(`_structural_free_energy_gate`, `_maturity`, `_apply_budget_and_cap`), which stay
on Neuron. The mutual exclusivity (signed/assembly return early; charge composes
with depression + budget) is now explicit in the selector instead of buried in
early returns.
"""

import numpy as np


def _closeness(n, v_pre):
    """Excitatory closeness p = clamp(theta / v_pre, 0, 1) (0 if v_pre <= 0)."""
    theta = n.threshold
    return min(max(theta / v_pre, 0.0), 1.0) if v_pre > 0 else 0.0


def bounded_signed_update(w, w_min, w_cap, gain, signal):
    """Shared direction-aware bounded weight kernel.

    For a positive weight w with lower/upper bounds w_min, w_cap, let
        q = clamp((w - w_min) / (w_cap - w_min), 0, 1).
    Upward movement uses H_up(q) = 1 - q^2 (largest at w_min, zero at w_cap);
    downward movement uses the REFLECTED H_down(q) = 1 - (1 - q)^2 (zero at w_min,
    maximal at w_cap). The reflection is required: a literal negative copy of the
    upward form (1 - (w/w_cap)^2) becomes zero at w_cap and would make a capped
    losing weight impossible to depress. Then
        dw = +gain * H_up(q)   where signal >= 0,
        dw = -gain * H_down(q) where signal <  0,
    and w_next = clip(w + dw, w_min, w_cap).

    `signal` is +1 (up) / -1 (down) per element (or a scalar); `gain` is a scalar
    or per-element array. Vectorized so the winner rule (mixed +1/-1 afferents) and
    competitive depression (all -1) share one implementation. `w_cap <= w_min` is a
    no-op (returns w unchanged) to avoid a divide-by-zero on the degenerate range.
    """
    w = np.asarray(w, dtype=float)
    if w_cap <= w_min:
        return w.copy()
    q = np.clip((w - w_min) / (w_cap - w_min), 0.0, 1.0)
    signal = np.asarray(signal, dtype=float)
    H_up = 1.0 - q ** 2
    H_down = 1.0 - (1.0 - q) ** 2
    dw = np.where(signal >= 0.0, gain * H_up, -gain * H_down)
    return np.clip(w + dw, w_min, w_cap)


class ExcitatoryRule:
    def on_fire(self, n, v_pre):
        raise NotImplementedError


class SignedSpikeRule(ExcitatoryRule):
    """Every positive synapse gets a +/-1 signed update (active potentiates, OFF
    depresses) via dw = eta * p * (1 - (w/w_cap)^2) * signal, no budget. When the
    structural free-energy gate is on it REPLACES p with the maturity brake."""

    def on_fire(self, n, v_pre):
        p = _closeness(n, v_pre)
        participating = n._last_input_spikes > 0.5
        pos = n._weights_array > 0
        if pos.any() and n.weight_cap > 0:
            w = n._weights_array[pos]
            signal = np.where(participating[pos], 1.0, -1.0)
            gain = (n.learning_rate * n._structural_free_energy_gate()
                    if n.structural_free_energy
                    else n.learning_rate * p)
            w_min = n.min_positive_weight if n.min_positive_weight is not None else 0.0
            # Shared bounded kernel: +1 afferents potentiate via H_up, OFF afferents
            # depress via the reflected H_down -- the SAME downward branch the
            # competitive-depression loser update uses (spec Section 6).
            n._weights_array[pos] = bounded_signed_update(
                w, w_min, n.weight_cap, gain, signal)


class AssemblyFlowCredit(ExcitatoryRule):
    """Flow-proportional credit for the E->I integrators: contributors potentiate
    toward the cap scaled by their flow share (normalized by MAX flow so the
    dominant driver gets the full rate); non-contributors decay toward the floor."""

    def on_fire(self, n, v_pre):
        p = _closeness(n, v_pre)
        pos = n._weights_array > 0
        w_max = n.excitatory_saturation_cap if n.excitatory_saturation_cap is not None else n.weight_cap
        if pos.any() and w_max > 0 and n._trace is not None:
            w = n._weights_array[pos]
            flow = n._trace[pos]
            fmax = float(flow.max())
            w_min = n.min_positive_weight if n.min_positive_weight is not None else 0.0
            if fmax > 0.0:
                fhat = flow / fmax                      # dominant driver -> 1.0
                contributed = flow > 0.0
                dw = np.where(
                    contributed,
                    n.learning_rate * p * fhat * (1.0 - (w * w) / w_max),
                    -n.learning_rate * p * n.assembly_decay_frac * (w - w_min),
                )
                n._weights_array[pos] = np.clip(w + dw, w_min, n.weight_cap)


class ChargeBasedRule(ExcitatoryRule):
    """Default when neither signed nor assembly is set: charge-based potentiation
    of active gates (optionally confidence-gated, maturing confidence), optional
    signed OFF-gate depression, then the shared budget/cap tail."""

    def on_fire(self, n, v_pre):
        p = _closeness(n, v_pre)
        participating = n._last_input_spikes > 0.5
        active = np.nonzero((n._weights_array > 0) & participating)[0]
        w_max = n.excitatory_saturation_cap if n.excitatory_saturation_cap is not None else n.weight_cap
        if w_max > 0 and active.size > 0:
            w = n._weights_array[active]
            if n.confidence_consolidation:
                # Confidence-gated potentiation: mature gates learn less (floor eta_min).
                C = n._confidence[active]
                eta = n.learning_rate * (n.eta_min + (1.0 - n.eta_min) * (1.0 - C))
                n._weights_array[active] = w + eta * p * (1.0 - (w * w) / w_max)
                # Mature confidence toward the (pre-update) gate's local maturity.
                n._confidence[active] = C + n.conf_beta * (n._maturity(w) - C)
            else:
                dw = n.learning_rate * p * (1.0 - (w * w) / w_max)
                n._weights_array[active] = w + dw
        # Signed OFF-gate depression ("4a"): push down positive gates whose input
        # did not participate this fire; confidence-protected when consolidation is on.
        if n.signed_depression and n.eta_off > 0.0:
            inactive = np.nonzero((n._weights_array > 0) & ~participating)[0]
            if inactive.size > 0:
                w_off = n._weights_array[inactive]
                w_min = n.min_positive_weight if n.min_positive_weight is not None else 0.0
                gate = (1.0 - n._confidence[inactive]) if n.confidence_consolidation else 1.0
                n._weights_array[inactive] = w_off - n.eta_off * p * gate * (w_off - w_min)
                n.signed_depression_events += 1
        n._apply_budget_and_cap()


# Stateless singletons -- shared across all neurons (they operate on the arg).
_SIGNED_SPIKE = SignedSpikeRule()
_ASSEMBLY_FLOW = AssemblyFlowCredit()
_CHARGE_BASED = ChargeBasedRule()


def select_excitatory_rule(n):
    """Map the neuron's current flags to the one active excitatory rule. Encodes
    the old mutual-exclusivity (signed/assembly took precedence and returned early;
    charge is the default) explicitly instead of as inline early returns."""
    if n.signed_spike_learning:
        return _SIGNED_SPIKE
    if n.assembly_flow_credit:
        return _ASSEMBLY_FLOW
    return _CHARGE_BASED
