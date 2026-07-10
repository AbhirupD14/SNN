"""Inhibitory-gate learning rules (REFACTOR_PLAN.md, Phase 3b).

The three per-discharge gate-magnitude rules from `Neuron.apply_inhibition`, moved
verbatim into strategy objects and selected by `select_inhibitory_rule`:

    inhibitory_delta_rule = False           -> SaturatingInhibition (converges to
                                               sqrt(w_max); uniform gates)
    delta_rule=True, mode="turnover"        -> DeltaTurnover (event-local
                                               strengthen minus size-proportional
                                               turnover; gates differentiate)
    delta_rule=True, mode="margin"          -> DeltaMargin (relax toward a fixed
                                               post-inhibition target; diagnostic)

Each returns the NEW gate magnitude w_new from local state only (this synapse's w,
the target's v_pre/theta, the arriving spike). The orchestration -- iterating
active negative gates, the flow-vs-one-shot discharge, the event records, and the
loser-depression hook -- stays in `apply_inhibition`. Bodies are byte-for-byte the
originals.
"""


class InhibitoryRule:
    def new_magnitude(self, n, w, v_pre, w_max, theta, p):
        """Return the new gate magnitude (>= 0) after this discharge event."""
        raise NotImplementedError


class SaturatingInhibition(InhibitoryRule):
    """Legacy: dw = eta * p * (1 - w^2/w_max); clip to [0, w_max]. Every gate
    converges to sqrt(w_max), so gates end up uniform. No-op if w_max <= 0."""

    def new_magnitude(self, n, w, v_pre, w_max, theta, p):
        if w_max > 0:
            dw = n.inhibitory_learning_rate * p * (1.0 - (w * w) / w_max)
            return min(max(w + dw, 0.0), w_max)
        return w


class DeltaTurnover(InhibitoryRule):
    """Default differentiating rule on the normalized gate u = w/G, G = sqrt(w_max):
    du = eta_up * p_t * (1 - u) - eta_down * u, p_t = clamp(v_pre/theta, 0, p_max).
    High-charge rivals strengthen; weak/dead targets drift down via turnover."""

    def new_magnitude(self, n, w, v_pre, w_max, theta, p):
        G = w_max ** 0.5 if w_max > 0 else 0.0
        u = w / G if G > 0 else 0.0
        p_t = min(max(v_pre / theta, 0.0), n.inhibitory_p_max) if theta > 0 else 0.0
        du = n.inhibitory_eta_up * p_t * (1.0 - u) - n.inhibitory_eta_down * u
        dw = du * G
        return min(max(w + dw, 0.0), G)


class DeltaMargin(InhibitoryRule):
    """Diagnostic: relax the gate toward s = clip(v_pre - margin_frac*theta, 0, G),
    the magnitude that would bring the target to a fixed post-inhibition level."""

    def new_magnitude(self, n, w, v_pre, w_max, theta, p):
        G = w_max ** 0.5 if w_max > 0 else 0.0
        target_post = n.inhibitory_margin_frac * theta
        s = min(max(v_pre - target_post, 0.0), G)
        dw = n.inhibitory_delta_eta * (s - w)
        return min(max(w + dw, 0.0), G)


_SATURATING = SaturatingInhibition()
_TURNOVER = DeltaTurnover()
_MARGIN = DeltaMargin()


def select_inhibitory_rule(n):
    """Map the neuron's current flags to the active inhibitory-gate rule."""
    if n.inhibitory_delta_rule:
        if n.inhibitory_rule_mode == "margin":
            return _MARGIN
        return _TURNOVER
    return _SATURATING
