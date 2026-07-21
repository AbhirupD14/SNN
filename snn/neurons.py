"""Neuron types for the conductance-based predictive-inhibition SNN model.

This model replaces the earlier subtractive **hard-wipe** inhibition with
**persistent inhibitory conductance**. An ``ExcitatoryNeuron`` integrates gathered
excitatory charge and its own inhibitory conductance *jointly* in one step, before
any threshold test. Inhibition therefore shunts an otherwise-instant spike within
the same integration boundary, and it *persists* (decaying) across voltage resets.

Five neuron abstractions live here:

* ``ExcitatoryNeuron`` -- conductance-based LIF with a local post-synaptic activity
  trace (a "calcium" trace) that survives voltage reset, plus the one nonnegative
  accumulating-weight learning rule. Used for BOTH roles that learn feedforward
  weights: the competitors (``L2E``, winner-take-all) and, in the ``rg`` topology,
  the noncompetitive ``L1E`` encoders. The learning rule and intrinsic configuration
  are identical; only WTA membership and fan-in differ.
* ``SourceNeuron`` -- an exogenous binary spike source (the retinal ganglion cell in
  the ``rg`` topology). No membrane, no inhibition, no plasticity.
* ``InhibitoryNeuron`` -- a stateless Boolean relay (``L2I_WTA`` and, in the
  ``enew_enabled=True`` comparison topology, the paired ``L1I`` cells). Its output
  is turned into a conductance pulse by the engine; the relay owns no membrane.
* ``SwitchInterneuron`` -- a charged two-branch coincidence interneuron. Bounded
  residual and paired-trace charge are each subthreshold alone and cross threshold
  only together; no global winner id is consulted.
* ``PredictiveInterneuron`` -- a pattern-specific relay paired one-to-one with an
  ``L2E`` competitor. It owns a nonnegative output weight vector onto the sensory
  ``L1E_s`` cells; each output synapse learns **strictly locally** from its own
  presynaptic event, its own target's activity trace, and its own weight.

Numeric scale is human-readable: the shared excitatory threshold is ``1000``.
"""

from __future__ import annotations

import math

import numpy as np

# --- Shared scientific constants -------------------------------------------
E_THRESHOLD = 1000.0
I_THRESHOLD = E_THRESHOLD / 3.0          # ~333.3333, a reported/visual invariant only
E_WEIGHT_CAP = 1000.0                    # shared accumulating-weight cap
INHIBITORY_SIGN = -1                     # display sign for inhibitory edges

DEFAULT_ETA = 0.01
DEFAULT_LEAK = 0.0
DEFAULT_REFRACTORY = 0

# --- Accumulating weight-update modes ----------------------------------------
# The ordinary E/L2E PRODUCTION default is now `linear_bounded` (promoted after a
# 32/32 fresh-seed confirmation): the per-synapse (1 - (w/w_max)^2) multiplier is
# dropped from the default E update, and the E hard cap is retained. `quadratic_bounded`
# is the HISTORICAL E rule, kept as a headless mode; `linear_nonnegative` is a cap-free
# diagnostic only. The C basal rule now MIRRORS this: production default `c_linear_bounded`
# (multiplier dropped, cap retained); `c_quadratic_bounded` is the historical C rule kept
# as a headless mode; `c_linear_nonnegative` is a cap-free diagnostic. None are dashboard controls.
E_UPDATE_MODES = ('quadratic_bounded', 'linear_bounded', 'linear_nonnegative')
C_UPDATE_MODES = ('c_quadratic_bounded', 'c_linear_bounded', 'c_linear_nonnegative')

# --- Conductance / trace defaults (documented, engine may override) ---------
# Membrane: C dV/dt = -g_L (V - E_L) - g_inh (V - E_inh) + I_exc, with C = 1,
# E_L = V_rest = 0, dt = 1. The existing per-step leak maps to a baseline leak
# conductance g_L = -ln(1 - leak_rate) so that with no inhibition and no input the
# update reduces EXACTLY to the historical V <- (1 - leak_rate) * V.
DEFAULT_E_INH = 0.0                      # shunting inhibition (E_inh = V_rest); must be <= V_rest
DEFAULT_ALPHA_INH = 0.6                  # inhibitory-conductance retention per step = exp(-dt/tau_inh)
# Local activity trace: a <- alpha_a * a + (beta_v * depol + beta_s * spike), clipped to [0, a_max].
DEFAULT_ALPHA_A = 0.85                   # activity-trace retention per step
DEFAULT_BETA_V = 0.30                    # trace gain on normalized sub-threshold depolarization
DEFAULT_BETA_S = 1.00                    # trace gain on an actual spike
DEFAULT_A_MAX = 1.0                      # trace bound


def leak_to_conductance(leak_rate: float) -> float:
    """Map a per-step retained-fraction leak to a baseline leak conductance g_L.

    exp(-g_L * dt) == (1 - leak_rate) at dt = 1, so g_L = -ln(1 - leak_rate).
    leak_rate == 0 -> g_L == 0 (a pure integrator; the g_total == 0 branch applies).
    """
    if not 0.0 <= leak_rate < 1.0:
        raise ValueError(f'leak_rate must be in [0, 1), got {leak_rate}')
    return 0.0 if leak_rate == 0.0 else -math.log(1.0 - leak_rate)


class ConductanceLIFNeuron:
    """Shared conductance-based leaky integrate-and-fire membrane -- intrinsic
    excitatory dynamics ONLY (no accumulating afferent vector, no learning rule).

    Two execution contracts share these mechanics:

    * the LEGACY whole-boundary path: the engine (1) gathers charge via
      ``gather_exc`` + inhibition via ``add_inhibition``, (2) calls ``integrate``
      once to advance ``V`` jointly, (3) tests ``can_fire``/``fire``,
      (4) ``update_trace`` (survives reset), (5) ``decay_conductance`` once,
      (6) ``advance_refractory``. This path is byte-preserved for existing presets;
    * the EVENT-RESOLVED path (added for the coincidence topology): gathered charge
      is frozen into ``remaining_excitation`` via ``freeze_drive``; the boundary is
      then resolved in analytic sub-boundary segments with ``advance_segment`` /
      ``crossing_time`` / ``fire(tau)``, and an inhibitory relay may issue a
      zero-latency ``hard_reset``. ``integrate`` is never called on this path.

    With ``C = 1``, ``E_L = V_rest = 0``, and inhibitory reversal ``E_inh`` the
    membrane obeys, over an elapsed segment ``Delta_tau`` under constant drive
    ``I_exc`` (the frozen packet interpreted as a rate over the unit interval):

        g       = g_L + g_inh
        g == 0  -> V <- V + I_exc * Delta_tau                      (pure integrator)
        g  > 0  -> V_inf = (g_L*E_L + g_inh*E_inh + I_exc) / g
                   V <- V_inf + (V - V_inf) * exp(-g * Delta_tau)

    A single ``advance_segment(1.0)`` with the same frozen drive reproduces the
    legacy ``integrate(dt=1.0)`` result exactly (see the Phase 1 equivalence tests),
    so the event loop cannot alter legacy E dynamics.
    """

    def __init__(self, nid, role, *,
                 threshold=E_THRESHOLD,
                 leak_rate=DEFAULT_LEAK, refractory_steps=DEFAULT_REFRACTORY,
                 e_inh=DEFAULT_E_INH, alpha_inh=DEFAULT_ALPHA_INH,
                 alpha_a=DEFAULT_ALPHA_A, beta_v=DEFAULT_BETA_V,
                 beta_s=DEFAULT_BETA_S, a_max=DEFAULT_A_MAX):
        self.id = nid
        self.role = role
        self.type = 'E'

        self.v_rest = 0.0
        self.V = 0.0
        self.threshold = float(threshold)

        self.leak_rate = float(leak_rate)
        self.g_L = leak_to_conductance(self.leak_rate)      # baseline leak conductance
        self.refractory_steps = int(refractory_steps)
        self.refractory_timer = 0

        # --- persistent inhibitory conductance ---
        if e_inh > self.v_rest:
            raise ValueError(f'E_inh must be <= V_rest, got {e_inh}')
        self.e_inh = float(e_inh)
        self.alpha_inh = float(alpha_inh)
        self.g_inh = 0.0

        # --- local post-synaptic activity trace (survives reset) ---
        self.alpha_a = float(alpha_a)
        self.beta_v = float(beta_v)
        self.beta_s = float(beta_s)
        self.a_max = float(a_max)
        self.a = 0.0

        # --- per-boundary gather + bookkeeping ---
        self.pending_exc = 0.0           # boundary-start gathering (legacy + event)
        self.remaining_excitation = 0.0  # frozen drive packet consumed by the event loop
        self.v_pre_reset = 0.0           # MAX depolarized endpoint this boundary (pre reset)
        self.spiked = False
        self.v_pre = 0.0
        self.spike_tau = None            # sub-boundary time of this boundary's spike
        self.fired_this_boundary = False

    # ------------------------------------------------------------------ views
    @property
    def potential(self):
        return self.V

    @property
    def activation(self):
        return self.V / self.threshold if self.threshold else 0.0

    # ----------------------------------------------------------- gather phase
    def gather_exc(self, charge):
        """Accumulate excitatory charge for THIS boundary. Geometry never appears
        here; delivered charge is the raw weighted sum."""
        self.pending_exc += float(charge)

    # Backwards-compatible alias used widely in the engine/tests.
    receive_acc = gather_exc

    def add_inhibition(self, g):
        """Add nonnegative inhibitory conductance for THIS boundary."""
        g = float(g)
        if g < 0:
            raise ValueError('inhibitory conductance increment must be >= 0')
        self.g_inh += g

    def freeze_drive(self):
        """Event path: freeze this boundary's gathered ``pending_exc`` into the
        constant ``remaining_excitation`` drive packet the segment loop consumes."""
        self.remaining_excitation += self.pending_exc
        self.pending_exc = 0.0
        return self.remaining_excitation

    def apply_charge_impulse(self, charge):
        """Apply an excitatory charge packet instantaneously to the membrane.

        This is distinct from :meth:`gather_exc`: gathered excitation is a constant
        current/rate integrated over elapsed boundary time, whereas an impulse is a
        capacitor-charge event at the current analytic ``tau``.  The model uses
        ``C = 1``, hence ``delta_V = charge``.  Firing remains the scheduler/cell's
        responsibility, including refractory and per-boundary gates.
        """
        charge = float(charge)
        if not math.isfinite(charge) or charge < 0.0:
            raise ValueError(f'excitatory charge impulse must be finite and >= 0, got {charge!r}')
        self.V += charge
        if self.V > self.v_pre_reset:
            self.v_pre_reset = self.V
        return self.V

    # --------------------------------------------------------- integration
    def integrate(self, dt=1.0):
        """LEGACY whole-boundary advance: combine leak, persistent inhibition, and
        the gathered excitatory input BEFORE any threshold test. Consumes
        pending_exc.

            g_total = g_L + g_inh
            g_total == 0 -> V <- V + Q_exc / C          (pure integrator, C = 1)
            else         -> I_exc = Q_exc / dt
                            V_inf = (g_L*E_L + g_inh*E_inh + I_exc) / g_total
                            V     <- V_inf + (V - V_inf) * exp(-g_total * dt)

        Records ``v_pre_reset`` (the post-integration membrane) so the local trace
        can be updated correctly even if the neuron then fires and resets.
        """
        Q = self.pending_exc
        self.pending_exc = 0.0
        g_total = self.g_L + self.g_inh
        if g_total == 0.0:
            self.V = self.V + Q                     # C = 1; degenerate no-conductance case
        else:
            i_exc = Q / dt
            v_inf = (self.g_L * self.v_rest + self.g_inh * self.e_inh + i_exc) / g_total
            self.V = v_inf + (self.V - v_inf) * math.exp(-g_total * dt)
        self.v_pre_reset = self.V
        return self.V

    # ------------------------------------------------- exact segment primitives
    def _v_inf(self):
        """Steady-state voltage for the current frozen drive + conductances; None
        when total conductance is zero (the pure-integrator branch)."""
        g = self.g_L + self.g_inh
        if g == 0.0:
            return None
        return (self.g_L * self.v_rest + self.g_inh * self.e_inh
                + self.remaining_excitation) / g

    def advance_segment(self, delta_tau):
        """Advance ``V`` by an elapsed sub-boundary segment ``delta_tau`` under the
        constant frozen drive ``remaining_excitation``, using the exact trajectory.
        Tracks ``v_pre_reset`` as the MAXIMUM depolarized endpoint reached this
        boundary (never cleared by a later hard reset), so the once-per-boundary
        activity-trace update still registers depolarization that was wiped."""
        delta_tau = float(delta_tau)
        if delta_tau < 0.0:
            raise ValueError(f'delta_tau must be >= 0, got {delta_tau}')
        g = self.g_L + self.g_inh
        if g == 0.0:
            self.V = self.V + self.remaining_excitation * delta_tau
        else:
            v_inf = (self.g_L * self.v_rest + self.g_inh * self.e_inh
                     + self.remaining_excitation) / g
            self.V = v_inf + (self.V - v_inf) * math.exp(-g * delta_tau)
        if self.V > self.v_pre_reset:
            self.v_pre_reset = self.V
        return self.V

    def crossing_time(self, remaining_interval):
        """Analytic elapsed time until ``V`` reaches ``threshold`` under the current
        frozen drive, or ``math.inf`` if it cannot within ``remaining_interval``.

        Encodes candidate VALIDITY (a purely local, per-cell computation -- never a
        winner policy): refractory must permit firing and the cell must not have
        already fired this boundary. Returns ``0.0`` when the segment begins at or
        above threshold (an immediate crossing)."""
        if self.refractory_timer != 0 or self.fired_this_boundary:
            return math.inf
        if self.V >= self.threshold:
            return 0.0
        g = self.g_L + self.g_inh
        i_exc = self.remaining_excitation
        if g > 0.0:
            v_inf = (self.g_L * self.v_rest + self.g_inh * self.e_inh + i_exc) / g
            if v_inf <= self.threshold:
                return math.inf                    # trajectory asymptotes below theta
            dtau = (1.0 / g) * math.log((v_inf - self.V) / (v_inf - self.threshold))
        else:
            if i_exc <= 0.0:
                return math.inf                    # no drive, no leak -> never crosses
            dtau = (self.threshold - self.V) / i_exc
        if dtau < 0.0:
            dtau = 0.0
        return dtau if dtau <= float(remaining_interval) else math.inf

    def can_fire(self):
        return self.refractory_timer == 0 and self.V >= self.threshold

    def fire(self, tau=None):
        """Threshold crossing: record v_pre, reset membrane to rest, arm refractory,
        consume the frozen drive packet, and mark this boundary spent. ``tau`` is the
        sub-boundary crossing time (``None`` on the legacy whole-boundary path). Does
        NOT clear ``g_inh`` or the activity trace ``a``."""
        self.v_pre = self.V
        self.spiked = True
        self.V = self.v_rest
        self.refractory_timer = self.refractory_steps
        self.spike_tau = tau
        self.remaining_excitation = 0.0
        self.fired_this_boundary = True
        return self.v_pre

    def hard_reset(self, tau, discard_drive=True):
        """Immediate zero-latency inhibitory reset at sub-boundary time ``tau``:
        wipe retained membrane evidence and (by default) discard the remainder of the
        frozen drive packet. Deliberately does NOT touch learned weights, the local
        trace, persistent inhibitory conductance, refractory state, or an
        already-emitted spike's ``spiked``/``spike_tau``/``v_pre``/``v_pre_reset``."""
        self.V = self.v_rest
        if discard_drive:
            self.remaining_excitation = 0.0
        return self.V

    # ------------------------------------------------------------- trace
    def update_trace(self):
        """Update the local activity trace from THIS cell's own state only.

            depol = clip((v_pre_reset - V_rest) / (threshold - V_rest), 0, 1)
            a <- clip(alpha_a * a + beta_v * depol + beta_s * spike, 0, a_max)

        Uses ``v_pre_reset`` (the max pre-reset depolarization) so a cell that fired
        and reset -- or was hard-reset after depolarizing -- still registers as
        "recently active". Carries no information about WHICH afferent supplied the
        charge.
        """
        span = self.threshold - self.v_rest
        depol = 0.0 if span <= 0 else (self.v_pre_reset - self.v_rest) / span
        depol = min(1.0, max(0.0, depol))
        incr = self.beta_v * depol + self.beta_s * (1.0 if self.spiked else 0.0)
        self.a = self.alpha_a * self.a + incr
        if self.a < 0.0:
            self.a = 0.0
        elif self.a > self.a_max:
            self.a = self.a_max
        return self.a

    def decay_conductance(self):
        """Retain/decay the inhibitory conductance exactly once per boundary."""
        self.g_inh *= self.alpha_inh
        if self.g_inh < 0.0:
            self.g_inh = 0.0

    def advance_refractory(self):
        """Count down the refractory timer once per completed boundary."""
        if self.spiked:
            return                                 # fired this step: the firing step does not count
        if self.refractory_timer > 0:
            self.refractory_timer -= 1

    def begin_event_boundary(self):
        """Event path: reset per-boundary transient membrane bookkeeping at boundary
        start. Persistent state (``g_inh``, activity trace, refractory timer, learned
        weights) is deliberately untouched. Membrane ``V`` persists across boundaries."""
        self.pending_exc = 0.0
        self.remaining_excitation = 0.0
        self.v_pre_reset = self.v_rest
        self.spiked = False
        self.fired_this_boundary = False
        self.spike_tau = None


class ExcitatoryNeuron(ConductanceLIFNeuron):
    """Ordinary conductance LIF unit with the one accumulating-weight learning rule.

    Adds a flat plastic afferent vector (``acc_weights`` + aligned distance factors)
    and the signed accumulating update on top of the shared membrane base. Used for
    both learning roles: the WTA competitors (``L2E``) and the noncompetitive
    ``e_encoder`` cells. ``e_pretrained`` uses this class with empty afferent arrays
    and ``learn=False``.
    """

    def __init__(self, nid, role, *, acc_weights, acc_distance_factor,
                 threshold=E_THRESHOLD, w_max=E_WEIGHT_CAP, w_floor=0.0,
                 leak_rate=DEFAULT_LEAK, refractory_steps=DEFAULT_REFRACTORY,
                 eta=DEFAULT_ETA, learn=True, update_mode='linear_bounded',
                 maturity_budget_frac=1.10,
                 e_inh=DEFAULT_E_INH, alpha_inh=DEFAULT_ALPHA_INH,
                 alpha_a=DEFAULT_ALPHA_A, beta_v=DEFAULT_BETA_V,
                 beta_s=DEFAULT_BETA_S, a_max=DEFAULT_A_MAX):
        super().__init__(nid, role, threshold=threshold, leak_rate=leak_rate,
                         refractory_steps=refractory_steps, e_inh=e_inh,
                         alpha_inh=alpha_inh, alpha_a=alpha_a, beta_v=beta_v,
                         beta_s=beta_s, a_max=a_max)
        self.w_max = float(w_max)
        self.w_floor = float(w_floor)            # lower clip on learned weights (default 0)
        if update_mode not in E_UPDATE_MODES:
            raise ValueError(f'update_mode must be one of {E_UPDATE_MODES}, got {update_mode!r}')
        self.update_mode = update_mode
        # Learning budget target as a multiple of the FIRING threshold. The rule drives
        # sum(acc_weights) toward maturity_budget_frac * threshold from below, so a value
        # > 1.0 lets a matured single-pattern specialist's active afferents overshoot the
        # firing threshold and fire in ONE integration boundary. This is decoupled from the
        # firing threshold itself (unchanged); < 1.0 would make specialists sub-threshold.
        if maturity_budget_frac < 1.0:
            raise ValueError(
                f'maturity_budget_frac must be >= 1.0, got {maturity_budget_frac}')
        self.maturity_budget_frac = float(maturity_budget_frac)
        self.acc_weights = np.asarray(acc_weights, dtype=float)
        self.acc_distance_factor = np.asarray(acc_distance_factor, dtype=float)
        if self.acc_weights.shape != self.acc_distance_factor.shape:
            raise ValueError('acc_weights and acc_distance_factor must align')
        self.eta = float(eta)
        self.learn = bool(learn)
        # --- opt-in learning-event instrumentation (off = byte-identical) ---
        self.record_updates = False
        self.update_log = []

    # -------------------------------------------------------------- learning
    def update_acc_weights(self, participation):
        """The one accumulating-weight rule. Runs when this neuron fires.

        PRODUCTION default is ``linear_bounded``:

            p          = maturity_budget_frac*threshold - sum(acc_weights)  # pre-update, signed
            signal_i   = +1 if afferent i spiked in the causal volley else -1
            base_i     = eta * p * signal_i * distance_factor_i
            delta_i    = base_i                                # linear_bounded (default)
            w_i        = clip(w_i + delta_i, w_floor, w_max)   # E cap retained, floor 0

        ``update_mode`` selects the variant:
          * ``linear_bounded``     -- delta = base; clip [w_floor, w_max]. **PRODUCTION
            default** (promoted after a 32/32 fresh-seed confirmation); no quadratic term;
          * ``quadratic_bounded``  -- delta = base * (1 - (w/w_max)^2); clip [w_floor, w_max].
            The HISTORICAL rule, kept as a headless mode;
          * ``linear_nonnegative`` -- delta = base; floor only (NO upper cap). Cap-free
            diagnostic only.

        ``w_floor`` (default 0) sets the lower clip: at 0 a fully depressed afferent can
        still recover additively if it later participates, but it holds no charge and is
        maximally plastic; a positive floor keeps a residual weight on every afferent.
        """
        if not self.learn:
            return
        w = self.acc_weights
        participation = np.asarray(participation, dtype=bool)
        p = self.maturity_budget_frac * self.threshold - float(w.sum())
        signal = np.where(participation, 1.0, -1.0)
        base = self.eta * p * signal * self.acc_distance_factor
        if self.update_mode == 'quadratic_bounded':
            delta = base * (1.0 - (w / self.w_max) ** 2)
        else:                                        # linear_bounded / linear_nonnegative
            delta = base
        w_before = w.copy() if self.record_updates else None
        if self.update_mode == 'linear_nonnegative':
            np.maximum(w + delta, self.w_floor, out=w)     # floor only; cap-free probe
        else:
            np.clip(w + delta, self.w_floor, self.w_max, out=w)
        if self.record_updates:
            self._log_acc_update(w_before, delta, p)

    def _log_acc_update(self, w_before, raw_delta, fe_pre):
        """Append one learning-event record (opt-in; never affects dynamics)."""
        w_after = self.acc_weights
        pre_sum, post_sum = float(w_before.sum()), float(w_after.sum())
        applied = w_after - w_before
        bounded = self.update_mode != 'linear_nonnegative'
        self.update_log.append(dict(
            cell=self.id, mode=self.update_mode,
            pre_sum=pre_sum, post_sum=post_sum,
            fe_pre=float(fe_pre), fe_post=float(self.threshold - post_sum),
            min_w=float(w_after.min()), max_w=float(w_after.max()),
            n_zero=int((w_after <= self.w_floor + 1e-12).sum()),
            n_at_max=int((w_after >= self.w_max - 1e-9).sum()) if bounded else 0,
            raw_delta_norm=float(np.linalg.norm(raw_delta)),
            applied_delta_norm=float(np.linalg.norm(applied)),
            crossed_theta=bool((pre_sum <= self.threshold < post_sum)
                               or (post_sum <= self.threshold < pre_sum))))


class DendriticCompartment:
    """A minimal named input compartment for a coincidence cell -- NOT a general
    dendritic tree. It owns its ordered source/edge ids and this boundary's transient
    delivery; only a PLASTIC compartment carries an aligned weight + distance vector.
    The cross-compartment coincidence gate belongs to the C cell, not here.

    * ``basal`` -- exactly one source, one weight, one distance factor (plastic);
    * ``apical`` -- one or more sources, no weight vector (structural Boolean gate).
    """

    def __init__(self, name, source_ids, edge_ids, *, weights=None,
                 distance_factors=None):
        if name not in ('basal', 'apical'):
            raise ValueError(f"compartment name must be 'basal' or 'apical', got {name!r}")
        self.name = name
        self.source_ids = list(source_ids)
        self.edge_ids = list(edge_ids)
        if len(self.source_ids) != len(self.edge_ids):
            raise ValueError('source_ids and edge_ids must align')
        self.plastic = weights is not None
        if self.plastic:
            self.weights = np.asarray(weights, dtype=float)
            if distance_factors is None:
                self.distance_factors = np.ones(len(self.source_ids))
            else:
                self.distance_factors = np.asarray(distance_factors, dtype=float)
            if not (self.weights.shape == self.distance_factors.shape
                    == (len(self.source_ids),)):
                raise ValueError('plastic compartment weights/distances must align with sources')
        else:
            self.weights = None
            self.distance_factors = None
        # transient current-boundary delivery (aligned lists)
        self.delivered_sources = []
        self.delivered_signals = []

    def clear(self):
        """Drop this boundary's delivered events (apical state never persists; basal
        persistence is the C cell's one-boundary eligibility, not the compartment's)."""
        self.delivered_sources = []
        self.delivered_signals = []

    def gather(self, source, signal=1.0):
        """Record a delivered event on this compartment for the current boundary."""
        self.delivered_sources.append(source)
        self.delivered_signals.append(float(signal))


class CoincidencePyramidalNeuron(ConductanceLIFNeuron):
    """Excitatory coincidence pyramidal cell: one learned basal afferent and one or
    more unweighted Boolean apical afferents, over the shared conductance-LIF membrane.

    Somatic charge is deposited only when a basal signal (current OR carried one
    boundary) coincides with a current apical event -- basal-only and apical-only
    trains add exactly zero charge, regardless of leak or presentation length.
    Intrinsic dynamics (threshold, leak, reset, refractory, conductance, activity
    trace) are ordinary E dynamics; only the single basal weight learns, and only when
    the cell actually fires.

    ``type = 'E'`` with role ``coincidence``. There is no accumulating flat-feedforward
    afferent vector, no ``update_acc_weights`` fallback, and no apical weight storage.
    """

    def __init__(self, nid, basal_source, basal_edge_id, *, apical_sources=(),
                 apical_edge_ids=(), basal_weight=0.0, basal_distance_factor=1.0,
                 w_max=E_WEIGHT_CAP, eta_c=DEFAULT_ETA, learn=True, use_fe=True,
                 update_mode='c_linear_bounded', maturity_budget_frac=1.10,
                 role='coincidence',
                 threshold=E_THRESHOLD, leak_rate=DEFAULT_LEAK,
                 refractory_steps=DEFAULT_REFRACTORY,
                 e_inh=DEFAULT_E_INH, alpha_inh=DEFAULT_ALPHA_INH,
                 alpha_a=DEFAULT_ALPHA_A, beta_v=DEFAULT_BETA_V,
                 beta_s=DEFAULT_BETA_S, a_max=DEFAULT_A_MAX):
        super().__init__(nid, role, threshold=threshold, leak_rate=leak_rate,
                         refractory_steps=refractory_steps, e_inh=e_inh,
                         alpha_inh=alpha_inh, alpha_a=alpha_a, beta_v=beta_v,
                         beta_s=beta_s, a_max=a_max)
        self.w_max = float(w_max)                     # C-specific basal cap (NOT e_weight_cap)
        self.eta_c = float(eta_c)
        self.learn = bool(learn)
        self.use_fe = bool(use_fe)                    # include the w1-budget fullness-error factor
        if update_mode not in C_UPDATE_MODES:
            raise ValueError(f'update_mode must be one of {C_UPDATE_MODES}, got {update_mode!r}')
        self.update_mode = update_mode
        # Basal-weight learning budget target as a multiple of the leak-corrected
        # full-boundary reference w1 = theta/kappa. C deposits are impulses, whose
        # from-reset threshold is theta, so this retained reference gives conservative
        # one-shot headroom. Firing threshold and the coincidence gate are untouched.
        if maturity_budget_frac < 1.0:
            raise ValueError(
                f'maturity_budget_frac must be >= 1.0, got {maturity_budget_frac}')
        self.maturity_budget_frac = float(maturity_budget_frac)
        self.record_updates = False                   # opt-in instrumentation (off = byte-identical)
        self.update_log = []
        self.basal = DendriticCompartment(
            'basal', [basal_source], [basal_edge_id],
            weights=[float(basal_weight)], distance_factors=[float(basal_distance_factor)])
        self.apical = DendriticCompartment(
            'apical', list(apical_sources), list(apical_edge_ids))

        # --- one-boundary basal eligibility (dendritic state, not membrane charge) ---
        self.basal_received = False
        self.basal_signal = 0.0
        self.basal_eligible = False                   # unconsumed basal event from t-1
        self.basal_eligible_signal = 0.0
        self.apical_sources = set()                   # current-boundary L2E ids
        self.apical_active = False
        self.coincidence_active = False
        self.coincidence_charge = 0.0
        # signal that produced the current deposit (causal state for learning)
        self._deposit_signal = 0.0
        # A deposit is an atomic per-boundary transaction, separate from Boolean gate
        # truth.  Delivery counters keep duplicate routing observable without allowing
        # it to alter membrane state.
        self.deposit_committed_this_boundary = False
        self.coincidence_deposit_count = 0
        self.coincidence_deposit_tau = None
        self.apical_delivery_count = 0
        self.apical_duplicate_count = 0

    # ------------------------------------------------------------------ views
    @property
    def basal_weight(self):
        return float(self.basal.weights[0])

    # ----------------------------------------------------- boundary lifecycle
    def begin_event_boundary(self):
        """Reset current-boundary receipts and transient membrane bookkeeping, but
        PRESERVE the basal eligibility carried from the previous boundary so it can be
        evaluated against this boundary's apical events."""
        super().begin_event_boundary()
        self.basal.clear()
        self.apical.clear()
        self.basal_received = False
        self.basal_signal = 0.0
        self.apical_sources = set()
        self.apical_active = False
        self.coincidence_active = False
        self.coincidence_charge = 0.0
        self._deposit_signal = 0.0
        self.deposit_committed_this_boundary = False
        self.coincidence_deposit_count = 0
        self.coincidence_deposit_tau = None
        self.apical_delivery_count = 0
        self.apical_duplicate_count = 0
        # basal_eligible / basal_eligible_signal are intentionally NOT cleared here.

    # -------------------------------------------------------- input delivery
    def gather_basal(self, source, signal=1.0):
        """Deliver a basal event this boundary. Does not depolarize the soma; it only
        makes basal signal available to the coincidence gate."""
        self.basal.gather(source, signal)

    def gather_apical(self, source):
        """Deliver an unweighted apical (Boolean permission) event this boundary."""
        self.apical.gather(source, 1.0)
        self.apical_delivery_count += 1
        if source in self.apical_sources:
            self.apical_duplicate_count += 1
        else:
            self.apical_sources.add(source)
        self.apical_active = bool(self.apical_sources)

    def deliver_basal(self, source, signal=1.0, tau=0.0):
        """Deliver basal input at ``tau`` and atomically try the coincidence gate."""
        self.gather_basal(source, signal)
        return self.resolve_dendrites(tau=tau)

    def deliver_apical(self, source, tau):
        """Deliver apical permission at ``tau`` and atomically try the C gate."""
        self.gather_apical(source)
        return self.resolve_dendrites(tau=tau)

    # ------------------------------------------------- dendritic gating rule
    def resolve_dendrites(self, tau=0.0):
        """Implement the coincidence truth table and one-boundary eligibility state
        machine, then commit any gated somatic charge impulse. Receipts processed at
        the same ``tau`` are order invariant; when arrival times differ, the second
        required receipt commits at its own (later) ``tau``.

        Deposits ``w_basal * s`` at most once iff (current OR carried basal) AND a
        current apical event. The impulse changes ``V`` immediately, including during
        refractory; firing is gated separately and the same basal event is never reused.
        """
        tau = float(tau)
        if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
            raise ValueError(f'coincidence deposit tau must be finite and in [0, 1], got {tau!r}')
        basal_received = len(self.basal.delivered_sources) > 0
        basal_signal = self.basal.delivered_signals[0] if basal_received else 0.0
        # ``gather_apical`` maintains raw/unique delivery diagnostics. Deriving the set
        # here as well keeps direct compartment-based diagnostic use deterministic.
        apical_sources = set(self.apical.delivered_sources)
        if apical_sources != self.apical_sources:
            self.apical_sources = apical_sources
        apical_active = bool(self.apical_sources)

        self.basal_received = basal_received
        self.basal_signal = basal_signal
        self.apical_active = apical_active

        # B = current OR carried basal availability; its signal prefers the current event.
        B = basal_received or self.basal_eligible
        b_signal = basal_signal if basal_received else self.basal_eligible_signal
        A = apical_active

        committed_charge = 0.0
        coincidence = bool(B and A)
        if coincidence and not self.deposit_committed_this_boundary:
            w = float(self.basal.weights[0])
            committed_charge = w * b_signal
            self._deposit_signal = b_signal
            self.apply_charge_impulse(committed_charge)
            self.deposit_committed_this_boundary = True
            self.coincidence_deposit_count = 1
            self.coincidence_deposit_tau = tau
            self.coincidence_charge = committed_charge
            # consume ALL basal availability; a participating current event is not carried.
            self.basal_eligible = False
            self.basal_eligible_signal = 0.0
        elif not self.deposit_committed_this_boundary:
            # carry a current, unconsumed basal event for EXACTLY the next boundary;
            # otherwise eligibility expires (no two-boundary survival).
            if basal_received:
                self.basal_eligible = True
                self.basal_eligible_signal = basal_signal
            else:
                self.basal_eligible = False
                self.basal_eligible_signal = 0.0

        # Once opened, the causal gate stays observable for the rest of this boundary;
        # later apical receipts cannot erase the committed deposit or its diagnostics.
        self.coincidence_active = self.coincidence_active or coincidence
        return committed_charge

    # ----------------------------------------------------------- firing gate
    def can_fire(self):
        """A C cell may fire only with refractory clear, a supra-threshold membrane,
        an ACTIVE current coincidence gate, and no prior spike this boundary."""
        return (self.refractory_timer == 0 and self.V >= self.threshold
                and self.coincidence_active and not self.fired_this_boundary)

    def crossing_time(self, remaining_interval):
        """C-local crossing time: infinity whenever the coincidence gate is closed, so
        a retained supra-threshold membrane cannot fire on a non-coincident boundary."""
        if not self.coincidence_active:
            return math.inf
        return super().crossing_time(remaining_interval)

    # -------------------------------------------------------------- learning
    def update_basal_weight(self):
        """The exact basal update, applied ONLY immediately after this cell fires and
        using the causal firing-boundary dendritic state and the PRE-update weight:

            kappa = 1                if g_L == 0 else (1 - e^{-g_L}) / g_L
            w1    = theta / kappa                 (leak-corrected budget reference)
            FE    = maturity_budget_frac * w1 - w (budget headroom; dropped if use_fe=False)
            dw    = eta_C * FE * A * (1 - (w / w_max)^2) * s * phi
            w     <- clip(w + dw, 0, w_max)

        The FE budget target retains the leak-corrected full-boundary reference ``w1``
        for developmental compatibility. Since a C coincidence is an instantaneous
        charge impulse, its actual from-reset firing condition is ``w*s >= theta``;
        the ``w1`` target therefore supplies conservative headroom. The deposit gate
        (basal AND apical) and firing threshold are unchanged.

        A is the Boolean apical gate (1 at a causal C spike), s the active basal signal
        of the causal deposit, phi the basal distance influence. There is no apical
        weight and no negative-participation update.

        ``update_mode`` selects the bounding variant (production default ``c_linear_bounded``):
          * ``c_linear_bounded``     -- dw = base (no multiplier); clip [0, w_max]. **DEFAULT**;
          * ``c_quadratic_bounded``  -- dw = base * (1 - (w/w_max)^2); clip [0, w_max]. Historical;
          * ``c_linear_nonnegative`` -- dw = base; floor 0 only, NO cap (diagnostic probe).

        ``use_fe=False`` drops the budget factor, leaving the soft bound to
        (1 - (w/w_max)^2) alone. FE is deliberately preserved by the ablation modes."""
        if not self.learn:
            return self.basal_weight
        w = float(self.basal.weights[0])
        A = 1.0 if self.apical_active else 0.0
        s = self._deposit_signal
        phi = float(self.basal.distance_factors[0])
        kappa = 1.0 if self.g_L == 0.0 else (1.0 - math.exp(-self.g_L)) / self.g_L
        w1 = self.threshold / kappa
        budget_fe = self.maturity_budget_frac * w1 - w
        fe = budget_fe if self.use_fe else 1.0
        base = self.eta_c * fe * A * s * phi
        if self.update_mode == 'c_quadratic_bounded':
            dw = base * (1.0 - (w / self.w_max) ** 2)
        else:                                        # c_linear_bounded / c_linear_nonnegative
            dw = base
        if self.update_mode == 'c_linear_nonnegative':
            w_new = max(0.0, w + dw)                  # floor only; cap-free diagnostic
        else:
            w_new = min(self.w_max, max(0.0, w + dw))
        self.basal.weights[0] = w_new
        if self.record_updates:
            self.update_log.append(dict(
                cell=self.id, mode=self.update_mode, w_pre=w, w_post=w_new,
                fe_pre=float(budget_fe), raw_dw=float(dw),
                applied_dw=float(w_new - w), at_cap=bool(
                    self.update_mode != 'c_linear_nonnegative' and w_new >= self.w_max - 1e-9)))
        return w_new


class SourceNeuron:
    """An exogenous binary spike source -- the retinal ganglion (RG) cell.

    Deliberately NOT an ``ExcitatoryNeuron``: it owns no membrane, no inhibitory
    conductance, no refractory state, and no learning rule, so there is nothing for
    cortical feedback to act on. Its spike is a pure function of the external input::

        spiked(t) = input_arrives(t) AND input_vec[pixel] > 0.5

    The engine sets that Boolean via ``present`` each boundary. The plastic weight on
    the RG -> L1E path is owned by the *postsynaptic* encoder, not by this cell, so RG
    supplies the presynaptic event and nothing else.

    "Uninhibited" here means uninhibited *by the modelled cortical feedback loop*
    (L1I / L2I / PI). It is not a claim that a biological retina lacks inhibitory
    circuitry -- retinal amacrine/horizontal inhibition is simply outside this model.
    """

    def __init__(self, nid, role='rg_source', *, threshold=E_THRESHOLD):
        self.id = nid
        self.role = role
        self.type = 'S'
        self.threshold = float(threshold)      # reported/visual invariant only
        self.spiked = False

    def present(self, active: bool):
        """Set this boundary's exogenous spike. The only way an RG cell ever fires."""
        self.spiked = bool(active)
        return self.spiked

    def clear(self):
        self.spiked = False

    @property
    def potential(self):
        return self.threshold if self.spiked else 0.0

    @property
    def activation(self):
        return 1.0 if self.spiked else 0.0

    @property
    def refractory_timer(self):
        return 0


class InhibitoryNeuron:
    """A stateless instant relay (``L2I_WTA`` and the comparison-topology ``L1I``).

    Any +1 input in the current causal sub-phase makes it fire immediately. It owns
    no membrane and no plasticity; the engine converts its firing into a persistent
    inhibitory *conductance* pulse on the target excitatory cells. The one-third
    threshold is a reported invariant only.
    """

    def __init__(self, nid, role, *, threshold=I_THRESHOLD):
        self.id = nid
        self.role = role
        self.type = 'I'
        self.threshold = float(threshold)          # reported only; there is no integrator
        self.received_signal = False
        self.spiked = False
        self.spike_tau = None                      # sub-boundary tau (event path); inherits driver's

    def receive(self):
        self.received_signal = True

    def resolve(self):
        if self.received_signal:
            self.spiked = True
        return self.spiked

    def clear(self):
        self.received_signal = False
        self.spiked = False
        self.spike_tau = None

    @property
    def potential(self):
        return self.threshold if self.spiked else 0.0

    @property
    def activation(self):
        return 1.0 if self.spiked else 0.0

    @property
    def refractory_timer(self):
        return 0


class SwitchInterneuron:
    """Charged two-branch local incumbent-release coincidence detector.

    Each switch is paired with exactly one L2 competitor. ``x`` is a scalar local
    eligibility trace written by a real spike from that competitor. ErrorE events add
    numeric ``residual_charge``. A sufficiently mature ``x`` opens a second local
    priming branch, ``trace_charge``. Both branch charges are individually capped below
    threshold, so repeated activity on either branch alone can never fire the cell;
    their sum is the visible membrane/coincidence potential.

    The engine resolves residual charge against the trace carried into the boundary,
    then records the current competitor spike for a future boundary. Consequently a
    new winner cannot combine with same-boundary residual activity and inhibit itself.

    A successful coincidence consumes the old trace, making the gate one-shot until
    another paired L2 spike primes it. This rejects either branch alone and avoids a
    sticky global winner variable while retaining a tunable local memory window.
    """

    def __init__(self, nid, role='switch', *, trace_decay=0.97,
                 trace_threshold=0.5, residual_charge_frac=0.55,
                 trace_charge_frac=0.90, branch_cap_frac=0.90,
                 threshold=I_THRESHOLD):
        self.id = nid
        self.role = role
        self.type = 'I'
        self.threshold = float(threshold)          # reported invariant only
        self.trace_decay = float(np.clip(trace_decay, 0.0, 1.0))
        self.trace_threshold = float(np.clip(trace_threshold, 0.0, 1.0))
        self.residual_charge_frac = max(0.0, float(residual_charge_frac))
        self.trace_charge_frac = max(0.0, float(trace_charge_frac))
        # Structural safety invariant: neither branch can reach threshold alone,
        # even if a programmatic caller bypasses the dashboard's bounded controls.
        safe_cap_frac = float(np.clip(branch_cap_frac, 0.0, 1.0 - 1e-9))
        self.branch_cap = safe_cap_frac * self.threshold
        self.x = 0.0
        self.residual_charge = 0.0
        self.trace_charge = 0.0
        self.V = 0.0
        self.v_pre_reset = 0.0
        self.residual_events = 0
        self.received_residual = False
        self.spiked = False

    def _refresh_potential(self):
        """Combine bounded branch charge without weakening the strict AND.

        The biochemical eligibility trace becomes electrical priming only after its
        own threshold. Below that threshold the residual branch remains visible but
        cannot be pushed supra-threshold by an ineligible trace.
        """
        if self.x >= self.trace_threshold:
            self.trace_charge = min(self.branch_cap,
                                    self.x * self.trace_charge_frac * self.threshold)
        else:
            self.trace_charge = 0.0
        self.V = self.residual_charge + self.trace_charge

    def begin_boundary(self):
        """Decay local eligibility and clear this boundary's transient state."""
        self.x *= self.trace_decay
        self.residual_charge = 0.0                    # current-boundary residual branch
        self.residual_events = 0
        self.received_residual = False
        self.spiked = False
        self.v_pre_reset = 0.0
        self._refresh_potential()

    def receive_residual(self):
        """Add one explicit ErrorE event as bounded numeric branch charge."""
        self.received_residual = True
        self.residual_events += 1
        dq = self.residual_charge_frac * self.threshold
        self.residual_charge = min(self.branch_cap, self.residual_charge + dq)
        self._refresh_potential()

    def resolve_residual(self):
        """Evaluate strict temporal AND against the pre-existing local trace."""
        if (self.received_residual and self.x >= self.trace_threshold
                and self.V >= self.threshold):
            self.spiked = True
            self.v_pre_reset = self.V
            self.x = 0.0                              # consume: one release per trace
            self.residual_charge = 0.0
            self.trace_charge = 0.0
            self.V = 0.0
        return self.spiked

    def prime(self):
        """Record an actual spike from this switch's paired L2 competitor."""
        self.x = 1.0
        # Do not refresh V here: current L2 spikes are recorded only for a future
        # boundary and must not combine electrically with same-boundary residuals.

    def clear(self):
        """Compatibility with relay cleanup; the engine calls begin_boundary()."""
        self.begin_boundary()

    @property
    def potential(self):
        return self.threshold if self.spiked else self.V

    @property
    def activation(self):
        return self.potential / self.threshold if self.threshold else 0.0

    @property
    def refractory_timer(self):
        return 0


class PredictiveInterneuron:
    """Pattern-specific predictive inhibitory relay, paired one-to-one with one L2E.

    It relays its paired competitor's win (a Boolean event), and owns a nonnegative
    output weight vector ``w[i]`` -- one candidate inhibitory synapse onto each
    sensory ``L1E_s[i]``. Each output synapse is **strictly local**:

    * potentiation uses only its own presynaptic event (``pre = 1`` on a real PI
      spike), its own target's local activity trace ``a_i``, and its own weight;
    * the inhibitory pulse it emits uses the **pre-update** weight;
    * a slow passive decay uses only the weight itself.

    The learning update is written element-wise: entry ``i`` reads and writes only
    ``w[i]`` and ``a_i``. It never sees another target's trace, the full L1 spike
    vector, a pattern label, or a central winner-row. The 8x9 array is a display
    convenience, not a shared computation.
    """

    def __init__(self, nid, role, n_targets, *, w_init=0.0, w_max=1.0,
                 eta=0.02, lt_decay=0.0, g_scale=6.0, threshold=I_THRESHOLD):
        self.id = nid
        self.role = role
        self.type = 'I'
        self.threshold = float(threshold)          # reported invariant only
        self.n_targets = int(n_targets)
        self.w = np.full(self.n_targets, float(w_init), dtype=float)
        self.w_max = float(w_max)
        self.eta = float(eta)
        self.lt_decay = float(lt_decay)
        self.g_scale = float(g_scale)
        self.received_signal = False
        self.spiked = False

    # ------------------------------------------------------------ relay events
    def receive(self):
        self.received_signal = True

    def resolve(self):
        if self.received_signal:
            self.spiked = True
        return self.spiked

    def clear(self):
        self.received_signal = False
        self.spiked = False

    # ------------------------------------------------------- inhibitory output
    def conductance_pulse(self):
        """Per-target inhibitory conductance increment from the PRE-update weights:
        ``g_scale * w``. Returns one value per L1E_s target (an array)."""
        return self.g_scale * self.w

    # ------------------------------------------------------- local plasticity
    def learn(self, target_traces):
        """Local potentiation on a real presynaptic (PI) event.

        ``target_traces[i]`` is the local activity trace of ``L1E_s[i]`` -- the only
        post-synaptic information synapse ``i`` uses. The update is element-wise:

            w[i] <- clip(w[i] + eta * a_i * (w_max - w[i]), 0, w_max)

        so synapse ``i`` touches only ``w[i]`` and ``a_i``. Uses the pre-update
        weight to compute the increment; it does not retroactively mature a first
        observation into a strong prediction.
        """
        a = np.asarray(target_traces, dtype=float)
        if a.shape != self.w.shape:
            raise ValueError('target_traces must align one-to-one with PI output synapses')
        self.w += self.eta * a * (self.w_max - self.w)
        np.clip(self.w, 0.0, self.w_max, out=self.w)

    def passive_decay(self):
        """Slow long-term decay for recovery from stale associations. Local: each
        weight decays based only on itself. Applied once per boundary."""
        if self.lt_decay:
            self.w *= (1.0 - self.lt_decay)

    # ------------------------------------------------------------------ views
    @property
    def potential(self):
        return self.threshold if self.spiked else 0.0

    @property
    def activation(self):
        return 1.0 if self.spiked else 0.0

    @property
    def refractory_timer(self):
        return 0
