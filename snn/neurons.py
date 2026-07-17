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


class ExcitatoryNeuron:
    """Conductance-based leaky integrate-and-fire unit with one learning rule.

    Per integration boundary the engine (1) gathers excitatory charge via
    ``gather_exc`` and inhibitory conductance via ``add_inhibition``, (2) calls
    ``integrate`` once to advance ``V`` jointly, (3) tests ``can_fire``/``fire``,
    (4) calls ``update_trace`` (survives reset), (5) calls ``decay_conductance``
    exactly once, (6) counts down refractory via ``advance_refractory``.
    """

    def __init__(self, nid, role, *, acc_weights, acc_distance_factor,
                 threshold=E_THRESHOLD, w_max=E_WEIGHT_CAP,
                 leak_rate=DEFAULT_LEAK, refractory_steps=DEFAULT_REFRACTORY,
                 eta=DEFAULT_ETA, learn=True,
                 e_inh=DEFAULT_E_INH, alpha_inh=DEFAULT_ALPHA_INH,
                 alpha_a=DEFAULT_ALPHA_A, beta_v=DEFAULT_BETA_V,
                 beta_s=DEFAULT_BETA_S, a_max=DEFAULT_A_MAX):
        self.id = nid
        self.role = role
        self.type = 'E'

        self.v_rest = 0.0
        self.V = 0.0
        self.threshold = float(threshold)
        self.w_max = float(w_max)

        self.leak_rate = float(leak_rate)
        self.g_L = leak_to_conductance(self.leak_rate)      # baseline leak conductance
        self.refractory_steps = int(refractory_steps)
        self.refractory_timer = 0

        self.acc_weights = np.asarray(acc_weights, dtype=float)
        self.acc_distance_factor = np.asarray(acc_distance_factor, dtype=float)
        if self.acc_weights.shape != self.acc_distance_factor.shape:
            raise ValueError('acc_weights and acc_distance_factor must align')
        self.eta = float(eta)
        self.learn = bool(learn)

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
        self.pending_exc = 0.0
        self.v_pre_reset = 0.0            # membrane after integration, before any reset
        self.spiked = False
        self.v_pre = 0.0

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

    # --------------------------------------------------------- integration
    def integrate(self, dt=1.0):
        """Advance ``V`` once, combining leak, persistent inhibition, and the
        gathered excitatory input BEFORE any threshold test. Consumes pending_exc.

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

    def can_fire(self):
        return self.refractory_timer == 0 and self.V >= self.threshold

    def fire(self):
        """Threshold crossing: record v_pre, reset membrane to rest, arm refractory.
        Does NOT clear ``g_inh`` or the activity trace ``a``."""
        self.v_pre = self.V
        self.spiked = True
        self.V = self.v_rest
        self.refractory_timer = self.refractory_steps
        return self.v_pre

    # ------------------------------------------------------------- trace
    def update_trace(self):
        """Update the local activity trace from THIS cell's own state only.

            depol = clip((v_pre_reset - V_rest) / (threshold - V_rest), 0, 1)
            a <- clip(alpha_a * a + beta_v * depol + beta_s * spike, 0, a_max)

        Uses ``v_pre_reset`` (captured at integration) so a cell that fired and
        reset still registers as "recently active". Carries no information about
        WHICH afferent supplied the charge.
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

    # -------------------------------------------------------------- learning
    def update_acc_weights(self, participation):
        """The one accumulating-weight rule. Runs when this neuron fires.

            p          = threshold - sum(acc_weights)          # pre-update, signed
            signal_i   = +1 if afferent i spiked in the causal volley else -1
            delta_i    = eta * p * signal_i * distance_factor_i * (1 - (w_i/w_max)**2)
            w_i        = clip(w_i + delta_i, 0, w_max)
        """
        if not self.learn:
            return
        w = self.acc_weights
        participation = np.asarray(participation, dtype=bool)
        p = self.threshold - float(w.sum())
        signal = np.where(participation, 1.0, -1.0)
        delta = self.eta * p * signal * self.acc_distance_factor * (1.0 - (w / self.w_max) ** 2)
        np.clip(w + delta, 0.0, self.w_max, out=w)


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

    def receive(self):
        self.received_signal = True

    def resolve(self):
        if self.received_signal:
            self.spiked = True
        return self.spiked

    def clear(self):
        self.received_signal = False
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
