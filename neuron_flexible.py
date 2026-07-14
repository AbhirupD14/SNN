"""
Single Neuron implementation for the SNN.

The class supports both construction styles used in the repo:

* fixed fan-in via ``Neuron(n_inputs=...)`` for simple layers/tests;
* dynamic fan-in via ``add_input_connection()`` + ``finalize_connections()`` for
  column construction where wiring is assembled in stages.
"""

import numpy as np

# A synapse counts as having "participated" in the current integration window if
# its archived trace accumulator carries any residual charge above this floor.
PARTICIPATION_EPS = 1e-6

# ---------------------------------------------------------------------------
# Fixed-point convention for gate weights, leak controls, and thresholds.
#
# The default numeric representation for these three categories is defined in
# integer / integer-rational form rather than as bare float literals. UNIT is the
# fixed-point scale for weight/threshold magnitudes (1 model unit == 1/UNIT);
# LEAK_SCALE is the denominator for leak-rate numerators.
UNIT = 1000
LEAK_SCALE = 1000


def to_units(x):
    """Float magnitude -> integer fixed-point units (nearest)."""
    return int(round(x * UNIT))


def to_float(u):
    """Integer fixed-point units -> float magnitude."""
    return u / UNIT


def leak_num(rate):
    """Leak fraction (float) -> integer leak numerator over LEAK_SCALE."""
    return int(round(rate * LEAK_SCALE))


def div_round(n, d):
    """Deterministic integer rounding division (round-half-away-from-zero)."""
    return (n + d // 2) // d if n >= 0 else -((-n + d // 2) // d)


def _distribution_entropy(x):
    """Shannon entropy (nats) of non-negative values treated as a distribution."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    s = x.sum()
    if x.size == 0 or s <= 0:
        return 0.0
    p = x / s
    return float(-(p * np.log(p)).sum())


def _concentration(x):
    """Herfindahl-Hirschman concentration of non-negative values."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    s = x.sum()
    if x.size == 0 or s <= 0:
        return 0.0
    p = x / s
    return float((p * p).sum())


# Smaller neuron components are imported after the fixed-point constants so
# snn.membrane can read LEAK_SCALE / leak_num without a circular-import
# failure. Neuron composes these and forwards to them via properties below.
from snn.entity import NeuralEntity      # noqa: E402
from snn.synapses import SynapseBank    # noqa: E402
from snn.membrane import Membrane        # noqa: E402
from snn.rules import (select_excitatory_rule, select_inhibitory_rule,  # noqa: E402
                       select_delivery, effective_weights, bounded_signed_update)


class Neuron(NeuralEntity):
    """
    A flexible spiking neuron model that allows dynamic specification of 
    input connections during layer construction, then locks connections 
    for simulation.
    
    Both excitatory and inhibitory neurons use the same dynamics;
    the difference is in the sign of their synaptic weights.
    """
    
    def __init__(self, n_inputs=None, threshold=1000 / UNIT, refractory_period=2,
                 weight_init_range=(-500 / UNIT, 500 / UNIT),
                 learning_rate=0.1, weight_cap=1000 / UNIT, leak_rate=10 / LEAK_SCALE,
                 inhibitory_learning_rate=0.05, inhibitory_weight_cap=None,
                 excitatory_saturation_cap=None,
                 confidence_init=0.10,
                 homeostasis=False, ca_rate=0.01, ca_target=0.02, ca_band=0.5,
                 homeo_up=0.01, homeo_down=0.01,
                 homeo_budget_min=None, homeo_budget_max=None):
        """
        Initialize a flexible neuron.

        Args:
            n_inputs (int|None): optional fixed fan-in. If provided, connections
                are initialized and finalized immediately. If omitted, call
                add_input_connection() and finalize_connections().
            threshold (float): Firing threshold (constant)
            refractory_period (int): Refractory period in time steps (1ms each)
            weight_init_range (tuple): initialization range for fixed fan-in use
            learning_rate (float): Amount weights increase when neuron fires (excitatory plasticity)
            weight_cap (float): Maximum absolute value for weights (weights clipped to [-weight_cap, weight_cap]).
                Also serves as w_max, the saturating magnitude ceiling for inhibitory plasticity.
            leak_rate (float): Fraction of potential that leaks away per time step (0 = no leak, 1 = full leak)
            inhibitory_learning_rate (float): eta for the inhibitory-discharge plasticity rule
                (see apply_inhibition). 0 disables inhibitory learning.
            confidence_init (float): initial value of the active per-synapse
                confidence array. The archived confidence-beta / confidence-gamma
                params were removed.
            homeostasis, ca_rate, ca_target, ca_band, homeo_up, homeo_down,
            homeo_budget_min, homeo_budget_max: homeostatic synaptic scaling.
        """
        super().__init__()   # NeuralEntity contract (entity_id)
        # Membrane scalars (potential, threshold, resting, refractory timer/period,
        # fixed-point leak numerator, v_sat, spike bookkeeping) live on a Membrane
        # sub-object; Neuron forwards to it via properties below so every existing
        # `self.potential` / `self.threshold` / ... site is unchanged. leak_rate and
        # v_sat are seeded here; the later `self.leak_rate = ...` / `self.v_sat = ...`
        # lines were removed as redundant.
        self._membrane = Membrane(threshold, 0.0, refractory_period, leak_rate,
                                  v_sat=None)

        # Vectorized afferent bank: owns the weight/trace/confidence/distance/
        # last-input arrays and the staged construction path. Neuron forwards to it
        # through the _weights_array/_trace/... properties below, so every existing
        # read/write site (internal and external) is unchanged. confidence_init is
        # set first because the bank reads it (via this owner) when allocating.
        self.confidence_init = float(confidence_init)
        self._bank = SynapseBank(self)
        self.last_excitatory_event = {}
        # Confidence-gated consolidation (opt-in). All flags off by default so
        # bare neurons / L1 / inhibitory neurons are unaffected.
        self.confidence_consolidation = False  # revive confidence, gate eta, decay
        self.loser_depression = False          # depress active gates on inhibitory loss
        self.conf_cap = None                   # effective mature gate value (None -> weight_cap)
        self.conf_beta = 0.05                  # confidence EMA rate toward maturity
        self.eta_min = 0.05                    # plasticity floor fraction for mature gates
        self.eta_loss = 0.01                   # loser-depression rate (dimensionless)
        self.loss_gamma = 2                    # protect-small / punish-large exponent
        self.conf_rho_active = 1e-5            # confidence decay while recently active
        self.conf_rho_dead = 1e-3              # confidence decay once dead past the grace window
        self.conf_ca_dead = 0.002              # ca below this counts the neuron as inactive
        self.conf_grace = 5000                 # inactive steps before dead-decay engages
        self.inactive_steps = 0                # long grace counter (see update())
        self.loser_depression_events = 0       # diagnostic counter
        self.redistribution_events = 0         # diagnostic counter (redistribution mode)
        # Flow-proportional assembly credit (opt-in; for the E->I "assembly
        # evidence" neurons L2I / L1I -- see _update_weights). On this neuron's
        # OWN fire, credit every incoming positive synapse in proportion to the
        # flow it delivered over the retention window (the per-synapse leaky
        # _trace), normalized so the DOMINANT driver gets the full learning rate;
        # synapses that delivered no flow are depressed toward the floor. This
        # fixes the last-volley-only credit that let a habitual winner's E->I
        # synapse stall below threshold (the L2I firing deadlock). Default OFF.
        self.assembly_flow_credit = False      # flow-proportional E->I credit on fire
        self.assembly_decay_frac = 0.5         # down-pressure on non-contributing gates
        # Signed-spike depression ("4a"; opt-in; see _update_weights). OFF
        # pixels deliver no charge but their positive gates
        # are depressed on this neuron's own fire. Default OFF -> baseline intact.
        self.signed_depression = False         # enable OFF-gate depression on fire
        self.eta_off = 0.0                     # depression rate for inactive gates
        self.signed_depression_events = 0      # diagnostic counter
        # Minimal signed-spike feedforward learning (opt-in; see _update_weights).
        # On fire, every POSITIVE feedforward synapse gets one local signed update
        # -- active inputs (+1) potentiate, inactive inputs (-1) depress -- with no
        # global weight budget. Replaces the whole potentiation / OFF-depression /
        # confidence / budget stack with a single equation. Default OFF.
        self.signed_spike_learning = False
        # Structural free-energy plasticity gate (opt-in; EXCITATORY postsynaptic
        # neurons only; see _structural_free_energy_gate and _update_weights). When
        # on, the signed-spike learning rate is scaled by a STRUCTURAL maturity
        # signal derived only from this neuron's own accumulated positive afferent
        # weight vs its threshold -- NOT from membrane voltage, rivals, or labels.
        # maturity = clamp(sum_positive_afferent_weights / threshold, 0, 1);
        # gate = max(eta_floor, 1 - maturity). An under-built neuron stays fully
        # plastic; one whose excitatory support already explains a threshold
        # crossing slows down (it is "minimizing free energy"), so a consolidated
        # specialist resists being reshaped when it fires during a later pattern.
        # This REPLACES the voltage closeness term p in the signed rule (the point
        # is to make the consolidation brake structural, not input/voltage-state
        # dependent). Default OFF -> signed rule uses p exactly as before.
        self.structural_free_energy = False
        self.structural_fe_eta_floor = 0.02    # plasticity floor for a mature neuron
        # Reset-by-subtraction on fire (opt-in; see fire()). Default OFF reproduces
        # the full reset-to-rest. When ON, a fired neuron keeps its residual
        # overshoot above threshold (standard LIF), floored at rest -- this attacks
        # the discharge asymmetry vs partially-inhibited losers (a fully-reset
        # winner would otherwise be overtaken by losers that ratcheted charge up).
        self.subtractive_reset = False
        # Hard-reset inhibition (opt-in; see apply_inhibition). Default OFF reproduces the small
        # subtractive/flow gate. When ON, a real inhibitory discharge (events
        # non-empty) clamps the loser's membrane back to rest AFTER the inhibitory
        # plasticity rule has read its pre-reset charge -- so the loser's
        # accumulated charge is consumed by local learning, then cleared. This
        # attacks loser charge carryover directly: unlike the winner (which resets
        # on fire), partially-inhibited losers keep charge and start the next race
        # ahead; a hard reset returns every non-winner to the same baseline so the
        # best-matched integrator rebuilds charge fastest and wins repeatedly.
        # hard_reset_clear_traces also zeroes the excitatory/inhibitory current
        # traces so no residual flow refills the membrane after the reset.
        self.l2i_hard_reset_losers = False
        self.hard_reset_clear_traces = True
        # Homeostatic synaptic scaling (third local system).
        self.homeostasis = homeostasis
        self.ca_rate = ca_rate
        self.ca_target = ca_target
        self.ca_band = ca_band
        self.homeo_up = homeo_up
        self.homeo_down = homeo_down
        self.homeo_budget_min = homeo_budget_min
        self.homeo_budget_max = homeo_budget_max
        self.ca = 0.0                 # slow EMA of the neuron's OWN firing ("calcium")
        self.homeo_budget = None      # homeostatic excitatory resource R (lazy-init)
        self.weight_budget = None      # optional homeostatic budget for positive weights
        self.min_positive_weight = None  # optional floor for positive weights (see _apply_budget_and_cap)
        self.learning_rate = learning_rate  # Weight increase amount when neuron fires (excitatory)
        self.weight_cap = weight_cap        # Maximum absolute value for weights (also w_max for inhibition)
        self.inhibitory_learning_rate = inhibitory_learning_rate  # eta for inhibitory plasticity
        # Saturation ceiling (w_max) for inhibitory gates, kept separate from the
        # feedforward weight_cap (defaults to weight_cap when None). See apply_inhibition.
        self.inhibitory_weight_cap = inhibitory_weight_cap
        # Saturation ceiling (w_max) for the EXCITATORY quadratic term, kept
        # separate from the hard clip weight_cap -- see this class for the
        # full rationale (sqrt(w_max) equilibrium; set to weight_cap**2 so a
        # habitually-participating synapse can reach the hard cap exactly).
        self.excitatory_saturation_cap = excitatory_saturation_cap
        # Membrane saturation ceiling (absolute, in the same units as potential;
        # None = unbounded, the baseline). A finite driving force / reversal
        # potential: charge accumulation cannot push the membrane above this, so
        # the potential cannot ratchet far past threshold between cycles. Local
        # and per-neuron; see receive_input. Off by default so the baseline is intact
        # (seeded to None on the Membrane in __init__; forwards there).
        # Sparse excitatory FLOW-RATE accumulation (opt-in; see receive_input,
        # advance_trace, and Current_Implementation_Methodology_Equations.md).
        # Default OFF -> instantaneous V += dot(weights, spikes) baseline. When on,
        # an input spike opens a decaying excitatory CURRENT trace (exc_trace) that
        # is integrated into the membrane over time using closed-form skipped-time
        # math, so weight behaves as a current/gate amplitude rather than a charge
        # packet deposited instantly.
        self.excitatory_flow_rate = False   # enable current-trace accumulation
        self.exc_trace_decay = 0.8          # per-timestep current decay d in [0, 1)
        self.exc_trace_normalized = True    # inject drive*(1-d) so total delivered ~= drive
        self.exc_trace = 0.0                # I: current trace amplitude
        self.exc_trace_last_t = 0           # last outer timestep the trace was advanced to
        # Inhibitory FLOW (opt-in; symmetric to the excitatory flow). When on, a real
        # inhibitory discharge injects the gate magnitude into a decaying inhibitory
        # CURRENT (inh_trace) that drains charge out of the membrane over several
        # steps (in update(), floored at rest), instead of subtracting it all at once
        # -- sustained suppression to counteract the continuous excitatory inflow.
        # Normalized injection w*(1-d) makes the TOTAL drained ~= w (same as the
        # one-shot hit, just spread over time); un-normalized injects w (total w/(1-d),
        # a stronger sustained bite). Does not change the stored gate weight.
        self.inhibitory_flow_rate = False
        self.inh_trace_decay = 0.8
        self.inh_trace_normalized = True
        self.inh_trace = 0.0                # pending inhibitory current (drains in update)
        # Inhibitory-gate plasticity rule (see apply_inhibition). inhibitory_delta_rule
        # False = legacy SATURATING rule (dw = eta*p*(1 - w^2/w_max); every gate
        # converges to the same ceiling sqrt(w_max), so gates end up uniform). True =
        # a differentiating local rule, selected by inhibitory_rule_mode. All rules
        # normalize the gate against a sub-threshold ceiling G = sqrt(w_max) (the
        # scale the saturating rule converges to) and share the same linear
        # floored-at-rest delivery. Neuron default stays legacy so bare-neuron tests
        # are unchanged; SimulationEngine defaults delta_rule True, mode "turnover".
        self.inhibitory_delta_rule = False
        # "turnover" (default): event-local strengthen/turnover on the NORMALIZED gate
        #   u = w/G:  du = eta_up*p_t*(1 - u) - eta_down*u,  p_t = clamp(v_pre/theta,
        #   0, p_max). High-charge targets strengthen more (eta_up term); every gate
        #   turns over proportional to size (eta_down term). No target voltage, no
        #   averages -- purely this event's v_pre, theta, w and spike.
        # "margin": relax the gate toward s = clamp(v_pre - margin_frac*theta, 0, G)
        #   (a hand-set post-inhibition target level); kept as a diagnostic.
        self.inhibitory_rule_mode = "turnover"
        self.inhibitory_eta_up = 0.02       # turnover: charge-driven strengthening
        self.inhibitory_eta_down = 0.005    # turnover: size-proportional decay
        self.inhibitory_p_max = 1.0         # turnover: cap on p_t = v_pre/theta
        self.inhibitory_margin_frac = 0.5   # margin mode: target_post = frac * theta
        self.inhibitory_delta_eta = 0.05    # margin mode: EMA rate toward s
        # Distance attenuation of DELIVERED excitatory drive (opt-in; see
        # receive_input). The weight is the learned gate strength; distance is a
        # fixed per-synapse DELIVERY attenuation that multiplies the injected drive
        # amplitude -- it does NOT change the stored weight, and does NOT enter the
        # trace decay/integration math. Per afferent:
        #   factor_i = (distance_ref / max(d_i, distance_min)) ** distance_power
        # so the effective drive is sum_i spike_i * w_i * factor_i. _distance holds
        # the per-afferent d_i (allocated to ones at finalize = no attenuation until
        # real functional distances are provided). Default OFF preserves behavior.
        self.distance_weighting = False
        self.distance_power = 2.0
        self.distance_ref = 1.0
        self.distance_min = 1.0
        self._distance = None               # per-afferent d_i (set at finalize)
        self.last_inhibitory_events = []    # debug records from the most recent apply_inhibition()
        # (last_spike_time / spiked live on the Membrane, seeded in its constructor;
        # Neuron forwards to them via the properties below.)

        if n_inputs is not None:
            weights = np.random.uniform(weight_init_range[0], weight_init_range[1], int(n_inputs))
            self._set_weights(weights, finalized=True)

    # --- Membrane forwarding (Phase 1) ----------------------------------------
    # The membrane scalars live on self._membrane; these properties keep every
    # existing `self.potential` / `self.threshold` / `self._leak_num` / ... site
    # (internal and external) working unchanged. leak_rate stays the fixed-point
    # accessor, now backed by the Membrane's integer numerator.
    @property
    def potential(self): return self._membrane.potential
    @potential.setter
    def potential(self, v): self._membrane.potential = v

    @property
    def threshold(self): return self._membrane.threshold
    @threshold.setter
    def threshold(self, v): self._membrane.threshold = v

    @property
    def resting_potential(self): return self._membrane.resting_potential
    @resting_potential.setter
    def resting_potential(self, v): self._membrane.resting_potential = v

    @property
    def refractory_period(self): return self._membrane.refractory_period
    @refractory_period.setter
    def refractory_period(self, v): self._membrane.refractory_period = v

    @property
    def refractory_timer(self): return self._membrane.refractory_timer
    @refractory_timer.setter
    def refractory_timer(self, v): self._membrane.refractory_timer = v

    @property
    def v_sat(self): return self._membrane.v_sat
    @v_sat.setter
    def v_sat(self, v): self._membrane.v_sat = v

    @property
    def spiked(self): return self._membrane.spiked
    @spiked.setter
    def spiked(self, v): self._membrane.spiked = v

    @property
    def last_spike_time(self): return self._membrane.last_spike_time
    @last_spike_time.setter
    def last_spike_time(self, v): self._membrane.last_spike_time = v

    @property
    def _leak_num(self): return self._membrane._leak_num
    @_leak_num.setter
    def _leak_num(self, v): self._membrane._leak_num = v

    @property
    def leak_rate(self):
        """Leak fraction per step, backed by the Membrane's integer numerator
        over LEAK_SCALE (fixed-point leak control)."""
        return self._membrane.leak_rate

    @leak_rate.setter
    def leak_rate(self, value):
        self._membrane.leak_rate = value

    # --- SynapseBank forwarding (Phase 1) -------------------------------------
    # The afferent arrays and construction state live on self._bank; these
    # properties keep every existing `self._weights_array` / `self._trace` / ...
    # site (and external `n._weights_array = ...` callers) working unchanged. The
    # getters return the bank's actual array objects, so in-place index mutation
    # (`self._weights_array[active] = ...`) still writes through to the bank.
    @property
    def _weights_array(self): return self._bank.weights_array
    @_weights_array.setter
    def _weights_array(self, v): self._bank.weights_array = v

    @property
    def _trace(self): return self._bank.trace
    @_trace.setter
    def _trace(self, v): self._bank.trace = v

    @property
    def _confidence(self): return self._bank.confidence
    @_confidence.setter
    def _confidence(self, v): self._bank.confidence = v

    @property
    def _distance(self): return self._bank.distance
    @_distance.setter
    def _distance(self, v): self._bank.distance = v

    @property
    def _last_input_spikes(self): return self._bank.last_input_spikes
    @_last_input_spikes.setter
    def _last_input_spikes(self, v): self._bank.last_input_spikes = v

    @property
    def _connections_finalized(self): return self._bank.connections_finalized
    @_connections_finalized.setter
    def _connections_finalized(self, v): self._bank.connections_finalized = v

    @property
    def _weights_list(self): return self._bank.weights_list
    @_weights_list.setter
    def _weights_list(self, v): self._bank.weights_list = v

    def add_input_connection(self, weight):
        """Add an input connection (positive excitatory / negative inhibitory).
        Only valid before finalization. Delegates to the SynapseBank."""
        self._bank.add_input_connection(weight)

    def finalize_connections(self):
        """Lock connections and materialize the arrays. Delegates to the bank."""
        self._bank.finalize()

    def _set_weights(self, weights, finalized=None):
        """Replace the afferent weight vector and resize aligned local state.
        Delegates to the SynapseBank (clip uses this neuron's current weight_cap)."""
        self._bank.set_weights(weights, finalized=finalized)

    def _ensure_finalized(self):
        """Internal method to check if connections are ready for simulation."""
        if not self._connections_finalized:
            raise RuntimeError("Connections not finalized. "
                             "Call finalize_connections() before simulation.")
        
    def advance_trace(self, t):
        """Lazily integrate the excitatory current trace forward to outer timestep
        `t` (closed-form skipped-time), adding the integrated current to the
        membrane and decaying the trace. This is the sparse equivalent of running
        `V += I; I *= d` once per timestep from exc_trace_last_t up to t, but in
        O(1). No-op outside flow-rate mode or when already at/past t. See the
        methodology doc's flow-rate section for the derivation."""
        if not self.excitatory_flow_rate:
            return
        dt = t - self.exc_trace_last_t
        if dt <= 0:
            return
        d = self.exc_trace_decay
        if self.exc_trace != 0.0:
            # Geometric sum of dt discrete `V += I; I *= d` steps. As d -> 1 the
            # closed form (1 - d^dt)/(1 - d) -> dt (its limit), which also avoids
            # the 1/(1-d) blow-up; d == 0 gives geom == 1 (contributes once).
            geom = float(dt) if abs(1.0 - d) < 1e-9 else (1.0 - d ** dt) / (1.0 - d)
            self._membrane.deposit(self.exc_trace * geom)
            self.exc_trace *= d ** dt
        self.exc_trace_last_t = t

    def receive_input(self, input_spikes, charge_scale=1.0, t=None):
        """
        Accumulate charge based on weighted inputs.

        Args:
            input_spikes (array-like): Binary array indicating which inputs spiked (1) or not (0)
            charge_scale (float): scales ONLY the delivered charge (potential
                increment and eligibility trace), not the participation signal --
                _last_input_spikes still records the binary input vector so the
                signed-spike learning rule sees the true participating synapses.
                Used by the chunked-charge L2 competition (l2_charge_chunks) to
                deliver weight_ji/K per inner sub-step; the default 1.0 leaves
                every existing caller unchanged.
            t (int|None): current outer timestep. Required for flow-rate mode
                (excitatory_flow_rate); ignored in the instantaneous baseline.
        """
        self._ensure_finalized()

        # Only accumulate charge if not in refractory period (flow mode too: a
        # refractory neuron neither advances its trace nor injects, so no hidden
        # current builds up while it is clamped -- consistent with instantaneous).
        if self.refractory_timer > 0 or len(self._weights_array) == 0:
            return
        input_spikes = np.asarray(input_spikes, dtype=float)

        # Delivery is a strategy (Phase 3c): distance attenuates the delivered
        # amplitude (never the stored weight), then flow-rate vs instantaneous
        # integrates it into the membrane. select_delivery matches the original
        # `excitatory_flow_rate and t is not None` guard.
        w_eff = effective_weights(self)
        select_delivery(self, t).deliver(self, input_spikes, charge_scale, t, w_eff)

    def apply_inhibition(self, inhibitory_spikes):
        """
        Deliver inhibitory-discharge events and run the inhibitory plasticity rule.

        Independent second learning system (the excitatory rule in _update_weights
        is untouched and fires only on a postsynaptic spike). An inhibitory synapse
        is any afferent with a negative weight, acting as an adaptive suppression
        gate. For every inhibitory synapse carrying a spike this step, per the
        algorithm (w = |weight|, w_max = inhibitory_weight_cap or weight_cap,
        theta = threshold):

            V_pre  = V ; V = V - w ; V_post = V
            p      = V_pre / theta
            dw     = eta * p * (1 - w^2 / w_max)   (saturating, local, gradient-free)
            w      = w + dw

        The gate strengthens most when it suppressed a near-threshold neuron and
        saturates as w -> w_max, so no global normalization is needed. Sign is
        preserved (weights stay negative; |w| grows). Note the quadratic term
        means the natural equilibrium (dw = 0) is w* = sqrt(w_max), not w_max
        itself, whenever w_max != 1 -- growth reverses for w > sqrt(w_max).

        Returns: list[dict] debug records (v_pre, v_post, theta, p, w_before,
        delta_w, w_after, index) -- also stored on self.last_inhibitory_events.
        """
        self._ensure_finalized()
        events = []
        self.last_inhibitory_events = events
        if self.refractory_timer > 0 or len(self._weights_array) == 0:
            return events

        spikes = np.asarray(inhibitory_spikes, dtype=float)
        theta = self.threshold
        w_max = self.inhibitory_weight_cap if self.inhibitory_weight_cap is not None else self.weight_cap
        # Loser-depression closeness signal: how close to firing this neuron was
        # BEFORE any inhibitory discharge this call (see _depress_losers).
        v_entry = float(self.potential)
        active = np.nonzero((self._weights_array < 0) & (spikes > 0.5))[0]
        for idx in active:
            w = -float(self._weights_array[idx])   # magnitude of the inhibitory gate
            v_pre = float(self.potential)
            if self.inhibitory_flow_rate:
                # Inhibitory FLOW: inject the gate into a decaying inhibitory current
                # that drains the membrane over subsequent steps (see update()),
                # rather than subtracting it all now. No instantaneous change here;
                # the learning rule below still uses v_pre.
                self.inh_trace += w * (1.0 - self.inh_trace_decay) if self.inh_trace_normalized else w
                v_post = v_pre
            else:
                # Linear one-shot discharge FLOORED at rest:
                # inhibition cannot push the membrane below resting potential.
                self.potential = max(self.potential - w, self.resting_potential)
                v_post = float(self.potential)
            # Normalized closeness to firing, clamped to [0, 1] (saturating rule;
            # also recorded in the event below).
            p = min(max(v_pre / theta, 0.0), 1.0) if theta > 0 else 0.0
            # Per-discharge gate learning is a strategy (Phase 3b): saturating vs
            # differentiating turnover/margin, selected by the neuron's flags. All
            # inputs are local to this event; it returns the new gate magnitude.
            w_new = select_inhibitory_rule(self).new_magnitude(self, w, v_pre, w_max, theta, p)
            self._weights_array[idx] = -w_new      # keep the inhibitory sign
            events.append(dict(index=int(idx), v_pre=v_pre, v_post=v_post,
                               theta=theta, p=p, w_before=w,
                               delta_w=w_new - w, w_after=w_new))
        # Loser depression: a real inhibitory discharge (events non-empty) means
        # this neuron was a suppressed near-winner; depress the active positive
        # feedforward gates that made it one. Opt-in; see _depress_losers.
        if self.loser_depression and events:
            self._depress_losers(v_entry)
        # Hard reset: AFTER the inhibitory plasticity rule (and loser depression)
        # have consumed the loser's pre-reset charge, clamp the membrane back to
        # rest so no charge carries into the next race. Ordering matters -- the
        # learning above already read v_pre; only the transient charge is cleared,
        # never the weights. Optionally zero the current traces so residual
        # excitatory/inhibitory flow can't refill the membrane after the reset.
        if self.l2i_hard_reset_losers and events:
            self.potential = self.resting_potential
            if self.hard_reset_clear_traces:
                self.exc_trace = 0.0
                self.inh_trace = 0.0
        return events

    def _depress_losers(self, v_pre_loss):
        """Depress the active positive feedforward gates after an L2I discharge.
        Nonlinear protect-small / punish-large, confidence-protected, and scaled by
        how close the neuron was to firing (p_loss). Inactive and mature gates are
        spared; the negative gate is never touched (positive weights only)."""
        theta = self.threshold
        p_loss = min(max(v_pre_loss / theta, 0.0), 1.0) if theta > 0 else 0.0
        if p_loss <= 0.0:
            return
        participating = self._last_input_spikes > 0.5
        active = np.nonzero((self._weights_array > 0) & participating)[0]
        if active.size == 0:
            return
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        w = self._weights_array[active]
        C = self._confidence[active]
        ratio = self._maturity(w)
        dw_minus = (self.eta_loss * p_loss * (1.0 - C)
                    * (ratio ** self.loss_gamma) * (w - w_min))
        # Clamp at the floor: the step is meant to decelerate INTO w_min, but a
        # large eta_loss (slider reaches 20) makes the multiplier exceed 1 and
        # overshoot below w_min into negatives. active is the positive-gate set,
        # so flooring here leaves the legitimate negative gate untouched.
        self._weights_array[active] = np.maximum(w - dw_minus, w_min)
        self.loser_depression_events += 1
        self._apply_budget_and_cap()

    def apply_competitive_reset(self, current_input=None,
                                competitive_weight_update="depression"):
        """Unweighted L2I competitive-reset event broadcast to every L2E neuron.

        There is no learned L2I->L2E gate,
        no inhibitory magnitude, and no negative afferent involved. The event has
        two parts:

          - a refractory-gated LOSER WEIGHT UPDATE selected by
            `competitive_weight_update`. The neuron that fired this step is in
            refractory (refractory_timer > 0 at arrival) and is the current WINNER:
            it skips the weight update entirely (winner protection). A non-refractory
            neuron is a LOSER and runs one of:
              * "redistribution": estimate how strongly its ACTIVE feedforward gates
                match the current input (p_match from the active effective-weight
                sum), move an incremental bounded amount OUT of those active gates,
                and place exactly that amount INTO its inactive (OFF) gates with
                headroom -- total positive feedforward mass is conserved. See
                _redistribute_off.
              * "depression": the retained one-sided competitive-depression baseline
                -- depress only participating positive gates through the reflected
                downward kernel, scaled by p_loss = clamp(V_pre/theta, 0, 1). OFF
                gates are never touched and mass is NOT conserved. See
                _depress_one_sided.
              * "none": hard-reset-only control -- no weight update at all.
          - an UNCONDITIONAL hard reset of the membrane to rest and clearing of the
            pending excitatory/inhibitory current traces (when hard_reset_clear_traces),
            performed in EVERY mode and even for a refractory winner.

        The refractory timer is left untouched (end-of-step update() decrements it),
        so with refractory=1 the winner is protected only for the current step and
        can compete again next step. `current_input` is the accepted feedforward
        input mask (same one the L2E charge path used); when None the neuron's own
        `_last_input_spikes` is used. Returns a diagnostic record (see below)."""
        self._ensure_finalized()
        if competitive_weight_update not in ("redistribution", "depression", "none"):
            raise ValueError("unknown competitive_weight_update mode: "
                             f"{competitive_weight_update!r}")
        theta = self.threshold
        v_pre = float(self.potential)
        refractory_at_arrival = self.refractory_timer > 0
        p_loss = min(max(v_pre / theta, 0.0), 1.0) if theta > 0 else 0.0

        # Input mask: the current accepted feedforward input used by the charge path.
        if current_input is None:
            participating = self._last_input_spikes > 0.5
        else:
            participating = np.asarray(current_input, dtype=float) > 0.5

        # Redistribution diagnostics (defaults for the non-redistribution paths).
        plasticity_applied = False
        active_indices: list[int] = []
        off_indices: list[int] = []
        active_effective_sum = 0.0
        p_match = 0.0
        candidate_release = 0.0
        transferred = 0.0
        active_before = np.zeros(0); active_delta = np.zeros(0); active_after = np.zeros(0)
        off_before = np.zeros(0); off_delta = np.zeros(0); off_after = np.zeros(0)
        # Depression / legacy-compatible diagnostics.
        depressed_indices: list[int] = []
        weights_before = np.zeros(0); delta_weights = np.zeros(0); weights_after = np.zeros(0)

        have_weights = (self.weight_cap > 0 and self._weights_array is not None
                        and len(self._weights_array) > 0)
        # Weight plasticity runs ONLY for a non-refractory loser (winner protection
        # comes solely from refractory_timer, never a one-hot winner check).
        if (not refractory_at_arrival) and have_weights:
            w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
            w_cap = self.weight_cap
            if competitive_weight_update == "redistribution":
                r = self._redistribute_off(participating, w_min, w_cap, theta)
                (plasticity_applied, active_indices, off_indices,
                 active_effective_sum, p_match, candidate_release, transferred,
                 active_before, active_delta, active_after,
                 off_before, off_delta, off_after) = r
            elif competitive_weight_update == "depression":
                d = self._depress_one_sided(participating, w_min, w_cap, p_loss)
                (plasticity_applied, depressed_indices,
                 weights_before, delta_weights, weights_after) = d
            # "none": hard reset only, no weight update.

        # Unconditional hard reset (every mode, even a refractory winner): zero the
        # membrane and pending current traces. The refractory timer is untouched.
        self.potential = self.resting_potential
        if self.hard_reset_clear_traces:
            self.exc_trace = 0.0
            self.inh_trace = 0.0
        v_post = float(self.potential)
        return dict(
            mode=competitive_weight_update,
            refractory_at_arrival=refractory_at_arrival,
            plasticity_applied=plasticity_applied,
            v_pre=v_pre, v_post=v_post, theta=theta,
            p_loss=p_loss, p_match=p_match,
            active_indices=active_indices, off_indices=off_indices,
            active_effective_sum=active_effective_sum,
            candidate_release=candidate_release, transferred=transferred,
            active_before=active_before, active_delta=active_delta, active_after=active_after,
            off_before=off_before, off_delta=off_delta, off_after=off_after,
            # Depression / legacy-compatible keys (empty in the other modes).
            depressed_indices=depressed_indices,
            weights_before=weights_before, delta_weights=delta_weights,
            weights_after=weights_after)

    def _depress_one_sided(self, participating, w_min, w_cap, p_loss):
        """Retained one-sided competitive-depression baseline (the "depression"
        mode). Depress only the ACTIVE positive feedforward gates (weight>0 AND the
        pixel participated) through the reflected downward kernel, scaled by
        p_loss = clamp(V_pre/theta, 0, 1) and the structural-FE gate when enabled.
        OFF gates are untouched and mass is NOT conserved. Returns
        (plasticity_applied, depressed_indices, weights_before, delta, weights_after)."""
        depressed_indices: list[int] = []
        weights_before = np.zeros(0); delta_weights = np.zeros(0); weights_after = np.zeros(0)
        if p_loss <= 0.0:
            return False, depressed_indices, weights_before, delta_weights, weights_after
        eligible = np.nonzero((self._weights_array > 0) & participating)[0]
        if eligible.size == 0:
            return False, depressed_indices, weights_before, delta_weights, weights_after
        w_before = self._weights_array[eligible].astype(float).copy()
        gate = (self._structural_free_energy_gate()
                if self.structural_free_energy else 1.0)
        gain = self.learning_rate * gate * p_loss
        signal = np.full(eligible.size, -1.0)
        w_after = bounded_signed_update(w_before, w_min, w_cap, gain, signal)
        self._weights_array[eligible] = w_after
        self.loser_depression_events += 1
        return (True, eligible.tolist(), w_before, w_after - w_before, w_after)

    def _redistribute_off(self, participating, w_min, w_cap, theta):
        """Conservative redistribution of a loser's ACTIVE feedforward gate capacity
        into its OFF gates (the "redistribution" mode; spec Sections 4-6).

        Structural competition signal (Section 4):
            c_ji = w_ji * delivery_factor_ji   (effective_weights; distance-aware)
            S_on = sum(c_ji for active gates); p_match = clamp(S_on/theta, 0, 1)
        Candidate active decrease (Section 5): per active gate, the reflected bounded
        downward kernel gain = learning_rate * p_match, clipped so no gate crosses
        w_min. Conservative OFF recruitment (Section 6): transfer T = min(R_candidate,
        C_off); if OFF capacity is short, scale every active decrease by T/R_candidate
        so untransferable resource stays in the active gates; place exactly T across
        OFF gates via H_up-weighted capped water-filling. Total positive feedforward
        mass is invariant. Returns the redistribution diagnostic tuple."""
        w = self._weights_array
        pos = w > 0
        active_idx = np.nonzero(pos & participating)[0]
        off_idx = np.nonzero(pos & ~participating)[0]
        active_before = w[active_idx].astype(float).copy()
        off_before = w[off_idx].astype(float).copy()

        # Structural competition signal from active DELIVERED support.
        active_effective_sum = 0.0
        p_match = 0.0
        if active_idx.size > 0 and theta > 0:
            w_eff = effective_weights(self)
            active_effective_sum = float(np.sum(np.maximum(w_eff[active_idx], 0.0)))
            p_match = min(max(active_effective_sum / theta, 0.0), 1.0)

        def _no_change():
            return (False, active_idx.tolist(), off_idx.tolist(),
                    active_effective_sum, p_match, 0.0, 0.0,
                    active_before, np.zeros_like(active_before), active_before.copy(),
                    off_before, np.zeros_like(off_before), off_before.copy())

        if active_idx.size == 0 or p_match <= 0.0 or off_idx.size == 0:
            return _no_change()

        # Candidate active decreases: reflected downward kernel, already clipped at
        # w_min by bounded_signed_update.
        gain = self.learning_rate * p_match
        signal = np.full(active_idx.size, -1.0)
        candidate_after = bounded_signed_update(active_before, w_min, w_cap, gain, signal)
        candidate_d = active_before - candidate_after   # >= 0
        R_candidate = float(candidate_d.sum())
        if R_candidate <= 0.0:
            return _no_change()

        # Conservative transfer: bounded by OFF capacity.
        capacity = np.maximum(w_cap - off_before, 0.0)
        C_off = float(capacity.sum())
        transfer = min(R_candidate, C_off)
        if transfer <= 0.0:
            return _no_change()

        scale = transfer / R_candidate
        applied_d = candidate_d * scale                 # sum == transfer
        active_after = active_before - applied_d
        off_alloc = self._allocate_off_transfer(off_before, w_min, w_cap, transfer)

        self._weights_array[active_idx] = active_after
        self._weights_array[off_idx] = off_before + off_alloc
        self.redistribution_events += 1
        return (True, active_idx.tolist(), off_idx.tolist(),
                active_effective_sum, p_match, R_candidate, transfer,
                active_before, active_after - active_before, active_after,
                off_before, off_alloc, off_before + off_alloc)

    def _allocate_off_transfer(self, w_off, w_min, w_cap, transfer):
        """Deterministic capped water-filling of exactly `transfer` across OFF gates
        (spec Section 6). Allocate proportional to the FIXED upward headroom weight
        H_up(q_k) = 1 - q_k^2 (q_k from the pre-transfer weight), respecting each
        gate's remaining capacity w_cap - w_k; redistribute any clipping remainder
        among gates that still have capacity using the same H_up weights until the
        full transfer is placed or every OFF gate is at w_cap. Returns the per-gate
        allocation (all >= 0, sum == min(transfer, total capacity))."""
        n = w_off.size
        alloc = np.zeros(n)
        remaining_cap = np.maximum(w_cap - w_off, 0.0).astype(float)
        left = min(float(transfer), float(remaining_cap.sum()))
        if left <= 0.0:
            return alloc
        if w_cap > w_min:
            q = np.clip((w_off - w_min) / (w_cap - w_min), 0.0, 1.0)
            H_up = 1.0 - q ** 2
        else:
            H_up = np.zeros(n)
        open_gate = remaining_cap > 1e-15
        for _ in range(n + 1):
            if left <= 1e-12 or not open_gate.any():
                break
            weights = np.where(open_gate, H_up, 0.0)
            wsum = float(weights.sum())
            if wsum <= 0.0:
                # All open gates have zero upward headroom (near cap): fall back to
                # capacity-proportional so the transfer still completes deterministically.
                weights = np.where(open_gate, remaining_cap, 0.0)
                wsum = float(weights.sum())
                if wsum <= 0.0:
                    break
            give = np.minimum(left * weights / wsum, remaining_cap)
            alloc += give
            remaining_cap -= give
            left -= float(give.sum())
            open_gate = remaining_cap > 1e-15
        # Float remainder: place in the gate with the most remaining capacity.
        if left > 1e-12 and open_gate.any():
            k = int(np.argmax(np.where(open_gate, remaining_cap, -1.0)))
            give = min(left, float(remaining_cap[k]))
            alloc[k] += give
        return alloc

    def update(self):
        """
        Update neuron state for next time step.
        Handles refractory period and potential leak to resting state.
        """
        self._ensure_finalized()

        # Homeostatic firing-rate sensor: slow EMA of this neuron's own spiking.
        self.ca += self.ca_rate * (float(self.spiked) - self.ca)

        # Activity-dependent confidence decay (long-term memory; opt-in). See
        # update() for the full rationale.
        if self.confidence_consolidation:
            self._decay_confidence()

        # Membrane leak + refractory countdown (fixed-point leak on the Membrane).
        # Returns True on a non-refractory (leak) step, where the eligibility-trace
        # decay and inhibitory-current drain -- which touch bank/neuron state, not
        # just the membrane -- also run.
        if self._membrane.leak_and_countdown():
            # Decay the eligibility trace with the same leak fraction.
            self._trace *= (1.0 - self._leak_num / LEAK_SCALE)
            # Inhibitory FLOW: drain the pending inhibitory current out of the
            # membrane this step (floored at rest), then decay it. Runs every step,
            # so a single discharge keeps suppressing the target for ~1/(1-decay)
            # steps instead of only on the step L2I fired. Not reset on fire -- the
            # pending inhibition drains fully regardless.
            if self.inhibitory_flow_rate and self.inh_trace > 0.0:
                self.potential = max(self.potential - self.inh_trace, self.resting_potential)
                self.inh_trace *= self.inh_trace_decay

        # Homeostatic synaptic scaling (slow, activity-driven, non-Hebbian).
        if self.homeostasis:
            self._homeostatic_scaling()

        # Reset spike flag for next time step
        self.spiked = False
        
    def check_threshold(self):
        """
        Check if neuron should fire based on threshold.

        Returns:
            bool: True if neuron fires, False otherwise
        """
        self._ensure_finalized()
        return self._membrane.check_threshold()
    
    def fire(self):
        """
        Handle spike event: capture charge, discharge (reset potential), start
        refractory period, and update weights from the captured charge.
        """
        self._ensure_finalized()

        # Capture charge BEFORE discharging (mirrors apply_inhibition's V_pre).
        # Weight learning below uses this v_pre unchanged, so reset semantics do
        # not alter what the fire teaches -- only the post-fire membrane state.
        v_pre = float(self.potential)

        # Membrane discharge on spike: reset (full or subtractive), arm refractory,
        # record spike time, mark spiked. See Membrane.fire_reset.
        self._membrane.fire_reset(self.subtractive_reset)

        # Flow-rate: discharge the excitatory current trace along with the membrane,
        # so no residual current keeps re-charging a just-fired neuron (and none
        # back-integrates when it leaves the refractory window). Consistent with the
        # full membrane reset above.
        if self.excitatory_flow_rate:
            self.exc_trace = 0.0

        # Update weights from the charge captured before discharge.
        self._update_weights(v_pre)

        # Evidence consumed: clear the trace for the next cycle (ARCHIVED bookkeeping).
        if self._trace is not None:
            self._trace = np.zeros_like(self._trace)

    def _update_weights(self, v_pre):
        """
        Charge-based excitatory weight update -- same algorithm and rationale for
        fixed and dynamic fan-in construction (same structure as
        apply_inhibition: capture charge, discharge, then
        dw = eta * p * (1 - w^2/w_max), with p = clamp(theta/v_pre, 0, 1) for
        excitation. w_max here is excitatory_saturation_cap (defaults to
        weight_cap when None), kept separate from the hard clip weight_cap so
        the sqrt(w_max) equilibrium can be made to land exactly on weight_cap
        -- see this class for the full rationale. Only synapses active in
        the most recent receive_input() call are updated
        (self._last_input_spikes), replacing the ARCHIVED trace-based
        participation and confidence-weighted credit-splitting rules.
        """
        if self._weights_array is None or len(self._weights_array) == 0:
            return
        # Polymorphic dispatch over the one active excitatory rule (Phase 3a). The
        # signed-spike / assembly-flow / charge-based branches now live as strategy
        # objects in snn/rules/excitatory.py; the selector encodes the old mutual
        # exclusivity (signed and assembly took precedence; charge is the default).
        select_excitatory_rule(self).on_fire(self, v_pre)

    def _positive_afferent_weight_sum(self):
        """Total learned excitatory support: the sum of this neuron's POSITIVE
        afferent weights (negative inhibitory gates such as L2I->L2E contribute 0).
        Input/label independent -- it is the whole positive weight vector, not only
        the currently-active afferents, so the structural brake below is a property
        of what the neuron HAS learned, not of the current input event."""
        if self._weights_array is None or len(self._weights_array) == 0:
            return 0.0
        return float(np.maximum(self._weights_array, 0.0).sum())

    def _structural_free_energy_gate(self):
        """Structural free-energy plasticity gate in [eta_floor, 1] (see __init__).
        maturity = clamp(sum_positive_afferent_weights / threshold, 0, 1);
        gate = max(eta_floor, 1 - maturity). Uses ONLY this neuron's own positive
        weights and its own threshold -- no membrane voltage, no rivals, no labels.
        An under-built neuron (sum << theta) gates ~1 (fully plastic); a neuron whose
        excitatory support already covers a threshold crossing (sum >= theta) gates
        at the floor (consolidated/stable)."""
        theta = self.threshold
        if theta <= 0:
            return 1.0
        maturity = min(max(self._positive_afferent_weight_sum() / theta, 0.0), 1.0)
        return max(self.structural_fe_eta_floor, 1.0 - maturity)

    def _maturity(self, w):
        """Local instantaneous maturity m in [0,1] of positive gate weights w:
        (w - w_min) / (w_conf_cap - w_min), clamped. w_min is min_positive_weight
        (or 0), w_conf_cap is conf_cap (the effective reachable mature value; falls
        back to weight_cap)."""
        w_min = self.min_positive_weight if self.min_positive_weight is not None else 0.0
        cap = self.conf_cap if self.conf_cap is not None else self.weight_cap
        if cap <= w_min:
            return np.zeros_like(w)
        return np.clip((w - w_min) / (cap - w_min), 0.0, 1.0)

    def _apply_budget_and_cap(self):
        """Shared tail: renormalize positive weights to the resource target, then
        the absolute cap. Under homeostasis the target is the homeostatic resource
        R; otherwise the fixed weight_budget. See _apply_budget_and_cap details,
        including the min_positive_weight floor applied below."""
        target = self._resource_target()
        if target is not None:
            pos = self._weights_array > 0
            total = float(self._weights_array[pos].sum())
            if total > 1e-9:
                self._weights_array[pos] *= target / total
        if self.min_positive_weight is not None:
            pos = self._weights_array > 0
            self._weights_array[pos] = np.maximum(self._weights_array[pos], self.min_positive_weight)
        self._weights_array = np.clip(self._weights_array, -self.weight_cap, self.weight_cap)

    def _resource_target(self):
        """Positive-weight budget to renormalize to: homeostatic resource R under
        homeostasis (lazy-init from current positive sum), else weight_budget."""
        if self.homeostasis:
            if self.homeo_budget is None:
                pos = self._weights_array > 0
                self.homeo_budget = float(self._weights_array[pos].sum()) if pos.any() else None
            return self.homeo_budget
        return self.weight_budget

    def _homeostatic_scaling(self):
        """Turrigiano-style synaptic scaling. Triggered by the neuron's OWN chronic firing rate
        (self.ca) leaving a target band, applies a FIXED multiplicative step (not a
        gradient, not global) to the excitatory resource, preserving relative
        weights so no pattern information is injected."""
        pos = self._weights_array > 0
        if not pos.any():
            return
        if self.homeo_budget is None:
            self.homeo_budget = float(self._weights_array[pos].sum())

        lo = self.ca_target * (1.0 - self.ca_band)
        hi = self.ca_target * (1.0 + self.ca_band)
        if self.ca < lo:
            self.homeo_budget *= (1.0 + self.homeo_up)      # chronically silent -> grow
        elif self.ca > hi:
            self.homeo_budget *= (1.0 - self.homeo_down)    # chronically over-active -> shrink
        if self.homeo_budget_min is not None:
            self.homeo_budget = max(self.homeo_budget, self.homeo_budget_min)
        if self.homeo_budget_max is not None:
            self.homeo_budget = min(self.homeo_budget, self.homeo_budget_max)

        total = float(self._weights_array[pos].sum())
        if total > 1e-9:
            self._weights_array[pos] *= self.homeo_budget / total
        self._weights_array = np.clip(self._weights_array, -self.weight_cap, self.weight_cap)

    def _decay_confidence(self):
        """Activity-dependent confidence decay."""
        if self._confidence is None:
            return
        if self.ca >= self.conf_ca_dead:
            self.inactive_steps = 0
            rho = self.conf_rho_active
        else:
            self.inactive_steps += 1
            rho = self.conf_rho_dead if self.inactive_steps >= self.conf_grace else 0.0
        if rho > 0.0:
            self._confidence *= (1.0 - rho)

    def plasticity_stats(self):
        """Summary statistics over the excitatory (positive-weight) afferents --
        weight/confidence entropy and concentration, plus current budget usage.
        See plasticity_stats on this class."""
        self._ensure_finalized()
        pos_mask = self._weights_array > 0
        pos_w = self._weights_array[pos_mask]
        pos_c = self._confidence[pos_mask]
        return dict(
            weight_entropy=_distribution_entropy(pos_w),
            confidence_entropy=_distribution_entropy(pos_c),
            weight_concentration=_concentration(pos_w),
            confidence_concentration=_concentration(pos_c),
            budget_used=float(pos_w.sum()),
        )
        
    def get_state(self):
        """
        Get current neuron state for monitoring/debugging.
        
        Returns:
            dict: Dictionary containing key state variables
        """
        self._ensure_finalized()
        
        return {
            'potential': self.potential,
            'refractory_timer': self.refractory_timer,
            'spiked': self.spiked,
            'weights': self._weights_array.copy() if self._weights_array is not None else np.array([]),
            'confidence': self._confidence.copy() if self._confidence is not None else np.array([]),
            'last_spike_time': self.last_spike_time
        }

    @property
    def weights(self):
        """Get the weights array."""
        self._ensure_finalized()
        return self._weights_array.copy()

    @weights.setter
    def weights(self, value):
        """Replace weights for fixed-fan-in setup and tests."""
        self._set_weights(value, finalized=True)

    @property
    def trace(self):
        """Archived eligibility trace, kept for inspection/test compatibility."""
        self._ensure_finalized()
        return self._trace.copy()

    @trace.setter
    def trace(self, value):
        arr = np.asarray(value, dtype=float)
        if self._weights_array is not None and len(arr) != len(self._weights_array):
            raise ValueError("trace length must match weights length")
        self._trace = arr.copy()

    @property
    def distance(self):
        """Per-afferent delivery distance d_i (aligned to weights). Used only for
        distance attenuation of the delivered drive; never changes stored weights."""
        self._ensure_finalized()
        return self._distance.copy() if self._distance is not None else None

    @distance.setter
    def distance(self, value):
        arr = np.asarray(value, dtype=float)
        if self._weights_array is not None and len(arr) != len(self._weights_array):
            raise ValueError("distance length must match weights length")
        self._distance = arr.copy()

    @property
    def confidence(self):
        """Get the per-synapse confidence array (aligned to weights)."""
        self._ensure_finalized()
        return self._confidence.copy()

    @confidence.setter
    def confidence(self, value):
        arr = np.asarray(value, dtype=float)
        if self._weights_array is not None and len(arr) != len(self._weights_array):
            raise ValueError("confidence length must match weights length")
        self._confidence = arr.copy()
        
    @property
    def n_inputs(self):
        """Get the number of input connections."""
        self._ensure_finalized()
        return len(self._weights_array)
