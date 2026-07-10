"""
SimulationEngine -- steppable wrapper around the spiking network.

Spikes are delivered immediately (no conduction delays). Every neuron has a
3D position for display; the L2 positions in L2_HOMES remain so the layout
is meaningful in the viewport, but they no longer imply any timing.

Learning architecture:
  - L1E neurons are treated as pre-trained pixel encoders: weights are fixed
    at [-1.0, 1.0] and learning_rate = 0.  They fire whenever the external
    pixel is active and they are not suppressed by their paired L1I neuron.
  - L2E neurons carry a homeostatic weight budget equal to the threshold.
    Only their positive (feedforward) incoming weights are counted; the
    inhibitory index-0 weight is excluded.  As learning strengthens active
    synapses, the budget normalisation weakens the others, producing
    competitive receptive-field emergence.
  - L1I / L2I (inhibitory) neurons carry NO budget.  Instead, each
    individual incoming weight is capped independently (no renormalization
    trades one synapse off against another), and BOTH follow the identical
    two-regime "assembly evidence integrator" policy, just scaled to their
    own layer's threshold (see the L2 / L1 competition note below for the
    full mechanism): every synapse is randomly initialized (never a repeated
    constant) at [0.25, 0.5] of its own neuron's threshold -- L1I from
    [0.25, 0.5] * thr_l1 (L1_EI_WEIGHT_INIT_LOW/_HIGH_FRAC), L2I from
    [0.25, 0.5] * threshold_l2 (L2_EI_WEIGHT_INIT_LOW/_HIGH_FRAC) -- and each
    synapse's LEARNING ceiling is that same neuron's own threshold (thr_l1 /
    threshold_l2), not a separate fixed constant. This lets timing dynamics
    train freely without distorting receptive fields.

With slow leak_l2 (~0.01) and small initial feedforward weights, L2E neurons
require many volleys to fire at first (classic LIF accumulation).  As
synapses specialise, they fire from a single volley (pattern integrator).
The charge-ring visualisation makes this transition directly observable.

L1 / L2 competition (source of winner-take-all, and of L1's feedback
suppression):
  Both inhibitory neurons -- L1I (one per pixel) and the shared L2I -- are
  "assembly evidence integrators" whose own behavior changes across training
  in two regimes, identically in mechanism, only differing in which
  threshold and which leak rate govern them (thr_l1/L1I_LEAK_RATE for L1I,
  threshold_l2/L2I_LEAK_RATE for L2I):
    EARLY -- each incoming synapse starts randomly well below that neuron's
    own threshold, and a fast membrane leak (much faster than the
    corresponding excitatory layer's leak_l1/leak_l2) sets a short
    evidence-retention window of a few volleys, so the inhibitory neuron only
    fires once several DIFFERENT excitatory neurons have spiked within that
    window and their contributions sum toward threshold -- a "round robin"
    phase where no single source is yet trusted. (The fraction/leak-rate pair
    is chosen so this ceiling -- w/(1-r) for a geometric series of
    volley-spaced contributions -- sits comfortably ABOVE threshold, not
    below it: raising a threshold without also scaling the weight range and/or
    leak rate can push the target past that ceiling entirely, making the
    round-robin phase permanently unreachable rather than merely slower --
    see L2I_Temporal_Integration.md for the full derivation.)
    LATER -- these synapses are not budgeted and each is individually capped
    at its own neuron's threshold (not a fixed sub-threshold ceiling), so as
    the existing, unmodified Hebbian rule repeatedly credits whichever
    excitatory neuron habitually co-occurs with a discharge, that synapse can
    grow all the way to threshold and become sufficient alone -- one spike
    from that now-trusted source then fires the inhibitory neuron
    immediately, the same LIF-accumulator -> pattern-integrator transition
    L2E itself goes through.
  Whenever L2I fires (either regime) it laterally inhibits the rest of the L2
  pool through the L2I->L2E gate (see Neuron.apply_inhibition); whenever an
  L1I fires it suppresses its paired L1E the same way. NOTE: L1I no longer uses
  the trainable-integrator regime by default -- with l1i_immediate_relay (default
  ON, see step() 2e) L1I is an immediate deterministic relay that fires on any
  nonzero L2E feedback, since the learned integrator was not useful and its
  accumulation introduced a feedback phase shift. The two-regime accumulator
  below still applies to L2I, and to L1I only when the relay flag is turned off. Subthreshold L2E
  neurons are left untouched, so charge accumulated across volleys is
  preserved and every unit can eventually win. Each L2I->L2E gate's strength
  is learned per neuron by the inhibitory-plasticity rule and saturates below
  threshold, so L2 competition self-organizes and cannot collapse to a
  permanent single winner. (An earlier version reset all non-winners to rest
  each step, which destroyed subthreshold evidence and locked the network to
  one neuron; a version after that let a single L2E spike fire L2I
  immediately from t=0 with no round-robin phase at all, and L1I was an
  instant integrator from t=0 the whole time -- see
  L2I_Temporal_Integration.md, now superseded by the two-regime design above,
  applied uniformly to both L1I and L2I.)
"""

from __future__ import annotations

import os
import sys
from collections import deque, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import InputLayer                       # noqa: E402
from cortical_column_flexible import CorticalColumn  # noqa: E402
from neuron_flexible import UNIT, LEAK_SCALE         # noqa: E402  fixed-point convention


PATTERNS = {
    'row 0':    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    'row 1':    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    'row 2':    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    'col 0':    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    'col 1':    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    'col 2':    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    'diag \\':  [1, 0, 0, 0, 1, 0, 0, 0, 1],
    'diag /':   [0, 0, 1, 0, 1, 0, 1, 0, 0],
}

# 3D layout for the 8 L2 output neurons — evenly spaced ring in the XY plane.
import math as _math
_R, _Z = 3.2, 4.0
L2_HOMES = [
    (round(_R * _math.cos(k * _math.pi / 4), 4),
     round(_R * _math.sin(k * _math.pi / 4), 4),
     _Z)
    for k in range(8)
]

N_PIX = 9
N_OUT = 8
GRID = 2.2
L2E_FANIN = 1 + N_PIX     # [local_I_placeholder, *pixels]
FREQ_WINDOW = 40
WEIGHT_EPS = 1e-6
LOG_MAX = 400

# Episode-based competition window (interpretation only -- see _update_episode).
# An episode groups the L2 spikes produced across one or more volley bursts, and
# the "winner" is resolved from that spike history only when the episode ends,
# instead of an instantaneous per-step argmax. These two knobs are the episode
# end conditions; neither touches learning, WTA, or membrane dynamics.
EPISODE_QUIET_K = 5    # Condition A: end after this many consecutive L2-silent steps (spec: 3-5)
EPISODE_MAX_LEN = 12   # Condition B: hard cap on episode length in steps (spec: 8-12)

# L2 lateral inhibition ("adaptive gate") parameters. Competition in L2 is
# produced by the shared inhibitory neuron L2I suppressing near-winners through
# the L2I->L2E synapse, NOT by a procedural hard reset. Each L2E owns one gate
# from L2I whose strength is learned by the inhibitory-plasticity rule
# (Neuron.apply_inhibition): it grows when it suppresses a neuron that was close
# to firing and saturates at L2_GATE_WMAX. These values were chosen by a
# parameter sweep (see the competition investigation) as the point giving the
# broadest participation with the most per-pattern differentiation; crucially
# L2_GATE_WMAX < thr_l2 so a saturated gate can't fully reset the membrane.
#
# FIXED-POINT SCALE (see neuron_flexible.py): the model runs at the
# UNIT-scaled fixed-point scale, so thresholds are the integers 1000 / 8000 and
# all charge-like magnitudes are scaled by UNIT. Scaling rules by role:
#   - LINEAR magnitudes (gate init, potentials, weights, clips) scale by UNIT.
#   - the learning rate eta scales by UNIT (dw must scale with w).
#   - the QUADRATIC saturation denominator w_max in dw = eta*p*(1 - w^2/w_max)
#     scales by UNIT**2, because w^2 scales by UNIT**2. The gate MAGNITUDE still
#     settles at its natural equilibrium sqrt(w_max) = UNIT*sqrt(1.5) ~= 1225,
#     which is still well below thr_l2 = 8000 -- the "< threshold, partial
#     discharge" property is preserved; only the numeric denominator is large.
L2_GATE_INIT = -500          # initial gate magnitude 500 == 0.5 * UNIT (linear)
L2_GATE_WMAX = 1500 * UNIT   # quadratic w_max == 1.5 * UNIT**2; equilibrium sqrt ~= 1225 (< thr_l2)
L2_GATE_ETA = 100            # inhibitory-plasticity eta == 0.1 * UNIT (scales with charge)

# L2E->L2I "assembly evidence" synapses.
#
# Two regimes, by design, not by accident:
#   EARLY in training, no single L2E->L2I synapse is anywhere near enough on
#   its own (see L2_EI_WEIGHT_INIT_LOW/_HIGH_FRAC: the random init range is
#   well below threshold_l2), so L2I only fires once several distinct L2E neurons
#   have spiked within its short retention window (L2I_LEAK_RATE). This is
#   what produces the initial "round robin": no one source is trusted yet, so
#   competition is decided by population consensus, not a single vote.
#   OVER TRAINING, the L2E->L2I weights are NOT budgeted (weight_budget=None,
#   homeostasis off for L2I) and each synapse's cap is the neuron's OWN
#   threshold_l2 (not a fixed sub-threshold ceiling) -- so as the existing,
#   unmodified Hebbian rule repeatedly credits the synapse from whichever L2E
#   habitually wins (and therefore habitually co-occurs with L2I's discharge
#   cycle: L2E fires -> L2I fires -> that L2E gets inhibited, repeat), that
#   one synapse can grow all the way up to threshold_l2 and become
#   sufficient on its own: once it has, a single spike from that specific,
#   now-trusted source fires L2I immediately, matching L2E's own transition
#   from many-volley LIF accumulation to single-volley pattern-integrator
#   firing (see the module docstring). The temporal-integration REQUIREMENT
#   (multiple contributors) is therefore a property of an undertrained
#   synapse, not a permanent structural ceiling -- see L2I_Temporal_Integration.md
#   for the original (now superseded) design that made it permanent.
# L2I_LEAK_RATE is still much faster than leak_l2 (L2E's slow, many-volley
# accumulator): L2I needs a SHORT retention window so that only spikes from
# roughly the same "burst" of population activity sum together while the
# synapses are still weak, while isolated single-neuron spikes decay away
# before a second volley's worth of unrelated evidence could pad them out.
#
# The init range is expressed as a FRACTION of threshold_l2, not an absolute
# constant, because the achievable ceiling of the round-robin phase depends on
# both together: with volley-spaced contributions decaying by r=(1-leak)^4
# per gap, n same-size contributions sum to at most w/(1-r) as n -> infinity
# (a geometric series). With L2I_LEAK_RATE=0.07, r=(0.93)^4~=0.748, so that
# ceiling is w/0.252 ~= 3.97*w -- population consensus can only ever ratchet
# threshold_l2 to just under 4*(the init weight), REGARDLESS of how many
# volleys pass. The mean init weight must therefore scale with threshold_l2
# (not stay fixed) or raising the threshold alone can make the ceiling fall
# below threshold_l2, at which point L2I could never fire at all via the
# round-robin route -- and since L2I's own E->I weights only update on ITS
# OWN fire, it would then never learn either (a silently dead network, not
# merely a slow one). [0.25, 0.5] * threshold_l2 keeps the same relative
# dynamics validated in L2I_Temporal_Integration.md (a single contribution at
# ~37.5% of threshold; 4 volley-spaced contributions needed to cross) at
# whatever threshold_l2 is actually configured to.
# Init fractions as integer rationals (1/4, 1/2) per the fixed-point convention.
L2_EI_WEIGHT_INIT_LOW_FRAC = 1 / 4   # low end of the random E->I init range,
L2_EI_WEIGHT_INIT_HIGH_FRAC = 1 / 2  # high end -- both as a fraction of threshold_l2.
                                     # The LEARNING ceiling is threshold_l2 itself
                                     # (see the per-neuron cap assignment below),
                                     # not this range -- this is an INITIAL value only.
L2I_LEAK_RATE = 70 / LEAK_SCALE      # L2I's own membrane/trace leak (>> leak_l2); 70/1000 == 0.07

# L2E->L1I feedback weights (the "quiet the inputs" loop). Governed by
# EXACTLY the same two-regime "assembly evidence" policy as L2E->L2I above,
# scaled to L1I's own threshold (thr_l1) instead of threshold_l2 -- L1I was
# previously an instant integrator from t=0 (every synapse initialized
# at-or-above its own threshold, so a single L2E spike always fired it
# immediately, with no round-robin phase and nothing left for the existing
# Hebbian rule to actually move). Now: synapses start randomly in
# [0.25, 0.5] * thr_l1 (L1_EI_WEIGHT_INIT_LOW/_HIGH_FRAC -- same fractions as
# L2I, so the same relative dynamics apply: ~37.5% of threshold from one
# spike, 4 volley-spaced spikes needed to cross), are NOT weight-budgeted,
# and are individually capped at thr_l1 itself, so a habitually-participating
# source can learn its way up to full self-sufficiency exactly like L2I does.
# L1I_LEAK_RATE (== L2I_LEAK_RATE numerically, kept as its own named constant
# since it governs a different neuron) gives the same ~1.49x reachable-ceiling
# headroom above threshold that L2I_LEAK_RATE gives L2I -- see the
# L2_EI_WEIGHT_INIT_LOW_FRAC note above for the derivation (it is scale-free:
# depends only on the weight/threshold fraction and the leak rate, not the
# absolute threshold value, so the identical fractions and leak rate carry
# over unchanged from threshold_l2=8 to thr_l1=1).
L1_EI_WEIGHT_INIT_LOW_FRAC = 1 / 4   # low end of the random L2E->L1I init range,
L1_EI_WEIGHT_INIT_HIGH_FRAC = 1 / 2  # high end -- both as a fraction of thr_l1.
                                      # Learning ceiling is thr_l1 itself (see the
                                      # per-neuron cap assignment below), not this.
L1I_LEAK_RATE = 70 / LEAK_SCALE      # L1I's own membrane/trace leak (>> leak_l1); 70/1000 == 0.07

# Learning rate for the charge-based excitatory rule (Neuron._update_weights),
# used by every positive-weight population: L1I's incoming weights, L2I's
# incoming weights, and L2E's feedforward weights. Expressed as a FRACTION of
# each neuron's OWN weight_cap (learning_rate = ETA_FRAC * weight_cap), not a
# shared absolute constant -- with an absolute constant, L1I (cap=thr_l1=1)
# and L2I (cap=threshold_l2=8) would take a wildly different NUMBER OF EVENTS
# to reach self-sufficiency even though they're meant to behave with the same
# relative pace (this was the actual cause of L1I "training too fast" /
# L2I's weights climbing too high: same absolute dw per event, an 8x smaller
# target). Scaling eta by weight_cap makes the FRACTIONAL growth per event
# (dw/w_max) independent of the neuron's absolute scale.
#
# ETA_FRAC=0.01 was chosen by simulating the saturating recursion
# f_{n+1} = f_n + ETA_FRAC*p*(1-f_n^2) (f = w/w_max) from the init-range
# midpoint (f=0.375) with a representative p~=0.6: it takes ~180 events for a
# habitually-participating synapse to reach 90% of its own cap -- a genuinely
# long "gatling gun" round-robin phase, not a handful of volleys. See
# Weight_Update_Unification.md for the full derivation and validation.
ETA_FRAC = 0.01

# Floor for L2E's feedforward (positive) weights after budget renormalization
# (see min_positive_weight in Neuron._apply_budget_and_cap). An order
# of magnitude below the feedforward init range (ff_weights ~ uniform[0.05,
# 0.20] below) so a pattern the neuron hasn't seen in a while still delivers
# some nonzero charge instead of literally none.
L2E_MIN_WEIGHT_FLOOR = 10   # fixed-point weight floor; 10 == 0.01 * UNIT (linear)

# L2E feedforward weight budget as a multiple of threshold_l2. The old 1x forced
# every neuron to the SAME total weight, flattening winner-vs-rival margins and
# causing pattern collisions (7/8 distinct). Loosening to 2x lets specialists
# separate -> a clean 8/8 distinct map with 0 dead units, at no dominance cost.
# See memory note one-to-one-and-lasting-inhibition for the frontier this sits on.
L2E_BUDGET_MULT = 2


class SimulationEngine:
    def __init__(self, seed: int = 1,
                 # Fixed-point scale (see neuron_flexible.py): the model
                 # runs at the UNIT-scaled scale, so thresholds are INTEGERS and
                 # every charge-like magnitude (potentials, weights, gate, clips)
                 # is scaled by UNIT. Leak rates stay dimensionless ratios stored
                 # as integer numerators over LEAK_SCALE.
                 threshold: int = 1 * UNIT,           # 1000
                 threshold_l2: int = 8 * UNIT,        # 8000
                 leak_l1: float = 100 / LEAK_SCALE,   # 0.10
                 leak_l2: float = 10 / LEAK_SCALE,    # 0.01
                 learning_rate: float = 0.05,
                 weight_cap: int = 1 * UNIT,          # 1000
                 refractory: int = 2,
                 volley_period: int = 4,
                 input_period: int | None = None,
                 cycle_period: int | None = None,
                 homeostasis: bool = False,
                 ca_rate: float = 0.01,
                 ca_target: float = 0.012,   # between a specialist's rate (~0.01) and a
                                             # tyrant's (~0.02): below it a unit is grown,
                                             # above it a unit is shrunk (see sweep in memory)
                 homeo_up: float = 0.01,
                 homeo_down: float = 0.01,
                 l2e_lr_frac: float | None = None,
                 l2i_lr_frac: float | None = None,
                 l1i_lr_frac: float | None = None,
                 l2_gate_eta: float | None = None,
                 l2i_threshold_frac: float = 1,   # unit multiplier (1 == unchanged)
                 l1i_threshold_frac: float = 1,   # unit multiplier (1 == unchanged)
                 ei_sat_mult: float = 1,          # unit multiplier (1 == unchanged)
                 l1i_ei_init_frac: float | None = None,
                 # Confidence-gated consolidation (see Claude_Confidence_Consolidation_Plan.md
                 # and neuron_flexible.Neuron). Enabled on L2E by default: a local, label-free
                 # consolidation rule that protects mature specialists, depresses
                 # losing near-winners, and keeps stale neurons reusable. All the
                 # numeric knobs below are DIMENSIONLESS (scale-invariant) except
                 # conf_cap, which is a weight and scales with thr_l2 via conf_cap_frac.
                 confidence_consolidation: bool = True,
                 loser_depression: bool = True,
                 conf_cap_frac: float = 1 / 3,   # effective mature gate value = frac * thr_l2
                 conf_beta: float = 0.05,        # confidence EMA rate toward maturity
                 eta_min: float = 0.05,          # plasticity floor fraction for mature gates
                 eta_loss: float = 0.01,         # loser-depression rate (dimensionless)
                 loss_gamma: float = 2,          # protect-small / punish-large exponent
                 conf_rho_active: float = 1e-5,  # confidence decay while recently active
                 conf_rho_dead: float = 1e-3,    # confidence decay once dead past grace
                 conf_ca_dead: float = 0.002,    # ca below this counts the neuron inactive
                 conf_grace: int = 5000,         # inactive steps before dead-decay engages
                 # Signed-spike depression ("4a"): OFF pixels deliver no charge but
                 # their positive gates are depressed on fire. L2E only; default off
                 # so the baseline is untouched. eta_off is the depression rate.
                 # ON by default: sharpens receptive-field margins, which is what
                 # keeps the 8 specialists distinct AND lifts sustained-presentation
                 # dominance (0.23->0.36) with no measured cost. eta_off is the rate.
                 signed_depression: bool = True,
                 eta_off: float = 0.20,          # OFF-gate depression rate (dimensionless)
                 # L2E feedforward weight budget (sum-renormalization competition).
                 # DEFAULT OFF: the project moved to the signed-spike rule below,
                 # whose -1 signal on inactive inputs supplies the downward pressure
                 # the budget used to impose (see signed_spike_learning). Under the
                 # signed rule this flag is inert anyway (_update_weights returns
                 # before _apply_budget_and_cap); it is kept as a knob so the older
                 # budget/charge regime can still be reconstructed for comparison.
                 l2e_budget: bool = False,
                 # Event-driven firing: resolve L2 competition EVERY step -- pick the
                 # single argmax winner among the threshold-crossers and inhibit the
                 # rest, once per timestep. DEFAULT ON: this per-step single-winner
                 # flow is the canonical L2 procedure. Turn OFF to fall back to the
                 # cycle-quantized regime (resolve the same argmax competition only on
                 # the intrinsic cycle boundary, once per cycle_period steps), which
                 # decouples winner timing from the input rate -- kept as a knob so
                 # that rate-decoupling regime can still be reconstructed for A/B.
                 event_driven: bool = True,
                 # L2 feedforward charge granularity: deliver this step's L1->L2E
                 # drive in K equal chunks (weight_ji/K per active synapse) WITHIN a
                 # frozen outer timestep, re-running the argmax WTA after each chunk
                 # and stopping at the first chunk that produces a threshold-crosser
                 # (consolidation-first: the earliest strong responder wins before
                 # rivals pile up charge). The clock does not advance and no
                 # leak/update runs between chunks. K=1 (default) delivers the full
                 # drive in one chunk and reproduces the un-chunked behavior exactly.
                 l2_charge_chunks: int = 1,
                 # L1I feedback firing mode. DEFAULT ON: L1I acts as an IMMEDIATE
                 # DETERMINISTIC RELAY -- any L1I that receives a nonzero L2E
                 # feedback signal fires in that same step, with NO membrane
                 # accumulation, NO learned-threshold crossing, and NO dependence on
                 # L1I feedback-weight training (the learned integrator introduced a
                 # phase shift and was not useful). Turn OFF to restore the trainable
                 # threshold-integrating L1I (fire only when accumulated feedback
                 # crosses L1I's own threshold). Either way L1I output stays binary
                 # and the L1I->L1E inhibition delivery path is unchanged.
                 l1i_immediate_relay: bool = True,
                 # Sparse excitatory FLOW-RATE accumulation (opt-in; see Neuron and
                 # step(), and the flow-rate section of the methodology doc). DEFAULT
                 # OFF -> instantaneous V += dot(weights, spikes). When ON, an input
                 # spike opens a decaying excitatory current trace that integrates
                 # into V over time (weight = current amplitude, not a charge packet)
                 # for the positive-charge integrators L2E / L2I / L1I (NOT L1E, the
                 # abstract sensory source; NOT L1I in immediate-relay mode). Flow
                 # mode is the finer temporal model, so it FORCES effective
                 # l2_charge_chunks = 1 (chunking is ignored while it is on).
                 excitatory_flow_rate: bool = True,
                 exc_trace_decay: float = 0.8,        # per-timestep current decay d
                 exc_trace_normalized: bool = True,   # inject drive*(1-d) so total ~= drive
                 # Flow-proportional assembly credit for the E->I integrators (L2I/L1I).
                 # On the inhibitory neuron's OWN fire, credit every incoming positive
                 # synapse in proportion to the flow it delivered over the retention
                 # window (per-synapse leaky trace), normalized so the dominant driver
                 # gets the full learning rate; non-contributors decay toward the floor.
                 # Breaks the L2I firing deadlock (a habitual winner's E->I synapse
                 # climbs to self-sufficiency instead of only the threshold-crossing
                 # spike taking credit). Default OFF -> legacy last-volley E->I credit.
                 assembly_flow_credit: bool = False,
                 assembly_decay_frac: float = 0.5,    # down-pressure on non-contributing gates
                 # Inhibitory FLOW (opt-in; symmetric to the excitatory flow). When on,
                 # a real L2I->L2E discharge injects the gate into a decaying inhibitory
                 # CURRENT that drains the target's charge out over ~1/(1-decay) steps
                 # (sustained suppression) instead of a one-shot subtraction. Applies
                 # to the same neurons as the excitatory flow (L2E/L2I/L1I, not L1E).
                 inhibitory_flow_rate: bool = False,
                 inh_trace_decay: float = 0.8,
                 inh_trace_normalized: bool = True,
                 # Inhibitory-gate plasticity rule (see Neuron.apply_inhibition).
                 # inhibitory_delta_rule False = legacy saturating rule (every L2I->L2E
                 # gate converges to the same ceiling sqrt(w_max) -> uniform). True
                 # (DEFAULT) = a differentiating local rule selected by
                 # inhibitory_rule_mode:
                 #   "turnover" (DEFAULT): on each discharge the NORMALIZED gate
                 #     u = w/G updates as du = eta_up*p_t*(1-u) - eta_down*u, with
                 #     p_t = clamp(v_pre/theta, 0, p_max). Purely event-local (no target
                 #     voltage, no averages): high-charge rivals accumulate stronger
                 #     gates while weak/dead targets drift down via size-proportional
                 #     turnover.
                 #   "margin": diagnostic; relax toward s = clamp(v_pre-margin*theta,0,G).
                 inhibitory_delta_rule: bool = True,
                 inhibitory_rule_mode: str = "turnover",
                 inhibitory_eta_up: float = 0.02,       # turnover strengthening rate
                 inhibitory_eta_down: float = 0.005,    # turnover size-proportional decay
                 inhibitory_p_max: float = 1.0,         # turnover cap on p_t = v_pre/theta
                 inhibitory_margin_frac: float = 0.5,   # margin mode: target_post = frac*theta
                 inhibitory_delta_eta: float = 0.05,    # margin mode: EMA rate
                 # Distance attenuation of DELIVERED excitatory drive (opt-in; see
                 # Neuron.receive_input and the methodology doc). When on, each
                 # afferent's delivered amplitude is multiplied by
                 # (distance_ref/max(d_i, distance_min))^distance_power -- weight =
                 # learned gate, distance = delivery attenuation, trace = temporal
                 # flow. It does NOT change stored weights or the trace math, and is
                 # not entangled with chunking. OFF by default (per-synapse distances
                 # default to 1.0 = no attenuation until real functional distances
                 # are provided). Applies in both flow-rate and instantaneous modes.
                 distance_weighting: bool = False,
                 distance_power: float = 2.0,
                 distance_ref: float = 1.0,
                 distance_min: float = 1.0,
                 # Lasting inhibition: replace the one-shot lateral discharge with a
                 # LEAKY decaying inhibitory field. L2I (a leaky integrator) pumps a
                 # shared field when it fires; the field hyperpolarizes the whole L2E
                 # pool every step and decays with inh_decay. A pattern's specialist
                 # keeps the field pumped (rivals stay suppressed for a finite window
                 # = sustained ownership); when the pattern changes the field decays
                 # and the new specialist takes over. inh_boost_frac scales the per-
                 # fire field increment as a fraction of threshold_l2.
                 lasting_inhibition: bool = False,
                 inh_decay: float = 0.15,
                 inh_boost_frac: float = 1.0,
                 # Reset-by-subtraction on L2E fire: on spike, potential -= threshold
                 # (floored at rest) instead of a full reset to rest. Standard LIF;
                 # leaves the winner its residual overshoot like partially-inhibited
                 # losers keep theirs, directly attacking the discharge asymmetry
                 # that drives the round-robin (see AGENT_HANDOFF.md sec 5-6). L2E
                 # only; default off so the baseline is untouched.
                 subtractive_reset: bool = False,
                 # Membrane saturation ceiling for L2E, as a multiple of thr_l2
                 # (None = unbounded, the baseline). Bounds accumulated charge so
                 # the membrane can't ratchet to ~2-3x threshold between cycles;
                 # this keeps the (small, capped) L2I->L2E gate in a range where
                 # it can actually regulate firing frequency -- inhibition, not a
                 # pile-up, sets who fires. Local finite driving force; L2E only.
                 v_sat_frac: float | None = None,
                 # Learnable L2I->L2E gate equilibrium, as a fraction of thr_l2
                 # (None = module default, ~0.15*thr). The gate grows by the
                 # inhibitory rule and settles at sqrt(w_max); this sets that
                 # target. Raising it toward/above 1.0 lets a fired owner's L2I
                 # discharge FLOOR its rivals to rest (they lose all their charge
                 # and cannot fire) -- the inhibitory winner-take-all lockout that
                 # makes ongoing plasticity safe (only the owner fires, so only
                 # the owner learns). Pair with v_sat so a rival's bounded charge
                 # is within reach of the gate. L2E only.
                 l2_gate_eq_frac: float | None = None,
                 # Minimal SIGNED-SPIKE feedforward learning (see
                 # Claude_Minimal_Signed_Spike_Learning_Prompt.md). Replaces the
                 # potentiation + OFF-depression + confidence + budget stack with a
                 # single local signed rule: on fire, active inputs (+1) potentiate
                 # and inactive inputs (-1) depress, dw = eta*p*(1-(w/w_cap)^2)*sig,
                 # bounded to [min_positive_weight, weight_cap], NO budget. Intended
                 # to be run with the compensating mechanisms off and refractory=0
                 # (the "minimal experiment"). L2E only. DEFAULT ON: this is now the
                 # project's canonical L2E learning rule. When on it takes over
                 # _update_weights entirely (returns early), so l2e_budget,
                 # confidence_consolidation, loser_depression and signed_depression
                 # are all bypassed for L2E regardless of their own defaults.
                 signed_spike_learning: bool = True,
                 # Capacity rule for the minimal experiment (see the prompt's
                 # "Threshold, Cap, Floor" section). l2e_weight_cap_frac sets each
                 # L2E positive feedforward weight cap to frac*thr_l2, so three
                 # maximally strong active afferents reach threshold (1/3 for the
                 # 3-pixel line patterns). Default 1.0 preserves the prior cap.
                 l2e_weight_cap_frac: float = 1.0,
                 # Positive-afferent weight FLOOR (L2E feedforward and E->I positive
                 # afferents). None keeps legacy behaviour (L2E floor =
                 # L2E_MIN_WEIGHT_FLOOR, E->I unfloored); the minimal experiment
                 # sets 1. Negative inhibitory gates are never floored here -- they
                 # are bounded by magnitude in apply_inhibition.
                 pos_weight_floor: int | None = None):
        # Decouple the SENSORY input rate from the INTRINSIC competition clock.
        # input_period: steps between external input bursts (how fast the world
        #   throws spikes at the network -- "nature" is slow, a "rave" is fast).
        # cycle_period: the intrinsic gamma-like clock that structures L2
        #   competition and L2I evidence integration. FIXED regardless of the
        #   input rate, so which pattern wins does not depend on how fast input
        #   arrives. Both default to volley_period, which reproduces the old
        #   input-locked behavior exactly (input_period == cycle_period), so
        #   existing callers/tests are unchanged.
        input_period = volley_period if input_period is None else input_period
        cycle_period = volley_period if cycle_period is None else cycle_period
        # Separate learning-rate controls for the E and I populations (Phase 1 of
        # the symmetry-breaking plan). Each defaults to the module constant that
        # was previously applied uniformly, so the defaults reproduce prior
        # behavior exactly. l2i_threshold_frac / l1i_threshold_frac (Phase 2)
        # scale each inhibitory neuron's OWN threshold; their E->I init range and
        # learning cap scale with that same (possibly lowered) threshold below,
        # so lowering an I threshold does not trivially overdrive it.
        l2e_lr_frac = ETA_FRAC if l2e_lr_frac is None else l2e_lr_frac
        l2i_lr_frac = ETA_FRAC if l2i_lr_frac is None else l2i_lr_frac
        l1i_lr_frac = ETA_FRAC if l1i_lr_frac is None else l1i_lr_frac
        l2_gate_eta = L2_GATE_ETA if l2_gate_eta is None else l2_gate_eta
        self.params = dict(seed=seed, threshold=threshold, threshold_l2=threshold_l2,
                           leak_l1=leak_l1, leak_l2=leak_l2,
                           learning_rate=learning_rate, weight_cap=weight_cap,
                           refractory=refractory, volley_period=volley_period,
                           input_period=input_period, cycle_period=cycle_period,
                           homeostasis=homeostasis, ca_rate=ca_rate, ca_target=ca_target,
                           homeo_up=homeo_up, homeo_down=homeo_down,
                           l2e_lr_frac=l2e_lr_frac, l2i_lr_frac=l2i_lr_frac,
                           l1i_lr_frac=l1i_lr_frac, l2_gate_eta=l2_gate_eta,
                           l2i_threshold_frac=l2i_threshold_frac,
                           l1i_threshold_frac=l1i_threshold_frac,
                           ei_sat_mult=ei_sat_mult,
                           l1i_ei_init_frac=l1i_ei_init_frac,
                           confidence_consolidation=confidence_consolidation,
                           loser_depression=loser_depression,
                           conf_cap_frac=conf_cap_frac, conf_beta=conf_beta,
                           eta_min=eta_min, eta_loss=eta_loss, loss_gamma=loss_gamma,
                           conf_rho_active=conf_rho_active, conf_rho_dead=conf_rho_dead,
                           conf_ca_dead=conf_ca_dead, conf_grace=conf_grace,
                           signed_depression=signed_depression, eta_off=eta_off,
                           l2e_budget=l2e_budget, event_driven=event_driven,
                           l2_charge_chunks=l2_charge_chunks,
                           l1i_immediate_relay=l1i_immediate_relay,
                           excitatory_flow_rate=excitatory_flow_rate,
                           exc_trace_decay=exc_trace_decay,
                           exc_trace_normalized=exc_trace_normalized,
                           assembly_flow_credit=assembly_flow_credit,
                           assembly_decay_frac=assembly_decay_frac,
                           inhibitory_flow_rate=inhibitory_flow_rate,
                           inh_trace_decay=inh_trace_decay,
                           inh_trace_normalized=inh_trace_normalized,
                           inhibitory_delta_rule=inhibitory_delta_rule,
                           inhibitory_rule_mode=inhibitory_rule_mode,
                           inhibitory_eta_up=inhibitory_eta_up,
                           inhibitory_eta_down=inhibitory_eta_down,
                           inhibitory_p_max=inhibitory_p_max,
                           inhibitory_margin_frac=inhibitory_margin_frac,
                           inhibitory_delta_eta=inhibitory_delta_eta,
                           distance_weighting=distance_weighting,
                           distance_power=distance_power,
                           distance_ref=distance_ref, distance_min=distance_min,
                           lasting_inhibition=lasting_inhibition, inh_decay=inh_decay,
                           inh_boost_frac=inh_boost_frac,
                           subtractive_reset=subtractive_reset,
                           v_sat_frac=v_sat_frac,
                           l2_gate_eq_frac=l2_gate_eq_frac,
                           signed_spike_learning=signed_spike_learning,
                           l2e_weight_cap_frac=l2e_weight_cap_frac,
                           pos_weight_floor=pos_weight_floor)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        p = self.params
        rng = np.random.default_rng(p['seed'])
        thr_l1 = p['threshold']      # L1 neurons fire on a single pixel hit
        thr_l2 = p['threshold_l2']   # L2 neurons must accumulate many volleys
        # Inhibitory-neuron thresholds (Phase 2): each defaults to its excitatory
        # layer's threshold (frac 1.0 -> unchanged). When lowered, the E->I init
        # range and learning cap below scale with THIS threshold, not the E one.
        thr_l2i = thr_l2 * p['l2i_threshold_frac']   # L2I's own firing threshold
        thr_l1i = thr_l1 * p['l1i_threshold_frac']   # each L1I's own firing threshold

        self.l1 = InputLayer(n_neurons=N_PIX, threshold=thr_l1,
                             refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                             weight_cap=thr_l1, leak_rate=p['leak_l1'],
                             n_feedback_inputs=N_OUT)
        # L1E: pre-trained pixel encoders — fixed weights, no learning.
        # Fixed-point scale: gate -1*UNIT, excitatory drive +1*UNIT, so one pixel
        # spike delivers UNIT charge == thr_l1 and fires the encoder in one hit.
        for e in self.l1.excitatory_neurons:
            e.weights = np.array([-1.0, 1.0]) * UNIT
            e.learning_rate = 0.0
            e.weight_budget = None
            # The local-inhibition gate (index 0, magnitude UNIT) is meant to sit
            # frozen at saturation: unscaled its quadratic factor 1 - w^2/w_max was
            # exactly 0 (w = w_max = weight_cap). At the UNIT scale that needs
            # w_max = UNIT*weight_cap == weight_cap^2, so the gate still can't drift.
            e.inhibitory_weight_cap = UNIT * e.weight_cap
        # L1I: incoming (L2E->L1I) weights start randomly in [0.25, 0.5] * thr_l1
        # -- well below L1I's own threshold, not at-or-above it -- so L1I
        # behaves as a genuine temporal integrator early on (several distinct
        # L2E spikes needed) exactly like L2I, rather than an instant relay.
        # See the L1_EI_WEIGHT_INIT_LOW_FRAC note above for the full
        # round-robin -> single-trusted-source rationale (identical mechanism
        # to L2I, just scaled to thr_l1). The per-synapse LEARNING ceiling is
        # thr_l1 itself, set per-neuron below.
        # L2E->L1I feedback init. Default (None) uses the [0.25,0.5] round-robin
        # integrator range. Setting l1i_ei_init_frac initializes EVERY feedback
        # synapse at that fraction of L1I's threshold -- at >= 1.0 a single spike
        # from ANY L2E winner fires EVERY L1I, giving synchronous global input
        # suppression (recognition -> quiet all inputs) instead of the per-winner
        # single-relay lockout that leaves most L1I subthreshold. The weights are
        # unbudgeted so they stay pinned there.
        if p['l1i_ei_init_frac'] is None:
            lo_frac, hi_frac = L1_EI_WEIGHT_INIT_LOW_FRAC, L1_EI_WEIGHT_INIT_HIGH_FRAC
        else:
            lo_frac = hi_frac = p['l1i_ei_init_frac']
        for inh in self.l1.inhibitory_neurons:
            inh.threshold = thr_l1i    # Phase 2: L1I's own (possibly lowered) threshold
            inh.weights = rng.uniform(lo_frac * thr_l1i, hi_frac * thr_l1i, size=N_OUT)
            inh.leak_rate = L1I_LEAK_RATE

        self.l2 = CorticalColumn(n_neurons=N_OUT, threshold=thr_l2,
                                 refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                                 weight_cap=thr_l2, leak_rate=p['leak_l2'])
        self.l2.setup_connectivity(n_feedforward_inputs=N_PIX, n_feedback_inputs=0)
        self.l2.finalize_connections()
        # L2I->L2E gates start weak; they are the real source of L2 competition
        # (see step 2c) and self-tune via inhibitory plasticity.
        self.l2.set_local_inhibition_weights(L2_GATE_INIT)
        # E→I weights start randomly in [0.25, 0.5] * threshold_l2 -- not a
        # repeated constant -- so the 8 sources aren't artificially tied at
        # t=0, and every value in that range is still well below threshold_l2
        # so early competition needs several distinct contributors (see the
        # L2_EI_WEIGHT_INIT_LOW_FRAC note above for the full round-robin ->
        # single-trusted-source rationale, and why this MUST scale with
        # threshold_l2 rather than being a fixed absolute constant). The
        # LEARNING ceiling for these synapses is threshold_l2 itself, set
        # per-neuron below -- much higher than this init range -- so growth
        # can carry a habitually-participating synapse to self-sufficiency.
        self.l2.set_lateral_excitation_weights(
            rng.uniform(L2_EI_WEIGHT_INIT_LOW_FRAC * thr_l2i,
                        L2_EI_WEIGHT_INIT_HIGH_FRAC * thr_l2i, size=N_OUT))
        # Small positive feedforward weights: neurons must accumulate across many
        # volleys initially (LIF phase), then specialise toward single-volley
        # firing (pattern integrator phase). Plain uniform random init (linear /
        # fixed-point scale, 50..200 == 0.05..0.20 * UNIT).
        ff_weights = rng.uniform(50, 200, size=(N_OUT, N_PIX))
        self.l2.set_feedforward_weights(ff_weights)
        self.l2.inhibitory_neuron.refractory_period = 0
        self.l2.inhibitory_neuron.threshold = thr_l2i   # Phase 2: L2I's own threshold
        # Short evidence-retention window: much faster than L2E's leak_l2 (its
        # slow multi-volley accumulator). See L2I_LEAK_RATE derivation above.
        self.l2.inhibitory_neuron.leak_rate = L2I_LEAK_RATE

        self.neurons: dict[str, object] = {}
        self.meta: dict[str, dict] = {}
        self._register_neurons()
        # Phase 2: reflect the (possibly lowered) inhibitory thresholds in meta
        # so the dashboard/serialization show each I neuron's real threshold.
        self.meta['L2I']['threshold'] = thr_l2i
        for i in range(N_PIX):
            self.meta[f'L1I{i}']['threshold'] = thr_l1i

        # Budget / cap assignment:
        #   L2E → budget = thr_l2 (positive feedforward weights only).
        #   L1E → no budget (weights fixed, learning disabled).
        #   L1I → cap = thr_l1 itself; L2I → cap = thr_l2 itself. Neither is a
        #   fixed sub-threshold constant: each synapse starts far below its
        #   own neuron's threshold (round-robin phase) but is free to learn
        #   all the way up to it, at which point that one synapse alone is
        #   sufficient (see the L1_EI_*/L2_EI_* notes above -- identical
        #   mechanism for both, just scaled to a different threshold).
        #   Neither L1I nor L2I is weight-budgeted (weight_budget stays None,
        #   homeostasis stays off below) -- each incoming synapse is capped
        #   independently, with no renormalization forcing them to trade off
        #   against each other.
        for nid, n in self.neurons.items():
            if self.meta[nid]['type'] == 'E' and nid.startswith('L2'):
                n.weight_budget = L2E_BUDGET_MULT * thr_l2 if p['l2e_budget'] else None
                # Per-afferent capacity rule (minimal experiment): the positive
                # feedforward weight cap is a fraction of THIS E neuron's threshold
                # (l2e_weight_cap_frac * thr_l2), so three maximally strong active
                # afferents can reach threshold (frac=1/3 for the 3-pixel lines).
                # learning_rate and excitatory_saturation_cap below auto-follow the
                # cap. Default 1.0 reproduces the prior cap == thr_l2.
                n.weight_cap = p['l2e_weight_cap_frac'] * thr_l2
                # Floor for feedforward weights: without this, heavy training on
                # one pattern erodes every OTHER pixel's weight toward 0 (budget
                # renormalization rescales participating and non-participating
                # synapses alike on every update), so switching patterns later
                # accumulates no charge even though L1E still fires correctly.
                # L2E_MIN_WEIGHT_FLOOR is an order of magnitude below the
                # feedforward init range (ff_weights ~ [0.05, 0.20]) -- small
                # enough not to distort normal competition, nonzero enough that
                # every pixel keeps some baseline responsiveness.
                n.min_positive_weight = (p['pos_weight_floor'] if p['pos_weight_floor'] is not None
                                         else L2E_MIN_WEIGHT_FLOOR)
                # Adaptive lateral-inhibition gate: dedicated (lower) saturation
                # ceiling and its own learning rate, independent of feedforward.
                # l2_gate_eq_frac (opt-in) retargets the gate equilibrium to
                # frac*thr_l2 (w_max = equilibrium**2, since equilibrium=sqrt(w_max));
                # raising it lets the discharge floor rivals for a WTA lockout.
                if p['l2_gate_eq_frac']:
                    n.inhibitory_weight_cap = (p['l2_gate_eq_frac'] * thr_l2) ** 2
                else:
                    n.inhibitory_weight_cap = L2_GATE_WMAX
                n.inhibitory_learning_rate = p['l2_gate_eta']   # Phase 1: gate plasticity eta
                # Charge-based excitatory rule (see Neuron._update_weights);
                # eta scaled to this neuron's own weight_cap -- see ETA_FRAC note.
                # Phase 1: L2E feedforward learning uses its own l2e_lr_frac.
                # (Both auto-scale: learning_rate = frac * weight_cap, and
                # weight_cap is UNIT-scaled, so eta scales by UNIT as required.)
                n.learning_rate = p['l2e_lr_frac'] * n.weight_cap
                # Feedforward quadratic saturation: the (w/w_cap)^2 form, i.e.
                # w_max = weight_cap**2 so the term is 1 - (w/weight_cap)^2 and the
                # equilibrium is weight_cap ITSELF (not sqrt). Moves the per-gate
                # equilibrium from ~2828 to weight_cap (=8000): gates stay far from
                # saturation for longer, so participating pixels can pile more
                # weight onto a neuron's key pixels before the budget clips the
                # sum -- sharper receptive fields / bigger winner-vs-rival margins.
                n.excitatory_saturation_cap = n.weight_cap ** 2
                # Homeostatic synaptic scaling: recruits silent units and tames
                # over-active ones by regulating each neuron's own firing rate to a
                # set-point (see Neuron._homeostatic_scaling). When on, this
                # REPLACES the fixed weight budget as the resource regulator, so the
                # total is set by activity, not a hard constant.
                n.homeostasis = p['homeostasis']
                n.ca_rate = p['ca_rate']
                n.ca_target = p['ca_target']
                n.homeo_up = p['homeo_up']
                n.homeo_down = p['homeo_down']
                n.homeo_budget_min = 500   # 0.5 * UNIT, fixed-point weight resource
                n.homeo_budget_max = 2 * thr_l2
                # Confidence-gated consolidation (see neuron_flexible.Neuron and
                # Claude_Confidence_Consolidation_Plan.md): local, label-free
                # protection of mature specialists + loser depression, on L2E only.
                # conf_cap is the effective reachable mature per-gate value: the 8
                # line primitives each have 3 active pixels, so a fully specialized
                # neuron carries ~thr_l2/3 on each of its active gates (conf_cap_frac
                # = 1/3). It scales with thr_l2, so maturity stays scale-invariant.
                n.confidence_consolidation = p['confidence_consolidation']
                n.loser_depression = p['loser_depression']
                n.conf_cap = p['conf_cap_frac'] * thr_l2
                n.conf_beta = p['conf_beta']
                n.eta_min = p['eta_min']
                n.eta_loss = p['eta_loss']
                n.loss_gamma = p['loss_gamma']
                n.conf_rho_active = p['conf_rho_active']
                n.conf_rho_dead = p['conf_rho_dead']
                n.conf_ca_dead = p['conf_ca_dead']
                n.conf_grace = p['conf_grace']
                # Signed-spike depression ("4a"), L2E only. Default off leaves the
                # existing consolidation stack (confidence + loser depression +
                # budget) untouched; when on, OFF gates are also depressed on fire.
                n.signed_depression = p['signed_depression']
                n.eta_off = p['eta_off']
                # Reset-by-subtraction on fire (L2E only; default off). Leaves the
                # winner its residual overshoot instead of a full reset to rest.
                n.subtractive_reset = p['subtractive_reset']
                # Membrane saturation ceiling (L2E only; None = unbounded). Bounds
                # accumulated charge near threshold so inhibition can regulate it.
                n.v_sat = p['v_sat_frac'] * thr_l2 if p['v_sat_frac'] else None
                # Minimal signed-spike feedforward learning (L2E only; default off).
                # When on it takes over _update_weights entirely (see the neuron).
                n.signed_spike_learning = p['signed_spike_learning']
            else:
                n.weight_budget = None
                if self.meta[nid]['type'] == 'I':
                    # Phase 2: the E->I learning ceiling scales with THIS
                    # inhibitory neuron's own (possibly lowered) threshold, not
                    # the E threshold -- otherwise a single source could exceed a
                    # lowered I threshold and trivially overdrive inhibition.
                    n.weight_cap = thr_l2i if nid.startswith('L2') else thr_l1i
                    # Charge-based excitatory rule for these incoming (E->I)
                    # weights too -- same eta-scaled-to-cap principle as L2E
                    # above. Phase 1: L2I and L1I get separate lr fractions.
                    lr_frac = p['l2i_lr_frac'] if nid.startswith('L2') else p['l1i_lr_frac']
                    n.learning_rate = lr_frac * n.weight_cap
                    # Decouple the saturation ceiling from the hard clip (see
                    # excitatory_saturation_cap in Neuron). The quadratic
                    # rule dw = eta*p*(1 - w^2/w_max) has its natural equilibrium
                    # (dw=0) at w = sqrt(w_max). With w_max = weight_cap**2 the
                    # equilibrium lands EXACTLY on the clip -- so the weight only
                    # ASYMPTOTES toward the cap (dw -> 0 as w -> cap) and never
                    # actually reaches it. ei_sat_mult > 1 pushes the equilibrium
                    # sqrt(w_max) = weight_cap * sqrt(ei_sat_mult) ABOVE the clip,
                    # so dw stays large as w passes the clip and the HARD clip
                    # catches it AT weight_cap in finite time -- the rule stays
                    # nonlinear/saturating in [0, cap] but now actually reaches
                    # the cap (a trained E->I synapse becomes a true single-source
                    # relay: one presynaptic spike is enough to fire the neuron).
                    n.excitatory_saturation_cap = (n.weight_cap ** 2) * p['ei_sat_mult']
                    # Positive-afferent floor for E->I weights (minimal experiment).
                    # Only floors positive weights; the inhibitory sign is untouched.
                    if p['pos_weight_floor'] is not None:
                        n.min_positive_weight = p['pos_weight_floor']
                    # Flow-proportional assembly credit on this inhibitory neuron's
                    # own fire (L2I / L1I): the E->I "assembly evidence" synapses.
                    n.assembly_flow_credit = p['assembly_flow_credit']
                    n.assembly_decay_frac = p['assembly_decay_frac']

        # Sparse excitatory flow-rate accumulation (opt-in): configure the current-
        # trace on the POSITIVE-charge integrators. Applies to L2E, L2I and L1I --
        # but NOT L1E (an abstract pretrained sensory source), and NOT L1I while it
        # is an immediate relay (the relay bypasses trace integration entirely).
        # Read from params: the engine-level self.excitatory_flow_rate attribute is
        # assigned later in _build.
        flow = p['excitatory_flow_rate']
        for nid, n in self.neurons.items():
            if nid.startswith('L1E'):
                n.excitatory_flow_rate = False
            elif nid.startswith('L1I'):
                n.excitatory_flow_rate = flow and not p['l1i_immediate_relay']
            else:                                    # L2E, L2I
                n.excitatory_flow_rate = flow
            n.exc_trace_decay = p['exc_trace_decay']
            n.exc_trace_normalized = p['exc_trace_normalized']
            n.exc_trace = 0.0
            n.exc_trace_last_t = 0
            # Inhibitory flow (independent of the excitatory flag): applies wherever a
            # neuron RECEIVES inhibition -- L2E (the L2I->L2E discharge). L1E is exempt;
            # L1I/L2I never receive apply_inhibition so the flag is moot for them.
            n.inhibitory_flow_rate = p['inhibitory_flow_rate'] and not nid.startswith('L1E')
            n.inh_trace_decay = p['inh_trace_decay']
            n.inh_trace_normalized = p['inh_trace_normalized']
            n.inh_trace = 0.0
            # Inhibitory-gate rule applies to any neuron carrying a learned negative
            # gate (L2E's L2I->L2E gate); frozen gates (eta=0, e.g. L1E) are inert
            # under either rule, so setting it uniformly is safe.
            n.inhibitory_delta_rule = p['inhibitory_delta_rule']
            n.inhibitory_rule_mode = p['inhibitory_rule_mode']
            n.inhibitory_eta_up = p['inhibitory_eta_up']
            n.inhibitory_eta_down = p['inhibitory_eta_down']
            n.inhibitory_p_max = p['inhibitory_p_max']
            n.inhibitory_margin_frac = p['inhibitory_margin_frac']
            n.inhibitory_delta_eta = p['inhibitory_delta_eta']
            # Distance attenuation of delivered drive (per-synapse d_i stays at its
            # default 1.0 -> factor 1 -> no attenuation until functional distances
            # are assigned; the toggle + params are wired and ready).
            n.distance_weighting = p['distance_weighting']
            n.distance_power = p['distance_power']
            n.distance_ref = p['distance_ref']
            n.distance_min = p['distance_min']

        self.l1i_hold = np.zeros(N_PIX)   # L1I spike latch: held until next volley
        self.input_vec = np.array(PATTERNS['row 0'], dtype=float)
        self.timestep = 0
        self.spiked = defaultdict(bool)
        self.freq = {nid: deque(maxlen=FREQ_WINDOW) for nid in self.neurons}
        self.emitted: list[str] = []   # synapse IDs that carried a spike this step
        self._pulses: dict[str, float] = {}
        self._holds: dict[str, float] = {}
        self.event_log: deque = deque(maxlen=LOG_MAX)
        self._log_seq = 0
        self._weights_snapshot = self._all_weights()
        self.changed_synapses: list[dict] = []
        self._confidence_snapshot = self._all_confidence()
        self.changed_confidence: list[dict] = []
        self.l2_drive: dict[str, float] = {}          # PRE-WTA snapshot (peak/margin)
        self.l2_charge: dict[str, float] = {}         # POST-inhibition charge (graph/export)
        self.l2_inh_phase_debug: list[dict] = []      # per-inhibited-L2E phase record (see step())
        self.winner: str | None = None
        self._inh_events: list[tuple] = []   # (neuron_id, event) from this step's discharges

        # Episode-based competition window (interpretation only; see _update_episode).
        self.episode_active = False
        self.episode_timer = 0
        self.episode_last_spike_time = -1
        self.episode_l2_spikes: list[tuple] = []   # list of (timestep, neuron_id)

        # Auto-cycle: rotate through the patterns, showing each for a short bounded
        # VISIT, and detect training per pattern. Sitting on one pattern can't
        # work -- continuous single-pattern presentation makes the L2 pool
        # round-robin (adaptive lateral inhibition spreads participation), so no
        # single neuron "wins every time". Specialization only shows as a stable
        # pattern->neuron map ACROSS visits. So each visit accumulates L2E spikes,
        # takes the argmax as that visit's winner, and a pattern is "trained" once
        # its winner is the same neuron for trained_streak consecutive rounds.
        # After each visit the cycle advances to the next pattern; when every
        # pattern is trained, auto-cycle disables itself (curriculum complete).
        self.event_driven = self.params['event_driven']   # fire on threshold crossing every step
        # L2 chunked-charge granularity (see step()): deliver this step's L1->L2E
        # drive in K equal chunks, resolving the argmax WTA after each. K=1
        # reproduces un-chunked delivery. l2_winner_chunk records which chunk
        # resolved the last competition (diagnostic; None if no winner fired).
        self.l2_charge_chunks = max(1, int(self.params['l2_charge_chunks']))
        self.l2_winner_chunk = None
        # L1I feedback firing mode (see step() 2e): immediate deterministic relay
        # (default) vs. trainable threshold integrator.
        self.l1i_immediate_relay = self.params['l1i_immediate_relay']
        # Sparse excitatory flow-rate accumulation (see step() and Neuron). When on
        # it forces effective l2_charge_chunks = 1 (see step()); per-neuron flow
        # flags are configured on the target integrators after neuron registration.
        self.excitatory_flow_rate = self.params['excitatory_flow_rate']
        # Lasting-inhibition state (see step()): a decaying shared inhibitory field.
        self.lasting_inhibition = self.params['lasting_inhibition']
        self.inh_decay = self.params['inh_decay']
        self.inh_boost = self.params['inh_boost_frac'] * self.params['threshold_l2']
        self.l2_inh_field = 0.0
        self.current_pattern = 'row 0'            # name backing self.input_vec
        self.auto_cycle = False
        self.visit_steps = max(1, self.params['cycle_period'])   # steps per pattern visit
        self.trained_streak = 3                   # consecutive same-winner ROUNDS = trained
        self._cycle_order = list(PATTERNS.keys())
        self._visit_step = 0                      # steps elapsed in the current visit
        self._visit_spikes = np.zeros(N_OUT, dtype=int)
        self._pattern_last_winner: dict[str, int | None] = {n: None for n in PATTERNS}
        self._pattern_streak: dict[str, int] = {n: 0 for n in PATTERNS}
        self._pattern_trained: dict[str, bool] = {n: False for n in PATTERNS}

        self._log('backend', f'network built (seed={p["seed"]}, immediate delivery, '
                             f'{len(self.neurons)} neurons, {len(self.synapses)} synapses)')

    def _register_neurons(self):
        for i in range(N_PIX):
            r, c = divmod(i, 3)
            nid = f'L1E{i}'
            self.neurons[nid] = self.l1.excitatory_neurons[i]
            self.meta[nid] = dict(id=nid, label=f'in {i}', layer='L1', type='E',
                                  threshold=self.params['threshold'],
                                  pos=[(c - 1) * GRID, (1 - r) * GRID, 0.0])
        for i in range(N_PIX):
            r, c = divmod(i, 3)
            nid = f'L1I{i}'
            self.neurons[nid] = self.l1.inhibitory_neurons[i]
            self.meta[nid] = dict(id=nid, label=f'inh {i}', layer='L1', type='I',
                                  threshold=self.params['threshold'],
                                  pos=[(c - 1) * GRID, (1 - r) * GRID, -2.0])
        for j in range(N_OUT):
            nid = f'L2E{j}'
            self.neurons[nid] = self.l2.excitatory_neurons[j]
            self.meta[nid] = dict(id=nid, label=f'out {j}', layer='L2', type='E',
                                  threshold=self.params['threshold_l2'], pos=list(L2_HOMES[j]))
        self.neurons['L2I'] = self.l2.inhibitory_neuron
        self.meta['L2I'] = dict(id='L2I', label='inhib', layer='L2', type='I',
                                threshold=self.params['threshold_l2'], pos=[0.0, 0.0, 6.0])

        self.synapses: list[dict] = []
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.synapses.append(dict(id=f'ff{i}->{j}', source=f'L1E{i}', target=f'L2E{j}', kind='feedforward'))
        for j in range(N_OUT):
            self.synapses.append(dict(id=f'inh->{j}', source='L2I', target=f'L2E{j}', kind='inhibition'))
        for j in range(N_OUT):
            self.synapses.append(dict(id=f'{j}->inh', source=f'L2E{j}', target='L2I', kind='excitation'))
        for i in range(N_PIX):
            self.synapses.append(dict(id=f'li{i}', source=f'L1I{i}', target=f'L1E{i}', kind='inhibition'))
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.synapses.append(dict(id=f'fb{j}->{i}', source=f'L2E{j}', target=f'L1I{i}', kind='feedback'))

    # --------------------------------------------------------------- controls
    def reset(self):
        self._build()

    # Parameters the dashboard is allowed to change live. Anything not listed here
    # is rejected so a stray key can't silently no-op or corrupt self.params.
    TUNABLE = ('signed_depression', 'eta_off', 'l2e_budget', 'l2e_lr_frac',
               'confidence_consolidation', 'loser_depression', 'eta_loss',
               'eta_min', 'conf_cap_frac', 'leak_l2', 'event_driven',
               'subtractive_reset', 'refractory', 'v_sat_frac',
               'signed_spike_learning', 'seed', 'l2_charge_chunks',
               'l1i_immediate_relay', 'excitatory_flow_rate', 'exc_trace_decay',
               'exc_trace_normalized', 'inhibitory_flow_rate', 'inh_trace_decay',
               'inh_trace_normalized', 'inhibitory_delta_rule', 'inhibitory_rule_mode',
               'inhibitory_eta_up', 'inhibitory_eta_down', 'inhibitory_p_max',
               'inhibitory_margin_frac', 'inhibitory_delta_eta',
               'distance_weighting', 'distance_power', 'distance_ref', 'distance_min',
               'assembly_flow_credit', 'assembly_decay_frac')

    def apply_config(self, overrides: dict):
        """Merge tunable overrides into self.params and rebuild the network in
        place (same object -- all external references stay valid). Rebuilding
        restarts learning from fresh weights, which is the intended semantics of
        changing a structural/plasticity parameter. Unknown keys are ignored;
        bool/int params are coerced from the JSON the frontend sends."""
        applied = {}
        for k, v in overrides.items():
            if k not in self.TUNABLE:
                continue
            if k in ('signed_depression', 'confidence_consolidation', 'loser_depression',
                     'l2e_budget', 'event_driven', 'subtractive_reset',
                     'signed_spike_learning', 'l1i_immediate_relay',
                     'excitatory_flow_rate', 'exc_trace_normalized',
                     'inhibitory_flow_rate', 'inh_trace_normalized',
                     'inhibitory_delta_rule', 'distance_weighting',
                     'assembly_flow_credit'):
                v = bool(v)
            elif k in ('seed', 'refractory', 'l2_charge_chunks'):
                v = int(v)
            elif k == 'inhibitory_rule_mode':
                v = str(v)
            else:
                v = float(v)
            self.params[k] = v
            applied[k] = v
        self._build()
        self._log('control', f'config applied: {applied}')
        return applied

    def set_pattern(self, name: str):
        if name not in PATTERNS:
            raise KeyError(name)
        self.input_vec = np.array(PATTERNS[name], dtype=float)
        self.current_pattern = name
        self._visit_step = 0                 # start a fresh visit window
        self._visit_spikes[:] = 0
        self._log('control', f'pattern set: {name}')

    def set_auto_cycle(self, enabled: bool, streak: int | None = None,
                       visit_steps: int | None = None):
        """Enable/disable auto-cycling and (optionally) its thresholds. Starts the
        cycle from the currently-shown pattern; per-pattern training history is
        cleared so a new run measures training fresh."""
        self.auto_cycle = bool(enabled)
        if streak is not None:
            self.trained_streak = max(1, int(streak))
        if visit_steps is not None:
            self.visit_steps = max(1, int(visit_steps))
        self._visit_step = 0
        self._visit_spikes[:] = 0
        for n in PATTERNS:
            self._pattern_last_winner[n] = None
            self._pattern_streak[n] = 0
            self._pattern_trained[n] = False
        self._log('control', f'auto-cycle {"on" if self.auto_cycle else "off"} '
                             f'(trained_streak={self.trained_streak}, '
                             f'visit_steps={self.visit_steps})')
        return dict(enabled=self.auto_cycle, streak=self.trained_streak,
                    visit_steps=self.visit_steps)

    def _auto_cycle_tick(self):
        """Called at the END of each step() while auto_cycle is on. Accumulates
        this visit's L2E spikes; when the visit window closes, resolves the visit
        winner (argmax), updates the current pattern's consecutive-round streak,
        marks it trained at the threshold, then advances to the next pattern.
        When every pattern is trained the cycle disables itself."""
        for j in range(N_OUT):
            if self.spiked[f'L2E{j}']:
                self._visit_spikes[j] += 1
        self._visit_step += 1
        if self._visit_step < self.visit_steps:
            return

        p = self.current_pattern
        winner = int(self._visit_spikes.argmax()) if self._visit_spikes.sum() > 0 else None
        prev = self._pattern_last_winner[p]
        if winner is not None and winner == prev:
            self._pattern_streak[p] += 1
        else:
            self._pattern_streak[p] = 1 if winner is not None else 0
        self._pattern_last_winner[p] = winner
        if not self._pattern_trained[p] and self._pattern_streak[p] >= self.trained_streak:
            self._pattern_trained[p] = True
            self._log('learning', f'auto-cycle: "{p}" TRAINED '
                                  f'-> L2E{winner} won {self._pattern_streak[p]} rounds')

        if all(self._pattern_trained.values()):
            self.auto_cycle = False
            self._log('control', 'auto-cycle: all patterns trained -- curriculum complete')
            self._visit_step = 0
            self._visit_spikes[:] = 0
            return

        order = self._cycle_order
        idx = order.index(p) if p in order else -1
        self.set_pattern(order[(idx + 1) % len(order)])   # resets the visit window

    def set_input(self, vec):
        self.input_vec = np.array(vec, dtype=float).reshape(N_PIX)
        self._log('control', 'input vector set')

    def toggle_pixel(self, i: int):
        self.input_vec[i] = 0.0 if self.input_vec[i] > 0.5 else 1.0

    def clear_input(self):
        self.input_vec = np.zeros(N_PIX)
        self._log('control', 'input cleared')

    def random_pattern(self):
        self.input_vec = (np.random.default_rng().random(N_PIX) > 0.5).astype(float)
        self._log('control', 'random input')

    def inject_noise(self, prob: float = 0.15):
        flip = np.random.default_rng().random(N_PIX) < prob
        self.input_vec = np.where(flip, 1.0 - self.input_vec, self.input_vec)
        self._log('control', f'noise injected (p={prob:.2f})')

    def stimulate(self, neuron_id: str, magnitude: float = 1.0, continuous: bool = False):
        if neuron_id not in self.neurons:
            raise KeyError(neuron_id)
        if continuous:
            self._holds.pop(neuron_id, None) if magnitude == 0 else self._holds.__setitem__(neuron_id, magnitude)
        else:
            self._pulses[neuron_id] = self._pulses.get(neuron_id, 0.0) + magnitude
        self._log('control', f'stimulate {neuron_id} (+{magnitude:g}{", hold" if continuous else ""})')

    def _check_l2_inhibition_phases(self, l2, v_start):
        """Build per-L2E charge-phase records for this step's real L2I->L2E
        discharges and warn (never crash) if the flow-rate ordering invariant is
        violated. The invariant: a real inhibition must not RAISE the target's
        charge (V_after_inhibition <= V_before_inhibition), and no same-timestep
        excitatory trace advance may push an inhibited L2E back above its
        post-inhibition charge (V_end is measured after leak, so it must be
        <= V_after_inhibition). Also flags any discharge not routed through the
        expected negative L2I->L2E gate at synapse index 0. Read-only: it inspects
        this step's inhibitory events and the current membrane; it never advances a
        trace or mutates a neuron."""
        self.l2_inh_phase_debug = []
        for nid, ev in self._inh_events:
            if not nid.startswith('L2E'):
                continue
            j = int(nid[3:])
            neuron = l2.excitatory_neurons[j]
            v_end = float(neuron.potential)
            self.l2_inh_phase_debug.append(dict(
                id=nid,
                v_start=round(v_start.get(j, 0.0), 3),
                v_after_trace_advance=round(float(getattr(neuron, '_dbg_v_after_advance', ev['v_pre'])), 3),
                v_before_inhibition=round(ev['v_pre'], 3),   # == V after trace + injection
                inhibition_w=round(ev['w_before'], 3),
                v_after_inhibition=round(ev['v_post'], 3),
                v_end=round(v_end, 3),
                gate_index=int(ev['index'])))
            if ev['v_post'] > ev['v_pre'] + 1e-6:
                self._log('warn', f"{nid}: inhibition RAISED charge "
                                  f"(v_pre={ev['v_pre']:.1f} -> v_post={ev['v_post']:.1f})")
            if v_end > ev['v_post'] + 1e-6:
                self._log('warn', f"{nid}: charge rose above post-inhibition within the "
                                  f"same timestep (v_post={ev['v_post']:.1f} -> v_end={v_end:.1f}) "
                                  f"-- unexpected second excitatory trace advance")
            if int(ev['index']) != 0:
                self._log('warn', f"{nid}: L2I discharge hit synapse {ev['index']}, "
                                  f"not the expected L2I->L2E gate at index 0")

    def _resolve_l2_competition(self, l2, l2e, t):
        """Attempt one standard argmax WTA resolution on the CURRENT L2E membrane
        state. If any L2E crossed threshold, the max-charge crosser fires, drives
        L2I, and (if L2I fires) the whole rest of the pool -- every non-winner, not
        just co-crossers -- is inhibited through the learned L2I->L2E gate. Mutates
        `l2e` in place (sets the winner's one-hot bit). Returns
        (l2i, inhibited, winner); winner is None and L2I is left UNTOUCHED when
        nobody crossed, so a caller running this per charge-chunk can drive L2I's
        no-winner integration exactly once after the chunk loop instead of K times.
        `t` is the current outer timestep (passed through to L2I flow-rate charge).
        """
        eligible = [j for j, e in enumerate(l2.excitatory_neurons) if e.check_threshold()]
        if not eligible:
            return 0.0, [], None
        winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
        l2.excitatory_neurons[winner].fire()
        l2e[winner] = 1.0
        # The winner drives L2I, which fires (E->I weight = thr_l2) and laterally
        # inhibits the whole rest of the pool.
        l2.inhibitory_neuron.receive_input(l2e, t=t)
        l2i = 1.0 if l2.inhibitory_neuron.check_threshold() else 0.0
        inhibited = []
        if l2i:
            l2.inhibitory_neuron.fire()
            inh_spk = np.zeros(L2E_FANIN)
            inh_spk[0] = 1.0                       # index 0 = the L2I->L2E gate
            for j in range(N_OUT):
                if j == winner:
                    continue                        # winner already fired / refractory
                # apply_inhibition no-ops on refractory neurons; sub-threshold
                # rivals (the real cause of the rotation) are now discharged too.
                events = l2.excitatory_neurons[j].apply_inhibition(inh_spk)
                for ev in events:
                    self._inh_events.append((f'L2E{j}', ev))
                if events:
                    inhibited.append(j)
        return l2i, inhibited, winner

    # ------------------------------------------------------------------- step
    def step(self) -> dict:
        l1, l2 = self.l1, self.l2
        t = self.timestep

        # 1. L1E: [paired I1's previous spike (local inhibition), external pixel].
        #    Pixels fire in bursts every input_period steps (the SENSORY rate).
        #    L2 competition resolves EVERY step by default (event_driven, the
        #    canonical per-step single-winner flow). With event_driven OFF it
        #    instead resolves once per cycle_period steps (the INTRINSIC clock),
        #    decoupling winner identity from how fast input arrives -- see the
        #    input_period/cycle_period note in __init__.
        #    Excitatory pixel drive goes through receive_input; the inhibitory
        #    discharge is delivered as its own event via apply_inhibition, which
        #    also runs the inhibitory-gate plasticity rule. The net membrane before
        #    threshold (ext - |w_inh|) is identical to the old summed delivery, so
        #    spike timing and event ordering are unchanged.
        input_arrives = (t % self.params['input_period'] == 0)
        cycle_boundary = (t % self.params['cycle_period'] == 0)
        self._inh_events = []
        for i, e in enumerate(l1.excitatory_neurons):
            ext = 1.0 if (input_arrives and self.input_vec[i] > 0.5) else 0.0
            e.receive_input(np.array([0.0, ext]))
            # Inhibition counteracts the external drive, so it is applied on the
            # same input steps; the hold persists from the last cycle's L1I
            # activity so refractory doesn't swallow it.
            inh = float(self.l1i_hold[i]) if input_arrives else 0.0
            if inh > 0.5:
                for ev in e.apply_inhibition(np.array([1.0, 0.0])):
                    self._inh_events.append((f'L1E{i}', ev))

        self._apply_stim()

        # 2a. L1E fires (no competition).
        l1e = np.array([1.0 if e.check_threshold() else 0.0 for e in l1.excitatory_neurons])
        for k, e in enumerate(l1.excitatory_neurons):
            if l1e[k]:
                e.fire()

        # 2b/2c. Deliver L1E->L2E feedforward charge and resolve L2 competition.
        #     When the winner fires, the shared inhibitory neuron L2I discharges the
        #     ENTIRE rest of the pool through its learned L2I->L2E gate -- not just
        #     the co-threshold-crossers. The sub-threshold rivals sitting JUST below
        #     threshold are the real cause of winner rotation; discharging the whole
        #     pool subtracts each rival's own learned gate magnitude so the race
        #     restarts closer to even and the best-matched integrator can win
        #     repeatedly (the precondition for consolidation). The gate stays below
        #     threshold_l2 (L2_GATE_WMAX < thr_l2), a PARTIAL discharge that
        #     preserves cross-volley evidence, not a hard reset. See
        #     _resolve_l2_competition for the argmax WTA body.
        #
        #     Chunked charge (l2_charge_chunks = K): this step's feedforward drive
        #     can arrive in K equal chunks (weight_ji/K per active synapse) WITHIN
        #     this frozen outer timestep. After each chunk the argmax WTA is
        #     re-attempted; the FIRST chunk that produces a threshold-crosser
        #     resolves the competition and the remaining chunks are skipped
        #     (consolidation-first: the earliest strong responder wins before rivals
        #     pile up charge). The clock does not advance and no leak/update runs
        #     between chunks. K=1 (default) delivers the full drive in a single chunk
        #     and reproduces the un-chunked behavior exactly.
        ff_vec = np.zeros(L2E_FANIN)
        for i in range(N_PIX):
            if l1e[i]:
                ff_vec[1 + i] = 1.0

        l2e = np.zeros(N_OUT)
        l2i = 0.0
        inhibited = []
        self.l2_winner_chunk = None
        # EXPLICIT PHASE ORDER (flow-rate; see the flow-rate section of the
        # methodology doc): per outer timestep each L2E goes
        #   V_start -> (receive_input: advance trace to t-1, inject new drive,
        #   one same-timestep integrate = V_before_inhibition) -> threshold/WTA ->
        #   L2I inhibition on non-winners (V_after_inhibition) -> leak (V_end).
        # No excitatory trace is advanced again for an inhibited L2E in the same
        # timestep (receive_input already set exc_trace_last_t = t, so any later
        # advance_trace(t) is a no-op). l2_drive is the PRE-WTA snapshot (peak/margin
        # consumers); l2_charge (captured after inhibition, below) is what the
        # dashboard/export shows so the inhibition dip is visible.
        l2_v_start = {j: float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}

        if self.lasting_inhibition:
            # Alternate mechanism (opt-in): deliver the full drive un-chunked and
            # resolve through a decaying shared inhibitory field. L2I pumps the
            # field when it fires; the field hyperpolarizes the whole L2E pool every
            # step and decays with inh_decay, so a pattern's specialist keeps the
            # field up and rivals stay locked out for a finite window.
            for e in l2.excitatory_neurons:
                e.receive_input(ff_vec, t=t)
            self.l2_drive = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}
            self.l2_inh_field *= (1.0 - self.inh_decay)
            field = self.l2_inh_field
            eligible = [j for j, e in enumerate(l2.excitatory_neurons)
                        if e.refractory_timer <= 0 and e.potential >= e.threshold + field]
            if eligible:
                winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
                l2.excitatory_neurons[winner].fire()
                l2e[winner] = 1.0
                inhibited = [j for j in eligible if j != winner]
                l2.inhibitory_neuron.receive_input(l2e, t=t)
                if l2.inhibitory_neuron.check_threshold():
                    l2.inhibitory_neuron.fire()
                    l2i = 1.0
                    self.l2_inh_field += self.inh_boost
        elif cycle_boundary or self.event_driven:
            # Standard per-step argmax competition, delivered in K charge chunks.
            # Competition resolves EVERY step (event_driven, default) OR once per
            # cycle (event_driven OFF: only on the intrinsic cycle boundary).
            # Flow-rate mode is the finer temporal model, so it FORCES effective
            # K = 1 (chunking is ignored while flow-rate is on) -- the trace itself
            # spreads a volley's charge over time.
            K = 1 if self.excitatory_flow_rate else self.l2_charge_chunks
            resolved = False
            for chunk in range(K):
                for e in l2.excitatory_neurons:
                    e.receive_input(ff_vec, charge_scale=1.0 / K, t=t)
                # Pre-WTA membrane snapshot at the current evaluation point (charge
                # viz / winner-margin diagnostics read this).
                self.l2_drive = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}
                l2i, inhibited, winner = self._resolve_l2_competition(l2, l2e, t)
                if winner is not None:
                    self.l2_winner_chunk = chunk
                    resolved = True
                    break
            if not resolved:
                # Full drive arrived and nobody crossed: L2I still integrates the
                # (empty) winner vector and may fire from residual charge.
                l2.inhibitory_neuron.receive_input(l2e, t=t)
                l2i = 1.0 if l2.inhibitory_neuron.check_threshold() else 0.0
                if l2i:
                    l2.inhibitory_neuron.fire()
        else:
            # Non-resolving step (event_driven OFF, between cycle boundaries): the
            # drive is delivered in full and accumulates on the membrane; the
            # competition waits for the next cycle boundary.
            for e in l2.excitatory_neurons:
                e.receive_input(ff_vec, t=t)
            self.l2_drive = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}

        # PHASE 7: record the graph/export charge AFTER inhibition (before leak), so
        # the dashboard "charge over time" shows the L2I->L2E discharge dip instead
        # of the pre-WTA value. This is a read-only snapshot -- it does NOT advance
        # traces or mutate any neuron.
        self.l2_charge = {f'L2E{j}': float(e.potential) for j, e in enumerate(l2.excitatory_neurons)}

        # 2d. Deliver L2E winner spike immediately to all L1I neurons (feedback).
        #     l2e is length N_OUT with a 1 at the winner index, matching each
        #     L1I neuron's N_OUT-dimensional afferent weight vector. (t carries the
        #     flow-rate current trace when L1I is a trainable integrator; ignored in
        #     immediate-relay mode, where L1I's flow flag is off.)
        for inh in l1.inhibitory_neurons:
            inh.receive_input(l2e, t=t)

        # 2e. L1I fires after receiving L2E feedback.
        if self.l1i_immediate_relay:
            # Immediate deterministic relay (default): any L1I that received a
            # nonzero L2E feedback signal fires THIS step -- no membrane
            # accumulation, no learned-threshold crossing, no dependence on L1I
            # feedback-weight training. l2e is delivered identically to every L1I
            # (2d), so a winner (l2e nonzero) makes every L1I relay. Output stays
            # binary and the downstream L1I->L1E inhibition path is unchanged.
            fb_present = 1.0 if np.any(l2e > 0) else 0.0
            l1i = np.full(len(l1.inhibitory_neurons), fb_present)
        else:
            # Trainable threshold integrator: fire only when accumulated feedback
            # crossed L1I's own (learned/leaky) threshold.
            l1i = np.array([1.0 if n.check_threshold() else 0.0 for n in l1.inhibitory_neurons])
        for k, n in enumerate(l1.inhibitory_neurons):
            if l1i[k]:
                n.fire()

        # 3. Collect synapse IDs that carried a spike this step (for edge flash).
        self.emitted = []
        for i in range(N_PIX):
            if l1e[i]:
                for j in range(N_OUT):
                    self.emitted.append(f'ff{i}->{j}')
        for j in range(N_OUT):
            if l2e[j]:
                for i in range(N_PIX):
                    self.emitted.append(f'fb{j}->{i}')
                self.emitted.append(f'{j}->inh')
        for j in inhibited:
            self.emitted.append(f'inh->{j}')
        for i in range(N_PIX):
            if l1i[i]:
                self.emitted.append(f'li{i}')

        # 4. Advance membrane state (leak + refractory countdown).
        for e in l1.excitatory_neurons:
            e.update()
        for n in l1.inhibitory_neurons:
            n.update()
        for e in l2.excitatory_neurons:
            e.update()
        l2.inhibitory_neuron.update()

        # Phase-order guard + per-L2E charge-phase diagnostics for this step's real
        # L2I->L2E discharges (V_end is post-leak here).
        self._check_l2_inhibition_phases(l2, l2_v_start)

        # 5. Bookkeeping.
        self._record_spikes(l1e, l1i, l2e, l2i)
        # Latch L1I activity so it blocks L1E on the NEXT input burst, not the
        # next time step (where refractory would silently swallow the
        # inhibition). L1I responds to the per-cycle winner, so latch on the
        # intrinsic cycle boundary; the hold is applied on input steps above.
        if cycle_boundary:
            self.l1i_hold = l1i
        self.timestep += 1
        self._detect_weight_changes()
        self._detect_confidence_changes()
        self._log_inhibitory_events()
        self._update_episode(l2e, t)
        if self.auto_cycle:
            self._auto_cycle_tick()
        return self.dynamic_state()

    def _log_inhibitory_events(self):
        """Surface inhibitory-discharge plasticity into the event log for the
        dashboard. Only events that actually moved a gate (|delta_w| > eps) are
        logged, so a saturated gate (delta_w == 0) doesn't flood the panel; the
        full per-event debug record always lives on each neuron's
        last_inhibitory_events."""
        for nid, ev in self._inh_events:
            if abs(ev['delta_w']) > WEIGHT_EPS:
                self._log('inhibition',
                          f"{nid} gate: Vpre={ev['v_pre']:.3f} θ={ev['theta']:.2f} "
                          f"p={ev['p']:.2f} |w| {ev['w_before']:.3f}->{ev['w_after']:.3f} "
                          f"(Δ={ev['delta_w']:+.4f})")

    def _apply_stim(self):
        for nid, mag in list(self._pulses.items()):
            self.neurons[nid].potential += mag
        self._pulses.clear()
        for nid, mag in self._holds.items():
            self.neurons[nid].potential += mag

    def _record_spikes(self, l1e, l1i, l2e, l2i):
        for i in range(N_PIX):
            self.spiked[f'L1E{i}'] = bool(l1e[i]); self.freq[f'L1E{i}'].append(l1e[i])
            self.spiked[f'L1I{i}'] = bool(l1i[i]); self.freq[f'L1I{i}'].append(l1i[i])
        for j in range(N_OUT):
            self.spiked[f'L2E{j}'] = bool(l2e[j]); self.freq[f'L2E{j}'].append(l2e[j])
        self.spiked['L2I'] = bool(l2i); self.freq['L2I'].append(l2i)

    def _update_episode(self, l2e, t):
        """
        Episode-based competition interpretation. This is the ONLY thing that
        changed relative to the old instantaneous winner readout: it decides
        *when* competition is considered resolved and *which* neuron is reported
        as the winner. It reads only l2e (this step's L2E spikes) and t, and
        writes only the episode_* fields and self.winner. It never touches a
        neuron, a weight, a potential, WTA, or the learning rule -- so LIF and
        plasticity are byte-for-byte unchanged.

        Structure:
          - An episode STARTS on a volley tick, but only if one is not already
            running (so a single episode can span several volleys up to T_max
            instead of being reset every volley).
          - While active, every L2E spike this step is appended to the history
            and the last-spike time is updated.
          - The episode ENDS on Condition A (K consecutive L2-silent steps) or
            Condition B (episode_timer reaches T_max), whichever comes first.
          - The winner is then resolved from the spike history alone
            (latest-spike, then most-spikes tiebreak) -- no argmax over membrane
            potentials, no global ranking.
        """
        volley = (t % self.params['volley_period'] == 0)
        if volley and not self.episode_active:
            self.episode_active = True
            self.episode_timer = 0
            self.episode_last_spike_time = -1
            self.episode_l2_spikes = []

        if not self.episode_active:
            return

        # Record this step's L2E spikes. WTA fires at most one L2E per step, but
        # we record generally so any co-firing would also be captured.
        for j in range(N_OUT):
            if l2e[j]:
                self.episode_l2_spikes.append((t, f'L2E{j}'))
                self.episode_last_spike_time = t
        self.episode_timer += 1

        # Condition A: silent for K consecutive steps (counted from the last
        # spike, or from episode start if nothing has fired yet).
        if self.episode_last_spike_time >= 0:
            silent = t - self.episode_last_spike_time
        else:
            silent = self.episode_timer - 1
        # Condition B: episode length cap.
        if silent >= EPISODE_QUIET_K or self.episode_timer >= EPISODE_MAX_LEN:
            self._resolve_episode()
            self.episode_active = False

    def _resolve_episode(self):
        """Resolve the episode winner from spike history only.

        Rule 1 (primary): the neuron with the LATEST spike time wins.
        Rule 2 (tiebreak): if several neurons share that latest spike time, the
        one with the MOST spikes over the whole episode wins.
        An episode with no L2 spikes leaves the previous winner untouched.
        """
        if not self.episode_l2_spikes:
            return
        latest_t = max(ts for ts, _ in self.episode_l2_spikes)
        last_spikers = [nid for ts, nid in self.episode_l2_spikes if ts == latest_t]
        if len(set(last_spikers)) > 1:
            counts: dict[str, int] = {}
            for _, nid in self.episode_l2_spikes:
                counts[nid] = counts.get(nid, 0) + 1
            winner = max(set(last_spikers), key=lambda n: counts[n])
        else:
            winner = last_spikers[0]
        if winner != self.winner:
            self._log('learning', f'episode winner -> {winner} '
                                  f'(spikes={len(self.episode_l2_spikes)}, last_t={latest_t})')
        self.winner = winner

    # ------------------------------------------------------------- weight diff
    def _all_weights(self) -> dict:
        w = {}
        for j in range(N_OUT):
            arr = self.l2.excitatory_neurons[j]._weights_array
            w[f'inh->{j}'] = float(arr[0])
            for i in range(N_PIX):
                w[f'ff{i}->{j}'] = float(arr[1 + i])
        iw = self.l2.inhibitory_neuron._weights_array
        for j in range(N_OUT):
            w[f'{j}->inh'] = float(iw[j])
        for i in range(N_PIX):
            w[f'li{i}'] = float(self.l1.excitatory_neurons[i].weights[0])
            fbw = self.l1.inhibitory_neurons[i].weights
            for j in range(N_OUT):
                w[f'fb{j}->{i}'] = float(fbw[j])
        return w

    def _detect_weight_changes(self):
        now = self._all_weights()
        self.changed_synapses = [dict(id=sid, weight=round(v, 4))
                                 for sid, v in now.items()
                                 if abs(v - self._weights_snapshot.get(sid, v)) > WEIGHT_EPS]
        self._weights_snapshot = now

    def _all_confidence(self) -> dict:
        """Per-synapse confidence for the L2E feedforward receptive fields, keyed by
        the same synapse ids as _all_weights (ff{i}->{j}). Confidence is the L2E
        neuron's trust that opening each EXCITATORY gate helps it fire (see
        neuron_flexible.Neuron), so only the feedforward (positive) synapses are reported --
        the negative L2I->L2E gate has its own inhibitory plasticity and no
        excitatory-trust value. The field is always safe to serialize because
        every neuron allocates a confidence vector during connection finalization."""
        c: dict[str, float] = {}
        for j in range(N_OUT):
            conf = self.l2.excitatory_neurons[j].confidence
            for i in range(N_PIX):
                c[f'ff{i}->{j}'] = float(conf[1 + i])
        return c

    def _detect_confidence_changes(self):
        now = self._all_confidence()
        self.changed_confidence = [dict(id=sid, confidence=round(v, 4))
                                   for sid, v in now.items()
                                   if abs(v - self._confidence_snapshot.get(sid, v)) > WEIGHT_EPS]
        self._confidence_snapshot = now

    def _budget_usage(self, nid: str):
        """(budget, budget_used) for an L2E neuron, else (None, None). budget_used
        is the current sum of positive (feedforward) weights vs its fixed budget."""
        n = self.neurons[nid]
        if self.meta[nid]['type'] == 'E' and nid.startswith('L2') and n.weight_budget is not None:
            w = n.weights
            return float(n.weight_budget), float(w[w > 0].sum())
        return None, None

    # ----------------------------------------------------------------- access
    def firing_freq(self, nid: str) -> float:
        d = self.freq[nid]
        return float(sum(d) / len(d)) if d else 0.0

    def activation(self, nid: str) -> float:
        thr = self.meta[nid]['threshold'] or 1.0
        return float(self.neurons[nid].potential / thr)

    def _log(self, kind: str, message: str):
        self._log_seq += 1
        self.event_log.append(dict(seq=self._log_seq, t=self.timestep, kind=kind, message=message))

    # ------------------------------------------------------------ serialization
    def topology(self) -> dict:
        weights = self._all_weights()
        confidence = self._all_confidence()
        neurons = [dict(**self.meta[nid]) for nid in self.neurons]
        synapses = [dict(**s, weight=round(weights.get(s['id'], 0.0), 4),
                         confidence=round(confidence[s['id']], 4) if s['id'] in confidence else None)
                    for s in self.synapses]
        return dict(neurons=neurons, synapses=synapses, layers=['L1', 'L2'],
                    patterns=list(PATTERNS.keys()),
                    pattern_vectors={k: list(map(int, v)) for k, v in PATTERNS.items()},
                    grid=dict(rows=3, cols=3), params=self.params)

    def dynamic_state(self) -> dict:
        neurons = []
        for nid, n in self.neurons.items():
            # Charge shown to the dashboard is the POST-inhibition snapshot
            # (l2_charge) so the L2I->L2E discharge dip is visible; fall back to the
            # pre-WTA drive, then the live potential. Read-only -- never advances a
            # flow-rate trace.
            pot = self.l2_charge.get(nid, self.l2_drive.get(nid, float(n.potential)))
            thr = self.meta[nid]['threshold'] or 1.0
            budget, budget_used = self._budget_usage(nid)
            neurons.append(dict(id=nid, potential=round(pot, 4),
                                activation=round(pot / thr, 4),
                                spiked=self.spiked[nid], freq=round(self.firing_freq(nid), 4),
                                refractory=int(n.refractory_timer),
                                budget=round(budget, 4) if budget is not None else None,
                                budget_used=round(budget_used, 4) if budget_used is not None else None,
                                assembly=(self.winner if nid == self.winner else None),
                                # Optional flow-rate diagnostic: current-trace amplitude.
                                **({'exc_trace': round(float(n.exc_trace), 4)}
                                   if self.excitatory_flow_rate and n.excitatory_flow_rate else {})))
        return dict(timestep=self.timestep, running=False, neurons=neurons,
                    changed_synapses=self.changed_synapses,
                    changed_confidence=self.changed_confidence,
                    emitted=self.emitted,
                    input=self.input_vec.astype(int).tolist(), winner=self.winner,
                    episode=dict(active=self.episode_active, timer=self.episode_timer,
                                 spikes=len(self.episode_l2_spikes),
                                 participants=sorted({nid for _, nid in self.episode_l2_spikes})),
                    autocycle=dict(enabled=self.auto_cycle, pattern=self.current_pattern,
                                   target=self.trained_streak, visit_steps=self.visit_steps,
                                   last_winner=self._pattern_last_winner.get(self.current_pattern),
                                   streak=self._pattern_streak.get(self.current_pattern, 0),
                                   trained=sum(1 for v in self._pattern_trained.values() if v),
                                   total=len(self._pattern_trained),
                                   trained_map={n: self._pattern_trained[n] for n in self._cycle_order}),
                    l2_charge_chunks=self.l2_charge_chunks,
                    l2_winner_chunk=self.l2_winner_chunk,
                    l2_inh_phases=self.l2_inh_phase_debug,
                    stats=self.stats(), log=list(self.event_log)[-12:])

    def stats(self) -> dict:
        active = sum(1 for nid in self.neurons if abs(self.neurons[nid].potential) > 1e-3)
        firing = sum(1 for nid in self.neurons if self.spiked[nid])
        pots = [self.neurons[nid].potential for nid in self.neurons]
        weights = list(self._all_weights().values())
        rate = float(np.mean([self.firing_freq(nid) for nid in self.neurons]))
        return dict(total=len(self.neurons), active=active, firing=firing,
                    avg_activation=round(float(np.mean(np.abs(pots))), 4),
                    firing_rate=round(rate, 4), avg_weight=round(float(np.mean(weights)), 4),
                    winner=self.winner)
