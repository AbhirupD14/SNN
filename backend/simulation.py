"""
SimulationEngine -- steppable wrapper around the spiking network.

Spikes are delivered immediately except for L1I feedback, which is held in a
one-step register so a spike at t suppresses its paired L1E at t+1. Every neuron
has a 3D position generated once per build as seeded local scatter in
`backend.layout`. The same coordinates are shown in the viewport and used to
derive dimensionless L1E->L2E distances. Those distances attenuate delivered
feedforward charge without changing stored weights. Positions do not imply a
conduction delay.

Learning architecture:
  - L1E neurons are treated as pre-trained pixel encoders: weights are fixed
    at [-1.0, 1.0] and learning_rate = 0.  They fire whenever the external
    pixel is active and they are not suppressed by their paired L1I neuron.
  - L2E neurons use signed input-spike learning by default. Active pixel gates
    potentiate, inactive pixel gates depress, and there is no positive weight
    budget on this path.
  - L1I / L2I incoming excitatory weights have no budget. Each gate is capped at
    its inhibitory neuron's own threshold, and both inhibitory thresholds share
    the L2I scale by default.
  - L1I is a trainable accumulator by default. Every L1I observes the same L2E
    winner stream, so the bank starts from copies of one random, task-independent
    feedback vector. Independent vectors create arbitrary pixel phase classes
    without any pixel-local signal that can repair them.
  - L1I credits every L2E contributor in its accumulation window, not only the
    final threshold-crossing winner. Its effective one-step refractory interval
    prevents consecutive feedback pulses. With constant input this converges to
    a synchronized fire/suppress rhythm that halves active L1E frequency.

L2E spikes train the positive recruitment weights into L2I. When L2I fires, it
broadcasts an unweighted hard-reset event to the L2E pool; there is no learned
L2I-to-L2E gate on this path. L1I supplies weighted feedback inhibition to L1E
through the delayed register.
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
from snn import NeuronConfig                          # noqa: E402  engine-sourced neuron config
from .layout import generate_layout


PATTERNS = {
    'row 1':    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    'col 1':    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    'diag \\':  [1, 0, 0, 0, 1, 0, 0, 0, 1],
    'diag /':   [0, 0, 1, 0, 1, 0, 1, 0, 0],
}

# Four center-crossing input primitives, but the L2E pool is DECOUPLED from the
# pattern count: N_OUT L2E neurons compete over len(PATTERNS) inputs. With N_OUT=8
# and 4 patterns that is 2x overcapacity -- multiple neurons vie for each pattern, so
# competition/recruitment dynamics (who wins, who differentiates, who dies) are
# visible. N_OUT sizes the L2E layer and everything downstream (ring, feedback fan-in,
# L1I afferents); PATTERNS is cycled by NAME and never indexes a neuron, so the two
# are independent.
N_PIX = 9
N_OUT = 8

# Active L2E fan-in: exactly N_PIX pixel afferents. There is NO index-0 local-I
# placeholder and no negative L2I->L2E gate -- L2 competition is now an unweighted
# hard reset plus a local loser update in Neuron.apply_competitive_reset.
L2E_FANIN = N_PIX
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
#   synapse, not a permanent structural ceiling.
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
# dynamics: one contribution starts near 37.5% of threshold and roughly four
# volley-spaced contributions are needed to cross, at
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
# scaled to L1I's own threshold on the L2I scale -- L1I was
# previously an instant integrator from t=0 (every synapse initialized
# at-or-above its own threshold, so a single L2E spike always fired it
# immediately, with no round-robin phase and nothing left for the existing
# Hebbian rule to actually move). Now: synapses start randomly in
# [0.25, 0.5] * thr_l1i (L1_EI_WEIGHT_INIT_LOW/_HIGH_FRAC -- same fractions as
# L2I, so the same relative dynamics apply: ~37.5% of threshold from one
# spike, 4 volley-spaced spikes needed to cross), are NOT weight-budgeted,
# and are individually capped at thr_l1i itself, so a habitually-participating
# source can learn its way up to full self-sufficiency exactly like L2I does.
# L1I_LEAK_RATE (== L2I_LEAK_RATE numerically, kept as its own named constant
# since it governs a different neuron) gives the same ~1.49x reachable-ceiling
# headroom above threshold that L2I_LEAK_RATE gives L2I -- see the
# L2_EI_WEIGHT_INIT_LOW_FRAC note above for the derivation (it is scale-free:
# depends only on the weight/threshold fraction and the leak rate, not the
# absolute threshold value, so the identical fractions and leak rate carry
# over unchanged at any shared L2I/L1I threshold scale).
L1_EI_WEIGHT_INIT_LOW_FRAC = 1 / 4   # low end of the random L2E->L1I init range,
L1_EI_WEIGHT_INIT_HIGH_FRAC = 1 / 2  # high end -- both as a fraction of thr_l1i.
                                      # Learning ceiling is thr_l1i itself (see the
                                      # per-neuron cap assignment below), not this.
L1I_LEAK_RATE = 70 / LEAK_SCALE      # L1I's own membrane/trace leak (>> leak_l1); 70/1000 == 0.07
# fire() is followed by update() in the same outer step, so a timer of 2 counts
# down to 1 immediately and blocks exactly the following step. This prevents
# consecutive feedback pulses and sets the trained constant-drive rhythm to 2:1.
L1I_FEEDBACK_REFRACTORY = 2

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
# long round-robin phase rather than a handful of volleys.
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

# Mean L2E feedforward weight at init (fixed-point). 125 == the mean of the legacy
# Uniform(50,200) init, preserved by the balanced mode so the LIF-accumulation scale
# is unchanged -- only the balance/bias of the init differs, not its magnitude.
L2E_INIT_MEAN = 125.0
L2E_INIT_JITTER = 0.05   # default jitter eps for the balanced init (Uniform(1-eps,1+eps))


def _sinkhorn_balance(Z, col_target, max_iters=1000, tol=1e-12):
    """Doubly-balance a POSITIVE matrix by alternating row/column normalization
    (Sinkhorn-Knopp). Returns a matrix with every row summing to 1 and every column
    summing to `col_target`. For an (m x n) matrix the consistent choice is
    col_target = m/n (total mass = m along both axes). Convergence is geometric for a
    strictly positive matrix (Sinkhorn's theorem); the loop stops early once both the
    row- and column-sum errors fall below `tol`. Permutation-EQUIVARIANT: permuting
    rows or columns of the input permutes the output identically (it uses only per-row
    and per-column sums, never a position)."""
    Z = np.array(Z, dtype=float)
    for _ in range(max_iters):
        Z *= 1.0 / Z.sum(axis=1, keepdims=True)           # rows -> sum 1
        Z *= col_target / Z.sum(axis=0, keepdims=True)     # cols -> sum col_target
        row_err = np.abs(Z.sum(axis=1) - 1.0).max()
        col_err = np.abs(Z.sum(axis=0) - col_target).max()
        if max(row_err, col_err) < tol:
            break
    return Z


def balanced_feedforward_init(rng, n_out, n_pix, jitter=L2E_INIT_JITTER,
                              target_mean=L2E_INIT_MEAN, max_iters=1000, tol=1e-12):
    """Task-INDEPENDENT, doubly-balanced L2E feedforward init (Sinkhorn-Knopp).

    1. Sample a narrow positive matrix Z[j,i] ~ Uniform(1-jitter, 1+jitter).
    2. Alternately normalize rows and columns until every L2E has equal total incoming
       weight AND every input pixel has equal total outgoing weight across L2E.
    3. Scale the balanced matrix so its overall MEAN entry == target_mean.

    Depends only on (rng, n_out, n_pix, jitter) -- it never inspects PATTERNS, never
    privileges the center pixel, and encodes no line templates. jitter=0 yields an
    exactly uniform matrix (every entry == target_mean, perfectly symmetric); jitter>0
    injects small UNBIASED differences for the learning/competition rules to amplify.
    Result: each L2E's incoming sum == target_mean*n_pix, each pixel's outgoing sum ==
    target_mean*n_out, so no neuron or pixel is privileged at t=0."""
    Z = rng.uniform(1.0 - jitter, 1.0 + jitter, size=(n_out, n_pix))
    B = _sinkhorn_balance(Z, n_out / n_pix, max_iters=max_iters, tol=tol)
    return B * (target_mean / B.mean())


def legacy_wide_feedforward_init(rng, n_out, n_pix):
    """ABLATION: the original wide-uniform init, Uniform(50, 200) (mean 125). Kept so
    the balanced init can be A/B'd against the historical unbalanced starting point."""
    return rng.uniform(50, 200, size=(n_out, n_pix))


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
                 # Membrane leak is opt-in for every trainable population.
                 # L2E, L2I, and L1I remain independently controllable.
                 leak_enabled: bool = False,
                 l2i_leak_enabled: bool = False,
                 # L1I is a pure accumulator by default when immediate relay is
                 # disabled. Its former fast evidence-window leak remains available
                 # as an explicit ablation rather than being imposed unconditionally.
                 l1i_leak_enabled: bool = False,
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
                 l2i_threshold_frac: float = 1,   # unit multiplier (1 == unchanged)
                 # L1I threshold as a fraction of L2I's resolved threshold.
                 # 1.0 makes the two inhibitory populations match exactly.
                 l1i_threshold_frac: float = 1,
                 ei_sat_mult: float = 1,          # unit multiplier (1 == unchanged)
                 l1i_ei_init_frac: float | None = None,
                 # Confidence-gated consolidation in neuron_flexible.Neuron.
                 # Enabled on L2E by default: a local, label-free
                 # consolidation rule that protects mature specialists, depresses
                 # losing near-winners, and keeps stale neurons reusable. All the
                 # numeric knobs below are DIMENSIONLESS (scale-invariant) except
                 # conf_cap, which is a weight and scales with thr_l2 via conf_cap_frac.
                 confidence_consolidation: bool = True,
                 loser_depression: bool = True,
                 # Loser weight-update rule for the L2I competitive-reset event
                 # after the unweighted competitive-reset event. Three modes:
                 #   "redistribution" -- conservative ON->OFF gate recruitment (new default);
                 #   "depression"     -- retained one-sided charge-scaled loser depression (A/B baseline);
                 #   "none"           -- hard-reset-only control (no loser weight update).
                 # None (absent) maps from the legacy loser_depression bool for old
                 # experiment files: True -> "depression", False -> "none".
                 competitive_weight_update: str | None = None,
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
                 # L1I feedback firing mode. DEFAULT OFF: L1I is a trainable
                 # threshold accumulator. When enabled it acts as an IMMEDIATE
                 # DETERMINISTIC RELAY: any L1I that receives a nonzero L2E
                 # feedback signal fires in that same step, with NO membrane
                 # accumulation, NO learned-threshold crossing, and NO dependence on
                 # L1I feedback-weight training (the learned integrator introduced a
                 # phase shift and was not useful). With the flag off, L1I fires only
                 # when accumulated feedback crosses its own threshold. Either way L1I output stays binary
                 # and the L1I->L1E inhibition delivery path is unchanged.
                 l1i_immediate_relay: bool = False,
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
                 # Per-L2I override of excitatory flow. None (default) = L2I follows
                 # excitatory_flow_rate like L2E. Set False to give the shared L2I
                 # INSTANT charge delivery (V += weight in one step) while L2E keeps
                 # flow: under flow, a single L2E->L2I spike is spread over decaying
                 # steps while L2I leaks, so its peak reaches only ~0.6x the weight --
                 # and since the weight is capped AT L2I's threshold, NO single spike
                 # can ever fire L2I (no one-shot single-source relay is possible).
                 # Instant delivery makes a trained (weight==theta) synapse fire L2I
                 # from one spike, the "now-trusted source relays L2I" mechanism.
                 l2i_excitatory_flow_rate: bool | None = None,
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
                 # Distance attenuation of DELIVERED L1E->L2E charge. Physical layout
                 # distances are divided by the nearest feedforward distance, producing
                 # dimensionless d >= 1. Delivery multiplies each stored weight by
                 # (distance_ref/max(d, distance_min))**distance_power. With the default
                 # ref=min=1 the nearest feedforward synapse delivers at full strength
                 # and all farther synapses are attenuated. Stored weights and learning
                 # deltas are untouched. Engine default OFF; dashboard default ON.
                 distance_weighting: bool = False,
                 distance_power: float = 2.0,
                 distance_ref: float = 1.0,
                 distance_min: float = 1.0,
                 # Seeded local scatter of the functional 3D layout.
                 # Default ON: L1 jitters around retinotopic anchors and L2 is sampled
                 # as a bounded cloud, so no layer is an exact grid, ring, or plane.
                 # OFF selects the deterministic regular grid/ring geometry.
                 # The scatter draws from an INDEPENDENT layout RNG spawned from the
                 # engine seed, so adding it does not shift weight-init draws.
                 layout_scatter_enabled: bool = True,
                 l1_xy_jitter: float = 0.9,     # capped below half a retinotopic cell
                 l1_z_jitter: float = 0.55,     # L1E out-of-plane scatter
                 l1i_pair_jitter: float = 1.2,  # extra L1I offset around its paired L1E
                 l2_xy_radius: float = 6.0,     # radius of the irregular L2E cloud
                 l2_z_jitter: float = 1.8,      # L2E height scatter around its layer center
                 layout_min_separation: float = 1.2,  # generic final spacing floor
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
                 # that drives round-robin firing. L2E only; default off.
                 subtractive_reset: bool = False,
                 # Hard-reset inhibition on losing L2E. When on, an L2I->L2E
                 # discharge clamps the non-winner's membrane back to rest AFTER the
                 # inhibitory plasticity rule has read its pre-reset charge, so loser
                 # charge carryover (the round-robin's engine) drops to zero. L2E
                 # only; default off keeps the small subtractive/flow gate.
                 l2i_hard_reset_losers: bool = False,
                 # Clear the excitatory/inhibitory current traces on a hard reset so
                 # no residual flow refills the membrane after it is clamped to rest.
                 hard_reset_clear_traces: bool = True,
                 # Membrane saturation ceiling for L2E, as a multiple of thr_l2
                 # (None = unbounded, the baseline). Bounds accumulated charge so
                 # the membrane can't ratchet to ~2-3x threshold between cycles;
                 # this keeps the (small, capped) L2I->L2E gate in a range where
                 # it can actually regulate firing frequency -- inhibition, not a
                 # pile-up, sets who fires. Local finite driving force; L2E only.
                 v_sat_frac: float | None = None,
                 # Minimal signed-spike feedforward learning. Replaces the
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
                 # Structural free-energy plasticity gate (see Neuron and the
                 # methodology doc). EXCITATORY postsynaptic neurons only (L2E
                 # feedforward). When on, the signed-spike learning rate is scaled by
                 # gate = max(eta_floor, 1 - clamp(sum_positive_afferents/theta, 0, 1))
                 # -- a structural, input/voltage-independent consolidation brake
                 # that REPLACES the voltage term p in the signed rule. Under-built
                 # L2E stay plastic; mature specialists slow down and resist being
                 # reshaped on a later pattern. Default OFF -> signed rule uses p.
                 structural_free_energy: bool = False,
                 structural_fe_eta_floor: float = 0.02,
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
                 pos_weight_floor: int | None = None,
                 # L2E feedforward INITIALIZATION (see balanced_feedforward_init).
                 # 'balanced': task-independent doubly-balanced Sinkhorn init
                 # -- narrow jitter, then row/col-normalized so every L2E has equal
                 # total incoming weight and every pixel equal total outgoing weight,
                 # scaled to the legacy mean (125). A FAIR developmental start: no
                 # neuron/pixel privileged, no task structure. 'legacy_wide' is the
                 # unconstrained Uniform(50,200) init and is the default.
                 l2e_init_mode: str = 'legacy_wide',
                 # Jitter eps for the balanced init: Z[j,i] ~ Uniform(1-eps, 1+eps).
                 # eps=0 -> exactly uniform (perfect symmetry); eps>0 -> small UNBIASED
                 # differences the learning rules must amplify. Ignored in legacy mode.
                 l2e_init_jitter: float = L2E_INIT_JITTER):
        # Decouple the SENSORY input rate from the INTRINSIC competition clock.
        # input_period: steps between external input samples. The default is 1:
        #   a held pixel supplies drive continuously, every simulation step.
        # cycle_period: the intrinsic gamma-like clock that structures L2
        #   competition and L2I evidence integration. FIXED regardless of the
        #   input rate. It retains volley_period as its default, independently of
        #   the now-continuous sensory drive.
        input_period = 1 if input_period is None else input_period
        cycle_period = volley_period if cycle_period is None else cycle_period
        # Separate learning-rate controls for the E and I populations (Phase 1 of
        # the symmetry-breaking plan). Each defaults to the module constant that
        # was previously applied uniformly, so the defaults reproduce prior
        # behavior exactly. l2i_threshold_frac scales L2I from threshold_l2;
        # l1i_threshold_frac then scales L1I from the resolved L2I threshold. Their E->I init range and
        # learning cap scale with that same (possibly lowered) threshold below,
        # so lowering an I threshold does not trivially overdrive it.
        l2e_lr_frac = ETA_FRAC if l2e_lr_frac is None else l2e_lr_frac
        l2i_lr_frac = ETA_FRAC if l2i_lr_frac is None else l2i_lr_frac
        l1i_lr_frac = ETA_FRAC if l1i_lr_frac is None else l1i_lr_frac
        # Resolve the competitive-reset loser rule. When absent, fall back to the
        # legacy loser_depression bool (True -> "depression", False -> "none") so old
        # experiment files replay unchanged; otherwise use the canonical mode flag.
        if competitive_weight_update is None:
            competitive_weight_update = "depression" if loser_depression else "none"
        if competitive_weight_update not in ("redistribution", "depression", "none"):
            raise ValueError("competitive_weight_update must be 'redistribution', "
                             f"'depression', or 'none'; got {competitive_weight_update!r}")
        self.params = dict(seed=seed, threshold=threshold, threshold_l2=threshold_l2,
                           leak_l1=leak_l1, leak_l2=leak_l2, leak_enabled=leak_enabled,
                           l2i_leak_enabled=l2i_leak_enabled,
                           l1i_leak_enabled=l1i_leak_enabled,
                           learning_rate=learning_rate, weight_cap=weight_cap,
                           refractory=refractory, volley_period=volley_period,
                           input_period=input_period, cycle_period=cycle_period,
                           homeostasis=homeostasis, ca_rate=ca_rate, ca_target=ca_target,
                           homeo_up=homeo_up, homeo_down=homeo_down,
                           l2e_lr_frac=l2e_lr_frac, l2i_lr_frac=l2i_lr_frac,
                           l1i_lr_frac=l1i_lr_frac,
                           l2i_threshold_frac=l2i_threshold_frac,
                           l1i_threshold_frac=l1i_threshold_frac,
                           ei_sat_mult=ei_sat_mult,
                           l1i_ei_init_frac=l1i_ei_init_frac,
                           confidence_consolidation=confidence_consolidation,
                           loser_depression=loser_depression,
                           competitive_weight_update=competitive_weight_update,
                           conf_cap_frac=conf_cap_frac, conf_beta=conf_beta,
                           eta_min=eta_min, eta_loss=eta_loss, loss_gamma=loss_gamma,
                           conf_rho_active=conf_rho_active, conf_rho_dead=conf_rho_dead,
                           conf_ca_dead=conf_ca_dead, conf_grace=conf_grace,
                           signed_depression=signed_depression, eta_off=eta_off,
                           l2e_budget=l2e_budget, event_driven=event_driven,
                           l2_charge_chunks=l2_charge_chunks,
                           l1i_immediate_relay=l1i_immediate_relay,
                           excitatory_flow_rate=excitatory_flow_rate,
                           l2i_excitatory_flow_rate=l2i_excitatory_flow_rate,
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
                           layout_scatter_enabled=layout_scatter_enabled,
                           l1_xy_jitter=l1_xy_jitter, l1_z_jitter=l1_z_jitter,
                           l1i_pair_jitter=l1i_pair_jitter,
                           l2_xy_radius=l2_xy_radius, l2_z_jitter=l2_z_jitter,
                           layout_min_separation=layout_min_separation,
                           lasting_inhibition=lasting_inhibition, inh_decay=inh_decay,
                           inh_boost_frac=inh_boost_frac,
                           subtractive_reset=subtractive_reset,
                           l2i_hard_reset_losers=l2i_hard_reset_losers,
                           hard_reset_clear_traces=hard_reset_clear_traces,
                           v_sat_frac=v_sat_frac,
                           signed_spike_learning=signed_spike_learning,
                           structural_free_energy=structural_free_energy,
                           structural_fe_eta_floor=structural_fe_eta_floor,
                           l2e_weight_cap_frac=l2e_weight_cap_frac,
                           pos_weight_floor=pos_weight_floor,
                           l2e_init_mode=l2e_init_mode,
                           l2e_init_jitter=l2e_init_jitter)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        p = self.params
        # NEUTERED (reversible): the trace-based flow-rate delivery is permanently
        # OFF. The task is now the minimal loop -- accumulate -> fire -> learn ->
        # inhibit -> chunk (instantaneous charge + chunked WTA race for timing).
        # The flow-rate / current-trace / assembly-flow CODE still exists, but this
        # is the single choke point both __init__ and apply_config funnel through,
        # so pinning the params here means no constructor arg or live config toggle
        # can re-enable trace accumulation. To restore, delete this block. See
        # snn/rules/delivery.py (FlowRateDelivery) and excitatory.py (AssemblyFlowCredit).
        p['excitatory_flow_rate'] = False
        p['l2i_excitatory_flow_rate'] = False
        p['inhibitory_flow_rate'] = False
        p['assembly_flow_credit'] = False
        # INDEPENDENT RNG streams for the three weight inits, derived deterministically
        # from the engine seed (SeedSequence.spawn gives statistically independent
        # children). Feedforward, L2E->L2I, and feedback thus no longer share/consume
        # one stream in a fixed order -- each is reproducible from the seed on its own,
        # and reordering or resizing one init cannot shift the others.
        # spawn(4): children 0-2 are byte-identical to the previous spawn(3), so
        # adding the layout stream does NOT shift the feedforward / E->I / feedback
        # weight draws. Child 3 is the INDEPENDENT layout RNG.
        rng_ff, rng_ei, rng_fb, rng_layout = (
            np.random.default_rng(s)
            for s in np.random.SeedSequence(p['seed']).spawn(4))
        # One source of truth for geometry: generate every neuron's final physical 3D
        # position once. _register_neurons exposes it to the renderer; _apply_distances
        # derives dimensionless feedforward distances without rescaling the positions.
        self.layout = generate_layout(p, rng_layout, N_PIX, N_OUT)
        thr_l1 = p['threshold']      # L1 neurons fire on a single pixel hit
        thr_l2 = p['threshold_l2']   # L2 neurons must accumulate many volleys
        # Inhibitory-neuron thresholds (Phase 2): each defaults to its excitatory
        # layer's threshold (frac 1.0 -> unchanged). When lowered, the E->I init
        # range and learning cap below scale with THIS threshold, not the E one.
        thr_l2i = thr_l2 * p['l2i_threshold_frac']   # L2I's own firing threshold
        thr_l1i = thr_l2i * p['l1i_threshold_frac']  # 1.0 => L1I exactly matches L2I

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
        # L2E->L1I feedback init. Default (None) draws each SOURCE gate in the
        # [0.25, 0.5] integrator range. l1i_ei_init_frac can collapse that range
        # to one fraction for ablation. The vector is shared across the L1I bank
        # below because these neurons receive the same global winner stream and
        # differ only in which L1E pixel they suppress.
        if p['l1i_ei_init_frac'] is None:
            lo_frac, hi_frac = L1_EI_WEIGHT_INIT_LOW_FRAC, L1_EI_WEIGHT_INIT_HIGH_FRAC
        else:
            lo_frac = hi_frac = p['l1i_ei_init_frac']
        # Every L1I receives the same L2E winner stream and is intended to suppress
        # its paired pixel in phase with the rest of the active input. Give the bank
        # one task-independent feedback vector; independent random vectors create
        # arbitrary phase groups with no pixel-local signal that could correct them.
        l1i_feedback_init = rng_fb.uniform(lo_frac * thr_l1i, hi_frac * thr_l1i,
                                           size=N_OUT)
        for inh in self.l1.inhibitory_neurons:
            inh.threshold = thr_l1i    # Phase 2: L1I's own (possibly lowered) threshold
            # InputLayer constructed L1I with L1E's cap. Raise the cap before
            # assigning L2I-scale feedback weights or SynapseBank.set_weights()
            # would clip every initialization value above the old L1E threshold.
            inh.weight_cap = thr_l1i
            inh.weights = l1i_feedback_init.copy()
            inh.leak_rate = L1I_LEAK_RATE if p['l1i_leak_enabled'] else 0.0
            inh.refractory_period = L1I_FEEDBACK_REFRACTORY

        # L2E leak is independently switchable from the shared L2I neuron's fast
        # evidence-window leak.
        eff_leak_l2 = p['leak_l2'] if p['leak_enabled'] else 0.0
        # Active L2 has NO learned L2I->L2E gate: build the column without the
        # index-0 local-I placeholder so each L2E has exactly N_PIX pixel afferents.
        # Competition is the unweighted hard reset + competitive depression in
        # _resolve_l2_competition.
        self.l2 = CorticalColumn(n_neurons=N_OUT, threshold=thr_l2,
                                 refractory_period=p['refractory'], learning_rate=p['learning_rate'],
                                 weight_cap=thr_l2, leak_rate=eff_leak_l2,
                                 include_local_inhibition=False)
        self.l2.setup_connectivity(n_feedforward_inputs=N_PIX, n_feedback_inputs=0)
        self.l2.finalize_connections()
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
            rng_ei.uniform(L2_EI_WEIGHT_INIT_LOW_FRAC * thr_l2i,
                           L2_EI_WEIGHT_INIT_HIGH_FRAC * thr_l2i, size=N_OUT))
        # Small positive feedforward weights: neurons must accumulate across many
        # volleys initially (LIF phase), then specialise toward single-volley firing
        # (pattern integrator phase). 'balanced' is the optional task-independent
        # Sinkhorn init (equal incoming/outgoing totals, mean 125). DEFAULT
        # 'legacy_wide' is the unconstrained Uniform(50,200) developmental draw.
        # Both keep mean 125 and draw from the dedicated feedforward stream (rng_ff).
        if p['l2e_init_mode'] == 'legacy_wide':
            ff_weights = legacy_wide_feedforward_init(rng_ff, N_OUT, N_PIX)
        else:  # 'balanced'
            ff_weights = balanced_feedforward_init(rng_ff, N_OUT, N_PIX,
                                                   jitter=p['l2e_init_jitter'],
                                                   target_mean=L2E_INIT_MEAN)
        self.l2.set_feedforward_weights(ff_weights)
        self.l2.inhibitory_neuron.refractory_period = 0
        self.l2.inhibitory_neuron.threshold = thr_l2i   # Phase 2: L2I's own threshold
        # Short evidence-retention window: much faster than L2E's leak_l2 (its
        # slow multi-volley accumulator), controlled by its own dashboard switch.
        self.l2.inhibitory_neuron.leak_rate = (
            L2I_LEAK_RATE if p['l2i_leak_enabled'] else 0.0
        )

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
                # L2 competition is an unweighted hard reset plus the selected
                # loser update in Neuron.apply_competitive_reset.
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
                # Confidence-gated consolidation in neuron_flexible.Neuron: local,
                # label-free
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
                # Hard-reset inhibition (L2E only; default off). A real L2I->L2E
                # discharge clamps the loser to rest after inhibitory learning has
                # read its pre-reset charge, eliminating loser charge carryover.
                n.l2i_hard_reset_losers = p['l2i_hard_reset_losers']
                n.hard_reset_clear_traces = p['hard_reset_clear_traces']
                # Membrane saturation ceiling (L2E only; None = unbounded). Bounds
                # accumulated charge near threshold so inhibition can regulate it.
                n.v_sat = p['v_sat_frac'] * thr_l2 if p['v_sat_frac'] else None
                # Minimal signed-spike feedforward learning (L2E only; default off).
                # When on it takes over _update_weights entirely (see the neuron).
                n.signed_spike_learning = p['signed_spike_learning']
                # Structural free-energy plasticity gate (L2E / excitatory only).
                # Scales the signed-spike eta by this neuron's own positive-afferent
                # maturity vs its threshold; default off leaves the p-scaled rule.
                n.structural_free_energy = p['structural_free_energy']
                n.structural_fe_eta_floor = p['structural_fe_eta_floor']
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
                    if nid.startswith('L1'):
                        # L1I membrane charge integrates a sequence of L2 winners.
                        # Credit every contributor in that same membrane window, not
                        # only the final winner that happened to cross threshold.
                        n.assembly_flow_credit = True
                        n.assembly_decay_frac = 0.0
                    else:
                        n.assembly_flow_credit = p['assembly_flow_credit']
                        n.assembly_decay_frac = p['assembly_decay_frac']

        # Uniform per-neuron delivery / flow-rate / inhibitory-gate-rule / distance
        # config. The engine's params are the SOURCE OF TRUTH; NeuronConfig is a
        # transport built from them (from_engine_params) and applied population-aware
        # to each neuron -- replacing the scattered per-attribute assignments. See
        # snn/config.py.
        cfg = NeuronConfig.from_engine_params(p)
        for nid, n in self.neurons.items():
            cfg.apply_to(n, is_l1e=nid.startswith('L1E'), is_l1i=nid.startswith('L1I'),
                         is_l2i=(nid == 'L2I'))
        # apply_to sets the delivery-distance flag on L2E; now derive its actual
        # per-synapse model distances from the final physical layout. Must run after
        # weights are finalized because set_weights resets distance to ones.
        self._apply_distances()

        # One-step feedback delay. L1I spikes produced at t inhibit paired L1E
        # neurons at t+1 only; the register is replaced every step.
        self.l1i_feedback_delay = np.zeros(N_PIX)
        initial_pattern = next(iter(PATTERNS))
        self.input_vec = np.array(PATTERNS[initial_pattern], dtype=float)
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
        self.l2_charge: dict[str, float] = {}         # POST-inhibition, pre-update diagnostic
        self.l2_inh_phase_debug: list[dict] = []      # per-inhibited-L2E phase record (see step())
        self.winner: str | None = None
        self._inh_events: list[tuple] = []   # (neuron_id, event) from this step's discharges
        self._reset_events: list[tuple] = []  # (neuron_id, record) from L2 competitive resets

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
        self.current_pattern = initial_pattern    # name backing self.input_vec
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
        pos = self.layout
        for i in range(N_PIX):
            nid = f'L1E{i}'
            self.neurons[nid] = self.l1.excitatory_neurons[i]
            self.meta[nid] = dict(id=nid, label=f'in {i}', layer='L1', type='E',
                                  threshold=self.params['threshold'],
                                  pos=pos[nid].tolist())
        for i in range(N_PIX):
            nid = f'L1I{i}'
            self.neurons[nid] = self.l1.inhibitory_neurons[i]
            self.meta[nid] = dict(id=nid, label=f'inh {i}', layer='L1', type='I',
                                  threshold=self.params['threshold'],
                                  pos=pos[nid].tolist())
        for j in range(N_OUT):
            nid = f'L2E{j}'
            self.neurons[nid] = self.l2.excitatory_neurons[j]
            self.meta[nid] = dict(id=nid, label=f'out {j}', layer='L2', type='E',
                                  threshold=self.params['threshold_l2'], pos=pos[nid].tolist())
        self.neurons['L2I'] = self.l2.inhibitory_neuron
        self.meta['L2I'] = dict(id='L2I', label='inhib', layer='L2', type='I',
                                threshold=self.params['threshold_l2'], pos=pos['L2I'].tolist())

        self.synapses: list[dict] = []
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.synapses.append(dict(id=f'ff{i}->{j}', source=f'L1E{i}', target=f'L2E{j}', kind='feedforward'))
        # Structural (unweighted) L2I->L2E reset fanout: conveys the binary
        # competitive_reset event, NOT a synaptic weight. It stays visible (the
        # inhibitory fanout still exists) but carries weight=null and never appears
        # in weight snapshots, weight-change tracking, or the weights graph.
        for j in range(N_OUT):
            self.synapses.append(dict(id=f'reset->{j}', source='L2I', target=f'L2E{j}',
                                      kind='reset_inhibition'))
        for j in range(N_OUT):
            self.synapses.append(dict(id=f'{j}->inh', source=f'L2E{j}', target='L2I', kind='excitation'))
        for i in range(N_PIX):
            self.synapses.append(dict(id=f'li{i}', source=f'L1I{i}', target=f'L1E{i}', kind='inhibition'))
        for j in range(N_OUT):
            for i in range(N_PIX):
                self.synapses.append(dict(id=f'fb{j}->{i}', source=f'L2E{j}', target=f'L1I{i}', kind='feedback'))

    def _apply_distances(self):
        """Derive L1E->L2E model distances from the final physical/display layout.

        Physical coordinates stay untouched. The physical feedforward matrix is
        divided by its global minimum, producing dimensionless model distances >= 1.
        With distance_ref=distance_min=1, effective delivery factors are therefore in
        (0, 1]: the nearest feedforward connection is neutral and farther connections
        deliver less charge. One global unit preserves both within- and between-neuron
        geometric variation. Stored weights are never modified.
        """
        l1e_pos = [self.layout[f'L1E{i}'] for i in range(N_PIX)]
        l2e_pos = [self.layout[f'L2E{j}'] for j in range(N_OUT)]
        physical = np.array([
            [float(np.linalg.norm(l2e_pos[j] - l1e_pos[i])) for i in range(N_PIX)]
            for j in range(N_OUT)
        ])
        dmin = float(physical.min())
        if not np.isfinite(dmin) or dmin <= 1e-9:
            raise ValueError('feedforward layout contains a zero or invalid distance')
        self.feedforward_distance_unit = dmin
        self.physical_ff_distances = physical
        model = physical / dmin

        for j in range(N_OUT):
            n = self.l2.excitatory_neurons[j]
            if n._weights_array is None or len(n._weights_array) != N_PIX:
                continue
            n.distance = model[j]

    # --------------------------------------------------------------- controls
    def reset(self):
        self._build()

    def reseed(self):
        """Draw a fresh random seed and rebuild from new random initial weights,
        preserving every other tunable -- so it works under ANY plasticity/config
        combination (only the seed-driven random draws change: L2E feedforward,
        E->I, L1I). Like reset(), this rebuilds the network and wipes learned
        state -- reseeding *is* a randomized reset. Returns the new seed (which
        becomes the current seed, so a subsequent Reset reproduces this network)."""
        self.params['seed'] = int(np.random.SeedSequence().generate_state(1)[0])
        self._build()
        return self.params['seed']

    # Parameters the dashboard is allowed to change live. Anything not listed here
    # is rejected so a stray key can't silently no-op or corrupt self.params.
    TUNABLE = ('signed_depression', 'eta_off', 'l2e_budget', 'l2e_lr_frac',
               'confidence_consolidation', 'loser_depression', 'eta_loss',
               'competitive_weight_update',
               'eta_min', 'conf_cap_frac', 'leak_l2', 'event_driven',
               'subtractive_reset', 'l2i_hard_reset_losers',
               'hard_reset_clear_traces', 'refractory', 'v_sat_frac',
               'signed_spike_learning', 'structural_free_energy',
               'structural_fe_eta_floor', 'seed', 'l2_charge_chunks',
               'l1i_immediate_relay', 'excitatory_flow_rate',
               'l2i_excitatory_flow_rate', 'exc_trace_decay',
               'exc_trace_normalized', 'inhibitory_flow_rate', 'inh_trace_decay',
               'inh_trace_normalized', 'inhibitory_delta_rule', 'inhibitory_rule_mode',
               'inhibitory_eta_up', 'inhibitory_eta_down', 'inhibitory_p_max',
               'inhibitory_margin_frac', 'inhibitory_delta_eta',
               'distance_weighting', 'distance_power', 'distance_ref', 'distance_min',
               'layout_scatter_enabled', 'l1_xy_jitter',
               'l1_z_jitter', 'l1i_pair_jitter', 'l2_xy_radius', 'l2_z_jitter',
               'layout_min_separation',
               'assembly_flow_credit', 'assembly_decay_frac',
               'l2e_init_mode', 'l2e_init_jitter', 'leak_enabled',
               'l2i_leak_enabled', 'l1i_leak_enabled')

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
                     'l2i_hard_reset_losers', 'hard_reset_clear_traces',
                     'signed_spike_learning', 'structural_free_energy',
                     'l1i_immediate_relay',
                     'excitatory_flow_rate', 'l2i_excitatory_flow_rate',
                     'exc_trace_normalized',
                     'inhibitory_flow_rate', 'inh_trace_normalized',
                     'inhibitory_delta_rule', 'distance_weighting',
                     'layout_scatter_enabled',
                     'assembly_flow_credit', 'leak_enabled', 'l2i_leak_enabled',
                     'l1i_leak_enabled'):
                v = bool(v)
            elif k in ('seed', 'refractory', 'l2_charge_chunks'):
                v = int(v)
            elif k == 'l2e_init_mode':
                # Exposed as a dashboard TOGGLE (bool): on -> balanced, off ->
                # legacy_wide. A raw string is also accepted (programmatic callers).
                v = ('balanced' if v else 'legacy_wide') if isinstance(v, bool) else str(v)
            elif k == 'inhibitory_rule_mode':
                v = str(v)
            elif k == 'competitive_weight_update':
                v = str(v)
                if v not in ('redistribution', 'depression', 'none'):
                    continue   # reject an unknown mode rather than corrupting params
            else:
                v = float(v)
            self.params[k] = v
            applied[k] = v
        self._build()
        self._log('control', f'config applied: {applied}')
        return applied

    def set_feedforward_weight(self, j: int, i: int, weight: float) -> float:
        """Manually set L2E neuron j's feedforward weight FROM pixel i (afferent index
        i; the active L2E has no index-0 placeholder), clipped to [0, this neuron's
        per-afferent cap]. Used by the dashboard RF panel to push a neuron toward
        winner/loser by hand -- typically while paused, then step/resume to see the
        effect. Returns the applied (clipped) value."""
        if not (0 <= j < N_OUT and 0 <= i < N_PIX):
            raise IndexError(f"L2E index j={j} or pixel i={i} out of range")
        n = self.l2.excitatory_neurons[j]
        cap = float(n.weight_cap)
        w = float(np.clip(weight, 0.0, cap))
        arr = n._weights_array.copy()
        arr[i] = w
        n._weights_array = arr
        self._log('control', f'L2E{j} <- pixel {i} weight set to {w:.1f}')
        return w

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

    def _check_l2_reset_phases(self, l2, v_start):
        """Build per-L2E reset-phase records for this step's competitive-reset events
        and warn (never crash) if the reset invariant is violated. The invariant: a
        competitive reset must leave the target at EXACT rest (v_post ==
        resting_potential) and no later same-timestep operation (leak, trace advance)
        may restore charge (v_end, measured after leak, must not rise above rest).
        Read-only: it inspects this step's reset records and the current membrane; it
        never mutates a neuron."""
        self.l2_inh_phase_debug = []
        for nid, rec in self._reset_events:
            j = int(nid[3:])
            neuron = l2.excitatory_neurons[j]
            v_end = float(neuron.potential)
            rest = float(neuron.resting_potential)
            self.l2_inh_phase_debug.append(dict(
                id=nid,
                v_start=round(v_start.get(j, 0.0), 3),
                v_pre=round(rec['v_pre'], 3),      # charge before the reset
                p_loss=round(rec['p_loss'], 3),
                # depression mode records p_loss/depressed gates; redistribution mode
                # records p_match and the ON->OFF transfer. Both are surfaced so the
                # phase panel reflects whichever loser rule ran.
                p_match=round(rec.get('p_match', 0.0), 3),
                refractory=bool(rec.get('refractory_at_arrival', False)),
                depressed=len(rec['depressed_indices']),
                transferred=round(rec.get('transferred', 0.0), 3),
                v_post=round(rec['v_post'], 3),    # after the reset (should be rest)
                v_end=round(v_end, 3)))
            if abs(rec['v_post'] - rest) > 1e-6:
                self._log('warn', f"{nid}: competitive reset did not reach rest "
                                  f"(v_post={rec['v_post']:.1f}, rest={rest:.1f})")
            if v_end > rest + 1e-6:
                self._log('warn', f"{nid}: charge rose above rest after reset within the "
                                  f"same timestep (v_end={v_end:.1f}) -- unexpected "
                                  f"post-reset recharge")

    def _resolve_l2_competition(self, l2, l2e, t, current_input):
        """Attempt one standard argmax WTA resolution on the CURRENT L2E membrane
        state. If any L2E crossed threshold, the max-charge crosser fires, drives
        L2I, and (if L2I fires) the unweighted competitive-reset event is broadcast
        to every L2E, including the winner.
        Each neuron unconditionally hard-resets its membrane/current traces; the
        loser weight update runs ONLY for the non-refractory neurons (the winner
        fired this step and is in refractory, so it is protected -- winner protection
        comes solely from the refractory timer, not a one-hot winner check). The
        resolved `competitive_weight_update` mode and the current feedforward input
        mask are passed to every reset. Mutates `l2e` in place (sets the winner's
        one-hot bit). Returns (l2i, inhibited, winner); winner is None and L2I is left
        UNTOUCHED when nobody crossed, so a caller running this per charge-chunk can
        drive L2I's no-winner integration exactly once after the chunk loop instead
        of K times. `t` is the current outer timestep (passed to L2I flow-rate charge).
        """
        eligible = [j for j, e in enumerate(l2.excitatory_neurons) if e.check_threshold()]
        if not eligible:
            return 0.0, [], None
        winner = max(eligible, key=lambda j: l2.excitatory_neurons[j].potential)
        l2.excitatory_neurons[winner].fire()
        l2e[winner] = 1.0
        # The winner drives L2I, which fires (E->I weight = thr_l2) and issues the
        # competitive-reset event to the WHOLE pool.
        l2.inhibitory_neuron.receive_input(l2e, t=t)
        l2i = 1.0 if l2.inhibitory_neuron.check_threshold() else 0.0
        inhibited = []
        mode = self.params['competitive_weight_update']
        if l2i:
            l2.inhibitory_neuron.fire()
            for j in range(N_OUT):
                # Broadcast to EVERY L2E, winner included. The hard reset is
                # unconditional; the refractory winner skips the loser weight update
                # internally (refractory_timer > 0 at arrival).
                rec = l2.excitatory_neurons[j].apply_competitive_reset(
                    current_input=current_input, competitive_weight_update=mode)
                self._reset_events.append((f'L2E{j}', rec))
                if j != winner:
                    inhibited.append(j)
        return l2i, inhibited, winner

    # ------------------------------------------------------------------- step
    def step(self) -> dict:
        l1, l2 = self.l1, self.l2
        t = self.timestep

        # 1. L1E: [paired I1's previous spike (local inhibition), external pixel].
        #    A held pixel supplies drive every input_period steps; input_period=1
        #    (default) is constant presentation.
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
        self._reset_events = []
        for i, e in enumerate(l1.excitatory_neurons):
            ext = 1.0 if (input_arrives and self.input_vec[i] > 0.5) else 0.0
            e.receive_input(np.array([0.0, ext]))
            # L1I feedback is a one-step delayed pulse. With constant presentation,
            # a pulse at t exactly cancels its paired pixel drive at t+1 and is then
            # consumed when this register is replaced at the end of the step.
            inh = float(self.l1i_feedback_delay[i]) if input_arrives else 0.0
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
        #     A threshold-crossing winner drives L2I. If L2I fires, it broadcasts an
        #     unweighted hard reset to the entire L2E pool. The refractory winner is
        #     protected from the loser weight update, but its transient charge is
        #     reset like every other L2E. See _resolve_l2_competition.
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
                ff_vec[i] = 1.0

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
        # consumers); l2_charge is the post-inhibition, pre-update diagnostic. The
        # dashboard reports the live end-of-step membrane for every population.
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
                l2i, inhibited, winner = self._resolve_l2_competition(l2, l2e, t, ff_vec)
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

        # PHASE 7: retain the charge AFTER inhibition and BEFORE the membrane update
        # for phase diagnostics. This read-only snapshot does not advance traces or
        # mutate any neuron; dynamic_state() reports the later live membrane phase.
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
            self.emitted.append(f'reset->{j}')
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

        # Reset-phase guard + per-L2E diagnostics for this step's competitive resets
        # (V_end is post-leak here; it must stay at rest).
        self._check_l2_reset_phases(l2, l2_v_start)

        # 5. Bookkeeping.
        self._record_spikes(l1e, l1i, l2e, l2i)
        # Queue this step's L1I spikes as one-step inhibition for t+1. Replacing
        # rather than OR-latching the register is what permits the E/silent/E rhythm.
        self.l1i_feedback_delay = l1i.copy()
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
            for i in range(N_PIX):
                w[f'ff{i}->{j}'] = float(arr[i])
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
                c[f'ff{i}->{j}'] = float(conf[i])
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
        # The structural reset fanout has no weight: serialize kind='reset_inhibition'
        # with weight=null so it never reads as a learned magnitude.
        synapses = [dict(**s,
                         weight=(None if s['kind'] == 'reset_inhibition'
                                 else round(weights.get(s['id'], 0.0), 4)),
                         confidence=round(confidence[s['id']], 4) if s['id'] in confidence else None)
                    for s in self.synapses]
        return dict(neurons=neurons, synapses=synapses, layers=['L1', 'L2'],
                    patterns=list(PATTERNS.keys()),
                    pattern_vectors={k: list(map(int, v)) for k, v in PATTERNS.items()},
                    grid=dict(rows=3, cols=3), params=self.params)

    def dynamic_state(self) -> dict:
        neurons = []
        for nid, n in self.neurons.items():
            # Report one coherent phase for every population: the live membrane at
            # the end of the completed step. Previously L2E used the pre-update
            # l2_charge snapshot while L2I used this post-update value, which hid
            # L2E leak but exposed L2I leak/reset in the same chart. l2_drive and
            # l2_charge remain available internally as phase diagnostics.
            pot = float(n.potential)
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
