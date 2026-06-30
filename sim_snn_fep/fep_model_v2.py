"""
fep_model_v2.py -- conductance-based spiking fabric (per docs/plans/2026-06-29-spiking-fabric-redesign.md).

Tileable fabric: Population (population.py) + Projection (projection.py) + this Network.
Goals this iteration: competition, learning, self-organisation, composition, hierarchy.

Architecture (each layer = an E population + a local inhibitory interneuron implementing
k-WTA via DELAYED feedback shunting inhibition -- no clamp, no argmax):

    L1 (input spike sources) --plastic STDP--> L2_E (features)  --plastic STDP--> L3_E (compositions)
                                                 ^  L2_I (WTA)                      ^  L3_I (WTA)

Everything talks in spike events with conduction delays; weights/thresholds are voltage-based.
v1 (`fep_model.py`) is left intact for comparison.
"""

import numpy as np
from population import Population
from projection import Projection

# ---- scales (dimensionless but consistent; E_E>theta>E_L=0>E_I) ----
THETA_E = 20.0
THETA_I = 10.0
THETA_SUFF = 0.30          # total incoming g a single volley needs to cross theta (maturity / one-volley fire)
PRESENT_TICKS = 80
INPUT_PERIOD = 4           # active inputs spike every 4 ticks (a volley train)
# v1's proven demand-driven gate rule (timing-blind Hebbian; STDP deferred to a later pass).
# The cell that wins a presentation grows its gates to the inputs that were active and prunes
# the rest; a per-neuron budget caps the total. Local + unsupervised; provably tiles these patterns.
GATE_MAX = 0.6             # gate ceiling (also Projection.w_max)
W_GROW = 0.30              # demand-driven widening toward GATE_MAX (saturating)
PRUNE_RATE = 0.40          # heterosynaptic withering of un-recruited gates per win
SCALE_BUDGET = 1.2         # per-neuron incoming-weight budget (slow synaptic scaling)
# Presentation-scale homeostasis (ports v1's proven (N-1):1 fatigue into conductance form).
# A cell that fired recently starts the next presentation hyperpolarised (g_adapt baseline),
# so quieter cells claim other patterns. STDP then holds each niche. This is the rate-balancer.
# Symmetric intrinsic homeostasis (ports v1's BOTH halves): a winner tires (fatigue up,
# hyperpolarising), an idle cell sensitises (fatigue down -> can go negative -> depolarising boost).
# The up:down ratio sets the equilibrium win-rate; (N-1):1 targets 1/N (v1's result).
FAT_UP = 0.35              # fatigue added to a cell that fired this presentation
FAT_DOWN = 0.05            # fatigue removed from a cell that stayed silent (~ FAT_UP/(N-1))
FAT_CLIP = (-1.5, 5.0)     # bound the homeostatic conductance both ways


class InputLayer:
    """L1: spike sources. Active indices emit a spike every INPUT_PERIOD ticks (a volley train)."""
    def __init__(self, n, tau_trace=15.0):
        self.n = n
        self.ptype = "E"
        self.trace = np.zeros(n)
        self.tau_trace = tau_trace
        self.active = []

    def set_pattern(self, idxs):
        self.active = list(idxs)

    def integrate(self, dt=1.0):
        self.trace *= max(0.0, 1.0 - dt / self.tau_trace)

    def fire(self, t):
        return np.array(self.active, dtype=int) if (t % INPUT_PERIOD == 0) else np.array([], dtype=int)

    def bump_traces(self, fired):
        if len(fired):
            self.trace[fired] += 1.0


class Network:
    def __init__(self, n_l1=9, n_l2=8, n_l3=4, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # populations
        self.L1 = InputLayer(n_l1)
        # E cells: mild within-presentation AHP + a slow tau so the fatigue baseline persists
        self.L2E = Population(n_l2, "E", THETA_E, theta_jitter=2.0, adapt_gain=1.0, tau_adapt=300.0)
        self.L2I = Population(1, "I", THETA_I, t_ref=1, noise=0.0, adapt_gain=0.0)
        self.L3E = Population(n_l3, "E", THETA_E, theta_jitter=2.0, adapt_gain=1.0, tau_adapt=300.0)
        self.L3I = Population(1, "I", THETA_I, t_ref=1, noise=0.0, adapt_gain=0.0)
        # slow per-cell fatigue (presentation-scale homeostasis)
        self.l2_fatigue = np.zeros(n_l2)
        self.l3_fatigue = np.zeros(n_l3)

        # Feed-forward gates. plastic=False -> the per-tick STDP no-ops; learning is the v1
        # demand-driven rule applied per-presentation in _consolidate(). Born moderate so cells
        # can fire from the start (then the prune+budget specialise them).
        self.P_L1_L2 = Projection(self.L1, self.L2E, delay=1, plastic=False, w_max=GATE_MAX,
                                  w_init=np.random.uniform(0.05, 0.15, (n_l2, n_l1)))
        self.P_L2_L3 = Projection(self.L2E, self.L3E, delay=2, plastic=False, w_max=GATE_MAX,
                                  w_init=np.random.uniform(0.05, 0.15, (n_l3, n_l2)))

        # fixed WTA scaffolds: E->I (drive) and I->E (shunting inhibition), with delays
        self.P_L2E_L2I = Projection(self.L2E, self.L2I, delay=1, plastic=False,
                                    w_init=np.full((1, n_l2), 0.6))
        self.P_L2I_L2E = Projection(self.L2I, self.L2E, delay=1, plastic=False,
                                    w_init=np.full((n_l2, 1), 2.5))
        self.P_L3E_L3I = Projection(self.L3E, self.L3I, delay=1, plastic=False,
                                    w_init=np.full((1, n_l3), 0.6))
        self.P_L3I_L3E = Projection(self.L3I, self.L3E, delay=1, plastic=False,
                                    w_init=np.full((n_l3, 1), 2.5))

        self.projections = [self.P_L1_L2, self.P_L2E_L2I, self.P_L2I_L2E,
                            self.P_L2_L3, self.P_L3E_L3I, self.P_L3I_L3E]
        self.pops = [self.L2E, self.L2I, self.L3E, self.L3I]

    # ---------------------------------------------------------- one presentation
    def present(self, active_indices, learn=True, ticks=PRESENT_TICKS):
        active = list(active_indices)
        self.L1.set_pattern(active)
        for p in self.pops:
            p.reset()
        # presentation-scale homeostasis: recent winners start hyperpolarised (training only).
        # During evaluation we read the pure gate-driven preference (no fatigue).
        if learn:
            self.L2E.g_adapt[:] = self.l2_fatigue
            self.L3E.g_adapt[:] = self.l3_fatigue
        else:
            self.L2E.g_adapt[:] = 0.0
            self.L3E.g_adapt[:] = 0.0
        l2_count = np.zeros(self.L2E.n, dtype=int)
        l3_count = np.zeros(self.L3E.n, dtype=int)

        for t in range(ticks):
            # integrate membranes (decay g, update V via driving force, decay traces)
            self.L1.integrate()
            for p in self.pops:
                p.integrate()
            # deliver conductance arrivals scheduled earlier (respecting delays)
            for proj in self.projections:
                proj.deliver()
            # spike events
            f_in = self.L1.fire(t)
            f_l2e = self.L2E.fire()
            f_l2i = self.L2I.fire()
            f_l3e = self.L3E.fire()
            f_l3i = self.L3I.fire()
            l2_count[f_l2e] += 1
            l3_count[f_l3e] += 1
            # (no per-tick STDP; learning is the v1 demand-driven rule, applied per-presentation)
            # bump activity traces for cells that fired
            self.L1.bump_traces(f_in)
            self.L2E.bump_traces(f_l2e)
            self.L3E.bump_traces(f_l3e)
            # schedule spikes into downstream projections (delayed conductance)
            self.P_L1_L2.schedule(f_in)
            self.P_L2E_L2I.schedule(f_l2e)
            self.P_L2I_L2E.schedule(f_l2i)
            self.P_L2_L3.schedule(f_l2e)
            self.P_L3E_L3I.schedule(f_l3e)
            self.P_L3I_L3E.schedule(f_l3i)

        if learn:
            self._consolidate(active, l2_count, l3_count)
        return l2_count, l3_count

    def _gate_update(self, W, post, active_pre, budget):
        """v1 demand-driven rule on one winning post-neuron's incoming gates:
        grow gates from inputs that were active this presentation, prune the rest,
        then cap the total to the budget. Local to that neuron."""
        row = W[post]
        for i in range(row.shape[0]):
            if i in active_pre:
                row[i] += W_GROW * (1.0 - row[i] / GATE_MAX)
            else:
                row[i] *= (1.0 - PRUNE_RATE)
        np.clip(row, 0.0, GATE_MAX, out=row)
        tot = row.sum()
        if tot > budget:
            row *= budget / tot

    def _consolidate(self, active, l2_count, l3_count):
        # L2: the winner (most-active cell, a readout of real spikes) learns this pattern.
        # Tie-break randomly among the most-active cells (no index bias toward cell 0).
        if l2_count.max() > 0:
            top = np.flatnonzero(l2_count == l2_count.max())
            j = int(np.random.choice(top))
            self._gate_update(self.P_L1_L2.W, j, set(active), SCALE_BUDGET)
        # L3: its winner composes the L2 cells that co-fired this presentation.
        if l3_count.max() > 0:
            top3 = np.flatnonzero(l3_count == l3_count.max())
            k = int(np.random.choice(top3))
            active_l2 = set(np.where(l2_count > 0)[0].tolist())
            self._gate_update(self.P_L2_L3.W, k, active_l2, SCALE_BUDGET)
        # symmetric homeostasis: fired -> tire (up); silent -> sensitise (down, may go negative=boost)
        fired2 = l2_count > 0
        self.l2_fatigue[fired2] += FAT_UP
        self.l2_fatigue[~fired2] -= FAT_DOWN
        np.clip(self.l2_fatigue, FAT_CLIP[0], FAT_CLIP[1], out=self.l2_fatigue)
        fired3 = l3_count > 0
        self.l3_fatigue[fired3] += FAT_UP
        self.l3_fatigue[~fired3] -= FAT_DOWN
        np.clip(self.l3_fatigue, FAT_CLIP[0], FAT_CLIP[1], out=self.l3_fatigue)

    # ---------------------------------------------------------- readouts (measurement only)
    def l2_maturity(self):
        return self.P_L1_L2.maturity(THETA_SUFF)

    def l3_maturity(self):
        return self.P_L2_L3.maturity(THETA_SUFF)
