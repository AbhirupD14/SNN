"""
projection.py -- a delayed, optionally-plastic bundle of synapses (pre -> post).

Dale's principle is structural: the sign is set by the PRESYNAPTIC population's type.
An E pre-population deposits into the post's excitatory conductance (g_E); an I
pre-population deposits into g_I. A neuron therefore only ever excites or only ever
inhibits -- never both.

Communication is by spike events with a conduction DELAY (a ring buffer): a presynaptic
spike at tick t adds its weight to the postsynaptic conductance at tick t+delay.

Learning is pair-based STDP, entirely local: synapse (j,i) changes using only its own
presynaptic eligibility trace x_i and postsynaptic trace y_j -- no objective, no gradient.
  * LTP (pre-before-post): on a post spike, W[j,i] += a_plus * x_i   (soft-bounded)
  * LTD (post-before-pre): on a pre  spike, W[j,i] -= a_minus * y_j  (soft-bounded)
"""

import numpy as np


class Projection:
    def __init__(self, pre, post, *, w_init, delay=1, plastic=False, w_max=1.0,
                 a_plus=0.02, a_minus=0.022):
        self.pre, self.post = pre, post
        self.W = np.array(w_init, dtype=float)          # shape (post.n, pre.n)
        assert self.W.shape == (post.n, pre.n)
        self.delay = max(1, int(delay))
        self.plastic = plastic
        self.w_max = w_max
        self.a_plus, self.a_minus = a_plus, a_minus
        self.target = "g_E" if pre.ptype == "E" else "g_I"   # Dale
        self.buf = np.zeros((self.delay + 1, post.n))         # ring buffer of pending conductance
        self.ptr = 0

    # ---- transmission (events with delay) ----
    def schedule(self, pre_fired):
        """A presynaptic spike volley deposits weight into the post conductance `delay` ticks later."""
        if len(pre_fired):
            slot = (self.ptr + self.delay) % (self.delay + 1)
            self.buf[slot] += self.W[:, pre_fired].sum(axis=1)

    def deliver(self):
        """Hand the conductance scheduled for *now* to the postsynaptic population, then advance."""
        getattr(self.post, self.target)[:] += self.buf[self.ptr]
        self.buf[self.ptr] = 0.0
        self.ptr = (self.ptr + 1) % (self.delay + 1)

    # ---- plasticity (local STDP) ----
    def stdp(self, pre_fired, post_fired):
        if not self.plastic:
            return
        if len(post_fired):   # LTP: recent pre (x) preceded this post spike
            self.W[post_fired, :] += self.a_plus * self.pre.trace[None, :] * \
                (1.0 - self.W[post_fired, :] / self.w_max)
        if len(pre_fired):    # LTD: recent post (y) preceded this pre spike
            self.W[:, pre_fired] -= self.a_minus * self.post.trace[:, None] * \
                (self.W[:, pre_fired] / self.w_max)
        np.clip(self.W, 0.0, self.w_max, out=self.W)

    def scale(self, budget):
        """Slow synaptic scaling: cap each post-neuron's incoming total (local homeostasis)."""
        if not self.plastic:
            return
        tot = self.W.sum(axis=1)
        over = tot > budget
        if over.any():
            self.W[over] *= (budget / tot[over])[:, None]

    # ---- observable maturity (not fed into dynamics) ----
    def maturity(self, theta_sufficiency):
        """How close each post-neuron's incoming gates are to firing on one volley."""
        return np.minimum(self.W.sum(axis=1) / theta_sufficiency, 1.0)
