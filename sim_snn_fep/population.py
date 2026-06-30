"""
population.py -- a conductance-based LIF population (the E or I cells of one tile).

Spikes are EVENTS (threshold crossings). Internal state is voltage/conductance based.
Every update below is local to a single neuron; numpy vectorisation is implementation
detail only -- there is no global objective, gradient, or cross-neuron read.

Membrane (conductance-based LIF, Ohm's-law driving force; the leak is FIXED):

    V += dt * [ g_L*(E_L - V) + g_E*(E_E - V) + g_I*(E_I - V) ]

g_E rises when excitatory spikes arrive (a synaptic weight is added) and decays to 0;
g_I likewise for inhibition. Because inputs scale with the driving force (E_* - V),
excitation saturates near E_E (no runaway overshoot) and inhibition is *shunting*
(divisive gain control) -- the biophysics the v1 clamp had to fake.

The LIF -> pure-integrator transition is preserved as the g_E/g_L ratio: nascent gates
add little g_E vs the fixed leak (leaky, many volleys to fire); matured gates make one
volley's g_E cross threshold before it leaks (pure integrator).
"""

import numpy as np


class Population:
    def __init__(self, n, ptype, theta, *, g_L=0.1, E_L=0.0, E_E=70.0, E_I=-10.0,
                 v_reset=0.0, t_ref=3, tau_E=3.0, tau_I=5.0, tau_trace=15.0,
                 tau_act=400.0, theta_jitter=0.0, noise=0.3,
                 tau_adapt=120.0, adapt_gain=2.0, E_adapt=-10.0):
        assert ptype in ("E", "I")
        self.n = n
        self.ptype = ptype
        self.g_L, self.E_L, self.E_E, self.E_I = g_L, E_L, E_E, E_I
        self.v_reset, self.t_ref = v_reset, t_ref
        self.tau_E, self.tau_I, self.tau_trace, self.tau_act = tau_E, tau_I, tau_trace, tau_act
        self.noise = noise
        # spike-frequency adaptation (AHP): a slow self-inhibitory conductance.
        # A frequent firer tires (g_adapt rises on each spike, decays slowly) and yields
        # the floor to quieter cells -> the population tiles the input space. Local; it is
        # the rate-balancer, NOT a threshold change (threshold stays fixed). Carries across
        # presentations (slow state); not cleared by reset().
        self.tau_adapt, self.adapt_gain, self.E_adapt = tau_adapt, adapt_gain, E_adapt
        self.g_adapt = np.zeros(n)

        self.v = np.full(n, E_L)
        self.g_E = np.zeros(n)
        self.g_I = np.zeros(n)
        self.theta = np.full(n, float(theta))
        if theta_jitter:
            self.theta += np.random.uniform(-theta_jitter, theta_jitter, n)
        self.refrac = np.zeros(n, dtype=int)
        self.trace = np.zeros(n)   # STDP eligibility: pre-trace for outgoing, post-trace for incoming
        self.act = np.zeros(n)     # slow activity trace (homeostasis / dead-unit recruitment)

    def reset(self):
        """Clear transient membrane state between presentations. Traces persist (slow)."""
        self.v[:] = self.E_L
        self.g_E[:] = 0.0
        self.g_I[:] = 0.0
        self.refrac[:] = 0

    def integrate(self, dt=1.0):
        # conductances decay toward zero (channels close)
        self.g_E *= max(0.0, 1.0 - dt / self.tau_E)
        self.g_I *= max(0.0, 1.0 - dt / self.tau_I)
        self.g_adapt *= max(0.0, 1.0 - dt / self.tau_adapt)
        # conductance-based membrane update (local, per-neuron; driving force)
        self.v += dt * (self.g_L * (self.E_L - self.v)
                        + self.g_E * (self.E_E - self.v)
                        + self.g_I * (self.E_I - self.v)
                        + self.g_adapt * (self.E_adapt - self.v))
        if self.noise:
            self.v += np.random.uniform(-self.noise, self.noise, self.n)
        # eligibility + activity traces decay
        self.trace *= max(0.0, 1.0 - dt / self.tau_trace)
        self.act *= max(0.0, 1.0 - dt / self.tau_act)
        self.refrac[self.refrac > 0] -= 1

    def fire(self):
        """Threshold crossings become spike events; reset + refractory. Returns the
        indices that fired this tick (each is a genuine per-neuron event -- no winner
        is selected by comparison)."""
        fired = np.where((self.v >= self.theta) & (self.refrac == 0))[0]
        self.v[fired] = self.v_reset
        self.refrac[fired] = self.t_ref
        self.g_adapt[fired] += self.adapt_gain   # AHP build-up (self-inhibition)
        return fired

    def bump_traces(self, fired):
        if len(fired):
            self.trace[fired] += 1.0
            self.act[fired] += 1.0
