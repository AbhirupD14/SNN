"""
Self-Organizing Spiking Neural Network core implementing the
"Selective Maturation Race".

The layer reaches a strict 1:1 localist code (one symbol -> one neuron) under a
*blocked* training curriculum, without catastrophic interference or symbol
collapse, using three purely-local, decentralized forces (no top-down daemon
touches the weights):

  A. Gated Homeostatic Bootstrap  (Abundance)  -- silent neurons self-amplify.
  B. Full Global Lateral Inhibition (Squeeze)  -- one spike blankets the pool.
  C. Selective Auto-Excitation Diagonal (Surge)-- the leader runs away & wins.

plus

  *  Selective Maturation Mask  -- the race winner freezes, retires, and drops
     its inhibitory output to 0, opening a "power vacuum" for the next block.
  *  Mass-Action Synaptic Bounding -- per-neuron L1 normalization (sum = W_total)
     so a surging winner strips its assembly down to N = 1.

Implementation is fully vectorized NumPy; every operation broadcasts across the
N-neuron pool. (The 1:1 port to torch is a literal symbol swap: np -> torch,
@ stays @, * stays *, np.where -> torch.where, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SNNConfig:
    # --- geometry ---------------------------------------------------------
    input_dim: int = 16          # flattened pixel grid
    n_neurons: int = 64          # L2 allocation pool (N)

    # --- mass-action synaptic bound (Section 3) ---------------------------
    w_total: float = 6.0         # sum_j W[j, i]  == W_total  (L1 mass)
    theta_rest: float = 3.0      # resting threshold = 0.5 * w_total
                                 # (only a high-fidelity match ignites)

    # --- lifecycle / maturation (Section 1) -------------------------------
    maturation_bar: float = 0.2  # nu_i crossing this within a block -> mature
    nu_beta: float = 0.10        # EMA rate for the per-block firing average

    # --- A. Gated homeostatic bootstrap (Section 2A) ----------------------
    nu_min: float = 0.01         # below this an un-matured neuron is "silent"
    nu_target: float = 0.01      # target rate used inside the boost gain
    alpha: float = 0.10          # homeostatic boost strength

    # --- B. Full global lateral inhibition (Section 2B) -------------------
    w_lateral: float = -5.0      # dense (sparsity = 1.0) off-diagonal squeeze
    # --- C. Selective auto-excitation diagonal (Section 2C) ---------------
    w_self: float = 2.0          # +W_ii surge, un-matured neurons only

    # --- membrane (LIF) ---------------------------------------------------
    v_leak: float = 0.90         # plastic-pool leak (matured = non-leaky, 1.0)
    v_floor: float = 0.0         # clamp; stops "trench-warfare" negative wells

    # --- plasticity (Hebbian / STDP race) ---------------------------------
    eta: float = 2.0             # aggressive Hebbian gain -> fast collapse to 1

    # --- substrate hygiene ------------------------------------------------
    sparse_substrate: bool = False  # if True, melt a block's spent runners-up
                                    # back into the fresh reservoir when the
                                    # race is won (keeps the pool sparse: only
                                    # retired neurons stay tuned).

    # --- runtime ----------------------------------------------------------
    max_block_steps: int = 400
    seed: int = 0


class SelfOrganizingSNNLayer:
    """A single self-stabilizing L2 layer of ``N`` competing spiking neurons."""

    def __init__(self, cfg: SNNConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        N, D = cfg.n_neurons, cfg.input_dim

        # Feedforward weights  W[input_j, neuron_i],  L1-normalized to W_total.
        self.W = np.empty((D, N))
        self._fresh_columns(np.arange(N))

        # Per-neuron dynamic state.
        self.v = np.zeros(N)                 # membrane potential
        self.nu = np.zeros(N)                # per-block EMA firing rate
        self.mature_mask = np.zeros(N, dtype=np.int8)
        self.spikes = np.zeros(N)            # last-step spike vector (for lateral)

        # Bookkeeping (read-only observation; never used to steer weights).
        self.tuned_pattern = np.full(N, -1, dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  Lateral connectivity (rebuilt each step from the maturation mask)   #
    # ------------------------------------------------------------------ #
    def _lateral_current(self, spikes: np.ndarray) -> np.ndarray:
        """I_lat[j] = sum_i L[j, i] * spikes_i, with matured neurons excised.

        L is 100% dense: off-diagonal = w_lateral (Squeeze), diagonal =
        +w_self for un-matured neurons only (Surge). A matured neuron's
        *output* is gated to 0 (the "power vacuum"); a matured neuron also
        receives no lateral current (a stable, isolated threshold detector).
        """
        cfg = self.cfg
        plastic = (self.mature_mask == 0).astype(np.float64)   # (N,)

        # Only un-matured presynaptic spikes exert any lateral force.
        eff = spikes * plastic                                  # (N,)

        # Global squeeze: every plastic spike blankets the whole pool.
        i_lat = cfg.w_lateral * (eff.sum() - eff)               # off-diagonal @ eff
        # Selective surge: a plastic neuron re-injects +w_self into itself.
        i_lat = i_lat + cfg.w_self * eff                        # diagonal @ eff

        # Matured neurons are mathematically excised from the landscape.
        i_lat = i_lat * plastic
        return i_lat

    # ------------------------------------------------------------------ #
    #  A. Gated Homeostatic Bootstrap                                      #
    # ------------------------------------------------------------------ #
    def _homeostatic_boost(self) -> None:
        """Silent un-matured neurons multiplicatively turn up their own gain.

        Scaling = 1 + alpha * (nu_target / max(0.001, nu)) * max(0, nu_min - nu)

        The scaffold melts the instant the neuron sparks (nu > nu_min): the
        ``max(0, nu_min - nu)`` gate -> 0 AND the next post-spike Hebbian step
        re-normalizes the inflated L1 mass back to W_total.
        """
        cfg = self.cfg
        nu = self.nu
        silent = (nu < cfg.nu_min) & (self.mature_mask == 0)

        scaling = 1.0 + (
            cfg.alpha
            * (cfg.nu_target / np.maximum(0.001, nu))
            * np.maximum(0.0, cfg.nu_min - nu)
        )
        scaling = np.where(silent, scaling, 1.0)               # (N,)
        # Broadcast the per-neuron gain across the input dimension.
        self.W *= scaling[None, :]

    # ------------------------------------------------------------------ #
    #  Mass-Action Hebbian collapse (Section 3)                            #
    # ------------------------------------------------------------------ #
    def _hebbian_update(self, x: np.ndarray, spikes: np.ndarray) -> None:
        """Un-matured spikers pull weight onto the active pattern, then the
        column is re-normalized to L1 = W_total.

        Because the surging winner spikes far more than the quenched runners-up,
        repeated normalization actively strips the shared mass off the trailers,
        collapsing the assembly to N = 1.
        """
        cfg = self.cfg
        learn = (spikes > 0) & (self.mature_mask == 0)         # (N,)
        if not learn.any():
            return

        # Additive Hebbian onto co-active inputs (outer product, broadcast).
        self.W += cfg.eta * np.outer(x, spikes * learn)        # (D, N)
        np.maximum(self.W, 0.0, out=self.W)                    # weights >= 0

        # Mass-action L1 bound -- only the columns that just learned.
        col = self.W.sum(axis=0, keepdims=True)
        col = np.where(col > 1e-9, col, 1.0)
        renorm = np.where(learn[None, :], cfg.w_total / col, 1.0)
        self.W *= renorm

    # ------------------------------------------------------------------ #
    #  Single simulation step                                              #
    # ------------------------------------------------------------------ #
    def step(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        plastic = (self.mature_mask == 0)

        # A. abundance: silent plastic neurons self-amplify before integrating.
        self._homeostatic_boost()

        # Feedforward drive  (W^T @ x), broadcast across the pool.
        i_ff = self.W.T @ x                                    # (N,)
        # B + C. lateral squeeze + selective surge from last step's spikes.
        i_lat = self._lateral_current(self.spikes)             # (N,)

        # Membrane integration. Matured neurons are non-leaky, isolated
        # threshold detectors (no leak, no lateral); plastic neurons leak.
        leak = np.where(plastic, cfg.v_leak, 1.0)
        self.v = leak * self.v + i_ff + i_lat
        np.maximum(self.v, cfg.v_floor, out=self.v)            # no negative wells

        # Spike + hard reset.
        spikes = (self.v >= cfg.theta_rest).astype(np.float64)
        self.v = np.where(spikes > 0, 0.0, self.v)

        # Per-block firing-rate EMA (plastic neurons only drive maturation).
        self.nu = (1.0 - cfg.nu_beta) * self.nu + cfg.nu_beta * spikes

        # Mass-action Hebbian collapse on the spikers.
        self._hebbian_update(x, spikes)

        self.spikes = spikes
        return spikes

    # ------------------------------------------------------------------ #
    #  Maturation check (the "race" verdict)                               #
    # ------------------------------------------------------------------ #
    def try_mature(self, pattern_id: int) -> int | None:
        """If any *plastic* neuron has crossed the stability bar this block,
        mature exactly one (the strongest), freeze it, and retire it.

        Returns the matured neuron index, else None.
        """
        cfg = self.cfg
        cand = (self.nu >= cfg.maturation_bar) & (self.mature_mask == 0)
        if not cand.any():
            return None

        # Deterministic single winner: strongest nu (ties -> lowest index).
        nu_masked = np.where(cand, self.nu, -np.inf)
        i = int(np.argmax(nu_masked))

        # Strict Retirement Isolation: freeze W, lr->0, drop lateral output->0.
        # (All three are realized structurally by mature_mask gating in step().)
        self.mature_mask[i] = 1
        self.tuned_pattern[i] = pattern_id
        self.v[i] = 0.0
        return i

    # ------------------------------------------------------------------ #
    #  Block boundary: reset transient plastic state                       #
    # ------------------------------------------------------------------ #
    def reset_transient(self) -> None:
        plastic = (self.mature_mask == 0)
        self.v = np.where(plastic, 0.0, self.v)
        self.nu = np.where(plastic, 0.0, self.nu)
        self.spikes = np.where(plastic, 0.0, self.spikes)

    def _fresh_columns(self, idx: np.ndarray) -> None:
        """(Re)initialize the given neuron columns to fresh, L1-normalized
        reservoir weights drawn from the layer's deterministic RNG."""
        D = self.cfg.input_dim
        W = self.rng.uniform(0.5, 1.0, size=(D, len(idx)))
        self.W[:, idx] = W * (self.cfg.w_total / W.sum(axis=0, keepdims=True))

    def reset_losers(self, winner: int) -> int:
        """Substrate hygiene (sparse mode): melt the runners-up of the block
        that just matured ``winner`` back into the fresh reservoir.

        A runner-up is any still-plastic neuron that fired this block (nu > 0)
        but did not win the race. Returning them to uniform weights stops the
        bootstrap-inflated, half-formed tuning from lingering in the pool, so
        only retired neurons remain selective.
        """
        losers = np.where((self.mature_mask == 0) & (self.nu > 0))[0]
        losers = losers[losers != winner]
        if losers.size:
            self._fresh_columns(losers)
            self.v[losers] = 0.0
            self.nu[losers] = 0.0
            self.spikes[losers] = 0.0
            self.tuned_pattern[losers] = -1
        return int(losers.size)

    # ------------------------------------------------------------------ #
    #  Read-only diagnostics                                               #
    # ------------------------------------------------------------------ #
    def response(self, x: np.ndarray) -> np.ndarray:
        """Feedforward drive of every neuron to pattern ``x`` (no dynamics)."""
        return self.W.T @ x
