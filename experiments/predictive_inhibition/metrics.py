"""Section 12 metrics, recomputed purely from recorded event history.

Everything here reads a `History` (the driver's saved npz or an in-memory Recorder
view) and returns plain floats/dicts, so `compute_all` reproduces every reported
number from the durable artifact alone (Experiment.md Section 8 requirement).
"""

import numpy as np

from .config import (F_INDEX, G_INDEX, X_TRIALS, Y_TRIALS, REPORT_ACQ, REPORT_REV)

N_OUT = 8
N_PIX = 9


class History:
    """Array view over a recorded run (npz file or a Recorder)."""

    _TS = ("pres_idx", "step_in_pres", "input_arrives", "label", "phase", "l2i",
           "delivered_any", "ext", "l1e", "l1i", "l2e", "actual_l2e", "delivered",
           "removed")
    _PRES = ("pres_label", "pres_phase", "pres_l2e_spikes", "pres_owner")
    _SCALARS = ("mode", "seed", "task", "ext_l1e_weight", "G", "gate", "theta_l1i")
    _MATS = ("initial_W", "final_W", "initial_ff", "final_ff",
             "snap_pres_idx", "snap_W", "snap_trace")

    def __init__(self, src):
        for k in self._SCALARS:
            v = np.asarray(src[k])
            setattr(self, k, v.item() if v.ndim == 0 else v)
        self.mode = str(self.mode); self.task = str(self.task); self.seed = int(self.seed)
        self.ext_l1e_weight = float(self.ext_l1e_weight)
        self.G = float(self.G); self.gate = float(self.gate)
        for k in self._TS + self._PRES + self._MATS:
            setattr(self, k, np.asarray(src[k]))
        self.label = self.label.astype(str); self.phase = self.phase.astype(str)
        self.pres_label = self.pres_label.astype(str)
        self.pres_phase = self.pres_phase.astype(str)

    @classmethod
    def load(cls, path):
        return cls(np.load(path, allow_pickle=False))


# ---- Section 12.2 suppression fraction --------------------------------------
def suppression_fraction(H, feature, pres_mask):
    """SF(feature, C) = Q/max(1e-12, O) over presentations C (a boolean mask over
    presentations). Excludes the first outer timestep of each presentation; counts
    only timesteps where the feature is externally active with a delivered pulse."""
    ts_in_C = pres_mask[H.pres_idx]
    active = ts_in_C & (H.step_in_pres >= 1) & (H.ext[:, feature] > 0.5)
    O = float(active.sum()) * H.ext_l1e_weight
    Q = float(H.removed[active, feature].sum())
    n = int(active.sum())
    sf = Q / max(1e-12, O)
    suppressed_rate = float((H.removed[active, feature] > 0).mean()) if n else 0.0
    spike_rate = float((H.l1e[active, feature] > 0.5).mean()) if n else 0.0
    return dict(sf=sf, O=O, Q=Q, n=n,
                suppressed_active_step_rate=suppressed_rate,
                l1e_spike_rate=spike_rate)


# ---- Section 12.3 contextual suppression contrasts --------------------------
def csc(H, pres_mask):
    """(CSC_F, CSC_G, CSC_primary) over the presentation window `pres_mask`."""
    def SF(feature, label):
        return suppression_fraction(H, feature, pres_mask & (H.pres_label == label))["sf"]
    csc_f = SF(F_INDEX, "X_with_F") - SF(F_INDEX, "Y_with_F")
    csc_g = SF(G_INDEX, "Y_without_F") - SF(G_INDEX, "X_without_F")
    return csc_f, csc_g, 0.5 * (csc_f + csc_g)


def _phase_mask(H):
    return H.pres_phase == "acquisition", H.pres_phase == "reversal"


def _last_n(mask, n):
    idx = np.nonzero(mask)[0]
    sel = idx[-n:] if len(idx) >= n else idx
    out = np.zeros(len(mask), bool)
    out[sel] = True
    return out


def _windows(mask, size):
    """Yield (window_index, boolean presentation-mask) for consecutive `size`-blocks
    of the presentations selected by `mask`, in order."""
    idx = np.nonzero(mask)[0]
    for w in range(len(idx) // size):
        m = np.zeros(len(mask), bool)
        m[idx[w * size:(w + 1) * size]] = True
        yield w, m


def csc_trajectory(H):
    """Windowed CSC_primary over acquisition (every REPORT_ACQ) and reversal (every
    REPORT_REV) presentations, in order."""
    acq, rev = _phase_mask(H)
    acq_traj = [(w, *csc(H, m)) for w, m in _windows(acq, REPORT_ACQ)]
    rev_traj = [(w, *csc(H, m)) for w, m in _windows(rev, REPORT_REV)]
    return acq_traj, rev_traj


# ---- Section 12.5 predictor matrix + source context -------------------------
def source_context(H):
    """Assign each L2E source X/Y/unassigned from its per-presentation spike rate on
    X vs Y trials over the FINAL 100 acquisition presentations (frozen for reversal)."""
    acq, _ = _phase_mask(H)
    acq_idx = np.nonzero(acq)[0][-100:]
    labels = H.pres_label[acq_idx]
    spikes = H.pres_l2e_spikes[acq_idx]
    xm = np.isin(labels, X_TRIALS); ym = np.isin(labels, Y_TRIALS)
    ctx = []
    for j in range(N_OUT):
        rx = spikes[xm, j].mean() if xm.any() else 0.0
        ry = spikes[ym, j].mean() if ym.any() else 0.0
        ctx.append("X" if rx > ry else ("Y" if ry > rx else "unassigned"))
    return ctx


def predictor_contrasts(W, ctx):
    """PWD_F, PWD_G from a predictor matrix W[j,i] and frozen source groups.
    Returns (PWD_F, PWD_G); None for a contrast whose group is empty."""
    Xj = [j for j in range(N_OUT) if ctx[j] == "X"]
    Yj = [j for j in range(N_OUT) if ctx[j] == "Y"]
    if not Xj or not Yj:
        return None, None
    pwd_f = float(W[Xj, F_INDEX].mean() - W[Yj, F_INDEX].mean())
    pwd_g = float(W[Yj, G_INDEX].mean() - W[Xj, G_INDEX].mean())
    return pwd_f, pwd_g


# ---- Section 12.4 L1I event classes -----------------------------------------
def event_class_spike_probs(H):
    local = H.l1e > 0.5
    fb = (H.delivered_any > 0.5)[:, None]
    spike = H.l1i > 0.5
    masks = dict(local_only=local & ~fb, feedback_only=~local & fb,
                 coincident=local & fb, none=~local & ~fb)
    return {name: dict(p_spike=(float(spike[m].mean()) if m.any() else None),
                       n=int(m.sum())) for name, m in masks.items()}


# ---- Section 12.6 ownership -------------------------------------------------
def ownership(H):
    owners = H.pres_owner
    distinct = len(set(int(o) for o in owners if o >= 0))
    dead_l2e = int((H.l2e.sum(axis=0) == 0).sum())
    # A tie is a presentation where >=2 L2E share the (positive) max spike count;
    # argmax resolved it to the lowest index (Section 12.6).
    spikes = H.pres_l2e_spikes
    maxv = spikes.max(axis=1)
    n_at_max = (spikes == maxv[:, None]).sum(axis=1)
    ties = int(((n_at_max > 1) & (maxv > 0)).sum())
    return dict(distinct_owners=distinct, dead_l2e=dead_l2e, tie_presentations=ties)


# ---- Section 12.7 recovery + silence ----------------------------------------
def recovery(H):
    """First reversal window whose CSC_primary is negative AND stays negative for 3
    consecutive windows; recovery time = presentations elapsed to that first window;
    else 'not_recovered' at the reversal budget."""
    _, rev_traj = csc_trajectory(H)
    prim = [t[3] for t in rev_traj]
    for w in range(len(prim) - 2):
        if prim[w] < 0 and prim[w + 1] < 0 and prim[w + 2] < 0:
            return dict(recovered=True, recovery_presentations=(w + 1) * REPORT_REV)
    return dict(recovered=False, recovery_presentations=len(prim) * REPORT_REV)


def silence(H):
    """Per reporting window (both phases), an L1E with >=1 active external
    opportunity but zero spikes is temporarily silent; >=5 consecutive such windows
    is permanent silence (within this budget). Returns per-feature counts/identities."""
    acq, rev = _phase_mask(H)
    all_windows = list(_windows(acq, REPORT_ACQ)) + list(_windows(rev, REPORT_REV))
    consec = np.zeros(N_PIX, int); max_consec = np.zeros(N_PIX, int)
    temp_silent = np.zeros(N_PIX, int)
    for _, m in all_windows:
        ts = m[H.pres_idx]
        for i in range(N_PIX):
            opp = ts & (H.ext[:, i] > 0.5)
            if opp.any():
                if (H.l1e[opp, i] > 0.5).sum() == 0:
                    temp_silent[i] += 1; consec[i] += 1
                    max_consec[i] = max(max_consec[i], consec[i])
                else:
                    consec[i] = 0
    permanent = [int(i) for i in range(N_PIX) if max_consec[i] >= 5]
    return dict(temporarily_silent_windows=temp_silent.tolist(),
                max_consecutive_silent=max_consec.tolist(),
                permanently_silenced=permanent)


# ---- Section 12.8 charge ----------------------------------------------------
def charge_summary(H):
    return dict(total_removed_charge=float(H.removed.sum()),
                l1i_spikes=int((H.l1i > 0.5).sum()),
                l1e_spikes=int((H.l1e > 0.5).sum()),
                l2e_spikes=int((H.l2e > 0.5).sum()),
                l2i_spikes=int((H.l2i > 0.5).sum()),
                charge_per_presentation=float(H.removed.sum()
                                              / max(1, len(H.pres_owner))))


# ---- top-level --------------------------------------------------------------
def compute_all(H):
    """Full metric bundle for one run, recomputed from recorded history."""
    out = dict(mode=H.mode, seed=H.seed, task=H.task)
    if H.task == "contextual":
        acq, rev = _phase_mask(H)
        last100_acq = _last_n(acq, 100)
        last100_rev = _last_n(rev, 100)
        cf, cg, cp = csc(H, last100_acq)
        out["primary_csc"] = dict(csc_f=cf, csc_g=cg, csc_primary=cp)  # Section 13 score
        rf, rg, rp = csc(H, last100_rev)
        out["reversal_csc"] = dict(csc_f=rf, csc_g=rg, csc_primary=rp)
        acq_traj, rev_traj = csc_trajectory(H)
        out["csc_trajectory"] = dict(acquisition=acq_traj, reversal=rev_traj)
        ctx = source_context(H)
        out["source_context"] = ctx
        pwd_f, pwd_g = predictor_contrasts(H.final_W, ctx)
        out["predictor_contrasts"] = dict(pwd_f=pwd_f, pwd_g=pwd_g)
        out["recovery"] = recovery(H)
        out["silence"] = silence(H)
    else:  # fourpattern
        # SF(i, pattern) for active features + the pattern-by-feature matrix.
        from backend.simulation import PATTERNS
        allp = np.ones(len(H.pres_owner), bool)
        sf_matrix = {}
        for name in PATTERNS:
            pm = allp & (H.pres_label == name)
            sf_matrix[name] = {i: suppression_fraction(H, i, pm)["sf"]
                               for i in range(N_PIX) if PATTERNS[name][i] > 0.5}
        out["fourpattern_sf"] = sf_matrix
    out["event_classes"] = event_class_spike_probs(H)
    out["ownership"] = ownership(H)
    out["charge"] = charge_summary(H)
    return out
