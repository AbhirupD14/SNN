"""Per-run driver and instrumentation (Experiment.md Sections 5, 7, 10, 11).

Runs ONE (mode, seed, task) through the one SimulationEngine, honouring the exact
protocol: one warm-up presentation (predictor off, excluded from metrics and tape),
dwell=20, input_period=1, blank=0. Records per-timestep event history sufficient to
recompute every Section 12 metric exactly, plus predictor-matrix snapshots at the
reporting boundaries. For time_shuffled_feedback it injects the permuted reference
tape via engine._feedback_override, one vector per measured timestep, per phase.
"""

import numpy as np

from backend.simulation import SimulationEngine, N_OUT, N_PIX
from .config import engine_overrides, MODES, DWELL
from .tape import ReferenceTape


class Recorder:
    """Accumulates per-timestep and per-presentation history; serializes to npz."""

    def __init__(self, mode, seed, task, engine):
        self.mode, self.seed, self.task = mode, seed, task
        self.ext_l1e_weight = float(engine.l1.excitatory_neurons[0].weights[1])  # +UNIT
        self.G = float(engine._l1i_G)
        self.gate = float(engine.params["predictive_output_gate_frac"]
                          * engine.params["threshold"])
        self.theta_l1i = float(engine.meta["L1I0"]["threshold"])
        self.fb_offset = int(engine._l1i_fb_offset)
        self.paired = bool(engine._l1i_paired)
        # per-timestep
        self.pres_idx, self.step_in_pres, self.input_arrives = [], [], []
        self.label, self.phase, self.l2i, self.delivered_any = [], [], [], []
        self.ext, self.l1e, self.l1i = [], [], []
        self.l2e, self.actual_l2e, self.delivered, self.removed = [], [], [], []
        # per-presentation
        self.pres_label, self.pres_phase = [], []
        self.pres_l2e_spikes, self.pres_owner = [], []
        # snapshots at reporting boundaries
        self.snap_pres_idx, self.snap_W, self.snap_trace = [], [], []
        self.initial_W = self._predictor_matrix(engine)
        self.initial_ff = self._l2_ff(engine)

    # ---- capture ------------------------------------------------------------
    def record_step(self, engine, pres_idx, label, phase, step, input_arrives):
        l1e = np.array([1.0 if engine.spiked[f"L1E{i}"] else 0.0 for i in range(N_PIX)])
        l1i = np.array([1.0 if engine.spiked[f"L1I{i}"] else 0.0 for i in range(N_PIX)])
        l2e = np.array([1.0 if engine.spiked[f"L2E{j}"] else 0.0 for j in range(N_OUT)])
        removed = np.zeros(N_PIX)
        for nid, ev in engine._inh_events:
            if nid.startswith("L1E"):
                removed[int(nid[3:])] += max(0.0, ev["v_pre"] - ev["v_post"])
        deliv = np.asarray(engine.delivered_feedback, dtype=float)
        self.pres_idx.append(pres_idx); self.step_in_pres.append(step)
        self.input_arrives.append(bool(input_arrives))
        self.label.append(label); self.phase.append(phase)
        self.l2i.append(1.0 if engine.spiked["L2I"] else 0.0)
        self.delivered_any.append(float(deliv.any()))
        self.ext.append(np.asarray(engine.input_vec, dtype=float).copy()
                        if input_arrives else np.zeros(N_PIX))
        self.l1e.append(l1e); self.l1i.append(l1i); self.l2e.append(l2e)
        self.actual_l2e.append(np.asarray(engine.actual_l2e, dtype=float).copy())
        self.delivered.append(deliv.copy()); self.removed.append(removed)

    def record_presentation(self, label, phase, l2e_spikes):
        l2e_spikes = np.asarray(l2e_spikes, dtype=float)
        total = float(l2e_spikes.sum())
        owner = int(l2e_spikes.argmax()) if total > 0 else -1   # -1 == None
        self.pres_label.append(label); self.pres_phase.append(phase)
        self.pres_l2e_spikes.append(l2e_spikes); self.pres_owner.append(owner)

    def snapshot(self, engine, pres_idx):
        self.snap_pres_idx.append(int(pres_idx))
        self.snap_W.append(self._predictor_matrix(engine))
        self.snap_trace.append(np.asarray(engine.l1i_trace, dtype=float).copy())

    # ---- helpers ------------------------------------------------------------
    def _predictor_matrix(self, engine):
        """Source-major W[j,i] = L2E_j -> L1I_i feedback weight (Section 12.5).
        Zeros when not paired (no feedback afferents to read)."""
        W = np.zeros((N_OUT, N_PIX))
        if engine._l1i_paired:
            off = engine._l1i_fb_offset
            for i in range(N_PIX):
                arr = engine.l1.inhibitory_neurons[i]._weights_array
                for j in range(N_OUT):
                    W[j, i] = arr[off + j]
        return W

    def _l2_ff(self, engine):
        return np.array([engine.l2.excitatory_neurons[j]._weights_array[0:N_PIX]
                         for j in range(N_OUT)])

    # ---- serialization ------------------------------------------------------
    def save(self, path, engine):
        self.final_W = self._predictor_matrix(engine)
        self.final_ff = self._l2_ff(engine)
        np.savez_compressed(
            path,
            mode=self.mode, seed=self.seed, task=self.task,
            dwell=DWELL, n_out=N_OUT, n_pix=N_PIX,
            ext_l1e_weight=self.ext_l1e_weight, G=self.G, gate=self.gate,
            theta_l1i=self.theta_l1i, fb_offset=self.fb_offset, paired=self.paired,
            pres_idx=np.array(self.pres_idx), step_in_pres=np.array(self.step_in_pres),
            input_arrives=np.array(self.input_arrives),
            label=np.array(self.label), phase=np.array(self.phase),
            l2i=np.array(self.l2i), delivered_any=np.array(self.delivered_any),
            ext=np.array(self.ext), l1e=np.array(self.l1e), l1i=np.array(self.l1i),
            l2e=np.array(self.l2e), actual_l2e=np.array(self.actual_l2e),
            delivered=np.array(self.delivered), removed=np.array(self.removed),
            pres_label=np.array(self.pres_label), pres_phase=np.array(self.pres_phase),
            pres_l2e_spikes=np.array(self.pres_l2e_spikes),
            pres_owner=np.array(self.pres_owner),
            snap_pres_idx=np.array(self.snap_pres_idx),
            snap_W=np.array(self.snap_W), snap_trace=np.array(self.snap_trace),
            initial_W=self.initial_W, final_W=self.final_W,
            initial_ff=self.initial_ff, final_ff=self.final_ff)


def _report_boundary(task, pres_idx, phase):
    """1-based reporting cadence per Section 10 (report AFTER this presentation)."""
    from .config import REPORT_FOURPATTERN, REPORT_ACQ, REPORT_REV
    n = pres_idx + 1
    if task == "fourpattern":
        return n % REPORT_FOURPATTERN == 0
    return n % (REPORT_REV if phase == "reversal" else REPORT_ACQ) == 0


def _present(engine, pres, rec, pres_idx, mode, replay_state):
    """Run one presentation (dwell steps). Records into `rec` when non-None. For a
    replay mode, sets _feedback_override to the next tape vector for this phase
    before each step and clears it after."""
    engine.set_input(pres["vector"])
    label, phase = pres["label"], pres["phase"]
    pres_l2e = np.zeros(N_OUT)
    for step in range(DWELL):
        input_arrives = (engine.timestep % engine.params["input_period"] == 0)
        if replay_state is not None:
            engine._feedback_override = replay_state.pop(phase)
        engine.step()
        engine._feedback_override = None
        if rec is not None:
            rec.record_step(engine, pres_idx, label, phase, step, input_arrives)
        pres_l2e += [1.0 if engine.spiked[f"L2E{j}"] else 0.0 for j in range(N_OUT)]
    if rec is not None:
        rec.record_presentation(label, phase, pres_l2e)


class _ReplayState:
    """Pops one shuffled tape vector per measured timestep, per phase."""

    def __init__(self, permuted):
        self._data = {ph: d["shuffled"] for ph, d in permuted.items()}
        self._pos = {ph: 0 for ph in self._data}

    def pop(self, phase):
        arr = self._data.get(phase)
        if arr is None or self._pos[phase] >= len(arr):
            return np.zeros(N_OUT)          # exhausted -> deliver nothing
        v = arr[self._pos[phase]]
        self._pos[phase] += 1
        return np.asarray(v, dtype=float)


def run_one(mode, seed, task, schedule, warmup, permuted_tape=None,
            build_tape=False, save_path=None):
    """Run one (mode, seed, task). Returns (Recorder, ReferenceTape|None, engine).

    - Warm-up (predictor off, no recording, no tape) then the measured schedule.
    - build_tape: also accumulate the live L2E vector per measured timestep (only
      meaningful for local_plus_feedback, to seed the shuffled control).
    - permuted_tape: for a replay mode, the {phase: dict(shuffled,...)} to inject.
    """
    engine = SimulationEngine(seed=seed, **engine_overrides(mode))
    rec = Recorder(mode, seed, task, engine)
    tape = ReferenceTape() if build_tape else None
    replay_state = _ReplayState(permuted_tape) if (MODES[mode]["replay"]
                                                   and permuted_tape) else None

    # Warm-up: duplicate of the first scheduled presentation. Predictor off; no
    # recording, no tape, no replay (time_shuffled gets its own live feedback here).
    saved_predictive = engine._predictive
    engine._predictive = False
    _present(engine, warmup, rec=None, pres_idx=-1, mode=mode, replay_state=None)
    engine._predictive = saved_predictive

    # Measured schedule.
    for pres_idx, pres in enumerate(schedule):
        _present(engine, pres, rec, pres_idx, mode, replay_state)
        if build_tape:
            # actual_l2e per measured timestep of this presentation, grouped by phase.
            base = len(rec.actual_l2e) - DWELL
            for k in range(DWELL):
                tape.record(pres["phase"], rec.actual_l2e[base + k])
        if _report_boundary(task, pres_idx, pres["phase"]):
            rec.snapshot(engine, pres_idx)
    # Final snapshot (in case the last presentation was not on a boundary).
    if not rec.snap_pres_idx or rec.snap_pres_idx[-1] != len(schedule) - 1:
        rec.snapshot(engine, len(schedule) - 1)

    if save_path is not None:
        rec.save(save_path, engine)
    return rec, tape, engine
