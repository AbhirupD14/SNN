"""
Headless experiment runner for finite-budget SNN consolidation experiments.

Runs the SAME SimulationEngine as the interactive dashboard, but with no browser
or WebSocket -- suitable for overnight / cluster runs under SLURM, tmux, systemd,
or nohup. It reads a JSON/YAML config, sweeps seeds x dwell x schedule x ablation,
runs each combo for a FIXED presentation budget (not unlimited training), and
writes durable, structured artifacts a supervisor can monitor without screen
scraping:

    experiments/runs/<timestamp>/
        config.json          # the resolved config actually run
        description.md        # name / description / hypothesis
        status.json           # live progress (overwritten frequently)
        events.jsonl          # append-only log of milestones / failures
        metrics.jsonl         # append-only per-checkpoint metrics (one JSON/line)
        summary.csv           # one row per completed combo
        checkpoints/          # per-combo final weight snapshots (.npz)
        plots/                # current_*.png (live) + summary_*.png (final)

    experiments/runs/current -> latest run dir (symlink)

The process is designed to OUTLIVE the agent/browser: it only writes files, never
serves; failures are recorded per-combo and the sweep continues; SIGTERM/SIGINT
are caught so a preempted run is marked "interrupted" with artifacts flushed.

CLI:
    PYTHONPATH=. .venv/bin/python -m experiments.runner --config experiments/config_example.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")   # headless: no display needed
import matplotlib.pyplot as plt

from backend.simulation import SimulationEngine, PATTERNS, N_OUT, N_PIX

# Minimal signed-spike preset (mirrors the dashboard default in backend/api.py).
# Configs and ablations override any of these.
DEFAULT_MINIMAL = dict(
    signed_spike_learning=True, l2e_budget=False, confidence_consolidation=False,
    loser_depression=False, signed_depression=False, homeostasis=False, refractory=0,
    l2e_weight_cap_frac=1 / 3, pos_weight_floor=1, l2i_threshold_frac=1 / 3,
    l1i_threshold_frac=1 / 3, l2e_lr_frac=0.02, ei_sat_mult=4.0,
)

STABLE_WINDOW = 20      # trailing presentations used to judge ownership stability
STABLE_THRESH = 0.8     # modal owner share needed to call a pattern "stable"
CHECKPOINT_EVERY = 10   # presentations between metric checkpoints
PLOT_HISTORY = 400      # rolling steps kept for the live raster/charge plots


# --------------------------------------------------------------------- config
def load_config(path: str) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def combos(cfg: dict):
    """Enumerate every (schedule, dwell, ablation, seed) combination."""
    schedules = cfg.get("schedules") or cfg.get("schedule") or ["interleaved"]
    if isinstance(schedules, str):
        schedules = [schedules]
    dwells = cfg.get("dwell_steps") or [8]
    seeds = cfg.get("seeds") or [1]
    ablations = cfg.get("ablations") or {"baseline": {}}
    if isinstance(ablations, list):        # list of names -> empty overrides
        ablations = {a: {} for a in ablations}
    for sched in schedules:
        for dwell in dwells:
            for abl_name, abl_over in ablations.items():
                for seed in seeds:
                    yield dict(schedule=sched, dwell=int(dwell),
                               ablation=abl_name, ablation_overrides=abl_over or {},
                               seed=int(seed))


# ------------------------------------------------------------------ artifacts
class Run:
    def __init__(self, cfg: dict, root: Path):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / ts
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "plots").mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2))
        (self.dir / "description.md").write_text(
            f"# {cfg.get('name','experiment')}\n\n"
            f"{cfg.get('description','')}\n\n"
            f"**Hypothesis:** {cfg.get('hypothesis','')}\n")
        self._metrics = open(self.dir / "metrics.jsonl", "a", buffering=1)
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self.status = dict(run=ts, name=cfg.get("name", ""),
                           hypothesis=cfg.get("hypothesis", ""),
                           state="starting", started=time.time(), updated=time.time(),
                           total_combos=0, done_combos=0, current={}, message="")
        self._update_symlink(root)
        self.write_status()

    def _update_symlink(self, root: Path):
        link = root / "current"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.dir.name)
        except OSError:
            pass   # symlinks may be unavailable on some filesystems; non-fatal

    def write_status(self, **kw):
        self.status.update(kw)
        self.status["updated"] = time.time()
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=2))
        tmp.replace(self.dir / "status.json")   # atomic

    def event(self, kind: str, **data):
        self._events.write(json.dumps(dict(t=time.time(), kind=kind, **data)) + "\n")

    def metric(self, record: dict):
        self._metrics.write(json.dumps(record) + "\n")

    def close(self):
        self._metrics.close()
        self._events.close()


# ------------------------------------------------------------------- metrics
def _neuron_ids():
    l2e = [f"L2E{j}" for j in range(N_OUT)]
    l1i = [f"L1I{i}" for i in range(N_PIX)]
    return l2e, l1i


def weight_saturation(engine) -> float:
    """Fraction of positive L2E feedforward weights within 5% of the cap."""
    cap = engine.l2.excitatory_neurons[0].weight_cap
    hi = 0; tot = 0
    for j in range(N_OUT):
        w = engine.l2.excitatory_neurons[j]._weights_array[1:1 + N_PIX]
        tot += len(w); hi += int((w >= 0.95 * cap).sum())
    return hi / tot if tot else 0.0


# ------------------------------------------------------------------- one combo
def run_combo(run: Run, cfg: dict, combo: dict, idx: int, total: int):
    patterns = cfg.get("patterns") or list(PATTERNS.keys())
    budget = int(cfg.get("max_presentations_per_pattern", 200))
    dwell = combo["dwell"]
    seed = combo["seed"]
    sched = combo["schedule"]
    tag = f"{sched}/d{dwell}/{combo['ablation']}/seed{seed}"

    overrides = {**DEFAULT_MINIMAL, **cfg.get("base_params", {}), **combo["ablation_overrides"]}
    engine = SimulationEngine(seed=seed, **overrides)
    thr_l2 = engine.params["threshold_l2"]
    thr_l2i = engine.l2.inhibitory_neuron.threshold
    thr_l1i = engine.l1.inhibitory_neurons[0].threshold
    l2e_ids, l1i_ids = _neuron_ids()

    # per-pattern presentation bookkeeping
    votes = {p: deque(maxlen=STABLE_WINDOW) for p in patterns}
    pres_count = {p: 0 for p in patterns}
    time_to_stable = {p: None for p in patterns}
    # cumulative fire counts (dead detection) + inhibitory charge tracking
    fire = Counter()
    l2i_fires = 0; l1i_any = 0; steps = 0
    l2i_peak = 0.0; l1i_peak = 0.0
    margin_hist = []
    # rolling buffers for live plots
    spk_buf = deque(maxlen=PLOT_HISTORY)     # (t, set of fired neuron ids)
    chg_buf = deque(maxlen=PLOT_HISTORY)     # (t, dict id->activation)
    owner_hist = {p: [] for p in patterns}   # (pres_index, owner) per pattern

    def order():
        """Yield the next pattern to present under the schedule until budget hit."""
        rng = np.random.default_rng(seed + 777)
        if sched == "blocked":
            for p in patterns:
                for _ in range(budget):
                    yield p
        elif sched == "mixed":
            while any(pres_count[p] < budget for p in patterns):
                pool = [p for p in patterns if pres_count[p] < budget]
                yield pool[int(rng.integers(len(pool)))]
        else:  # interleaved (default)
            while any(pres_count[p] < budget for p in patterns):
                for p in patterns:
                    if pres_count[p] < budget:
                        yield p

    def present(name):
        nonlocal steps, l2i_fires, l1i_any, l2i_peak, l1i_peak
        engine.set_pattern(name)
        first = None
        for _ in range(dwell):
            engine.step(); steps += 1
            drive = engine.l2_drive
            # winner margin at the decisive step (top1 - top2)/thr
            vals = sorted(drive.values(), reverse=True)
            if len(vals) >= 2:
                margin_hist.append((vals[0] - vals[1]) / thr_l2)
            fired = set()
            for j, nid in enumerate(l2e_ids):
                if engine.spiked[nid]:
                    fire[nid] += 1; fired.add(nid)
                    if first is None:
                        first = j
            if engine.spiked["L2I"]:
                l2i_fires += 1; fired.add("L2I")
            for nid in l1i_ids:
                if engine.spiked[nid]:
                    fire[nid] += 1; l1i_any += 1; fired.add(nid)
            l2i_peak = max(l2i_peak, engine.l2.inhibitory_neuron.potential)
            l1i_peak = max(l1i_peak, max(n.potential for n in engine.l1.inhibitory_neurons))
            spk_buf.append((steps, fired))
            chg_buf.append((steps, {**drive,
                                    "L2I": engine.l2.inhibitory_neuron.potential}))
        return first

    def ownership():
        owner = {}; cons = {}
        for p in patterns:
            c = Counter(votes[p])
            if c:
                top, n = c.most_common(1)[0]
                owner[p] = top; cons[p] = n / sum(c.values())
            else:
                owner[p] = None; cons[p] = 0.0
        owners = [o for o in owner.values() if o is not None]
        distinct = len(set(owners))
        collisions = sum(n for n in Counter(owners).values() if n > 1)
        return owner, cons, distinct, collisions

    def checkpoint(done):
        owner, cons, distinct, collisions = ownership()
        dead_l2e = sum(1 for i in l2e_ids if fire[i] == 0)
        dead_l1i = sum(1 for i in l1i_ids if fire[i] == 0)
        rec = dict(combo=tag, seed=seed, schedule=sched, dwell=dwell,
                   ablation=combo["ablation"], presentations=dict(pres_count),
                   total_presentations=sum(pres_count.values()), steps=steps,
                   owner={p: owner[p] for p in patterns},
                   consistency={p: round(cons[p], 3) for p in patterns},
                   mean_consistency=round(float(np.mean(list(cons.values()))), 3),
                   distinct_owners=distinct, collisions=collisions,
                   time_to_stable={p: time_to_stable[p] for p in patterns},
                   dead_l2e=dead_l2e, dead_l2i=int(l2i_fires == 0), dead_l1i=dead_l1i,
                   l2i_fire_rate=round(l2i_fires / max(1, steps), 4),
                   l1i_fire_rate=round(l1i_any / max(1, steps * N_PIX), 4),
                   l2i_charge_ratio=round(l2i_peak / thr_l2i, 3),
                   l1i_charge_ratio=round(l1i_peak / thr_l1i, 3),
                   winner_margin=round(float(np.mean(margin_hist[-500:])) if margin_hist else 0.0, 4),
                   weight_saturation=round(weight_saturation(engine), 3),
                   done=done)
        run.metric(rec)
        run.write_status(current=dict(combo=tag, seed=seed, schedule=sched, dwell=dwell,
                                       ablation=combo["ablation"], steps=steps,
                                       total_presentations=rec["total_presentations"],
                                       budget_per_pattern=budget,
                                       mean_consistency=rec["mean_consistency"],
                                       distinct_owners=distinct),
                         done_combos=idx, total_combos=total,
                         message=f"combo {idx+1}/{total}: {tag}")
        _render_live_plots(run, tag, spk_buf, chg_buf, owner_hist, patterns,
                           l2e_ids, thr_l2, engine)
        return rec

    run.event("combo_start", combo=tag, budget=budget, patterns=patterns)
    since_ckpt = 0
    for name in order():
        w = present(name)
        pres_count[name] += 1
        if w is not None:
            votes[name].append(w)
        owner_c = Counter(votes[name])
        cur_owner = owner_c.most_common(1)[0][0] if owner_c else None
        owner_hist[name].append((pres_count[name], cur_owner))
        # finite-time stability check
        if (time_to_stable[name] is None and len(votes[name]) >= STABLE_WINDOW
                and owner_c.most_common(1)[0][1] / sum(owner_c.values()) >= STABLE_THRESH):
            time_to_stable[name] = pres_count[name]
            run.event("pattern_stable", combo=tag, pattern=name,
                      presentation=pres_count[name], owner=cur_owner)
        since_ckpt += 1
        if since_ckpt >= CHECKPOINT_EVERY:
            since_ckpt = 0
            checkpoint(done=False)

    rec = checkpoint(done=True)
    # final weight snapshot
    W = np.stack([engine.l2.excitatory_neurons[j]._weights_array for j in range(N_OUT)])
    np.savez_compressed(run.dir / "checkpoints" / f"{tag.replace('/','_')}.npz", weights=W)

    unstable = [p for p in patterns if time_to_stable[p] is None]
    failure = None if not unstable else f"not stable within {budget} presentations: {unstable}"
    run.event("combo_end", combo=tag, failure=failure,
              time_to_stable=time_to_stable, distinct=rec["distinct_owners"])
    return dict(schedule=sched, dwell=dwell, ablation=combo["ablation"], seed=seed,
                distinct_owners=rec["distinct_owners"], mean_consistency=rec["mean_consistency"],
                collisions=rec["collisions"], dead_l2e=rec["dead_l2e"],
                dead_l1i=rec["dead_l1i"], dead_l2i=rec["dead_l2i"],
                l2i_fire_rate=rec["l2i_fire_rate"], l1i_fire_rate=rec["l1i_fire_rate"],
                l2i_charge_ratio=rec["l2i_charge_ratio"], l1i_charge_ratio=rec["l1i_charge_ratio"],
                winner_margin=rec["winner_margin"], weight_saturation=rec["weight_saturation"],
                mean_time_to_stable=round(float(np.mean([time_to_stable[p] for p in patterns
                                                         if time_to_stable[p] is not None]))
                                          if any(time_to_stable[p] for p in patterns) else -1, 1),
                stable_patterns=sum(1 for p in patterns if time_to_stable[p] is not None),
                n_patterns=len(patterns), failure_reason=failure or "")


# --------------------------------------------------------------------- plots
def _render_live_plots(run, tag, spk_buf, chg_buf, owner_hist, patterns, l2e_ids, thr_l2, engine):
    """Overwrite the 'current' PNGs so the read-only viewer always shows the combo
    running right now. Reconstructable purely from these durable files."""
    try:
        pdir = run.dir / "plots"
        # 1. spikes-only raster
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ids = l2e_ids + ["L2I"] + [f"L1I{i}" for i in range(N_PIX)]
        yidx = {nid: k for k, nid in enumerate(ids)}
        xs, ys = [], []
        for (t, fired) in spk_buf:
            for nid in fired:
                if nid in yidx:
                    xs.append(t); ys.append(yidx[nid])
        ax.scatter(xs, ys, s=6, marker="|")
        ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids, fontsize=6)
        ax.set_xlabel("timestep"); ax.set_title(f"spike raster (spikes only) — {tag}")
        fig.tight_layout(); fig.savefig(pdir / "current_raster.png", dpi=90); plt.close(fig)

        # 2. charge over time (L2E + L2I, V/thr)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        if chg_buf:
            ts = [t for t, _ in chg_buf]
            for nid in l2e_ids:
                ax.plot(ts, [c.get(nid, 0) / thr_l2 for _, c in chg_buf], lw=0.8, label=nid)
            ax.axhline(1.0, ls="--", c="k", lw=0.8)
        ax.set_ylabel("charge V/θ"); ax.set_xlabel("timestep")
        ax.set_title(f"charge over time (L2E) — {tag}"); ax.legend(fontsize=5, ncol=4)
        fig.tight_layout(); fig.savefig(pdir / "current_charge.png", dpi=90); plt.close(fig)

        # 3. weights over time proxy: current L2E feedforward RF bars (snapshot)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        cap = engine.l2.excitatory_neurons[0].weight_cap
        M = np.stack([engine.l2.excitatory_neurons[j]._weights_array[1:1 + N_PIX] / cap
                      for j in range(N_OUT)])
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(N_OUT)); ax.set_yticklabels(l2e_ids, fontsize=6)
        ax.set_xlabel("pixel"); ax.set_title(f"L2E feedforward weights ÷ cap — {tag}")
        fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
        fig.savefig(pdir / "current_weights.png", dpi=90); plt.close(fig)

        # 4. ownership map over time
        fig, ax = plt.subplots(figsize=(9, 3.2))
        for p in patterns:
            if owner_hist[p]:
                xs = [i for i, _ in owner_hist[p]]
                ys = [(-1 if o is None else o) for _, o in owner_hist[p]]
                ax.plot(xs, ys, lw=1, label=p)
        ax.set_xlabel("presentation #"); ax.set_ylabel("owner L2E idx (-1=none)")
        ax.set_yticks(range(-1, N_OUT)); ax.set_title(f"ownership over time — {tag}")
        ax.legend(fontsize=6, ncol=4)
        fig.tight_layout(); fig.savefig(pdir / "current_ownership.png", dpi=90); plt.close(fig)
    except Exception as e:
        run.event("plot_error", error=str(e))


def _render_summary_plots(run, rows):
    try:
        pdir = run.dir / "plots"
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"{r['schedule'][:3]}/d{r['dwell']}/{r['ablation'][:6]}/s{r['seed']}" for r in rows]
        ax.bar(range(len(rows)), [r["mean_consistency"] for r in rows])
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.axhline(STABLE_THRESH, ls="--", c="r", lw=0.8)
        ax.set_ylabel("mean ownership consistency"); ax.set_title("per-combo consistency")
        fig.tight_layout(); fig.savefig(pdir / "summary_consistency.png", dpi=90); plt.close(fig)
    except Exception as e:
        run.event("plot_error", error=str(e))


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Headless SNN experiment runner")
    ap.add_argument("--config", required=True, help="JSON or YAML experiment config")
    ap.add_argument("--out", default=None, help="override output root (default from config or experiments/runs)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.out or cfg.get("output_directory")
                or (Path(__file__).parent / "runs"))
    root.mkdir(parents=True, exist_ok=True)
    run = Run(cfg, root)

    all_combos = list(combos(cfg))
    run.write_status(state="running", total_combos=len(all_combos))
    run.event("run_start", combos=len(all_combos), config=cfg.get("name", ""))

    # graceful shutdown: mark interrupted, flush, exit (artifacts preserved).
    def _sig(signum, frame):
        run.write_status(state="interrupted", message=f"signal {signum}")
        run.event("interrupted", signal=signum)
        run.close()
        sys.exit(1)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    rows = []
    summary_path = run.dir / "summary.csv"
    for idx, combo in enumerate(all_combos):
        try:
            row = run_combo(run, cfg, combo, idx, len(all_combos))
        except Exception:
            tb = traceback.format_exc()
            run.event("combo_failed", combo=str(combo), error=tb)
            row = dict(schedule=combo["schedule"], dwell=combo["dwell"],
                       ablation=combo["ablation"], seed=combo["seed"],
                       failure_reason="EXCEPTION: " + tb.strip().splitlines()[-1])
        rows.append(row)
        # rewrite summary.csv after every combo (durable partial results)
        cols = sorted({k for r in rows for k in r})
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    _render_summary_plots(run, [r for r in rows if "mean_consistency" in r])
    run.write_status(state="completed", done_combos=len(all_combos),
                     message="all combos finished")
    run.event("run_end", combos=len(all_combos))
    run.close()
    print(f"[runner] done. artifacts in {run.dir}")


if __name__ == "__main__":
    main()
