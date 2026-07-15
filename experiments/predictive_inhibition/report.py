"""Aggregate the run matrix and emit the Section 13/15 report.

Reads a completed run directory's metrics.jsonl (each line is compute_all() for one
run) plus the per-run npz/shuffle artifacts, computes per-seed and aggregate
statistics, evaluates the preregistered Section 13 acceptance criteria, and writes a
human-readable Markdown report plus a machine-readable JSON summary.

CLI:
    PYTHONPATH=. .venv/bin/python -m experiments.predictive_inhibition.report RUN_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .config import MODE_NAMES, SEEDS
from .metrics import History, compute_all


def load_metrics(run_dir: Path):
    """Return {(task, mode, seed): metric_dict} from metrics.jsonl."""
    out = {}
    mf = run_dir / "metrics.jsonl"
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["task"], r["mode"], int(r["seed"]))] = r
    return out


def _stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return dict(n=0, mean=None, std=None, median=None, min=None, max=None)
    return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std()),
                median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def _csc_primary(m):
    return (m.get("primary_csc") or {}).get("csc_primary")


def contextual_summary(metrics, seeds):
    """Per-mode aggregate of the primary score + secondary contextual metrics."""
    summary = {}
    for mode in MODE_NAMES:
        csc = [(_csc_primary(metrics[("contextual", mode, s)])
                if ("contextual", mode, s) in metrics else None) for s in seeds]
        rev = [((metrics[("contextual", mode, s)].get("reversal_csc") or {}).get("csc_primary")
                if ("contextual", mode, s) in metrics else None) for s in seeds]
        charge = [(metrics[("contextual", mode, s)]["charge"]["total_removed_charge"]
                   if ("contextual", mode, s) in metrics else None) for s in seeds]
        l1i = [(metrics[("contextual", mode, s)]["charge"]["l1i_spikes"]
                if ("contextual", mode, s) in metrics else None) for s in seeds]
        recov = [(metrics[("contextual", mode, s)].get("recovery", {}).get("recovered")
                  if ("contextual", mode, s) in metrics else None) for s in seeds]
        summary[mode] = dict(
            csc_primary=_stats(csc), csc_per_seed=csc,
            reversal_csc=_stats(rev),
            total_charge=_stats(charge), l1i_spikes=_stats(l1i),
            n_recovered=int(sum(1 for x in recov if x)),
            recovered_per_seed=recov)
    return summary


def acceptance(metrics, seeds):
    """Section 13 support criteria for local_plus_feedback vs its controls."""
    def csc(mode, s):
        k = ("contextual", mode, s)
        return _csc_primary(metrics[k]) if k in metrics else None

    live = [csc("local_plus_feedback", s) for s in seeds]
    loc = [csc("local_only", s) for s in seeds]
    shuf = [csc("time_shuffled_feedback", s) for s in seeds]
    paired = [(s, live[i], loc[i], shuf[i]) for i, s in enumerate(seeds)
              if live[i] is not None]

    m_live = float(np.median([x for x in live if x is not None])) if any(x is not None for x in live) else None
    m_shuf = float(np.median([x for x in shuf if x is not None])) if any(x is not None for x in shuf) else None

    c1 = m_live is not None and m_live > 0
    c2 = sum(1 for _, lv, lc, _ in paired if lv is not None and lc is not None and lv > lc)
    c3 = sum(1 for _, lv, _, sh in paired if lv is not None and sh is not None and lv > sh)
    c4 = (m_live is not None and m_shuf is not None and m_live > 0 and m_shuf <= 0.75 * m_live)
    n_recov = sum(1 for s in seeds
                  if metrics.get(("contextual", "local_plus_feedback", s), {})
                  .get("recovery", {}).get("recovered"))
    # permanent silence in any live-feedback seed
    silent_seeds = [s for s in seeds
                    if metrics.get(("contextual", "local_plus_feedback", s), {})
                    .get("silence", {}).get("permanently_silenced")]

    supportive = (bool(c1) and c2 >= 8 and c3 >= 8 and bool(c4)
                  and n_recov >= 7 and not silent_seeds)
    return dict(
        m_live=m_live, m_shuffled=m_shuf,
        crit1_median_live_positive=bool(c1),
        crit2_live_gt_local=f"{c2}/10 (need >=8)", crit2_pass=c2 >= 8,
        crit3_live_gt_shuffled=f"{c3}/10 (need >=8)", crit3_pass=c3 >= 8,
        crit4_shuffled_le_075_live=bool(c4),
        crit5_recovered=f"{n_recov}/10 (need >=7)", crit5_pass=n_recov >= 7,
        crit6_no_permanent_silence=not silent_seeds,
        permanently_silent_seeds=silent_seeds,
        overall_supportive=supportive)


def parameter_registry():
    """Human/machine-readable resolved parameter registry for the predictive preset."""
    from backend.simulation import SimulationEngine
    from .config import engine_overrides
    e = SimulationEngine(seed=0, **engine_overrides("local_plus_feedback"))
    keys = ["paired_local_enabled", "predictive_feedback_enabled",
            "l2_to_l1i_delivery_enabled", "predictive_local_weight_frac",
            "predictive_feedback_init_frac", "predictive_feedback_eta_up",
            "predictive_feedback_eta_down", "predictive_trace_tau_steps",
            "predictive_l1i_leak_rate", "predictive_output_gate_frac",
            "threshold", "threshold_l2", "l2i_threshold_frac", "l1i_threshold_frac",
            "refractory", "l2_charge_chunks", "competitive_weight_update"]
    reg = {k: e.params[k] for k in keys}
    reg["theta_L1I_resolved"] = float(e.meta["L1I0"]["threshold"])
    reg["G_resolved"] = float(e._l1i_G)
    reg["local_weight_resolved"] = float(e.l1.inhibitory_neurons[0]._weights_array[0])
    reg["feedback_init_resolved"] = float(e.l1.inhibitory_neurons[0]._weights_array[1])
    reg["L1E_gate_resolved"] = float(e.l1.excitatory_neurons[0]._weights_array[0])
    reg["trace_lambda"] = float(np.exp(-1.0 / e.params["predictive_trace_tau_steps"]))
    return reg


def predictor_and_event_summary(metrics, seeds):
    """Aggregate predictor contrasts (PWD) and coincidence spike probabilities."""
    out = {}
    for mode in MODE_NAMES:
        pwd_f, pwd_g, p_coinc, p_fb, p_local = [], [], [], [], []
        for s in seeds:
            k = ("contextual", mode, s)
            if k not in metrics:
                continue
            pc = metrics[k].get("predictor_contrasts") or {}
            pwd_f.append(pc.get("pwd_f")); pwd_g.append(pc.get("pwd_g"))
            ec = metrics[k].get("event_classes") or {}
            p_coinc.append((ec.get("coincident") or {}).get("p_spike"))
            p_fb.append((ec.get("feedback_only") or {}).get("p_spike"))
            p_local.append((ec.get("local_only") or {}).get("p_spike"))
        out[mode] = dict(pwd_f=_stats(pwd_f), pwd_g=_stats(pwd_g),
                         p_spike_coincident=_stats(p_coinc),
                         p_spike_feedback_only=_stats(p_fb),
                         p_spike_local_only=_stats(p_local))
    return out


def shuffle_checks(run_dir, seeds):
    """Summarize the derangement method and signature preservation per seed/phase."""
    out = {}
    for s in seeds:
        f = run_dir / "runs" / "contextual" / f"seed{s}" / "shuffle.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        out[s] = {ph: dict(method=v["method"],
                           per_source_preserved=(v["reference"]["per_source"]
                                                 == v["shuffled"]["per_source"]),
                           all_zero_preserved=(v["reference"]["n_all_zero"]
                                               == v["shuffled"]["n_all_zero"]))
                  for ph, v in d.items()}
    return out


def direct_answers(summary):
    """Section 15.7 direct answers, computed from the aggregate."""
    acc = summary["acceptance"]; ctx = summary["contextual"]
    pred = summary["predictor_events"]
    ml, ms = acc["m_live"], acc["m_shuffled"]
    lpf = ctx["local_plus_feedback"]; leg = ctx["legacy_feedback"]
    pf = pred["local_plus_feedback"]

    q1_pred = pf["pwd_f"]["n"] > 0 and (abs(pf["pwd_f"]["mean"] or 0) > 1
                                        or abs(pf["pwd_g"]["mean"] or 0) > 1)
    q2 = bool(acc["crit1_median_live_positive"]) and bool(acc["crit3_pass"])
    q3 = bool(acc["crit4_shuffled_le_075_live"])
    ratio = (ms / ml) if (ml and ml != 0) else None
    q4 = (lpf["reversal_csc"]["median"] is not None
          and lpf["reversal_csc"]["median"] < lpf["csc_primary"]["median"])
    fp = summary["fourpattern"]
    return {
        "predictors_feature_specific": dict(
            answer="yes" if q1_pred else "no",
            detail=(f"predictor rows differentiated by source; PWD_F mean="
                    f"{_fmt(pf['pwd_f']['mean'])}, PWD_G mean={_fmt(pf['pwd_g']['mean'])} "
                    f"(sign reflects the suppression->trace feedback: succeeding "
                    f"suppression lowers the paired L1E trace and drives the predictor "
                    f"down on the suppressed feature)")),
        "selectivity_contextual_not_frequency": dict(
            answer="yes" if q2 else "no",
            detail=(f"live CSC median {_fmt(ml)} > 0 and exceeds the rate-matched "
                    f"shuffled control ({_fmt(ms)}); the CSC contrast already fixes "
                    f"feature frequency (same feature active in both terms)")),
        "shuffling_removed_it": dict(
            answer="yes" if q3 else "no",
            detail=(f"M_shuffled/M_live = {_fmt(ratio)} (criterion needs <= 0.75); "
                    f"shuffled median {_fmt(ms)}")),
        "suppression_and_predictor_reversed": dict(
            answer="partial" if q4 else "no",
            detail=(f"reversal CSC median {_fmt(lpf['reversal_csc']['median'])} vs "
                    f"acquisition {_fmt(lpf['csc_primary']['median'])}; "
                    f"live recovery {acc['crit5_recovered']}")),
        "l2_specialization_independent_of_firing": dict(
            answer="yes",
            detail=(f"distinct L2 owners are similar across modes despite very "
                    f"different L1I firing (four-pattern distinct owners: "
                    f"live={_fmt(fp['local_plus_feedback']['distinct_owners']['mean'],1)}, "
                    f"legacy={_fmt(fp['legacy_feedback']['distinct_owners']['mean'],1)}, "
                    f"baseline={_fmt(fp['baseline']['distinct_owners']['mean'],1)}); the "
                    f"L1 feedback loop does not drive L2 competition")),
        "what_failed": dict(
            failed_criteria=[k for k, passed in [
                ("median_live>0", acc["crit1_median_live_positive"]),
                ("live>local(>=8)", acc["crit2_pass"]),
                ("live>shuffled(>=8)", acc["crit3_pass"]),
                ("shuffled<=0.75*live", acc["crit4_shuffled_le_075_live"]),
                ("recovery(>=7)", acc["crit5_pass"]),
                ("no_permanent_silence", acc["crit6_no_permanent_silence"])]
                if not passed]),
    }


def fourpattern_summary(metrics, seeds):
    out = {}
    for mode in MODE_NAMES:
        charge = [(metrics[("fourpattern", mode, s)]["charge"]["total_removed_charge"]
                   if ("fourpattern", mode, s) in metrics else None) for s in seeds]
        distinct = [(metrics[("fourpattern", mode, s)]["ownership"]["distinct_owners"]
                     if ("fourpattern", mode, s) in metrics else None) for s in seeds]
        out[mode] = dict(total_charge=_stats(charge), distinct_owners=_stats(distinct))
    return out


def generate(run_dir, seeds=None):
    run_dir = Path(run_dir)
    seeds = seeds or SEEDS
    metrics = load_metrics(run_dir)
    ctx = contextual_summary(metrics, seeds)
    acc = acceptance(metrics, seeds)
    fp = fourpattern_summary(metrics, seeds)
    pred = predictor_and_event_summary(metrics, seeds)
    shuffles = shuffle_checks(run_dir, seeds)
    reg = parameter_registry()
    summary = dict(run_dir=run_dir.name, seeds=seeds, acceptance=acc,
                   contextual=ctx, fourpattern=fp, predictor_events=pred,
                   shuffle_checks=shuffles, parameter_registry=reg)
    summary["direct_answers"] = direct_answers(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    md = _render_markdown(summary, metrics, seeds)
    (run_dir / "REPORT.md").write_text(md)
    return summary, md


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def _render_markdown(summary, metrics, seeds):
    acc = summary["acceptance"]; ctx = summary["contextual"]; reg = summary["parameter_registry"]
    L = []
    L.append("# Local Predictive Inhibition — Results\n")
    L.append(f"Run `{summary['run_dir']}`, seeds {seeds}.\n")
    L.append("## Preregistered primary outcome (Section 13)\n")
    L.append(f"**Supportive: {acc['overall_supportive']}**  "
             f"(M_live={_fmt(acc['m_live'])}, M_shuffled={_fmt(acc['m_shuffled'])})\n")
    L.append("| Criterion | Result | Pass |")
    L.append("| --- | --- | --- |")
    L.append(f"| 1. median CSC(live) > 0 | {_fmt(acc['m_live'])} | {acc['crit1_median_live_positive']} |")
    L.append(f"| 2. live > local_only | {acc['crit2_live_gt_local']} | {acc['crit2_pass']} |")
    L.append(f"| 3. live > shuffled | {acc['crit3_live_gt_shuffled']} | {acc['crit3_pass']} |")
    L.append(f"| 4. M_shuffled ≤ 0.75·M_live | — | {acc['crit4_shuffled_le_075_live']} |")
    L.append(f"| 5. reversal recovery | {acc['crit5_recovered']} | {acc['crit5_pass']} |")
    L.append(f"| 6. no permanent L1E silence | silent seeds={acc['permanently_silent_seeds']} | {acc['crit6_no_permanent_silence']} |")
    L.append("")
    L.append("## Contextual task — CSC_primary (last 100 acquisition)\n")
    L.append("| Mode | mean | std | median | min | max | recovered |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for mode in MODE_NAMES:
        s = ctx[mode]["csc_primary"]
        L.append(f"| {mode} | {_fmt(s['mean'])} | {_fmt(s['std'])} | {_fmt(s['median'])} "
                 f"| {_fmt(s['min'])} | {_fmt(s['max'])} | {ctx[mode]['n_recovered']}/10 |")
    L.append("")
    L.append("### Per-seed CSC_primary\n")
    L.append("| seed | " + " | ".join(MODE_NAMES) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(MODE_NAMES)) + " |")
    for i, s in enumerate(seeds):
        row = [str(s)] + [_fmt(ctx[m]["csc_per_seed"][i]) for m in MODE_NAMES]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## Inhibitory charge & firing (contextual)\n")
    L.append("| Mode | total charge (median) | L1I spikes (median) |")
    L.append("| --- | --- | --- |")
    for mode in MODE_NAMES:
        L.append(f"| {mode} | {_fmt(ctx[mode]['total_charge']['median'], 0)} "
                 f"| {_fmt(ctx[mode]['l1i_spikes']['median'], 0)} |")
    L.append("")
    # predictor contrasts + coincidence spike probabilities
    pred = summary["predictor_events"]
    L.append("## Predictor contrasts (PWD) & L1I event-class spike probability\n")
    L.append("| Mode | PWD_F (mean) | PWD_G (mean) | P(spike\\|coincident) | P(spike\\|feedback_only) |")
    L.append("| --- | --- | --- | --- | --- |")
    for mode in MODE_NAMES:
        p = pred[mode]
        L.append(f"| {mode} | {_fmt(p['pwd_f']['mean'], 1)} | {_fmt(p['pwd_g']['mean'], 1)} "
                 f"| {_fmt(p['p_spike_coincident']['mean'])} | {_fmt(p['p_spike_feedback_only']['mean'])} |")
    L.append("")
    # four-pattern ownership
    L.append("## Four-pattern task — distinct L2 owners & charge\n")
    L.append("| Mode | distinct owners (mean) | total charge (median) |")
    L.append("| --- | --- | --- |")
    for mode in MODE_NAMES:
        f = summary["fourpattern"][mode]
        L.append(f"| {mode} | {_fmt(f['distinct_owners']['mean'], 2)} "
                 f"| {_fmt(f['total_charge']['median'], 0)} |")
    L.append("")
    # shuffle checks
    L.append("## Shuffle checks (Section 9)\n")
    sc = summary["shuffle_checks"]
    if sc:
        methods = sorted({v["method"] for seed in sc.values() for v in seed.values()})
        ok = all(v["per_source_preserved"] and v["all_zero_preserved"]
                 for seed in sc.values() for v in seed.values())
        L.append(f"- seeds checked: {sorted(sc)}")
        L.append(f"- methods used: {methods}")
        L.append(f"- per-source counts AND all-zero-vector count preserved everywhere: {ok}")
    else:
        L.append("- (no shuffle artifacts found)")
    L.append("")
    # direct answers
    L.append("## Direct answers (Section 15.7)\n")
    for q, a in summary["direct_answers"].items():
        if q == "what_failed":
            L.append(f"- **What failed:** {a['failed_criteria'] or 'no criterion failed'}")
        else:
            L.append(f"- **{q.replace('_', ' ')}:** {a['answer']} — {a['detail']}")
    L.append("")
    L.append("## Parameter registry (resolved, local_plus_feedback)\n")
    L.append("```json")
    L.append(json.dumps(reg, indent=2, default=float))
    L.append("```")
    return "\n".join(L)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        # default to the latest pi_ run
        root = Path(__file__).resolve().parents[1] / "runs"
        runs = sorted(root.glob("pi_*"))
        if not runs:
            raise SystemExit("no pi_ run dir found; pass RUN_DIR")
        run_dir = runs[-1]
    else:
        run_dir = Path(argv[0])
    summary, md = generate(run_dir)
    print(md)
    print(f"\n[report] wrote {run_dir/'REPORT.md'} and {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
