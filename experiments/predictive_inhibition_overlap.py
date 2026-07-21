"""Overlap symmetry-breaking validation for local predictive inhibition.

Runs the deterministic row -> column -> row schedule on the PI direct topology
(``enew_enabled=False``) across multiple seeds, with the required controls
(predictive conductance off, plasticity off, fast vs slow association). It measures,
without hardcoding overlap pixels into any learning rule:

* Phase A: which L2E competitor becomes the incumbent, and whether its paired PI
  cell develops strong output synapses primarily onto the row's active pixels.
* Phase B: whether the incumbent PI conductance suppresses the shared feature more
  than the column's novel features, whether a different L2E can win, and how much
  the incumbent PI's synapses onto the novel features grow (contamination).
* Phase C: whether the original detector still responds to the row after decay.
* Sparsity of the dense 72-candidate PI projection.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.predictive_inhibition_overlap
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_PIX, N_OUT

STRONG_FRAC = 0.25          # PI weight (fraction of cap) counted as a "strong" candidate


def active_pixels(name):
    return {i for i, v in enumerate(PATTERNS[name]) if v}


def _dominant(winners):
    c = Counter(w for w in winners if w is not None)
    return (c.most_common(1)[0][0] if c else None), c


def _phase(engine, pattern, steps):
    engine.set_pattern(pattern)
    winners = []
    for _ in range(steps):
        d = engine.step()
        winners.append(d['winner'])
    dom, counts = _dominant(winners)
    return dict(winners=winners, dominant=dom, counts=counts)


def run_schedule(seed, pattern_a='row 1', pattern_b='col 1',
                 steps=(3000, 3000, 2000), **cfg):
    """One row->column->row run. Returns a metrics dict (JSON-serializable)."""
    e = SimulationEngine(seed=seed, topology='pi', **cfg)
    a_pix, b_pix = active_pixels(pattern_a), active_pixels(pattern_b)
    shared = sorted(a_pix & b_pix)
    novel = sorted(b_pix - a_pix)

    A = _phase(e, pattern_a, steps[0])
    incumbent = A['dominant']
    inc_j = int(incumbent[3:]) if incumbent else 0
    pi_after_a = e.pi[inc_j].w.copy()

    # --- Phase B: overlapping pattern, tracking contamination over time ---
    e.set_pattern(pattern_b)
    b_winners = []
    contam_novel = []           # incumbent PI weight summed over novel pixels, per step
    g_shared_series, g_novel_series = [], []
    first_rival_step = None
    for t in range(steps[1]):
        d = e.step()
        b_winners.append(d['winner'])
        w = e.pi[inc_j].w
        contam_novel.append(float(w[novel].sum()))
        gi = {n['id']: n.get('g_inh', 0.0) for n in d['neurons']}
        g_shared_series.append(float(np.mean([gi[f'L1E{i}'] for i in shared])))
        g_novel_series.append(float(np.mean([gi[f'L1E{i}'] for i in novel])))
        if first_rival_step is None and d['winner'] not in (None, incumbent):
            first_rival_step = t
    B_dom, B_counts = _dominant(b_winners)
    pi_after_b = e.pi[inc_j].w.copy()

    C = _phase(e, pattern_a, steps[2])

    # --- sparsity of the whole dense candidate projection ---
    cap = float(e.params['pi_w_max'])
    W = np.array([pi.w for pi in e.pi])                 # 8 x 9
    strong = W > STRONG_FRAC * cap
    fanout = strong.sum(axis=1)

    return dict(
        seed=seed, config=cfg,
        pattern_a=pattern_a, pattern_b=pattern_b, shared=shared, novel=novel,
        incumbent=incumbent, incumbent_j=inc_j,
        phaseA_counts=dict(A['counts']),
        pi_after_a=[round(x, 4) for x in pi_after_a.tolist()],
        pi_after_a_on_active=round(float(pi_after_a[sorted(a_pix)].sum()), 4),
        pi_after_a_off_active=round(float(pi_after_a[sorted(set(range(N_PIX)) - a_pix)].sum()), 4),
        phaseB_dominant=B_dom, phaseB_counts=dict(B_counts),
        symmetry_break=bool(B_dom is not None and B_dom != incumbent),
        first_rival_step=first_rival_step,
        pi_after_b=[round(x, 4) for x in pi_after_b.tolist()],
        contam_novel_start=round(contam_novel[0], 4) if contam_novel else 0.0,
        contam_novel_end=round(contam_novel[-1], 4) if contam_novel else 0.0,
        g_shared_mean=round(float(np.mean(g_shared_series)), 4),
        g_novel_mean=round(float(np.mean(g_novel_series)), 4),
        shared_over_novel_g=round(float(np.mean(g_shared_series) / (np.mean(g_novel_series) + 1e-9)), 3),
        phaseC_dominant=C['dominant'], phaseC_counts=dict(C['counts']),
        incumbent_recovers=bool(C['dominant'] == incumbent),
        sparsity_strong_total=int(strong.sum()),
        sparsity_fanout=[int(x) for x in fanout.tolist()],
        sparsity_incumbent_fanout=int(fanout[inc_j]),
    )


def summarize(runs):
    def frac(key):
        return round(float(np.mean([bool(r[key]) for r in runs])), 3)
    return dict(
        n=len(runs),
        symmetry_break_rate=frac('symmetry_break'),
        recover_rate=frac('incumbent_recovers'),
        mean_shared_over_novel_g=round(float(np.mean([r['shared_over_novel_g'] for r in runs])), 3),
        mean_contam_end=round(float(np.mean([r['contam_novel_end'] for r in runs])), 4),
        mean_pi_active=round(float(np.mean([r['pi_after_a_on_active'] for r in runs])), 4),
        mean_pi_off=round(float(np.mean([r['pi_after_a_off_active'] for r in runs])), 4),
        mean_strong_synapses=round(float(np.mean([r['sparsity_strong_total'] for r in runs])), 2),
    )


CONDITIONS = {
    'full_default':      dict(),
    'conductance_off':   dict(pi_conductance_enabled=False),
    'plasticity_off':    dict(pi_plasticity_enabled=False),
    'fast_association':  dict(pi_eta=0.10),
    'slow_association':  dict(pi_eta=0.005),
}


def main(seeds=(1, 2, 3, 4, 5), out_path=None):
    report = {}
    for cond, cfg in CONDITIONS.items():
        runs = [run_schedule(seed, **cfg) for seed in seeds]
        report[cond] = dict(summary=summarize(runs), runs=runs)
        s = report[cond]['summary']
        print(f'[{cond:16s}] break={s["symmetry_break_rate"]:.2f} '
              f'recover={s["recover_rate"]:.2f} '
              f'g_shared/g_novel={s["mean_shared_over_novel_g"]:.2f} '
              f'PI_active={s["mean_pi_active"]:.2f} PI_off={s["mean_pi_off"]:.2f} '
              f'contam_end={s["mean_contam_end"]:.3f} strong={s["mean_strong_synapses"]:.1f}')
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), 'predictive_inhibition_results.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {out_path}')
    return report


if __name__ == '__main__':
    main()
