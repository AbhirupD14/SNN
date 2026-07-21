"""Does an explicit, uninhibitable RG source layer buy useful symmetry breaking?

Deterministic, headless comparison of four conditions on the `old` cortical topology:

  1. ``old``                  -- frozen direct external->L1E drive (the incumbent).
  2. ``rg_frozen``            -- explicit RG layer + the extra delay-1 hop, RG->L1E
                                weights FROZEN at the new ~61-unit initialization.
  3. ``rg_plastic``           -- the built-in `rg` behaviour (plastic RG->L1E).
  4. ``rg_plastic_equal_init``-- plastic RG->L1E, all nine weights identical at init.

The two controls are deliberately NOT blurred:

  * ``rg_frozen`` uses the new ~61-unit initialization, so it changes BOTH the topology/
    delay AND the L1 cadence relative to `old`. It is the "RG structure without RG
    learning" control, not a charge-matched one.
  * ``rg_frozen_matched`` (also run) freezes RG->L1E at the old SENSORY_WEIGHT
    (theta/3 = 333), reproducing `old`'s per-event L1 charge exactly. THAT is the
    control that isolates only topology + the extra delay, with L1 cadence held fixed.

Every condition shares the same L2E initialization at a given seed (the engine draws
competitor jitter before encoder jitter), so differences are not a reshuffled L2 seed.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.rg_timing_symmetry
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

from backend.simulation import (
    SimulationEngine, PATTERNS, N_PIX, N_OUT, FF_INIT_MEAN, SENSORY_WEIGHT,
)

# Training horizons long enough for the TWO-hop developmental path: an L1E starting at
# w~61 needs ~23 RG events for its first spike, and only then can L2 begin to learn.
DWELL = 1500                 # boundaries per pattern phase
SEEDS = (1, 2, 3, 4, 5)
SELECTIVITY_STRONG = 0.25    # fraction of the weight cap counted as a "strong" afferent

def _flatten_encoder_distance(engine):
    """Experimental override (NOT an engine feature, NOT used by the built-in preset):
    set every RG->L1E geometric learning-rate factor to 1.0.

    ``enc_init_jitter=False`` equalizes the initial WEIGHTS but leaves the per-synapse
    1/d^2 learning-rate multiplier asymmetric, because each RG_i -> L1E_i distance
    differs (the L1E end of the layout is jittered). This condition removes that last
    per-synapse asymmetry too, which is what attributes the L1 phase split to a cause."""
    for c in engine.encoders:
        c.acc_distance_factor[:] = 1.0


CONDITIONS = {
    'old':                    dict(topology='old'),
    'rg_frozen':              dict(topology='rg', enc_plasticity_enabled=False),
    'rg_frozen_matched':      dict(topology='rg', enc_plasticity_enabled=False,
                                   enc_w_init=SENSORY_WEIGHT, enc_init_jitter=False),
    'rg_plastic':             dict(topology='rg'),
    'rg_plastic_equal_init':  dict(topology='rg', enc_init_jitter=False),
    # Attribution control: equal init AND equal geometric learning rate.
    'rg_plastic_symmetric':   dict(topology='rg', enc_init_jitter=False,
                                   _post_build=_flatten_encoder_distance),
}

# Overlap schedules. Each pair shares the centre pixel (4) and nothing else.
SCHEDULES = {
    'row_sustained':  ['row 1'],
    'row_col_row':    ['row 1', 'col 1', 'row 1'],
    'diag_pair':      ['diag \\', 'diag /', 'diag \\'],
}


def active_pixels(name):
    return [i for i, v in enumerate(PATTERNS[name]) if v]


# ------------------------------------------------------------------ recording
def _record(engine, pattern, steps, log):
    """Advance ``steps`` boundaries on ``pattern``, appending raw per-boundary events."""
    engine.set_pattern(pattern)
    phase_start = engine.timestep
    for _ in range(steps):
        d = engine.step()
        t = d['timestep']
        spk = {n['id'] for n in d['neurons'] if n['spiked']}
        log['rg_events'] += len([s for s in spk if s.startswith('RG')])
        for s in spk:
            if s.startswith('L1E'):
                log['l1_spikes'].append((t, s))
                log['first'].setdefault(s, t)
            elif s.startswith('L2E'):
                log['l2_spikes'].append((t, s, pattern))
                log['first'].setdefault(s, t)
            elif s.startswith('RG'):
                log['first'].setdefault(s, t)
        # L2 learning events: which causal L1 afferents drove this update.
        if d['winner'] and any(s.startswith('L2E') and s == d['winner'] for s in spk):
            win = next(c for c in engine.l2e if c.id == d['winner'])
            part = engine._participation(win)
            log['l2_updates'].append(dict(t=t, winner=d['winner'], pattern=pattern,
                                          afferents=[win.ff_src[i]
                                                     for i, v in enumerate(part) if v]))
        # Timing of the inhibitory arrival onto L1 caused by an L2 winner.
        arr = [p['boundary'] for p in d['inhibitory_pulses']
               if p['target'].startswith('L1E')]
        if arr:
            log['l1_inh_arrivals'].append(t)
        log['boundaries'] += 1
        # RG->L1E weight trajectory, sampled sparsely to keep the artifact small.
        if engine.encoders and (t % 50 == 0):
            log['w_traj'].append([t] + [round(float(c.acc_weights[0]), 3)
                                        for c in engine.encoders])
    log['phases'].append(dict(pattern=pattern, start=phase_start + 1, end=engine.timestep))


def run_schedule(seed, schedule, cfg):
    cfg = dict(cfg)
    post_build = cfg.pop('_post_build', None)
    engine = SimulationEngine(seed=seed, **cfg)
    if post_build is not None:
        post_build(engine)
    log = dict(rg_events=0, boundaries=0, l1_spikes=[], l2_spikes=[], l2_updates=[],
               l1_inh_arrivals=[], first={}, w_traj=[], phases=[])
    for pattern in SCHEDULES[schedule]:
        _record(engine, pattern, DWELL, log)
    return engine, log


# ------------------------------------------------------------------ measures
def _cadence(times):
    """Mean inter-spike interval over a list of boundary indices."""
    if len(times) < 2:
        return None
    return float(np.mean(np.diff(sorted(times))))


def _phase_dispersion(l1_spikes, pixels, window):
    """Circular-ish dispersion of the three active L1 channels' spike phases.

    For each boundary in which ANY of the tracked channels fired, record which fired.
    Returns the mean pairwise |offset| between channels' spike trains -- 0 means the
    channels are locked in the same boundary, larger means they have desynchronized.
    """
    lo, hi = window
    trains = {p: sorted(t for t, s in l1_spikes if s == f'L1E{p}' and lo <= t <= hi)
              for p in pixels}
    offs = []
    ps = [p for p in pixels if len(trains[p]) > 1]
    for a in range(len(ps)):
        for b in range(a + 1, len(ps)):
            ta, tb = trains[ps[a]], trains[ps[b]]
            if not ta or not tb:
                continue
            # For each spike of a, distance to the nearest spike of b.
            tb_arr = np.array(tb)
            offs.append(float(np.mean([np.min(np.abs(tb_arr - x)) for x in ta])))
    return float(np.mean(offs)) if offs else None


def _sync_fraction(l1_spikes, pixels, window):
    """Fraction of active-L1 boundaries in which ALL tracked channels fired together."""
    lo, hi = window
    per_t = {}
    for t, s in l1_spikes:
        if lo <= t <= hi and s.startswith('L1E'):
            p = int(s[3:])
            if p in pixels:
                per_t.setdefault(t, set()).add(p)
    if not per_t:
        return None
    full = sum(1 for v in per_t.values() if len(v) == len(pixels))
    return full / len(per_t)


def _escape_latency(l1_spikes, inh_arrivals, window):
    """Boundaries from an L1 inhibitory arrival to the next L1E spike of any channel."""
    lo, hi = window
    l1_t = sorted({t for t, _ in l1_spikes if lo <= t <= hi})
    lat = []
    for a in inh_arrivals:
        if not (lo <= a <= hi):
            continue
        nxt = [t for t in l1_t if t > a]
        if nxt:
            lat.append(nxt[0] - a)
    return float(np.mean(lat)) if lat else None


def _winner_stats(l2_spikes, window):
    lo, hi = window
    seq = [w for t, w, _ in l2_spikes if lo <= t <= hi]
    if not seq:
        return dict(dominant=None, dominance=0.0, turnovers=0, mean_dwell=None,
                    n_wins=0, distinct=0)
    c = Counter(seq)
    dom, n = c.most_common(1)[0]
    turnovers = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    # Mean dwell = mean run length of consecutive wins by the same cell.
    runs, cur = [], 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return dict(dominant=dom, dominance=n / len(seq), turnovers=turnovers,
                mean_dwell=float(np.mean(runs)), n_wins=len(seq), distinct=len(c))


def _selectivity(engine):
    """Per-competitor L2 receptive-field structure over its nine L1 afferents."""
    cap = float(engine.params['e_weight_cap'])
    out = {}
    for c in engine.l2e:
        w = np.asarray(c.acc_weights, dtype=float)
        if w.size != N_PIX:
            continue
        norm = w / cap
        strong = [i for i, v in enumerate(norm) if v >= SELECTIVITY_STRONG]
        out[c.id] = dict(strong=strong, n_strong=len(strong),
                         top=float(norm.max()), mean=float(norm.mean()))
    return out


def analyze(engine, log, schedule):
    phases = log['phases']
    first_phase = (phases[0]['start'], phases[0]['end'])
    last_phase = (phases[-1]['start'], phases[-1]['end'])
    pix = active_pixels(SCHEDULES[schedule][0])
    early = (first_phase[0], first_phase[0] + DWELL // 4)
    mature = (last_phase[1] - DWELL // 4, last_phase[1])

    firsts = log['first']
    first_rg = min((v for k, v in firsts.items() if k.startswith('RG')), default=None)
    first_l1 = min((v for k, v in firsts.items() if k.startswith('L1E')), default=None)
    first_l2 = min((v for k, v in firsts.items() if k.startswith('L2E')), default=None)

    # L2 update composition: how many of the schedule's active features drove it.
    comp = Counter()
    for u in log['l2_updates']:
        comp[len([a for a in u['afferents'] if int(a[3:]) in pix])] += 1
    total_u = sum(comp.values()) or 1

    per_phase = []
    for ph in phases:
        w = (ph['start'], ph['end'])
        ws = _winner_stats(log['l2_spikes'], w)
        ws['pattern'] = ph['pattern']
        per_phase.append(ws)

    l1_cad_early = _cadence([t for t, s in log['l1_spikes']
                             if early[0] <= t <= early[1] and int(s[3:]) in pix])
    l1_cad_mature = _cadence([t for t, s in log['l1_spikes']
                              if mature[0] <= t <= mature[1] and int(s[3:]) in pix])

    sel = _selectivity(engine)
    dead = [cid for cid in sel if cid not in {w for _, w, _ in log['l2_spikes']}]
    w_final = ([round(float(c.acc_weights[0]), 3) for c in engine.encoders]
               if engine.encoders else None)
    # Saturation: first sampled boundary at which every active channel is >=95% of cap.
    sat = None
    if engine.encoders:
        cap = float(engine.params['e_weight_cap'])
        for row in log['w_traj']:
            if all(row[1 + p] >= 0.95 * cap for p in pix):
                sat = row[0]
                break

    return dict(
        boundaries=log['boundaries'], rg_events=log['rg_events'],
        l1_spike_count=len(log['l1_spikes']), l2_spike_count=len(log['l2_spikes']),
        first_rg=first_rg, first_l1=first_l1, first_l2=first_l2,
        l1_cadence_early=l1_cad_early, l1_cadence_mature=l1_cad_mature,
        phase_dispersion_early=_phase_dispersion(log['l1_spikes'], pix, early),
        phase_dispersion_mature=_phase_dispersion(log['l1_spikes'], pix, mature),
        sync_fraction_early=_sync_fraction(log['l1_spikes'], pix, early),
        sync_fraction_mature=_sync_fraction(log['l1_spikes'], pix, mature),
        l1_escape_latency=_escape_latency(log['l1_spikes'], log['l1_inh_arrivals'], mature),
        n_l1_inh_arrivals=len(log['l1_inh_arrivals']),
        winner_overall=_winner_stats(log['l2_spikes'], (1, log['boundaries'])),
        winner_per_phase=per_phase,
        l2_update_feature_counts={str(k): v / total_u for k, v in sorted(comp.items())},
        n_l2_updates=len(log['l2_updates']),
        selectivity=sel, dead_l2=dead, n_dead_l2=len(dead),
        mean_strong_per_l2=float(np.mean([v['n_strong'] for v in sel.values()])) if sel else None,
        rg_w_final=w_final, rg_w_saturation_t=sat,
        rg_w_spread_final=(round(max(w_final) - min(w_final), 3) if w_final else None),
    )


# ------------------------------------------------------------------ verdicts
def classify(summary):
    """Name the outcome honestly, using the distinctions the brief asks for."""
    if summary['mean_first_l2'] is None or (summary['mean_n_l2_updates'] or 0) < 5:
        return 'developmental_deadlock'
    verdicts = []
    if summary['mean_dominance'] >= 0.9 and summary['mean_distinct_winners'] <= 1.2:
        verdicts.append('winner_tyranny')
    if summary['mean_single_feature_frac'] >= 0.6:
        verdicts.append('single_feature_collapse')
    if (summary['mean_sync_fraction_mature'] is not None
            and summary['mean_sync_fraction_mature'] < 0.6
            and summary['mean_phase_dispersion_mature'] is not None
            and summary['mean_phase_dispersion_mature'] > 0.5):
        verdicts.append('temporal_phase_breaking')
    if (summary['mean_row_col_distinct'] is not None
            and summary['mean_row_col_distinct'] >= 0.5
            and summary['mean_recover_rate'] >= 0.5):
        verdicts.append('useful_assembly_symmetry_breaking')
    return '+'.join(verdicts) if verdicts else 'phase_locked_no_break'


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize(runs, schedule):
    n = len(runs)
    per_phase_dom = [[p['dominant'] for p in r['winner_per_phase']] for r in runs]
    # row->col->row: did the column phase recruit a DIFFERENT owner, and did the row
    # owner come back afterwards?
    row_col_distinct = recover = None
    if len(SCHEDULES[schedule]) == 3:
        distinct = [1.0 if (d[0] is not None and d[1] is not None and d[0] != d[1]) else 0.0
                    for d in per_phase_dom]
        rec = [1.0 if (d[0] is not None and d[0] == d[2]) else 0.0 for d in per_phase_dom]
        row_col_distinct, recover = float(np.mean(distinct)), float(np.mean(rec))

    def frac(k):
        return _mean([r['l2_update_feature_counts'].get(k, 0.0) for r in runs])

    s = dict(
        n_seeds=n, schedule=schedule,
        mean_boundaries=_mean([r['boundaries'] for r in runs]),
        mean_rg_events=_mean([r['rg_events'] for r in runs]),
        mean_l1_spikes=_mean([r['l1_spike_count'] for r in runs]),
        mean_l2_spikes=_mean([r['l2_spike_count'] for r in runs]),
        mean_first_rg=_mean([r['first_rg'] for r in runs]),
        mean_first_l1=_mean([r['first_l1'] for r in runs]),
        mean_first_l2=_mean([r['first_l2'] for r in runs]),
        mean_l1_cadence_early=_mean([r['l1_cadence_early'] for r in runs]),
        mean_l1_cadence_mature=_mean([r['l1_cadence_mature'] for r in runs]),
        mean_phase_dispersion_early=_mean([r['phase_dispersion_early'] for r in runs]),
        mean_phase_dispersion_mature=_mean([r['phase_dispersion_mature'] for r in runs]),
        mean_sync_fraction_early=_mean([r['sync_fraction_early'] for r in runs]),
        mean_sync_fraction_mature=_mean([r['sync_fraction_mature'] for r in runs]),
        mean_l1_escape_latency=_mean([r['l1_escape_latency'] for r in runs]),
        mean_dominance=_mean([r['winner_overall']['dominance'] for r in runs]),
        mean_dwell=_mean([r['winner_overall']['mean_dwell'] for r in runs]),
        mean_turnovers=_mean([r['winner_overall']['turnovers'] for r in runs]),
        mean_distinct_winners=_mean([r['winner_overall']['distinct'] for r in runs]),
        mean_n_l2_updates=_mean([r['n_l2_updates'] for r in runs]),
        mean_single_feature_frac=frac('1'),
        mean_two_feature_frac=frac('2'),
        mean_three_feature_frac=frac('3'),
        mean_dead_l2=_mean([r['n_dead_l2'] for r in runs]),
        mean_strong_per_l2=_mean([r['mean_strong_per_l2'] for r in runs]),
        mean_rg_w_spread=_mean([r['rg_w_spread_final'] for r in runs]),
        mean_rg_saturation_t=_mean([r['rg_w_saturation_t'] for r in runs]),
        mean_row_col_distinct=row_col_distinct,
        mean_recover_rate=recover if recover is not None else 0.0,
    )
    s['verdict'] = classify(s)
    return s


def main(seeds=SEEDS, out_path=None):
    report = {}
    for schedule in SCHEDULES:
        report[schedule] = {}
        print(f'\n=== schedule: {schedule} '
              f'({" -> ".join(SCHEDULES[schedule])}, {DWELL} boundaries/phase) ===')
        for cond, cfg in CONDITIONS.items():
            runs = []
            for seed in seeds:
                engine, log = run_schedule(seed, schedule, cfg)
                runs.append(analyze(engine, log, schedule))
            s = summarize(runs, schedule)
            report[schedule][cond] = dict(summary=s, runs=runs)
            print(f'[{cond:22s}] firstL1={_fmt(s["mean_first_l1"])} '
                  f'firstL2={_fmt(s["mean_first_l2"])} '
                  f'L1cad {_fmt(s["mean_l1_cadence_early"])}->{_fmt(s["mean_l1_cadence_mature"])} '
                  f'sync={_fmt(s["mean_sync_fraction_mature"])} '
                  f'disp={_fmt(s["mean_phase_dispersion_mature"])} '
                  f'dom={_fmt(s["mean_dominance"])} dwell={_fmt(s["mean_dwell"])} '
                  f'1feat={_fmt(s["mean_single_feature_frac"])} '
                  f'dead={_fmt(s["mean_dead_l2"])} '
                  f'-> {s["verdict"]}')
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), 'rg_timing_results.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {out_path}')
    return report


def _fmt(v):
    return '--' if v is None else (f'{v:.2f}' if isinstance(v, float) else str(v))


if __name__ == '__main__':
    main()
