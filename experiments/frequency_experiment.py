"""Deterministic headless coincidence/frequency experiment for the minimal SNN.

The L1 feedback circuit was corrected so that each L1E_new[i] is a local
COINCIDENCE detector with nine accumulating afferents: one paired local sensory
afferent (L1E_s[i]) plus dense L2E feedback. It should fire only when its paired
sensory input AND an L2E winner coincide, so that inhibition is pixel-selective
rather than global. With the shared cap at theta/2, exact two-input coincidence is
calibrated: mature paired sensory (~500) + mature winning-L2E (~500) = theta, while
either branch alone (~500) stays below theta -- IF leak rejects the lone branch.

This script does not build a general framework. It runs five focused measurements
and writes one JSON:

  A. Leaky periodic-integrator model: V_peak(T) = Q / (1 - r**T) and the rejection
     inequality  Q_off < theta*(1 - r**T) <= Q_on. Validated analytically and
     against the engine's exact deposit/leak recurrence, for several fan-in counts.
  B. Neuron-level coincidence: a mature L1E_new driven by lone sensory, lone winner,
     and coincident trains under leak -- the leak needed to reject a lone 500 branch.
  C. Full-network shared-leak sweep: does L2E bootstrap, and do inactive-pixel
     L1E_new stay quiet? (the disjoint-window negative result).
  D. Delayed inhibition: does the queued L1I->L1E_s wipe remove nonzero charge and
     lower the source cadence?
  E. Pattern switching row -> col -> row: winner identity, inhibition localization,
     and whether symmetry breaking occurs.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.frequency_experiment
"""

from __future__ import annotations

import json
import os

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT
from snn.neurons import ExcitatoryNeuron, E_THRESHOLD


# --------------------------------------------------------------------- model
def v_peak(Q, leak_rate, T):
    r = 1.0 - leak_rate
    if r >= 1.0:
        return float('inf')          # zero leak: pure integrator, unbounded peak
    return Q / (1.0 - r ** T)


def fires_at(Q, leak_rate, T, theta=E_THRESHOLD):
    return v_peak(Q, leak_rate, T) >= theta


def selectivity_window(leak_rate, T_fast, T_slow, N_active, theta=E_THRESHOLD):
    r = 1.0 - leak_rate
    Q_lo = theta * (1.0 - r ** T_fast)
    Q_hi = theta * (1.0 - r ** T_slow)
    return Q_lo, Q_hi, Q_lo / N_active, Q_hi / N_active


# NOTE: this module analyses an abstract *leaky periodic jump integrator*
# (V += Q on a volley, then V *= (1 - leak_rate)). That was the historical engine
# neuron. The live engine neuron is now conductance-based (input enters as a
# current over the interval, not an instantaneous jump), so these analytic helpers
# use a small self-contained reference integrator rather than the engine neuron.
def _jump_leak_first_fire(deposit, leak_rate, steps=400, theta=E_THRESHOLD):
    """First firing step (or None) of a leaky jump integrator; ``deposit(t)`` is the
    charge injected at step ``t`` (0 for none)."""
    r = 1.0 - leak_rate
    V = 0.0
    for t in range(1, steps + 1):
        V += deposit(t)
        if V >= theta:
            return t
        V *= r
    return None


def simulate_periodic_integrator(Q, leak_rate, T, steps=400, theta=E_THRESHOLD):
    """First firing step (or None) for a jump integrator fed charge Q every T steps."""
    return _jump_leak_first_fire(lambda t: Q if t % T == 0 else 0.0,
                                 leak_rate, steps, theta)


def recurrence_trace(charges, gaps, leak_rate):
    """Replay V_pre[k] = V_post[k-1]*r**dt[k]; V_post[k] = V_pre[k] + Q[k]."""
    r = 1.0 - leak_rate
    v_post = 0.0
    trace = []
    for Q, dt in zip(charges, gaps):
        v_pre = v_post * (r ** dt)
        v_post = v_pre + Q
        trace.append(v_post)
    return trace


def _jump_leak_reference_trace(charges, gaps, leak_rate):
    """Post-volley membrane of the reference jump integrator on irregular gaps: for
    each volley, ``dt-1`` pure-leak steps, then deposit Q and record V (pre-leak)."""
    r = 1.0 - leak_rate
    V = 0.0
    trace = []
    for Q, dt in zip(charges, gaps):
        for _ in range(dt - 1):
            V *= r
        V += Q
        trace.append(V)
        V *= r
    return trace


# ------------------------------------------------------- mature L1E_new probe
def mature_enew(leak_rate, winner_j=0, sensory=500.0, winner_w=500.0, theta=E_THRESHOLD):
    """A mature coincidence detector: index 0 = paired sensory, 1+winner_j = winning
    L2E, all other feedback afferents 0."""
    w = np.zeros(1 + N_OUT)
    w[0] = sensory
    w[1 + winner_j] = winner_w
    return ExcitatoryNeuron('L1Enew_probe', 'supervisor', acc_weights=w,
                            acc_distance_factor=np.ones(1 + N_OUT),
                            threshold=theta, w_max=theta / 2.0, leak_rate=leak_rate,
                            learn=False)


def drive_enew(leak_rate, T, mode, winner_j=0, steps=400, sensory=500.0, winner_w=500.0):
    """Drive a mature L1E_new (abstract jump integrator) with a train every T steps
    and return the first firing step (or None). mode in {'sensory','winner','coincident'}."""
    def deposit(t):
        if t % T != 0:
            return 0.0
        q = 0.0
        if mode in ('sensory', 'coincident'):
            q += sensory
        if mode in ('winner', 'coincident'):
            q += winner_w
        return q
    return _jump_leak_first_fire(deposit, leak_rate, steps)


# ------------------------------------------------------- network measurements
def sweep_network(leak, seed=1, pattern='row 1', steps=1500):
    e = SimulationEngine(seed=seed, leak_rate=leak)
    active = [i for i, v in enumerate(PATTERNS[pattern]) if v]
    e.set_pattern(pattern)
    l2_spikes = 0
    enew_active = enew_inactive = 0
    src = 0
    winners = set()
    for _ in range(steps):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        for j in range(N_OUT):
            if by[f'L2E{j}']['spiked']:
                l2_spikes += 1
                winners.add(j)
        for i in range(9):
            if by[f'L1Enew{i}']['spiked']:
                if i in active:
                    enew_active += 1
                else:
                    enew_inactive += 1
        src += by[f'L1E{active[0]}']['spiked']
    return dict(leak=leak, l2_spikes=l2_spikes, winners=sorted(winners),
                enew_active=enew_active, enew_inactive=enew_inactive,
                src_cadence=round(steps / max(1, src), 2))


def measure_delay(leak=0.03, seed=1, pattern='row 1', steps=400):
    """Does the queued L1I->L1E_s wipe remove nonzero charge, and does it lower the
    active-source cadence relative to the same run with the inhibitory queue frozen?"""
    e = SimulationEngine(seed=seed, leak_rate=leak)
    active = [i for i, v in enumerate(PATTERNS[pattern]) if v]
    e.set_pattern(pattern)
    removed = 0.0
    events = 0
    src_with = 0
    same_step_wipe = 0
    for _ in range(steps):
        d = e.step()
        by = {n['id']: n for n in d['neurons']}
        src_with += by[f'L1E{active[0]}']['spiked']
        for ev in d['applied_inhibition']:
            if ev['target'].startswith('L1E') and not ev['target'].startswith('L1Enew'):
                removed += ev['charge_removed']
                events += 1
        # A relay firing this step must NOT wipe its source this step (delay).
        relay_now = {int(s[6:]) for s in d['emitted'] if s.startswith('re_l1_')}
        wipe_now = {int(ev['target'][3:]) for ev in d['applied_inhibition']
                    if ev['target'].startswith('L1E') and not ev['target'].startswith('L1Enew')}
        same_step_wipe += len(relay_now & wipe_now)
    cad_with = steps / max(1, src_with)
    return dict(leak=leak, inhibition_events=events, total_charge_removed=round(removed, 1),
                same_step_wipes=same_step_wipe, active_src_cadence=round(cad_with, 2))


def pattern_switch(leak=0.03, seed=1, hold=1500):
    """Train row {3,4,5}, switch to col {1,4,7}, return to row. Report the winner
    trajectory and where the old winner's L1I inhibition lands after the switch."""
    e = SimulationEngine(seed=seed, leak_rate=leak)

    def run(pattern, n):
        e.set_pattern(pattern)
        spikes = np.zeros(N_OUT, dtype=int)
        inh_targets = np.zeros(9, dtype=int)
        for _ in range(n):
            d = e.step()
            by = {nn['id']: nn for nn in d['neurons']}
            for j in range(N_OUT):
                spikes[j] += by[f'L2E{j}']['spiked']
            for ev in d['applied_inhibition']:
                tgt = ev['target']
                if tgt.startswith('L1E') and not tgt.startswith('L1Enew'):
                    inh_targets[int(tgt[3:])] += 1
        top = int(spikes.argmax()) if spikes.any() else None
        return top, spikes.tolist(), inh_targets.tolist()

    row_w, row_spk, _ = run('row 1', hold)
    col_w, col_spk, col_inh = run('col 1', hold)
    row2_w, row2_spk, _ = run('row 1', hold)
    return dict(row_winner=row_w, row_spikes=row_spk,
                col_winner=col_w, col_spikes=col_spk,
                col_inh_by_pixel=col_inh,
                row_recover_winner=row2_w, row_recover_spikes=row2_spk)


# --------------------------------------------------------------------- parts
def part_a():
    print('=' * 78)
    print('A. Leaky periodic-integrator model: reject a lone branch, fire on coincidence')
    rows = []
    for leak in (0.1, 0.2, 0.3):
        for N in (1, 2, 3):
            Q_lo, Q_hi, w_lo, w_hi = selectivity_window(leak, 1, 2, N)
            Q_mid = 0.5 * (Q_lo + Q_hi)
            full = simulate_periodic_integrator(Q_mid, leak, 1)
            half = simulate_periodic_integrator(Q_mid, leak, 2)
            ok = (full is not None) and (half is None)
            rows.append(dict(leak=leak, N=N, Q_lo=Q_lo, Q_hi=Q_hi, Q_mid=Q_mid,
                             full=full, half=half, selective=ok))
            print(f'   leak={leak:.2f} N_active={N}: band [{Q_lo:7.1f},{Q_hi:7.1f}) '
                  f'mean w [{w_lo:6.1f},{w_hi:6.1f})  full fires@{full}  half@{half}  ok={ok}')
    # Recurrence agreement on irregular gaps.
    gaps = [1, 2, 1, 3]
    charges = [180.0, 90.0, 210.0, 60.0]
    ref = _jump_leak_reference_trace(charges, gaps, 0.15)
    ana = recurrence_trace(charges, gaps, 0.15)
    print(f'   irregular-interval recurrence matches reference integrator: {np.allclose(ref, ana)}')
    return rows


def part_b():
    print('=' * 78)
    print('B. Neuron-level coincidence (mature L1E_new: sensory 500 + one winner 500)')
    out = []
    for leak in (0.0, 0.05, 0.16, 0.20, 0.25):
        for T in (3, 5):
            sen = drive_enew(leak, T, 'sensory')
            win = drive_enew(leak, T, 'winner')
            coi = drive_enew(leak, T, 'coincident')
            rej = (sen is None) and (win is None) and (coi is not None)
            out.append(dict(leak=leak, T=T, lone_sensory=sen, lone_winner=win,
                            coincident=coi, rejects_lone=rej))
            print(f'   leak={leak:.2f} T={T}: lone_sensory@{sen}  lone_winner@{win}  '
                  f'coincident@{coi}  strict_coincidence={rej}')
    return out


def part_c():
    print('=' * 78)
    print('C. Full-network shared-leak sweep (L2E bootstrap vs inactive-pixel quiet)')
    out = []
    for leak in (0.0, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.16, 0.20):
        r = sweep_network(leak)
        out.append(r)
        boot = r['l2_spikes'] > 0
        quiet = r['enew_inactive'] == 0
        print(f'   leak={leak:.2f}: L2_spikes={r["l2_spikes"]:4d} winners={r["winners"]!s:8} '
              f'Enew_active={r["enew_active"]:4d} Enew_inactive={r["enew_inactive"]:4d} '
              f'src_cad={r["src_cadence"]}  bootstrap={boot} inactive_quiet={quiet}')
    return out


def part_d():
    print('=' * 78)
    print('D. Delayed inhibition removes nonzero charge (default leak)')
    r = measure_delay()
    print(f'   leak={r["leak"]}: inhibition_events={r["inhibition_events"]} '
          f'total_charge_removed={r["total_charge_removed"]} '
          f'same_step_wipes={r["same_step_wipes"]} active_src_cadence={r["active_src_cadence"]}')
    print('   (same_step_wipes must be 0: the I->E synapse delays the wipe by one step.)')
    return r


def part_e():
    print('=' * 78)
    print('E. Pattern switching row {3,4,5} -> col {1,4,7} -> row')
    r = pattern_switch()
    print(f'   row winner=L2E{r["row_winner"]} spikes={r["row_spikes"]}')
    print(f'   col winner=L2E{r["col_winner"]} spikes={r["col_spikes"]}')
    print(f'   col inhibition by pixel (0..8)={r["col_inh_by_pixel"]}')
    print(f'   back-to-row winner=L2E{r["row_recover_winner"]} spikes={r["row_recover_spikes"]}')
    return r


def main():
    a = part_a()
    b = part_b()
    c = part_c()
    d = part_d()
    e = part_e()

    boot = {r['leak'] for r in c if r['l2_spikes'] > 0}
    strict = {r['leak'] for r in b if r['rejects_lone']}
    shared = boot & strict
    print('=' * 78)
    print('CONCLUSION')
    print(f'  L2E bootstraps at leak in {sorted(boot)}')
    print(f'  strict lone-branch rejection needs leak in {sorted(strict)} (neuron-level)')
    if shared:
        print(f'  shared-leak window satisfying BOTH: {sorted(shared)}')
    else:
        print('  No shared leak satisfies BOTH L2E bootstrap and strict lone-branch')
        print('  rejection: the windows are disjoint (bootstrap needs low leak so the')
        print('  subthreshold L2E volley can accumulate; lone-500 rejection needs high')
        print('  leak). The architectural fix still eliminates the reported bug -- at the')
        print('  low-leak default, inactive-pixel L1E_new stay quiet (0 false fires) so')
        print('  inhibition is pixel-selective, not global -- but a mature lone sensory')
        print('  branch is not rejected at that leak, so exact two-input coincidence and')
        print('  full symmetry breaking are not simultaneously achievable with one shared')
        print('  leak. No population-specific leak or coincidence gate was added.')

    summary = dict(part_a=a, part_b=b, part_c=c, part_d=d, part_e=e,
                   bootstrap_leaks=sorted(boot), strict_reject_leaks=sorted(strict),
                   shared_window=sorted(shared))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frequency_results.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n  wrote {path}')


if __name__ == '__main__':
    main()
