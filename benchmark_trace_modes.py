"""
A/B benchmark: activity vs. confidence excitatory trace on the 8-line task.

This drives the FULL network from the dashboard engine (backend.simulation.
SimulationEngine): per-pixel feedforward fan-in, volley input, the excitatory
weight budget, AND the adaptive lateral-inhibition gate (the learned L2I->L2E
competition that actually lets more than one neuron win). Both arms are built
with an IDENTICAL seed and identical hyperparameters; the ONLY difference is
Neuron.trace_mode on the L2E neurons:

    "activity"   -- dw = lr * trace * sign(w)            (recent-activity credit)
    "confidence" -- dw = lr * (c_i / sum c_active)       (confidence-weighted credit)

so any difference in specialization is attributable to the trace semantics.

A weak-competition control (the plain test_8line network, budget only, no
adaptive gate) is also run to show that without real competition NEITHER mode
can tile -- confidence is a credit-assignment rule, not a competition mechanism.

Reported per arm:
  * assignment matrix          pattern -> winning L2 neuron
  * distinct winner count      how many of the 8 neurons are used (8 = perfect tiling)
  * receptive-field selectivity  winner's active-pixel weight / silent-pixel weight
  * single-volley latency      steps until the winner first fires on a fresh volley
  * convergence speed          epoch at which the assignment stops changing
  * budget concentration       HHI of each winner's positive weights (higher = sharper)
  * confidence concentration   HHI of each winner's confidence

Everything is local, spike-driven, and gradient-free -- this script only drives
inputs and reads out state; it never supervises learning.

Run:  .venv/bin/python benchmark_trace_modes.py
Plots (if matplotlib is present) are written to sweep_results/.
"""

import os
import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_PIX, N_OUT
from neuron import _concentration

NAMES = list(PATTERNS)
VECTORS = {k: np.array(v, dtype=float) for k, v in PATTERNS.items()}
ACTIVE = {k: [i for i, v in enumerate(vec) if v > 0.5] for k, vec in VECTORS.items()}
SILENT = {k: [i for i, v in enumerate(vec) if v <= 0.5] for k, vec in VECTORS.items()}

STEPS_PER_PATTERN = 24     # training steps per pattern per epoch (several volleys)
EVAL_WINDOW = 20           # steps in a fresh read-only presentation


# ------------------------------------------------------------------- dynamics
def _reset_dynamics(eng):
    """Zero membranes / refractory / eligibility traces and restart the volley
    clock, WITHOUT touching learned weights or confidence."""
    for n in eng.l1.excitatory_neurons + eng.l1.inhibitory_neurons:
        n.potential, n.refractory_timer, n.spiked = 0.0, 0, False
        n.trace = np.zeros_like(n.trace)
    for n in eng.l2.excitatory_neurons + [eng.l2.inhibitory_neuron]:
        n.potential, n.refractory_timer, n.spiked = 0.0, 0, False
        n._trace = np.zeros_like(n._trace)
    eng.l1i_hold = np.zeros(N_PIX)
    eng.timestep = 0


def _snap(eng):
    return ([e._weights_array.copy() for e in eng.l2.excitatory_neurons],
            [e._confidence.copy() for e in eng.l2.excitatory_neurons],
            eng.l2.inhibitory_neuron._weights_array.copy())


def _restore(eng, snap):
    ws, cs, iw = snap
    for e, w, c in zip(eng.l2.excitatory_neurons, ws, cs):
        e._weights_array, e._confidence = w.copy(), c.copy()
    eng.l2.inhibitory_neuron._weights_array = iw.copy()


def eval_pattern(eng, name):
    """Read-only presentation: reset dynamics, drive the pattern for EVAL_WINDOW
    steps counting L2E spikes, then restore all learned state (confidence updates
    on every spike, so freezing the learning rate alone would not be read-only)."""
    snap = _snap(eng)
    _reset_dynamics(eng)
    eng.set_pattern(name)
    counts = np.zeros(N_OUT, dtype=int)
    first = None
    for t in range(EVAL_WINDOW):
        eng.step()
        for j in range(N_OUT):
            if eng.spiked[f'L2E{j}']:
                counts[j] += 1
                if first is None:
                    first = t
    _restore(eng, snap)
    return counts, first


def assignment(eng):
    winners, latencies = {}, {}
    for name in NAMES:
        counts, first = eval_pattern(eng, name)
        winners[name] = int(np.argmax(counts)) if counts.sum() > 0 else None
        latencies[name] = first
    return winners, latencies


# --------------------------------------------------------------------- metrics
def selectivity_ratio(eng, winners):
    ratios = []
    for name, w in winners.items():
        if w is None:
            continue
        ff = eng.l2.excitatory_neurons[w]._weights_array[1:1 + N_PIX]
        a = ff[ACTIVE[name]].mean()
        s = ff[SILENT[name]].mean()
        ratios.append(a / s if s > 1e-9 else np.inf)
    finite = [r for r in ratios if np.isfinite(r)]
    return float(np.mean(finite)) if finite else float('inf')


def concentrations(eng, winners):
    used = sorted({w for w in winners.values() if w is not None})
    w_hhi, c_hhi = [], []
    for j in used:
        e = eng.l2.excitatory_neurons[j]
        ff = e._weights_array[1:1 + N_PIX]
        conf = e._confidence[1:1 + N_PIX]
        w_hhi.append(_concentration(ff[ff > 0]))
        c_hhi.append(_concentration(conf))
    return (float(np.mean(w_hhi)) if w_hhi else 0.0,
            float(np.mean(c_hhi)) if c_hhi else 0.0)


def run_arm(trace_mode, seed, epochs, use_gate=True):
    eng = SimulationEngine(seed=seed, trace_mode=trace_mode)
    if not use_gate:
        # Weak-competition control: disable the adaptive lateral-inhibition gate
        # learning so only the weight budget remains as a counter-force.
        for e in eng.l2.excitatory_neurons:
            e.inhibitory_learning_rate = 0.0

    history, prev, stabilized_at = [], None, None
    for epoch in range(epochs):
        for name in NAMES:
            eng.set_pattern(name)
            for _ in range(STEPS_PER_PATTERN):
                eng.step()
        winners, _ = assignment(eng)
        distinct = len({w for w in winners.values() if w is not None})
        history.append((epoch, distinct))
        cur = tuple(winners[n] for n in NAMES)
        stabilized_at = epoch if (cur == prev and stabilized_at is None) else \
            (None if cur != prev else stabilized_at)
        prev = cur

    winners, latencies = assignment(eng)
    lat = [v for v in latencies.values() if v is not None]
    w_hhi, c_hhi = concentrations(eng, winners)
    return dict(mode=trace_mode, gate=use_gate, winners=winners, latencies=latencies,
                distinct=len({w for w in winners.values() if w is not None}),
                selectivity=selectivity_ratio(eng, winners),
                budget_conc=w_hhi, conf_conc=c_hhi,
                mean_latency=float(np.mean(lat)) if lat else float('nan'),
                stabilized_at=stabilized_at, history=history, eng=eng)


# ---------------------------------------------------------------------- report
def print_arm(r):
    tag = 'adaptive gate' if r['gate'] else 'budget only (control)'
    print(f"\n--- trace_mode = {r['mode']}  [{tag}] ---")
    for name in NAMES:
        w, lat = r['winners'][name], r['latencies'][name]
        print(f"    {name:7s} -> {'E'+str(w) if w is not None else 'none':>5s}   "
              f"latency {lat if lat is not None else '-'}")
    print(f"  distinct winners        : {r['distinct']} / {N_OUT}")
    print(f"  selectivity ratio       : {r['selectivity']:.2f}x (active/silent weight)")
    print(f"  single-volley latency   : {r['mean_latency']:.2f} steps (mean, lower=faster)")
    print(f"  convergence (stabilized): epoch {r['stabilized_at']}")
    print(f"  budget concentration    : {r['budget_conc']:.3f} (HHI; uniform={1/N_PIX:.3f})")
    print(f"  confidence concentration: {r['conf_conc']:.3f} (HHI; uniform={1/N_PIX:.3f})")


def maybe_plot(res_a, res_c, outdir='sweep_results'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"\n(matplotlib unavailable: {e}; skipping plots)")
        return
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for r, c in ((res_a, 'tab:orange'), (res_c, 'tab:blue')):
        ep, dist = zip(*r['history'])
        axes[0].plot(ep, dist, label=r['mode'], color=c, lw=2)
    axes[0].set(title='Distinct winners vs epoch', xlabel='epoch',
                ylabel='distinct L2 winners', ylim=(0, N_OUT + 0.5))
    axes[0].legend(); axes[0].grid(alpha=0.3)

    W = np.array([e._weights_array[1:1 + N_PIX] for e in res_c['eng'].l2.excitatory_neurons])
    im = axes[1].imshow(W, aspect='auto', cmap='magma')
    axes[1].set(title='confidence: L2 feedforward weights', xlabel='pixel', ylabel='L2 neuron')
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    labels = ['distinct/8', 'selectivity', 'budget HHI', 'conf HHI']
    x = np.arange(len(labels)); w = 0.38
    va = [res_a['distinct'] / N_OUT, res_a['selectivity'], res_a['budget_conc'], res_a['conf_conc']]
    vc = [res_c['distinct'] / N_OUT, res_c['selectivity'], res_c['budget_conc'], res_c['conf_conc']]
    axes[2].bar(x - w / 2, va, w, label='activity', color='tab:orange')
    axes[2].bar(x + w / 2, vc, w, label='confidence', color='tab:blue')
    axes[2].set(title='Summary metrics', xticks=x); axes[2].set_xticklabels(labels, rotation=20)
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, 'trace_mode_ab.png')
    fig.savefig(path, dpi=110)
    print(f"\nsaved comparison figure -> {path}")


def main(seed=1, epochs=40):
    print(f"8-line A/B benchmark on the full engine "
          f"(seed={seed}, epochs={epochs}; identical wiring/competition, only trace_mode differs)")

    print("\n############ WEAK-COMPETITION CONTROL (weight budget only, no adaptive gate) ############")
    ctrl_a = run_arm('activity', seed, epochs, use_gate=False)
    ctrl_c = run_arm('confidence', seed, epochs, use_gate=False)
    print_arm(ctrl_a); print_arm(ctrl_c)

    print("\n############ FULL COMPETITION (weight budget + adaptive lateral-inhibition gate) ############")
    res_a = run_arm('activity', seed, epochs, use_gate=True)
    res_c = run_arm('confidence', seed, epochs, use_gate=True)
    print_arm(res_a); print_arm(res_c)

    print("\n=== verdict (full competition) ===")
    print(f"  distinct winners : activity {res_a['distinct']}  vs  confidence {res_c['distinct']}  (of {N_OUT})")
    print(f"  selectivity      : activity {res_a['selectivity']:.2f}x vs confidence {res_c['selectivity']:.2f}x")
    print(f"  budget HHI       : activity {res_a['budget_conc']:.3f} vs confidence {res_c['budget_conc']:.3f}")
    print(f"  confidence HHI   : activity {res_a['conf_conc']:.3f} vs confidence {res_c['conf_conc']:.3f}")
    print(f"  mean latency     : activity {res_a['mean_latency']:.2f} vs confidence {res_c['mean_latency']:.2f}")
    maybe_plot(res_a, res_c)
    return res_a, res_c


if __name__ == "__main__":
    main()
