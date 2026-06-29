"""
generate_viz.py -- bake the network's learning history into a self-contained
high-fidelity dashboard (index.html).

Trains the model once (deterministic seed) and records:
  * per-epoch maturity (the LIF -> pure-integrator curve),
  * per-epoch win distribution (firing frequencies),
  * detailed spiking-burst traces of every pattern at several checkpoints from
    nascent -> mature, captured in observation mode (plastic=False),
  * steps-to-fire per pattern at each checkpoint (the direct LIF->integrator
    signature: many burst-steps when leaky, ~1 when a pure integrator),
  * the feed-forward gate matrix + feedback squelch matrix at each checkpoint.

Injected into viz_template.html (replacing the __DATA__ token).
"""

import json
import numpy as np

import fep_model as fm
from fep_model import FEPSNN

SEED = 0
EPOCHS = 300
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 40, 100, 299]
FREQ_WINDOW = 100

PATTERNS = {
    "Row 0":  [0, 1, 2], "Row 1":  [3, 4, 5], "Row 2":  [6, 7, 8],
    "Col 0":  [0, 3, 6], "Col 1":  [1, 4, 7], "Col 2":  [2, 5, 8],
    "Diag 1": [0, 4, 8], "Diag 2": [2, 4, 6],
}


def r(x, n=2):
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def strip_trace(trace):
    out = []
    for s in trace:
        out.append({
            "step": s["step"], "stage": s["stage"], "winner": s["winner"],
            "l2_fired": s["l2_fired"], "brake": s["brake"], "free_energy": s["free_energy"],
            "spk_l1e": s["spk_l1e"], "spk_l1i": s["spk_l1i"],
            "v_l1e": r(s["v_l1e"], 1), "v_l1i": r(s["v_l1i"], 1),
            "v_l2e": r(s["v_l2e"], 1), "v_l2i": r(s["v_l2i"], 1),
        })
    return out


def capture_checkpoint(snn, epoch):
    # Save the live transient state so probing does not perturb training.
    saved = (snn.v_l1e.copy(), snn.v_l1i.copy(), snn.v_l2e.copy(), snn.v_l2i.copy(),
             snn.adapt.copy(), snn.refrac.copy(), snn.silence.copy())

    traces, winners, steps = {}, {}, {}
    for name, idx in PATTERNS.items():
        snn.reset_membranes()                       # clean, independent probe
        w = snn.process_event(idx, record=True, plastic=False)
        winners[name] = (None if w is None else int(w))
        tr = snn.last_trace
        traces[name] = strip_trace(tr)
        sf = next((s["step"] + 1 for s in tr if s["l2_fired"] is not None), None)
        steps[name] = sf

    (snn.v_l1e[:], snn.v_l1i[:], snn.v_l2e[:], snn.v_l2i[:],
     snn.adapt[:], snn.refrac[:], snn.silence[:]) = saved

    avg_mat = float(np.mean(snn.maturity))
    valid = [s for s in steps.values() if s is not None]
    avg_steps = float(np.mean(valid)) if valid else None
    if avg_mat < 0.4:
        label = "Nascent - Leaky Integrate & Fire"
    elif avg_mat < 0.85:
        label = "Specialising"
    elif avg_mat < 0.99:
        label = "Maturing"
    else:
        label = "Mature - Pure Integrator"
    return {
        "epoch": epoch, "label": label,
        "avg_maturity": round(avg_mat, 3),
        "avg_steps": (round(avg_steps, 2) if avg_steps is not None else None),
        "maturity": r(snn.maturity.tolist(), 3),
        "theta_l2e": r(snn.theta_l2e.tolist(), 1),
        "weights_l1e_l2e": r(snn.weights_l1e_l2e.tolist(), 2),
        "weights_l2e_l1i": r(snn.weights_l2e_l1i.tolist(), 2),
        "traces": traces, "winners": winners, "steps_to_fire": steps,
    }


def main():
    np.random.seed(SEED)
    snn = FEPSNN()

    maturity_hist, avg_mat_hist = [], []
    win_counts = np.zeros(fm.N_L2_E)
    checkpoints = []

    if 0 in CHECKPOINTS:
        checkpoints.append(capture_checkpoint(snn, 0))

    names = list(PATTERNS.keys())
    for epoch in range(EPOCHS):
        order = names[:]
        np.random.shuffle(order)
        for name in order:
            w = snn.process_event(PATTERNS[name])
            if w is not None and epoch >= EPOCHS - FREQ_WINDOW:
                win_counts[w] += 1
        maturity_hist.append(r(snn.maturity.tolist(), 3))
        avg_mat_hist.append(round(float(np.mean(snn.maturity)), 3))
        if epoch != 0 and epoch in CHECKPOINTS:
            checkpoints.append(capture_checkpoint(snn, epoch))

    # Final niche mapping (robust: mode owner over several noisy probes).
    from collections import Counter
    mapping = {}
    for name, idx in PATTERNS.items():
        owners = [snn.volleys_to_fire(idx)[1] for _ in range(7)]
        owners = [o for o in owners if o is not None]
        mapping[name] = (Counter(owners).most_common(1)[0][0] if owners else None)

    total = win_counts.sum()
    win_freq = (win_counts / total).tolist() if total > 0 else win_counts.tolist()

    data = {
        "meta": {
            "N_L1E": fm.N_L1_E, "N_L1I": fm.N_L1_I, "N_L2E": fm.N_L2_E, "N_L2I": fm.N_L2_I,
            "BASELINE_THETA": fm.BASELINE_THETA, "LEAK_FRACTION": fm.LEAK_FRACTION,
            "MATURITY_THRESHOLD": fm.MATURITY_THRESHOLD, "SYNAPTIC_BUDGET": fm.SYNAPTIC_BUDGET,
            "ADAPT_GAIN": fm.ADAPT_GAIN, "W_GATE_MAX": fm.W_L1E_L2E_MAX,
            "W_L2E_L1I": fm.W_L2E_L1I, "T_STEPS": fm.T_STEPS, "EPOCHS": EPOCHS,
            "patterns": PATTERNS, "pattern_order": names,
        },
        "history": {"maturity": maturity_hist, "avg_maturity": avg_mat_hist},
        "checkpoints": checkpoints,
        "final": {"mapping": mapping, "win_freq": r(win_freq, 3),
                  "win_counts": [int(c) for c in win_counts]},
    }

    with open("viz_template.html") as f:
        template = f.read()
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open("index.html", "w") as f:
        f.write(html)

    uniq = len(set(v for v in mapping.values() if v is not None))
    print(f"Niche mapping: {mapping}")
    print(f"Unique symbols: {uniq}/{fm.N_L2_E}")
    print(f"Checkpoint steps-to-fire (avg): " +
          ", ".join(f"ep{c['epoch']}={c['avg_steps']}" for c in checkpoints))
    print(f"Wrote index.html ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
