# From Spikes to Symbols — Sources Appendix

The deck is grounded in this repository's own architecture and in a set of
well-established computational-neuroscience motifs. Motifs are cited as *biological
inspiration*, not as a claim of faithful simulation of any specific cell type or circuit.

## Primary grounding — repository documents (this work)

| Document | What it grounds in the deck |
|---|---|
| `README.md` | Overall architecture, the four 3×3 line primitives, the local learning rule, network topology (L1 encoders → L2 integrators, shared L2I). |
| `Current_Implementation_Methodology_Equations.md` | The equations behind every mechanism in the *default* dashboard configuration: LIF integration, threshold/firing, signed-spike plasticity, the competitive reset. |
| `Architecture_Decisions_Report.md` | Decision-by-decision log with locality and biological-plausibility assessment for each mechanism; the collapse/tyrant failure mode and its fix. |
| `Inhibitory_Off_Weight_Recruitment_Spec.md` | The competitive-reset + conservative ON→OFF redistribution/recruitment rule used on slide 7 ("losing capacity is reallocated"). |
| `PaulVersion.md` | The "1 vs Z" paradigm spec (LIF, WTA, STDP, homeostasis) and its explicit non-goals (no backprop, no batch training). |
| `docs/DASHBOARD.md` | The live FastAPI + WebSocket + Three.js dashboard referenced in the architecture summary. |
| Core code — `neuron_flexible.py`, `cortical_column_flexible.py`, `backend/simulation.py`, `snn/rules/` | The actual neuron model, per-pixel feedforward fan-in, simulation engine, and the shared bounded learning kernels. |

## Scientific motifs referenced (biological inspiration)

- **Leaky integrate-and-fire (LIF) neurons** — the standard reduced spiking-neuron model
  (e.g. Gerstner & Kistler, *Spiking Neuron Models*). Grounds slides 4 and the membrane
  dynamics throughout.
- **Center-surround retinal receptive fields & lateral inhibition** — Kuffler's
  center-ON/surround-OFF ganglion cells; Hartline's lateral inhibition. Grounds slide 5.
- **Cortical winner-take-all microcircuits** — the canonical soft-competition motif
  (Douglas & Martin; Amari's neural-field WTA). Grounds slide 7.
- **Spike-timing / eligibility-trace plasticity** — Hebbian learning gated by spike
  timing (Bi & Poo). Grounds the local excitatory rule.
- **Inhibitory plasticity (iSTDP) for E/I balance** — learning inhibitory synapses to
  stabilize excitation/inhibition. Grounds the inhibitory-discharge mechanism in the
  decision log.
- **Homeostatic synaptic scaling** — slow, multiplicative, activity-dependent gain
  control (Turrigiano). Referenced as a recruitment/anti-tyranny force.
- **Sparse, highly selective coding** — the idea of highly selective neurons standing for
  specific patterns (Quiroga et al.). Grounds the "specialist becomes a symbol" framing.

## What is *not* claimed

- No numeric speed, energy, or memory benchmarks are asserted; hardware efficiency is
  named as an *expected source of gain* and flagged as future physical validation
  (slide 18).
- The motifs above are computational abstractions. Several constants in the working code
  are task-fit rather than derived from biological measurement — see
  `Architecture_Decisions_Report.md §6` for the honest list of simplifications.
- The deck presents the *theory* the architecture is designed to enable; the measured
  system today is a small two-layer network learning four 3×3 line primitives.

## Companion files

- `from_spikes_to_symbols.html` — editable slideshow source.
- `From_Spikes_to_Symbols.pdf` — final 16:9 deck.
- `architecture_summary_onepage.html` / `Architecture_Summary_OnePage.pdf` — one-page summary.
- `SPEAKER_NOTES.md` — full per-slide speaker notes.
- `build.py` — renders the HTML sources to PDF (`weasyprint`).
