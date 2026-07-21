# Open Problem: Temporal AND Semantics and Winner Turnover

> **Historical (pre-conductance).** This document analyses the earlier subtractive
> hard-wipe model. Inhibition is now persistent conductance and winner turnover is
> addressed by local predictive inhibition; see
> `Current_Implementation_Methodology_Equations.md`. Kept for context only.


## Status

The July 2026 backend rebuild fixed two concrete defects in the L1 feedback
circuit:

1. `L1E_new[i]` now receives a real local synapse from its paired `L1E_s[i]`, so
   inactive pixels do not train toward every L2 winner.
2. `L1I[i]` still relays instantly, but its `I -> E` output is delivered one
   timestep later, after fresh sensory charge and before the next sensory
   threshold check. Inhibition now removes real charge instead of landing on an
   already-reset source.

These changes make inhibition spatially selective during a fixed pattern. They
do **not** yet make `L1E_new` a reliable temporal AND gate, and they do not
produce emergent winner turnover when the input pattern changes.

## Intended Boolean relation

For a pixel `i`, define:

- `S_i(t)`: the paired sensory neuron spikes in the evaluation window;
- `W_j(t)`: L2 competitor `j` wins in that window;
- `A_ij`: the learned feedback association between pixel `i` and winner `j`.

The intended coincidence detector is

```text
C_i(t) = S_i(t) AND OR_j(A_ij AND W_j(t)).
```

Its relay should then veto the paired sensory source in the following window:

```text
L1E_s_allowed_i(t + 1) = sensory_drive_i(t + 1) AND NOT C_i(t).
```

At the default excitatory threshold `theta = 1000` and shared weight cap
`w_max = theta/2 = 500`, a mature detector is intended to implement this relation
with two half-threshold inputs:

| Paired sensory | Associated winner | Charge | Expected output |
| --- | --- | ---: | --- |
| 0 | 0 | 0 | silent |
| 1 | 0 | 500 | silent |
| 0 | 1 | 500 | silent |
| 1 | 1 | 1000 | spike |

This truth table is valid for simultaneous deposits into an empty membrane. A
leaky integrate-and-fire neuron also retains state from earlier timesteps, so it
does not automatically obey the table over a sequence of events.

## Observed failure

The production default uses a shared leak of `0.03`. That value allows L2E to
bootstrap by accumulating sub-threshold feedforward volleys, but it also lets an
`L1E_new` detector accumulate repeated inputs from one branch. In temporal terms,
the implemented detector can behave like

```text
sensory_now OR sensory_recently OR feedback_recently
```

rather than strict sensory-and-feedback coincidence.

This distinction is hidden during a fixed row presentation: inactive pixels
receive no local sensory spikes, and the small initial feedback input alone is
insufficient, so inactive `L1E_new` neurons remain quiet. It becomes visible on a
pattern change. When switching from row `{3,4,5}` to column `{1,4,7}`, the new
exclusive pixels `1` and `7` repeatedly deliver their nearly half-threshold local
sensory charge. At the shared low leak, that lone branch eventually fires their
coincidence detectors. Their I relays then inhibit the very sources that were
supposed to remain available to a rival.

Measured row -> column -> row runs therefore retain the same L2 winner across all
three phases. This is not emergent symmetry breaking.

## Why one shared leak is currently infeasible

Let `r = 1 - leak_rate`. For periodic charge `Q` arriving every `T` timesteps, the
pre-threshold peak is

```text
V_peak = Q / (1 - r^T).
```

A valid temporal coincidence window requires

```text
Q_off < theta * (1 - r^T) <= Q_on,
```

where `Q_off` is the largest unmatched branch and `Q_on` is the matched sensory
plus feedback charge. For a mature unmatched half-threshold branch,

```text
r^T < 1/2.
```

The current experiment measured two disjoint operating regions:

- L2E cold bootstrap requires shared leak `<= 0.04`.
- Strict rejection of a repeated mature 500-unit lone branch requires leak
  approximately `>= 0.16` for the measured cadence.

The production value `0.03` deliberately keeps the full network alive and fixes
the original global-training defect, but it cannot also provide strict temporal
AND semantics.

This is a role conflict: L2E is a slow evidence accumulator, while L1E_new is
intended to be a fast coincidence detector. Sharing one membrane time constant
forces incompatible behavior.

## Association bootstrap is a second problem

Even with a faster `L1E_new` leak, a weak incumbent feedback weight plus a newly
active local sensory input is a real physical coincidence. Repetition can cause
the incumbent to learn the new pixel before a rival wins. The desired behavior
therefore depends on a separation of timescales:

1. New-exclusive sensory sources remain available.
2. A rival responds and wins.
3. The rival's feedback association consolidates.
4. This happens before the incumbent recruits those new coincidence detectors.

If role-specific leak does not create that window, the remaining limitation is
not the Boolean gate itself. It is the relationship between association learning,
winner-only feedforward learning, and consolidation.

## Next evidence-driven experiment

The next minimal design change to test is a role-specific membrane time constant:

- retain the low leak needed by `L1E_s` and `L2E` accumulation;
- give `L1E_new` a faster, fixed and documented leak derived from measured event
  intervals;
- keep the shared threshold `1000` and shared accumulating-weight cap `500`;
- avoid adding another dashboard control until a feasible value is demonstrated.

This is a scientific change, not yet part of production. It should be accepted
only if sequence-level tests establish the gate contract:

| Event sequence | Required detector behavior |
| --- | --- |
| repeated sensory only | never fires |
| repeated feedback only | never fires |
| sensory and feedback outside the coincidence window | never fires |
| paired sensory and associated feedback in the window | fires |
| repeated valid coincidences during bootstrap | can learn without false lone-branch firing |

After that contract passes, the row -> column -> row experiment must still show
that new-exclusive sources remain uninhibited long enough for a different L2
competitor to emerge. If it does not, candidate changes such as slower feedback
consolidation, local sub-threshold coincidence learning, or loser learning must be
evaluated explicitly. No winner should be removed through an engine-level
override, and no negative result should be hidden behind a heuristic.

## Reproduction and evidence

Run:

```bash
PYTHONPATH=. .venv/bin/python -m experiments.frequency_experiment
```

The machine-readable measurements are written to
`experiments/frequency_results.json`. The implemented topology, timestep order,
equations, and current negative result are also recorded in
`Current_Implementation_Methodology_Equations.md`.

The separate problem of an established L2E winner that crosses threshold in one
volley, before frequency modulation can act, and the proposed intrinsic-adaptation
experiment are documented in `docs/INTRINSIC_ADAPTATION_DESIGN.md`.
