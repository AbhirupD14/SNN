# Implemented RG residual/error pathway

![Residual/error pathway architecture](residual_error_pathway_flowchart.svg)

This diagram describes the implemented `rg_residual` preset. It is deliberately
separate from the legacy `rg` preset, which places RG in front of the old cortical
topology. `rg_residual` instead preserves L1E as the complete evidence representation
and applies predictive inhibition to a separate ErrorE sheet.

## Population and projection counts

The circuit contains **52 cells**:

- 9 uninhibitable retinal sources (`RG0..RG8`)
- 9 plastic, noncompetitive evidence encoders (`L1E0..L1E8`)
- 9 excitatory residual cells (`ErrorE0..ErrorE8`)
- 8 categorical competitors (`L2E0..L2E7`)
- 8 paired inhibitory predictors (`PI0..PI7`)
- 8 paired inhibitory switch cells (`SwitchI0..SwitchI7`)
- 1 shared inhibitory WTA relay (`L2I`)

That is 35 excitatory/source cells and **17 inhibitory interneurons**. There is no
global `NoveltyE` cell.

The circuit contains **274 directed internal projections**:

| Projection | Count | Meaning |
| --- | ---: | --- |
| `RG_i -> L1E_i` | 9 | Paired plastic sensory excitation |
| `L1E_i -> ErrorE_i` | 9 | Paired evidence copy |
| `L1E_i -> L2E_j` | 72 | Dense plastic recognition excitation |
| `L2E_j -> PI_j` | 8 | Paired predictor relay |
| `PI_j --| ErrorE_i` | 72 | Dense learned predictive inhibition |
| `ErrorE_i -> SwitchI_j` | 72 | Dense residual broadcast |
| `L2E_j -> SwitchI_j` | 8 | Paired excitation carrying local trace `x_j` |
| `SwitchI_j --| L2E_j` | 8 | Paired incumbent-only inhibition |
| `L2E_j -> L2I` | 8 | Shared WTA drive |
| `L2I --| L2E_j` | 8 | Fast global L2 competition |

## Familiar input

For the familiar row `{3,4,5}`, RG and L1E supply the complete row to every L2
competitor. The established owner `L2E3` wins the ordinary WTA and activates `PI3`.
Its learned inhibition shunts the predicted `ErrorE3`, `ErrorE4`, and `ErrorE5`
responses. With the residual sheet quiet, no SwitchI receives both coincidence
branches, so `L2E3` remains the single categorical owner.

## Overlapping input

For column `{1,4,7}` following the row, L1E still supplies the complete column to all
L2 competitors. The row prediction explains shared feature 4, while `ErrorE1` and
`ErrorE7` fire. Those residual spikes broadcast to every SwitchI. Only `SwitchI3`
should also have the recent local trace produced by an actual `L2E3` spike, so it
temporarily inhibits only `L2E3`. The switch does not select a replacement: the
remaining L2 cells use the complete L1 column and the shared L2I competition to
select exactly one rival.

## Unresolved timing assumptions

The local causal mechanics are implemented and regression-tested, but long-horizon
classification success has not been established. These questions remain open:

- Does the bounded two-branch SwitchI charge calibration generalize beyond the tested
  one- and two-feature residual volleys without becoming too permissive or too quiet?
- Does error-driven incumbent inhibition arrive before the incumbent contaminates
  its weights with novel features?
- After a rival wins, does its predictor learn the complete pattern quickly enough
  for residual activity to disappear?
- Can the circuit settle to one stable final winner rather than oscillating?

The recent-winner state is intended to be a local decaying trace on each real
`L2E_j -> SwitchI_j` pathway: an L2E spike sets `x_j` to one, and otherwise it decays
as `x_j <- rho * x_j`. ErrorE events add visible residual branch charge to every
connected switch. Both the residual and trace-priming branches are bounded below the
inhibitory threshold individually. It is not implemented as a global or sticky winner ID.
