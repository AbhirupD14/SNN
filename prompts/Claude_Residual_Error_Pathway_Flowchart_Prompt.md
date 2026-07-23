# Create a graphical residual/error-pathway architecture flowchart

## Objective

Create a polished, genuinely graphical vector flowchart of the **proposed
classification-preserving residual/error topology** described below. This is a design
visualization task only: do not implement the neural circuit and do not modify the
current `pi`, `old`, or `rg` presets.

The resulting image must make the cell layout and E/I signal flow understandable
without relying on ASCII art. It must be clear enough that we can inspect the proposed
topology together and decide whether it is worth implementing.

## Deliverables

Create:

1. `docs/residual_error_pathway_flowchart.svg` — the primary, self-contained,
   publication-quality vector image.
2. `docs/residual_error_pathway_flowchart.md` — a short companion explaining how to
   read the image, the exact population/edge counts, and the unresolved timing
   assumptions.

Optionally create a PNG preview only if a conversion tool is already installed and
works immediately. The SVG is the required artifact. Do not install Playwright,
Chromium, a headless browser, or any new dependency merely to render it; author and
validate the SVG directly.

The SVG should use a wide landscape canvas around 1600×1100 logical pixels, remain
legible when scaled down, include a descriptive `<title>` and `<desc>` for
accessibility, and contain no external fonts, scripts, network resources, or raster
images.

## Important distinction from the implemented network

The current built-in `rg` preset is RG added to the **old** cortical topology:

```text
RG -> plastic L1E -> L2E WTA -> every L1I -> every paired L1E inhibited
```

That is not the circuit to draw as the main flowchart. Put a small warning badge in
the image:

> PROPOSED RESIDUAL DESIGN — not the current `rg` preset

The proposed circuit moves predictive inhibition off the main L1 representation and
onto a separate 3×3 residual sheet.

## Exact populations to draw

Use visually distinct population containers arranged from left/bottom upstream to
right/top downstream:

| Population | Count | IDs | Class/role |
| --- | ---: | --- | --- |
| Retinal sources | 9 | `RG0..RG8` | Source (`S`), exogenous and uninhibitable |
| Main evidence grid | 9 | `L1E0..L1E8` | Excitatory (`E`), plastic noncompetitive encoders |
| Residual/error grid | 9 | `ErrorE0..ErrorE8` | Excitatory (`E`), paired feature-error cells |
| Categorical pool | 8 | `L2E0..L2E7` | Excitatory (`E`), deterministic single-winner WTA |
| Predictors | 8 | `PI0..PI7` | Inhibitory (`I`), one paired with each L2E |
| Incumbent switch cells | 8 | `SwitchI0..SwitchI7` | Inhibitory (`I`), one paired with each L2E |
| WTA relay | 1 | `L2I` | Inhibitory (`I`), shared fast L2 competition |

Total: **52 cells**, including **17 inhibitory interneurons**. There is deliberately
no global `NoveltyE` cell in this version. Residual spikes project directly to the
switch bank so novelty remains distributed and local.

Render each 9-cell feature population as a recognizable 3×3 sheet. Render each
8-cell L2/PI/Switch population as an eight-cell ring or aligned indexed bank so the
pairing `L2E_j <-> PI_j <-> SwitchI_j` is visually obvious. Put the shared `L2I` in
the center of the L2 ring.

## Exact projections to draw

Every connection below must appear, with its sign, plasticity, and multiplicity.
Use representative indexed paths plus clearly labelled fanout/fanin bundles rather
than drawing 274 individual lines as spaghetti.

| Projection | Count | Sign | Plasticity/meaning |
| --- | ---: | ---: | --- |
| `RG_i -> L1E_i` | 9 | `+` | Plastic paired sensory afferent |
| `L1E_i -> ErrorE_i` | 9 | `+` | Paired evidence copy into residual sheet |
| `L1E_i -> L2E_j` | 72 | `+` | Dense plastic full-pattern recognition path |
| `L2E_j -> PI_j` | 8 | `+` | Paired winner-to-predictor relay |
| `PI_j -> ErrorE_i` | 72 | `-` | Dense learned predictive inhibition; targets ErrorE, not L1E |
| `ErrorE_i -> SwitchI_j` | 72 | `+` | Dense residual evidence delivered to every switch cell |
| `L2E_j -> SwitchI_j` | 8 | `+` | Paired local winner-eligibility trace `x_j` |
| `SwitchI_j -> L2E_j` | 8 | `-` | Paired incumbent-only inhibition |
| `L2E_j -> L2I` | 8 | `+` | Winner drives shared WTA relay |
| `L2I -> L2E_j` | 8 | `-` | Fast global L2 competition |

Total: **274 directed internal projections**.

Do not draw or imply any negative `E -> E` connection. Every inhibitory action must
originate from an inhibitory cell:

- `PI_j --| ErrorE_i`
- `SwitchI_j --| L2E_j`
- `L2I --| L2E_j`

Excitatory cells may excite inhibitory cells, which then provide inhibition.

## The two parallel L1 roles

Make this separation the visual center of the diagram:

### Main evidence path

```text
RG -> L1E -> L2E
```

The complete input remains available for classification. Predictive inhibition does
**not** shunt L1E in this topology.

### Residual/error path

```text
L1E -> ErrorE
PI --| ErrorE
```

Each `ErrorE_i` represents approximately:

```text
present feature i - feature i predicted by the current/recent L2 owner
```

This must be shown as conductance-based cancellation/shunting, not arithmetic
subtraction performed by the engine.

## Show the local recent-winner mechanism precisely

The label “recent L2 winner” must not look like a global engine variable or a router.
Show a small local state marker `x_j` beside every paired `L2E_j -> SwitchI_j`
connection:

```text
on an actual L2E_j spike:  x_j <- 1
otherwise:                 x_j <- rho * x_j
```

`SwitchI_j` is a temporal coincidence detector:

```text
paired local winner trace x_j
AND
current residual ErrorE activity
-> SwitchI_j fires
-> paired L2E_j is temporarily inhibited
```

All ErrorE spikes broadcast to all connected SwitchI cells. The engine does not route
the error only to the incumbent. Only the switch cell whose paired winner trace is
still active should reach threshold. Use a small inset truth table:

| Winner trace `x_j` | Residual input | `SwitchI_j` |
| --- | --- | --- |
| 0 | 0 | silent |
| 1 | 0 | silent |
| 0 | 1 | silent |
| 1 | 1 | fires |

Add a visible caution next to this inset:

> Timing assumption to validate: this temporal AND must reject repeated activity on
> either branch alone and must never consult a global/sticky winner ID.

## Add two causal scenario panels

Under the main architecture, add two smaller left-to-right sequence panels.

### A. Familiar input: stable classification

Use row `{3,4,5}` as the example:

1. `RG3,4,5 -> L1E3,4,5` supplies the full row.
2. The established row owner `L2E3` wins the ordinary WTA.
3. `PI3` predicts features `{3,4,5}` and inhibits paired ErrorE cells.
4. The residual sheet is quiet.
5. No SwitchI receives both required branches.
6. `L2E3` remains the single categorical owner.

### B. Overlapping new input: incumbent release

Use column `{1,4,7}` after the row owner:

1. L1E still supplies the complete column `{1,4,7}` to every L2 competitor.
2. The row prediction explains/suppresses shared ErrorE4.
3. Novel `ErrorE1` and `ErrorE7` fire.
4. Their spikes broadcast to all SwitchI cells.
5. Only `SwitchI3`, locally primed by the recent `L2E3` trace, fires.
6. `SwitchI3 --| L2E3` temporarily removes the incumbent from the next competition.
7. Other L2 cells compete using the **full** L1 column, and exactly one rival wins.

Do not imply that the rival is chosen by the switch circuit. The switch suppresses
only the incumbent; ordinary L2 WTA chooses the replacement.

## Visual language

Use a consistent, high-contrast legend:

- RG/source cells: amber/gold.
- Excitatory cells: teal/green.
- Inhibitory cells: magenta/red.
- Positive excitation: green solid arrow with arrowhead.
- Plastic positive excitation: green solid arrow plus a small weight/learning marker.
- Fixed inhibition: red line ending in a T-bar.
- Learned predictive inhibition: purple or magenta line ending in a T-bar, labelled
  `learned w_ji`.
- Local eligibility trace: amber dashed line or badge labelled `x_j`, but retain a
  positive arrow where the actual L2E-to-Switch excitation is shown.
- Dense bundles: a tapered band or bracket labelled, for example, `dense 9×8 (+)`;
  also show at least one explicit representative indexed edge.
- Paired 1:1 projections: parallel aligned lines or a bracket labelled `paired 1:1`.

Include a compact counts box and a legend explaining `+`, T-bar inhibition, plastic,
dense, paired, and trace. Ensure labels never overlap edges at normal viewing size.

## Accuracy constraints

The flowchart must make all of these points visually unambiguous:

1. RGC is upstream and cannot be inhibited.
2. L1E remains the full evidence representation.
3. PI inhibits ErrorE, not L1E.
4. ErrorE is excitatory residual evidence.
5. A SwitchI is a real inhibitory interneuron, not a negative E-to-E edge.
6. Residual events broadcast to all SwitchI connections; weights/state and threshold
   determine response—there is no selective engine routing.
7. SwitchI inhibits only its paired incumbent L2E.
8. Shared L2I still enforces exactly one L2 winner.
9. The “recent winner” is a local decaying trace from a real L2E spike.
10. The circuit is proposed and its temporal coincidence dynamics remain unverified.

## Companion Markdown

The companion file should embed the SVG and briefly list:

- the 52-cell population count;
- the 274 directed-projection count;
- the familiar-input sequence;
- the overlapping-input sequence;
- the unresolved timing/race questions:
  - Can `SwitchI` implement a strict temporal AND without false single-branch firing?
  - Does error-driven incumbent inhibition arrive before the incumbent contaminates
    its weights with novel features?
  - After a rival wins, does its predictor learn the complete pattern quickly enough
    for residual activity to disappear?
  - Can the circuit settle to one stable final winner rather than oscillating?

Do not claim that these questions have been simulated or solved.

## Verification

After creating the files:

1. Parse the SVG as XML using an already installed standard-library or project tool.
2. Check that every population name, count, projection multiplicity, sign, and
   plasticity label listed above appears in the SVG.
3. Confirm the companion Markdown embeds the correct relative SVG path.
4. Run `git diff --check`.
5. Report the exact files created and any visual validation that could not be done.

Do not change simulation code, tests, golden fixtures, experiment outputs, or current
architecture documentation in this visualization-only task.
