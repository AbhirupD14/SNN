# Four presentation demonstration recordings — reproduction and interpretation notes

Four **independent** presentation videos of the live SNN dashboard. There is no combined
montage and no fifth compilation file.

| # | Video | Claim | Duration |
|---|---|---|---|
| 1 | `uncertainty_frequency_demo` | frequency/cadence as an **observable candidate** uncertainty signal, plus a walk through the circuit's anatomy | 97.7 s |
| 2 | `continuous_learning_demo` | four patterns learned sequentially in **one** network, no reset between phases | 58.6 s |
| 3 | `unbiased_learning_demo` | a 6 000-boundary incumbent still loses the next pattern | 79.0 s |
| 4 | `scaling_demo` | 9×9 tiled hierarchy → 9×18 two-tower hierarchy (two chapters, one recording) | 187.5 s |

Everything on screen is derived from the run: topology counts, edge-kind counts and owner ids
are read from `GET /api/state` and from the dynamic frames the recording browser actually
rendered; the cadence and turnover numbers come from a deterministic headless preflight of the
**same** engine configuration. No caption contains a hard-coded scientific value.

---

## 1. Setup and commands

```bash
# one-time
npm install --save-dev playwright
npx playwright install chromium

# deterministic headless measurement -> presentation_assets/preflight_measurements.json
PYTHONPATH=. .venv/bin/python scripts/preflight_demo_measurements.py --compare-uncapped
#   (or: npm run preflight)

# record all four videos in ONE invocation
npm run record-demo
```

Optional interfaces:

```bash
npm run record-demo -- --only uncertainty            # subset: uncertainty|continuous|unbiased|scaling
npm run record-demo -- --only continuous,unbiased
npm run record-demo -- --preflight                   # run the preflight first, then record
npm run record-demo -- --port 8000 --strict-port     # fail instead of falling back to a free port
npm run record-demo -- --keep-work                   # keep the intermediate Playwright video dir
```

The default `npm run record-demo` produces the **complete four-video set**.
`preflight_measurements.json` must exist first (it lives under `presentation_assets/`, so
wiping that directory means re-running the preflight before recording).

### Server command and readiness

The workflow starts and stops **its own** Uvicorn:

```bash
PYTHONPATH=<repo> .venv/bin/python -m uvicorn backend.api:app \
  --host 127.0.0.1 --port <port> --log-level warning
```

Before any recorded action, all five gates must pass (`waitDashboardReady`, and
`Recorder.configure` for every later topology change):

1. `GET /api/state` returns HTTP 200 with a topology payload;
2. `#st-conn-label` reads exactly `connected`;
3. the expected topology has arrived (node count asserted per video);
4. `#scene canvas` exists **and** `#pixel-grid .pixel` has exactly `rows × cols` cells for
   the active input shape (9, 81 or 162);
5. a dynamic frame has rendered at least once (`#st-status` is no longer `—`).

Everything after that waits on explicit predicates — engine `timestep`, `running`, node/edge
counts, overlay `hidden` attributes, filter-checkbox states, the inspector's `.insp-id` — never
on a bare sleep. The only sleeps are the deliberate presentation pauses (2–3 s at the start and
end of each video, plus caption read-time holds).

### Port policy — this run used 8010, not 8000

`--port` defaults to 8000. If 8000 is occupied the workflow **never terminates that
process**. It probes it: if it is an SNN dashboard *already running at seed 1* it is reused
(and left running afterwards); otherwise the recorder logs what it found and moves to an
automatically chosen free port. `--strict-port` turns the fallback into a hard failure
instead.

On this machine port 8000 was serving the developer's own dashboard at seed **2858281940**,
which is not the seed these demonstrations require, so all four videos were recorded on port
**8010**. The 8000 dashboard was untouched and was still running afterwards.

### `.claude/dashboard_seed.txt` contract

`backend/api.py` reads that file **once at import time** to construct its engine, so the file
must say `1` before the recording server starts. `SeedFile` in `scripts/demo_support.mjs`
captures the exact prior contents (including "did not exist"), writes `1`, and restores
byte-for-byte in a `finally` block.

Verified for this run: previous contents `2858281940` → restored to `2858281940`
(10 bytes). Recorded in `recording_manifest.json.seed_file`.

> Caveat worth knowing: while the recorder runs, that file transiently reads `1`. A
> **running** dashboard is unaffected (it read its seed at import), but restarting the
> developer's dashboard during a recording would give it seed 1 until the recorder restores
> the file.

---

## 2. Versions, seed, environment

| Component | Version |
|---|---|
| Node | v24.16.0 |
| Playwright | 1.62.1 |
| Chromium (headless) | 151.0.7922.34 (Chrome Headless Shell, Playwright build) |
| FFmpeg | 6.1.1-3ubuntu5 (system, `/usr/bin/ffmpeg`) |
| FFprobe | 6.1.1-3ubuntu5 (system, `/usr/bin/ffprobe`) |
| Python | `.venv/bin/python` 3.12 |
| OS | Linux 6.9.3-76060903-generic |

**Fixed seed: `1`** for every video, matching the measured experiments.
`POST /api/reseed` is never called. The recorder asserts
`GET /api/state → topology.params.seed == 1` before it records anything and aborts otherwise.

Browser viewport **and** `recordVideo.size` are both exactly **1920×1080**. Headless Chromium,
so no browser chrome appears. A fresh browser **context** is created per video, so each WebM
is genuinely independent.

---

## 3. Recording-only presentation chrome

Injected by Playwright (`OVERLAY_CSS` + `OVERLAY_JS` in `scripts/record_simulation_demos.mjs`):

* `cursor: none` and zero-width scrollbars — removes the pointer artifact and scrollbar clutter;
* a title card, a centred chapter chip, a bottom caption bar, and a top-right fast-forward badge,
  all styled from the dashboard's own design tokens;
* a **read-only** WebSocket tap that records `{timestep, winner}` for each **distinct** engine
  timestep the browser rendered. It sends nothing and mutates nothing.

Everything scientifically relevant stays visible — the top bar (Status / Timestep / Speed /
Winner / FPS / connection), the input sheet, pattern buttons, the raster legend and firing-rate
gutter, receptive-field values, and the right-hand neuron inspector.

The chapter chip is centred under the top bar specifically so it does not cover the raster's
pinned left gutter or the left sidebar's transport controls. The caption bar overlays the
bottom of the page; in the raster views it covers the last few inhibitory lanes, never the
`RG` / `L1E` lanes the cadence claim rests on.

### Display filters — full connectivity

**"Hide weak synapses" (`#f-weak`) is turned OFF in every recording.** It is ON by default and
suppresses any *weighted* edge whose magnitude falls below 25 % of the weight cap
(`WEAK = 0.25` in `frontend/renderer.js`), which progressively erases the losing competitors'
feedforward afferents as one owner consolidates. With it off, the complete graph is drawn — in
`rg_coincidence` that is all 72 `L1E→L2E` feedforward edges at every stage, not just the 66 that
stay above the threshold after 1 500 boundaries.

`Only active neurons` and `Isolate winning assembly` are also forced OFF; `Show RG layer`,
`Show L1 + residual`, `Show Layer 2` and `Show inhibitory` are forced ON.
`Recorder.setSceneFilters` clicks the real controls (so the renderer's own change handler runs)
and asserts the resulting checked state.

Weightless structural edges — the 72 apical gates and the 17 relay / 17 hard-reset edges — were
never affected by the weak filter (`renderer.js` exempts `weight == null` edges); they were
simply hard to read inside the full graph, which is what video 1's anatomy walk addresses.

### Left sidebar and the inspector panel

Two standing display requirements, held for the whole of every recording:

* **The left sidebar is pinned to the top.** Playwright scrolls a control into view before
  clicking it, so toggling the display filters (and the anatomy walk's layer toggles) pushes
  the input pattern grid off screen. `Recorder.scrollSidebarTop` resets `#sidebar-left` after
  every such interaction and after every topology change, so the pixel grid and the pattern
  buttons — what is actually being presented — stay visible throughout.
* **A neuron is always selected**, so the right-hand inspector reports live threshold, charge,
  spike state, firing frequency and synaptic weights instead of showing its "Select a neuron in
  the viewport" placeholder. There is no DOM control for this: `NeuronRenderer._handleClick`
  raycasts from the pointer onto the visible meshes, so the only way in is a pointer event on
  the canvas over a neuron body. `Recorder.captureSelectionSpots` screenshots the scene and
  ranks candidate points by `luminance × saturation` — neuron bodies are emissive and strongly
  coloured, while the in-scene legend is near-neutral grey and would otherwise supply hundreds
  of unclickable points — spreading accepted points 22 px apart so one bright blob cannot
  consume the whole probe budget. `Recorder.selectNeuron` then dispatches a
  `pointerdown`/`pointerup` pair per candidate and reads the resulting `.insp-id`, walking a
  priority list of id prefixes under a click/time budget.

  Selected in this run: **L2E1** (video 1), **ccE3** (videos 2 and 3), **L2c00E2** then
  **L3c00E1** (video 4, re-selected after the chapter-B topology change).

### Ordering: all setup happens behind the title card

Per video the driver (1) screenshots the scene while nothing covers it, (2) raises that video's
title card, (3) sets the display filters, (4) probes for and selects a neuron, then (5) pads to
at least the 3 s title hold before handing off to the shot plan. The screenshot is deliberately
split from the clicking for exactly this reason: the screenshot needs an unobstructed canvas,
but the clicks derived from it can be dispatched later from *behind* the card (the overlay is
`pointer-events: none`, so they still reach the canvas). Filter toggling and click probing are
therefore never visible; each recording opens on the dashboard for a beat, fades to the title
card, and reveals a fully prepared view.

Each probe is one `page.evaluate` that dispatches the pointer pair and reads the inspector id in
the same round trip. Doing it with `page.mouse.*` costs four round trips and held the title card
for ~16 s on the coincidence graph; the single-evaluate form exercises the identical handler
path (`_handleClick` reads only `clientX`/`clientY`, and `inspector.select()` renders
synchronously) at roughly a quarter of the latency.

None of this touches simulation behaviour: no engine parameter, no control, no view state.

**The Topology Editor is not opened in any video.** It rendered the `NetworkSpec` structure
rather than live learned state, and its dense label layout added nothing the live 3D view and
the caption counts do not already carry. Node and edge counts are read from `GET /api/state`.

---

## 4. Engine configuration per video

Every video runs on the ordinary dashboard server, so every engine carries the two
`DASHBOARD_OVERRIDES` construction values that `POST /api/config` cannot reach:
`e_weight_cap_frac = 0.5` (the θ/2 pattern-detector ceiling) and `cc_e_count = 8`.
`relay_weight_cap_frac = 1.0`, `dual_fe_e = dual_fe_wte = 0.001` and `dual_fe_B = 5.0` are
engine defaults and are not overridden.

### Videos 1 and 4 — dashboard fast-maturation rates

```text
dual_fe_fes       True
eta               4.0
c_eta             16.0
leak_rate         0.0
refractory_steps  0
input_period      0        AUTO -> one volley per resolved causal chain
e_weight_cap_frac 0.5      (construction override)
topology          rg_coincidence (video 1) | tiled_cc then two_tower_composition (video 4)
```

### Videos 2 and 3 — the published `rg_direct_cc4` confirmation candidate

```text
topology          rg_direct_cc4
dual_fe_fes       True
dual_fe_B         5.0      B = 5   (engine default)
eta               1.0      = 0.01  * m, m = 100
c_eta             0.5      = 0.005 * m, m = 100
leak_rate         0.0
refractory_steps  0
input_period      0        AUTO; resolves to 1 on this graph (no feedback loop)
e_weight_cap_frac 0.5      (construction override)
```

`B = 5`, `m = 100` is the confirmation candidate selected by the declared rule in
`experiments/dual_fe_cc4_consolidation.py` (`select_confirmation_candidate`: smallest passing
`m`, then `B` closest to 5) and pinned by
`tests/test_dual_fe_cc4_experiment.py::test_candidate_b5_m100_assigns_four_distinct_owners_one_to_one`.
Both values were verified against the checkout before use.

#### Difference from the published headless experiment, and why it does not matter

`experiments/dual_fe_cc4_consolidation.py::build_engine` leaves `e_weight_cap_frac` at the
engine default `None` (uncapped). The dashboard applies `0.5`. The ceiling **does** bind —
the long-dwell incumbent's peak weight is 668.4 uncapped vs exactly 500.0 (= θ/2) capped —
but it changes no reported result. Measured both ways
(`preflight_demo_measurements.py --compare-uncapped`, stored under
`preflight_measurements.json.uncapped_comparison`):

| | capped (recorded) | uncapped (published experiment) |
|---|---|---|
| four-pattern owners | `ccE0, ccE1, ccE3, ccE2` | `ccE0, ccE1, ccE3, ccE2` |
| cold recall | all four reproduced | all four reproduced |
| long-dwell incumbent → new owner | `ccE0 → ccE1` | `ccE0 → ccE1` |
| turnover latency | eligible presentation 2 | eligible presentation 2 |

Pinned by `tests/test_demo_preflight.py::test_theta_over_two_ceiling_does_not_change_the_cc4_scientific_result`.

---

## 5. Per-video protocol, measurements and shot list

### Video 1 — `uncertainty_frequency_demo` (97.7 s, 2 932 frames)

**Topology** `rg_coincidence` — 45 nodes / 196 edges (live), resolved input period 1.
**Stimulus** `row 1` held sustained for the whole run (active pixels 3, 4, 5 → `RG3/4/5`,
`L1E3/4/5`). Single reset at the start; nothing else is changed. **Inspected cell: `L2E1`.**

**Live graph composition** (read from `GET /api/state`, shown in the opening caption):

| Population / edge kind | Count |
|---|---|
| RG sources / L1E / L1C / L1I | 9 / 9 / 9 / 9 |
| **L2E ordinary competitors** | **8** |
| L2I (single shared WTA relay) | 1 |
| `pretrained_excitation` RG→L1E | 9 |
| `feedforward` L1E→L2E (learned) | **72** = 9 × 8 |
| `basal_excitation` L1E→L1C (learned) | 9 |
| `apical_excitation` L2E→L1C (unweighted Boolean gate) | **72** = 8 × 9 |
| `relay_excitation` | 17 |
| `hard_reset_inhibition` | 17 |

**Anatomy walk.** Because both the 8-way WTA and the apical gate fan are hard to read inside
the full 196-edge graph, video 1 peels the circuit apart with the dashboard's own layer
filters, holding each state long enough to count:

1. **Full graph, weak synapses shown** — every edge kind captioned with its live count.
2. **Layer 2 alone** (`Show RG layer` and `Show L1 + residual` unchecked) — the **8** competing
   ordinary-E cells radiating from the **one** shared L2I relay. Caption states that the WTA is
   emergent rather than a policy: the first L2E to threshold fires, its relay drives L2I, and
   L2I hard-resets all 8 in the same boundary; every competitor is present and eligible on every
   boundary, only the winner survives it.
3. **The apical gates** (`Show L1` back on, `Show inhibitory` off) — the pink fan of
   **72 = 8 L2E × 9 L1C** unweighted Boolean permissions. Caption states that these carry no
   charge and never learn; they only grant permission, and a L1C deposits its learned basal
   charge exactly when basal evidence and top-down permission coincide.
4. **Full circuit restored** — with the explicit note that this preset has **no per-feature
   gating** (that was a separate, removed variant); the gates here are the 72 apical permissions
   plus the 17 hard-reset inhibitory edges.

**Measured cadence (headless preflight, seed 1, 2 400 boundaries):**

| Quantity | Value |
|---|---|
| L1E emission per RG volley, first 400 boundaries | **0.795** (per cell: L1E3 0.790, L1E4 0.800, L1E5 0.795) |
| L1E emission per RG volley, last 400 boundaries | **0.500** (0.500 / 0.500 / 0.500) |
| training-inclusive aggregate L1E/RG over 2 400 boundaries | 0.5906 |
| cadence-lock boundary | **915** |
| early sample (`L1E3`, boundaries 1–40) | `0111111110111111110111111110111111110111` |
| late sample (`L1E3`, last 40) | `1010101010101010101010101010101010101010` |

**Cadence-lock criterion (declared):** the first boundary from which *every* active `L1E`
alternates fire/silent with no repeat through the end of the run.

The 0.500 late rate reproduces the mature-window figure recorded in `README.md` for this
preset; the dashboard's fast-maturation rates reach it at ~915 boundaries instead of ~2 250.

**Shot list.** Title card (~8 s, covering setup) → reset 45-node network, full connectivity,
all edge counts (4.3 s) → **anatomy walk**, steps 1–4 above (~20 s) → `row 1` applied, 220
boundaries at 120 steps/s → open **Spike Raster**, caption the early grouped cadence → run to
boundary 1 500 (the full live history window) so one raster frame holds early *and* late cadence
→ pause → evidence caption + uncertainty qualification (6.5 s) → **zoom the still raster ×4**,
showing RG solid gold every boundary against L1E teal every *other* boundary (5 s) → zoom out,
close raster → **Receptive Fields** (the weight view, 4.2 s) → back on the live paused network
(5.2 s) → closing summary (3 s).

### Video 2 — `continuous_learning_demo` (58.6 s, 1 757 frames)

**Topology** `rg_direct_cc4` — 14 nodes / 44 edges (live). **Inspected cell: `ccE3`.**
**Order** `row 1` → `col 1` → `diag \` → `diag /`, **800 boundaries each** at 120 steps/s.
**One reset**, at the very beginning. Between phases: no reset, no reseed, no rebuild, no
config change — only `POST /api/pattern`.

**Owners observed in this recording** (modal winner over the dynamic frames the browser
rendered during each phase): `row 1 → ccE0`, `col 1 → ccE1`, `diag \ → ccE3`,
`diag / → ccE2` — 4 distinct owners across 4 patterns, identical to the headless preflight
(`matches_preflight: true`), whose cold recall reproduces all four assignments without a reset.
The single stale-owner frame at each switch is the incumbent's one carry-over presentation,
discussed under video 3.

Fewer eligible presentations than boundaries simply means the remaining boundaries had **no**
competitor fire at all; those are not counted as presentations.

**Shot list.** Title card (covering setup) → 14-node network with counts and the B/m/η/C-η line
(2.5 s) → phases 1 and 2 on the 3D view, each captioned with the phase number, dwell, and the
already-owned assignments so far → at phase 3 switch to **Receptive Fields** so a *new*
competitor's weights can be seen forming while the earlier owners' weights visibly persist →
phase 4 in the same view → paused summary listing all four measured pattern→owner assignments,
the distinct-owner count, and the preflight cross-check (6.5 s).

### Video 3 — `unbiased_learning_demo` (79.0 s, 2 369 frames)

Same topology and configuration as video 2. **Inspected cell: `ccE3`.**

**Protocol.** Reset to seed 1 → establish the four-pattern network (`row 1`, `col 1`,
`diag \`, `diag /`, 800 boundaries each) → hold **`row 1`** for **6 014 boundaries**
(timestep 3 200 → 9 214) → switch to **`col 1`** with no reset or rebuild → 809 boundaries.
Input sources stay equal-frequency throughout: no class reweighting, no balanced sampling, no
pattern-specific learning rate. Nothing but `POST /api/pattern` changes at the switch.

**Measured in this recording:**

| Quantity | Value |
|---|---|
| trained owners before the dwell | `row 1 → ccE0`, `col 1 → ccE1`, `diag \ → ccE3`, `diag / → ccE2` |
| long dwell | 6 014 boundaries on `row 1` (7.5× any other pattern's exposure) |
| incumbent after the dwell | **ccE0** |
| new owner after the switch | **ccE1** |
| turnover | eligible presentation **2** of 809 |
| post-switch winner counts | `ccE0 1`, `ccE1 808` |

**Consolidation criterion (declared before measuring):** the turnover presentation is the
first eligible presentation (a boundary on which some competitor fired) whose winner differs
from the long-dwell incumbent **and** which begins an unbroken run of that same winner through
the end of the post-switch phase.

**Was the switch literally first-presentation instantaneous? No.** The over-exposed incumbent
wins exactly **one** carry-over presentation, then never again. The honest description is
*"immediate turnover in 2 eligible presentations after a 6 000-boundary dwell"* — the word
"instant" is not used anywhere in the video. No learning rate or neural rule was changed to
make the transition look faster.

**Independent cross-checks at seed 1**, both reproducing `ccE0 → ccE1`:

* the same four-pattern-then-dwell protocol run headless → turnover at eligible presentation 2;
* the published `long_dwell_stress` protocol in `experiments/dual_fe_cc4_consolidation.py`
  (fresh engine, 6 000 boundaries on `row 1`, then `col 1`) → `ccE0 → ccE1`,
  `turnover_after_long_dwell: true`, `incumbent_locked: false`, matching the stored artifact
  `experiments/runs/dual_fe_cc4/20260723-135744-.../long_dwell_stress_candidate.json`.

**Compressing the uninformative middle.** Watching 6 000 identical frames is pointless, so the
middle is fast-forwarded: the runner is paused and boundaries are advanced with `POST
/api/step` (~330–450 boundaries/s), while a top-right badge reads
`dwell "row 1" ▸▸ 7,6xx / 8,7xx boundaries` and the top-bar **Timestep** readout climbs. The
boundaries genuinely execute — nothing is skipped or simulated. Shown at 1×: the first 200
boundaries of training phase 1, the first 450 and last 450 boundaries of the long dwell, and
**all 809** post-switch boundaries (the part that carries the claim).

**Shot list.** Title card (covering setup) → config caption stating equal-frequency inputs
(2.5 s) → step 1, four-pattern training with per-phase badges, fast-forwarded → "network
established" caption → step 2, long dwell: 450 boundaries at 1×, fast-forward badge over
~5 100, 450 boundaries at 1× → incumbent caption (3.2 s) → step 3, switch to `col 1`, 809
boundaries at 1× → turnover measurement caption with criterion, the explicit "Not
instantaneous" statement, and both cross-checks (7 s) → paused **Receptive Fields** summary:
`ccE0` still holding its `row 1` weights, `ccE1` now owning `col 1` (7.5 s).

### Video 4 — `scaling_demo` (187.5 s, 5 624 frames) — two chapters

**Chapter A — `tiled_cc`.** Live counts **191 nodes / 1 052 edges**, 9×9 RGC sheet tiled into
nine 3×3 patches, column layers `L1 3×3 · L2 1×1` (9 L1 columns + 1 L2 column).
**Inspected cell: `L2c00E2`.** Four patches are driven independently — (0,0) `row 1`,
(1,1) `diag \`, (2,2) `col 1`, (0,2) `diag /` — each applied one at a time with its own caption,
then 260 boundaries at 25 steps/s, then a held full-graph beat with another 160 boundaries.

**Chapter B — `two_tower_composition`.** Live counts **393 nodes / 2 162 edges**, 9×18 input
surface, column layers `L1 3×6 · L2 1×2 · L3 1×1` — **18 L1 columns (nine per tower), 2 L2
columns, 1 L3 composition column**. **Inspected cell: `L3c00E1`** (re-selected after the
topology change, before the chapter card). The whole-sheet glyphs `V`, `A`, `7` are driven
through `POST /api/pattern` (they cross the tower seam, so no per-patch pattern can express
them), 78 boundaries each at 6 steps/s, then a held full-graph beat.

Growth across the chapters: 191 → 393 nodes, 1 052 → 2 162 edges, same column motif, same
generic child→parent connector.

**Measured glyph owners (headless preflight, 600 boundaries per glyph, seed 1):**

| Glyph | Tower-L2 top ordinary-E owner | L3 top ordinary-E owner |
|---|---|---|
| `V` | `T0L2c00E6` | `L3c00E1` |
| `A` | `T1L2c00E3` | `L3c00E1` |
| `7` | `T1L2c00E5` | `L3c00E1` |
| | **3 distinct** | **1 distinct** |

This reproduces the documented result in `docs/TWO_TOWER_COMPOSITION.md`: the towers separate
the glyphs internally, the single L3 column does not.

**Shot list.** Title card (covering setup) → chapter A: live `tiled_cc` counts and tiling
(2.5 s), four patch assignments captioned one by one, 260 boundaries live, full-graph beat
(2.6 s) → chapter B card (3 s) → live `two_tower_composition` counts, column layers and the
growth from chapter A (3.3 s) → `V`, `A`, `7` driven in turn, each captioned with its measured
tower-L2 owner → full two-tower graph beat (3.2 s) → closing scope statement (8 s).

---

## 6. Scientific qualifications

### Frequency / uncertainty (video 1)

> The circuit exposes a frequency/cadence signal associated with confirmed bottom-up and
> top-down agreement. It is an observable substrate for uncertainty — **not** a calibrated
> confidence score, **not** a probability. Cadence also depends on causal-loop latency and
> input pacing.

This wording appears on screen in the evidence caption and again in the closing summary. It
follows `docs/SUMMER_2026_CIPP_ACCOMPLISHMENTS.md` ("observable substrate ... not calibrated
uncertainty"), `docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md` (the cadence is set by loop latency
and input pacing — `period = 2 × latency`), and
`docs/STANDING_PROBLEMS_AND_HANDOFF_PRIORITIES.md` (frequency halving is **not yet validated**
as a certainty signal).

The claim rests on the **Spike Raster**, which shows discrete spikes and carries a per-neuron
firing-rate bar in its pinned left gutter. The Charge / time view is named on screen as
membrane charge / threshold crossings / inhibition and is explicitly **not** presented as a
frequency estimator. Learned weights are shown in the Receptive Fields view.

**On "feature gates".** `rg_coincidence` has no per-feature gating mechanism. A dedicated
per-feature gated tiled variant (`tiled_cc_feature_gated`) was implemented and then **removed**
— see "Rejected direction: per-feature gated tiled columns" in
`Current_Implementation_Methodology_Equations.md`; git history is the archive. The gating that
does exist in this preset, and which the anatomy walk shows explicitly, is (a) the 72 unweighted
Boolean **apical permissions** `L2E → L1C` and (b) the 17 **hard-reset inhibitory** edges. The
caption says this in as many words so the video cannot be read as claiming feature gating.

### Continuous learning (video 2)

"Continuous" means the same live network and the same learned state persist across pattern
changes. The recording performs exactly one reset, at the start; between phases only
`POST /api/pattern` is called. Owner ids are read from the recorded run, not hard-coded.

### Exposure-bias resistance (video 3)

The word "instant" is not used. The measured latency — turnover on eligible presentation 2 of
809, with one incumbent carry-over presentation — is stated on screen together with the
declared criterion. Inputs remain equal-frequency; no rebalancing, reweighting, or
pattern-specific rate exists anywhere in the workflow.

### Two-tower L3 (video 4)

> Scope: this demonstrates graph and lower-encoder scaling. It does **not** claim the L3
> composition column distinguishes V / A / 7 — the two scalar Eor streams make the tested L3
> inputs non-identifiable.

Shown on screen alongside the measurement that produces it (3 distinct tower-L2 owners, 1
shared L3 owner). This is the preserved negative result `representation_not_identifiable` from
`docs/TWO_TOWER_COMPOSITION.md`, not a bug in the demo.

---

## 7. Verification performed

For each of the four MP4s (`recording_manifest.json.videos[].checks`, all `true`):

* WebM non-empty and playable; the raw Playwright recordings are preserved untouched in
  `presentation_assets/raw_webm/` before conversion;
* `ffprobe` asserts codec `h264`, pixel format `yuv420p`, width `1920`, height `1080`,
  frame rate exactly `30`, duration positive and within the planned chapter length;
* decode check: `ffprobe -count_packets` returns > 0 packets (2 932 / 1 757 / 2 369 / 5 624);
* thumbnail is a valid 1920×1080 PNG;
* beginning, representative middle and end frames were extracted and visually inspected for
  blank pages, browser error screens, dialogs, clipped overlays, cursor artifacts and
  misleading labels — none found. Specifically confirmed: the opening is the title card (not
  filter toggling or click probing), the left sidebar shows the input pattern grid throughout,
  the right-hand inspector is populated throughout, the anatomy-walk frames show 8 L2E
  competitors plus 1 L2I and the 72-edge apical fan, and no Topology Editor appears in any video;
* the scientific state predicates recorded during each run were compared against what the
  captions claim (video 1's edge-kind counts equal the live topology; video 2
  `matches_preflight: true`; video 3's recorded turnover presentation 2 equals the preflight's;
  video 4's live node/edge counts equal the asserted ones);
* the four MP4 paths are distinct and no combined montage exists
  (`independent_outputs: true`).

Thumbnails are taken at a timestamp that carries the central claim, never a title card or a
blank transition: video 1 at 50.82 s (the zoomed raster: RG every boundary, L1E every other),
video 2 at 56.81 s (four distinct receptive fields + the assignment summary), video 3 at
71.07 s (`ccE0` holding `row 1` beside `ccE1` owning `col 1`), video 4 at 181.84 s (the
two-tower graph with the scope statement).

### Tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_dual_fe_cc4_experiment.py \
  tests/test_coincidence_experiment.py \
  tests/test_two_tower_composition.py \
  tests/test_tiled_cc_dashboard_contract.py
# 78 passed

.venv/bin/python -m pytest tests/test_demo_preflight.py -q
# 12 passed  (recording-support coverage)

.venv/bin/python -m pytest tests/ -q --ignore=tests/nest
# 665 passed, 2 warnings (pre-existing FastAPI on_event deprecations)
```

All four suites named in the specification exist under their given names; no substitution was
needed. `tests/nest/` was excluded because it requires the separate NEST environment and is
unrelated to this work (that subtree also carries pre-existing uncommitted changes, preserved).

`tests/test_demo_preflight.py` pins the recording-support contract: the recorded configuration
really is the published `B=5, m=100` candidate, the measurement functions return the fields the
captions consume, the measurements are deterministic at seed 1, the θ/2 ceiling does not change
the CC4 result, and the two-tower measurement still reports the documented **negative** L3
result.

---

## 8. Limitations, compromises and dependencies

* **Rendering throughput, not simulation speed, sets the pace of video 4.** A `two_tower_composition`
  dynamic frame is ~72 KiB and Chromium sustains only ~5–6 frames/s of it (measured; `tiled_cc`
  ~27/s, `rg_direct_cc4` ~400/s). The server itself manages ~190 frames/s on the two-tower graph
  (step 373/s, serialize 399/s). Requesting the 120 steps/s cap there does not run the model
  faster — it queues frames the viewer sees as a stutter followed by a long catch-up. The
  recorder therefore paces the run loop per graph (`SPEED_TILED = 25`, `SPEED_TWO_TOWER = 6`)
  so every rendered frame is real time. This is a **presentation** compromise only; it changes
  no engine behaviour. A first attempt at 120 steps/s produced a 424 s video of mostly stalled
  frames.
* **The title card holds ~8–10 s rather than 3 s**, because all setup (display filters, neuron
  probing) happens behind it. That is deliberate: the alternative is showing the setup.
* **Neuron selection is best-effort under a budget.** The probe walks a priority list of id
  prefixes but settles for the best neuron found within 8 s. On the two-tower graph each probe
  is slow, so the selected cell varies between runs (this run reached `L3c00E1`, an earlier one
  stopped at a tower-L1 cell). Any populated inspector satisfies the requirement, so this is
  not treated as a failure.
* **Turning off the weak-synapse filter costs a little legibility in the dense views.** With all
  196 edges drawn, `rg_coincidence`'s full graph is visually busy — which is precisely why the
  anatomy walk isolates Layer 2 and the apical fan rather than relying on the full view alone.
  Edge opacity still scales with learned weight (`0.04 + 0.32 × magnitude`), so weak edges are
  drawn faint rather than hidden; structural edges sit at a fixed 0.22.
* **Boundary counts differ slightly from the nominal dwell.** `runVisible` polls `GET /api/state`
  at 100 ms and pauses once the target is reached, so ~6–14 extra boundaries execute at 120
  steps/s (e.g. 809 rather than 800 post-switch boundaries). Captions quote the **actual**
  recorded count.
* **Rendered frames vs. boundaries.** The WebSocket tap counts one entry per distinct engine
  timestep and only counts a presentation as *eligible* when some competitor fired. Boundaries
  with no winner are correctly excluded. An earlier version counted the duplicate dynamic frames
  the server broadcasts on every control call, which inflated the post-switch tally and put the
  apparent turnover at presentation 4; the timestep dedupe fixed this and the recorded latency
  now matches the headless preflight exactly.
* **Video 4's camera framing** is the dashboard's own fit-to-topology camera; it is not
  hand-framed. Counts come from `GET /api/state`, not from reading the picture.
* **`cadence_lock_boundary = 915`** is measured over a 2 400-boundary window. It is the first
  boundary from which alternation holds *to the end of that window*; it is not a proof of
  permanence beyond it.
* **Video 4's glyph phases are short** (78 boundaries each) — enough to show the sheet driving
  the towers, not enough to train them. The glyph→owner assignments quoted on screen come from
  the 600-boundary headless preflight and are labelled as such.
* **Determinism caveats.** The simulation is bit-deterministic at seed 1 and every scientific
  number reproduces exactly. Video *durations* are not bit-reproducible: they depend on browser
  render throughput and HTTP round-trip times on the recording machine.
* **Dependencies.** System FFmpeg/FFprobe 6.1.1 were used (no `ffmpeg-static` needed). Playwright
  1.62.1 and its Chromium 151.0.7922.34 were installed locally under `node_modules/` and
  `~/.cache/ms-playwright/`; `node_modules/` and `presentation_assets/.playwright_video/` were
  added to `.gitignore`.
* **Nothing was committed, pushed, published, or uploaded.** The historical singular
  `simulation_demo.*` naming is superseded and no such file was created.
* **Neural behaviour was not modified.** `backend/simulation.py`, `snn/neurons.py`, the learning
  equations, thresholds, topology builders and scheduler semantics are untouched. The only
  repository changes are the new recording workflow, its test, and two `.gitignore` lines.

---

## 9. Generated files

| File | Size |
|---|---|
| `presentation_assets/uncertainty_frequency_demo.mp4` | 11.69 MB |
| `presentation_assets/uncertainty_frequency_demo.webm` | 9.36 MB |
| `presentation_assets/uncertainty_frequency_demo_thumbnail.png` | 1.27 MB |
| `presentation_assets/continuous_learning_demo.mp4` | 5.39 MB |
| `presentation_assets/continuous_learning_demo.webm` | 5.84 MB |
| `presentation_assets/continuous_learning_demo_thumbnail.png` | 0.62 MB |
| `presentation_assets/unbiased_learning_demo.mp4` | 7.86 MB |
| `presentation_assets/unbiased_learning_demo.webm` | 7.27 MB |
| `presentation_assets/unbiased_learning_demo_thumbnail.png` | 0.49 MB |
| `presentation_assets/scaling_demo.mp4` | 21.55 MB |
| `presentation_assets/scaling_demo.webm` | 20.87 MB |
| `presentation_assets/scaling_demo_thumbnail.png` | 1.13 MB |
| `presentation_assets/raw_webm/*.webm` (4 preserved raw recordings) | 42 MB total |
| `presentation_assets/preflight_measurements.json` | 18.3 KB |
| `presentation_assets/recording_manifest.json` | 9.6 KB |
| `presentation_assets/recording_notes.md` | this file |
| `presentation_assets/recording_server.log` | 0 B (clean run) |

Source added:

| File | Purpose |
|---|---|
| `package.json` | `npm run record-demo`, `npm run preflight`, Playwright dev dependency |
| `scripts/record_simulation_demos.mjs` | the four shot plans, overlay chrome, display filters, neuron selection, orchestration |
| `scripts/demo_support.mjs` | server lifecycle, seed-file contract, REST client, bright-spot picker, FFmpeg/FFprobe |
| `scripts/preflight_demo_measurements.py` | deterministic headless measurement of every quoted number |
| `tests/test_demo_preflight.py` | focused coverage for the preflight contract |
| `.gitignore` | + `node_modules/`, `presentation_assets/.playwright_video/` |

All four MP4 files are independent demonstrations. There is no combined montage.
