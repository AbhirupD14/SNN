# Two 9x9 Towers Feeding One L3 Composition Column (V / A / 7)

**Scope.** This report covers **Task 4 only** of
`prompts/Claude_Final_Experiments_Prompt.md`. Tasks 1–3 of that prompt (adaptive hold-out
B sweep, interleaving/charge-wipe ablation, fabric constraint matrix) were **not
executed** and nothing here should be read as evidence about them. The omnibus plan in
that file is superseded; Task 4 was re-specified against the **current live contract**
before execution.

**Run.** `experiments/runs/two_tower_composition/20260728-171108-two_tower/`
Seed 1, `B = 5.0` (reference), deterministic, ~102 s of compute, zero non-finite values.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Does the two-tower graph build and validate under the existing rules? | **Yes** — 393 nodes / 2162 edges, composed only from `build_cortical_column` / `connect_rgc_patch` / `connect_columns`. |
| Do the lower towers learn V, A and 7? | **Yes.** All active L1 columns, both L2 columns, every Eor path and every non-top C reach full readiness on all three glyphs, in 2.0k–3.7k boundaries per glyph. |
| Do the two towers *internally* separate the three glyphs? | **Yes, cleanly.** Each tower's L2 assigns a distinct ordinary-E owner per glyph, and cold recall reproduces all six assignments exactly (6/6). |
| Can L3 distinguish V, A and 7? | **No.** All three glyphs collapse onto **one** L3 owner (`L3c00E1`) in training *and* in cold recall (0/3 distinct). |
| Is that a dwell/tuning problem or a structural one? | **Structural.** The complete ordered L3 input trace is **byte-identical** for the three glyphs. There is nothing at L3's input to separate. |

**The principal finding is a preserved negative result: `representation_not_identifiable`.**
The single-Eor column output is a hard information bottleneck. Each tower *knows* which
glyph it is looking at — its L2 ordinary-E bank holds three distinct owners — and the one
Eor relay discards that identity before L3 ever sees it. L3 is handed the same two source
identities, the same event count, the same timing and the same delivered charge for all
three glyphs.

### Architectural interpretation

The current fabric is intended as an edge/feature extractor and compressor, not the final
classifier. On that intended role, the lower-tower result is positive: both encoders learned
and cold-recalled distinct internal representations for V, A, and 7.

The negative result applies to the tested latent interface. Compressing each complete
`9x9` tower to one scalar Eor stream removed every V/A/7 distinction before the shared L3
column. A future divergent semantic decoder could expand an adequately distributed latent
code, but no decoder can distinguish cases whose complete input streams are already
identical. The experiment therefore establishes that **two tower-level Eor streams are not
an adequate complete decoder input for these glyphs**; it does not establish that the
feature extractor failed or that semantic expansion is impossible.

The proposed encoder/decoder interpretation is a hypothesis, not an implemented result.
No decoder, symbol-creation stage, or reconstruction objective was tested. See
`docs/ENCODER_DECODER_ARCHITECTURE_HYPOTHESIS.md`.

---

## 2. Live contract actually used

Task 4 was updated from the prompt's historical reference block to the current dashboard /
engine contract before execution. Every value below is **asserted on the constructed
engine** by `assert_engine_contract()` before a single boundary is stepped, and recorded in
`config.json` and `composition_topology.json`.

```text
topology            two_tower_composition   (custom spec via engine.apply_topology)
cc_e_count          8
dual_fe_fes         True
dual_fe_e / wte     0.001 / 0.001
dual_fe_B           5.0                     (the reference value)
eta                 4.0
c_eta               16.0
leak_rate           0.0
refractory_steps    0
input_period        0                       AUTO -> one volley per resolved causal chain
e_weight_cap_frac   0.5                     theta/2 pattern-detector ceiling
relay_weight_cap_frac 1.0                   theta one-afferent ceiling
eor_w_init_frac     1.0    +  eor_plasticity_enabled False
                                            Eor is a FIXED NON-PLASTIC relay at theta
c_feedback_reset    True                    top-down delay-1 feedback reset enabled
```

Derived and verified at construction: **feedback loop latency = 3**, so the resolved input
period is 3 (one presentation per resolved chain). This is the same loop depth as
`tiled_cc`; adding the L2->L3 stage does not lengthen the shortest child->parent chain.

Two consequences of the updated contract are worth stating plainly, because they change
what the prompt's milestones mean:

* **Eor readiness is structural by construction.** With the bank frozen at `theta`, one
  ordinary-E winner always drives its Eor to exactly threshold. The one-event maturity test
  is therefore trivially satisfied and only the *observed* reliability is informative. It
  was measured anyway, and came out at **1.0** on every path (see §6).
* **The `theta/2` detector ceiling defines the L3 problem.** Every L3 ordinary E owns
  exactly **two** afferents (`T0L2c00Eor`, `T1L2c00Eor`), each capped at `theta/2`. A
  matured L3 detector therefore needs **both** towers to deliver on the same boundary and
  can never be driven by one tower alone. That is a designed property, and the experiment
  confirms it: the matured owner's components are exactly `{500.0, 500.0}` against
  `theta = 1000`.

---

## 3. Topology

`backend.network_spec.two_tower_composition_spec(cc_e_count=8)`, exposed as the
`two_tower_composition` **built-in preset**.

> The experiment was first run through the ordinary custom-topology path with the graph
> deliberately unregistered. It was promoted to a built-in afterwards, on request, so the
> same graph can be driven live from the dashboard. The two construction paths are
> **bit-identical** — same node order, same seeded layout, same initial weights (asserted
> in `test_preset_path_is_bit_identical_to_custom_topology_path`) — and the artifacts were
> regenerated through the preset path to the same `topology_fingerprint`
> `228712668542acf9…` and the same owners, boundaries and verdict. Promotion changed no
> result.

It is the only built-in whose input surface is **not** 81 pixels, so the tiled family's
single-surface assumption was replaced by a per-preset declaration
(`TILED_PRESET_INPUT_SHAPE` / `tiled_preset_input_size`), read by engine size resolution,
the fixed-input guard and the preset store alike.

```text
Tower 0 (left):   9x9 RGC -> 9 L1 classic CC -> T0L2c00 -.
                                                          >-- L3c00
Tower 1 (right):  9x9 RGC -> 9 L1 classic CC -> T1L2c00 -'
```

| | value |
|---|---|
| input sheet | one validated `9x18` surface, row-major `RGC0..RGC161` |
| columns | 18 L1 (global metadata cols 0–2 left, 3–5 right), 2 L2, 1 L3 |
| per column | 8 ordinary E + 1 Eor + 1 C + 1 I |
| nodes / edges | **393 / 2162** (derived from the built graph, then asserted) |
| L2->L3 link | the generic `connect_columns` rule only: `child.L2.Eor -> every L3 E` (feedforward) and `every L3 E -> child.L2.C` (apical) |
| dormancy | both tower C cells are now **non-dormant**; `L3c00C` is the dormant one (`has_parent = false`, 0 apical inputs) |
| fingerprint | `topology_fingerprint` = SHA-256 of the validated `current_spec()` with display `pos` removed; verified identical across seeds |

Structural tests (`tests/test_two_tower_composition.py`) check the *connectivity*, not just
totals: 162 unique RGC pixels with disjoint half ownership, every RGC feeding only its own
`3x3` patch's E bank, every L1 linking only to its own tower's L2, both and only the two L2
columns feeding L3, column-local I resets, apical fan-in equal to exactly the declared
parent, zero apical input on the L3 C, no feature-relay archetype or variant, and
deterministic fresh-object construction. A cross-tower L1->L2 edge is rejected by
`validate_spec`.

Layout and metadata are fully metadata-driven, so the recorded replay renders both towers
and L3 in the existing player with no new renderer.

---

## 4. Glyphs

Three deterministic one-pixel-wide binary glyphs on the `9x18` sheet (`|` marks the seam
between the two `9x9` fields). Coordinates, vectors, per-half counts and ASCII are stored
in `composition_glyphs.json` and asserted before training.

```text
V  (18 px, 9|9, 6 patches)      A  (26 px, 13|13, 8 patches)   7  (26 px, 9|17, 8 patches)
#........|........#            ........#|#........            #########|#########
.#.......|.......#.            .......#.|.#.......            .........|........#
..#......|......#..            ......#..|..#......            .........|........#
...#.....|.....#...            .....#...|...#.....            .........|........#
....#....|....#....            ....#####|#####....            .........|........#
.....#...|...#.....            ...#.....|.....#...            .........|........#
......#..|..#......            ..#......|......#..            .........|........#
.......#.|.#.......            .#.......|.......#.            .........|........#
........#|#........            #........|........#            .........|........#
```

`V`: legs `(r, r)` and `(r, 17-r)`. `A`: legs `(r, 8-r)` and `(r, 9+r)` plus crossbar
`(4, 4..13)`. `7`: full top row `(0, 0..17)` plus the far-right column `(r, 17)`.

V and A deliberately share diagonal structure and differ by the crossbar and the apex
direction. Every active patch of every glyph carries **≥ 3** active pixels, so a
`theta/2`-capped L1 detector can always be driven over threshold by its own patch. Verified
pre-training: in-sheet coordinates, correct half membership, three mutually distinct
vectors, every active pixel in exactly one patch, and ASCII matching the binary vector.

---

## 5. Representation-separability preflight

**Design.** A separate run, never mixed into the composition condition. L3 **learning only**
was disabled on its ordinary E and Eor by metadata; L3 kept firing, kept its feedforward
input, and kept driving the towers' C cells apically. It was never disconnected and its
feedback was never suppressed — only its plastic flags were cleared. The 24 frozen L3
plastic edges were verified byte-unchanged at the end of training
(`l3_frozen_weight_invariant: {n_edges: 24, drifted: 0}`).

The lower towers were then trained on V, A and 7 sequentially to the lower-tower gate
(A–F; `G` is unreachable by construction when L3 is frozen and must not hold the phase
hostage). All three glyphs reached it: **4501 / 1916 / 2470** boundaries.

For each glyph a **fresh identical engine** was built, the snapshot transferred by stable
edge id, all learning frozen, and the glyph presented from a cold dynamic state for 1000
boundaries. Weight invariance was asserted after every probe; no probe engine was reused
across glyphs.

**Result — the L3 input signature is identical for all three glyphs.**

| glyph | L2 owners (left / right) | L3 events | per source | co-occurrence | inter-event interval |
|---|---|---:|---|---|---|
| V | `T0L2c00E6` / `T1L2c00E7` | 332 | 166 / 166 | both, on all 166 boundaries | 6 |
| A | `T0L2c00E4` / `T1L2c00E3` | 332 | 166 / 166 | both, on all 166 boundaries | 6 |
| 7 | `T0L2c00E3` / `T1L2c00E5` | 332 | 166 / 166 | both, on all 166 boundaries | 6 |

Classification over the common 1000-boundary window:

```text
verdict           identical
identifiable      false
source_collision  true      (all glyphs present the same L3 source set)
pairwise          V/A, V/7, A/7 -- exact_trace_match: true, late_summary_match: true
declared L3 sources  ["T0L2c00Eor", "T1L2c00Eor"]
```

Because the **complete ordered** `(relative_boundary, source_L2_Eor_id, delivered_charge)`
streams match exactly — not merely their late summaries — this is not
`transient_only_separability`. It is a full collision at the engine's own resolution.

The inter-event interval of **6** is the frequency-halving cadence: `2 x` the loop latency
of 3, which is exactly what the enabled top-down feedback reset produces on a confirmed
column. It is the same for all three glyphs, so even the timing channel carries no glyph
identity.

Note the second column of that table carefully. **The towers did separate the glyphs.** Six
distinct L2 owners across three glyphs and two towers, all recovered from a cold frozen
state. The information exists one layer below L3 and is destroyed by the single Eor.

---

## 6. Composition training (the bounded negative probe)

Condition: reference `B = 5.0`, no transition wipe, seed 1, L3 learning **enabled**,
30 000 boundaries per glyph, canonical order V -> A -> 7. Training continues to the next
glyph after a timeout; none occurred.

**Every glyph reached the complete composition gate, including L3 maturity.**

| glyph | outcome | boundaries | milestone first reached (absolute boundary) |
|---|---|---:|---|
| V | `full_gate_ready` | 3658 | A/B/C 652, D 1044, E 1861, G 2744, F 3656 |
| A | `full_gate_ready` | 2002 | A/B/C 4516, D 4569, G 4664, E 5647, F 5658 |
| 7 | `full_gate_ready` | 2382 | A/B/C 6664, D 6882, G 7166, E 7483, F 8040 |

Gate items: **A** every active L1 column stably owned; **B** each active L1 owner one-event
mature from its patch's active RGCs; **C** each active L1 owner->Eor path ready; **D** both
L2 columns stably owned and one-event mature; **E** both L2 owner->Eor paths ready; **F**
every active non-top C one-shot ready; **G** L3 stably owned and one-event mature from its
participating L2 Eors. Ownership is the established event criterion (trailing 50 winner
events, dominance ≥ 0.95, three consecutive non-overlapping windows, reassessed every
boundary — no permanent latch). The full conjunction had to hold on 3 consecutive
boundaries.

Measured readiness at phase end:

* **L1** — every active column stably owned and one-event mature; owner->Eor reliability
  **1.0** over 50 eligible delivery boundaries on every path.
* **L2** — both towers stably owned and one-event mature (active charge 1282–1607 against
  `theta = 1000`), Eor reliability **1.0**.
* **C** — every active non-top C at basal weight exactly `1000.0 = theta`, i.e. one-shot
  capable, with hundreds of committed deposits and spikes each. `L3c00C` is reported and
  **excluded by metadata** (`has_parent = false`): 0 opportunities, 0 deposits, basal
  parked at its `theta/4` init.
* **L3** — owner one-event mature with components exactly `{T0L2c00Eor: 500.0,
  T1L2c00Eor: 500.0}` = `theta`. L3 Eor reliability 1.0 (reported, not gated).

**And yet the mapping collapses.**

| glyph | left L2 owner | right L2 owner | L3 owner (train) | L3 owner (cold recall) | recall match |
|---|---|---|---|---|---|
| V | `T0L2c00E6` | `T1L2c00E7` | `L3c00E1` | `L3c00E1` | yes |
| A | `T0L2c00E4` | `T1L2c00E3` | `L3c00E1` | `L3c00E1` | yes |
| 7 | `T0L2c00E3` | `T1L2c00E5` | `L3c00E1` | `L3c00E1` | yes |

`L3_distinct_mapping = false`. The requirement of **three distinct stable L3 owners fails
1 / 3**. Cold recall "matches training" for every glyph, which here is a restatement of the
failure rather than a success: the same neuron is recalled because the same neuron was
trained, for all three glyphs.

**Mechanism, measured not assumed.** `learning_event_counts.csv` shows that across the
whole composition run exactly **one** L3 cell ever learned:

```text
L3c00E1   V: 352 updates   A: 310 updates   7: 295 updates
(the other seven L3 ordinary E: zero updates, zero spikes)
```

L3's eight competitors differ only in seeded initial jitter on their two afferents
(row sums 483.4–515.8 at seed 1, with `L3c00E1` the largest at 515.8). Since the delivered
input is *identical* for all three glyphs, the first-spike latency race has the same winner
every time; the local WTA hard-resets the other seven, and the incumbent's weights then
grow to the `theta/2` ceiling on both afferents. There is no evidence for the winner to be
wrong about, and no signal any other competitor could specialize on.

Per the prompt's declared gate, the seed sweep was **stopped after seed 1** because the
failure is structural, not seed-specific (`seed_sweep_stopped.remaining_seeds = [2, 3, 4]`).

---

## 7. Failure catalog

`failure_catalog.json`, using the prompt's stable taxonomy.

| taxon | first seen | smallest reproducing condition | structural? | narrowest future action |
|---|---|---|---|---|
| `representation_not_identifiable` | L3, `L3c00`, seed 1 | frozen cold probes of V/A/7 on the two-tower graph, `B = 5.0` | **yes** | expose more than one source identity per tower to L3 — **not attempted here** |
| `representation_not_identifiable` | L3, `L3c00`, seed 1 | composition training, `B = 5.0` | **yes** | as above |
| `L2_mapping_collision` | L3, `L3c00`, seed 1 | cold recall, `B = 5.0` | **yes** | as above |

None of these is parameter-sensitive. Lowering `B`, lengthening the dwell, changing the
seed or adding presentations cannot create a distinction that is absent from the input
stream: the three signatures are byte-identical, so any learning rule reading only that
stream is deciding between glyphs on no evidence.

**Not evaluable in this run**

* The prompt's second composition condition ("the selected low-B/no-wipe condition from
  Task 1"). Task 1 was not executed, so `low_B_candidate` is recorded as **`null` with a
  reason** rather than substituted with an undeclared `B`. This is recorded in
  `config.json` and in `aggregate_summary.json -> completion.not_evaluable`.
* Seeds 2–4, deliberately not run under the declared structural-failure gate.
* Any charge-wipe diagnostic. The intervention belongs to Task 2 and was **not** used here;
  the prompt forbids it as an undeclared composition aid, and both required no-wipe
  conditions must be preserved first.

---

## 8. What this does and does not show

**Validated by this experiment**

1. The generic cortical-column rules compose to a three-layer, two-tower hierarchy with no
   new mechanism: `connect_columns` emitted the L2->L3 stage unchanged, and the tiled
   validator, layout and replay recorder all accepted it as-is.
2. Declaring L3 as the towers' parent correctly *activates* both tower C cells and moves
   dormancy up to L3 — confirmed by metadata, by zero apical fan-in on `L3c00C`, and by
   both tower C cells maturing to one-shot.
3. A three-layer fabric under the live contract matures end to end in a few thousand
   boundaries: L1 ownership, `theta/2` detector maturity, fixed-Eor throughput at
   reliability 1.0, L2 ownership, C one-shot readiness, and structural L3 maturity.
4. Both towers hold a genuine, cold-recallable 1-to-1 glyph->owner map at L2.

**The structural constraint this experiment isolates**

5. One Eor per column collapses the whole ordinary-E owner identity into a single output
   channel. At L1->L2 that loss is partly repaired, because L2 sees *which nine children*
   fired and the active-child pattern still differs per glyph. At L2->L3 there is no such
   repair: with two towers there are only two children, both always active for every glyph,
   so the source-identity code has exactly one state. **The composition column cannot
   discriminate what its children refuse to tell it.**
6. Local single-winner WTA at L3 does not create information. It converts an identical
   input into a deterministic seed-decided incumbent.

**Untested here** — whether the collision would persist with more than two children per
composition column (where active-child *subsets* could vary per glyph), and whether any
mechanism outside the current rule set repairs it. Deliberately not attempted: this task
forbids feature gates, direct identity and new neural mechanisms, and the negative result
is more useful preserved than tuned away.

---

## 9. Artifacts and reproduction

Parent run: `experiments/runs/two_tower_composition/20260728-171108-two_tower/`

```text
config.json                  full condition config, engine contract, ownership/readiness
                             criteria, recording policy, git commit + dirty flag,
                             low_B_candidate = null + reason
status.json                  per-phase completion
composition_topology.json    validated spec, counts, fingerprint, engine audit, towers,
                             dormant-C declaration
composition_glyphs.json      sorted coordinates, 162-bit vectors, half counts, patches,
                             ASCII renderings, active L1 columns per glyph
preflight_signatures.json    full ordered L3 input traces + taus + summaries + verdict
composition_results.csv      one row per (condition, seed, glyph)
learning_event_counts.csv    per-cell drained learning-event aggregates per glyph phase
failure_catalog.json         structured taxonomy records
aggregate_summary.json       schema_version / run_id / repository / config / structure /
                             preflight / composition / failures / artifacts / completion
cells/<slug>/                atomic status.json marker + config hash + full result.json
replays/<slug>/              manifest.json, replay.snn.jsonl, metrics.csv, summary.json
```

Replay accounting (recorded, not estimated): stride `record_every = 60` derived from the
planned 90 000-boundary budget and a 1500-frame cap; preflight wrote 152 frames / 11.6 MB
over 8887 actual boundaries, composition 138 frames / 10.4 MB over 8042. Scientific metrics
used **every** engine boundary; replay sampling never fed an analysis. Both replays load
and reconstruct weights through the existing recorder API (393 neurons, 2162 synapses,
markers for glyph start/end, each milestone first-reach, the learning freeze and the
separability verdict).

```bash
# Full run (creates and prints RUN_DIR)
PYTHONPATH=. .venv/bin/python experiments/two_tower_composition.py \
  --phase all --seeds 1-4 --glyph-timeout 30000 --recall-boundaries 1000 \
  --output-root experiments/runs/two_tower_composition

# Later phase into an existing parent run (resume skips only a cell whose atomic
# marker says completed AND whose recorded config hash matches)
PYTHONPATH=. .venv/bin/python experiments/two_tower_composition.py \
  --phase composition --run-dir "$RUN_DIR" --resume --seeds 1-4

# Implementation smoke test -- NEVER a scientific result
PYTHONPATH=. .venv/bin/python experiments/two_tower_composition.py --quick --phase all

# Tests
PYTHONPATH=. .venv/bin/python -m pytest tests/test_two_tower_composition.py
```

Source: `backend/network_spec.py` (`two_tower_composition_spec`, `TILED_PRESET_INPUT_SHAPE`),
`backend/simulation.py` + `backend/presets.py` + `backend/dashboard_config.py` (preset
registration and per-preset input-surface resolution),
`experiments/two_tower_analysis.py` (pure analysis),
`experiments/two_tower_composition.py` (orchestrator),
`tests/test_two_tower_composition.py`, `tests/test_preset_registry_contract.py`.

### Driving it live

Select **"Two-Tower Composition · 9×18 · 18 L1 / 2 L2 / 1 L3"** in the dashboard's Topology
control. Applying it rebuilds the input to 162 pixels and the patch grid to `3x6`.

The pattern buttons then offer **seven** stimuli at two different scopes:

| button | scope | click behaviour |
|---|---|---|
| `row 1`, `col 1`, `diag \`, `diag /` | 3×3 **local** feature | embeds into the *selected* patch and composes with the other 17 |
| `V`, `A`, `7` | **whole-sheet** glyph (dashed border) | drives the entire 9×18 surface and clears any per-patch composition |

The glyphs exist because no per-patch pattern can express a stimulus that crosses the tower
seam — which is the whole point of a composition graph. They are the same vectors the
experiment trains on: `backend.network_spec.SHEET_GLYPH_SHEETS` is the single source of
truth, and `experiments/two_tower_analysis.py` re-exports it rather than re-defining it, so
the buttons and the reported results cannot drift apart.

The bank is keyed on the **input shape**, not the preset name: any tiled graph with a 9×18
surface gets `V/A/7`, and every 9×9 preset reports an empty `sheet_patterns` list and is
untouched. `set_patch_pattern` refuses a glyph with an explanatory error rather than a bare
`KeyError`.

To reproduce the reported operating point, set `eta = 4`, `c_eta = 16`, leak `0`,
refractory `0`, input period `0`, dual FE/FES on and feedback reset on — the dashboard
defaults already match, except that the startup patch assignment is the two-patch
`tiled_cc` demo, which the first glyph click clears anyway.

No production stepping/scheduling rule, neuron equation, learning rule, golden baseline or
existing experiment was modified. The preset registration (added on request after the
experiment concluded) changes which topologies the dashboard offers and replaces the tiled
family's single-81-pixel-surface assumption with a per-preset declaration; it is
bit-identical for every pre-existing preset.
