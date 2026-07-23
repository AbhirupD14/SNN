# Basic L1 Consolidation Headless Experiment

`experiments/basic_consolidation.py` asks whether each of the nine 3x3 receptive fields can
consolidate the four canonical local patterns onto a **stable, unique one-to-one mapping**
between patterns and ordinary L1 E competitors, for both current presets:

- `tiled_cc` — eight ordinary competing E per L1 column;
- `tiled_cc_l1_4` — four ordinary competing E per L1 column.

It is a headless scientific experiment (no FastAPI/WebSocket/browser). It records evidence
through the shared replay recorder and **does not modify any network dynamic, learning
equation, threshold, delay, scheduler rule, or topology preset**. Analysis lives in the pure,
independently tested `experiments/consolidation_analysis.py`.

> A replay is an observation artifact; a training timeout is never reinterpreted as
> consolidation; intact and feedback-disabled results are never pooled.

## Two matched conditions (reported separately)

1. **`intact`** — the complete preset unchanged (parent-E apical, C firing, `C -> I`, I hard
   reset). Primary result, because feedback is part of the deployed system.
2. **`feedback_disabled`** — the predictive `C -> I -> ordinary-E` consequence disabled while
   local WTA and all learning are preserved. Causal L1 control.

### How feedback is disabled (and WTA proven intact)

`disable_column_c_to_i(engine)` selects edges purely by `projection == 'column_c_to_i'`
metadata (never id parsing) and removes only those from the engine's relay-dispatch adjacency
(`_relayexc_out`). It touches **no** `column_e_to_i` (E→I) or `column_i_to_e` (I→E) edge, so
local WTA is fully preserved, and it changes no weight/threshold/delay/scheduler rule. The
exact disabled edge ids (nine L1 columns + one L2 column = 10) are recorded in the manifest,
replay header, and summary conditions, so a viewer cannot mistake the route for active.
`intact` runs are byte-identical to an unmodified engine.

`tests/test_basic_consolidation.py::test_feedback_disable_selective` proves: an ordinary-E
winner still drives its local I and resets its own E bank; a C spike causes **zero**
resets in the disabled condition (intact produces C-driven resets); intact is byte-identical;
no cross-column path changes.

## Input construction (all nine patches)

Not `set_pattern()` (which fills one patch). `all_nine_patch_input()` embeds the same 3x3
`PATTERNS` vector into every one of the nine declared patches via `embed_patch_pattern()`,
driven entirely by tiled metadata. `verify_all_nine_input()` asserts, before training: the
vector length matches the input shape; every RGC belongs to exactly one patch; each patch
contains exactly the selected pattern; no cross-patch indexing error; all nine columns are in
the analysis.

## Operational definition of consolidation

A pure, tested rule (`assess_ownership`) over each column's sequence of **ordinary-E winner
events** (one per boundary the column produced an ordinary-E winner; silent boundaries are not
events):

- **ownership window**: the most recent 50 winner events (`--window`);
- **minimum evidence**: all 50 events must exist per window (a silent/rare column cannot pass);
- **owner dominance**: ≥ `0.95` of a window's events belong to one ordinary E (`--dominance`);
- **stable windows**: the same owner satisfies the criterion for 3 consecutive non-overlapping
  windows (`--stable-windows`);
- evaluated on the **trailing** events, so a later turnover un-consolidates a column (no
  permanent latch of the first apparent owner);
- **strict diagnostic** reported separately: whether the owner had `1.00` in the final window;
- **finite per-pattern timeout** (`--train-timeout`), recorded in the manifest.

Uniqueness is **not** part of the phase stopping condition. A stable owner may duplicate a
previous pattern's owner — that is a mapping collision, exposed in the mapping checks (not
trained around).

## Mapping and capacity

After four training phases, per L1 column: all four patterns consolidated; each pattern has
exactly one stable owner; the four owner ids are distinct within the column; and per-preset
capacity — `tiled_cc`: four distinct owners + exactly four unassigned competitors;
`tiled_cc_l1_4`: four distinct owners + zero unassigned. The **run-level Basic gate passes
only when all nine columns pass**; per-column results are always reported so an aggregate
cannot hide a failed field. Neuron indices are arbitrary local labels — never required to
match across columns/seeds/topologies/conditions.

## Frozen recall

Learning is frozen across every plastic object (`freeze_learning`: every ordinary L1 E, Eor,
L2 E, and C basal learner), snapshotted by stable edge id and asserted **exactly unchanged**
through recall.

- **A. Cold-state recall (primary retention).** For each pattern: a fresh engine with the same
  topology/seed/condition/params; the complete trained plastic state transferred **by stable
  edge id** (`transfer_plastic_weights`, exact, rejects id mismatch; no membrane/refractory/
  queue/conductance/winner transfer); learning frozen; only that pattern presented from a
  fresh dynamic state. The recalled owner must match the training owner and be unique across
  patterns.
- **B. Sequential frozen recall (dynamical stress).** One frozen post-training engine, all
  four patterns presented in canonical order without resetting dynamic state. Reported
  separately so carryover cannot contaminate the cold-state result.

Neither probe updates any learned weight (asserted pre/post equal).

## Artifacts

One shared-recorder directory per `(topology, condition, seed)` run:
`manifest.json`, `replay.snn.jsonl`, `metrics.csv`, `summary.json` (see
`docs/REPLAY_RECORDER.md`). `metrics.csv` is long form (run/seed/topology/condition/order,
phase, pattern, timestep, column, neuron, winner-event count, dominance, owner/runner-up,
stable-window count, consolidation, acquisition time, timeout, collision, assigned/unassigned
counts, recall owner-match, owner incoming-weight total + per-afferent weights). Replay markers
include train start, per-column consolidation, timeout, collision, learning freeze, cold
recall, sequential recall, and final result. Batch runs also write `aggregate_summary.json` +
`aggregate_metrics.csv`, keeping intact and feedback-disabled separate.

## CLI

```
PYTHONPATH=. .venv/bin/python experiments/basic_consolidation.py \
  --topology both --condition both --seeds 1-8 \
  --pattern-order row1,col1,diag_back,diag_fwd \
  --window 50 --dominance 0.95 --stable-windows 3 \
  --train-timeout 3000 --recall-timeout 1500 --recall-block 400 \
  --record-stride 20 --checkpoint-every 50 \
  --output-root experiments/runs/basic_consolidation --resume
```

`--summarize-only` (re)writes the aggregate over an existing output root and is safe to run
while runs are still active — incomplete runs are labeled, never scored. `--resume` skips
cells that already have a completed run dir. `--workers` is reserved (runs are currently
sequential; default 1).

### Background invocation

```bash
nohup env PYTHONPATH=. .venv/bin/python experiments/basic_consolidation.py \
  --topology both --condition both --seeds 1-8 --resume \
  --output-root experiments/runs/basic_consolidation \
  > experiments/runs/basic_consolidation/sweep.log 2>&1 &
```

The recorder streams/flushes at a bounded interval and the batch writes each run's artifacts
before moving on, so the sweep is resumable and its partial artifacts stay parseable.

## Deliberately NOT in this experiment

Continuous/interleaved learning, noise invariance, composition, NEST, the dashboard replay
player, and any model fix in response to a failed result — all separate work.
