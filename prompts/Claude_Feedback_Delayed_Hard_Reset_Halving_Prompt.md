# Implementation prompt: delayed feedback hard reset — top-down `C→I` suppression and frequency halving

## Goal

Give the coincidence cell (`C`) in a tiled cortical column a real downstream effect: when a
pattern is confirmed at the parent level, the column's confirmed re-fire is **suppressed on
the next boundary**, dropping the column's output toward the frequency-halving cadence.

Today, once a column's winning ordinary `E`, its `Eor`, and its `C` are trained, all three
fire on **every** boundary and `C`'s spike is inert — its only edge (`C→I`,
`column_c_to_i`) is swallowed by the once-per-boundary relay guard, because the ordinary-E
WTA already fired that same `I` earlier in the boundary. `C` currently changes no membrane,
no weight (other than its own basal), and no downstream state.

The fix routes `C`'s inhibition through the **same** column `I` neuron, but as a
**delay-1, boundary-start** hard reset rather than the mid-loop immediate reset the WTA
uses. A trained ordinary E crosses at `tau≈0`; a same-boundary reset arriving at `C`'s
later `tau` lands on an already-fired cell (`fire()` set `V=v_rest`) and is a no-op. Only a
reset applied at the **start** of the next boundary — after `freeze_drive`, before the
event loop — can prevent the trained instant-integrator from re-crossing.

The timing rule for this change is:

> The WTA `E→I` reset is **zero-latency** (it must pick one winner inside the boundary).
> The feedback `C→I` reset is **delay-1** (a real top-down round-trip): the `I` volley that
> `C` triggers at boundary `t` records a pending hard reset that is applied at the start of
> boundary `t+1`, before any ordinary E in that column freezes and crosses.

## Scope

Keep this a small, additive extension of the event-resolved path. Reuse:

- the existing single per-column `I` (`i_relay`) and its existing `column_c_to_i` and
  `column_i_to_e` edges — **no new nodes or edges are required**;
- the existing `hard_reset(tau, discard_drive=True)` primitive (`snn/neurons.py:407`);
- the existing delay-1 buffer/rotate pattern (`_exc_next`/`_basal_next`) as the template
  for a new `_hardreset_next` queue;
- the existing `_hardreset_out` adjacency (`backend/simulation.py:981`) so the delayed
  reset targets exactly the column's ordinary-E bank the WTA already resets.

Do not add:

- persistent inhibitory conductance (`g_inh`) anywhere on this path — this topology is
  hard-reset only, and `g_inh` must remain identically `0` for tiled columns;
- a second/dedicated feedback interneuron — reuse the one column `I`;
- a same-boundary immediate reset from `C`;
- any change to the WTA `E→I` timing, the once-per-boundary guard for the WTA volley, or
  the scheduler selection policy;
- ID-prefix or layer heuristics to identify the feedback edge.

Do not commit or push unless explicitly asked after review.

## Read before editing

Read:

- `backend/simulation.py`, especially `_event_step` (~1266-1365), `_begin_event_step`
  (~1367), the delay-1 rotate block (~1279-1286), the `freeze_drive` loop (~1320-1321),
  `_fire_event_cell` (~1385-1411), `_drive_event_relays` (~1446-1472), the adjacency build
  in `_build` (`_relayexc_out` ~972/988, `_hardreset_out` ~981/1003), and the role/column
  maps `_role_of`/`_column_of`/`_ordinary_e_ids` (~1065-1067);
- `snn/neurons.py`: `hard_reset` (407), `fire` (393), `crossing_time` (363),
  `begin_event_boundary` (454, "`V` persists across boundaries"), `freeze_drive` (273);
- `backend/network_spec.py`: `build_cortical_column` / `build_column` (593-646), the edge
  kinds `relay_excitation` (165) and `hard_reset_inhibition` (190), and `connect_columns`
  (661-676);
- `tests/test_coincidence_cell.py` and the tiled-cc golden `tests/golden/` fixtures.

Inspect the full working tree and preserve unrelated changes. Record the tiled-cc / event
baseline before editing.

## Root cause (from the event-loop audit)

In a tiled column, `I` is driven by **both** the ordinary E bank (`column_e_to_i`,
`relay_excitation`) and by `C` (`column_c_to_i`, `relay_excitation`), and `I` hard-resets
the ordinary E bank (`column_i_to_e`, `hard_reset_inhibition`). Every relay is resolved
inside the sub-boundary loop by `_drive_event_relays`, which fires a relay **at most once
per boundary** (`backend/simulation.py:1453`: `if relay is None or relay.spiked: continue`).

Trained-regime timeline for one column with input every boundary (`input_period=1`):

| boundary | fires | note |
|---|---|---|
| `t`   | `E_win` | ordinary E wins; `E→I` fires `I` immediately → hard-resets losers (WTA); `E→Eor` scheduled (delay-1) |
| `t+1` | `Eor`   | `Eor→parent.E` scheduled (delay-1); `Eor→C` basal scheduled (delay-1) |
| `t+2` | `parent.E`, `C` | parent fires from `Eor@t+1`; apical→`C` zero-latency; basal from `Eor@t+1` arrives; `C` fires. `C→I` reaches `I`, but `I.spiked` is already `True` (from `E_win@t+2`, which fired first at `tau≈0`) → **`C→I` dropped by the guard** |

`C` fires but produces nothing: the guard eats its trigger, and even if it did not, `I`'s
immediate reset would land at `C`'s late `tau`, after `E_win@t+2` already fired and reset
to `v_rest` (`fire()`, `neurons.py:400`). A same-boundary reset cannot suppress a `tau≈0`
crosser. The column therefore fires on every boundary and never halves.

`g_inh` is not the lever and is not involved: it is identically `0` on this path
(`_sched_inh` is only reachable from the synchronous `step()` behind the
`if self.event_resolved` gate at `1105`; the event path never charges it). Do not
reintroduce it.

## Fix — a delay-1 boundary-start feedback reset from the same `I`

`C→I` becomes a **feedback trigger**: when `C` fires and drives `I`, instead of attempting
the guarded same-boundary reset, record `I`'s `column_i_to_e` targets as a **pending hard
reset** for the next boundary. Apply the pending resets at the start of the next boundary,
after this boundary's drive is frozen and before the event loop, so the trained ordinary-E
bank is wiped (`V=v_rest`, frozen drive discarded) and cannot cross.

The WTA path (`E→I`, immediate, guarded) is untouched. The two uses of the one `I` differ
only in *when* the reset lands: WTA = this boundary at the winner's `tau`; feedback = next
boundary at boundary start.

### Required code changes (`backend/simulation.py`)

1. **Identify feedback triggers at build time.** In `_build`, alongside `_relayexc_out`
   (~988), record which `relay_excitation` edges are the column feedback path. Gate on the
   edge **projection** `column_c_to_i` (the single source of truth emitted by
   `build_column`, `network_spec.py:641`), never on role/layer/id heuristics. Store e.g.
   `self._feedback_trigger_edges: set[str]` (edge ids) or a
   `self._feedback_relay_of: dict[source_id, list[(relay_id, edge_id)]]`. Ordinary
   `column_e_to_i` edges stay in the immediate WTA path only.

   Rationale for projection gating: `rg_coincidence`'s `L1C→L1I` is also a C-driven relay
   but is intentionally **immediate**; it carries no `column_c_to_i` projection and its
   `_role_of` is `None` (no `column_role` metadata). Gating on `column_c_to_i` touches
   `build_column` columns only.

2. **Add the delay-1 pending-reset queue.** Add `self._hardreset_next: dict[target_id,
   list[(relay_id, edge_id)]]= {}` next to `_basal_next` (~589). Snapshot and clear it in
   the delay-1 rotate block of `_event_step` (~1279-1286), exactly like `_basal_next →
   basal_now`:

   ```text
   hardreset_now   = self._hardreset_next
   self._hardreset_next = {}
   ```

3. **Route the `C→I` trigger to the queue.** In `_drive_event_relays` (~1446), when the
   `(rid, re_eid)` for `source_id` is a feedback trigger (step 1), do **not** apply the
   `relay.spiked` guard and do **not** call `hard_reset` now. Instead:
   - record the feedback volley (emit `re_eid`, and optionally set an observable
     `feedback_relay_fired` flag / counter for tests — see idempotency below);
   - append `I`'s `self._hardreset_out.get(rid, [])` entries into `self._hardreset_next`
     for delivery next boundary.

   Keep the existing immediate branch for non-feedback triggers (ordinary `E→I`)
   byte-identical, including its guard and its `hard_reset(tau)` at the winner's `tau`.

4. **Apply the pending resets at boundary start, after freeze.** In `_event_step`,
   immediately **after** the `for n in self.exc.values(): n.freeze_drive()` loop
   (`~1320-1321`) and **before** the crossing-capture diagnostic / event loop (`~1328`),
   drain `hardreset_now`: for each `(tgt, edge_id)` call `self.exc[tgt].hard_reset(0.0)`
   (i.e. `tau=0`, `discard_drive=True`) and append a `hard_reset_events` record tagged as
   feedback (distinct `kind`, e.g. `'feedback_hard_reset'`, with `outer_boundary`, `tgt`,
   `edge_id`, `v_before`, `drive_before`). This runs after `_begin_event_step` (~1277)
   cleared the per-boundary log, so appends belong to the current boundary.

   Placement is load-bearing: it must be **after** `freeze_drive` (so the just-assembled
   `remaining_excitation` packet — delay-1 RGC + external input — is discarded) and
   **before** the loop (so the E cannot cross). With `V=v_rest` and
   `remaining_excitation=0`, `crossing_time` returns `inf` (`v_inf ≤ threshold`), so the
   E is silent that boundary. No ordinary E receives new excitation mid-loop, so once wiped
   it stays silent.

### Idempotency and reporting

- Schedule at most **one** feedback reset per `(relay, boundary)`. A column has one `C` and
  one `I`, so this is naturally once; enforce it (e.g. a per-boundary
  `feedback_scheduled` guard on the relay, cleared in `begin_event_boundary`/`clear`) so a
  future multi-C column cannot double-schedule.
- Applying the same target's hard reset more than once is harmless (idempotent wipe), but
  do not rely on that to hide duplicate scheduling; keep the scheduling side single-shot
  and observable.
- Expose enough for tests: a feedback-volley count/flag, and `hard_reset_events` entries
  whose `kind` distinguishes feedback resets from WTA resets.
- The full column E bank is reset because `_hardreset_out[I]` already lists every
  `column_i_to_e` target (one per ordinary E). Do not reset a subset — a partial wipe could
  promote a loser E to winner. Do not reset `Eor` or `C`.

### Config gate

Add `DEFAULTS['c_feedback_reset'] = True` (validate as bool in `SimulationEngine.__init__`,
add to the config allowlist / serialized config). When `False`, `column_c_to_i` reverts to
the current guarded no-op (behavior byte-identical to today) — this is the A/B control for
the halving test and the rollback path. When `True` (default), the delayed feedback reset
is active.

## Do NOT change

- The WTA `E→I` path: same-boundary, immediate `hard_reset` at the winner's `tau`, still
  guarded to fire `I` once per boundary for the WTA volley.
- `hard_reset` semantics (`V=v_rest`, discard drive; touches no refractory/trace/weights/
  `g_inh`), `fire`, `crossing_time`, `freeze_drive`/`advance_segment` continuous-drive
  meaning, `begin_event_boundary` `V`-persistence, or the `BoundaryEventScheduler` policy.
- Any feedforward/basal/pretrained/apical delay. `Eor→C` basal stays delay-1; `parent→C`
  apical stays zero-latency; `E→Eor`, `Eor→parent` stay delay-1.
- The coincidence deposit gate, one-boundary basal carried eligibility, `settle_eligibility`.
- `g_inh` / persistent inhibition anywhere on the event path — it stays `0`.
- The synchronous `step()` path and the four non-tiled topologies' inhibition.

## Tests

Add/extend Python tests (interpreter is `.venv/bin/python`; there is no bare `python`):

- **Queue/timing unit test.** With `c_feedback_reset=True`, fire a column `C` on boundary
  `t`; assert nothing is hard-reset on `t` by the feedback path, that `_hardreset_next`
  holds the column E-bank targets, and that on `t+1` — after `freeze_drive`, before the
  loop — every ordinary E of that column is at `V=v_rest` with `remaining_excitation=0` and
  a matching `hard_reset_events` entry with the feedback `kind`.
- **Suppression / crossing test.** Give a trained ordinary E a supra-threshold single-
  boundary drive; with a pending feedback reset it must **not** fire that boundary
  (`crossing_time == inf`); on the following boundary with no pending reset it fires
  normally. Prove the reset defeats a `tau≈0` crosser (the whole point).
- **Same-boundary is a no-op (regression guard).** Assert that a `C`-triggered reset applied
  at `C`'s `tau` in the same boundary would not suppress the already-fired winner —
  encode this as the reason the reset is deferred (e.g. by asserting the winner already had
  `spiked=True` / `V==v_rest` before `C` resolves).
- **WTA preserved.** With `c_feedback_reset` both `True` and `False`, the intra-boundary
  winner selection and loser resets are identical for a boundary with no pending feedback;
  exactly one ordinary-E winner per column per boundary.
- **Both-relays-on-one-E is safe.** A boundary where a pending feedback reset lands on the
  column E bank and the WTA `I` also fires: assert no double-count, no `g_inh`, idempotent
  final state.
- **Halving cadence integration test.** On `tiled_cc` (or the smallest tiled topology),
  train to the "fires every boundary" steady state, then measure the confirmed column's
  ordinary-E output rate with `c_feedback_reset=True` vs `False` over the repo's standard
  evaluation window. Assert the confirmed column fires **strictly less often** with feedback
  on, at the cadence set by the feedback round-trip, and that an **unconfirmed** column
  (no parent recognition → no `C` spike → no pending reset) is unchanged. Report the
  measured ratio; do not hard-code exactly `1/2` unless the measured steady state supports
  it — the exact ratio is set by the round-trip latency and the loop re-times as spikes are
  suppressed.
- **`g_inh` stays zero.** Assert `max |g_inh|` over the run is `0.0` for all E/C in the
  tiled topology, with the feature on.

## Goldens & regression

- Non-tiled goldens (`pi`, `old`, `rg`, `rg_residual`, and `rg_coincidence`) contain no
  `column_c_to_i` edge and **must stay bit-exact**. In particular confirm `rg_coincidence`'s
  `L1C→L1I` remains immediate — if it moves, the feedback gating leaked past the
  `column_c_to_i` projection; stop and report.
- With `c_feedback_reset=False`, every tiled-cc golden must stay bit-exact (prove the gate
  is a true no-op when off).
- With the default `c_feedback_reset=True`, regenerate only the tiled-cc captured
  baselines/fixtures (intentional dynamics change), with a one-line note that a confirmed
  column now skips its re-fire one boundary after `C` fires.
- Full suite green: `.venv/bin/python -m pytest tests/ -q`.

## Guardrails

The delay-1 boundary-start reset applies to the `column_c_to_i` feedback trigger **only**.
Do not defer or otherwise alter the WTA `E→I` reset — it must stay same-boundary or the
column loses winner selection. Do not fire, defer, or reset via `g_inh`. Do not reset a
subset of the E bank, and do not reset `Eor`/`C`. Do not let the feedback trigger fire a
second **immediate** `I` volley. If a non-tiled or `c_feedback_reset=False` result regresses
in any way not explained by "a confirmed tiled column now skips its re-fire one boundary
after `C` fires," stop and report rather than adjusting tolerances or goldens to mask it.

## Execution and handoff

Run:

1. the tiled-cc / event baseline before and after;
2. new/updated unit + integration tests;
3. golden regression with the gate both off (bit-exact) and on (re-baselined);
4. the full suite;
5. `git diff --check`.

Report:

- exact changed files;
- the feedback-edge gating signal (`column_c_to_i` projection) and how non-tiled C→I paths
  are excluded;
- the `_hardreset_next` queue lifecycle and the exact insertion point relative to
  `freeze_drive` and the event loop;
- what state the feedback reset wipes (`V`, frozen drive) and what it leaves;
- the measured confirmed-vs-unconfirmed output-rate ratio and the evaluation window;
- proof `g_inh` stays `0` and non-tiled/`off` goldens stay bit-exact;
- focused and full test commands/counts.

Do not commit or push.
