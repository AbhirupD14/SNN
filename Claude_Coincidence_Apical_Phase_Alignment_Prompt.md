# Implementation prompt: bring coincidence (C) cells into phase — zero-latency apical drive

## Goal

Make a coincidence pyramidal cell (`L1C_i`, `CoincidencePyramidalNeuron`) fire in the **same
outer boundary** as the L2E spike that gives it apical permission, instead of one boundary
later. Today L1C trails L2E by exactly one boundary; this is the "one phase off" symptom
seen on the dashboard charge-over-time chart (L2E / L2I / L1E fire in one phase, L1C in the
next). The mature circuit must consequently recover the intended frequency-halving cadence;
do not implement same-boundary reporting without the causal same-`tau` membrane effect.

The timing rule for this change is:

> A synaptic transmission delay determines **when a signal arrives**. Once a C cell has both
> basal availability and apical permission at `(boundary, tau)`, its gated charge is deposited
> into the soma at that same `tau`; it is not spread over the unused part of the boundary.

## Dependency (read first)

**This change only produces a same-`tau` L1C spike once the C-cell one-shot maturity change
has landed** (the linear C rule + `w₁`-based budget headroom that lets a single coincidence
deposit cross θ — see the `c_maturity_budget_frac` work). Phase alignment delivers apical
permission mid-loop and applies an instantaneous gated charge impulse `q = w·s`. With the
model's `C = 1`, the immediate membrane transition is

```text
V(tau+) = V(tau-) + q
```

and the cell can fire at that same `tau` iff `V(tau+) ≥ θ`, its coincidence gate is active,
and refractory/fired-state checks permit it. The old full-boundary condition `w·κ ≥ θ`
does **not** describe a mid-boundary impulse: `κ` belongs to integrating a constant drive
over a full boundary. Existing `w₁`-based maturity headroom may remain as the learning
budget and is conservative for an impulse, but tests must use the actual impulse condition
above.

An immature subthreshold impulse may leave retained membrane charge, but it does not
necessarily fire one boundary later. A C cell can fire only when a new valid coincidence
opens its gate; document and test that behavior instead of asserting an unconditional `+1`
boundary trail.

## Root cause (from the event-loop audit)

In `rg_coincidence`, `L1C_i` is driven by **basal ← L1E_i** (paired) and **apical ← all
L2E_j** (dense). Every feedforward/dendritic hop carries a fixed **1-boundary delay**
(`_emit_event_outputs` schedules to `_basal_next` / `_apical_next`), whereas inhibitory
relays are **zero-latency** (`_drive_event_relays`, same boundary as their driver). The
phase error comes from that apical delay. Once the delay is removed, the existing
full-boundary drive-packet representation is a second issue: it cannot represent an
instantaneous mid-boundary charge deposit and must not be reused for this event. Steady-state
timeline for one pixel:

| boundary | fires | note |
|---|---|---|
| `t`   | RG_i   | input present |
| `t+1` | L1E_i  | pretrained relay; emits basal→L1C for `t+2` |
| `t+2` | L2E_j  | wins WTA; basal now at L1C, but **apical scheduled for `t+3`** → L1C carries basal eligibility |
| `t+3` | L1C_i  | carried basal + (now-arrived) apical → coincidence → deposit → fire |

So **L1C fires at `t+3`, one boundary after L2E (`t+2`)**. The binding delay is the
**apical edge**; the basal delay is absorbed by the one-boundary carried eligibility and is
NOT the lever. L2I fires *with* L2E because it is a zero-latency relay; L1C does not because
its apical permission travels the delay-1 dendritic path.

## Fix — zero-latency apical permission and a same-`tau` C charge impulse

When an apical source (L2E) fires inside the sub-boundary event loop, deliver its apical
event to each target C cell **at the same `tau`** (the way `relay_excitation` already works).
If basal availability is present, atomically open the coincidence gate, consume the basal
eligibility, and apply `q = w·s` as an instantaneous somatic charge impulse.

Do **not** implement the mid-loop deposit with `gather_exc(q)` followed by `freeze_drive()`.
`remaining_excitation` is a constant current/rate integrated by `advance_segment` over
elapsed time. Adding `q` there at a late `tau` applies only a fraction of the intended
charge before the boundary ends and does not make a one-shot crossing immediate.

Add a small shared membrane primitive with explicit impulse semantics, for example
`ConductanceLIFNeuron.apply_charge_impulse(q)`. It must:

- require finite, non-negative `q` for this excitatory use;
- update `V <- V + q` immediately (`C = 1`);
- update `v_pre_reset` to include the post-impulse value;
- leave `pending_exc` and `remaining_excitation` unchanged; and
- accrue membrane charge during refractory while leaving firing prohibited by the existing
  refractory gate.

Use this primitive for gated C deposits only in this change. Do not silently change the
integration semantics of ordinary E/feedforward arrivals or legacy topologies.

**No scheduler policy change should be needed.** After the impulse, the existing scheduler
recomputes crossing times from live state. A mature C cell is now supra-threshold, so its
next local `crossing_time` is exactly `0.0` and it is selected at the current absolute
`tau`. L2E is causally first because its already-processed spike invokes apical delivery;
this is not a node-order fiction. Stable order remains relevant only among other genuinely
same-time candidates.

This does **not** create a two-feedforward-hops-in-one-boundary path: `L1E→L2E` stays
delay-1, so L2E still fires one boundary after L1E; only the single `L2E→L1C` apical hop
becomes a zero-latency append after L2E's spike — structurally identical to `L2E→L2I`.

### Required code changes (`backend/simulation.py`)

1. **Remove apical from the delay-1 path.** In `_emit_event_outputs` (~line 1226-1228),
   delete the block that schedules `_apical_next`. Apical is no longer a delay-1 output.
   Remove the now-dead `_apical_next` rotation/delivery in `_event_step` (~line 1088,
   1105-1109) and the buffer field if nothing else uses it. (Basal stays delay-1 —
   unchanged.)

2. **Add a zero-latency apical drive** analogous to `_drive_event_relays`. In
   `_fire_event_cell` (~line 1191), after `_emit_event_outputs(cell.id)` and alongside
   `_drive_event_relays(cell.id, tau)`, call a new `_drive_event_apical(cell.id, tau)` that,
   for every apical edge out of `cell.id` (use the structured apical adjacency built in
   `_build`, e.g. `self._apical_out`), calls a C-cell method such as
   `c.deliver_apical(source_id, tau)`. That method records the receipt and attempts one
   atomic gate transition. If basal is available, it commits one charge impulse at `tau`.
   It must not call `freeze_drive()`.

   Order within `_fire_event_cell`: emit delay-1 outputs → drive apical (zero-latency) →
   drive relays (zero-latency). Delivering apical before the L2I hard-reset is correct: the
   apical is the Boolean fact that L2E fired, and L1C is not itself a hard-reset target.

3. **Keep the pre-loop dendritic pass** (`_event_step` ~line 1130). With apical no
   longer pre-delivered, that pass now only processes basal (carry/eligibility bookkeeping)
   and never deposits; the actual coincidence deposit happens in the mid-loop apical drive.
   Confirm the basal carried-eligibility state machine still behaves: at the driver boundary
   L1C has current basal, pre-loop resolve sets `basal_eligible=True`; the mid-loop
   coincidence must consume that eligibility (set it False) so it does not also carry to the
   next boundary. Refactor resolution if necessary so both boundary-start delivery and
   mid-loop delivery call one `_try_commit_coincidence(tau)` transition. This transition
   must also preserve processing-order invariance for basal and apical receipts at the same
   `tau`. When their physical arrival times differ, commit at the later arrival
   `max(tau_basal, tau_apical)`, because that is the first instant both requirements are
   active.

4. **Make same-`tau` firing explicit and auditable.** The C impulse is applied inside the
   already-processed L2E spike callback. On the scheduler's next recomputation, a mature C
   cell must return `dtau == 0.0` and fire with `L1C.spike_tau == L2E.spike_tau`. Do not add
   an epsilon, advance the clock, force the C spike flag, or directly bypass the scheduler's
   normal `fire()`/learning/output path.

### Idempotency without hiding routing bugs

Apical permission is a Boolean OR across valid apical sources, so multiple distinct L2E
events in one boundary still authorize at most one C deposit. However, idempotency must not
silently make duplicate routing look correct.

- Add a dedicated `deposit_committed_this_boundary` flag, cleared only by
  `begin_event_boundary`. Do **not** reuse `coincidence_active`: gate truth and transaction
  history are different state.
- The first transition satisfying basal AND apical commits exactly one impulse, records its
  charge and `tau`, consumes basal availability, and leaves `coincidence_active=True` for the
  remainder of that boundary.
- Later distinct apical sources update observable source state but cannot commit again.
- A repeated delivery of the same source/edge must be harmless to membrane state **and
  observable** through a duplicate-delivery counter or diagnostic. Reject duplicate parallel
  apical edges at topology-build time where possible.
- Expose or otherwise test `coincidence_deposit_count` (0 or 1),
  `coincidence_deposit_tau`, raw apical delivery count, unique apical sources, and duplicate
  count. Normal WTA integration tests must assert that unexpected duplicate count is zero.
- Never overwrite the committed `coincidence_charge`/deposit `tau` with zero merely because
  a later receipt did not commit another deposit.

### Do NOT change

- The coincidence **deposit gate** itself (basal AND apical required to deposit `w·s`;
  basal-only / apical-only deposit zero).
- The one-boundary basal carried-eligibility mechanism, `can_fire`, `crossing_time`
  (`inf` unless `coincidence_active`), the continuous-drive meaning of
  `freeze_drive`/`advance_segment`, or the `BoundaryEventScheduler` selection policy.
- Basal edge delay (stays delay-1) and all feedforward/pretrained delays (stay delay-1).
- Ordinary E-cell synaptic integration and the four non-coincidence topology paths. The new
  impulse primitive is used only when a C coincidence gate commits in this change.

## Tests

- Add isolated **impulse primitive tests** at early and late `tau`, with zero and nonzero
  leak. Assert an immediate `V` jump by exactly `q`, unchanged frozen/pending drive, correct
  `v_pre_reset`, and no dependence on the remaining boundary duration.
- Add a **phase-alignment test** on `rg_coincidence`: drive a matured C cell (basal weight
  at the one-shot saturation level) and assert both the same outer boundary and exact causal
  timing: `L2E_j.spike_tau == L1C_i.spike_tau`, with L2E already processed before the C
  delivery callback. Assert the C's committed deposit `tau` equals both spike times.
- Include an **immature counter-case** where `V(tau-) + w·s < θ`: it must not spike on
  that coincidence. Do not assert that it automatically spikes one boundary later; show
  that firing requires a later valid coincidence.
- Add **no-double-deposit tests** separating (a) multiple distinct valid apical sources and
  (b) duplicate delivery of the same source/edge. Both produce exactly one impulse, at most
  one spike and one basal-weight update. The duplicate case must also increment an
  observable diagnostic rather than disappearing silently.
- Add a late-source regression with L2E firing near the end of the boundary. A mature C cell
  must still jump and fire at exactly that late `tau`; this specifically prevents accidental
  reintroduction of `freeze_drive()`/remaining-interval semantics.
- Add or strengthen the mature cadence integration test so the learned C-mediated circuit
  demonstrates the intended frequency halving (output frequency ratio `1/2` over the
  established evaluation window, using the repository's existing tolerance/definition).
  Same-boundary spike flags alone are not sufficient acceptance.
- Update `tests/test_coincidence_cell.py`, `tests/test_coincidence_turnover_sweep.py`, and
  `tests/test_coincidence_protocol.py` where they assert the old `+1` apical timing or
  delay-1 apical scheduling. Update expected `emitted`/spike-history phases accordingly.
  Do not weaken assertions — re-derive the expected boundary indices.

## Goldens & regression

- The four topology goldens `pi`, `old`, `rg`, `rg_residual` contain **no C cells and must
  stay bit-exact**. If any moves, apical logic leaked outside the coincidence path — stop
  and report.
- Regenerate only coincidence-specific captured baselines/fixtures (intentional timing
  change), with a one-line note that L1C is now in phase with its L2E driver.
- **Re-validate the turnover circuit.** This changes C-cell timing, which the event-resolved
  turnover behavior depends on; confirm the column-turnover phase still passes (per the
  `c_eta` one-shot tuning, turnover was 16/16 at `c_eta` 0.005/0.01). If turnover regresses,
  report it — do not silently retune.
- Full suite green: `.venv/bin/python -m pytest tests/ -q` (venv interpreter is `.venv`;
  there is no bare `python`/`pytest` on PATH).

## Guardrails

Zero-latency applies to the **apical permission hop and resulting gated C impulse only**; do
not collapse any feedforward/basal/pretrained delay. Do not fire a C cell before the L2E
that permits it. The causal callback establishes L2E first and the scheduler then observes
the C's zero-time crossing at the same `tau`; do not rely on node order to manufacture that
causality. If a non-turnover coincidence test regresses in a way not explained by "L1C now
fires in the same boundary as its L2E driver," stop and report rather than adjusting
tolerances or goldens to mask it.
