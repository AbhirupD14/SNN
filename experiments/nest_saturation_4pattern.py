#!/usr/bin/env python
"""Train the NEST prototype to saturation on all four 3x3 patterns, adaptively.

    .nest-env/bin/python experiments/nest_saturation_4pattern.py

The problem this solves
+++++++++++++++++++++++

There is no timestep count to run for. NEST runs in milliseconds and the number of volleys
a cell needs to train is not known in advance -- it depends on the learning rate, the
initial weights and how often the cell happens to win. So the run cannot be scheduled; it
has to be DRIVEN BY CONVERGENCE.

Each pattern is therefore presented in volleys until that pattern is learned, then the next
pattern starts. "Learned" is a declared, measured predicate (see `Convergence`), not a fixed
volley budget, and every phase records how many volleys it actually took.

Topology
++++++++

A 3x6 input sheet tiled into TWO 3x3 patches, so there are two L1 columns feeding one L2
column. Two is the minimum that lets L2 train at all: with a single L1 column each L2
competitor has fan-in 1, its lone afferent always participates, and the +/-1 participation
term has nothing to discriminate. With fan-in 2 the L2 bank can actually select, which is
what lets the C -> I feedback pathway train.

Both patches are driven with the SAME pattern in each phase, so L2 receives two coincident
Eor inputs per volley.

What is expected
++++++++++++++++

Four patterns presented in sequence should leave each L1 column with FOUR DIFFERENT owners,
one per pattern. That is the claim under test; the script reports what actually happened,
including collisions, rather than asserting success.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nest_backend.engine import RunResult, Stimulus, pattern_pixels  # noqa: E402
from nest_backend.recording import (  # noqa: E402
    aggregate_multiplicity, by_column, by_role, coincidence_stats, collect_charge,
    collect_inputs, collect_spikes, collect_weight_changes, eor_input_multiplicity,
    firing_pattern, reset_stats, winner_multiplicity,
)
from nest_backend.replay_adapter import write_replay  # noqa: E402
from nest_backend.topology import E_THRESHOLD, NestTiledNetwork, Timescales  # noqa: E402

PATTERNS = ["row 1", "col 1", "diag \\", "diag /"]
OUT_ROOT = REPO_ROOT / "experiments" / "runs" / "nest_saturation"


class Convergence:
    """The declared predicate for "this pattern is learned".

    Two conditions, both measured:

    1. **Stable single owner.** For the last `window` presentations, every driven L1 column
       produced exactly ONE ordinary-E winner, and it was the SAME cell throughout. One
       winner alone is not enough -- a column that rotates between winners has not settled
       on an owner.

    2. **C basal matured.** The coincidence cell's basal weight has reached
       `c_target_frac * theta`. The one-shot recognition condition is `w_basal >= theta`
       (equations doc section 2.3), so this is what makes C able to confirm at all.

    `Eor` is deliberately NOT a criterion: it is frozen at theta by design
    (`eor_plasticity_enabled = False`), so it is at its final value from t=0 and can never
    "become trained".
    """

    def __init__(self, window: int = 4, c_target_frac: float = 0.95):
        self.window = window
        self.c_target = c_target_frac * E_THRESHOLD

    def check(self, winner_history: dict, c_weights: dict) -> dict:
        stable, owners = True, {}
        for column, hist in winner_history.items():
            recent = hist[-self.window:]
            if len(recent) < self.window:
                stable = False
                continue
            singles = [w[0] for w in recent if len(w) == 1]
            if len(singles) != self.window or len(set(singles)) != 1:
                stable = False
            else:
                owners[column] = singles[0]
        if not winner_history:
            stable = False

        matured = {cid: (w >= self.c_target) for cid, w in c_weights.items()}
        c_ok = bool(matured) and all(matured.values())
        return {
            "converged": bool(stable and c_ok),
            "stable_single_owner": stable,
            "owners": owners,
            "c_matured": c_ok,
            "c_weights": {k: round(v, 2) for k, v in c_weights.items()},
        }


class Trainer:
    """Drives one plastic NEST network through the four-pattern curriculum."""

    def __init__(self, args):
        import nest  # noqa: PLC0415

        self.nest = nest
        self.args = args
        self.ts = Timescales(h=args.h, base_ff=1.0, spread=args.spread,
                             presentation=args.presentation)
        # No multimeter during TRAINING. Full-resolution sampling of an 8.9-second training
        # run is tens of millions of samples held in memory and then discarded -- only the
        # demo window is ever converted. `demo_pass` attaches the meter at its own boundary.
        self.net = NestTiledNetwork(
            seed=args.seed, timescales=self.ts, shape=(3, 6), jitter_ms=args.jitter,
            learning=True, charge_interval_ms=None,
            c_basal_weight=None,
            # Registered at construction because a weight_recorder is a synapse-model
            # COMMON PROPERTY and cannot be attached to connections that already exist.
            # It records nothing until its window is opened.
            record_weights=args.record_training,
        )
        if args.record_training:
            self.net.attach_weight_recording(start_ms=0.0)
            if args.charge_interval:
                self.net.attach_charge_recording(args.charge_interval, start_ms=0.0)
        self.t_ms = 0.0
        self.spikes_seen = 0
        self.presentations: list = []
        self.driven_columns = [c for c in self.net.column_ids() if c.startswith("L1")]
        self.role_of = {n: m.get("column_role") for n, m in self.net.node_meta.items()}
        self.column_of = {n: m.get("column_id") for n, m in self.net.node_meta.items()}

    # ------------------------------------------------------------------ weights
    def plastic_weights(self) -> dict:
        """One kernel query per synapse model -- see `NestTiledNetwork.plastic_weights`.

        This runs inside the convergence check, once per block, for the whole run. Querying
        edge by edge made it ~93% of the wall clock.
        """
        return self.net.plastic_weights()

    def c_basal_weights(self) -> dict:
        weights = self.plastic_weights()
        out = {}
        for edge in self.net.spec["edges"]:
            if edge["kind"] == "basal_excitation" and edge["id"] in weights:
                # Only the L1 coincidence cells; the top C is dormant by construction.
                if self.net.node_meta[edge["target"]].get("has_parent"):
                    out[edge["target"]] = weights[edge["id"]]
        return out

    # ----------------------------------------------------------------- stimulus
    def present(self, pattern: str, n_volleys: int) -> None:
        """Append `n_volleys` presentations of `pattern` to BOTH patches and run them.

        Volleys are scheduled at `t + (k+1)*D`, so the LAST one sits on the closing instant
        of the interval this call simulates and is still pending when the call returns --
        NEST delivers it in the next `Simulate`. Inside training that is invisible (the next
        block delivers it, on time and at the right spacing), but at a boundary it matters;
        see `flush_pending`.
        """
        schedule: dict = defaultdict(list)
        for patch in ((0, 0), (0, 1)):
            for pixel in pattern_pixels(self.net, patch[0], patch[1], pattern):
                for k in range(n_volleys):
                    schedule[pixel].append(
                        round(self.t_ms + (k + 1) * self.ts.presentation, 6))
        # Every volley, logged as it is scheduled, so `--record-training` can label the
        # whole run without reconstructing the curriculum after the fact.
        for k in range(n_volleys):
            self.presentations.append({
                "window": len(self.presentations),
                "t_ms": round(self.t_ms + (k + 1) * self.ts.presentation, 6),
                "patches": {"0,0": pattern, "0,1": pattern},
            })
        self.net.set_input_schedule(schedule)
        duration = n_volleys * self.ts.presentation
        self.net.simulate(duration)
        self.t_ms = round(self.t_ms + duration, 6)

    def flush_pending(self) -> None:
        """Deliver the volley `present` left sitting on the closing instant, and nothing else.

        The generators are deliberately NOT rescheduled: every time they still hold is now in
        the past except that one, so a plain `simulate` delivers exactly it and gives its
        cascade a full presentation interval to complete. That makes `t_ms` a clean boundary
        -- no volley pending across it in either direction.
        """
        self.net.simulate(self.ts.presentation)
        self.t_ms = round(self.t_ms + self.ts.presentation, 6)

    def new_winners(self, n_volleys: int, block_start: float) -> dict:
        """Per driven L1 column, the winner set of each presentation in the last block."""
        spikes = collect_spikes(self.net)[self.spikes_seen:]
        self.spikes_seen += len(spikes)
        per: dict = defaultdict(lambda: defaultdict(set))
        for t, nid in spikes:
            if self.role_of.get(nid) != "E":
                continue
            column = self.column_of.get(nid)
            if column not in self.driven_columns:
                continue
            k = int((t - block_start) // self.ts.presentation)
            per[column][k].add(nid)
        return {c: [sorted(per[c][k]) for k in sorted(per[c])] for c in per}

    # -------------------------------------------------------------------- train
    def run(self) -> dict:
        criterion = Convergence(window=self.args.stable_window,
                                c_target_frac=self.args.c_target_frac)
        phases = []
        winner_history: dict = defaultdict(list)

        for pattern in PATTERNS:
            # Each pattern gets a FRESH convergence history: an owner that was stable for
            # the previous pattern says nothing about this one.
            winner_history = defaultdict(list)
            volleys = 0
            started = time.perf_counter()
            status = "max_volleys"
            verdict = {}

            while volleys < self.args.max_volleys:
                block_start = self.t_ms
                self.present(pattern, self.args.block)
                volleys += self.args.block
                for column, blocks in self.new_winners(self.args.block, block_start).items():
                    winner_history[column].extend(blocks)

                verdict = criterion.check(dict(winner_history), self.c_basal_weights())
                if verdict["converged"]:
                    status = "converged"
                    break

            phases.append({
                "pattern": pattern,
                "status": status,
                "volleys": volleys,
                "t_end_ms": self.t_ms,
                "wall_clock_s": round(time.perf_counter() - started, 2),
                **verdict,
            })
            print(f"  {pattern:<8} {status:<12} after {volleys:>4} volleys "
                  f"(t={self.t_ms:.0f} ms)  owners={verdict.get('owners')}  "
                  f"C={verdict.get('c_weights')}")

        return {"phases": phases}


def demo_pass(trainer, volleys_per_pattern: int) -> RunResult:
    """Present all four patterns to the TRAINED network and package it as a RunResult.

    The training run itself is far too long to view -- ~580 volleys and 8.7 seconds of
    simulated time. What is worth watching is the trained network RESPONDING: which
    competitor now owns each pattern, and how the charge climbs to threshold in the owner
    while the losers are reset.

    Weights keep learning during the demo (they are the same plastic synapses), but over a
    handful of volleys per pattern the movement is negligible compared with training. The
    artifact records that honestly -- `changed_synapses` stays empty because the adapter
    reports header weights as authoritative, and the header carries the POST-TRAINING
    weights.
    """
    net = trainer.net
    ts = trainer.ts

    # Training's final volley is still pending (see `present`). Delivered inside the demo it
    # would open the artifact with a TRAINING volley -- a pattern no marker describes, so the
    # viewer meets an unlabelled volley of the previous pattern before the demo begins. Drain
    # it here, where it belongs.
    trainer.flush_pending()

    # Record from here on. At `--charge-interval` equal to `h` every tick carries a sample,
    # so every tick becomes a frame and the artifact is a complete state history of the
    # demo rather than a subsample of it.
    if trainer.args.charge_interval:
        net.attach_charge_recording(trainer.args.charge_interval, start_ms=trainer.t_ms)

    presentations = []
    window = 0
    for pattern in PATTERNS:
        # `t_ms` advances only when the block is simulated, so the per-volley offset has to
        # come from `k`. Reusing the block's start for all of them collapses the volleys onto
        # one timestamp, and the adapter resolves a frame to the LAST presentation at or
        # before it -- which would label the whole block with its final window index and give
        # the volleys in between no marker at all.
        for k in range(volleys_per_pattern):
            presentations.append({
                "window": window,
                "t_ms": round(trainer.t_ms + (k + 1) * ts.presentation, 6),
                "patches": {"0,0": pattern, "0,1": pattern},
            })
            window += 1
        trainer.present(pattern, volleys_per_pattern)

    # And symmetrically at the end: without this the demo's own last volley is never
    # simulated, so the final pattern is recorded one volley short.
    trainer.flush_pending()

    # The demo's origin is its FIRST VOLLEY, not the moment training stopped -- `present`
    # puts the first volley a full D later. Taking the earlier instant would open the
    # artifact on a blank presentation interval, and -- worse -- window k of the metrics
    # below would be window k-1 of the markers, since `window_index` floors against the t0
    # it is handed.
    origin_ms = presentations[0]["t_ms"]

    def after(rows):
        return [(t, nid) for t, nid in rows if t >= origin_ms]

    spikes = after(collect_spikes(net))
    inputs = after(collect_inputs(net))
    n_windows = len(presentations)

    metrics = {
        "input_events": len(inputs),
        "input_schedule": [{"t": t, "id": nid} for t, nid in inputs],
        "spike_count": len(spikes),
        "spikes": [{"t": t, "id": nid} for t, nid in spikes],
        "by_role": {r: sum(c.values()) for r, c in by_role(net, spikes).items()},
        "by_role_detail": by_role(net, spikes),
        "by_column": by_column(net, spikes),
        "winner_multiplicity": winner_multiplicity(net, spikes, origin_ms, ts.presentation),
        "eor_input_multiplicity": eor_input_multiplicity(net),
        "coincidence": coincidence_stats(net),
        "resets": reset_stats(net),
        "firing_pattern": firing_pattern(net, spikes, origin_ms, ts.presentation, n_windows),
    }
    metrics["winner_multiplicity_overall"] = aggregate_multiplicity(
        metrics["winner_multiplicity"])

    charge = {nid: {t: v for t, v in samples.items() if t >= origin_ms}
              for nid, samples in collect_charge(net).items()}

    manifest = net.manifest()
    manifest["feedback_loop_latency_ms"] = round(net.feedback_loop_latency_ms(), 6)
    return RunResult(
        name="saturation_demo",
        manifest=manifest,
        stimulus={p: f"{volleys_per_pattern} volleys" for p in PATTERNS},
        metrics=metrics,
        wall_clock_s=0.0,
        settings={
            "seed": trainer.args.seed,
            "t0_ms": origin_ms,
            "period_ms": ts.presentation,
            "duration_ms": round(trainer.t_ms - origin_ms, 6),
            "n_presentations": n_windows,
            "learning": "dual FE/FES (weights are POST-TRAINING)",
            "note": "demonstration pass over the trained network, not the training run",
        },
        topology=net.dashboard_topology(),
        outgoing_edges=net.outgoing_edges(),
        presentations=presentations,
        charge=charge,
    )


def training_pass(trainer) -> RunResult:
    """Package the WHOLE training run -- every volley of convergence -- as a RunResult.

    The demo pass answers "what did the trained network do"; this answers "how did it get
    there", which is the question the saturation experiment is actually about. Owners are
    selected, contested and re-selected over hundreds of volleys, and none of that is
    visible in a post-training snapshot.

    Weight updates ride along as `weight_changes`, so `changed_synapses` is populated and
    the weights-over-time and receptive-field panels animate instead of sitting static.
    """
    net = trainer.net
    ts = trainer.ts
    trainer.flush_pending()          # deliver the last scheduled volley before converting

    spikes = collect_spikes(net)
    inputs = collect_inputs(net)
    presentations = trainer.presentations
    origin_ms = presentations[0]["t_ms"] if presentations else 0.0
    n_windows = len(presentations)

    metrics = {
        "input_events": len(inputs),
        "input_schedule": [{"t": t, "id": nid} for t, nid in inputs],
        "spike_count": len(spikes),
        "spikes": [{"t": t, "id": nid} for t, nid in spikes],
        "by_role": {r: sum(c.values()) for r, c in by_role(net, spikes).items()},
        "by_role_detail": by_role(net, spikes),
        "by_column": by_column(net, spikes),
        "winner_multiplicity": winner_multiplicity(net, spikes, origin_ms, ts.presentation),
        "eor_input_multiplicity": eor_input_multiplicity(net),
        "coincidence": coincidence_stats(net),
        "resets": reset_stats(net),
        "firing_pattern": firing_pattern(net, spikes, origin_ms, ts.presentation, n_windows),
    }
    metrics["winner_multiplicity_overall"] = aggregate_multiplicity(
        metrics["winner_multiplicity"])

    manifest = net.manifest()
    manifest["feedback_loop_latency_ms"] = round(net.feedback_loop_latency_ms(), 6)
    return RunResult(
        name="saturation_training",
        manifest=manifest,
        stimulus={p: "trained to convergence" for p in PATTERNS},
        metrics=metrics,
        wall_clock_s=0.0,
        settings={
            "seed": trainer.args.seed,
            "t0_ms": origin_ms,
            "period_ms": ts.presentation,
            "duration_ms": round(trainer.t_ms - origin_ms, 6),
            "n_presentations": n_windows,
            "learning": "dual FE/FES, RECORDED (weights move during this artifact)",
            "note": "the training run itself, not a demonstration over a trained network",
        },
        topology=net.dashboard_topology(),
        outgoing_edges=net.outgoing_edges(),
        presentations=presentations,
        charge=collect_charge(net),
        weight_changes=collect_weight_changes(net),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--spread", type=float, default=2.0)
    parser.add_argument("--presentation", type=float, default=60.0)
    parser.add_argument("--jitter", type=float, default=3.0)
    parser.add_argument("--block", type=int, default=10,
                        help="volleys per convergence check")
    parser.add_argument("--max-volleys", type=int, default=600,
                        help="safety cap per pattern; reported when hit")
    parser.add_argument("--stable-window", type=int, default=4)
    parser.add_argument("--c-target-frac", type=float, default=0.95)
    parser.add_argument("--charge-interval", type=float, default=None,
                        help="ms between multimeter charge samples (omit for none)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--demo-volleys", type=int, default=3,
                        help="volleys per pattern in the viewable demo pass (0 = skip)")
    parser.add_argument("--record-training", action="store_true",
                        help="record the TRAINING run itself -- every volley of convergence, "
                             "with weight updates -- instead of only the post-training demo. "
                             "This is what makes learning visible in the replay: the demo "
                             "pass shows an already-trained network responding.")
    args = parser.parse_args(argv)

    out_dir = args.out or (OUT_ROOT / time.strftime("%Y%m%dT%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("NEST 4-pattern saturation -- 3x6 sheet, two L1 columns, L2 fan-in 2")
    print(f"h={args.h} ms  jitter={args.jitter} ms  D={args.presentation} ms  "
          f"block={args.block}  cap={args.max_volleys}\n")

    trainer = Trainer(args)
    print(f"topology: {trainer.net.manifest()['counts']}  "
          f"columns={trainer.net.column_ids()}\n")

    initial = trainer.plastic_weights()
    result = trainer.run()
    final = trainer.plastic_weights()

    owners_by_pattern = {p["pattern"]: p.get("owners", {}) for p in result["phases"]}
    per_column: dict = defaultdict(dict)
    for pattern, owners in owners_by_pattern.items():
        for column, owner in owners.items():
            per_column[column][pattern] = owner

    print("\n--- owners per column ------------------------------------------------")
    for column in sorted(per_column):
        mapping = per_column[column]
        distinct = len(set(mapping.values()))
        print(f"  {column}: {distinct}/4 distinct owners")
        for pattern in PATTERNS:
            print(f"      {pattern:<8} -> {mapping.get(pattern, '(not converged)')}")

    training_info = None
    if args.record_training:
        print("\nrecording the TRAINING run (every volley, with weight updates) ...")
        training = training_pass(trainer)
        training_info = write_replay(training, out_dir / "training.snn.jsonl")
        print(f"  training replay -> {training_info['path']}  "
              f"({training_info['frames']} frames, {training_info['bytes']:,} bytes, "
              f"{len(training.weight_changes):,} weight updates)")

    demo_info = None
    if args.demo_volleys > 0:
        print(f"\nrecording demo pass ({args.demo_volleys} volleys/pattern) ...")
        demo = demo_pass(trainer, args.demo_volleys)
        demo_info = write_replay(demo, out_dir / "replay.snn.jsonl")
        demo_info["winner_multiplicity"] = demo.metrics["winner_multiplicity_overall"]
        print(f"  replay -> {demo_info['path']}  "
              f"({demo_info['frames']} frames, {demo_info['bytes']:,} bytes)")

    charge = collect_charge(trainer.net) if args.charge_interval else {}
    report = {
        "settings": vars(args) | {"out": str(out_dir)},
        "manifest": trainer.net.manifest(),
        "phases": result["phases"],
        "owners_by_column": {k: dict(v) for k, v in per_column.items()},
        "distinct_owners_by_column": {k: len(set(v.values())) for k, v in per_column.items()},
        "total_ms": trainer.t_ms,
        "weights": {
            "initial": {k: round(v, 4) for k, v in initial.items()},
            "final": {k: round(v, 4) for k, v in final.items()},
            "moved": sum(1 for k in initial if abs(final[k] - initial[k]) > 1e-9),
            "tracked": len(initial),
        },
        "charge_samples": {k: len(v) for k, v in charge.items()},
        "demo_replay": demo_info,
        "training_replay": training_info,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    if charge:
        (out_dir / "charge.json").write_text(
            json.dumps({k: {str(t): v for t, v in s.items()} for k, s in charge.items()},
                       sort_keys=True, default=str) + "\n")

    print(f"\nweights moved: {report['weights']['moved']}/{report['weights']['tracked']}")
    print(f"total simulated: {trainer.t_ms:.0f} ms")
    print(f"report -> {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
