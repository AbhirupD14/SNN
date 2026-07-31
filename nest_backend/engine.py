"""Construct, stimulate, run and record the translated `tiled_cc` network through PyNEST.

This is a thin driver, not a second simulator: it builds a `NestTiledNetwork`, converts a
patch/pattern stimulus into a NEST spike schedule, calls `Simulate`, and hands the recorder
output to `nest_backend.recording`. NEST owns the clock, delivery order, delays, buffering,
recording and thread/MPI execution throughout.

Learning is FROZEN here by construction -- every connection uses `static_synapse`, so no
weight can change inside NEST and none is touched from Python between calls. Phase 4 is what
introduces plastic synapse models.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .recording import collect_charge, summarize
from .topology import NestTiledNetwork, Timescales

# The four center-crossing 3x3 local features, mirrored from `backend.simulation.PATTERNS`.
# Imported lazily in `pattern_pixels` so this module stays importable without the engine.
PATTERN_NAMES = ("row 1", "col 1", "diag \\", "diag /")


@dataclass
class Stimulus:
    """A set of `(patch_row, patch_col) -> pattern name` assignments."""

    patches: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {f"{r},{c}": name for (r, c), name in sorted(self.patches.items())}


def pattern_pixels(net, patch_row: int, patch_col: int, pattern: str) -> list:
    """RGC ids activated by `pattern` inside one 3x3 patch.

    The patch's nine pixels come back from `net.patch_pixels` in patch-local row-major
    order, which is exactly the order `PATTERNS` indexes, so the mask applies directly.
    """
    from backend.simulation import PATTERNS  # noqa: PLC0415

    if pattern not in PATTERNS:
        raise KeyError(f"unknown 3x3 pattern {pattern!r}; have {sorted(PATTERNS)}")
    mask = PATTERNS[pattern]
    pixels = net.patch_pixels(patch_row, patch_col)
    if len(pixels) != len(mask):
        raise ValueError(f"patch has {len(pixels)} pixels but mask has {len(mask)}")
    return [pid for pid, on in zip(pixels, mask) if on]


def build_schedule(net, stimulus: Stimulus, *, t0: float, period: float,
                   n_presentations: int) -> dict:
    """`{rgc_id: [times]}` -- every active pixel of every assigned patch, once per window.

    All pixels of a presentation are emitted at the SAME time. The arrival spread that
    breaks the latency tie is produced downstream by the geometric conduction delays
    (`topology.assign_delays`), not by staggering the source, so the stimulus stays a clean
    synchronous volley exactly as the reference engine presents it.
    """
    schedule: dict = {}
    times = [round(t0 + k * period, 10) for k in range(n_presentations)]
    for (patch_row, patch_col), pattern in stimulus.patches.items():
        for pixel in pattern_pixels(net, patch_row, patch_col, pattern):
            schedule.setdefault(pixel, []).extend(times)
    return schedule


@dataclass
class RunResult:
    name: str
    manifest: dict
    stimulus: dict
    metrics: dict
    wall_clock_s: float
    settings: dict
    # The dashboard topology payload and the outgoing-edge index, captured while the NEST
    # network still exists. Carrying them here makes the artifact SELF-CONTAINED, so the
    # replay adapter is a pure artifact -> replay conversion and never has to rebuild NEST
    # (which would make "deterministic output for identical input" untestable).
    topology: dict = field(default_factory=dict)
    outgoing_edges: dict = field(default_factory=dict)
    presentations: list = field(default_factory=list)
    # Sampled charge traces from an attached NEST multimeter, or {} when none was
    # attached. This is NEST's own recording of declared model state -- never a
    # reconstruction -- and it is what turns the dashboard's charge panel from
    # "not recorded" into a real trace.
    charge: dict = field(default_factory=dict)
    # `[(t_ms, edge_id, weight)]` from a NEST weight_recorder, or [] when none was
    # registered. Event-driven rather than sampled: each entry is an actual weight update at
    # the moment it happened, which is what lets a replay show learning instead of a static
    # post-hoc weight set.
    weight_changes: list = field(default_factory=list)


def run_case(name: str, stimulus: Stimulus, *, seed: int = 1,
             timescales: Timescales | None = None, n_presentations: int = 8,
             dispersion_enabled: bool = True, feedback_enabled: bool = True,
             threads: int = 1, settle_windows: float = 2.0,
             switch_to: Stimulus | None = None,
             c_basal_weight: float | None = None,
             charge_interval_ms: float | None = None,
             shape: tuple | None = None, jitter_ms: float = 0.0,
             reset_suppression_ms: float = 0.0, learning: bool = False) -> RunResult:
    """Build, stimulate and run one case, returning its full record.

    `switch_to` presents a second stimulus for the back half of the run WITHOUT rebuilding
    the network -- suite case 5. The switch happens by rewriting the spike generators'
    schedules only; nodes, connections, weights and accumulated state all persist.

    `settle_windows` extends the simulation past the last presentation so late feedback,
    confirmation resets and coincidence deposits land inside the recorded interval rather
    than being truncated.
    """
    ts = timescales or Timescales()
    net = NestTiledNetwork(seed=seed, timescales=ts, dispersion_enabled=dispersion_enabled,
                           feedback_enabled=feedback_enabled, threads=threads,
                           c_basal_weight=c_basal_weight,
                           charge_interval_ms=charge_interval_ms,
                           shape=shape, jitter_ms=jitter_ms,
                           reset_suppression_ms=reset_suppression_ms, learning=learning)

    t0 = ts.presentation  # start one full window in, so nothing lands at t=0
    period = ts.presentation

    if switch_to is None:
        schedule = build_schedule(net, stimulus, t0=t0, period=period,
                                  n_presentations=n_presentations)
        total_windows = n_presentations
    else:
        first = n_presentations // 2
        second = n_presentations - first
        schedule = build_schedule(net, stimulus, t0=t0, period=period,
                                  n_presentations=first)
        second_t0 = round(t0 + first * period, 10)
        for pixel, times in build_schedule(net, switch_to, t0=second_t0, period=period,
                                           n_presentations=second).items():
            schedule.setdefault(pixel, []).extend(times)
        for pixel in schedule:
            schedule[pixel] = sorted(set(schedule[pixel]))
        total_windows = n_presentations

    net.set_input_schedule(schedule)
    duration = t0 + (total_windows + settle_windows) * period

    started = time.perf_counter()
    net.simulate(duration)
    wall_clock = time.perf_counter() - started

    metrics = summarize(net, t0=t0, period=period, n_windows=total_windows)
    manifest = net.manifest()
    manifest["feedback_loop_latency_ms"] = round(net.feedback_loop_latency_ms(), 6)
    topology = net.dashboard_topology()
    outgoing = net.outgoing_edges()
    charge = collect_charge(net)

    # The presentation schedule, as PLANNED. Each entry is what was actually scheduled on
    # the generators, so a replay can mark window boundaries and name the pattern being
    # shown without inferring either from the spike record.
    presentations = []
    for window in range(total_windows):
        start = round(t0 + window * period, 10)
        if switch_to is None:
            shown = stimulus
        else:
            shown = stimulus if window < (n_presentations // 2) else switch_to
        presentations.append({
            "window": window,
            "t_ms": start,
            "patches": shown.describe(),
        })

    describe = dict(stimulus.describe())
    if switch_to is not None:
        describe = {"first_half": stimulus.describe(), "second_half": switch_to.describe()}

    return RunResult(
        name=name,
        manifest=manifest,
        stimulus=describe,
        metrics=metrics,
        wall_clock_s=round(wall_clock, 4),
        settings={
            "seed": seed,
            "n_presentations": n_presentations,
            "t0_ms": t0,
            "period_ms": period,
            "duration_ms": duration,
            "settle_windows": settle_windows,
            "dispersion_enabled": dispersion_enabled,
            "feedback_enabled": feedback_enabled,
            "threads": threads,
            "learning": ("dual FE/FES on feedforward + C basal" if learning
                         else "frozen (static_synapse throughout)"),
            "c_basal_weight_override": c_basal_weight,
            "charge_interval_ms": charge_interval_ms,
            "shape": list(shape) if shape else [9, 9],
            "jitter_ms": jitter_ms,
            "reset_suppression_ms": reset_suppression_ms,
        },
        topology=topology,
        outgoing_edges=outgoing,
        presentations=presentations,
        charge=charge,
    )
