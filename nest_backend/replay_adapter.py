"""Convert a completed NEST run into the repository's versioned `snn.replay` artifact.

This is an **adapter**, not a simulator. It reads a finished
`nest_backend.engine.RunResult` and serialises what NEST actually recorded into the
existing schema-1 replay contract, so the dashboard's existing **Load Test** player can
open it with no second renderer and no browser-side science.

What it will not do
+++++++++++++++++++

* It never reconstructs `q`, `q_pre`, eligibility, refractory or membrane charge from the
  spike record. Those histories were not sampled, so every frame reports them as
  **unavailable** (the field is omitted, and the frame's declared `state_availability`
  map says why) rather than as zero or as a repeated final value.
* It never collapses simultaneous winners. Every sender at a timestamp is preserved.
* It never invents learning. A frozen manifest keeps header weights authoritative and
  `changed_synapses` empty. For a learning run with a registered recorder,
  `changed_synapses` carries the updates NEST actually emitted, at the ticks it emitted
  them -- recorded events, never an interpolated trajectory between them. Learning without
  a recorder is labelled as such; an empty change list is not relabelled frozen.

What it does derive
+++++++++++++++++++

Only one field, by an exact documented rule: `emitted` (the edge-pulse list the renderer
uses) is the set of outgoing edges of the nodes that spiked at that tick. A spike
traverses exactly its source's outgoing edges, so this is exact rather than heuristic.

Run:
    .nest-env/bin/python -m nest_backend.replay_adapter --case 01_center_patch_row
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nest_backend.topology import Timescales  # noqa: E402
from experiments.replay_recorder import (  # noqa: E402
    REC_CHECKPOINT,
    REC_FRAME,
    REC_HEADER,
    REC_MARKER,
    REC_RESULT,
    REPLAY_SCHEMA_NAME,
    REPLAY_SCHEMA_VERSION,
    _reject_nonfinite,
)

# Bumped independently of the replay schema: it identifies THIS adapter's mapping
# decisions, not the artifact format, so a mapping change is auditable without implying a
# schema break.
ADAPTER_VERSION = 1

# Tolerance for "this timestamp lies on the resolution grid". NEST times come back as
# doubles; anything further off than this is a real error, not float noise.
GRID_TOLERANCE_MS = 1e-9


class AdapterError(ValueError):
    """Raised when a NEST artifact cannot be represented faithfully."""


# --------------------------------------------------------------------------- time
def to_tick(t_ms: float, h: float) -> int:
    """`round(t_ms / h)`, rejecting any timestamp that is not on the resolution grid.

    Failing loudly matters: a silently rounded timestamp would move a spike to a
    neighbouring frame and quietly change what the dashboard shows.
    """
    if not math.isfinite(t_ms):
        raise AdapterError(f"non-finite timestamp {t_ms!r}")
    ticks = t_ms / h
    nearest = round(ticks)
    if abs(ticks - nearest) * h > GRID_TOLERANCE_MS:
        raise AdapterError(
            f"timestamp {t_ms} ms is not on the h={h} ms resolution grid "
            f"(off by {abs(ticks - nearest) * h:g} ms)"
        )
    return int(nearest)


def to_ms(tick: int, h: float) -> float:
    return round(tick * h, 10)


# --------------------------------------------------------------------- provenance
def git_info(repo_dir: Path = REPO_ROOT) -> dict:
    """Commit and dirty flag. A git failure never fails a conversion."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True,
                              text=True, timeout=15)
    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
        if commit.returncode != 0:
            return {"commit": None, "dirty": None}
        return {
            "commit": commit.stdout.strip() or None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except Exception:
        return {"commit": None, "dirty": None}


def provenance(result) -> dict:
    """Everything §4.1 requires to reproduce the run."""
    manifest = result.manifest
    versions = manifest.get("versions", {})
    timescales = manifest.get("timescales", {})
    kernel = manifest.get("kernel", {})
    return {
        "adapter_version": ADAPTER_VERSION,
        "replay_schema": REPLAY_SCHEMA_NAME,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "engine": "nest",
        # Which SEMANTIC profile produced this run, carried into the artifact rather than
        # left in the manifest only. A replay outlives the process that made it, and
        # `impulse_characterization` results must never be readable as CIPP-engine claims
        # once a second profile exists.
        #
        # The NAME alone is not enough. Without the parameters, the implementation
        # disposition and the conduction envelope, a reader cannot tell a Phase 2 scaffold
        # from a promoted engine, nor which values were provisional. All of it travels.
        "engine_profile": manifest.get("engine_profile"),
        "profile": manifest.get("profile"),
        "implementation_disposition": (manifest.get("profile") or {}).get("disposition"),
        "causal_arrival_envelope": manifest.get("causal_arrival_envelope"),
        "accepted_differences": manifest.get("accepted_differences"),
        "case": result.name,
        "nest_version": versions.get("nest"),
        "nestml_version": versions.get("nestml"),
        "nest_module": versions.get("module"),
        "model_fingerprint": versions.get("model_fingerprint"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": git_info(),
        "spec_hash": manifest.get("spec_hash"),
        "seed": manifest.get("seed"),
        "resolution_h_ms": timescales.get("h_ms"),
        "L_wta_ms": timescales.get("L_wta_ms"),
        "S_target_ms": timescales.get("S_target_ms"),
        "S_realized": manifest.get("delays", {}).get("realized_spread_ms"),
        "D_ms": timescales.get("D_ms"),
        "ratio_S_over_Lwta": timescales.get("ratio_S_over_Lwta"),
        "ratio_D_over_S": timescales.get("ratio_D_over_S"),
        "threads": kernel.get("local_num_threads"),
        "mpi_ranks": kernel.get("num_processes"),
        "mpi_available": False,
        "dispersion_enabled": manifest.get("dispersion_enabled"),
        "feedback_enabled": manifest.get("feedback_enabled"),
        "c_basal_weight_override": manifest.get("c_basal_weight_override"),
        "learning_mode": (
            "frozen" if not manifest.get("learning") else
            "unrecorded" if not (manifest.get("weight_state") or {}).get("updates_recorded") else
            "recorded" if result.weight_changes else "recorded_no_updates"
        ),
        "weight_state": manifest.get("weight_state"),
        "converted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ----------------------------------------------------------------- availability
# Declared once and carried on every frame so a consumer never has to guess whether a
# missing value means "zero" or "we did not look".
BASE_AVAILABILITY = {
    "spiked": "recorded",
    "input": "recorded",
    "emitted": "derived",
    "potential": "unavailable",
    "activation": "unavailable",
    "v_pre_reset": "unavailable",
    "freq": "unavailable",
    "refractory": "unavailable",
    "g_inh": "unavailable",
    "trace": "unavailable",
    "coincidence_charge": "unavailable",
    "changed_synapses": "recorded",
}

# Kept for callers that import the old name.
STATE_AVAILABILITY = dict(BASE_AVAILABILITY)

# Which dashboard field each NESTML recordable populates, and how. `derived` is used only
# where an exact rule turns the recorded value into the field: `activation` is `q / theta`,
# `refractory` is the remaining refractory interval expressed in the timesteps the panel
# labels it with. Everything else is a straight rename.
RECORDABLE_FIELDS = {
    "q": (("potential", "recorded"), ("activation", "derived")),
    "q_pre": (("v_pre_reset", "recorded"),),
    "refr_until": (("refractory", "derived"),),
    "basal_charge": (("coincidence_charge", "recorded"),),
}


def availability(recorded_vars) -> dict:
    """Declared provenance per field, given which recordables this run actually sampled.

    None of this is intrinsically unrecordable in NEST -- `q`, `q_pre`, `refr_until`,
    `basal_charge` and the lock timers are all declared NESTML recordables, and a
    `multimeter` samples them. Whether a particular artifact HAS them is a property of the
    run, so the declaration is computed from the recorded variable names rather than
    hard-coded or inferred from a single boolean.

    `freq`, `g_inh` and `trace` stay unavailable for a reason worth distinguishing from
    "we did not look": these models have no such state. Under `leak_rate = 0` with no
    persistent inhibitory conductance there IS no `g_inh` to sample.
    """
    out = dict(BASE_AVAILABILITY)
    for var in recorded_vars or ():
        for field, provenance in RECORDABLE_FIELDS.get(var, ()):
            out[field] = provenance
    return out


def activity_availability(freq_recorded: bool, resets_recorded: bool) -> dict:
    """Provenance for the two fields derived from the SPIKE record rather than a meter.

    Both follow exact rules over data that is fully recorded, which is what separates them
    from a reconstruction: `hard_reset_events` is edge traversal (identical in kind to
    `emitted`), and `freq` is a count over a declared window.
    """
    out = {}
    if freq_recorded:
        out["freq"] = "derived"
    if resets_recorded:
        out["hard_reset_events"] = "derived"
    # These have no counterpart in the models at all -- not unsampled, absent. Under
    # `leak_rate = 0` with no persistent inhibitory conductance there is no `g_inh`, and
    # the NESTML cells carry no eligibility trace variable.
    out["g_inh"] = "n/a"
    out["trace"] = "n/a"
    out["inhibitory_pulses"] = "n/a"
    return out


UNAVAILABLE_NOTE = (
    "This NEST run recorded a complete spike record. Fields marked `unavailable` were "
    "never sampled -- they are not zero, and they were not reconstructed from spikes. "
    "Charge CAN be recorded (q and q_pre are NESTML recordables sampled by a multimeter); "
    "run with --charge-interval to include it."
)


# --------------------------------------------------------------------- conversion
def _input_vector(topology: dict, patches: dict) -> list:
    """The 81-cell RGC input sheet for a presentation, from node metadata only.

    Built by looking up each active pattern pixel's RGC node and reading its `pixel`
    index -- never by parsing an id and never by re-deriving the tiling arithmetic.
    """
    from backend.simulation import PATTERNS  # noqa: PLC0415

    grid = topology.get("grid", {})
    size = int(grid.get("rows", 9)) * int(grid.get("cols", 9))
    vector = [0] * size

    by_patch: dict = {}
    for node in topology["neurons"]:
        if node.get("archetype") != "rg_source":
            continue
        key = (node.get("patch_row"), node.get("patch_col"))
        by_patch.setdefault(key, []).append(node)

    for label, pattern_name in (patches or {}).items():
        row, col = (int(x) for x in str(label).split(","))
        mask = PATTERNS.get(pattern_name)
        if mask is None:
            continue
        pixels = sorted(by_patch.get((row, col), []),
                        key=lambda n: (n.get("patch_local_row", 0),
                                       n.get("patch_local_col", 0)))
        for node, on in zip(pixels, mask):
            if on and node.get("pixel") is not None:
                vector[int(node["pixel"])] = 1
    return vector


FREQ_WINDOW_PRESENTATIONS = 4


def _rolling_freq(spikes_by_tick: dict, ticks: list, h: float,
                  window_ms: float, period_ms: float) -> dict:
    """`{tick: {node: freq}}` -- spikes in the trailing window, per presentation's worth.

    `freq = spikes_in_window / FREQ_WINDOW_PRESENTATIONS`, clamped to 1. A cell that wins
    every volley reads 1.0; one that never fires reads 0.0. Returns `{}` when the run
    declares no presentation period, because then there is no unit to normalise against and
    a made-up one would be worse than the field being honestly absent.
    """
    if window_ms <= 0 or period_ms <= 0 or not ticks:
        return {}
    span = max(1, int(round(window_ms / h)))
    spiking = sorted(spikes_by_tick)
    out: dict = {}
    for tick in ticks:
        counts: dict = {}
        lo = tick - span
        for st in spiking:
            if st > tick:
                break
            if st > lo:
                for nid in spikes_by_tick[st]:
                    counts[nid] = counts.get(nid, 0) + 1
        if counts:
            out[tick] = {nid: round(min(1.0, c / FREQ_WINDOW_PRESENTATIONS), 4)
                         for nid, c in counts.items()}
    return out


def _reset_events(topology: dict, spikes_by_tick: dict, h: float) -> dict:
    """`{tick: [{source, target, edge}]}` -- hard resets delivered at each tick.

    An I spike traverses precisely its outgoing `hard_reset_inhibition` edges and lands one
    edge delay later. That is the same exact derivation `emitted` uses, applied to the one
    inhibitory pathway these models have.
    """
    edges = [s for s in topology.get("synapses", ())
             if s.get("kind") == "hard_reset_inhibition"]
    if not edges:
        return {}
    by_source: dict = {}
    for edge in edges:
        delay_ticks = max(1, int(round(float(edge.get("nest_delay_ms") or h) / h)))
        by_source.setdefault(edge["source"], []).append(
            (edge["target"], edge["id"], delay_ticks))

    out: dict = {}
    for tick, fired in spikes_by_tick.items():
        for nid in fired:
            for target, edge_id, delay_ticks in by_source.get(nid, ()):
                out.setdefault(tick + delay_ticks, []).append(
                    {"source": nid, "target": target, "edge": edge_id})
    return {t: sorted(v, key=lambda e: e["target"]) for t, v in out.items()}


def _top_winner(winners_by_column: dict, column_of: dict):
    """The single winner to highlight: deepest column layer first, then stable id order.

    Purely a display choice for the legacy one-winner field. Every winner remains visible
    through `column_winners` and its own `spiked` flag, so nothing is hidden by picking one.
    """
    if not winners_by_column:
        return None
    def rank(column):
        return (column.startswith("L2"), column)
    best = max(winners_by_column, key=rank)
    return sorted(winners_by_column[best])[0]


def _window_of(t_ms: float, presentations: list) -> tuple:
    """(window index, patches) for a timestamp: the latest presentation at or before it."""
    current = (None, {})
    for entry in presentations:
        if t_ms + GRID_TOLERANCE_MS >= entry["t_ms"]:
            current = (entry["window"], entry["patches"])
        else:
            break
    return current


def build_records(result) -> list:
    """Convert a `RunResult` into the ordered list of replay records."""
    topology = result.topology
    if not topology:
        raise AdapterError("run result carries no topology payload")

    manifest = result.manifest
    h = float(manifest["timescales"]["h_ms"])
    node_ids = [n["id"] for n in topology["neurons"]]
    node_set = set(node_ids)
    role_of = {n["id"]: n.get("column_role") for n in topology["neurons"]}
    column_of = {n["id"]: n.get("column_id") for n in topology["neurons"]}
    outgoing = result.outgoing_edges or {}
    presentations = result.presentations or []

    prov = provenance(result)

    # Sampled charge, when a multimeter was attached. Its sample times become frame ticks
    # too, so the charge view is continuous rather than only defined where a spike
    # happened -- a multimeter sample IS an explicitly recorded observation.
    charge = result.charge or {}
    threshold_of = {n["id"]: float(n.get("threshold") or 0.0)
                    for n in topology["neurons"]}
    charge_by_tick: dict = {}
    recorded_vars: set = set()
    for nid, samples in charge.items():
        for t_ms, values in samples.items():
            charge_by_tick.setdefault(to_tick(float(t_ms), h), {})[nid] = values
            recorded_vars.update(values)
    charge_recorded = bool(charge_by_tick)

    # Derived-activity provenance is a property of the RUN, so it is settled here with the
    # rest of the declaration rather than from whether any event happened to occur: `freq`
    # is derivable whenever the run declares a presentation period to normalise against,
    # and hard resets whenever the topology has the edges to traverse.
    period = float((result.settings or {}).get("period_ms") or 0.0)
    freq_window_ms = FREQ_WINDOW_PRESENTATIONS * period if period > 0 else 0.0
    has_reset_edges = any(s.get("kind") == "hard_reset_inhibition"
                          for s in topology.get("synapses", ()))

    # Recorded weight updates, keyed by the tick they happened on. Their ticks become frame
    # ticks too: a weight change is an observation, and a run that learns but shows no frame
    # where it learned would be the same failure as a spike with no frame.
    changes_by_tick: dict = {}
    for t_ms, edge_id, weight in (result.weight_changes or ()):
        changes_by_tick.setdefault(to_tick(float(t_ms), h), []).append(
            {"id": edge_id, "weight": round(float(weight), 6)})
    weights_recorded = bool(changes_by_tick)
    # The two weight snapshots the artifact can carry, kept distinct: what the HEADER
    # topology holds (the baseline the deltas apply to, when the producer captured one) and
    # the separately labelled final materialized set, when the producer captured that.
    final_weights = getattr(result, "final_weights", None) or {}
    header_weight_state = (topology.get("nest") or {}).get("weight_state")
    state_availability = availability(recorded_vars) | activity_availability(
        freq_window_ms > 0, has_reset_edges)

    records: list = []
    records.append({
        "record": REC_HEADER,
        "schema": REPLAY_SCHEMA_NAME,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_id": f"nest-{result.name}",
        "experiment": f"nest_3x3:{result.name}",
        "created_utc": prov["converted_utc"],
        "seed": manifest.get("seed"),
        "topology_name": topology.get("params", {}).get("topology_name"),
        "preset": topology.get("params", {}).get("topology"),
        "conditions": {
            # The legacy player reads this key; for NEST the meaningful condition is
            # whether the translated C -> I confirmation pathway was connected.
            "hierarchical_feedback": bool(manifest.get("feedback_enabled")),
            "engine": "nest",
            "learning": (
                "frozen" if not manifest.get("learning") else
                "recorded" if (manifest.get("weight_state") or {}).get("updates_recorded")
                else "unrecorded"
            ),
            "dispersion_enabled": manifest.get("dispersion_enabled"),
            "c_basal_weight_override": manifest.get("c_basal_weight_override"),
        },
        "schedule": {
            "presentations": presentations,
            "t0_ms": result.settings.get("t0_ms"),
            "period_ms": result.settings.get("period_ms"),
            "duration_ms": result.settings.get("duration_ms"),
        },
        "recording": {
            # Frames are event-driven, not strided: a frame exists for every tick that
            # carried an observation. `record_every` is 1 in the sense that nothing
            # observed was dropped, but ticks with no events have no frame at all.
            "record_every": 1,
            "checkpoint_every": 0,
            "frame_policy": "event-driven: one frame per NEST tick carrying an observation",
            "state_availability": state_availability,
            "charge_interval_ms": manifest.get("charge_interval_ms"),
            "charge_recorded": charge_recorded,
            "unavailable_note": UNAVAILABLE_NOTE,
        },
        "nest": prov,
        "neuron_order": node_ids,
        "synapse_order": [s["id"] for s in topology["synapses"]],
        "topology": topology,
    })

    # ---- gather observations by tick -------------------------------------------
    spikes_by_tick: dict = {}
    for spike in result.metrics.get("spikes", []):
        nid = spike["id"]
        if nid not in node_set:
            raise AdapterError(f"spike from unknown repository node id {nid!r}")
        spikes_by_tick.setdefault(to_tick(float(spike["t"]), h), set()).add(nid)

    inputs_by_tick: dict = {}
    for event in result.metrics.get("input_schedule", []):
        nid = event["id"]
        if nid not in node_set:
            raise AdapterError(f"input event from unknown repository node id {nid!r}")
        inputs_by_tick.setdefault(to_tick(float(event["t"]), h), set()).add(nid)

    presentation_ticks = {to_tick(float(p["t_ms"]), h): p for p in presentations}

    ticks = sorted(set(spikes_by_tick) | set(inputs_by_tick) | set(presentation_ticks)
                   | set(charge_by_tick) | set(changes_by_tick))
    if not ticks:
        raise AdapterError("no observations to convert -- the run recorded nothing")

    # Per-window winner sets, so a frame can report the multiplicity of the window it is
    # part of without the consumer having to aggregate frames itself.
    multiplicity = result.metrics.get("winner_multiplicity", {})

    # ---- markers ----------------------------------------------------------------
    frame_index = 0
    previous_patches = None
    for entry in presentations:
        kind = "presentation" if previous_patches is None or \
            entry["patches"] == previous_patches else "pattern_switch"
        records.append({
            "record": REC_MARKER,
            "kind": kind,
            "frame_index": 0,
            "timestep": to_tick(float(entry["t_ms"]), h),
            "annotation": {"phase": "nest", "pattern": json.dumps(entry["patches"]),
                           "tags": [], "notes": None},
            "data": {"window": entry["window"], "t_ms": entry["t_ms"],
                     "patches": entry["patches"]},
        })
        previous_patches = entry["patches"]

    # ---- derived activity ---------------------------------------------------------
    # Rolling firing rate. The live engine's `firing_freq` is the mean of a 40-TIMESTEP
    # spike history, where a timestep is a presentation boundary. A NEST timestep is `h`,
    # so porting the number would measure 0.4 ms at h=0.01 -- a cell that fires once per
    # 15 ms volley would read 0 for 97% of frames and the renderer would stay unlit. The
    # window is therefore declared in MILLISECONDS and normalised against the stimulus:
    # one spike per presentation over the trailing `FREQ_WINDOW_PRESENTATIONS` counts as
    # fully active. Exact, deterministic, and stated in the artifact rather than implied.
    freq_by_tick = _rolling_freq(spikes_by_tick, ticks, h, freq_window_ms, period)

    # A hard reset is what inhibition IS in these models -- the I cell clears its targets'
    # accumulated charge outright, with no persistent conductance (there is no `g_inh` to
    # increment). So it is reported as `hard_reset_events`, the field the live engine uses
    # for exactly the same mechanism, and never as an `inhibitory_pulses` conductance
    # increment it does not have. The derivation is the same exact rule as `emitted`: an
    # I spike traverses precisely its outgoing reset edges, arriving one delay later.
    resets_by_tick = _reset_events(topology, spikes_by_tick, h)

    for tick in ticks:
        t_ms = to_ms(tick, h)
        window, patches = _window_of(t_ms, presentations)
        cortical = spikes_by_tick.get(tick, set())
        arrived = inputs_by_tick.get(tick, set())
        # An RGC source IS a spiking node: its `spike_generator` emitting at this tick is
        # the retinal cell firing. Those events land on a SEPARATE NEST recorder from the
        # cortical spikes (generators are devices, not model neurons), so they have to be
        # unioned in explicitly -- otherwise the whole 9x9 input surface stays dark in the
        # renderer even though the events were recorded.
        fired = cortical | arrived

        # EVERY node appears in EVERY frame, even when it did nothing.
        #
        # This is not padding. `applyDynamic` rebuilds `stateById` from `dynamic.neurons`,
        # so a node omitted from a frame would keep the PREVIOUS frame's state -- its
        # `spiked` flag would persist and a past spike would be shown as current. Emitting
        # the full roster is what makes a backward seek truthful.
        #
        # Only `spiked` is carried, because only `spiked` was recorded. The unsampled
        # fields are OMITTED rather than sent as null: the header's and frame's
        # `state_availability` map is the authority on why they are missing, and reading an
        # absent field yields `undefined`, which every consumer already treats as unknown
        # via `!= null`. Omitting them keeps the artifact small enough to replay in a
        # browser without ever implying a value.
        # `q` and `activation = q / theta` come from a NEST multimeter when one was
        # attached, and are simply ABSENT otherwise. They are never interpolated between
        # samples and never carried over from a neighbouring tick: a node with no sample at
        # this tick reports no charge, which the consumer reads as unknown.
        sampled = charge_by_tick.get(tick, {})
        freqs = freq_by_tick.get(tick, {})
        neurons = []
        for nid in node_ids:
            record = {"id": nid, "spiked": nid in fired}
            if freq_window_ms > 0:
                record["freq"] = freqs.get(nid, 0.0)
            values = sampled.get(nid)
            if values:
                theta = threshold_of.get(nid) or 0.0
                if "q" in values:
                    q = float(values["q"])
                    record["potential"] = round(q, 4)
                    if theta:
                        record["activation"] = round(q / theta, 4)
                if "q_pre" in values:
                    record["v_pre_reset"] = round(float(values["q_pre"]), 4)
                if "basal_charge" in values:
                    record["coincidence_charge"] = round(float(values["basal_charge"]), 4)
                # The coincidence gate, read off the deadlines the C cell actually holds.
                # `basal_until` / `apical_until` ARE the eligibility windows -- the cell is
                # basally eligible exactly while `t < basal_until` -- so the inspector's
                # gate card is a direct reading of recorded state, not a reconstruction.
                if "basal_until" in values:
                    record["basal_eligible"] = t_ms < float(values["basal_until"])
                if "apical_until" in values:
                    record["apical_active"] = t_ms < float(values["apical_until"])
                if "basal_until" in values and "apical_until" in values:
                    record["coincidence_active"] = (
                        record["basal_eligible"] and record["apical_active"])
                if "refr_until" in values:
                    # `refr_until` is an ABSOLUTE deadline in ms; the panel labels this field
                    # "<n> steps", so the remaining interval is converted to timesteps. Once
                    # the deadline has passed the remainder is clamped at 0 -- a negative
                    # count would read as refractoriness rather than the absence of it.
                    remaining = float(values["refr_until"]) - t_ms
                    record["refractory"] = max(0, int(round(remaining / h)))
                # Every recordable NEST handed over, under its NESTML name and unrounded.
                # The mapped fields above exist for the panels that already speak this
                # vocabulary; this block exists so nothing NEST sampled is lost in the
                # translation -- lock timers and `q_confirm` have no dashboard field yet.
                # 6 decimals: times live on the `h` grid (h >= 0.001 ms) so they are exact,
                # and charges are accumulated sums against a threshold of ~1000, where the
                # 7th decimal is far below anything that changes behaviour. The float64
                # repr NEST hands back is ~18 characters per value, on every node of every
                # tick -- unrounded it is the single largest term in the artifact.
                record["nest_state"] = {k: round(float(v), 6)
                                        for k, v in sorted(values.items())}
            neurons.append(record)

        # Exact derivation: a spike traverses precisely its source's outgoing edges.
        emitted = sorted({eid for nid in fired for eid in outgoing.get(nid, ())})

        # Every ordinary-E that fired in this column at this tick. NOT reduced to one.
        winners_by_column: dict = {}
        for nid in sorted(fired):
            if role_of.get(nid) == "E" and column_of.get(nid):
                winners_by_column.setdefault(column_of[nid], []).append(nid)

        frame_index += 1
        dynamic = {
            "timestep": tick,
            "running": False,
            "neurons": neurons,
            # Recorded weight updates at this tick. Empty for a frozen run, where header
            # weights are authoritative; populated whenever a weight_recorder was watching.
            "changed_synapses": changes_by_tick.get(tick, []),
            "emitted": emitted,
            # Empty because these models HAVE no persistent inhibitory conductance to
            # increment, not because it went unrecorded. The inhibition they do have is the
            # hard reset below.
            "inhibitory_pulses": [],
            "hard_reset_events": resets_by_tick.get(tick, []),
            "latency_ties": [],
            # Legacy single-winner-per-column highlight, populated with the FIRST winner by
            # stable id order. The complete set is in `nest.column_winners`, and every
            # winner is independently visible through its own `spiked` flag, so no winner
            # is lost by this field's shape.
            "column_winners": {col: {"id": ids[0], "tau": None}
                               for col, ids in winners_by_column.items()},
            "input": _input_vector(topology, patches),
            # The top-of-hierarchy winner, so the status pill reads a cell rather than a
            # dash. Deepest layer first: an L2 winner outranks an L1 one.
            "winner": _top_winner(winners_by_column, column_of),
            "stats": {
                "total": len(node_ids),
                # `active` is now answerable whenever charge was sampled: a node is active
                # if it holds charge. Still None for a spikes-only run, where it is unknown
                # rather than zero.
                "active": (sum(1 for n in neurons if (n.get("potential") or 0) > 0)
                           if sampled else None),
                "firing": len(fired),        # RGC sources included: they are firing cells
                "firing_rate": (round(len(fired) / len(node_ids), 4) if node_ids else None),
                "winner": _top_winner(winners_by_column, column_of),
            },
            "log": [],
            "nest": {
                "t_ms": t_ms,
                "tick": tick,
                "resolution_h_ms": h,
                "window": window,
                "patches": patches,
                "threads": manifest.get("kernel", {}).get("local_num_threads"),
                "case": result.name,
                # Kept separate here so the two recorders stay auditable, even though
                # both drive `spiked` above.
                "input_events": sorted(arrived),
                "spike_events": sorted(cortical),
                "column_winners": {col: list(ids) for col, ids in winners_by_column.items()},
                "winner_multiplicity": {col: len(ids)
                                        for col, ids in winners_by_column.items()},
                "window_multiplicity": {
                    col: multiplicity.get(col, {}).get("mean")
                    for col in winners_by_column
                },
                "state_availability": state_availability,
                "charge_recorded": charge_recorded,
            },
        }

        records.append({
            "record": REC_FRAME,
            "frame_index": frame_index,
            "timestep": tick,
            "record_every": 1,
            "annotation": {
                "phase": "nest",
                "pattern": json.dumps(patches) if patches else None,
                "tags": ["nest", "learning" if manifest.get("learning") else "frozen"],
                "notes": None,
            },
            "dynamic": dynamic,
        })

    # ---- result -------------------------------------------------------------------
    records.append({
        "record": REC_RESULT,
        "status": "completed",
        "case": result.name,
        "nest": prov,
        "stimulus": result.stimulus,
        "settings": result.settings,
        "wall_clock_s": result.wall_clock_s,
        "measurements": {
            "spike_count": result.metrics.get("spike_count"),
            "by_role": result.metrics.get("by_role"),
            "by_column": result.metrics.get("by_column"),
            "winner_multiplicity_overall": result.metrics.get("winner_multiplicity_overall"),
            "winner_multiplicity": result.metrics.get("winner_multiplicity"),
            "eor_input_multiplicity": result.metrics.get("eor_input_multiplicity"),
            "coincidence": result.metrics.get("coincidence"),
            "firing_pattern": result.metrics.get("firing_pattern"),
            "weights": {
                # `state` describes THESE values, and only these. A learning run that
                # attached no final snapshot is `not_captured`, never `initial_frozen`:
                # the header of such a run may itself hold a materialized snapshot (the
                # saturation demo pass does), and calling that pair "initial_frozen"
                # would state the opposite of both halves of the artifact.
                "state": ("materialized_final" if final_weights
                          else "initial_frozen" if not manifest.get("learning")
                          else "not_captured"),
                "header_weight_state": header_weight_state,
                "logical_weights_flushed": bool(
                    (manifest.get("weight_state") or {}).get("logical_weights_flushed")
                ),
                "values": final_weights or None,
            },
        },
        "frames": frame_index,
    })

    return records


def write_replay(result, path: Path) -> dict:
    """Serialise a `RunResult` to `path` as newline-delimited JSON.

    Every record is checked for non-finite numbers BEFORE any bytes are written, using the
    recorder's own validator, so a malformed artifact is never produced.
    """
    records = build_records(result)
    for index, record in enumerate(records):
        _reject_nonfinite(record, f"$[{index}]")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    frames = sum(1 for r in records if r["record"] == REC_FRAME)
    return {
        "path": str(path),
        "records": len(records),
        "frames": frames,
        "markers": sum(1 for r in records if r["record"] == REC_MARKER),
        "bytes": path.stat().st_size,
    }


# ---------------------------------------------------------------------------- CLI
# `extra` is passed straight to run_case, so a case can pin the timescales and levers it
# needs. `wta_solved` is the configuration under which single-winner WTA holds on 8/8 seeds.
CASES = {
    "wta_solved_3x3": (
        {(0, 0): "row 1"},
        {"shape": (3, 3), "jitter_ms": 3.0,
         "timescales": Timescales(h=0.001, base_ff=1.0, spread=2.0, presentation=60.0)},
    ),
    "wta_unsolved_3x3": (
        {(0, 0): "row 1"},
        {"shape": (3, 3), "jitter_ms": 0.0,
         "timescales": Timescales(h=0.1, base_ff=1.0, spread=2.0, presentation=60.0)},
    ),
    "01_center_patch_row": ({(1, 1): "row 1"}, {}),
    "02_center_patch_col": ({(1, 1): "col 1"}, {}),
    "03_two_independent_patches": ({(0, 0): "row 1", (2, 2): "col 1"}, {}),
    "04_all_nine_patches": ({(r, c): "row 1" for r in range(3) for c in range(3)}, {}),
    "06b_feedback_on_matured_c": ({(1, 1): "row 1"}, {"c_basal_weight": 1000.0}),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="01_center_patch_row", choices=sorted(CASES))
    parser.add_argument("--presentations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--charge-interval", type=float, default=None, metavar="MS",
        help="attach a NEST multimeter sampling q/q_pre every MS milliseconds (must be a "
             "multiple of h). Makes the dashboard charge panel real instead of "
             "'not recorded'. Omit to keep the artifact small.")
    args = parser.parse_args(argv)

    from .engine import Stimulus, run_case  # noqa: PLC0415

    patches, extra = CASES[args.case]
    result = run_case(args.case, Stimulus(patches), seed=args.seed,
                      n_presentations=args.presentations,
                      charge_interval_ms=args.charge_interval, **extra)

    out = args.out or (REPO_ROOT / "experiments" / "runs" / "nest_3x3" /
                       f"replay_{args.case}" / "replay.snn.jsonl")
    info = write_replay(result, out)
    print(json.dumps(info, indent=2))
    print(f"\nOpen it with the dashboard's Load Test control:\n  {info['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
