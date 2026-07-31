"""Turn NEST recorder output into the metrics prompt section 10 requires.

Everything here selects components through topology metadata (`column_id`, `column_role`,
`layer`, `archetype`), never by parsing display names.

The two metrics that carry the scientific weight are:

`winner_multiplicity`
    Distinct ordinary-E cells that fired in one column during one presentation window.
    The reference engine guarantees exactly 1 -- the first eligible crossing recruits its
    column's `I` at that same tau, which hard-resets the whole local bank and cancels every
    later prospective crossing. Under NEST the reset cannot arrive sooner than `L_wta`, so
    a value above 1 is the direct measurement of WTA loss.

`eor_input_multiplicity`
    Corroborator. `Eor` is frozen at theta with afferent weights at theta, so it
    contractually fires on ONE afferent. `q_pre / theta` at an `Eor` firing is therefore
    the number of column winners that drove it. `Eor` is not plastic, so a multiplicity
    above 1 breaks nothing numerically -- which is exactly why it has to be measured rather
    than inferred from "it still fires".
"""
from __future__ import annotations

import math
from collections import defaultdict


def collect_spikes(net) -> list:
    """[(time_ms, repo_node_id), ...] sorted by time, from the authoritative recorder."""
    events = net.recorder.get("events")
    out = [(float(t), net.id_of_gid[int(g)])
           for t, g in zip(events["times"], events["senders"])]
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def collect_inputs(net) -> list:
    events = net.input_recorder.get("events")
    out = [(float(t), net.id_of_gid[int(g)])
           for t, g in zip(events["times"], events["senders"])]
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def collect_charge(net) -> dict:
    """`{node_id: {tick_ms: {var: float}}}` -- EVERY variable the multimeters recorded.

    The variable names are taken from the recorded events themselves rather than from a
    fixed list here, so whatever `_attach_multimeter` asked NEST for arrives intact. A
    hard-coded pair would silently drop `refr_until`, `basal_charge` and the rest even
    though the multimeter had already sampled them.

    Returns `{}` when no multimeter was attached, which the adapter reports as
    "unavailable" rather than as zero. The samples are NEST's own recording of declared
    model state -- nothing here reconstructs or interpolates.
    """
    out: dict = {}
    for meter in getattr(net, "multimeters", {}).values():
        events = meter.get("events")
        senders, times = events["senders"], events["times"]
        for key in events:
            if key in ("senders", "times"):
                continue
            for gid, t, value in zip(senders, times, events[key]):
                nid = net.id_of_gid.get(int(gid))
                if nid is None:
                    continue
                out.setdefault(nid, {}).setdefault(round(float(t), 6), {})[key] = float(value)
    return out


def collect_weight_changes(net) -> list:
    """`[(t_ms, edge_id, weight)]` from the weight recorder, in time order.

    Empty when no recorder was registered or its window never opened, which the adapter
    reports as "learning not recorded" rather than as "no weights changed" -- the two are
    different claims and only one of them is checkable.

    Edge identity comes from the (source, target) endpoints, the same mapping
    `plastic_weights` uses; this topology has no parallel edges.
    """
    recorder = getattr(net, "weight_recorder", None)
    if recorder is None:
        return []
    events = recorder.get("events")
    times, senders = events.get("times", []), events.get("senders", [])
    targets, weights = events.get("targets", []), events.get("weights", [])

    edge_of = {pair: edge_id for edge_id, pair in net.conn_of.items()}
    out = []
    for t, src, tgt, w in zip(times, senders, targets, weights):
        edge_id = edge_of.get((int(src), int(tgt)))
        if edge_id is not None:
            out.append((round(float(t), 6), edge_id, float(w)))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def window_index(t: float, t0: float, period: float) -> int:
    """Which presentation window a spike time falls in. Windows are [t0+kD, t0+(k+1)D)."""
    if period <= 0:
        return 0
    return int(math.floor((t - t0) / period))


def by_role(net, spikes: list) -> dict:
    """role -> {node_id: spike count}. Roles come from `column_role` metadata."""
    out: dict = defaultdict(lambda: defaultdict(int))
    for _t, nid in spikes:
        role = net.node_meta[nid].get("column_role") or net.node_meta[nid]["archetype"]
        out[role][nid] += 1
    return {role: dict(counts) for role, counts in out.items()}


def by_column(net, spikes: list) -> dict:
    """column_id -> role -> spike count."""
    out: dict = defaultdict(lambda: defaultdict(int))
    for _t, nid in spikes:
        meta = net.node_meta[nid]
        column = meta.get("column_id")
        if not column:
            continue
        out[column][meta.get("column_role") or meta["archetype"]] += 1
    return {col: dict(roles) for col, roles in out.items()}


def winner_multiplicity(net, spikes: list, t0: float, period: float) -> dict:
    """The primary WTA observable.

    Returns per-column statistics over presentation windows in which that column produced
    at least one ordinary-E spike. A column that never fires contributes no windows, so a
    silent column cannot flatter the mean.
    """
    per_column_window: dict = defaultdict(lambda: defaultdict(set))
    for t, nid in spikes:
        meta = net.node_meta[nid]
        if meta.get("column_role") != "E":
            continue
        column = meta["column_id"]
        per_column_window[column][window_index(t, t0, period)].add(nid)

    summary: dict = {}
    for column, windows in per_column_window.items():
        counts = [len(winners) for winners in windows.values()]
        summary[column] = {
            "windows_active": len(counts),
            "mean": round(sum(counts) / len(counts), 4) if counts else 0.0,
            "max": max(counts) if counts else 0,
            "min": min(counts) if counts else 0,
            "single_winner_windows": sum(1 for c in counts if c == 1),
            "fraction_single_winner": (
                round(sum(1 for c in counts if c == 1) / len(counts), 4) if counts else None
            ),
        }
    return summary


def aggregate_multiplicity(summary: dict) -> dict:
    """Collapse the per-column winner-multiplicity summary into headline numbers."""
    if not summary:
        return {"columns_active": 0, "mean": None, "max": None,
                "fraction_single_winner": None}
    windows = sum(s["windows_active"] for s in summary.values())
    weighted = sum(s["mean"] * s["windows_active"] for s in summary.values())
    singles = sum(s["single_winner_windows"] for s in summary.values())
    return {
        "columns_active": len(summary),
        "windows_total": windows,
        "mean": round(weighted / windows, 4) if windows else None,
        "max": max(s["max"] for s in summary.values()),
        "fraction_single_winner": round(singles / windows, 4) if windows else None,
    }


def eor_input_multiplicity(net) -> dict:
    """`q_pre / theta` at each `Eor`'s most recent firing.

    This is a snapshot of the LAST firing rather than a per-firing history: the model keeps
    one pre-reset snapshot, and adding a per-spike log would mean recording state NEST does
    not natively carry. It is reported as the corroborating observable, with
    `winner_multiplicity` (which is per-window and complete) as the primary one.
    """
    from .topology import E_THRESHOLD  # noqa: PLC0415

    out = {}
    for nid in net.ids_where(column_role="Eor"):
        cell = net.gid_of[nid]
        q_pre = float(cell.get("q_pre"))
        n_spikes = int(cell.get("n_spikes"))
        out[nid] = {
            "n_spikes": n_spikes,
            "q_pre": q_pre,
            "last_input_multiplicity": (round(q_pre / E_THRESHOLD, 4) if n_spikes else None),
        }
    return out


def coincidence_stats(net) -> dict:
    """Per-`C` deposit / eligibility bookkeeping, split by arrival ORDER.

    `n_deposit_basal_first` is the carried-eligibility path -- the one that was dead in the
    reference engine before the settle-phase fix. Reporting it separately is what makes a
    dead carry visible in recordings instead of having to be inferred.
    """
    out = {}
    for nid in net.ids_where(archetype="e_coincidence"):
        cell = net.gid_of[nid]
        # Afferent COUNTS are reported alongside the event counts, because the two are
        # easy to confuse and the confusion is consequential: `basal_steps` is the number of
        # TIMESTEPS on which the basal port received anything, not the number of basal
        # afferents. A C cell has exactly ONE basal afferent (its column's Eor) and
        # `cc_e_count` apical afferents (every parent-column E competitor), so a low apical
        # step count means the PARENT rarely fired -- never that the cell is under-wired.
        n_basal_aff = sum(1 for e in net.spec["edges"]
                          if e["kind"] == "basal_excitation" and e["target"] == nid)
        n_apical_aff = sum(1 for e in net.spec["edges"]
                           if e["kind"] == "apical_excitation" and e["target"] == nid)
        out[nid] = {
            "has_parent": bool(net.node_meta[nid].get("has_parent")),
            "basal_afferents": n_basal_aff,
            "apical_afferents": n_apical_aff,
            "basal_steps": int(cell.get("n_basal_steps")),
            "apical_steps": int(cell.get("n_apical_steps")),
            "deposits": int(cell.get("n_deposits")),
            "deposit_basal_first": int(cell.get("n_deposit_basal_first")),
            "deposit_apical_first": int(cell.get("n_deposit_apical_first")),
            "deposit_locked_out": int(cell.get("n_deposit_locked_out")),
            "basal_expired": int(cell.get("n_basal_expired")),
            "basal_superseded": int(cell.get("n_basal_superseded")),
            "spikes": int(cell.get("n_spikes")),
            "q": float(cell.get("q")),
        }
    return out


def reset_stats(net) -> dict:
    """Hard resets received per ordinary-E, and volleys emitted / suppressed per `I`."""
    resets = {}
    for nid in net.ids_where(column_role="E"):
        cell = net.gid_of[nid]
        resets[nid] = {
            "resets_received": int(cell.get("n_resets")),
            "spikes": int(cell.get("n_spikes")),
            "q": float(cell.get("q")),
            "q_pre": float(cell.get("q_pre")),
            "blocked_by_refractory": int(cell.get("n_blocked_by_refr")),
        }
    relays = {}
    for nid in net.ids_where(column_role="I"):
        cell = net.gid_of[nid]
        relays[nid] = {
            "spikes": int(cell.get("n_spikes")),
            "suppressed_by_lockout": int(cell.get("n_suppressed")),
        }
    return {"ordinary_e": resets, "i_relay": relays}


def firing_pattern(net, spikes: list, t0: float, period: float, n_windows: int) -> dict:
    """Per-column fire/silent string over presentation windows -- the cadence observable.

    The reference engine predicts strict `1010` alternation when the presentation period
    equals the graph's own feedback loop latency. That is a claim about RELATIVE timing,
    which NEST owns natively, so it should survive a faithful port even though absolute
    spike times will not match.
    """
    active: dict = defaultdict(set)
    for t, nid in spikes:
        meta = net.node_meta[nid]
        if meta.get("column_role") != "E":
            continue
        active[meta["column_id"]].add(window_index(t, t0, period))
    return {
        column: "".join("1" if k in windows else "0" for k in range(n_windows))
        for column, windows in sorted(active.items())
    }


def summarize(net, *, t0: float, period: float, n_windows: int) -> dict:
    """The full per-case record required by prompt section 10."""
    spikes = collect_spikes(net)
    inputs = collect_inputs(net)
    multiplicity = winner_multiplicity(net, spikes, t0, period)
    return {
        "input_events": len(inputs),
        "input_schedule": [{"t": t, "id": nid} for t, nid in inputs],
        "spike_count": len(spikes),
        "spikes": [{"t": t, "id": nid} for t, nid in spikes],
        "by_role": {role: sum(counts.values()) for role, counts in by_role(net, spikes).items()},
        "by_role_detail": by_role(net, spikes),
        "by_column": by_column(net, spikes),
        "winner_multiplicity": multiplicity,
        "winner_multiplicity_overall": aggregate_multiplicity(multiplicity),
        "eor_input_multiplicity": eor_input_multiplicity(net),
        "coincidence": coincidence_stats(net),
        "resets": reset_stats(net),
        "firing_pattern": firing_pattern(net, spikes, t0, period, n_windows),
    }
