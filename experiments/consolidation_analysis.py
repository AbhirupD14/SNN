"""Pure analysis utilities for the Basic L1 consolidation experiment.

Everything here is deterministic and free of engine/dashboard state so it can be unit
tested in isolation:

- :func:`all_nine_patch_input` / :func:`verify_all_nine_input` build and validate the
  81-element input that embeds one 3x3 pattern into all nine declared patches;
- :func:`ordinary_e_ids_by_column` selects exactly ordinary L1 E competitors;
- :func:`assess_ownership` turns a sequence of ordinary-E winner events into a
  consolidation verdict (window / dominance / stable-window rule);
- :func:`capacity_accounting` and :func:`column_mapping_checks` evaluate the one-to-one
  mapping and per-preset capacity.

None of these functions define scientific truth on their own; they implement the operational
definitions declared in the experiment prompt and are configurable by the caller.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from backend.network_spec import embed_patch_pattern

# The four canonical local 3x3 stimuli, imported from production so the harness never
# re-defines them. Kept as a name tuple for explicit, ordered iteration.
CANONICAL_PATTERN_ORDER = ("row 1", "col 1", "diag \\", "diag /")

# Primary operational defaults (all CLI-overridable and recorded in every artifact).
DEFAULT_WINDOW = 50
DEFAULT_DOMINANCE = 0.95
DEFAULT_STABLE_WINDOWS = 3


# ----------------------------------------------------------------- input construction
def _grid_shape(meta: Mapping[str, Any]) -> tuple:
    g = meta["grid_shape"]
    return int(g["rows"]), int(g["cols"])


def all_nine_patch_input(tiled_meta: Mapping[str, Any], pattern_name: str,
                         patterns: Mapping[str, Sequence[int]]) -> list:
    """Embed ``patterns[pattern_name]`` into ALL nine declared 3x3 patches, returning an
    ``input_rows*input_cols`` int vector. Patches are disjoint, so this is a plain union.

    Driven entirely by tiled metadata (input/patch/grid shapes) via
    :func:`embed_patch_pattern`; never a hard-coded flat-index formula.
    """
    ishape = (int(tiled_meta["input_shape"]["rows"]), int(tiled_meta["input_shape"]["cols"]))
    pshape = (int(tiled_meta["patch_shape"]["rows"]), int(tiled_meta["patch_shape"]["cols"]))
    grows, gcols = _grid_shape(tiled_meta)
    local = patterns[pattern_name]
    out = [0] * (ishape[0] * ishape[1])
    for pr in range(grows):
        for pc in range(gcols):
            embedded = embed_patch_pattern(ishape, pshape, (pr, pc), local)
            for i, v in enumerate(embedded):
                if v:
                    out[i] = 1
    return out


def verify_all_nine_input(tiled_meta: Mapping[str, Any], vec: Sequence[int],
                          pattern_name: str, patterns: Mapping[str, Sequence[int]],
                          rgc_nodes: Sequence[Mapping[str, Any]]) -> dict:
    """Assert the five pre-training input invariants. Raises ``AssertionError`` on any
    violation; returns a small report dict on success.

    ``rgc_nodes`` are the RGC source nodes carrying ``patch_id`` and ``pixel`` metadata.
    """
    ishape = (int(tiled_meta["input_shape"]["rows"]), int(tiled_meta["input_shape"]["cols"]))
    pshape = (int(tiled_meta["patch_shape"]["rows"]), int(tiled_meta["patch_shape"]["cols"]))
    grows, gcols = _grid_shape(tiled_meta)
    n = ishape[0] * ishape[1]

    # (1) length matches input shape
    assert len(vec) == n, f"input length {len(vec)} != {n}"

    # (2) every RGC belongs to exactly one patch
    patch_of_pixel: dict = {}
    for node in rgc_nodes:
        pid = node.get("patch_id")
        pix = node.get("pixel")
        assert pid is not None and pix is not None, f"RGC {node.get('id')} missing patch/pixel"
        assert pix not in patch_of_pixel, f"pixel {pix} claimed by two RGCs"
        patch_of_pixel[pix] = pid

    # (3) each of the nine patches contains exactly the selected local pattern, and
    # (4) there is no cross-patch indexing error (pixels outside patches stay 0)
    expected = [0] * n
    for pr in range(grows):
        for pc in range(gcols):
            emb = embed_patch_pattern(ishape, pshape, (pr, pc), patterns[pattern_name])
            for i, v in enumerate(emb):
                if v:
                    expected[i] = 1
    assert list(int(x) for x in vec) == expected, "input does not match all-nine embedding"

    n_patches = grows * gcols
    return {"n_pixels": n, "n_patches": n_patches,
            "active_pixels": int(sum(1 for x in vec if x))}


# ----------------------------------------------------------------- candidate selection
def ordinary_e_ids_by_column(role_of: Mapping[str, Optional[str]],
                             column_of: Mapping[str, Optional[str]],
                             layer_of: Mapping[str, Optional[str]],
                             layer: str = "L1") -> dict:
    """Return ``{column_id: [ordinary-E neuron ids]}`` for the requested layer.

    Selection is by explicit node metadata only: role == 'E' (ordinary competitor, never
    Eor/C/I), the requested layer, and a real column id. RGC/Eor/C/I/L2 are excluded.
    """
    out: dict = {}
    for nid, role in role_of.items():
        if role != "E":
            continue
        if layer_of.get(nid) != layer:
            continue
        cid = column_of.get(nid)
        if cid is None:
            continue
        out.setdefault(cid, []).append(nid)
    for cid in out:
        out[cid].sort()
    return out


# ----------------------------------------------------------------- ownership assessment
@dataclass
class OwnershipVerdict:
    consolidated: bool
    owner: Optional[str]
    reason: str
    n_events: int
    windows: list = field(default_factory=list)       # per-window {owner, dominance, count}
    final_window_dominance: Optional[float] = None
    strict_unanimous_final: bool = False
    runner_up: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "consolidated": self.consolidated, "owner": self.owner, "reason": self.reason,
            "n_events": self.n_events, "windows": self.windows,
            "final_window_dominance": self.final_window_dominance,
            "strict_unanimous_final": self.strict_unanimous_final,
            "runner_up": self.runner_up,
        }


def assess_ownership(events: Sequence[str], *, window: int = DEFAULT_WINDOW,
                     dominance: float = DEFAULT_DOMINANCE,
                     stable_windows: int = DEFAULT_STABLE_WINDOWS) -> OwnershipVerdict:
    """Decide whether a column has a stable owner for the current pattern.

    ``events`` is the ordered sequence of ordinary-E *winner events* (neuron ids), one per
    boundary on which that column produced an ordinary-E winner — silent boundaries are not
    events. Consolidation requires the ``stable_windows`` most recent non-overlapping
    windows of ``window`` events to each be dominated (>= ``dominance``) by the *same*
    owner. Evaluating on the trailing events means a later turnover un-consolidates a
    column (no permanent latch of the first apparent owner).
    """
    if window < 1 or stable_windows < 1:
        raise ValueError("window and stable_windows must be >= 1")
    n = len(events)
    need = window * stable_windows
    if n < need:
        return OwnershipVerdict(False, None, "insufficient_events", n)

    tail = list(events[-need:])
    windows_meta = []
    owners = []
    for k in range(stable_windows):
        chunk = tail[k * window:(k + 1) * window]
        counts = Counter(chunk)
        owner, cnt = counts.most_common(1)[0]
        dom = cnt / window
        windows_meta.append({"owner": owner, "dominance": round(dom, 4), "count": cnt})
        owners.append((owner, dom))

    final_owner, final_dom = owners[-1]
    all_dominant = all(dom >= dominance for _, dom in owners)
    same_owner = len({o for o, _ in owners}) == 1
    consolidated = all_dominant and same_owner
    # runner-up in the final window (for the confusion table)
    final_counts = Counter(tail[(stable_windows - 1) * window:])
    ranked = final_counts.most_common(2)
    runner_up = ranked[1][0] if len(ranked) > 1 else None

    reason = "consolidated" if consolidated else (
        "owner_turnover" if all_dominant and not same_owner else "below_dominance")
    return OwnershipVerdict(
        consolidated=consolidated,
        owner=final_owner if consolidated else None,
        reason=reason, n_events=n, windows=windows_meta,
        final_window_dominance=round(final_dom, 4),
        strict_unanimous_final=(final_dom >= 1.0),
        runner_up=runner_up)


# ----------------------------------------------------------------- mapping / capacity
def capacity_accounting(owners_by_pattern: Mapping[str, Optional[str]],
                        n_competitors: int) -> dict:
    """Distinct-owner and assigned/unassigned accounting for one column.

    ``owners_by_pattern`` maps each pattern to its consolidated owner (or ``None``). A
    ``None`` owner means that pattern did not consolidate and cannot be counted.
    """
    assigned = [o for o in owners_by_pattern.values() if o is not None]
    distinct = sorted(set(assigned))
    n_patterns = len(owners_by_pattern)
    return {
        "n_competitors": int(n_competitors),
        "n_patterns": n_patterns,
        "n_consolidated": len(assigned),
        "distinct_owners": distinct,
        "n_distinct_owners": len(distinct),
        "assigned_competitors": len(distinct),
        "unassigned_competitors": int(n_competitors) - len(distinct),
        "all_owners_distinct": len(distinct) == len(assigned) and len(assigned) == n_patterns,
    }


def column_mapping_checks(owners_by_pattern: Mapping[str, Optional[str]],
                          n_competitors: int, *, expected_patterns: int = 4) -> dict:
    """Per-column one-to-one and capacity checks.

    Returns a dict of named boolean checks plus the capacity accounting. The column passes
    only when every check is True.
    """
    cap = capacity_accounting(owners_by_pattern, n_competitors)
    all_consolidated = (cap["n_consolidated"] == expected_patterns
                        and cap["n_patterns"] == expected_patterns)
    distinct = cap["all_owners_distinct"]
    # a mapping collision = two patterns sharing an owner within this column
    owner_counts = Counter(o for o in owners_by_pattern.values() if o is not None)
    collisions = {o: c for o, c in owner_counts.items() if c > 1}
    # per-preset capacity expectation
    unassigned = cap["unassigned_competitors"]
    if n_competitors == expected_patterns:
        capacity_ok = (cap["n_distinct_owners"] == expected_patterns and unassigned == 0)
    else:
        capacity_ok = (cap["n_distinct_owners"] == expected_patterns
                       and unassigned == n_competitors - expected_patterns)

    checks = {
        "all_patterns_consolidated": all_consolidated,
        "each_pattern_one_owner": all_consolidated,   # one owner per consolidated pattern
        "owners_distinct": distinct,
        "no_collision": len(collisions) == 0,
        "capacity_ok": capacity_ok,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "capacity": cap,
        "collisions": collisions,
    }
