"""Pure analysis utilities for the two-tower V/A/7 composition experiment (Task 4).

Everything here is deterministic and free of engine/dashboard state so it can be unit
tested in isolation:

- glyph geometry (:func:`glyph_coordinates`, :func:`glyph_vector`, :func:`glyph_ascii`,
  :func:`verify_glyphs`) for the ``9x18`` two-tower input sheet;
- structural one-event maturity from rest (:func:`active_charge`,
  :func:`one_event_maturity`);
- observed feedforward throughput reliability (:class:`ThroughputTracker`);
- C one-shot readiness bookkeeping (:func:`c_readiness`);
- the L3 representation-separability classification (:class:`L3Signature`,
  :func:`classify_signatures`);
- the shared failure taxonomy and replay slug helper.

None of these functions define scientific truth on their own; they implement the
operational definitions declared in ``prompts/Claude_Final_Experiments_Prompt.md`` Task 4
and are configurable by the caller.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from backend.network_spec import (
    SHEET_GLYPH_SHEETS, sheet_glyph_bank, sheet_glyph_coordinates, sheet_glyph_names,
)

# ------------------------------------------------------------------ glyph geometry
# One validated 9x18 sheet holding two lateral 9x9 fields. Pixel = row * 18 + col, so
# columns 0..8 belong to tower 0 (left) and columns 9..17 to tower 1 (right). Pixel
# ownership is disjoint: a coordinate belongs to exactly one field.
GLYPH_ROWS = 9
GLYPH_COLS = 18
GLYPH_SEAM = GLYPH_COLS // 2                # first column of the right field

# The glyph geometry lives in ``backend.network_spec`` (the topology's own whole-sheet
# stimulus bank, which the dashboard also drives). This module never re-defines it: it
# re-exports and adds the presentation / verification helpers the experiment needs, so the
# report, the tests and the live pattern buttons can never drift apart.
GLYPH_STROKES = SHEET_GLYPH_SHEETS[(GLYPH_ROWS, GLYPH_COLS)]
CANONICAL_GLYPH_ORDER = sheet_glyph_names(GLYPH_ROWS, GLYPH_COLS)


def glyph_coordinates(glyph: str) -> list:
    """Sorted, de-duplicated ``(row, col)`` coordinates of one glyph."""
    return sheet_glyph_coordinates(GLYPH_ROWS, GLYPH_COLS, glyph)


def glyph_vector(glyph: str) -> list:
    """The ``9*18`` binary input vector (row-major pixel order) for one glyph."""
    bank = sheet_glyph_bank(GLYPH_ROWS, GLYPH_COLS)
    if glyph not in bank:
        raise KeyError(f"unknown glyph {glyph!r}; expected one of {CANONICAL_GLYPH_ORDER}")
    return bank[glyph]


def glyph_ascii(glyph: str, *, on: str = "#", off: str = ".", seam: str = "|") -> list:
    """One ASCII row per sheet row, with an explicit seam marker between the two fields."""
    coords = set(glyph_coordinates(glyph))
    rows = []
    for r in range(GLYPH_ROWS):
        left = "".join(on if (r, c) in coords else off for c in range(GLYPH_SEAM))
        right = "".join(on if (r, c) in coords else off
                        for c in range(GLYPH_SEAM, GLYPH_COLS))
        rows.append(f"{left}{seam}{right}")
    return rows


def glyph_patches(glyph: str, patch_rows: int = 3, patch_cols: int = 3) -> list:
    """Sorted ``(patch_row, patch_col)`` patches this glyph actually activates."""
    return sorted({(r // patch_rows, c // patch_cols)
                   for r, c in glyph_coordinates(glyph)})


def glyph_half_counts(glyph: str) -> dict:
    """Active-pixel count in each 9x9 field."""
    coords = glyph_coordinates(glyph)
    return {"left": sum(1 for _r, c in coords if c < GLYPH_SEAM),
            "right": sum(1 for _r, c in coords if c >= GLYPH_SEAM)}


def verify_glyphs(glyphs: Sequence[str] = CANONICAL_GLYPH_ORDER) -> dict:
    """Assert the declared pre-training glyph invariants. Raises ``AssertionError`` on any
    violation; returns a small report dict on success.

    Checks: in-sheet coordinates, correct half membership, mutually distinct vectors,
    every active pixel in exactly one patch, and the stored ASCII matching the vector.
    """
    report: dict = {}
    seen_vectors: dict[tuple, str] = {}
    for g in glyphs:
        coords = glyph_coordinates(g)
        assert coords, f"glyph {g!r} has no active pixels"
        for r, c in coords:
            assert 0 <= r < GLYPH_ROWS and 0 <= c < GLYPH_COLS, \
                f"glyph {g!r} coordinate ({r},{c}) outside the {GLYPH_ROWS}x{GLYPH_COLS} sheet"
        halves = glyph_half_counts(g)
        assert halves["left"] + halves["right"] == len(coords), \
            f"glyph {g!r} half counts do not partition its pixels"

        vec = glyph_vector(g)
        assert sum(vec) == len(coords), f"glyph {g!r} vector/coordinate count mismatch"
        key = tuple(vec)
        assert key not in seen_vectors, \
            f"glyph {g!r} is identical to {seen_vectors.get(key)!r}"
        seen_vectors[key] = g

        # every active pixel belongs to exactly one declared patch
        patch_of: dict[int, tuple] = {}
        for r, c in coords:
            pix = r * GLYPH_COLS + c
            assert pix not in patch_of, f"pixel {pix} claimed twice in glyph {g!r}"
            patch_of[pix] = (r // 3, c // 3)

        # the stored picture must match the binary vector exactly
        art = glyph_ascii(g)
        assert len(art) == GLYPH_ROWS, f"glyph {g!r} ASCII row count"
        for r, line in enumerate(art):
            cells = line.replace("|", "")
            assert len(cells) == GLYPH_COLS, f"glyph {g!r} ASCII row {r} width"
            for c, ch in enumerate(cells):
                assert (ch == "#") == bool(vec[r * GLYPH_COLS + c]), \
                    f"glyph {g!r} ASCII disagrees with the vector at ({r},{c})"

        report[g] = {"coordinates": [list(x) for x in coords],
                     "n_active": len(coords),
                     "half_counts": halves,
                     "patches": [list(p) for p in glyph_patches(g)],
                     "ascii": art}
    return report


# --------------------------------------------------- structural one-event maturity
def active_charge(cell: Any, active_source_ids: Iterable[str]) -> float:
    """Total live feedforward charge this cell would receive from one full volley of
    ``active_source_ids`` (pre-update weights, raw delivery -- the distance factor is a
    LEARNING-RATE multiplier in this engine, never a delivery scale)."""
    active = set(active_source_ids)
    return sum(float(w) for src, w in zip(cell.ff_src, cell.acc_weights) if src in active)


def one_event_maturity(cell: Any, active_source_ids: Iterable[str], *,
                       tol: float = 1e-9) -> dict:
    """Structural counterfactual from rest: can ONE delivered packet from
    ``active_source_ids`` cross this cell's threshold?

    Valid only under the declared regime (``leak_rate=0``, no inhibitory conductance, a
    resting target). Deliberately separate from observed live response, which is also
    affected by retained V, inhibition, refractory state and event coalescing.
    """
    active = sorted(set(active_source_ids))
    q = active_charge(cell, active)
    thr = float(cell.threshold)
    components = {src: round(float(w), 6)
                  for src, w in zip(cell.ff_src, cell.acc_weights) if src in set(active)}
    return {"cell": cell.id, "active_charge": round(q, 6), "threshold": round(thr, 6),
            "margin": round(q - thr, 6), "mature": bool(q >= thr - tol),
            "active_sources": active, "components": components}


# --------------------------------------------------- observed throughput reliability
class ThroughputTracker:
    """Causal source-event -> delivery-boundary reliability for one (source, target) path.

    The engine schedules a plastic feedforward delivery at ``t+1`` for a source spike at
    ``t``, so a source event at boundary ``t`` is *eligible* on delivery boundary ``t+1``.
    Several source events may coalesce onto one delivery boundary; both denominators are
    preserved and reliability uses delivery boundaries.
    """

    def __init__(self, window: int = 50):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.eligible_source_events = 0
        self._pending: set = set()                 # delivery boundaries not yet resolved
        self._delivered: deque = deque(maxlen=self.window)   # bool: target spiked
        self.total_delivery_boundaries = 0
        self.total_target_spikes = 0
        self.taus: deque = deque(maxlen=self.window)

    def note_source_event(self, boundary: int) -> None:
        """Record a causal source spike at ``boundary`` (delivery lands at ``boundary+1``)."""
        self.eligible_source_events += 1
        self._pending.add(int(boundary) + 1)

    def note_boundary(self, boundary: int, target_spiked: bool,
                      tau: Optional[float] = None) -> bool:
        """Resolve ``boundary``. Returns True when it was an eligible delivery boundary."""
        b = int(boundary)
        if b not in self._pending:
            return False
        self._pending.discard(b)
        self.total_delivery_boundaries += 1
        self._delivered.append(bool(target_spiked))
        if target_spiked:
            self.total_target_spikes += 1
            if tau is not None:
                self.taus.append(float(tau))
        return True

    @property
    def recent_delivery_boundaries(self) -> int:
        return len(self._delivered)

    @property
    def recent_target_spikes(self) -> int:
        return sum(1 for x in self._delivered if x)

    def reliability(self) -> Optional[float]:
        """Target-spike fraction over the most recent ``window`` eligible delivery
        boundaries, or None when none have been observed yet."""
        if not self._delivered:
            return None
        return self.recent_target_spikes / len(self._delivered)

    def ready(self, *, min_boundaries: int = 50, threshold: float = 0.95) -> bool:
        r = self.reliability()
        return (r is not None and len(self._delivered) >= min_boundaries
                and r >= threshold)

    def report(self, *, min_boundaries: int = 50, threshold: float = 0.95) -> dict:
        taus = sorted(self.taus)
        med = None if not taus else (
            taus[len(taus) // 2] if len(taus) % 2
            else 0.5 * (taus[len(taus) // 2 - 1] + taus[len(taus) // 2]))
        return {
            "eligible_source_events": self.eligible_source_events,
            "eligible_delivery_boundaries": self.total_delivery_boundaries,
            "recent_delivery_boundaries": self.recent_delivery_boundaries,
            "recent_target_spike_boundaries": self.recent_target_spikes,
            "target_spike_boundaries": self.total_target_spikes,
            "response_reliability": (None if self.reliability() is None
                                     else round(self.reliability(), 6)),
            "median_tau": (None if med is None else round(med, 9)),
            "min_tau": (None if not taus else round(taus[0], 9)),
            "max_tau": (None if not taus else round(taus[-1], 9)),
            "window": self.window,
            "ready": self.ready(min_boundaries=min_boundaries, threshold=threshold),
        }


# --------------------------------------------------------------- C readiness
def c_readiness(basal_weight: float, threshold: float, *, deposits: int, spikes: int,
                updates: int, opportunities: int = 0, tol: float = 1e-9) -> dict:
    """Coincidence-cell one-shot readiness under ``leak_rate=0``.

    Requires at least one committed deposit, at least one C spike or drained learning
    event, and a live basal weight meeting the impulse one-shot condition
    ``w >= theta``. Opportunity / deposit / spike / update counts stay separate: the
    absence of a spike is never evidence that no opportunity occurred.
    """
    impulse_ok = float(basal_weight) >= float(threshold) - tol
    return {
        "basal_weight": round(float(basal_weight), 6),
        "threshold": round(float(threshold), 6),
        "margin": round(float(basal_weight) - float(threshold), 6),
        "impulse_one_shot": bool(impulse_ok),
        "opportunities": int(opportunities),
        "deposits": int(deposits),
        "spikes": int(spikes),
        "updates": int(updates),
        "ready": bool(impulse_ok and deposits >= 1 and (spikes >= 1 or updates >= 1)),
    }


# ------------------------------------------------- L3 representation separability
@dataclass
class L3Signature:
    """Everything the graph actually exposes to L3 for one glyph.

    ``events`` are ordered ``(relative_delivery_boundary, source_L2_Eor_id,
    delivered_charge)`` tuples; ``taus`` is the sub-boundary spike time of the source
    event that scheduled each delivery, in the same order.
    """
    glyph: str
    events: list = field(default_factory=list)
    taus: list = field(default_factory=list)
    observed_boundaries: int = 0
    source_ids: list = field(default_factory=list)

    def exact_trace(self) -> tuple:
        return tuple((int(b), str(s), round(float(q), 6)) for b, s, q in self.events)

    def truncated(self, boundaries: int) -> "L3Signature":
        keep = [(b, s, q) for (b, s, q) in self.events if b <= boundaries]
        taus = [t for (b, _s, _q), t in zip(self.events, self.taus) if b <= boundaries]
        return L3Signature(self.glyph, keep, taus, boundaries, list(self.source_ids))

    def late(self, frac: float = 0.5) -> "L3Signature":
        """The trailing ``frac`` of the observation window (startup transients removed)."""
        cut = int(math.floor(self.observed_boundaries * (1.0 - frac)))
        keep = [(b, s, q) for (b, s, q) in self.events if b > cut]
        taus = [t for (b, _s, _q), t in zip(self.events, self.taus) if b > cut]
        sig = L3Signature(self.glyph, keep, taus, self.observed_boundaries,
                          list(self.source_ids))
        return sig

    def summary(self) -> dict:
        """Looser summaries: per-source event counts, inter-event-interval distribution,
        and per-boundary source co-occurrence. Exact traces can differ by a startup
        transient alone, so these are reported alongside the strict comparison."""
        per_source = Counter(s for _b, s, _q in self.events)
        by_boundary: dict = {}
        for b, s, _q in self.events:
            by_boundary.setdefault(b, set()).add(s)
        cooccurrence = Counter(tuple(sorted(v)) for v in by_boundary.values())
        ieis: dict = {}
        for src in sorted(per_source):
            bs = sorted(b for b, s, _q in self.events if s == src)
            ieis[src] = Counter(b2 - b1 for b1, b2 in zip(bs, bs[1:]))
        return {
            "n_events": len(self.events),
            "observed_boundaries": self.observed_boundaries,
            "source_participation": sorted(per_source),
            "events_per_source": {k: int(v) for k, v in sorted(per_source.items())},
            "cooccurrence_by_boundary": {"+".join(k): int(v)
                                         for k, v in sorted(cooccurrence.items())},
            "inter_event_intervals": {k: {str(i): int(n) for i, n in sorted(v.items())}
                                      for k, v in ieis.items()},
        }


def classify_signatures(signatures: Mapping[str, L3Signature], *,
                        late_frac: float = 0.5) -> dict:
    """Compare the L3 input signatures of every glyph pair.

    Verdicts:

    ``identical``
        every pair's complete ordered tuple stream matches over the common window --
        L3 cannot distinguish the glyphs at the engine's own resolution.
    ``transient_only_separability``
        some exact traces differ, but every pair's LATE steady summary matches. Not a
        robustly identifiable representation.
    ``separable``
        at least one pair differs in its late steady summary.

    ``source_collision`` reports the stricter structural fact: whether all glyphs present
    the same set of L3 source identities.
    """
    names = list(signatures)
    if len(names) < 2:
        raise ValueError("need at least two signatures to classify separability")
    common = min(s.observed_boundaries for s in signatures.values())

    pairs = []
    exact_all_match = True
    late_all_match = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = signatures[a].truncated(common), signatures[b].truncated(common)
            exact = sa.exact_trace() == sb.exact_trace()
            la, lb = signatures[a].late(late_frac).summary(), signatures[b].late(late_frac).summary()
            late_match = (la["events_per_source"] == lb["events_per_source"]
                          and la["cooccurrence_by_boundary"] == lb["cooccurrence_by_boundary"]
                          and la["inter_event_intervals"] == lb["inter_event_intervals"])
            exact_all_match &= exact
            late_all_match &= late_match
            pairs.append({"pair": [a, b], "exact_trace_match": bool(exact),
                          "late_summary_match": bool(late_match),
                          "n_events": [len(sa.events), len(sb.events)]})

    if exact_all_match:
        verdict = "identical"
    elif late_all_match:
        verdict = "transient_only_separability"
    else:
        verdict = "separable"

    participation = {g: sorted({s for _b, s, _q in sig.events})
                     for g, sig in signatures.items()}
    distinct_participation = {tuple(v) for v in participation.values()}
    return {
        "verdict": verdict,
        "identifiable": verdict == "separable",
        "common_observation_boundaries": common,
        "late_fraction": late_frac,
        "pairs": pairs,
        "source_participation": participation,
        "source_collision": len(distinct_participation) == 1,
        "declared_l3_source_ids": sorted(
            {s for sig in signatures.values() for s in sig.source_ids}),
    }


# --------------------------------------------------------------- failure taxonomy
# The stable vocabulary declared in the experiments prompt. Every recorded failure must
# use one of these taxa (``timeout_unclassified`` only when nothing narrower applies).
FAILURE_TAXONOMY = (
    "no_firing_or_bootstrap_deadlock",
    "stable_but_nonunique_owner",
    "incumbent_absorbs_multiple_patterns",
    "transition_residual_charge",
    "learning_event_starvation",
    "Eor_maturation_bottleneck",
    "C_coincidence_starvation",
    "L2_evidence_collapse",
    "L2_mapping_collision",
    "recall_drift_or_forgetting",
    "order_sensitive_tie",
    "capacity_exhaustion",
    "representation_not_identifiable",
    "nonfinite_or_numerical_failure",
    "timeout_unclassified",
)


def failure_record(taxon: str, **fields) -> dict:
    """One structured failure-catalog entry. Rejects any taxon outside the vocabulary."""
    if taxon not in FAILURE_TAXONOMY:
        raise ValueError(f"unknown failure taxon {taxon!r}; expected one of {FAILURE_TAXONOMY}")
    rec = {"taxon": taxon}
    rec.update(fields)
    return rec


# --------------------------------------------------------------- replay slugs
_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def condition_slug(prefix: str, **parts) -> str:
    """Deterministic replay-subdirectory slug, e.g.
    ``condition_slug('composition', condition='reference', B=5.0, seed=1)`` ->
    ``composition-condition_reference-b_5p0-seed_1``. Dots become ``p`` so the slug stays
    one filesystem-safe token."""
    def norm(x):
        s = str(x).strip().lower().replace(".", "p")
        s = _SLUG_BAD.sub("_", s).strip("_")
        return s or "na"

    tail = "-".join(f"{norm(k)}_{norm(v)}" for k, v in parts.items())
    return f"{norm(prefix)}-{tail}" if tail else norm(prefix)


class SlugRegistry:
    """Rejects duplicate replay slugs inside one parent run."""

    def __init__(self):
        self._seen: set = set()

    def claim(self, slug: str) -> str:
        if slug in self._seen:
            raise ValueError(f"duplicate replay slug {slug!r}")
        self._seen.add(slug)
        return slug
