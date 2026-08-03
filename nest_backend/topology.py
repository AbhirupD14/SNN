"""Translate the canonical `tiled_cc` NetworkSpec into a NEST network.

Scope (prompt section 4): ONLY `backend.network_spec.tiled_cc_spec(cc_e_count=8)` -- the
9x9 RGC surface, nine 3x3 patches, nine L1 columns, one L2 column, eight ordinary-E
competitors per column. 191 nodes, 1052 directed edges. No other preset is translated.

Nothing here hand-writes a second copy of the graph: nodes, edges, roles, projections and
column metadata all come from the spec, and the initial weights come from the validated
Python engine itself, so "frozen" really means "frozen at the engine's own deterministic
initialization" rather than at a re-derived approximation of it.

Timing policy (prompt section 5) is implemented in `Timescales` and `assign_delays`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict, field, replace

from nest_backend.profiles import (  # noqa: E402
    PROFILE_CIPP_CONTINUOUS, PROFILE_IMPULSE, EngineProfile, get_profile,
)
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reference parameter set (`Current_Implementation_Methodology_Equations.md` section 9).
E_THRESHOLD = 1000.0
I_THRESHOLD_FRAC = 1.0 / 3.0

# Profile names and configuration live in `nest_backend.profiles` and are imported above.
# There is deliberately NO module-global "current profile": it is instance state on the
# network, so two networks can declare different profiles in one process without hidden
# shared state.

# The NESTML synapse models that carry a plastic `w`. Named once here because
# `plastic_weights` queries the kernel BY MODEL rather than edge by edge, and the two lists
# would otherwise drift apart silently.
PLASTIC_SYNAPSE_MODELS = (
    "plastic_feedforward_synapse__with_event_accumulator",
    "plastic_basal_synapse__with_event_coincidence",
)

# Pathways that receive geometric arrival dispersion. Only the two genuinely
# spatial feedforward projections do: an RGC surface projecting into a column, and a
# column projecting into its parent. Within-column pathways are compact by construction
# and are given the flat base hop.
DISPERSED_PROJECTIONS = ("rg_to_column", "column_to_column_ff")


class LeakNotSupported(ValueError):
    """Raised when a configuration would invalidate the event-only node models.

    The NESTML models are an EXACT port only while `leak_rate == 0` and no persistent
    inhibitory conductance is in play, because that is what degenerates the reference
    conductance-LIF membrane to a pure integrator (see `event_accumulator.nestml`).
    Translating anything else would silently replace a scientific equation.
    """


@dataclass(frozen=True)
class Timescales:
    """The three declared timescales of prompt section 5.1, plus the base hop.

    The physical claim being encoded is that local lateral inhibition closes far faster
    than a cell integrates its feedforward volley. NEST has no zero delay -- the floor is
    the resolution `h` -- so the separation is expressed as an explicit ratio and recorded
    with every run, rather than idealised away.

    Defaults give `L_wta : S : D = 0.2 : 2.0 : 20.0 = 1 : 10 : 100`.
    """

    h: float = 0.1               # kernel resolution (ms); the minimum any delay may take
    base_ff: float = 1.0         # flat conduction delay of one feedforward hop (ms)
    spread: float = 2.0          # S: target arrival spread within one volley (ms)
    presentation: float = 20.0   # D: presentation interval, volley to volley (ms)

    @property
    def l_wta(self) -> float:
        """E -> I -> E lateral inhibition loop: one `h` per hop, the minimum possible."""
        return 2.0 * self.h

    @property
    def ratio_s_over_lwta(self) -> float:
        return self.spread / self.l_wta

    @property
    def ratio_d_over_s(self) -> float:
        return self.presentation / self.spread if self.spread > 0 else math.inf

    def quantize(self, delay_ms: float) -> float:
        """Round a delay up onto the `h` grid, never below `h` itself."""
        steps = max(1, int(round(delay_ms / self.h)))
        return round(steps * self.h, 10)

    def validate(self) -> None:
        if self.h <= 0:
            raise ValueError(f"resolution h must be positive, got {self.h}")
        if self.base_ff < self.h:
            raise ValueError(f"base_ff {self.base_ff} is below the resolution {self.h}")
        if self.presentation <= self.spread:
            raise ValueError(
                f"presentation D={self.presentation} must exceed the spread S={self.spread}"
            )

    def describe(self) -> dict:
        return {
            "h_ms": self.h,
            "L_wta_ms": self.l_wta,
            "S_target_ms": self.spread,
            "D_ms": self.presentation,
            "base_ff_ms": self.base_ff,
            "ratio_S_over_Lwta": round(self.ratio_s_over_lwta, 4),
            "ratio_D_over_S": round(self.ratio_d_over_s, 4),
        }


@dataclass
class DelayAssignment:
    """Per-edge delay plus the per-pathway dispersion bookkeeping it came from."""

    by_edge: dict = field(default_factory=dict)          # edge_id -> delay (ms)
    realized_spread: dict = field(default_factory=dict)  # projection -> realized S (ms)
    gain: dict = field(default_factory=dict)             # projection -> ms per unit distance
    histogram: dict = field(default_factory=dict)        # projection -> {delay: count}


def reference_overrides(seed: int = 1) -> dict:
    """The documented reference configuration, restricted to what this prototype scopes.

    Learning rates are included even though Phases 0-3 run frozen: they determine nothing
    about the initial weights, but recording them keeps the manifest honest about which
    configuration the frozen weights were initialized under.
    """
    return {
        "topology": "tiled_cc",
        "cc_e_count": 8,
        "dual_fe_fes": True,
        "eta": 4.0,
        "c_eta": 16.0,
        "leak_rate": 0.0,
        "refractory_steps": 0,
        "input_period": 0,
        "e_threshold": E_THRESHOLD,
        "c_feedback_reset": True,
        # Reference learning constants (equations doc section 9). They are inert while
        # `learning=False`, but the plastic synapses read them, so they are pinned here
        # rather than left to the engine defaults (which are the older linear-rule values).
        "dual_fe_B": 5.0,
        "dual_fe_e": 0.001,
        "dual_fe_wte": 0.001,
        "e_weight_cap_frac": 0.5,
        "relay_weight_cap_frac": 1.0,
        "eor_plasticity_enabled": False,
    }


def build_reference_engine(seed: int = 1, shape: tuple | None = None):
    """Instantiate the validated Python engine purely as a WEIGHT AND GEOMETRY SOURCE.

    This is not a co-simulation and the engine is never stepped here. Reading its
    deterministic initialization is what makes "structural parity" and "equation parity"
    checkable: the NEST network starts from byte-identical weights, so any behavioural
    difference is attributable to timing rather than to a re-derived initializer.
    """
    from backend.simulation import SimulationEngine  # noqa: PLC0415

    from backend.network_spec import tiled_cc_spec  # noqa: PLC0415

    overrides = reference_overrides(seed)
    engine = SimulationEngine(seed=seed, **overrides)
    if shape is not None:
        # A SMALLER member of the same canonical family, from the same spec function --
        # not a different topology. `input_rows=patch_rows=3` gives exactly one L1 column
        # over one 3x3 patch plus the L2 column: 31 nodes, 140 edges, the full
        # E -> Eor -> L2E -> apical -> C -> I hierarchy, and the same graph-derived loop
        # latency L = 3. Everything the 9x9 tests can be tested here, faster and with the
        # patch/tiling confound removed.
        rows, cols = shape
        # The PATCH stays 3x3 -- that is the canonical local feature size and the size the
        # four 3x3 patterns are defined on. Only the input SHEET changes, which is what
        # sets the number of L1 columns: 3x3 -> one column, 3x6 -> two, 9x9 -> nine.
        engine.apply_topology(tiled_cc_spec(cc_e_count=8, input_rows=rows, input_cols=cols,
                                            patch_rows=3, patch_cols=3))
    if float(engine.params["leak_rate"]) != 0.0:
        raise LeakNotSupported(
            f"leak_rate={engine.params['leak_rate']} -- the NESTML event models are an "
            "exact port only at leak_rate=0 (pure integrator). Refusing to translate."
        )
    return engine


def extract_weights(engine) -> dict:
    """edge_id -> initial weight, for every weighted edge kind in the spec.

    Unweighted edge kinds (apical permission, hard-reset inhibition) are deliberately
    absent: their delivered value is ignored by the receiving model, and inventing a
    weight for them would imply a scientific quantity that does not exist.
    """
    weights: dict = {}
    for edge_id, (cell, widx) in engine._ff_weight_ref.items():
        weights[edge_id] = float(cell.acc_weights[widx])
    for edge_id, (cell, bidx) in engine._basal_weight_ref.items():
        weights[edge_id] = float(cell.basal.weights[bidx])
    return weights


def extract_phi(engine) -> dict:
    """edge_id -> the per-synapse distance influence phi_i, from the engine itself.

    Geometry is a LEARNING-RATE multiplier only and never scales delivered charge
    (`Current_Implementation_Methodology_Equations.md` section 4.3). Reading it off the
    engine keeps the per-target `d_ref` normalisation exactly as the reference computed it,
    instead of re-deriving a formula that could drift.
    """
    phi: dict = {}
    for edge_id, (cell, widx) in engine._ff_weight_ref.items():
        phi[edge_id] = float(cell.acc_distance_factor[widx])
    for edge_id, (cell, bidx) in engine._basal_weight_ref.items():
        phi[edge_id] = float(cell.basal.distance_factor[bidx]) \
            if hasattr(cell.basal, "distance_factor") else 1.0
    return phi


def node_positions(engine) -> dict:
    """node id -> 3D position, from the engine's own seeded functional layout."""
    return {nid: [float(x) for x in pos] for nid, pos in engine.pos.items()}


def _distance(pos: dict, a: str, b: str) -> float:
    pa, pb = pos.get(a), pos.get(b)
    if pa is None or pb is None:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def assign_uniform_delays(spec: dict, profile) -> DelayAssignment:
    """Every equivalent edge in one projection gets the SAME delay, from the profile.

    The reference `cipp_continuous` policy. Geometry never reaches this function: layout
    distance keeps its one legitimate role as the learning-rate multiplier `phi`, and
    delivery time comes from the declared per-projection constant, ceiling-quantized so a
    delay is never shortened below what was asked for.

    Phase 1 is why this is safe: with a continuous competitor, latency after arrival varied
    by 1.8e-15 ms across a 16x range of conduction delay, so total drive alone orders the
    crossings. The impulse profile needed dispersion to break ties; this one does not.
    """
    by_class = {
        "rg_to_column": profile.delays.rg_to_column_ms,
        "column_to_column_ff": profile.delays.column_to_column_ff_ms,
        "column_e_to_eor": profile.delays.column_e_to_eor_ms,
        "column_e_to_i": profile.delays.column_e_to_i_ms,
        "column_i_to_e": profile.delays.column_i_to_e_ms,
        "column_c_to_i": profile.delays.column_c_to_i_ms,
        "column_to_column_apical": profile.delays.column_to_column_apical_ms,
        "column_eor_to_c_basal": profile.delays.column_eor_to_c_basal_ms,
    }
    assignment = DelayAssignment()
    for edge in spec["edges"]:
        projection = edge.get("projection")
        if projection not in by_class:
            raise ValueError(
                f"projection {projection!r} has no declared delay in the profile; "
                f"a reference profile may not fall back to geometry"
            )
        assignment.by_edge[edge["id"]] = profile.quantize(by_class[projection])
    for projection, requested in by_class.items():
        quantized = profile.quantize(requested)
        assignment.realized_spread[projection] = {
            "per_target_min_ms": quantized, "per_target_max_ms": quantized,
            "per_target_median_ms": quantized, "population_ms": 0.0,
            "requested_ms": requested,
        }
        assignment.gain[projection] = 0.0     # no distance term at all
    return assignment


def assign_delays(spec: dict, pos: dict, ts: Timescales,
                  *, dispersion_enabled: bool = True,
                  jitter_ms: float = 0.0, seed: int = 1) -> DelayAssignment:
    """Assign every edge a NEST-valid delay per prompt section 5.3.

        delay(e) = base_hop(kind) + round(gain * (d(e) - d_min) / h) * h

    `gain` is solved per dispersed pathway so the REALIZED spread matches the target `S`.
    It is not derived blindly from raw distance: the layout separates layers on `z`, so
    inter-layer distance is dominated by the layer gap and the informative within-patch
    variation is the smaller term riding on top of it. Normalising per pathway isolates
    that variation, and the realized spread is recorded rather than assumed.

    Setting delays from geometry does NOT violate the standing rule that distance is a
    learning-rate multiplier only and never scales delivered charge
    (`Current_Implementation_Methodology_Equations.md` section 4.3): delay changes WHEN
    charge arrives, never HOW MUCH. It is nonetheless a new semantic and is declared as
    one in the report.

    `dispersion_enabled=False` collapses every volley to a simultaneous impulse. That is
    the control condition for prompt section 5.2: with no arrival ordering there is
    nothing for fast inhibition to arbitrate, and single-winner WTA is expected to fail
    however small `h` becomes.
    """
    ts.validate()
    assignment = DelayAssignment()

    # Seeded per-synapse conduction jitter (prompt section 5.3).
    #
    # WHY THIS EXISTS. Pure geometric dispersion gives every competitor in a column
    # essentially the SAME arrival schedule, because the pixels sit at similar distances
    # from all of them. With 3 active pixels there are then only ~3 distinct arrival times
    # in the whole column, so at most ~3 distinguishable crossing times -- and 8
    # near-degenerate competitors cannot be separated into 8 classes by 3 events. That, not
    # the loop latency, is what caps winner multiplicity.
    #
    # Jitter gives each (source, target) PAIR its own offset, so the column sees up to
    # n_active x n_competitors distinct arrival times and ties can actually break. It is
    # drawn from the run seed and is exactly reproducible.
    #
    # This is conduction-delay variability, which is real: axonal arrival times are not
    # identical across a projection. It never scales delivered charge.
    rng = None
    if jitter_ms > 0:
        import numpy as _np  # noqa: PLC0415
        rng = _np.random.default_rng(seed)

    # Base hop per edge kind / projection. Apical permission, WTA recruitment and the
    # hard-reset pathway all take the MINIMUM delay: they are the "fast loop" whose
    # zero-latency idealisation this prototype replaces with a ratio.
    def base_hop(edge: dict) -> float:
        kind = edge["kind"]
        if kind in ("apical_excitation", "relay_excitation", "hard_reset_inhibition"):
            return ts.h
        return ts.base_ff

    # Per-pathway distance range, computed only over the dispersed projections.
    ranges: dict = {}
    per_target_span: dict = {}
    if dispersion_enabled:
        by_target: dict = {}
        for edge in spec["edges"]:
            projection = edge.get("projection")
            if projection not in DISPERSED_PROJECTIONS:
                continue
            d = _distance(pos, edge["source"], edge["target"])
            lo, hi = ranges.get(projection, (math.inf, -math.inf))
            ranges[projection] = (min(lo, d), max(hi, d))
            by_target.setdefault((projection, edge["target"]), []).append(d)
        # Calibrate on the PER-TARGET afferent span, not the population span.
        #
        # `S` is defined as the arrival spread within ONE volley at ONE cell -- that is the
        # spread a latency race can actually resolve. The population-wide span is larger
        # (it also contains the distance between different columns), so calibrating on it
        # would silently deliver a smaller per-target spread than requested. The median
        # target's span is used so one unusually compact or stretched cell cannot set the
        # scale for the whole pathway.
        for (projection, _target), distances in by_target.items():
            per_target_span.setdefault(projection, []).append(max(distances) - min(distances))

    for projection, (lo, hi) in ranges.items():
        spans = sorted(s for s in per_target_span.get(projection, []) if s > 1e-12)
        if spans:
            median_span = spans[len(spans) // 2]
            assignment.gain[projection] = ts.spread / median_span
        else:
            span = hi - lo
            assignment.gain[projection] = (ts.spread / span) if span > 1e-12 else 0.0

    for edge in spec["edges"]:
        projection = edge.get("projection")
        delay = base_hop(edge)
        if dispersion_enabled and projection in ranges:
            lo, _hi = ranges[projection]
            d = _distance(pos, edge["source"], edge["target"])
            delay += assignment.gain[projection] * (d - lo)
        if rng is not None and projection in DISPERSED_PROJECTIONS:
            delay += float(rng.uniform(0.0, jitter_ms))
        delay = ts.quantize(delay)
        assignment.by_edge[edge["id"]] = delay
        bucket = assignment.histogram.setdefault(projection or edge["kind"], {})
        bucket[delay] = bucket.get(delay, 0) + 1

    # Realized spread, measured on the quantized delays actually used. Both figures are
    # reported because they answer different questions: `per_target_median` is the one `S`
    # denotes and the one a latency race can resolve; `population` is the pathway's total
    # delay envelope and is what NEST's max_delay reflects.
    per_projection: dict = {}
    per_projection_target: dict = {}
    for edge in spec["edges"]:
        projection = edge.get("projection")
        if projection in ranges:
            delay = assignment.by_edge[edge["id"]]
            per_projection.setdefault(projection, []).append(delay)
            per_projection_target.setdefault((projection, edge["target"]), []).append(delay)

    target_spans: dict = {}
    for (projection, _target), delays in per_projection_target.items():
        target_spans.setdefault(projection, []).append(round(max(delays) - min(delays), 10))

    for projection, delays in per_projection.items():
        spans = sorted(target_spans.get(projection, []))
        assignment.realized_spread[projection] = {
            "per_target_median_ms": spans[len(spans) // 2] if spans else 0.0,
            "per_target_min_ms": spans[0] if spans else 0.0,
            "per_target_max_ms": spans[-1] if spans else 0.0,
            "population_ms": round(max(delays) - min(delays), 10),
        }

    return assignment


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


class NestTiledNetwork:
    """A constructed NEST realisation of the canonical `tiled_cc` topology.

    External labels are the repository's own node and edge IDs. The manifest maps every
    one of them to its NEST node / connection representation, so tests and reports select
    components through topology metadata and never by parsing display names.
    """

    def __init__(self, *, seed: int = 1, timescales: Timescales | None = None,
                 dispersion_enabled: bool = True, feedback_enabled: bool = True,
                 threads: int = 1, c_basal_weight: float | None = None,
                 charge_interval_ms: float | None = None,
                 shape: tuple | None = None, jitter_ms: float = 0.0,
                 reset_suppression_ms: float = 0.0, learning: bool = False,
                 record_weights: bool = False,
                 profile: str | EngineProfile = PROFILE_IMPULSE):
        """`c_basal_weight` installs a MATURED frozen basal weight on every `C`.

        The engine initializes `C` basal at theta/4 with a theta ceiling, and the one-shot
        recognition condition is `w_basal >= theta`. Under frozen learning `C` therefore
        can never fire, which in turn means the `C -> I` confirmation pathway never
        triggers and anything downstream of it -- notably the feedback cadence law -- is
        untestable.

        Setting this to `theta` freezes `C` at its MATURE value instead of its initial one.
        That is still frozen learning: no weight moves during the run. It isolates the
        cadence question from the learning question, so a Phase 3 result about feedback
        timing does not have to wait on Phase 4.
        """
        import nest  # noqa: PLC0415

        from backend.network_spec import tiled_cc_spec  # noqa: PLC0415
        from . import build_models  # noqa: PLC0415

        self.nest = nest
        # INSTANCE state, never a module global: two networks must be able to declare
        # different profiles in one process without hidden shared state. Resolved FIRST,
        # because it owns the kernel resolution and the delay policy that follow.
        self.profile = profile if isinstance(profile, EngineProfile) else get_profile(profile)
        self.is_continuous = self.profile.name == PROFILE_CIPP_CONTINUOUS

        self.ts = timescales or Timescales()
        if self.is_continuous:
            self.profile.validate()
            if jitter_ms:
                raise ValueError(
                    f"{self.profile.name} is the reference profile and is jitter-free; "
                    f"jitter_ms={jitter_ms} requires a separately named ablation profile"
                )
            # The profile owns the kernel resolution; `Timescales.h` is the impulse
            # profile's knob and must not silently override it.
            self.ts = replace(self.ts, h=self.profile.resolution_h_ms)
            # `Timescales.validate` enforces `presentation > spread` and `base_ff >= h`.
            # Those are IMPULSE constraints on geometric dispersion and the flat base hop,
            # neither of which exists here: the continuous profile has uniform delays and
            # takes no spread at all. Applying them would make an impulse-only rule part of
            # continuous physiology. Only the resolution is checked.
            if self.ts.h <= 0:
                raise ValueError(f"resolution h must be positive, got {self.ts.h}")
        else:
            self.ts.validate()
        self.seed = int(seed)
        self.dispersion_enabled = bool(dispersion_enabled)
        self.feedback_enabled = bool(feedback_enabled)

        self.shape = shape
        self.jitter_ms = float(jitter_ms)
        self.reset_suppression_ms = float(reset_suppression_ms)
        # When True, feedforward and C-basal connections use the co-generated plastic
        # synapse models and their targets use the PAIRED neuron variants. `Eor` stays
        # static in both modes: it is frozen at theta by design.
        self.learning = bool(learning)
        if shape is None:
            self.spec = tiled_cc_spec(cc_e_count=8)
        else:
            rows, cols = shape
            self.spec = tiled_cc_spec(cc_e_count=8, input_rows=rows, input_cols=cols,
                                      patch_rows=3, patch_cols=3)
        self.engine = build_reference_engine(seed=seed, shape=shape)
        self.pos = node_positions(self.engine)
        self.weights = extract_weights(self.engine)
        self.phi_of = extract_phi(self.engine)
        # Learning constants, read from the engine's resolved parameters so the NEST rule
        # and the reference rule are driven by ONE source of truth.
        params = self.engine.params
        self.eta = float(params["eta"])
        self.c_eta = float(params["c_eta"])
        self.dual_fe_B = float(params["dual_fe_B"])
        self.dual_fe_e = float(params["dual_fe_e"])
        self.dual_fe_wte = float(params["dual_fe_wte"])
        self.e_weight_cap_frac = float(params["e_weight_cap_frac"] or 0.5)
        self.relay_weight_cap_frac = float(params["relay_weight_cap_frac"])
        self.c_basal_weight = c_basal_weight
        if c_basal_weight is not None:
            for edge in self.spec["edges"]:
                if edge["kind"] == "basal_excitation":
                    self.weights[edge["id"]] = float(c_basal_weight)
        if self.is_continuous:
            # Uniform by projection, from the profile, ceiling-quantized. Geometry is not
            # consulted at all; `self.phi_of` keeps distance's one legitimate role as the
            # learning-rate multiplier.
            self.delays = assign_uniform_delays(self.spec, self.profile)
        else:
            self.delays = assign_delays(self.spec, self.pos, self.ts,
                                        dispersion_enabled=self.dispersion_enabled,
                                        jitter_ms=self.jitter_ms, seed=seed)

        # One volley's width in NEST time -- the analogue of the reference's "THIS
        # boundary" for the participation test. Taken from the REALIZED per-target arrival
        # spread plus a margin, and clamped below the presentation interval so the previous
        # volley can never be counted as participating.
        if self.is_continuous:
            # Learning membership comes from the PROFILE and nothing else. The impulse
            # formula below reads `ts.spread`, `ts.base_ff` and `ts.presentation`, so it
            # carries the stimulus schedule straight into plastic `tau_volley` -- measured
            # at presentation 3.0 ms -> 1.5 ms and 5.0 ms -> 2.5 ms. A test pair that
            # happens to sit on the same side of the `presentation * 0.5` clamp does not
            # see it, which is how it survived the first pass.
            self.tau_volley_ms = float(self.profile.causal_volley.separation_ms)
        else:
            realized = self.delays.realized_spread.get("rg_to_column", {})
            spread = float(realized.get("per_target_max_ms", self.ts.spread) or self.ts.spread)
            self.tau_volley_ms = min(spread * 1.5 + self.ts.base_ff,
                                     self.ts.presentation * 0.5)

        self.node_meta = {n["id"]: n for n in self.spec["nodes"]}
        self.edge_meta = {e["id"]: e for e in self.spec["edges"]}

        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": self.ts.h,
            "rng_seed": self.seed,
            "local_num_threads": int(threads),
            # Deterministic construction: NEST must not silently reorder or randomise.
            "overwrite_files": True,
        })
        build_models.install_into_kernel()

        self.gid_of: dict = {}        # repo node id -> NEST NodeCollection (size 1)
        self.id_of_gid: dict = {}     # NEST global id -> repo node id
        self.generators: dict = {}    # RGC id -> spike_generator NodeCollection
        self._receptors: dict = {}    # model name -> {port name: receptor index}

        self.charge_interval_ms = charge_interval_ms
        self.multimeters: dict = {}
        self._create_nodes()
        # BEFORE `_connect_edges`, and it has to be: a `weight_recorder` is a COMMON
        # PROPERTY of the synapse model, baked into each connection as it is created. Wiring
        # it afterwards records nothing, silently. It is created holding `start` past the end
        # of any run, so it costs nothing until `attach_weight_recording` opens the window.
        self.weight_recorder = None
        if record_weights:
            self._attach_weight_recorder()
        self._connect_edges()
        self._attach_recorders()
        if charge_interval_ms:
            self.attach_charge_recording(charge_interval_ms)

    # ---------------------------------------------------------------- construction
    def _receptor_map(self, model: str) -> dict:
        """Port name -> receptor index, read from the generated model, never hardcoded.

        Two NESTML conventions matter here. Receptor names are UPPERCASED in the generated
        metadata, and a model with only ONE spike port publishes no `receptor_types` at
        all -- for those, connections must omit `receptor_type` entirely rather than pass
        a guessed index. `event_relay` is the single-port case.
        """
        if model not in self._receptors:
            defaults = self.nest.GetDefaults(model)
            raw = defaults.get("receptor_types") or {}
            self._receptors[model] = {str(k).lower(): int(v) for k, v in raw.items()}
        return self._receptors[model]

    def _model_for(self, node: dict) -> tuple[str, dict]:
        """Choose the NESTML model and parameters for a spec node, BY METADATA ONLY.

        Selection reads `archetype` and `column_role`; it never parses an id or a display
        label. The dormant top-column `C` is identified by `has_parent == False`, exactly
        as the reference implementation requires.
        """
        archetype = node["archetype"]
        role = node.get("column_role")
        theta = E_THRESHOLD

        # THE Phase 2 separation. `window = self.ts.presentation` is the audit's P0
        # "physiology coupled to the experiment controller": it makes dendritic coincidence
        # and relay behaviour change when the experiment changes how often it presents
        # input. It survives ONLY in the historical impulse branch, where the recorded
        # findings depend on it. The continuous profile takes every window from
        # `profile.coincidence`, which carries no schedule at all.
        if self.is_continuous:
            return self._model_for_continuous(archetype, role, theta)

        window = self.ts.presentation

        if archetype == "rg_source":
            return "spike_generator", {}
        if archetype == "i_relay":
            # Stateless inhibitory relay: theta/3, at most one WTA volley per window.
            return "event_relay", {"theta": theta * I_THRESHOLD_FRAC,
                                   "t_lockout": window}
        if archetype == "e_coincidence":
            if self.learning:
                return "event_coincidence__with_plastic_basal_synapse", {
                    "theta": theta,
                    "t_ref": 0.0,
                    "tau_basal": window,
                    "tau_apical": window,
                    "tau_deposit_lock": window,
                }
            return "event_coincidence", {
                "theta": theta,
                "t_ref": 0.0,
                "tau_basal": window,
                "tau_apical": window,
                "tau_deposit_lock": window,
            }
        if archetype == "e_latency_competitor":
            if role == "Eor":
                # Frozen relay at theta with afferent weights at theta: one upstream event
                # produces one output spike. No lockout -- Eor is not rate limited.
                return "event_relay", {"theta": theta, "t_lockout": 0.0}
            model = ("event_accumulator__with_plastic_feedforward_synapse"
                     if self.learning else "event_accumulator")
            return model, {"theta": theta, "t_ref": 0.0,
                           "t_ref_reset": float(self.reset_suppression_ms)}
        raise ValueError(f"unhandled archetype {archetype!r} for node {node['id']!r}")

    def _model_for_continuous(self, archetype: str, role: str | None, theta: float):
        """Model and parameters for `cipp_continuous`, sourced ONLY from the profile.

        No value here is derived from the stimulus schedule -- that is what makes two
        networks differing only in presentation period byte-equivalent in their physics.

        SCOPE. Phase 2 makes the profile the source of the parameters it owns. Swapping the
        competitor to the continuous membrane model itself is Phase 3 ("ordinary E, Eor and
        WTA"), which also has to introduce the causal-volley state and the separate
        `wta_reset` / `prediction_credit` ports. The competitor entry below therefore still
        names the accumulator, and says so, rather than silently implying Phase 3 is done.
        """
        coincidence = self.profile.coincidence
        membrane = self.profile.membrane

        if archetype == "rg_source":
            return "spike_generator", {}
        if archetype == "i_relay":
            # The relay's OWN lockout. It had been taken from the C soma's deposit dead
            # time, which gave that field two roles in two unrelated cell types.
            return "event_relay", {"theta": theta * I_THRESHOLD_FRAC,
                                   "t_lockout": float(self.profile.relay.wta_lockout_ms)}
        if archetype == "e_coincidence":
            params = {
                "theta": theta,
                # The C cell's own refractory, not the competitor's. Borrowing
                # `MembranePolicy.t_ref_ms` spread continuous-competitor physiology onto a
                # different cell type.
                "t_ref": float(coincidence.c_refractory_ms),
                "tau_basal": float(coincidence.basal_window_ms),
                "tau_apical": float(coincidence.apical_window_ms),
                "tau_deposit_lock": float(coincidence.deposit_dead_time_ms),
            }
            model = ("event_coincidence__with_plastic_basal_synapse"
                     if self.learning else "event_coincidence")
            return model, params
        if archetype == "e_latency_competitor":
            if role == "Eor":
                return "event_relay", {"theta": theta, "t_lockout": 0.0}
            # PHASE 3 REPLACES THIS with `profile.membrane.model`. Kept as the accumulator
            # for now so the profile integration can be tested on a constructible network;
            # `reset_suppression_ms` is deliberately NOT read here, because it is a
            # timer-based stand-in for the counted prediction credit Phase 4 introduces.
            model = ("event_accumulator__with_plastic_feedforward_synapse"
                     if self.learning else "event_accumulator")
            # `membrane.t_ref_ms` is the TARGET continuous competitor's refractory and is
            # applied here because the accumulator scaffold stands in for that cell. It is
            # not reused for any other cell type.
            return model, {"theta": theta, "t_ref": float(membrane.t_ref_ms),
                           "t_ref_reset": 0.0}
        raise ValueError(f"unhandled archetype {archetype!r}")

    def _create_nodes(self) -> None:
        nest = self.nest
        for node in self.spec["nodes"]:
            model, params = self._model_for(node)
            if model == "spike_generator":
                gen = nest.Create("spike_generator", 1, {"spike_times": []})
                self.generators[node["id"]] = gen
                if self.learning:
                    # NEST forbids MIXED synapse types on a device's outgoing connections,
                    # and with learning on an RGC drives both plastic feedforward edges and
                    # the (static) input recorder. The canonical fix is a `parrot_neuron`
                    # relay: it repeats every spike it receives one-for-one, and being a
                    # real neuron it may have heterogeneous outgoing synapses.
                    #
                    # The parrot IS the RGC cell as far as identity and recording go; the
                    # generator is only its schedule. That also reads truer to the
                    # reference, where an RGC is an exogenous binary source.
                    #
                    # Cost: one extra hop, so an RGC spike lands `h` after its scheduled
                    # time. Used ONLY in learning mode, so every frozen-mode result and
                    # fixture already recorded stays byte-identical.
                    parrot = nest.Create("parrot_neuron", 1)
                    nest.Connect(gen, parrot, syn_spec={
                        "synapse_model": "static_synapse", "weight": 1.0,
                        "delay": self.ts.h})
                    self.gid_of[node["id"]] = parrot
                    self.id_of_gid[int(parrot.global_id)] = node["id"]
                else:
                    self.gid_of[node["id"]] = gen
                    self.id_of_gid[int(gen.global_id)] = node["id"]
                continue
            cell = nest.Create(model, 1, params)
            self.gid_of[node["id"]] = cell
            self.id_of_gid[int(cell.global_id)] = node["id"]

    def _target_port(self, edge: dict) -> str:
        """Which named input port an edge kind lands on.

        The `column_c_to_i` confirmation pathway is routed to `event_relay`'s separate
        `confirm` port so it is NOT swallowed by the once-per-window WTA lockout, per
        `Current_Implementation_Methodology_Equations.md` section 7. Selection is by
        PROJECTION, so it cannot be confused with the lateral `column_e_to_i` volley that
        shares the same edge kind.
        """
        kind = edge["kind"]
        if kind == "hard_reset_inhibition":
            return "reset"
        if kind == "basal_excitation":
            return "basal"
        if kind == "apical_excitation":
            return "apical"
        if edge.get("projection") == "column_c_to_i":
            return "confirm"
        return "exc"

    def _edge_weight(self, edge: dict) -> float:
        """Delivered weight for an edge.

        Weighted kinds take the engine's own initial weight. Unweighted kinds carry a
        placeholder that the receiving model ignores entirely:

          - `apical_excitation` is Boolean permission; `event_coincidence` never reads it.
          - `hard_reset_inhibition` clears state unconditionally; the value is unused. It
            is emitted as a POSITIVE number because the reset is not a subtractive
            inhibitory current -- calling it -1 would imply a charge contribution that the
            reference `hard_reset` does not make.
          - `relay_excitation` drives a threshold relay whose theta is theta/3, so one
            event must cross. It is given `theta` rather than a learned value because the
            reference `I` owns no weight at all.
        """
        kind = edge["kind"]
        if kind in ("feedforward", "basal_excitation"):
            try:
                return float(self.weights[edge["id"]])
            except KeyError as exc:
                raise KeyError(
                    f"no engine weight for {kind} edge {edge['id']!r} -- the spec and the "
                    "engine disagree about which edges are plastic"
                ) from exc
        if kind == "relay_excitation":
            return E_THRESHOLD
        return 1.0

    def _syn_spec(self, edge: dict) -> dict:
        """Synapse model + parameters for one edge.

        Plastic ONLY where the reference is plastic:
          * `rg_to_column` and `column_to_column_ff` -- ordinary-E feedforward detectors;
          * `column_eor_to_c_basal` -- the C cell's single learned basal weight.

        `column_e_to_eor` is deliberately STATIC. `Eor` is frozen at theta by design
        (`eor_plasticity_enabled = False`), so making it plastic would be a change to the
        reference model, not a port of it.
        """
        projection = edge.get("projection")
        delay = self.delays.by_edge[edge["id"]]
        weight = self._edge_weight(edge)

        if self.learning and projection in ("rg_to_column", "column_to_column_ff"):
            return {
                "synapse_model": PLASTIC_SYNAPSE_MODELS[0],
                "w": weight, "delay": delay,
                "eta": float(self.eta), "theta": E_THRESHOLD, "B": float(self.dual_fe_B),
                "e_floor": float(self.dual_fe_e), "w_te": float(self.dual_fe_wte),
                "w_cap": E_THRESHOLD * float(self.e_weight_cap_frac),
                "phi": float(self.phi_of.get(edge["id"], 1.0)),
                "tau_volley": float(self.tau_volley_ms),
            }
        if self.learning and projection == "column_eor_to_c_basal":
            return {
                "synapse_model": PLASTIC_SYNAPSE_MODELS[1],
                "w": weight, "delay": delay,
                "eta_c": float(self.c_eta), "theta": E_THRESHOLD,
                "B": float(self.dual_fe_B), "e_floor": float(self.dual_fe_e),
                "w_te": float(self.dual_fe_wte),
                # C's ceiling is theta, not theta/2: it must reach one-shot recognition.
                "w_cap": E_THRESHOLD * float(self.relay_weight_cap_frac),
                "phi": 1.0,   # column-local basal edges carry no distance penalty
            }
        return {"synapse_model": "static_synapse", "weight": weight, "delay": delay}

    def _connect_edges(self) -> None:
        nest = self.nest
        self.conn_of: dict = {}
        self.skipped_edges: list = []
        for edge in self.spec["edges"]:
            # The feedback control condition (suite case 6) disables the translated
            # top-down confirmation pathway. It is selected by PROJECTION, not by id.
            if not self.feedback_enabled and edge.get("projection") == "column_c_to_i":
                self.skipped_edges.append(edge["id"])
                continue
            source = self.gid_of[edge["source"]]
            target = self.gid_of[edge["target"]]
            target_model = self.node_meta[edge["target"]]
            model_name, _ = self._model_for(target_model)
            receptors = self._receptor_map(model_name)
            port = self._target_port(edge)
            syn_spec = self._syn_spec(edge)
            if receptors:
                if port not in receptors:
                    raise KeyError(
                        f"model {model_name!r} has no port {port!r} for edge "
                        f"{edge['id']!r}; available: {sorted(receptors)}"
                    )
                syn_spec["receptor_type"] = receptors[port]
            elif port != "exc":
                # A single-port model can only accept the default excitatory pathway. An
                # edge kind that wanted `reset`/`basal`/`apical` here would be silently
                # mis-routed, so fail loudly instead.
                raise KeyError(
                    f"edge {edge['id']!r} (kind {edge['kind']!r}) needs port {port!r} but "
                    f"target model {model_name!r} publishes only one unnamed spike port"
                )
            nest.Connect(source, target, "one_to_one", syn_spec=syn_spec)
            self.conn_of[edge["id"]] = (int(source.global_id), int(target.global_id))

    def _attach_recorders(self) -> None:
        """One spike recorder for every non-source node; senders map back via `id_of_gid`."""
        nest = self.nest
        self.recorder = nest.Create("spike_recorder")
        recorded = [nid for nid, n in self.node_meta.items()
                    if n["archetype"] != "rg_source"]
        self.recorded_ids = recorded
        population = None
        for nid in recorded:
            population = self.gid_of[nid] if population is None else population + self.gid_of[nid]
        nest.Connect(population, self.recorder)
        # A second recorder on the generators makes the realized input schedule auditable
        # rather than merely intended.
        # Record whatever actually represents the RGC cell: the parrot in learning mode,
        # the generator itself otherwise.
        self.input_recorder = nest.Create("spike_recorder")
        gen_pop = None
        for nid in self.generators:
            source = self.gid_of[nid]
            gen_pop = source if gen_pop is None else gen_pop + source
        nest.Connect(gen_pop, self.input_recorder)

    # ------------------------------------------------------------------- manifest
    def manifest(self) -> dict:
        """Everything a run needs to be reproducible and every ID mapping, per section 4."""
        import nest  # noqa: PLC0415

        from . import build_models  # noqa: PLC0415

        stamp = json.loads((REPO_ROOT / ".nest-build" / "build_stamp.json").read_text()) \
            if (REPO_ROOT / ".nest-build" / "build_stamp.json").is_file() else {}
        return {
            # Which semantic profile produced this run, read from the INSTANCE. Every
            # artifact carries it plus the complete unit-bearing parameter set, so a result
            # can never be read as a CIPP-engine claim when it came from the prototype.
            "engine_profile": self.profile.name,
            "profile": self.profile.describe(),
            # Matched causal path pairs, so a window claim can be checked against the
            # graph rather than against independently combined projection extrema.
            "causal_arrival_envelope": self.causal_arrival_envelope(),
            # The differences a reader must not mistake for results.
            "accepted_differences": self.accepted_differences(),
            "topology": self.spec["name"],
            "spec_hash": spec_hash(self.spec),
            "seed": self.seed,
            "counts": {
                "nodes": len(self.spec["nodes"]),
                "edges": len(self.spec["edges"]),
                "connected_edges": len(self.conn_of),
                "skipped_edges": len(self.skipped_edges),
            },
            "timescales": self.ts.describe(),
            "shape": list(self.shape) if self.shape else [9, 9],
            "jitter_ms": self.jitter_ms,
            "reset_suppression_ms": self.reset_suppression_ms,
            "learning": self.learning,
            "tau_volley_ms": self.tau_volley_ms,
            "eta": self.eta,
            "c_eta": self.c_eta,
            "dispersion_enabled": self.dispersion_enabled,
            "feedback_enabled": self.feedback_enabled,
            "c_basal_weight_override": self.c_basal_weight,
            "charge_interval_ms": self.charge_interval_ms,
            "delays": {
                "realized_spread_ms": self.delays.realized_spread,
                "gain_ms_per_unit_distance": self.delays.gain,
                "histogram": {k: {str(d): c for d, c in sorted(v.items())}
                              for k, v in self.delays.histogram.items()},
                "min_ms": min(self.delays.by_edge.values()),
                "max_ms": max(self.delays.by_edge.values()),
            },
            "kernel": {
                "resolution_ms": float(nest.resolution),
                "min_delay_ms": float(nest.min_delay),
                "max_delay_ms": float(nest.max_delay),
                "local_num_threads": int(nest.local_num_threads),
                "num_processes": int(nest.num_processes),
            },
            "versions": {
                "nest": stamp.get("nest_version"),
                "nestml": stamp.get("nestml_version"),
                "module": stamp.get("module_name"),
                "model_fingerprint": stamp.get("fingerprint"),
            },
            "id_map": {nid: int(nc.global_id) for nid, nc in self.gid_of.items()},
            "edge_map": self.conn_of,
        }

    # Beyond any run this repository simulates, so a recorder registered at construction
    # captures nothing until a window is opened for it.
    _NEVER_MS = 1.0e12

    def _attach_weight_recorder(self) -> None:
        """Register one `weight_recorder` as the common property of every plastic model.

        A weight recorder is EVENT-DRIVEN, not sampled: the synapse emits when it updates,
        so what comes back is exactly the set of weight changes at exactly the times they
        happened. That is the right shape for this learning rule, whose weights move in
        discrete steps, and it is the only affordable way to see learning -- polling 163
        connections per tick through `GetConnections` is what made a 3.7-second simulation
        take 18 minutes.
        """
        nest = self.nest
        self.weight_recorder = nest.Create("weight_recorder", 1, {"start": self._NEVER_MS})
        for model in PLASTIC_SYNAPSE_MODELS:
            nest.SetDefaults(model, {"weight_recorder": self.weight_recorder})

    def attach_weight_recording(self, start_ms: float | None = None) -> None:
        """Open the weight recorder's window, from `start_ms` (default: now).

        Requires `record_weights=True` at construction -- the common property cannot be
        attached retroactively, because existing connections already carry the old one.
        """
        if self.weight_recorder is None:
            raise ValueError(
                "no weight recorder: build the network with record_weights=True "
                "(a weight_recorder is a synapse-model common property and must be "
                "registered before the connections are created)"
            )
        now = float(self.nest.biological_time)
        self.weight_recorder.set({"start": float(start_ms if start_ms is not None else now),
                                  "stop": self._NEVER_MS})

    def attach_charge_recording(self, interval_ms: float, start_ms: float | None = None) -> None:
        """Sample EVERY declared recordable from every model neuron, from `start_ms` on.

        Callable AFTER construction, and after simulating, which is what makes it possible
        to record a long training run's short viewable tail at full resolution. Sampling
        every `h` for the whole of an 8.9-second training run would mean tens of millions of
        samples held in memory and then thrown away; attaching the meter at the boundary
        records the same thing for the window anyone will actually look at.

        `start_ms` defaults to now, which for a freshly-built network is 0.

        Devices created between `Simulate` calls are ordinary NEST practice -- the kernel is
        rebuilt on the next call. Nothing here alters connectivity between model neurons,
        consumes RNG, or delivers events; see `test_nest_charge_recording.py`.
        """
        self.charge_interval_ms = interval_ms
        self._attach_multimeter(start_ms)

    def _attach_multimeter(self, start_ms: float | None = None) -> None:
        """Sample EVERY declared recordable from every model neuron, if requested.

        One multimeter per MODEL, because a NEST multimeter records a fixed list of
        variable names and the three models expose different sets:

            event_accumulator  q, q_pre, refr_until
            event_relay        q, q_pre, q_confirm, lock_until
            event_coincidence  q, q_pre, refr_until, basal_charge,
                               basal_until, apical_until, deposit_lock_until

        The list is read from the model's own `recordables` rather than hard-coded, so a
        variable added to a NESTML model is sampled without anyone having to remember to
        widen this. Recording only `q` and `q_pre` -- the previous behaviour -- left
        refractory state and the coincidence cell's basal charge reported as `unavailable`
        when NEST was in fact willing to hand them over.
        """
        self.multimeters = {}
        interval = self.charge_interval_ms
        if not interval:
            return
        steps = interval / self.ts.h
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"charge_interval_ms={interval} is not a multiple of h={self.ts.h}"
            )
        nest = self.nest
        by_model: dict = {}
        for nid, meta in self.node_meta.items():
            if meta["archetype"] == "rg_source":
                continue
            model, _params = self._model_for(meta)
            by_model.setdefault(model, []).append(nid)

        for model, ids in by_model.items():
            # NESTML emits a `codegen_placeholder__for_<synapse>` recordable per attached
            # plastic synapse type. It is a code-generation artifact, not model state -- it
            # is constant zero -- so recording it would put a meaningless column in every
            # frame of every artifact.
            recordables = sorted(
                name for name in (str(v) for v in nest.GetDefaults(model)["recordables"])
                if not name.startswith("codegen_placeholder")
            )
            if not recordables:
                raise ValueError(f"model {model} declares no recordables to sample")
            params = {"interval": float(interval), "record_from": recordables}
            if start_ms is not None:
                params["start"] = float(start_ms)
            meter = nest.Create("multimeter", 1, params)
            population = None
            for nid in ids:
                population = self.gid_of[nid] if population is None \
                    else population + self.gid_of[nid]
            nest.Connect(meter, population)
            self.multimeters[model] = meter

    # ------------------------------------------------------- dashboard topology
    def dashboard_topology(self) -> dict:
        """The dashboard's existing topology payload, describing the NEST network.

        Built from the reference engine's own `topology()` rather than hand-assembled: the
        graph, node metadata, positions, tiling block, grid and pattern bank are the SAME
        canonical spec, so reusing them guarantees the repository IDs, roles and column
        metadata round-trip exactly and cannot drift from the live dashboard's vocabulary.

        Only what NEST actually changes is overridden:

        * every weighted edge carries the FROZEN NEST weight;
        * every edge carries its realized NEST delay in ms;
        * `params` is marked as the NEST engine and carries the timescales;
        * an additive `nest` block carries provenance and the validity envelope.

        Unweighted edge kinds keep the reference engine's `None`. NEST delivers a
        placeholder value on those connections (the models ignore it), and reporting a
        placeholder as a weight would present an unavailable quantity as a real one.
        """
        topo = self.engine.topology()

        for synapse in topo["synapses"]:
            edge_id = synapse["id"]
            if edge_id in self.weights:
                synapse["weight"] = round(float(self.weights[edge_id]), 6)
            synapse["nest_delay_ms"] = self.delays.by_edge.get(edge_id)
            meta = self.edge_meta.get(edge_id, {})
            if meta.get("projection"):
                synapse.setdefault("projection", meta["projection"])

        params = topo.setdefault("params", {})
        params["engine"] = "nest"
        params["nest_resolution_ms"] = self.ts.h
        params["nest_presentation_ms"] = self.ts.presentation
        # The legacy integer feedback-loop latency is a BOUNDARY count and means nothing
        # here; the NEST analogue is a duration. Replace it rather than let a stale
        # integer be read as if it still applied.
        params["feedback_loop_latency"] = None
        params["feedback_loop_latency_ms"] = round(self.feedback_loop_latency_ms(), 6)

        topo["nest"] = {
            "manifest": self.manifest(),
            "validity_envelope": {
                "leak_rate": 0.0,
                "persistent_inhibitory_conductance": False,
                "note": (
                    "The NESTML event models are an exact port ONLY at leak_rate = 0 with "
                    "no persistent inhibitory conductance, which is what makes the "
                    "reference conductance-LIF membrane a pure integrator."
                ),
            },
            "learning": "frozen",
            # The profile's implementation status travels with the DASHBOARD payload too,
            # not only the manifest and replay provenance: someone reading the topology
            # panel must not conclude that a continuous CIPP engine produced this.
            "engine_profile": self.profile.name,
            "implementation_disposition": asdict(self.profile.disposition),
            "known_differences": [
                self.profile.disposition.summary,
                f"Active competitor model: {self.profile.disposition.active_competitor_model} "
                f"in {self.profile.disposition.active_units}. Profile membrane model "
                f"{self.profile.disposition.target_membrane_model} is a TARGET, not active "
                f"physics.",
                "mechanical_profile_promoted = "
                f"{self.profile.disposition.mechanical_profile_promoted}: the mechanical "
                "acceptance suite has not passed.",
                *self.profile.disposition.known_defects,
                "Native NEST delivery produces MULTIPLE ordinary-E winners in many "
                "column/presentation windows; the reference engine guarantees one.",
                "The strict 1010 feedback cadence law does not reproduce.",
                "Weights are frozen static_synapse values; no learning occurs.",
                "MPI is not available in the installed conda NEST build.",
            ],
        }
        return topo

    def _walk_paths(self, start_ids, projection_sequence):
        """Every CONNECTED path from `start_ids` following `projection_sequence` in order.

        Connectivity is checked on real edge endpoints: an edge only extends a path if its
        `source` is the node the path currently sits on. The superseded implementation
        grouped edges by projection, took projection-wide extrema and summed them, which
        can combine a delay from one column with a delay from another and never verifies
        that the hops meet at all.
        """
        by_source: dict = {}
        for edge in self.spec["edges"]:
            if edge["id"] in self.delays.by_edge:
                by_source.setdefault(edge["source"], []).append(edge)

        paths = [{"nodes": [nid], "edges": [], "delays": []} for nid in start_ids]
        for projection in projection_sequence:
            extended = []
            for path in paths:
                for edge in by_source.get(path["nodes"][-1], ()):
                    if edge.get("projection") != projection:
                        continue
                    extended.append({
                        "nodes": path["nodes"] + [edge["target"]],
                        "edges": path["edges"] + [edge["id"]],
                        "delays": path["delays"] + [float(self.delays.by_edge[edge["id"]])],
                    })
            paths = extended
            if not paths:
                break
        for path in paths:
            path["path_delay_ms"] = sum(path["delays"])
        return paths

    def accepted_differences(self) -> list:
        """Declared divergences that travel with every artifact from this network.

        A reader opening a replay months later has only the artifact. These are the claims
        it must not make on the artifact's behalf.
        """
        disposition = self.profile.disposition
        out = [
            {"area": "implementation status", "difference": disposition.summary,
             "mechanical_profile_promoted": disposition.mechanical_profile_promoted},
            {"area": "exact ties",
             "difference": (f"exact_tie_policy={self.profile.wta.exact_tie_policy}; "
                            "implicit GID/creation-order arbitration is forbidden")},
            {"area": "causal envelope",
             "difference": ("conduction only; parent membrane/processing latency is not "
                            "included, so the reported skew is a lower bound")},
        ]
        for defect in disposition.known_defects:
            out.append({"area": "known defect", "difference": defect})
        if disposition.deferred_to_phase_3_or_later:
            out.append({"area": "deferred",
                        "difference": list(disposition.deferred_to_phase_3_or_later)})
        return out

    def causal_arrival_envelope(self, *, max_pairs_recorded: int = 8) -> dict:
        """CONDUCTION-ONLY arrival envelope, over MATCHED basal/apical evidence pairs.

        A coincidence is a claim about ONE C cell receiving both arms of ONE causal branch.
        Basal and apical evidence reach it along different routes::

            basal:   RGC -> L1 E -> L1 Eor -> C
            apical:  RGC -> L1 E -> L1 Eor -> L2 E -> C

        Summarising the two families independently -- 144 basal paths against 2304 apical
        paths, taking extrema of each -- can pair a basal arrival at one C with an apical
        arrival at a DIFFERENT C, or one driven by a different RGC entirely. Those are not
        coincidences and their difference is not a skew. So pairs are matched here on both
        the target C and the shared causal prefix `RGC -> E -> Eor`, and the skew is
        computed per matched pair.

        WHAT THIS IS NOT. Conduction only. It omits the parent competitor's membrane
        crossing latency on the apical arm -- the time L2 E takes to reach threshold before
        emitting the apical event -- which is unknown until the Phase 3 continuous model and
        its drive envelope exist. Phase 1's pA weights cannot supply it: that report states
        explicitly that those units are not mapped to repository charge units. The reported
        skew is therefore a LOWER BOUND, and full window validation is deferred, not done.
        """
        rgc_ids = [n["id"] for n in self.spec["nodes"] if n["archetype"] == "rg_source"]
        basal = self._walk_paths(rgc_ids, ("rg_to_column", "column_e_to_eor",
                                           "column_eor_to_c_basal"))
        apical = self._walk_paths(rgc_ids, ("rg_to_column", "column_e_to_eor",
                                            "column_to_column_ff",
                                            "column_to_column_apical"))

        # Key on the shared causal prefix AND the target C: same source, same competitor,
        # same local relay, same coincidence cell.
        def key(path):
            return (tuple(path["nodes"][:3]), path["nodes"][-1])

        basal_by_key: dict = {}
        for path in basal:
            basal_by_key.setdefault(key(path), []).append(path)

        pairs = []
        for apical_path in apical:
            for basal_path in basal_by_key.get(key(apical_path), ()):
                pairs.append({
                    "source_rgc": apical_path["nodes"][0],
                    "competitor": apical_path["nodes"][1],
                    "relay": apical_path["nodes"][2],
                    "target_c": apical_path["nodes"][-1],
                    "shared_prefix_nodes": apical_path["nodes"][:3],
                    "basal": basal_path,
                    "apical": apical_path,
                    "conduction_skew_ms": (apical_path["path_delay_ms"]
                                           - basal_path["path_delay_ms"]),
                })

        skews = [pair["conduction_skew_ms"] for pair in pairs]
        totals_basal = [pair["basal"]["path_delay_ms"] for pair in pairs]
        totals_apical = [pair["apical"]["path_delay_ms"] for pair in pairs]

        first_hop = [float(self.delays.by_edge[e["id"]]) for e in self.spec["edges"]
                     if e.get("projection") == "rg_to_column" and e["id"] in self.delays.by_edge]
        return {
            "kind": "conduction_only",
            "matched_pairs": len(pairs),
            # A basal path with no apical partner at the same C from the same branch is not
            # evidence of a coincidence and is excluded from the envelope; the count is
            # reported so an unexpected structure is visible rather than silently dropped.
            "unmatched_basal_paths": len(basal) - len({id(pair["basal"]) for pair in pairs}),
            "basal_paths_total": len(basal),
            "apical_paths_total": len(apical),
            "pairs_recorded": pairs[:max_pairs_recorded],
            "basal_arrival_ms": ({"min_ms": min(totals_basal), "max_ms": max(totals_basal)}
                                 if pairs else None),
            "apical_arrival_ms": ({"min_ms": min(totals_apical), "max_ms": max(totals_apical)}
                                  if pairs else None),
            "conduction_skew_ms": (max(skews) if pairs else None),
            "conduction_skew_range_ms": ({"min": min(skews), "max": max(skews)}
                                         if pairs else None),
            "within_volley_spread_ms": (max(first_hop) - min(first_hop)) if first_hop else 0.0,
            "processing_latency_included": False,
            "deferred": (
                "parent competitor membrane/processing latency on the apical arm is NOT "
                "included and is unverified until the Phase 3 continuous model and drive "
                "envelope exist; the reported skew is a LOWER BOUND"
            ),
            "window_validation_status": "provisional_deferred_to_phase_3",
            "method": ("connected edge sequences walked from RGC sources, then MATCHED on "
                       "shared causal prefix and target C; per-path node ids, edge ids and "
                       "delays recorded"),
        }

    def validate_against_schedule(self, *, min_inter_volley_interval_ms: float) -> dict:
        """Check the profile's windows against THIS graph and THIS schedule.

        Schedule information is consumed for VALIDATION ONLY. Nothing here mutates or
        derives neuron physiology from it -- that separation is the point of Phase 2.
        """
        envelope = self.causal_arrival_envelope()
        # Structural profile checks always run.
        self.profile.validate()
        # The causal-volley separation check is legitimate on conduction alone: it bounds
        # learning membership against arrival spread and the schedule, neither of which
        # involves the parent's membrane latency.
        self.profile.validate_schedule(
            within_volley_spread_ms=envelope["within_volley_spread_ms"],
            min_inter_volley_interval_ms=min_inter_volley_interval_ms)
        # The coincidence WINDOW check is NOT run against conduction alone. The true skew
        # includes the parent's membrane crossing latency, which Phase 2 cannot measure, so
        # validating against a lower bound would produce a pass that means nothing. Only
        # the cross-volley bound -- which does not depend on the missing term -- is checked.
        self.profile.coincidence.validate_cross_volley_only(
            min_inter_volley_interval_ms=min_inter_volley_interval_ms)
        return envelope

    def plastic_weights(self) -> dict:
        """`{edge_id: w}` for every plastic edge -- ONE kernel query per synapse model.

        The obvious implementation, `GetConnections(source=.., target=..)` once per edge,
        is correct and pathologically slow. Each call scans the kernel's connection tables,
        and the per-call cost GROWS as a run proceeds. Measured on the 4-pattern saturation
        run at h=0.01: the per-edge form cost 0.4 s at the first convergence check and
        3.6 s by the eighth, ~93 % of a wall clock in which NEST's own stepping accounted
        for about 3.7 seconds total. Summed over the run that is the difference between
        ~18 minutes and ~1. One query per model is 110-130x faster and returns identical
        numbers.

        `(source, target)` identifies an edge because this topology has no parallel edges;
        that is asserted rather than assumed, since a future spec could add one and the
        failure would otherwise be a silently mismatched weight.
        """
        nest = self.nest
        by_pair: dict = {}
        for edge_id, pair in self.conn_of.items():
            by_pair.setdefault(pair, []).append(edge_id)

        out: dict = {}
        for model in PLASTIC_SYNAPSE_MODELS:
            conns = nest.GetConnections(synapse_model=model)
            if not len(conns):
                continue
            data = conns.get(["source", "target", "w"])
            columns = [data[k] if isinstance(data[k], (list, tuple)) else [data[k]]
                       for k in ("source", "target", "w")]
            for src, tgt, w in zip(*columns):
                edge_ids = by_pair.get((int(src), int(tgt)))
                if not edge_ids:
                    continue
                if len(edge_ids) > 1:
                    raise ValueError(
                        f"parallel edges {edge_ids} share source {src} -> target {tgt}; "
                        "weights cannot be attributed by endpoint alone"
                    )
                out[edge_ids[0]] = float(w)
        return out

    def outgoing_edges(self) -> dict:
        """node id -> [edge ids leaving it]. Used to pulse edges from a recorded spike."""
        out: dict = {}
        for edge in self.spec["edges"]:
            if edge["id"] in self.conn_of:
                out.setdefault(edge["source"], []).append(edge["id"])
        return {k: sorted(v) for k, v in out.items()}

    # ------------------------------------------------------------- role selection
    def ids_where(self, **criteria) -> list:
        """Select node ids by topology METADATA, never by parsing a display name."""
        out = []
        for nid, meta in self.node_meta.items():
            if all(meta.get(k) == v for k, v in criteria.items()):
                out.append(nid)
        return sorted(out)

    def column_ids(self, layer: str | None = None) -> list:
        cols = {meta.get("column_id") for meta in self.node_meta.values()
                if meta.get("column_id") and (layer is None or meta.get("layer") == layer)}
        return sorted(c for c in cols if c)

    def patch_pixels(self, patch_row: int, patch_col: int) -> list:
        """The nine RGC ids of one 3x3 input patch, in patch-local row-major order."""
        pixels = [meta for meta in self.node_meta.values()
                  if meta["archetype"] == "rg_source"
                  and meta.get("patch_row") == patch_row
                  and meta.get("patch_col") == patch_col]
        pixels.sort(key=lambda m: (m["patch_local_row"], m["patch_local_col"]))
        return [m["id"] for m in pixels]

    def feedback_loop_latency_ms(self) -> float:
        """Loop latency L in NEST time, DERIVED FROM THE GRAPH -- never from a preset name.

        The reference engine derives `L = h_hops + 1` boundaries for the path
        `E -> Eor -> parent E`, giving L = 3 for `tiled_cc`. Its NEST-time analogue is the
        sum of the actual assigned delays along that same structural path, plus the fast
        apical / confirmation / reset hops that close the loop:

            E -> Eor            (base_ff)
            Eor -> parent E     (base_ff + dispersion)
            parent E -> C       (apical, h)
            C -> I              (h)
            I -> E              (h)
        """
        def mean_delay(projection: str) -> float:
            vals = [self.delays.by_edge[e["id"]] for e in self.spec["edges"]
                    if e.get("projection") == projection]
            return sum(vals) / len(vals) if vals else 0.0

        return (mean_delay("column_e_to_eor")
                + mean_delay("column_to_column_ff")
                + mean_delay("column_to_column_apical")
                + mean_delay("column_c_to_i")
                + mean_delay("column_i_to_e"))

    # ------------------------------------------------------------------ stimulation
    def set_input_schedule(self, schedule: dict) -> None:
        """`{rgc_id: [spike times in ms]}`; every generator is set, absent ones cleared.

        Times are snapped onto the `h` grid. NEST rejects a spike time it cannot represent
        at the current resolution, and a derived period (such as the graph's own feedback
        loop latency) is almost never a whole number of steps. Snapping here keeps that a
        property of the stimulus rather than a trap every caller has to remember.
        """
        # ATOMIC: snap, then validate, then install. Mutating the generators first left a
        # rejected schedule installed when validation raised -- the caller saw an exception
        # and the kernel silently held the bad times.
        snapped = {}
        for nid in self.generators:
            snapped[nid] = sorted({round(round(float(t) / self.ts.h) * self.ts.h, 10)
                                   for t in schedule.get(nid, ())})

        # Validation happens on the REAL run path, not in a helper a test has to remember
        # to call. It reads the schedule and checks it against the profile's windows and
        # causal-volley separation; it never derives or mutates physiology from it. The
        # interval is measured from the ACTUAL scheduled times, so an irregular schedule is
        # checked on its smallest gap rather than on a nominal period.
        if self.is_continuous:
            interval = self._minimum_inter_volley_interval(snapped)
            if interval is not None:
                self.validate_against_schedule(min_inter_volley_interval_ms=interval)

        # Only now is anything installed.
        for nid, gen in self.generators.items():
            gen.set({"spike_times": snapped[nid]})

    @staticmethod
    def _minimum_inter_volley_interval(schedule: dict) -> float | None:
        """Smallest gap between DISTINCT volley times across the whole schedule.

        Times shared by several generators are one volley, so the set is collapsed first.
        Returns None when fewer than two volleys exist and there is nothing to check.
        """
        moments = sorted({t for times in schedule.values() for t in times})
        if len(moments) < 2:
            return None
        return min(b - a for a, b in zip(moments, moments[1:]))


    def simulate(self, duration_ms: float) -> None:
        self.nest.Simulate(float(duration_ms))
