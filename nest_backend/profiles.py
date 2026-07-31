"""Named engine profiles and the immutable physical-time configuration for each.

Phase 2 of `prompts/Claude_NEST_CIPP_Semantic_Repair_Prompt.md`.

Two profiles, kept apart on purpose (audit section 6):

`impulse_characterization`
    The existing event-accumulator backend. A useful characterization prototype with
    recorded semantic divergences from CIPP. Its parameters are preserved EXACTLY as they
    were, including the two known defects below, so its recorded findings stay
    reproducible. It is not a production CIPP engine and must not be promoted by tuning.

`cipp_continuous`
    The repair target: continuous postsynaptic dynamics, uniform reference conduction,
    physical coincidence windows, causal-volley learning state and counted prediction
    credits. Feasibility for the timing mechanism was established in
    `docs/CIPP_PHASE1_FEASIBILITY_REPORT.md`.

Every parameter below carries a UNIT and exactly ONE semantic role. Where the impulse
profile gave a parameter two roles -- most importantly the presentation interval, which it
used as both a stimulus schedule and a dendritic time constant -- the roles are separated
here and the stimulus one is removed from the neuron model entirely.

The profile objects are frozen. A run's physics is fixed when the profile is chosen, not
adjusted while an experiment is looking for an outcome.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclasses_fields

PROFILE_IMPULSE = "impulse_characterization"
PROFILE_CIPP_CONTINUOUS = "cipp_continuous"


class QuantizationPolicy:
    """How a requested delay is placed on the resolution grid.

    `CEIL` is the correct semantics and the one every docstring in this repository has
    always claimed. Its contract, stated over the ACTUAL float that comes back:

        result >= requested_delay
        result >= h
        result is the smallest representable integer multiple k*h satisfying both

    `ROUND` is the historical behaviour of `Timescales.quantize` -- a documented defect
    (audit P1). It is retained ONLY for `impulse_characterization`, whose recorded results
    were produced under it. Measured on the canonical 3x6 graph, switching that profile to
    CEIL would lengthen 88 of 254 edge delays, so silently "fixing" it would invalidate
    every negative finding the prototype exists to preserve. It is named a defect wherever
    it appears and is never presented as CIPP-correct.

    Why integer ticks and no final rounding
    +++++++++++++++++++++++++++++++++++++++

    An earlier version subtracted a fixed `1e-9` tolerance in tick units and rounded the
    result to ten decimals. Both steps could return a float strictly BELOW the request::

        0.10000000000001 -> 0.1        (tolerance swallowed the excess)
        3 * 0.1          -> 0.3        (rounding undid 0.30000000000000004)

    A contract that says "never shorter" may not return a smaller float. So the tick count
    is chosen by comparing the candidate products directly -- `(k-1)*h` and `k*h` against
    the request -- which also stops `0.30000000000000004` from spuriously selecting tick 4.
    The returned value is exactly `k * h`, never re-rounded.
    """

    CEIL = "ceil"
    ROUND = "round"
    ALL = (CEIL, ROUND)

    @staticmethod
    def apply(policy: str, delay_ms: float, h: float) -> float:
        """Place `delay_ms` on the `h` grid, never returning less than one `h`."""
        if not math.isfinite(h) or h <= 0:
            raise ValueError(f"resolution h must be finite and positive, got {h!r}")
        if not math.isfinite(delay_ms):
            raise ValueError(f"delay must be finite, got {delay_ms!r}")
        if delay_ms < 0:
            raise ValueError(f"delay must be non-negative, got {delay_ms!r}")

        if policy == QuantizationPolicy.ROUND:
            # Preserved historical behaviour, including its rounding to 10 decimals.
            return round(max(1, int(round(delay_ms / h))) * h, 10)
        if policy != QuantizationPolicy.CEIL:
            raise ValueError(f"unknown quantization policy {policy!r}")

        # Provisional tick count, then corrected by direct comparison so the answer is
        # decided by the floats that will actually be returned rather than by the quotient.
        k = max(1, math.ceil(delay_ms / h))
        while k > 1 and (k - 1) * h >= delay_ms:
            k -= 1
        while k * h < delay_ms:
            k += 1
        return k * h


@dataclass(frozen=True)
class DelayPolicy:
    """Conduction delay by projection class. Reference profile: UNIFORM within a class.

    The audit's P0 finding was that geometry and random jitter had come to DECIDE content:
    the backend converts layout coordinates into feedforward delays and the A/B scripts add
    up to 3 ms of per-synapse jitter to recover a single winner. Phase 1 showed that neither
    is needed -- with a continuous competitor, total drive alone orders the crossings, and
    latency after arrival varies by 1.8e-15 ms across a 16x range of conduction delay.

    So in the reference profile every equivalent edge in one projection gets the SAME delay.
    Layout distance keeps its one legitimate role, as a learning-rate multiplier (`phi`),
    and never touches delivery time. Geometry- and jitter-based delays remain available
    only through explicitly named ablation profiles.
    """

    rg_to_column_ms: float = 1.0          # retina -> column feedforward hop
    column_to_column_ff_ms: float = 1.0   # column -> parent column feedforward hop
    column_e_to_eor_ms: float = 1.0       # competitor -> column output relay
    column_e_to_i_ms: float = 0.01        # E -> I, the fast lateral arm
    column_i_to_e_ms: float = 0.01        # I -> E, the fast lateral return
    column_c_to_i_ms: float = 0.01        # C confirmation -> I
    column_to_column_apical_ms: float = 1.0
    column_eor_to_c_basal_ms: float = 1.0

    geometry_scales_delay: bool = False   # reference: geometry NEVER sets delay
    jitter_ms: float = 0.0                # reference: exactly zero

    def loop_latency_ms(self, relay_crossing_ms: float = 0.0) -> float:
        """Measured E->I->E latency. The relay's own crossing time is the floor.

        Phase 1 measured this rather than assuming it, and it is what sets the WTA
        operating envelope: two drives resolve to one winner only when their crossing gap
        exceeds this value.
        """
        return self.column_e_to_i_ms + relay_crossing_ms + self.column_i_to_e_ms


@dataclass(frozen=True)
class MembranePolicy:
    """Continuous competitor physiology. Units: mV, pF, ms.

    Shape and time constant come from Phase 1's candidate, `iaf_psc_exp_ps`: exact
    subthreshold integration between events with the off-grid crossing found by
    regula-falsi root finding.
    """

    model: str = "iaf_psc_exp_ps"
    v_threshold_mv: float = -55.0
    v_rest_mv: float = -70.0
    v_reset_mv: float = -70.0
    c_m_pf: float = 250.0
    tau_m_ms: float = 10.0
    tau_syn_ex_ms: float = 2.0        # excitatory synaptic-current time constant
    tau_syn_in_ms: float = 2.0
    t_ref_ms: float = 2.0             # firing refractory; gates FIRING, not accumulation

    @property
    def threshold_gap_mv(self) -> float:
        return self.v_threshold_mv - self.v_rest_mv


@dataclass(frozen=True)
class CoincidencePolicy:
    """C-cell windows, in PHYSICAL milliseconds and independent of stimulus pacing.

    This is the direct repair of audit P0 "cell physiology is coupled to the experiment
    controller". The impulse profile sets `tau_basal`, `tau_apical`, `tau_deposit_lock` and
    the I relay's `t_lockout` all equal to the presentation interval `D`, so changing how
    often the experiment presents input changes dendritic coincidence physiology, and a TTL
    exactly equal to `D` can pair an apical from one presentation with a basal from the
    adjacent one at the inclusive boundary.

    Here each window is a property of the CELL with one role:

    * `basal_window_ms` -- how long a delivered basal token stays eligible to pair;
    * `apical_window_ms` -- how long an apical depolarisation stays eligible to pair;
    * `deposit_dead_time_ms` -- minimum interval between two somatic deposits, so one
      prolonged coincidence cannot deposit repeatedly. A DEAD TIME, not a TTL, and
      deliberately unrelated to input pacing.
    """

    basal_window_ms: float = 5.0
    apical_window_ms: float = 5.0
    deposit_dead_time_ms: float = 1.0

    # PROVISIONAL, and marked as such. 5 ms is not a measured CIPP physiological constant;
    # it is a placeholder wide enough for the canonical graph's causal skew. The binding
    # constraints are checked in `validate`, which requires a MEASURED skew rather than
    # inferring one, and a minimum inter-volley interval so a window cannot pair across
    # volleys. Phase 1's convenient model defaults must not silently become CIPP physiology.
    provisional: bool = True

    def validate(self, *, causal_skew_ms: float,
                 min_inter_volley_interval_ms: float | None = None) -> None:
        """Windows must admit the graph's MEASURED causal skew and exclude cross-volley pairing.

        `causal_skew_ms` must be derived from the traversed basal and apical paths of the
        translated graph -- see `NestTiledNetwork.causal_arrival_envelope`. An earlier
        version estimated it as `abs(column_to_column_apical_ms - column_eor_to_c_basal_ms)`,
        which subtracts two TERMINAL edges and is not the causal skew at all: basal and
        apical evidence traverse different paths through local E->Eor, feedforward, parent
        integration and the return apical arm.
        """
        if not math.isfinite(causal_skew_ms) or causal_skew_ms < 0:
            raise ValueError(f"causal skew must be finite and non-negative, got {causal_skew_ms!r}")
        for name, value in (("basal_window_ms", self.basal_window_ms),
                            ("apical_window_ms", self.apical_window_ms),
                            ("deposit_dead_time_ms", self.deposit_dead_time_ms)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

        if min(self.basal_window_ms, self.apical_window_ms) <= causal_skew_ms:
            raise ValueError(
                f"coincidence windows ({self.basal_window_ms}, {self.apical_window_ms} ms) "
                f"cannot span the measured causal arrival skew of {causal_skew_ms} ms"
            )
        if self.deposit_dead_time_ms >= min(self.basal_window_ms, self.apical_window_ms):
            raise ValueError(
                f"deposit dead time {self.deposit_dead_time_ms} ms is not shorter than the "
                f"eligibility window; the window would be unreachable"
            )
        if min_inter_volley_interval_ms is not None:
            # A window at least as long as the gap between volleys can pair an apical from
            # one presentation with a basal from the next -- the inclusive-boundary failure
            # the impulse profile has by construction, since its windows ARE that gap.
            widest = max(self.basal_window_ms, self.apical_window_ms)
            if widest >= min_inter_volley_interval_ms:
                raise ValueError(
                    f"coincidence window {widest} ms is not shorter than the minimum "
                    f"inter-volley interval {min_inter_volley_interval_ms} ms; it could "
                    f"pair evidence across volleys"
                )


@dataclass(frozen=True)
class CausalVolleyPolicy:
    """What counts as "this volley" for LEARNING MEMBERSHIP only.

    Scope is deliberately narrow. This rule decides the participation sign `s_i` in the
    dual FE/FES update -- whether an afferent belongs to the causal volley that fired the
    cell -- and it decides nothing else. It is not a neuron time constant, it does not gate
    coincidence, and it does not schedule anything.

    `separation_ms` is a STRUCTURAL quantity: it must exceed the arrival spread of one
    volley and stay below the smallest interval between volleys, so the preceding volley is
    excluded. With uniform delays the spread within a projection is zero, so this reduces
    to a small tolerance rather than the impulse profile's `tau_volley` derived from
    geometric spread and then reused across projections with different delays.

    Participation is an INTERVAL, `[participate_from, participate_until]`, in the
    connection's delay-adjusted time coordinate. The impulse profile stores only the upper
    bound and arms it at send time, which is why an afferent that has not yet arrived can
    be scored `+1` (audit P1, contract 4).
    """

    separation_ms: float = 0.5
    require_arrival_before_firing: bool = True   # the missing lower bound


@dataclass(frozen=True)
class PredictionCreditPolicy:
    """One confirmation creates one credit, consumed by the next eligible evidence volley.

    The repair of audit P0 "confirmation feedback is time-triggered state erasure". A reset
    event can only clear charge that has already arrived, and a fixed refractory merely
    covers an expected volley -- both make one prediction's meaning depend on a timer and
    on presentation pacing. A counted credit does not: it survives arbitrary silence and is
    spent by the next eligible evidence, whenever that arrives.
    """

    capacity: int = 1                      # credits a column may hold at once
    credits_per_confirmation: int = 1
    consumed_per_volley: int = 1
    silence_consumes_credit: bool = False  # silence must NOT spend a credit
    wta_reset_affects_credit: bool = False # WTA traffic neither creates nor consumes one
    # OVERFLOW IS SATURATION, declared rather than left to the implementation. A column
    # already holding `capacity` credits that receives another confirmation stays at
    # `capacity`; the surplus is dropped, not queued. Queuing would let a burst of
    # confirmations suppress an unbounded run of later evidence, which is a different
    # scientific claim from "one prediction suppresses the next evidence".
    overflow: str = "saturate"

    VALID_OVERFLOW = ("saturate",)

    def validate(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be at least one credit, got {self.capacity}")
        if self.credits_per_confirmation < 1:
            raise ValueError("a confirmation must create >= 1 credit")
        if self.consumed_per_volley < 1:
            raise ValueError("a volley must consume >= 1 credit")
        if self.overflow not in self.VALID_OVERFLOW:
            raise ValueError(
                f"unknown overflow behaviour {self.overflow!r}; have {self.VALID_OVERFLOW}"
            )
        if self.consumed_per_volley > self.capacity:
            raise ValueError(
                f"a volley consumes {self.consumed_per_volley} credits but the column can "
                f"hold only {self.capacity}; no volley could ever be suppressed"
            )
        if self.credits_per_confirmation > self.capacity:
            raise ValueError(
                f"a confirmation creates {self.credits_per_confirmation} credits but "
                f"capacity is {self.capacity}; under saturation the surplus is silently "
                f"dropped, so the setting cannot be represented coherently"
            )

    def after_confirmation(self, held: int) -> int:
        """Credits held after one confirmation, under the declared overflow behaviour."""
        return min(self.capacity, held + self.credits_per_confirmation)


@dataclass(frozen=True)
class WtaPolicy:
    """What happens when two competitors cross at the same instant.

    The custom engine arbitrates crossings within `1e-12` NORMALISED outer-boundary time
    using stable repository node order, and records the event in `latency_ties`. That
    tolerance is expressed in normalised boundary units, not milliseconds, and translating
    it without a declared mapping would invent a physical constant.

    Phase 1's built-in NEST microcircuit has no arbiter and correctly reports an exact tie
    as unresolved. So until a local arbiter is specified, `cipp_continuous` declares the
    honest state: unresolved.

    `implicit_gid_tie_break_allowed` is present to be asserted `False`. NEST GID, creation
    order, connection insertion order, geometry and jitter may NEVER decide a winner --
    Phase 1 found exactly that bug (`min((time, index))` resolving ties by node index) and
    Phase 1 gate 6 now proves the decision is invariant to GID and creation order.
    """

    exact_tie_policy: str = "unresolved"
    implicit_gid_tie_break_allowed: bool = False
    # Undeclared on purpose. A physical tolerance requires a stated mapping from the custom
    # engine's normalised-time arbiter, a stable ordering key, and an observable tie record.
    # That is Phase 3 work; a number here now would be a guess presented as physics.
    tie_tolerance_ms: float | None = None
    tie_ordering_key: str | None = None

    VALID_POLICIES = ("unresolved", "local_arbiter")

    def validate(self) -> None:
        if self.exact_tie_policy not in self.VALID_POLICIES:
            raise ValueError(
                f"unknown exact_tie_policy {self.exact_tie_policy!r}; "
                f"have {self.VALID_POLICIES}"
            )
        if self.implicit_gid_tie_break_allowed:
            raise ValueError(
                "implicit GID/creation-order tie-breaking is never permitted: a winner "
                "manufactured from node order is not a scientific result"
            )
        if self.exact_tie_policy == "local_arbiter":
            if self.tie_tolerance_ms is None or self.tie_ordering_key is None:
                raise ValueError(
                    "a local arbiter must declare both a physical tie tolerance (ms) and "
                    "a stable ordering key before it may be selected"
                )
        elif self.tie_tolerance_ms is not None:
            raise ValueError(
                "exact_tie_policy='unresolved' must not carry a tie tolerance; an "
                "undeclared arbiter with a tolerance is an implicit tie-breaker"
            )


@dataclass(frozen=True)
class EngineProfile:
    """One immutable physical-time configuration. Frozen: a run's physics is fixed."""

    name: str
    resolution_h_ms: float
    quantization: str
    delays: DelayPolicy = field(default_factory=DelayPolicy)
    membrane: MembranePolicy = field(default_factory=MembranePolicy)
    coincidence: CoincidencePolicy = field(default_factory=CoincidencePolicy)
    causal_volley: CausalVolleyPolicy = field(default_factory=CausalVolleyPolicy)
    prediction: PredictionCreditPolicy = field(default_factory=PredictionCreditPolicy)
    wta: WtaPolicy = field(default_factory=WtaPolicy)
    # Declared, not implied: no neuron parameter in this profile is derived from the
    # stimulus schedule. The experiment controller may schedule presentations; it may not
    # set dendritic TTLs, relay lockout, or coincidence physiology.
    neuron_physics_independent_of_pacing: bool = True

    def quantize(self, delay_ms: float) -> float:
        return QuantizationPolicy.apply(self.quantization, delay_ms, self.resolution_h_ms)

    def delay_fields(self) -> dict:
        """Every projection delay by name, so validation cannot miss one by omission."""
        return {f.name: getattr(self.delays, f.name)
                for f in dataclasses_fields(self.delays)
                if f.name.endswith("_ms") and f.name != "jitter_ms"}

    def validate(self, *, causal_skew_ms: float | None = None,
                 min_inter_volley_interval_ms: float | None = None) -> None:
        """Structural validation. Graph-dependent checks need measured inputs.

        `causal_skew_ms` is omitted for a bare profile check and supplied at run
        construction, where the traversed basal/apical paths are known. A profile cannot
        validate its coincidence windows against a graph it has not seen.
        """
        if not math.isfinite(self.resolution_h_ms) or self.resolution_h_ms <= 0:
            raise ValueError(f"resolution must be finite and positive, got {self.resolution_h_ms!r}")
        if self.quantization not in QuantizationPolicy.ALL:
            raise ValueError(f"unknown quantization policy {self.quantization!r}")

        # EVERY projection delay, not just the lateral pair.
        for name, value in self.delay_fields().items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
            if value < self.resolution_h_ms:
                raise ValueError(
                    f"{name} = {value} ms is below the resolution {self.resolution_h_ms} ms; "
                    f"NEST has no sub-h delay"
                )
        if not math.isfinite(self.delays.jitter_ms) or self.delays.jitter_ms < 0:
            raise ValueError(f"jitter must be finite and non-negative, got {self.delays.jitter_ms!r}")

        membrane = self.membrane
        if not membrane.v_threshold_mv > membrane.v_rest_mv:
            raise ValueError(
                f"threshold {membrane.v_threshold_mv} mV must exceed rest "
                f"{membrane.v_rest_mv} mV or the cell fires without input"
            )
        if not membrane.v_threshold_mv > membrane.v_reset_mv:
            raise ValueError("reset must sit below threshold")
        for name, value in (("tau_m_ms", membrane.tau_m_ms),
                            ("tau_syn_ex_ms", membrane.tau_syn_ex_ms),
                            ("tau_syn_in_ms", membrane.tau_syn_in_ms),
                            ("c_m_pf", membrane.c_m_pf)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"membrane {name} must be finite and positive, got {value!r}")
        if not math.isfinite(membrane.t_ref_ms) or membrane.t_ref_ms < 0:
            raise ValueError(f"refractory must be finite and non-negative, got {membrane.t_ref_ms!r}")

        rule = self.causal_volley
        if not math.isfinite(rule.separation_ms) or rule.separation_ms <= 0:
            raise ValueError(f"causal-volley separation must be positive, got {rule.separation_ms!r}")

        self.prediction.validate()
        self.wta.validate()
        if causal_skew_ms is not None:
            self.coincidence.validate(
                causal_skew_ms=causal_skew_ms,
                min_inter_volley_interval_ms=min_inter_volley_interval_ms)

    def validate_schedule(self, *, within_volley_spread_ms: float,
                          min_inter_volley_interval_ms: float) -> None:
        """`within_volley_spread < separation_ms < minimum_inter_volley_interval`.

        Consumes schedule information for VALIDATION ONLY. Nothing here mutates or derives
        neuron physiology from the schedule -- that separation is the whole point of Phase 2.
        """
        separation = self.causal_volley.separation_ms
        if not within_volley_spread_ms < separation:
            raise ValueError(
                f"causal-volley separation {separation} ms does not exceed the measured "
                f"within-volley arrival spread {within_volley_spread_ms} ms; afferents of "
                f"one volley would be split across two"
            )
        if not separation < min_inter_volley_interval_ms:
            raise ValueError(
                f"causal-volley separation {separation} ms is not below the minimum "
                f"inter-volley interval {min_inter_volley_interval_ms} ms; the preceding "
                f"volley would be counted as causal"
            )

    def describe(self) -> dict:
        """Flat, serialisable, unit-bearing -- for the run manifest and replay header."""
        out = asdict(self)
        out["loop_latency_ms_floor"] = self.delays.loop_latency_ms()
        return out


# `h` is chosen from the COMMUNICATION-DELAY ENVELOPE, not from crossing precision.
# Phase 1 gate 3 showed the crossing time is already converged at h=0.1 (spread 3.55e-15 ms
# across h from 0.1 to 0.001), so a finer grid buys no timing accuracy. What it does buy is
# a lower floor on `d_ei`/`d_ie`, and those set the loop latency that sets the WTA envelope:
#
#     loop 0.2042 ms -> resolves drive ratios up to 0.769
#     loop 0.0242 ms -> resolves drive ratios up to 0.9625
#
# h = 0.01 puts the loop at ~0.0242 ms, which resolves competitors differing by under 4%.
CIPP_CONTINUOUS = EngineProfile(
    name=PROFILE_CIPP_CONTINUOUS,
    resolution_h_ms=0.01,
    quantization=QuantizationPolicy.CEIL,
)

# The existing backend, described rather than redefined. Its two known defects are named
# here so they are visible in every manifest instead of living only in the audit.
IMPULSE_CHARACTERIZATION = EngineProfile(
    name=PROFILE_IMPULSE,
    resolution_h_ms=0.1,
    quantization=QuantizationPolicy.ROUND,      # DEFECT, preserved for reproducibility
    delays=DelayPolicy(
        # The prototype's lateral arms sit at the resolution floor -- one `h` per hop, which
        # is where its declared `L_wta = 2h` comes from. Stating 0.01 here while the profile
        # runs at h=0.1 would describe a circuit NEST cannot build, and `validate` says so.
        column_e_to_i_ms=0.1,
        column_i_to_e_ms=0.1,
        column_c_to_i_ms=0.1,
        geometry_scales_delay=True,             # DEFECT: layout coordinates set delay
        jitter_ms=0.0,                          # the DEFAULT; A/B runs override it to 3.0
    ),
    neuron_physics_independent_of_pacing=False,  # DEFECT: windows are set from D
)

PROFILES = {p.name: p for p in (CIPP_CONTINUOUS, IMPULSE_CHARACTERIZATION)}


def get_profile(name: str) -> EngineProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown engine profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]
