"""Phase 0 regression probes for the CIPP semantic repair.

Source: `docs/CIPP_CUSTOM_ENGINE_TO_NEST_SEMANTIC_AUDIT.md`, implemented per
`prompts/Claude_NEST_CIPP_Semantic_Repair_Prompt.md` Phase 0.

These probes FREEZE EVIDENCE. Each pins one audit finding to an executable measurement, so
the repair is judged against a fixed target rather than a prose claim.

Reading the markers
+++++++++++++++++++

`xfail(strict=True, raises=AssertionError)` states the CORRECT post-repair CIPP contract
and is expected to fail against today's `impulse_characterization` profile.

Both keywords carry weight:

* `strict=True` -- when the repair lands the probe passes, pytest reports XPASS as a
  FAILURE, and whoever fixed it must return here and drop the marker. A non-strict xfail
  would let a repaired contract go silently unrecorded.
* `raises=AssertionError` -- an expected failure may ONLY come from the measurement
  disagreeing with the contract. A typo, a missing fixture, a renamed NEST parameter or any
  other API error raises something else, which pytest then reports as a hard ERROR instead
  of quietly counting as "expected". Without this, a probe that never actually ran would
  look identical to a probe that ran and correctly failed.

Probes with no marker document behaviour that is already correct and must not regress.

Every probe states its measured numbers in its docstring, so the size of each divergence is
readable without running anything.
"""
from __future__ import annotations

import pytest

from nest_backend.profiles import PROFILE_IMPULSE

E_THRESHOLD = 1000.0

# Every probe in this file characterises the CURRENT backend. When `cipp_continuous`
# exists it gets its own suite; these must keep measuring the profile whose findings they
# froze, or the audit trail is lost.
PROFILE_UNDER_TEST = PROFILE_IMPULSE


# --------------------------------------------------------------------------- helpers
def _ports(nest, model):
    return nest.GetDefaults(model).get("receptor_types") or {}


def _accumulator(nest, theta=E_THRESHOLD, record=("q", "q_pre")):
    """One `event_accumulator` with a spike recorder and a per-step multimeter."""
    node = nest.Create("event_accumulator", 1, {"theta": theta, "q_reset": 0.0})
    recorder = nest.Create("spike_recorder")
    meter = nest.Create("multimeter", 1, {"interval": nest.resolution,
                                          "record_from": list(record)})
    nest.Connect(node, recorder)
    nest.Connect(meter, node)
    return node, recorder, meter


def _drive(nest, node, times, charge, delay=1.0, port="EXC"):
    gen = nest.Create("spike_generator", 1, {"spike_times": [float(t) for t in times]})
    spec = {"weight": float(charge), "delay": float(delay)}
    ports = _ports(nest, "event_accumulator")
    if ports:
        spec["receptor_type"] = ports[port]
    nest.Connect(gen, node, syn_spec=spec)
    return gen


def _small_net(presentation=15.0, h=0.1, jitter=0.0, learning=False):
    from nest_backend.topology import NestTiledNetwork, Timescales
    return NestTiledNetwork(
        seed=1, shape=(3, 6), jitter_ms=jitter, learning=learning,
        timescales=Timescales(h=h, base_ff=1.0, spread=2.0, presentation=presentation),
    )


# A NESTML synapse generated as a pair only attaches to its paired neuron. Connecting it
# to the plain `event_accumulator` does not raise -- it aborts the interpreter -- so the
# names are bound together here rather than spelled out at each call site.
PLASTIC_SYN = "plastic_feedforward_synapse__with_event_accumulator"
PLASTIC_TARGET = "event_accumulator__with_plastic_feedforward_synapse"


def _plastic_target(nest, theta=E_THRESHOLD):
    """The paired accumulator, plus a spike recorder."""
    node = nest.Create(PLASTIC_TARGET, 1, {"theta": theta, "q_reset": 0.0})
    recorder = nest.Create("spike_recorder")
    nest.Connect(node, recorder)
    return node, recorder, _ports(nest, PLASTIC_TARGET)


def _hard_driver(nest, target, ports, at_ms):
    """A static input strong enough to fire `target` one delay later."""
    drv = nest.Create("spike_generator", 1, {"spike_times": [float(at_ms)]})
    spec = {"weight": E_THRESHOLD, "delay": 1.0}
    if ports:
        spec["receptor_type"] = ports["EXC"]
    nest.Connect(drv, target, syn_spec=spec)
    return drv


# =====================================================================================
# CONTRACT 1 -- ordinary-E `I_accq` is the current causal volley
# =====================================================================================
def test_c1_retained_charge_is_the_full_accumulated_total(fresh_kernel):
    """The RETAINED total is correct today and must stay correct after the repair.

    Three 400-charge volleys into theta=1000, measured::

        q just before firing   =  800.0
        q_pre at firing        = 1200.0   <- correct: the whole retained charge

    The repair ADDS a causal-volley quantity; it must not damage the pure-integrator sum,
    which is the thing the impulse prototype gets exactly right (audit section 2). This
    probe and `test_c1_causal_volley_...` are complementary, not contradictory: 1200 is
    the right answer for `q_pre`, and 400 is the right answer for a DIFFERENT variable
    that does not exist yet.
    """
    nest = fresh_kernel(0.1)
    node, recorder, meter = _accumulator(nest)
    _drive(nest, node, [10.0, 20.0, 30.0], charge=400.0)
    nest.Simulate(40.0)

    assert len(recorder.get("events")["times"]) == 1, "must fire on the third volley"
    events = meter.get("events")
    just_before = [q for t, q in zip(events["times"], events["q"]) if 30.5 < t < 31.0]
    assert just_before and just_before[-1] == pytest.approx(800.0)
    assert max(events["q_pre"]) == pytest.approx(1200.0), (
        "retained charge at firing must be the full accumulated total"
    )


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P0: the model exposes no causal-volley quantity at all. The Python oracle "
    "reports V_pre=1200 AND I_accq=400 for this stimulus -- two separate numbers. NEST "
    "has only `q_pre`=1200 and feeds that to FE. Repair requires a second state variable "
    "`q_causal_volley`, declared recordable, snapshotted at firing."))
def test_c1_causal_volley_charge_is_recorded_separately_from_the_total(fresh_kernel):
    """`I_accq` is the FIRING volley's delivered packet, not the retained membrane total.

    Measured, both engines, three 400-charge volleys into theta=1000::

        Python oracle at firing:  V_pre = 1200.0   I_accq =  400.0
        NEST at firing:           q_pre = 1200.0   q_causal_volley = (absent)

    FE is a function of `I_accq`, so this is a 3x error in the learning drive of every
    cell that crosses threshold after accumulating more than one presentation.
    """
    nest = fresh_kernel(0.1)
    recordables = [str(v) for v in nest.GetDefaults("event_accumulator")["recordables"]]
    # Asserted, not looked up: a missing recordable must fail the CONTRACT, not raise a
    # NEST KeyError that `raises=AssertionError` would report as a hard error.
    assert "q_causal_volley" in recordables, (
        f"event_accumulator must declare a causal-volley recordable; has {recordables}"
    )

    node, recorder, meter = _accumulator(nest, record=("q", "q_pre", "q_causal_volley"))
    _drive(nest, node, [10.0, 20.0, 30.0], charge=400.0)
    nest.Simulate(40.0)

    events = meter.get("events")
    at_firing = [v for t, v in zip(events["times"], events["q_causal_volley"])
                 if 31.0 <= t <= 31.2]
    assert at_firing and at_firing[0] == pytest.approx(400.0), (
        f"I_accq must be the firing volley's packet (400), got {at_firing}"
    )
    assert max(events["q_pre"]) == pytest.approx(1200.0), (
        "and the retained total must STILL be 1200 -- both quantities, separately"
    )


# =====================================================================================
# CONTRACT 2 -- C accumulation vs one-shot
# =====================================================================================
def _coincidence(nest, basal_weight, times, tau=5.0):
    ports = _ports(nest, "event_coincidence")
    cell = nest.Create("event_coincidence", 1, {
        "theta": E_THRESHOLD, "t_ref": 0.0,
        "tau_basal": tau, "tau_apical": tau, "tau_deposit_lock": 1.0,
    })
    recorder = nest.Create("spike_recorder")
    meter = nest.Create("multimeter", 1, {"interval": 0.1, "record_from": ["q"]})
    nest.Connect(cell, recorder)
    nest.Connect(meter, cell)
    basal = nest.Create("spike_generator", 1, {"spike_times": list(times)})
    apical = nest.Create("spike_generator", 1, {"spike_times": list(times)})
    nest.Connect(basal, cell, syn_spec={"weight": basal_weight, "delay": 1.0,
                                        "receptor_type": ports["BASAL"]})
    nest.Connect(apical, cell, syn_spec={"weight": 1.0, "delay": 1.0,
                                         "receptor_type": ports["APICAL"]})
    return cell, recorder, meter


def test_c2_c_soma_accumulates_repeated_subthreshold_deposits(fresh_kernel):
    """A C at basal `theta/4` is NOT one-shot, but it is not inert either.

    The NEST report claimed such a C 'can never fire under frozen learning'; the audit
    showed that confuses 'not one-shot' with 'cannot accumulate'. Four valid coincidences
    of theta/4 reach theta. Audit measurements on the full circuit::

        8 presentations:    2 deposits, 0 spikes
        20 presentations:   5 deposits, 1 spike
        100 presentations: 26 deposits, 6 spikes

    UNMARKED: the soma already accumulates correctly. Frozen because Phase 5 rewrites the
    coincidence cell and must not lose it.
    """
    nest = fresh_kernel(0.1)
    _cell, recorder, meter = _coincidence(nest, E_THRESHOLD / 4, [10.0, 30.0, 50.0, 70.0])
    nest.Simulate(90.0)

    assert list(recorder.get("events")["times"]), (
        "a theta/4 C must reach threshold by cumulative deposit"
    )
    assert max(meter.get("events")["q"]) > E_THRESHOLD / 4, "deposits must accumulate"


def test_c2_c_at_full_theta_is_one_shot(fresh_kernel):
    """The contrast case: basal at `theta` fires on the FIRST valid coincidence.

    With the probe above this separates the three conditions the audit says are currently
    confounded: immature accumulation, mature one-shot, and feedback effect.
    """
    nest = fresh_kernel(0.1)
    _cell, recorder, _meter = _coincidence(nest, E_THRESHOLD, [10.0])
    nest.Simulate(30.0)
    assert len(recorder.get("events")["times"]) == 1, "mature C is one-shot"


# =====================================================================================
# CONTRACT 3 -- owner identity is reported, not just multiplicity
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P0: `winner_multiplicity` reduces each window's winner set to `len(winners)`, "
    "discarding identity; `firing_pattern` keeps only fire/silent. The assembled pair "
    "report therefore cannot answer 'which cell owned this pattern, how stably, and did "
    "it agree with the Python owner'. Measured divergence the current report cannot "
    "surface: Python owner L1c00E4 against NEST owners E7,E4,E4,E7,E6,E6,E7 across seven "
    "windows -- three distinct owners, labelled 'matched'. Repair must carry per-window "
    "winner identity, distinct-owner count, modal owner, stability and agreement through "
    "to the emitted artifact (Phase 7)."))
@pytest.mark.parametrize("python_owner_index, expect_agreement", [(1, True), (0, False)])
def test_c3_pair_report_carries_owner_identity_and_agreement(
        fresh_kernel, python_owner_index, expect_agreement):
    """The EMITTED A/B section must carry the representational outcome, not just counts.

    This exercises `build_nest_section` -- the assembly path that produces the NEST half
    of `pair.json` -- rather than the metrics helper underneath it, because the artifact
    shape is what a reader actually gets.

    The synthetic history reproduces the recorded divergence: exactly one winner in every
    window, so multiplicity is a flat 1.0 and looks perfect, while the OWNER rotates.
    Three windows with owners [E0, E1, E1], modal owner E1, stability 2/3.

    Required section, per column::

        winners_by_window  = {0: [E0], 1: [E1], 2: [E1]}
        distinct_owners    = 2
        owner              = E1        (modal)
        stability          = 2/3
        agrees_with_python = True when the supplied Python owner is E1
                             False when it is E0

    Both cases are run. An implementation that hard-codes agreement, or derives it from
    anything other than the modal owner, passes one and fails the other.
    """
    from experiments.nest_vs_engine_ab import build_nest_section

    fresh_kernel(0.1)
    net = _small_net()
    column = next(c for c in net.column_ids() if c.startswith("L1"))
    e_cells = sorted(n for n, m in net.node_meta.items()
                     if m.get("column_id") == column and m.get("column_role") == "E")
    first, second = e_cells[0], e_cells[1]

    period = 15.0
    spikes = [(5.0, first), (20.0, second), (35.0, second)]
    supplied_owner = e_cells[python_owner_index]

    section = build_nest_section(net, spikes, t0=0.0, period=period,
                                 python_owners={column: supplied_owner})

    assert "columns" in section, f"pair section must report per column; got {sorted(section)}"
    assert column in section["columns"], "the driven column must appear in the section"
    report = section["columns"][column]

    required = ("winners_by_window", "distinct_owners", "owner", "stability",
                "agrees_with_python")
    missing = [k for k in required if report.get(k) is None]
    assert not missing, (
        f"emitted NEST section omits {missing}; it carries {sorted(report)}"
    )

    assert report["winners_by_window"] == {0: [first], 1: [second], 2: [second]}, (
        f"per-window identities wrong: {report['winners_by_window']}"
    )
    assert report["distinct_owners"] == 2, (
        f"two different cells won; distinct_owners = {report['distinct_owners']}"
    )
    assert report["owner"] == second, f"modal owner should be {second}"
    assert report["stability"] == pytest.approx(2 / 3), (
        f"the modal owner held 2 of 3 windows; stability = {report['stability']}"
    )
    assert report["agrees_with_python"] is expect_agreement, (
        f"supplied Python owner {supplied_owner} against modal NEST owner "
        f"{report['owner']}: agreement must be {expect_agreement}"
    )
    # And the thing multiplicity alone would have said about this run: perfect.
    assert section["winner_multiplicity"]["mean"] == pytest.approx(1.0), (
        "multiplicity is a flat 1.0 across a rotating owner -- which is exactly why it "
        "cannot stand in for identity"
    )


# =====================================================================================
# CONTRACT 4 -- a late same-volley afferent is NOT a participant
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P1: `plastic_feedforward_synapse` arms `participate_until = t + tau_volley` in "
    "the PRESYNAPTIC handler, i.e. at send time, and the post handler tests only "
    "`t <= participate_until`. There is no lower bound, so an afferent whose event has "
    "been sent but has NOT YET ARRIVED is scored +1. Repair requires tracking an interval "
    "with both `participate_from` and `participate_until` in the connection's "
    "delay-adjusted time coordinate."))
def test_c4_afferent_arriving_after_the_post_spike_is_depressed(fresh_kernel):
    """`s_i = +1` requires the event to have REACHED the target before it fired.

    Schedule: the afferent is emitted at t=10 with a 5 ms delay, so it lands at t=15. A
    separate strong input fires the target at t=12 -- three milliseconds BEFORE the
    afferent arrives. That afferent contributed nothing to the firing and must receive
    -1, so its weight must DECREASE from 250.
    """
    import nest as _nest

    nest = fresh_kernel(0.1)
    target, recorder, ports = _plastic_target(nest)

    # The second presynaptic spike at t=100 exists ONLY to force materialization of the
    # post-triggered update. Without it this probe measures the flush gap (contract 6) --
    # the weight stays at exactly 250.0 because the synapse is never revisited -- and the
    # SIGN question it is meant to isolate is never reached. It is far outside the first
    # spike's volley, so it cannot itself contribute a +1.
    late = nest.Create("spike_generator", 1, {"spike_times": [10.0, 100.0]})
    spec = {"synapse_model": PLASTIC_SYN, "w": 250.0, "delay": 5.0, "tau_volley": 3.0}
    if ports:
        spec["receptor_type"] = ports["EXC"]
    nest.Connect(late, target, syn_spec=spec)
    _hard_driver(nest, target, ports, 11.0)
    nest.Simulate(140.0)

    fired = [float(t) for t in recorder.get("events")["times"]]
    assert fired and fired[0] == pytest.approx(12.0, abs=0.3), (
        f"setup: target must fire at ~12 ms, before the late afferent lands; got {fired}"
    )
    final = float(_nest.GetConnections(source=late, target=target).get("w"))
    assert final != 250.0, (
        "setup: the post update never materialized, so this probe measured the flush gap "
        "(contract 6) rather than the participation sign"
    )
    assert final < 250.0, (
        f"a late afferent must be depressed (s_i=-1); weight went 250.0 -> {final}, "
        "i.e. it was rewarded as a causal participant despite arriving after the firing"
    )


# =====================================================================================
# CONTRACT 5 -- multiple deferred posts use each post's own historical FE
# =====================================================================================
def test_c5_two_posts_between_pre_spikes_each_use_their_own_snapshot(fresh_kernel):
    """Two postsynaptic spikes between two presynaptic spikes -- PASSES.

    The audit called this "not proven wrong, but not accepted": the generated C++ iterates
    archived post spikes and reads continuous post state at each post time, so deferred
    materialization CAN be equivalent, but nothing established it. This probe establishes
    it for the two-post case.

    The logical transaction is two SEQUENTIAL updates: each post reads ITS OWN historical
    `I_accq`, and each uses the synapse weight as evolved by the previous update. Both
    historical charges are read back from the target's recorded `q_pre` rather than
    assumed -- an earlier version of this probe hard-coded them as (1000, 1000), forgot
    that the plastic afferent itself deposits 250 at t=6, and so compared NEST against the
    wrong number. The measured charges are 1250 and 1000.

    Measured::

        oracle over the recorded snapshots (1250, 1000) = 247.1679082206652
        NEST                                            = 247.1679082206652

    Exact to the last bit, which is the evidence the audit asked for.
    """
    import nest as _nest

    nest = fresh_kernel(0.1)
    target, recorder, ports = _plastic_target(nest)
    meter = nest.Create("multimeter", 1, {"interval": nest.resolution,
                                          "record_from": ["q_pre"]})
    nest.Connect(meter, target)

    w0, eta, B, e_floor, w_te, w_cap = 250.0, 4.0, 5.0, 0.001, 0.001, 500.0
    pre = nest.Create("spike_generator", 1, {"spike_times": [5.0, 60.0]})
    spec = {"synapse_model": PLASTIC_SYN, "w": w0, "delay": 1.0, "tau_volley": 3.0,
            "eta": eta, "theta": E_THRESHOLD, "B": B, "e_floor": e_floor,
            "w_te": w_te, "w_cap": w_cap, "phi": 1.0}
    if ports:
        spec["receptor_type"] = ports["EXC"]
    nest.Connect(pre, target, syn_spec=spec)
    for at in (19.0, 39.0):
        _hard_driver(nest, target, ports, at)
    nest.Simulate(80.0)

    fired = [float(t) for t in recorder.get("events")["times"]]
    assert len(fired) == 2, f"setup: target must fire exactly twice, got {fired}"

    # The snapshot each post actually carried, read off the target rather than assumed.
    events = meter.get("events")
    snapshots = []
    for spike_time in fired:
        at = [v for t, v in zip(events["times"], events["q_pre"])
              if spike_time <= t < spike_time + 0.15]
        assert at, f"setup: no q_pre sample at the spike at {spike_time}"
        snapshots.append(float(at[0]))
    assert snapshots == pytest.approx([1250.0, 1000.0]), (
        f"setup: expected historical charges [1250, 1000], measured {snapshots}"
    )

    def update(w, i_accq):
        fe = e_floor + (1.0 - e_floor) / (1.0 + B * (i_accq / E_THRESHOLD - 0.5) ** 2)
        fes = w_te + (1.0 - w_te) / (1.0 + B * (2.0 * w / E_THRESHOLD - 0.5) ** 2)
        return min(max(w + eta * fe * fes * -1.0 * 1.0, w_te), w_cap)

    # Both posts fall far outside the pre spike's 3 ms volley, so both score s_i = -1.
    expected = w0
    for snapshot in snapshots:
        expected = update(expected, snapshot)

    actual = float(_nest.GetConnections(source=pre, target=target).get("w"))
    assert actual == pytest.approx(expected, rel=1e-12), (
        f"two sequential post updates must give {expected!r}, got {actual!r}"
    )


# =====================================================================================
# CONTRACT 6 -- a silent afferent's negative updates are still materialized
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P1: NEST advances a synapse on its PRESYNAPTIC events. An afferent that never "
    "spikes again is never revisited, so its post-triggered negative updates are never "
    "materialized and a weight read returns the stale value. NEST's own e-prop example "
    "forces presynaptic activity before final weight readout, which is precedent for "
    "requiring an explicit flush -- not evidence that stale reads are harmless. Repair "
    "requires a non-perturbing flush before checkpoint, convergence decision, weight "
    "export, replay header or final metric."))
def test_c6_silent_afferent_weight_includes_post_triggered_updates(fresh_kernel):
    """One presynaptic spike, then three postsynaptic spikes, then read the weight.

    The afferent never fires again, so nothing ever revisits the synapse. Its three `-1`
    updates are logically real and must appear in any exported weight.

    Asserted as an EXACT value, not merely `< 250`. Contract 5 established that NEST
    applies the sequential dual FE/FES updates correctly once it is provoked into doing
    so, so the only thing separating the stored weight from the logical weight here is the
    flush. That makes the target computable rather than approximate::

        historical I_accq = 1250, 1000, 1000
        logical final weight = 245.38819342785618
        stored weight        = 250.0   (untouched)

    A `< 250` assertion would also be satisfied by a partial or wrongly-signed flush.
    """
    import nest as _nest

    nest = fresh_kernel(0.1)
    target, recorder, ports = _plastic_target(nest)
    meter = nest.Create("multimeter", 1, {"interval": nest.resolution,
                                          "record_from": ["q_pre"]})
    nest.Connect(meter, target)

    w0, eta, B, e_floor, w_te, w_cap = 250.0, 4.0, 5.0, 0.001, 0.001, 500.0
    pre = nest.Create("spike_generator", 1, {"spike_times": [5.0]})
    spec = {"synapse_model": PLASTIC_SYN, "w": w0, "delay": 1.0, "tau_volley": 3.0,
            "eta": eta, "theta": E_THRESHOLD, "B": B, "e_floor": e_floor,
            "w_te": w_te, "w_cap": w_cap, "phi": 1.0}
    if ports:
        spec["receptor_type"] = ports["EXC"]
    nest.Connect(pre, target, syn_spec=spec)
    for at in (19.0, 39.0, 59.0):
        _hard_driver(nest, target, ports, at)
    nest.Simulate(80.0)

    fired = [float(t) for t in recorder.get("events")["times"]]
    assert len(fired) == 3, f"setup: target must fire three times, got {fired}"

    events = meter.get("events")
    snapshots = []
    for spike_time in fired:
        at = [v for t, v in zip(events["times"], events["q_pre"])
              if spike_time <= t < spike_time + 0.15]
        assert at, f"setup: no q_pre sample at the spike at {spike_time}"
        snapshots.append(float(at[0]))
    assert snapshots == pytest.approx([1250.0, 1000.0, 1000.0]), (
        f"setup: expected historical charges [1250, 1000, 1000], measured {snapshots}"
    )

    def update(w, i_accq):
        fe = e_floor + (1.0 - e_floor) / (1.0 + B * (i_accq / E_THRESHOLD - 0.5) ** 2)
        fes = w_te + (1.0 - w_te) / (1.0 + B * (2.0 * w / E_THRESHOLD - 0.5) ** 2)
        return min(max(w + eta * fe * fes * -1.0 * 1.0, w_te), w_cap)

    expected = w0
    for snapshot in snapshots:
        expected = update(expected, snapshot)

    actual = float(_nest.GetConnections(source=pre, target=target).get("w"))
    assert actual == pytest.approx(expected, rel=1e-12), (
        f"the logical weight after three post-triggered updates is {expected!r}; "
        f"the stored weight is {actual!r} -- the updates were never materialized"
    )


# =====================================================================================
# CONTRACT 7 -- prediction is a counted credit, not a timer
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P0: a reset event can only clear charge that has ALREADY arrived, and "
    "`t_ref_reset` suppresses a fixed duration. Both make one prediction's meaning depend "
    "on a timer and on presentation pacing. The CIPP contract is that one confirmation "
    "creates one counted credit which survives arbitrary silence and is consumed by the "
    "NEXT eligible evidence volley. Repair requires a dedicated prediction_credit port and "
    "local counted state, distinct from wta_reset (Phase 4)."))
@pytest.mark.parametrize("silence_ms", [20.0, 50.0, 130.0])
def test_c7_prediction_credit_survives_silence_and_suppresses_next_evidence(
        fresh_kernel, silence_ms):
    """Acceptance schedule from the brief::

        evidence -> confirmation -> long silence -> evidence -> evidence
        expected:     credit stored              suppressed   accepted

    The accepted/suppressed sequence must be IDENTICAL at every silence duration. A fixed
    refractory timer that merely covers an expected volley is not equivalent, which is
    exactly what varying `silence_ms` detects.
    """
    nest = fresh_kernel(0.1)
    node, recorder, _meter = _accumulator(nest)

    t_evidence_1 = 10.0
    t_confirm = t_evidence_1 + 5.0
    t_evidence_2 = t_confirm + silence_ms
    t_evidence_3 = t_evidence_2 + 20.0

    # Each evidence volley alone is exactly threshold, so each would fire unsuppressed.
    _drive(nest, node, [t_evidence_1, t_evidence_2, t_evidence_3], charge=E_THRESHOLD)
    # One confirmation, arriving during silence.
    _drive(nest, node, [t_confirm], charge=1.0, port="RESET")

    nest.Simulate(t_evidence_3 + 30.0)
    fired = sorted(float(t) for t in recorder.get("events")["times"])

    def near(t):
        return any(abs(f - (t + 1.0)) < 1.5 for f in fired)

    assert near(t_evidence_1), f"evidence 1 must fire; spikes={fired}"
    assert not near(t_evidence_2), (
        f"evidence 2 must be SUPPRESSED by the stored credit after {silence_ms} ms of "
        f"silence; spikes={fired}"
    )
    assert near(t_evidence_3), f"evidence 3 must be accepted again; spikes={fired}"


# =====================================================================================
# CONTRACT 8 -- cell physiology is independent of presentation interval
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P0: `topology.py` sets `window = self.ts.presentation` and passes it verbatim "
    "as `tau_basal`, `tau_apical`, `tau_deposit_lock` and the I relay's `t_lockout`. "
    "Changing how often the experiment presents input therefore changes dendritic "
    "coincidence physiology and relay behaviour. A TTL equal to D can also pair an apical "
    "from one presentation with a basal from the adjacent one at the inclusive boundary. "
    "Repair requires physical windows defined independently of stimulus pacing (Phase 2)."))
def test_c8_cell_physiology_is_independent_of_presentation_interval(fresh_kernel):
    """Coincidence and relay windows are properties of the CELL, not of the experiment.

    Two networks differing ONLY in `presentation` must expose identical neuron physiology.
    Measured today: every `tau_basal`/`tau_apical`/`tau_deposit_lock`/`t_lockout` tracks D.
    """
    def windows(presentation):
        net = _small_net(presentation=presentation)
        out = {}
        for nid, meta in net.node_meta.items():
            _model, params = net._model_for(meta)
            for key in ("tau_basal", "tau_apical", "tau_deposit_lock", "t_lockout"):
                if key in params:
                    out[(nid, key)] = params[key]
        return out

    fast, slow = windows(10.0), windows(40.0)
    assert fast and slow, "setup: the probe topology must expose windowed physiology"
    differing = {k: (fast[k], slow[k]) for k in fast if fast[k] != slow[k]}
    assert not differing, (
        f"{len(differing)} neuron parameters changed with the presentation interval, "
        f"e.g. {dict(list(differing.items())[:3])}"
    )


# =====================================================================================
# CONTRACT 9 -- delay quantization is a ceiling
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P1: `Timescales.quantize` documents 'round a delay UP onto the h grid' but "
    "uses Python `round()`. Measured at h=0.1: 0.14 -> 0.10, 0.15 -> 0.10, 0.24 -> 0.20, "
    "0.25 -> 0.20. Each SHORTENS a causal delay below what was requested, moving an event "
    "earlier in causal order. Repair requires ceil-to-grid with an explicit float "
    "tolerance and property tests either side of half-grid points."))
def test_c9_delay_quantization_never_shortens_a_requested_delay(fresh_kernel):
    """A shortened delay reorders causality silently, which this engine must never do."""
    from nest_backend.topology import Timescales

    ts = Timescales(h=0.1, base_ff=1.0, spread=2.0, presentation=20.0)
    offenders = [(r, ts.quantize(r)) for r in
                 (0.11, 0.14, 0.15, 0.16, 0.24, 0.25, 0.26, 1.01, 1.049, 1.05)
                 if ts.quantize(r) < r - 1e-12]
    assert not offenders, (
        "quantize shortened these delays: "
        + ", ".join(f"{r} -> {g}" for r, g in offenders)
    )


def test_c9_delay_quantization_never_returns_below_the_resolution(fresh_kernel):
    """The half of the contract that DOES hold today, pinned so the repair keeps it."""
    from nest_backend.topology import Timescales

    ts = Timescales(h=0.1, base_ff=1.0, spread=2.0, presentation=20.0)
    for requested in (0.0, 1e-9, 0.01, 0.05):
        assert ts.quantize(requested) >= 0.1 - 1e-12


# =====================================================================================
# SUPPLEMENTARY -- artifact weight export (audit P1, not one of the nine required probes)
# =====================================================================================
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "AUDIT P1: `dashboard_topology()` fills displayed weights from `self.weights`, the "
    "CONSTRUCTION-TIME dictionary, and never calls `plastic_weights()`. A post-training "
    "artifact therefore ships initial weights while claiming to carry trained ones -- "
    "observed as `L1c00_eor_c` = 250 in a replay header against 1000 in the adjacent "
    "report. Repair requires a materialized live-weight snapshot at artifact boundaries "
    "(Phase 7)."))
def test_supp_dashboard_topology_reports_live_weights(fresh_kernel):
    """What a learning run exports must be what the kernel currently holds."""
    import nest as _nest

    fresh_kernel(0.1)
    net = _small_net(learning=True)
    live = net.plastic_weights()
    assert live, "setup: a learning network must expose plastic weights"
    edge_id, original = next(iter(live.items()))
    src, tgt = net.conn_of[edge_id]
    _nest.GetConnections(source=_nest.NodeCollection([src]),
                         target=_nest.NodeCollection([tgt])).set({"w": original + 111.0})

    exported = {syn["id"]: syn.get("weight")
                for syn in net.dashboard_topology()["synapses"]}
    assert exported[edge_id] == pytest.approx(original + 111.0), (
        f"exported weight for {edge_id} is {exported[edge_id]}, "
        f"but the kernel holds {original + 111.0}"
    )


# =====================================================================================
# Profile guard
# =====================================================================================
def test_probes_measure_the_impulse_characterization_profile(fresh_kernel):
    """These probes characterise the CURRENT backend, and must keep doing so.

    When `cipp_continuous` exists it gets its own suite. If this file were repointed at
    the new profile, every xfail above would flip for the wrong reason and the audit trail
    would be lost.
    """
    from nest_backend.profiles import PROFILE_CIPP_CONTINUOUS

    net = _small_net()
    manifest = net.manifest()
    assert manifest["engine_profile"] == PROFILE_UNDER_TEST == PROFILE_IMPULSE, (
        f"probes expect {PROFILE_UNDER_TEST}, manifest says "
        f"{manifest.get('engine_profile')}"
    )
    # The DEFAULT must stay the prototype: `cipp_continuous` may not be claimed by a
    # network that did not explicitly ask for it, until the mechanical gate passes.
    assert manifest["engine_profile"] != PROFILE_CIPP_CONTINUOUS
    counts = manifest["counts"]
    assert counts["nodes"] == 51 and counts["edges"] == 254, (
        "the 3x6 probe topology changed shape; re-ground the probes before trusting them"
    )


def test_engine_profile_reaches_the_replay_artifact(fresh_kernel):
    """The profile must survive into the ARTIFACT, not stop at the manifest.

    A replay outlives the process that produced it. If the profile is only in the run
    manifest, an `impulse_characterization` artifact opened later is indistinguishable
    from a CIPP-engine one, which is precisely the confusion the two-profile split exists
    to prevent.
    """
    from nest_backend.replay_adapter import provenance
    from nest_backend.profiles import PROFILE_IMPULSE

    fresh_kernel(0.1)
    net = _small_net()

    class _Result:
        name = "profile_probe"
        manifest = net.manifest()
        weight_changes: list = []      # read by provenance() to label learning_mode

    prov = provenance(_Result())
    assert prov["engine_profile"] == PROFILE_IMPULSE, (
        f"replay provenance must carry the engine profile; got "
        f"{prov.get('engine_profile')!r}"
    )
