"""Charge recording via a NEST multimeter, and proof that observing does not perturb.

The dashboard prompt allows adding "an explicit observation mechanism owned by NEST" when
historical state is needed, provided the observer changes nothing. A multimeter is exactly
that: `q` and `q_pre` are declared NESTML recordables on every committed model.
"""
from __future__ import annotations

import pytest

from nest_backend.engine import Stimulus, run_case
from nest_backend.recording import collect_charge
from nest_backend.replay_adapter import availability, build_records

CENTER = Stimulus({(1, 1): "row 1"})


def test_models_declare_charge_as_recordable(fresh_kernel):
    nest = fresh_kernel()
    for model in ("event_accumulator", "event_relay", "event_coincidence"):
        recordables = [str(r) for r in nest.GetDefaults(model)["recordables"]]
        assert "q" in recordables, f"{model} must expose accumulated charge"
        assert "q_pre" in recordables


def test_charge_is_actually_sampled():
    result = run_case("charge", CENTER, n_presentations=5, charge_interval_ms=1.0)
    assert result.charge, "a multimeter must produce samples"
    # 80 ordinary E + 10 Eor + 10 C + 10 I. RGC are spike_generators, not model neurons.
    assert len(result.charge) == 110
    trace = result.charge["L1c11E4"]
    assert len(trace) > 50
    assert max(v["q"] for v in trace.values()) > 0, "the driven column must accumulate"


def test_charge_trace_shows_the_pure_integrator_holding_flat():
    """No leak: between arrivals the charge must not decay at all."""
    result = run_case("charge", CENTER, n_presentations=5, charge_interval_ms=1.0)
    trace = sorted(result.charge["L1c11E4"].items())
    plateau = [q for _t, v in trace for q in [v["q"]] if 25.0 <= _t <= 40.0]
    assert plateau, "expected samples in the inter-volley interval"
    assert len(set(round(q, 6) for q in plateau)) == 1, (
        f"charge must hold perfectly flat with leak_rate=0, saw {sorted(set(plateau))}"
    )


def test_recorded_charge_reaches_the_replay_frames():
    result = run_case("charge", CENTER, n_presentations=5, charge_interval_ms=1.0)
    records = build_records(result)
    header = records[0]
    assert header["recording"]["charge_recorded"] is True
    assert header["recording"]["state_availability"]["potential"] == "recorded"
    assert header["recording"]["state_availability"]["activation"] == "derived"

    frames = [r for r in records if r["record"] == "frame"]
    with_charge = [f for f in frames
                   if any(n.get("potential") is not None for n in f["dynamic"]["neurons"])]
    assert with_charge, "sampled charge must appear on frames"


def test_without_a_multimeter_charge_stays_honestly_unavailable():
    result = run_case("nocharge", CENTER, n_presentations=5)
    assert result.charge == {}
    header = build_records(result)[0]
    assert header["recording"]["charge_recorded"] is False
    assert header["recording"]["state_availability"]["potential"] == "unavailable"


def test_availability_declaration_tracks_the_run():
    """The declaration follows the RECORDED VARIABLE SET, not a single yes/no flag.

    A run that sampled `q` but not `refr_until` must not claim refractory state, and a run
    that sampled both must not hide the one it has.
    """
    assert availability(())["potential"] == "unavailable"
    assert availability(())["refractory"] == "unavailable"

    charge_only = availability({"q", "q_pre"})
    assert charge_only["potential"] == "recorded"
    assert charge_only["activation"] == "derived"
    assert charge_only["v_pre_reset"] == "recorded"
    assert charge_only["refractory"] == "unavailable"
    assert charge_only["coincidence_charge"] == "unavailable"

    everything = availability({"q", "q_pre", "refr_until", "basal_charge", "lock_until"})
    assert everything["refractory"] == "derived"
    assert everything["coincidence_charge"] == "recorded"
    # No NESTML counterpart exists for these, so no amount of recording makes them appear.
    for field in ("freq", "g_inh", "trace"):
        assert everything[field] == "unavailable"


def test_every_declared_recordable_is_sampled_and_reaches_the_frame():
    """The multimeter asks for the model's full `recordables`, and none is dropped en route.

    Recording only `q`/`q_pre` used to report refractory state and the coincidence cell's
    basal charge as `unavailable` when NEST was willing to hand them over.
    """
    result = run_case("full", CENTER, n_presentations=3, charge_interval_ms=1.0)
    sampled = {var for samples in result.charge.values()
               for values in samples.values() for var in values}
    assert {"q", "q_pre", "refr_until", "basal_charge"} <= sampled
    assert not any(v.startswith("codegen_placeholder") for v in sampled), (
        "codegen placeholders are a NESTML artifact, not model state")

    header = build_records(result)[0]
    declared = header["recording"]["state_availability"]
    assert declared["refractory"] == "derived"
    assert declared["coincidence_charge"] == "recorded"

    frames = [r for r in build_records(result) if r["record"] == "frame"]
    states = [n["nest_state"] for f in frames for n in f["dynamic"]["neurons"]
              if n.get("nest_state")]
    assert states, "sampled state must reach the frames"
    assert any("basal_charge" in s for s in states)
    assert any("refr_until" in s for s in states)


def test_charge_is_never_interpolated_between_samples():
    """A tick with no sample reports no charge, rather than the neighbouring value."""
    result = run_case("charge", CENTER, n_presentations=5, charge_interval_ms=1.0)
    records = build_records(result)
    h = result.manifest["timescales"]["h_ms"]
    sampled_ticks = {round(float(t) / h) for s in result.charge.values() for t in s}
    for frame in records:
        if frame.get("record") != "frame":
            continue
        has_charge = any(n.get("potential") is not None
                         for n in frame["dynamic"]["neurons"])
        if has_charge:
            assert frame["timestep"] in sampled_ticks, (
                f"frame at tick {frame['timestep']} reports charge but was never sampled"
            )


@pytest.mark.parametrize("interval", [0.5, 1.0, 2.0])
def test_observing_charge_does_not_change_the_simulation(interval):
    """THE noninterference proof: a multimeter is passive.

    Identical seed and schedule, with and without the multimeter. Spikes, per-column
    counts, winner multiplicity and coincidence counters must all be untouched.
    """
    plain = run_case("plain", CENTER, n_presentations=5)
    observed = run_case("observed", CENTER, n_presentations=5,
                        charge_interval_ms=interval)

    assert observed.metrics["spikes"] == plain.metrics["spikes"]
    assert observed.metrics["by_column"] == plain.metrics["by_column"]
    assert observed.metrics["winner_multiplicity"] == plain.metrics["winner_multiplicity"]
    assert observed.metrics["coincidence"] == plain.metrics["coincidence"]
    assert observed.metrics["resets"] == plain.metrics["resets"]
    assert observed.metrics["eor_input_multiplicity"] == plain.metrics["eor_input_multiplicity"]


def test_a_charge_interval_off_the_resolution_grid_is_refused():
    from nest_backend.topology import NestTiledNetwork

    with pytest.raises(ValueError, match="not a multiple of h"):
        NestTiledNetwork(seed=1, charge_interval_ms=0.15)
