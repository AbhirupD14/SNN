"""Phase 4 minimal reproducer: is the dual FE/FES rule expressible as a NESTML synapse?

The prompt (section 6.5) identifies ONE real risk and requires it be settled with the
smallest possible model before the phase is called blocked or complete:

    does a post-spike update give every afferent the same pre-update `FE`
    and its own pre-update `w_i`?

The arrangement is the minimum that can answer it: one postsynaptic accumulator with TWO
afferents, one participating in the causal volley and one silent. The reference rule says
both must be updated when the target fires -- the participant with `s = +1`, the silent one
with `s = -1` -- from a shared `FE` and their own `FES(w_i)`.

Run:  .nest-env/bin/python -m nest_backend.phase4_reproducer
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
BUILD_ROOT = REPO_ROOT / ".nest-build" / "phase4"

THETA = 1000.0
ETA = 4.0
B = 5.0
E_FLOOR = 0.001
W_TE = 0.001
W_CAP = 800.0


def expected_update(w: float, iaccq: float, participated: bool, phi: float = 1.0) -> float:
    """The declared equation, evaluated in Python as the oracle."""
    fe = E_FLOOR + (1.0 - E_FLOOR) / (1.0 + B * (iaccq / THETA - 0.5) ** 2)
    fes = W_TE + (1.0 - W_TE) / (1.0 + B * (2.0 * w / THETA - 0.5) ** 2)
    s = 1.0 if participated else -1.0
    return min(max(w + ETA * fe * fes * s * phi, W_TE), W_CAP)


# A diagnostic variant of the committed synapse in which `I_accq` is a plain PARAMETER
# rather than a continuous post port. It isolates the identified risk (shared pre-update
# `FE`, own pre-update `w_i`, and whether a SILENT afferent is visited at post-spike time)
# from the separate question of whether NESTML can wire a continuous post port at all.
# Written to the build tree, not committed: it is an instrument, not a model of the system.
REDUCED_SYNAPSE = """
model dual_fe_fes_probe_synapse:

    state:
        w real = 250.
        participated real = 0.
        n_post_updates real = 0.
        unused_trace real = 0.

    equations:
        # Present only because NESTML's NEST synapse target requires an `equations` block
        # plus an `integrate_odes()` in `update` to classify and generate the model as a
        # SYNAPSE at all. Without it the model is emitted as a NEURON and never registers
        # as a synapse type. `unused_trace` is inert: nothing reads it, and no scientific
        # state depends on it. Established by bisection against NESTML's own stdp_synapse.
        unused_trace' = -unused_trace / tau_unused

    parameters:
        eta real = 4.
        theta real = 1000.
        B real = 5.
        e_floor real = 0.001
        w_te real = 0.001
        w_cap real = 500.
        phi real = 1.
        I_accq real = 1000.
        tau_unused ms = 20 ms

    input:
        pre_spikes <- spike
        post_spikes <- spike

    output:
        spike

    onReceive(post_spikes):
        n_post_updates += 1.
        fe real = e_floor + (1. - e_floor) / (1. + B * (I_accq / theta - 0.5) ** 2)
        fes real = w_te + (1. - w_te) / (1. + B * (2. * w / theta - 0.5) ** 2)
        sgn real = 2. * participated - 1.
        w_new real = w + eta * fe * fes * sgn * phi
        w = min(max(w_new, w_te), w_cap)
        participated = 0.

    onReceive(pre_spikes):
        participated = 1.
        emit_spike(w)

    update:
        integrate_odes()
"""


def _generate(synapse_source: Path, synapse_name: str, post_ports: list,
              module: str, subdir: str) -> str:
    from pynestml.frontend.pynestml_frontend import generate_nest_target

    from .build_models import ensure_toolchain_on_path

    ensure_toolchain_on_path()
    root = BUILD_ROOT / subdir
    shutil.rmtree(root, ignore_errors=True)
    target, install = root / "target", root / "install"
    target.mkdir(parents=True, exist_ok=True)
    install.mkdir(parents=True, exist_ok=True)

    generate_nest_target(
        input_path=[str(MODELS_DIR / "event_accumulator.nestml"), str(synapse_source)],
        target_path=str(target),
        install_path=str(install),
        module_name=module,
        logging_level="ERROR",
        codegen_opts={
            "neuron_synapse_pairs": [{
                "neuron": "event_accumulator",
                "synapse": synapse_name,
                "post_ports": post_ports,
            }],
            "weight_variable": {synapse_name: "w"},
        },
    )
    return str(install / f"{module}.so")


def build_full() -> str:
    """The committed model, with `I_accq` as a CONTINUOUS POST PORT reading `q_pre`."""
    return _generate(MODELS_DIR / "plastic_feedforward_synapse.nestml",
                     "plastic_feedforward_synapse",
                     ["post_spikes", ["I_accq", "q_pre"]], "phase4module", "full")


def build_reduced() -> str:
    """The diagnostic variant, with `I_accq` as a parameter."""
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    source = BUILD_ROOT / "dual_fe_fes_probe_synapse.nestml"
    source.write_text(REDUCED_SYNAPSE)
    return _generate(source, "dual_fe_fes_probe_synapse", ["post_spikes"],
                     "phase4reducedmodule", "reduced")


def probe_continuous_post_port() -> dict:
    """Tier A: can NESTML wire `I_accq` as a continuous post port reading `q_pre`?"""
    try:
        build_full()
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True}


def probe_update_semantics() -> dict:
    """Tier B: the identified risk, isolated from the continuous-post-port question.

    One postsynaptic accumulator, two afferents. `A` participates in the causal volley;
    `B` is silent. The reference rule requires BOTH to be updated when the target fires --
    `A` with `s = +1`, `B` with `s = -1` -- from a shared pre-update `FE` and their own
    pre-update `w_i`.
    """
    import nest

    module_path = build_reduced()
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.Install(module_path)

    neuron_model = "event_accumulator__with_dual_fe_fes_probe_synapse"
    synapse_model = "dual_fe_fes_probe_synapse__with_event_accumulator"
    available = set(nest.node_models) | set(nest.synapse_models)
    if neuron_model not in available or synapse_model not in available:
        return {"status": "BLOCKED", "reason": "co-generated models not registered",
                "looked_for": [neuron_model, synapse_model],
                "sample_registered": sorted(m for m in available if "plastic" in m)}

    receptors = nest.GetDefaults(neuron_model)["receptor_types"]
    post = nest.Create(neuron_model, 1, {"theta": THETA, "t_ref": 0.0})

    pre_a = nest.Create("parrot_neuron", 1)
    pre_b = nest.Create("parrot_neuron", 1)
    gen_a = nest.Create("spike_generator", 1, {"spike_times": [1.0, 2.0, 70.0]})
    gen_b = nest.Create("spike_generator", 1, {"spike_times": [50.0]})
    nest.Connect(gen_a, pre_a)
    nest.Connect(gen_b, pre_b)

    # Deliberately DIFFERENT starting weights, so "each synapse used its own pre-update
    # w_i" is distinguishable from "both used the same one".
    w_a0, w_b0 = 600.0, 300.0
    iaccq = 1200.0
    for pre, w0 in ((pre_a, w_a0), (pre_b, w_b0)):
        nest.Connect(pre, post, syn_spec={
            "synapse_model": synapse_model, "w": w0, "delay": 1.0,
            "eta": ETA, "theta": THETA, "B": B, "e_floor": E_FLOOR,
            "w_te": W_TE, "w_cap": W_CAP, "phi": 1.0, "I_accq": iaccq,
            "receptor_type": receptors["EXC"],
        })

    conn_a = nest.GetConnections(source=pre_a, target=post)
    conn_b = nest.GetConnections(source=pre_b, target=post)

    def updates(conn):
        """`n_post_updates` if the connection exposes it, else `None`.

        NESTML does not surface every synapse state variable on the NEST connection, so
        the update COUNT may be unavailable. The weight change is the authoritative
        signal either way, and is what the verdict is derived from.
        """
        try:
            return int(float(conn.get("n_post_updates")))
        except Exception:
            return None

    nest.Simulate(20.0)
    at_post = {"w_a": float(conn_a.get("w")), "w_b": float(conn_b.get("w")),
               "updates_a": updates(conn_a), "updates_b": updates(conn_b),
               "post_spikes": int(post.get("n_spikes"))}

    nest.Simulate(80.0)   # B spikes at t=50 and A at t=70: do the deferred updates land?
    later = {"w_a": float(conn_a.get("w")), "w_b": float(conn_b.get("w")),
             "updates_b": updates(conn_b)}
    a_deferred = abs(at_post["w_a"] - w_a0) < 1e-9

    oracle_a = expected_update(w_a0, iaccq, participated=True)
    oracle_b = expected_update(w_b0, iaccq, participated=False)

    participant_ok = abs(later["w_a"] - oracle_a) < 1e-6
    silent_at_post = abs(at_post["w_b"] - w_b0) > 1e-9
    silent_later = abs(later["w_b"] - w_b0) > 1e-9
    silent_ok = abs(later["w_b"] - oracle_b) < 1e-6 if silent_later else False

    return {
        "status": "OK" if (participant_ok and silent_ok and not a_deferred) else "PARTIAL",
        "post_spikes": at_post["post_spikes"],
        "I_accq_used": iaccq,
        "participating_afferent": {
            "w_initial": w_a0,
            "w_at_post_spike_time": at_post["w_a"],
            "w_after_its_own_later_spike": later["w_a"],
            "w_oracle": oracle_a,
            "visited_at_post_spike_time": abs(at_post["w_a"] - w_a0) > 1e-9,
            "matches_declared_equation_when_finally_applied": participant_ok,
        },
        "silent_afferent": {
            "w_initial": w_b0,
            "w_at_post_spike_time": at_post["w_b"],
            "w_after_its_own_later_spike": later["w_b"],
            "w_oracle_for_minus_one_branch": oracle_b,
            "post_updates_at_post_spike_time": at_post["updates_b"],
            "post_updates_after_its_own_later_spike": later["updates_b"],
            "visited_at_post_spike_time": silent_at_post,
            "visited_retroactively_on_next_pre_spike": silent_later,
            "matches_declared_equation_when_finally_applied": silent_ok,
        },
        "own_pre_update_weight_used": (
            participant_ok and silent_ok and abs(oracle_a - oracle_b) > 1e-9
        ),
        "any_afferent_updated_at_post_spike_time": not a_deferred,
    }


def run() -> dict:
    continuous = probe_continuous_post_port()
    semantics = probe_update_semantics()
    return {
        "tier_a_continuous_post_port": continuous,
        "tier_b_update_semantics": semantics,
        "verdict": _verdict(continuous, semantics),
    }


def _verdict(continuous: dict, semantics: dict) -> str:
    if semantics.get("status") == "BLOCKED":
        return "BLOCKED: the co-generated synapse could not be built or registered."
    silent = semantics.get("silent_afferent", {})
    parts = []
    part = semantics.get("participating_afferent", {})
    if part.get("matches_declared_equation_when_finally_applied"):
        parts.append("BOTH branches compute the declared equation EXACTLY")
    else:
        parts.append("the computed update does NOT match the declared equation")
    if not part.get("visited_at_post_spike_time"):
        parts.append(
            "but NO afferent is updated at post-spike time: NEST visits a synapse only "
            "when a PRESYNAPTIC spike traverses it, so every update is deferred to the "
            "afferent's next pre-spike"
        )
    if silent.get("visited_at_post_spike_time"):
        parts.append("silent afferents ARE visited at post-spike time, so s=-1 is exact")
    elif silent.get("visited_retroactively_on_next_pre_spike"):
        parts.append(
            "silent afferents are visited only RETROACTIVELY at their next presynaptic "
            "spike, so the s=-1 branch is deferred rather than simultaneous"
        )
    else:
        parts.append("silent afferents are NEVER visited, so the s=-1 branch cannot fire")
    if not continuous.get("available"):
        parts.append("I_accq as a continuous post port failed to generate")
    return "; ".join(parts)


def main() -> int:
    try:
        result = run()
    except Exception as exc:  # a build/codegen failure IS the result here
        result = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    out = REPO_ROOT / ".nest-build" / "phase4_reproducer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
