"""Environment, build and model-hygiene checks (prompt sections 8.4 and 11)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "nest_backend" / "models"

# The no-ODE mandate applies to the EVENT NODE models. `*_synapse.nestml` is carved out for
# one specific, documented reason: NESTML's NEST synapse target does not generate a usable
# synapse without an `equations` block plus `integrate_odes()`. The trace it integrates is
# inert -- no scientific state reads it -- and `test_synapse_ode_is_inert` below pins that.
NODE_MODELS = sorted(p for p in MODELS_DIR.glob("*.nestml") if "synapse" not in p.name)
SYNAPSE_MODELS = sorted(MODELS_DIR.glob("*_synapse.nestml"))


def test_nest_and_nestml_versions_are_the_pinned_pair():
    import nest
    import pynestml

    assert str(nest.__version__).startswith("3.10"), (
        f"environment-nest.yml pins nest-simulator 3.10, got {nest.__version__}"
    )
    assert str(pynestml.__version__).startswith("8.3"), (
        f"environment-nest.yml pins nestml 8.3.0, got {pynestml.__version__}"
    )


def test_manifests_pin_the_versions_that_are_actually_installed():
    """The manifest is only useful if it reproduces what was verified."""
    import nest
    import pynestml

    manifest = (REPO_ROOT / "environment-nest.yml").read_text()
    assert f"nest-simulator={str(nest.__version__)[:4]}" in manifest
    assert f"nestml=={pynestml.__version__}" in manifest


def test_generated_module_builds_and_loads(fresh_kernel):
    nest = fresh_kernel()
    for model in ("event_accumulator", "event_relay", "event_coincidence"):
        assert model in nest.Models(), f"{model} was not registered by the module"


def test_build_stamp_records_the_toolchain():
    stamp = json.loads((REPO_ROOT / ".nest-build" / "build_stamp.json").read_text())
    for key in ("nest_version", "nestml_version", "platform", "fingerprint", "models"):
        assert stamp.get(key), f"build stamp is missing {key}"
    # 3 event-node models + 2 plastic synapse models.
    assert len(stamp["models"]) == 5
    assert len(stamp["synapse_pairs"]) == 2


@pytest.mark.parametrize("model_file", NODE_MODELS)
def test_no_ode_or_integration_construct_in_committed_models(model_file):
    """The no-ODE mandate is checkable, so check it rather than trusting review.

    This is the mechanical half of prompt section 2. Forbidden constructs are ODE
    integration and the continuous/kernel machinery that implies a sampled trajectory.
    `onCondition` is also rejected: it is evaluated once per timestep, which would move the
    threshold test out of the event path and back onto a periodic scan.
    """
    source = model_file.read_text()
    stripped = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    forbidden = [
        r"\bintegrate_odes\b",
        r"\bequations\s*:",
        r"\bconvolve\s*\(",
        r"\bkernel\b\s+\w+\s*=",
        r"<-\s*continuous",
        r"\bonCondition\b",
        r"\bdelta\s*\(",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, stripped), (
            f"{model_file.name} contains forbidden construct matching {pattern!r} -- the "
            "prototype must perform no scientific integration"
        )


@pytest.mark.parametrize("model_file", NODE_MODELS)
def test_update_block_is_scientifically_empty(model_file):
    """An `update` block may exist only because the NEST target requires one."""
    source = model_file.read_text()
    if "update:" not in source:
        return
    body = source.split("update:", 1)[1]
    code = [line.strip() for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")]
    assert len(code) <= 1, (
        f"{model_file.name} has a non-trivial update block: {code}. Every scientific "
        "state transition must happen in an event handler."
    )
    if code:
        assert code[0].startswith("dummy"), f"unexpected update body in {model_file.name}: {code}"


def _code_only(model_file):
    """Model source with comments stripped, so prose about STDP is not mistaken for STDP."""
    return "\n".join(
        line for line in model_file.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize("model_file", SYNAPSE_MODELS)
def test_synapse_ode_is_inert_and_cannot_decay(model_file):
    """The synapse's mandatory ODE must carry no scientific meaning AND no time constant.

    It exists only because NESTML will not generate a synapse without an `equations` block.
    Its derivative must be exactly zero: a DECAYING inert variable would be cosmetically
    indistinguishable from an STDP eligibility trace, and an invitation to wire one in.
    """
    code = _code_only(model_file)
    assert "codegen_placeholder' = 0" in code.replace(" ", "").replace(
        "codegen_placeholder'=0", "codegen_placeholder' = 0"
    ) or "codegen_placeholder' = 0.0 / ms" in code, (
        "the placeholder ODE must have a literally zero derivative"
    )
    body = code.split("onReceive", 1)[1] if "onReceive" in code else ""
    assert "codegen_placeholder" not in body, (
        "no event handler may read the placeholder -- if one does, the synapse has acquired "
        "real ODE-driven behaviour and the no-integration claim no longer holds"
    )


@pytest.mark.parametrize("model_file", SYNAPSE_MODELS)
def test_learning_rule_is_not_stdp(model_file):
    """The rule must never acquire a dependence on the pre/post interval.

    Using NEST's post-spike archive as a SCHEDULING mechanism is fine -- it only defers when
    a handler runs. Adopting STDP's RULE is not. STDP is by definition
    `dw = f(t_post - t_pre)`, realised with decaying traces; dual FE/FES has no such term.

    The specific way it could creep in is `s_i`: answering "did I participate in the causal
    volley?" with a time window would be a rectangular Delta-t kernel, and once tuned,
    nearest-neighbour STDP in disguise. Participation must stay a Boolean flag set on
    pre-spike and cleared at post-spike.
    """
    code = _code_only(model_file)
    forbidden = {
        r"\bexp\s*\(": "an exponential kernel is the signature of an STDP trace",
        r"\btrace\b": "an eligibility trace is STDP machinery",
        r"\bpow\s*\(|\*\*\s*\(?\s*-": "a decaying power law is a graded Delta-t kernel",
    }
    for pattern, why in forbidden.items():
        assert not re.search(pattern, code), (
            f"{model_file.name} contains {pattern!r}: {why}. Grading the update by the "
            "pre/post interval is a change of scientific algorithm and must be declared as "
            "one -- not introduced through the synapse."
        )

    # Only the DETECTOR rule has a participation term. The C basal rule deliberately has
    # none: `Current_Implementation_Methodology_Equations.md` section 4.2 states there is
    # "no negative-participation term", because C fires only on a valid coincidence with
    # its own causal basal source, so A and s are both structurally 1. Requiring a window
    # there would be requiring a term the reference does not have.
    if "participated" in code or "participate_until" in code:
        # A flat participation WINDOW is legitimate and required: the reference scopes s_i
        # to exactly one boundary (backend/simulation.py::_participation). What must never
        # appear is a magnitude that varies with the interval.
        assert "participate_until" in code, (
            "participation must be a one-volley membership window, not 'since the last "
            "firing': an ordinary E cell needs several volleys to reach threshold, so an "
            "unscoped flag would award +1 to afferents the reference excludes as 'from the "
            "preceding boundary'"
        )
        assert re.search(r"sgn\s*real\s*=\s*-1\.0", code) and re.search(r"sgn\s*=\s*1\.0", code), (
            "the participation signal must be exactly +1 or -1. A graded or decaying value "
            "would make the magnitude interval-dependent, which IS STDP."
        )
        assert not re.search(r"sgn\s*=.*(participate_until|t\s*-\s*t)", code), (
            "the signal must not be computed FROM the interval -- only selected by it"
        )


def test_thread_and_mpi_capability_is_reported_not_assumed(fresh_kernel):
    nest = fresh_kernel()
    assert int(nest.num_processes) >= 1
    nest.ResetKernel()
    nest.local_num_threads = 2
    assert int(nest.local_num_threads) == 2, "threads are configurable in this build"
