"""Generate, compile and install the committed NESTML event models as a NEST module.

Run as ``python -m nest_backend.build_models`` from the repository root, using the
interpreter of the dedicated NEST environment (``.nest-env/bin/python``).

Everything this writes -- generated C++, CMake caches, the built shared object -- is a
gitignored artifact under ``.nest-build/``. Only the ``.nestml`` sources in
``nest_backend/models/`` are committed.

Idempotence: the build is skipped when the module is already present and no model source
is newer than it. Pass ``--force`` to rebuild unconditionally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import sysconfig
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
BUILD_ROOT = REPO_ROOT / ".nest-build"
TARGET_DIR = BUILD_ROOT / "target"
INSTALL_DIR = BUILD_ROOT / "install"
STAMP_PATH = BUILD_ROOT / "build_stamp.json"

MODULE_NAME = "snnevents"

# The neuron models the prototype needs.
NEURON_MODELS = (
    "event_accumulator.nestml",
    "event_relay.nestml",
    "event_coincidence.nestml",
)

# Plastic synapse models, and the neuron each is CO-GENERATED with.
#
# NESTML generates a synapse together with one specific neuron, producing renamed pair
# models (`<neuron>__with_<synapse>` / `<synapse>__with_<neuron>`). That pairing is what
# gives the synapse access to the neuron's declared post state -- here `q_pre`, the
# pre-reset accumulated charge the rule reads as `I_accq`.
#
# `event_relay` is deliberately absent: `Eor` is frozen at theta by design
# (`eor_plasticity_enabled = False` in the reference), so its bank never learns and needs
# no plastic pairing.
SYNAPSE_PAIRS = (
    {
        "synapse": "plastic_feedforward_synapse.nestml",
        "neuron": "event_accumulator",
        "synapse_model": "plastic_feedforward_synapse",
        # `post_spikes` carries the firing event; the tuple maps the synapse's continuous
        # `I_accq` port onto the neuron's `q_pre` state variable.
        "post_ports": ["post_spikes", ["I_accq", "q_pre"]],
    },
    {
        "synapse": "plastic_basal_synapse.nestml",
        "neuron": "event_coincidence",
        "synapse_model": "plastic_basal_synapse",
        "post_ports": ["post_spikes", ["I_accq", "q_pre"]],
    },
)


def paired_names(pair: dict) -> tuple:
    """(neuron model name, synapse model name) as NESTML will register them."""
    return (f"{pair['neuron']}__with_{pair['synapse_model']}",
            f"{pair['synapse_model']}__with_{pair['neuron']}")


def ensure_toolchain_on_path() -> None:
    """Put the active environment's ``bin/`` first on PATH.

    NESTML shells out to ``cmake`` and ``make``; when the environment's interpreter is
    invoked by absolute path (which is how ``setup_nest_env.sh`` and the tests do it),
    that ``bin/`` is NOT on PATH and the build fails with a bare ``FileNotFoundError:
    cmake``. Fixing it here rather than in a wrapper script keeps ``python -m
    nest_backend.build_models`` working on its own.
    """
    env_bin = Path(sys.prefix) / "bin"
    if env_bin.is_dir():
        os.environ["PATH"] = f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def model_sources() -> list[Path]:
    names = list(NEURON_MODELS) + [p["synapse"] for p in SYNAPSE_PAIRS]
    missing = [name for name in names if not (MODELS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing committed NESTML sources: {missing}")
    return [MODELS_DIR / name for name in names]


def sources_fingerprint() -> str:
    """Hash of the model sources plus the toolchain identity.

    The NEST/NESTML versions are part of the fingerprint because a generated module is
    only valid for the NEST it was compiled against.
    """
    h = hashlib.sha256()
    for path in sorted(model_sources()):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    try:
        import nest  # noqa: PLC0415

        h.update(str(nest.__version__).encode())
    except Exception:  # pragma: no cover - only when NEST is absent
        h.update(b"nest-unavailable")
    try:
        import pynestml  # noqa: PLC0415

        h.update(str(getattr(pynestml, "__version__", "unknown")).encode())
    except Exception:  # pragma: no cover
        h.update(b"nestml-unavailable")
    h.update(sysconfig.get_platform().encode())
    return h.hexdigest()


def module_is_current(fingerprint: str) -> bool:
    if not STAMP_PATH.is_file():
        return False
    try:
        stamp = json.loads(STAMP_PATH.read_text())
    except (OSError, ValueError):
        return False
    if stamp.get("fingerprint") != fingerprint:
        return False
    so_path = INSTALL_DIR / f"{MODULE_NAME}module.so"
    return so_path.is_file()


def build(force: bool = False) -> dict:
    ensure_toolchain_on_path()
    fingerprint = sources_fingerprint()

    if not force and module_is_current(fingerprint):
        stamp = json.loads(STAMP_PATH.read_text())
        print(f"[build_models] up to date ({MODULE_NAME}); nothing to do")
        return stamp

    from pynestml.frontend.pynestml_frontend import generate_nest_target  # noqa: PLC0415

    # A stale target tree makes NESTML regenerate against old CMake state; start clean.
    shutil.rmtree(TARGET_DIR, ignore_errors=True)
    shutil.rmtree(INSTALL_DIR, ignore_errors=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[build_models] generating + compiling {len(NEURON_MODELS)} models -> {MODULE_NAME}")
    generate_nest_target(
        input_path=[str(p) for p in model_sources()],
        target_path=str(TARGET_DIR),
        install_path=str(INSTALL_DIR),
        module_name=f"{MODULE_NAME}module",
        logging_level="WARNING",
        codegen_opts={
            "neuron_synapse_pairs": [
                {"neuron": p["neuron"], "synapse": p["synapse_model"],
                 "post_ports": p["post_ports"]}
                for p in SYNAPSE_PAIRS
            ],
            # NOT `delay_variable`: declaring one that NESTML does not accept as a delay
            # makes pairing fall back to NEURON generation, silently. The connection's own
            # delay is used instead.
            "weight_variable": {p["synapse_model"]: "w" for p in SYNAPSE_PAIRS},
        },
    )
    elapsed = time.time() - started

    import nest  # noqa: PLC0415
    import pynestml  # noqa: PLC0415

    stamp = {
        "fingerprint": fingerprint,
        "module_name": f"{MODULE_NAME}module",
        "install_path": str(INSTALL_DIR),
        "models": list(NEURON_MODELS) + [p["synapse"] for p in SYNAPSE_PAIRS],
        "synapse_pairs": [paired_names(p) for p in SYNAPSE_PAIRS],
        "nest_version": str(nest.__version__),
        "nestml_version": str(getattr(pynestml, "__version__", "unknown")),
        "platform": sysconfig.get_platform(),
        "build_seconds": round(elapsed, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    STAMP_PATH.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    print(f"[build_models] built in {elapsed:.1f}s -> {INSTALL_DIR}")
    return stamp


def install_into_kernel() -> str:
    """Make the built module available to the current NEST kernel.

    Loads by ABSOLUTE PATH. `nest.Install("name")` resolves through the dynamic loader,
    which consults LD_LIBRARY_PATH as captured at process start -- so neither appending to
    `sys.path` nor mutating `os.environ` mid-process makes an out-of-tree module findable.
    Passing a path containing a separator makes `dlopen` treat it as a path directly,
    which keeps build artifacts out of the NEST prefix and out of git.

    Returns the module name. Safe to call repeatedly within one process: NEST raises on a
    double install, and that case is swallowed.
    """
    import nest  # noqa: PLC0415

    module = f"{MODULE_NAME}module"
    so_path = INSTALL_DIR / f"{module}.so"
    if not so_path.is_file():
        raise FileNotFoundError(
            f"{so_path} is missing -- run `python -m nest_backend.build_models` first"
        )
    try:
        nest.Install(str(so_path))
    except Exception as exc:  # nest raises a generic error on double-install
        if "already" not in str(exc).lower():
            raise
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args(argv)
    stamp = build(force=args.force)
    print(json.dumps(stamp, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
