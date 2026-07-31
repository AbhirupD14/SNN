#!/usr/bin/env python
"""Verify the isolated NEST/NESTML environment end to end.

Fails (non-zero exit) unless it can, in order (prompt section 8.4):

  1. import `nest`
  2. print NEST build/version information
  3. import the NESTML toolchain
  4. generate and compile the smallest committed event model
  5. load the generated module with `nest.Install`
  6. deliver at least one spike through it
  7. record the resulting output spike
  8. report thread and MPI capabilities WITHOUT assuming either exists

Step 6 is the one that matters most: it proves NESTML can `emit_spike()` directly from an
`onReceive` handler, with an empty `update` block and no ODE. Every downstream phase
assumes that, so it is checked here rather than discovered later.

Run:  .nest-env/bin/python scripts/verify_nest_install.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []
REPORT: dict = {}


def check(label: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
                return None
            print(f"  ok    {label}")
            return result

        return wrapper

    return decorator


@check("import nest")
def step_import_nest():
    import nest

    return nest


@check("NEST build / version information")
def step_nest_version(nest):
    info = {
        "version": str(nest.__version__),
        "prefix": sys.prefix,
    }
    # `nest.ll_api` / kernel attributes vary between releases; read defensively.
    try:
        nest.ResetKernel()
        info["resolution_default_ms"] = float(nest.resolution)
        info["min_delay_ms"] = float(nest.min_delay)
        info["max_delay_ms"] = float(nest.max_delay)
    except Exception as exc:
        info["kernel_probe_error"] = str(exc)
    REPORT["nest"] = info
    for key, value in info.items():
        print(f"        {key}: {value}")
    return info


@check("import the NESTML toolchain")
def step_import_nestml():
    import pynestml
    from pynestml.frontend.pynestml_frontend import generate_nest_target  # noqa: F401

    version = str(getattr(pynestml, "__version__", "unknown"))
    REPORT["nestml"] = {"version": version}
    print(f"        version: {version}")
    return pynestml


@check("generate + compile the committed event models")
def step_build():
    from nest_backend import build_models

    stamp = build_models.build()
    REPORT["build"] = stamp
    print(f"        module: {stamp['module_name']}  models: {len(stamp['models'])}")
    return stamp


@check("load the generated module with nest.Install")
def step_install(nest):
    from nest_backend import build_models

    nest.ResetKernel()
    module = build_models.install_into_kernel()
    REPORT["module_loaded"] = module
    return module


@check("deliver a spike through the model and record the output")
def step_roundtrip(nest):
    """Two sub-threshold events then one that crosses -> exactly one spike.

    This simultaneously proves event-driven emission (the spike appears at the arrival
    timestamp of the third event, not at a step boundary chosen by an integrator) and
    that accumulation is a pure sum with no leak between the events.
    """
    nest.ResetKernel()
    from nest_backend import build_models

    build_models.install_into_kernel()

    resolution = 0.1
    nest.resolution = resolution
    theta = 1000.0

    cell = nest.Create("event_accumulator", 1, {"theta": theta, "t_ref": 0.0})
    gen = nest.Create("spike_generator", 1, {"spike_times": [1.0, 2.0, 3.0]})
    rec = nest.Create("spike_recorder")
    nest.Connect(gen, cell, syn_spec={"weight": theta / 2.5, "delay": resolution,
                                      "receptor_type": 1})
    nest.Connect(cell, rec)
    nest.Simulate(10.0)

    times = [float(x) for x in rec.get("events")["times"]]
    expected = 3.0 + resolution
    if len(times) != 1:
        raise AssertionError(f"expected exactly 1 output spike, got {times}")
    if abs(times[0] - expected) > 1e-9:
        raise AssertionError(f"expected emission at arrival {expected}, got {times[0]}")
    if int(cell.get("n_spikes")) != 1:
        raise AssertionError(f"n_spikes state is {cell.get('n_spikes')}, expected 1")

    REPORT["roundtrip"] = {
        "resolution_ms": resolution,
        "output_spike_times_ms": times,
        "emitted_from": "onReceive handler (no ODE, no onCondition)",
    }
    print(f"        output spike at {times[0]} ms (= arrival of the crossing event)")
    return times


@check("report thread and MPI capability (without assuming either)")
def step_capabilities(nest):
    caps: dict = {}
    nest.ResetKernel()
    try:
        nest.local_num_threads = 2
        caps["threads_settable"] = True
        caps["local_num_threads_probe"] = int(nest.local_num_threads)
    except Exception as exc:
        caps["threads_settable"] = False
        caps["threads_error"] = str(exc)
    nest.ResetKernel()
    for attr in ("num_processes", "total_num_virtual_procs"):
        try:
            caps[attr] = int(getattr(nest, attr))
        except Exception as exc:
            caps[f"{attr}_error"] = str(exc)
    try:
        import mpi4py  # noqa: PLC0415

        caps["mpi4py"] = str(getattr(mpi4py, "__version__", "present"))
    except Exception:
        caps["mpi4py"] = None

    # MPI support is DETECTED BY RUNNING, never inferred.
    #
    # An earlier version of this check inferred MPI from "mpi4py imports and mpirun
    # exists", and that inference was WRONG: the conda-forge nest-simulator 3.10 package
    # depends on openmpi and ships `mpirun`, but the NEST library itself is NOT compiled
    # with MPI support. The failure only appears when NEST is actually launched under a
    # multi-rank launcher, where it aborts with "NEST was not compiled with MPI support".
    #
    # The honest test is to launch two ranks and see what happens. Absence is reported as
    # absence -- this prototype never assumes a capability it has not exercised.
    caps["nest_has_mpi"] = False
    caps["nest_has_mpi_source"] = "not probed"
    mpirun = Path(sys.prefix) / "bin" / "mpirun"
    if mpirun.is_file():
        probe = subprocess.run(
            [str(mpirun), "--oversubscribe", "-np", "2", sys.executable, "-c",
             "import nest; print('RANKS', nest.num_processes)"],
            capture_output=True, text=True, timeout=300,
        )
        combined = (probe.stdout or "") + (probe.stderr or "")
        if "not compiled with MPI support" in combined:
            caps["nest_has_mpi"] = False
            caps["nest_has_mpi_source"] = "probed: NEST is not compiled with MPI support"
        elif probe.returncode == 0 and "RANKS 2" in combined:
            caps["nest_has_mpi"] = True
            caps["nest_has_mpi_source"] = "probed: two ranks started successfully"
        else:
            caps["nest_has_mpi_source"] = f"probed: inconclusive (rc={probe.returncode})"
    else:
        caps["nest_has_mpi_source"] = "mpirun not present in this environment"
    REPORT["capabilities"] = caps
    for key, value in caps.items():
        print(f"        {key}: {value}")
    return caps


def main() -> int:
    print("NEST / NESTML environment verification")
    print("=" * 70)

    nest = step_import_nest()
    if nest is None:
        print("\nFATAL: NEST is not importable in this interpreter.")
        print("Run ./scripts/setup_nest_env.sh and use .nest-env/bin/python")
        return 1

    step_nest_version(nest)
    step_import_nestml()
    step_build()
    step_install(nest)
    step_roundtrip(nest)
    step_capabilities(nest)

    print("=" * 70)
    out = REPO_ROOT / ".nest-build" / "verify_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(REPORT, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report written to {out}")

    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
