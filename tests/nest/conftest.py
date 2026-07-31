"""Collection guard for the NEST prototype tests.

The ordinary repository test suite runs on `.venv`, where NEST is deliberately absent, so
these tests SKIP with an explicit reason there. A dedicated NEST invocation must instead
treat absence as a FAILURE -- set `NEST_TESTS_REQUIRED=1` (which
`scripts/setup_nest_env.sh` and the documented test command both do) to turn the skip into
a hard error.
"""
from __future__ import annotations

import os

import pytest

_REQUIRED = os.environ.get("NEST_TESTS_REQUIRED") == "1"
_REASON = (
    "NEST is not available in this interpreter. These tests characterise the experimental "
    "NEST/NESTML backend and are expected to be skipped on the repository's ordinary "
    ".venv. Run them with: .nest-env/bin/python -m pytest tests/nest/ "
    "(after ./scripts/setup_nest_env.sh)"
)

def _nest_available() -> bool:
    """True only if the NEST SIMULATOR is importable -- not merely something named `nest`."""
    try:
        import nest  # noqa: PLC0415
    except Exception:
        return False
    return hasattr(nest, "Create") and hasattr(nest, "Simulate")


NEST_AVAILABLE = _nest_available()

if not NEST_AVAILABLE and _REQUIRED:
    raise RuntimeError("NEST_TESTS_REQUIRED=1 but the NEST simulator is not importable")


def pytest_ignore_collect(collection_path, config):
    """Exclude this directory from collection when NEST is absent.

    `pytest.skip(allow_module_level=True)` raised from a conftest is reported as a
    COLLECTION ERROR rather than a skip, and assigning `collect_ignore_glob` in a
    conditional branch proved unreliable here. This hook is the documented mechanism and
    is the one that actually produces a clean, silent skip on the repository's ordinary
    `.venv`, where NEST is deliberately absent.
    """
    return not NEST_AVAILABLE


@pytest.fixture(scope="session", autouse=True)
def _built_module():
    """Build the committed NESTML models once per session, before any test runs."""
    from nest_backend import build_models

    build_models.build()
    return build_models


@pytest.fixture
def fresh_kernel():
    """A reset NEST kernel with the prototype module installed.

    Returns a callable taking the resolution, because several contract tests need to set
    the resolution BEFORE creating nodes.
    """
    import nest

    from nest_backend import build_models

    def _make(resolution: float = 0.1, threads: int = 1):
        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": resolution,
            "local_num_threads": threads,
            "rng_seed": 1,
        })
        build_models.install_into_kernel()
        return nest

    return _make
