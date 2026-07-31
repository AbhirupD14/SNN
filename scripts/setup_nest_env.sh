#!/usr/bin/env bash
#
# Non-interactive, idempotent setup for the isolated NEST/NESTML prototype environment.
#
# Contract (prompt section 8.3):
#   - never invokes sudo
#   - never modifies shell startup files
#   - creates ONLY the dedicated local environment at .nest-env/
#   - installs the pinned dependency set from environment-nest.yml
#   - builds the committed NESTML models
#   - runs scripts/verify_nest_install.py
#   - prints the exact command needed to use the environment
#
# It deliberately does NOT touch the repository's existing .venv or the system Python.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.nest-env"
TOOLS_DIR="${REPO_ROOT}/.nest-tools"
MICROMAMBA="${TOOLS_DIR}/bin/micromamba"
MANIFEST="${REPO_ROOT}/environment-nest.yml"

SKIP_BUILD=0
PROBE_PIP=0
for arg in "$@"; do
    case "$arg" in
        --skip-model-build) SKIP_BUILD=1 ;;
        --probe-pip-fallback) PROBE_PIP=1 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '\033[1;34m[setup-nest]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup-nest] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------------------
# 1. Locate or bootstrap an environment manager.
# --------------------------------------------------------------------------------------
find_env_manager() {
    for candidate in micromamba mamba conda; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$(command -v "$candidate")"
            return 0
        fi
    done
    if [[ -x "$MICROMAMBA" ]]; then
        echo "$MICROMAMBA"
        return 0
    fi
    return 1
}

bootstrap_micromamba() {
    log "no conda/mamba/micromamba found; bootstrapping a local micromamba (no sudo, no PATH changes)"
    local uname_s uname_m plat
    uname_s="$(uname -s)"; uname_m="$(uname -m)"
    case "${uname_s}/${uname_m}" in
        Linux/x86_64)  plat=linux-64 ;;
        Linux/aarch64) plat=linux-aarch64 ;;
        Darwin/x86_64) plat=osx-64 ;;
        Darwin/arm64)  plat=osx-arm64 ;;
        *) die "unsupported platform ${uname_s}/${uname_m}. Install NEST manually: https://nest-simulator.readthedocs.io/en/stable/installation/index.html" ;;
    esac
    command -v curl >/dev/null 2>&1 || die "curl is required to bootstrap micromamba. Install micromamba manually: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    mkdir -p "$TOOLS_DIR"
    ( cd "$TOOLS_DIR" && curl -sSL --max-time 300 "https://micro.mamba.pm/api/micromamba/${plat}/latest" | tar -xj bin/micromamba ) \
        || die "micromamba download failed. Manual alternative: https://nest-simulator.readthedocs.io/en/stable/installation/index.html"
    [[ -x "$MICROMAMBA" ]] || die "micromamba bootstrap produced no executable at ${MICROMAMBA}"
}

if ! ENV_MANAGER="$(find_env_manager)"; then
    bootstrap_micromamba
    ENV_MANAGER="$MICROMAMBA"
fi
log "environment manager: ${ENV_MANAGER}"

# --------------------------------------------------------------------------------------
# 2. Create / update the dedicated local prefix. Idempotent: re-running converges.
# --------------------------------------------------------------------------------------
export MAMBA_ROOT_PREFIX="${TOOLS_DIR}/mmroot"
mkdir -p "$MAMBA_ROOT_PREFIX"

[[ -f "$MANIFEST" ]] || die "missing manifest ${MANIFEST}"

case "$(basename "$ENV_MANAGER")" in
    micromamba) CREATE_CMD=("$ENV_MANAGER" create --yes --prefix "$ENV_PREFIX" --file "$MANIFEST") ;;
    mamba|conda) CREATE_CMD=("$ENV_MANAGER" env create --yes --prefix "$ENV_PREFIX" --file "$MANIFEST") ;;
    *) die "unrecognised environment manager $(basename "$ENV_MANAGER")" ;;
esac

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    log "environment already exists at ${ENV_PREFIX}; converging to the manifest"
    case "$(basename "$ENV_MANAGER")" in
        micromamba) "$ENV_MANAGER" install --yes --prefix "$ENV_PREFIX" --file "$MANIFEST" ;;
        mamba|conda) "$ENV_MANAGER" env update --prefix "$ENV_PREFIX" --file "$MANIFEST" ;;
    esac
else
    log "creating ${ENV_PREFIX} from ${MANIFEST} (this compiles nothing but downloads a lot)"
    "${CREATE_CMD[@]}"
fi

ENV_PY="${ENV_PREFIX}/bin/python"
[[ -x "$ENV_PY" ]] || die "environment creation did not produce ${ENV_PY}"

# --------------------------------------------------------------------------------------
# 3. Optional: reproduce the pip-only failure as a documented negative result.
# --------------------------------------------------------------------------------------
if [[ "$PROBE_PIP" == "1" ]]; then
    log "probing the pip-only fallback (expected to fail; recording the exact error)"
    PIP_PROBE="${TOOLS_DIR}/pip-probe"
    rm -rf "$PIP_PROBE"
    python3 -m venv "$PIP_PROBE"
    set +e
    "${PIP_PROBE}/bin/pip" install -r "${REPO_ROOT}/requirements-nest.txt" 2>&1 | tail -30
    "${PIP_PROBE}/bin/python" -c "import nest; print(nest.__version__)" 2>&1 | tail -10
    "${PIP_PROBE}/bin/python" -c "
import subprocess, shutil
print('nest-config on PATH:', shutil.which('nest-config'))
" 2>&1 | tail -5
    set -e
    log "pip probe finished; record the output above in docs/NEST_EVENT_DRIVEN_3X3_REPORT.md"
fi

# --------------------------------------------------------------------------------------
# 4. Build the committed NESTML models.
# --------------------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "0" ]]; then
    log "building committed NESTML models"
    ( cd "$REPO_ROOT" && "$ENV_PY" -m nest_backend.build_models ) \
        || die "NESTML model build failed. Re-run with --skip-model-build to inspect the environment, then see nest_backend/README.md"
else
    log "skipping NESTML model build (--skip-model-build)"
fi

# --------------------------------------------------------------------------------------
# 5. Verify.
# --------------------------------------------------------------------------------------
log "running scripts/verify_nest_install.py"
( cd "$REPO_ROOT" && "$ENV_PY" scripts/verify_nest_install.py ) \
    || die "verification failed -- see the output above"

cat <<EOF

$(printf '\033[1;32m[setup-nest] OK\033[0m')

Use the environment by invoking its interpreter directly (no activation, no shell edits):

    ${ENV_PREFIX}/bin/python -m pytest tests/nest/ -v

    ${ENV_PREFIX}/bin/python experiments/nest_3x3_suite.py

For an interactive shell with it on PATH:

    export PATH="${ENV_PREFIX}/bin:\$PATH"

EOF
