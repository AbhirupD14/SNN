#!/bin/bash
#SBATCH --job-name=snn-consolidation
#SBATCH --output=experiments/runs/slurm-%j.out
#SBATCH --error=experiments/runs/slurm-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Example SLURM batch script for an overnight finite-budget sweep.
# Submit from the repo root:   sbatch experiments/run.sbatch [config.json]
# The Python process is owned by SLURM, so it outlives any agent/ssh session.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIG="${1:-experiments/config_example.json}"
PY="${PY:-.venv/bin/python}"

echo "[sbatch] repo   : $(pwd)"
echo "[sbatch] branch : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "[sbatch] commit : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[sbatch] config : $CONFIG"

PYTHONPATH=. "$PY" -m experiments.runner --config "$CONFIG"
