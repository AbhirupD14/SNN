# SNN Experiment Suite (headless / cluster)

A separate, headless suite for **finite-budget, overnight/cluster** experiments on
the same `SimulationEngine` the dashboard uses. It has **no browser dependency**:
the runner writes durable artifacts, and a **read-only** service exposes them.

- **Interactive dashboard** (`backend/` + `frontend/`) = algorithm development,
  stepping, pausing, live parameter changes.
- **This suite** (`experiments/`) = unattended runs that produce durable files and
  a read-only view. The Python process is owned by SLURM/tmux/systemd/nohup, so it
  **outlives the agent and the browser**.

## Layout

```
experiments/
  runner.py          # headless runner (CLI entrypoint: python -m experiments.runner)
  server.py          # read-only FastAPI service (window into runs/)
  viewer/index.html  # read-only browser page (polls the service, no controls)
  config_example.json
  run.sbatch         # example SLURM script
  runs/              # artifacts (git-ignored), one dir per run + `current` symlink
```

Each run writes:

```
experiments/runs/<timestamp>/
  config.json  description.md  status.json
  events.jsonl  metrics.jsonl  summary.csv
  checkpoints/*.npz   plots/current_*.png  plots/summary_*.png
experiments/runs/current -> <timestamp>     # latest run
```

`status.json`, `events.jsonl`, `metrics.jsonl`, and `summary.csv` are written
incrementally so a supervisor can monitor **without screen-scraping**, and the
whole dashboard reconstructs from these files.

## Config

JSON or YAML. Keys: `name`, `description`, `hypothesis`, `patterns` (optional
subset), `seeds`, `dwell_steps`, `schedules` (`interleaved` | `blocked` | `mixed`),
`max_presentations_per_pattern` (the finite budget), `ablations`
(`{name: {engine_param: value, ...}}`), `base_params` (engine overrides applied to
every combo), `output_directory`. Any `SimulationEngine.__init__` parameter can be
set in `base_params` or an ablation. See `config_example.json`.

The runner sweeps every `schedule x dwell x ablation x seed` combination, each for
the fixed budget, and records **time-to-stable-owner**, ownership consistency,
collisions, dead L2E/L2I/L1I, inhibitory firing rates and charge/threshold ratios,
winner margin, weight saturation, and a `failure_reason` when a pattern does not
stabilize within budget.

## Run it

Foreground (local sanity check):

```bash
PYTHONPATH=. .venv/bin/python -m experiments.runner --config experiments/config_example.json
```

### nohup (survives ssh disconnect)

```bash
PYTHONPATH=. nohup .venv/bin/python -m experiments.runner \
    --config experiments/config_example.json > experiments/runs/runner.log 2>&1 &
disown
```

### tmux (detachable, inspectable)

```bash
tmux new -s snn
PYTHONPATH=. .venv/bin/python -m experiments.runner --config experiments/config_example.json
#   detach: Ctrl-b d      reattach: tmux attach -t snn
```

### SLURM

```bash
sbatch experiments/run.sbatch experiments/config_example.json
squeue -u "$USER"          # watch the job
```

### systemd --user (optional, for a long-lived node)

```bash
systemd-run --user --unit=snn-exp \
    .venv/bin/python -m experiments.runner --config experiments/config_example.json
journalctl --user -u snn-exp -f
```

## Read-only dashboard

Start the viewer service (independent of the runner; can be started/stopped/
restarted any time — it only reads files):

```bash
EXPERIMENTS_RUNS_DIR=experiments/runs \
    PYTHONPATH=. .venv/bin/uvicorn experiments.server:app --host 0.0.0.0 --port 8010
```

Open `http://<host>:8010`. It shows the hypothesis, live status, recent events,
the spikes-only raster / charge / weights / ownership plots for the combo running
now, and the completed-runs summary table. It has **no** pause/step/reset/param
controls by design.

## Monitoring without the browser (for a supervisor agent)

```bash
cat  experiments/runs/current/status.json          # live progress + current combo
tail -n 20 experiments/runs/current/events.jsonl    # milestones, failures, stalls
tail -n 5  experiments/runs/current/metrics.jsonl    # latest metric checkpoint
column -s, -t experiments/runs/current/summary.csv   # completed combos
```

`status.json.state` is one of `starting | running | completed | interrupted`.
On SIGTERM/SIGINT (e.g. SLURM preemption) the runner marks `interrupted` and flushes,
so partial artifacts and `summary.csv` are always usable for the morning review.
