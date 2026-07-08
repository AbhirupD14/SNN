# Claude Experiment Suite Prompt

You are working in the SNN repo.

## Goal

First preserve the current code state, then create a separate experimental suite folder for headless cluster experiments and a read-only dashboard view into those experiments.

## Purpose And Operating Model

This experimental suite is meant to support overnight and long-running finite-budget experiments on a cluster.

The local interactive dashboard remains the place for algorithm development, debugging, stepping, pausing, changing parameters live, and trying new ideas. The new `experiments/` suite is different: it should run unattended, produce durable artifacts, and expose a read-only browser view so results can be inspected later or during a run.

Gemma or another smaller cluster agent will use this suite as an operator/supervisor, not as the owner of the simulation process.

Gemma's expected workflow:

1. SSH into or operate on the cluster.
2. Verify the repo, branch, commit, environment, and config.
3. Start the experiment runner through SLURM, `tmux`, `systemd --user`, or `nohup`.
4. Start or verify the read-only dashboard service.
5. Periodically inspect `status.json`, `events.jsonl`, `metrics.jsonl`, `summary.csv`, logs, process status, and dashboard health.
6. Report progress, crashes, stalls, dead neurons, failed seeds, and finite-time consolidation results.
7. Preserve all artifacts so the morning review can happen even if Gemma disconnects or crashes.

Therefore, Claude should build the experiment suite so that:

- the runner can continue if Gemma disconnects
- the browser can disconnect without affecting the run
- every run has enough structured files for Gemma to monitor without screen scraping
- the dashboard can be read-only and reconstructed from saved artifacts
- failures are recorded, not hidden or overwritten
- fixed-budget results are first-class, because the goal is to understand consolidation timing, not wait indefinitely for eventual success

## Part 1: Preserve Current Code

1. Inspect the repo state:
   - `git status`
   - current branch
   - recent commits
   - any untracked files
2. Run the existing test/check commands that are appropriate for this repo.
3. Commit the current working state with a clear message, such as:
   - `Save current simplified SNN dashboard and learning state`
4. Push the branch to the remote.

Do not rewrite history. Do not discard uncommitted work. If there are unrelated files or unclear changes, report them before committing.

## Part 2: Add A Separate Experimental Suite

Create a new folder in this repo:

```text
experiments/
```

This should be separate from the interactive frontend/backend simulation code. The existing dashboard remains for local interactive algorithm development. The new experimental suite is for cluster/headless overnight runs.

## Design Requirements

### 1. Headless Runner

- Runs the same core `SimulationEngine`/model logic as the dashboard.
- Does not depend on a browser or WebSocket connection.
- Accepts experiment configs from JSON or YAML files.
- Runs fixed-budget experiments, not unlimited training.
- Supports seed sweeps, dwell sweeps, schedule sweeps, and ablations.
- Saves checkpoints and structured logs.

### 2. Read-Only Experiment Service

- Serves current experiment state to a browser.
- Browser is only a window into the experiment.
- No pause, step, reset, or live parameter mutation.
- It should show what is currently running and why.

### 3. Experiment Config

Experiment config should include:

- `name`
- `description`
- `hypothesis`
- `schedule`
- `seeds`
- `dwell_steps`
- `max_presentations_per_pattern`
- `ablations`
- `metrics`
- `output_directory`

Example:

```json
{
  "name": "finite_time_row_consolidation_sweep",
  "description": "Tests whether row patterns consolidate within a fixed presentation budget under low initialization ranges and no refractory period.",
  "hypothesis": "Lower initialization variance should reduce seed luck while inhibition stabilizes owners within 200 presentations per pattern.",
  "seeds": [1, 2, 3, 4, 5, 6, 7, 8],
  "dwell_steps": [4, 8, 16],
  "schedules": ["blocked", "mixed", "interleaved"],
  "max_presentations_per_pattern": 200
}
```

### 4. Output Artifacts

Write artifacts in a stable structure:

```text
experiments/runs/<timestamp>/
  config.json
  description.md
  status.json
  events.jsonl
  metrics.jsonl
  summary.csv
  checkpoints/
  plots/
```

Also maintain this if practical:

```text
experiments/runs/current -> latest active run
```

### 5. Metrics To Log

At minimum:

- seed
- current pattern
- current timestep
- current presentation count
- L2E owner per pattern
- ownership consistency over time
- time-to-stable-owner
- owner collisions
- dead/silent L2E, L2I, and L1I neurons
- L2I and L1I firing rates
- charge vs threshold for inhibitory neurons
- winner margin over time
- weight saturation percentage
- failure reason if consolidation does not happen within budget

### 6. Plots And Views

The read-only experiment dashboard should expose:

- run description/hypothesis
- current status
- logs/events
- raster plot, spikes only
- charge-over-time plots
- weight-over-time plots
- ownership map over time
- summary table of completed runs

### 7. Cluster Execution

Add a simple way to run this on a cluster:

- a CLI entrypoint
- an example config
- optionally a SLURM sbatch script
- documentation explaining `tmux`, `nohup`, and SLURM usage

## Important Architecture Rule

The experiment process must outlive the agent and browser. Claude/Gemma may start or monitor it, but the Python process should be owned by SLURM, `tmux`, `systemd`, or `nohup`.

If the browser disconnects or the agent dies, the experiment must keep running.

Keep the first implementation minimal. Do not refactor the entire repo. Reuse the existing simulation code where possible, but keep the experimental suite clearly separated from the interactive dashboard.
