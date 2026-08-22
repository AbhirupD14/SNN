"""Headless, finite-budget SNN experiment suite (cluster/overnight runs).

Separate from the interactive dashboard (backend/ + frontend/). The runner
(experiments.runner) produces durable artifacts under experiments/runs/<ts>/;
the read-only service (experiments.server) is a browser window into them and
never controls the simulation.
"""
