"""Local Predictive Inhibition experiment (Experiment.md).

Driver, schedules, reference tape/replay, instrumentation, and metrics for the five
named modes. This package is the experiment CONTROLLER; it does not define a second
simulator or configuration system -- it constructs the one SimulationEngine with the
mode's feature flags and reads its state. Run artifacts land under experiments/runs/
(gitignored).
"""
