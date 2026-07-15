"""Modes, feature/context definitions, and constants (Experiment.md Sections 5, 10).

The five named modes are exactly the flag combinations in Section 5. The feature
flag combination -- never a frontend label -- determines behaviour: each mode is a
dict of SimulationEngine constructor overrides layered on the dashboard base.
"""

from backend.dashboard_config import DASHBOARD_OVERRIDES

# The dashboard preset is the shared base for every mode (same L2 network,
# feedforward learning, competition parameters). Modes only add the L1-feedback-loop
# flags, so the L2 side is identical across all five (Section 5).
BASE_OVERRIDES = dict(DASHBOARD_OVERRIDES)

# Section 5 mode table. paired == paired_local_enabled; predictor ==
# predictive_feedback_enabled; delivery == l2_to_l1i_delivery_enabled. `replay`
# marks the mode whose delivered feedback is the shuffled reference tape (the engine
# flags match local_plus_feedback; the driver injects _feedback_override per step).
MODES = {
    "baseline": dict(
        flags=dict(paired_local_enabled=False, predictive_feedback_enabled=False,
                   l2_to_l1i_delivery_enabled=False),
        replay=False,
        desc="no-L1-feedback-loop scientific control (L1I receives no events)"),
    "legacy_feedback": dict(
        flags=dict(paired_local_enabled=False, predictive_feedback_enabled=False,
                   l2_to_l1i_delivery_enabled=True),
        replay=False,
        desc="bit-exact current system: legacy L1I assembly-credit feedback"),
    "local_only": dict(
        flags=dict(paired_local_enabled=True, predictive_feedback_enabled=False,
                   l2_to_l1i_delivery_enabled=False),
        replay=False,
        desc="isolated-local negative control (paired local, no feedback)"),
    "local_plus_feedback": dict(
        flags=dict(paired_local_enabled=True, predictive_feedback_enabled=True,
                   l2_to_l1i_delivery_enabled=True),
        replay=False,
        desc="paired local + live L2E feedback + Section 6 predictor"),
    "time_shuffled_feedback": dict(
        flags=dict(paired_local_enabled=True, predictive_feedback_enabled=True,
                   l2_to_l1i_delivery_enabled=True),
        replay=True,
        desc="rate-matched contextual control: predictor over the shuffled tape"),
}

MODE_NAMES = list(MODES.keys())


def engine_overrides(mode):
    """Full SimulationEngine override dict for `mode` (base + mode flags)."""
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; known: {MODE_NAMES}")
    return {**BASE_OVERRIDES, **MODES[mode]["flags"]}


# ---- contextual task features (Section 10.2) --------------------------------
F_INDEX = 4     # center feature F
G_INDEX = 3     # middle-left feature G

# Matched-marginal contextual stimuli (row-major 3x3). P(F)=P(G)=0.5;
# P(F|X)=0.8, P(F|Y)=0.2; G has the opposite conditional relationship.
CONTEXT_VECTORS = {
    "X_with_F":    [1, 0, 1, 0, 1, 0, 0, 0, 0],   # indices 0,2,4
    "X_without_F": [1, 0, 1, 1, 0, 0, 0, 0, 0],   # indices 0,2,3
    "Y_with_F":    [0, 0, 0, 0, 1, 0, 1, 0, 1],   # indices 4,6,8
    "Y_without_F": [0, 0, 0, 1, 0, 0, 1, 0, 1],   # indices 3,6,8
}

# Which trial types carry feature F / feature G externally active.
TRIALS_WITH_F = ("X_with_F", "Y_with_F")
TRIALS_WITH_G = ("X_without_F", "Y_without_F")   # G active exactly when F absent here
X_TRIALS = ("X_with_F", "X_without_F")
Y_TRIALS = ("Y_with_F", "Y_without_F")

# Acquisition block composition (100 presentations): P(F|X)=0.8, P(F|Y)=0.2.
ACQ_BLOCK = {"X_with_F": 40, "X_without_F": 10, "Y_with_F": 10, "Y_without_F": 40}
# Reversal block composition (100 presentations): the conditionals flip.
REV_BLOCK = {"X_with_F": 10, "X_without_F": 40, "Y_with_F": 40, "Y_without_F": 10}

N_ACQ_BLOCKS = 5     # 500 acquisition presentations
N_REV_BLOCKS = 3     # 300 reversal presentations

# ---- protocol constants (Section 10) ----------------------------------------
DWELL = 20               # outer timesteps per presentation
INPUT_PERIOD = 1         # a pulse every step
BLANK_INTERVAL = 0       # no blank between presentations
SEEDS = list(range(10))  # seeds 0..9

# reporting cadence (presentations)
REPORT_FOURPATTERN = 20
REPORT_ACQ = 25
REPORT_REV = 10

# four-pattern task (Section 10.1)
FOURPATTERN_ORDER = ["row 1", "col 1", "diag \\", "diag /"]
FOURPATTERN_PER_PATTERN = 100    # 400 measured presentations total

# permutation / tape RNG offsets (Sections 9.2, 10.2)
PERM_SEED_OFFSET = 10_000
SCHED_SEED_OFFSET = 20_000
