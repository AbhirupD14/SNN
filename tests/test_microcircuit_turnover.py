"""Regression coverage for the single 3x3 microcircuit turnover mechanism.

Demonstrates that the ``rg_coincidence`` circuit's feature-specific inhibition
(paired ``L1E_i -> L1C_i -> L1I_i -> L1E_i``) enables pattern turnover: switching to a
pattern that shares only the center feature transiently suppresses the center relay,
leaves the novel relays active, and hands ownership to a different competitor. This is the
capability the tiled whole-bank column reset removed.

Kept short (single seed) so it stays in the default suite; the full multi-seed evidence is
``experiments/microcircuit_turnover.py`` / ``coincidence_turnover_sweep.py``.
"""

from __future__ import annotations

from experiments.microcircuit_turnover import run_microcircuit, CENTER_PIXEL


def test_microcircuit_four_pattern_turnover():
    res = run_microcircuit(seed=1, dwell=4000, early=200, final_window=500)
    assert res["passed"], res["checks"]
    # four distinct consolidated owners
    assert len(set(res["owners"])) == 4
    assert all(p["final_dominance"] >= 0.8 for p in res["phases"])


def test_center_relay_suppressed_novel_active_on_switch():
    res = run_microcircuit(seed=1, dwell=4000, early=200, final_window=500)
    switches = res["phases"][1:]
    assert switches, "expected switch phases"
    for p in switches:
        # a different competitor won (turnover)
        assert p["turnover_from_prev"], p["pattern"]
        # the shared center feature relay is suppressed relative to the novel relays
        assert p["center_suppressed_vs_novel"], p["pattern"]
        assert p["early_center_relay_fires"] < p["early_novel_mean"], p["pattern"]
        # every novel feature relay stays active through the switch
        assert all(v > 0 for v in p["early_novel_relay_fires"].values()), p["pattern"]


def test_all_patterns_share_center_pixel():
    # the mechanism relies on the center being the shared/predicted feature
    from backend.simulation import PATTERNS
    for name, vec in PATTERNS.items():
        assert vec[CENTER_PIXEL] > 0.5, name
