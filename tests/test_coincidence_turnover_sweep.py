import numpy as np
import pytest

from backend.simulation import SimulationEngine
from experiments.coincidence_turnover_sweep import (
    normalize_l2_initial_totals,
    reverse_l2_scheduler_order,
    run_condition,
)


def test_normalized_init_has_equal_totals_positive_fe_and_preserves_direction():
    e = SimulationEngine(seed=7, topology="rg_coincidence")
    before = [cell.acc_weights.copy() for cell in e.latency_competitors]

    result = normalize_l2_initial_totals(e, 0.8)

    assert result["free_energy"] == pytest.approx(0.2 * e.params["e_threshold"])
    for old, cell in zip(before, e.latency_competitors):
        assert cell.acc_weights.sum() == pytest.approx(0.8 * e.params["e_threshold"])
        assert cell.acc_weights / cell.acc_weights[0] == pytest.approx(old / old[0])
        # Ordinary-E init is cap-free: weights are finite and nonnegative, bounded by the
        # FE budget total (not a per-synapse ceiling).
        assert np.all(np.isfinite(cell.acc_weights)) and np.all(cell.acc_weights >= 0.0)


@pytest.mark.parametrize("rho", [-0.1, 0.0, 1.0, 1.1])
def test_normalized_init_rejects_nonpositive_free_energy(rho):
    e = SimulationEngine(seed=1, topology="rg_coincidence")
    with pytest.raises(ValueError):
        normalize_l2_initial_totals(e, rho)


def test_reverse_order_changes_only_l2_scheduler_positions():
    e = SimulationEngine(seed=1, topology="rg_coincidence")
    before = list(e.order)
    l2_before = [nid for nid in before if nid.startswith("L2E")]

    reverse_l2_scheduler_order(e)

    assert [nid for nid in e.order if nid.startswith("L2E")] == list(reversed(l2_before))
    assert [nid for nid in e.order if not nid.startswith("L2E")] == [
        nid for nid in before if not nid.startswith("L2E")
    ]


def test_short_protocol_reports_scientific_metrics():
    result = run_condition(1, 0.55, 0.01, dwell=20, final_window=10)

    assert result["initialization"]["free_energy"] > 0.0
    assert result["negative_initial_fe"] is False
    assert result["row_train"]["steps"] == 20
    assert result["column"]["pattern"] == "col 1"
    assert result["row_return"]["pattern"] == "row 1"
    assert set(result["column"]["first100_l1_counts"]) == {"1", "4", "7"}
    assert isinstance(result["full_success"], bool)
