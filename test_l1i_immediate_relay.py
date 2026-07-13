"""
Focused test for the L1I immediate-relay feedback mode (l1i_immediate_relay).

In this mode L1I is a deterministic feedback relay, NOT a trainable
threshold-integrating neuron: any L1I that receives a nonzero L2E feedback signal
fires in that same outer timestep, regardless of its learned threshold or its
feedback-weight training. The contract pinned here:

  * Default OFF: a fresh engine uses the trainable threshold accumulator.
  * On the step an L2E winner fires, EVERY L1I fires -- even when L1I is sabotaged
    (feedback weights zeroed, threshold set impossibly high) so that the legacy
    integrator could never cross threshold. This proves the firing decision does
    not depend on membrane accumulation, learned threshold, or weight training.
  * With no L2E winner (no nonzero feedback) L1I does NOT fire.
  * Flag OFF restores the integrator: under the same sabotage L1I never fires on
    that step.

    PYTHONPATH=. .venv/bin/python test_l1i_immediate_relay.py
"""

from __future__ import annotations

import numpy as np

from backend.simulation import SimulationEngine, PATTERNS, N_OUT, N_PIX


def _sabotage_l1i(engine):
    """Make L1I un-fireable as an integrator: zero every feedback weight and set an
    impossibly high threshold. In relay mode this must NOT matter."""
    for inh in engine.l1.inhibitory_neurons:
        inh.weights = np.zeros(N_OUT)
        inh.threshold = 1e12


def _run_to_first_l2e_winner(engine, pattern='row 1', max_steps=300):
    """Step (holding one pattern) until any L2E fires; record, per step, whether
    any L2E fired and how many L1I fired. Returns (winner_step, per_step_l1i_counts,
    l2e_fired_flags)."""
    engine.set_pattern(pattern)
    l1i_counts, l2e_flags = [], []
    winner_step = None
    for s in range(max_steps):
        engine.step()
        l2e_fired = any(engine.spiked[f'L2E{j}'] for j in range(N_OUT))
        l1i_fired = sum(1 for i in range(N_PIX) if engine.spiked[f'L1I{i}'])
        l2e_flags.append(l2e_fired)
        l1i_counts.append(l1i_fired)
        if l2e_fired and winner_step is None:
            winner_step = s
            break
    return winner_step, l1i_counts, l2e_flags


def test_default_is_trainable_accumulator():
    e = SimulationEngine(seed=1)
    assert e.l1i_immediate_relay is False, e.l1i_immediate_relay
    print("PASS: l1i_immediate_relay defaults to False (trainable accumulator)")


def test_relay_fires_all_l1i_on_winner_despite_sabotage():
    """Relay mode: on the first L2E-winner step, every L1I fires even though the
    feedback weights are zero and the threshold is impossibly high."""
    e = SimulationEngine(seed=1, l1i_immediate_relay=True)
    _sabotage_l1i(e)
    winner_step, l1i_counts, _ = _run_to_first_l2e_winner(e)
    assert winner_step is not None, "no L2E winner appeared within the step budget"
    # Every L1I fired on the winner step (weight/threshold-independent relay).
    assert l1i_counts[winner_step] == N_PIX, \
        f"relay: expected all {N_PIX} L1I to fire on the winner step, got {l1i_counts[winner_step]}"
    # Before the winner there was no nonzero feedback, so no L1I fired.
    assert all(c == 0 for c in l1i_counts[:winner_step]), \
        f"relay: L1I fired before any L2E winner: {l1i_counts[:winner_step]}"
    print(f"PASS: relay fires all {N_PIX} L1I on the winner step (step {winner_step}) "
          f"with zeroed weights + huge threshold; silent beforehand")


def test_integrator_does_not_fire_under_same_sabotage():
    """Flag OFF: the legacy integrator cannot cross the impossibly-high threshold,
    so no L1I fires on the winner step -- the direct contrast with relay mode."""
    e = SimulationEngine(seed=1, l1i_immediate_relay=False)
    _sabotage_l1i(e)
    winner_step, l1i_counts, _ = _run_to_first_l2e_winner(e)
    assert winner_step is not None, "no L2E winner appeared within the step budget"
    assert l1i_counts[winner_step] == 0, \
        f"integrator: expected 0 L1I fires under sabotage, got {l1i_counts[winner_step]}"
    print(f"PASS: integrator mode fires 0 L1I on the winner step (step {winner_step}) "
          f"under the same sabotage (learned-threshold crossing required)")


def test_relay_requires_nonzero_feedback():
    """A single isolated step with no L2E winner produces no L1I firing in relay
    mode: the relay is gated on nonzero feedback, not free-running."""
    e = SimulationEngine(seed=1, l1i_immediate_relay=True)
    # Clear the input so nothing drives L2E to threshold; no winner -> no feedback.
    e.clear_input()
    fired_any = False
    for _ in range(4):
        e.step()
        if any(e.spiked[f'L2E{j}'] for j in range(N_OUT)):
            fired_any = True
            break
        assert sum(e.spiked[f'L1I{i}'] for i in range(N_PIX)) == 0, \
            "relay L1I fired with no L2E winner (no nonzero feedback)"
    assert not fired_any, "expected no L2E winner with empty input"
    print("PASS: relay L1I do not fire when there is no nonzero L2E feedback")


def main():
    test_default_is_trainable_accumulator()
    test_relay_fires_all_l1i_on_winner_despite_sabotage()
    test_integrator_does_not_fire_under_same_sabotage()
    test_relay_requires_nonzero_feedback()
    print("\nALL L1I IMMEDIATE-RELAY TESTS PASSED")


if __name__ == "__main__":
    main()
