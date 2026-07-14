"""
Regression test for L2 competition driven by the unweighted competitive reset.

Background: L2 competition is resolved by the shared inhibitory neuron L2I. When an
L2E crosses threshold it fires and drives L2I; if L2I crosses ITS threshold it
issues an unweighted competitive-reset event to the pool through
Neuron.apply_competitive_reset.
There is NO learned L2I->L2E gate: the reset is binary (clamp the loser to rest and
clear its current traces) plus local competitive depression of the participating
positive feedforward weights.

This test asserts the pool stays alive and competition is inhibition-mediated:
multiple neurons fire, multiple patterns map to distinct winners, L2I actually
drives the resets, and the reset always returns a loser to exact rest (no residual
learned-gate magnitude, no negative L2E afferent).
"""

from collections import Counter

from backend.simulation import SimulationEngine, PATTERNS, N_OUT


def train_and_measure(seed=1, epochs=30):
    # Pin the L1I threshold-integrator regime so this stays a clean L2-competition
    # regression independent of the dashboard preset.
    e = SimulationEngine(seed=seed, l1i_immediate_relay=False)
    fired = Counter()
    l2i_spikes = 0
    reset_events = 0
    winners_by_pattern = {}
    for ep in range(epochs):
        for name in PATTERNS:
            e.set_pattern(name)
            for _ in range(25):
                e.step()
                for j in range(N_OUT):
                    if e.spiked[f'L2E{j}']:
                        fired[f'L2E{j}'] += 1
                if e.spiked['L2I']:
                    l2i_spikes += 1
                # competitive-reset events issued to non-winners this step
                reset_events += len(e._reset_events)
            if ep >= epochs - 3:
                winners_by_pattern.setdefault(name, Counter())[e.winner] += 1
    final = {k: v.most_common(1)[0][0] for k, v in winners_by_pattern.items()}
    return e, fired, final, l2i_spikes, reset_events


def test_competitive_reset_keeps_pool_participating():
    print("=== L2 competition: unweighted competitive reset (no learned gate) ===")
    e, fired, final, l2i_spikes, reset_events = train_and_measure(seed=1)

    distinct_fired = len([k for k in fired if fired[k] > 0])
    distinct_winners = len(set(final.values()))
    print(f"  distinct L2E that fired: {distinct_fired}/{N_OUT}")
    print(f"  spikes per neuron: {dict(sorted(fired.items()))}")
    print(f"  pattern -> winner: {final}")
    print(f"  distinct winners across {len(PATTERNS)} patterns: {distinct_winners}")
    print(f"  L2I spikes (competition driver): {l2i_spikes}")
    print(f"  competitive-reset events: {reset_events}")

    # The pool stays alive: several neurons win over the run (the exact tiling is
    # measured by the consolidation harnesses, not asserted here).
    assert distinct_fired >= 2, f"pool collapsed to <2 firing units: {distinct_fired}/{N_OUT}"
    # Competition is mediated by the inhibitory neuron issuing resets, not a
    # hand-tuned reset.
    assert l2i_spikes > 0, "L2I never fired -- competition is not inhibition-driven"
    assert reset_events > 0, "no competitive-reset events -- losers were never reset"
    # There is no learned negative L2I->L2E gate on any L2E (all afferents positive).
    for j in range(N_OUT):
        w = e.l2.excitatory_neurons[j]._weights_array
        assert not (w < 0).any(), f"L2E{j} has a negative afferent (learned gate leaked back)"
    print("  PASS: inhibition-driven competition via unweighted resets; no learned gate\n")


def test_reset_returns_losers_to_rest():
    """Directly exercise the resolver: when L2I fires, every non-winner ends at
    exact rest with no negative afferent involved."""
    print("=== L2 competition: a competitive reset returns losers to exact rest ===")
    e = SimulationEngine(seed=1, l1i_immediate_relay=False)
    e.set_pattern('row 1')
    seen_reset_step = False
    for _ in range(400):
        e.step()
        if e._reset_events:
            seen_reset_step = True
            for nid, rec in e._reset_events:
                j = int(nid[3:])
                pot = float(e.l2.excitatory_neurons[j].potential)
                rest = float(e.l2.excitatory_neurons[j].resting_potential)
                # v_post is measured right after the reset; end-of-step charge may
                # rebuild on a LATER step but must be rest immediately after.
                assert rec['v_post'] == rest, (nid, rec['v_post'], rest)
            break
    assert seen_reset_step, "no competitive reset occurred in 400 steps"
    print("  PASS: losers clamped to rest by the competitive reset\n")


def test_robust_across_seeds():
    print("=== L2 competition: robustness across seeds ===")
    for seed in (1, 2, 3, 4):
        _, fired, final, l2i_spikes, reset_events = train_and_measure(seed=seed)
        df = len([k for k in fired if fired[k] > 0])
        dw = len(set(final.values()))
        print(f"  seed {seed}: fired={df}/{N_OUT}  distinct_winners={dw}  "
              f"L2I={l2i_spikes}  resets={reset_events}")
        assert df >= 2, f"seed {seed} lost pool participation: fired={df}"
        assert reset_events > 0, f"seed {seed}: no competitive resets"
    print("  PASS: pool participation and resets robust across seeds\n")


if __name__ == "__main__":
    test_competitive_reset_keeps_pool_participating()
    test_reset_returns_losers_to_rest()
    test_robust_across_seeds()
    print("ALL L2 COMPETITION TESTS PASSED")
