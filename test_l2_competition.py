"""
Regression test for L2 competition driven by adaptive lateral inhibition.

Background: L2 used to resolve competition with a procedural hard reset -- when
one neuron fired, every other L2E was forced to resting potential the same step.
That destroyed the subthreshold charge losing neurons had accumulated across
volleys, so only one neuron ever fired, only it ever learned, and every pattern
collapsed onto that single winner.

The fix routes suppression through the real L2I->L2E pathway instead: when
several L2E cross threshold together, one fires and drives the shared inhibitory
neuron L2I, which laterally inhibits the OTHER threshold-crossers (near-winners)
through an adaptive gate (Neuron.apply_inhibition). Subthreshold accumulators are
left untouched, so every unit can eventually win, and the gate strengths are
learned rather than hand-tuned.

This test asserts the collapse is gone: multiple neurons fire, multiple patterns
map to distinct winners, L2I actually mediates the competition, and the gates
adapt away from their initial value.
"""

from collections import Counter

from backend.simulation import SimulationEngine, PATTERNS, N_OUT, L2_GATE_INIT


def train_and_measure(seed=1, epochs=30):
    # This test isolates the L2I-mediated lateral-inhibition competition (see the
    # module docstring); its distinct-winner bars were calibrated against the L1I
    # threshold-integrator regime. L1I is now an immediate relay by default, which
    # is an orthogonal change that shifts these emergent counts per-seed, so pin
    # the integrator regime here to keep this a clean L2-competition regression.
    # The relay default is covered by test_l1i_immediate_relay.py.
    e = SimulationEngine(seed=seed, l1i_immediate_relay=False)
    fired = Counter()
    l2i_spikes = 0
    gate_discharges = 0
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
                # inhibitory-gate discharge events routed to L2E this step
                gate_discharges += sum(1 for nid, _ in e._inh_events if nid.startswith('L2E'))
            if ep >= epochs - 3:
                winners_by_pattern.setdefault(name, Counter())[e.winner] += 1
    final = {k: v.most_common(1)[0][0] for k, v in winners_by_pattern.items()}
    gates = [-float(e.l2.excitatory_neurons[j]._weights_array[0]) for j in range(N_OUT)]
    return e, fired, final, l2i_spikes, gate_discharges, gates


def test_no_single_winner_collapse():
    print("=== L2 competition: adaptive lateral inhibition (no hard reset) ===")
    e, fired, final, l2i_spikes, gate_discharges, gates = train_and_measure(seed=1)

    distinct_fired = len([k for k in fired if fired[k] > 0])
    distinct_winners = len(set(final.values()))
    print(f"  distinct L2E that fired: {distinct_fired}/8")
    print(f"  spikes per neuron: {dict(sorted(fired.items()))}")
    print(f"  pattern -> winner: {final}")
    print(f"  distinct winners across 8 patterns: {distinct_winners}")
    print(f"  L2I spikes (competition driver): {l2i_spikes}")
    print(f"  L2 gate discharge events: {gate_discharges}")
    print(f"  learned gate magnitudes: {[round(g, 2) for g in gates]}")

    # 1. The collapse is gone: many neurons participate (was 1/8).
    assert distinct_fired >= 4, f"expected broad participation, got {distinct_fired}/8"
    # 2. Patterns no longer converge to a single neuron (was 1 distinct winner).
    assert distinct_winners >= 4, f"expected pattern differentiation, got {distinct_winners}"
    # 3. Competition is actually mediated by the inhibitory neuron, not a reset.
    assert l2i_spikes > 0, "L2I never fired -- competition is not inhibition-driven"
    assert gate_discharges > 0, "no L2 gate discharges -- inhibition not routed through the synapse"
    # 4. Gates are ADAPTIVE: at least some grew away from the initial magnitude.
    init_mag = abs(L2_GATE_INIT)
    assert any(g > init_mag + 1e-3 for g in gates), "inhibitory gates did not adapt"
    # 5. No gate exceeds its (sub-threshold) ceiling, so it can't act as a full reset.
    thr2 = e.params['threshold_l2']
    assert all(g < thr2 for g in gates), "a gate reached threshold strength (reset-equivalent)"
    print("  PASS: participation restored, patterns differentiate, competition is inhibition-driven\n")


def test_robust_across_seeds():
    print("=== L2 competition: robustness across seeds ===")
    for seed in (1, 2, 3, 4):
        _, fired, final, _, _, _ = train_and_measure(seed=seed)
        df = len([k for k in fired if fired[k] > 0])
        dw = len(set(final.values()))
        print(f"  seed {seed}: fired={df}/8  distinct_winners={dw}")
        assert df >= 3 and dw >= 3, f"seed {seed} collapsed: fired={df}, winners={dw}"
    print("  PASS: no seed collapses to a single winner\n")


if __name__ == "__main__":
    test_no_single_winner_collapse()
    test_robust_across_seeds()
    print("ALL L2 COMPETITION TESTS PASSED")
