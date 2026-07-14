"""Phase 1 contract: the paired L1E_i -> L1I_i topology (Local Predictive Inhibition).

OFF (default): every L1I keeps its legacy 8-element [L2E0..L2E7] afferent array; no
local edges exist; _all_weights is byte-identical in shape to legacy; determinism
holds. ON: each L1I has 9 afferents [local, fb0..fb7]; exactly nine local edges
exist, each L1E_i->L1I_i with NO cross-pairs; all eight feedback afferents are
retained at the correct offset; the fixed local weights never change under stepping.

The bit-exact OFF equivalence against the committed golden baseline is enforced by
tests/golden/test_golden_equiv.py; this test adds the structural / no-cross-pair /
fixed-local assertions and a same-seed determinism check.

    PYTHONPATH=. .venv/bin/python test_paired_local_topology.py
"""
import numpy as np

from backend.simulation import SimulationEngine, N_OUT, N_PIX
from backend.dashboard_config import DASHBOARD_OVERRIDES

INPUT = [0, 0, 0, 1, 1, 1, 0, 0, 0]   # "row 1"


def _run(engine, steps=30):
    engine.set_input(INPUT)
    for _ in range(steps):
        engine.step()
    return engine._all_weights()


def test_disabled_is_legacy():
    e = SimulationEngine(seed=3, **DASHBOARD_OVERRIDES)   # paired off by default
    assert e._l1i_paired is False and e._l1i_fb_offset == 0
    # 8-element legacy L1I arrays.
    assert all(len(n._weights_array) == N_OUT for n in e.l1.inhibitory_neurons)
    # No local edges / no local weight keys.
    assert not any(s['kind'] == 'local_evidence' for s in e.synapses)
    w = e._all_weights()
    assert not any(k.startswith('local') for k in w)
    for i in range(N_PIX):
        for j in range(N_OUT):
            assert f'fb{j}->{i}' in w
    # Determinism: same seed/config/schedule -> identical final weights.
    w1 = _run(SimulationEngine(seed=3, **DASHBOARD_OVERRIDES))
    w2 = _run(SimulationEngine(seed=3, **DASHBOARD_OVERRIDES))
    assert w1.keys() == w2.keys()
    assert all(w1[k] == w2[k] for k in w1), "disabled path is nondeterministic"
    print("  disabled: 8-afferent legacy L1I, no local edges, deterministic: OK")


def test_enabled_topology():
    e = SimulationEngine(seed=3, **{**DASHBOARD_OVERRIDES,
                                    'paired_local_enabled': True,
                                    'predictive_feedback_enabled': True})
    assert e._l1i_paired is True and e._l1i_fb_offset == 1
    # 9-element afferent arrays: [local, fb0..fb7].
    assert all(len(n._weights_array) == N_OUT + 1 for n in e.l1.inhibitory_neurons)
    # Exactly nine local edges, each L1E_i->L1I_i, NO cross-pairs.
    locals_ = [s for s in e.synapses if s['kind'] == 'local_evidence']
    assert len(locals_) == N_PIX, f"expected {N_PIX} local edges, got {len(locals_)}"
    seen = set()
    for s in locals_:
        i = int(s['id'][len('local'):])
        assert s['source'] == f'L1E{i}' and s['target'] == f'L1I{i}', f"cross-pair: {s}"
        seen.add(i)
    assert seen == set(range(N_PIX)), "local edges do not cover pixels 0..8 exactly once"
    # Every enabled L1I retains all eight feedback afferents at the correct offset.
    w = e._all_weights()
    for i in range(N_PIX):
        assert f'local{i}' in w
        for j in range(N_OUT):
            assert f'fb{j}->{i}' in w
    # _all_weights fb offset matches the raw array (index 1..8 are feedback).
    for i in range(N_PIX):
        arr = e.l1.inhibitory_neurons[i]._weights_array
        assert w[f'local{i}'] == float(arr[0])
        for j in range(N_OUT):
            assert w[f'fb{j}->{i}'] == float(arr[1 + j])
    print("  enabled: 9 afferents, nine non-crossed local edges, fb offset 1: OK")


def test_local_weights_never_change():
    e = SimulationEngine(seed=5, **{**DASHBOARD_OVERRIDES,
                                    'paired_local_enabled': True,
                                    'predictive_feedback_enabled': True})
    before = np.array([n._weights_array[0] for n in e.l1.inhibitory_neurons])
    _run(e, steps=60)
    after = np.array([n._weights_array[0] for n in e.l1.inhibitory_neurons])
    assert np.array_equal(before, after), f"local afferent moved: {before} -> {after}"
    # All identical to the declared 0.40 * G magnitude.
    assert np.allclose(before, 0.40 * e._l1i_G)
    print("  local afferent fixed at 0.40*G across 60 steps: OK")


def main():
    test_disabled_is_legacy()
    test_enabled_topology()
    test_local_weights_never_change()
    print("PASS: paired-local topology contract (Phase 1)")


if __name__ == "__main__":
    main()
