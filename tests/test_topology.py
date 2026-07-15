"""Exact population and edge counts, absence of the forbidden L2E->L1I path, and
the frozen subtractive-gate invariant.
"""

from collections import Counter

import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT
from snn.neurons import E_THRESHOLD


@pytest.fixture
def engine():
    return SimulationEngine(seed=1)


def test_population_counts(engine):
    topo = engine.topology()
    by = Counter((n['layer'], n['type']) for n in topo['neurons'])
    assert by[('L1', 'E')] == 18          # 9 L1E_s + 9 L1E_new
    assert by[('L1', 'I')] == 9
    assert by[('L2', 'E')] == 8
    assert by[('L2', 'I')] == 1
    assert len(topo['neurons']) == 36


def test_roles_present(engine):
    roles = Counter(n['role'] for n in engine.topology()['neurons'])
    assert roles['source'] == 9
    assert roles['supervisor'] == 9
    assert roles['competitor'] == 8
    assert roles['relay'] == 10           # 9 L1I + 1 L2I


def test_internal_edge_counts(engine):
    kinds = Counter(s['kind'] for s in engine.topology()['synapses'])
    assert kinds['feedforward'] == N_PIX * N_OUT       # 72
    assert kinds['feedback'] == N_OUT * N_PIX          # 72
    assert kinds['coincidence_local'] == N_PIX         # 9 paired sensory afferents
    assert kinds['relay_excitation'] == N_PIX + N_OUT  # 9 + 8 = 17
    assert kinds['inhibition'] == N_PIX + N_OUT        # 9 + 8 = 17
    assert sum(kinds.values()) == 187


def test_one_local_coincidence_edge_per_pixel(engine):
    local = [s for s in engine.topology()['synapses'] if s['kind'] == 'coincidence_local']
    assert len(local) == N_PIX
    # Exactly one L1E_s[i] -> L1E_new[i] edge per pixel, paired one-to-one.
    pairs = {(s['source'], s['target']) for s in local}
    assert pairs == {(f'L1E{i}', f'L1Enew{i}') for i in range(N_PIX)}


def test_no_l2e_to_l1i_edges(engine):
    for s in engine.topology()['synapses']:
        assert not (s['source'].startswith('L2E') and s['target'].startswith('L1I'))
    # Feedback must target the supervisory L1E_new population, never L1I directly.
    fb = [s for s in engine.topology()['synapses'] if s['kind'] == 'feedback']
    assert all(s['target'].startswith('L1Enew') for s in fb)


def test_subtractive_gates_equal_threshold_and_frozen(engine):
    inh = [s for s in engine.topology()['synapses'] if s['kind'] == 'inhibition']
    assert len(inh) == 17
    for s in inh:
        assert s['weight'] == pytest.approx(E_THRESHOLD)
        assert s['sign'] == -1
    # The engine has no subtractive plasticity: source E neurons never touch a gate.
    for n in (*engine.l1e_s, *engine.l2e):
        assert n.subt_magnitude == pytest.approx(E_THRESHOLD)
        assert n.learn in (True, False)   # only acc_weights learn; gate is a scalar constant


def test_relay_edges_are_structural(engine):
    relays = [s for s in engine.topology()['synapses'] if s['kind'] == 'relay_excitation']
    assert all(s['weight'] is None for s in relays)     # structural, no learned magnitude
    # E->I direction only.
    for s in relays:
        assert s['source'][:3] in ('L1E', 'L2E') and s['target'] in \
            {f'L1I{i}' for i in range(N_PIX)} | {'L2I'}
