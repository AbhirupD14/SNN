"""The 'old' dense global-inhibition topology (topology='old').

Nine paired L1I relays, fed DENSELY by every L2E (every L2E -> every L1I), each
projecting a paired inhibitory conductance onto its own L1E_s. The single L2 winner
therefore drives all nine L1I, shunting every L1E_s on the next boundary -- global
inhibition gated by the winner. There is no L1E_new coincidence population and no
PI cell in this topology. These tests pin the structure, the dense wiring, the
winner-gated global inhibition, and clean toggling against the PI topology.
"""

from collections import Counter

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT


@pytest.fixture
def old():
    return SimulationEngine(seed=1, topology='old')


def test_default_topology_is_pi(old):
    assert SimulationEngine(seed=1).params['topology'] == 'pi'
    assert old.params['topology'] == 'old'
    assert old.topology()['params']['topology'] == 'old'


def test_old_population_counts(old):
    topo = old.topology()
    by = Counter((n['layer'], n['type']) for n in topo['neurons'])
    assert by[('L1', 'E')] == 9            # L1E_s sources
    assert by[('L1', 'I')] == 9            # nine paired L1I relays
    assert by[('L2', 'E')] == 8            # competitors
    assert by[('L2', 'I')] == 1            # only L2I_WTA
    assert len(topo['neurons']) == 27
    roles = Counter(n['role'] for n in topo['neurons'])
    assert roles['source'] == 9
    assert roles['relay'] == 10            # 9 L1I + 1 L2I
    assert 'predictor' not in roles        # no PI cells
    assert 'supervisor' not in roles       # no L1E_new
    assert not any(n['id'].startswith(('L1Enew', 'PI')) for n in topo['neurons'])
    assert len(old.l1i) == 9 and old.pi == []


def test_old_edge_counts(old):
    kinds = Counter(s['kind'] for s in old.topology()['synapses'])
    assert kinds['feedforward'] == N_PIX * N_OUT               # 72
    assert kinds['relay_excitation'] == N_OUT * N_PIX + N_OUT  # 72 dense L2E->L1I + 8 L2E->L2I
    assert kinds['inhibition'] == N_PIX + N_OUT                # 9 paired L1I->L1E + 8 WTA
    assert kinds['predictive_inhibition'] == 0
    assert kinds['feedback'] == 0 and kinds['coincidence_local'] == 0
    assert sum(kinds.values()) == 169


def test_l2e_to_l1i_is_dense(old):
    re = [s for s in old.topology()['synapses']
          if s['kind'] == 'relay_excitation' and s['target'].startswith('L1I')]
    assert len(re) == N_OUT * N_PIX        # every L2E -> every L1I
    assert {(s['source'], s['target']) for s in re} == \
        {(f'L2E{j}', f'L1I{i}') for j in range(N_OUT) for i in range(N_PIX)}


def test_l1i_to_l1e_is_paired_conductance(old):
    inh = [s for s in old.topology()['synapses']
           if s['kind'] == 'inhibition' and s['target'].startswith('L1E')]
    assert len(inh) == N_PIX
    assert {(s['source'], s['target']) for s in inh} == \
        {(f'L1I{i}', f'L1E{i}') for i in range(N_PIX)}
    for s in inh:
        assert s['sign'] == -1
        assert s['weight'] is None         # structural conductance, no learned magnitude


def test_winner_drives_global_inhibition(old):
    """A single L2 winner fires all nine L1I this boundary; every L1E_s receives a
    'global_l1i' conductance pulse on the NEXT boundary (winner-gated global inhib)."""
    old.set_pattern('row 1')
    hit = None
    for _ in range(400):
        d = old.step()
        if d['winner'] is not None:
            fired = [n['id'] for n in d['neurons'] if n['id'].startswith('L1I') and n['spiked']]
            if len(fired) == N_PIX:
                hit = d
                break
    assert hit is not None, 'no full L1I volley observed'
    # No same-boundary inhibition on L1E: the pulse is delayed one step.
    assert not any(p['target'].startswith('L1E') for p in hit['inhibitory_pulses'])
    d2 = old.step()
    # Inhibitory conductance onto a sensory (non-competitor) target is tagged 'inhibition'.
    pulses = [p for p in d2['inhibitory_pulses']
              if p['kind'] == 'inhibition' and p['target'].startswith('L1E')]
    assert {p['target'] for p in pulses} == {f'L1E{i}' for i in range(N_PIX)}
    assert all(p['conductance_increment'] > 0 for p in pulses)


def test_toggle_between_pi_and_old_rebuilds_cleanly():
    eng = SimulationEngine(seed=1, topology='pi')
    assert len(eng.topology()['neurons']) == 26
    eng.set_pattern('row 1')
    for _ in range(200):
        eng.step()
    assert any(np.any(pi.w > 0) for pi in eng.pi)          # PI learned something
    # Switch to the old topology.
    assert eng.apply_config({'topology': 'old'}) == ['topology']
    assert len(eng.topology()['neurons']) == 27
    assert eng.pi == [] and len(eng.l1i) == 9
    assert Counter(n['role'] for n in eng.topology()['neurons'])['relay'] == 10
    # Switch back; PI weights are wiped by the rebuild.
    assert eng.apply_config({'topology': 'pi'}) == ['topology']
    assert len(eng.topology()['neurons']) == 26
    for pi in eng.pi:
        assert np.all(pi.w == 0.0)


def test_invalid_topology_rejected():
    with pytest.raises(ValueError):
        SimulationEngine(seed=1, topology='nonsense')
