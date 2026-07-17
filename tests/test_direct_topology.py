"""The predictive-inhibition (PI) direct topology, selected by topology='pi'.

The network drops the L1I comparison population and adds eight pattern-specific
predictive interneurons ``PI[j]``, paired one-to-one
with the competitors ``L2E[j]``. Each PI owns nine candidate inhibitory output
synapses onto the sensory ``L1E_s`` cells (72 candidates), learned strictly
locally. There is NO global winner-to-all-L1I shortcut. These tests pin that
structure and clean toggling.
"""

from collections import Counter

import numpy as np
import pytest

from backend.simulation import SimulationEngine, N_PIX, N_OUT


@pytest.fixture
def direct():
    return SimulationEngine(seed=1, topology='pi')


def test_topology_defaults_to_pi(direct):
    assert SimulationEngine(seed=1).params['topology'] == 'pi'
    assert direct.params['topology'] == 'pi'
    assert direct.topology()['params']['topology'] == 'pi'


def test_direct_population_counts(direct):
    topo = direct.topology()
    by = Counter((n['layer'], n['type']) for n in topo['neurons'])
    assert by[('L1', 'E')] == 9            # only L1E_s
    assert by[('L2', 'E')] == 8            # competitors
    assert by[('L2', 'I')] == 9            # 8 PI + 1 L2I_WTA
    assert len(topo['neurons']) == 26
    roles = Counter(n['role'] for n in topo['neurons'])
    assert roles['predictor'] == 8         # eight predictive interneurons
    assert roles['relay'] == 1             # only L2I_WTA is a relay
    assert 'supervisor' not in roles       # no L1E_new
    assert not any(n['id'].startswith('L1Enew') for n in topo['neurons'])
    assert not any(n['id'].startswith('L1I') for n in topo['neurons'])
    assert len(direct.pi) == 8 and direct.l1i == []


def test_each_l2e_drives_only_its_paired_pi(direct):
    re_pi = [s for s in direct.topology()['synapses']
             if s['kind'] == 'relay_excitation' and s['target'].startswith('PI')]
    assert len(re_pi) == N_OUT             # exactly one relay edge per PI
    assert {(s['source'], s['target']) for s in re_pi} == \
        {(f'L2E{j}', f'PI{j}') for j in range(N_OUT)}      # strictly paired 1:1


def test_each_pi_has_nine_candidate_outputs(direct):
    pi_out = [s for s in direct.topology()['synapses']
              if s['kind'] == 'predictive_inhibition']
    assert len(pi_out) == N_OUT * N_PIX    # 72 candidate synapses
    for j in range(N_OUT):
        targets = {s['target'] for s in pi_out if s['source'] == f'PI{j}'}
        assert targets == {f'L1E{i}' for i in range(N_PIX)}   # dense onto all L1E_s
    for s in pi_out:
        assert s['sign'] == -1
        assert s['weight'] is not None     # live per-synapse PI weight (starts at 0)


def test_direct_edge_counts(direct):
    kinds = Counter(s['kind'] for s in direct.topology()['synapses'])
    assert kinds['feedforward'] == N_PIX * N_OUT               # 72
    assert kinds['relay_excitation'] == N_OUT + N_OUT          # 8 (L2E->PI) + 8 (L2E->L2I)
    assert kinds['predictive_inhibition'] == N_OUT * N_PIX     # 72
    assert kinds['inhibition'] == N_OUT                        # L2I->L2E (WTA conductance)
    assert kinds['feedback'] == 0 and kinds['coincidence_local'] == 0
    assert sum(kinds.values()) == 168


def test_no_global_winner_to_all_l1i_shortcut(direct):
    # The retired direct branch fired every winner into nine Boolean L1I relays.
    # That population and those edges must be gone.
    for s in direct.topology()['synapses']:
        assert not s['target'].startswith('L1I')
        assert not (s['source'].startswith('L2E') and s['target'].startswith('L1E')
                    and not s['target'].startswith('L1Enew'))   # no direct L2E->L1E_s


def test_pi_weights_start_at_zero(direct):
    for pi in direct.pi:
        assert np.all(pi.w == 0.0)         # no meaningful inhibition before learning


def test_toggle_rebuilds_and_resets_pi_weights():
    eng = SimulationEngine(seed=1, topology='pi')
    eng.set_pattern('row 1')
    for _ in range(200):
        eng.step()
    assert any(np.any(pi.w > 0) for pi in eng.pi)   # learned something
    # Switch to the old topology and back.
    assert eng.apply_config({'topology': 'old'}) == ['topology']
    assert len(eng.topology()['neurons']) == 27
    assert Counter(n['role'] for n in eng.topology()['neurons'])['relay'] == 10
    assert eng.apply_config({'topology': 'pi'}) == ['topology']
    assert len(eng.topology()['neurons']) == 26
    for pi in eng.pi:                                # rebuild wipes learned weights
        assert np.all(pi.w == 0.0)
