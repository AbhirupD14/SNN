"""Tiled cortical-column input surface, patch embedding, and dimension configuration
(Phase 3). Legacy 9/8 behavior must remain untouched.
"""

import numpy as np
import pytest

from backend.simulation import SimulationEngine, TILED_CC_INPUT


def _active_rgc_patches(engine):
    """The set of patch ids whose RGCs are currently active."""
    patches = set()
    for n in engine.spec['nodes']:
        if n['archetype'] == 'rg_source' and engine.input_vec[n['pixel']] > 0.5:
            patches.add(n['patch_id'])
    return patches


def test_tiled_engine_owns_81_element_input():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    assert e.n_pix == TILED_CC_INPUT == 81
    assert e.input_vec.shape == (81,) and e.input_vec.sum() == 0.0    # boots blank
    assert e.current_pattern is None


def test_global_pixels_map_to_correct_rgc_and_patch():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    by_pixel = {n['pixel']: n for n in e.spec['nodes'] if n['archetype'] == 'rg_source'}
    assert by_pixel[0]['input_row'] == 0 and by_pixel[0]['input_col'] == 0
    assert by_pixel[0]['patch_id'] == 0
    assert by_pixel[80]['input_row'] == 8 and by_pixel[80]['input_col'] == 8
    assert by_pixel[80]['patch_id'] == 8               # bottom-right 3x3 patch
    assert by_pixel[40]['patch_id'] == 4               # center patch (1,1)


def test_local_pattern_activates_only_selected_patch():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    for pr in range(3):
        for pc in range(3):
            e.set_patch(pr, pc)
            e.set_pattern('row 1')
            assert _active_rgc_patches(e) == {pr * 3 + pc}     # ONLY that patch fires
            assert e.input_vec.sum() == 3


def test_full_diagonal_activates_three_diagonal_subpatches():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    v = np.zeros(81)
    for r in range(9):
        v[r * 9 + r] = 1.0                             # full 9x9 main diagonal
    e.set_input(v)
    # the diagonal crosses exactly patches (0,0)=0, (1,1)=4, (2,2)=8
    assert _active_rgc_patches(e) == {0, 4, 8}
    # and only the three diagonal L1 columns ever win under it
    won = set()
    for _ in range(40):
        d = e.step()
        won |= set(d['column_winners'])
    assert won <= {'L1c00', 'L1c11', 'L1c22', 'L2c00'}
    assert won & {'L1c00', 'L1c11', 'L1c22'}           # at least one diagonal column wins


def test_cc_e_count_changes_every_column():
    e = SimulationEngine(seed=1, topology='tiled_cc', cc_e_count=5)
    counts = {}
    for n in e.spec['nodes']:
        if n.get('column_role') == 'E':
            counts[n['column_id']] = counts.get(n['column_id'], 0) + 1
    assert set(counts.values()) == {5}
    assert len(e.topology()['neurons']) == 10 * 5 + 111


def test_reset_and_reseed_preserve_dims():
    e = SimulationEngine(seed=1, topology='tiled_cc', cc_e_count=6)
    e.reset()
    assert e.n_pix == 81 and e.cc_e_count == 6
    e.reseed()
    assert e.n_pix == 81 and e.cc_e_count == 6
    assert len(e.topology()['neurons']) == 10 * 6 + 111


def test_dashboard_switch_legacy_tiled_legacy_resolves_dims():
    e = SimulationEngine(seed=1, topology='rg_coincidence')       # 9-input coincidence graph
    assert (e.n_pix, len(e.topology()['neurons'])) == (9, 45)
    e.apply_config({'topology': 'tiled_cc'})
    assert (e.n_pix, len(e.topology()['neurons'])) == (81, 191)
    assert e.input_vec.shape == (81,) and e.input_vec.sum() == 0.0     # old input cleared
    e.apply_config({'topology': 'rg_coincidence'})
    assert (e.n_pix, len(e.topology()['neurons'])) == (9, 45)          # 9-input graph restored


def test_invalid_input_and_patch_fail_loudly():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    with pytest.raises(ValueError):
        e.set_input([1, 0, 1])                          # wrong length
    with pytest.raises(ValueError):
        e.set_patch(3, 0)                               # out of the 3x3 patch grid
    with pytest.raises(ValueError):
        e.set_patch(0, 5)


def test_explicit_n_pix_override_rejected_for_tiled():
    with pytest.raises(ValueError):
        SimulationEngine(seed=1, topology='tiled_cc', n_pix=9)
    with pytest.raises(ValueError):
        SimulationEngine(seed=1, topology='tiled_cc', cc_e_count=0)


def test_topology_pattern_bank_is_topology_sized():
    e = SimulationEngine(seed=1, topology='tiled_cc')
    topo = e.topology()
    # never advertises a 9-length vector to an 81-input engine
    assert all(len(v) == 81 for v in topo['pattern_vectors'].values())
    assert set(topo['patterns']) == {'row 1', 'col 1', 'diag \\', 'diag /'}
    # legacy engine keeps the 3x3 vectors
    leg = SimulationEngine(seed=1, topology='rg_coincidence').topology()
    assert all(len(v) == 9 for v in leg['pattern_vectors'].values())


def test_legacy_presets_unaffected_by_tiled_additions():
    e = SimulationEngine(seed=1, topology='rg_coincidence')
    assert e.tiled_meta is None
    d = e.step()
    assert d['column_winners'] == {}                    # empty for legacy graphs
    assert 'tiling' not in e.topology()                 # no tiled block on legacy topo
