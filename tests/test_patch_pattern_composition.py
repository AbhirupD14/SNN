"""Per-patch pattern composition on the 9x9 tiled system: driving different 3x3 patches
with different patterns at once, changing them independently, and confirming each L1 column
learns continuously under the dual FE/FES rule (with the coincidence cells left inert).
Covers the engine API, its serialization, the REST endpoints, and the learning behavior.
"""

import asyncio
from collections import Counter

import numpy as np
import pytest

from backend.simulation import SimulationEngine, PATTERNS


def tiled(**kw):
    return SimulationEngine(seed=1, topology='tiled_cc', leak_rate=0.0,
                            refractory_steps=0, **kw)


# =============================================================== engine API
def test_compose_union_of_patches():
    e = tiled()
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'diag \\')
    assert e.patch_pattern_map() == [
        {'row': 0, 'col': 0, 'name': 'row 1'},
        {'row': 2, 'col': 2, 'name': 'diag \\'}]
    # input is the pixelwise union: two 3-pixel patches -> 6 lit pixels.
    assert int(e.input_vec.sum()) == 6
    # clearing one patch leaves the other intact.
    e.set_patch_pattern(0, 0, None)
    assert e.patch_pattern_map() == [{'row': 2, 'col': 2, 'name': 'diag \\'}]
    assert int(e.input_vec.sum()) == 3
    e.clear_patch_patterns()
    assert e.patch_pattern_map() == [] and int(e.input_vec.sum()) == 0


def test_patch_pattern_validation():
    e = tiled()
    with pytest.raises(ValueError):
        e.set_patch_pattern(9, 9, 'row 1')          # out of the 3x3 patch grid
    with pytest.raises(KeyError):
        e.set_patch_pattern(0, 0, 'nope')           # unknown pattern
    # non-tiled topology has no patches
    flat = SimulationEngine(topology='rg_direct_cc4', leak_rate=0.0)
    with pytest.raises(ValueError):
        flat.set_patch_pattern(0, 0, 'row 1')


def test_topology_exposes_patch_patterns():
    e = tiled()
    e.set_patch_pattern(1, 1, 'col 1')
    tiling = e.topology()['tiling']
    assert tiling['patch_patterns'] == [{'row': 1, 'col': 1, 'name': 'col 1'}]


def test_set_pattern_and_manual_edits_keep_map_honest():
    e = tiled()
    e.set_patch(1, 1)
    e.set_pattern('row 1')                           # single named pattern -> current patch only
    assert e.patch_pattern_map() == [{'row': 1, 'col': 1, 'name': 'row 1'}]
    # a manual pixel edit no longer matches any patch map, so the map is dropped.
    e.toggle_pixel(0)
    assert e.patch_pattern_map() == []
    # set_input / clear likewise drop the composition view.
    e.set_patch_pattern(0, 0, 'diag /')
    e.clear_input()
    assert e.patch_pattern_map() == [] and int(e.input_vec.sum()) == 0


def test_composition_survives_rebuild_on_apply():
    e = tiled(dual_fe_fes=False)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(2, 2, 'col 1')
    # toggling the dual rule rebuilds (wipes learned state) but keeps the composed stimulus.
    e.apply_config({'dual_fe_fes': True})
    assert e.patch_pattern_map() == [
        {'row': 0, 'col': 0, 'name': 'row 1'}, {'row': 2, 'col': 2, 'name': 'col 1'}]
    assert int(e.input_vec.sum()) == 6


# ================================================ dual FE/FES learning behavior
def _run_collect(e, steps, cols):
    counts = {cid: Counter() for cid in cols}
    for _ in range(steps):
        e.step()
        for cid in cols:
            w = e.column_winners.get(cid)
            if w:
                counts[cid][w['id']] += 1
    return counts


def test_different_patches_learn_different_patterns_at_once():
    e = tiled(dual_fe_fes=True, eta=1.0, c_eta=0.5)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(1, 1, 'col 1')
    e.set_patch_pattern(2, 2, 'diag \\')
    counts = _run_collect(e, 350, ['L1c00', 'L1c11', 'L1c22', 'L1c01'])
    owners = {}
    for cid in ('L1c00', 'L1c11', 'L1c22'):
        top, n = counts[cid].most_common(1)[0]
        owners[cid] = top
        assert n / sum(counts[cid].values()) >= 0.9      # each patch consolidates one owner
    assert counts['L1c01'] == Counter()                  # unstimulated column stays silent
    # each stimulated column has its OWN owner (independent local WTAs).
    assert len(owners) == 3
    # RGC inputs never frequency-halved: all active pixels fired every boundary.
    active = [i for i in range(e.n_pix) if e.input_vec[i] > 0.5]
    fires = Counter()
    for _ in range(30):
        e.step()
        for i in active:
            if e.spiked[f'RGC{i}']:
                fires[i] += 1
    assert set(fires.values()) == {30}


def test_changing_one_patch_turns_over_only_that_column():
    e = tiled(dual_fe_fes=True, eta=1.0, c_eta=0.5)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(1, 1, 'col 1')
    c1 = _run_collect(e, 300, ['L1c00', 'L1c11'])
    o00, o11 = c1['L1c00'].most_common(1)[0][0], c1['L1c11'].most_common(1)[0][0]
    # change ONLY patch (0,0); patch (1,1) keeps its pattern.
    e.set_patch_pattern(0, 0, 'diag /')
    c2 = _run_collect(e, 300, ['L1c00', 'L1c11'])
    assert c2['L1c00'].most_common(1)[0][0] != o00       # the changed patch turns over
    assert c2['L1c11'].most_common(1)[0][0] == o11        # the untouched patch is stable


def test_l2_composes_the_l1_outputs():
    e = tiled(dual_fe_fes=True, eta=1.0, c_eta=0.5)
    e.set_patch_pattern(0, 0, 'row 1')
    e.set_patch_pattern(1, 1, 'col 1')
    l2 = Counter()
    for _ in range(600):
        e.step()
        w = e.column_winners.get('L2c00')
        if w:
            l2[w['id']] += 1
    assert l2                                            # the L2 composition column is active


# ============================================================== REST endpoints
def test_api_patch_pattern_endpoints_compose_and_clear():
    import backend.api as api

    async def scenario():
        prior = api.engine.params['topology']
        api.engine.apply_config({'topology': 'tiled_cc', 'dual_fe_fes': True})
        await api.clear_patch_patterns()   # shared engine may carry the startup composition
        try:
            r1 = await api.set_patch_pattern(api.PatchPatternBody(row=0, col=0, name='row 1'))
            r2 = await api.set_patch_pattern(api.PatchPatternBody(row=1, col=1, name='col 1'))
            cleared_one = await api.set_patch_pattern(
                api.PatchPatternBody(row=0, col=0, name=None))
            cleared_all = await api.clear_patch_patterns()
            bad = await api.set_patch_pattern(api.PatchPatternBody(row=9, col=9, name='row 1'))
            return r1, r2, cleared_one, cleared_all, bad
        finally:
            api.engine.apply_config({'topology': prior})

    r1, r2, cleared_one, cleared_all, bad = asyncio.run(scenario())
    assert {'row': 0, 'col': 0, 'name': 'row 1'} in r2['patch_patterns']
    assert {'row': 1, 'col': 1, 'name': 'col 1'} in r2['patch_patterns']
    assert cleared_one['patch_patterns'] == [{'row': 1, 'col': 1, 'name': 'col 1'}]
    assert cleared_all['patch_patterns'] == []
    assert getattr(bad, 'status_code', None) == 400     # out-of-range patch -> 400
