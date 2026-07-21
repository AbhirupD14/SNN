"""Server-side preset persistence: built-in listing, save/load/delete round-trip,
name sanitization, reserved-name and validity guards.
"""

import pytest

from backend import presets as ps
from backend.network_spec import preset_spec, SpecError

N_PIX, N_OUT = 9, 8


@pytest.fixture(autouse=True)
def _tmp_preset_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, 'PRESET_DIR', str(tmp_path / 'presets'))


def _mini():
    return {'name': 'mini',
            'nodes': [{'id': 'S0', 'archetype': 'e_sensory', 'pixel': 0},
                      {'id': 'C0', 'archetype': 'e_competitor'}],
            'edges': [{'source': 'S0', 'target': 'C0', 'kind': 'feedforward'}]}


def test_builtins_always_listed_first():
    lst = ps.list_presets(N_PIX, N_OUT)
    assert [p['name'] for p in lst[:4]] == ['pi', 'old', 'rg', 'rg_residual']
    assert all(p['builtin'] for p in lst[:4])


def test_builtin_specs_loadable():
    for name in ('pi', 'old', 'rg', 'rg_residual'):
        spec = ps.load_spec(name, N_PIX, N_OUT)
        assert spec['name'] == name and spec['nodes']


def test_save_load_delete_round_trip():
    name = ps.save_preset('My Graph 1', _mini(), N_PIX)
    assert name == 'My Graph 1'
    listed = {p['name'] for p in ps.list_presets(N_PIX, N_OUT)}
    assert 'My Graph 1' in listed
    spec = ps.load_spec('My Graph 1', N_PIX, N_OUT)
    assert spec['name'] == 'My Graph 1' and len(spec['nodes']) == 2
    assert ps.delete_preset('My Graph 1') is True
    assert 'My Graph 1' not in {p['name'] for p in ps.list_presets(N_PIX, N_OUT)}


def test_reserved_and_invalid_names_rejected():
    with pytest.raises(ValueError):
        ps.save_preset('pi', _mini(), N_PIX)          # reserved built-in
    with pytest.raises(ValueError):
        ps.save_preset('!!!', _mini(), N_PIX)         # sanitizes to empty


def test_invalid_spec_not_saved():
    bad = {'nodes': [{'id': 'X', 'archetype': 'nope'}], 'edges': []}
    with pytest.raises(SpecError):
        ps.save_preset('bad', bad, N_PIX)
    assert 'bad' not in {p['name'] for p in ps.list_presets(N_PIX, N_OUT)}


def test_builtins_cannot_be_deleted():
    for name in ('pi', 'old', 'rg', 'rg_residual'):
        assert ps.delete_preset(name) is False


def test_missing_preset_raises_keyerror():
    with pytest.raises(KeyError):
        ps.load_spec('does-not-exist', N_PIX, N_OUT)
