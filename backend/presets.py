"""Server-side persistence for topology presets.

Built-in presets ('pi', 'old', 'rg', 'rg_residual') are generated on demand from
``network_spec``. User
presets are NetworkSpec JSON files under ``.claude/presets/`` (next to the persisted
dashboard seed), so a saved topology survives a server restart and can be reloaded.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from .network_spec import (
    preset_spec, validate_spec, PRESETS, tiled_input_size, tiled_cc_spec,
    TILED_CC_DEFAULTS, TILED_PRESETS,
)
from .layout import generate_layout, generate_tiled_layout


def _resolved_dims(name: str, n_pix: int, n_out: int) -> tuple[int, int]:
    """Each preset's own construction dimensions -- so listing/loading a tiled preset
    never inherits the currently active engine's n_pix by accident."""
    if name in TILED_PRESETS:
        return (TILED_CC_DEFAULTS["input_rows"] * TILED_CC_DEFAULTS["input_cols"], n_out)
    return (n_pix, n_out)


def _with_layout_positions(spec: dict, n_pix: int, n_out: int) -> dict:
    """Attach the seeded functional 3D positions to a preset's nodes so the editor shows
    its real 3D structure. Display only -- the engine's build path fills positions from
    its own seeded layout, so this never affects simulated behaviour. Tiled specs use the
    metadata-driven tiled layout so their columns load at their real positions."""
    if tiled_input_size(spec) is not None:
        pos = generate_tiled_layout(np.random.default_rng(1), spec)
    else:
        pos = generate_layout(np.random.default_rng(1), n_pix, n_out)
    for node in spec["nodes"]:
        if node.get("pos") is None and node["id"] in pos:
            node["pos"] = [round(float(x), 4) for x in pos[node["id"]]]
    return spec

_STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".claude")
PRESET_DIR = os.path.join(_STATE_DIR, "presets")

BUILTINS = PRESETS                 # single source of truth for built-in names
_SAFE_NAME = re.compile(r"[^A-Za-z0-9 _-]+")
MAX_NAME = 48


def _sanitize(name: str) -> str:
    name = _SAFE_NAME.sub("", (name or "").strip())[:MAX_NAME].strip()
    return name


def _path(name: str) -> str:
    # File name is the sanitized display name; safe chars only, so it maps 1:1.
    return os.path.join(PRESET_DIR, f"{name}.json")


def builtin_spec(name: str, n_pix: int, n_out: int) -> dict:
    return preset_spec(name, n_pix, n_out)


def list_presets(n_pix: int, n_out: int) -> list[dict]:
    """Built-in presets first, then saved user presets (alphabetical). Each entry is a
    lightweight summary; the full spec is fetched on load. Node/edge counts use each
    preset's OWN resolved construction dimensions (tiled_cc reports its 191/1052, not a
    9-input miscount from the active engine)."""
    out = []
    for b in BUILTINS:
        bp, bo = _resolved_dims(b, n_pix, n_out)
        spec = preset_spec(b, bp, bo)
        out.append(dict(name=b, builtin=True,
                        nodes=len(spec["nodes"]), edges=len(spec["edges"])))
    try:
        names = sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json"))
    except OSError:
        names = []
    for name in names:
        try:
            with open(_path(name)) as f:
                spec = json.load(f)
            out.append(dict(name=name, builtin=False,
                            nodes=len(spec.get("nodes", [])), edges=len(spec.get("edges", []))))
        except (OSError, ValueError):
            continue
    return out


def load_spec(name: str, n_pix: int, n_out: int) -> dict:
    """Return the NetworkSpec for a built-in or saved preset. Raises KeyError if the
    saved preset is missing, ValueError if it is corrupt/invalid."""
    if name in BUILTINS:
        bp, bo = _resolved_dims(name, n_pix, n_out)
        return _with_layout_positions(preset_spec(name, bp, bo), bp, bo)
    clean = _sanitize(name)
    if not clean:
        raise KeyError(name)
    try:
        with open(_path(clean)) as f:
            spec = json.load(f)
    except FileNotFoundError as e:
        raise KeyError(name) from e
    # A tiled saved spec is validated at its OWN declared input size (e.g. 81), not the
    # currently active engine's n_pix, so it never fails a stale pixel-range check.
    spec = validate_spec(spec, tiled_input_size(spec) or n_pix)
    spec["name"] = clean
    return spec


def save_preset(name: str, spec: dict, n_pix: int) -> str:
    """Validate and persist ``spec`` under a sanitized ``name``. Returns the stored
    name. Reserved built-in names are rejected."""
    clean = _sanitize(name)
    if not clean:
        raise ValueError("preset name must contain at least one letter or digit")
    if clean in BUILTINS:
        raise ValueError(f"'{clean}' is a reserved built-in preset name")
    # Tiled specs are validated (and persisted) at their own declared input size.
    norm = validate_spec(spec, tiled_input_size(spec) or n_pix)
    norm["name"] = clean
    os.makedirs(PRESET_DIR, exist_ok=True)
    with open(_path(clean), "w") as f:
        json.dump(norm, f, indent=2)
    return clean


def delete_preset(name: str) -> bool:
    """Delete a saved preset. Built-ins cannot be deleted. Returns True if removed."""
    clean = _sanitize(name)
    if clean in BUILTINS or not clean:
        return False
    try:
        os.remove(_path(clean))
        return True
    except OSError:
        return False
