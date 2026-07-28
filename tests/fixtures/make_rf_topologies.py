"""Generate the receptive-field panel's topology fixtures FROM THE REAL ENGINE
(never hand-maintained).

Writes one committed artifact next to this script:

    rf_topologies.json    {preset: {neurons, synapses, params, grid, tiling}}

The JS model test (tests/receptive.model.test.mjs) builds the panel model from these
exact payloads, so the browser panel is exercised against every current input-surface
shape -- the 3x3 legacy presets, the 9x9 tiled hierarchy and the 9x18 two-tower
composition graph -- rather than a hand-written stub that can drift from the engine.

Only the fields the panel actually reads are stored, so the fixture stays small and a
change to unrelated serialization does not churn it.

Regenerate after a topology/serialization change::

    PYTHONPATH=. .venv/bin/python tests/fixtures/make_rf_topologies.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.simulation import SimulationEngine        # noqa: E402

# One preset per distinct input-surface shape the panel must lay out correctly.
PRESETS = ("rg_coincidence", "rg_direct_cc4", "tiled_cc", "two_tower_composition")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_topologies.json")


def main() -> int:
    out = {}
    for name in PRESETS:
        topo = SimulationEngine(seed=1, topology=name).topology()
        out[name] = {
            "neurons": topo["neurons"],
            "synapses": topo["synapses"],
            "params": topo["params"],
            "grid": topo["grid"],
            "tiling": topo.get("tiling"),
        }
    with open(OUT, "w") as f:
        json.dump(out, f, sort_keys=True)
    for name, payload in out.items():
        print(f"{name}: {len(payload['neurons'])} neurons, {len(payload['synapses'])} synapses")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
