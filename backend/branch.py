"""Replay-branch request layer: the strict compatibility contract and the request
orchestration behind ``POST /api/replay/branch-weights``.

A branch reconstructs the recorded synaptic weights at a selected replay frame and installs
them into a freshly reset live engine (see ``SimulationEngine.branch_from_weights``). It is a
NEW BRANCH from learned weights, never an exact continuation: only weights (and, optionally,
the raw input vector) are restored; all transient neuron/event/RNG/timestep state is reset.

This module is deliberately DOM-free and framework-free so it is unit-testable without an
HTTP client. ``api.py`` calls :func:`apply_branch_request`, then broadcasts; it never
duplicates any validation here.

SECURITY: the replay artifact is untrusted. The recorded topology/config the browser sends is
compared -- as plain data -- against the live engine's own authoritative topology; nothing in
the payload is ever treated as a path, and no file is read or code evaluated. The backend does
NOT trust the browser's "compatible" assertion: it re-derives the live side itself.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from experiments.replay_recorder import REPLAY_SCHEMA_NAME, REPLAY_SCHEMA_VERSION

# The branch payload's own schema (independent of the replay schema it is derived from).
BRANCH_SCHEMA_NAME = "snn.branch"
BRANCH_SCHEMA_VERSION = 1

# Model-affecting engine parameters compared for compatibility. Every value here is present in
# a serialized topology's ``params`` block and changes the computation (threshold, leak,
# refractory, inhibition, delay, learning rates, the active dual FE/FES rule + its parameters,
# and the construction dimensions). Excluded on purpose (documented in docs/REPLAY_PLAYER.md):
# display-only state (selected_patch, patch_patterns, current input, timestep, running), the
# recorder metadata, and the headless-only base update modes (e_weight_update_mode /
# c_weight_update_mode) which are NOT represented in a serialized topology payload.
_MODEL_PARAM_KEYS = (
    "e_threshold", "leak_rate", "refractory_steps", "eta", "c_eta",
    "alpha_inh", "alpha_inh_l1", "alpha_a", "beta_v", "beta_s", "a_max", "e_inh",
    "pi_eta", "pi_w_max", "pi_lt_decay", "pi_g_scale",
    "pi_conductance_enabled", "pi_plasticity_enabled", "l2i_g_scale",
    "enc_plasticity_enabled", "enc_w_init", "residual_exc_scale",
    "switch_trace_decay", "switch_trace_threshold", "switch_residual_charge_frac",
    "switch_trace_charge_frac", "switch_branch_cap_frac", "switch_g_scale",
    "switch_conductance_enabled", "l2_init_total_frac",
    "dual_fe_fes", "dual_fe_e", "dual_fe_wte", "dual_fe_B",
    "n_pix", "n_out", "cc_e_count", "ff_init_mean", "synaptic_delay",
    "e_weight_cap", "e_maturity_budget_frac", "topology", "topology_name",
    "is_custom_topology",
)

# Neuron identity/metadata fields the computation depends on (archetype/role/class/layer and
# the cortical-column grouping). Display-only labels/positions are excluded.
_NEURON_KEYS = ("archetype", "role", "type", "layer",
                "column_id", "column_role", "column_index")

# Tiling metadata that shapes the graph (family/variant/dimensions/column membership). The
# display-only ``selected_patch`` and ``patch_patterns`` are excluded.
_TILING_KEYS = ("family", "variant", "input_shape", "patch_shape", "grid_shape",
                "column_layers", "columns", "cc_e_count")


class BranchError(ValueError):
    """A branch request that must be refused. ``status_code`` is the HTTP status the API
    should return (400 for a bad/unsupported request, 409 for an incompatible live engine)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _num_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def build_branch_contract(topology: Mapping[str, Any]) -> dict:
    """Extract the canonical compatibility contract from a serialized topology payload
    (``engine.topology()`` shape). Pure: reads only the given mapping. Weight VALUES are
    excluded (they are exactly what a branch replaces); only the SET of weighted/mutable
    synapses is kept."""
    if not isinstance(topology, Mapping):
        raise BranchError("recorded topology is not an object")
    neurons_in = topology.get("neurons")
    synapses_in = topology.get("synapses")
    params_in = topology.get("params")
    if not isinstance(neurons_in, list) or not isinstance(synapses_in, list) \
            or not isinstance(params_in, Mapping):
        raise BranchError("recorded topology is missing neurons/synapses/params")

    neurons = {}
    for n in neurons_in:
        nid = n.get("id")
        if not isinstance(nid, str):
            raise BranchError("recorded topology has a neuron without a string id")
        neurons[nid] = {k: n.get(k) for k in _NEURON_KEYS}

    synapses = {}
    weighted = set()
    for s in synapses_in:
        sid = s.get("id")
        if not isinstance(sid, str):
            raise BranchError("recorded topology has a synapse without a string id")
        synapses[sid] = {"source": s.get("source"), "target": s.get("target"),
                         "kind": s.get("kind"), "sign": s.get("sign")}
        if s.get("weight") is not None:
            weighted.add(sid)

    params = {k: params_in.get(k) for k in _MODEL_PARAM_KEYS}
    tiling_in = topology.get("tiling")
    tiling = ({k: tiling_in.get(k) for k in _TILING_KEYS}
              if isinstance(tiling_in, Mapping) else None)
    return {
        "seed": params_in.get("seed"),
        "neurons": neurons,
        "synapses": synapses,
        "weighted": weighted,
        "params": params,
        "tiling": tiling,
    }


def compare_contracts(recorded: Mapping[str, Any], live: Mapping[str, Any]) -> Optional[str]:
    """Return ``None`` if the recorded contract is compatible with the live engine, else a
    precise, actionable message describing the FIRST meaningful mismatch. Never mutates."""
    # ---- seed: required (positions/distance-factors are seed-dependent, non-restored) ----
    if not _num_eq(recorded.get("seed"), live.get("seed")):
        return (f"seed differs (recorded {recorded.get('seed')} vs live {live.get('seed')}); "
                f"reseed/reset the live engine to seed {recorded.get('seed')} before branching "
                f"(a branch does not resume the recorded RNG stream)")

    # ---- neuron identity + role/column metadata ----
    rN, lN = recorded.get("neurons", {}), live.get("neurons", {})
    only_rec = set(rN) - set(lN)
    only_live = set(lN) - set(rN)
    if only_rec:
        return (f"live graph is missing {len(only_rec)} recorded neuron(s), e.g. "
                f"{sorted(only_rec)[:5]}; load the matching topology before branching")
    if only_live:
        return (f"live graph has {len(only_live)} neuron(s) not in the recording, e.g. "
                f"{sorted(only_live)[:5]}; load the matching topology before branching")
    for nid in sorted(rN):
        for k in _NEURON_KEYS:
            if rN[nid].get(k) != lN[nid].get(k):
                return (f"neuron {nid!r} {k} differs (recorded {rN[nid].get(k)!r} vs live "
                        f"{lN[nid].get(k)!r}); load the matching topology before branching")

    # ---- synapse identity/kind + the weighted set ----
    rS, lS = recorded.get("synapses", {}), live.get("synapses", {})
    only_rec = set(rS) - set(lS)
    only_live = set(lS) - set(rS)
    if only_rec:
        return (f"live graph is missing {len(only_rec)} recorded synapse(s), e.g. "
                f"{sorted(only_rec)[:5]}; load the matching topology before branching")
    if only_live:
        return (f"live graph has {len(only_live)} synapse(s) not in the recording, e.g. "
                f"{sorted(only_live)[:5]}; load the matching topology before branching")
    for sid in sorted(rS):
        for k in ("source", "target", "kind", "sign"):
            if rS[sid].get(k) != lS[sid].get(k):
                return (f"synapse {sid!r} {k} differs (recorded {rS[sid].get(k)!r} vs live "
                        f"{lS[sid].get(k)!r}); load the matching topology before branching")
    if set(recorded.get("weighted", ())) != set(live.get("weighted", ())):
        diff = (set(recorded.get("weighted", ())) ^ set(live.get("weighted", ())))
        return (f"the set of weighted synapses differs ({len(diff)} edge(s), e.g. "
                f"{sorted(diff)[:5]}); load the matching topology before branching")

    # ---- model-affecting parameters ----
    rP, lP = recorded.get("params", {}), live.get("params", {})
    for k in _MODEL_PARAM_KEYS:
        if not _num_eq(rP.get(k), lP.get(k)):
            return (f"model parameter {k!r} differs (recorded {rP.get(k)!r} vs live "
                    f"{lP.get(k)!r}); set it to {rP.get(k)!r} (or load the matching preset) "
                    f"before branching")

    # ---- tiling family/variant/dimensions/columns ----
    rT, lT = recorded.get("tiling"), live.get("tiling")
    if (rT is None) != (lT is None):
        return ("tiled-topology metadata is present on one side only; load the matching "
                "topology before branching")
    if rT is not None:
        for k in _TILING_KEYS:
            if rT.get(k) != lT.get(k):
                return (f"tiling {k} differs; load the matching tiled topology before branching")
    return None


def apply_branch_request(engine: Any, payload: Mapping[str, Any]) -> dict:
    """Validate a branch payload and, if valid, branch the engine. Raises :class:`BranchError`
    (HTTP status attached) on any problem, leaving the engine unchanged. Does NOT touch the
    runner/connection manager -- the API layer owns pause/broadcast around this call.

    The caller MUST have paused the runner first. Order: validate schemas, then topology/config
    compatibility (no mutation), then hand the snapshot to the engine which validates every
    weight/input BEFORE it resets or installs anything.
    """
    if not isinstance(payload, Mapping):
        raise BranchError("branch request body must be a JSON object")

    # ---- branch + replay schema ----
    if payload.get("branch_schema") != BRANCH_SCHEMA_NAME:
        raise BranchError(f"unsupported branch schema {payload.get('branch_schema')!r} "
                          f"(expected {BRANCH_SCHEMA_NAME!r})")
    if payload.get("branch_schema_version") != BRANCH_SCHEMA_VERSION:
        raise BranchError(f"unsupported branch schema version "
                          f"{payload.get('branch_schema_version')!r} "
                          f"(this server supports {BRANCH_SCHEMA_VERSION})")
    if payload.get("replay_schema") != REPLAY_SCHEMA_NAME:
        raise BranchError(f"unsupported replay schema {payload.get('replay_schema')!r} "
                          f"(expected {REPLAY_SCHEMA_NAME!r})")
    if payload.get("replay_schema_version") != REPLAY_SCHEMA_VERSION:
        raise BranchError(f"unsupported replay schema version "
                          f"{payload.get('replay_schema_version')!r} "
                          f"(this server supports {REPLAY_SCHEMA_VERSION})")

    weights = payload.get("weights")
    if not isinstance(weights, Mapping):
        raise BranchError("branch request is missing the weights object")
    # Bound the request by the live graph BEFORE any deeper validation so an oversized payload
    # is refused cheaply (an exact-match snapshot can never exceed the live synapse count).
    if len(weights) > len(engine.synapses):
        raise BranchError(f"weight snapshot has {len(weights)} entries but the live graph has "
                          f"only {len(engine.synapses)} synapses")
    inp = payload.get("input")
    if inp is not None:
        if not isinstance(inp, list):
            raise BranchError("input must be a list of 0/1 pixels")
        if len(inp) > engine.n_pix:
            raise BranchError(f"input vector has {len(inp)} pixels but the live graph has "
                              f"only {engine.n_pix}")

    # ---- topology/config compatibility (independent of the browser's assertion) ----
    recorded_topology = payload.get("recorded_topology")
    if not isinstance(recorded_topology, Mapping):
        raise BranchError("branch request is missing the recorded topology contract")
    recorded_contract = build_branch_contract(recorded_topology)
    live_contract = build_branch_contract(engine.topology())
    mismatch = compare_contracts(recorded_contract, live_contract)
    if mismatch is not None:
        raise BranchError(mismatch, status_code=409)

    # ---- hand to the engine: it validates every weight/input, then resets + installs ----
    provenance = {
        "run_id": (payload.get("source") or {}).get("run_id"),
        "experiment": (payload.get("source") or {}).get("experiment"),
        "seed": recorded_contract.get("seed"),
        "frame_index": payload.get("frame_index"),
        "timestep": payload.get("timestep"),
        "precision": payload.get("precision"),
    }
    try:
        result = engine.branch_from_weights(
            dict(weights),
            input_vector=list(inp) if inp is not None else None,
            restore_input=bool(payload.get("restore_input", True)),
            provenance=provenance,
        )
    except ValueError as e:
        raise BranchError(str(e))          # snapshot/input invalid: engine left unchanged

    return {
        "branched": True,
        "branch_schema": BRANCH_SCHEMA_NAME,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "precision": payload.get("precision"),
        "use_checkpoint": bool(payload.get("use_checkpoint", False)),
        **result,
    }
