"""Experimental NEST/NESTML event-driven backend for the canonical `tiled_cc` topology.

This package is a CHARACTERISATION PROTOTYPE, not a replacement for
`backend.simulation.SimulationEngine`. The validated Python engine remains the oracle and
is left completely unchanged; nothing here is imported by it.

Scope is deliberately narrow (see `prompts/Claude_NEST_Event_Driven_3x3_Engine_Prompt.md`):
only `backend.network_spec.tiled_cc_spec(cc_e_count=8)` is translated, and only under the
scoped configuration `leak_rate = 0` with no persistent inhibitory conductance -- the
condition that makes an event-only accumulator an EXACT port rather than an approximation.

Importing this package does not require NEST. The submodules that do
(`build_models`, `engine`, `recording`) import it lazily so that `topology` can be
exercised -- and its structural output checked against the canonical `NetworkSpec` -- from
the ordinary repository interpreter.
"""

__all__ = ["build_models", "engine", "recording", "topology"]
