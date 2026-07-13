"""NeuronConfig -- structured neuron configuration (REFACTOR_PLAN.md, Phase 3d).

SimulationEngine is the SOURCE OF TRUTH for configs and defaults: its __init__
signature defines the defaults, `self.params` is the resolved config, and the
dashboard reads its values back from there. This object does NOT introduce a
competing set of defaults -- it is a *transport* built FROM the engine's params
(`from_engine_params`) and applied to each neuron, replacing the scattered
`n.attr = p['key']` assignments in `_build` with one authoritative, population-aware
application. If the engine gains/renames a knob, this dataclass is the one place the
per-neuron wiring changes.

Scope (Phase 3d): the delivery / flow-rate / inhibitory-gate-rule / distance block
that `_build` applies UNIFORMLY to every neuron. The per-population LEARNING config
(L2E feedforward vs L2I/L1I E->I, caps, thresholds) stays inline in `_build` for now
-- it is genuinely population-specific -- and can fold in later without changing this
contract.
"""

from dataclasses import dataclass, fields


# Param keys this config mirrors 1:1 from the engine (the source of truth).
_ENGINE_KEYS = (
    "excitatory_flow_rate", "l2i_excitatory_flow_rate",
    "exc_trace_decay", "exc_trace_normalized",
    "inhibitory_flow_rate", "inh_trace_decay", "inh_trace_normalized",
    "inhibitory_delta_rule", "inhibitory_rule_mode", "inhibitory_eta_up",
    "inhibitory_eta_down", "inhibitory_p_max", "inhibitory_margin_frac",
    "inhibitory_delta_eta", "distance_weighting", "distance_power",
    "distance_ref", "distance_min", "l1i_immediate_relay",
)


@dataclass(frozen=True)
class NeuronConfig:
    excitatory_flow_rate: bool
    l2i_excitatory_flow_rate: bool | None
    exc_trace_decay: float
    exc_trace_normalized: bool
    inhibitory_flow_rate: bool
    inh_trace_decay: float
    inh_trace_normalized: bool
    inhibitory_delta_rule: bool
    inhibitory_rule_mode: str
    inhibitory_eta_up: float
    inhibitory_eta_down: float
    inhibitory_p_max: float
    inhibitory_margin_frac: float
    inhibitory_delta_eta: float
    distance_weighting: bool
    distance_power: float
    distance_ref: float
    distance_min: float
    l1i_immediate_relay: bool

    @classmethod
    def from_engine_params(cls, params):
        """Build from the engine's resolved params dict -- the engine is the single
        source of truth; this only reads, it defines no defaults of its own."""
        return cls(**{k: params[k] for k in _ENGINE_KEYS})

    def apply_to(self, neuron, *, is_l1e, is_l1i, is_l2i=False):
        """Apply the uniform per-neuron config, population-aware, exactly as the old
        inline `_build` block did.

        - Excitatory flow-rate is off for L1E (abstract sensory source) and for an
          L1I acting as an immediate relay (it bypasses trace integration).
        - Inhibitory flow-rate applies wherever a neuron RECEIVES inhibition
          (everything except L1E).
        - The inhibitory-gate rule applies uniformly (frozen/never-inhibited neurons
          are inert under it, so uniform setting is safe).
        - Distance weighting applies ONLY to L2E. L2E is the only population with real
          per-synapse source->target geometry (its distance array is set from the 3D
          layout); every other neuron keeps distance = 1.0, and with distance_ref != 1
          a uniform flag would scale their delivered charge by (distance_ref)^power
          (e.g. ~55x at ref=7.472) -- which silently amplifies L1E pixel drive and
          breaks L1I->L1E inhibition. So gate it to L2E.
        """
        is_l2e = not (is_l1e or is_l1i or is_l2i)
        if is_l1e:
            neuron.excitatory_flow_rate = False
        elif is_l1i:
            neuron.excitatory_flow_rate = self.excitatory_flow_rate and not self.l1i_immediate_relay
        elif is_l2i and self.l2i_excitatory_flow_rate is not None:
            # Per-L2I override: instant charge delivery (False) lets a trained
            # L2E->L2I synapse fire L2I from a single spike (single-source relay),
            # which flow makes impossible (see the engine's l2i_excitatory_flow_rate).
            neuron.excitatory_flow_rate = self.l2i_excitatory_flow_rate
        else:                                    # L2E (and L2I when no override)
            neuron.excitatory_flow_rate = self.excitatory_flow_rate
        neuron.exc_trace_decay = self.exc_trace_decay
        neuron.exc_trace_normalized = self.exc_trace_normalized
        neuron.exc_trace = 0.0
        neuron.exc_trace_last_t = 0

        neuron.inhibitory_flow_rate = self.inhibitory_flow_rate and not is_l1e
        neuron.inh_trace_decay = self.inh_trace_decay
        neuron.inh_trace_normalized = self.inh_trace_normalized
        neuron.inh_trace = 0.0

        neuron.inhibitory_delta_rule = self.inhibitory_delta_rule
        neuron.inhibitory_rule_mode = self.inhibitory_rule_mode
        neuron.inhibitory_eta_up = self.inhibitory_eta_up
        neuron.inhibitory_eta_down = self.inhibitory_eta_down
        neuron.inhibitory_p_max = self.inhibitory_p_max
        neuron.inhibitory_margin_frac = self.inhibitory_margin_frac
        neuron.inhibitory_delta_eta = self.inhibitory_delta_eta

        neuron.distance_weighting = self.distance_weighting and is_l2e
        neuron.distance_power = self.distance_power
        neuron.distance_ref = self.distance_ref
        neuron.distance_min = self.distance_min


# Sanity: dataclass fields and the engine-key list must stay aligned.
assert tuple(f.name for f in fields(NeuronConfig)) == _ENGINE_KEYS
