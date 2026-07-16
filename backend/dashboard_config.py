"""The dashboard preset and the controls exposed in the UI.

This model has a richer surface than the earlier hard-wipe model: predictive
inhibition deliberately separates several timescales (activity-trace decay,
inhibitory *association* rate, inhibitory conductance *expression* magnitude, and
inhibitory conductance decay), and the spec requires these to be inspectable and
independently controllable, plus two ablation toggles (predictive conductance and
inhibitory plasticity). The shared E threshold (1000), the inhibitory reversal
(shunting, E_inh=0), and per-projection initialization stay structural.
"""

DASHBOARD_OVERRIDES: dict = {}


CONFIG_SPEC = [
    {"key": "leak_rate", "label": "Leak rate", "kind": "range",
     "min": 0.0, "max": 0.5, "step": 0.005,
     "desc": "Per-step membrane decay for every excitatory neuron, mapped to a "
             "baseline leak conductance g_L = -ln(1 - leak_rate). With no inhibition "
             "and no input, integration reduces exactly to V <- (1 - leak_rate) * V."},
    {"key": "refractory_steps", "label": "Refractory steps", "kind": "range",
     "min": 0, "max": 5, "step": 1,
     "desc": "Steps after firing during which an excitatory neuron cannot fire."},
    {"key": "eta", "label": "Excitatory learning rate", "kind": "range",
     "min": 0.001, "max": 0.05, "step": 0.001,
     "desc": "Shared accumulating-weight (feedforward / coincidence) learning rate."},
    {"key": "e_weight_cap", "label": "Excitatory weight cap", "kind": "range",
     "min": 200, "max": 2000, "step": 50,
     "desc": "The shared per-synapse accumulating-weight cap (theta = 1000)."},
    {"key": "enew_enabled", "label": "L1E_new coincidence topology", "kind": "toggle",
     "desc": "ON: retained 36-neuron L1E_new/L1I coincidence comparison topology. "
             "OFF: the 26-neuron predictive-inhibition (PI) experiment -- eight "
             "pattern-specific PI cells paired 1:1 with L2E, each with 9 locally "
             "plastic inhibitory outputs onto L1E_s. Applying rebuilds the network."},

    # --- membrane conductance / trace ---
    {"key": "alpha_inh", "label": "WTA conductance retention (L2E)", "kind": "range",
     "min": 0.0, "max": 0.98, "step": 0.02,
     "desc": "Per-step retention of inhibitory conductance on L2E (the L2I_WTA "
             "target). Kept FAST so winner-take-all does not itself drive turnover."},
    {"key": "alpha_inh_l1", "label": "Predictive conductance retention (L1)", "kind": "range",
     "min": 0.0, "max": 0.98, "step": 0.02,
     "desc": "Per-step retention of inhibitory conductance on L1E_s (the predictive "
             "PI / legacy L1I target). The symmetry-breaking lever: the shared-feature "
             "shunt must persist across a rival's accumulation window (~0.95)."},
    {"key": "alpha_a", "label": "Activity-trace retention", "kind": "range",
     "min": 0.0, "max": 0.98, "step": 0.02,
     "desc": "Per-step retention of each L1 cell's local activity trace, which lets "
             "a PI cell learn onto features that were active before it fired."},

    # --- predictive inhibition (PI) ---
    {"key": "pi_eta", "label": "PI association rate (slow)", "kind": "range",
     "min": 0.0, "max": 0.2, "step": 0.005,
     "desc": "Local inhibitory learning rate. Kept SLOW so one overlapping "
             "presentation does not let an incumbent PI learn every novel feature."},
    {"key": "pi_g_scale", "label": "PI conductance / weight (fast)", "kind": "range",
     "min": 0.0, "max": 20.0, "step": 0.5,
     "desc": "Inhibitory conductance expressed per unit PI synaptic weight. Expression "
             "is immediate once a mature synapse activates (fast), unlike association."},
    {"key": "l2i_g_scale", "label": "L2I_WTA conductance", "kind": "range",
     "min": 0.0, "max": 30.0, "step": 1.0,
     "desc": "Magnitude of the global winner-take-all inhibitory conductance pulse "
             "onto all L2E (suppresses non-winners on the next boundary)."},
    {"key": "pi_conductance_enabled", "label": "Express PI conductance", "kind": "toggle",
     "desc": "Ablation control: when OFF, PI cells still learn but express no "
             "inhibitory conductance onto L1E_s (predictive inhibition disabled)."},
    {"key": "pi_plasticity_enabled", "label": "PI plasticity", "kind": "toggle",
     "desc": "Ablation control: when OFF, PI output synapses do not learn (frozen at "
             "their initial zero weights)."},
]


def config_values(params):
    """Return current values in the representation expected by the controls."""
    return {item["key"]: params[item["key"]] for item in CONFIG_SPEC}
