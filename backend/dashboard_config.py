"""The one dashboard preset and the small set of controls exposed in the UI.

SimulationEngine owns general-purpose defaults.  This file describes only the
specific experiment launched by ``backend.api``.  Keeping this separate makes it
possible to read the active model without searching through HTTP endpoint code.
"""

DASHBOARD_OVERRIDES = {
    "refractory": 1,
    "l2e_lr_frac": 0.02,
    "l2i_threshold_frac": 1 / 3,
    "ei_sat_mult": 4.0,
    "confidence_consolidation": False,
    "competitive_weight_update": "redistribution",
    "signed_depression": False,
    "l2_charge_chunks": 20,
    "excitatory_flow_rate": False,
    "l2i_excitatory_flow_rate": False,
    "assembly_decay_frac": 0.0,
    "inhibitory_delta_rule": False,
    "distance_weighting": True,
    "l2i_hard_reset_losers": True,
    "structural_free_energy": True,
    "l2e_weight_cap_frac": 1 / 3,
    "pos_weight_floor": 1,
}


# Only controls that affect the active dashboard model belong here. Experimental
# mechanisms remain callable from Python but no longer crowd the browser.
CONFIG_SPEC = [
    {"key": "l2_charge_chunks", "label": "Charge chunks", "kind": "range",
     "min": 1, "max": 32, "step": 1,
     "desc": "Substeps used for the frozen-timestep L2 winner race."},
    {"key": "competitive_weight_update", "label": "Loser update", "kind": "select",
     "options": [
         {"value": "redistribution", "label": "Redistribute into OFF gates"},
         {"value": "depression", "label": "Depress active gates"},
         {"value": "none", "label": "Reset only"},
     ],
     "desc": "Weight rule applied to non-refractory L2 losers after competitive reset."},
    {"key": "distance_weighting", "label": "Distance attenuation", "kind": "toggle",
     "desc": "Scale delivered L1E→L2E charge by the normalized inverse-square distance."},
    {"key": "distance_power", "label": "Distance power", "kind": "range",
     "min": 0.0, "max": 4.0, "step": 0.5,
     "desc": "Exponent used by the distance-delivery factor."},
    {"key": "layout_scatter_enabled", "label": "Scattered layout", "kind": "toggle",
     "desc": "Use seeded irregular neuron positions instead of the regular reference layout."},
    {"key": "event_driven", "label": "Resolve every step", "kind": "toggle",
     "desc": "Run L2 competition each timestep instead of once per cycle."},
    {"key": "refractory", "label": "Refractory steps", "kind": "range",
     "min": 0, "max": 3, "step": 1,
     "desc": "A value of 1 protects the current winner from the loser update."},
    {"key": "l2e_lr_frac", "label": "L2E learning-rate fraction", "kind": "range",
     "min": 0.005, "max": 0.1, "step": 0.005,
     "desc": "L2E learning rate as a fraction of its per-afferent weight cap."},
    {"key": "l2e_init_mode", "label": "Balanced initialization", "kind": "toggle",
     "desc": "Use Sinkhorn-balanced L2E weights instead of the wide random initialization."},
    {"key": "l2e_init_jitter", "label": "Balanced-init jitter", "kind": "range",
     "min": 0.0, "max": 0.2, "step": 0.005,
     "desc": "Seed variation used before balancing; only active in balanced mode."},
    {"key": "l1i_immediate_relay", "label": "Immediate L1I relay", "kind": "toggle",
     "desc": "Relay any L2E feedback immediately instead of accumulating to threshold."},
    {"key": "leak_enabled", "label": "L2E leak", "kind": "toggle",
     "desc": "Enable membrane leak on L2 excitatory neurons."},
    {"key": "leak_l2", "label": "L2E leak rate", "kind": "range",
     "min": 0.001, "max": 0.05, "step": 0.001,
     "desc": "Per-step L2E membrane decay when leak is enabled."},
    {"key": "l2i_leak_enabled", "label": "L2I leak", "kind": "toggle",
     "desc": "Enable membrane leak on the shared L2 inhibitory neuron."},
    {"key": "l1i_leak_enabled", "label": "L1I leak", "kind": "toggle",
     "desc": "Enable membrane leak on trainable L1 inhibitory accumulators."},
]


def config_values(params):
    """Return current values in the representation expected by the controls."""
    values = {item["key"]: params[item["key"]] for item in CONFIG_SPEC}
    values["l2e_init_mode"] = values["l2e_init_mode"] == "balanced"
    return values
