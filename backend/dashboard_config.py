"""The active dashboard preset and declarative controls exposed in the UI.

The browser starts on the validated RG coincidence turnover configuration and offers
exactly the three current built-in topologies (rg_coincidence, tiled_cc, tiled_cc_l1_4).
Only controls that affect at least one of those three presets are exposed; every other
engine parameter is a construction-time / headless-only setting.
"""

# The browser opens directly on the validated coincidence turnover experiment.
# SimulationEngine's general-purpose default is also ``rg_coincidence``; this dictionary is
# the explicit dashboard startup preset. It sets only retained (editable) controls -- no
# per-synapse weight cap (ordinary-E learning is cap-free; the FE budget saturates totals).
DASHBOARD_OVERRIDES: dict = {
    "topology": "rg_coincidence",
    "eta": 0.01,
    "c_eta": 0.005,
    "l2_init_total_frac": 0.95,
    "leak_rate": 0.0,
    "refractory_steps": 0,
}


CONFIG_SPEC = [
    {"key": "topology", "label": "Topology", "kind": "select",
     "options": [{"value": "rg_coincidence", "label": "Coincidence · 3×3"},
                 {"value": "tiled_cc", "label": "Tiled CC · 9×9 · 8 E/column"},
                 {"value": "tiled_cc_l1_4", "label": "Tiled CC · 9×9 · L1 4 E / L2 8 E"},
                 {"value": "tiled_cc_feature_gated",
                  "label": "9×9 Feature-Gated CC (L1=8)"}],
     "desc": "rg_coincidence: the 3x3 coincidence circuit -- pretrained RG->L1E, "
             "coincidence L1C cells (one learned basal + eight unweighted apical), "
             "immediate hard-reset L1I/L2I relays, and an emergent first-spike-latency L2 "
             "WTA. tiled_cc: the 9x9 tiled cortical-column hierarchy -- a 9x9 RGC surface "
             "tiled into nine 3x3 patches, one L1 column per patch (eight ordinary E + Eor "
             "+ coincidence C + relay I each) arranged 3x3, and one L2 column receiving all "
             "nine L1 Eor outputs (191 nodes / 1052 edges). tiled_cc_l1_4: the same "
             "hierarchy with a shallower L1 (four ordinary E per L1 column, eight in L2) -- "
             "155 nodes / 620 edges. tiled_cc_feature_gated: the eight-competitor variant "
             "that restores rg_coincidence's feature-specific inhibition -- nine fixed "
             "feature relays per 3x3 RF, each with a paired coincidence C and feature "
             "inhibitory If that suppresses only its own relay, plus a separate WTA-only I "
             "per L1 module (424 nodes / 1932 edges). Selecting a tiled preset rebuilds the "
             "input to 81 pixels. Applying rebuilds the network and wipes learned state. "
             "(Use the Topology Editor for arbitrary graphs and saved presets.)"},
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
     "desc": "Accumulating feedforward learning rate for ordinary E/L2E/Eor cells. "
             "Learning is cap-free: as an incoming row's total approaches the FE budget "
             "B = maturity_budget_frac*theta the update vanishes, so weights saturate "
             "without any per-synapse ceiling."},
    {"key": "c_eta", "label": "C basal learning rate", "kind": "range",
     "min": 0.0005, "max": 0.01, "step": 0.0005, "decimals": 4,
     "desc": "Coincidence-cell basal learning rate. Under the one-shot budget rule a "
             "16-seed sweep gives 16/16 turnover/recovery at 0.005 while maturing to "
             "one-shot firing ~5x faster than the historical 0.001."},
    {"key": "l2_init_total_frac", "label": "L2 initial total / threshold", "kind": "range",
     "min": 0.5, "max": 0.99, "step": 0.01,
     "desc": "For latency-WTA L2E/Eor cells, normalize each seeded afferent row to this "
             "fraction of theta. 0.95 gives equal positive initial FE=0.05*theta "
             "while preserving within-row jitter that breaks symmetry."},
]


def config_values(params):
    """Return current values in the representation expected by the controls."""
    return {item["key"]: params[item["key"]] for item in CONFIG_SPEC}
