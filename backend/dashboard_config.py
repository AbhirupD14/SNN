"""The active dashboard preset and declarative controls exposed in the UI.

The browser opens on the dual FE/FES rule running on the 9x9 tiled cortical-column hierarchy,
so different 3x3 patches can be driven by different patterns at once and changed independently
while every L1 column keeps learning; the L2 column composes their outputs. It offers the five
built-in topologies (rg_coincidence, tiled_cc, tiled_cc_l1_4, tiled_cc_feature_gated,
rg_direct_cc4). Only controls that affect at least one preset are exposed; every other engine
parameter is a construction-time / headless-only setting.
"""

# The browser opens directly on the tiled_cc hierarchy with the dual FE/FES rule at its
# confirmation parameters (B=5, LR multiplier m=100 -> eta=1.0, c_eta=0.5): each 3x3 patch's
# L1 column self-organizes an owner for whatever pattern is driven into that patch, patches are
# independent, and the top L2 column composes the nine L1 outputs. (The coincidence C cells are
# present but largely inert under this rule -- the per-column WTA runs through the ordinary
# E -> I -> E hard-reset loop.) ``backend.api`` pre-loads a small two-patch composition at
# startup so the effect is visible on Play. This is only the dashboard STARTUP preset;
# SimulationEngine's general-purpose default stays rg_coincidence with the production learning
# rule (dual_fe_fes off), so the engine default, flag-off bit-exactness, and every golden are
# unchanged. B=5 is the default dual_fe_B (a construction parameter, not a dashboard control).
# For the minimal single-column demo, select "3x3 Direct CC - 4 E + WTA I".
DASHBOARD_OVERRIDES: dict = {
    "topology": "tiled_cc",
    "dual_fe_fes": True,
    "eta": 1.0,
    "c_eta": 0.5,
    "leak_rate": 0.0,
    "refractory_steps": 0,
}

# Patches (row, col) -> pattern pre-loaded at server startup so the dashboard opens on a live
# composition (two 3x3 patches running two different patterns). Applied only when the startup
# preset is tiled; a Reset keeps them (the engine carries the per-patch map across rebuilds).
DASHBOARD_STARTUP_PATCH_PATTERNS = [(0, 0, "row 1"), (2, 2, "col 1")]


CONFIG_SPEC = [
    {"key": "topology", "label": "Topology", "kind": "select",
     "options": [{"value": "rg_coincidence", "label": "Coincidence · 3×3"},
                 {"value": "tiled_cc", "label": "Tiled CC · 9×9 · 8 E/column"},
                 {"value": "tiled_cc_l1_4", "label": "Tiled CC · 9×9 · L1 4 E / L2 8 E"},
                 {"value": "tiled_cc_feature_gated",
                  "label": "9×9 Feature-Gated CC (L1=8)"},
                 {"value": "rg_direct_cc4", "label": "3×3 Direct CC · 4 E + WTA I"}],
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
             "input to 81 pixels. rg_direct_cc4: the direct 3x3 experimental column -- a "
             "3x3 RGC surface densely feeding four ordinary latency-E competitors, each "
             "driving one central WTA I that hard-resets the four E; no feature relay, "
             "coincidence C, feature-specific I, Eor, or hierarchical feedback (14 nodes / "
             "44 edges). Applying rebuilds the network and wipes learned state. "
             "(Use the Topology Editor for arbitrary graphs and saved presets.)"},
    {"key": "dual_fe_fes", "label": "Dual FE/FES learning (experimental)", "kind": "toggle",
     "desc": "Experimental self-regulating learning rule. When ON, BOTH ordinary/latency-E "
             "feedforward learning (LR=excitatory rate) AND coincidence basal learning "
             "(LR=C rate) switch to the inverse-quadratic dual node/synapse free-energy rule: "
             "FE = e + (1-e)/(1 + B((Iaccq/theta)-0.5)^2) shared by the neuron, "
             "FES = wte + (1-wte)/(1 + B((2w/theta)-0.5)^2) per synapse, dw = LR·FE·FES·signal·"
             "influence, floor wte and NO upper cap. Plastic weights reinitialize at the FES "
             "middle theta/4. Reference e=wte=0.001, B=5. Applying rebuilds the network and "
             "WIPES all learned state."},
    {"key": "leak_rate", "label": "Leak rate", "kind": "range",
     "min": 0.0, "max": 0.5, "step": 0.005,
     "desc": "Per-step membrane decay for every excitatory neuron, mapped to a "
             "baseline leak conductance g_L = -ln(1 - leak_rate). With no inhibition "
             "and no input, integration reduces exactly to V <- (1 - leak_rate) * V."},
    {"key": "refractory_steps", "label": "Refractory steps", "kind": "range",
     "min": 0, "max": 5, "step": 1,
     "desc": "Steps after firing during which an excitatory neuron cannot fire."},
    {"key": "eta", "label": "Excitatory learning rate", "kind": "range",
     "min": 0.001, "max": 1.0, "step": 0.001,
     "desc": "Accumulating feedforward learning rate for ordinary E/L2E/Eor cells. "
             "Production (linear_fe) learning is cap-free: as an incoming row's total "
             "approaches the FE budget B = maturity_budget_frac*theta the update vanishes, "
             "so weights saturate without any per-synapse ceiling. With the DUAL FE/FES "
             "rule ON, weights start at the FES middle theta/4 and a fresh competitor is "
             "deliberately sub-threshold (it accumulates over two boundaries and only "
             "MATURES into a one-boundary instant integrator as its active weights grow); "
             "at eta=0.01 that maturation is very slow, so raise eta toward ~1.0 to watch a "
             "direct column consolidate and turn over within a live session."},
    {"key": "c_eta", "label": "C basal learning rate", "kind": "range",
     "min": 0.0005, "max": 5.0, "step": 0.0005, "decimals": 4,
     "desc": "Coincidence-cell basal learning rate. Under the one-shot budget rule a "
             "16-seed sweep gives 16/16 turnover/recovery at 0.005. With the DUAL FE/FES "
             "rule ON, the single C basal weight uses the coincidence-cell references (FE "
             "peaks at Iaccq=theta, FES at w=theta/2) and must itself climb to ~theta for "
             "one-shot recognition; it starts at theta/4, so raise c_eta toward ~5 to mature "
             "it within a live session. The C's firing/attention effect is still gated by "
             "how often bottom-up (Eor) meets top-down (L2 apical), so it tightens only as "
             "the whole hierarchy consolidates."},
    {"key": "l2_init_total_frac", "label": "L2 initial total / threshold", "kind": "range",
     "min": 0.5, "max": 0.99, "step": 0.01,
     "desc": "For latency-WTA L2E/Eor cells, normalize each seeded afferent row to this "
             "fraction of theta. 0.95 gives equal positive initial FE=0.05*theta "
             "while preserving within-row jitter that breaks symmetry."},
]


def config_values(params):
    """Return current values in the representation expected by the controls."""
    return {item["key"]: params[item["key"]] for item in CONFIG_SPEC}
