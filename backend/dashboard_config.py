"""The active dashboard preset and declarative controls exposed in the UI.

The browser opens on the dual FE/FES rule running on the 9x9 tiled cortical-column hierarchy,
so different 3x3 patches can be driven by different patterns at once and changed independently
while every L1 column keeps learning; the L2 column composes their outputs. It offers the six
built-in topologies (rg_coincidence, tiled_cc, tiled_cc_l1_4, tiled_cc_direct_identity,
tiled_cc_double_eor, rg_direct_cc4). Only controls that affect at least one preset are exposed; every other engine
parameter is a construction-time / headless-only setting.
"""

# The browser opens directly on the tiled_cc hierarchy with the dual FE/FES rule at FAST
# maturation rates (B=5, eta=4.0, c_eta=16.0): each 3x3 patch's L1 column self-organizes an
# owner for whatever pattern is driven into that patch, patches are independent, and the top
# L2 column composes the nine L1 outputs. These rates (vs the historical eta=1.0, c_eta=0.5)
# mature BOTH the L2 feedforward AND the coincidence C fast enough that top-down feedback is
# active within a few thousand live steps. Exact presentation-level alternation is supplied
# by AUTO pacing below, not by a seed-specific period-2 attractor: the historical
# volley-every-boundary regime produced either fragile ``1010`` aliasing or the stable
# loop-latency cadence ``111000``. See docs/FEEDBACK_CADENCE_AND_LOOP_LATENCY.md.
# ``backend.api`` pre-loads a small two-patch composition at startup so
# the effect is visible on Play. This is only the dashboard STARTUP preset; SimulationEngine's
# general-purpose default stays rg_coincidence with the production learning rule (dual_fe_fes
# off), so the engine default and flag-off goldens are unchanged. B=5 is the default dual_fe_B
# (a construction parameter, not a dashboard control). The dual rule itself has no upper cap,
# but the structural theta/2 detector and theta relay ceilings still bind after each update.
# For the minimal single-column demo, select "3x3 Direct CC - 4 E + WTA I".
DASHBOARD_OVERRIDES: dict = {
    "topology": "tiled_cc",
    "dual_fe_fes": True,
    "eta": 4.0,
    # C basal rate chosen so the coincidence cell finishes maturing SHORTLY AFTER the
    # ordinary-E pool, not ~40x later. Measured on the 9x9 tiled surface, two active
    # patches: the E bank reaches its theta/2 ceiling at ~400 boundaries and the C basal
    # reaches 95% of its theta target at ~711 (at the previous 2.0 it took ~3500, and
    # before the column-local distance fix it never got there at all). C must still finish
    # AFTER E -- it confirms an owner the pool has already settled on, so a C that matured
    # first would be confirming an unstable winner.
    "c_eta": 16.0,
    # AUTO: present one volley per RESOLVED causal chain, by matching the graph's own
    # feedback-loop latency (derived structurally, so it re-tracks on every topology
    # change). A real cortical loop resolves far faster than the input changes, so one
    # presentation per resolved chain is the physically meaningful regime; the historical
    # input_period=1 keeps ~L presentations overlapping in flight, which is an artifact of
    # the unit-delay discretization and makes the feedback cadence depend on wiring depth.
    "input_period": 0,
    "leak_rate": 0.0,
    "refractory_steps": 0,
    # Per-synapse ceiling at theta/2 on pattern-detector feedforward weights: no single
    # afferent can drive a competitor to threshold, so each becomes a true integrator that
    # needs >= 2 evidence volleys to fire. NB with leak_rate=0 the two volleys may come from
    # one source over two boundaries; raise leak to demand coincidence.
    #
    # Eor and the coincidence C basal do NOT follow the theta/2 rule -- each fires on a
    # SINGLE afferent (Eor relays whichever ordinary E won its column; a C deposit is its lone
    # basal event), so theta/2 would leave them unable to fire at all. They are bounded at
    # theta instead by the engine default ``relay_weight_cap_frac=1.0``: one afferent reaches
    # the threshold exactly and can never overshoot it.
    "e_weight_cap_frac": 0.5,
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
                 {"value": "tiled_cc_direct_identity",
                  "label": "Tiled CC Direct Identity · 9×9 · 8 E/column"},
                 {"value": "tiled_cc_double_eor",
                  "label": "Tiled CC Double-Eor · 9×9 (latency probe)"},
                 {"value": "rg_direct_cc4", "label": "3×3 Direct CC · 4 E + WTA I"},
                 {"value": "two_tower_composition",
                  "label": "Two-Tower Composition · 9×18 · 18 L1 / 2 L2 / 1 L3"}],
     "desc": "rg_coincidence: the 3x3 coincidence circuit -- pretrained RG->L1E, "
             "coincidence L1C cells (one learned basal + eight unweighted apical), "
             "immediate hard-reset L1I/L2I relays, and an emergent first-spike-latency L2 "
             "WTA. tiled_cc: the 9x9 tiled cortical-column hierarchy -- a 9x9 RGC surface "
             "tiled into nine 3x3 patches, one L1 column per patch (eight ordinary E + Eor "
             "+ coincidence C + relay I each) arranged 3x3, and one L2 column receiving all "
             "nine L1 Eor outputs (191 nodes / 1052 edges). tiled_cc_l1_4: the same "
             "hierarchy with a shallower L1 (four ordinary E per L1 column, eight in L2) -- "
             "155 nodes / 620 edges. tiled_cc_direct_identity: the same hierarchy with NO "
             "Eor relay -- every ordinary E projects directly to every parent ordinary E, so "
             "the parent owns a distinct plastic weight per (child column, child winner) "
             "source address instead of one pooled 'this column fired' event, and each "
             "column's C owns one learned basal per local E (181 nodes / 1546 edges). "
             "Selecting a tiled preset rebuilds the "
             "input to 81 pixels. rg_direct_cc4: the direct 3x3 experimental column -- a "
             "3x3 RGC surface densely feeding four ordinary latency-E competitors, each "
             "driving one central WTA I that hard-resets the four E; no feature relay, "
             "coincidence C, feature-specific I, Eor, or hierarchical feedback (14 nodes / "
             "44 edges). two_tower_composition: TWO 9x9 towers side by side on one 9x18 "
             "input sheet (18 L1 columns, nine per tower), each tower feeding its own L2 "
             "column, and BOTH L2 columns feeding one L3 composition column through the "
             "same generic child->parent rule -- 393 nodes / 2162 edges, three column "
             "layers. Selecting it rebuilds the input to 162 pixels and the patch grid to "
             "3x6. Because each tower L2 now has a parent, its C is no longer dormant; L3's "
             "C is. Note the structural result it was built to test: L3 sees only TWO "
             "source identities (the two tower Eors), so distinct glyphs that differ below "
             "L2 arrive identical at L3 -- see docs/TWO_TOWER_COMPOSITION.md. "
             "Applying rebuilds the network and wipes learned state. "
             "(Use the Topology Editor for arbitrary graphs and saved presets.)"},
    {"key": "dual_fe_fes", "label": "Dual FE/FES learning (experimental)", "kind": "toggle",
     "desc": "Experimental self-regulating learning rule. When ON, BOTH ordinary/latency-E "
             "feedforward learning (LR=excitatory rate) AND coincidence basal learning "
             "(LR=C rate) switch to the inverse-quadratic dual node/synapse free-energy rule: "
             "FE = e + (1-e)/(1 + B((Iaccq/theta)-0.5)^2) shared by the neuron, "
             "FES = wte + (1-wte)/(1 + B((2w/theta)-0.5)^2) per synapse, dw = LR·FE·FES·signal·"
             "influence, floor wte and no upper cap in the RULE itself. Plastic weights "
             "reinitialize at the FES middle theta/4. The structural per-synapse ceilings "
             "still apply on top: theta/2 for pattern detectors, theta for the one-afferent "
             "Eor and C basal. Reference e=wte=0.001, B=5. Applying rebuilds the network and "
             "WIPES all learned state."},
    {"key": "c_feedback_reset", "label": "Top-down feedback reset (halving)", "kind": "toggle",
     "desc": "Tiled cortical columns only. When ON, a column's coincidence C cell -- once its "
             "pattern is confirmed at the parent level -- drives the column's own I to schedule "
             "a DELAY-1 hard reset of the ordinary-E bank, applied at the start of the next "
             "boundary (after drive is frozen, before the event loop). This suppresses the "
             "trained instant integrator's redundant re-fire, dropping the confirmed column "
             "toward the frequency-halving cadence. The zero-latency WTA E->I reset is "
             "untouched and no persistent inhibitory conductance is used (hard-reset only). "
             "OFF reverts C->I to the guarded same-boundary no-op. Applying rebuilds the "
             "network and wipes learned state."},
    {"key": "input_period", "label": "Input period (0 = auto, one per resolved chain)",
     "kind": "range", "min": 0, "max": 8, "step": 1,
     "desc": "Boundaries between RGC volleys. **0 = AUTO (default)**: match the graph's own "
             "feedback-loop latency L, derived structurally, so exactly ONE presentation is "
             "in flight at a time and the pacing re-tracks automatically on every topology "
             "change (L is 2 for tiled_cc_direct_identity, 3 for tiled_cc / tiled_cc_l1_4, "
             "4 for tiled_cc_double_eor). Each volley's confirmation then lands exactly on "
             "its successor's drive packet and cancels it, so presentations alternate "
             "fire/silent EXACTLY -- measured strict 1010 on 12/12 seeds at all three loop "
             "depths. A real cortical loop resolves far faster than the input changes, so "
             "one presentation per resolved chain is the physically meaningful regime. "
             "1 (the historical value) presents a volley EVERY boundary, leaving ~L "
             "presentations overlapping in flight; the confirmed column then runs L-on/L-off "
             "(period 2L) and the halving survives only where L %% input_period == 0, so the "
             "cadence reflects wiring depth rather than certainty. Setting a value by hand "
             "does NOT re-track on a topology change. Unlike every other control this one is "
             "runtime-only: applying it does not rebuild the network or wipe learned state, "
             "so a trained column can be re-paced live."},
    {"key": "leak_rate", "label": "Leak rate", "kind": "range",
     "min": 0.0, "max": 0.5, "step": 0.005,
     "desc": "Per-step membrane decay for every excitatory neuron, mapped to a "
             "baseline leak conductance g_L = -ln(1 - leak_rate). With no inhibition "
             "and no input, integration reduces exactly to V <- (1 - leak_rate) * V."},
    {"key": "refractory_steps", "label": "Refractory steps", "kind": "range",
     "min": 0, "max": 5, "step": 1,
     "desc": "Steps after firing during which an excitatory neuron cannot fire."},
    {"key": "eta", "label": "Excitatory learning rate", "kind": "range",
     "min": 0.001, "max": 5.0, "step": 0.001,
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
     "min": 0.0005, "max": 25.0, "step": 0.0005, "decimals": 4,
     "desc": "Coincidence-cell basal learning rate. Under the one-shot budget rule a "
             "16-seed sweep gives 16/16 turnover/recovery at 0.005. With the DUAL FE/FES "
             "rule ON, the single C basal weight uses the coincidence-cell references (FE "
             "peaks at Iaccq=theta, FES at w=theta/2) and must itself climb to ~theta for "
             "one-shot recognition; it starts at theta/4. The dashboard default 16.0 makes "
             "it finish maturing shortly AFTER the ordinary-E pool (~711 vs ~400 boundaries "
             "on the two-patch 9x9 surface) instead of ~40x later; at that rate the trained "
             "tiled_cc column reaches an EXACT fire/silent 1010 alternation. It is capped AT "
             "theta (its one-shot target), so a matured C deposits exactly threshold and no "
             "more. The C's firing/attention effect is still gated by "
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
