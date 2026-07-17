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
    {"key": "topology", "label": "Topology", "kind": "select",
     "options": [{"value": "pi", "label": "Predictive inhibition (PI)"},
                 {"value": "old", "label": "Old dense global inhibition"},
                 {"value": "rg", "label": "Retinal ganglion source layer (RG)"},
                 {"value": "rg_residual", "label": "RG residual/error pathway"}],
     "desc": "pi: the 26-neuron predictive-inhibition experiment -- eight "
             "pattern-specific PI cells paired 1:1 with L2E, each with 9 locally "
             "plastic inhibitory outputs onto L1E_s. old: the 27-neuron original "
             "topology -- nine paired L1I relays, densely fed by every L2E, so the "
             "winner shunts every L1E_s (global inhibition). rg: the 36-neuron source-"
             "layer experiment -- old's cortex with nine uninhibitable RG cells ahead "
             "of L1 and plastic 1:1 RG->L1E synapses, so a held edge keeps supplying "
             "retinal evidence while L1 is shunted and L1E must LEARN its afferent. "
             "rg_residual: the 52-cell residual circuit -- L1E remains uninhibited, "
             "PI predicts onto a separate ErrorE sheet, and local traced SwitchI "
             "cells release only the incumbent before ordinary L2 WTA re-competes. "
             "Applying rebuilds the network. (Use the Topology Editor for arbitrary "
             "graphs and presets.)"},

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

    # --- plastic encoder feedforward (the RG -> L1E path; 'rg' topology only) ---
    {"key": "enc_plasticity_enabled", "label": "RG->L1E plasticity", "kind": "toggle",
     "desc": "Ablation control (rg topology): when OFF, the nine RG->L1E synapses are "
             "frozen at their initial weight, isolating the RG layer's topology and "
             "extra delay from the effect of L1 learning its sensory afferent."},
    {"key": "enc_init_jitter", "label": "RG->L1E init jitter", "kind": "toggle",
     "desc": "Control (rg topology): when OFF, all nine RG->L1E weights start at "
             "exactly the same value instead of the shared seeded narrow jitter. Tests "
             "whether L1 phase splitting is learned/dynamic or merely injected by "
             "initialization asymmetry."},

    # --- residual/error circuit ---
    {"key": "residual_exc_scale", "label": "L1E->ErrorE copy / threshold", "kind": "range",
     "min": 0.5, "max": 2.0, "step": 0.05,
     "desc": "Fixed charge delivered to paired ErrorE per L1E spike, as a multiple "
             "of the excitatory threshold. This path is structural, not plastic."},
    {"key": "switch_trace_decay", "label": "Switch winner-trace retention", "kind": "range",
     "min": 0.0, "max": 0.99, "step": 0.01,
     "desc": "Per-boundary retention rho of each SwitchI cell's strictly local "
             "eligibility trace x_j. Only a real paired L2E_j spike writes it."},
    {"key": "switch_trace_threshold", "label": "Switch trace threshold", "kind": "range",
     "min": 0.0, "max": 1.0, "step": 0.05,
     "desc": "Minimum pre-existing local x_j required alongside a current residual "
             "event for SwitchI_j to fire."},
    {"key": "switch_residual_charge_frac", "label": "Switch residual charge / θI", "kind": "range",
     "min": 0.05, "max": 0.9, "step": 0.05,
     "desc": "Numeric charge added to every connected SwitchI by one ErrorE spike, "
             "as a fraction of the inhibitory threshold. The branch is capped below "
             "threshold, so residual activity alone cannot fire SwitchI."},
    {"key": "switch_trace_charge_frac", "label": "Switch trace priming / θI", "kind": "range",
     "min": 0.05, "max": 0.9, "step": 0.05,
     "desc": "Maximum numeric priming charge opened by the paired local L2 trace. "
             "This branch is also capped below threshold; only residual plus priming "
             "can cross θI."},
    {"key": "switch_g_scale", "label": "Switch incumbent conductance", "kind": "range",
     "min": 0.0, "max": 30.0, "step": 1.0,
     "desc": "Paired inhibitory conductance scheduled onto the incumbent L2E when "
             "its SwitchI temporal AND fires."},
    {"key": "switch_conductance_enabled", "label": "Express SwitchI conductance", "kind": "toggle",
     "desc": "Ablation: SwitchI coincidence events and local traces remain visible, "
             "but no incumbent-inhibitory pulse is emitted."},
]


def config_values(params):
    """Return current values in the representation expected by the controls."""
    return {item["key"]: params[item["key"]] for item in CONFIG_SPEC}
