"""The dashboard preset and the small set of controls exposed in the UI.

The scientific model has one path, so the configuration surface is intentionally
tiny: leak and refractory (explicitly requested as sliders), plus the shared
learning rate and shared weight cap. Everything else -- topology sizes, the shared
E threshold (fixed at 1000), the inhibitory threshold (theta/3), subtractive gate
magnitudes, and per-projection initialization -- is derived or structural and is
not browser-configurable.
"""

# The active experiment runs on plain engine defaults (leak 0, refractory 0). The
# seed is supplied separately by backend.api from persisted state.
DASHBOARD_OVERRIDES: dict = {}


CONFIG_SPEC = [
    {"key": "leak_rate", "label": "Leak rate", "kind": "range",
     "min": 0.0, "max": 0.5, "step": 0.005,
     "desc": "Per-step membrane decay for every excitatory neuron. Default 0 exposes "
             "the pure-integrator baseline; a nonzero value makes integration "
             "rate-sensitive (see the frequency experiment)."},
    {"key": "refractory_steps", "label": "Refractory steps", "kind": "range",
     "min": 0, "max": 5, "step": 1,
     "desc": "Steps after firing during which an excitatory neuron cannot fire. "
             "Default 0."},
    {"key": "eta", "label": "Learning rate (eta)", "kind": "range",
     "min": 0.001, "max": 0.05, "step": 0.001,
     "desc": "Shared accumulating-weight learning rate."},
    {"key": "e_weight_cap", "label": "Shared weight cap", "kind": "range",
     "min": 200, "max": 2000, "step": 50,
     "desc": "The one shared per-synapse accumulating-weight cap (theta = 1000)."},
]


def config_values(params):
    """Return current values in the representation expected by the controls."""
    return {item["key"]: params[item["key"]] for item in CONFIG_SPEC}
