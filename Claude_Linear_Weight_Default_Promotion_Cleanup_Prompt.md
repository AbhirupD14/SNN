# Claude Prompt: Linear Weight Default Promotion and Cleanup

Finish the linear-weight ablation package, clean the remaining documentation and
composition-probe issues, and promote the validated ordinary E/L2E equation to the
production default.

Read these first:

- `Claude_Linear_Weight_Update_Ablation_Plan.md`
- `docs/LINEAR_WEIGHT_ABLATION_REPORT.md`
- `docs/LINEAR_WEIGHT_ABLATION_STATUS.md`
- `experiments/linear_ablation.py`
- `snn/neurons.py`
- `backend/simulation.py`

Inspect the complete working-tree diff before editing. Work in phases and run focused
tests after each phase.

## Approved production decision

- Ordinary accumulating E/L2E learning becomes linear-bounded.
- Keep the ordinary E hard cap.
- Keep the E weight floor at zero.
- Keep C learning quadratic-bounded.
- Keep the C temporal cap.
- Keep FE enabled for both E and C.
- Do not change learning rates, thresholds, initialization, WTA, inhibition, event
  ordering, topology, leak, refractory behavior, or presentation durations.

The new production E equation is:

```text
FE = theta - sum_j(w_j)
s_i = +1 if afferent i participated, otherwise -1
dw_i = eta * FE * s_i * influence_i
w_i <- clip(w_i + dw_i, 0, w_max)
```

The removed factor is:

```text
1 - (w_i / w_max)^2
```

This factor must no longer appear in the default ordinary E/L2E update.

The C equation remains unchanged:

```text
FE_C = theta - w_basal
dw_basal = eta_C * FE_C * apical
           * (1 - (w_basal / w_C_max)^2)
           * s_basal * influence_basal
w_basal <- clip(w_basal + dw_basal, 0, w_C_max)
```

## Phase 1 — Clean the status documentation

`docs/LINEAR_WEIGHT_ABLATION_STATUS.md` currently contains revised conclusions followed
by stale historical conclusions that contradict them.

Remove or clearly segregate the superseded material. The final status document must have
one authoritative conclusion:

- Linear-bounded ordinary E/L2E: validated on 32/32 fresh seeds and selected as the new
  default.
- E cap: retained; seed 2005 demonstrated that it can be active under linear E.
- C quadratic term: retained because removing it degraded the novelty window.
- C temporal cap: retained because cap-free C's temporal margin trends toward zero,
  although no explicit invariant failure occurred in 20,000 events.
- Composition: learned-weight marginality and membrane recency both contribute; the
  experiment does not by itself prove that k-WTA is required.

Remove stale claims that:

- the E cap is removable or inert;
- cap-free C produced an observed invariant failure;
- the C safety margin was already literally crossed;
- composition has already proven that k-WTA is required;
- production defaults were not changed.

After this task, production defaults are intentionally changing, so all documentation
must state that clearly.

## Phase 2 — Tighten the controlled composition probe

Repair the controlled-state probe in `experiments/linear_ablation.py`.

Requirements:

1. Explicitly disable learning during both diagnostic composition probes, or at minimum
   during the controlled probe.
2. Freeze every plastic object relevant to `rg_coincidence`, including:
   - ordinary E/L2E accumulating weights;
   - C basal weights;
   - any other plastic state that could change during probe execution.
3. Snapshot learned weights before the probe and assert that they are byte-identical
   afterward.
4. For the controlled-state probe:
   - reset/equalize L2E membrane voltage and inhibitory conductance;
   - preserve learned weights;
   - capture the first boundary on which the plus-sign L2 drive is delivered;
   - do not continue to the last driven boundary merely because no neuron crosses.
5. For the carried-state probe:
   - preserve the real membrane state;
   - capture the first plus-driven boundary before the first winner's reset.
6. Keep WTA unchanged and do not permit diagnostic co-firing.
7. Rename the reconstructed field `v_after_drive`.
   - It is currently calculated as `V + frozen_excitation`, which is not an actual
     captured voltage and can ignore inhibitory conductance.
   - Use an accurate name such as `projected_uninhibited_end_v`, or remove it.
8. Continue recording:
   - boundary-start V;
   - frozen excitation;
   - active-input weight sum;
   - inhibitory conductance;
   - refractory state;
   - crossing time;
   - finite status.
9. Add tests proving:
   - the first driven boundary is selected;
   - no learned weights change during either probe;
   - controlled-state membrane equalization is applied;
   - instrumentation remains behaviorally inert when disabled.

Regenerate `phase4_composition.json` and update the report only if the corrected probe
changes any result.

## Phase 3 — Promote linear-bounded E to the production default

Change the ordinary E/L2E production default from `quadratic_bounded` to
`linear_bounded`.

Update all relevant default layers consistently:

- `ExcitatoryNeuron` constructor default;
- `SimulationEngine.DEFAULTS["e_weight_update_mode"]`;
- helper constructors and topology rebuild paths;
- serialization/config fallback behavior;
- tests that assert production defaults;
- comments and docstrings describing the default rule;
- exported or public metadata that names the default.

Do not add an update-mode dashboard control. The dashboard should simply inherit the new
production engine default.

Keep these headless modes available for experiments and regression comparisons:

- `quadratic_bounded` — historical E rule;
- `linear_bounded` — new production E rule;
- `linear_nonnegative` — cap-free diagnostic only.

Do not rename historical result conditions:

- Condition A remains the historical quadratic-E baseline.
- Condition B remains linear-bounded E and now represents the promoted production rule.

Update their descriptions so this distinction is unmistakable.

## Phase 4 — Update equations and documentation everywhere

Search the repository for all descriptions of the ordinary accumulating E equation,
including occurrences of:

- `1 - (w/w_max)^2`
- `1 − (w/w_max)²`
- `quadratic_bounded`
- "production rule"
- "default rule"
- E/L2E learning equations

Update every current production description to the new linear-bounded equation.

Preserve the nonlinear term in places that describe:

- the historical quadratic E ablation mode;
- the C basal learning equation;
- archived experimental results.

Clearly distinguish:

- new production E equation;
- historical quadratic E mode;
- unchanged quadratic C equation.

Revise the technical specification and ablation documents so they state:

- why linear E was selected;
- that it passed 32/32 fresh-seed gates;
- that the E cap remains necessary as a safety bound;
- that C was intentionally not changed;
- the exact commit-level production decision.

## Phase 5 — Baselines and regression tests

Because the production E default is intentionally changing, do not preserve obsolete
default fingerprints by silently forcing the historical mode.

Instead:

1. Keep an explicit historical-quadratic regression test using
   `e_weight_update_mode="quadratic_bounded"`.
2. Add an explicit new-production regression test using the default mode and prove it
   matches explicitly requested `linear_bounded`.
3. Prove default E updates use exactly:

   ```text
   dw_i = eta * (theta - sum(w)) * s_i * influence_i
   ```

   followed by clipping to `[0, w_max]`.
4. Prove the default E update contains no quadratic multiplier.
5. Prove the default E cap still clips correctly.
6. Prove the default C equation remains quadratic-bounded and byte-compatible with its
   previous production behavior.
7. Prove reset, reseed and custom-topology rebuilds retain the selected mode.
8. Prove the dashboard control surface remains unchanged.
9. Regenerate any baseline fingerprints whose meaning is "current production default."
10. Do not rewrite unrelated goldens unless the intentional default equation genuinely
    changes their learned dynamics.

Run:

- focused neuron equation tests;
- coincidence-cell tests;
- ablation tests;
- dashboard/config tests;
- serialization tests;
- topology and event-scheduler tests;
- the complete test suite;
- `git diff --check`.

## Phase 6 — Final report

Update:

- `docs/LINEAR_WEIGHT_ABLATION_REPORT.md`
- `docs/LINEAR_WEIGHT_ABLATION_STATUS.md`
- any technical specification containing the old production E equation
- affected experiment JSON and reproduction commands

The final report must clearly separate:

- historical quadratic-E baseline A;
- validated linear-E candidate B;
- the newly promoted production default;
- unchanged C behavior;
- retained E and C caps;
- experimental modes that remain headless.

Do not claim "no production default changed." State that the ordinary E/L2E learning
equation was intentionally changed after the 32-seed confirmation.

At completion, provide:

- the exact old and new E equations;
- the unchanged C equation;
- every changed file;
- test totals and commands;
- regenerated experiment artifacts;
- confirmation that the dashboard surface did not change;
- confirmation that E cap, C quadratic behavior and C cap remain active;
- any remaining uncertainty about composition.

Do not commit or publish unless separately instructed.
