# Claude Frontend Inhibition Diagnosis Prompt

Diagnose this from the current frontend/dashboard state before changing code.

## Observed Behavior

- I trained the network in pattern blocks until some patterns were learned.
- Row 0 and row 1 consolidated cleanly.
- Row 2 is still split between two competing L2E neurons, which is acceptable for now.
- The problem is that the relevant L2I neuron is not accumulating enough charge from those two L2E competitors to reach threshold and fire.
- Some L1I neurons appear dead even though they should be firing in phase with the other L1I neurons.

## What I Want

Use the frontend/dashboard and backend state to explain why this is happening. Do not assume the current implementation is correct. Inspect the actual live values.

## 1. L2E -> L2I Path

- Which L2E neurons are firing for row 2?
- Which L2I neuron should receive from them?
- What are the L2E -> L2I weights for those connections?
- What is the L2I threshold?
- How much charge does that L2I neuron receive per timestep when one or both competing L2E neurons fire?
- Does leak or reset erase too much charge before it can cross threshold?
- Are the two row-2 L2E competitors firing close enough in time to summate in L2I, or are their spikes too far apart?

## 2. L1I Behavior

- Which L1I neurons are dead?
- Which L2E neurons feed those L1I neurons?
- Are their incoming weights initialized too low, clipped, not learning, or disconnected?
- Are the corresponding L2E neurons actually firing often enough to drive them?
- Is the L1I threshold too high relative to incoming L2E -> L1I weights?
- Are dead L1I neurons being reset, inhibited, leaked, or skipped by update ordering?

## 3. Scale Consistency

- Check that charge, thresholds, weights, floors, caps, and initialization ranges are still in the same numeric scale.
- Confirm whether the one-third-of-E-threshold rule is actually applied consistently:
  - positive afferent weight cap = E threshold / 3
  - I neuron threshold = E threshold / 3
  - positive afferent floor = 1
- If an inhibitory neuron needs multiple incoming spikes to fire, explain whether that is intended or a scaling bug.

## 4. Connectivity And Locality

- Verify the local connectivity rules:
  - L2I should only receive from the appropriate local L2E sources.
  - L1I should receive from the intended L2E sources and project back locally.
- Check for indexing errors, row/column mapping mistakes, dead fan-in, or incorrect source-target assignment.
- Report whether the issue is biological/dynamical or simply a wiring/scaling bug.

## 5. Frontend Instrumentation

If the current frontend cannot answer these questions clearly, add minimal debugging views or overlays that show:

- live membrane charge vs threshold for each L2I and L1I neuron
- incoming spike events into each inhibitory neuron
- incoming weight values into each inhibitory neuron
- last-fired timestep for each inhibitory neuron
- per-neuron dead/silent status over a rolling window

Do not add broad new features. Add only the smallest instrumentation needed to explain the observed failure.

## 6. Frontend Coordinate Consistency

There is also a visualization mismatch that needs diagnosis and cleanup:

- When I select or present the leftmost column pattern, the input grid correctly shows the leftmost column.
- The weight heatmaps also show the active/learned structure on the left.
- But the rendered neuron/RF view shows it on the right side.
- This is likely just an axis reversal or coordinate transform mismatch, but it makes debugging confusing.

Please inspect the frontend rendering path and make the coordinate convention consistent across:

- input grid
- rendered neuron/RF view
- heatmaps
- any enlarged chart/pop-up views

Do not change the learned weights or backend pattern definitions just to fix the visual. Treat this as a frontend visualization coordinate issue unless the data proves otherwise.

Report:

- where the flip is introduced
- which coordinate convention you chose
- which files were changed
- how you verified that left/right and top/bottom now match across all views

## 7. Seed Sensitivity And Finite-Time Consolidation

There is also an algorithmic/theoretical issue that needs to be considered while diagnosing the current behavior.

Observation:

- Certain random seeds for weight initialization help cases where we get clean consolidation of patterns that do not share common pixels.
- Given enough timesteps, this may balance out, but we cannot rely on arbitrarily long training.
- For deployment, the network needs a reasonable finite-time consolidation guarantee or at least a measurable bound under stated assumptions.

Please diagnose whether the current learning dynamics are overly dependent on lucky initial weights.

Questions to answer:

- How much does ownership consistency vary across initialization seeds?
- Do row/column patterns with no shared pixels converge faster, slower, or less reliably than patterns with overlap?
- Are some seeds only helping because they create a larger initial margin for one future owner?
- Does inhibition amplify early random margins before the correct receptive field has enough time to form?
- Are there cases where two neurons keep fighting because their initial weights are too similar, their inputs are too symmetric, or inhibition is too weak/late?
- Does the current update rule provide a monotonic local advantage to the neuron that best matches the presented pattern, or can random initialization dominate for too long?
- Under the current thresholds, weight caps, floors, learning rate, and leak, what is the rough expected number of presentations needed for a correct owner to become stable?

What to test:

- Run a small seed sweep with fixed hyperparameters and report time-to-stable-ownership, not just final ownership.
- Track first-owner margin over time: winner charge or score minus runner-up charge or score for each pattern.
- Track whether ownership stabilizes before weights saturate.
- Compare consolidation time for:
  - non-overlapping patterns
  - partially overlapping patterns
  - strongly competing patterns
- Report failures where ownership does not stabilize within a fixed presentation budget.

What not to do:

- Do not solve this by allowing unbounded training time.
- Do not tune seed-by-seed.
- Do not add non-local supervision or global winner assignment.
- Do not hide the issue behind dashboard-only metrics.

The goal is not necessarily a full proof yet, but we need to move toward a finite-time story: what assumptions make consolidation likely, what parameters control convergence time, and what local mechanism could improve the bound if the current mechanism has no reliable one.

## Output

- First, give a diagnosis based on the live frontend/backend values.
- Then list likely causes ranked by evidence.
- Then suggest the smallest code or parameter change to test first.
- Do not tune blindly. Every proposed change should be tied to a measured failure mode.
