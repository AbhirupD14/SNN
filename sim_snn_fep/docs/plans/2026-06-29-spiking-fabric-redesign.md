# Plan — Conductance-Based Spiking Fabric (v2)

Author: review/plan pass 2026-06-29 (Opus 4.8). Supersedes the current `fep_model.py` dynamics
(keeps the LIF→pure-integrator *training idea*). Target: get the **fabric** right — behavior first,
scalable and tileable, future features (sequences, prediction) emerge from scale, not hand-engineered now.

## 0. Goals & non-goals (this iteration)

**Goals (must demonstrate):** competition, learning, self-organization, **composition**, **hierarchy**.
**Means:** spike-event communication; voltage/conductance internal state; STDP (timing-based, unsupervised);
synaptic delays; conductance-based membrane with shunting inhibition; the LIF→pure-integrator transition preserved.
**Hard constraints (unchanged):** every rule neuron-local / synapse-local; **no supervision, no teacher, no global
error/objective, no argmax-as-mechanism**; Dale's principle (a neuron is E or I, never both).
**Non-goals (defer):** sequence memory, predictive coding, neuromodulation, spatial tiling, dendrites, large scale.
Build the *unit of fabric* such that these can be added later by scaling/interaction.

## 1. Paradigm shifts (what changes vs current code)

| Current (v1) | v2 |
|---|---|
| Synchronous tick; "winner = max membrane among supra-threshold this tick" (argmax-ish) | **Per-neuron threshold crossing emits a spike event**; no selection step. Competition is emergent. |
| Inhibition = instantaneous clamp `v→θ` ("free-energy wipe") | **Delayed feedback shunting inhibition** (E→I→E with conduction delays) implements k-WTA *dynamically*. |
| Current-additive membrane (`v += w`) | **Conductance-based LIF** with reversal potentials + driving force (shunting, saturation, gain control). |
| Zero delays; everything same tick | **Per-projection synaptic delays** (delay lines). Delays *create* the WTA timing window and enable later sequence learning. |
| On-winner pre∧post coincidence + 45% heterosynaptic prune; `fired_l2e` bug | **Pair-based STDP** with eligibility traces (timing-asymmetric, local) + slow synaptic scaling. |
| L2→L3 driven by analog membrane (`v_l2e>=0`) | **L3 driven by L2 spike events** through delayed plastic projections. |
| L3 recruitment = `l2_winner % 4` | **L3 self-organizes** from its own STDP + its own local activity-driven recruitment. |
| Observation runs full plasticity, Δ huge | Incremental STDP ⇒ a probe barely perturbs; optional frozen-plasticity flag *as labeled measurement tooling only*. |

## 2. Neuron model (conductance-based LIF, voltage units)

State per E/I neuron: membrane `V`, excitatory conductance `g_E`, inhibitory conductance `g_I`,
fixed threshold `θ`, refractory timer, slow activity trace `a` (for homeostasis).

Per dt (small, fixed; dimensionless but consistent — document a ms mapping):
```
dV = dt/τ_m * [ g_L*(E_L - V) + g_E*(E_E - V) + g_I*(E_I - V) ]      # conductance dynamics, driving force
V += dV
g_E -= dt/τ_E * g_E ;  g_I -= dt/τ_I * g_I                           # conductances decay to 0
if refrac == 0 and V >= θ:  emit spike; V = V_reset; refrac = T_ref  # spike EVENT
```
Constants (voltage frame): `E_L = 0` (rest), `E_E > θ` (e.g. 70), `E_I ≤ E_L` (e.g. −10 → hyperpolarizing,
or = E_L for pure shunting), `θ = 50`, `V_reset = 0`. `g_L` fixed (the leak).

**LIF→pure-integrator preserved (the training mechanism):** a presynaptic spike adds `w` to `g_E`. Nascent
(small `w`): each volley's `g_E` bump is small vs `g_L` → leak wins → many arrivals needed (leaky). Mature
(large `w`): one volley's `g_E` drives `V` past `θ` before it leaks → fires on one volley (pure integrator).
The transition is now the **`g_E/g_L` ratio growing with gate maturity** — the biologically standard
"high-conductance vs leaky" distinction. `maturity_j = Σ_i w_ij / (gate sum that makes one volley reach θ)`
stays an observable; **NOT fed into `g_L`** (leak fixed, per your constraint). Driving force also fixes the
v1 "free-energy overshoot": `V` saturates toward `E_E`, so overshoot is naturally small — no clamp needed.

## 3. Synapse model (conductance increment + delay + STDP)

Each projection P(pre-pop → post-pop): weight matrix `W` (sparse-capable), delay `d` (scalar or per-synapse),
sign fixed by **pre-pop type** (E-pop → adds to `g_E`; I-pop → adds to `g_I`) ⇒ Dale enforced structurally.

- **Transmission (event):** when pre neuron `i` spikes at time `t`, schedule a deposit `g_*[j] += W[j,i]`
  at `t + d` for every post `j` (ring-buffer delay line per projection).
- **STDP (pair-based, local, on spike events) on plastic projections only:**
  - maintain pre-trace `x_i` (bumped on pre spike, decays τ_x) and post-trace `y_j` (bumped on post spike, decays τ_y).
  - on **post spike** j: for incoming synapses, `W[j,i] += A+ * x_i` (LTP: pre-before-post).
  - on **pre spike** i (arrival): `W[j,i] -= A- * y_j` (LTD: post-before-pre).
  - soft bounds: keep `W∈[0, w_max]` (e.g. multiply increments by `(1−W/w_max)` for LTP, `(W/w_max)` is implicit) → stable.
- **Synaptic scaling (slow homeostasis, per post-neuron, local):** every K presentations, if `Σ_i W[j,i] > budget`,
  multiply that neuron's incoming weights by `budget/Σ` — but **slowly** (small step), not instant. (Turrigiano; fixes A11 timescale.)
- **No heterosynaptic 45% prune.** "Pruning" is now LTD (timing) + scaling. `PRUNE_RATE`, `W_GAIN` retired;
  hyperparameters become `A+`, `A-`, `τ_x`, `τ_y`, scaling rate, `budget` (you said decay/pruning are free to change).

## 4. Competition = delayed feedback shunting inhibition (replaces clamp & argmax)

Per layer: E-population + one (or few) I-interneuron(s).
- `E→I` projection: small delay `d1`, strong enough that a single (or few) E spikes drive I over its θ.
- `I→E` projection: delay `d2`, deposits into **`g_I`** of all E (or a neighborhood) → **shunting** (divisive) inhibition.
- The **first** E cell to cross θ spikes first, triggers I after `d1+d2`, which shunts the slower E cells before
  they reach θ. ⇒ **k-WTA emerges from timing**, not from a max-comparison or a clamp. `k` is tuned by I strength/threshold/delay.
- This is the principled, event-based, conductance-based replacement for the v1 brake. No global read; fully local
  (each neuron only integrates the conductances delivered to it).

## 5. Hierarchy & composition (the centerpiece)

- L1 (input) → L2 (features) → L3 (compositions), each a layer-tile (§4 competition + §3 STDP feedforward gates).
- **Composition emerges:** when a composite stimulus makes L2 features A and B fire near-coincidentally, both their
  spikes precede the L3 post-spike → STDP potentiates **both** A→L3 and B→L3 ⇒ that L3 cell *binds* {A,B}. Decompose a
  trained L3 cell by reading its strongest L2 inputs = its constituent features. This is the falsifiable test of composition.
- **Hierarchy** = stack tiles (L3→L4…); the same Population/Projection mechanism repeats. Delays make *temporal*
  composition (sequences) reachable later without architectural change — deferred, but the fabric supports it.
- L2-spike→L3 fixes A15/A16; L3's own STDP + local recruitment fixes A17.

## 6. Tileable architecture (clean abstraction — rebuild what the deleted OOP attempt aimed at, correctly)

- `Population(N, type∈{E,I}, θ, params)` — conductance-LIF state arrays; `step(dt)`; emits spike indices.
- `Projection(pre, post, W, delay, plastic, stdp_params)` — delay line; `deliver(spikes)`; `apply_stdp(pre_spk, post_spk)`;
  sign from `pre.type`.
- `Network` — owns populations + projections; per dt: integrate populations → collect spikes → push into projections →
  deliver arrivals → run STDP on spiking synapses → homeostasis on a slow clock.
- Adding a layer/tile = add a Population + Projections. **Scalable & tileable by construction.** Keep numpy-vectorized
  *within* a projection for speed, but the *mechanism* is still per-spike-event local (no global objective).

## 7. Self-organization & dead-unit recruitment (local, unsupervised)

- Each neuron keeps a slow activity trace `a_j`. If `a_j` stays below a floor (chronically silent), the neuron
  **up-regulates its own incoming weights** (intrinsic homeostatic scaling-up) until it captures a niche. Local,
  driven only by the neuron's own firing history — **no cross-layer signal** (fixes A17). Threshold stays fixed.

## 8. Bug-fix mapping (from the review / registry)

| Registry | Fix in v2 |
|---|---|
| A15 analog L2→L3 | L3 driven by L2 **spike events** via delayed projection |
| A16 `fired_l2e` never set / L3 LTD-only | STDP uses real spike times; LTP occurs on genuine L2 pre-spikes |
| A17 `winner%4` recruitment | L3 self-organizes; recruitment from its **own** activity trace |
| A18 probe mutates state | incremental STDP ⇒ negligible per-probe Δ; optional labeled freeze for clean figures |
| A19 unclamped 1.0 weights / dead L1I | bounded STDP weights; I-neurons calibrated to actually fire (§4) |
| A7 clamp-to-θ | removed → delayed shunting inhibition |
| A6 max-membrane winner | removed → no selection; emergent first-to-fire |
| A2 zero delay / A1 lockstep | explicit delays + fine dt (full event-driven queue = future option) |
| A4 current-additive membrane | conductance-based with driving force + shunting |

## 9. Constraints honored (checklist)
- [x] Spike-event communication; [x] voltage/conductance internal state; [x] STDP unsupervised & local;
- [x] delays; [x] conductance membrane + shunting; [x] LIF→pure-integrator retained (g_E/g_L);
- [x] firings as unique events (no multi-event-per-tick selection); [x] Dale; [x] no teacher/argmax/global error;
- [x] composition & hierarchy central; [x] tileable/scalable; [x] hyperparameters (rates) tunable.

## 10. Behavioral validation targets (behavior is the deliverable)
1. **Competition:** a layer yields sparse k-WTA winners, emergent (no clamp); vary I strength → k changes predictably.
2. **Learning / LIF→integrator:** gates grow via STDP; volleys-to-fire for a learned cell drops many→~1; maturity→~1.
3. **Self-organization:** L2 cells specialize to distinct features unsupervised; no dead units; robust to seed.
4. **Composition (key):** present composite stimuli; an L3 cell becomes selective for a specific L2 combination;
   its top L2 inputs = the constituent features (decomposable). Selectivity ≫ chance.
5. **Hierarchy:** the L2→L3 story repeats for L3→L4 (smoke test on 1 extra tile).
6. **Stability:** no runaway, no oscillation, no dead units; a probe leaves consolidated state ~intact.

## 11. Open decisions (need your call before implementation)
- **D1 Time model:** fine-dt clock (recommended v1 — simple, tileable) vs full event-driven priority queue (more
  faithful continuous time, heavier). 
- **D2 Input encoding:** keep discrete "presentations" but as brief **spike volleys with delays** (recommended) vs
  continuous Poisson spike-train input (more biological, needed for sequences later).
- **D3 k-WTA target:** how sparse per layer (k=1 like v1, or k>1 to allow composition to see multiple active features)?
  Composition likely needs **k>1 at L2** so multiple features co-fire for L3 to bind.
- **D4 Units:** dimensionless-but-consistent (recommended) vs real ms/mV mapping now.
- **D5 Scope:** confirm rebuild as fresh `fep_model_v2.py` (+ `population.py`, `projection.py`) leaving v1 intact for comparison.

## 12. Risks
- STDP + conductance + delays is a substantial rewrite; **stable composition is not guaranteed** and will need tuning
  (A±, τ's, delays, I strength). Plan for an iteration loop on the validation targets, not a one-shot.
- k-WTA timing (delays vs I strength) is the delicate part — too strong/fast = single winner (no composition inputs);
  too weak/slow = runaway. Budget tuning time here.
- Composition emerging depends on **input statistics** (composites must actually co-occur) — the stimulus protocol (D2/D3) is load-bearing.

## 13. Implementation status — 2026-06-29 (build pass)

**Built & runs:** `population.py` (conductance LIF + driving force + shunting + spike-frequency
adaptation), `projection.py` (delay lines + local pair-STDP + synaptic scaling + Dale by pre-type),
`fep_model_v2.py` (L1→L2→L3 with delayed-feedback-inhibition WTA, spike-driven L2→L3, deprivation
recruitment, maturity readouts). All **architectural** goals met: spike-event comms, conduction
delays, conductance/voltage state, STDP, no clamp, no argmax, Dale, tileable. The four v1 bugs
(A15–A19) are fixed *by construction*.

**NOT yet working — dynamics do not self-organise.** L2 collapses to a single shared detector /
single tyrant (1/8 unique winners; gates saturate to w_max; maturity→1.0 for all). Root cause:
competitive STDP map formation is sensitive and needs proper **per-neuron rate homeostasis** to tile
the input space; adding spike-frequency adaptation alone did not fix it.

**Concrete next levers (priority order for the tuning session):**
1. **Diehl–Cook-style homeostatic excitability** — strongest known fix: make the per-neuron adaptation
   strong/slow enough that a cell that just claimed a pattern is suppressed across the *next several
   presentations*, so others claim other patterns; tune steady win-rate ≈ 1/N. (Conductance analog of
   v1's threshold homeostasis, which reached 8/8.)
2. **LTP/LTD balance + harder soft bound** — gates all peg at w_max; raise a_minus vs a_plus or add
   weight-dependent depression so non-pattern synapses are actively removed.
3. **Inhibition timing/strength** — confirm L2I fires in 1–2 ticks and shunts hard enough that ~1 L2
   cell fires per volley; instrument per-presentation L2 spike counts.
4. **Recruitment** — uniform +RECRUIT to all gates pushes toward generalists; make it gentler /
   pattern-specific or gate on near-threshold activity only.
5. Only after L2 tiles cleanly: validate L3 composition.

Matches the plan's stated risk. The fabric is the durable deliverable; tuning is a focused loop on the five levers.

### 13b. Learning-rule swap result — 2026-06-29 (second build pass)

Per user request, **swapped STDP out for v1's proven demand-driven gate rule** (grow active-input
gates on a win, prune the rest, budget cap) inside the v2 conductance fabric, plus added **symmetric
intrinsic homeostasis** (winner tires / idle cell sensitises, (N-1):1 — both halves of v1's rule) and
a random tie-break for the learning winner. STDP transmission/delays retained; plasticity is the v1 rule.

**Result: partial. 2–4/8 unique winners (seed-sensitive); does NOT robustly tile (v1 got 8/8).**
Best observed: one cell cleanly specialised (e.g. Row0→correct gates [0,1,2]), the rest partial.

Diagnosis of why v2 is harder to tile than v1 (all evidence-based this session):
- v1's winner was a clean **deterministic first-to-fire**; v2's "winner" is determined by conductance
  dynamics + **delayed feedback inhibition**, which isn't crisp enough — multiple cells fire per volley
  and the argmax-of-spike-count winner is noisy. The homeostasis interacts with this less cleanly.
- The conductance scale, inhibition strength/delay, and the homeostasis ratio must be tuned **jointly**;
  found neither the monopoly nor the random-rotation regime gives clean tiling at tried values.
- **Practical blocker:** pure-Python at T_STEPS=80 × thousands of presentations is slow (~minutes/run,
  some runs time out), which makes the tuning loop expensive. Speeding up the sim (vectorise the tick
  loop, or cut T_STEPS, or event-driven) is a prerequisite for efficient tuning.

**Honest status:** the v2 *fabric* is built and correct; competitive self-organisation in it is **not yet
working**. v1 (`fep_model.py`) remains the working competitive/hierarchy substrate. Next session should
(1) speed up the sim, then (2) jointly tune WTA crispness (inhibition delay/strength so ~1 cell fires
per volley) + homeostasis ratio, validating per-presentation spike counts — not guess parameters blind.
