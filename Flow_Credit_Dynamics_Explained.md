# Flow Credit Dynamics — A Plain-Language Walkthrough

This document explains, from the ground up, how **flow-rate charge** and
**flow-proportional assembly credit** work together in the network. The
methodology sheet (`Current_Implementation_Methodology_Equations.md`) has the exact
equations; this one is the intuition behind them. No equations are required to
follow the story, but the few that appear are translated into words.

---

## The one-sentence version

> A synapse behaves like a **tap of flowing water** rather than a thrown bucket,
> and when a neuron finally fires, it hands out **learning credit in proportion to
> how much each tap has been flowing** — with the single biggest contributor
> getting full credit, so it can grow strong enough to drive the neuron **by
> itself**.

Everything below unpacks that sentence.

---

## The three characters you need

The full network has more parts, but flow credit only involves three:

- **L1E** — the input pixels. When a pattern is shown, the relevant pixels fire
  over and over, like a strobe light held on. Think of them as **faucets** upstream.
- **L2E** — the output/"concept" neurons. Each one is learning to respond to one
  input pattern (a line, a diagonal, etc.).
- **L2I** — a single shared **inhibitory** neuron. Its job is to enforce
  competition: when one L2E wins, L2I fires and suppresses the others. Without L2I
  firing reliably, there is no competition and the whole thing collapses into
  "everybody wins," which is the same as nobody learning anything distinct.

The star of this document is the connection **L2E → L2I** (an "E→I" synapse): the
wire that lets an excitatory winner *tell* the inhibitor to fire. Getting that wire
to grow correctly is the entire problem.

---

## Part 1 — What "flow" means

### The old picture: a thrown bucket

In the simplest spiking model, a synapse with weight `w` works like this: when the
input spikes, you **dump `w` units of charge onto the membrane instantly**, all at
once, and it's gone the next moment. One spike = one bucket of water thrown at a
wall. Splash, done.

### The new picture: a tap left running

**Flow-rate accumulation** reinterprets the same weight `w` as the **size of a tap**,
not the size of a bucket. When the input spikes, you don't dump charge — you
**open a tap** that then trickles charge into the membrane over the next several
time-steps, and the flow gradually decays (the tap slowly closes on its own).

Concretely, each neuron carries a hidden quantity called the **current trace** `I`
(in code: `exc_trace`). On a spike it jumps up; every step after that, the membrane
drinks in the current (`V += I`) and the current fades a little (`I *= d`, where `d`
is the decay, default 0.8). So a single input spike keeps feeding the membrane for a
while instead of vanishing immediately.

Two consequences matter for the rest of the story:

1. **Charge builds up over time.** Because a held pattern re-fires its pixels every
   few steps, the taps stay open and the membrane fills like a bucket under a
   running faucet. A neuron can even **cross its firing threshold on a step where no
   new input arrived**, purely from charge still flowing in from earlier spikes.
2. **"How much has this synapse been flowing lately" becomes a real, measurable
   quantity.** That quantity is what credit will be shared by. Hold onto this.

> Why bother with taps instead of buckets? Because real signals arrive over time,
> and a tap naturally integrates evidence: a synapse that has been steadily driving
> the neuron for a while has delivered a lot of water, even if it isn't spiking on
> the *exact* step the neuron happens to fire.

---

## Part 2 — The problem flow credit solves (the "L2I firing deadlock")

Here is the trap the network fell into **before** flow credit existed.

For competition to work, **L2I must fire**. L2I fires when its incoming E→I wires
are strong enough to push it over threshold. Those wires learn by a simple rule:
*"when I (L2I) fire, strengthen the synapses that helped make me fire."* Fine — but
which synapses get the credit?

The **old rule gave credit only to the synapse that spiked on the exact final
step** — the "last volley." This breaks badly in a world of flowing taps:

- L2I often crosses threshold on a step with **no fresh input spike at all** (the
  charge was still flowing in from before). On that step, "who spiked last" is empty
  or misleading, so credit lands on the wrong wire — or nowhere.
- The **true dominant driver** — the L2E specialist that has been steadily flowing
  into L2I for many steps — keeps getting **skipped**. It never gets rewarded for
  the sustained work it actually did.

The result was a **deadlock**: no single E→I wire ever grew strong enough to fire
L2I on its own. In tests, the strongest E→I wire **stalled at about half of
threshold** (~631 out of ~1143) and L2I basically went **silent**. No L2I firing →
no competition → no clean winners. The engine had the right architecture but a
credit-assignment bug that starved the one wire it most needed.

---

## Part 3 — Flow credit (the fix)

Flow credit changes the answer to *"which synapses get the credit?"* from **"whoever
spiked last"** to **"whoever has been flowing the most."**

Three ingredients:

### 1. A memory of recent flow (the eligibility trace)

Each incoming synapse keeps a small running tally called an **eligibility trace**
(in code: `_trace`). Every time that input participates, its tally goes up; every
step, the tally **decays at the neuron's own leak rate**. That last detail is the
elegant part: the trace fades on **the same clock the membrane forgets charge**, so
the trace measures flow **over exactly the window that actually mattered for
firing**. It is a short-term memory of "how much has this tap been running lately."

### 2. Credit is handed out in proportion to flow — on the neuron's *own* fire

When L2I fires, it looks at all its incoming excitatory wires and their eligibility
traces (their recent flow). Then:

- Every **contributor** (a wire that delivered some flow) is **strengthened**, in
  proportion to its share of the flow.
- Every **non-contributor** (a wire that delivered nothing this window) is gently
  **weakened**, drifting back toward the floor.

This is purely local and event-driven: it happens on L2I's **own** spike, using only
its **own** incoming wires and their **own** traces. Nothing looks up a pattern
label, a rival neuron, or a global average.

### 3. Normalize by the *biggest* contributor, not the total — the crucial choice

This is the subtle decision that actually breaks the deadlock. When sharing credit,
you could divide by the **sum** of all flows, or by the **maximum** single flow. The
code divides by the **maximum**.

- **Divide by the sum** → credit is split into fractions that add to 1. If five
  wires contribute, each gets ~20%. Nobody ever reaches full strength; you get five
  permanently mediocre wires. The deadlock persists in a different costume.
- **Divide by the maximum** → the single **dominant driver gets the *full* learning
  rate every time**, and weaker ones get a proportional slice. The dominant wire
  therefore **climbs all the way to the cap** and becomes **self-sufficient**: one
  spike from that one L2E specialist is now enough to fire L2I by itself.

That self-sufficiency is the whole goal. It means a consolidated specialist can
drive the inhibitor in rhythm, competition runs, and — because only the winner fires
and therefore only the winner learns — ongoing plasticity stays stable.

---

## Part 4 — One cycle, start to finish

Picture the pattern "row 0" held on the input. Follow the L2E specialist that has
started to prefer it — call it **A** — and the shared inhibitor **L2I**.

1. **Pixels strobe.** Row-0 pixels fire every few steps. Through flow-rate taps,
   they keep charge flowing into every L2E, including A.
2. **A wins and fires.** A fills fastest (its feedforward RF matches row 0), crosses
   threshold, and spikes. Its **A→L2I** tap opens and starts flowing into L2I.
3. **Flow accumulates in L2I.** Over the next steps A keeps winning and firing, so
   the A→L2I tap keeps running. L2I's eligibility trace for that wire grows large,
   while other wires (from L2E neurons that rarely fire on row 0) stay near zero.
4. **L2I finally fires** — possibly on a quiet step, purely from accumulated flow.
   Now flow credit runs: A→L2I was **by far the dominant flow**, so it is normalized
   to 1.0 and gets the **full** learning-rate boost. The idle wires get nudged down.
5. **Repeat.** Every L2I fire, the A→L2I wire gets full credit again. It climbs
   steadily to the cap. Soon a **single** spike from A fires L2I on its own — no
   pile-up from others needed.
6. **Competition is now self-driving.** A fires → A alone fires L2I → L2I suppresses
   the rival L2E neurons → A keeps its win. A stable specialist plus a rhythmically
   firing inhibitor, with no external "punish the losers" signal required.

In validation this shows up as L2I firing on a steady ~16-step rhythm with the
winner's E→I wire matured to threshold — exactly what stayed broken under the old
last-volley credit.

---

## Part 5 — The equations, translated

From the methodology sheet, for L2I's incoming positive wires on its own fire:

```
phi_i      = trace_i                     # recent flow through wire i (eligibility)
phi_max    = max over wires of phi_i     # the single biggest flow
p          = clamp(theta / v_pre, 0, 1)  # "how cleanly did I just cross threshold"

if wire i delivered flow (phi_i > 0):
    strengthen it, scaled by its flow share phi_i / phi_max
else (delivered nothing):
    weaken it gently toward the floor
```

Word for word:

- **`phi_i = trace_i`** — "how much has wire *i* been flowing, over the window that
  matters." (Built up on each input volley, decayed at the neuron's leak.)
- **`phi_i / phi_max`** — "wire *i*'s flow **relative to the biggest** flow." The
  biggest contributor gets 1.0 (full credit); a wire flowing half as much gets 0.5.
  **This ratio is the deadlock-breaker.**
- **`p = clamp(theta / v_pre, 0, 1)`** — a modesty factor. If the neuron fired with a
  membrane barely at threshold, `p ≈ 1` (learn fully); if it fired with a big
  overshoot, `p < 1` (learn a bit less). It just keeps the step sane.
- **The "else" branch** — wires that did nothing this window decay toward the floor,
  so credit concentrates on the wires actually doing the driving.

The strengthening also has the usual **saturating cap**, so a wire slows as it nears
the ceiling and never runs away.

---

## Part 6 — What you would see on the dashboard

- **With assembly flow credit OFF:** L2I fires rarely or erratically; the strongest
  L2E→L2I weight plateaus **below** L2I's threshold (it stalls around half). On the
  weights view, no E→I wire ever reaches the top of its range.
- **With it ON:** one L2E specialist's L2E→L2I wire **climbs to the cap**, and L2I
  begins firing on a **regular rhythm**. Toggle it live (`Assembly flow credit (E→I)`
  in Model Config) on a single held pattern and watch the top E→I weight cross
  threshold and L2I's firing become periodic.

---

## Why this fits the ideology

Flow credit is deliberately **local and self-organizing**:

- It runs on a neuron's **own** spike, using only its **own** incoming wires and
  their **own** flow traces.
- It needs **no pattern labels, no rival lookup, no global average, no external
  supervisor**. A wire is rewarded strictly for the flow it delivered.
- It is the mechanism that lets an inhibitor become self-sufficient **from evidence
  alone**, which is why it can replace externally-imposed consolidation crutches
  (such as the now-archived "loser depression" signal) with something the network
  discovers on its own.

That is the point: competition and consolidation should **emerge** from each
neuron minimizing its own surprise with the information in front of it — not from a
referee reaching in to punish the losers.
