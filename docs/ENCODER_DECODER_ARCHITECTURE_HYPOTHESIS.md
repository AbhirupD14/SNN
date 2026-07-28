# Encoder–Decoder Architecture Hypothesis

**Status:** architectural interpretation and future hypothesis
**Implementation status:** encoder/feature-extraction fabric exists; semantic decoder does
not
**Last updated:** 2026-07-28

## Core interpretation

The current cortical-column fabric is not intended to be the final classifier.

Its present role is closer to a hierarchical edge and feature extractor:

```text
raw spatial input
-> local edge/pattern competition
-> sparse column activity
-> progressively compressed feature representation
```

A separate future network would consume that compressed representation and diverge into a
larger semantic space:

```text
compressed latent feature code
-> divergent associations / compositions
-> symbol candidates
-> semantic representation
```

The analogy is an autoencoder-like shape: information contracts through an encoder and
later expands through a decoder. It is not yet a demonstrated autoencoder. There is no
implemented decoder, reconstruction objective, end-to-end error signal, or measured symbol
creation stage.

The current scientific questions should therefore be separated:

1. **Encoding:** Can the fabric learn stable local edge/feature representations, compress
   them, and transmit a useful latent code?
2. **Decoding:** Can another network expand that latent code into distinct symbols and
   richer semantics?

Only the first question is currently implemented. The second remains a hypothesis because
there is not enough project time to build and test the decoding network.

## Role of the current column

Within one classic column:

- ordinary E neurons compete to learn local input patterns;
- the winning ordinary E is the local latent identity;
- fixed Eor reports that the column recognized something;
- C combines local basal evidence with higher-layer apical confirmation;
- feedback can remove a redundant future output event;
- output cadence is intended eventually to represent certainty or attention, not content
  identity.

The fabric thus separates two kinds of information:

| Channel | Intended meaning |
| --- | --- |
| Which column or population is active | feature content and spatial source |
| Which ordinary E won locally | finer latent identity inside that column |
| Eor event | compressed “this column recognized a feature” message |
| Output frequency/cadence | proposed certainty, use, learning, or guided-attention state |

Classic Eor is deliberately a compression boundary. Its fixed `theta` bank makes it a
reliable relay for column activity, but a single Eor event does not preserve which ordinary
E won.

That loss is not automatically a defect. It is acceptable wherever the decoder only needs
the distributed pattern of *which columns* were active. It becomes a hard constraint when
several semantically different inputs produce the same complete distributed Eor pattern.

## The information-preservation requirement

Compression may reduce dimensionality, but a downstream decoder can separate two inputs
only if their latent observations remain distinguishable.

Let `z(x)` be the complete event stream exposed to the future decoder for input `x`. For a
deterministic downstream network:

```text
z(x) != z(y)    separation may be learnable
z(x) == z(y)    no downstream learner can distinguish x from y from that input alone
```

The “complete event stream” includes every source identity, event time, charge, and other
declared decoder-visible state. Adding more decoder neurons, training longer, changing the
seed, or increasing the learning rate cannot recover a distinction absent from `z`.

This gives the encoder a concrete interface obligation:

> Before semantic decoding is attempted, the compressed latent population must preserve
> the distinctions the decoder will be asked to learn.

The code need not preserve pixels or local ordinary-E addresses verbatim. A lower
dimensional distributed code is sufficient. But the code cannot collapse all
task-relevant cases onto one identical observable state.

## Reinterpreting the two-tower V/A/7 experiment

The two-tower experiment tested:

```text
left 9x9 encoder -> one left L2 column  \
                                            -> one classic L3 column
right 9x9 encoder -> one right L2 column /
```

The lower towers succeeded as feature extractors:

- every active L1 column consolidated;
- both L2 columns matured;
- the left and right towers assigned different L2 ordinary-E owners to V, A, and 7;
- cold recall recovered all six tower-level assignments;
- Eor and C paths operated reliably.

The failure occurred only at the interface presented to L3. Each whole tower was reduced
to one Eor source. All three glyphs activated both towers with the same cadence, so L3 saw
the same two sources, event counts, timing, and charge for V, A, and 7. Their complete L3
input traces were byte-identical.

The correct interpretation is:

- **not:** the edge extractor failed to learn V/A/7 structure;
- **not:** a future divergent decoder is impossible;
- **not:** the present fabric should already be judged as an end-to-end classifier;
- **yes:** two scalar tower-level Eor streams are insufficient as the complete latent code
  for distinguishing these three glyphs;
- **yes:** compressing an entire `9x9` tower to one content-free activity event crossed the
  information limit for this decoding task.

The two-tower result therefore identifies a decoder-interface boundary. It does not reject
the encoder–decoder architecture.

## Hypothesized semantic expansion

A future decoder would be a separate network downstream of a distributed latent
population. Conceptually, it could:

1. receive sparse feature-presence events from multiple spatial columns or regions;
2. expand each compact latent combination onto a larger pool of semantic candidates;
3. use competition to allocate different stable units or assemblies to different
   combinations;
4. associate increasingly composed combinations with symbols;
5. preserve certainty/attention as a modulatory cadence rather than using cadence as the
   identity code.

The likely content carrier is a **constellation of source-addressed latent events** across
many columns, not one scalar Eor for an entire visual half. Divergence can create a larger
semantic population, but it cannot invent distinctions after all content-bearing source
structure has been pooled away.

Several interface families remain conceivable:

- retain more spatial Eor channels before the compression boundary;
- decode from a population of intermediate columns rather than one final Eor per tower;
- expose an explicitly identity-bearing latent channel while keeping Eor for
  certainty/control;
- combine several compressed regional symbols whose active-source subsets differ across
  inputs.

These are hypotheses, not approved implementation directions. In particular, this note
does not authorize feature gates, direct-identity wiring at every boundary, a new learning
equation, or using frequency to encode both content and certainty.

## Why frequency should remain separate from content

The intended future interpretation of firing cadence is:

- high activity: active learning, active use, novelty, or guided attention;
- slower alternating activity: a confirmed representation requiring less repeated
  evidence.

Using Eor timing or frequency to recover the local winner identity would overload the same
channel with content and certainty. The two-tower experiment also found identical late
cadence for V, A, and 7, so frequency did not preserve identity in that case.

The working hypothesis is therefore:

```text
source/population pattern -> what is represented
frequency/attention state -> how certain or actively used it is
```

Whether the cadence truly tracks certainty rather than regular auto-pacing remains an open
experiment.

## Claim boundary

### Supported now

- The current network can act as a local and hierarchical feature extractor.
- Classic Eor reliably emits a compressed column-active event.
- The two `9x9` towers learned distinct internal V/A/7 codes and cold-recalled them.
- A decoder cannot distinguish V/A/7 when its entire input is the measured identical pair
  of tower Eor streams.
- Direct identity demonstrates that retaining a richer source alphabet can make controlled
  compositions distinguishable one layer higher.

### Hypothesized, not tested

- A divergent downstream network can expand an adequate distributed latent code into
  stable symbols.
- Semantic units can be allocated without an end-to-end global error signal.
- The resulting decoder will generalize across position, scale, noise, or new
  compositions.
- Alternating frequency will provide a useful certainty/attention signal to that decoder.
- A sparse latent interface can preserve enough content without the density cost of
  transmitting every local winner identity everywhere.
- A decoder can reconstruct input structure; reconstruction may not be necessary if symbol
  creation rather than pixel recovery is the objective.

No classifier, semantic decoder, or reconstruction result should be claimed until those
mechanisms exist and are evaluated.

## Future acceptance criteria

If the decoder is pursued later, test the interface before tuning the decoder:

1. Freeze a trained encoder.
2. Record complete decoder-visible latent signatures for every target symbol and declared
   distractor.
3. Reject the interface if target cases have identical signatures.
4. Train the decoder only after identifiability is established.
5. Require distinct stable semantic owners or assemblies plus cold frozen recall.
6. Test whether removing one latent source destroys only the semantics it supports.
7. Separate content accuracy from certainty/attention cadence.
8. Compare novel compositions with memorized ones.
9. Preserve negative results where compression removes required distinctions.

This ordering prevents decoder tuning from being used to conceal an insufficient latent
alphabet.

## Documentation rule

Describe current higher-layer failures in terms of the role actually being tested:

- “the extractor failed” only when lower feature ownership or recall fails;
- “the latent interface is not identifiable” when lower codes differ but decoder-visible
  streams are identical;
- “the decoder failed” only after a decoder exists, receives distinguishable inputs, and
  still fails its learning/recall criteria.

Until then, the measured seed-1 two-tower result is a successful encoder-scaling result and
a negative latent-interface result—not a failed end-to-end classifier.

## Related evidence

- [`TWO_TOWER_COMPOSITION.md`](TWO_TOWER_COMPOSITION.md)
- [`FABRIC_CONSTRAINTS_AND_OPERATING_ENVELOPE.md`](FABRIC_CONSTRAINTS_AND_OPERATING_ENVELOPE.md)
- [`DIRECT_IDENTITY_TILED_TOPOLOGY.md`](DIRECT_IDENTITY_TILED_TOPOLOGY.md)
- [`FEEDBACK_CADENCE_AND_LOOP_LATENCY.md`](FEEDBACK_CADENCE_AND_LOOP_LATENCY.md)
- [`../Current_Implementation_Methodology_Equations.md`](../Current_Implementation_Methodology_Equations.md)
