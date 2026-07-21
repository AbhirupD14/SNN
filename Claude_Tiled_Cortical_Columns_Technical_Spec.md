# Technical Specification: Tiled Cortical Columns Preset

## Instructions to the implementer

Implement the new tiled cortical-column topology described here as a new built-in
preset while preserving every existing preset and its observable behavior. This is a
large repository-wide change, but it is primarily a graph-construction, metadata,
layout, input-shape, serialization, and dashboard-generalization task. Reuse the current
neuron equations, event-resolved scheduler, plastic feedforward rule, coincidence-cell
rule, and immediate hard-reset relay mechanics.

Do not silently redesign the science to obtain a preferred plot. Do not add lateral
connections, soft competition, multi-winner competition, a `delta_tau` admission rule,
a queue-driven simulator, special Eor threshold logic, global pattern labels, or
supervisory winner selection. The deferred event-scheduler and multi-winner composition
questions are recorded separately in
`docs/EVENT_DRIVEN_MULTIWINNER_COMPOSITION_PROBLEM.md`; they are explicitly out of
scope for this implementation.

The existing presets must remain available and functional:

```text
pi
old
rg
rg_residual
rg_coincidence
```

The canonical name of the new built-in preset in this specification is:

```text
tiled_cc
```

Use this name consistently in the backend preset registry, configuration API,
dashboard selector, persistence layer, documentation, and tests.

## Scientific objective

Replace the current one-neuron-per-pixel abstraction with reusable cortical-column
tiles while retaining the current mechanics inside each tile. The first concrete
hierarchy has:

- an exogenous 9x9 RGC input surface;
- nine spatial 3x3 RGC patches;
- one L1 cortical column per patch, arranged as a 3x3 tile grid;
- one L2 cortical column receiving the nine L1 column outputs;
- a configurable number `N` of ordinary competing E neurons per column;
- one ordinary plastic output E neuron (`Eor`) per column;
- one coincidence neuron (`C`) per column;
- one immediate inhibitory relay (`I`) per column;
- no lateral connections between columns in the same layer;
- current hard single-winner WTA inside each column;
- feedforward output from a child column through its Eor;
- feedback from every ordinary E in a parent column to the child C's apical branch.

The topology must be generated from reusable tile and connector rules so headless
experiments can change `N`, create smaller fixtures, or add greater hierarchical depth
without copying a hard-coded 191-node graph. The dashboard must consume the same graph
and metadata as the headless engine; it must not reconstruct a separate idea of a
column in JavaScript.

## Terminology

Use these terms consistently to avoid the ambiguity of “upstream”:

- **child/lower column**: closer to the RGC input;
- **parent/higher column**: farther from the RGC input;
- **ordinary E**: one of the `N` plastic excitatory neurons that participates in the
  column's local hard WTA;
- **Eor**: the column's single output excitatory neuron; it has exactly the ordinary E
  intrinsic and learning behavior and differs only by connectivity and metadata;
- **C**: the coincidence pyramidal neuron with one learned basal input and zero or more
  unweighted Boolean apical inputs;
- **I**: the column's one-shot immediate inhibitory relay;
- **tile**: the nodes and internal edges of one cortical column;
- **RGC patch**: one 3x3 spatial subset of the global 9x9 RGC surface;
- **column link**: one directed child-to-parent feedforward relationship plus the
  corresponding parent-to-child apical feedback relationship.

“Eor” is a role name, not a Boolean logic primitive. It must not receive a hard-coded
OR operation, a fixed supra-threshold packet, a special threshold, a special cap, or a
special firing shortcut.

## Hard behavioral contracts

### Ordinary E and Eor parity

Ordinary E and Eor must use the same `ExcitatoryNeuron` implementation and the same:

- threshold and resting/reset potential;
- leak/conductance integration;
- refractory behavior;
- analytic event crossing calculation;
- activity trace;
- plastic feedforward weight vector;
- target-owned causal participation vector;
- accumulating excitatory weight update;
- weight floor, weight cap, maturity budget, learning rate, and update mode;
- delay-one feedforward emission.

Their only behavioral difference must emerge from their edges:

- ordinary E drives and is reset by local I;
- Eor does not drive local WTA and is not a target of local I;
- ordinary E feeds local Eor;
- Eor feeds parent ordinary E and local C basal;
- Eor never supplies apical feedback.

It is acceptable to reuse/refactor the existing event-resolved plastic-E archetype or
to introduce metadata that distinguishes `E` and `Eor`, but do not duplicate the
ordinary E equation. If separate archetype names are used for display roles, both must
instantiate the same class through the same construction path and be covered by a
numerical parity test.

The current event path updates ordinary feedforward weights only when a fired cell is
registered in `_latency_ids`. Generalize this safely: every fired event-resolved
ordinary plastic E target must learn from its own `_ff_deliv_now` participation,
including Eor. Learning eligibility must follow the plastic target's declared mechanics,
not an ID prefix, layer name, or hard-coded “L2 competitor” list. Existing
`e_latency_competitor` behavior must remain unchanged.

### C-cell behavior

Reuse `CoincidencePyramidalNeuron` without weakening its gate:

- exactly one learned basal edge from the same column's Eor;
- apical inputs are unweighted Boolean permissions;
- lower-column C receives one apical edge from every ordinary E in every explicitly
  connected parent column;
- parent Eor never connects to child C apical;
- basal-only and apical-only input deposit exactly zero somatic charge;
- a valid gate deposits exactly the causal basal weight times basal signal, once;
- deposit is an immediate impulse at the later required receipt's `tau`;
- a mature C may fire at that same `tau`;
- C learns only its basal weight and only when C fires;
- duplicate apical deliveries remain observable but cannot duplicate charge;
- the one-boundary basal eligibility semantics remain unchanged.

Every column, including a top column, contains C. A column with no parent has no
apical edges. Its C is intentionally dormant:

- it still has exactly one local Eor basal edge;
- basal receipts may enter/expire through the existing eligibility state machine;
- it must never deposit charge or fire without apical input;
- it must never learn its basal weight;
- zero apical inputs are legal only when column metadata declares that the column has
  no parent (or an equivalent explicit dormant/top-column marker).

Do not globally relax coincidence validation in a way that allows an accidentally
unwired non-top C. A C in a column with one or more declared parents must receive the
complete expected apical bank.

### Local inhibitory relay and hard WTA

Each column has exactly one I. Its edges are:

```text
every ordinary E -> I      relay_excitation
C -> I                     relay_excitation
I -> every ordinary E      hard_reset_inhibition
```

There are no I edges to Eor, C, RGC, a parent column, or another column's E population.
The current immediate relay behavior is retained:

- the first eligible ordinary E spike recruits I at the same `tau`;
- I hard-resets every ordinary E in its own column, including the cell whose emitted
  spike caused the relay;
- the emitted winner spike and its learning survive its own reset;
- later prospective crossings in that column are cancelled;
- I emits at most once per outer boundary;
- a later same-boundary ordinary E or C input does not create an inhibitory burst or a
  second reset event;
- if C fires before an ordinary E crossing, C may recruit I and suppress that local
  E population before it emits;
- if the column I has already fired, a later C spike cannot retroactively cancel an
  already-emitted E spike.

WTA must emerge from first-spike latency plus these local edges. Do not add a centralized
per-column winner selector. Different columns are independent: one winner in an L1
column must not reset candidates in another L1 column. Multiple columns may therefore
each produce one local winner in the same outer boundary even though each individual
column remains hard single-winner.

### RGC behavior

Every RGC is the existing `rg_source` abstraction:

- owns exactly one global retinal pixel;
- has no membrane, learned weight, or incoming edge;
- emits on every input boundary where its pixel is active;
- is not affected by cortical inhibition;
- schedules its outgoing feedforward events with the normal one-boundary internal
  projection delay.

The RGC does not learn. The receiving ordinary E owns and learns every `RGC -> E`
feedforward weight.

## Reusable graph construction API

Implement graph construction as reusable functions/data structures, not a runtime
“super-neuron” that hides internal neurons from the engine. The engine, serializer, and
dashboard must continue to see explicit nodes and explicit directed edges.

A recommended shape is:

```python
@dataclass(frozen=True)
class ColumnHandles:
    column_id: str
    layer: str
    row: int
    col: int
    e_ids: tuple[str, ...]
    eor_id: str
    c_id: str
    i_id: str

def build_cortical_column(..., n_e: int, has_parent: bool) -> ColumnHandles: ...
def connect_rgc_patch(..., rg_ids, column: ColumnHandles) -> None: ...
def connect_columns(..., child: ColumnHandles, parent: ColumnHandles) -> None: ...
def tiled_cc_spec(..., e_count: int = 8) -> dict: ...
```

The exact names may follow repository style, but the separation of responsibilities is
required:

1. The tile builder creates only one column's nodes and internal edges.
2. The RGC connector creates only RGC-to-ordinary-E feedforward edges.
3. The column connector creates only child-Eor-to-parent-E feedforward edges and
   parent-E-to-child-C apical edges.
4. The preset builder declares the spatial hierarchy and composes these rules.

Calling the tile builder twice must not share mutable lists or state. Node and edge IDs
must be deterministic and unique. Simulation logic must never parse the generated IDs to
recover layer, role, position, parentage, or patch membership; it must use metadata.

### Canonical first-preset dimensions

Use these defaults:

```text
input rows                 9
input columns              9
patch rows                 3
patch columns              3
L1 column grid rows        3
L1 column grid columns     3
L1 columns                 9
L2 columns                 1
ordinary E per column N    8 (headless/configurable)
```

The global input index is row-major:

```text
pixel = global_row * 9 + global_col
```

Patch IDs are row-major over the 3x3 patch grid. A patch at `(patch_row, patch_col)`
contains global rows `3*patch_row ... 3*patch_row+2` and global columns
`3*patch_col ... 3*patch_col+2`. There must be no accidental cross-patch RGC edge.

For arbitrary `N`, the canonical graph has:

```text
RGC nodes                         81
nodes per column                  N + 3
columns                           10
total nodes                       81 + 10(N + 3) = 10N + 111

RGC -> L1 ordinary E edges        81N
internal edges per column         3N + 2
all internal edges                30N + 20
L1 Eor -> L2 ordinary E edges     9N
L2 ordinary E -> L1 C apicals     9N
total edges                       129N + 20
```

At the default `N = 8`, require exactly:

```text
191 nodes
1052 directed edges
```

These counts are acceptance tests, not merely documentation.

### Internal column rule

For a column with ordinary cells `E[0:N]`, output `Eor`, coincidence cell `C`, and
relay `I`, generate exactly:

```text
for each i:
    E[i] -> Eor    feedforward
    E[i] -> I      relay_excitation
    I -> E[i]      hard_reset_inhibition

Eor -> C           basal_excitation
C -> I             relay_excitation
```

The tile builder must not generate:

- `E[i] -> E[j]` lateral edges;
- `I -> Eor`;
- `I -> C`;
- `Eor -> I`;
- `Eor -> C` apical;
- `C -> E[i]`;
- any edge to another column.

### RGC-to-column rule

Given exactly the nine RGC IDs in one spatial patch and one L1 column, generate the
complete bipartite projection:

```text
for rg in patch.rg_ids:
    for e in column.e_ids:
        rg -> e     feedforward
```

This is `9N` plastic feedforward edges per L1 column and `81N` over all nine L1
columns. Plasticity belongs to the receiving ordinary E. Do not generate RGC edges to
Eor, C, or I.

### Column-to-column rule

For an explicit child/parent pair, generate:

```text
for parent_e in parent.e_ids:
    child.Eor -> parent_e    feedforward
    parent_e -> child.C      apical_excitation
```

Thus one pair contributes `2 * parent.N` edges. For the canonical 9-to-1 hierarchy,
the L2 ordinary E population has exactly nine Eor afferents per cell, and every L1 C
has exactly `N` apical afferents from the L2 ordinary E population.

Do not generate:

- child ordinary E directly to parent ordinary E;
- child C or I to parent;
- parent Eor to child C;
- parent I or C feedback;
- same-layer/lateral child-to-child edges;
- an apical edge into the top L2 C.

The connector should be composable for future depth. If a later L3 column is connected
above L2, calling the same function must make L2 C non-dormant by supplying L3 ordinary
E apicals. Do not hard-code the words `L1` and `L2` into the connection algorithm.

## NetworkSpec and metadata contract

The current NetworkSpec normalization drops most unknown metadata. Extend the schema in
a controlled, validated way so the topology remains the single source of truth for
headless construction, layout, editor round trips, and dashboard grouping.

### Top-level metadata

The new built-in spec must serialize a topology metadata object equivalent to:

```json
{
  "family": "tiled_cortical_columns",
  "input_shape": {"rows": 9, "cols": 9},
  "patch_shape": {"rows": 3, "cols": 3},
  "column_layers": [
    {"layer": "L1", "rows": 3, "cols": 3},
    {"layer": "L2", "rows": 1, "cols": 1}
  ],
  "columns": [
    {
      "id": "...",
      "layer": "L1",
      "row": 0,
      "col": 0,
      "e_count": 8,
      "parent_ids": ["..."]
    }
  ]
}
```

The exact JSON field nesting may be adjusted, but it must unambiguously provide input
shape, patch shape, layer tiling, column membership, column position, ordinary-E count,
and parentage. `validate_spec()`, `current_spec()`, saved presets, loaded presets,
`topology()`, REST state, and websocket topology messages must preserve it.

### Node metadata

Every column node must expose at least:

```text
column_id
column_role       one of: E, Eor, C, I
column_index      0..N-1 for ordinary E; null/absent otherwise
column_row
column_col
layer
```

Every RGC must expose at least:

```text
pixel             global row-major input ownership index
input_row
input_col
patch_id
patch_row
patch_col
patch_local_row
patch_local_col
```

It is useful for the ordinary L1 E nodes to expose their input patch and for all nodes
to expose a stable display label, but IDs remain opaque. Preserve these recognized
fields through validation and editor save/load cycles.

### Edge metadata

Add a stable projection-family field for generated edges, for example:

```text
rg_to_column
column_e_to_eor
column_e_to_i
column_i_to_e
column_eor_to_c_basal
column_c_to_i
column_to_column_ff
column_to_column_apical
```

The existing edge `kind` remains authoritative for simulation dispatch. Projection
family is metadata for validation, distance normalization, layout/debugging, and
dashboard filtering; it must not create a second delivery mechanism.

### Tiled-topology validation

Add structural validation that checks at least:

- input shape is positive and matches the number/range of RGC-owned pixels;
- patch shape tiles the 9x9 input exactly for the canonical preset;
- every RGC owns one unique global pixel and belongs to exactly one patch;
- every declared column contains exactly `N` ordinary E, one Eor, one C, and one I;
- each ordinary E, Eor, C, and I belongs to exactly one column;
- every column has the complete internal edge set and no duplicate internal edge;
- I resets exactly its own ordinary E bank and no external cell;
- every L1 ordinary E receives all nine and only the nine RGCs in its declared patch;
- every declared child/parent pair has the complete two-way feedforward/apical rule;
- Eor is never an apical source;
- a non-top C has exactly one basal plus the complete parent-E apical bank;
- a top/dormant C has exactly one basal and zero apicals;
- no generated same-layer lateral projection exists;
- no edge targets an RGC;
- all hard-reset targets use event-resolved E semantics;
- duplicate basal/apical source-target pairs remain illegal.

Generic custom graphs that are not in the tiled family should retain their current
validation rules. Existing `rg_coincidence` C cells must still require at least one
apical input. Do not make the dormant-top exception based on a node ID such as `L2C`.

## Event timing and causal sequence

Retain these timing rules:

- external RGC presentation occurs at the current outer boundary;
- every `feedforward` edge emits to a delay-one queue;
- every `basal_excitation` edge emits to the delay-one basal queue;
- `apical_excitation` from a firing parent ordinary E is delivered at the source's
  current analytic `tau`;
- `relay_excitation` into I is zero latency;
- `hard_reset_inhibition` from I is zero latency;
- no spike traverses two feedforward edges in one outer boundary.

A representative path is:

```text
t      RGC spikes
t+1    one or more L1 ordinary E targets integrate; each active column admits at most
       one local winner; that winner learns and schedules local Eor charge
t+2    L1 Eor integrates its local winner event; if it fires, it learns, schedules
       charge to every L2 ordinary E, and schedules basal eligibility to local L1 C
t+3    L2 ordinary E integrates child-Eor events; the first local L2 winner fires,
       learns, recruits L2 I, and supplies same-tau apical permission to connected L1 C
       cells whose basal event is current/carried
```

Actual firing may require retained evidence across multiple presentations because Eor
is an ordinary learned integrator, not a fixed relay. The above is a causal minimum-hop
illustration, not a promise that every target crosses on its first packet.

The scheduler may continue scanning all membranes to find the next crossing. Do not
replace it with a priority queue in this task.

## Feedforward initialization, geometry, and learning

### Target-local fan-in handling

The current general initialization derives one raw mean from global `n_pix`, which is
not sufficient for a hierarchy whose plastic targets have fan-ins of 9, `N`, and 9.
Do not initialize all tiled projections using `theta * fraction / 81` merely because
the retina has 81 pixels.

Every ordinary E and Eor in `tiled_cc` must initialize with the same event-plastic E
policy applied to its actual incoming feedforward row. Preserve seeded narrow jitter,
then use the existing bounded proportional normalization so each target's initial total
is:

```text
min(l2_init_total_frac * threshold, fan_in * per_afferent_cap)
```

This applies uniformly to:

- nine RGC afferents into each L1 ordinary E;
- `N` local ordinary-E afferents into each Eor;
- nine child-Eor afferents into each L2 ordinary E.

The name `l2_init_total_frac` is now misleading for a reusable event-plastic E row.
Either introduce a backward-compatible public alias such as
`event_e_init_total_frac` or clearly generalize its documented meaning without changing
legacy numeric behavior. Existing `rg_coincidence` initialization must remain bit
identical.

### Distance factors

Functional positions affect learning through inverse-square distance factors. The new
graph contains short within-column feedforward edges and much longer inter-layer edges.
If all use the same target archetype, the current “minimum distance per target
archetype” reference can make one projection family rescale another accidentally.

For the tiled preset, normalize distance factors within a stable projection scope. A
recommended rule is per plastic target: use the closest positive incoming feedforward
distance for that target as its reference. An acceptable alternative is an explicit
projection-family/layer scope, provided no within-column edge controls the learning
rate of RGC-to-column or column-to-column edges.

Do not globally change legacy distance normalization. Gate the new scope through
validated topology/projection metadata and pin old preset goldens.

Basal C distance influence should likewise be normalized without making one spatially
distant column incapable of learning merely because another column's Eor and C happen
to be closer in the functional layout. The delivered C charge remains exactly the
basal weight; geometry changes learning only.

### Learning assertions

Tests must prove:

- RGC weights themselves do not exist or change;
- every ordinary E owns and can update all of its incoming weights;
- Eor owns and can update all local-E incoming weights;
- L2 ordinary E owns and can update all nine child-Eor incoming weights;
- inactive afferents receive the ordinary rule's negative signal only when their own
  target fires;
- participation remains per target and per arrival boundary across arbitrary depth;
- C learning changes only its one basal edge on C firing;
- a dormant top C never changes its basal weight;
- weight editing through `/api/weight` works for ordinary E, Eor, L2 E, and C basal
  with the correct cap.

## Preset dimensions and configuration

The existing engine treats `n_pix` and `n_out` as construction-time values and the
dashboard applies topology changes without resizing them. A 9x9 preset cannot be
implemented by leaving `n_pix = 9`, and switching back to a legacy preset must not
silently build an 81-input version of that preset.

Add an explicit preset-construction descriptor or equivalent mechanism. Required
behavior:

- `SimulationEngine()` keeps its current default `pi`, `n_pix=9`, `n_out=8` behavior;
- `SimulationEngine(topology='tiled_cc')` defaults to an 81-input tiled graph;
- `cc_e_count` is a construction parameter with default 8 and must be `>= 1`;
- `cc_e_count`, not legacy `n_out`, sizes every tiled column's ordinary-E bank;
- legacy `n_out` retains its current meaning for the five existing presets and may
  remain present in public compatibility fields while `tiled_cc` reports its explicit
  column counts;
- headless callers can select another `cc_e_count` without editing the preset builder;
- selecting `tiled_cc` in the dashboard rebuilds input storage to 81 values;
- selecting a legacy built-in afterward restores that built-in's normal 9/8 dashboard
  construction dimensions;
- explicit headless size overrides for existing general-size tests continue to work;
- a topology change that changes input size clears the old input safely rather than
  truncating/reusing it;
- reset and reseed preserve the currently selected preset's resolved dimensions;
- saved/custom tiled specs retain their input shape and column metadata;
- invalid dimension combinations fail loudly with a useful message.

The canonical built-in `tiled_cc` has a fixed 9x9 input surface. An explicit headless
`n_pix` override that disagrees with 81 must be rejected rather than producing a partly
tiled graph. Future shapes should be expressed through a generalized tiled builder or
custom spec metadata, not by letting `n_pix` disagree with `input_shape`.

Do not solve this with preset-name checks scattered through API routes, the frontend,
layout, and simulation. Centralize dimension resolution in a preset descriptor or
topology metadata contract.

Expose `cc_e_count`, input shape, patch shape, and column counts in public topology
parameters. The dashboard may expose `cc_e_count` as an integer/range construction
control; applying it must rebuild and wipe learned state with the same warning semantics
as other structural changes.

## Headless input and experiment contract

Headless operation is mandatory and must not import FastAPI, websocket, DOM, or frontend
modules.

At minimum support:

```python
engine = SimulationEngine(seed=1, topology='tiled_cc', cc_e_count=8)
engine.set_input(vector_of_length_81)
engine.step()
```

Provide topology-generic helpers or experiment utilities for embedding a local 3x3
pattern into a selected patch. A recommended interface is:

```python
embed_patch_pattern(
    input_shape=(9, 9),
    patch_shape=(3, 3),
    patch=(row, col),
    local_vector=[...9 values...],
) -> list[int]
```

The helper must be pure and usable without constructing the dashboard. It must validate
shape and patch bounds. The engine's ordinary `set_input()` must continue accepting any
valid 81-element vector, including a diagonal spanning multiple patches.

For the initial isolation experiment, expose the existing 3x3 `row`, `column`, and
diagonal patterns embedded into one selectable patch. Default the dashboard patch
selector to the center patch `(1,1)`, but do not hard-code training mechanics to that
patch. Full-grid multi-patch patterns may be added later; the topology must already
accept them.

Pattern definitions must be topology-sized. `topology()['patterns']` and
`pattern_vectors` must never advertise a 9-element vector while an 81-input engine is
active, and `set_pattern()` must resolve against the active topology's pattern bank.
Legacy presets keep the existing four 3x3 vectors and names. Random, clear, noise, and
direct `set_input()` operations must use the active input size.

Create a small headless smoke/diagnostic experiment that records, without claiming
premature scientific convergence:

- per-column ordinary-E spike counts and winners;
- Eor spike counts;
- C deposit/spike counts;
- I spike/reset counts;
- feedforward and basal weight movement;
- cross-column reset violations (must be zero);
- top-C deposits/spikes (must be zero);
- active versus inactive patch behavior.

Machine-readable results should identify seed, `N`, input patch, pattern, boundaries,
and resolved topology dimensions.

## Functional and display layout

Do not extend the current ID-specific `generate_layout()` by relying on names such as
`L1E4` or `L2I`. Add a metadata-driven tiled layout path while leaving the existing
layout path and RNG draw order untouched for legacy presets.

The canonical visual/functional organization should make hierarchy and tile ownership
obvious:

- RGC: one 9x9 plane with visible 3x3 patch boundaries;
- L1: nine column centers arranged 3x3 and aligned with their RGC patches;
- within each L1 column: ordinary E cells in a compact ring/grid, Eor toward the next
  layer, C on a feedback/basal side, and I on the inhibitory side;
- L2: one column centered above/downstream of the L1 tile array;
- within L2: the same local motif and role offsets as every other column;
- top L2 C remains visible and labeled dormant rather than silently omitted.

The layout algorithm must be parameterized by column metadata and `N`; it must not
assume eight E cells, exactly two layers, or one-character layer numbers. Repeated
columns use the same local motif translated to a column center.

Separate RNG concerns carefully:

- legacy preset layout and weight goldens must remain bit exact;
- adding tiled nodes must not consume RNG draws used by legacy presets;
- the tiled layout must be deterministic for a seed;
- if layout jitter influences learning, it is scientific state and must be reproducible;
- frontend-only spacing must never feed back into backend learning distances.

At `N=8`, no two generated nodes may occupy the zero placeholder or the same functional
position. Camera fitting must encompass the entire hierarchy.

## Serialization and runtime diagnostics

Static topology payloads must include the tiling/input metadata described above.
Dynamic payloads must remain backward compatible and add tiled fields rather than
changing existing field meanings.

The current single `winner` field cannot describe ten independent local WTAs. Preserve
it for existing presets, but add a tiled diagnostic such as:

```json
"column_winners": {
  "L1...": {"id": "...E3", "tau": 0.42},
  "L2...": {"id": "...E1", "tau": 0.77}
}
```

Requirements:

- record only ordinary E winners, not Eor or C;
- at most one ordinary E winner per column per boundary;
- independent columns may all be present in the mapping;
- reset the mapping at boundary start;
- derive it from actual emitted spikes and column metadata, not from membrane maxima;
- preserve exact/near tie diagnostics from the event scheduler;
- preserve `hard_reset_events` with source I, target E, boundary, and `tau`;
- expose Eor as an ordinary plastic E in neuron diagnostics;
- expose C basal/apical/deposit diagnostics unchanged;
- expose a top-C dormant/parent-count marker through static metadata.

Do not overload the persistent legacy `winner` highlight to mean a whole-column state.

## Dashboard requirements

The dashboard must remain usable for both the new 191-node default graph and all
existing presets. It must derive dimensions, groups, labels, and receptive-field shapes
from topology metadata.

Add `tiled_cc` as a selectable preset, but do not change the repository's current
dashboard startup preset merely as a side effect of this implementation. A startup
default change requires a separate explicit decision.

### Input panel

- Replace the hard-coded nine-cell creation loop with `topology.grid` or explicit
  input-shape metadata.
- Render all 81 cells as a 9x9 grid for `tiled_cc`.
- Draw strong boundaries between the nine 3x3 patches.
- Toggling a cell must address its global row-major pixel index.
- Visually indicate the selected patch used by local pattern buttons.
- Local row/column/diagonal buttons embed into the selected patch and send a complete
  81-element vector.
- Current 3x3 behavior must remain unchanged for legacy presets.
- RGC input flashing must be found through pixel ownership metadata, not an assumed
  `L1E${i}` ID.

### 3D topology view

- Render column grouping from `column_id`/`column_role` metadata.
- Add column/layer filters or selection focus so 1052 edges do not become an
  unreadable hairball.
- At minimum allow focusing one column and its parent/children while retaining a way
  to show the whole hierarchy.
- Distinguish ordinary E, Eor, C, and I in labels/tooltips while retaining excitatory
  versus inhibitory color semantics.
- Highlight winners per column from `column_winners`; do not show only one global halo.
- Structural edges remain visible under the weak-weight filter.
- Edge flashes continue to use actual `emitted` edge IDs.
- Existing preset filters and rendering remain functional.

### Activity/heat-map view

Replace the assumption of one global L2 competitor bank with per-column activity:

- one compact ordinary-E heat map per column;
- current winner label per column;
- Eor, C, and I status adjacent to that column;
- columns grouped by layer and tile coordinates;
- no implication that the most recently reported legacy winner fired this boundary.

For legacy presets, retain the existing L2 heat map or render an equivalent one-column
compatibility view.

### Spike raster and charge chart

The raster and charge views currently create one lane per neuron, which is unwieldy at
191 nodes. Add metadata-driven grouping/filtering:

- select all, one layer, one column, or one role;
- keep stable ordering by layer, column coordinates, column role, and local index;
- clearly label Eor, C, and I lanes;
- retain rolling history semantics and do not drop simulation frames merely because a
  group is hidden;
- keep C immediate spike rendering honest: a full-height spike marker represents an
  emitted spike, while end-of-boundary charge may already be reset;
- preserve legacy display behavior.

The chart only samples outer-boundary state. Do not claim it proves same-`tau` timing;
the inspector/runtime diagnostics carry `spike_tau` and deposit `tau`.

### Weights-over-time view

- Permit selecting every plastic ordinary E and every Eor, not only roles currently
  named `competitor`/`encoder`.
- Include C basal as a distinct single-weight target or provide an explicit C-basal
  mode.
- Label afferents from metadata: retinal coordinate for RGC, local E index for Eor
  inputs, and child column ID/coordinate for parent ordinary-E inputs.
- Use the actual shared E cap for ordinary E/Eor and the C-specific cap for basal.
- Changing targets may continue to start a new browser-side history, but label that
  behavior honestly; do not call the 1500-sample buffer “full training history.”
- Preserve manual weight editing and correct clipping.

### Receptive-field view

Remove the fixed `N_PIX = 9`, fixed top-three, and single global pixel-map assumptions.
Render a target according to its incoming projection:

- L1 ordinary E: a 3x3 local retinal patch using global pixel metadata;
- L2 ordinary E: a 3x3 grid of the nine child-column Eor inputs;
- Eor: its `N` local ordinary-E afferents in a compact metadata-labeled grid/list;
- C: one basal weight plus a read-only list/count of unweighted apical sources, or route
  C to the inspector rather than pretending apicals are weighted RF cells.

Any “dead” or reachability badge must use the target's actual afferents and declared
input semantics. The old `top3 < threshold` heuristic must not be applied blindly to
Eor or arbitrary `N`. If a correct topology-generic reachability test is not available,
omit the badge for the new roles and state why.

### Inspector and event log

Show:

- column ID, layer, tile position, role, and local E index;
- incoming projection family and learned weights;
- per-boundary column winner state;
- Eor ordinary membrane/learning diagnostics;
- C basal weight, basal eligibility, apical sources, deposit count/charge/`tau`, and
  dormant status;
- I one-shot relay state and hard-reset targets;
- event log messages that name column and role without parsing IDs.

## Topology editor and preset persistence

The new built-in must appear in preset lists and load into the topology editor. The
editor need not become a specialized cortical-column authoring wizard in this phase,
but it must not destroy tiling metadata on round trip.

Required behavior:

- built-in `tiled_cc` loads with its real metadata-driven positions;
- current spec export preserves top-level, node, and edge projection metadata;
- saving and reloading a tiled custom preset preserves input dimensions and columns;
- loading a saved 81-input tiled preset resizes the engine input safely;
- malformed edited tiled graphs fail validation with a column-specific error;
- legacy user presets remain loadable;
- reserved built-in naming includes `tiled_cc`;
- preset listing computes node/edge counts using each preset's resolved construction
  dimensions rather than the currently active engine's `n_pix` by accident.

## Backward compatibility requirements

This work is additive. The following are hard requirements:

- all five current built-ins retain their existing node/edge counts at defaults;
- all existing 9/8 golden snapshots remain bit exact unless a test is demonstrably
  asserting an incorrect dashboard label rather than simulation behavior;
- legacy synchronous presets remain on their synchronous path;
- `rg_coincidence` remains event-resolved and keeps its immediate C deposit semantics;
- existing single global `winner` behavior remains available to legacy consumers;
- existing 3x3 patterns and controls remain unchanged when a 3x3 preset is active;
- existing generic non-default-size tests continue to pass;
- existing custom NetworkSpecs that contain no tiled metadata validate as before;
- adding the new layout path consumes no RNG draws in legacy construction;
- no current edge kind changes meaning;
- no existing neuron class receives preset-name-specific branches.

Do not rewrite existing preset builders through the new column abstraction as part of
this task. They model different circuits and are protected by regression tests.

## Expected repository touchpoints

Trace actual call sites before editing; do not treat this list as permission to bypass
an affected path. The main expected areas are:

- `backend/network_spec.py`: preset registry, reusable tile/connectors, archetype/edge
  compatibility, metadata preservation, tiled-family validation;
- `backend/simulation.py`: preset construction dimensions, event-plastic E
  registration/learning, generic hierarchy dispatch, column-winner diagnostics,
  initialization/distance scope, topology serialization;
- `backend/layout.py`: separate metadata-driven tiled functional layout while preserving
  the legacy generator and RNG behavior;
- `backend/presets.py`: built-in listing/loading, per-preset dimensions, saved metadata
  round trips;
- `backend/dashboard_config.py`: new selector entry and structural `cc_e_count` control;
- `backend/api.py`, `backend/serializer.py`, and `backend/websocket.py`: only where
  construction-size changes or additive topology/dynamic fields require transport;
- `snn/neurons.py`: ordinary E/C equations should normally remain unchanged; edit only
  if a genuinely generic registration hook is required, never to add Eor-specific math;
- `frontend/controls.js`: topology-sized input grid, patch selection, metadata-based RGC
  flashing, topology-specific patterns;
- `frontend/renderer.js`: column grouping/focus, role rendering, per-column winners,
  large-edge filtering;
- `frontend/charts.js`, `frontend/raster.js`, `frontend/charge.js`,
  `frontend/weights.js`, `frontend/receptive.js`, and `frontend/inspector.js`:
  column-aware dashboard semantics;
- `frontend/index.html` and `frontend/style.css`: patch/column controls, layout, legends,
  and scalable views;
- `experiments/`: deterministic tiled smoke/acceptance probe and machine-readable output;
- `tests/`: new builder, validation, event, learning, input, layout, protocol, API,
  preset, and frontend/static-contract coverage;
- `README.md` and current technical methodology documentation: new topology and exact
  semantics.

Do not key execution behavior to `tiled_cc` when topology metadata or archetype
capabilities can express it generically. Preset-name branching is appropriate only at
the centralized preset-construction/dimension-selection boundary.

## Implementation phases

### Phase 0: Baseline and invariants

Before implementation:

1. Run the full suite and record the baseline pass count.
2. Identify golden/serialization tests for all five presets.
3. Add no behavior changes in this phase.
4. Keep the unrelated working-tree changes intact.

### Phase 1: Declarative tile builder and metadata

1. Add the reusable column handle/tile builder.
2. Add RGC-patch and child-parent connector functions.
3. Add the canonical `tiled_cc` builder with `N` configurable.
4. Add/preserve top-level, node, and edge projection metadata.
5. Add tiled-family structural validation.
6. Add node/edge count and exact connectivity tests.
7. Keep it constructible without running the dashboard.

### Phase 2: Generic event-plastic E execution

1. Ensure ordinary E and Eor instantiate the same E implementation.
2. Generalize event-path learning to every event-resolved plastic E.
3. Build generic source-indexed feedforward dispatch at arbitrary depth.
4. Keep feedforward/basal delay one and apical/relay/reset zero latency.
5. Add local column-winner diagnostics.
6. Support top C with explicit dormant metadata and zero apicals.
7. Verify one-shot shared-I behavior exactly as specified.

### Phase 3: Initialization, geometry, and headless input

1. Add target-fan-in-aware normalized initialization for tiled event E rows.
2. Add projection/target-scoped distance normalization without changing legacy math.
3. Add preset-specific dimension resolution and `cc_e_count`.
4. Add the 9x9 input shape and pure patch embedding helper.
5. Add a headless smoke experiment and machine-readable diagnostics.

### Phase 4: Tiled functional layout

1. Add a metadata-driven layout for arbitrary column count/depth and `N`.
2. Preserve the legacy layout path and RNG order.
3. Serialize all positions and group metadata.
4. Add non-overlap, determinism, and hierarchy tests.

### Phase 5: Dashboard generalization

1. Make the input grid topology-sized and patch-aware.
2. Add column-aware 3D focus/filtering and winner display.
3. Add per-column activity summaries.
4. Add raster/charge column filters.
5. Make weights and RF views recognize ordinary E, Eor, and C basal.
6. Update inspector/event log semantics.
7. Verify all legacy dashboard views still load without JavaScript errors.

### Phase 6: Documentation and full verification

1. Update README/current methodology with the new preset and exact counts.
2. Document Eor as an ordinary learned E, not Boolean OR.
3. Document the dormant top C and the shared one-shot I consequence.
4. Document that multi-winner composition and pure queue scheduling remain deferred.
5. Run focused tests, headless experiment, full suite, and frontend smoke checks.
6. Report measured behavior honestly; do not tune hidden constants solely to force a
   desired plot.

## Required focused tests

### Builder and validation

- default `N=8` yields exactly 191 nodes and 1052 edges;
- arbitrary `N` follows `10N+111` nodes and `129N+20` edges;
- each column has exactly `N` E, one Eor, one C, one I;
- each L1 patch contains exactly nine unique RGCs;
- RGC-to-column connectivity is complete within a patch and absent across patches;
- every internal edge set is exact;
- each L1 Eor reaches all L2 ordinary E;
- all L2 ordinary E reach each L1 C apically;
- no Eor is an apical source;
- L2 C has one basal and zero apicals and is valid only because it has no parent;
- malformed missing/extra/cross-column edges are rejected;
- metadata survives validate/export/save/load round trips.

### Neuron parity and learning

- ordinary E and Eor with identical state/input have identical crossing, spike, reset,
  trace, and weight-update numerics;
- Eor learns its causal local-E volley on its own firing;
- RGC-targeted ordinary E learns while RGC remains immutable;
- L2 ordinary E learns only child Eors delivered to that target/boundary;
- inactive afferents are not confused across targets or hierarchy hops;
- C basal uses its pre-update weight for deposit and post-fire learning remains local;
- top C weight never changes.

### Local WTA and event timing

- at most one ordinary E spikes per column per boundary;
- two independent L1 columns can each emit one winner in the same boundary;
- the first local E crossing recruits only its local I;
- local I resets its entire local ordinary-E bank, including the emitted winner;
- no reset target belongs to another column;
- Eor is not reset by local I;
- C can recruit the same local I;
- I remains one-shot if E and C both drive it in one boundary;
- exact ties within one column use stable event ordering and still yield one winner;
- exact ties across independent columns allow both columns to produce a winner at the
  same `tau`;
- every feedforward hop adds one outer-boundary delay;
- parent-E apical delivery and a resulting mature child-C spike share `tau`;
- top C never deposits or fires over a long basal-only run.

### Input and configuration

- `SimulationEngine(topology='tiled_cc')` owns an 81-element input;
- global pixel 0 and pixel 80 map to the correct RGCs/patches;
- patch embedding maps every local index correctly for all nine patches;
- a full 9x9 diagonal activates the correct three diagonal subpatches;
- a local pattern activates only the selected patch's RGCs;
- `cc_e_count` changes every column consistently;
- reset/reseed preserve 81/N;
- dashboard switching `legacy -> tiled_cc -> legacy` resolves dimensions correctly;
- invalid input vector lengths and invalid patch coordinates fail loudly.

### Layout and serialization

- every tiled node has a nonzero explicit position;
- positions are deterministic for a seed and distinct within a column;
- column centers follow declared layer/tile coordinates;
- public topology contains input, patch, column, role, and projection metadata;
- dynamic state contains per-column actual winners;
- legacy topology and dynamic payload keys remain valid.

### Dashboard smoke

- 9x9 input renders 81 clickable cells with 3x3 patch separators;
- legacy input renders nine cells;
- selecting a patch embeds local pattern buttons correctly;
- 191 nodes and 1052 edges load without a frontend exception;
- focusing a column makes its local motif inspectable;
- multiple same-boundary column winners are visible;
- ordinary E, Eor, and C basal weights can be selected/charted/edited appropriately;
- raster and charge filters do not lose hidden history;
- top C is visibly dormant rather than missing;
- switching among every built-in preset leaves the dashboard operational.

## Scientific acceptance experiment

Run at least one deterministic headless isolation experiment with only the center 3x3
patch active. It must demonstrate mechanical, not necessarily fully trained scientific,
correctness:

1. Only the center patch's RGCs emit.
2. Only the paired center L1 column receives RGC feedforward events.
3. Its ordinary E bank obeys local hard WTA.
4. Its emitted ordinary-E event reaches local Eor one boundary later.
5. Eor owns and updates its incoming weights when it fires.
6. An Eor spike reaches every L2 ordinary E one boundary later.
7. L2 obeys its own local hard WTA.
8. The L2 winner supplies unweighted apical permission to all nine L1 C cells, but only
   C cells with local basal eligibility can deposit.
9. C recruits its own column's shared I only.
10. The L2 top C remains dormant.
11. No inactive L1 column receives a hard reset from the active column's I.

Then run a two-patch mechanical probe and confirm that two L1 columns can each produce a
local winner while the one L2 column remains single-winner. This is not the deferred
row-plus-column multi-winner composition experiment; it validates independent tiled
columns under current local WTA.

Report boundaries, spike IDs/roles, column winners with `tau`, emitted edges, reset
targets, C deposits, and weight changes. A dashboard screenshot is supplemental; the
headless trace is the authoritative acceptance evidence.

## Explicit non-goals

Do not implement in this task:

- more than one ordinary-E winner inside a column per boundary;
- row-plus-column composition learning;
- fixed or adaptive k-WTA;
- `delta_tau` co-winner admission;
- learned lateral inhibition;
- lateral E-to-E or CC-to-CC connections;
- residual-evidence competition;
- continuous recurrent settling;
- a priority-queue discrete-event scheduler;
- special Boolean OR behavior for Eor;
- special fixed RGC-to-E weights;
- feedback from Eor;
- apical input to the top C without an actual parent column;
- rewriting legacy presets as cortical columns;
- hiding malformed graphs with supervisory repair passes.

## Definition of done

The work is complete only when:

1. `tiled_cc` is a real built-in preset generated from reusable tile/connector rules.
2. The default graph has exactly 191 nodes and 1052 explicit directed edges.
3. Headless experiments can construct, drive, step, inspect, reset, and reseed it.
4. `N` is configurable without editing topology code.
5. The RGC-to-column and child-to-parent rules are structurally validated.
6. Eor is numerically the same ordinary plastic E and differs only by connectivity.
7. Every column uses current immediate hard WTA through its own shared I.
8. Parent ordinary E, never Eor, supplies child C apical feedback.
9. The top C is explicitly valid and observably dormant.
10. Per-column winners and resets are causally observable.
11. The 9x9/patch hierarchy is usable in the dashboard without fixed-3x3 assumptions.
12. Layout and views are column-aware and remain usable at the default graph size.
13. Saved specs and topology-editor round trips preserve all tiling metadata.
14. All existing presets, goldens, APIs, experiments, and dashboard views remain
    functional.
15. Focused tests, the headless acceptance trace, the full suite, and frontend smoke
    checks pass.
16. Documentation clearly separates this single-winner tiled hierarchy from the
    deferred multi-winner composition and pure discrete-event work.
