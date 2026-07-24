The Wrong Machine:
Why Numerical Computation Fails at Event-Driven Systems,
and the Unexpected Benefits of a New Computational Paradigm

=================================================================================

I. The Problem

There exists a class of computational problems that resists every machine we
have built. They are not exotic -- they arise in any system where many
components are coupled through sparse, event-driven interactions, where the
state of one component depends on the precise timing of events from others,
and where the pattern of coupling shifts over time. Neural circuits are the
canonical example, but the pattern recurs across domains: agent-based
economic models where trades propagate through a network of interacting
firms, network security simulations where alerts cascade through
infrastructure graphs, discrete-event physical systems such as collision
detection in granular materials, and epidemiological models where individual
infection events ripple through contact networks.

These systems share a defining shape. Computation is driven not by a regular
clock sweeping over a dense array, but by the arrival of discrete events at
unpredictable locations and times. The graph of interactions is sparse,
dynamic, and cyclic -- feedback loops are the norm. And the semantics of the
system demand causal fidelity: an event at t=100.1 must be resolved before an
event at t=100.2, because the earlier event may alter the conditions under
which the later event is evaluated.

At the heart of the matter is a representational chasm. These systems
represent information in a way that has no native expression in the
machines we have built. Closing this gap is not a question of faster clocks
or wider vector units. It requires a different computational substrate.


II. Dense Binary Versus Causal Events

Every computer we have constructed treats presence and absence as equals.
A 0 is a bit. A 1 is a bit. Both occupy memory. Both consume a clock cycle.
To know that nothing happened at position 7, you must visit position 7 and
read the 0. The 0 costs as much as the 1 -- same fetch, same register,
same instruction. The hardware makes no distinction between "neuron fired"
and "neuron did not fire." Both are data.

An event is not a bit. It is a causal assertion. When a neuron fires, it
generates a spike -- a signal that says this symbol is active at this
moment. When a neuron does not fire, it generates nothing. There is no
zero to read, no address to decode, no memory to fetch. The non-event is
not a representation. It is the null state from which events emerge.

This is not an optimization layered on a binary substrate. It is a
different algebra. In a dense binary system, information is measured in
bits and every possible state occupies a location in memory. In an
event-driven system, information is measured in causal assertions, and
only the events that actually occur consume any representation at all.
The 99% of a network that is quiescent at any moment costs nothing --
not because we optimized it away, but because it was never there to
begin with.

The consequence for simulation is stark. On current hardware, a network
of 100,000 nodes must visit all 100,000 at every timestep to discover
that only 1,000 generated events. The discovery itself burns the budget.
On a GPU, a warp of 32 threads executes in lockstep: if one thread has
work, all 32 spin through the same instruction stream while 31 sit idle.
Sparsity becomes waste, and waste becomes the dominant cost.


III. Spatial Representation: Location as Information

In every computational system we have built, location is incidental. A
value at address 0x1000 and the same value at address 0x2000 are
functionally identical. The arithmetic logic unit does not ask where its
operands came from. The programmer does not care which register held the
intermediate result. Location is a retrieval mechanism, never meaning.
If two computations produce the same bit pattern, they are
interchangeable regardless of the paths taken.

In an event-driven spatial system, location IS information. Two neurons
with identical dynamics, identical threshold, and identical weights are
not the same entity -- because they occupy different positions in the
connectivity graph. Their inputs arrive from different sources. Their
outputs travel to different targets. When L1_FE[4] fires, the fact that
it was neuron 4 and not neuron 7 is part of the signal. The identity of
the sender cannot be stripped away without destroying what the event
means.

This runs deeper than distinguishing one neuron from another. The path a
signal takes to reach its destination carries information. If L1_CFE[3]
receives a spike that traveled L1_FE[2] -> L1_CFE[3] versus one that
arrived via L2_FE[5] -> L1_CFE[3], those are different events even if
the final voltage deflection is identical. The routing history is
content. Different paths represent different causal chains, different
contexts, different meanings.

At the highest level, a spike represents a symbol -- and the symbol is the
product of all the paths that converged on that neuron. It is not "output
equals f of input." It is: this event, at this specific location, arrived
via these specific pathways through the network, and that exact
combination of spatial identity and causal history IS the representation.
The symbol is embodied in the geometry of the network. It cannot be
extracted and stored as a value without destroying what it means.

This is an information channel our machines have never used. A spike
carries no number -- it is one bit of presence -- and yet it can
represent anything. Happy. Cup. Hungry. The concept of justice. The
representational capacity is not bounded by the bit-width of the events.
It is bounded by the combinatorial space of possible paths through the
connectivity graph -- a space that grows exponentially with network depth,
not linearly with bit precision. The spike is not a value. It is a
pointer to an entire causal graph.

Current hardware cannot express this. A centralized processor reading
passive memory destroys the spatial information in the act of retrieval.
The address tells you where to fetch, but the fact that the data was
*there* and not *somewhere else* is lost in translation. We have built
machines where value is the sole carrier of meaning, identity is
transparent, and location is a routing artifact. A machine that treats
location as a first-class computational primitive -- where the wire
carries meaning, not just data -- would open a channel of representation
that our paradigm cannot access.


IV. Time: Synchrony as Binding

There is a second dimension that carries meaning in these systems, and it
is equally foreign to our machines: time.

In an event-driven spatial system, events that occur at the same moment
are bound together. If L1_FE[0] and L1_FE[5] both fire at t=100.0, then
those two symbols are co-active, coupled by their temporal coincidence.
The synchrony itself carries information. It says: these events belong
together. They were produced by the same stimulus, the same context, the
same moment in the world.

This is not the time of a computer. A digital program has no concept of
how long computation takes. Run it at 10 Hz or 10 GHz -- the same
instructions execute, the same bits emerge. Time is a cost parameter
to be minimized, not a semantic dimension. The program's output contains
no trace of the duration of its execution.

In a system bound to real-world events, time is meaning. An event at
t=100.1 occurs after an event at t=100.0, and the ordering is causal --
the earlier event may have caused the later one. Downstream coincidence
detectors read the temporal proximity of their inputs and use it to make
decisions: fire if both inputs arrived within 2 ms, remain silent if
they are 10 ms apart. The time is not an implementation artifact. It is
the substrate of computation.

This is fundamentally incompatible with the way our hardware approaches
time. A von Neumann machine advances through a program counter, not
through real-world moments. An event-driven simulator must reconstruct
causal ordering from timestamps -- a priority queue that sorts events
chronologically before processing them. The heap can deliver correctness:
100,000 nodes generating temporally synchronous events in different
portions of the network will be processed in the correct order. But it
does so by layering an event chronology on top of a machine that has no
native concept of time. The overhead of that translation -- heap
operations, pointer-chasing through sparse graphs, cache-incoherent
delivery -- consumes nearly all of the available throughput. The machine
is not computing the network's dynamics. It is computing the scheduling
of events on a substrate that was never designed to represent them.

A native event-driven architecture would not simulate time. It would
advance through it. Events would arrive at their targets at wire speed,
with latency determined by the physical layout of the circuit rather
than the depth of a software queue. Synchrony would be a physical
property of the system, not a data field on a packet. The binding of
events through temporal coincidence would emerge from the hardware rather
than being imposed by a scheduler.


V. The Architecture This Demands

The mismatch is not a performance gap. It is a representational gap.

We have built machines where memory, computation, and communication are
separate -- where a central processor fetches passive data, operates on
it, and writes it back. The event-driven spatial paradigm fuses these
into a single substrate. A neuron is simultaneously a memory (its
weights), a processor (its threshold dynamics), and a router (its
outgoing connections). The spike is not a packet to be delivered. It is
a voltage on a wire. There is no address to decode, no pointer to chase,
no heap to manage. The connectivity IS the wiring.

This architecture would represent information through three channels that
our current machines cannot express:

First, the event itself -- a causal assertion, not a bit. Presence
without absence. The null state costs nothing because it is not
represented.

Second, spatial location -- the wire IS the information. Which neuron
fired, and which path the signal took, cannot be separated from the
meaning of the event. Identity and content are the same thing.

Third, synchrony -- events that occur together are bound together. Time
is not a cost parameter but a semantic dimension. The system advances
through real moments, and the temporal structure of its operation IS
the computation.

A machine built on these principles would not merely accelerate neural
simulation. It would represent a new class of computational device -- one
in which the problems we now simulate at enormous cost become the natural
idiom, and the representational channels of spatial identity and temporal
coincidence become available for the first time in engineered hardware.


VI. Beyond Numbers: Space and Time as Untapped Information Channels

There is a larger claim here, and it is worth stating directly: space
and time are not merely implementation details that a better architecture
would handle more efficiently. They are information channels in their own
right -- channels that current computation has never used, and that may
prove more powerful than the numerical representation we have relied on
for seventy years.

Consider what a number can do. A 32-bit float encodes a single scalar
value. It is self-contained: the value 0.734 carries its meaning
internally, independent of where it is stored or how it was computed.
It is a point. A single number, one location in a vast address space.

Now consider what a spike in a spatial architecture can do. The spike
itself is one bit of presence. But the identity of the neuron that fired
is not one bit -- it is log(N) bits of selection across the entire
network, and that selection was produced by every event that preceded
it. The same single spike can represent "cup" or "hungry" or "danger"
depending on where in the network it occurred, and "where" is not a
label attached to the spike -- it IS the spike. The representation is
not a value stored at an address. It is an address that carries value.

This inverts the relationship between content and container. In a
numerical system, the container (memory address) is a retrieval handle
for the content (the value). In a spatial system, the container IS the
content. You do not look up what neuron 473 means. Neuron 473 means
neuron 473 -- the sum of every causal path that converges there. The
spatial structure of the network is not a substrate that hosts
computation. It is itself the representation.

Time adds a second orthogonal dimension. When two neurons fire at the
same moment, they are coupled -- their synchrony asserts a relationship
between two symbols that would not exist if they fired at different
times. This binding through temporal coincidence is a form of
information that has no equivalent in a serial processor. A CPU can
compute that event A and event B both occurred. It cannot make them
*occur together* as a physical fact of the system's operation. The
simultaneity is simulated, not instantiated. In a native event-driven
architecture, the simultaneity is real -- it is a physical property
of parallel wires carrying spikes in the same clock cycle. The
hardware does not record that A and B were synchronous. It experiences
their synchrony.

Together, space and time form a representational space that is
categorically different from numerical computation. A 32-bit number
selects one of four billion values. A system with N spatially distinct
units and T temporal resolution selects from a combinatorial space of
paths through a graph -- a space that grows with the topology of the
network, not the bit-width of its registers. The same hardware resources
that store a single float in a conventional machine could, in a spatial
event-driven architecture, encode a symbol whose meaning is the product
of thousands of causal events distributed across both space and time.

This is not a quantitative advantage. It is a qualitative one. We are
not proposing a faster way to do arithmetic. We are pointing to a form
of representation that numbers cannot capture -- a form that biology has
used for hundreds of millions of years and that our machines have never
accessed. The fact that it has gone untapped is not evidence of its
irrelevance. It is evidence that we have been building one kind of
machine, and this kind requires another.


VII. Conclusion

The gap cannot be closed by incremental improvement. Faster clocks, wider
SIMD units, and deeper caches address the wrong problem. The problem is
not that our hardware is too slow. It is that we are using a machine
built for dense, array-shaped, address-based computation to simulate a
system that is sparse, event-driven, spatially-grounded, and causally
ordered -- and the translation between these two models consumes
essentially all of the available capacity.

More than this: the translation strips away the very channels of
information that make the system powerful. The spatial identity of the
sender, the causal path the signal traveled, the temporal coupling of
co-active events -- all are discarded by the act of reducing a spike to
a data packet. We are not just simulating inefficiently. We are
simulating the wrong thing.

The architecture this problem deserves does not yet exist. But its shape
is becoming clear. It would be a machine where events drive computation,
where wires carry meaning, where time is not simulated but inhabited,
and where the information channels of space and synchrony are available
as first-class computational primitives. It would not be a faster
version of what we have. It would be a different class of machine
entirely -- and the problems it would unlock may extend far beyond the
neural circuits that first revealed the gap.