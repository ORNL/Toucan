---
title: How Toucan Works
parent: User Guide
nav_order: 3
---

# How Toucan Works

Toucan simulates the evolution of grain-structure in a parallel-in-time (PinT) fashion. Its thermal input is not a temperature field at every time step. Instead, each RDF event describes one melt/cool cycle at one Cartesian-grid site. Knowing all events a priori allows
Toucan to build a space-time cellular-automata graph from those events and then advance grain capture events through that graph.

## 1. Obtain reduced thermal data

Each RDF record supplies a local flat grid index, melting time, liquidus
crossing time during cooling, and positive cooling-rate magnitude. Multiple
records can refer to the same spatial site when it remelts. Toucan obtains RDF
in one of two built-in ways:

- `Thermal/File` reads an existing RDF, or reads SRDF and asks Stork to
  interpolate and trim it.
- When Toucan is built with `TOUCAN_ENABLE_CONDOR=ON`, `Thermal/Condor` runs
  Condor in memory to generate RDF or SRDF. SRDF follows the same Stork
  interpolation path.

The first RDF header establishes the cellular-automata spacing and x-y domain.
Later layers must retain those grid properties. A coupled application can also
construct RDF directly; the complete producer/consumer contract is in
[Thermal sources]({{ '/reference/thermal-sources/#coupling-contract' | relative_url }}).

## 2. Initialize material and domain state

Toucan creates random orthonormal crystallographic orientation bases, an x-y
MPI decomposition, and a circular substrate buffer two windows deep. It fills
the initial baseplate with block-shaped grains at the configured nominal grain
size, assigning orientations through a shuffled orientation list. Independent
RNG seeds control orientation/nucleation generation and the substrate shuffle.

`Strong` MPI mode partitions a supplied RDF domain across the automatically
chosen two-dimensional Cartesian process grid. `OneToOne` treats each rank's
RDF as its local partition. Neighbor ranks exchange cellular-automata state at
x-y partition boundaries; `Comm` selects host or device communication buffers.

## 3. Advance DECA events

For a layer, Toucan initializes a Kokkos-resident grid from the RDF events and
the current substrate. A thermal event becomes a nucleation site with probability

\\[
p = \rho_n\,\Delta x^3,
\\]

where \\(\rho_n\\) is `Nucleation/Density` and \\(\Delta x\\) is the RDF grid
spacing. Its critical undercooling is sampled from the configured normal
distribution and converted to a nucleation time using the local cooling rate.

DECA keeps a steering vector of cells whose capture events can affect their 26
neighbors. Each iteration computes candidate capture times from the grain
geometry, local cooling history, and the power-law interface response

\\[
V = A(\Delta T)^B.
\\]

Here \\(V\\) is interface velocity, \\(\Delta T\\) is undercooling, and
\\(A\\) and \\(B\\) are the power-law coefficients. Cells adopt the
grain associated with the earliest candidate capture. MPI boundary state is
synchronized between iterations, and an all-reduce determines when no rank has
active steering events. This event-driven progression is the parallel-in-time
method described by the DECA and Toucan papers.

## 4. Preserve state and write output

For every spatial site, Toucan orders repeat events by melt time and connects
each event to its later event at that site. After a layer finishes, the last
event at each site is copied into the rolling substrate, so the next layer
inherits its grain state. As the build moves upward, Toucan reuses the two
substrate windows and asynchronously writes completed regions. The [output reference]({{ '/reference/output/' | relative_url }}) describes the volume and slice selectors, file formats, and optional grain identity fields.

## Performance portability

Toucan represents its simulation arrays with Kokkos views and performs
nucleation, event processing, scans, reductions, state transfer, and packing in
Kokkos kernels. The active `Kokkos::DefaultExecutionSpace` becomes Toucan's
device space at compile time. MPI owns decomposition and inter-rank transfer;
the `Host` and `Device` communication settings control whether messages are
staged through host mirrors.

The same source can therefore be built for different Kokkos backends, but each
binary uses the backends enabled in the Kokkos installation against which it
was compiled.
