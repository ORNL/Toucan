---
title: CLI and Library API
parent: Reference
nav_order: 3
---

# CLI and Library API

## Command line

The installed executable accepts one settings file:

```sh
mpiexec -n <ranks> /path/to/bin/Toucan ToucanSettings.json
```

Toucan initializes MPI and Kokkos, so use an MPI launcher for single- and
multi-rank runs. Rank 0 parses the JSON and broadcasts the materialized settings
to every rank. The process working directory is the base for relative thermal
paths, Condor inputs, the Condor `cache/` directory, and a relative output
directory.

During a run, rank 0 prints each layer and the thermal/microstructure stage.
Every rank prints its DECA iteration and active steering-vector size.

## Aggregate header

Include the installed aggregate header and link the CMake target
`Toucan_Core`:

```cpp
#include <Toucan_Core.hpp>
```

The application owns MPI and Kokkos initialization/finalization. Construct
Toucan objects only while both runtimes are active.

## Parse and broadcast settings

```cpp
Toucan::IO::FileReader input;
if (rank == 0) {
  input.Initialize("ToucanSettings.json");
}
input.Broadcast(0, MPI_COMM_WORLD);
```

`FileReader::Initialize` accepts either a filename or an already parsed
`nlohmann::json` object. It separates the top-level blocks into
`irf_json`, `rng_json`, `domain_json`, `mpi_json`,
`nucleation_json`, `substrate_json`, `thermal_json`, and
`output_json`. `Pack`, `Unpack`, and `Broadcast` carry
that parsed state between MPI ranks.

## Select stored grain fields

Grain metadata is a compile-time template choice. The executable reads the two
`Output/Grains` booleans first, then instantiates one alias:

| Alias | State propagated through capture, substrate transfer, and MPI |
|:--|:--|
| `BaseGrainIdentifier` | Orientation ID only. |
| `UniqueOnlyGrainIdentifier` | Orientation ID, repeat ID, and nucleating-rank ID. |
| `UnderresolvedOnlyGrainIdentifier` | Orientation ID and under-resolution flag. |
| `UniqueUnderresolvedGrainIdentifier` | All fields. |

The same `GrainID` alias must be used for `Mpi_Dual`,
`Substrate_Dual`, thermal routing, initialization, and every layer call.

## Construct simulation state

```cpp
template<template<class> class GrainID>
int run(const Toucan::IO::FileReader& input) {
  Toucan::Structs::Mpi_Dual<GrainID> mpi(MPI_COMM_WORLD);
  Toucan::Structs::Sim_Dual sim;
  Toucan::Structs::RNG_Dual rng;
  Toucan::Structs::Orientations_Dual orientations;
  Toucan::Structs::Substrate_Dual<GrainID> substrate;

  sim.rank = mpi.rank;
  // ...
}
```

`Mpi_Dual` creates a non-periodic two-dimensional Cartesian topology.
`Sim_Dual` holds host/device simulation settings and output controls.
`RNG_Dual` owns random-number state and the shuffled orientation list.
`Orientations_Dual` stores orientation bases and face projections.
`Substrate_Dual` stores the two-window rolling grain state.

## Obtain thermal RDF

The built-in JSON router returns a polymorphic source:

```cpp
auto thermal = Toucan::Thermal::Routing<float, Toucan::device_space, GrainID>
(
    input.thermal_json, mpi
);

auto rdf = thermal->GetRDF(layer);
```

`Routing` always recognizes `Thermal/File`. It recognizes `Thermal/Condor` when
Toucan was configured with `TOUCAN_ENABLE_CONDOR=ON`; a Condor-disabled build
reports a targeted runtime error if an input requests that source.
For another thermal solver, derive from `ThermalSource` or construct RDF
directly in the driver; see the
[coupling contract]({{ '/reference/thermal-sources/#coupling-contract' | relative_url }}).

## Initialize once

Obtain the first RDF before initialization:

```cpp
auto first_rdf = thermal->GetRDF(0);

Toucan::Run::Initialize(
    input, mpi, substrate, sim, orientations, rng, first_rdf);
```

`Initialize` reads simulation settings using the first RDF header, creates
the orientation table, establishes MPI bounds, allocates the rolling substrate,
and generates its two initial windows. It also writes `Orientations.csv`
on rank 0.

## Advance layers

The application uses three signals for each layer:

```cpp
Toucan::Run::FromLayer(
    mpi, substrate, sim, orientations, rng,
    Toucan::Run::CoupledSignal::READ, rdf);

Toucan::Run::FromLayer(
    mpi, substrate, sim, orientations, rng,
    Toucan::Run::CoupledSignal::SIMULATE);

Toucan::Run::FromLayer(
    mpi, substrate, sim, orientations, rng,
    Toucan::Run::CoupledSignal::CLEAR);
```

| Signal | Operation |
|:--|:--|
| `READ` | Compare the layer header with the first header, apply MPI decomposition, copy RDF fields into a cellular-automata grid, order repeated events at each site, and retain the grid. |
| `SIMULATE` | Initialize the layer from substrate/nucleation state, run DECA until the global steering set is empty, transfer final site states to the substrate, and schedule completed-window output. |
| `CLEAR` | Release all retained layer grids. |
| `DONE` | Release grids, schedule remaining output regions, and wait for the final asynchronous write. |

`SIMULATE` selects retained grids cyclically by the substrate layer number.
The executable uses one `READ`, `SIMULATE`, and `CLEAR`
sequence per layer.

After the loop, finish output before destroying state:

```cpp
Toucan::Run::FromLayer(
    mpi, substrate, sim, orientations, rng,
    Toucan::Run::CoupledSignal::DONE);
```

All ranks must make lifecycle calls in the same order because construction,
simulation, and output selection contain MPI collectives and neighbor
communication.
