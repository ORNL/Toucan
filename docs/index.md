---
title: Home
nav_order: 1
description: Toucan is a performance-portable, parallel-in-time cellular automata application for additive manufacturing microstructure simulation.
---

![Toucan](assets/images/toucan-wordmark.png){: .hero-logo }

Toucan is a C++17 application and header-only library for predicting grain
structure during additive manufacturing solidification. It implements the
Discrete Event inspired Cellular Automata (DECA) algorithm with Kokkos and MPI,
and advances the cellular automata from reduced thermal histories supplied via the Reduced Data Format (RDF).

[Install Toucan](docs/installation/){: .btn .btn-primary }
[Quick start](docs/quick-start/){: .btn }
[Read the papers](about/papers/){: .btn }

## What Toucan provides

- A performance portable, MPI-enabled implementation of the DECA algorithm for grain growth.
- A heterogenous grain nucleation model based on local undercooling.
- Kokkos data structures and kernels for CPU and accelerator execution spaces.
- Two-dimensional MPI decomposition with host- or device-buffer communication.
- An abstract thermal-input interface (`ThermalSource`) whose implementations
  must return RDF:
  - It is currently coupled to Stork for RDF and SRDF file input.
  - It is currently coupled to Condor for in-memory thermal solves.
- Extensible grain identifier types that make it easy to track orientation IDs
  and, when requested, unique grains through repeat and nucleating-rank IDs.
- Layer-by-layer state transfer through a moving substrate window.
- CSV or XDMF output, with volume or planar selection and optional grain
  metadata.

## At a glance

| Need | Start here |
|:--|:--|
| Build the executable and installable headers | [Installation](docs/installation/) |
| Run from a JSON settings file | [Quick start](docs/quick-start/) |
| Understand the simulation workflow | [How Toucan works](docs/how-toucan-works/) |
| Write a complete settings file | [JSON input](reference/json-input/) |
| Select a thermal source | [Thermal sources](reference/thermal-sources/) |
| Integrate Toucan in C++ | [Library API](reference/api/) |
| Interpret generated files | [Output](reference/output/) |
| Cite DECA or Toucan | [Papers](about/papers/) |

## Contributors

- [Benjamin Stump](https://www.ornl.gov/staff-profile/benjamin-c-stump)
- [Matt Rolchigo](https://www.ornl.gov/staff-profile/matt-r-rolchigo)
- [Daniel Arndt](https://www.ornl.gov/staff-profile/daniel-arndt)
- [Sam Reeve](https://www.ornl.gov/staff-profile/sam-t-reeve)


## License

Toucan source and these original documentation assets are distributed under the
repository's BSD 3-Clause License.