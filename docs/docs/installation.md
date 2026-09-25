---
title: Installation
parent: User Guide
nav_order: 1
---

# Installation

Toucan is a C++17 header-only library with a command-line executable. The build
installs the public headers, CMake package files, and `Toucan` executable.

## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- [Kokkos](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)
- MPI with C++ support
- OpenMP
- [Stork](https://github.com/ORNL/Stork)
- [Condor](https://github.com/ORNL/Condor), optional and required only for in-memory thermal solves
- [nlohmann/json](https://github.com/nlohmann/json)

Kokkos determines Toucan's default host and device execution spaces. Configure
Kokkos for the CPU or accelerator backend that the coupled application will
use. Kokkos, MPI, OpenMP, Stork, and nlohmann/json are always required. Condor
is required only when `TOUCAN_ENABLE_CONDOR` is enabled.

## Configure and build

From the repository root:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/stork;/path/to/condor;/path/to/json" \
  -DTOUCAN_ENABLE_CONDOR=ON
cmake --build build --parallel
cmake --install build
```

Add MPI's package prefix to `CMAKE_PREFIX_PATH` when CMake cannot find it through
the compiler wrapper or the system package configuration. Accelerator builds
also require the appropriate Kokkos compiler wrapper and architecture flags.

### Build without Condor

Condor support is enabled by default. Disable it for a file-input-only build:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/stork;/path/to/json" \
  -DTOUCAN_ENABLE_CONDOR=OFF
cmake --build build --parallel
cmake --install build
```

With `TOUCAN_ENABLE_CONDOR=ON`, configuration fails if CMake cannot find
Condor. With it set to `OFF`, Toucan does not search for or link Condor and the
`Thermal/File` source remains available. Selecting `Thermal/Condor` in an input
file passed to a Condor-disabled build reports that Toucan must be rebuilt with
`-DTOUCAN_ENABLE_CONDOR=ON`.

The installed command is:

```sh
./install/bin/Toucan
```

With no settings argument, it prints `Usage: Toucan <ToucanSettings.json>`.

## Use the repository helper

The included `make.sh` records one ORNL developer-machine configuration. It
cleans `build/`, uses dependency locations under the invoking user's home
directory, applies AMD CPU and CUDA architecture flags, and builds with 24
jobs. Review those paths, compiler flags, and architecture choices before using
the helper on another system. The explicit CMake commands above are the
portable starting point.

## Consume the installed headers

The installed package is named `ToucanCore` and exports the target
`Toucan_Core`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(toucan_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ToucanCore CONFIG REQUIRED)

add_executable(toucan_driver main.cpp)
target_link_libraries(toucan_driver PRIVATE Toucan_Core)
```

Configure the consumer with Toucan and its dependencies on
`CMAKE_PREFIX_PATH`. `ToucanCoreConfig.cmake` requests Kokkos, MPI, OpenMP,
Stork, and nlohmann/json before loading the exported target. It requests Condor
only when that Toucan installation was built with Condor support. The package
sets `ToucanCore_WITH_CONDOR` to `ON` or `OFF`, and the exported target
propagates the corresponding `TOUCAN_ENABLE_CONDOR=1` or
`TOUCAN_ENABLE_CONDOR=0` compile definition to consumers.
