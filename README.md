# Toucan

Toucan is a performance-portable C++ application and header-only library for
predicting grain structure during additive manufacturing solidification. It
uses Kokkos and MPI to advance cellular automata from reduced thermal histories
without stepping through a complete temperature field.

Thermal data can be read from Stork RDF or SRDF files. Toucan can also run
Condor in memory when built with Condor support. Its thermal-source interface
allows coupled applications to provide RDF data directly.

The implementation and its scaling behavior are presented in
[Stump et al. (2025)](https://doi.org/10.1016/j.commatsci.2025.113684).

## Documentation

The full installation guide, quick start, input reference, thermal coupling
guide, and API documentation are on the
[Toucan documentation site](https://ornl.github.io/Toucan/).

## Installation

Toucan requires Kokkos, MPI, OpenMP, Stork, and nlohmann/json. Condor is
optional. See the
[installation guide](https://ornl.github.io/Toucan/docs/installation/) for
dependency and configuration details.

After setting the dependency paths in `make.sh`, build and install with:

```sh
./make.sh
```

Set `TOUCAN_ENABLE_CONDOR=OFF` during CMake configuration for a file-input-only
build without Condor.

## Usage

Run Toucan with an MPI launcher and a JSON settings file:

```sh
mpiexec -n <ranks> ./build/install/bin/Toucan ToucanSettings.json
```

A downstream CMake project can import the installed header target with:

```cmake
find_package(ToucanCore CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE Toucan_Core)
```

## Contributors

- [Benjamin Stump](https://www.ornl.gov/staff-profile/benjamin-c-stump)
- [Matt Rolchigo](https://www.ornl.gov/staff-profile/matt-rolchigo)
- [Daniel Arndt](https://www.ornl.gov/staff-profile/daniel-arndt)
- [Sam Reeve](https://www.ornl.gov/staff-profile/sam-t-reeve)

## Contributing

Contributions are welcome, including new thermal sources, coupling workflows,
validation cases, performance improvements, and output capabilities.
