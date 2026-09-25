#!/bin/bash

# Clean previous build
rm -rf build
clear

# Set environment variables
export NVCC_WRAPPER_DEFAULT_COMPILER=mpic++

# Set environment variable for Stork installation path
export STORK_DIR=$HOME/stork/build/install

# Set environment variable for 3DThesis installation path
export CONDOR_DIR=$HOME/condor/build/install

# Set environment variable for MPI installation path
export MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi

# Set environment variable for the JSON library
export JSON_DIR=$HOME/json/build/install

# Create build directory
mkdir -p build
pushd build

# CMake configuration
cmake \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX=install \
  -D CMAKE_CXX_FLAGS="-fopenmp -O3 -ffast-math -march=znver3 -mtune=znver3" \
  -D CMAKE_PREFIX_PATH="${KOKKOS_DIR};${MPI_DIR};${STORK_DIR};${CONDOR_DIR};${JSON_DIR}" \
  -D TOUCAN_ENABLE_CONDOR=ON \
  -D CMAKE_CUDA_ARCHITECTURES="86" \
  ..

# Build the project
make -j24 install  # Use all available CPU cores for faster compilation

# Return to original directory
popd
