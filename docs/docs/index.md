---
title: User Guide
nav_order: 2
has_children: true
---

# User Guide

Use this guide to build Toucan, run the command-line application, and understand
how thermal histories become grain-structure output.

1. [Install Toucan](installation/) and its Kokkos, MPI, Stork, OpenMP, and
   nlohmann/json dependencies, plus Condor when in-memory thermal solves are
   enabled.
2. Follow the [quick start](quick-start/) to run `Toucan` with a JSON input.
3. Read [How Toucan works](how-toucan-works/) for the RDF-to-grain simulation
   pipeline and DECA layer lifecycle.

The [reference](../reference/) documents every active top-level JSON block,
thermal-source variants and coupling contract, output fields, and the C++ API.
