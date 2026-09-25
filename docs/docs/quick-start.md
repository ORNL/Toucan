---
title: Quick Start
parent: User Guide
nav_order: 2
---

# Quick Start

## 1. Choose thermal input

Prepare either:

- Stork `.stork` files containing RDF or SRDF data; or
- Condor parameter input files that a Condor-enabled Toucan build will execute
  in memory.

The complete examples in [Thermal sources]({{ '/reference/thermal-sources/' | relative_url }})
show both forms. The compact example below reads one RDF file named
`thermal/layer0.stork`.

## 2. Write `ToucanSettings.json`

```json
{
  "IRF": {
    "function": "power",
    "coefficients": { "A": 1.0e-7, "B": 2.0 }
  },
  "RNG": {
    "Seed": 12345,
    "Orientation": 12345,
    "Substrate": 67890
  },
  "Domain": {
    "NumberOfLayers": 1,
    "LayerOffset": 0.00004,
    "WindowHeight": 0.0002
  },
  "MPI": {
    "Mode": "Strong",
    "Comm": "Host"
  },
  "Nucleation": {
    "Density": 1.0e12,
    "MeanUndercooling": 5.0,
    "StdUndercooling": 1.0
  },
  "Substrate": {
    "MeanBaseplateGrainSize": 0.00004,
    "NumGrainOrientations": 1000
  },
  "Thermal": {
    "File": {
      "String": "thermal/layer0",
      "Layers": 1,
      "Ranks": 1,
      "RDF": { "Reuse": false },
      "Precision": "float"
    }
  },
  "Output": {
    "Directory": "ToucanOutput",
    "Format": "xdmf",
    "Selection": {
      "Dims": "XYZ",
      "Stride": 1,
      "Offset": 0
    },
    "Grains": {
      "UniqueIdentifiers": false,
      "UnderResolved": false
    }
  }
}
```

`IRF` may instead be a filename containing the IRF object. Numeric values above
illustrate the schema; use material and process values appropriate to the
simulation.

## 3. Run

From the case directory, where relative paths in the settings file resolve:

```sh
mpiexec -n 1 /path/to/toucan/install/bin/Toucan ToucanSettings.json
```

Rank 0 reads and validates the JSON, then broadcasts the parsed settings to the
other ranks. Toucan creates the configured output directory, writes
`Orientations.csv`, advances each layer, and writes the selected microstructure
regions in CSV or XDMF form.

## Next steps

- Use [JSON input]({{ '/reference/json-input/' | relative_url }}) for field meanings and defaults.
- Use [Thermal sources]({{ '/reference/thermal-sources/' | relative_url }}) for layer/rank
  patterns, SRDF interpolation, Condor coupling, caching, and reuse.
- Use [Output]({{ '/reference/output/' | relative_url }}) to select volumes, planes, and
  optional grain metadata.
