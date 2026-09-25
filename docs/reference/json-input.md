---
title: JSON Input
parent: Reference
nav_order: 1
---

# JSON Input

The `Toucan` executable accepts the path to one JSON document. Rank 0 reads the
file, materializes any supported inline/file references, validates the blocks,
and broadcasts the parsed data to all MPI ranks.

JSON does not support comments. Field names and string values are
case-sensitive.

## Top-level structure

```json
{
  "IRF": {},
  "RNG": {},
  "Domain": {},
  "MPI": {},
  "Nucleation": {},
  "Substrate": {},
  "Thermal": {},
  "Debug": {},
  "Output": {}
}
```

`IRF`, `RNG`, `Domain`, `MPI`, `Nucleation`, `Thermal`, and `Output` are used by
the command-line workflow. `Substrate` and `Debug` are optional and supply
defaults when omitted. See [Thermal sources]({{ '/reference/thermal-sources/' | relative_url }})
for the contents of `Thermal`.

## `IRF`

The interface response function (IRF) relates growth velocity to undercooling.
It can be an inline object:

```json
"IRF": {
  "function": "power",
  "coefficients": {
    "A": 1.0e-7,
    "B": 2.0
  }
}
```

or a path to a JSON file containing that object:

```json
"IRF": "material/IN718.json"
```

| Field | Required | Type | Meaning |
|:--|:--|:--|:--|
| `function` | yes | string | Must be `power`, representing \\(V=A(\Delta T)^B\\). |
| `coefficients/A` | yes | number | Power-law leading coefficient. |
| `coefficients/B` | yes | number | Power-law exponent. |

## `RNG`

```json
"RNG": {
  "Seed": 12345,
  "Orientation": 12345,
  "Substrate": 67890
}
```

| Field | Required | Default | Meaning |
|:--|:--|:--|:--|
| `Seed` | no | `0` | Shared fallback seed. |
| `Orientation` | no | `Seed` | Seed for orientations and the nucleation RNG stream. |
| `Substrate` | no | `Orientation` | Seed for the shuffled substrate orientation list. |

The `RNG` object itself is required and must be nonempty.

## `Domain`

```json
"Domain": {
  "NumberOfLayers": 20,
  "LayerOffset": 0.00004,
  "WindowHeight": 0.0002
}
```

| Field | Required | Type | Meaning |
|:--|:--|:--|:--|
| `NumberOfLayers` | yes | unsigned integer | Number of microstructure layers to advance. |
| `LayerOffset` | yes | number | Vertical distance in meters between layers. |
| `WindowHeight` | yes | number | Vertical distance in meters represented by one rolling substrate window. |

The grid spacing and x-y bounds come from the first RDF header rather than this
block.

## `MPI`

```json
"MPI": {
  "Mode": "Strong",
  "Comm": "Host"
}
```

| Field | Required | Values | Meaning |
|:--|:--|:--|:--|
| `Mode` | yes | `Strong`, `OneToOne` | `Strong` divides one supplied RDF domain across the two-dimensional process grid. `OneToOne` uses the per-rank RDF partition. |
| `Comm` | yes | `Host`, `Device` | Selects host-staged or device communication buffers. |

Toucan obtains the two-dimensional process-grid dimensions with
`MPI_Dims_create` and uses non-periodic x-y neighbors.

## `Nucleation`

```json
"Nucleation": {
  "Density": 1.0e12,
  "MeanUndercooling": 5.0,
  "StdUndercooling": 1.0
}
```

| Field | Required | Type | Meaning |
|:--|:--|:--|:--|
| `Density` | yes | number | Volumetric nucleation-site density in m\\(^{-3}\\). |
| `MeanUndercooling` | yes | number | Mean of the critical-undercooling distribution in kelvin. |
| `StdUndercooling` | yes | number | Standard deviation of that distribution in kelvin. |

Toucan multiplies `Density` by the cubic RDF grid spacing to obtain the
per-cell nucleation probability.

## `Substrate`

```json
"Substrate": {
  "MeanBaseplateGrainSize": 0.00004,
  "NumGrainOrientations": 1000
}
```

| Field | Default | Meaning |
|:--|:--|:--|
| `MeanBaseplateGrainSize` | `1.0e-5` | Nominal baseplate grain size in meters. |
| `NumGrainOrientations` | `65534` | Number of crystallographic orientation bases to generate. |

Toucan converts the baseplate grain size to a number of grid cells using the
RDF spacing. It writes every generated orientation basis to
`Orientations.csv`.

## `Debug`

```json
"Debug": {
  "MaxIterations": 100
}
```

| Field | Default | Meaning |
|:--|:--|:--|
| `MaxIterations` | effectively unlimited | Maximum number of DECA iterations performed for each layer. |

This optional block limits each layer independently. A value of `0` skips DECA
propagation, while `1` performs iteration zero and then stops. When the limit
is reached, Toucan continues with the partially advanced grid so it can be
inspected through the normal output path.

## `Output`

```json
"Output": {
  "Directory": "ToucanOutput",
  "Format": "xdmf",
  "Selection": {
    "Dims": "XYZ",
    "Stride": { "X": 2, "Y": 2, "Z": 1 },
    "Offset": { "X": 0, "Y": 0, "Z": 0 }
  },
  "Grains": {
    "UniqueIdentifiers": true,
    "UnderResolved": true
  }
}
```

| Field | Required | Default | Meaning |
|:--|:--|:--|:--|
| `Directory` | yes | — | Output directory, created when settings are initialized. |
| `Format` | no | `xdmf` | `xdmf` or `csv`. |
| `Selection/Dims` | yes | — | `None`, `XYZ`, `XY`, `XZ`, or `YZ`. |
| `Selection/Stride` | no | `1` per axis | Positive output sampling interval. |
| `Selection/Offset` | no | `0` per axis | Starting global index for the sampling sequence. |
| `Grains/UniqueIdentifiers` | no | `false` | Include repeat and nucleating-rank IDs. |
| `Grains/UnderResolved` | no | `false` | Include the stored under-resolution flag. |

`Selection` and `Grains` are required objects. `Stride` and `Offset` each accept
either one unsigned integer applied to all axes or an object containing any of
the case-sensitive `X`, `Y`, and `Z` keys. For every explicitly paired axis,
`Offset` is less than `Stride`.

`XYZ` emits sampled volumes. `XY`, `XZ`, and `YZ` emit planes. When the normal
axis is not explicitly selected, `XZ` and `YZ` use the center index, while `XY`
uses the layer spacing along z. `None` disables microstructure-region output;
the orientation table is still generated during initialization.
