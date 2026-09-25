---
title: Output
parent: Reference
nav_order: 4
---

# Output

Toucan writes an orientation table during initialization and rank-local
microstructure regions as rolling substrate windows become complete. Output
coordinates use the RDF origin/resolution and are written in the same spatial
units as the input (meters for Toucan's documented JSON interface).

## Orientation table

Rank 0 writes `Orientations.csv` in the configured output directory. Each row
contains:

```text
dirID,d1x,d1y,d1z,d2x,d2y,d2z,d3x,d3y,d3z
```

The three vectors form one randomly generated orthonormal orientation basis.
Rows are written in orientation-array order. The current orientation file labels
those rows `1` through `NumGrainOrientations`, while grain arrays store
their orientation index in `dirID`.

## Region selection

`Output/Selection/Dims` controls geometry:

| Value | Result |
|:--|:--|
| `XYZ` | One sampled three-dimensional region per completed window. |
| `XY` | One or more constant-z planes. |
| `XZ` | One or more constant-y planes. |
| `YZ` | One or more constant-x planes. |
| `None` | No microstructure-region files. |

`Stride` samples indices independently along x, y, and z. `Offset`
selects indices congruent to that offset modulo the stride. Plane filenames
include `.Slice.Xn`, `.Slice.Yn`, or `.Slice.Zn` before the
extension. Output is rank-local:

```text
Rank.<rank>.Output.<window>.csv
Rank.<rank>.Output.<window>.xmf
Rank.<rank>.Output.<window>.<attribute>.bin
```

Slice suffixes appear between `<window>` and the extension. Rank and window
numbers are zero-padded according to the MPI size and configured layer count.
Internal MPI halo cells are omitted from the selected rank region.

## CSV

CSV output produces one file per selected rank-local region. Its base columns
are:

```text
x,y,z,dirID
```

Coordinates are physical positions. Enabling `UniqueIdentifiers` appends
`repeatID` and `nucleatedRankID`. Enabling `UnderResolved`
appends `underResolved` as `0` or `1`. A baseplate grain uses
`-1` for the CSV nucleating-rank field. CSV writes only substrate cells
that have received a simulated grid state.

## XDMF

XDMF output produces an `.xmf` metadata file plus raw binary arrays. The XDMF
topology is a 3D rectilinear mesh with origin and spacing derived from the RDF
header and output stride. The always-present node attribute is `dirID`.

Optional attributes are:

- `repeatID` and `nucleatedRankID` when `UniqueIdentifiers` is enabled;
- `underResolved` when `UnderResolved` is enabled.

Each attribute is stored as a contiguous signed 32-bit integer array in a
sibling file named from the region base plus the attribute name and `.bin`.
The x index varies fastest in these output arrays. Integer value `-1`
represents cells outside the transferred grid and baseplate nucleating-rank
IDs.

## Grain identity

`dirID` identifies crystallographic orientation and can repeat across grains.
When unique identifiers are enabled, the tuple of direction ID, repeat ID, and
nucleating rank is the propagated grain identity used by Toucan. Nucleated
grains receive a repeat counter for their orientation on their MPI rank;
baseplate grains use their generated repeat index and have no nucleating rank.
The under-resolution field is set when the selected capture time precedes the
captured cell's liquidus-crossing time, then propagates with the grain state.
