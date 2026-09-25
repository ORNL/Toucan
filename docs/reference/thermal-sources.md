---
title: Thermal Sources
parent: Reference
nav_order: 2
---

# Thermal Sources

Every Toucan layer is driven by a Stork Reduced Data Format (RDF) object:

```cpp
Stork::Structs::RDF_Dual<float>
```

The command-line application can read Stork files or, when built with Condor
support, run Condor in memory. A coupled application can instead produce the
same RDF object directly. SRDF is an input to Stork interpolation; RDF is the
interface consumed by Toucan.

## Built-in source selection

The top-level `Thermal` object selects exactly one source:

```json
{
  "Thermal": {
    "File": {}
  }
}
```

or:

```json
{
  "Thermal": {
    "Condor": {}
  }
}
```

The `Condor` selection is available only when Toucan is configured with
`-DTOUCAN_ENABLE_CONDOR=ON`, which is the default. A build configured with
`-DTOUCAN_ENABLE_CONDOR=OFF` supports `File` without requiring Condor and
reports a targeted runtime error if an input requests `Condor`.

Both source classes implement:

```cpp
Stork::Structs::RDF_Dual<ToucanFloat> GetRDF(uint32_t layer);
```

For layer 0, the application calls `GetRDF(0)`, uses that object for one-time
simulation initialization, and then reads it as the first layer. It calls
`GetRDF(layer)` once for each later layer in increasing order. A source with
fewer thermal inputs than Toucan layers cycles through its configured
layer/file sequence.

## Stork files

`Thermal/File` reads RDF directly or reads SRDF and uses Stork to interpolate
it to RDF:

```json
"Thermal": {
  "File": {
    "String": "thermal/rank-RANK/layer-LAYER",
    "Layers": [0, 19],
    "Ranks": 4,
    "Precision": "double",
    "SRDF": {
      "FineFactor": 4,
      "NormalizeTimes": false,
      "TimeScale": 0.01,
      "Reuse": true
    }
  }
}
```

| Field | Form/default | Meaning |
|:--|:--|:--|
| `String` | string or array of strings | One or more file patterns. |
| `Files` | array of strings | Alternative to `String`. |
| `Layers` | integer count, or `[start, stop]`; default `1` | File-layer numbers. A count `N` means `0` through `N-1`; a pair is inclusive. |
| `Ranks` | integer count, or `[start, stop]`; default MPI size | File-rank numbers, mapped to MPI ranks in sequence. |
| `Precision` | `float` or `double`; default `double` | Floating type stored in the Stork file. Toucan converts it to its `float` RDF type. |
| `RDF` | object | Select direct RDF input. |
| `SRDF` | object | Select SRDF input followed by Stork interpolation and trimming. |

Select one of `RDF` and `SRDF`. The older `"Format": "RDF"` or
`"Format": "SRDF"` selector is also accepted, with mode settings placed in the
`File` object.

Settings used by either mode:

| Field | Default | Meaning |
|:--|:--|:--|
| `Reuse` | `false` | Retain loaded RDF or SRDF objects and reuse them when the configured layer sequence cycles. |
| `NormalizeTimes` | `false` | Normalize event times after loading the RDF or SRDF. |
| `TimeScale` | `0.01` | Maximum time gap passed to the selected Stork normalization operation. |

Settings used only by SRDF:

| Field | Default | Meaning |
|:--|:--|:--|
| `FineFactor` | required | Integer spatial refinement factor passed to Stork; use `1` through `255`. |

`Reuse`, `NormalizeTimes`, and `TimeScale` may be inside the selected mode
object or at the `File` level; the mode-object value takes precedence. When an
`SRDF` object selects the mode, its required `FineFactor` must be inside that
object. With the legacy `"Format": "SRDF"` selector, place `FineFactor` at the
`File` level.

### File patterns

Toucan appends `.stork` when the pattern omits it. `RANK` and
`LAYER` are decimal-number tokens and can occur in a directory component or
filename:

```text
thermal/rank-RANK/layer-LAYER.stork
thermal/rank-RANK-layer-LAYER.stork
```

Relative patterns are resolved from the process working directory. Absolute
patterns are resolved from the filesystem root. The selected rank number and
layer number determine the matching file. Each path component can contain at
most one token of each kind, and adjacent `RANKLAYER` or `LAYERRANK`
tokens need literal text between them.

Direct RDF input is optionally normalized with
`Stork::Run::Normalize_RDF_Times` and copied into Toucan's execution space.
SRDF is loaded, optionally normalized with
`Stork::Run::Normalize_SRDF_Times`, copied to the thermal execution space,
interpolated and trimmed to the rank's x-y partition, and trimmed in z.

## In-memory Condor

This source requires a Toucan build configured with
`-DTOUCAN_ENABLE_CONDOR=ON`. `Thermal/Condor` runs Condor from a sequence of
parameter inputs:

```json
"Thermal": {
  "Condor": {
    "Files": [
      "condor/layer0.json",
      "condor/layer1.json"
    ],
    "Precision": "double",
    "SRDF": {
      "FineFactor": 4,
      "NormalizeTimes": true,
      "TimeScale": 0.01,
      "Reuse": false,
      "Cache": true
    }
  }
}
```

| Field | Form/default | Meaning |
|:--|:--|:--|
| `Files` | required array of strings | Condor parameter inputs. Simulation layers cycle through the array. |
| `Precision` | `float` or `double`; default `double` | Floating type used for Condor SRDF generation. Direct RDF always has Toucan's `float` type. |
| `RDF` | object | Run Condor directly into RDF. |
| `SRDF` | object | Run Condor into SRDF, then convert it with Stork. |

Select one of `RDF` and `SRDF`. Their mode objects accept:

| Field | Default | Meaning |
|:--|:--|:--|
| `Reuse` | `false` | Retain generated data for later cycles through the input list. |
| `Cache` | `false` | Read or write a Stork binary cache under `cache/` in the working directory. |

For SRDF, `FineFactor` is required and uses the range `1` through `255`,
`NormalizeTimes` defaults to `true`, and `TimeScale` defaults to `0.01`.
Normalization uses `Stork::Run::Normalize_SRDF_Times`.

Cache stems contain the Condor input stem and a hash of its normalized path,
MPI size, Cartesian dimensions, coordinates, and rank. RDF and SRDF use
different suffixes. A matching cache is read instead of running Condor.

## Coupling contract

This section is the contract for a thermal code that supplies data without
going through the built-in File or Condor sources.

### Interface and type

The extension point is:

```cpp
template<class ToucanFloat, class ToucanSpace>
class Toucan::Thermal::ThermalSource {
public:
  using DualRDF = Stork::Structs::RDF_Dual<ToucanFloat>;
  virtual ~ThermalSource() = default;
  virtual DualRDF GetRDF(uint32_t layer) = 0;
};
```

Toucan's executable uses `ToucanFloat = float` and
`ToucanSpace = Toucan::device_space`. `Thermal::Routing` is only the
JSON factory for the two built-in sources; it does not register custom source
names. A custom coupled driver constructs its provider directly and feeds each
returned RDF through the Toucan layer lifecycle.

### RDF header

Both `rdf.host_header` and `rdf.device_header` must contain the same
regular-grid metadata before Toucan reads the layer:

| Accessor | Type | Meaning |
|:--|:--|:--|
| `global_i0()`, `global_j0()`, `global_k0()` | `uint32_t` | Global indices represented by local index `(0,0,0)`. |
| `local_inum()`, `local_jnum()`, `local_knum()` | `uint32_t` | Local node counts in x, y, and z. |
| `global_x0()`, `global_y0()`, `global_z0()` | `float` | Physical coordinates of global grid index `(0,0,0)`. |
| `gridResolution()` | `float` | Uniform Cartesian spacing. |

Toucan combines these values as:

\\[
(x,y,z)=(x_0,y_0,z_0)+\Delta x(i,j,k).
\\]

Spatial values must use meters to match `Domain/LayerOffset`,
`Domain/WindowHeight`, and `Substrate/MeanBaseplateGrainSize`.

The first layer fixes `global_i0`, `global_j0`, `global_x0`,
`global_y0`, `local_inum`, `local_jnum`, and
`gridResolution`. Every later layer must preserve those values. Layer z
origins and extents may change. The layer check requires
`global_z0 >= -Domain/WindowHeight`.

### RDF events and layout

Set `rdf.numEvents` to the logical event count. Allocate enough storage and
populate every entry in `[0, numEvents)`: one `p` and three floating values per
event. Toucan uses `numEvents`, not view capacity, as the iteration bound.

| Accessor | Meaning |
|:--|:--|
| `p(n)` | Local flat point index. |
| `tm(n)` | Time of the upward melt crossing. |
| `tl(n)` | Time of the downward liquidus crossing during cooling. |
| `cr(n)` | Positive cooling-rate magnitude for the downward crossing. |

Local indices are zero-based and flatten with z fastest:

\\[
p=i(N_jN_k)+jN_k+k.
\\]

Each `p(n)` must identify a node inside the header extents. A site can have
multiple records for remelting; Toucan groups same-site records and orders them
by `tm` when it creates the space-time grid, so producer record order is not
used as the cycle order. Each record represents melting followed by cooling
through liquidus, so its crossings are ordered `tm(n) < tl(n)`. Toucan uses the
cooling-rate magnitude to convert a sampled critical undercooling into
nucleation time:

\\[
t_\mathrm{nucl}=t_l+\frac{\Delta T_\mathrm{crit}}{cr}.
\\]

RDF carries no unit metadata. Times, cooling rates, nucleation undercooling, and
the IRF coefficients must use one consistent time/temperature system. With
undercooling in kelvin and time in seconds, `cr` is in K/s and `A`
must give velocity in m/s when applied in the configured power law.

### Memory spaces and ownership

Toucan reads initialization and consistency values from `host_header`. It
reads event indices, event values, and the device header from the device side
during grid construction. A producer that fills on the host can establish the
required device data with:

```cpp
RDF rdf;
rdf.numEvents = count;
rdf.template Make_Data_Views<Toucan::host_space>(count);

// Fill rdf.host_header and rdf.host_data[0..count).

rdf.template Make_Data_Mirrors<Toucan::host_space, Toucan::device_space>();
rdf.template Copy_All<Toucan::host_space, Toucan::device_space>();
```

`Make_Data_Views` and `Make_Data_Mirrors` allocate without filling
values; the producer fills every logical entry before the copy. A producer that
generates RDF on the device instead must still copy the header to
`host_header` before `Initialize` or `READ`.

`GetRDF` returns the RDF object by value. Its Kokkos views retain their
allocations through the returned object. Do not modify or recycle those
allocations until the corresponding `READ` call returns. `READ` accepts
the RDF by non-const reference and may replace its device data handles while
performing MPI decomposition; the simulation grid owns the copied fields after
`READ`.

### MPI partition contract

`Mpi_Dual` creates a non-periodic, two-dimensional Cartesian communicator
with `MPI_Dims_create`. Thermal production and Toucan lifecycle calls occur
on every rank.

- In `Strong` mode, Toucan decomposes the supplied x-y RDF domain over that
  Cartesian grid.
- In `OneToOne` mode, each rank supplies its local RDF partition. At an
  internal x or y boundary, Toucan treats the outer plane in that direction as
  the neighbor/halo plane and owns the remaining local range.

In `Strong` mode, every rank supplies the same RDF domain for Toucan to
partition. In `OneToOne` mode, each rank preserves its own required x-y header
values across layers and the per-rank partitions align at their shared halo
planes. All ranks call the layer lifecycle in the same collective order. A
file-based OneToOne run uses `Ranks` and `RANK` to map one Stork partition to
each MPI rank.

### Call order and lifetime

Initialize MPI and Kokkos before constructing Toucan, Stork, or Kokkos-view
objects. The supported simulation order is:

```cpp
auto rdf0 = thermal.GetRDF(0);
Toucan::Run::Initialize(
    input, mpi, substrate, sim, orientations, rng, rdf0);

for (uint32_t layer = 0; layer < sim.hostSim.numLayers(); ++layer) {
  auto rdf = (layer == 0) ? rdf0 : thermal.GetRDF(layer);
  Toucan::Run::FromLayer(
      mpi, substrate, sim, orientations, rng,
      Toucan::Run::CoupledSignal::READ, rdf);
  Toucan::Run::FromLayer(
      mpi, substrate, sim, orientations, rng,
      Toucan::Run::CoupledSignal::SIMULATE);
  Toucan::Run::FromLayer(
      mpi, substrate, sim, orientations, rng,
      Toucan::Run::CoupledSignal::CLEAR);
}

Toucan::Run::FromLayer(
    mpi, substrate, sim, orientations, rng,
    Toucan::Run::CoupledSignal::DONE);
```

`Initialize` is called once using the first RDF. For each layer, `READ`
constructs and retains the simulation grid, `SIMULATE` consumes that grid and
updates the substrate, and `CLEAR` releases retained grids. `DONE`
writes remaining output and waits for the last asynchronous output operation.
Keep all simulation objects alive until it returns; finalize Kokkos and MPI
afterward.

### Minimal producer

This skeleton shows the data handoff. The thermal solver supplies the header
and event values:

```cpp
class MyThermalSource final
    : public Toucan::Thermal::ThermalSource<
          float, Toucan::device_space> {
public:
  using RDF = Stork::Structs::RDF_Dual<float>;

  RDF GetRDF(uint32_t layer) override {
    RDF rdf;
    const uint32_t count = event_count(layer);
    rdf.numEvents = count;

    auto& h = rdf.host_header;
    fill_header(layer, h);

    rdf.template Make_Data_Views<Toucan::host_space>(count);
    for (uint32_t n = 0; n < count; ++n) {
      rdf.host_data.p(n)  = local_point(layer, n);
      rdf.host_data.tm(n) = melt_time(layer, n);
      rdf.host_data.tl(n) = liquidus_time(layer, n);
      rdf.host_data.cr(n) = cooling_rate_magnitude(layer, n);
    }

    rdf.template Make_Data_Mirrors<Toucan::host_space, Toucan::device_space>();
    rdf.template Copy_All<Toucan::host_space, Toucan::device_space>();
    
    return rdf;
  }
};
```

Producing Stork RDF `.stork` files and selecting `Thermal/File/RDF` is
the equivalent file-based coupling path.
