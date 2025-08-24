## Hotspot Benchmark

This directory contains the Hotspot (2D transient heat diffusion) benchmark based on Rodinia, with two implementations:
- `Managed/`: Uses CUDA Unified Memory (UM), supports `cudaMemAdvise`/`cudaMemPrefetchAsync`, and provides a `multistream` build target.
- `UnManaged/`: Uses explicit device memory and copies.

Both versions include wall-clock timing; at program end it prints: `Total elapsed time: X.XXX s`.

Note: The `multistream` in this implementation is mainly for multi-stream prefetch of managed arrays; kernels are launched on the default stream by default. The `Managed/Makefile` provides the `multistream` target; `prefetch` is an alias for compatibility.

## Workload Overview

### Mathematical Model and Algorithm (kept detailed)
- **Continuous equation**  

```math
\rho c \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + P $$
```

  where $T$ is temperature, $P$ is local power, and $\rho,c,k$ are density, specific heat, and thermal conductivity.
- **Explicit finite difference (FDM)**  
  Partition the chip into a $G\times G$ grid with time step $\Delta t$. Use a **4-neighbor (4-neigh) stencil** to approximate the Laplacian:
  
```math
T_{i,j} \leftarrow T_{i,j}
 + \kappa\,\frac{\Delta t}{\Delta x^{2}}
 \left[\,T_{i+1,j}+T_{i-1,j}+T_{i,j+1}+T_{i,j-1}-4T_{i,j}\,\right]
 + \text{power term}
```

- **Stability condition**  
  The explicit scheme requires $\kappa\,\Delta t/\Delta x^2 \le 0.25$. The program uses constants that satisfy this condition.
- **Boundary condition**  
  The chip edge is modeled with convective cooling; coefficients are embedded and need no user configuration.

### GPU Parallel Mapping
- **Threads/blocks**: Each 16×16 tile is processed by one thread block; each thread handles one grid point.
- **Shared memory buffering**: Threads within the same block share neighborhood data to reduce global accesses.
- **Time tiling – `pyramid_height (H)`**: Each kernel computes H steps in sequence, reducing write-backs; shared memory needs grow with H.
- **Multi-stream (`MULTISTREAM`)**: Create multiple CUDA streams between data prefetch and computation to overlap migration and compute.

### 4-neigh stencil and `pyramid_height`
````text
    (i-1,j)
       ↑
(i,j-1) ← (i,j) → (i,j+1)
       ↓
    (i+1,j)
````
- When `pyramid_height = 1`, only the above 4-neighborhood is needed.
- When `pyramid_height = H`, the dependency expands outward by H layers, forming an inverted pyramid.

### Use Cases
- Quickly evaluate hot spots of SoC / 3D-IC.
- Provide input to heatsink design, power scheduling, and timing–thermal co-analysis.


## Program Input
Hotspot uses **plain ASCII floating-point text**; one value per line, stored in **row-major** order.

| File | Role | Lines | How to generate |
|------|------|-------|-----------------|
| `temp_1024`  | Initial temperature (°C) | `G×G` | `data/inputGen/hotspotex.cpp` |
| `power_1024` | Power (W/cell) | `G×G` | `data/inputGen/hotspotver.cpp` |
| `out.txt`    | Final temperature (°C) | `G×G` | Program output |

Generate large inputs:
```bash
cd hotspot/data/inputGen
make                      # build hotspotex, hotspotver
./hotspotex 18900 temp_18900   # temperature matrix
./hotspotver 18900 power_18900 # power matrix
```


## Directory Structure
- `Managed/`   Managed-memory version (cudaMallocManaged); supports AB/RM/PL/PF and `multistream` target.
- `UnManaged/` Explicit-memory version (cudaMalloc + cudaMemcpy).
- `data/`      Example inputs and generators.


## Build
```bash
# Default (single stream)
make -C hotspot/Managed      # build Managed/hotspot
make -C hotspot/UnManaged    # build UnManaged/hotspot

# Multi-stream (MULTISTREAM) variant (alias: prefetch)
make -C hotspot/Managed multistream
```


## Run
### Managed version
```bash
./hotspot/Managed/hotspot <grid> <pyramid_height> <sim_time> \
                          <temp_file> <power_file> <output_file> \
                          [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
```
- `<grid>`             Grid side length `G` (rows equal columns).
- `<pyramid_height>`   Consecutive time steps `H` per kernel (>0).
- `<sim_time>`         Total number of iterations.
- UM parameters (any combination):
  - `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` specifies which device/CPU will access the region
  - `RM dev|cpu`: `cudaMemAdviseSetReadMostly` indicates the region is mostly read-only
  - `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation` suggests the initial residency
  - `PF dev|cpu`: `cudaMemPrefetchAsync`; in single-stream build runs in default stream order then synchronizes; `dev` is device id (e.g., `0`), `cpu` is host
- If built with the `multistream` target, prefetch runs on multiple streams and can overlap with kernel execution (kernel still launched on default stream).

### UnManaged version

```bash
./hotspot/UnManaged/hotspot <grid> <pyramid_height> <sim_time> \
                            <temp_file> <power_file> <output_file>
```

## Program Output
- Stdout: At startup prints total memory size (MiB/GiB); at the end prints `Total elapsed time: X.XXX s`.
- File output: After run, generates `output.out` in the current directory, writing sampled `temp` every 1000 points.

## Memory vs. Parameters (derivation)
Let `grid_rows = grid_cols = G`, `size = G²`. Main GPU arrays:
- `MatrixTemp[0]`, `MatrixTemp[1]`, `MatrixPower` (float).
- VRAM bytes: `Mem_bytes ≈ 4 × (2·size + size) = 12·G²`  
  GiB ≈ `Mem_bytes / 2^30`.
- Example: 4 GiB → `G ≈ 18 900`; formula: `G ≈ 9 486 × √Mem_GiB`.

The program prints grid/block partitioning and total elapsed time at startup to help verify configuration.

## Notes
- Preserve the legacy Rodinia macro `RD_WG_SIZE_0`:
  ```bash
  make -C hotspot/Managed KERNEL_DIM="-DRD_WG_SIZE_0=16"
  ```
- Choose `pyramid_height` such that `BLOCK_SIZE - 2·H > 0` (default `BLOCK_SIZE=16`).


