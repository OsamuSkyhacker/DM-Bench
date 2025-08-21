## Hotspot Benchmark

This directory provides a Hotspot (2-D transient thermal diffusion) benchmark derived from Rodinia with two implementations:
- `Managed/`: Uses CUDA Unified Memory (UM), supports `cudaMemAdvise`/`cudaMemPrefetchAsync`, offers a `multistream` make target.
- `UnManaged/`: Uses explicit device memory and copy.

Both versions include wall-clock timing; at termination the program prints `Total elapsed time: X.XXX s`.

Note: The `multistream` build primarily prefetches UM arrays in multiple streams; the kernel is still launched in the default stream. Target `prefetch` is an alias.

## Workload Overview

### Mathematical Model & Algorithm
- **Continuous equation**
  
  ```math
  \rho c \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + P 
  ```
  where $T$ is temperature, $P$ power density, and $\rho,c,k$ are density, specific heat, and conductivity.
- **Explicit Finite Difference (FDM)**
  Divide the chip into a $G\times G$ grid and use a 4-neighbour stencil:
  ```math
  T_{i,j} \leftarrow T_{i,j} + \kappa\,\Delta t/\Delta x^{2}\,[T_{i+1,j}+T_{i-1,j}+T_{i,j+1}+T_{i,j-1}-4T_{i,j}] + \text{power term} 
  ```
- **Stability**: Requires $\kappa\Delta t /\Delta x^{2} \le 0.25$. Constants satisfy this.
- **Boundary**: Convective cooling on chip edges is built-in; no user config needed.

### GPU Parallel Mapping
- **Threads/blocks**: Each 16×16 tile processed by one block; one thread per cell.
- **Shared memory buffering**: Threads share neighbourhood data inside the block.
- **Time tiling – `pyramid_height` (H)**: One kernel computes H steps consecutively; shared-mem needs grow with H.
- **Multi-stream (`MULTISTREAM`)**: Create multiple CUDA streams to overlap data migration and computation.

### 4-Neighbour Stencil & `pyramid_height`
[`diagram omitted`]

`pyramid_height = 1` uses immediate four neighbours; `H>1` expands dependency outward H layers.

### Use Cases
- Quick evaluation of SoC / 3D-IC hot spots.
- Input for heatsink design, power scheduling, thermal-timing coupling analysis.

## Program Input
Hotspot uses plain ASCII float text, one value per line, row-major.

| File | Role | Lines | Generator |
|------|------|-------|-----------|
| `temp_G`   | initial temperature (°C) | $G^2$ | `data/inputGen/hotspotex.cpp` |
| `power_G`  | power (W)               | $G^2$ | `data/inputGen/hotspotver.cpp` |
| `out.txt`  | final temperature       | $G^2$ | program output |

Generate large inputs:
```bash
cd hotspot/data/inputGen
make
./hotspotex 18900 temp_18900
./hotspotver 18900 power_18900
```

## Directory Structure
- `Managed/`   UM version (cudaMallocManaged); supports AB/RM/PL/PF and `multistream`.
- `UnManaged/` Explicit version (cudaMalloc + cudaMemcpy).
- `data/`      Sample inputs and generators.

## Build
```bash
# Default (single stream)
make -C hotspot/Managed
make -C hotspot/UnManaged

# Multi-stream variant
make -C hotspot/Managed multistream
```

## Run
### Managed
```bash
./hotspot/Managed/hotspot <grid> <pyramid_height> <sim_time> \
                          <temp_file> <power_file> <output_file> \
                          [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
```
Arguments:
- `<grid>`: grid size $G$.
- `<pyramid_height>`: H (>0).
- `<sim_time>`: total iterations.
- UM switches (any combination): `AB` accessor, `RM` read-mostly, `PL` preferred location, `PF` prefetch.
- Device id is GPU index or `cpu`.
- With `multistream` build, prefetch happens in multiple streams overlapping the kernel.

### UnManaged
```bash
./hotspot/UnManaged/hotspot <grid> <pyramid_height> <sim_time> \
                            <temp_file> <power_file> <output_file>
```

## Program Output
- Stdout: total memory size (MiB/GiB) at start and `Total elapsed time: X.XXX s` at end.
- File: `output.out` containing sampled temperatures (every 1000 points).

## Memory vs Parameters
Main GPU arrays (`float`): two temps + power.
Memory bytes: `Mem_bytes ≈ 12·G²` (GiB = Mem_bytes / 2^30).
Example: 4 GiB → `G ≈ 18 900`.

## Notes
- Keep legacy macro `RD_WG_SIZE_0`:
```bash
make -C hotspot/Managed KERNEL_DIM="-DRD_WG_SIZE_0=16"
```
- Choose `pyramid_height` s.t. `BLOCK_SIZE - 2·H > 0` (`BLOCK_SIZE=16`).
