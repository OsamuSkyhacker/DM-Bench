## FDTD-2D Benchmark

This directory contains the FDTD-2D (2D finite-difference time-domain) benchmark based on Rodinia/PolyBench, with two implementations:
- `Managed/`: Uses CUDA Unified Memory (UM), supporting `cudaMemAdvise`/`cudaMemPrefetchAsync`.
- `UnManaged/`: Uses explicit device memory and copies.

Both versions include wall-clock timing; at program end it prints: `Total elapsed time: X.XXX s`.

Note: This task currently does not provide a compile-time multi-stream switch (`MULTISTREAM`). In the Managed version, `PF` is single-stream prefetch followed by `cudaDeviceSynchronize()`.

## Workload Overview

### Mathematical Model and Algorithm
- **Maxwell equations (TEz mode)**  
  In the source-free, charge-free case, the 2D TEz mode can be written as:
```math
  \frac{\partial E_z}{\partial t}=\frac{1}{\varepsilon}\left(\frac{\partial H_y}{\partial x}-\frac{\partial H_x}{\partial y}\right),\quad
  \frac{\partial H_x}{\partial t}=-\frac{1}{\mu}\frac{\partial E_z}{\partial y},\quad
  \frac{\partial H_y}{\partial t}=\frac{1}{\mu}\frac{\partial E_z}{\partial x}
```

- **Yee grid & explicit differences**  
  Discretize space-time with $\Delta x, \Delta y, \Delta t$. Electric field (E) and magnetic field (H) are stored at staggered half-cell positions, forming a **Yee grid**. Update scheme:

```math
Hx(i,j) \leftarrow Hx(i,j) - \frac{\Delta t}{\mu} \cdot \frac{Ez(i,j+1) - Ez(i,j)}{\Delta y}
```

```math
Hy(i,j) \leftarrow Hy(i,j) + \frac{\Delta t}{\mu} \cdot \frac{Ez(i+1,j) - Ez(i,j)}{\Delta x}
```

```math
Ez(i,j) \leftarrow Ez(i,j) + \frac{\Delta t}{\varepsilon} \cdot \left[\frac{Hy(i,j) - Hy(i-1,j)}{\Delta x} - \frac{Hx(i,j) - Hx(i,j-1)}{\Delta y}\right]
```

- **Stability (Courant condition)**  
  Must satisfy $c \Delta t \le 1 / \sqrt{ (1/\Delta x)^2 + (1/\Delta y)^2 }$. The program uses the same safety factor as Rodinia.
- **Boundary condition**  
  Simplified to PEC (Perfect Electric Conductor); E-field on the boundary is forced to 0.

### GPU Parallel Mapping
- Threads/blocks: default `DIM_THREAD_BLOCK_X=32`, `DIM_THREAD_BLOCK_Y=8` (can be changed via compile-time macros).
- Kernel partition: For each time step, update `ey` → `ex` → `hz` in sequence (naming follows Rodinia).
- Memory layout: Main arrays `ex`, `ey`, `hz` and auxiliary `_fict_` (length `tmax`).

### Parameters / Inputs
- Size parameters: `NX`, `NY` (spatial grid dimensions), `tmax` (number of time steps). All are `#define` in source and can be overridden via compile macros.
- Runtime UM parameters (Managed only): `AB`/`RM`/`PL`/`PF` (see “Run”).
- External inputs: No files are required; initial values and `_fict_` are generated inside the program.

## Directory Structure
- `Managed/`: `fdtd2d.cu`, `Makefile`, `run`.
- `UnManaged/`: `fdtd2d.cu`, `Makefile`, `run`.
- Sample output file: After default run, `file.txt` will be generated in the respective directory (see “Program Output”).

## Build
- Default build and clean:
  ```bash
  make -C Managed
  make -C UnManaged
  make -C Managed clean
  make -C UnManaged clean
  ```
- Compile-time parameters (examples):
  - Grid size: `-DNX=<n> -DNY=<n>`; time steps: `-Dtmax=<t>`.
  - Block dimensions: `-DDIM_THREAD_BLOCK_X=32 -DDIM_THREAD_BLOCK_Y=8`.
  - Examples:
    ```bash
    make -C Managed   CFLAGS="-DNX=20000 -DNY=20000 -Dtmax=200"
    make -C UnManaged CFLAGS="-DNX=1200  -DNY=1200  -Dtmax=5"
    ```


## Run

### Managed
- Executable: `Managed/main`
- Usage:
  ```bash
  ./main [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
  ```
- UM parameter notes (independent and combinable):
  - `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` specifies which device/CPU will access the region
  - `RM dev|cpu`: `cudaMemAdviseSetReadMostly` indicates the region is mostly read-only
  - `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation` suggests the initial residency
  - `PF dev|cpu`: `cudaMemPrefetchAsync`; with single-stream build it executes in default stream order and then synchronizes; `dev` is device id (e.g., `0`), `cpu` is host
- Examples:
  ```bash
  # Prefer GPU0 residency and prefetch to GPU0 (single stream)
  ./main PL 0 PF 0

  # Mark read-mostly and accessed by CPU; no prefetch
  ./main RM cpu AB cpu
  ```

### UnManaged
- Executable: `UnManaged/main`
- Usage:
  ```bash
  ./main
  ```

## Program Input
- No external files; data are generated inside the program.

## Program Output
- Stdout: At startup prints total memory size (MiB/GiB); at the end prints `Total elapsed time: X.XXX s`.
- File output (enabled by default in both versions): After run, generates `file.txt` in the current directory, writing sampled `hz` every 1000 points; overwrites if exists, creates if not. For large-scale performance tests, it is recommended to disable or comment out this output to avoid I/O interference.

## Memory vs. Parameters (derivation)
- All main arrays are `float` (`sizeof(float)=4`):
  - `_fict_`: `tmax`
  - `ex`: `NX × (NY+1)`
  - `ey`: `(NX+1) × NY`
  - `hz`: `NX × NY`
- Total bytes:
  ```
  Mem_bytes = 4 × [ tmax + NX·(NY+1) + (NX+1)·NY + NX·NY ]
            = 4·tmax + 12·NX·NY + 4·(NX+NY)
  ```
- When `NX,NY` are large: `Mem_bytes ≈ 12·NX·NY`, `Mem_GiB = Mem_bytes / 2^30`.
- Let `NX=NY=n`. For a target memory `M_GiB`, the suggested grid edge length:
  ```
  n ≈ sqrt( M_GiB × 2^30 / 12 )
  ```
- Reference suggestions (rounded to nearby integers):

  | Target Memory (GiB) | Recommended n |
  |---------------------|---------------|
  | 1  | ≈ 9,462  |
  | 4  | ≈ 18,924 |
  | 8  | ≈ 26,745 |
  | 12 | ≈ 32,769 |
  | 16 | ≈ 37,848 |
  | 20 | ≈ 42,331 |
  | 24 | ≈ 46,368 |
  | 30 | ≈ 51,826 |

- Note: nvidia-smi includes extra overhead for context and UM alignment; observed VRAM usage is usually slightly higher than the theoretical array sum. The `tmax` and `NX+NY` terms have smaller impact at large scales.

## Notes
- UM switches `AB/RM/PL/PF` are independent and combinable (Managed only); the device used with `PF` is recommended to match the compute GPU.
- This task does not provide `MULTISTREAM`. For multi-stream prefetch/overlap, refer to the multi-stream implementation in `BFS/Managed`.
- For large-scale parameters ensure sufficient host/GPU memory; if OOM occurs, reduce `NX/NY/tmax` or disable file writes.
- When customizing `NX/NY/tmax/DIM_THREAD_BLOCK_*`, ensure device limits are not exceeded; use `cudaDeviceSynchronize()` at critical stages if necessary.
- Terminology is unified: this document consistently uses `Mem_bytes`; capacities are in GiB/MiB (`GiB=2^30`, `MiB=2^20`).


