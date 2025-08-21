## FDTD-2D Benchmark

This directory provides a 2-D Finite-Difference Time-Domain benchmark (FDTD-2D) based on Rodinia/PolyBench with two implementations:
- `Managed/`: Uses CUDA Unified Memory, supports `cudaMemAdvise`/`cudaMemPrefetchAsync`.
- `UnManaged/`: Uses explicit device memory and copy.

Both versions include wall-clock timing; at the end the program prints `Total elapsed time: X.XXX s`.

Note: No compile-time multi-stream switch (`MULTISTREAM`) is provided. `PF` in the Managed version issues single-stream prefetch followed by `cudaDeviceSynchronize()`.

## Workload Overview

### Mathematical Model & Algorithm
- **Maxwell Equations (TEz mode)**
  In source-free regions the 2-D TEz mode is:
  $$
  \partial_t E_z = \frac{1}{\varepsilon}(\partial_x H_y - \partial_y H_x),\quad
  \partial_t H_x = -\frac{1}{\mu}\partial_y E_z,\quad
  \partial_t H_y =  \frac{1}{\mu}\partial_x E_z
  $$
- **Yee Grid & Explicit Difference**
  Space-time is discretised into \(\Delta x,\Delta y,\Delta t\); electric (E) and magnetic (H) fields are staggered (Yee grid).
- **Stability (Courant)**: Ensure \(c\Delta t \le 1/\sqrt{(1/\Delta x)^2+(1/\Delta y)^2}\). Safe factors match Rodinia.
- **Boundary**: Simplified PEC — E field at boundaries forced to 0.

### GPU Parallel Mapping
- Threads/blocks: default `DIM_THREAD_BLOCK_X=32`, `DIM_THREAD_BLOCK_Y=8`.
- Kernel order per time step: `ey` → `ex` → `hz` (Rodinia naming).
- Arrays: `ex`, `ey`, `hz` plus helper `_fict_` (`tmax` length).

### Parameters & Inputs
- Dimensions: `NX`, `NY`, time steps `tmax` — compile-time `#define` (override with compile flags).
- Runtime UM switches (Managed only): `AB`/`RM`/`PL`/`PF`.
- No external files; initial values and `_fict_` generated internally.

## Directory Structure
- `Managed/`  : `fdtd2d.cu`, `Makefile`, `run`.
- `UnManaged/`: same.
- Output file `file.txt` is produced by default (see “Program Output”).

## Build
- Default build & clean:
```bash
make -C Managed
make -C UnManaged
make -C Managed clean
make -C UnManaged clean
```
- Compile-time options (examples):
```bash
# Grid size and time steps
make -C Managed   CFLAGS="-DNX=20000 -DNY=20000 -Dtmax=200"
make -C UnManaged CFLAGS="-DNX=1200  -DNY=1200  -Dtmax=5"
```

## Run

### Managed
Executable: `Managed/main`
```bash
./main [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
```
UM switches (independent, combinable):
- `AB dev|cpu`: `cudaMemAdviseSetAccessedBy`.
- `RM dev|cpu`: `cudaMemAdviseSetReadMostly`.
- `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation`.
- `PF dev|cpu`: `cudaMemPrefetchAsync`, single stream; `dev` GPU id, `cpu` host.

Examples:
```bash
# Prefer and prefetch to GPU0
./main PL 0 PF 0
# Read-mostly on CPU, no prefetch
./main RM cpu AB cpu
```

### UnManaged
Executable: `UnManaged/main`
```bash
./main
```

## Program Input
None — data generated internally.

## Program Output
- Stdout: total memory (MiB/GiB) and `Total elapsed time: X.XXX s`.
- File: `file.txt` sampling `hz` every 1000 points; enabled by default. Disable for large-scale performance tests to avoid I/O.

## Memory vs Parameters
Main arrays (float=4 B): `_fict_` (`tmax`), `ex` (`NX×(NY+1)`), `ey`(`(NX+1)×NY`), `hz`(`NX×NY`).
Total bytes:
```text
Mem_bytes = 4·[ tmax + NX·(NY+1) + (NX+1)·NY + NX·NY ]
          = 4·tmax + 12·NX·NY + 4·(NX+NY)
```
For large `NX,NY`: `≈12·NX·NY`; GiB = Mem_bytes / 2^30.
If `NX=NY=n`, target GiB →
```text
n ≈ sqrt( GiB·2^30 / 12 )
```

| Target GiB | n |
|------------|-------|
| 1 | ≈9 462 |
| 4 | ≈18 924 |
| … | … |

## Notes
- UM switches are independent; `PF` device should match compute GPU.
- No `MULTISTREAM` target here; refer to `BFS/Managed` for ideas if needed.
- Ensure sufficient memory for large cases; reduce `NX/NY/tmax` or disable file output if OOM.
- When customising `NX/NY/tmax/DIM_THREAD_BLOCK_*`, stay within device limits; sync as necessary.
