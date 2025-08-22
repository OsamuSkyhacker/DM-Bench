## Pathfinder Benchmark

This directory contains a GPU Pathfinder benchmark derived from Rodinia with two implementations:
- `Managed/`: Uses CUDA Unified Memory (UM) and supports `cudaMemAdvise`/`cudaMemPrefetchAsync`; also provides a multi-stream variant.
- `UnManaged/`: Uses explicit device memory allocation and copy.

The program measures full wall-clock time and prints `Total elapsed time: X.XXX s` when finished.

## Workload Overview

### Algorithm
Dynamic programming for the minimum-cost path on a 2-D grid. Starting from the first row the computation advances row by row; the cost of a cell is the minimum of its three predecessors (upper-left, upper, upper-right) plus its own cost.

Key symbols: `pyramid_height` (time-tile depth of each kernel launch), `HALO` (radius of the neighbourhood, 1 for Pathfinder), `BLOCK_SIZE` (thread-block width).

### GPU Parallel Mapping
- **Thread/block mapping**: The grid is tiled; each thread handles one cell and uses shared memory to buffer its neighbourhood.
- **Time/space tiling**: Each kernel advances `pyramid_height` steps. The valid area of a block shrinks accordingly, governed by `smallBlockCol/Row`. Blocks cooperate through halos of width `HALO*pyramid_height`.
- **Multi-stream**: Built with `-DMULTISTREAM`, prefetch (`PF`) is issued in several streams while the kernel runs in a separate stream to overlap migration with computation.

### Parameters & Inputs
- `<cols>` (number of columns), `<rows>` (rows), `<pyramid_height>` (time steps per batch).
- No external input file; the cost grid and boundary are initialised inside the program.

## Directory Structure
- `Managed/`  : `pathfinder.cu`, `Makefile`, `README`, `run`
- `UnManaged/`: same-named files.

## Build

### Managed (default)
```bash
make -C Managed
```

### Managed multi-stream variant (`-DMULTISTREAM`)
```bash
make -C Managed multistream   # alias target "prefetch" is identical
```

### UnManaged
```bash
make -C UnManaged
```

### Clean
```bash
make -C Managed clean
make -C UnManaged clean
```

## Run

### Managed
Executable: `Managed/pathfinder`

Usage:
```bash
./pathfinder <cols> <rows> <pyramid_height> [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
```
Arguments:
- `<cols>`, `<rows>`, `<pyramid_height>`: dimensions and time tile (see constraint below).
- `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` for the given device/CPU (e.g. `AB 0`).
- `RM dev|cpu`: `cudaMemAdviseSetReadMostly`.
- `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation`.
- `PF dev|cpu`: `cudaMemPrefetchAsync` to the target device/CPU.
  - **Single-stream build**: prefetch in default stream and sync, smaller cold-start jitter.
  - **Multi-stream build**: prefetch arrays in multiple streams; kernel runs in another stream to overlap.
- Device selection: no explicit `DEV` arg; program uses GPU 0 by default (change in code if needed).

Examples:
```bash
cd Managed
# Run on GPU0, enable AccessedBy & Prefetch
./pathfinder 5000 966000 100 AB 0 PF 0
# Prefer CPU residence but still prefetch to GPU0
./pathfinder 5000 966000 100 PL cpu PF 0
```

### UnManaged
Executable: `UnManaged/pathfinder`
```bash
./pathfinder <cols> <rows> <pyramid_height>
```

## Single- vs Multi-stream Behaviour of PF/AB/RM/PL
- **Single stream**: `AB/RM/PL` take effect immediately. `PF` prefetches in default stream and syncs, no overlap.
- **Multi-stream**: same for `AB/RM/PL`; `PF` prefetches different arrays in parallel streams while kernel runs elsewhere, enabling overlap.

## Program Input
None — data are internally generated.

## Program Output
- Stdout: key parameters and `Total elapsed time: X.XXX s`.
- File: if `BENCH_PRINT` is defined the final row is written to `result.txt` (off by default to avoid I/O overhead).

## Memory vs Parameters (estimation)
Allocations (int = 4 B):
- `gpuResult[0]`, `gpuResult[1]`: each `cols * 4`
- `gpuWall`: `(rows*cols - cols) * 4`

Approximate total:
```text
Mem_bytes ≈ 4 · (rows·cols + cols)
```

Estimating rows from target VRAM GiB:
```text
rows ≈ floor( GiB · 2^30 / (4·cols) ) - 1
```

Example (24 GiB device, cols=5000): `rows ≈ 1 288 489`.

## Suggested Parameters (1 GiB → 30 GiB)
Fix `cols=5000`, `pyramid_height=100`; see table below (template `./pathfinder 5000 <rows> 100 AB 0 PF 0`).

| Target GiB | rows |
|------------|-------|
| 1  |  53 686 |
| 2  | 107 373 |
| … | … |
| 30 | 1 610 611 |

## Notes
- Constraint: `BLOCK_SIZE - 2·HALO·pyramid_height > 0` (`BLOCK_SIZE=256`, `HALO=1` ⇒ max ≈127).
- Grid partition: `smallBlockCol = BLOCK_SIZE - 2·HALO·pyramid_height`, `blockCols = ceil(cols / smallBlockCol)`.
- `BENCH_PRINT` greatly slows execution and uses memory; enable only for small runs.
- `AB/RM/PL/PF` are independent; e.g. `AB 0` just informs that GPU0 will access the region, it does not move data — combine with `PF 0` to move.

## Output Switch
Writing `result.txt` is controlled by `#define BENCH_PRINT` in `Managed/pathfinder.cu`:
- Enable: define the macro to write final result.
- Disable: comment it out (default).


