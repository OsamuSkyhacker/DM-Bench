## Pathfinder Benchmark

This directory contains the GPU Pathfinder benchmark based on Rodinia, with two implementations:
- `Managed/`: Uses CUDA Unified Memory (managed memory), supports strategies such as `cudaMemAdvise`/`cudaMemPrefetchAsync`, and provides a multi-stream variant.
- `UnManaged/`: Uses explicit device memory allocation and copies.

The program includes wall-clock runtime statistics and prints: `Total elapsed time: X.XXX s` upon completion.

## Workload Overview
### Mathematical Model and Algorithm
- Dynamic programming (DP) for the minimum-cost path on a grid: starting from the first row, proceed row by row. The accumulated cost at the current cell equals the minimum among the three neighboring cells in the previous row (up, up-left, up-right) plus the current cell's cost.
- Key symbols: `pyramid_height` (temporal blocking steps, affecting how many time steps a kernel advances consecutively), `HALO` (neighborhood radius, equals 1 for Pathfinder), `BLOCK_SIZE` (thread-block side length, determining each block's valid compute region).

- State definition: Given a cost grid `C` (size `rows×cols`), define an accumulated cost grid `D` of the same size, where `D[r][c]` denotes the minimum accumulated cost to reach `(r,c)` when starting from some column in the first row.
- Boundary initialization: The first row has no dependency, `D[0][c] = C[0][c]` (`c=0..cols-1`).
- Recurrence (`r=1..rows-1, c=0..cols-1`):
  - `D[r][c] = C[r][c] + min( D[r-1][c-1], D[r-1][c], D[r-1][c+1] )`.
  - Out-of-bound neighbors (`c-1<0` or `c+1≥cols`) are treated as unreachable (regarded as `+∞`).
- Goal and output: The minimum cost in the final row `min_c D[rows-1][c]` is the top-down optimal path cost; alternatively, keep the entire row `D[rows-1][·]` for later selection (this implementation typically outputs only the last row).
- Path recovery (optional): From the minimal-cost position `c*` in the last row, backtrack to the first row. At each step choose among up, up-left, up-right that attains the current `D[r][c]` (if ties exist, set a priority). The current implementation does not write the path, only the minimal-cost array.
- Numeric types: Costs and accumulations use `int`. Consider wider types if input scale/costs may overflow.
- Relation to `HALO`/`pyramid_height`: `HALO=1` means one advancement step needs the three-neighborhood `[c-1,c,c+1]` in the previous row. When temporal blocking advances `H=pyramid_height` steps per kernel, the valid compute region must shrink by `HALO·H` on each horizontal side:
  - `smallBlockCol = BLOCK_SIZE - 2·HALO·pyramid_height` (must satisfy `> 0`).
  - Blocks coordinate via boundary expansion and overlap to satisfy cross-step dependencies; kernels use shared memory to buffer neighborhoods and reduce global memory accesses.
- Complexity:
  - Time complexity `O(rows·cols)`, with constant reads and one write per grid point.
  - Space complexity depends on the implementation; this implementation keeps `gpuWall` (input cost grid) and uses double-buffering `gpuResult[2]` (previous row and current row).

```text
Boundary initialization:
  D[0][c] = C[0][c]                          (c = 0..cols-1)

Recurrence (r = 1..rows-1, c = 0..cols-1):
  D[r][c] = C[r][c] + min( D[r-1][c-1], D[r-1][c], D[r-1][c+1] )
  # Out-of-bound neighbors treated as +∞

Output:
  min_cost = min_c D[rows-1][c]
```

### GPU Parallel Mapping
- Thread/block mapping: 2D tiling, each thread handles one grid point; use shared memory to buffer the neighborhood.
- Time/space tiling: Each kernel advances `pyramid_height` steps; the valid region shrinks and is controlled by `smallBlockCol/Row`. Inter-block coordination is determined by boundary expansion `HALO·pyramid_height`.
- Multi-stream: When built with `-DMULTISTREAM`, PF prefetch runs on multiple streams, and the kernel is launched in a separate stream to overlap with prefetch.

### Parameters / Inputs
- Size parameters: `<cols>` (columns), `<rows>` (rows), `<pyramid_height>` (temporal block size).
- Program input: No external files; the cost grid and boundaries are initialized inside the program.

## Directory Structure
- `Managed/`
  - `pathfinder.cu`, `Makefile`, `README`, `run`
- `UnManaged/`
  - `pathfinder.cu`, `Makefile`, `README`, `run`

## Build

- Managed-memory version (default build)
```bash
make -C Managed
```

- Managed-memory version (multi-stream variant: `-DMULTISTREAM`)
```bash
make -C Managed multistream     # Compatible target: prefetch is equivalent to multistream
```

- UnManaged version
```bash
make -C UnManaged
```

- Clean
```bash
make -C Managed clean
make -C UnManaged clean
```

## Run

### Managed-memory version (Managed)
- Executable: `Managed/pathfinder`
- Invocation:
```bash
./pathfinder <cols> <rows> <pyramid_height> [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
```
- Argument description:
  - `<cols>`: number of columns; `<rows>`: number of rows; `<pyramid_height>`: number of time steps per batch (see constraints).
  - `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` specifies which device/CPU will access the region
  - `RM dev|cpu`: `cudaMemAdviseSetReadMostly` indicates the region is mostly read-only
  - `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation` suggests the initial residency
  - `PF dev|cpu`: `cudaMemPrefetchAsync`; in single-stream build it runs in default stream order then synchronizes; in `MULTISTREAM` it prefetches different arrays in separate streams and overlaps with the kernel.

- Examples:
```bash
cd Managed
# Run on GPU0 with AccessedBy and Prefetch
./pathfinder 5000 966000 100 AB 0 PF 0

# Set preferred location to CPU and still prefetch to GPU0 (for experimental comparison)
./pathfinder 5000 966000 100 PL cpu PF 0
```

### UnManaged version
- Executable: `UnManaged/pathfinder`
- Invocation:
```bash
./pathfinder <cols> <rows> <pyramid_height>
```

## PF/AB/RM/PL behavior under single vs multi-stream
- Single stream (default build):
  - `AB/RM/PL` take effect immediately before launch.
  - With `PF` set, prefetches in default stream order and synchronizes. Cold-start jitter is small, but it cannot overlap with computation.
- Multi-stream (`-DMULTISTREAM`):
  - `AB/RM/PL` behave the same.
  - With `PF` set, prefetches major arrays in different streams; the kernel runs in a dedicated stream, allowing prefetch and compute to overlap and shorten startup latency.

## Program Input
- No external files; data are generated/initialized inside the program.

## Program Output
- Stdout: key parameters and `Total elapsed time: X.XXX s`.
- File output: When `BENCH_PRINT` is enabled, writes `result.txt` (overwrite if exists, create if not). Disabled by default to avoid I/O interference.

## Memory vs. Parameters (derivation)

Allocations (int is 4 bytes):
- `gpuResult[0]`, `gpuResult[1]`: each `cols * sizeof(int)`
- `gpuWall`: `(rows * cols - cols) * sizeof(int)`

Approximate total (Bytes):
```
Mem_bytes ≈ 4 · (rows·cols + cols)
```

Estimate parameters from target VRAM (GiB, base 2^30):
```
rows ≈ floor( GiB · 2^30 / (4·cols) ) - 1
```

Example (24 GiB device, cols=5000):
```
rows ≈ floor(24 · 2^30 / (4 · 5000)) - 1 ≈ 1,288,489
```

## Notes
- `pyramid_height` must satisfy: `BLOCK_SIZE - 2·HALO·pyramid_height > 0` (current `BLOCK_SIZE=256, HALO=1` → max ≈ 127).
- Grid partitioning: `smallBlockCol = BLOCK_SIZE - 2·HALO·pyramid_height`, `blockCols = ceil(cols / smallBlockCol)`.
- `BENCH_PRINT` output greatly slows execution and consumes memory; enable only at small scales (disabled by default).
- `AB/RM/PL/PF` are independent and can be combined; for example, `AB 0` indicates “notify that this memory segment will be accessed by GPU0”; it does not move data by itself—use `PF 0` to migrate.

## Result Output and Feature Switches
- Writing `result.txt` is controlled by the `#define BENCH_PRINT` macro in the source:
  - Enabled (define `#define BENCH_PRINT` at the top of `Managed/pathfinder.cu`): at program end, the final row is written to `result.txt` in the current working directory (create if missing; overwrite if exists).
  - Disabled (comment out the macro): do not write files to avoid I/O interference in large-scale performance tests (default recommendation).


