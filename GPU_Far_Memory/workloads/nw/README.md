## Needleman–Wunsch Benchmark
This directory contains two implementations of the Needleman–Wunsch (NW) global sequence alignment: `Managed/` (Unified Memory) and `UnManaged/` (explicit copies). Both include wall-clock timing and print total time like `Total elapsed time: X.XXX s` at the end.

- `Managed/` supports multi-stream (`-DMULTISTREAM`): run `cudaMemPrefetchAsync` and kernel launches in separate streams to overlap prefetch and compute.
- `UnManaged/` does not provide `MULTISTREAM` or UM strategies (AB/RM/PL/PF).

## Workload Overview
### Mathematical Model and Algorithm
NW computes the globally optimal alignment of two sequences via dynamic programming. Let sequences `A=a1…aN`, `B=b1…bN`. Define the score matrix `H` (called `itemsets` in the source) of size `(N+1)×(N+1)`:
```text
Boundary initialization:
  H[0][0] = 0
  H[0][j] = - j * penalty            (j = 1..N)
  H[i][0] = - i * penalty            (i = 1..N)

Recurrence (i=1..N, j=1..N):
  match_mismatch = H[i-1][j-1] + s(ai, bj)
  delete_from_A  = H[i-1][j]   - penalty
  insert_into_A  = H[i][j-1]   - penalty
  H[i][j]        = max(match_mismatch, delete_from_A, insert_into_A)

where s(x,y) is the substitution score from an integer-approximate BLOSUM62 (array `referrence`).
The optimal alignment score is H[N][N].
```
Traceback recovers the alignment path from `(N,N)`: if `H[i][j]==H[i-1][j-1]+s(ai,bj)` go diagonal; if equal to `H[i-1][j]-penalty` go up; if equal to `H[i][j-1]-penalty` go left. This implementation uses linear gap penalties (non-affine).
For verification, the `UnManaged` version enables `TRACEBACK` by default to write results to `result.txt`; the `Managed` version can selectively output via `-DBENCH_PRINT`.

Key parameters:
- `N`: length of both sequences (internally uses `N+1` as dimension). Must satisfy `N%16==0`.
- `penalty`: positive integer gap penalty.
- `BLOCK_SIZE`: thread block dimension macro (default 16), affects tile size and shared memory usage.

### GPU Parallel Mapping
- Parallel strategy: The matrix is tiled into `BLOCK_SIZE×BLOCK_SIZE` tiles and advanced along anti-diagonals in two phases (`needle_cuda_shared_1` and `needle_cuda_shared_2`). Within each tile, shared memory buffers `temp` and `ref` are used and two intra-block diagonal sweeps are performed.
- Memory access: Global memory is accessed in row-major order via shared memory buffers; no time tiling.
- Multi-stream (Managed optional): If `-DMULTISTREAM` is enabled, three streams are used: two for `cudaMemPrefetchAsync` of the main arrays (when `PF` is set) and one for kernel launch, enabling overlap. Otherwise, state “none”.
- Major data structures and dominant terms: integer 2D arrays `itemsets` and `referrence` (4 bytes per element), dimensions approx `(N+1)×(N+1)`, dominating VRAM usage.

### Parameters / Inputs
- Size-related parameters: `N` and `penalty` (VRAM dominated by `N`).
- External input files: none. Two sequences of length `N` are randomly generated inside the program (kept consistent with the original README).

## Directory Structure
- `Managed/needle.cu`: Unified Memory main program (includes UM flag parsing, timing, memory estimation, optional multi-stream).
- `Managed/needle_kernel.cu`: core CUDA kernels (two-phase diagonal sweeps).
- `Managed/needle.h`: macros related to `BLOCK_SIZE`.
- `Managed/Makefile`: build script (with `multistream` and compatible alias `prefetch`).
- `Managed/run`: example run script.
- `UnManaged/needle.cu`: explicit-copy main program (keeps legacy `TRACEBACK` file output).
- `UnManaged/needle_kernel.cu`, `UnManaged/needle.h`, `UnManaged/Makefile`, `UnManaged/run`: corresponding kernels, headers, and build scripts.

(Effective info preserved from the original README: sequences are randomly generated; `N` must be a multiple of 16; workgroup size can be tuned via `RD_WG_SIZE_0` and related macros.)

## Build
Default build and clean:
```bash
make -C Managed
make -C UnManaged
make -C Managed clean
make -C UnManaged clean
```

Compile-time options:
- Thread/block/grid macros (defaults in source):
  - `-DRD_WG_SIZE_0=<n>` (or `-DRD_WG_SIZE_0_0`/`-DRD_WG_SIZE`): set `BLOCK_SIZE`, default `16`, common values `16/32`; affects parallel granularity and shared memory usage, not VRAM size.
- Output control macros:
  - Managed: with `-DBENCH_PRINT`, write `result.txt` (traceback sequences). By default no file to reduce I/O.
  - UnManaged: source enables `#define TRACEBACK` by default, writing `result.txt`; to disable, edit `UnManaged/needle.cu` and comment out the macro (no equivalent compile-time flag).
- Multi-stream (Managed only):
  - `-DMULTISTREAM`: run `cudaMemPrefetchAsync` in multiple streams (if `PF` enabled) and launch kernels in a dedicated stream.
  - `Managed/Makefile` target: `multistream`; alias `prefetch` is preserved.

Example (custom block size with result writeback):
```bash
make -C Managed CFLAGS="-DRD_WG_SIZE_0=32 -DBENCH_PRINT"
```

Note: All `Makefile` recipe lines must use TABs (not spaces).

## Run
### Managed
Command:
```bash
./needle N penalty [UM flags]
```
- Workload parameters:
  - `N`: sequence length (must satisfy `N%16==0`). Internally uses dimension `N+1`.
  - `penalty`: positive integer.
- UM strategies:
  - `DEV gpu_id`: select device (e.g., `0`).
  - `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` specifies which device/CPU will access the region
  - `RM dev|cpu`: `cudaMemAdviseSetReadMostly` indicates the region is mostly read-only
  - `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation` suggests the initial residency
  - `PF dev|cpu`: `cudaMemPrefetchAsync`; single-stream runs in default order and then syncs; with `MULTISTREAM`, prefetches arrays in separate streams and overlaps with kernels.

Examples:
```bash
# Single stream + AB to GPU0 + PL to CPU + prefetch to GPU0
./needle 32768 10 AB 0 PL cpu PF 0

# Multi-stream (requires multistream build) + set AB/RM/PL/PF to GPU0
./needle 32768 10 AB 0 RM 0 PL 0 PF 0

# ReadMostly only
./needle 16384 10 RM 0
```

Suggested `N` for different VRAM targets (rounded down to multiple of 16):

| Target VRAM (GiB) | Suggested N |
|---|---|
| 1  | 11584 |
| 4  | 23168 |
| 8  | 32768 |
| 12 | 40128 |
| 16 | 46336 |
| 20 | 51760 |
| 24 | 56736 |
| 30 | 63472 |

Note: See “Memory vs. Parameters (derivation)”; `N` must be a multiple of 16.

### UnManaged
Command:
```bash
./needle N penalty
```
Examples (same scales as Managed for comparison):
```bash
./needle 32768 10
./needle 16384 10
```

## Program Input (if any)
No external files. Two sequences of length `N` are randomly generated inside the program (consistent with the original README).

## Program Output
- Stdout: at least includes `Total elapsed time: X.XXX s`.
- File output:
  - Managed: with `-DBENCH_PRINT`, write `result.txt` (traceback sequences); overwrite if exists; disabled by default to avoid I/O.
  - UnManaged: writes `result.txt` by default (controlled by `TRACEBACK` macro in source).

## Memory vs. Parameters (derivation)
Main allocations (4-byte ints):
```
itemsets   : (N+1) × (N+1) × 4
referrence : (N+1) × (N+1) × 4

Mem_bytes = 2 × (N+1) × (N+1) × 4 ≈ 8 × N^2
Mem_GiB   = Mem_bytes / 2^30
```
From target VRAM `M_GiB`, derive `N`:
```
N ≈ floor_to_16( sqrt( M_GiB × 2^30 / 8 ) )
```
Note: In practice, `nvidia-smi` includes context, driver, and UM alignment overheads, often above the theoretical sum of arrays; trust the program’s startup estimate and `nvidia-smi` observation.

## Suggestions
- Start with default `BLOCK_SIZE` 16; increasing to 32 may improve throughput but watch shared memory and stack/register usage.
- For very large scales, try the Managed version first to leverage on-demand migration and optional prefetch/multi-stream overlap.

## Notes
- UM switches `AB/RM/PL/PF` are independent and combinable (Managed only). The `PF` device should match the compute GPU.
- Multi-stream (Managed only): with `-DMULTISTREAM`, apply advises in a multi-stream context and split prefetch/compute to overlap.
- Large-scale parameters: `size_t` is used in `Managed/needle.cu` for products and byte counts to avoid overflow; when kernels still depend on `int` indices, avoid out-of-range accesses.
- When changing block/grid, ensure device limits are not exceeded; keep `cudaDeviceSynchronize()` after critical phases as needed.
- This README consistently uses `Mem_bytes`, GiB/MiB (`GiB=2^30`, `MiB=2^20`); relative paths are wrapped in backticks (e.g., `Managed/Makefile`).


