## Needleman–Wunsch Benchmark
This directory contains two implementations of the Needleman–Wunsch (NW) global sequence alignment: `Managed/` (Unified Memory) and `UnManaged/` (explicit copy). Both include wall-clock timing; at termination the program prints `Total elapsed time: X.XXX s`.

- `Managed/` supports multi-stream (`-DMULTISTREAM`): `cudaMemPrefetchAsync` and kernel launches run in separate streams to overlap prefetch and computation.
- `UnManaged/` does not provide `MULTISTREAM` or UM strategies (AB/RM/PL/PF).

## Workload Overview
### Algorithm
NW is a classic global sequence alignment (dynamic programming) algorithm. Given two sequences of length `N`, an `(N+1)×(N+1)` score matrix `itemsets` is built. Using a substitution matrix (`referrence`, an integer approximation of BLOSUM62) and a gap penalty `penalty`, the algorithm proceeds along wavefronts (diagonals) to compute the optimal global alignment score.

Key parameters:
- `N`: sequence length (the program uses `N+1` internally); must satisfy `N%16==0`.
- `penalty`: positive integer gap penalty.
- `BLOCK_SIZE`: thread-block dimension macro (default 16); affects tile size and shared memory use.

### GPU Parallel Mapping
- **Parallel strategy**: The matrix is tiled into `BLOCK_SIZE×BLOCK_SIZE` blocks. Two phases advance along diagonals (`needle_cuda_shared_1` and `needle_cuda_shared_2`). Within a tile, shared memory buffers `temp` and `ref` are used for two stage diagonal updates.
- **Memory access**: Global memory is accessed row-major; no time tiling.
- **Multi-stream (optional in `Managed/`)**: With `-DMULTISTREAM`, three streams are used—two for `cudaMemPrefetchAsync` (effective when `PF` enabled) on main arrays, and one for kernel launch—to overlap prefetch and computation. Otherwise “none”.
- **Data structures & dominant size**: Two int matrices `itemsets` and `referrence` (4 B each element) of roughly `(N+1)×(N+1)` dominate VRAM usage.

### Parameters & Inputs
- Scale parameters: `N` and `penalty` (VRAM is mainly driven by `N`).
- External files: None. Two sequences of length `N` are randomly generated internally (kept per original README).

## Directory Structure
- `Managed/needle.cu`: Unified Memory main program (UM flag parsing, timing, VRAM estimate, optional multi-stream).
- `Managed/needle_kernel.cu`: core CUDA kernel (two-phase diagonal updates).
- `Managed/needle.h`: `BLOCK_SIZE` macro.
- `Managed/Makefile`: build script (`multistream` and alias `prefetch`).
- `Managed/run`: sample script.
- `UnManaged/needle.cu`: explicit-copy main program (legacy `TRACEBACK` prints to file).
- `UnManaged/needle_kernel.cu`, `UnManaged/needle.h`, `UnManaged/Makefile`, `UnManaged/run`: corresponding kernel, macros, build script.

(The original README’s valid info is preserved: sequences are random; `N` must be a multiple of 16; work-group size can be tuned via `RD_WG_SIZE_0` etc.)

## Build
Default build & clean:
```bash
make -C Managed
make -C UnManaged
make -C Managed clean
make -C UnManaged clean
```

Compile-time options:
- Thread-block/grid macros (see source for defaults):
  - `-DRD_WG_SIZE_0=<n>` (or `-DRD_WG_SIZE_0_0`, `-DRD_WG_SIZE`): sets `BLOCK_SIZE` (default 16; common values 16/32). Affects parallel granularity & shared memory, not VRAM.
- Output control macros:
  - Managed: with `-DBENCH_PRINT` writes `result.txt` (traceback sequence); default off to reduce I/O.
  - UnManaged: `TRACEBACK` macro enabled in source writes `result.txt`; to disable, comment it in `UnManaged/needle.cu` (no equivalent compile flag).
- Multi-stream (Managed only):
  - `-DMULTISTREAM`: executes `cudaMemPrefetchAsync` (when `PF` set) in multiple streams and launches kernel in its own stream.
  - `Managed/Makefile` target `multistream`; alias `prefetch` kept.

Example (custom block size & result write):
```bash
make -C Managed CFLAGS="-DRD_WG_SIZE_0=32 -DBENCH_PRINT"
```

Note: All Makefile recipe lines must use TAB, not spaces.

## Run
### Managed
Command line:
```bash
./needle N penalty [UM flags]
```
- Workload parameters:
  - `N`: sequence length (`N%16==0`). Internally dimension is `N+1`.
  - `penalty`: positive integer.
- UM strategy flags:
  - `DEV gpu_id`: select compute device (e.g. `0`).
  - `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` designate accessor.
  - `RM dev|cpu`: `cudaMemAdviseSetReadMostly` designate mostly-read.
  - `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation` set preferred residence.
  - `PF dev|cpu`: `cudaMemPrefetchAsync`; single-stream build prefetches sequentially then syncs; `MULTISTREAM` build prefetches in split streams overlapping kernel.

Examples:
```bash
# Single stream: AB to GPU0 + PL to CPU + prefetch to GPU0
./needle 32768 10 AB 0 PL cpu PF 0

# Multi-stream build: AB/RM/PL/PF all to GPU0
./needle 32768 10 AB 0 RM 0 PL 0 PF 0

# ReadMostly only
./needle 16384 10 RM 0
```

Suggested `N` for different VRAM targets (rounded down to multiple of 16):

| Target GiB | N |
|---|---|
| 1  | 11 584 |
| 4  | 23 168 |
| 8  | 32 768 |
| 12 | 40 128 |
| 16 | 46 336 |
| 20 | 51 760 |
| 24 | 56 736 |
| 30 | 63 472 |

See “Memory vs Parameters” for derivation; `N` must be a multiple of 16.

### UnManaged
Command line:
```bash
./needle N penalty
```
Examples (same sizes as Managed for comparison):
```bash
./needle 32768 10
./needle 16384 10
```

## Program Input
None. Two sequences of length `N` are generated randomly inside the program (same as original README).

## Program Output
- Stdout: at least `Total elapsed time: X.XXX s`.
- File output:
  - Managed: with `-DBENCH_PRINT` writes `result.txt` (traceback); overwrite if exists; default off.
  - UnManaged: writes `result.txt` by default (controlled by `TRACEBACK` macro).

## Memory vs Parameters
Main VRAM allocations (`int`=4 B):
```text
itemsets   : (N+1)×(N+1)×4
referrence : (N+1)×(N+1)×4

Mem_bytes = 2×(N+1)×(N+1)×4 ≈ 8·N²
Mem_GiB   = Mem_bytes / 2^30
```
Target VRAM `M_GiB` →
```text
N ≈ floor_to_16( sqrt( M_GiB · 2^30 / 8 ) )
```
Note: Actual usage observed by `nvidia-smi` includes context, driver, UM alignment overhead; usually slightly higher than theoretical arrays. Use program-printed estimate and observed numbers as reference.

## Suggestions
- Start with default `BLOCK_SIZE` 16; raising to 32 may increase throughput but watch shared memory and register usage.
- For very large cases, test the Managed version first to leverage on-demand migration and optional prefetch/multi-stream overlap.

## Notes
- UM switches `AB/RM/PL/PF` are independent and combinable (Managed only); `PF` device should match compute GPU.
- Multi-stream (`Managed` only): enable `-DMULTISTREAM` to advise and prefetch in split streams overlapping computation.
- Large parameters: `size_t` is used for byte counts to avoid overflow; kernel indices still rely on `int`—ensure no overflow.
- When changing block/grid sizes, stay within device limits; retain `cudaDeviceSynchronize()` at critical points.
- This README consistently uses `Mem_bytes` and GiB/MiB (`GiB=2^30`, `MiB=2^20`). Relative paths are quoted with backticks (e.g., `Managed/Makefile`).


