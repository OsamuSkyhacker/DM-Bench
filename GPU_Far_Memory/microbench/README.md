## GPU Far Memory Microbench

### Introduction
microbench provides two types of CUDA memcpy microbenchmarks: bandwidth and latency. It covers Host↔Device as well as Device↔Device, supports two host memory modes—pageable and pinned (including write-combined)—and can be combined with numactl via external scripts for NUMA binding. All results are emitted as CSV for convenient visualization and comparison.

### Directory Structure
- Makefile: builds two executables
- copy_bandwidth_test.cu: bandwidth test (H2D/D2H/D2D), outputs bandwidth_results.csv
- copy_latency_test.cu: latency test (H2D/D2H/D2D), outputs latency_results.csv (or custom)
- gpu_mem_bandwidth_benchmark.sh: batch bandwidth test script (with NUMA binding and CSV renaming)
- gpu_mem_latency_benchmark.sh: batch latency test script (with NUMA binding and CSV renaming)

### Prerequisites
- CUDA Toolkit (nvcc/runtime) with NVIDIA driver installed
- Optional: numactl (if NUMA binding is needed)
- Default compile flag is `-arch=sm_80`; modify `NVCC_FLAGS` in the Makefile for other GPU architectures

### Build
```bash
cd "GPU far memory/microbench"
make          # builds copy_latency_test and copy_bandwidth_test
make clean    # removes executables
```

### Executable Usage
#### 1) Bandwidth: copy_bandwidth_test
- Direction: h2d | d2h | d2d (if omitted, all three are measured)
- Host memory: pageable | pinned (default pinned); optional wc (effective only when pinned)
- Device: `--device <GPU_ID>`
- Data sizes: by default 2^10–2^30 bytes (1 KB–1 GB); edit MIN_POWER/MAX_POWER in the source to change
- Output: `bandwidth_results.csv` with fields `TestDirection,DataSize(KB),Bandwidth(GB/s)`

Examples:
```bash
# H2D, pinned+wc, measure H2D only, device 0
./copy_bandwidth_test pinned wc h2d --device 0

# D2H, pinned (no wc)
./copy_bandwidth_test pinned d2h --device 0

# D2D bandwidth
./copy_bandwidth_test d2d --device 0

# With NUMA binding (if needed)
numactl --membind=0 --cpunodebind=0 ./copy_bandwidth_test pinned h2d --device 0
```

#### 2) Latency: copy_latency_test
- Direction: h2d | d2h | d2d (if omitted, all three are measured)
- Host memory: pageable | pinned (default pinned); optional wc (effective only when pinned)
- Device: `--device <GPU_ID>`
- Iterations: `--iters <N>` (per-size iteration count, default 10000)
- Data range: `--range <min_pow> <max_pow>` (sizes are 2^pow bytes, default 2^1–2^16)
- CSV file name: `--csv <file>` (default latency_results.csv)
- Output: `Direction,DataSize(Bytes),AvgLatency(us)`

Examples:
```bash
# H2D, range 2^10–2^30 B, 10,000 iterations
./copy_latency_test h2d --device 0 --range 10 30 --iters 10000 --csv latency_results.csv

# D2H, pinned+wc, custom CSV name
./copy_latency_test d2h pinned wc --device 0 --csv d2h_wc.csv

# D2D latency
./copy_latency_test d2d --device 0

# With NUMA binding (if needed)
numactl --membind=0 --cpunodebind=0 ./copy_latency_test h2d --device 0 --range 10 30
```

### Batch Scripts
- gpu_mem_bandwidth_benchmark.sh: runs bandwidth tests in a loop and automatically renames CSV files.
- gpu_mem_latency_benchmark.sh: runs latency tests in a loop and automatically renames CSV files.

Key variables in the scripts (edit at the top of each script):
- `device`: CUDA device ID
- `numactl_node` / `NODE_LOCAL`: NUMA binding nodes
- `iterations`: number of overall repetitions (CSV files are renamed with the iteration index)
- `iter_per_size` (latency script only): iterations per data size

Run examples:
```bash
bash gpu_mem_bandwidth_benchmark.sh
bash gpu_mem_latency_benchmark.sh
```

### Notes
- Write-Combined (wc) requires a newer CUDA runtime; the source includes version guards and falls back to regular pinned memory on older runtimes.
- The bandwidth test uses a 1GB CPU cache flush buffer (FLUSH_SIZE) by default; adjust in the source as needed.
- For stability, keep the GPU idle, fix power/frequency if possible, and do a short warm-up before measurements.

### Results & Analysis
CSV files are generated in the current directory as described by each program/script. Use any tool (Python/Excel/gnuplot) for aggregation and visualization.


