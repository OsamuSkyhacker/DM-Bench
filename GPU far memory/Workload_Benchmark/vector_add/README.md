## vector_add (GPU Far Memory Microbenchmark)

### Overview
This directory provides a minimal vector-add microbenchmark designed to evaluate end-to-end latency across different memory paths and strategies on GPUs. It covers:
- cudaMemcpy (host pinned + device memory)
- Zero-Copy (host pinned mapped memory, device direct access)
- Unified Memory (UM)
- UM + Prefetch (PF)
- UM + cudaMemAdvise (AccessedBy / PreferredLocation / ReadMostly)

NUMA/PMem batch scripts and CSV outputs are included to facilitate systematic testing.

### Layout
```
vector_add/
  Managed/        # Unified Memory version (vector_add_um)
  Unmanaged/      # Traditional/Zero-Copy version (vector_add)
  bench_both/     # Batch scripts and example results
    bench_all.sh
    runtime_local_dram.csv
    runtime_remote_dram.csv
    runtime_local_pmem.csv
    runtime_all.csv        # optional summary from other sweeps
  legacy_code/    # Historical scripts/kernels (reference only)
```

### Requirements
- Linux + CUDA Toolkit
- A GPU architecture supported by your toolchain (Makefiles default to `-arch=sm_80`; adjust as needed)
- Optional: `numactl` for memory node and CPU affinity control
- Optional: a machine with multiple NUMA nodes and/or PMem

### Build
Build both variants separately:
```bash
cd Unmanaged && make        # produces ./vector_add
cd ../Managed && make       # produces ./vector_add_um
```
If your GPU architecture differs, change `-arch=sm_80` in both `Managed/Makefile` and `Unmanaged/Makefile` (e.g., `sm_70`, `sm_90`).

### Run
#### 1) Unmanaged (cudaMemcpy vs Zero-Copy)
Executable: `./vector_add`

- Flags
  - `-d, --device <id>`: select GPU
  - `-r, --ratio <double>`: size derived from a fraction of device memory (consistently estimated as 3 buffers)
  - `-p, --policy <map|nonmap>`: switch between Zero-Copy and cudaMemcpy
  - Positional: you can also specify `n` directly (number of elements; lower priority than `--ratio`)

- Examples
```bash
# Traditional cudaMemcpy (recommended when oversub ≤ 1)
./vector_add -d 0 -r 0.975 -p nonmap

# Zero-Copy (device directly accesses mapped host memory)
./vector_add -d 0 -r 0.975 -p map
```

- Sample output
```
[GPU0] n=... elements  Nonmap latency: 997.072 ms
[GPU0] n=... elements  Map latency:    719.775 ms
```

#### 2) Managed (Unified Memory and strategies)
Executable: `./vector_add_um`

- Flags
  - `-d, --device <id>`: select GPU
  - `-r, --ratio <double>`: size derived from a fraction of device memory (still estimated as 3× buffers)
  - `-pf`: prefetch `a/b/o` to the GPU within the timing window
  - `--advise <kind> <tgt>`:
    - `ab gpuX`: `cudaMemAdviseSetAccessedBy` (target `gpuX`)
    - `pl cpu|gpuX`: `cudaMemAdviseSetPreferredLocation` (CPU or a specific GPU)
    - `rm`: `cudaMemAdviseSetReadMostly` (calls are commented out by default in the source)
  - Positional: you can also specify `n` directly

- Timing semantics
  - Perform warm-up (W=10). The measured window includes: applying advices, optional prefetch, kernel compute, then prefetching output `o` back to CPU.
  - Repeats R=1 by default (end-to-end scenario).

- Examples
```bash
# Basic UM
./vector_add_um -d 0 -r 0.975

# UM + Prefetch
./vector_add_um -d 0 -r 0.975 -pf

# UM + AccessedBy(GPU0)
./vector_add_um -d 0 -r 0.975 --advise ab gpu0

# UM + PreferredLocation(cpu)
./vector_add_um -d 0 -r 0.975 --advise pl cpu
```

- Sample output
```
[GPU0] n=...  avg latency=1741.351 ms  (R=1 W=10)
```

### NUMA/PMem batch script
Script: `bench_both/bench_all.sh`

- Purpose: For local DRAM, remote NUMA DRAM, and local PMem, sweep multiple oversubscription ratios across five strategies (cudaMemcpy / Zero-Copy / UM / UM+PF / UM+AB) and write three CSV files.
- Key variables (tweak at top of the script):
  - `VEC_COPY="./vector_add"`, `VEC_UM="./vector_add_um"`
  - `GPU_ID=0`
  - `NODE_LOCAL=0`, `NODE_REMOTE=1`, `NODE_PMEM=2`
- Oversubscription points: `0.3 0.7 1.0 1.5 2.0`. The actual `--ratio` passed to executables subtracts `0.025` to keep size estimation consistent across paths.
- Output files:
  - `runtime_local_dram.csv`
  - `runtime_remote_dram.csv`
  - `runtime_local_pmem.csv`
  - Header: `Oversub,cudaMemcpy,ZeroCopy,UM,UM+PF,UM+AB` (missing strategies are filled with `NA`).

Run:
```bash
cd bench_both
./bench_all.sh
```

### CSV example (excerpt)
`bench_both/runtime_local_dram.csv`
```
Oversub,cudaMemcpy,ZeroCopy,UM,UM+PF,UM+AB
0.3,281.069,202.806,748.438,492.864,205.003
...
```

### Notes and FAQ
- Architecture: If your GPU differs from `sm_80`, update both Makefiles accordingly.
- For oversub > 1, `cudaMemcpy` is skipped by default (CSV shows `NA`) to avoid OOM.
- Timing uses CUDA events. Under UM, the window includes migrating `o` back to CPU. For Zero-Copy/cudaMemcpy, each path’s transfer is measured accordingly.
- `--advise rm` (ReadMostly) calls are commented out in the current UM source; enable them if needed.
- A multi-NUMA environment and `numactl` are needed to fully reproduce the experiments. Without PMem, you can ignore the PMem section or set `NODE_PMEM` to the local node for an apples-to-apples comparison.

### Porting to other workloads
- Add a `bench_both/` folder under each workload and reuse this script/CSV naming.
- Binary naming convention: Unmanaged as `<workload>`, Managed as `<workload>_um`; use `--policy map|nonmap` to switch Zero-Copy vs cudaMemcpy where available.
- Unify program output to print latency like `123.456 ms` to make regex extraction consistent.


