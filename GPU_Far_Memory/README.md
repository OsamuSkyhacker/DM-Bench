# GPU Far Memory Benchmarking and Automation Framework 
## Project Overview

This project provides a unified, configurable, and automated benchmarking framework for GPU Far Memory scenarios:
- Uses scripts in `bench/` to execute a matrix defined by `plan.yaml`/`workloads.yaml` across different NUMA nodes and UM strategies (AccessedBy/PreferredLocation/ReadMostly/Prefetch), and collects standardized runtimes (normalized to `xxx.xxx ms`).
- Achieves oversubscription by precisely occupying VRAM using `DummyCudaMalloc/gpu_mem_hog`. The hog size is computed by a closed-form equation based on real-time `Free0`, baseline payload `B_algo`, and fixed/global overheads.
- Treats `vector_add/` as a special microbenchmark (distinct CLI from other workloads); `microbench/` provides memcpy bandwidth/latency microbenchmarks for auxiliary analysis.
- Key contributions: unified test framework and automation (`bench/`); unified timing and parsing; NUMA binding; UM strategy combinations (AB/PL/RM/PF) and argument mapping (including necessary CUDA-side changes); closed-form oversub calculation and `gpu_mem_hog` control; `plan.yaml`/`workloads.yaml` configuration system; unified wrappers per workload; dataset generators; and the `vector_add/` and `microbench/` subprojects.

## What's Included and Directory Layout
- bench/ (unified benchmarking framework)
  - bin/: `bench_plan.sh` (orchestrator), `build_all_workloads.sh`/`clean_all_workloads.sh`, `generate_datasets.sh`/`clean_generated_datasets.sh`
  - lib/: `bench_utils.sh` (timing capture/NUMA wrapper), `hog_ctl.sh` (hog process control), `math.sh` (closed-form computation), `yaml.sh` (multi-level parsing)
  - workloads/: unified CLI wrappers (`backprop.sh`, `bfs.sh`, `fdtd2d.sh`, `hotspot.sh`, `nw.sh`, `pathfinder.sh`)
  - configs/: `plan.yaml` (test matrix), `workloads.yaml` (capabilities/baseline/args)
  - results/: output root (CSV/logs/tmp/.xtrace)
  - special/vector_add/: special scripts and configs for `vector_add`
- workloads/ (workload codebase)
  - backprop, BFS, FDTD-2D, hotspot, nw, pathfinder: Managed/UnManaged versions with their README/run scripts (some include generators)
  - DummyCudaMalloc/: `gpu_mem_hog` (VRAM hogging utility for oversub)
  - vector_add/: Unmanaged/Managed implementations and batch scripts (`bench_both/`)
- microbench/ (standalone microbenchmarks): CUDA memcpy bandwidth/latency tests and batch scripts (NUMA bindable), output CSV
- Outputs and logs: unified under `bench/results/<workload>/<timestamp>/`
- Configurables: `WB_ROOT` (override workloads path), `plan.yaml`, `workloads.yaml`; UM combinations are defined by `um_profiles`

## Prerequisites
- OS: x86-64 Linux (e.g., Ubuntu 20.04/22.04)
- GPU & Driver: NVIDIA GPU with a driver matching CUDA Toolkit
- CUDA Toolkit: includes nvcc and runtime (Makefiles default to `-arch=sm_80`; adjust as needed)
- Optional: `numactl` (NUMA binding), Python3+PyYAML or `yq` (fallback parsing for non-inline YAML)
- nvidia-smi: for reading `Free0`
- Build tools: make, gcc/g++

## Quick Start

- Build all workloads (excluding vector_add):
```bash
bash GPU_Far_Memory/bench/bin/build_all_workloads.sh
```
- Generate datasets (if needed for BFS/Hotspot):
```bash
bash GPU_Far_Memory/bench/bin/generate_datasets.sh
```
- Run the test plan:
```bash
bash GPU_Far_Memory/bench/bin/bench_plan.sh
```

## Cleaning Up
- Clean all workload build artifacts:
```bash
bash GPU_Far_Memory/bench/bin/clean_all_workloads.sh
```
- Clean generated datasets:
```bash
bash GPU_Far_Memory/bench/bin/clean_generated_datasets.sh
```

## Configuration Files

- `bench/configs/plan.yaml`
  - **gpu_id**: select GPU ID
  - **nodes**: NUMA node labels and bindings, e.g., `local: {membind: 0, cpunodebind: 0}`
  - **ratios**: oversubscription ratios
  - **workloads**: workloads to run (e.g., `backprop`, `bfs`)
  - **modes**: `unmanaged` and/or `um`
  - **um_profiles**: UM combos (pf/ab/pl/rm), executed only when `modes` contains `um`

- `bench/configs/workloads.yaml`
  - **gpu.task_overhead_mb/global_task_mb/free_adjust_mb**: global parameters for hog calculation
  - **workloads.<name>.baseline_payload_mb**: baseline “algorithm payload” size (MB)
  - **execs/args/parse**: binaries, fixed arguments, and time parsing regex per workload

## Subproject Overview (see each README for details)

### microbench/
- memcpy bandwidth and latency microbenchmarks for Host-to-Device (H2D), Device-to-Host (D2H), and Device-to-Device (D2D); supports `pageable`/`pinned`/`pinned+wc` host memory.
- Can be combined with `numactl` for NUMA binding; results written as CSV for easy visualization.
- See `GPU_Far_Memory/microbench/README.md` for details.

### vector_add/
- Minimal end-to-end vector-add microbenchmark with `Unmanaged` (cudaMemcpy/Zero-Copy) and `Managed` (UM + PF/Advise) variants plus batch scripts (`bench_both/`).
- Its CLI differs from other workloads; treated as a special case in the unified framework.
- See `GPU_Far_Memory/workloads/vector_add/README.md` for details.

### Workload and Tool Usage Guide
- backprop: `GPU_Far_Memory/workloads/backprop/README.md`
- BFS: `GPU_Far_Memory/workloads/BFS/README.md`
- FDTD-2D: `GPU_Far_Memory/workloads/FDTD-2D/README.md`
- hotspot: `GPU_Far_Memory/workloads/hotspot/README.md`
- nw (Needleman–Wunsch): `GPU_Far_Memory/workloads/nw/README.md`
- pathfinder: `GPU_Far_Memory/workloads/pathfinder/README.md`
- vector_add (special microbenchmark): `GPU_Far_Memory/workloads/vector_add/README.md`
- DummyCudaMalloc (hog utility): `GPU_Far_Memory/workloads/DummyCudaMalloc/README.md`

Tip: `GPU_Far_Memory/workloads/DummyCudaMalloc` can be built standalone via `make` to produce `gpu_mem_hog`. The framework invokes and manages it automatically.

## Execution Flow Overview
1. For each `(workload, ratio)`, compute hog size via the closed-form equation and start `gpu_mem_hog`.
2. Iterate `nodes`, wrapping execution with `numactl --membind/--cpunodebind`.
3. Execute `unmanaged` and/or UM variants as defined by `modes` and `um_profiles`.
4. Capture runtime to CSV; write logs and temporary outputs.

## Results and Artifacts
- Root: `GPU_Far_Memory/bench/results/<workload>/<timestamp>/`
- CSV (per node): `runtime_<node>.csv`
- Logs: `<workload>.<node>.<variant>.log` (variant like `unmanaged` or a UM profile name)
- Temporary outputs (for time parsing):
  - Unmanaged: `tmp.unmanaged.<node>.txt`
  - UM: `tmp.um.<node>.<profile>.txt`
- Command tracing (if enabled): `<workload>.<node>.<variant>.xtrace.log`

Sample CSV (columns match the execution matrix; missing strategies show NA):
```
Oversub,unmanaged,um_pf,um_ab
1.0,1234.567,890.123,456.789
```

## Debugging and Tracing
- Enable debug prints:
```bash
export BENCH_DEBUG=1
```
- Enable command-level tracing (separate `.xtrace.log` without polluting tmp output):
```bash
export BENCH_TRACE=1
```
- At the end of each log, locate `[tmp]` to open the corresponding `tmp.*.txt` for raw stdout/stderr.
- Hog process log: `/tmp/hog.<gpu_id>.log` (includes “Actually allocated”).

## Run a Specific Combination
To run a single combination:
```bash
numactl --membind=<node> --cpunodebind=<node> bash GPU_Far_Memory/bench/workloads/<workload>.sh run --mode <unmanaged|um> --gpu <id>
```
Example:
```bash
numactl --membind=0 --cpunodebind=0 bash GPU_Far_Memory/bench/workloads/backprop.sh run --mode unmanaged --gpu 0
```

## Terms & Abbreviations
- Free0: GPU free memory (MB) read before executing a ratio
- FreeAdj: correction to Free0 (MB), from `workloads.yaml.free_adjust_mb`
- B_algo: baseline “algorithm payload” (MB) per workload, from `workloads.yaml`
- O (task_overhead_mb): per-process fixed VRAM overhead (MB), from `workloads.yaml`
- F_task (global_task_mb): per-task global overhead (MB) when the GPU has work
- R: oversub ratio (`plan.yaml.ratios`)
- hog: memory occupied by DummyCudaMalloc to create external pressure

## CUDA Memory Advice (cudaMemAdvise) Semantics
- AB (AccessedBy): `--um-ab cpu|gpuN` → `cudaMemAdviseSetAccessedBy`
- PL (PreferredLocation): `--um-pl cpu|gpuN` → `cudaMemAdviseSetPreferredLocation`
- RM (ReadMostly): `--um-rm cpu|gpuN` → `cudaMemAdviseSetReadMostly`
- PF (Prefetch): `--um-pf` → `cudaMemPrefetchAsync(..., gpu_id)`
Note: `gpuX` is replaced at runtime by the current `gpu_id`. You may also specify an explicit `gpuN`.

## Oversubscription Closed‑Form Equation
Compute target hog size (in MB; GB only for display):
```
x_GB = (Free0 - FreeAdj - F_task - (B_algo / R + O) - O) / 1024
```
- `x_GB` is converted back to MB when passed to `gpu_mem_hog`.
- Example: `Free0=12000, FreeAdj=0, F_task=3, B_algo=10248, R=1.0, O=251` → `x_GB ≈ 1.261`.

Why ratios > 1.0 are meaningful
- The hog method is a practical trick: it reduces the free VRAM externally, without changing the workload’s own working set. When the oversubscription ratio R ≤ 1.0, the effective free VRAM after hogging is still ≥ B_algo (plus overheads), so the entire working set can reside on the GPU. In this regime, UM will not be forced to migrate pages, and Unmanaged also fits; runtimes are therefore close to baseline.
- Only when R > 1.0 does the effective free VRAM become < B_algo (plus overheads). Unmanaged commonly OOMs (as noted), while UM triggers on-demand migrations/page faults; this is where far-memory behavior and performance differences appear and the experiment becomes meaningful.
- Recommendation: choose R > 1.0 for far-memory studies; use 0 < R ≤ 1.0 mainly for sanity checks or environment validation.

## Troubleshooting
- UM not executed: ensure `modes` contains `um` and `um_profiles` is non-empty; the initial debug output should list profile names. Without PyYAML/yq, only inline `nodes/um_profiles` are supported.
- Node parsing failure: install PyYAML or `yq`; or write `nodes` in inline form `{membind: N, cpunodebind: M}`.
- Empty CSV: check `.log` and `tmp.*.txt` for `... ms` or `... s`; use `export BENCH_TRACE=1` to capture `.xtrace.log`.
- Unmanaged: NA is expected when oversubscription ratio > 1.0 (out-of-memory).
- Binary not found: set `export WB_ROOT=<your_path>/workloads` (path must not contain spaces).
- numactl unavailable: the script falls back to direct execution without NUMA binding.

## plan.yaml Specification

- gpu_id: integer GPU index used for both workloads and hog binding.
- nodes: mapping of node labels to bindings; inline YAML recommended for robust fallback parsing:
```yaml
nodes:
  local:  {membind: 0, cpunodebind: 0}
  remote: {membind: 1, cpunodebind: 0}
  pmem:   {membind: 2, cpunodebind: 0}
```
- Only explicitly declared labels will be executed.
- ratios: list of oversub ratios (float or integer), e.g., `[1.0, 1.3, 1.7, 2.0]`.
- workloads: names must match `workloads.yaml.workloads.<name>` exactly.
- modes: `unmanaged` and/or `um`.
  - UM combinations are executed only when `um` is included and `um_profiles` is non-empty.
- um_profiles: list of UM strategy combos; dynamically builds UM variants and CSV columns.
  - Fields:
    - name: unique identifier (used in log suffix and CSV header)
    - pf: boolean, enable `cudaMemPrefetchAsync`
    - ab/pl/rm: target string among `none|cpu|gpuX|gpuN`
      - gpuX: replaced by current `gpu_id` (e.g., `gpu0`)
      - gpuN: explicitly select a GPU (e.g., `gpu1`)
      - cpu/none: bind CPU or skip advise respectively
  - Example:
```yaml
um_profiles:
  - name: um_base
    pf: false
    ab: none
    pl: none
    rm: none
  - name: um_pf
    pf: true
    ab: none
    pl: none
    rm: none
  - name: um_ab_gpu0_pl_cpu
    pf: false
    ab: gpu0
    pl: cpu
    rm: none
  - name: um_rm_gpu0
    pf: false
    ab: none
    pl: none
    rm: gpu0
```

Note: if `nodes` is not inline, installing PyYAML or `yq` is recommended to avoid limitations of the awk fallback.

## workloads.yaml Specification

- gpu: global GPU parameters for hog calculation and Free0 correction.
  - id: default GPU ID (currently overridden by `plan.yaml.gpu_id`)
  - task_overhead_mb (O): per-process fixed VRAM overhead (MB)
  - global_task_mb (F_task): global overhead (MB) when the GPU has any work
  - free_adjust_mb (FreeAdj): correction for observed Free0 (MB); effective free ≈ Free0 - FreeAdj
- workloads.<name>: per-workload configuration.
  - execs.unmanaged/managed: binary paths, typically using `${WB_ROOT}/...`; `WB_ROOT` defaults to `GPU_Far_Memory/workloads` and can be overridden via environment variable
  - baseline_payload_mb (B_algo): baseline payload (MB), used by the closed-form hog computation
  - supports: capability flags (informational)
    - um/ab/pl/rm/pf: whether the workload supports UM and each advise/prefetch
  - args.fixed: fixed CLI arguments appended when running the workload (e.g., Hotspot dataset paths and `/dev/null`)
  - Optional fields:
    - args.input: additional input for certain workloads (e.g., BFS `graph64M.txt`)
  - parse.sec_regex: regex to parse seconds; framework converts to `xxx.xxx ms`

Example (excerpt):
```yaml
gpu:
  task_overhead_mb: 251
  global_task_mb: 3
  free_adjust_mb: 10

workloads:
  backprop:
    execs:
      unmanaged: "${WB_ROOT}/backprop/UnManaged/backprop"
      managed:   "${WB_ROOT}/backprop/Managed/backprop"
    baseline_payload_mb: 10248
    supports: {um: true, ab: true, pl: true, rm: true, pf: true}
    args:
      fixed: ["1048560", "1280"]
    parse:
      sec_regex: "Total elapsed time: ([0-9]+\\.[0-9]+) s"
```

## FAQ
- If CSV shows NA, inspect `.log` and `tmp.*.txt` for parseable time (`... ms` or `... s`). For `oversub > 1.0` under UnManaged, NA due to OOM is expected.
- If wrapper reports missing binary, ensure `WB_ROOT` points to the correct path, or set `export WB_ROOT=<your_path>/workloads` (paths must not contain spaces).

## Acknowledgments

- The CUDA implementations of backprop, BFS, FDTD-2D, hotspot, nw, and pathfinder reference benchmarks from UVMSmart (UVM Smart, `https://github.com/DebashisGanguly/gpgpu-sim`).
- Additional thanks to Rodinia Benchmark Suite (3.1) for classic benchmarks and documentation (`http://lava.cs.virginia.edu/wiki/rodinia`).
