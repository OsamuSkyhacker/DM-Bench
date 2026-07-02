# GPU Far-Memory Benchmarking & Automation Framework (DM-Bench / GPU_Far_Memory)

A unified, configurable, and automated benchmarking framework for **GPU far-memory / oversubscribed VRAM** scenarios.  
It benchmarks multiple workloads under **NUMA placements** and **CUDA Unified Memory (UM) strategies** (AccessedBy / PreferredLocation / ReadMostly / Prefetch), with standardized timing capture and reproducible experiment plans.

## Why this repo
- **One framework, many workloads**: run a test matrix defined in YAML across multiple GPU workloads with a unified CLI.
- **UM strategy matrix**: AB/PL/RM/PF combinations via `um_profiles`, plus consistent argument mapping (and required CUDA-side support).
- **NUMA binding**: optional `numactl` binding for CPU/memory placement control.
- **Controlled oversubscription**: precisely occupy VRAM using `gpu_mem_hog` (`DummyCudaMalloc/`) with a closed-form hog size computed from:
  - real-time `Free0` (via `nvidia-smi`)
  - baseline payload `B_algo`
  - fixed/global overheads
- **Standardized outputs**: normalized runtime format (`xxx.xxx ms`) and unified result layout under `bench/results/...`.
- **Microbench support**: standalone memcpy bandwidth/latency microbenchmarks for auxiliary analysis.

## Repository layout
- `bench/` — **the automation framework**
  - `bin/bench_plan.sh` orchestrates the matrix defined by `configs/plan.yaml`
  - `lib/` timing/NUMA wrapper, hog control, math, YAML parsing
  - `workloads/` unified wrappers (`backprop.sh`, `bfs.sh`, `fdtd2d.sh`, `hotspot.sh`, `nw.sh`, `pathfinder.sh`)
  - `configs/` experiment plan + workload capabilities (`plan.yaml`, `workloads.yaml`)
  - `results/` CSV/logs/tmp/.xtrace outputs (auto-generated)
  - `special/vector_add/` special handling for `vector_add` (different CLI)
- `workloads/` — workload codebase (Managed/UnManaged variants + generators)
  - `DummyCudaMalloc/` — `gpu_mem_hog` (VRAM hog utility)
  - `vector_add/` — special microbenchmark (separate pipeline)
- `microbench/` — standalone CUDA memcpy bandwidth/latency tests (NUMA bindable)

## Prerequisites
- Linux x86-64 (Ubuntu 20.04/22.04 recommended)
- NVIDIA GPU + driver compatible with your CUDA Toolkit
- CUDA Toolkit (nvcc + runtime)
- Build tools: `make`, `gcc/g++`
- Optional: `numactl` (NUMA binding)
- Optional YAML helpers: Python3+PyYAML or `yq`
- `nvidia-smi` (for reading `Free0`)

> Note: Makefiles default to `-arch=sm_80`. Adjust if your GPU differs.

## Quickstart (3 commands)
### 1) Build workloads (excluding vector_add)
```bash
bash GPU_Far_Memory/bench/bin/build_all_workloads.sh
````

### 2) Generate datasets (only if your plan includes these workloads)

```bash
bash GPU_Far_Memory/bench/bin/generate_datasets.sh
```

### 3) Run the experiment plan

```bash
bash GPU_Far_Memory/bench/bin/bench_plan.sh
```

## Outputs

All results are written to:

```
bench/results/<workload>/<timestamp>/
```

Typical artifacts include:

* CSV with normalized runtime (`xxx.xxx ms`)
* logs / raw outputs
* temporary files and optional `.xtrace` for debugging

## Configuration

### `bench/configs/plan.yaml`

Defines the experiment matrix (e.g., workloads × NUMA nodes × UM profiles × oversub settings).

### `bench/configs/workloads.yaml`

Declares per-workload capabilities and parameters, including:

* baseline payload size `B_algo`
* valid UM knobs and argument mapping
* wrapper entrypoints

### UM profiles

UM combinations are defined by `um_profiles` (AB/PL/RM/PF).
If a workload needs CUDA-side changes to support certain hints, ensure the corresponding Managed implementation is enabled.

## Special cases

* `vector_add/` is treated as a special microbenchmark with a distinct CLI and scripts under:

  * `bench/special/vector_add/`

## Cleaning

Clean all workload build artifacts:

```bash
bash GPU_Far_Memory/bench/bin/clean_all_workloads.sh
```

Clean generated datasets:

```bash
bash GPU_Far_Memory/bench/bin/clean_generated_datasets.sh
```

## What to cite 

This repo provides:

* a unified benchmarking framework (`bench/`) with YAML-defined test plans
* reproducible NUMA + UM strategy sweeps
* controlled VRAM oversubscription via `gpu_mem_hog` and closed-form sizing
* standardized parsing and results organization

