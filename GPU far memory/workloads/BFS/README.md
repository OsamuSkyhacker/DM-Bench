## BFS Benchmark

This directory contains a GPU-based BFS (Breadth-First Search) benchmark derived from Rodinia with two implementations:
- `Managed/`: Uses CUDA Unified Memory, supports strategies such as `cudaMemAdvise` / `cudaMemPrefetchAsync`, and provides a multi-stream variant via `MULTISTREAM`.
- `UnManaged/`: Uses explicit device memory allocation and copy.

Both versions include wall-clock timing; at the end of execution the program prints `Total elapsed time: X.XXX s`.

## Workload Overview

### Algorithm
- BFS performs level-synchronous traversal on an unweighted graph: starting from a source vertex, it explores the graph layer by layer, relaxing the shortest distance of each reachable vertex.
- Input graph is stored as an adjacency list: an array of nodes records the start offset and edge count for each vertex; an edge array stores the destination vertex (an optional weight field is ignored in BFS).
- Output is the level (distance) of each vertex from the source (`-1`/`INT_MAX` for unreachable, depending on implementation).

### GPU Parallel Mapping
- Typical level-synchronous scheme:
  - Maintain three boolean/flag arrays by vertex: `graph_mask` (current frontier), `updating_graph_mask` (next frontier buffer), `graph_visited` (visited flag).
  - Kernel-1: For vertices in the current frontier, scan their neighbors; unvisited neighbors are marked in `updating_graph_mask` and their `cost` (level) is written.
  - Kernel-2: Move `updating_graph_mask` to `graph_mask`, clear the buffer, and check whether a new frontier exists to decide continuation.
- Threads/blocks: Usually “one or several vertices per thread”, grid/block shaped by `numNodes`. High throughput can be achieved without shared memory.
- Multi-stream (`MULTISTREAM`, Managed only): Assign several prefetch streams to different data structures (nodes/edges/state arrays); the kernel runs in a dedicated stream, overlapping data migration with computation.

### Parameters & Inputs
- Runtime parameters:
  - `Managed/`: `./bfs <input_file> [DEV gpu_id] [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]`
  - `UnManaged/`: `./bfs <input_file> [DEV gpu_id]`
- Input file: mandatory. Format is described in “Program Input”.
- Key sizes: number of vertices `N=numNodes`, number of edges `M=totalEdges` (average degree `d≈M/N` describes density).

## Directory Structure
- `Managed/`
  - `bfs.cu`, `kernel.cu`, `kernel2.cu`
  - `Makefile`, `Makefile_nvidia`
  - `run` (example script)
- `UnManaged/`
  - `bfs.cu`, `kernel.cu`, `kernel2.cu`
  - `Makefile`, `Makefile_nvidia`
  - `run`
- `bfs_data/`
  - Example datasets: `graph256k.txt`, `graph16M.txt`, etc.
  - Generator: `bfs_data/inputGen/graphgen`, script: `bfs_data/gen_dataset.sh`

## Build

- Default build and clean:
```bash
make -C Managed
make -C UnManaged
make -C Managed clean
make -C UnManaged clean
```

- Compile-time options:
  - Multi-stream: `-DMULTISTREAM` (enabled via `make -C Managed multistream`). It prefetches multiple arrays concurrently with `cudaMemPrefetchAsync` while launching kernels in a separate stream for overlap. A backward-compatible target `prefetch` is equivalent to `multistream`.
  - Output control: This task always produces `result.txt`; no extra switch.
  - Block dimension: No general `RD_WG_SIZE_0` macro is exposed; modify the source if you want to experiment with different block sizes.

Example:
```bash
# Enable multi-stream (Managed)
make -C Managed multistream
```

## Run

### Managed
- Executable: `Managed/bfs`
- Command line:
```bash
./bfs <input_file> [DEV gpu_id] [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]
```
- Arguments:
  - `DEV gpu_id`: Select compute device (e.g. `0`).
  - `AB dev|cpu`: `cudaMemAdviseSetAccessedBy` — specify which device/CPU will access the region.
  - `RM dev|cpu`: `cudaMemAdviseSetReadMostly` — hint that the region is mostly read-only.
  - `PL dev|cpu`: `cudaMemAdviseSetPreferredLocation` — suggest initial residency.
  - `PF dev|cpu`: `cudaMemPrefetchAsync`; with single-stream build it runs in the default stream then synchronizes; with `MULTISTREAM` it prefetches arrays in separate streams and overlaps with kernels.

Examples:
```bash
cd Managed
# Run on GPU0 with access hint and prefetch
./bfs ../bfs_data/graph16M.txt DEV 0 AB 0 PF 0
# Prefer CPU residency and prefetch to GPU0
./bfs ../bfs_data/graph16M.txt DEV 0 PL cpu PF 0
```

Memory suggestion should follow “actual allocation” instead of file size. For this implementation (GPU stores only `dest`, no edge weight):
- `Mem_bytes = 15·N + 4·E`, where `N=numNodes`, `E=edge_list_size` (fourth line in file).
- According to generator rule `E ≈ 6·N`, approximate `Mem_bytes ≈ 39·N`.
- `Mem_GiB ≈ Mem_bytes / 2^30`. The value shown by nvidia-smi includes context/UM alignment overhead and is usually higher.
- Examples:
  - `graphgen 262144 256k` (N≈2.62e5, E≈1.57e6) → `Mem_bytes ≈ 10.2 MB` (arrays only).
  - `graphgen 1048576 1M` (N≈1.05e6, E≈6.29e6) → `Mem_bytes ≈ 39 MiB`.
  - `graphgen 16777216 16M` (N≈1.68e7, E≈1.01e8) → theoretical `≈0.61 GiB`; observed ~0.82 GiB (includes overhead).
- For more accuracy, after reading the file plug the printed `no_of_nodes` and `edge_list_size` into `Mem_bytes = 15·N + 4·E`.

### UnManaged
- Executable: `UnManaged/bfs`
- Command line:
```bash
./bfs <input_file> [DEV gpu_id]
```
Example:
```bash
cd UnManaged
./bfs ../bfs_data/graph256k.txt DEV 0
```

## Program Input
- Graph format (relative path examples in `bfs_data/`):
  1. First line: number of vertices `numNodes`
  2. Next `numNodes` lines: `startIndex no_of_edges`
     - `startIndex`: start offset in edge list
     - `no_of_edges`: number of edges of the vertex
  3. Blank line: source vertex ID (integer). Current implementation often fixes `source=0` (ignores this value).
  4. Blank line: total edge count `totalEdges`
  5. Next `totalEdges` lines: `dest weight` (`weight` is unused in BFS; storage depends on implementation)

## Program Output
- Stdout:
  - `Start traversing the tree`, `Kernel Executed <k> times`, etc.
  - `Total elapsed time: X.XXX s` (wall-clock)
- Result file: `result.txt`, each line like `i) cost:<distance>` (overwrite if exists).

## Memory vs. Parameters (estimation)
- Let:
  - `N = numNodes`, `M = totalEdges`
  - 1 int = 4 B
- Main allocations (typical implementation, check source for exact):
  - Node array (offset + edge count): `8·N`
  - Edge array:
    - If only `dest`: `4·M`
    - If `dest`+`weight`: `8·M`
  - BFS state arrays (`cost`, `graph_mask`, `updating_graph_mask`, `graph_visited`, etc.): `≈16·N`
- Therefore (for current implementation: GPU stores only `dest`):
```text
Mem_bytes = 8·N        // graph_nodes: Node{int,int}
           + 4·E       // graph_edges: int
           + 3·N       // three bool flag arrays
           + 4·N       // cost: int
         ≈ 15·N + 4·E
Mem_GiB   = Mem_bytes / 2^30
```
- With generator rule `E ≈ 6·N`, approximate `Mem_bytes ≈ 39·N`, hence target VRAM `M_GiB` implies:
```text
N ≈ (M_GiB · 2^30) / 39
```
- If you modified the implementation to also store `weight` on GPU, replace edge term with `8·E`, giving `Mem_bytes ≈ 19·N + 8·E ≈ 67·N`.

## Notes
- `AB/RM/PL/PF` are independent switches and can be freely combined; the `PF` device should match the compute GPU.
- `MULTISTREAM` is a compile-time switch (`make -C Managed multistream`). When enabled it prefetches arrays in separate streams and launches kernels in their own stream to overlap migration and computation.
- If you change block/grid dimensions yourself, ensure they do not exceed device limits and check for overflow at large scales (use `size_t` for indices and sizes).

