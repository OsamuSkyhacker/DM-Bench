# DummyCudaMalloc / gpu_mem_hog

This sample program is used to allocate and hold a specified amount of GPU memory, making it convenient to intentionally reduce available VRAM when studying oversubscription.

## Build

```bash
make            # Build with default NVCC options
# Or customize NVCC options, for example:
make NVCC_FLAGS="-O3 -std=c++17"
```

After compilation, an executable named `gpu_mem_hog` will be generated.

## Run

```bash
./gpu_mem_hog <GB_to_allocate> [device_id]   # Directly specify the GPU index
./gpu_mem_hog list                           # List all GPUs and exit
# You can also specify the default GPU via the GPU_DEV environment variable
export GPU_DEV=1 && ./gpu_mem_hog 8          # Equivalent to ./gpu_mem_hog 8 1
```

Examples:

```bash
./gpu_mem_hog 10       # Allocate and lock 10 GB VRAM on GPU 0
./gpu_mem_hog 6 1      # Allocate and lock 6 GB VRAM on GPU 1
```

The program will keep the memory allocated until it receives Ctrl-C (SIGINT) or `kill` (SIGTERM); it then releases the memory and exits.

## Typical workflow

1. Use `nvidia-smi` to check the current free memory `free_MB`.
2. Compute the desired remaining VRAM. For example, if the GPU has 24 GB total and you want to leave 8 GB free, allocate `free_MB - 8GB`.
3. Run `gpu_mem_hog` to occupy VRAM, then start the target workload (e.g., the BFS benchmark).
4. By changing the allocation size you can simulate various oversubscription scenarios such as 110%, 125%, 150%, etc.
5. After testing, press `Ctrl-C` to terminate `gpu_mem_hog`; the memory is released immediately.

