// gpu_mem_hog.cu
//
// Purpose: Allocate GPU VRAM (in GB) and keep it resident until Ctrl-C (SIGINT) or SIGTERM is received.
//          Uses cudaMalloc (non-Unified Memory) to ensure this portion of memory stays on the GPU and is not migrated by the driver.
//
// Build:   nvcc -O2 -std=c++11 gpu_mem_hog.cu -o gpu_mem_hog
// Run:     ./gpu_mem_hog <memory_to_allocate_GB> [device_id]
// Example: ./gpu_mem_hog 10           // Allocate 10 GB on GPU0 by default
//          ./gpu_mem_hog 6 1          // Allocate 6 GB on GPU1
//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <csignal>
#include <thread>
#include <chrono>

static volatile std::sig_atomic_t g_keepRunning = 1;
void handler(int) { g_keepRunning = 0; }

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <GB_to_allocate> [device_id]\n"
                  << "       " << argv[0] << " list                # list available GPUs\n";
        return 1;
    }

    // If the user enters "list" or "-l", print GPU devices and exit
    std::string arg1(argv[1]);
    if (arg1 == "list" || arg1 == "-l" || arg1 == "--list") {
        int cnt = 0; cudaGetDeviceCount(&cnt);
        for (int i = 0; i < cnt; ++i) {
            cudaDeviceProp prop; cudaGetDeviceProperties(&prop, i);
            std::cout << "GPU" << i << ": " << prop.name << ", sm:" << prop.major << "." << prop.minor << "\n";
        }
        return 0;
    }

    double gb_to_alloc = 0.0;
    int dev = 0;
    try {
        gb_to_alloc = std::stod(argv[1]);
        // Parse device ID: prefer command-line arg, then GPU_DEV env var, otherwise default to 0
        if (argc >= 3) {
            dev = std::stoi(argv[2]);
        } else if (const char* env = std::getenv("GPU_DEV")) {
            dev = std::stoi(env);
        }
    } catch (const std::exception&) {
        std::cerr << "Invalid argument. Usage: " << argv[0] << " <GB_to_allocate> [device_id]\n";
        return 1;
    }

    // Register signal handlers early so a SIGTERM during allocation still exits cleanly
    std::signal(SIGINT,  handler);
    std::signal(SIGTERM, handler);

    // A failed cudaSetDevice would silently fall back to GPU0; treat it as fatal instead
    cudaError_t err_dev = cudaSetDevice(dev);
    if (err_dev != cudaSuccess) {
        std::cerr << "cudaSetDevice(" << dev << ") failed: "
                  << cudaGetErrorString(err_dev) << "\n";
        return 2;
    }

    size_t bytes_to_alloc = static_cast<size_t>(gb_to_alloc * 1024.0 * 1024.0 * 1024.0);

    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    std::cout << "GPU" << dev << " total memory: " << total_b / (1024*1024) << " MB, "
              << "current free: " << free_b / (1024*1024) << " MB\n";
    std::cout << "Attempting to allocate " << gb_to_alloc << " GB...\n";

    const size_t CHUNK = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // 1 GB per chunk
    std::vector<void*> ptrs;
    size_t alloced = 0;

    while (alloced < bytes_to_alloc) {
        size_t this_chunk = std::min(CHUNK, bytes_to_alloc - alloced);
        void *p = nullptr;
        cudaError_t err = cudaMalloc(&p, this_chunk);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed, already successfully allocated "
                      << alloced / (1024*1024) << " MB: "
                      << cudaGetErrorString(err) << "\n";
            break;
        }
        cudaMemset(p, 0, this_chunk);       // Touch memory to ensure physical allocation
        ptrs.push_back(p);
        alloced += this_chunk;
    }

    // On shortfall, do NOT print the "Actually allocated" readiness marker:
    // the framework greps for it to decide the hog is ready, and a partial hog
    // would silently invalidate the oversubscription ratio.
    if (alloced < bytes_to_alloc) {
        std::cerr << "Allocation failed: got " << alloced / (1024*1024)
                  << " MB of " << bytes_to_alloc / (1024*1024) << " MB requested\n";
        for (void *p : ptrs) cudaFree(p);
        return 3;
    }

    std::cout << "Actually allocated " << alloced / (1024*1024) << " MB\n"
              << "Press Ctrl-C to free and exit.\n";

    while (g_keepRunning) std::this_thread::sleep_for(std::chrono::seconds(1));

    for (void *p : ptrs) cudaFree(p);
    std::cout << "Memory released, exiting.\n";
    return 0;
}
