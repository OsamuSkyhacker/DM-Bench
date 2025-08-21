// main_updated.cpp – replace --map/--nonmap with --policy <map|nonmap>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// -------------------- helpers --------------------
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at " << file << ':' << line << " -> "
                  << cudaGetErrorString(err) << " (" << func << ")\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Kernel Error at " << file << ':' << line << " -> "
                  << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

float measure(std::function<void(cudaStream_t)> fn, cudaStream_t s,
             int rpt = 100, int warm = 100)
{
    cudaEvent_t st, ed; float ms = 0.f;
    CHECK_CUDA_ERROR(cudaEventCreate(&st));
    CHECK_CUDA_ERROR(cudaEventCreate(&ed));
    for (int i = 0; i < warm; ++i) fn(s);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s));
    CHECK_CUDA_ERROR(cudaEventRecord(st, s));
    for (int i = 0; i < rpt; ++i) fn(s);
    CHECK_CUDA_ERROR(cudaEventRecord(ed, s));
    CHECK_CUDA_ERROR(cudaEventSynchronize(ed));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, st, ed));
    CHECK_CUDA_ERROR(cudaEventDestroy(st));
    CHECK_CUDA_ERROR(cudaEventDestroy(ed));
    return ms / rpt;
}

__global__ void vec_add(float* o, const float* a, const float* b, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < n; i += step) o[i] = a[i] + b[i];
}

void launch_nonmap(float* h_out, const float* h_a, const float* h_b,
                   float* d_out, float* d_a, float* d_b, uint32_t n,
                   cudaStream_t s)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, n * sizeof(float),
                                     cudaMemcpyHostToDevice, s));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, h_b, n * sizeof(float),
                                     cudaMemcpyHostToDevice, s));
    dim3 blk{1024}, grd{32};
    vec_add<<<grd, blk, 0, s>>>(d_out, d_a, d_b, n);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_out, d_out, n * sizeof(float),
                                     cudaMemcpyDeviceToHost, s));
}

void launch_map(float* d_out, float* d_a, float* d_b, uint32_t n,
                cudaStream_t s)
{
    dim3 blk{1024}, grd{32};
    vec_add<<<grd, blk, 0, s>>>(d_out, d_a, d_b, n);
    CHECK_LAST_CUDA_ERROR();
}

inline void fill(float* p, uint32_t n, float v)
{ for (uint32_t i = 0; i < n; ++i) p[i] = v; }
inline bool check_buf(const float* p, uint32_t n, float v)
{ for (uint32_t i = 0; i < n; ++i) if (p[i] != v) return false; return true; }

int main(int argc, char* argv[])
{
    constexpr int R{10}, W{10};

    /* ------------ CLI ------------ */
    int64_t n64 = 10'000'000; double ratio = 1.0; bool use_ratio = false;
    int dev = 0;                  // GPU id
    enum class Policy { BOTH, MAP, NONMAP } policy = Policy::BOTH;

    for (int i = 1; i < argc; ++i)
    {
        if (!::strcmp(argv[i], "--device") || !::strcmp(argv[i], "-d"))
            dev = std::stoi(argv[++i]);
        else if (!::strcmp(argv[i], "--ratio") || !::strcmp(argv[i], "-r"))
        { ratio = std::stod(argv[++i]); use_ratio = true; }
        else if (!::strcmp(argv[i], "--policy") || !::strcmp(argv[i], "-p"))
        {
            const char* v = argv[++i];
            if (!::strcmp(v, "map"))    policy = Policy::MAP;
            else if (!::strcmp(v, "nonmap")) policy = Policy::NONMAP;
            else throw std::runtime_error{"policy must be map|nonmap"};
        }
        else if (i == 1) n64 = std::stoll(argv[i]); // positional n
    }

    int cnt = 0; CHECK_CUDA_ERROR(cudaGetDeviceCount(&cnt));
    if (dev < 0 || dev >= cnt) throw std::runtime_error{"invalid device"};
    CHECK_CUDA_ERROR(cudaSetDevice(dev));

    if (use_ratio)
    {
        cudaDeviceProp pp{}; CHECK_CUDA_ERROR(cudaGetDeviceProperties(&pp, dev));
        n64 = static_cast<int64_t>(ratio * pp.totalGlobalMem / (3.0 * sizeof(float)));
    }
    uint32_t n = static_cast<uint32_t>(std::min<int64_t>(n64, 0xFFFFFFFFull));

    cudaStream_t stream; CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // allocate memory
    float *h_a, *h_b, *h_out;   // host non‑mapped
    float *d_a, *d_b, *d_out;   // device
    float *a_a, *a_b, *a_out;   // host mapped + device ptrs
    float *m_a, *m_b, *m_out;

    CHECK_CUDA_ERROR(cudaMallocHost(&h_a, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_b, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_out, bytes));

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, bytes));

    CHECK_CUDA_ERROR(cudaHostAlloc(&a_a, bytes, cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&a_b, bytes, cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&a_out, bytes, cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_a, a_a, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_b, a_b, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_out, a_out, 0));

    float v1 = 1.f, v2 = 1.f, v0 = 0.f, v_ref = 2.f;

    auto run_nonmap = [&](){
        fill(h_a, n, v1); fill(h_b, n, v2); fill(h_out, n, v0);
        launch_nonmap(h_out, h_a, h_b, d_out, d_a, d_b, n, stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (!check_buf(h_out, n, v_ref)) throw std::runtime_error{"verify failed"};
        auto fn = std::bind(launch_nonmap, h_out, h_a, h_b,
                            d_out, d_a, d_b, n, std::placeholders::_1);
        float lat = measure(fn, stream, R, W);
        std::cout << std::fixed << std::setprecision(3)
                  << "[GPU" << dev << "] Nonmap latency: " << lat << " ms\n";
    };

    auto run_map = [&](){
        fill(a_a, n, v1); fill(a_b, n, v2); fill(a_out, n, v0);
        launch_map(m_out, m_a, m_b, n, stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (!check_buf(a_out, n, v_ref)) throw std::runtime_error{"verify failed"};
        auto fn = std::bind(launch_map, m_out, m_a, m_b, n, std::placeholders::_1);
        float lat = measure(fn, stream, R, W);
        std::cout << std::fixed << std::setprecision(3)
                  << "[GPU" << dev << "] Map    latency: " << lat << " ms\n";
    };

    if (policy == Policy::BOTH || policy == Policy::NONMAP) run_nonmap();
    if (policy == Policy::BOTH || policy == Policy::MAP)    run_map();

    // free
    CHECK_CUDA_ERROR(cudaFree(d_a)); CHECK_CUDA_ERROR(cudaFree(d_b)); CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaFreeHost(h_a)); CHECK_CUDA_ERROR(cudaFreeHost(h_b)); CHECK_CUDA_ERROR(cudaFreeHost(h_out));
    CHECK_CUDA_ERROR(cudaFreeHost(a_a)); CHECK_CUDA_ERROR(cudaFreeHost(a_b)); CHECK_CUDA_ERROR(cudaFreeHost(a_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return 0;
}
