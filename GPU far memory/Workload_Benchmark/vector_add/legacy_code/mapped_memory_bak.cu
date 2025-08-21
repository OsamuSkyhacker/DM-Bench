#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

__global__ void float_addition(float* output, float const* input_1,
                               float const* input_2, uint32_t n)
{
    const uint32_t idx{blockDim.x * blockIdx.x + threadIdx.x};
    const uint32_t stride{blockDim.x * gridDim.x};
    for (uint32_t i{idx}; i < n; i += stride)
    {
        output[i] = input_1[i] + input_2[i];
    }
}

void launch_float_addition_non_mapped_pinned_memory(
    float* h_output, float const* h_input_1, float const* h_input_2,
    float* d_output, float* d_input_1, float* d_input_2, uint32_t n,
    cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input_1, h_input_1, n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input_2, h_input_2, n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    float_addition<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_output, d_input_1, d_input_2, n);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
}

void launch_float_addition_mapped_pinned_memory(float* d_output,
                                                float* d_input_1,
                                                float* d_input_2, uint32_t n,
                                                cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    float_addition<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_output, d_input_1, d_input_2, n);
    CHECK_LAST_CUDA_ERROR();
}

void initialize_host_memory(float* h_buffer, uint32_t n, float value)
{
    for (int i{0}; i < n; ++i)
    {
        h_buffer[i] = value;
    }
}

bool verify_host_memory(float* h_buffer, uint32_t n, float value)
{
    for (int i{0}; i < n; ++i)
    {
        if (h_buffer[i] != value)
        {
            return false;
        }
    }
    return true;
}

// -------------------------- 新 main -----------------------------
int main(int argc, char* argv[])
{
    constexpr int kNumRepeats{10};
    constexpr int kNumWarmups{10};

    // ------------------ CLI 解析 ------------------
    int64_t n64       = 10'000'000;   // 默认元素数
    bool    use_ratio = false;
    double  ratio     = 1.0;
    int     device_id = 0;            // 默认 GPU 0

    // 先扫描 --device/-d
    for (int i = 1; i < argc - 1; ++i)
    {
        if (::strcmp(argv[i], "--device") == 0 || ::strcmp(argv[i], "-d") == 0)
        {
            device_id = std::stoi(argv[i + 1]);
        }
    }

    // 设置设备并校验
    int device_count = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error{"无效的 device_id"};
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    // 解析 n 或 ratio
    if (argc >= 2 && ::strcmp(argv[1], "--ratio") != 0 && ::strcmp(argv[1], "-r") != 0 &&
        ::strcmp(argv[1], "--device") != 0 && ::strcmp(argv[1], "-d") != 0)
    {
        n64 = std::stoll(argv[1]);
    }
    for (int i = 1; i < argc - 1; ++i)
    {
        if (::strcmp(argv[i], "--ratio") == 0 || ::strcmp(argv[i], "-r") == 0)
        {
            ratio     = std::stod(argv[i + 1]);
            use_ratio = true;
            break;
        }
    }

    // ----------------- ratio 模式 ----------------
    if (use_ratio)
    {
        cudaDeviceProp prop{};
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
        size_t const total_bytes = prop.totalGlobalMem;
        size_t const per_elem    = 3ull * sizeof(float); // 3 arrays
        n64 = static_cast<int64_t>(ratio * static_cast<double>(total_bytes) /
                                   static_cast<double>(per_elem));
        if (n64 <= 0)
            throw std::runtime_error{"ratio 导致 n <= 0"};
        std::cout << "[auto-ratio] GPU" << device_id << " totalGlobalMem="
                  << total_bytes / (1024.0 * 1024.0) << " MiB, ratio="
                  << ratio << " -> n=" << n64 << '\n';
    }
    else
    {
        std::cout << "[explicit] n=" << n64 << " (GPU" << device_id << ")\n";
    }

    if (n64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error{"n 超过 uint32_t 范围"};
    uint32_t const n = static_cast<uint32_t>(n64);


    /* -------------------- CUDA 初始化 -------------------------- */
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    float const v_input_1{1.0f}, v_input_2{1.0f};
    float const v_output{0.0f};
    float const v_output_ref{v_input_1 + v_input_2};

    cudaDeviceProp prop{};
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory)
        throw std::runtime_error{"Device does not support mapped memory."};

    /* ---------------------- 内存分配 --------------------------- */
    float *h_input_1, *h_input_2, *h_output;
    float *d_input_1, *d_input_2, *d_output;
    float *a_input_1, *a_input_2, *a_output;
    float *m_input_1, *m_input_2, *m_output;

    size_t const bytes = static_cast<size_t>(n) * sizeof(float);

    CHECK_CUDA_ERROR(cudaMallocHost(&h_input_1, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input_2, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, bytes));

    CHECK_CUDA_ERROR(cudaMalloc(&d_input_1, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_2, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output , bytes));

    CHECK_CUDA_ERROR(cudaHostAlloc(&a_input_1, bytes, cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&a_input_2, bytes, cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&a_output , bytes, cudaHostAllocMapped));

    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_1, a_input_1, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_2, a_input_2, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_output , a_output , 0));

    /* -------------------- 正确性验证 --------------------------- */
    initialize_host_memory(h_input_1, n, v_input_1);
    initialize_host_memory(h_input_2, n, v_input_2);
    initialize_host_memory(h_output , n, v_output );
    launch_float_addition_non_mapped_pinned_memory(h_output, h_input_1, h_input_2,
                                                   d_output, d_input_1, d_input_2,
                                                   n, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    assert(verify_host_memory(h_output, n, v_output_ref));

    initialize_host_memory(a_input_1, n, v_input_1);
    initialize_host_memory(a_input_2, n, v_input_2);
    initialize_host_memory(a_output , n, v_output );
    launch_float_addition_mapped_pinned_memory(m_output, m_input_1, m_input_2,
                                               n, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    assert(verify_host_memory(a_output, n, v_output_ref));

    /* -------------------- 性能测量 ----------------------------- */
    std::function<void(cudaStream_t)> fn_non_mapped = std::bind(
        launch_float_addition_non_mapped_pinned_memory, h_output, h_input_1,
        h_input_2, d_output, d_input_1, d_input_2, n, std::placeholders::_1);

    std::function<void(cudaStream_t)> fn_mapped = std::bind(
        launch_float_addition_mapped_pinned_memory, m_output, m_input_1,
        m_input_2, n, std::placeholders::_1);

    float lat_non_mapped = measure_performance(fn_non_mapped, stream,
                                               kNumRepeats, kNumWarmups);
    float lat_mapped     = measure_performance(fn_mapped    , stream,
                                               kNumRepeats, kNumWarmups);

    std::cout << std::fixed << std::setprecision(3)
              << "CUDA Kernel With Non-Mapped Pinned Memory Latency: "
              << lat_non_mapped << " ms\n"
              << "CUDA Kernel With Mapped Pinned Memory Latency:     "
              << lat_mapped     << " ms\n";

    // ---------------------- 资源释放 ---------------------------
    CHECK_CUDA_ERROR(cudaFree(d_input_1));
    CHECK_CUDA_ERROR(cudaFree(d_input_2));
    CHECK_CUDA_ERROR(cudaFree(d_output ));
    CHECK_CUDA_ERROR(cudaFreeHost(h_input_1));
    CHECK_CUDA_ERROR(cudaFreeHost(h_input_2));
    CHECK_CUDA_ERROR(cudaFreeHost(h_output ));
    CHECK_CUDA_ERROR(cudaFreeHost(a_input_1));
    CHECK_CUDA_ERROR(cudaFreeHost(a_input_2));
    CHECK_CUDA_ERROR(cudaFreeHost(a_output ));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}


// int main()
// {
//     constexpr int const num_repeats{10};
//     constexpr int const num_warmups{10};

//     constexpr int const n{10000000}; 
//     cudaStream_t stream;
//     CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

//     float const v_input_1{1.0f};
//     float const v_input_2{1.0f};
//     float const v_output{0.0f};
//     float const v_output_reference{v_input_1 + v_input_2};

//     cudaDeviceProp prop;
//     CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
//     if (!prop.canMapHostMemory)
//     {
//         throw std::runtime_error{"Device does not supported mapped memory."};
//     }

//     float *h_input_1, *h_input_2, *h_output;
//     float *d_input_1, *d_input_2, *d_output;

//     float *a_input_1, *a_input_2, *a_output;
//     float *m_input_1, *m_input_2, *m_output;

//     CHECK_CUDA_ERROR(cudaMallocHost(&h_input_1, n * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMallocHost(&h_input_2, n * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMallocHost(&h_output, n * sizeof(float)));

//     CHECK_CUDA_ERROR(cudaMalloc(&d_input_1, n * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_input_2, n * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

//     CHECK_CUDA_ERROR(
//         cudaHostAlloc(&a_input_1, n * sizeof(float), cudaHostAllocMapped));
//     CHECK_CUDA_ERROR(
//         cudaHostAlloc(&a_input_2, n * sizeof(float), cudaHostAllocMapped));
//     CHECK_CUDA_ERROR(
//         cudaHostAlloc(&a_output, n * sizeof(float), cudaHostAllocMapped));

//     CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_1, a_input_1, 0));
//     CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_2, a_input_2, 0));
//     CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_output, a_output, 0));

//     // Verify the implementation correctness.
//     initialize_host_memory(h_input_1, n, v_input_1);
//     initialize_host_memory(h_input_2, n, v_input_2);
//     initialize_host_memory(h_output, n, v_output);
//     launch_float_addition_non_mapped_pinned_memory(
//         h_output, h_input_1, h_input_2, d_output, d_input_1, d_input_2, n,
//         stream);
//     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
//     assert(verify_host_memory(h_output, n, v_output_reference));

//     initialize_host_memory(a_input_1, n, v_input_1);
//     initialize_host_memory(a_input_2, n, v_input_2);
//     initialize_host_memory(a_output, n, v_output);
//     launch_float_addition_mapped_pinned_memory(m_output, m_input_1, m_input_2,
//                                                n, stream);
//     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
//     assert(verify_host_memory(a_output, n, v_output_reference));

//     // Measure latencies.
//     std::function<void(cudaStream_t)> function_non_mapped_pinned_memory{
//         std::bind(launch_float_addition_non_mapped_pinned_memory, h_output,
//                   h_input_1, h_input_2, d_output, d_input_1, d_input_2, n,
//                   std::placeholders::_1)};
//     std::function<void(cudaStream_t)> function_mapped_pinned_memory{
//         std::bind(launch_float_addition_mapped_pinned_memory, m_output,
//                   m_input_1, m_input_2, n, std::placeholders::_1)};
//     float const latency_non_mapped_pinned_memory{measure_performance(
//         function_non_mapped_pinned_memory, stream, num_repeats, num_warmups)};
//     float const latency_mapped_pinned_memory{measure_performance(
//         function_mapped_pinned_memory, stream, num_repeats, num_warmups)};
//     std::cout << std::fixed << std::setprecision(3)
//               << "CUDA Kernel With Non-Mapped Pinned Memory Latency: "
//               << latency_non_mapped_pinned_memory << " ms" << std::endl;
//     std::cout << std::fixed << std::setprecision(3)
//               << "CUDA Kernel With Mapped Pinned Memory Latency: "
//               << latency_mapped_pinned_memory << " ms" << std::endl;

//     CHECK_CUDA_ERROR(cudaFree(d_input_1));
//     CHECK_CUDA_ERROR(cudaFree(d_input_2));
//     CHECK_CUDA_ERROR(cudaFree(d_output));
//     CHECK_CUDA_ERROR(cudaFreeHost(h_input_1));
//     CHECK_CUDA_ERROR(cudaFreeHost(h_input_2));
//     CHECK_CUDA_ERROR(cudaFreeHost(h_output));
//     CHECK_CUDA_ERROR(cudaFreeHost(a_input_1));
//     CHECK_CUDA_ERROR(cudaFreeHost(a_input_2));
//     CHECK_CUDA_ERROR(cudaFreeHost(a_output));
//     CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

//     return 0;
// }