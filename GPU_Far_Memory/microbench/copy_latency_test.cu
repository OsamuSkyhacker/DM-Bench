/******************************************************************************
 * Latency Test for cudaMemcpy
 *
 * Measures average per-call latency (micro-seconds) for:
 *   - Host→Device  (H2D)
 *   - Device→Host  (D2H)
 *   - Device→Device(D2D, same GPU)
 *
 * Usage (argv, all可混用):
 *   --device N     选择 GPU N
 *   h2d|d2h|d2d    只测某方向，默认全测
 *   pageable       Host 内存用 malloc
 *   pinned         Host 内存用 cudaMallocHost / cudaHostAlloc (默认)
 *   wc             pinned 时加 Write-Combined
 *   --iters 50000  修改每尺寸迭代次数
 *   --csv  <file>  指定 CSV 输出文件名，默认 latency_results.csv
 *
 * 编译:  nvcc -O2 latencyTest.cu -o latencyTest
 *****************************************************************************/
 #include <cuda_runtime.h>
 #include <cstdio>
 #include <cstdlib>
 #include <cstring>
 #include <fstream>
 #include <iostream>
 #include <vector>
 #include <chrono>
 
 #define CUDA_CHECK(call)                                          \
     do {                                                          \
         cudaError_t err = (call);                                 \
         if (err != cudaSuccess) {                                 \
             fprintf(stderr, "[CUDA] %s:%d  %s -> %s\n",           \
                     __FILE__, __LINE__, #call,                    \
                     cudaGetErrorString(err));                     \
             std::exit(EXIT_FAILURE);                              \
         }                                                         \
     } while (0)
 
 //———————————————— 参数与默认值 ————————————————//
 static int    MIN_POWER   = 1;   // 2^1  = 2  B
 static int    MAX_POWER   = 16;  // 2^16 = 64 KB
 static size_t LAT_ITER    = 10000;
 static bool   pinned      = true;
 static bool   wc_flag     = false;
 static bool   doH2D = true, doD2H = true, doD2D = true;
 static int    deviceIdx   = 0;
 static std::string csvName = "latency_results.csv";
 
 //———————————————— 计时工具 ————————————————//
 // 返回毫秒
 static float elapsedMs(cudaEvent_t start, cudaEvent_t stop)
 {
     float ms = 0.f;
     CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
     return ms;
 }
 
 //———————————————— 单向延迟测试模板 ————————————————//
 template <typename CopyFunc>
 static double runLatency(size_t bytes, CopyFunc copier,
                          bool needFlushHost = false)
 {
     // 预创建事件
     cudaEvent_t evStart, evStop;
     CUDA_CHECK(cudaEventCreate(&evStart));
     CUDA_CHECK(cudaEventCreate(&evStop));
 
     // 预热一次
     copier();  CUDA_CHECK(cudaDeviceSynchronize());
 
     // 真正计时
     CUDA_CHECK(cudaEventRecord(evStart));
     for (size_t i = 0; i < LAT_ITER; ++i) {
         copier();
         if (needFlushHost) {
             // 轻量刷新 L1/L2：写一个 dummy 地址
             asm volatile("" ::: "memory");
         }
     }
     CUDA_CHECK(cudaEventRecord(evStop));
     CUDA_CHECK(cudaEventSynchronize(evStop));
 
     double totalUs = elapsedMs(evStart, evStop) * 1000.0;  // ms→us
     CUDA_CHECK(cudaEventDestroy(evStart));
     CUDA_CHECK(cudaEventDestroy(evStop));
 
     return totalUs / static_cast<double>(LAT_ITER);        // 平均单次
 }
 
 //———————————————— 具体方向实现 ————————————————//
 static double latencyH2D(size_t bytes)
 {
     // host buffer
     unsigned char* hBuf = nullptr;
     if (pinned) {
 #if CUDART_VERSION >= 12000
         unsigned int flags = wc_flag ? cudaHostAllocWriteCombined : 0;
         CUDA_CHECK(cudaHostAlloc(&hBuf, bytes, flags));
 #else
         CUDA_CHECK(cudaMallocHost(&hBuf, bytes));
 #endif
     } else {
         hBuf = (unsigned char*)std::malloc(bytes);
         if (!hBuf) { std::exit(EXIT_FAILURE); }
     }
     // device buffer
     unsigned char* dBuf = nullptr;
     CUDA_CHECK(cudaMalloc(&dBuf, bytes));
 
     auto copier = [&]() {
         if (pinned)
             CUDA_CHECK(cudaMemcpyAsync(dBuf, hBuf, bytes,
                            cudaMemcpyHostToDevice));
         else
             CUDA_CHECK(cudaMemcpy(dBuf, hBuf, bytes,
                            cudaMemcpyHostToDevice));
     };
     double us = runLatency(bytes, copier, /*flush*/ !pinned);
 
     if (pinned) CUDA_CHECK(cudaFreeHost(hBuf)); else std::free(hBuf);
     CUDA_CHECK(cudaFree(dBuf));
     return us;
 }
 
 static double latencyD2H(size_t bytes)
 {
     unsigned char* hBuf = nullptr;
     if (pinned) {
 #if CUDART_VERSION >= 12000
         unsigned int flags = wc_flag ? cudaHostAllocWriteCombined : 0;
         CUDA_CHECK(cudaHostAlloc(&hBuf, bytes, flags));
 #else
         CUDA_CHECK(cudaMallocHost(&hBuf, bytes));
 #endif
     } else {
         hBuf = (unsigned char*)std::malloc(bytes);
         if (!hBuf) { std::exit(EXIT_FAILURE); }
     }
     unsigned char* dBuf = nullptr;
     CUDA_CHECK(cudaMalloc(&dBuf, bytes));
     // init device data
     CUDA_CHECK(cudaMemset(dBuf, 1, bytes));
 
     auto copier = [&]() {
         if (pinned)
             CUDA_CHECK(cudaMemcpyAsync(hBuf, dBuf, bytes,
                            cudaMemcpyDeviceToHost));
         else
             CUDA_CHECK(cudaMemcpy(hBuf, dBuf, bytes,
                            cudaMemcpyDeviceToHost));
     };
     double us = runLatency(bytes, copier, /*flush*/ !pinned);
 
     if (pinned) CUDA_CHECK(cudaFreeHost(hBuf)); else std::free(hBuf);
     CUDA_CHECK(cudaFree(dBuf));
     return us;
 }
 
 static double latencyD2D(size_t bytes)
 {
     unsigned char *dSrc = nullptr, *dDst = nullptr;
     CUDA_CHECK(cudaMalloc(&dSrc, bytes));
     CUDA_CHECK(cudaMalloc(&dDst, bytes));
     CUDA_CHECK(cudaMemset(dSrc, 2, bytes));
 
     auto copier = [&]() {
         CUDA_CHECK(cudaMemcpy(dDst, dSrc, bytes,
                               cudaMemcpyDeviceToDevice));
     };
     double us = runLatency(bytes, copier);
 
     CUDA_CHECK(cudaFree(dSrc));
     CUDA_CHECK(cudaFree(dDst));
     return us;
 }
 
 //———————————————— CLI 解析 ————————————————//
 static void parseArgs(int argc, char** argv)
 {
     for (int i = 1; i < argc; ++i) {
         if (!strcmp(argv[i], "--device") && i + 1 < argc) {
             deviceIdx = std::atoi(argv[++i]);
         } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
             LAT_ITER = std::strtoull(argv[++i], nullptr, 10);
         } else if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
             csvName = argv[++i];
         } else if (!strcmp(argv[i], "h2d")) { doD2H = doD2D = false; }
         else if (!strcmp(argv[i], "d2h"))   { doH2D = doD2D = false; }
         else if (!strcmp(argv[i], "d2d"))   { doH2D = doD2H = false; }
         else if (!strcmp(argv[i], "pageable")) { pinned = false; }
         else if (!strcmp(argv[i], "pinned"))   { pinned = true;  }
         else if (!strcmp(argv[i], "wc"))       { wc_flag = true; }
         else if (!strcmp(argv[i], "--range") && i + 2 < argc) {
             MIN_POWER = std::atoi(argv[++i]);
             MAX_POWER = std::atoi(argv[++i]);
         } else {
             fprintf(stderr, "Unknown arg: %s\n", argv[i]);
             std::exit(EXIT_FAILURE);
         }
     }
 }
 
 //———————————————— 主函数 ————————————————//
 int main(int argc, char** argv)
 {
     parseArgs(argc, argv);
     CUDA_CHECK(cudaSetDevice(deviceIdx));
 
     std::ofstream csv(csvName, std::ios::trunc);
     if (!csv) { std::cerr << "CSV open fail\n"; return -1; }
     csv << "Direction,DataSize(Bytes),AvgLatency(us)\n";
 
     for (int p = MIN_POWER; p <= MAX_POWER; ++p) {
         size_t sz = size_t(1) << p;
         if (doH2D) {
             double us = latencyH2D(sz);
             csv << "H2D," << sz << ',' << us << '\n';
             std::cout << "[H2D] " << sz << " B  " << us << " us\n";
         }
         if (doD2H) {
             double us = latencyD2H(sz);
             csv << "D2H," << sz << ',' << us << '\n';
             std::cout << "[D2H] " << sz << " B  " << us << " us\n";
         }
         if (doD2D) {
             double us = latencyD2D(sz);
             csv << "D2D," << sz << ',' << us << '\n';
             std::cout << "[D2D] " << sz << " B  " << us << " us\n";
         }
     }
     csv.close();
     std::cout << "==> Done. Results saved to " << csvName << '\n';
     return 0;
 }
 

 
