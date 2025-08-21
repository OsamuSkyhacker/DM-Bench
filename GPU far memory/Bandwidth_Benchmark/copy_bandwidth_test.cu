/******************************************************************************
 * Simplified Bandwidth Test
 * 
 * References NVIDIA's bandwidthTest sample logic to measure:
 *   - Host->Device (H2D)
 *   - Device->Host (D2H)
 *   - Device->Device (D2D)
 * with either Pinned or Pageable (malloc) host memory.
 *
 * Generates CSV output for further analysis.
 *
 *****************************************************************************/
 #include <cuda_runtime.h>
 #include <cstdio>
 #include <cstdlib>
 #include <cstring>
 #include <iostream>
 #include <fstream>
 
 // 为了使用高精度计时接口 (StopWatchInterface等)，可参考 helper_functions.h / helper_cuda.h
 // 但这里我们用最简单的 CPU 端计时 + CUDA Event 结合。
 
 //----------------------------- 宏和全局变量 ---------------------------//
 static const int    MEMCOPY_ITERATIONS = 100;
 static const size_t FLUSH_SIZE         = 1024 * 1024 * 1024; // 1GB CPU缓存刷写
 static const bool   DEFAULT_WC         = false; // 是否使用写结合 pinned
 
 // 测试的数据范围: 1KB ~ 1GB (可根据需要修改)
 static const int    MIN_POWER = 10;  // 2^10 = 1KB
 static const int    MAX_POWER = 30;  // 2^30 = 1GB
 
 // 如果页面锁定，可以用写结合 (wc) 的方式来提升写性能 (需要较新CUDA)
 #ifndef CUDART_VERSION
 #error CUDART_VERSION Undefined!
 #endif
 
 // 全局缓存flush数组
 char* flush_buf = nullptr;
 
 //----------------------------- 辅助函数 ---------------------------//
 static void checkCudaError(cudaError_t err, const char* msg)
 {
     if (err != cudaSuccess) {
         std::cerr << "[Error] " << msg << ": "
                   << cudaGetErrorString(err) << std::endl;
         std::exit(EXIT_FAILURE);
     }
 }
 
 // 刷新 CPU 缓存 (在 pageable 模式下，每次传输后可以调用)
 static void flushCPUCache(size_t iter)
 {
     if (!flush_buf) return; // 如果尚未分配缓存区，直接返回
     // 简单 memset，模拟把不同数据写入 flush_buf，从而逼迫 CPU cache 失效
     std::memset(flush_buf, (int)iter, FLUSH_SIZE);
 }
 
 //----------------------------- 核心带宽测试函数 ---------------------------//
 /**
  * @brief Host -> Device (H2D) 带宽测试
  * 
  * @param dataSize     测试数据大小 (字节)
  * @param pinned       是否使用 pinned 内存
  * @param wc           是否使用写结合 pinned (仅当 pinned=true 时有效)
  * @return 计算得到的带宽 (GB/s)
  */
 static float measureHtoDBandwidth(size_t dataSize, bool pinned, bool wc)
 {
     // 分配 host 内存
     unsigned char* h_data = nullptr;
     if (pinned) {
 #if (CUDART_VERSION >= 2020)
         unsigned int flags = wc ? cudaHostAllocWriteCombined : 0;
         checkCudaError(cudaHostAlloc((void**)&h_data, dataSize, flags),
                        "cudaHostAlloc(H2D)");
 #else
         checkCudaError(cudaMallocHost((void**)&h_data, dataSize),
                        "cudaMallocHost(H2D)");
 #endif
     } else {
         h_data = (unsigned char*)std::malloc(dataSize);
         if (!h_data) {
             std::cerr << "Not enough host memory for H2D test!\n";
             std::exit(EXIT_FAILURE);
         }
     }
 
     // 初始化 host 数据
     for (size_t i = 0; i < dataSize; i++) {
         h_data[i] = (unsigned char)(i & 0xFF);
     }
 
     // 分配 device 内存
     unsigned char* d_data = nullptr;
     checkCudaError(cudaMalloc((void**)&d_data, dataSize), "cudaMalloc(H2D)");
 
     // 为计时创建 CUDA Event
     cudaEvent_t start, stop;
     checkCudaError(cudaEventCreate(&start), "cudaEventCreate(H2D start)");
     checkCudaError(cudaEventCreate(&stop),  "cudaEventCreate(H2D stop)");
 
     float totalMs = 0.0f;
 
     // 如果是 pinned，使用 GPU 事件 + 异步 memcpy 测量 GPU 端时间
     // 否则使用 CPU 端简单累计
     if (pinned) {
         // 记录开始
         checkCudaError(cudaEventRecord(start), "cudaEventRecord(H2D start)");
 
         // 执行多次拷贝
         for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
             // 异步拷贝 Host->Device
             checkCudaError(cudaMemcpyAsync(d_data, h_data, dataSize,
                                            cudaMemcpyHostToDevice),
                            "cudaMemcpyAsync(H2D pinned)");
         }
 
         checkCudaError(cudaEventRecord(stop), "cudaEventRecord(H2D stop)");
         checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize(H2D)");
 
         checkCudaError(cudaEventElapsedTime(&totalMs, start, stop),
                        "cudaEventElapsedTime(H2D pinned)");
     } else {
         // pageable 模式: 简单用 CPU 端计时即可
         // 这里也可以用 CUDA event 结合 sync，但一般 pageable 传输常被CPU端计时
         for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
             float singleMs = 0.0f;
             float startMs  = 0.0f;
 
             // 记录start
             checkCudaError(cudaEventRecord(start), "cudaEventRecord(H2D start)");
             checkCudaError(cudaEventSynchronize(start), "cudaEventSync(H2D start)");
 
             // 拷贝
             checkCudaError(cudaMemcpy(d_data, h_data, dataSize,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(H2D pageable)");
 
             // 记录stop
             checkCudaError(cudaEventRecord(stop), "cudaEventRecord(H2D stop)");
             checkCudaError(cudaEventSynchronize(stop), "cudaEventSync(H2D stop)");
 
             checkCudaError(cudaEventElapsedTime(&singleMs, start, stop),
                            "cudaEventElapsedTime(H2D pageable)");
 
             totalMs += singleMs;
 
             // 刷新CPU cache，模拟更真实场景
             flushCPUCache(i);
         }
     }
 
     // 计算带宽(GB/s): (dataSize * MEMCOPY_ITERATIONS) / totalTime
     // dataSize单位: 字节; totalTime单位: ms => 转化为s => ×(1e-3)
     float totalSec = totalMs * 1e-3f;
     float totalBytes = (float)dataSize * (float)MEMCOPY_ITERATIONS;
     float bandwidthGBs = totalBytes / (1e9f * totalSec);
 
     // 释放资源
     checkCudaError(cudaEventDestroy(stop),  "cudaEventDestroy(H2D stop)");
     checkCudaError(cudaEventDestroy(start), "cudaEventDestroy(H2D start)");
 
     if (pinned) {
 #if (CUDART_VERSION >= 2020)
         checkCudaError(cudaFreeHost(h_data), "cudaFreeHost(H2D pinned)");
 #else
         checkCudaError(cudaFreeHost(h_data), "cudaFreeHost(H2D pinned legacy)");
 #endif
     } else {
         std::free(h_data);
     }
 
     checkCudaError(cudaFree(d_data), "cudaFree(H2D)");
 
     return bandwidthGBs;
 }
 
 /**
  * @brief Device -> Host (D2H) 带宽测试
  * 
  * @param dataSize     测试数据大小 (字节)
  * @param pinned       是否使用 pinned 内存
  * @param wc           是否使用写结合 pinned (仅当 pinned=true 时有效)
  * @return 计算得到的带宽 (GB/s)
  */
 static float measureDtoHBandwidth(size_t dataSize, bool pinned, bool wc)
 {
     // 分配 host 内存
     unsigned char* h_data = nullptr;
     if (pinned) {
 #if (CUDART_VERSION >= 2020)
         unsigned int flags = wc ? cudaHostAllocWriteCombined : 0;
         checkCudaError(cudaHostAlloc((void**)&h_data, dataSize, flags),
                        "cudaHostAlloc(D2H)");
 #else
         checkCudaError(cudaMallocHost((void**)&h_data, dataSize),
                        "cudaMallocHost(D2H)");
 #endif
     } else {
         h_data = (unsigned char*)std::malloc(dataSize);
         if (!h_data) {
             std::cerr << "Not enough host memory for D2H test!\n";
             std::exit(EXIT_FAILURE);
         }
     }
 
     // 分配 device 内存 & 初始化
     unsigned char* d_data = nullptr;
     checkCudaError(cudaMalloc((void**)&d_data, dataSize), "cudaMalloc(D2H)");
     // 给 device 写一些数据
     // std::unique_ptr<unsigned char[]> temp(new unsigned char[dataSize]);
     unsigned char* temp = new unsigned char[dataSize];
     for (size_t i = 0; i < dataSize; i++) {
         temp[i] = (unsigned char)(i & 0xFF);
     }
     checkCudaError(cudaMemcpy(d_data, temp, dataSize,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy init(D2H)");
     delete[] temp;
     // 计时
     cudaEvent_t start, stop;
     checkCudaError(cudaEventCreate(&start), "cudaEventCreate(D2H start)");
     checkCudaError(cudaEventCreate(&stop),  "cudaEventCreate(D2H stop)");
 
     float totalMs = 0.0f;
 
     if (pinned) {
         checkCudaError(cudaEventRecord(start), "cudaEventRecord(D2H start)");
         for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
             checkCudaError(cudaMemcpyAsync(h_data, d_data, dataSize,
                                            cudaMemcpyDeviceToHost),
                            "cudaMemcpyAsync(D2H pinned)");
         }
         checkCudaError(cudaEventRecord(stop), "cudaEventRecord(D2H stop)");
         checkCudaError(cudaEventSynchronize(stop), "cudaEventSync(D2H)");
         checkCudaError(cudaEventElapsedTime(&totalMs, start, stop),
                        "cudaEventElapsedTime(D2H pinned)");
     } else {
         for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
             float singleMs = 0.0f;
             checkCudaError(cudaEventRecord(start), "cudaEventRecord(D2H start)");
             checkCudaError(cudaEventSynchronize(start), "cudaEventSync(D2H start)");
 
             checkCudaError(cudaMemcpy(h_data, d_data, dataSize,
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy(D2H pageable)");
 
             checkCudaError(cudaEventRecord(stop), "cudaEventRecord(D2H stop)");
             checkCudaError(cudaEventSynchronize(stop), "cudaEventSync(D2H stop)");
 
             checkCudaError(cudaEventElapsedTime(&singleMs, start, stop),
                            "cudaEventElapsedTime(D2H pageable)");
             totalMs += singleMs;
             flushCPUCache(i);
         }
     }
 
     float totalSec = totalMs * 1e-3f;
     float totalBytes = (float)dataSize * (float)MEMCOPY_ITERATIONS;
     float bandwidthGBs = totalBytes / (1e9f * totalSec);
 
     checkCudaError(cudaEventDestroy(stop),  "cudaEventDestroy(D2H stop)");
     checkCudaError(cudaEventDestroy(start), "cudaEventDestroy(D2H start)");
 
     if (pinned) {
 #if (CUDART_VERSION >= 2020)
         checkCudaError(cudaFreeHost(h_data), "cudaFreeHost(D2H pinned)");
 #else
         checkCudaError(cudaFreeHost(h_data), "cudaFreeHost(D2H pinned legacy)");
 #endif
     } else {
         std::free(h_data);
     }
 
     checkCudaError(cudaFree(d_data), "cudaFree(D2H)");
     return bandwidthGBs;
 }
 
 /**
  * @brief Device -> Device (D2D) 带宽测试
  * 
  * @param dataSize     测试数据大小 (字节)
  * @return 计算得到的带宽 (GB/s)
  */
 static float measureDtoDBandwidth(size_t dataSize)
 {
     // 分配 host 内存用于初始化
     unsigned char* h_data = (unsigned char*)std::malloc(dataSize);
     if (!h_data) {
         std::cerr << "Not enough host memory for D2D test!\n";
         std::exit(EXIT_FAILURE);
     }
     for (size_t i = 0; i < dataSize; i++) {
         h_data[i] = (unsigned char)(i & 0xFF);
     }
 
     // 分配 device 内存
     unsigned char *d_src = nullptr, *d_dst = nullptr;
     checkCudaError(cudaMalloc((void**)&d_src, dataSize), "cudaMalloc(D2D src)");
     checkCudaError(cudaMalloc((void**)&d_dst, dataSize), "cudaMalloc(D2D dst)");
 
     // 初始化 d_src
     checkCudaError(cudaMemcpy(d_src, h_data, dataSize, cudaMemcpyHostToDevice),
                    "cudaMemcpy init(D2D)");
 
     cudaEvent_t start, stop;
     checkCudaError(cudaEventCreate(&start), "cudaEventCreate(D2D start)");
     checkCudaError(cudaEventCreate(&stop),  "cudaEventCreate(D2D stop)");
 
     // 计时
     float totalMs = 0.0f;
     checkCudaError(cudaEventRecord(start), "cudaEventRecord(D2D start)");
     for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
         checkCudaError(cudaMemcpy(d_dst, d_src, dataSize, cudaMemcpyDeviceToDevice),
                        "cudaMemcpy(D2D)");
     }
     checkCudaError(cudaEventRecord(stop), "cudaEventRecord(D2D stop)");
     checkCudaError(cudaEventSynchronize(stop), "cudaEventSync(D2D)");
     checkCudaError(cudaEventElapsedTime(&totalMs, start, stop),
                    "cudaEventElapsedTime(D2D)");
 
     float totalSec = totalMs * 1e-3f;
     // D2D 一次拷贝传输 dataSize 字节，但 src->dst 只是一份传输，所以带宽计算单向即可
     float totalBytes = (float)dataSize * (float)MEMCOPY_ITERATIONS;
     float bandwidthGBs = totalBytes / (1e9f * totalSec);
 
     // 释放
     std::free(h_data);
     checkCudaError(cudaFree(d_src), "cudaFree(D2D src)");
     checkCudaError(cudaFree(d_dst), "cudaFree(D2D dst)");
     checkCudaError(cudaEventDestroy(stop),  "cudaEventDestroy(D2D stop)");
     checkCudaError(cudaEventDestroy(start), "cudaEventDestroy(D2D start)");
 
     return bandwidthGBs;
 }
 
 //----------------------------- 主函数: 测试并输出 CSV ---------------------------//
 int main(int argc, char* argv[])
 {
     // 默认选择第一个 GPU 设备
     int deviceIndex = 0;
 
     // 1. 简单解析：选择测试方向
     bool doH2D = true;
     bool doD2H = true;
     bool doD2D = true;
     bool pinned = true;      // 默认 pinned
     bool wc     = DEFAULT_WC; // 默认不启用写结合
 
     // 2. 解析命令行参数，选择 GPU 设备
     for (int i = 1; i < argc; i++) {
         if (!strcmp(argv[i], "--device")) {
             // 如果传入 --device 参数，选择指定的 GPU 设备
             if (i + 1 < argc) {
                 deviceIndex = atoi(argv[i + 1]);  // 获取指定的 GPU 索引
                 i++;  // 跳过下一个参数，因为它是设备索引
             } else {
                 std::cerr << "Error: --device requires a device index as argument.\n";
                 return -1;
             }
         } else if (!strcmp(argv[i], "h2d")) {
             doD2H = false; doD2D = false; // 仅测 h2d
         } else if (!strcmp(argv[i], "d2h")) {
             doH2D = false; doD2D = false; // 仅测 d2h
         } else if (!strcmp(argv[i], "d2d")) {
             doH2D = false; doD2H = false; // 仅测 d2d
         } else if (!strcmp(argv[i], "pageable")) {
             pinned = false;
         } else if (!strcmp(argv[i], "pinned")) {
             pinned = true;
         } else if (!strcmp(argv[i], "wc")) {
             wc = true;
         }
     }
 
     // 选择 GPU 设备
     checkCudaError(cudaSetDevice(deviceIndex), "cudaSetDevice()");
 
 
     // 分配 CPU flush_buf
     flush_buf = (char*)malloc(FLUSH_SIZE);
     if (!flush_buf) {
         std::cerr << "Not enough memory for flush_buf.\n";
         std::exit(EXIT_FAILURE);
     }
 
     // 3. 打开 CSV 文件输出
     std::ofstream outFile("bandwidth_results.csv", std::ios::trunc);
     if (!outFile.is_open()) {
         std::cerr << "Failed to open CSV file.\n";
         return -1;
     }
     // 写 CSV 标题行
     outFile << "TestDirection,DataSize(KB),Bandwidth(GB/s)\n";
 
     // 4. 测试数据范围 1KB ~ 256MB，进行带宽测试
     for (int power = MIN_POWER; power <= MAX_POWER; power++) {
         size_t dataSize = (size_t)1 << power; // 2^power
         // Host->Device
         if (doH2D) {
             float bw = measureHtoDBandwidth(dataSize, pinned, wc);
             outFile << "H2D," << (dataSize >> 10) << "," << bw << "\n";
             std::cout << "[H2D] " << (dataSize >> 10) << " KB: " << bw << " GB/s\n";
         }
         // Device->Host
         if (doD2H) {
             float bw = measureDtoHBandwidth(dataSize, pinned, wc);
             outFile << "D2H," << (dataSize >> 10) << "," << bw << "\n";
             std::cout << "[D2H] " << (dataSize >> 10) << " KB: " << bw << " GB/s\n";
         }
         // Device->Device
         if (doD2D) {
             float bw = measureDtoDBandwidth(dataSize);
             outFile << "D2D," << (dataSize >> 10) << "," << bw << "\n";
             std::cout << "[D2D] " << (dataSize >> 10) << " KB: " << bw << " GB/s\n";
         }
     }
 
     outFile.close();
     // 释放 flush_buf
     free(flush_buf);
 
     std::cout << "==> Done. Results saved to bandwidth_results.csv\n";
     return 0;
 }
