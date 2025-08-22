/********************************************************************
 *  vector_add_um.cu  ——  Unified-Memory 版向量加基准
 *  nvcc -O3 -std=c++14 -arch=sm_80 vector_add_um.cu -o vector_add_um
 ********************************************************************/
 #include <cuda_runtime.h>
 #include <cassert>
 #include <cstring>
 #include <functional>
 #include <iomanip>
 #include <iostream>
 #include <map>
 #include <stdexcept>
 #include <string>
 #include <vector>
 
 #define CHECK(cudaStmt) do {                                        \
     cudaError_t _e = cudaStmt;                                      \
     if (_e != cudaSuccess) {                                        \
         std::cerr << "CUDA Error: " << cudaGetErrorString(_e)       \
                   << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
         std::exit(EXIT_FAILURE);                                    \
     }                                                               \
 } while (0)
 
 /* ----------------------- kernel ----------------------- */
 __global__ void vec_add(float* o, const float* a, const float* b, uint32_t n) {
     uint32_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
     uint32_t step = blockDim.x * gridDim.x;
     for (uint32_t i = idx; i < n; i += step) o[i] = a[i] + b[i];
 }
 
 /* ----------------------- helpers ---------------------- */
 float time_ms(std::function<void(cudaStream_t)> fn,
               cudaStream_t s, int repeat = 100, int warmup = 100) {
     cudaEvent_t st, ed; float ms = 0.f;
     CHECK(cudaEventCreate(&st)); CHECK(cudaEventCreate(&ed));
     for (int i = 0; i < warmup; ++i) fn(s);
     CHECK(cudaStreamSynchronize(s));
     CHECK(cudaEventRecord(st, s));
     for (int i = 0; i < repeat; ++i) fn(s);
     CHECK(cudaEventRecord(ed, s));
     CHECK(cudaEventSynchronize(ed));
     CHECK(cudaEventElapsedTime(&ms, st, ed));
     CHECK(cudaEventDestroy(st)); CHECK(cudaEventDestroy(ed));
     return ms / repeat;
 }
 inline void fill(float* p, uint32_t n, float v) {
     for (uint32_t i = 0; i < n; ++i) p[i] = v;
 }
 inline bool verify(const float* p, uint32_t n, float v) {
     for (uint32_t i = 0; i < n; ++i) if (p[i] != v) return false;
     return true;
 }
 
 /* --------------------- main ---------------------------- */
 int main(int argc, char* argv[]) {
     /* --- default parameters --- */
     int     dev   = 0;
     int64_t n64   = 10'000'000;
     bool    use_ratio = false;
     double  ratio = 1.0;
     bool    use_pf = false;                         // --pf
     struct Advise { enum Kind{AB,PL,RM}; Kind k; int arg; };
     std::vector<Advise> advises;                    // collected advices
 
     /* --- CLI parsing --- */
     for (int i = 1; i < argc; ++i) {
         if (!std::strcmp(argv[i], "-d") || !std::strcmp(argv[i],"--device"))
             dev = std::stoi(argv[++i]);
         else if (!std::strcmp(argv[i], "-r") || !std::strcmp(argv[i],"--ratio")) {
             ratio = std::stod(argv[++i]); use_ratio = true;
         }
         else if (!std::strcmp(argv[i], "-pf"))  use_pf = true;
         else if (!std::strcmp(argv[i], "--advise")) {
             const char* kind = argv[++i];
             const char* tgt  = argv[++i];
             if (!std::strcmp(kind, "ab"))
                 advises.push_back({Advise::AB,
                                    std::atoi(tgt+3)});      // gpuX
             else if (!std::strcmp(kind, "pl")) {
                 if (!std::strncmp(tgt,"gpu",3))
                     advises.push_back({Advise::PL, std::atoi(tgt+3)});
                 else if (!std::strcmp(tgt,"cpu"))
                     advises.push_back({Advise::PL, -1});
                 else throw std::runtime_error{"pl target must cpu|gpuX"};
             }
             else if (!std::strcmp(kind, "rm"))
                 advises.push_back({Advise::RM, 0});
             else throw std::runtime_error{"unknown advise kind"};
         }
         else if (i == 1) n64 = std::stoll(argv[i]);
         else throw std::runtime_error{"unknown arg: " + std::string(argv[i])};
     }
 
     int devcnt=0; CHECK(cudaGetDeviceCount(&devcnt));
     if (dev<0||dev>=devcnt) throw std::runtime_error{"invalid GPU id"};
     CHECK(cudaSetDevice(dev));
 
     /* ratio→n：仍按 3× 缓冲估算一致的 n */
     if (use_ratio) {
         cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, dev));
         n64 = static_cast<int64_t>( ratio * prop.totalGlobalMem /
                                     (3.0 * sizeof(float)) );
     }
     uint32_t n = static_cast<uint32_t>(std::min<int64_t>(n64, 0xFFFFFFFFull));
     size_t   bytes = static_cast<size_t>(n) * sizeof(float);
 
     /* --- allocate Unified Memory --- */
     float *a,*b,*o;
     CHECK(cudaMallocManaged(&a, bytes));
     CHECK(cudaMallocManaged(&b, bytes));
     CHECK(cudaMallocManaged(&o, bytes));
 
     cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
 
    /* --- apply cudaMemAdvise --- */
    for (auto &ad : advises) {
        if (ad.k == Advise::AB) {
            CHECK(cudaMemAdvise(a, bytes, cudaMemAdviseSetAccessedBy, ad.arg));
            CHECK(cudaMemAdvise(b, bytes, cudaMemAdviseSetAccessedBy, ad.arg));
            CHECK(cudaMemAdvise(o, bytes, cudaMemAdviseSetAccessedBy, ad.arg));
        }
        else if (ad.k == Advise::PL) {
            CHECK(cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, ad.arg));
            CHECK(cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, ad.arg));
            CHECK(cudaMemAdvise(o, bytes, cudaMemAdviseSetPreferredLocation, ad.arg));
        }
        else if (ad.k == Advise::RM) {
            CHECK(cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, 0));
            CHECK(cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, 0));
        }
    }

 
     /* --- optional prefetch --- */
     if (use_pf) {
         CHECK(cudaMemPrefetchAsync(a, bytes, dev, stream));
         CHECK(cudaMemPrefetchAsync(b, bytes, dev, stream));
         CHECK(cudaMemPrefetchAsync(o, bytes, dev, stream));
     }
 
     /* --- warm-up & timed run --- */
     float v1=1.f,v2=1.f,v0=0.f,v_ref=2.f;
     fill(a,n,v1); fill(b,n,v2); fill(o,n,v0);
     dim3 blk(1024), grd(32);
     vec_add<<<grd,blk,0,stream>>>(o,a,b,n);
     CHECK(cudaStreamSynchronize(stream));
     if (!verify(o,n,v_ref)) throw std::runtime_error{"verify failed"};
 
     auto fn = [&](cudaStream_t s){
         vec_add<<<grd,blk,0,s>>>(o,a,b,n);
     };
     double lat = time_ms(fn, stream, /*repeat=*/10, /*warmup=*/10);
     std::cout<<std::fixed<<std::setprecision(3)
              <<"[GPU"<<dev<<"] n="<<n<<" elements  latency="<<lat<<" ms\n";
 
     /* --- cleanup --- */
     CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(o));
     CHECK(cudaStreamDestroy(stream));
     return 0;
 }
 