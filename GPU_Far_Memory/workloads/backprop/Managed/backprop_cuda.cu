

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <ctype.h>
// 全局 flag 定义（Managed 版本）
int FLAG_AB = 0, FLAG_RM = 0, FLAG_PL = 0, FLAG_PF = 0;
int DEV_AB  = -1, DEV_RM = -1, DEV_PL = -1, DEV_PF = -1;

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

/* ---------------- 结果与传输功能开关 ----------------
  取消下面的宏以启用相应功能：
   #define HOST_TRANSFER_ONLY   // 仅把数据转移到 Host，不落盘
   #define WRITE_RESULT         // 把数据转移到 Host，并写 result.txt
  兼容：若定义 DUMP_RESULT，则等同于 WRITE_RESULT
 +------------------------------------------------- */
// #define HOST_TRANSFER_ONLY
// #define WRITE_RESULT
#ifdef DUMP_RESULT
#define WRITE_RESULT
#endif

////////////////////////////////////////////////////////////////////////////////
extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);

extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}




////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  int m = 0;  /* 已不再使用，可保留以免改动太多 */
  /* 直接使用统一内存中的数据结构指针 */
  float *input_hidden_cuda      = net->input_weights[0];      // (in+1)*(hid+1)
  float *input_prev_weights_cuda= net->input_prev_weights[0]; // (in+1)*(hid+1)
  float *input_cuda             = net->input_units;           // (in+1)
  float *output_hidden_cuda     = net->hidden_units;          // (hid+1) – kernel 实际未用
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda      = net->hidden_delta;          // (hid+1)
  float  sum;
  // [old Rodinia Benchmark code]
  // dim3  grid( 1 , num_blocks);
  // dim3  threads(16 , 16);

  /* New Code Start */
  /* 2025-07-14 */
  /* 以 TILE_X × TILE_Y 线程块，把 hid/in 划分成二维网格 */
  dim3 threads(TILE_X, TILE_Y);
  int blocks_x = (hid + TILE_X - 1) / TILE_X;   // 隐层 tile 数
  int blocks_y =  in / TILE_Y;                  // 输入 tile 数（in 已保证能被 16 整除）
  dim3 grid(blocks_x, blocks_y);
  int blocks_xy = blocks_x * blocks_y;          // 总 blocks 数，用于 partial 缓冲区
  /* New Code End */

  cudaMallocManaged((void**)&hidden_partial_sum, blocks_x * hid * sizeof(float));
  /* host 聚合直接访问 unified memory */
  partial_sum = hidden_partial_sum;

  /* ---------------- 根据命令行 flag 对各大数组进行 MEMADVISE / PREFETCH ---------------- */
  auto apply_advise = [&](void* ptr, size_t bytes){
      if(FLAG_AB) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, DEV_AB);
      if(FLAG_RM) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly,  DEV_RM);
      if(FLAG_PL) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, DEV_PL);
      if(FLAG_PF) cudaMemPrefetchAsync(ptr, bytes, DEV_PF, 0);
  };

  size_t bytes_in_hidden = (size_t)(in + 1) * (hid + 1) * sizeof(float);
  size_t bytes_in        = (size_t)(in + 1) * sizeof(float);
  size_t bytes_hid       = (size_t)(hid + 1) * sizeof(float);
  size_t bytes_partial   = (size_t)blocks_x * hid * sizeof(float);

  apply_advise(input_hidden_cuda, bytes_in_hidden);
  apply_advise(input_prev_weights_cuda, bytes_in_hidden);
  apply_advise(input_cuda, bytes_in);
  apply_advise(output_hidden_cuda, bytes_hid);
  apply_advise(hidden_delta_cuda, bytes_hid);
  apply_advise(hidden_partial_sum, bytes_partial);

  /* 归零缓存，配合 kernel 中 atomicAdd */
  cudaMemset(hidden_partial_sum, 0, bytes_partial);

  if(FLAG_PF) cudaDeviceSynchronize();    // 等待 Prefetch 完成
  
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  
  /* 统一内存：无需显式 cudaMemcpy */

  
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 
  cudaDeviceSynchronize();
  
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  
  /* hidden_partial_sum 位于统一内存，同步后可直接访问 */
  cudaDeviceSynchronize();
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    // for (int k = 0; k < num_blocks; k++) {	
    //   sum += partial_sum[k * hid + j-1] ;
    // }
    for (int k = 0; k < blocks_x; ++k) {
      sum += partial_sum[k * hid + j - 1];
    }

	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  #endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

  /* 计算权重元素总数供可选调试块使用 */
  size_t total_elems = (size_t)(in + 1) * (hid + 1);

#if defined(HOST_TRANSFER_ONLY) || defined(WRITE_RESULT)
  /* ---------- 仅转移到 Host（UM 预取），或转移后写文件 ---------- */
  float *host_weights = input_hidden_cuda;   /* 统一内存，Host 可直接访问 */
  cudaMemPrefetchAsync(host_weights, total_elems * sizeof(float), cudaCpuDeviceId);
#endif

#ifdef WRITE_RESULT
  /* 写入结果到 result.txt（存在则覆盖，不存在则创建） */
  FILE *fp = fopen("result.txt", "w");
  if (fp) {
      fprintf(fp, "Input_units:\n");
      for (int i = 0; i < in + 1; ++i)
          fprintf(fp, "%f ", net->input_units[i]);
      fprintf(fp, "\nInput_weight_one_dim:\n");

      size_t idx = 0;
      for (int k = 0; k <= in; ++k) {
          for (int j = 0; j <= hid; ++j) {
              fprintf(fp, "%f ", input_hidden_cuda[idx++]);
          }
          fprintf(fp, "\n");
      }
      fclose(fp);
  }
#endif

    
  cudaFree(hidden_partial_sum);
  /* 展平数组已在复制后立即释放，如启用 DUMP_RESULT 也会释放临时 host_weights */

#endif   
  
}