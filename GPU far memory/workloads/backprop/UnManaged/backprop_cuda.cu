

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <ctype.h>
int FLAG_AB = 0, FLAG_RM = 0, FLAG_PL = 0, FLAG_PF = 0;
int DEV_AB = -1, DEV_RM = -1, DEV_PL = -1, DEV_PF = -1;

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
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float  sum;
  /* 展开数组已弃用，直接逐行 memcpy */
  float *input_weights_one_dim;     // 新增：一次性展平权重
  float *input_weights_prev_one_dim;

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
  /* New Code End */

  /* 不再为一维展开数组 malloc，节省 host 内存 */
  partial_sum = (float *) malloc(blocks_x * hid * sizeof(float));

 
  /* 一次性展平权重到一维数组，再整体复制到 GPU，随后释放，峰值内存瞬间占用 */

  size_t total_elems = (size_t)(in + 1) * (hid + 1);
  input_weights_one_dim        = (float*) malloc(total_elems * sizeof(float));
  input_weights_prev_one_dim   = (float*) malloc(total_elems * sizeof(float));

  if (!input_weights_one_dim || !input_weights_prev_one_dim) {
      fprintf(stderr, "Host malloc for weight flatten failed\n");
      exit(EXIT_FAILURE);
  }

  size_t idx = 0;
  for (int k = 0; k <= in; ++k) {
      for (int j = 0; j <= hid; ++j) {
          input_weights_one_dim[idx]      = net->input_weights[k][j];
          input_weights_prev_one_dim[idx] = net->input_prev_weights[k][j];
          ++idx;
      }
  }

  cudaMalloc((void**)&input_hidden_cuda,        total_elems * sizeof(float));
  cudaMalloc((void**)&input_prev_weights_cuda,  total_elems * sizeof(float));

  cudaMemcpy(input_hidden_cuda,       input_weights_one_dim,       total_elems * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,  total_elems * sizeof(float), cudaMemcpyHostToDevice);

  /* 立即释放展平数组，降低常驻内存 */
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
  
  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  // input_hidden_cuda 已在上面分配
  // cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  cudaMalloc((void**) &hidden_partial_sum, blocks_x * hid * sizeof(float));
  cudaMemset(hidden_partial_sum, 0, blocks_x * hid * sizeof(float));
  
  
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  
  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  /* 权重已整体复制完成 */

  
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 
  cudaThreadSynchronize();
  
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  
  // cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(partial_sum, hidden_partial_sum, blocks_x * hid * sizeof(float), cudaMemcpyDeviceToHost);
     
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

  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  /* input_prev_weights_cuda 已在前面分配并写入，无需再次 cudaMalloc / cudaMemcpy */
  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);


  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

#if defined(HOST_TRANSFER_ONLY) || defined(WRITE_RESULT)
  /* ---------- 将 GPU 数据拷回 Host ---------- */
  float *host_weights = (float*)malloc(total_elems * sizeof(float));
  if (host_weights) {
      cudaMemcpy(host_weights, input_hidden_cuda,
                 total_elems * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(net->input_units, input_cuda,
                 (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  } else {
      fprintf(stderr, "[WARN] host_weights malloc failed, skip host copy.\n");
  }
#endif

#ifdef WRITE_RESULT
  /* 写入结果到 result.txt（存在则覆盖） */
  if (host_weights) {
      FILE *fp = fopen("result.txt", "w");
      if (fp) {
          fprintf(fp, "Input_units:\n");
          for (int i = 0; i < in + 1; ++i)
              fprintf(fp, "%f ", net->input_units[i]);
          fprintf(fp, "\nInput_weight_one_dim:\n");

          size_t idx = 0;
          for (int k = 0; k <= in; ++k) {
              for (int j = 0; j <= hid; ++j) {
                  fprintf(fp, "%f ", host_weights[idx++]);
              }
              fprintf(fp, "\n");
          }
          fclose(fp);
      }
  }
#endif

#if defined(HOST_TRANSFER_ONLY) || defined(WRITE_RESULT)
  if (host_weights) free(host_weights);
#endif

    
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  free(partial_sum);
  /* 展平数组已在复制后立即释放，如启用 DUMP_RESULT 也会释放临时 host_weights */

#endif   
  
}