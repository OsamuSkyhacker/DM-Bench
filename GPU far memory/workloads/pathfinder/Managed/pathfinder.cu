#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* gpuWall;
int* gpuResult[2];
#define M_SEED 9
int pyramid_height;

// Unified Memory advise/prefetch flags (Managed version)
int FLAG_AB = 0, FLAG_RM = 0, FLAG_PL = 0, FLAG_PF = 0;
int DEV_AB  = -1, DEV_RM = -1, DEV_PL = -1, DEV_PF = -1; // device id or cudaCpuDeviceId
#ifndef cudaCpuDeviceId
#define cudaCpuDeviceId (-1)
#endif

static double wall_time()
{
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

void
init(int argc, char** argv)
{
    if(argc < 4){
        printf("Usage: dynproc row_len col_len pyramid_height [AB dev|cpu] [RM dev|cpu] [PL dev|cpu] [PF dev|cpu]\n");
        exit(0);
    }
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);

    // Parse optional UM flags starting from argv[4]
    FLAG_AB = FLAG_RM = FLAG_PL = FLAG_PF = 0;
    DEV_AB = DEV_RM = DEV_PL = DEV_PF = -1;
    int idx = 4;
    while(idx < argc){
        if (strcmp(argv[idx], "AB") == 0 && idx + 1 < argc) {
            FLAG_AB = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0) DEV_AB = cudaCpuDeviceId; else DEV_AB = atoi(argv[idx+1]);
            idx += 2; continue;
        }
        if (strcmp(argv[idx], "RM") == 0 && idx + 1 < argc) {
            FLAG_RM = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0) DEV_RM = cudaCpuDeviceId; else DEV_RM = atoi(argv[idx+1]);
            idx += 2; continue;
        }
        if (strcmp(argv[idx], "PL") == 0 && idx + 1 < argc) {
            FLAG_PL = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0) DEV_PL = cudaCpuDeviceId; else DEV_PL = atoi(argv[idx+1]);
            idx += 2; continue;
        }
        if (strcmp(argv[idx], "PF") == 0 && idx + 1 < argc) {
            FLAG_PF = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0) DEV_PF = cudaCpuDeviceId; else DEV_PF = atoi(argv[idx+1]);
            idx += 2; continue;
        }
        fprintf(stderr, "Unknown or incomplete flag near '%s'\n", argv[idx]);
        exit(1);
    }
        checkCuda(cudaMallocManaged((void**)&gpuResult[0], sizeof(int)*cols), "cudaMallocManaged(gpuResult[0])");
        checkCuda(cudaMallocManaged((void**)&gpuResult[1], sizeof(int)*cols), "cudaMallocManaged(gpuResult[1])");
        {
            size_t numWall = (size_t)rows * (size_t)cols - (size_t)cols;
            checkCuda(cudaMallocManaged((void**)&gpuWall, sizeof(int)*numWall), "cudaMallocManaged(gpuWall)");
        }
	
	int seed = M_SEED;
	srand(seed);

    for (int i = 0; i < rows; i++)
        {
        	for (int j = 0; j < cols; j++)
        	{
            if (i == 0) {
                gpuResult[0][j] = rand() % 10;
            } else {
                size_t off = ((size_t)(i-1)) * (size_t)cols + (size_t)j;
                gpuWall[off] = rand() %10;
            }
        	}
    	}
#ifdef BENCH_PRINT
    // 打印大规模数据会非常慢且占内存，建议仅在小规模下启用
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? (BLOCK_SIZE-1-(blkXmax-(cols-1))) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  long long indexWall = (long long)cols * (long long)(startStep + i) + (long long)xidx;
                  result[tx] = shortest + gpuWall[indexWall];
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);  

#ifdef MULTISTREAM
    cudaStream_t stream3; // kernel stream
    cudaStreamCreate(&stream3);
#endif

        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
#ifdef MULTISTREAM
            dynproc_kernel<<<dimGrid, dimBlock, 0, stream3>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
#else
	    dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
#endif
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    double t_start = wall_time();
    run(argc,argv);
    double t_end = wall_time();
    printf("Total elapsed time: %.3f s\n", t_end - t_start);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int size = rows*cols;

    // --------- Apply MemAdvise and Prefetch according to flags ---------
    auto apply_advise = [&](void* ptr, size_t bytes){
        if(FLAG_AB) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, DEV_AB);
        if(FLAG_RM) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly,  DEV_RM);
        if(FLAG_PL) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, DEV_PL);
    };
    apply_advise(gpuResult[0], sizeof(int)*cols);
    apply_advise(gpuResult[1], sizeof(int)*cols);
    apply_advise(gpuWall,      sizeof(int)*(size-cols));

#ifndef MULTISTREAM
    if(FLAG_PF){
        cudaMemPrefetchAsync(gpuResult[0], sizeof(int)*cols, DEV_PF, 0);
        cudaMemPrefetchAsync(gpuResult[1], sizeof(int)*cols, DEV_PF, 0);
        cudaMemPrefetchAsync(gpuWall,      sizeof(int)*(size-cols), DEV_PF, 0);
        cudaDeviceSynchronize();
    }
#endif

#ifdef MULTISTREAM
    cudaStream_t stream1; cudaStreamCreate(&stream1);
    cudaStream_t stream2; cudaStreamCreate(&stream2);
    if(FLAG_PF){
        cudaMemPrefetchAsync( gpuResult[0], sizeof(int)*cols, DEV_PF, stream1);
        cudaMemPrefetchAsync( gpuWall,      sizeof(int)*(size-cols), DEV_PF, stream2);
    }
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

    cudaDeviceSynchronize();

#ifdef BENCH_PRINT
    // Write final result row to file (create if missing, overwrite if exists)
    {
        FILE *fp = fopen("result.txt", "w");
        if (fp) {
            for (int i = 0; i < cols; i++)
                fprintf(fp, "%d ", gpuResult[final_ret][i]);
            fprintf(fp, "\n");
            fclose(fp);
            printf("Result stored in result.txt\n");
        } else {
            fprintf(stderr, "Failed to open result.txt for writing\n");
        }
    }
#endif


    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    // no host-side giant data buffer

}

