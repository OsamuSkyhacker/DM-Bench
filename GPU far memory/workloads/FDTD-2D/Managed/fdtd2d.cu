#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <strings.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
//#define tmax 500
//#define NX 2048
//#define NY 2048

#define tmax 5
#define NX 23168
#define NY 23168


/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
	for (t=0; t < tmax; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       		for (j = 0; j < NY; j++)
			{
       			ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}

/*
void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}
*/


__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		if (i == 0) 
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)//, DATA_TYPE* hz_outputFromGpu)
{
	//double t_start, t_end;
/*
	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);

	cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
	cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
*/
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

	//t_start = rtclock();

	for(int t = 0; t< tmax; t++)
	{
		fdtd_step1_kernel<<<grid,block>>>(_fict_, ex, ey, hz, t);
		cudaThreadSynchronize();
		fdtd_step2_kernel<<<grid,block>>>(ex, ey, hz, t);
		cudaThreadSynchronize();
		fdtd_step3_kernel<<<grid,block>>>(ex, ey, hz, t);
		cudaThreadSynchronize();
	}
	
	//t_end = rtclock();
    	//fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	//cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);	
		
	//cudaFree(_fict_gpu);
	//cudaFree(ex_gpu);
	//cudaFree(ey_gpu);
	//cudaFree(hz_gpu);
}


static int FLAG_AB = 0, FLAG_RM = 0, FLAG_PL = 0, FLAG_PF = 0;
static int DEV_AB  = -1, DEV_RM = -1, DEV_PL = -1, DEV_PF = -1;
#ifndef cudaCpuDeviceId
#define cudaCpuDeviceId (-1)
#endif

static double wall_time()
{
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv)
{
	//double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	//DATA_TYPE* hz_outputFromGpu;

/*
	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
*/	
    cudaMallocManaged(&_fict_, tmax*sizeof(DATA_TYPE));
    cudaMallocManaged(&ex, NX*(NY+1)*sizeof(DATA_TYPE));
    cudaMallocManaged(&ey, (NX+1)*NY*sizeof(DATA_TYPE));
    cudaMallocManaged(&hz, NX*NY*sizeof(DATA_TYPE));
	
	//hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

    init_arrays(_fict_, ex, ey, hz);

    double t_start = wall_time();

    // Parse UM flags
    {
        FLAG_AB = FLAG_RM = FLAG_PL = FLAG_PF = 0;
        DEV_AB = DEV_RM = DEV_PL = DEV_PF = -1;
        int idx = 1;
        while (idx < argc) {
            if (strcmp(argv[idx], "AB") == 0 && idx + 1 < argc) {
                FLAG_AB = 1; DEV_AB = (strcasecmp(argv[idx+1], "cpu") == 0) ? cudaCpuDeviceId : atoi(argv[idx+1]); idx += 2; continue;
            }
            if (strcmp(argv[idx], "RM") == 0 && idx + 1 < argc) {
                FLAG_RM = 1; DEV_RM = (strcasecmp(argv[idx+1], "cpu") == 0) ? cudaCpuDeviceId : atoi(argv[idx+1]); idx += 2; continue;
            }
            if (strcmp(argv[idx], "PL") == 0 && idx + 1 < argc) {
                FLAG_PL = 1; DEV_PL = (strcasecmp(argv[idx+1], "cpu") == 0) ? cudaCpuDeviceId : atoi(argv[idx+1]); idx += 2; continue;
            }
            if (strcmp(argv[idx], "PF") == 0 && idx + 1 < argc) {
                FLAG_PF = 1; DEV_PF = (strcasecmp(argv[idx+1], "cpu") == 0) ? cudaCpuDeviceId : atoi(argv[idx+1]); idx += 2; continue;
            }
            fprintf(stderr, "Unknown or incomplete flag near '%s'\n", argv[idx]);
            return 1;
        }
    }

    // Apply Advise/Prefetch
    {
        auto apply_advise = [&](void* ptr, size_t bytes){
            if (FLAG_AB) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, DEV_AB);
            if (FLAG_RM) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, DEV_RM);
            if (FLAG_PL) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, DEV_PL);
        };
        size_t bytes_fict = (size_t)tmax * sizeof(DATA_TYPE);
        size_t bytes_ex   = (size_t)NX * (size_t)(NY + 1) * sizeof(DATA_TYPE);
        size_t bytes_ey   = (size_t)(NX + 1) * (size_t)NY * sizeof(DATA_TYPE);
        size_t bytes_hz   = (size_t)NX * (size_t)NY * sizeof(DATA_TYPE);
        apply_advise(_fict_, bytes_fict);
        apply_advise(ex,     bytes_ex);
        apply_advise(ey,     bytes_ey);
        apply_advise(hz,     bytes_hz);
        if (FLAG_PF) {
            cudaMemPrefetchAsync(_fict_, bytes_fict, DEV_PF, 0);
            cudaMemPrefetchAsync(ex,     bytes_ex,   DEV_PF, 0);
            cudaMemPrefetchAsync(ey,     bytes_ey,   DEV_PF, 0);
            cudaMemPrefetchAsync(hz,     bytes_hz,   DEV_PF, 0);
            cudaDeviceSynchronize();
        }
    }

	//GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz);//, hz_outputFromGpu);

	//t_start = rtclock();
	//runFdtd(_fict_, ex, ey, hz);
	//t_end = rtclock();
	
	//fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	//compareResults(hz, hz_outputFromGpu);

    {
        size_t bytes = sizeof(DATA_TYPE) * ((size_t)tmax
                     + (size_t)NX * (size_t)(NY + 1)
                     + (size_t)(NX + 1) * (size_t)NY
                     + (size_t)NX * (size_t)NY);
        double mib = (double)bytes / (1024.0 * 1024.0);
        double gib = (double)bytes / (1024.0 * 1024.0 * 1024.0);
        printf("-------------Size: %.3f MiB (%.3f GiB)--------------\n", mib, gib);
    }
	FILE *fp;

	fp = fopen("file.txt","w");

	for(int i = 0; i < NX*NY; i+= 1000) {
		fprintf(fp, "%lf\n", hz[i]);
	}
	
	fclose(fp);

	cudaFree(_fict_);
	cudaFree(ex);
	cudaFree(ey);
    cudaFree(hz);
	//free(hz_outputFromGpu);

    double t_end = wall_time();
    printf("Total elapsed time: %.3f seconds\n", t_end - t_start);
    return 0;
}

