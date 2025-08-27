#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>
#include <strings.h>
#ifndef cudaCpuDeviceId
#define cudaCpuDeviceId (-1)
#endif

// includes, kernels
#include "needle_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// Unified Memory flags (Managed)
static int FLAG_AB = 0; // AccessedBy
static int FLAG_RM = 0; // ReadMostly
static int FLAG_PL = 0; // PreferredLocation
static int FLAG_PF = 0; // Prefetch

static int DEV_AB = 0;
static int DEV_RM = 0; // reserved for symmetry
static int DEV_PL = 0;
static int DEV_PF = 0;


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

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

  printf("WG size of kernel = %d \n", BLOCK_SIZE);
  double __t0 = gettime();
  runTest( argc, argv);
  double __t1 = gettime();
  printf("Total wall time: %.6f seconds\n", __t1 - __t0);

  return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> [UM flags]\n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	fprintf(stderr, "\t[UM flags] (Managed only, hotspot-style): AB dev | RM dev | PL dev | PF dev (dev can be gpu id or 'cpu')\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
        int max_rows, max_cols, penalty;
	int *itemsets,  *referrence;
	size_t size;
	
    
    	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc >= 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    	else{
		usage(argc, argv);
    	}
	
	if(atoi(argv[1])%16!=0){
		fprintf(stderr,"The dimension values must be a multiple of 16\n");
		exit(1);
	}
	

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
    
	size = (size_t)max_cols * (size_t)max_rows;

	cudaMallocManaged(&referrence, sizeof(int)*size);
	cudaMallocManaged(&itemsets, sizeof(int)*size);
	

	if (!itemsets)
		fprintf(stderr, "error: can not allocate memory");

    	srand ( 7 );
	
	
    	for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			itemsets[i*max_cols+j] = 0;
		}
	}
	
	printf("Start Needleman-Wunsch\n");
	
	for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
       		itemsets[i*max_cols] = rand() % 10 + 1;
	}
    	for( int j=1; j< max_cols ; j++){    //please define your own sequence.
       		itemsets[j] = rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
			referrence[i*max_cols+j] = blosum62[itemsets[i*max_cols]][itemsets[j]];
		}
	}

    	for( int i = 1; i< max_rows ; i++)
       		itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       		itemsets[j] = -j * penalty;

    // Parse optional UM flags from argv[3..] (hotspot-style)
    {
        FLAG_AB = FLAG_RM = FLAG_PL = FLAG_PF = 0;
        DEV_AB = DEV_RM = DEV_PL = DEV_PF = -1;
        int idx = 3;
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
            exit(1);
        }
    }

    {
        size_t gpu_bytes = 2ull * size * sizeof(int);
        double gpu_gib = (double)gpu_bytes / (1024.0*1024.0*1024.0);
        printf("Estimated GPU Unified Memory: %.2f GiB (%zu bytes)\n", gpu_gib, gpu_bytes);
    }

    auto apply_advise = [&](void* ptr, size_t bytes, cudaStream_t pf_stream){
        if (FLAG_AB) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, DEV_AB);
        if (FLAG_RM) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly, DEV_RM);
        if (FLAG_PL) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, DEV_PL);
        if (FLAG_PF) cudaMemPrefetchAsync(ptr, bytes, DEV_PF, pf_stream);
    };

#ifdef MULTISTREAM
    cudaStream_t stream1; cudaStreamCreate(&stream1);
    cudaStream_t stream2; cudaStreamCreate(&stream2);
    cudaStream_t stream3; cudaStreamCreate(&stream3);
    apply_advise(referrence, sizeof(int)*size, stream1);
    apply_advise(itemsets,   sizeof(int)*size, stream2);
#else
    apply_advise(referrence, sizeof(int)*size, 0);
    apply_advise(itemsets,   sizeof(int)*size, 0);
#endif

        dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;

	printf("Processing top-left matrix\n");
	
	//process top-left matrix
	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
#ifdef MULTISTREAM
		needle_cuda_shared_1<<<dimGrid, dimBlock, 0, stream3>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#else
		needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#endif
	}
	
	printf("Processing bottom-right matrix\n");

    	//process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
#ifdef MULTISTREAM
		needle_cuda_shared_2<<<dimGrid, dimBlock, 0, stream3>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#else
		needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence, itemsets, max_cols, penalty, i, block_width);
#endif
	}


	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
#ifdef MULTISTREAM
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
#endif
	
#ifdef BENCH_PRINT
	
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
    
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           		break;
		if ( i > 0 && j > 0 ){
			nw = itemsets[(i - 1) * max_cols + j - 1];
		    	w  = itemsets[ i * max_cols + j - 1 ];
            		n  = itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    	nw = n = LIMIT;
		    	w  = itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    	nw = w = LIMIT;
            	   	n  = itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            		traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
			{i--; j--; continue;}

        	else if(traceback == w )
			{j--; continue;}

        	else if(traceback == n )
			{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

#endif

	cudaFree(referrence);
	cudaFree(itemsets);

}

