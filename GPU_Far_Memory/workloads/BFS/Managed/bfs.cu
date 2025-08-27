/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <strings.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

static double wall_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define MAX_THREADS_PER_BLOCK 512

/* ------------ Unified Memory flags (Managed version) ------------ */
int FLAG_AB = 0, FLAG_RM = 0, FLAG_PL = 0, FLAG_PF = 0;
int DEV_AB  = -1, DEV_RM = -1, DEV_PL = -1, DEV_PF = -1;

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
    double t_start = wall_time();
    BFSGraph( argc, argv);
    double t_end = wall_time();
    printf("Total elapsed time: %.3f seconds\n", t_end - t_start);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file> [AB dev] [RM dev] [PL dev] [PF dev]\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

    char *input_f;
	if(argc<2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];

    /* ----------- Unified Memory flag parsing ----------- */
    FLAG_AB = FLAG_RM = FLAG_PL = FLAG_PF = 0;
    DEV_AB = DEV_RM = DEV_PL = DEV_PF = -1;

    int compute_dev = -1;   /* 用于 cudaSetDevice */
    int idx = 2;
    while(idx < argc) {
        if (strcmp(argv[idx], "DEV") == 0 && idx + 1 < argc) {
            compute_dev = atoi(argv[idx+1]);
            idx += 2;
            continue;
        }
        if (strcmp(argv[idx], "AB") == 0 && idx + 1 < argc) {
            FLAG_AB = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0)
                DEV_AB = cudaCpuDeviceId;
            else {
                DEV_AB = atoi(argv[idx+1]);
                if(compute_dev==-1) compute_dev = DEV_AB;  /* 记录首个 GPU 作为计算设备 */
            }
            idx += 2;
        } else if (strcmp(argv[idx], "RM") == 0 && idx + 1 < argc) {
            FLAG_RM = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0)
                DEV_RM = cudaCpuDeviceId;
            else {
                DEV_RM = atoi(argv[idx+1]);
                if(compute_dev==-1) compute_dev = DEV_RM;
            }
            idx += 2;
        } else if (strcmp(argv[idx], "PL") == 0 && idx + 1 < argc) {
            FLAG_PL = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0)
                DEV_PL = cudaCpuDeviceId;
            else {
                DEV_PL = atoi(argv[idx+1]);
                if(compute_dev==-1) compute_dev = DEV_PL;
            }
            idx += 2;
        } else if (strcmp(argv[idx], "PF") == 0 && idx + 1 < argc) {
            FLAG_PF = 1;
            if (strcasecmp(argv[idx+1], "cpu") == 0)
                DEV_PF = cudaCpuDeviceId;
            else {
                DEV_PF = atoi(argv[idx+1]);
                if(compute_dev==-1) compute_dev = DEV_PF;
            }
            idx += 2;
        } else {
            fprintf(stderr, "Unknown or incomplete flag near '%s'\n", argv[idx]);
            exit(1);
        }
    }

    /* ----------- 设置计算设备 ----------- */
    if(compute_dev == -1) {
        cudaGetDevice(&compute_dev);  // 保持默认
    }
    cudaSetDevice(compute_dev);
    printf("[Info] Using GPU%d for computation\n", compute_dev);

	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}


	//Allocate the Node list
	Node* graph_nodes;
	cudaMallocManaged(  &graph_nodes, sizeof(Node)*no_of_nodes) ;

	//Allocate the Mask
	bool* graph_mask;
	cudaMallocManaged( &graph_mask, sizeof(bool)*no_of_nodes) ;

	bool* updating_graph_mask;
	cudaMallocManaged( &updating_graph_mask, sizeof(bool)*no_of_nodes) ;

	//Allocate the Visited nodes array
	bool* graph_visited;
	cudaMallocManaged( &graph_visited, sizeof(bool)*no_of_nodes) ;

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
                fscanf(fp,"%d %d",&start,&edgeno);
		graph_nodes[i].starting = start;
		graph_nodes[i].no_of_edges = edgeno;
		graph_mask[i]=false;
		updating_graph_mask[i]=false;
		graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	graph_mask[source]=true;
	graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	//Allocate the Edge List
	int* graph_edges;
	cudaMallocManaged( &graph_edges, sizeof(int)*edge_list_size) ;

	int id,edgeCost;
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&edgeCost);
		graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

	// allocate mem for the result
	int* cost;
	cudaMallocManaged( (void**) &cost, sizeof(int)*no_of_nodes);

	for(int i=0;i<no_of_nodes;i++)
		cost[i]=-1;
	cost[source]=0;
	
        //make a bool to check if the execution is over
        bool *d_over;
        cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Copied Everything to GPU memory\n");

    /* ------------- Apply cudaMemAdvise ------------- */
    {
        auto apply_advise = [&](void* ptr, size_t bytes){
            if(FLAG_AB) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, DEV_AB);
            if(FLAG_RM) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly,  DEV_RM);
            if(FLAG_PL) cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, DEV_PL);
        };

        apply_advise(graph_nodes, sizeof(Node)*no_of_nodes);
        apply_advise(graph_edges, sizeof(int)*edge_list_size);
        size_t bytes_bool = sizeof(bool)*no_of_nodes;
        apply_advise(graph_mask, bytes_bool);
        apply_advise(updating_graph_mask, bytes_bool);
        apply_advise(graph_visited, bytes_bool);
        apply_advise(cost, sizeof(int)*no_of_nodes);
    }

#ifndef MULTISTREAM
    // 单流情形：如启用 PF，则在默认流上预取并同步
    if(FLAG_PF){
        cudaMemPrefetchAsync( graph_nodes, sizeof(Node)*no_of_nodes, DEV_PF, 0);
        cudaMemPrefetchAsync( graph_edges, sizeof(int)*edge_list_size, DEV_PF, 0);
        size_t bytes_bool = sizeof(bool)*no_of_nodes;
        cudaMemPrefetchAsync( graph_mask, bytes_bool, DEV_PF, 0);
        cudaMemPrefetchAsync( updating_graph_mask, bytes_bool, DEV_PF, 0);
        cudaMemPrefetchAsync( graph_visited, bytes_bool, DEV_PF, 0);
        cudaMemPrefetchAsync( cost, sizeof(int)*no_of_nodes, DEV_PF, 0);
        cudaDeviceSynchronize();
    }
#endif

#ifdef MULTISTREAM
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);

	cudaStream_t stream3;
	cudaStreamCreate(&stream3);

	cudaStream_t stream4;
	cudaStreamCreate(&stream4);

	cudaStream_t stream5;
	cudaStreamCreate(&stream5);

	cudaStream_t stream6;
	cudaStreamCreate(&stream6);

    cudaStream_t stream7;
    cudaStreamCreate(&stream7);

    // 多流情形：如启用 PF，则在多流上预取到 DEV_PF；未启用 PF 则不做隐式预取
    if(FLAG_PF){
        cudaMemPrefetchAsync( graph_nodes, sizeof(Node)*no_of_nodes, DEV_PF, stream1);
        cudaMemPrefetchAsync( graph_edges, sizeof(int)*edge_list_size, DEV_PF, stream2);
        cudaMemPrefetchAsync( graph_mask, sizeof(bool)*no_of_nodes, DEV_PF, stream3);
        cudaMemPrefetchAsync( updating_graph_mask, sizeof(bool)*no_of_nodes, DEV_PF, stream4);
        cudaMemPrefetchAsync( graph_visited, sizeof(bool)*no_of_nodes, DEV_PF, stream5);
        cudaMemPrefetchAsync( cost, sizeof(int)*no_of_nodes, DEV_PF, stream6);
    }
#endif

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
        bool stop;	
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
                //if no thread changes this value then the loop stops
                stop=false;
                cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;

#ifdef MULTISTREAM
		Kernel<<< grid, threads, 0, stream7>>>( graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes);
		// check if kernel execution generated and error
		

        Kernel2<<< grid, threads, 0, stream7>>>( graph_mask, updating_graph_mask, graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
#else
		Kernel<<< grid, threads, 0 >>>( graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes);
		// check if kernel execution generated and error
		

		Kernel2<<< grid, threads, 0 >>>( graph_mask, updating_graph_mask, graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
		
#endif        

                cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
		k++;
	}
	while(stop); //if no thread changes this value then the loop stops


        cudaDeviceSynchronize();

	printf("Kernel Executed %d times\n",k);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	cudaFree(graph_nodes);
	cudaFree(graph_edges);
	cudaFree(graph_mask);
	cudaFree(updating_graph_mask);
	cudaFree(graph_visited);
	cudaFree(cost);
	cudaFree(d_over);
}
