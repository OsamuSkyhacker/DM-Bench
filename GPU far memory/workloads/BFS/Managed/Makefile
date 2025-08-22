#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -std=c++11 -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -o bfs 

multistream:
	$(NVCC) $(NVCC_FLAGS) -DMULTISTREAM bfs.cu -lcudart -o bfs 

# 兼容旧名（PREF）
prefetch:
	$(NVCC) $(NVCC_FLAGS) -DMULTISTREAM bfs.cu -lcudart -o bfs 

clean:
	rm -f bfs *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
