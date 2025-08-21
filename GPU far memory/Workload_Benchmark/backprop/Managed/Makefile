#!/usr/bin/make

# C compiler
# 原 C 编译器（仍用于小工具）
CC = gcc
CC_FLAGS = -g  -O2
CC_FLAGS += -fopenmp

# CUDA compiler
NVCC = nvcc
# NVCC_FLAGS = -g -Wno-deprecated-gpu-targets
NVCC_FLAGS = -g -arch=sm_86 -rdc=true -Xcompiler -fopenmp -Wno-deprecated-gpu-targets

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o
	$(NVCC) $(NVCC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -lcudart -lm -o backprop

# 默认规则取消，采用显式规则避免 gcc 处理含 CUDA API 的源文件

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop.c -o backprop.o

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
