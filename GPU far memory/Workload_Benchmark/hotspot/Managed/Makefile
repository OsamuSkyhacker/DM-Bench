#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

SRC = hotspot.cu

EXE = hotspot

default:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) hotspot.cu -lcudart -o hotspot

multistream:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DMULTISTREAM hotspot.cu -lcudart -o hotspot

# 兼容旧名
prefetch:
	$(NVCC) $(NVCC_FLAGS) $(KERNEL_DIM) -DMULTISTREAM hotspot.cu -lcudart -o hotspot

clean:
	rm -f hotspot *.txt gpgpusim_power_report* _cuobjdump_*
