#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets

default:
	$(NVCC) $(NVCC_FLAGS) pathfinder.cu -lcudart -o pathfinder

multistream:
	$(NVCC) $(NVCC_FLAGS) -DMULTISTREAM pathfinder.cu -lcudart -o pathfinder

# 兼容旧名（PREF）
prefetch:
	$(NVCC) $(NVCC_FLAGS) -DMULTISTREAM pathfinder.cu -lcudart -o pathfinder

clean:
	rm -f pathfinder *.txt gpgpusim_power_report* _cuobjdump_*
