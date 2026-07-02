#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -o fdtd2d

clean:
	rm -rf fdtd2d main gpgpusim_power_report* _cuobjdump_* *.txt
