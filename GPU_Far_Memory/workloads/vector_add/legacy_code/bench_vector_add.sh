#!/bin/bash

# 设置设备和测试参数
device=0
iterations=1
numactl_node=0  # 设置为 NUMA 节点 1

numactl --membind=$numactl_node ./vector_add.cu --device $device

