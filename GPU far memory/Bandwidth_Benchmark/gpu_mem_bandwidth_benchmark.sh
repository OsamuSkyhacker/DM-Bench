#!/bin/bash

# 设置设备和测试参数
device=0
iterations=1
numactl_node=0  # 设置为 NUMA 节点 1
NODE_LOCAL=0  # 设置为 NUMA 节点 0

# H2D 和 D2H 测试，不带写结合（wc）
for ((i=0; i<$iterations; i++))
do
    echo "Running H2D test, iteration $i"
    # 运行 H2D 测试，未使用 wc，指定使用 NUMA 节点
    numactl --membind=$numactl_node --cpunodebind=$NODE_LOCAL ./copy_bandwidth_test pinned h2d --device $device
    mv ./bandwidth_results.csv ./bandwidth_results_pinned_h2d_dram_$i.csv

    echo "Running D2H test, iteration $i"
    # 运行 D2H 测试，未使用 wc，指定使用 NUMA 节点
    numactl --membind=$numactl_node --cpunodebind=$NODE_LOCAL ./copy_bandwidth_test pinned d2h --device $device
    mv ./bandwidth_results.csv ./bandwidth_results_pinned_d2h_dram_$i.csv
done

# H2D 和 D2H 测试，带写结合（wc）
for ((i=0; i<$iterations; i++))
do
    echo "Running H2D test with wc, iteration $i"
    # 运行 H2D 测试，启用 wc，指定使用 NUMA 节点
    numactl --membind=$numactl_node --cpunodebind=$NODE_LOCAL ./copy_bandwidth_test pinned wc h2d --device $device
    mv ./bandwidth_results.csv ./bandwidth_results_pinned_h2d_dram_wc_$i.csv

    echo "Running D2H test with wc, iteration $i"
    # 运行 D2H 测试，启用 wc，指定使用 NUMA 节点
    numactl --membind=$numactl_node --cpunodebind=$NODE_LOCAL ./copy_bandwidth_test pinned wc d2h --device $device
    mv ./bandwidth_results.csv ./bandwidth_results_pinned_d2h_dram_wc_$i.csv
done

echo "Benchmarking complete!"
