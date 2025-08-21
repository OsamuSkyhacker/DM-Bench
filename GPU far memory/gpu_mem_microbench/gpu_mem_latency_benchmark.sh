#!/bin/bash
###############################################################################
# gpu_mem_latency_benchmark.sh
#
# 按 NUMA 绑定方式批量运行 cudaMemcpy 延迟测试
#  - H2D / D2H    （可按需加 D2D）
#  - pinned / pinned+WC
#  - 数据范围 1 KB–1 GB (--range 10 30)
#  - 结果 CSV 自动重命名保存
#
# 依赖：
#   1. 已编译好的 latencyTest 可执行文件（见前述 C++ 代码）
#   2. numactl (可选；如无 NUMA 需求可删掉 numactl 行)
###############################################################################

#---------------------------- 参数配置 ----------------------------#
device=0            # CUDA 设备号
iterations=1        # 整体重复次数（CSV 文件也跟着重命名）
numactl_node=0      # NUMA 节点，如果不想绑定，置空即可
NODE_LOCAL=0       # 本地 DRAM 节点
iter_per_size=100 # latencyTest 内部每个数据大小循环次数 (--iters)

# latencyTest 可执行文件路径
lat_exe=./copy_latency_test

#---------------------------- 函数封装 ----------------------------#
run_test () {
    local mem_flag=$1       # "pinned" | "pinned wc"
    local direction=$2      # "h2d" | "d2h" | "d2d"
    local tag=$3            # 文件名 tag

    echo "Running ${direction^^} test (${mem_flag}), tag=$tag"

    # 组合命令行
    cmd=("${lat_exe}")
    # memory flag
    if [[ $mem_flag == "pinned" ]]; then
        cmd+=("pinned")
    else
        cmd+=("pinned" "wc")
    fi
    # copy direction
    cmd+=("${direction}")
    # others
    cmd+=("--device" "${device}" \
          "--range" 10 30 \
          "--iters" "${iter_per_size}" \
          "--csv" "latency_results.csv")

    # 若需要 NUMA 绑定
    if [[ -n $numactl_node ]]; then
        numactl --membind=${numactl_node} --cpunodebind=${NODE_LOCAL} "${cmd[@]}"
    else
        "${cmd[@]}"
    fi

    # 重命名结果文件
    mv latency_results.csv "latency_${tag}_$iteration.csv"
}

#---------------------------- 主循环 ----------------------------#
for ((iteration=0; iteration<iterations; iteration++))
do
    ###### 不带 WC ######
    run_test "pinned" "h2d" "pinned_h2d_dram"
    run_test "pinned" "d2h" "pinned_d2h_dram"

    ###### 带 Write-Combined (WC) ######
    run_test "pinned wc" "h2d" "pinned_wc_h2d_dram"
    run_test "pinned wc" "d2h" "pinned_wc_d2h_dram"

    # 若需要测 D2D，可取消注释
    # run_test "pinned"       "d2d" "d2d"
done

echo "Latency benchmarking complete!"
