#!/usr/bin/env bash
# bench_um_pf.sh —— 统一测试 UM 与 UM+PF 两种策略（所有节点），写单一 CSV
# 伪代码执行顺序：
#   numactl --membind=0 ./vector_add_um [-pf] ...
#   numactl --membind=1 ./vector_add_um [-pf] ...
#   numactl --membind=2 ./vector_add_um [-pf] ...

set -euo pipefail

# ---------- 自行调整的路径 / 参数 ----------
VEC_UM="./vector_add_um"   # Unified-Memory 可执行文件
GPU_ID=0                   # CUDA 设备 ID
NODE_LOCAL=0               # 本地 DRAM
NODE_REMOTE=1              # 远端 NUMA-DRAM
NODE_PMEM=2                # 本地 PMem

CSV_ALL="runtime_all.csv"  # 统一输出

# ---------- 写表头 ----------
echo "Oversub,Local_UM,Local_UM_PF,Remote_UM,Remote_UM_PF,PMEM_UM,PMEM_UM_PF" > "$CSV_ALL"

# ---------- 工具函数：执行命令并抓取 “number ms” ----------
run() {
    local cmd="$1"
    # 只提取形如 “123.456 ms” 的数字；失败则写 NA
    local out
    out=$($cmd 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+ ms' | awk '{print $1}')
    printf "%s" "${out:-NA}"
}

# ---------- 主循环 ----------
for ratio in $(seq 0.1 0.1 2.5); do
    printf "%.1f" "$ratio" >  tmp_line
    real_ratio=$(awk "BEGIN{printf \"%.3f\", $ratio - 0.025}")

    # -------- node-0 ----------
    printf "," >> tmp_line
    run "numactl --membind=$NODE_LOCAL --cpunodebind=${NODE_LOCAL} $VEC_UM -d $GPU_ID -r $real_ratio"               >> tmp_line
    printf "," >> tmp_line
    run "numactl --membind=$NODE_LOCAL --cpunodebind=${NODE_LOCAL} $VEC_UM -d $GPU_ID -r $real_ratio -pf"           >> tmp_line

    # -------- node-1 ----------
    printf "," >> tmp_line
    run "numactl --membind=$NODE_REMOTE --cpunodebind=${NODE_LOCAL} $VEC_UM -d $GPU_ID -r $real_ratio"               >> tmp_line
    printf "," >> tmp_line
    run "numactl --membind=$NODE_REMOTE --cpunodebind=${NODE_LOCAL} $VEC_UM -d $GPU_ID -r $real_ratio -pf"           >> tmp_line

    # -------- node-2 ----------
    printf "," >> tmp_line
    run "numactl --membind=$NODE_PMEM --cpunodebind=${NODE_LOCAL}  $VEC_UM -d $GPU_ID -r $real_ratio"               >> tmp_line
    printf "," >> tmp_line
    run "numactl --membind=$NODE_PMEM --cpunodebind=${NODE_LOCAL}  $VEC_UM -d $GPU_ID -r $real_ratio -pf"           >> tmp_line

    cat tmp_line >> "$CSV_ALL"
    echo        >> "$CSV_ALL"
    rm tmp_line
done

echo "✅  完成！全部结果已写入 $CSV_ALL"

