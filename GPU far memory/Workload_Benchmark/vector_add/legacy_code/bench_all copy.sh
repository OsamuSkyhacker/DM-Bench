#!/usr/bin/env bash
# bench_all.sh  ——  自动跑表（ratio 需 −0.025，cudaMemcpy 仅 r<=1）

set -euo pipefail

# --------- 可按实际路径调整 ----------
VEC_COPY="./vector_add"          # non-map / map
VEC_UM="./vector_add_um"         # Unified Memory
GPU_ID=0
NODE_LOCAL=0
NODE_REMOTE=1
NODE_PMEM=2

# --------- 输出文件 ----------
CSV_LOCAL="runtime_local_dram.csv"
CSV_NUMA="runtime_remote_dram.csv"
CSV_PMEM="runtime_local_pmem.csv"

echo "Oversub,cudaMemcpy,ZeroCopy,UM,UM+PF,UM+AB" | tee   $CSV_LOCAL > $CSV_NUMA
echo "Oversub,cudaMemcpy,ZeroCopy,UM,UM+PF,UM+AB" >> $CSV_LOCAL
echo "Oversub,cudaMemcpy,ZeroCopy,UM,UM+PF,UM+AB" >  $CSV_PMEM

# --------- 小工具：执行并抽出 “number ms” ----------
run() {
    local cmd="$1"
    out=$($cmd 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+ ms' | awk '{print $1}')
    printf "%s" "${out:-NA}"
}

iterate() {
    local node=$1 csv=$2
    for nominal in 0.3 0.7 1.0 1.5 2.0; do
        # 真正传给程序的 ratio
        real_ratio=$(awk "BEGIN{printf \"%.3f\", $nominal - 0.025}")

        printf "%s" "$nominal" >  tmp_line
        printf "," >> tmp_line

        # ---------- cudaMemcpy ----------
        if awk "BEGIN{exit !($nominal <= 1.0)}"; then
            printf "%s" "$(run "numactl --membind=$node --cpunodebind=0 $VEC_COPY -d $GPU_ID -r $real_ratio -p nonmap")" >> tmp_line
        else
            printf "NA" >> tmp_line
        fi
        printf "," >> tmp_line

        # ---------- Zero-Copy ----------
        printf "%s" "$(run "numactl --membind=$node --cpunodebind=0 $VEC_COPY -d $GPU_ID -r $real_ratio -p map")" >> tmp_line
        printf "," >> tmp_line

        # ---------- UM ----------
        printf "%s" "$(run "numactl --membind=$node --cpunodebind=0 $VEC_UM  -d $GPU_ID -r $real_ratio")" >> tmp_line
        printf "," >> tmp_line

        # ---------- UM + PF ----------
        printf "%s" "$(run "numactl --membind=$node --cpunodebind=0 $VEC_UM  -d $GPU_ID -r $real_ratio -pf")" >> tmp_line
        printf "," >> tmp_line

        # ---------- UM + AB ----------
        printf "%s" "$(run "numactl --membind=$node --cpunodebind=0 $VEC_UM  -d $GPU_ID -r $real_ratio --advise ab gpu$GPU_ID")" >> tmp_line

        cat tmp_line >> "$csv"
        echo        >> "$csv"
        rm tmp_line
    done
}

echo "▶  Local DRAM (node $NODE_LOCAL)"
iterate $NODE_LOCAL $CSV_LOCAL
echo "▶  Remote NUMA DRAM (node $NODE_REMOTE)"
iterate $NODE_REMOTE $CSV_NUMA
echo "▶  Local PMem (node $NODE_PMEM)"
iterate $NODE_PMEM $CSV_PMEM

echo "✅  完成！结果已写入:"
echo "   $CSV_LOCAL"
echo "   $CSV_NUMA"
echo "   $CSV_PMEM"
