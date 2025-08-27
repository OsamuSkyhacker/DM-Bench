#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"

. "$ROOT/bench/lib/bench_utils.sh"
. "$ROOT/bench/lib/hog_ctl.sh"
. "$ROOT/bench/lib/math.sh"

PLAN="$ROOT/bench/configs/plan.yaml"
WY="$ROOT/bench/configs/workloads.yaml"

# 简化解析：用 grep/awk 拿到几个关键列表（生产可换 yq）
gpu_id=$(grep -E '^gpu_id:' "$PLAN" | awk -F: '{gsub(/ /,""); print $2}')
ratios=($(grep -E '^ratios:' -A0 "$PLAN" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' '))
workloads=($(grep -E '^workloads:' -A0 "$PLAN" | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' ,'))
modes=($(grep -E '^modes:' -A0 "$PLAN" | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' ,'))

declare -A nodes_membind nodes_cpubind
while read -r tag rest; do
  [[ -z "$tag" ]] && continue
  case "$tag" in
    local:) nodes_membind[local]=$(echo "$rest" | grep -Eo 'membind: [0-9]+' | awk '{print $2}'); nodes_cpubind[local]=$(echo "$rest" | grep -Eo 'cpunodebind: [0-9]+' | awk '{print $2}');;
    remote:) nodes_membind[remote]=$(echo "$rest" | grep -Eo 'membind: [0-9]+' | awk '{print $2}'); nodes_cpubind[remote]=$(echo "$rest" | grep -Eo 'cpunodebind: [0-9]+' | awk '{print $2}');;
    pmem:) nodes_membind[pmem]=$(echo "$rest" | grep -Eo 'membind: [0-9]+' | awk '{print $2}'); nodes_cpubind[pmem]=$(echo "$rest" | grep -Eo 'cpunodebind: [0-9]+' | awk '{print $2}');;
  esac
done < <(grep -E '^(local|remote|pmem):' -A0 "$PLAN")

# 读取 total/free；按 Free0 口径计算 hog
read -r _ Free0 < <(gpu_mem_info_mb "$gpu_id")

for wl in "${workloads[@]}"; do
  # 读取参数
  B=$(grep -A8 "^\s*${wl}:" "$WY" | grep baseline_payload_mb | awk -F: '{gsub(/ /,""); print $2}' | head -n1)
  Over=$(grep -E '^\s*task_overhead_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  GTask=$(grep -E '^\s*global_task_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  FreeAdj=$(grep -E '^\s*free_adjust_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')

  ts=$(date +%Y%m%d_%H%M%S)
  outdir="$ROOT/bench/results/$wl/$ts"
  mkdir -p "$outdir"

  # CSV 表头
  header=("Oversub" "unmanaged" "um_base" "um_pf" "um_ab" "um_pl_cpu" "um_rm")
  for tag in local remote pmem; do
    csv="$outdir/runtime_${tag}.csv"
    csv_write_header "$csv" "${header[*]}"
  done

  for r in "${ratios[@]}"; do
    x=$(calc_hog_gb_from_free0 "$Free0" "$FreeAdj" "$B" "$r" "$Over" "$GTask")
    HOG_BIN="$WB_ROOT/DummyCudaMalloc/gpu_mem_hog"
    pid=""; if [[ "$x" != "0.000" ]]; then pid=$(start_hog "$HOG_BIN" "$gpu_id" "$x"); wait_hog_ready "$gpu_id" 5 || true; fi
    trap '[[ -n "$pid" ]] && stop_hog "$pid"' EXIT

    for tag in local remote pmem; do
      membind=${nodes_membind[$tag]}; cpubind=${nodes_cpubind[$tag]}
      # unmanaged
      cmd_un="$ROOT/bench/workloads/${wl}.sh run --mode unmanaged --gpu $gpu_id"
      ms_un=$(grab_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_un")")
      # um_base
      cmd_b="$ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id"
      ms_b=$(grab_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_b")")
      # um_pf
      cmd_pf="$ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-pf"
      ms_pf=$(grab_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_pf")")
      # um_ab
      cmd_ab="$ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-ab gpu$gpu_id"
      ms_ab=$(grab_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_ab")")
      # um_pl_cpu
      cmd_pl="$ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-pl cpu"
      ms_pl=$(grab_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_pl")")
      # um_rm（修正为带目标）
      cmd_rm="$ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-rm gpu$gpu_id"
      ms_rm=$(grab_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_rm")")

      line="$r,${ms_un:-NA},${ms_b:-NA},${ms_pf:-NA},${ms_ab:-NA},${ms_pl:-NA},${ms_rm:-NA}"
      csv_write_line "$outdir/runtime_${tag}.csv" "$line"
    done

    [[ -n "$pid" ]] && stop_hog "$pid"
  done
done

echo "done. see $ROOT/bench/results/"


