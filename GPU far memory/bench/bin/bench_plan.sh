#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"
export WB_ROOT

. "$ROOT/bench/lib/bench_utils.sh"
. "$ROOT/bench/lib/hog_ctl.sh"
. "$ROOT/bench/lib/math.sh"

PLAN="$ROOT/bench/configs/plan.yaml"
WY="$ROOT/bench/configs/workloads.yaml"

# 简化解析：用 grep/awk 拿到几个关键列表（生产可换 yq）
gpu_id=$(grep -E '^gpu_id:' "$PLAN" | awk -F: '{gsub(/ /,""); print $2}')
ratios=($(grep -E '^ratios:' -A0 "$PLAN" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' '))
workloads=($(grep -E '^workloads:' -A0 "$PLAN" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' '))
modes=($(grep -E '^modes:' -A0 "$PLAN" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' '))

debug_log "gpu_id=$gpu_id"
debug_log "ratios=${ratios[*]:-<empty>}"
debug_log "workloads=${workloads[*]:-<empty>}"
debug_log "modes=${modes[*]:-<empty>}"

# 脚本入口时清理历史残留 hog，并注册退出清理
stop_hog_for_gpu "$gpu_id" || true
trap 'stop_hog_for_gpu "$gpu_id" || true' EXIT

# 解析 NUMA 节点（容忍缩进）
declare -A nodes_membind nodes_cpubind
node_tags=()
while read -r tag rest; do
  [[ -z "${tag:-}" ]] && continue
  case "$tag" in
    local:|remote:|pmem:)
      name="${tag%:}"
      nodes_membind["$name"]=$(echo "$rest" | grep -Eo 'membind:[[:space:]]*[0-9]+' | awk '{print $2}')
      nodes_cpubind["$name"]=$(echo "$rest" | grep -Eo 'cpunodebind:[[:space:]]*[0-9]+' | awk '{print $2}')
      node_tags+=("$name")
      ;;
  esac
done < <(grep -E '^[[:space:]]*(local|remote|pmem):' "$PLAN")
debug_log "node_tags=${node_tags[*]:-<empty>}"

if [[ ${#workloads[@]} -eq 0 ]]; then echo "[error] workloads empty"; exit 2; fi
if [[ ${#ratios[@]} -eq 0 ]]; then echo "[error] ratios empty"; exit 2; fi
if [[ ${#node_tags[@]} -eq 0 ]]; then echo "[error] node_tags empty"; exit 2; fi

# 读取 total/free；按 Free0 口径计算 hog
read -r _ Free0 < <(gpu_mem_info_mb "$gpu_id")
debug_log "Free0=$Free0"

for wl in "${workloads[@]}"; do
  # 读取参数
  B=$(grep -A8 "^\s*${wl}:" "$WY" | grep baseline_payload_mb | awk -F: '{gsub(/ /,""); print $2}' | head -n1)
  Over=$(grep -E '^\s*task_overhead_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  GTask=$(grep -E '^\s*global_task_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  FreeAdj=$(grep -E '^\s*free_adjust_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  debug_log "wl=$wl B=$B Over=$Over GTask=$GTask FreeAdj=$FreeAdj"

  ts=$(date +%Y%m%d_%H%M%S)
  outdir="$ROOT/bench/results/$wl/$ts"
  mkdir -p "$outdir"

  # CSV 表头
  header=("Oversub" "unmanaged" "um_base" "um_pf" "um_ab" "um_pl_cpu" "um_rm")
  for tag in "${node_tags[@]}"; do
    csv="$outdir/runtime_${tag}.csv"
    csv_write_header "$csv" "${header[*]}"
  done

  for r in "${ratios[@]}"; do
    x=$(calc_hog_gb_from_free0 "$Free0" "$FreeAdj" "$B" "$r" "$Over" "$GTask")
    x_mb=$(awk -v g="$x" 'BEGIN{printf "%d", g*1024.0 + 0.5}')
    debug_log "ratio=$r hog_gb=$x hog_mb=$x_mb"
    HOG_BIN="$WB_ROOT/DummyCudaMalloc/gpu_mem_hog"
    pid=""; if [[ "$x_mb" != "0" ]]; then pid=$(start_hog "$HOG_BIN" "$gpu_id" "$x_mb"); wait_hog_ready "$gpu_id" 5 || true; fi

    for tag in "${node_tags[@]}"; do
      membind="${nodes_membind[$tag]:-}"
      cpubind="${nodes_cpubind[$tag]:-}"
      if [[ -z "$membind" || -z "$cpubind" ]]; then
        echo "[warn] skip node '$tag' (membind/cpunodebind missing)"
        continue
      fi
      # unmanaged
      cmd_un="bash $ROOT/bench/workloads/${wl}.sh run --mode unmanaged --gpu $gpu_id"
      ms_un=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_un")" "$outdir/${wl}.${tag}.unmanaged.log")
      # um_base
      cmd_b="bash $ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id"
      ms_b=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_b")" "$outdir/${wl}.${tag}.um_base.log")
      # um_pf
      cmd_pf="bash $ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-pf"
      ms_pf=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_pf")" "$outdir/${wl}.${tag}.um_pf.log")
      # um_ab
      cmd_ab="bash $ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-ab gpu$gpu_id"
      ms_ab=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_ab")" "$outdir/${wl}.${tag}.um_ab.log")
      # um_pl_cpu
      cmd_pl="bash $ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-pl cpu"
      ms_pl=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_pl")" "$outdir/${wl}.${tag}.um_pl_cpu.log")
      # um_rm（修正为带目标）
      cmd_rm="bash $ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id --um-rm gpu$gpu_id"
      ms_rm=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_rm")" "$outdir/${wl}.${tag}.um_rm.log")

      line="$r,${ms_un:-NA},${ms_b:-NA},${ms_pf:-NA},${ms_ab:-NA},${ms_pl:-NA},${ms_rm:-NA}"
      csv_write_line "$outdir/runtime_${tag}.csv" "$line"
    done

    [[ -n "$pid" ]] && stop_hog "$pid"
  done
done

echo "done. see $ROOT/bench/results/"


