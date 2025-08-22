#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"

. "$ROOT/bench/lib/bench_utils.sh"
. "$ROOT/bench/lib/hog_ctl.sh"
. "$ROOT/bench/lib/math.sh"

usage(){
  cat <<USAGE
usage: $0 <workload> --node <local|remote|pmem> --ratio <R> --mode <unmanaged|um> [--profile <name>] [--gpu <id>]
USAGE
}

wl=""; node_tag=""; ratio=""; mode=""; profile="um_base"; gpu_id="${GPU_ID:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node) node_tag="$2"; shift 2;;
    --ratio) ratio="$2"; shift 2;;
    --mode) mode="$2"; shift 2;;
    --profile) profile="$2"; shift 2;;
    --gpu) gpu_id="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) if [[ -z "$wl" ]]; then wl="$1"; shift; else echo "unknown arg: $1"; exit 2; fi;;
  esac
done

[[ -z "$wl" || -z "$node_tag" || -z "$ratio" || -z "$mode" ]] && { usage; exit 2; }

# 简化：读取 workloads.yaml 中的 B 和全局参数
WY="$ROOT/bench/configs/workloads.yaml"
Sctx=1024; Delta=0; Sr=512
read -r T Free0 < <(gpu_mem_info_mb "$gpu_id")

# 为演示：从 workloads.yaml 粗略抓 B
B=$(grep -A6 "^\s*${wl}:" "$WY" | grep baseline_unmanaged_vram_mb | awk -F: '{gsub(/ /,""); print $2}' | head -n1)
B=${B:-0}

H=$(calc_hog_mb "$T" "$B" "$ratio" "$Sctx" "$Delta" "$Free0" "$Sr")

HOG_BIN="$WB_ROOT/DummyCudaMalloc/gpu_mem_hog"
pid=""
if [[ "$H" -gt 0 ]]; then
  pid=$(start_hog "$HOG_BIN" "$gpu_id" "$H")
  wait_hog_ready "$gpu_id" 5 || true
fi

trap '[[ -n "$pid" ]] && stop_hog "$pid"' EXIT

case "$node_tag" in
  local) membind=0; cpubind=0;;
  remote) membind=1; cpubind=0;;
  pmem) membind=2; cpubind=0;;
  *) echo "unknown node $node_tag"; exit 2;;
esac

WL_SH="$ROOT/bench/workloads/${wl}.sh"
[[ -x "$WL_SH" ]] || { echo "workload script not found: $WL_SH"; exit 2; }

# 仅示例：按 profile 注入 UM 选项（默认 um_base 无选项）
um_args=()
if [[ "$profile" == "um_pf" ]]; then um_args+=(--um-pf); fi
if [[ "$profile" == "um_ab" ]]; then um_args+=(--um-ab "gpu$gpu_id"); fi
if [[ "$profile" == "um_pl_cpu" ]]; then um_args+=(--um-pl cpu); fi
if [[ "$profile" == "um_rm" ]]; then um_args+=(--um-rm "gpu$gpu_id"); fi

cmd="$WL_SH run --mode $mode --gpu $gpu_id ${um_args[*]}"
wrapped=$(numa_wrap "$membind" "$cpubind" "$cmd")
ms=$(grab_ms "$wrapped")
echo "${ms:-NA}"


