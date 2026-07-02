#!/usr/bin/env bash
set -euo pipefail

# 路径与环境兜底
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"
export WB_ROOT

# pathfinder 统一接口封装
# 用法：run --mode <unmanaged|um> --gpu <id> --fixed-args "<rows> <cols> <pyr_height>" [--um-*]

usage(){ echo "usage: $0 run --mode <unmanaged|um> --gpu <id> --fixed-args \"<rows> <cols> <pyr>\" [--um-*]"; }

# 将 UM 目标设备 (cpu|gpuN) 映射为程序内设备号。
# 负载进程经 CUDA_VISIBLE_DEVICES 绑定到单卡后，程序内可见的 GPU 编号恒为 0；
# 指向其他物理卡的 gpuN 不受支持，回退为当前卡并告警。
map_dev() {
  case "$1" in
    cpu) echo cpu ;;
    "gpu${gpu}") echo 0 ;;
    *) echo "[warn] um target '$1' != gpu${gpu}, mapping to current GPU" >&2; echo 0 ;;
  esac
}

run(){
  local mode="" gpu="" fixed_args=""
  local ab="none" pl="none" rm="none" pf=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode) mode="$2"; shift 2;;
      --gpu) gpu="$2"; shift 2;;
      --fixed-args) fixed_args="$2"; shift 2;;
      --um-ab) ab="$2"; shift 2;;
      --um-pl) pl="$2"; shift 2;;
      --um-rm) rm="$2"; shift 2;;
      --um-pf) pf=1; shift 1;;
      *) echo "unknown arg: $1"; exit 2;;
    esac
  done
  [[ -z "$mode" || -z "$gpu" || -z "$fixed_args" ]] && { usage; exit 2; }

  # 将负载绑定到目标 GPU（编号与 nvidia-smi 一致），绑定后程序内设备号恒为 0
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${gpu}"

  local exe cmd; read -r rows cols pyr <<<"$fixed_args"
  if [[ "$mode" == "unmanaged" ]]; then
    exe="${WB_ROOT}/pathfinder/UnManaged/pathfinder"
    cmd=("$exe" "$cols" "$rows" "$pyr")
  elif [[ "$mode" == "um" ]]; then
    exe="${WB_ROOT}/pathfinder/Managed/pathfinder"
    cmd=("$exe" "$cols" "$rows" "$pyr")
    [[ "$ab" != "none" ]] && cmd+=(AB "$(map_dev "$ab")")
    [[ "$pl" != "none" ]] && cmd+=(PL "$(map_dev "$pl")")
    [[ "$rm" != "none" ]] && cmd+=(RM "$(map_dev "$rm")")
    [[ $pf -eq 1 ]] && cmd+=(PF 0)
  else
    echo "unsupported"; exit 3
  fi

  out=$("${cmd[@]}" 2>/dev/null || true)
  sec=$(printf "%s\n" "$out" | grep -Eo 'Total elapsed time: [0-9]+\.[0-9]+ s' | awk '{print $4}')
  if [[ -n "$sec" ]]; then
    awk -v s="$sec" 'BEGIN{printf "%.3f ms\n", s*1000.0}'
  else
    printf "%s\n" "$out" | grep -Eo '[0-9]+\.[0-9]+ ms' | head -n1 || true
  fi
}

case "${1:-}" in
  run) shift; run "$@";;
  *) usage; exit 2;;
esac


