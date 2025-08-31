#!/usr/bin/env bash
set -euo pipefail

# 路径与环境兜底
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"
export WB_ROOT

# backprop 真实 CLI 封装
# 用法同一：run --mode <unmanaged|um> --gpu <id> [--fixed-args "IN HID"] [--um-*]

usage() { echo "usage: $0 run --mode <unmanaged|um> --gpu <id> [--fixed-args \"IN HID\"] [--um-*]"; }

run() {
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
  [[ -z "$mode" || -z "$gpu" ]] && { usage; exit 2; }

  # 默认固定参数（与 configs 对齐）：IN HID
  if [[ -z "$fixed_args" ]]; then
    fixed_args="1048560 1280"
  fi

  local exe
  if [[ "$mode" == "unmanaged" ]]; then
    exe="${WB_ROOT}/backprop/UnManaged/backprop"
    # backprop UnManaged 二进制接受两个位置参数：in hid
    read -r in hid <<<"$fixed_args"
    cmd=("$exe" "$in" "$hid")
  elif [[ "$mode" == "um" ]]; then
    exe="${WB_ROOT}/backprop/Managed/backprop"
    read -r in hid <<<"$fixed_args"
    cmd=("$exe" "$in" "$hid")
    [[ "$ab" != "none" ]] && cmd+=(AB "${ab#gpu}")
    [[ "$pl" != "none" ]] && cmd+=(PL "$pl")
    [[ "$rm" != "none" ]] && cmd+=(RM "${rm#gpu}")
    [[ $pf -eq 1 ]] && cmd+=(PF "${gpu}")
    # 多流不支持，已移除
  else
    echo "unsupported"; exit 3
  fi

  out=$("${cmd[@]}" 2>&1 || true)
  # 统一解析：匹配 "X[.Y] (ms|s|sec|seconds)"，转换为毫秒
  match=$(printf "%s\n" "$out" | grep -Eo '([0-9]+(\.[0-9]+)?) (ms|s|sec|seconds)' | tail -n1 || true)
  if [[ -n "${match:-}" ]]; then
    val=$(printf "%s" "$match" | awk '{print $1}')
    unit=$(printf "%s" "$match" | awk '{print $2}')
    if [[ "$unit" == "ms" ]]; then
      awk -v v="$val" 'BEGIN{printf "%.3f ms\n", v+0.0}'
    else
      awk -v v="$val" 'BEGIN{printf "%.3f ms\n", v*1000.0}'
    fi
  fi
}

case "${1:-}" in
  run) shift; run "$@";;
  *) usage; exit 2;;
esac


