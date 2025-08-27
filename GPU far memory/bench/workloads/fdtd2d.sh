#!/usr/bin/env bash
set -euo pipefail

# fdtd2d 统一接口封装
# 用法：run --mode <unmanaged|um> --gpu <id> [--um-ab <gpuX|cpu|none>] [--um-pl <gpuX|cpu|none>] [--um-rm <gpuX|cpu|none>] [--um-pf]

usage(){ echo "usage: $0 run --mode <unmanaged|um> --gpu <id> [--um-*]"; }

run(){
  local mode="" gpu=""; local ab="none" pl="none" rm="none" pf=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode) mode="$2"; shift 2;;
      --gpu) gpu="$2"; shift 2;;
      --um-ab) ab="$2"; shift 2;;
      --um-pl) pl="$2"; shift 2;;
      --um-rm) rm="$2"; shift 2;;
      --um-pf) pf=1; shift 1;;
      *) echo "unknown arg: $1"; exit 2;;
    esac
  done
  [[ -z "$mode" || -z "$gpu" ]] && { usage; exit 2; }

  local exe cmd
  if [[ "$mode" == "unmanaged" ]]; then
    exe="${WB_ROOT}/FDTD-2D/UnManaged/fdtd2d"
    cmd=("$exe")
  elif [[ "$mode" == "um" ]]; then
    exe="${WB_ROOT}/FDTD-2D/Managed/fdtd2d"
    cmd=("$exe")
    [[ "$ab" != "none" ]] && cmd+=(AB "${ab#gpu}")
    [[ "$pl" != "none" ]] && cmd+=(PL "$pl")
    [[ "$rm" != "none" ]] && cmd+=(RM "${rm#gpu}")
    [[ $pf -eq 1 ]] && cmd+=(PF "${gpu}")
  else
    echo "unsupported"; exit 3
  fi

  out=("$(${cmd[@]} 2>&1 || true)")
  match=$(printf "%s\n" "${out[@]}" | grep -Eo '([0-9]+(\.[0-9]+)?) (ms|s|sec|seconds)' | head -n1 || true)
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


