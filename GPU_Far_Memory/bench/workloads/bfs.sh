#!/usr/bin/env bash
set -euo pipefail

# 统一外部接口：
#   bfs.sh run --mode <unmanaged|um> --gpu <id> [--input <path>] \
#              [--um-ab <gpuX|cpu|none>] [--um-pl <gpuX|cpu|none>] \
#              [--um-rm <gpuX|cpu|none>] [--um-pf]

usage() { echo "usage: $0 run --mode <unmanaged|um> --gpu <id> [--input <path>] [--um-...]"; }

run() {
  local mode="" gpu="" input=""; local ab="none" pl="none" rm="none" pf=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode) mode="$2"; shift 2;;
      --gpu) gpu="$2"; shift 2;;
      --input) input="$2"; shift 2;;
      --um-ab) ab="$2"; shift 2;;
      --um-pl) pl="$2"; shift 2;;
      --um-rm) rm="$2"; shift 2;;
      --um-pf) pf=1; shift 1;;
      *) echo "unknown arg: $1"; exit 2;;
    esac
  done
  [[ -z "$mode" || -z "$gpu" ]] && { usage; exit 2; }

  local exe
  if [[ "$mode" == "unmanaged" ]]; then
    exe="${WB_ROOT}/BFS/UnManaged/bfs"
    [[ -z "$input" ]] && input="${WB_ROOT}/BFS/bfs_data/graph16M.txt"
    cmd=("$exe" "$input" DEV "$gpu")
  elif [[ "$mode" == "um" ]]; then
    exe="${WB_ROOT}/BFS/Managed/bfs"
    [[ -z "$input" ]] && input="${WB_ROOT}/BFS/bfs_data/graph16M.txt"
    cmd=("$exe" "$input" DEV "$gpu")
    [[ "$ab" != "none" ]] && cmd+=(AB "${ab#gpu}")
    [[ "$pl" != "none" ]] && cmd+=(PL "$pl")
    [[ "$rm" != "none" ]] && cmd+=(RM "${rm#gpu}")
    [[ $pf -eq 1 ]] && cmd+=(PF "${gpu}")
    # streams 为编译期开关，这里忽略
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


