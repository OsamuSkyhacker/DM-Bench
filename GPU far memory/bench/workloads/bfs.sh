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

  # 执行并将秒转毫秒打印一行 "xxx.xxx ms" 供上层抓取
  out=$("${cmd[@]}" 2>/dev/null || true)
  sec=$(printf "%s\n" "$out" | grep -Eo 'Total elapsed time: [0-9]+\.[0-9]+ s' | awk '{print $4}')
  if [[ -n "$sec" ]]; then
    awk -v s="$sec" 'BEGIN{printf "%.3f ms\n", s*1000.0}'
  else
    # 若程序自身已输出 ms，则透传第一处
    printf "%s\n" "$out" | grep -Eo '[0-9]+\.[0-9]+ ms' | head -n1 || true
  fi
}

case "${1:-}" in
  run) shift; run "$@";;
  *) usage; exit 2;;
esac


