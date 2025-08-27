#!/usr/bin/env bash
set -euo pipefail

# pathfinder 统一接口封装
# 用法：run --mode <unmanaged|um> --gpu <id> --fixed-args "<rows> <cols> <pyr_height>" [--um-*]

usage(){ echo "usage: $0 run --mode <unmanaged|um> --gpu <id> --fixed-args \"<rows> <cols> <pyr>\" [--um-*]"; }

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
  local exe cmd; read -r rows cols pyr <<<"$fixed_args"
  if [[ "$mode" == "unmanaged" ]]; then
    exe="${WB_ROOT}/pathfinder/UnManaged/pathfinder"
    cmd=("$exe" "$cols" "$rows" "$pyr")
  elif [[ "$mode" == "um" ]]; then
    exe="${WB_ROOT}/pathfinder/Managed/pathfinder"
    cmd=("$exe" "$cols" "$rows" "$pyr")
    [[ "$ab" != "none" ]] && cmd+=(AB "${ab#gpu}")
    [[ "$pl" != "none" ]] && cmd+=(PL "$pl")
    [[ "$rm" != "none" ]] && cmd+=(RM "${rm#gpu}")
    [[ $pf -eq 1 ]] && cmd+=(PF "${gpu}")
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


