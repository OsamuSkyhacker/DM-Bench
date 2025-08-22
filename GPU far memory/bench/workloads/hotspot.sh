#!/usr/bin/env bash
set -euo pipefail

# hotspot 统一接口封装
# 用法：run --mode <unmanaged|um> --gpu <id> --fixed-args "<rows_cols> <pyr_height> <sim_time> <temp_file> <power_file> <out_file>" [--um-*]

usage(){ echo "usage: $0 run --mode <unmanaged|um> --gpu <id> --fixed-args \"<rows_cols> <pyr> <sim> <temp> <power> <out>\" [--um-*]"; }

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

  local exe cmd; read -r rows_cols pyr sim tfile pfile ofile <<<"$fixed_args"
  if [[ "$mode" == "unmanaged" ]]; then
    exe="${WB_ROOT}/hotspot/UnManaged/hotspot"
    cmd=("$exe" "$rows_cols" "$pyr" "$sim" "$tfile" "$pfile" "$ofile")
  elif [[ "$mode" == "um" ]]; then
    exe="${WB_ROOT}/hotspot/Managed/hotspot"
    cmd=("$exe" "$rows_cols" "$pyr" "$sim" "$tfile" "$pfile" "$ofile")
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


