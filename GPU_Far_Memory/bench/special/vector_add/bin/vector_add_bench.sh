#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"

. "$ROOT/bench/lib/bench_utils.sh"

usage(){ cat <<U
usage: $0 --node <local|remote|pmem> --gpu <id> --ratio <r> [--pf] [--ab] [--pl cpu|gpuX]
U
}

node=local; gpu=0; ratio=1.0; pf=0; ab=0; pl="none"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node) node="$2"; shift 2;;
    --gpu) gpu="$2"; shift 2;;
    --ratio|-r) ratio="$2"; shift 2;;
    --pf) pf=1; shift 1;;
    --ab) ab=1; shift 1;;
    --pl) pl="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown: $1"; exit 2;;
  esac
done

VEC_UM="${WB_ROOT}/vector_add/Managed/vector_add_um"

case "$node" in
  local) membind=0; cpubind=0;;
  remote) membind=1; cpubind=0;;
  pmem) membind=2; cpubind=0;;
  *) echo "unknown node"; exit 2;;
esac

args=("$VEC_UM" -d "$gpu" -r "$ratio")
[[ $pf -eq 1 ]] && args+=(-pf)
[[ $ab -eq 1 ]] && args+=(--advise ab "gpu$gpu")
[[ "$pl" != "none" ]] && args+=(--advise pl "$pl")

cmd="$(numa_wrap "$membind" "$cpubind" "${args[*]}")"
ms=$(grab_ms "$cmd")
echo "${ms:-NA}"


