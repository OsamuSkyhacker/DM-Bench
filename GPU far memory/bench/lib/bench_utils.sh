#!/usr/bin/env bash
set -euo pipefail

# numactl 包装
numa_wrap() {
  local membind="$1"; shift
  local cpubind="$1"; shift
  local cmd="$*"
  echo "numactl --membind=${membind} --cpunodebind=${cpubind} ${cmd}"
}

# 抓取 "xx.xx ms"，失败返回空
grab_ms() {
  local cmd="$1"
  ${SHELL:-/bin/bash} -lc "$cmd" 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+ ms' | awk '{print $1}' || true
}

# 简单 CSV 写入
csv_write_header() { # <csv_path> <header_line>
  local csv="$1"; shift; echo "$*" > "$csv"
}
csv_write_line() { # <csv_path> <line>
  local csv="$1"; shift; echo "$*" >> "$csv"
}

# 查询 GPU total/free（MB），优先 nvidia-smi
gpu_mem_info_mb() {
  local gpu_id="${1:-0}"
  local q
  q=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || true)
  if [[ -n "$q" ]]; then
    echo "$q" | awk -F',' '{gsub(/ /,""); printf "%d %d\n", $1, $2}'
    return 0
  fi
  # 兜底：返回 0 0
  echo "0 0"
}


