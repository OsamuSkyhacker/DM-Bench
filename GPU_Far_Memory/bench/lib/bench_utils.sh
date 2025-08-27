#!/usr/bin/env bash
set -euo pipefail

# numactl 包装（若系统无 numactl 则直接返回原命令）
numa_wrap() {
  local membind="$1"; shift
  local cpubind="$1"; shift
  local cmd="$*"
  if command -v numactl >/dev/null 2>&1; then
    echo "numactl --membind=${membind} --cpunodebind=${cpubind} ${cmd}"
  else
    echo "$cmd"
  fi
}

# 简单调试输出
debug_log() {
  if [[ "${BENCH_DEBUG:-0}" == "1" ]]; then
    echo "[debug] $*" >&2
  fi
}

# 抓取 "xx.xx ms"，失败返回空；不屏蔽 stderr，便于调试
grab_ms() {
  local cmd="$1"
  debug_log "exec: $cmd"
  ${SHELL:-/bin/bash} -lc "$cmd" | grep -Eo '[0-9]+\.[0-9]+ ms' | awk '{print $1}' || true
}

# 运行命令并将全部输出追加到日志，同时抽取 ms 值
run_and_capture_ms() { # <cmd_string> <log_file>
  local cmd="$1"; shift
  local log="$1"
  debug_log "exec(logged): $cmd -> $log"
  ${SHELL:-/bin/bash} -lc "$cmd" 2>&1 | tee -a "$log" | grep -Eo '[0-9]+\.[0-9]+ ms' | awk '{print $1}' || true
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


