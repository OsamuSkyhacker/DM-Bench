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
  ${SHELL:-/bin/bash} -lc "$cmd" | grep -Eo '([0-9]+(\.[0-9]+)?) ms' | awk '{print $1}' || true
}

# 运行命令并将全部输出追加到日志，同时抽取 ms 值
run_and_capture_ms() { # <cmd_string> <log_file> [capture_file]
  local cmd="$1"; shift
  local log="$1"; shift || true
  local capture_file="${1:-}"
  debug_log "exec(logged): $cmd -> $log"
  printf "[time] %s\n" "$(date -Ins)" >> "$log"
  printf "[cmd ] %s\n" "$cmd" >> "$log"
  # 执行命令到临时文件，执行完再解析（避免管道与回显干扰）
  local tmp out ms ret lines matchline xtrace_log xtrace_fd_used
  if [[ -n "$capture_file" ]]; then
    tmp="$capture_file"
  else
    tmp="$(mktemp -t bench_out.XXXXXX)"
  fi
  set +e
  xtrace_fd_used=0
  if [[ "${BENCH_TRACE:-0}" == "1" ]]; then
    xtrace_log="${log%.log}.xtrace.log"
    exec 9> "$xtrace_log"
    export BASH_XTRACEFD=9
    xtrace_fd_used=1
    set -x
  fi
  eval "$cmd" >"$tmp" 2>&1
  ret=$?
  if [[ $xtrace_fd_used -eq 1 ]]; then
    set +x
    exec 9>&-
    unset BASH_XTRACEFD
    printf "[xtrace] %s\n" "$xtrace_log" >> "$log"
  fi
  set -e
  out="$(cat "$tmp")"
  printf "%s\n" "$out" >> "$log"
  lines=$(wc -l < "$tmp" | awk '{print $1}')
  printf "[exit ] %s\n" "$ret" >> "$log"
  printf "[lines] %s\n" "$lines" >> "$log"
  # 预览前 5 行
  sed -n '1,5p' "$tmp" | sed 's/^/[out ] /' >> "$log"
  # 匹配到的时间行
  matchline=$(grep -Eo '([0-9]+(\.[0-9]+)?) (ms|s|sec|seconds)' "$tmp" | tail -n1 || true)
  if [[ -n "${matchline:-}" ]]; then printf "[match] %s\n" "$matchline" >> "$log"; fi
  # 提取 ms 数值
  ms=$(grep -Eo '([0-9]+(\.[0-9]+)?) ms' "$tmp" | awk '{print $1}' | tail -n1 || true)
  if [[ -z "${ms:-}" ]]; then
    # 尝试以秒为单位的输出，转换为 ms
    local sec
    sec=$(grep -Eo '([0-9]+(\.[0-9]+)?) (s|sec|seconds)' "$tmp" | awk '{print $1}' | tail -n1 || true)
    if [[ -n "${sec:-}" ]]; then
      ms=$(awk -v s="$sec" 'BEGIN{printf "%.3f", s*1000.0}')
    fi
  fi
  if [[ -n "${ms:-}" ]]; then
    printf "[result] %s ms\n" "$ms" >> "$log"
  else
    printf "[result] NA\n" >> "$log"
  fi
  # 保留临时文件，便于排查（日志中记录路径）
  printf "[tmp ] %s\n" "$tmp" >> "$log"
  echo "$ms"
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


