#!/usr/bin/env bash
set -euo pipefail

start_hog() { # <hog_bin> <gpu_id> <hog_mb>
  local hog_bin="$1"; shift
  local gpu_id="$1"; shift
  local hog_mb="$1"
  local hog_gb
  hog_gb=$(awk -v m="$hog_mb" 'BEGIN{printf "%.3f", m/1024.0}')
  # PCI_BUS_ID 保证 hog 与负载（CUDA_VISIBLE_DEVICES）、nvidia-smi 使用同一套 GPU 编号
  nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID "$hog_bin" "$hog_gb" "$gpu_id" >"/tmp/hog.${gpu_id}.log" 2>&1 &
  echo $!
}

stop_hog() { # <pid>
  local pid="$1"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3; do
    if kill -0 "$pid" 2>/dev/null; then sleep 0.3; else return 0; fi
  done
  kill -KILL "$pid" 2>/dev/null || true
}

stop_hog_for_gpu() { # <gpu_id>
  local gpu_id="$1"
  local pids
  pids=$(pgrep -af "gpu_mem_hog" | awk -v gid="$gpu_id" '$NF==gid {print $1}')
  for pid in $pids; do
    stop_hog "$pid"
  done
}

wait_hog_ready() { # <gpu_id> <timeout_sec>
  local gpu_id="$1"; shift
  local timeout="${1:-5}"
  local log="/tmp/hog.${gpu_id}.log"
  local t=0
  while [[ $t -lt $timeout ]]; do
    if [[ -f "$log" ]]; then
      if grep -q "Actually allocated" "$log"; then
        return 0
      fi
      # hog 分配不足/设备错误时会打印失败信息并退出，不能视为就绪
      if grep -Eq "Allocation failed|cudaSetDevice.*failed" "$log"; then
        return 1
      fi
    fi
    sleep 1; t=$((t+1))
  done
  return 1
}


