#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"

echo "[clean] WB_ROOT=$WB_ROOT"

declare -a FAILS=()

clean_dir() { # <dir>
  local d="$1"
  if [[ -f "$d/Makefile" ]]; then
    echo "[clean] make -C $d clean"
    make -C "$d" clean || FAILS+=("$d")
  else
    echo "[skip ] no Makefile: $d"
  fi
}

# Backprop
clean_dir "$WB_ROOT/backprop/Managed"
clean_dir "$WB_ROOT/backprop/UnManaged"

# BFS (+ generator)
clean_dir "$WB_ROOT/BFS/Managed"
clean_dir "$WB_ROOT/BFS/UnManaged"
clean_dir "$WB_ROOT/BFS/bfs_data/inputGen"

# FDTD-2D
clean_dir "$WB_ROOT/FDTD-2D/Managed"
clean_dir "$WB_ROOT/FDTD-2D/UnManaged"

# hotspot (+ generator)
clean_dir "$WB_ROOT/hotspot/Managed"
clean_dir "$WB_ROOT/hotspot/UnManaged"
clean_dir "$WB_ROOT/hotspot/data/inputGen"

# nw
clean_dir "$WB_ROOT/nw/Managed"
clean_dir "$WB_ROOT/nw/UnManaged"

# pathfinder
clean_dir "$WB_ROOT/pathfinder/Managed"
clean_dir "$WB_ROOT/pathfinder/UnManaged"

# DummyCudaMalloc (standalone)
clean_dir "$WB_ROOT/DummyCudaMalloc"

# vector_add is intentionally skipped (legacy/special)
echo "[skip ] vector_add (legacy/special)"

if (( ${#FAILS[@]} )); then
  echo "[done ] with failures:" >&2
  printf '  - %s\n' "${FAILS[@]}" >&2
  exit 1
fi
echo "[done ] all workloads cleaned successfully"


