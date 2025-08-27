#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"
MAKE_JOBS="${MAKE_JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"

echo "[build] WB_ROOT=$WB_ROOT"

declare -a FAILS=()

build_dir() { # <dir>
  local d="$1"
  if [[ -f "$d/Makefile" ]]; then
    echo "[build] make -C $d -j$MAKE_JOBS"
    make -C "$d" -j"$MAKE_JOBS" || FAILS+=("$d")
  else
    echo "[skip ] no Makefile: $d"
  fi
}

# Backprop
build_dir "$WB_ROOT/backprop/Managed"
build_dir "$WB_ROOT/backprop/UnManaged"

# BFS (+ generator)
build_dir "$WB_ROOT/BFS/Managed"
build_dir "$WB_ROOT/BFS/UnManaged"
build_dir "$WB_ROOT/BFS/bfs_data/inputGen"

# FDTD-2D
build_dir "$WB_ROOT/FDTD-2D/Managed"
build_dir "$WB_ROOT/FDTD-2D/UnManaged"

# hotspot (+ generator)
build_dir "$WB_ROOT/hotspot/Managed"
build_dir "$WB_ROOT/hotspot/UnManaged"
build_dir "$WB_ROOT/hotspot/data/inputGen"

# nw
build_dir "$WB_ROOT/nw/Managed"
build_dir "$WB_ROOT/nw/UnManaged"

# pathfinder
build_dir "$WB_ROOT/pathfinder/Managed"
build_dir "$WB_ROOT/pathfinder/UnManaged"

# DummyCudaMalloc (standalone)
build_dir "$WB_ROOT/DummyCudaMalloc"

# vector_add is intentionally skipped (legacy/special)
echo "[skip ] vector_add (legacy/special)"

if (( ${#FAILS[@]} )); then
  echo "[done ] with failures:" >&2
  printf '  - %s\n' "${FAILS[@]}" >&2
  exit 1
fi
echo "[done ] all workloads built successfully"


