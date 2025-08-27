#!/usr/bin/env bash
set -euo pipefail

# ============ Developer editable list ============
# Relative to WB_ROOT. Add/remove items as needed.
# Hotspot generated datasets (examples)
HOTSPOT_FILES=(
  "hotspot/data/temp_8192"
  "hotspot/data/power_8192"
  # "hotspot/data/temp_18900"
  # "hotspot/data/power_18900"
)

# BFS generated graphs (examples)
BFS_FILES=(
  "BFS/bfs_data/graph64M.txt"
  # "BFS/bfs_data/graph16M.txt"
)
# ================================================

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"

echo "[clean-datasets] WB_ROOT=$WB_ROOT"

remove_if_exists() {
  local rel="$1"
  local abs="$WB_ROOT/$rel"
  # safety: ensure target is inside WB_ROOT
  case "$abs" in
    "$WB_ROOT"/*) ;;
    *) echo "[skip ] unsafe path: $abs"; return 0;;
  esac
  if [[ -e "$abs" ]]; then
    rm -f "$abs" && echo "[rm   ] $abs" || echo "[fail] $abs" >&2
  else
    echo "[miss] $abs"
  fi
}

for f in "${HOTSPOT_FILES[@]}"; do remove_if_exists "$f"; done
for f in "${BFS_FILES[@]}"; do remove_if_exists "$f"; done

echo "[done] dataset files cleaned (see messages above)"


