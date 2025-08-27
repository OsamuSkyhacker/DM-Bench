#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"

echo "[gen ] WB_ROOT=$WB_ROOT"

# Hotspot: run hotspotex / hotspotver to produce temp_8192 / power_8192 (or whatever header selects)
if [[ -x "$WB_ROOT/hotspot/data/inputGen/hotspotex" ]]; then
  ( cd "$WB_ROOT/hotspot/data" && ./inputGen/hotspotex )
else
  echo "[warn] hotspotex not built: $WB_ROOT/hotspot/data/inputGen/hotspotex"
fi

if [[ -x "$WB_ROOT/hotspot/data/inputGen/hotspotver" ]]; then
  ( cd "$WB_ROOT/hotspot/data" && ./inputGen/hotspotver )
else
  echo "[warn] hotspotver not built: $WB_ROOT/hotspot/data/inputGen/hotspotver"
fi

# BFS: build graph64M.txt using graphgen (two-scale params guessed: 65536*1024)
if [[ -x "$WB_ROOT/BFS/bfs_data/inputGen/graphgen" ]]; then
  ( cd "$WB_ROOT/BFS/bfs_data" && ./inputGen/graphgen 67108864 64M )
  echo "[gen ] BFS graph generated: $WB_ROOT/BFS/bfs_data/graph64M.txt"
else
  echo "[warn] graphgen not built: $WB_ROOT/BFS/bfs_data/inputGen/graphgen"
fi

echo "[done] dataset generation script completed"


