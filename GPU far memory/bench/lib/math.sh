#!/usr/bin/env bash
set -euo pipefail

# 基于实时 Free0 的闭式解（支持修正项 free_adjust_mb）：
# F_task + (B_algo/R + O) + (H_bytes + O) = (Free0 - FreeAdj)
# 其中 H_bytes = 1024 * x_GB ⇒ x_GB = (Free0 - FreeAdj - F_task - (B_algo/R + O) - O) / 1024
# 返回 x_GB（保留三位小数，且不小于 0）
calc_hog_gb_from_free0() { # Free0_mb FreeAdj_mb B_algo_mb R task_overhead_mb global_task_mb
  awk -v F0="$1" -v FA="$2" -v B="$3" -v R="$4" -v O="$5" -v FT="$6" '
    BEGIN{
      num = (F0 - FA) - FT - (B / R + O) - O;
      x = num / 1024.0;
      if (x < 0) x = 0;
      printf "%.3f\n", x;
    }'
}


