#!/usr/bin/env bash
set -euo pipefail

# 按 B/R 口径计算需要占用的显存（MB），并按当前空闲裁剪
# calc_hog_mb T_mb B_mb R Sctx_mb Delta_mb Free0_mb Sr_mb
calc_hog_mb() {
  awk -v T="$1" -v B="$2" -v R="$3" -v S="$4" -v D="$5" -v F0="$6" -v Sr="$7" '
    BEGIN{
      free_target = B / R + S + D;
      H = T - free_target; if (H < 0) H = 0;
      maxH = F0 - Sr; if (maxH < 0) maxH = 0;
      if (H > maxH) H = maxH;
      printf "%.0f\n", H;
    }'
}


