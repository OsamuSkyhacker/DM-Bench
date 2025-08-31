#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WB_ROOT="${WB_ROOT:-$ROOT/workloads}"
export WB_ROOT

. "$ROOT/bench/lib/bench_utils.sh"
. "$ROOT/bench/lib/hog_ctl.sh"
. "$ROOT/bench/lib/math.sh"

PLAN="$ROOT/bench/configs/plan.yaml"
WY="$ROOT/bench/configs/workloads.yaml"

die() { echo "[error] $*" >&2; exit 2; }
warn() { echo "[warn] $*" >&2; }
parse_scalar() { # <file> <key>
  local file="$1"; local key="$2"; local out=""
  out=$(python3 - "$file" "$key" <<'PY' || true
import sys
try:
  import yaml
except Exception:
  sys.exit(1)
fn, key = sys.argv[1], sys.argv[2]
with open(fn,'r') as f:
  d = yaml.safe_load(f) or {}
v = d.get(key, '')
print(v)
PY
)
  if [[ -n "${out:-}" ]]; then echo "$out"; return 0; fi
  if command -v yq >/dev/null 2>&1; then
    yq -r ".$key" "$file" 2>/dev/null || true
  else
    grep -E "^$key:" "$file" | awk -F: '{gsub(/ /,""); print $2}' | head -n1 || true
  fi
}
parse_list() { # <file> <key>
  local file="$1"; local key="$2"; local out=""
  out=$(python3 - "$file" "$key" <<'PY' || true
import sys
try:
  import yaml
except Exception:
  sys.exit(1)
fn, key = sys.argv[1], sys.argv[2]
with open(fn,'r') as f:
  d = yaml.safe_load(f) or {}
arr = d.get(key) or []
if isinstance(arr, list):
  print(' '.join(str(x) for x in arr))
PY
)
  if [[ -n "${out:-}" ]]; then echo "$out"; return 0; fi
  if command -v yq >/dev/null 2>&1; then
    yq -r ".$key[]" "$file" 2>/dev/null | tr '\n' ' ' || true
  else
    grep -E "^$key:" -A0 "$file" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' ' || true
  fi
}
declare -A nodes_membind nodes_cpubind
node_tags=()
parse_nodes() {
  # Python+PyYAML
  while IFS=$'\t' read -r name membind cpubind; do
    [[ -z "${name:-}" ]] && continue
    nodes_membind["$name"]="${membind:-}"
    nodes_cpubind["$name"]="${cpubind:-}"
    node_tags+=("$name")
  done < <(python3 - <<'PY' "$PLAN" || true
import sys
try:
  import yaml
except Exception:
  sys.exit(1)
fn = sys.argv[1]
with open(fn,'r') as f:
  data = yaml.safe_load(f) or {}
nodes = data.get('nodes') or {}
if isinstance(nodes, dict):
  for name, cfg in nodes.items():
    if isinstance(cfg, dict):
      membind = str(cfg.get('membind','')).strip()
      cpubind = str(cfg.get('cpunodebind','')).strip()
    else:
      membind = ''
      cpubind = ''
    print(f"{name}\t{membind}\t{cpubind}")
PY
  )
  [[ ${#node_tags[@]} -gt 0 ]] && return 0
  # yq
  if command -v yq >/dev/null 2>&1; then
    while IFS=$'\t' read -r name membind cpubind; do
      [[ -z "${name:-}" ]] && continue
      [[ "$name" =~ ^# ]] && continue
      nodes_membind["$name"]="${membind:-}"
      nodes_cpubind["$name"]="${cpubind:-}"
      node_tags+=("$name")
    done < <(yq -r '.nodes | to_entries[] | "\(.key)\t\(.value.membind)\t\(.value.cpunodebind)"' "$PLAN" 2>/dev/null || true)
  fi
  [[ ${#node_tags[@]} -gt 0 ]] && return 0
  # awk fallback: inline only
  while IFS=$'\t' read -r name membind cpubind; do
    [[ -z "${name:-}" ]] && continue
    nodes_membind["$name"]="${membind:-}"
    nodes_cpubind["$name"]="${cpubind:-}"
    node_tags+=("$name")
  done < <(awk '
    BEGIN { inblk=0 }
    /^nodes:/ { inblk=1; next }
    /^[^[:space:]].*:/{ if(inblk){ inblk=0 } }
    { if(inblk) print $0 }
  ' "$PLAN" | sed -E 's/#.*$//' | sed -E 's/^\s+//g' | grep -E '^[^:]+:\s*\{[^}]*\}' | awk -F'[: ,{}]+' '
    {
      name=$1; mb=""; cb="";
      for(i=1;i<=NF;i++){
        if($i=="membind") mb=$(i+1);
        if($i=="cpunodebind") cb=$(i+1);
      }
      if(name!="" && mb!="" && cb!="") printf "%s\t%s\t%s\n", name, mb, cb
    }' || true)
}
um_profile_names=()
declare -A UM_PF UM_AB UM_PL UM_RM
parse_um_profiles() {
  um_profile_names=(); UM_PF=(); UM_AB=(); UM_PL=(); UM_RM=()
  while IFS=$'\t' read -r _name _pf _ab _pl _rm; do
    [[ -z "${_name:-}" ]] && continue
    um_profile_names+=("$_name")
    UM_PF["$_name"]="${_pf:-false}"
    UM_AB["$_name"]="${_ab:-none}"
    UM_PL["$_name"]="${_pl:-none}"
    UM_RM["$_name"]="${_rm:-none}"
  done < <(python3 - <<'PY' "$PLAN" || true
import sys
try:
  import yaml
except Exception:
  sys.exit(1)
fn = sys.argv[1]
with open(fn,'r') as f:
  data = yaml.safe_load(f) or {}
profiles = data.get('um_profiles') or []
for p in profiles:
  name = str(p.get('name','')).strip()
  if not name:
    continue
  pf = p.get('pf', False)
  pf_str = 'true' if (pf is True or str(pf).lower()=='true') else 'false'
  ab = str(p.get('ab','none')).strip() or 'none'
  pl = str(p.get('pl','none')).strip() or 'none'
  rm = str(p.get('rm','none')).strip() or 'none'
  print('\t'.join([name, pf_str, ab, pl, rm]))
PY
  )
  if [[ ${#um_profile_names[@]} -gt 0 ]]; then return 0; fi
  if command -v yq >/dev/null 2>&1; then
    while IFS=$'\t' read -r _name _pf _ab _pl _rm; do
      [[ -z "${_name:-}" ]] && continue
      um_profile_names+=("$_name")
      UM_PF["$_name"]="${_pf:-false}"
      UM_AB["$_name"]="${_ab:-none}"
      UM_PL["$_name"]="${_pl:-none}"
      UM_RM["$_name"]="${_rm:-none}"
    done < <(yq -r '.um_profiles[] | [.name, (.pf // false), (.ab // "none"), (.pl // "none"), (.rm // "none")] | @tsv' "$PLAN" 2>/dev/null || true)
  fi
  if [[ ${#um_profile_names[@]} -gt 0 ]]; then return 0; fi
  # awk 兜底解析（仅支持扁平字段 name/pf/ab/pl/rm）
  while IFS=$'\t' read -r _name _pf _ab _pl _rm; do
    [[ -z "${_name:-}" ]] && continue
    um_profile_names+=("$_name")
    UM_PF["$_name"]="${_pf:-false}"
    UM_AB["$_name"]="${_ab:-none}"
    UM_PL["$_name"]="${_pl:-none}"
    UM_RM["$_name"]="${_rm:-none}"
  done < <(awk '
    BEGIN{inblk=0; have=0; name=""; pf="false"; ab="none"; pl="none"; rm="none"}
    /^um_profiles:/ {inblk=1; next}
    inblk==1 && /^[^[:space:]].*:/{inblk=0}
    inblk==1 {
      gsub(/#.*/ , "");
      if ($0 ~ /^\s*-\s*name:/) {
        if (have==1 && name!="") {printf "%s\t%s\t%s\t%s\t%s\n", name, pf, ab, pl, rm}
        have=1; pf="false"; ab="none"; pl="none"; rm="none";
        sub(/.*name:/, ""); gsub(/^\s+|\s+$/, ""); name=$0; next
      }
      if ($0 ~ /pf:/) {sub(/.*pf:/, ""); gsub(/^\s+|\s+$/, ""); pf=tolower($0); if(pf!="true") pf="false"}
      if ($0 ~ /ab:/) {sub(/.*ab:/, ""); gsub(/^\s+|\s+$/, ""); ab=$0}
      if ($0 ~ /pl:/) {sub(/.*pl:/, ""); gsub(/^\s+|\s+$/, ""); pl=$0}
      if ($0 ~ /rm:/) {sub(/.*rm:/, ""); gsub(/^\s+|\s+$/, ""); rm=$0}
    }
    END{ if (have==1 && name!="") printf "%s\t%s\t%s\t%s\t%s\n", name, pf, ab, pl, rm }
  ' "$PLAN" || true)
}

# 解析 workloads.yaml: 获取 args.input（若存在）
get_wl_input() { # <wy_file> <wl>
  local file="$1"; local wl="$2"; local out=""
  out=$(python3 - "$file" "$wl" <<'PY' || true
import sys
try:
  import yaml
except Exception:
  sys.exit(1)
fn, wl = sys.argv[1], sys.argv[2]
with open(fn,'r') as f:
  d = yaml.safe_load(f) or {}
w = (d.get('workloads') or {}).get(wl) or {}
args = w.get('args') or {}
val = args.get('input','')
if val is None:
  val = ''
print(val)
PY
  )
  if [[ -n "${out:-}" ]]; then echo "$out"; return 0; fi
  if command -v yq >/dev/null 2>&1; then
    yq -r ".workloads.$wl.args.input // \"\"" "$file" 2>/dev/null || true
  else
    grep -A16 -E "^\s*$wl:" "$file" | grep -E "^\s*input:" | sed -E 's/^[^:]+:\s*//' | tr -d '"' | head -n1 || true
  fi
}

# 解析 workloads.yaml: 获取 args.fixed（若存在，拼接为空格分隔一行）
get_wl_fixed() { # <wy_file> <wl>
  local file="$1"; local wl="$2"; local out=""
  out=$(python3 - "$file" "$wl" <<'PY' || true
import sys
try:
  import yaml
except Exception:
  sys.exit(1)
fn, wl = sys.argv[1], sys.argv[2]
with open(fn,'r') as f:
  d = yaml.safe_load(f) or {}
w = (d.get('workloads') or {}).get(wl) or {}
args = w.get('args') or {}
fixed = args.get('fixed') or []
if isinstance(fixed, list):
  print(' '.join(str(x) for x in fixed))
PY
  )
  if [[ -n "${out:-}" ]]; then echo "$out"; return 0; fi
  if command -v yq >/dev/null 2>&1; then
    yq -r ".workloads.$wl.args.fixed[]" "$file" 2>/dev/null | tr '\n' ' ' || true
  else
    # 仅支持简化：单行 [a,b,c]
    local line
    line=$(grep -A16 -E "^\s*$wl:" "$file" | grep -E "^\s*fixed:" | head -n1 || true)
    if [[ -n "${line:-}" ]]; then
      echo "$line" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' ' | tr -d '"' | tr -s ' '
    fi
  fi
}

# 读取 plan 关键字段
gpu_id="$(parse_scalar "$PLAN" "gpu_id")"; gpu_id="${gpu_id:-0}"
read -r -a ratios <<<"$(parse_list "$PLAN" "ratios" | tr -s ' ')"
read -r -a workloads <<<"$(parse_list "$PLAN" "workloads" | tr -s ' ')"
read -r -a modes <<<"$(parse_list "$PLAN" "modes" | tr -s ' ')"

debug_log "gpu_id=$gpu_id"
debug_log "ratios=${ratios[*]:-<empty>}"
debug_log "workloads=${workloads[*]:-<empty>}"
debug_log "modes=${modes[*]:-<empty>}"

parse_nodes
debug_log "node_tags=${node_tags[*]:-<empty>}"
[[ ${#workloads[@]} -eq 0 ]] && die "workloads empty"
[[ ${#ratios[@]} -eq 0 ]] && die "ratios empty"
[[ ${#node_tags[@]} -eq 0 ]] && die "node_tags empty"

# 脚本入口时清理历史残留 hog，并注册退出清理
stop_hog_for_gpu "$gpu_id" || true
trap 'stop_hog_for_gpu "$gpu_id" || true' EXIT

# 解析 modes：决定是否跑 unmanaged 和/或 um
do_unmanaged=0; do_um=0
for m in "${modes[@]}"; do
  if [[ "$m" == "unmanaged" ]]; then do_unmanaged=1; fi
  if [[ "$m" == "um" ]]; then do_um=1; fi
done

um_profile_names=()
declare -A UM_PF UM_AB UM_PL UM_RM
if [[ $do_um -eq 1 ]]; then
  parse_um_profiles || true
fi

for wl in "${workloads[@]}"; do
  # 读取参数
  B=$(grep -A8 "^\s*${wl}:" "$WY" | grep baseline_payload_mb | awk -F: '{gsub(/ /,""); print $2}' | head -n1)
  Over=$(grep -E '^\s*task_overhead_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  GTask=$(grep -E '^\s*global_task_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  FreeAdj=$(grep -E '^\s*free_adjust_mb:' "$WY" | awk -F: '{gsub(/ /,""); print $2}')
  wl_input="$(get_wl_input "$WY" "$wl" | sed 's/#.*$//')"
  wl_fixed="$(get_wl_fixed "$WY" "$wl" | sed 's/#.*$//')"
  B=$(printf "%s" "$B" | sed 's/#.*$//')
  Over=$(printf "%s" "$Over" | sed 's/#.*$//')
  GTask=$(printf "%s" "$GTask" | sed 's/#.*$//')
  FreeAdj=$(printf "%s" "$FreeAdj" | sed 's/#.*$//')
  debug_log "wl=$wl B=$B Over=$Over GTask=$GTask FreeAdj=$FreeAdj input=${wl_input:-<none>} fixed=${wl_fixed:-<none>}"

  ts=$(date +%Y%m%d_%H%M%S)
  outdir="$ROOT/bench/results/$wl/$ts"
  mkdir -p "$outdir"

  # 构建运行配置与 CSV 表头
  run_names=("Oversub")
  if [[ $do_unmanaged -eq 1 ]]; then run_names+=("unmanaged"); fi
  if [[ $do_um -eq 1 && ${#um_profile_names[@]} -gt 0 ]]; then run_names+=("${um_profile_names[@]}"); fi

  for tag in "${node_tags[@]}"; do
    csv="$outdir/runtime_${tag}.csv"
    csv_write_header "$csv" "${run_names[*]}"
  done

  for r in "${ratios[@]}"; do
    # 每个 ratio 前实时读取 Free0
    read -r _ Free0 < <(gpu_mem_info_mb "$gpu_id")
    Free0=$(printf "%s" "$Free0" | sed 's/#.*$//')
    debug_log "Free0=$Free0 (ratio=$r)"
    x=$(calc_hog_gb_from_free0 "$Free0" "$FreeAdj" "$B" "$r" "$Over" "$GTask")
    x_mb=$(awk -v g="$x" 'BEGIN{printf "%d", g*1024.0 + 0.5}')
    debug_log "ratio=$r hog_gb=$x hog_mb=$x_mb"
    HOG_BIN="$WB_ROOT/DummyCudaMalloc/gpu_mem_hog"
    pid=""; if [[ "$x_mb" != "0" ]]; then pid=$(start_hog "$HOG_BIN" "$gpu_id" "$x_mb"); wait_hog_ready "$gpu_id" 5 || true; fi

    for tag in "${node_tags[@]}"; do
      membind="${nodes_membind[$tag]:-}"
      cpubind="${nodes_cpubind[$tag]:-}"
      if [[ -z "$membind" || -z "$cpubind" ]]; then
        echo "[warn] skip node '$tag' (membind/cpunodebind missing)"
        continue
      fi

      values=("$r")
      # unmanaged
      if [[ $do_unmanaged -eq 1 ]]; then
        cmd_un="bash $ROOT/bench/workloads/${wl}.sh run --mode unmanaged --gpu $gpu_id"
        if [[ -n "${wl_input:-}" ]]; then cmd_un+=" --input \"$wl_input\""; fi
        if [[ -n "${wl_fixed:-}" ]]; then cmd_un+=" --fixed-args \"$wl_fixed\""; fi
        log_un="$outdir/${wl}.${tag}.unmanaged.log"
        # 运行元信息
        {
          echo "[time] $(date -Ins)"
          echo "[meta] wl=$wl mode=unmanaged node=$tag membind=$membind cpunodebind=$cpubind gpu_id=$gpu_id"
          echo "[meta] ratio=$r Free0=$Free0 FreeAdj=$FreeAdj B=$B O=$Over GTask=$GTask"
          echo "[meta] hog_gb=$x hog_mb=$x_mb HOG_BIN=$HOG_BIN hog_pid=${pid:-}"
          echo "[meta] input=${wl_input:-} fixed=${wl_fixed:-}"
        } >> "$log_un"
        cap_un="$outdir/tmp.unmanaged.${tag}.txt"
        ms_un=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_un")" "$log_un" "$cap_un")
        values+=("${ms_un:-NA}")
      fi
      # um variants（按 plan.yaml 的 um_profiles 动态执行）
      if [[ $do_um -eq 1 && ${#um_profile_names[@]} -gt 0 ]]; then
        for prof in "${um_profile_names[@]}"; do
          extra_args=""
          # pf
          if [[ "${UM_PF[$prof]:-false}" == "true" ]]; then
            extra_args+=" --um-pf"
          fi
          # ab/pl/rm：支持 cpu|gpuX|gpuN|none，gpuX 用当前 gpu_id 替换
          _ab="${UM_AB[$prof]:-none}"; _pl="${UM_PL[$prof]:-none}"; _rm="${UM_RM[$prof]:-none}"
          case "$_ab" in gpuX) _ab="gpu$gpu_id";; esac
          case "$_pl" in gpuX) _pl="gpu$gpu_id";; esac
          case "$_rm" in gpuX) _rm="gpu$gpu_id";; esac
          if [[ "$_ab" != "none" && -n "$_ab" ]]; then extra_args+=" --um-ab $_ab"; fi
          if [[ "$_pl" != "none" && -n "$_pl" ]]; then extra_args+=" --um-pl $_pl"; fi
          if [[ "$_rm" != "none" && -n "$_rm" ]]; then extra_args+=" --um-rm $_rm"; fi

          cmd_um="bash $ROOT/bench/workloads/${wl}.sh run --mode um --gpu $gpu_id$extra_args"
          if [[ -n "${wl_input:-}" ]]; then cmd_um+=" --input \"$wl_input\""; fi
          if [[ -n "${wl_fixed:-}" ]]; then cmd_um+=" --fixed-args \"$wl_fixed\""; fi
          log_um="$outdir/${wl}.${tag}.${prof}.log"
          {
            echo "[time] $(date -Ins)"
            echo "[meta] wl=$wl mode=um profile=$prof node=$tag membind=$membind cpunodebind=$cpubind gpu_id=$gpu_id"
            echo "[meta] ratio=$r Free0=$Free0 FreeAdj=$FreeAdj B=$B O=$Over GTask=$GTask"
            echo "[meta] hog_gb=$x hog_mb=$x_mb HOG_BIN=$HOG_BIN hog_pid=${pid:-}"
            echo "[meta] pf=${UM_PF[$prof]:-false} ab=$_ab pl=$_pl rm=$_rm"
            echo "[meta] input=${wl_input:-} fixed=${wl_fixed:-}"
          } >> "$log_um"
          cap_um="$outdir/tmp.um.${tag}.${prof}.txt"
          ms_um=$(run_and_capture_ms "$(numa_wrap "$membind" "$cpubind" "$cmd_um")" "$log_um" "$cap_um")
          values+=("${ms_um:-NA}")
        done
      fi

      csv_write_line "$outdir/runtime_${tag}.csv" "${values[*]}"
    done

    [[ -n "$pid" ]] && stop_hog "$pid"
  done
done

echo "done. see $ROOT/bench/results/"


