#!/usr/bin/env bash
set -euo pipefail

# 轻量封装：优先用 yq，兜底用 python -c

yaml_get() { # <file> <jq_like_path>
  local file="$1"; shift
  local path="$*"
  if command -v yq >/dev/null 2>&1; then
    yq -r "$path" "$file"
  else
    python3 - <<PY "$file" "$path"
import sys, json
try:
  import yaml
except Exception:
  print('')
  sys.exit(0)
fn, path = sys.argv[1], sys.argv[2]
with open(fn,'r') as f:
  data = yaml.safe_load(f)
def get(d, keys):
  cur = d
  for k in keys:
    if isinstance(cur, list):
      i = int(k)
      cur = cur[i]
    else:
      cur = cur.get(k, None)
      if cur is None:
        return ''
  return cur
keys = [k for k in path.replace('.',' ').split() if k]
val = get(data, keys)
if isinstance(val, (dict,list)):
  print(json.dumps(val))
else:
  print(val if val is not None else '')
PY
  fi
}


