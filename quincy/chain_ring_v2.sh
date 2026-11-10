#!/bin/bash
# #440 ring arm, second attempt (host-kept ring results + host-checked ring writes). Waits for the first chain's last arm
# (the 27B two-tree 10x10) to have its eval, then: 16-agent ring smoke v2 (gate 14/16) -> tesseract side 4 v2. fabe, 2026-11-10.
set -uo pipefail
cd "$(dirname "$0")/.."
until ls runs/q-10x10-torus-sum-tree-verify-1110/eval-t*.json >/dev/null 2>&1; do sleep 60; done
for p in 9460 9461 9462 9463; do until curl -sf --max-time 5 http://127.0.0.1:$p/v1/models >/dev/null; do sleep 30; done; done
one() { CFG=config.quincy-$2.toml bash quincy/run_one.sh "$1"; }
exact() { .venv/bin/python - "$1" <<'PY'
import json, sys, glob
f = sorted(glob.glob(f"runs/{sys.argv[1]}/eval-t*.json"))
r = json.load(open(f[-1]))["results"] if f else []
print(sum(x.get("sum_score", 0) == 1 for x in r))
PY
}
one q-tesseract2-sum-ring-smoke-v2-1110 ring-tesseract2-v2
if [ "$(exact q-tesseract2-sum-ring-smoke-v2-1110)" -ge 14 ]; then
  one q-tesseract4-sum-ring-v2-1110 ring-tesseract4-v2
else
  echo "== ring smoke v2 below 14/16: skipping the 256-agent tesseract"
fi
echo "== ring v2 chain done $(date -Is)"
