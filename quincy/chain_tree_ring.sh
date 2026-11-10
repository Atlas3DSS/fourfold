#!/bin/bash
# #440: the Qwen3.8-27B tree and ring arms on four Flare servers (9460-9463), one after another, each with its eval. fabe, 2026-11-10.
# Order: 4x4 tree smoke (gate) -> 10x10 one tree -> ring smoke on 16 agents (gate) -> tesseract side 4 ring -> 10x10 two trees.
# A smoke that gets fewer than 14/16 exact stops the chain before the large arm it guards. Re-running resumes each run.
set -uo pipefail
cd "$(dirname "$0")/.."
for p in 9460 9461 9462 9463; do until curl -sf --max-time 5 http://127.0.0.1:$p/v1/models >/dev/null; do sleep 30; done; done
echo "== servers up $(date -Is)"
one() { CFG=config.quincy-$2.toml bash quincy/run_one.sh "$1"; }
exact() { .venv/bin/python - "$1" <<'PY'
import json, sys, glob
f = sorted(glob.glob(f"runs/{sys.argv[1]}/eval-t*.json"))
r = json.load(open(f[-1]))["results"] if f else []
print(sum(x.get("sum_score", 0) == 1 for x in r))
PY
}
one q-4x4-torus-sum-tree-smoke-1110 tree-4x4
if [ "$(exact q-4x4-torus-sum-tree-smoke-1110)" -ge 14 ]; then
  one q-10x10-torus-sum-tree-told-1110 tree-10x10
  TREE_OK=1
else
  echo "== tree smoke below 14/16: skipping the 10x10 tree arms"; TREE_OK=0
fi
one q-tesseract2-sum-ring-smoke-1110 ring-tesseract2
if [ "$(exact q-tesseract2-sum-ring-smoke-1110)" -ge 14 ]; then
  one q-tesseract4-sum-ring-1110 ring-tesseract4
else
  echo "== ring smoke below 14/16: skipping the 256-agent tesseract"
fi
[ "$TREE_OK" = 1 ] && one q-10x10-torus-sum-tree-verify-1110 tree2-10x10
echo "== chain done $(date -Is)"
