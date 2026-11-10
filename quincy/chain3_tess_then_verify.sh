#!/bin/bash
# #440, 2026-11-10: the 27B ring fix first. The 256-agent tesseract with host-checked ring writes (its 16-agent smoke passed
# 16/16), then the paused two-tree 10x10 resumes from its last tick. fabe.
set -uo pipefail
cd "$(dirname "$0")/.."
for p in 9460 9461 9462 9463; do until curl -sf --max-time 5 http://127.0.0.1:$p/v1/models >/dev/null; do sleep 30; done; done
CFG=config.quincy-ring-tesseract4-v2.toml bash quincy/run_one.sh q-tesseract4-sum-ring-v2-1110
CFG=config.quincy-tree2-10x10.toml bash quincy/run_one.sh q-10x10-torus-sum-tree-verify-1110
echo "== chain3 done $(date -Is)"
