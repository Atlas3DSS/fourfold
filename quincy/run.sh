#!/bin/bash
# The full fourfold run on Quincy, meant to run as an `sb run` job on Quincy (CPU only,
# it talks to the Flare servers on 127.0.0.1:9400-9406):
#
#   sb run --name fourfold-quincy-run --host quincy --threads 2 --mem 4 --hours 10 \
#     --cwd /mnt/storage16tb/fourfold/fourfold -- bash quincy/run.sh
#
# 1. smoke: waits for all endpoints, then fills all 28 slots once; aborts on any failure
# 2. sim:   100 ticks, 4x7 grid, resumable (python sim.py --resume <run_id>)
# 3. eval:  final questions to all 28 agents, scored against the true wiring
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-config.quincy.toml}
RUN_ID=${RUN_ID:-quincy-$(date +%Y%m%d-%H%M%S)}
PY=.venv/bin/python
echo "== smoke $(date -Is)"
$PY quincy/smoke.py "$CFG" --wait "${WAIT_MIN:-480}" --full
echo "== sim $RUN_ID $(date -Is)"
if [ -f "runs/$RUN_ID/ticks.jsonl" ]; then
  $PY sim.py --resume "$RUN_ID"
else
  $PY sim.py --config "$CFG" --run-id "$RUN_ID"
fi
echo "== eval $(date -Is)"
$PY eval.py "runs/$RUN_ID"
echo "== done $(date -Is)  results in runs/$RUN_ID"
