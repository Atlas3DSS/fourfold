#!/bin/bash
# One fourfold run plus its eval, outside the queue (for a run that shares the servers with the queue's run).
#   bash quincy/run_one.sh RUN_ID [sim.py args...]      e.g. --rows 10 --cols 10 --topology torus --set prime="..."
# re-running with the same RUN_ID resumes it.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-config.quincy.toml}
id=$1; shift
if [ -f "runs/$id/ticks.jsonl" ]; then
  echo "== $id resume $(date -Is)"; .venv/bin/python sim.py --resume "$id"
else
  echo "== $id start $(date -Is)"; .venv/bin/python sim.py --config "$CFG" --run-id "$id" "$@"
fi
echo "== $id eval $(date -Is)"
.venv/bin/python eval.py "runs/$id" || echo "eval failed for $id"
echo "== $id done $(date -Is)"
