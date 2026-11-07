#!/bin/bash
# Size ladder: one fourfold run per shape, one after another, each followed by its eval.
# Shapes are ROWSxCOLS or ROWSxCOLS:torus. The slot count is fixed by the servers in the
# config, so a grid bigger than the slots just takes longer per tick.
#
#   CFG=config.quincy.toml TICKS=100 bash quincy/ladder.sh 8x8 10x10 12x12 12x12:torus
#
# runs are named ladder-<shape>-<stamp>; a stopped one resumes with
#   .venv/bin/python sim.py --resume <run_id>
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-config.quincy.toml}
TICKS=${TICKS:-100}
PY=.venv/bin/python
STAMP=$(date +%Y%m%d-%H%M)
$PY quincy/smoke.py "$CFG" --wait "${WAIT_MIN:-60}" --full --min-up "${MIN_UP:-1}"
for shape in "$@"; do
  dims=${shape%%:*}; topo=grid; [[ $shape == *:torus ]] && topo=torus
  rows=${dims%x*}; cols=${dims#*x}
  id="ladder-${rows}x${cols}-${topo}-$STAMP"
  echo "== $id start $(date -Is)"
  $PY sim.py --config "$CFG" --run-id "$id" --rows "$rows" --cols "$cols" --topology "$topo" --ticks "$TICKS"
  echo "== $id eval $(date -Is)"
  $PY eval.py "runs/$id" || echo "eval failed for $id"
  echo "== $id done $(date -Is)"
done
