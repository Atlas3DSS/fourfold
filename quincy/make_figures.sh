#!/bin/bash
# erz0r-style figures for one run: replay.gif (first 50 ticks) and map.png (needs an eval), into the run dir.
#   bash quincy/make_figures.sh runs/<run>
set -euo pipefail
cd "$(dirname "$0")/.."
run=${1%/}
.venv/bin/python -I experiments/quincy/render_replay.py "$run" "$run/replay.gif"
if ls "$run"/eval-t*.json >/dev/null 2>&1; then
  .venv/bin/python -I experiments/quincy/render_map.py "$run" "$run/map.png"
else
  echo "no eval yet: skipped map.png"
fi
