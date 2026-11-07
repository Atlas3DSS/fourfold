#!/bin/bash
# Run fourfold experiments from an editable queue file, one at a time, each followed by its eval.
# Edit runs/QUEUE while this runs: the next line is read only when the current run ends.
#
#   line formats (blank lines and # comments are skipped):
#     resume <run_id>                       continue a stopped run to its configured tick count
#     <ROWS>x<COLS>[:torus] [@name] [--set k=v ...]   a new run from $CFG, e.g.
#                                            10x10:torus @noprime --set prime=''
#     stop                                   end the runner
#
# finished lines move to runs/QUEUE.done with a timestamp. an empty queue waits for new lines.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-config.quincy.toml}
PY=.venv/bin/python
Q=runs/QUEUE; DONE=runs/QUEUE.done
touch "$Q" "$DONE"
while true; do
  line=$(grep -v -E '^\s*(#|$)' "$Q" | head -1)
  if [ -z "$line" ]; then sleep 60; continue; fi
  # drop that first live line from the queue before running it, so an edit can't re-run it
  awk -v L="$line" 'done || $0 != L {print; next} {done=1}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  echo "$(date -Is) start $line" >> "$DONE"
  [ "$line" = stop ] && { echo "== queue stopped $(date -Is)"; exit 0; }
  $PY quincy/smoke.py "$CFG" --wait "${WAIT_MIN:-120}" --min-up "${MIN_UP:-6}" || { echo "== smoke failed, line put back"; sed -i "1i $line" "$Q"; sleep 300; continue; }
  if [[ $line == resume\ * ]]; then
    id=${line#resume }
    echo "== $id resume $(date -Is)"
    $PY sim.py --resume "$id"
  else
    read -r shape rest <<< "$line"
    dims=${shape%%:*}; topo=grid; [[ $shape == *:torus ]] && topo=torus
    rows=${dims%x*}; cols=${dims#*x}
    tag=""
    if [[ $rest == @* ]]; then read -r tag rest <<< "$rest"; tag=${tag#@}; fi
    id="q-${rows}x${cols}-${topo}${tag:+-$tag}-$(date +%m%d-%H%M)"
    echo "== $id start $(date -Is)  ($line)"
    eval "set -- $rest"
    $PY sim.py --config "$CFG" --run-id "$id" --rows "$rows" --cols "$cols" --topology "$topo" "$@"
  fi
  rc=$?
  echo "== ${id} sim exit $rc, eval $(date -Is)"
  $PY eval.py "runs/$id" || echo "eval failed for $id"
  echo "$(date -Is) done  $line -> $id (sim rc $rc)" >> "$DONE"
  echo "== $id done $(date -Is)"
done
