#!/bin/bash
# One Flare server that takes its turn in the gpu-workboard's normal queue, for when every card is full.
# (`sb run --gpu` holds a batch job outside the board until a card already has room, so it never gets in line.)
# Run it as a tracked CPU job, it does the board's submit / acquire / start / heartbeat / finish itself:
#   sb run --name fourfold-flare-0 --host quincy --threads 2 --mem 16 --hours 48 -- bash quincy/flare_board.sh 0 16
# args: server index (port 9400+i), hours the server may run once it has its card.
set -u
i=$1; HOURS=${2:-16}
port=$((9400 + i))
D=/mnt/storage16tb/fourfold/flare
WB=$(command -v gpu-workboard || echo "$HOME/.local/bin/gpu-workboard")
field() { python3 -c "import json,sys; d=json.load(sys.stdin); j=d.get('job') or {}; print(d.get('$1') or j.get('$1') or '')"; }

bid=$("$WB" submit --project ripple-test --label "fourfold-flare-$i" --vram-gib 31 --duration-min $((HOURS * 60)) \
      --lane batch --protected --idle-ok --gpu any --json --note "fourfold flare server $i, port $port" | field id)
[ -n "$bid" ] || { echo "submit failed"; exit 1; }
echo "== board job $bid queued $(date -Is)"
pid=""; state=queued
bye() {
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; fi
  if [ "$state" = queued ]; then "$WB" cancel "$bid"; else "$WB" finish "$bid" --outcome released --detail "stopped"; fi
  exit 0
}
trap bye TERM INT

uuid=""
while [ -z "$uuid" ]; do
  out=$("$WB" acquire --json "$bid"); rc=$?
  if [ $rc -eq 0 ]; then
    uuid=$(echo "$out" | field gpu_uuid); [ -n "$uuid" ] || uuid=$(echo "$out" | field assigned_uuid)
    [ -n "$uuid" ] || { echo "lease without a card: $out"; state=leased; bye; }
    state=leased
  elif [ $rc -ne 3 ]; then
    echo "acquire failed (rc $rc): $out"
  fi
  [ -n "$uuid" ] || { sleep 30 & wait $!; }
done
echo "== card $uuid $(date -Is)"

cd "$D" || { "$WB" finish "$bid" --outcome failed --detail "no $D"; exit 1; }
CUDA_VISIBLE_DEVICES=$uuid timeout "${HOURS}h" "$D/flare" run "$D/pkg/model.safetensors" \
  --publisher-keys "$D/keys/development-publisher.json" --trust-store "$D/keys/trust.toml" \
  --verify-weights --gpus 0 --memory-limit 31G \
  --max-concurrency 4 --kv-cache-limit concurrency --mtp-draft-tokens 3 --no-mtp-adaptive-verification \
  --prefix-cache-memory 6G --embedding-table-storage ram --scheduler-policy balanced \
  --served-model-name qwen38-27b --host 127.0.0.1 --port "$port" &
pid=$!
"$WB" start "$bid" --pid "$pid" ${SB_UNIT:+--unit "$SB_UNIT"}
while kill -0 "$pid" 2>/dev/null; do
  "$WB" heartbeat "$bid" --pid "$pid" >/dev/null || echo "heartbeat failed $(date -Is)"
  sleep 30 & wait $!
done
wait "$pid"; rc=$?
if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then "$WB" finish "$bid" --outcome completed --detail "server ended (rc $rc)"
else "$WB" finish "$bid" --outcome failed --detail "flare exited rc $rc"; fi
