#!/bin/bash
# Keep a gpu-workboard lease of ours alive while something else uses the card, so handing it over does not lose it
# to the queue. Heartbeats with this script's pid, which the board accepts for a non-preemptible lease.
#   sb run --name fourfold-lease-hold-5 --host quincy --threads 1 --mem 1 --hours 24 -- bash quincy/lease_hold.sh BOARD_ID
# Handover: start this; `kill -STOP` that server's flare_board.sh (so its exit trap does not finish the lease); kill its
# flare process. The borrower runs with CUDA_VISIBLE_DEVICES=<card>. To give the card back to the board: stop this job,
# then `sb stop` the old fourfold-flare job (systemd continues the wrapper and its trap finishes the lease).
set -u
bid=$1
WB=$(command -v gpu-workboard || echo "$HOME/.local/bin/gpu-workboard")
trap 'exit 0' TERM INT
"$WB" start "$bid" --pid $$ || { echo "start refused"; exit 1; }
echo "== holding $bid as pid $$ $(date -Is)"
while true; do
  "$WB" heartbeat "$bid" --pid $$ >/dev/null || echo "heartbeat failed $(date -Is)"
  sleep 30 & wait $!
done
