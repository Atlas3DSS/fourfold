#!/bin/bash
# Queue one Flare server per Quincy V100 through Switchboard + gpu-workboard.
# Each is a Qwen3.8-27B sm_70 package with 32k context and 4 slots: 7 x 4 = 28 slots.
# The board assigns the cards (CUDA_VISIBLE_DEVICES), so a job waits in the queue
# until a card frees up; it never preempts a running lease.
#
#   quincy/flare_up.sh            # queue all 7 (ports 9400-9406)
#   quincy/flare_up.sh 3          # queue only the first 3
#   HOURS=12 quincy/flare_up.sh
#
# stop:  sb jobs   then   sb stop <id>   for each fourfold-flare-* job
set -euo pipefail
N=${1:-7}
HOURS=${HOURS:-14}
D=/mnt/storage16tb/fourfold/flare
for i in $(seq 0 $((N - 1))); do
  port=$((9400 + i))
  sb run --name "fourfold-flare-$i" --host quincy --gpu 31 --threads 2 --mem 16 --hours "$HOURS" \
    --lane batch --protected --cwd "$D" -- \
    "$D/flare" run "$D/pkg/model.safetensors" \
      --publisher-keys "$D/keys/development-publisher.json" --trust-store "$D/keys/trust.toml" \
      --verify-weights --gpus 0 --memory-limit 31G \
      --max-concurrency 4 --kv-cache-limit concurrency --mtp-draft-tokens 3 --no-mtp-adaptive-verification \
      --prefix-cache-memory 6G --embedding-table-storage ram --scheduler-policy balanced \
      --served-model-name qwen38-27b --host 127.0.0.1 --port "$port"
done
sb jobs | grep fourfold-flare || true
