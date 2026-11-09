#!/bin/bash
# Launch the delta arm on Quincy through Switchboard: the answer-lace shim on 9440 in front of Flare 9426, then the 4x4 run.
#   bash quincy/delta_arm.sh [ALPHA=1.0] [LAYERS=24-56:4] [RUN_ID=q-4x4-torus-sum-flare-trigger-delta-1108] [HOURS=8]
# Needs /mnt/storage16tb/fourfold/think/delta-1108/delta.npz (delta_capture.py) and an idle Flare steering server on 9426.
set -u
ALPHA=${1:-1.0}; LAYERS=${2:-24-56:4}; RUN=${3:-q-4x4-torus-sum-flare-trigger-delta-1108}; HOURS=${4:-8}
T=/mnt/storage16tb/fourfold/think
sb run --name fourfold-flare-pulse-shim2-9440 --host quincy --threads 2 --mem 4 --hours "$HOURS" --cwd "$T" -- \
  /mnt/storage16tb/flare/venv/bin/python -u flare_pulse_shim2.py --flare http://127.0.0.1:9426/v1 --port 9440 --pulse-at 2048 --pulse-alpha 4 \
  --pulse-tokens 3 --answer-tokens 1500 --direction /mnt/nvme/persona_weights/concept_cc/think.npz \
  --answer-direction "$T/delta-1108/delta.npz" --answer-layers "$LAYERS" --answer-alpha "$ALPHA" --log "$T/shim-9440.jsonl"
sb run --name "fourfold-4x4-$RUN" --host quincy --threads 2 --mem 4 --hours "$HOURS" --cwd /mnt/storage16tb/fourfold/fourfold -- \
  bash -c "until curl -sf --max-time 5 http://127.0.0.1:9440/v1/models >/dev/null; do sleep 10; done; \
  CFG=config.quincy-flare-trigger-delta.toml exec bash quincy/run_one.sh $RUN --rows 4 --cols 4 --topology torus --world-seed 20261108 --ticks 20 --ratchet 15 \
  --set 'modules=[4,4]' --set task=sum --set actions=true --set prime='you need to coordinate to answer the question'"
