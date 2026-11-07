# fourfold on Quincy (Flare, 7x V100)

Staged 2026-11-07. Code: `/mnt/storage16tb/fourfold/fourfold` on Quincy (mirror of
`~/dev_genius/ripple-test/fourfold`, branch `flare-quincy`). Flare: `/mnt/storage16tb/fourfold/flare`
(binary `flare` = flare-public-4528c220, `pkg/` = the sm_70 Qwen3.8-27B release package as
hardlinks, `keys/` = its development publisher key and trust store; signatures verified).

## Shape
- 7 Flare servers, one per V100, ports 9400-9406, each 32k context x 4 slots = 28 slots.
- 4x7 grid = 28 agents. Agent i is pinned to server i % 7 (its prefix cache stays on one card),
  at most 4 in flight per server (`endpoint_concurrency`).
- 100 ticks, then `eval.py`. Thinking off (`chat_template_kwargs.enable_thinking = false`, owner call:
  thinking tails grew every tick), `max_tokens` 4096 per tick.

## Launch (only with the owner's go)
```
cd ~/dev_genius/ripple-test/fourfold
quincy/flare_up.sh                       # queues 7 protected GPU jobs; each starts when a card frees
sb run --name fourfold-quincy-run --host quincy --threads 2 --mem 4 --hours 16 \
  --cwd /mnt/storage16tb/fourfold/fourfold -- bash quincy/run.sh
```
`run.sh` waits for all 7 servers, fills all 28 slots once as a smoke test (aborts on any bad
reply), then runs the sim and the eval. Resume a stopped run with `RUN_ID=<id> bash quincy/run.sh`.

## More than 28 agents
- No Flare change needed: make the grid bigger (e.g. rows=7, cols=8 = 56). The slots stay 28, so
  each tick runs in waves and takes about agents/28 times as long.
- More slots per card needs a new package: `tools/prepare_replay.py --context 8192 --concurrency 12..16`.
  The sm_70 fork's README limits it to 4 concurrent requests, so that is unvalidated Flare work.
  KV estimate: 64 KiB/token (16 full-attention layers) + ~150 MB of linear-attention state per slot,
  in about 7-9 GiB of free VRAM per card: about 12-16 slots at 6-8k context, so 84-112 per Quincy.

## Run log
- 2026-11-07 21:51Z `quincy-4x7-grid`: 28 agents, 6 servers (24 slots), thinking off. ~25 s/tick, 1-3 agents
  with a rejected value per tick. (`quincy-4x7-grid-lowthink-partial`: 5 ticks at low thinking, stopped
  because the slowest reply grew from 28 s to 111 s by tick 5.)
- queued `fourfold-ladder`: starts when that run and its eval finish; `config.quincy.toml` (all 7 servers,
  least-loaded routing, so the 7th card joins when chemistry frees it): 10x10 then 20x20, 100 ticks each.
- next: 20x20 torus. The server jobs end ~2026-11-08 11:30Z (14 h limit), so renew them before it:
  `sb stop` the old fourfold-flare-* jobs, then `quincy/flare_up.sh`.

- 2026-11-07 23:19Z `q-10x10-torus-primed-open-1107-1519` (queue): 10x10 torus, primed, seed 20261107, 1000 ticks.
- 2026-11-07 23:31Z `q-10x10-torus-five-questions-1107` (job fourfold-five-q, `quincy/run_one.sh`): same torus and seed,
  prime "you might not be alone. there are five questions you can ask: what are you? who am i? who are we? what are
  we? where are we?", 1000 ticks, sharing the 28 slots with the open torus.
  Both were started before --ticks was recorded in the config: resume them with `--ticks 1000`
  (queue line `resume <id> --ticks 1000`, or `.venv/bin/python sim.py --resume <id> --ticks 1000`).
- Board: http://127.0.0.1:8820/ and the Neuropedia tab /fourfold.
