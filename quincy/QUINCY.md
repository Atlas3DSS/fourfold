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

## Next builds (owner, 2026-11-07), in order
1. One-island long-memory test `q-4x4-torus-longmemory-test-1107` (16 agents, `memory = "conversation"`): does
   Flare report cache hits (usage.prompt_tokens_details.cached_tokens), and what is the host-RAM footprint per agent?
2. Tiered state scheduler: GPU (in flight) / RAM (next waves) / NVMe (`--prefix-cache-disk`, cold), prefetching the next
   wave during the current one, with a pinned hot set instead of LRU (LRU is worst-case for a fixed cycle).
3. Event-driven ticking, an opt-in labelled variant: agents with unchanged inputs and no pending results sit out a tick
   (optionally wake every k ticks), compared against the every-tick baseline.
Quincy goes to 768 GB RAM next week: ~650 GB of parked agent state, ~720 agents at ~12K tokens each.

## STOPPED (2026-11-08, owner: fix looping thoughts and run-on replies first)
Everything below is paused. Stopped: fourfold-queue, fourfold-eval-t100, fourfold-network-sum (ticks.jsonl ends at t69 of
100; resume with `.venv/bin/python sim.py --resume q-4x28-arrays-sum-actions-1107`), fourfold-flare-0..5, and fabe's
queued fourfold-gpu-session. The long-memory test had already finished (t20). All 7 V100 leases are released.
runs/QUEUE still holds its 4 lines. Restart only on the owner's go: servers first (flare_up.sh), then the network-sum resume, then the queue.

## Thinking xhigh queue (owner 2026-11-08: "run all our experiments with thinking xhigh", log it all per the norm)
Job fourfold-xhigh-queue on Quincy: Q=runs/QUEUE-XHIGH CFG=config.quincy-xhigh.toml MIN_UP=5 bash quincy/queue.sh, servers
fourfold-flare-0..5 (quincy/flare_board.sh, 16 h leases: when they end, sim exits 3, queue.sh puts the run back and waits; relaunch
the servers). The template allows xhigh, medium, low; "high" is a 400. A bare tick-1 reply: 16,351 tokens, 276 s, so about 20-25
min a tick at 112 agents. fabe-integration is building thinking control (the flare-6 slot is theirs).

## Haiku and coordinate runs (2026-11-08)
Done on devbox (only devbox has a logged-in Claude Code): q-4x28-arrays-sum-actions-haiku-1108, q-10x10-torus-primed-haiku-1108.
Coordinate on Haiku: q-4x28-arrays-sum-coordinate-haiku-1108 (devbox). flare_board.sh joins the workboard queue directly because
`sb run --gpu` batch jobs wait outside it while every card is full.

## State at handoff (2026-11-08 03:15Z / Nov 7 7:15 PM PST)
Running on Quincy (`sb jobs`):
- `fourfold-network-sum` → run `q-4x28-arrays-sum-actions-1107`: 112 agents, 7 arrays 4x4, seams every 2 ticks,
  sum task (answer 55279), five actions, 100 ticks, ~4 min/tick; tick 15 at 03:00Z → done ~08:45Z, then its eval.
- `fourfold-longmem-test` → `q-4x4-torus-longmemory-test-1107`: 16 agents, memory=conversation, on 9405, 20 ticks.
  **Problem: Flare reports 0 cached tokens and ticks slow as history grows (prompt ~6.3K tokens at t7).** Find out
  why (chat template re-rendering past replies? 512-token checkpoint granularity? eviction under shared load?)
  before building the tiered scheduler (#321). Owner: any Flare change goes in OUR fork, never the shared Flare tree
  chemistry uses.
- `fourfold-queue` (job db6ae3) waits for the network task, then runs runs/QUEUE: resume both tori to 150 with
  `--ratchet 15`, resume the 10x10 grid, then the ASCII-hint torus. Owner rule: cap 150; every 15 ticks a run must
  be interesting (sim.interest: talk / discovery / task progress) or it is evicted (reason in runs/<id>/ratchet.jsonl).
- `fourfold-eval-t100`: scores both tori at tick 100 when they get there.
- `fourfold-flare-0..5` on ports 9400-9405 (30 h limit from 23:00Z Nov 7). Card GPU-f2b6541e (server 6, port 9406)
  is lent to fabe-integration: lace package prepare (job fourfold-lace-prepare), then validation on 9410, then the
  wide-slot measurement (8/16/32 slots per V100 via persona-wide; results → quincy/WIDE-SLOTS.md). When they
  release the card, bring server 6 back: `HOURS=24 quincy/flare_up.sh` queues all 7, so instead launch only
  fourfold-flare-6 (copy its sb run line from flare_up.sh with i=6).
- Lace run (Sun Wukong on one agent): payloads in quincy/payloads/ (fabe), per-agent file via
  `quincy/lace_request.py payloads.json wukong <agent0based>`; set `--set llm.agent_extra_body_file=<file>`; serve
  the steering package (flare-steer). Viewer: add {"<run>": <agent0based>} to fourfold-board/trace.json for the
  influence view.
Done tonight: public fork https://github.com/Atlas3DSS/fourfold (branch quincy, EXPERIMENTS-QUINCY.md), board
http://127.0.0.1:8820/ + Neuropedia tab /fourfold (public), share PNG share/fourfold-x.png (+ make_fourfold_x.py).
Neuropedia: the Fourfold tab lines in experiments/flare/neuropedia/neuropedia_server.py + fourfold.html are
UNCOMMITTED in that tree, next to the neuropedia agents' own uncommitted Persona Atlas work; leave them for those
agents to commit by path.
Work items: #317 (this), #321 tiered scheduler, #322 event-driven ticking, #237 Ripple.
