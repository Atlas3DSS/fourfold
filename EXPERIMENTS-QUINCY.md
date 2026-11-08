# fourfold on Quincy: a hundred small models wired into a torus

This is a fork of **[erz0r/fourfold](https://github.com/erz0r/fourfold)** (MIT, © 2026 erz0r).
The idea, the simulator, the prompts, the scoring and the original showcase run
([`examples/sol-50`](examples/sol-50): nine gpt-6.1-sol agents that worked out they were in a 3×3 grid)
are erz0r's. Read their [README](README.md) first; everything below builds on it.

We took the same harness to much larger worlds with a much weaker local model. We ran up to 100 copies of
Qwen3.8-27B, on flat grids and on a **torus** (a grid whose edges wrap round, so nobody sits at an edge),
and tried a few small hints in the prompt. Everything runs on **Quincy**, a box with seven Tesla V100 cards;
each concurrent agent slot on it is a "quincy". Status: **in progress** (last updated 2026-11-07). This page
is updated as runs finish.

**Watch it live:** [neuropedia.atlas3dss.com/fourfold](https://neuropedia.atlas3dss.com/fourfold) is a 3D
replay of every run that you can bend from a flat grid into a torus. It refreshes every few minutes.

## The story so far

1. **Left alone, the weak model concludes it is alone.** With erz0r's prompt unchanged, all 28 agents in a 4×7
   grid answered "how many processes are there?" with **1**, after 100 ticks. Sol's nine agents were talking
   by tick 6.
2. **"You might not be alone." makes them look, but not talk.** One added sentence doubles the agents that
   reason about other processes, and they start running experiments to find out. But in 23 ticks on a grid
   and 22 on a torus, **none** of 100 agents put readable text on a wire.
3. **Give them five questions and they start talking, in ASCII, on their own.** We added: *"there are five
   questions you can ask: what are you? who am i? who are we? what are we? where are we?"* Nobody mentioned
   ASCII or how to send text. Within 11 ticks, **31 of 100 agents** had written readable text into their
   ports, starting at tick 1. By tick 12, **24 question texts were arriving** at other agents' inputs. The very
   same torus with the same wiring and only "you might not be alone." produced **zero**.
4. **Thinking made it too slow.** With reasoning on, each tick waited for the slowest agent, and those
   replies kept getting longer. Everything here runs with thinking off.

What they actually wrote, decoded from their port values (tick and agent that first sent each):

| tick | agent | on the wire |
|---|---|---|
| 1 | 37 | `Hello` |
| 2 | 46 | `who are you?`  ·  `what are we?` |
| 2 | 59 | `what are you?`  ·  `who am i?`  ·  `who are we?` |
| 3 | 54 | `ASK: WHO AM I?` |
| 4 | 36 | `ANSWER` |
| 5 | 89 | `where are we?` |
| 5 | 69 | `process 69a17d8f` |
| 6 | 23 | `I am a process` |

Agents are numbered from 1, as in erz0r's README and the scorer. A port holds 32 hex characters (16 bytes), so each
message is the ASCII bytes plus zero padding. Agent 46 sent
`who are you?` as `0000000077686f2061726520796f753f` (padding first), and agent 23 sent `I am a process` as
`4920616d20612070726f636573730000` (padding last). Many early attempts were rejected because the agents
miscounted the padding (34 hex characters instead of 32). The simulator refuses a value of the wrong length and
tells the agent why on the next tick.

## What we told them (more than the original did)

The original gave nine gpt-6.1-sol agents the minimal prompt in `prompts/system.txt`, high reasoning effort, and
nothing else: no hint that anyone else exists and no suggestion of how to talk. Our agents are a much smaller
model, and **we scaffolded them more than the original did**. Every addition, so each result is read against
the right baseline:

| | original (gpt-6.1-sol) | ours (Qwen3.8-27B) |
|---|---|---|
| system prompt | `prompts/system.txt` | the same file, unchanged |
| told others might exist | no | in primed runs: "you might not be alone." |
| given questions to ask | no | in the five-questions run: "you might not be alone. there are five questions you can ask: what are you? who am i? who are we? what are we? where are we?" |
| told how to talk | no | only in the queued ASCII run: "you might not be alone. a port value can also be read as ascii text, two hex characters per byte, which is one way to send a message." |
| reasoning | high effort | thinking **off**: less help, because thinking made 100-agent ticks too slow |
| world | 3×3 flat grid | 4×7 and 10×10 flat grids, and 10×10 tori |

The added sentence goes in as one new paragraph after the port description, and nothing else in the prompt
changes. Only the unprimed runs are like-for-like with the original experiment. Read every primed result as
"with a hint".

## Results in detail

### Text on the wire, run by run

Agents writing readable text (at least 4 printable characters, at least 3 letters, not a fill like `3333`)
in the same window, the first 11 ticks, so runs of different lengths compare fairly:

| run | world | what they were told | agents writing text, first 11 ticks |
|---|---|---|---|
| `quincy-4x7-grid` | 4×7 grid, 28 agents | original prompt | 0 of 28 (2 in all 100 ticks, both noise like `\|"9o\|"9o`) |
| `10x10-grid-unprimed-partial` | 10×10 grid | original prompt | 1 of 100 (`u6XO`, noise; 4 ticks only) |
| `ladder-10x10-grid-20261107-1435` | 10×10 grid | "you might not be alone." | **0** of 100 |
| `q-10x10-torus-primed-open-1107-1519` | 10×10 torus | "you might not be alone." | **0** of 100 |
| `q-10x10-torus-five-questions-1107` | 10×10 torus, same wiring | + the five questions | **31** of 100 |

`python experiments/quincy/ascii_text.py experiments/quincy/runs/*` reproduces this table and lists every message.

### 1. Thinking is too slow at this scale, and its tail grows

With Qwen's low reasoning effort, the slowest reply in a tick went from 28 s at tick 1 to 111 s at tick 5, and
the median reply from 711 to 2,247 tokens. Agents think longer as their notes fill up, and a synchronous
tick waits for the slowest agent. With thinking off, a 28-agent tick takes about 22 s and stays there.

### 2. Without help, every weak agent concluded it was alone

`quincy-4x7-grid` ran 100 ticks with the original prompt. All 28 agents answered the final count question
with **1**, describing themselves as an "Isolated instance", a "Sole process", or a "Single process with internal feedback
loops". The only credit they earned was for calling their dead edge ports dead (eval total 0.05, against 0.60
for the sol showcase).

The information was there:
- half of them received a neighbour's id on an input and wrote it down;
- several saw their own values come back and called it "self-feedback".

They settled into a quiet equilibrium. The share of agents still changing their outputs fell from 0.54 to about
0.29, and their logs read "All inputs unchanged… No action required. Monitoring."

### 3. "You might not be alone." makes them look

By tick 5 of a 10×10 grid, 62% of agents (on a broad keyword count) were writing about other processes, against
25–29% without the sentence. They designed experiments: "Goal: determine if I am alone or interacting with
other processes", and "q2/q3 = echo of p1/p2 to test if inputs are from other agents or fixed".

A shared broadcast format appeared without coordination: an agent's id followed by a tick counter
(`80ffedc6…02`, `…03`, …), which neighbours then track. Most echo tests miss, though. An echo only gets back
to its sender if it goes out on the same port the value came in on, and most agents echo across ports.

### 4. On a torus, more of them find a neighbour's name

On a torus every agent has four live neighbours, where an edge agent on a flat grid has three or two. At about
tick 22, 65% of agents on the primed torus had written a true neighbour's id into their notes, against 55%
on the primed flat grid at tick 23.

### The same pictures as the original, at our scale

These are drawn the way erz0r's `examples/sol-50` figures are:
- the replay uses `view.py`'s own tile renderer (180 px per tile, first 50 ticks, ~0.34 s per tick, status bar cropped);
- the map figure follows `map.png`'s layout and colours.

One visible difference: the original was rendered with Consolas, which this Linux machine lacks, so the tile
text is DejaVu Sans Mono and wraps a few characters differently. `experiments/quincy/render_replay.py` and
`render_map.py` make both for any run, grid or torus. The torus figures will be added when those runs are scored.

**4×7 grid, original prompt, thinking off: first 50 ticks**

![replay of the 4x7 run's first 50 ticks](experiments/quincy/runs/quincy-4x7-grid/replay.gif)

**Its final answers: no agent drew a map, and all 28 said they were alone**

![the 4x7 run's map figure: no map drawn, 0 of 28 placed](experiments/quincy/runs/quincy-4x7-grid/map.png)

### All numbers

From `python experiments/quincy/analyze.py experiments/quincy/runs/*`. Runs still in progress are snapshots
taken on 2026-11-07, at the tick shown.

| run | world | agents | ticks | thinking | told | s/tick | active | social | names a neighbour | says "alone" | eval total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `quincy-4x7-grid` | 4×7 grid | 28 | 100 | off | original | 22 | 0.29 | 0.11 | 0.50 | **1.00** | **0.05** |
| `quincy-4x7-grid-lowthink-partial` | 4×7 grid | 28 | 5 | low | original | 80 | 0.85 | 0.32 | 0.57 | | |
| `10x10-grid-unprimed-partial` | 10×10 grid | 100 | 4 | off | original | 58 | 0.52 | 0.06 | 0.53 | | |
| `ladder-10x10-grid-20261107-1435` | 10×10 grid | 100 | 23 (paused) | off | not alone | 76 | 0.46 | 0.34 | 0.55 | | |
| `q-10x10-torus-primed-open-1107-1519` | 10×10 torus | 100 | 22 (running) | off | not alone | 120* | 0.49 | 0.29 | 0.65 | | |
| `q-10x10-torus-five-questions-1107` | 10×10 torus | 100 | 12 (running) | off | + 5 questions | 154* | 0.47 | 0.32 | 0.62 | | |
| showcase `sol-50` (erz0r) | 3×3 grid | 9 | 50 | high | original | | | | | 0.00 | 0.60 |

\* The two tori share the same servers, and one card is on loan to the lace work below, so their ticks are slower.

Column definitions (from `analyze.py`):
- **active:** share of agents whose outputs changed, averaged over the last 10 ticks.
- **social:** share of agents whose notes talk about other processes, at the last tick.
- **names a neighbour:** share of agents that wrote a true neighbour's id into their notes, whether or not
  they understood it as a name.
- **says "alone":** share of agents whose final count answer was 1.

Raw data is in [`experiments/quincy/runs/`](experiments/quincy/runs). `ticks.jsonl.gz` holds every prompt, reply,
log and port value; `eval-t*.json` holds the final answers and scores.

## What's next

- **Score the tori at tick 100** and draw their replay and map figures.
- **A torus told about ASCII but given no questions**, to separate "they were given words" from "they were told how to send words".
- **Thinking back on, small grid** (running): the 4×7 grid with the same wiring as the original-prompt run, low
  reasoning effort, 50 ticks (sol's length), once with the original prompt and once with "you might not be
  alone.". Does thinking alone get the weak model talking, without the five questions?
- **One agent with a neural lace.** [Neuropedia](https://neuropedia.atlas3dss.com)'s persona laces (the Sun
  Wukong lace first) nudge a handful of neurons in layers 34 to 62 through Flare's per-request activation
  steering. Only one agent gets it; nothing in its prompt changes. The viewer's influence colouring will show
  whether anything it does spreads across the torus.
- **Bigger worlds** (20×20 = 400 agents). They take longer per tick but need no new hardware.

## Setup

| | original showcase | this fork |
|---|---|---|
| model | gpt-6.1-sol, high reasoning effort | Qwen3.8-27B (NVFP4 MLP), thinking off unless noted |
| serving | OpenAI API | [Flare](https://github.com/Atlas3DSS/flare-local-inference), the SM70 (V100) port of [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) |
| hardware | hosted | 7 × Tesla V100 32 GB, one Flare server per card |
| agents | 9 (3×3 grid) | 28 (4×7) and 100 (10×10) |
| topology | flat grid, edge ports dead | flat grid and torus (edges wrap, no dead ports) |

**Slots.** The V100 Flare package serves 32,768 tokens per request and at most 4 requests at once per card
(its attention kernel is fixed at four), so Quincy has 28 slots. Lowering the context does not add slots on
this hardware. A world can still be bigger than 28 agents: the extra agents wait for a slot, and a tick takes longer.

## What changed in the code

All additions are opt-in. With an upstream `config.toml`, `sim.py` and `eval.py` behave as before (same world
for the same seed, same prompts, same scores).

- **Several endpoints** (`sim.py`, `LLMAgent`). `llm.base_url` may be a list:
  - `routing = "pinned"` sends agent *i* to endpoint *i mod n*;
  - `routing = "least_loaded"` sends each call to the live server with the fewest calls in flight, skips a
    server that refuses connections for 30 s, and picks it up again after;
  - `endpoint_concurrency` caps calls in flight per server.
- **Torus** (`topology = "torus"`, `--topology torus`). A boundary side wraps to the opposite edge, so every
  port is live. A seed gives the same ids and port shuffles in both shapes. `eval.py` scores a torus fairly:
  - arrangement: half credit for the right dimensions, half for saying the edges wrap;
  - position: credit for saying there is no edge or corner to sit on;
  - map: matched under every cyclic shift as well as the 8 rotations and reflections, since no cell is special.
- **Prime** (`prime = "..."`). Extra sentence(s) added after the port description in the system prompt. Empty by default.
- **Per-agent request extras** (`llm.agent_extra_body` / `agent_extra_body_file`). Extra fields sent with one
  agent's requests only, which is how one agent gets a lace.
- **Overrides** (`--rows`, `--cols`, `--ticks`, `--set key=value`) for new runs, so one config file serves a whole queue.
- **Quincy tooling** (`quincy/`). It covers the servers (`flare_up.sh`), a preflight that fills every slot once
  (`smoke.py`), and run, queue and figure scripts. `QUINCY.md` is the operator's runbook. These refer to our
  hosts; the code changes above are general.
- **Analysis and figures** (`experiments/quincy/`):
  - `analyze.py` and `ascii_text.py` produce every number on this page;
  - `render_replay.py` and `render_map.py` draw erz0r-style figures;
  - `viewer/torus.html`, built by `build_viewer.py` from `viz_data.py`, is the 3D replay. Its influence trace
    is strict: a value counts only if it first appeared at the source agent, or at an agent the source had
    already reached.

## Reproduce

```
pip install -r requirements.txt
python sim.py --agent echo --ticks 30 --topology torus --rows 5 --cols 5      # plumbing, no model
# any OpenAI-compatible servers; list several to spread agents over them
python sim.py --config config.quincy.toml --rows 10 --cols 10 --topology torus --world-seed 20261107 \
    --set prime="you might not be alone." --set llm.base_url='["http://host:8000/v1"]'
python eval.py runs/<run_id>
python experiments/quincy/analyze.py runs/<run_id>
python experiments/quincy/ascii_text.py runs/<run_id>
```

## Credits

- **[erz0r/fourfold](https://github.com/erz0r/fourfold)**, the experiment, harness, prompts, scoring and the
  original sol-50 result. This fork keeps their MIT license and their code as the base.
- **[Flare](https://github.com/NVIDIA/flare-local-inference)** (NVIDIA) and its
  [SM70 port](https://github.com/Atlas3DSS/flare-local-inference), the inference server for the V100s.
- **[Neuropedia](https://neuropedia.atlas3dss.com)**, the neuron map and persona laces for the lace experiment.
- **Qwen3.8-27B** (Qwen team), the agents' model.
