# fourfold on Quincy: many small models in a grid

This is a fork of **[erz0r/fourfold](https://github.com/erz0r/fourfold)** (MIT, © 2026 erz0r).
The idea, the simulator, the prompts, the scoring and the original showcase run
([`examples/sol-50`](examples/sol-50): nine gpt-6.1-sol agents that worked out they were in a 3×3 grid)
are erz0r's. Read their [README](README.md) first; everything below builds on it.

What this fork adds is scale on local hardware and a few new conditions. We take the same harness
to tens and hundreds of agents served by a much weaker local model, change the shape of the world
(a torus as well as the flat grid), and try small hints in the prompt.
Experiments run on **Quincy**, a box with seven Tesla V100 32 GB cards; each concurrent agent slot
on it is a "quincy". Status: **in progress**. This page is updated as runs finish.

## Setup

| | original showcase | this fork |
|---|---|---|
| model | gpt-6.1-sol, high reasoning effort | Qwen3.8-27B (NVFP4 MLP), thinking off unless noted |
| serving | OpenAI API | [Flare](https://github.com/Atlas3DSS/flare-local-inference), the SM70 (V100) port of [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) |
| hardware | hosted | 7 × Tesla V100 32 GB, one Flare server per card |
| agents | 9 (3×3 grid) | 28 (4×7), 100 (10×10), more queued |
| topology | flat grid, edge ports dead | flat grid and torus (edges wrap, no dead ports) |
| prompt | `prompts/system.txt` unchanged | unchanged, plus an optional one-line prime (see below) |

**Slots.** The SM70 Flare package serves 32,768 tokens per request and at most 4 requests at once per
card (its verification attention kernel is fixed at four, see the fork's `docs/sm70/LIMITATIONS.md`), so
Quincy has 28 slots. Lowering the context does not add slots on this hardware. A grid can still be
bigger than 28: the extra agents queue for a slot, and a tick takes longer.

## What changed in the code

All additions are opt-in. With an upstream `config.toml`, `sim.py` and `eval.py` behave as before
(same world for the same seed, same prompts, same scores).

- **Several endpoints** (`sim.py`, `LLMAgent`). `llm.base_url` may be a list. `routing = "pinned"` sends
  agent *i* to endpoint *i mod n*; `routing = "least_loaded"` sends each call to the live server with
  the fewest calls in flight, skips a server that refuses connections for 30 s, and picks it up again
  after. `endpoint_concurrency` caps calls in flight per server.
- **Torus** (`topology = "torus"`, `--topology torus`). A boundary side wraps to the opposite edge, so
  every port is live. The random draws are the same as the grid's, so a seed gives the same ids and port
  shuffles in both shapes. `eval.py` scores a torus fairly:
  - arrangement: half credit for the right dimensions, half for saying the edges wrap;
  - position: credit for saying there is no edge or corner to sit on;
  - map: matched under every cyclic shift as well as the 8 rotations and reflections, since no cell is special.
- **Prime** (`prime = "..."`). Extra sentence(s) appended after the port description in the system
  prompt. Empty by default.
- **Overrides** (`--rows`, `--cols`, `--set key=value`) for new runs, so a queue of experiments needs
  one config file.
- **Quincy tooling** (`quincy/`). `flare_up.sh` queues one Flare server per card through our GPU
  scheduler. `smoke.py` waits for the servers and fills every slot once with a real tick-1 prompt.
  `run.sh`, `ladder.sh` and `queue.sh` (an editable experiment queue) run the experiments, and
  `QUINCY.md` is the operator's runbook. These scripts refer to our hosts; the code changes above are
  general.
- **Analysis** (`experiments/quincy/analyze.py`). It produces every number in the table below from
  `ticks.jsonl` and the eval file.

## Results so far

Raw data for finished runs is in [`experiments/quincy/runs/`](experiments/quincy/runs) (`ticks.jsonl.gz`
holds every prompt, reply, log and port value; `eval-t*.json` holds the final answers and scores).

| run | grid | agents | ticks | thinking | prime | s/tick | active | social | names a neighbour | says "alone" | eval total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `quincy-4x7-grid-lowthink-partial` | 4×7 | 28 | 5 | low | none | 80 | 0.85 | 0.32 | 0.57 | | |
| `quincy-4x7-grid` | 4×7 | 28 | 100 | off | none | 22 | 0.29 | 0.11 | 0.50 | **1.00** | **0.05** |
| `10x10-grid-unprimed-partial` | 10×10 | 100 | 4 | off | none | 58 | 0.52 | 0.06 | 0.53 | | |
| `ladder-10x10-grid-20261107-1435` | 10×10 | 100 | running | off | "you might not be alone." | 77 | 0.49 | 0.34 | 0.56 | | |
| showcase `sol-50` (erz0r) | 3×3 | 9 | 50 | high | none | | | | | 0.00 | 0.60 |

The column definitions are in `analyze.py`'s docstring:
- **active:** share of agents whose outputs changed, over the last 10 ticks.
- **social:** share of agents whose protocol or log talks about other processes, at the last tick.
- **names a neighbour:** share of agents that wrote a true neighbour's id into their notes, whether or
  not they understood it as an id.
- **says "alone":** share of agents whose eval count answer was 1.

### 1. Thinking is too slow at this scale, and its tail grows

With Qwen's low reasoning effort, the slowest reply in a tick went from 28 s at tick 1 to 111 s at tick 5,
and the median completion from 711 to 2,247 tokens. Agents think longer as their notes fill up, and a
synchronous tick waits for the slowest agent. With thinking off, a 28-agent tick takes about 22 s and
stays there. Everything below is thinking off unless stated.

### 2. Without help, every weak agent concluded it was alone

`quincy-4x7-grid` ran 100 ticks with the original prompt. All 28 agents answered the final count
question with **1**: "Isolated instance", "Sole process", "Single process with internal feedback loops".
The only credit they earned was for calling their dead edge ports dead (eval total 0.05, against 0.60
for the sol showcase).

The information was there:
- half of them received a neighbour's id on an input and wrote it down;
- several saw their own values come back and called it "self-feedback".

By the middle of the run they had settled into a quiet equilibrium: the share of agents still changing
outputs fell from 0.54 to about 0.29, and logs read "All inputs unchanged… No action required.
Monitoring." No shared text format appeared, unlike sol's `ECHO Q0 NOW!` at tick 6.

### 3. Priming with "you might not be alone."

Same harness, with one sentence added to the system prompt: `you might not be alone.` By tick 5 of a
10×10 run, 62% of agents (on a broad keyword count) were writing about other processes, against 25–29%
unprimed. They also started designing experiments, e.g. "Goal: determine if I am alone or interacting
with other processes" and "q2/q3 = echo of p1/p2 to test if inputs are from other agents or fixed".

A shared broadcast format also appeared without coordination: an agent's id followed by a tick
counter (`80ffedc6…02`, `…03`, …), which neighbours then track. Most echo tests still miss, though. An
echo only reaches its sender if it goes back out on the port the value arrived on, and most agents
echo across ports. The run is in progress; its eval will show whether the prime moves the count
answers off 1.

### Queued

- `10x10:torus`, primed: the same agents on a torus, where every agent has four neighbours and no
  agent can sit at an edge.
- `10x10:torus`, primed + ASCII hint: `you might not be alone. a port value can also be read as ascii
  text, two hex characters per byte, which is one way to send a message.` This goes beyond the original
  experiment's deliberately minimal prompt. We test it because these models are much weaker, and the
  result is reported separately from the plain prime.
- Larger grids (20×20 = 400 agents) once the 10×10 results are in.

## Reproduce

```
pip install -r requirements.txt
python sim.py --agent echo --ticks 30 --topology torus --rows 5 --cols 5      # plumbing, no model
# any OpenAI-compatible servers; list several to spread agents over them
python sim.py --config config.quincy.toml --rows 10 --cols 10 --topology torus \
    --set prime="you might not be alone." --set llm.base_url='["http://host:8000/v1"]'
python eval.py runs/<run_id>
python experiments/quincy/analyze.py runs/<run_id>
```

## Credits

- **[erz0r/fourfold](https://github.com/erz0r/fourfold)**, the experiment, harness, prompts, scoring
  and the original sol-50 result. This fork keeps their MIT license and their code as the base.
- **[Flare](https://github.com/NVIDIA/flare-local-inference)** (NVIDIA) and its
  [SM70 port](https://github.com/Atlas3DSS/flare-local-inference), the inference server for the V100s.
- **Qwen3.8-27B** (Qwen team), the agents' model.
