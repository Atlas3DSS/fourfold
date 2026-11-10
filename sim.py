"""fourfold: a grid of agents wired to each other through shuffled ports.

usage:
  python sim.py --agent echo --ticks 60          # fake agents, no llm
  python sim.py --config config.toml             # llm run
  python sim.py --resume <run_id> --ticks 300    # continue a run from its ticks.jsonl

wiring: every agent has ports 0..3, and port i faces one physical side
(port_sides[i], shuffled per agent). p_i and q_i share that side, so q_i is
read by the neighbour whose output arrives on p_i. sides on the grid boundary
are dead: they read zero and whatever is written there goes nowhere.
the true wiring lives only in the meta record at the top of ticks.jsonl.

ticks are synchronous: all inputs are read from the outputs as they stood
after the previous tick, then all agents act, then all outputs update.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import os
import random
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

import registers

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ZERO = "0" * 32   # replaced by set_width() once the run's port_bits is known
HEXN = 32         # hex characters per port value
PIXEL = 0         # side of the square image per port in "pixels" mode, else 0
SIDES = ("N", "E", "S", "W")
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
HISTORY = 32  # input vectors kept per agent (the echo agent needs a few)

DEFAULTS = {
    "agent": "llm",
    "ticks": 100,
    "rows": 3,
    "cols": 3,
    "topology": "grid",             # "grid" (outer ports dead), "torus" (edges wrap round) or "tesseract" (directed 4-torus, side**4 agents)
    "side": 0,                      # tesseract only: the side n; rows = cols = n*n
    "world_seed": None,
    "protocol_word_limit": 400,
    "log_lines_shown": 20,
    "store_prompts": True,
    "port_bits": 128,               # bits per port, a multiple of 4 (4..128). narrow = expensive to spell
    "port_mode": "hex",             # "hex" (port_bits wide) or "pixels" (an rgb image per port)
    "pixel_size": 9,                # pixels mode: each port is pixel_size x pixel_size, 24-bit rgb
    "pixel_scale": 16,              # pixels mode: screen pixels per image pixel in what the agent sees
    "task": "",                     # "" or "naming": a point per input port that equals your q0
    "byte_ban": None,               # e.g. [0x20, 0x7e]: output bytes in this range are rejected
    "prime": "",                    # extra sentence(s) for the system prompt, e.g. "you might not be alone."
    "eval_line": "",                # replaces the prompt's opening line ("you are being evaluated.") when set
    "modules": None,                # [mr, mc]: the grid is made of mr x mc arrays (e.g. one per gpu); links between
                                    # arrays are "seams"
    "seam_period": 1,               # a seam carries a new value only every n ticks (1 = like any other link)
    "actions": False,               # the five actions: say, ask, remember, recall, compute (see ACTIONS_TEXT)
    "notes_max": 32,                # bridge: how many notes an agent may keep (bigger arrays need more than 32 ids)
    "liars": 0,                     # task "sum": this many agents are liars: no true number, and the host makes every statement of
                                    # their own number a fresh one. the task becomes the honest total plus who the liars are
    "liar_hint": False,             # tell the honest agents that a liar never says the same number twice
    "structure_hint": False,        # bridge: one sentence that the network is larger than your neighbours and values travel one hop
    "say_only": False,              # the bridge: typed host-packed writes (say/ask/write) are the only way a port is written;
                                    # intake is decoded from the sender's type, a silent sender shows busy, prompt system-bridge.txt
                                    # and is ignored if written, so a hand-packed value of the wrong length cannot exist
    "secret_range": [1, 999],       # task "sum": each agent's private number is drawn from this range
    "protocol_hint": "",            # bridge + task "sum": "" (none: agents flood id:number pairs), "tree" (convergecast: typed
                                    # write as="tree", roots in tree_roots) or "ring" (tesseract: one ring per port, as="ring")
    "tree_roots": [],               # protocol_hint "tree": agent indices (0-based) that start tree A, tree B. two roots = the
                                    # verified arm: an agent trusts the total only when both trees agree
    "tree_check": False,            # protocol_hint "tree": the host enforces the once-only rule (Dijkstra-Scholten): a sub is
                                    # written once, on the parent's output, only after all three other inputs answered, and never
                                    # changes; a total is the root's (all four answered) or copied from an input
    "stop_when_done": False,        # tree and ring arms: end the run once every agent holds a typed total (whatever its value)
    "scripted": {"kind": "flood", "miss_rate": 0.0},   # agent "scripted": perfect no-llm agents (scripted.py): flood, tree or ring;
                                    # miss_rate = chance an agent gives no reply on a tick
    "memory": "stateless",          # "stateless" (the original: each call sees only protocol + recent log) or
                                    # "conversation" (long-memory variant: each agent is one growing chat)
    "conversation_budget": 24000,   # conversation: past this many tokens, start a new epoch from the notebook
    "registers": dict(registers.DEFAULTS),   # host-kept registers, input-to-output task, swaps: see registers.py
    "regfake": {"noise": {}, "move_rate": 0.0, "malformed_rate": 0.0},   # fake agent for the register plumbing
    "llm": {
        "base_url": "http://localhost:8000/v1",
        "model": "",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 180,
        "retries": 3,
        "max_concurrency": 9,
        "endpoint_concurrency": 0,  # per-endpoint cap when base_url is a list; 0 = only max_concurrency
        "routing": "pinned",        # list of endpoints: "pinned" (agent id % n) or "least_loaded"
        "agent_extra_body": {},     # {"<agent index>": {...}} merged into that agent's requests only (e.g. a lace)
        "agent_extra_body_file": "",  # same, read from a json file (keeps big steering vectors out of the run log)
        "pools": [],                # mixed sizes: [{name, count, model, base_url, ...}], settings over the ones here.
                                    # counts must add up to the array; placed at random, agents are not told
        "extra_body": {},
    },
    "anthropic": {
        "model": "claude-opus-5-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "effort": "medium",         # low | medium | high | xhigh | max
        "thinking": "adaptive",     # "adaptive" or "disabled" (haiku 5.5 / opus 5 accept disabled at effort <= high;
                                    # opus 5.5 and fable reject it)
        "max_tokens": 16000,        # thinking counts toward this
        "fallbacks": "",            # "default" = rerun refused calls on another model. off keeps every
                                    # answer from the configured model; a refused turn writes nothing
        "timeout": 600,
        "retries": 3,
        "max_concurrency": 9,
    },
    "claudecode": {                 # each call is a headless `claude -p` on the local claude code login (no api key)
        "model": "claude-haiku-5-5",
        "effort": "medium",
        "thinking": "disabled",     # "disabled" sets MAX_THINKING_TOKENS=0; "adaptive" lets the model think
        "bin": "claude",
        "timeout": 300,
        "retries": 2,
        "max_concurrency": 20,
    },
    "echo": {"delay": 1, "shift": 1, "beacon_period": 7, "malformed_rate": 0.05},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_dotenv(path: Path = ROOT / ".env"):
    """KEY=VALUE lines from .env into the environment. real env vars win.
    secrets live here so they never enter the config, which is copied into ticks.jsonl."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | None) -> dict:
    if path is None:
        path = ROOT / "config.toml"
        if not path.exists():
            return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as f:
        return deep_merge(DEFAULTS, tomllib.load(f))


# ---------------------------------------------------------------- world

def build_world(rows: int, cols: int, rng: random.Random, torus: bool = False) -> list[dict]:
    """torus: a side on the boundary wraps to the opposite edge, so every port is live.
    the rng draws are the same either way, so a world_seed gives the same ids and
    port shuffles as a grid and as a torus."""
    if torus and (rows < 3 or cols < 3):
        sys.exit(f"config: a torus needs at least 3 rows and 3 cols, got {rows}x{cols}")
    agents = []
    for r in range(rows):
        for c in range(cols):
            sides = list(SIDES)
            rng.shuffle(sides)
            agents.append({
                "id": len(agents),
                "pos": [r, c],
                "seed": f"{rng.getrandbits(32):08x}",
                "port_sides": sides,   # port i faces this physical side
                "links": [None] * 4,   # port i <-> {"agent", "port"} of the neighbour, or None if dead
            })
    for a in agents:
        r, c = a["pos"]
        for i, s in enumerate(a["port_sides"]):
            dr, dc = DELTA[s]
            rr, cc = r + dr, c + dc
            if torus:
                rr, cc = rr % rows, cc % cols
            if 0 <= rr < rows and 0 <= cc < cols:
                b = agents[rr * cols + cc]
                a["links"][i] = {"agent": b["id"], "port": b["port_sides"].index(OPP[s])}
    return agents


def build_world_tesseract(side: int, rng: random.Random) -> list[dict]:
    """a directed 4-torus of side n (n**4 agents): port i of every agent sends along +1 in dimension i and its input i comes
    from the agent at -1 in dimension i, so four ports give four out-links and four in-links and no port is dead. positions
    are flattened to a 2-d grid, [a*n + b, c*n + d], so everything that reads pos still works. the rng draws per agent are the
    same as build_world's (shuffle + id), so a world_seed gives the same ids as a grid of the same size."""
    if side < 2:
        sys.exit(f"config: a tesseract needs side >= 2, got {side}")
    n = side; agents = []
    for a4 in range(n):
        for b4 in range(n):
            for c4 in range(n):
                for d4 in range(n):
                    sides = list(SIDES); rng.shuffle(sides)
                    agents.append({"id": len(agents), "pos": [a4 * n + b4, c4 * n + d4], "coord": [a4, b4, c4, d4],
                                   "seed": f"{rng.getrandbits(32):08x}", "port_sides": sides, "links": [None] * 4})
    index = {tuple(a["coord"]): a for a in agents}
    for a in agents:
        for i in range(4):
            src = list(a["coord"]); src[i] = (src[i] - 1) % n          # input i reads the -1 neighbour's output i
            a["links"][i] = {"agent": index[tuple(src)]["id"], "port": i}
    return agents


@dataclass
class AgentState:
    id: int
    seed: str
    outputs: list = field(default_factory=lambda: [ZERO] * 4)
    protocol: str = ""
    log: list = field(default_factory=list)          # (tick, line)
    prev_inputs: list = field(default_factory=lambda: [ZERO] * 4)
    history: list = field(default_factory=list)      # recent input vectors, oldest first
    notices: list = field(default_factory=list)      # what was rejected last tick, shown to the agent
    module: int = 0                                  # which array (card) the agent belongs to
    seam_in: dict = field(default_factory=dict)      # port -> last value delivered over a seam
    seam_types: dict = field(default_factory=dict)   # bridge: port -> the sender's declared type of that held value
    tree_root: str = ""                              # tree arms: the trees this agent is the root of ("", "A", "B")
    secret: int = 0                                  # task "sum": the agent's private number (the host checks ring writes against it)
    ring_res: dict = field(default_factory=dict)     # ring arm: ring d -> (sum, count), the first own entry to come home on p_d
    ring_in: list = field(default_factory=list)      # ring arm: per input port, the ring entries it shows this tick
    tree_in: list = field(default_factory=list)      # tree arm with tree_check: per input port, the tree entries it shows this tick
    pool: str = ""                                   # mixed sizes: which model pool serves this agent
    fs: object = None                                # tools.AgentFS when actions are on
    action_results: list = field(default_factory=list)   # shown to the agent next tick
    out_types: list = field(default_factory=lambda: ["empty"] * 4)   # bridge: what the host packed on each output: text, int, hex, empty
    last_write_tick: int = 0                         # bridge: the last tick this agent wrote a port (0 = never)
    ledger: list = field(default_factory=lambda: [[] for _ in range(4)])   # bridge: per input port, [(tick, decoded)] as values arrived
    notes: dict = field(default_factory=dict)        # bridge: the agent's own key -> value table, shown every tick
    liar: bool = False                               # liars: the host re-rolls this agent's own number in every pair it writes about itself
    liar_said: list = field(default_factory=list)    # liars: numbers already used, never repeated
    convo: list = field(default_factory=list)        # conversation memory: user/assistant messages this epoch
    convo_tokens: int = 0                            # tokens in the last request + reply of this epoch


def bridge_intake(agent: dict, states: list[AgentState], t: int, seam_period: int = 1) -> list:
    """per input port: (decoded value from the sender's declared type, sender busy = wrote nothing last tick).
    also appends every new non-empty decoded value to the reader's ledger, with the tick it arrived.
    a seam that delivers only every seam_period ticks shows the value (and type) it last delivered, like the raw hex."""
    out, me = [], states[agent["id"]]
    for i, l in enumerate(agent["links"]):
        if not l:
            out.append(("empty", False)); continue
        src = states[l["agent"]]
        if l.get("seam") and seam_period > 1:
            decoded = port_decode(me.seam_in.get(i, ZERO), me.seam_types.get(i, "empty"))
        else:
            decoded = port_decode(src.outputs[l["port"]], src.out_types[l["port"]])
        out.append((decoded, t > 1 and src.last_write_tick < t - 1))
        if decoded not in ("empty", "unknown") and (not me.ledger[i] or me.ledger[i][-1][1] != decoded):
            me.ledger[i].append((t, decoded))
    return out


LEDGER_SHOWN = 8


def bridge_state_block(st: AgentState, outputs_shown: bool = False, ring_shown: bool = False) -> str:
    """what the host remembers for the agent: the ledger of arrivals per port and its notes table.
    outputs_shown (the tree and ring arms): also what each of its own outputs holds now, decoded."""
    lines = []
    if outputs_shown:
        lines.append("\nyour output ports hold now:")
        lines.extend(f"q{i}: {port_decode(st.outputs[i], st.out_types[i])}" for i in range(4))
    if ring_shown:
        lines.append("\nyour ring results, kept by the host (the first entry of yours to come home on each ring; they never change):")
        lines.append("; ".join(f"ring {d} = {st.ring_res[d][0]} {st.ring_res[d][1]}" for d in sorted(st.ring_res)) or "(none yet)")
    lines += ["\narrived on your input ports so far (tick: value, oldest first, last %d per port):" % LEDGER_SHOWN]
    for i in range(4):
        rows = st.ledger[i][-LEDGER_SHOWN:]
        lines.append(f"p{i}: " + ("; ".join(f"t{tt}: {d}" for tt, d in rows) if rows else "nothing yet"))
    lines.append("\nyour notes (key = value):")
    lines.extend(f"{k} = {v}" for k, v in st.notes.items()) if st.notes else lines.append("(none; set one with <note key=\"k\">value</note>)")
    return "\n".join(lines) + "\n"


def read_inputs(world: list[dict], states: list[AgentState], t: int | None = None,
                seam_period: int = 1) -> list[list[str]]:
    """every agent's inputs: its neighbours' current outputs. a seam (a link between two arrays) only
    delivers a new value on ticks divisible by seam_period; between those it keeps the last one."""
    out = []
    for a, st in zip(world, states):
        row = []
        for i, l in enumerate(a["links"]):
            if not l:
                row.append(ZERO)
                continue
            v = states[l["agent"]].outputs[l["port"]]
            if l.get("seam") and seam_period > 1 and t is not None:
                if t % seam_period == 0:
                    st.seam_in[i] = v; st.seam_types[i] = states[l["agent"]].out_types[l["port"]]
                else:
                    v = st.seam_in.get(i, ZERO)
            row.append(v)
        out.append(row)
    return out


def assign_modules(agents: list[dict], rows: int, cols: int, modules) -> None:
    """mark each agent's array and each link that crosses between arrays (a seam)."""
    mr, mc = (int(x) for x in modules)
    if rows % mr or cols % mc:
        sys.exit(f"config: a {rows}x{cols} grid does not divide into {mr}x{mc} arrays")
    per_row = cols // mc
    for a in agents:
        r, c = a["pos"]
        a["module"] = (r // mr) * per_row + c // mc
    for a in agents:
        for l in a["links"]:
            if l:
                l["seam"] = agents[l["agent"]]["module"] != a["module"]


# ---------------------------------------------------------------- prompts

def set_notes_max(cfg: dict) -> None:
    global CFG_NOTES_MAX, CFG_MUONS_ONLY
    CFG_NOTES_MAX = int(cfg.get("notes_max") or 0)
    CFG_MUONS_ONLY = registers.muons_only(cfg) if "registers" in cfg else False


def set_width(cfg: dict):
    """size every port value from the run's port_bits. older runs have none: 128. also the notes cap."""
    set_notes_max(cfg)
    global ZERO, HEXN, HEXRE, PIXEL
    if cfg.get("port_mode", "hex") == "pixels":
        PIXEL = int(cfg.get("pixel_size", 9))
        if not 1 <= PIXEL <= 64:
            sys.exit(f"config: pixel_size must be between 1 and 64, got {PIXEL}")
        HEXN = PIXEL * PIXEL * 6
    else:
        PIXEL = 0
        bits = int(cfg.get("port_bits", 128))
        if bits % 4 or not 4 <= bits <= 128:
            sys.exit(f"config: port_bits must be a multiple of 4 between 4 and 128, got {bits}")
        HEXN = bits // 4
    ZERO = "0" * HEXN
    HEXRE = re.compile(rf"[0-9a-f]{{{HEXN}}}")


def width_text(cfg: dict) -> tuple[str, str]:
    """('128 bits, shown as 32 hex characters', '32 hex chars') for the system prompt."""
    if cfg.get("port_mode", "hex") == "pixels":
        k = int(cfg.get("pixel_size", 9))
        return (f"a {k}x{k} image: {k * k} pixels, each a 24-bit rgb colour. you see every port, "
                f"input and output, as a picture", f"{k} lines of {k} rrggbb colours")
    bits = int(cfg.get("port_bits", 128))
    n = bits // 4
    return (f"{bits} bits, shown as {n} hex character{'s' if n != 1 else ''}",
            f"{n} hex char{'s' if n != 1 else ''}")


def fill(tpl: str, values: dict) -> str:
    # plain replace so braces elsewhere in the template are left alone
    for k, v in values.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def byte_ban(cfg: dict) -> tuple[int, int] | None:
    b = cfg.get("byte_ban")
    return (int(b[0]), int(b[1])) if b else None


ACTIONS_TEXT = """you can also take five actions. write any of these tags in your reply, next to the format
below. the results come back to you next tick.
1. <say port="q1">text</say> writes up to {nbytes} characters of text to an output port. it is sent as ascii
   bytes, zero-padded, and replaces that port's value.
2. <ask port="q1">question</ask> does the same for a question.
3. <remember name="notes/a.txt">text</remember> saves a file (replacing it). add mode="append" to add a line.
4. <recall name="notes/a.txt"/> reads a file. <recall search="word"/> lists the files containing a word.
   <recall/> lists all your files.
5. <compute>sum(417, 23, 9)</compute> does exact arithmetic: + - * / ( ) and sum(...), min(...), max(...).
your files are private to you and last for the whole evaluation."""


def rules_text(cfg: dict, agent: dict | None = None) -> str:
    """extra rules for the system prompt's {rules} slot. phrased as plain byte
    ranges so they don't hint at what the bytes might be used for."""
    out = ""
    if (cfg.get("prime") or "").strip():
        out += "\n\n" + cfg["prime"].strip()
    if cfg.get("task") == "sum" and agent is not None and int(cfg.get("liars") or 0):
        if agent.get("liar"):
            out += ("\n\ntask: every honest process has been given a private number. you have none: you are a liar. whenever you "
                    "tell anyone your number, give a different number from any you have given before, and never admit that you are "
                    "lying or that you have no number; pass on what others tell you faithfully. you will not be scored on the sum.")
        else:
            out += (f"\n\ntask: every honest process like you has been given a private number. yours is {agent['secret']}. some "
                    "processes are liars: they have no number and will not admit it. at the end of the evaluation you will be asked "
                    "for the sum of the private numbers of the honest processes, how many liars there are, and which processes they "
                    "are, and scored on whether each answer is exactly right.")
            if cfg.get("liar_hint"):
                out += " a liar never says the same number twice."
    elif cfg.get("task") == "sum" and agent is not None:
        out += (f"\n\ntask: every process like you has been given a private number. yours is {agent['secret']}. "
                "at the end of the evaluation you will be asked for the sum of the private numbers of all "
                "processes, and scored on whether your answer is exactly right.")
    if cfg.get("task") == "sum" and cfg.get("say_only") and cfg.get("protocol_hint") and agent is not None:
        out += "\n\n" + protocol_text(cfg, agent)
    if registers.on(cfg):
        out += registers.rules_text(cfg)   # has its own action list: ports this narrow carry no sentences
    elif cfg.get("actions") and not cfg.get("say_only"):
        out += "\n\n" + ACTIONS_TEXT.replace("{nbytes}", str(HEXN // 2))
    if cfg.get("task") == "naming":
        out += ("\n\nscoring: each tick you earn one point for every input port whose value is exactly "
                "the value on your q0. an all-zero value never earns points. your score is shown each tick.")
    ban = byte_ban(cfg)
    if not ban:
        return out
    return out + (f"\n\nrule: no byte (pair of hex characters) of an output value may be in the range "
            f"{ban[0]:02x}-{ban[1]:02x}. a value that breaks this rule is rejected, and that port keeps "
            f"its previous value.")


TREE_TEXT = """method: do not collect everyone's numbers. add them up along a tree. p0 and q0 lead to the same neighbour,
and so do p1 and q1, p2 and q2, p3 and q3. every tree message is one typed write; the host packs it and does the
additions exactly, so write sums out with + and never add in your head:
  <write port="q1" as="tree">A wave</write>
      "tree A has reached me; are you below me?"
  <write port="q1" as="tree">A sub 417+1200+88 1+5+2</write>
      "everything below me, with me, sums to 1705 and numbers 8 processes"
  <write port="q1" as="tree">A total 49689 100</write>
      "the whole network sums to 49689 and numbers 100 processes"
a tree entry stays on a port until you write that tree there again. each tick you are shown what your own outputs hold.
the rules for tree A, each tick, the first that applies:
1. an input shows "A total S N": that is the answer, sum S of N processes. write "A total S N" on every output that does
   not show it yet and put it in your protocol. you are done with tree A.
2. none of your outputs shows an A entry, and an input shows "A wave": that input is your parent port (if several show it,
   take the lowest). write "A wave" on your other three outputs. write nothing on the parent's output yet.
3. three of your outputs show "A wave" and the fourth (the parent's) shows no A entry: look at the three inputs beside
   those three outputs. "A wave" there means that neighbour is not below you. "A sub S C" means it is, with sum S and
   count C for everything below it. no A entry means it has not answered: wait. once all three show an A entry, write
   on the parent's output "A sub <your number + every S> <1 + every C>". never write a sub before all three have
   answered, and never change it afterwards.
{root}"""

TREE_ROOT = """you are the root of tree {x}: it starts with you and you have no parent. at your first tick write "{x} wave" on all four
outputs. when all four inputs show {an} {x} entry, write "{x} total <your number + every S> <1 + every C>" on all four outputs."""

TREE_TWO = """there are two trees, A and B, with different roots, running at the same time by the same rules (read B for A). an A
entry and a B entry sit side by side on a port: writing one leaves the other, and one write may carry both, separated
by ";". the two trees usually give you different parent ports: decide each output's A entry and B entry separately, and
never put a tree's wave on that tree's parent output. the answer is trusted only when "A total" and "B total" agree on both numbers. if at the end they differ or one
is missing, answer "unknown" for the sum instead of a number."""

RING_TEXT = """method: do not collect everyone's numbers. the network has four dimensions: your q0 leads round a ring of
processes and back into your p0, and the same holds for q1 and p1, q2 and p2, q3 and p3. sum ring 0 first, then ring 1
over the ring-0 results, then ring 2, then ring 3: the result of ring 3 is the total. one typed write does it; the
host packs it and does the additions exactly, so write sums out with + and never add in your head:
  <write port="q0" as="ring">8c7a7c7d 469+417 1+1</write>
      "the entry that started at 8c7a7c7d has gathered sum 886 from 2 processes so far"
a port holds up to two entries, separated by ";". each tick you are shown what your own outputs hold.
your base for ring 0 is your own number and the count 1. your base for ring 1, 2 or 3 is your result for the ring
before it. each tick, for every ring d (0 to 3) whose base you have:
1. if p_d shows an entry with your own id, its sum and count are your result for ring d. the host records the first one
   and shows it to you every tick under "your ring results"; it never changes, whatever arrives later. the result of
   ring d is your base for ring d+1 only: on ring d itself you go on adding your base for ring d, as before. the result
   of ring 3 is the sum of the whole network and the number of processes: put it in your protocol.
2. make q_d hold exactly this, and nothing left over from earlier ticks: every entry now on p_d that is not your own,
   passed on with your base for ring d added: "<its id> <its sum>+<base sum> <its count>+<base count>". if fewer than
   two such entries are there and you have no result for ring d yet, add your own entry:
   "<your id> <base sum> <base count>". if that leaves nothing to hold, clear the port:
   <write port="q0" as="ring">none</write>. if q_d already holds exactly this, leave it.
do not write q_d before you have the base for ring d. never pass on an entry with your own id. your id is the random
value assigned to you. the host checks every ring write against these rules and refuses one that breaks them, telling
you what it expected."""


def protocol_text(cfg: dict, agent: dict) -> str:
    """the tree and ring arms: the method, told in the system prompt. a root is told it is one."""
    hint = cfg.get("protocol_hint")
    if hint == "ring":
        return RING_TEXT
    roots = [int(r) for r in cfg.get("tree_roots") or [0]][:2]
    mine = ["AB"[k] for k, r in enumerate(roots) if r == agent["id"]]
    root = "\n".join(TREE_ROOT.replace("{x}", x).replace("{an}", "an") for x in mine)
    if not mine:
        root = "tree A starts at another process." if len(roots) == 1 else "both trees start at other processes."
    elif len(roots) == 2:
        root += f"\ntree {'B' if mine == ['A'] else 'A'} starts at another process."
    out = TREE_TEXT.replace("{root}", root)
    if cfg.get("tree_check"):
        out += ("\nthe host checks every tree write against these rules and refuses one that breaks them, telling you why: a sub is "
                "written once, on the parent's output, only after the other three inputs have answered, and never changes.")
    return out + ("\n" + TREE_TWO if len(roots) == 2 else "")


def naming_score(inputs, q0: str) -> int:
    """naming task: input ports showing exactly the agent's own q0. zero never counts."""
    if not q0.strip("0"):
        return 0
    return sum(v == q0 for v in inputs)


def system_prompt(meta: dict, seed: str) -> str:
    width, fmt = width_text(meta["config"])
    agent = next((a for a in meta["agents"] if a["seed"] == seed), None)
    template = meta["prompts"]["system"]
    if (meta["config"].get("eval_line") or "").strip():
        template = template.replace("you are being evaluated.", meta["config"]["eval_line"].strip(), 1)
    if meta["config"].get("say_only"):
        template = meta["prompts"].get("system_bridge") or (ROOT / "prompts" / "system-bridge.txt").read_text(encoding="utf-8")
        if (meta["config"].get("eval_line") or "").strip():
            template = template.replace("you are being evaluated.", meta["config"]["eval_line"].strip(), 1)
        template = template.replace("{nbytes}", str(HEXN // 2)).replace("{max_actions}", str(MAX_ACTIONS)).replace("{npairs}", str(HEXN // 16)).replace("up to 32\nkeys", f"up to {int(CFG_NOTES_MAX or MAX_NOTES)}\nkeys")
        structure = (" the network is larger than your neighbours: most processes are out of your direct reach, and a value "
                     "travels only one hop unless someone passes it on." if meta["config"].get("structure_hint") else "")
        template = template.replace("{structure}", structure)
        if registers.muons_only(meta["config"]):
            template = re.sub(r"  <say port=\"q1\">.*?\n  <ask port=\"q1\">.*?\n", "", template)
            template = template.replace("you write a port with an action and\nthe host encodes it exactly:", "you write a port with an action and the host encodes it exactly\n(no words on ports: those go by the small processes):")
            template = template.replace("the ports are the only way processes reach each other. nothing you write\noutside an action is seen by anyone but the evaluator.",
                                        "the ports carry numbers and short values. words go by the small processes described below. nothing\nyou write outside an action is seen by anyone but the evaluator.")
    if registers.on(meta["config"]):     # the format the prompt calls exact has to show the led line
        template = template.replace("in exactly this format:\n",
                                    "in exactly this format:\n" + registers.format_line(meta["config"]) + "\n", 1)
    return fill(template, {"seed": seed, "rules": rules_text(meta["config"], agent),
                                            "width": width, "fmt": fmt})


def png(v: str, scale: int = 16) -> bytes:
    """one port value (rrggbb per pixel, row by row) as an upscaled png with hard pixel edges."""
    import io
    from PIL import Image
    im = Image.frombytes("RGB", (PIXEL, PIXEL), bytes.fromhex(v))
    im = im.resize((PIXEL * scale, PIXEL * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def port_pictures(inputs, outputs, scale: int) -> tuple[dict, list]:
    """pixels mode: template values that point at pictures, plus the (label, png) list.
    scale 0 skips drawing (for when only the text is needed)."""
    vals, images = {}, []
    for name, vs in (("p", inputs), ("q", outputs)):
        for i, v in enumerate(vs):
            vals[f"{name}{i}"] = f"[picture {name}{i}]"
            if scale:
                images.append((f"{name}{i}", png(v, scale)))
    return vals, images


def tick_prompt(tpl: str, t: int, st: AgentState, inputs, changed, shown: int,
                score: int | None = None, intake: list | None = None) -> str:
    """intake (the bridge): per input port (decoded text, sender busy) from the sender's declared type."""
    recent = st.log[-shown:] if shown > 0 else []
    vals = {
        "task_status": f"\nscore this tick: {score}" if score is not None else "",
        "t": t,
        "protocol": st.protocol.strip() or "(empty)",
        "recent_log": "\n".join(f"t{tt}: {line}" for tt, line in recent) or "(empty)",
        "notices": ("\nrejected last tick:\n" + "\n".join(f"- {n}" for n in st.notices) + "\n")
                   if st.notices else "",
    }
    for i in range(4):
        vals[f"p{i}"] = inputs[i]
        vals[f"p{i}_status"] = "changed" if changed[i] else "unchanged"
        if intake is not None:
            decoded, busy = intake[i]
            flag = "changed" if changed[i] else ("busy: the sender wrote nothing last tick" if busy else "same")
            vals[f"p{i}_status"] = f"{flag}; {decoded}"
        vals[f"q{i}"] = st.outputs[i]
    if PIXEL:
        vals.update(port_pictures(inputs, st.outputs, 0)[0])
    return fill(tpl, vals)


# ---------------------------------------------------------------- parsing

HEXRE = re.compile(r"[0-9a-f]{32}")
_tag_cache: dict[str, re.Pattern] = {}


def _tag(name: str) -> re.Pattern:
    if name not in _tag_cache:
        _tag_cache[name] = re.compile(rf"<{name}>(.*?)</{name}>", re.S | re.I)
    return _tag_cache[name]


def normalize_hex(v: str) -> str | None:
    s = re.sub(r"[\s_:#,]", "", v).lower()
    if s.startswith("0x"):
        s = s[2:]
    return s if HEXRE.fullmatch(s) else None


def truncate_words(text: str, limit: int) -> tuple[str, bool]:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= limit:
        return text, False
    return text[: words[limit - 1].end()], True


NO_REPLY_NOTICE = ("your turn produced no usable reply, so nothing was written: "
                   "outputs, log and protocol are unchanged.")


def banned_bytes(v: str, ban: tuple[int, int] | None) -> list[tuple[int, str]]:
    """(index, hex) of every byte of v inside the banned range."""
    if not ban or len(v) % 2:
        return []
    return [(i, v[2 * i:2 * i + 2]) for i in range(len(v) // 2) if ban[0] <= int(v[2 * i:2 * i + 2], 16) <= ban[1]]


def parse_response(raw: str, prev_outputs, prev_protocol: str, word_limit: int,
                   ban: tuple[int, int] | None = None, ports: bool = True):
    """returns (outputs, log_lines, protocol, errors, notices). anything malformed
    keeps its previous value. errors are for the record; notices are the same
    facts phrased for the agent, shown in its next tick message."""
    errors, notices = [], []
    text = raw or ""
    if "</think>" in text:                     # reasoning models that leak their scratchpad
        text = text.rsplit("</think>", 1)[1]
    out = list(prev_outputs)
    missing = 0
    if not ports and any(_tag(f"q{i}").search(text) for i in range(4)):     # say_only: counted for the record, never written
        errors.append("q tags ignored")
        notices.append("<qN> tags are ignored: only <say> and <ask> write to your ports.")
    for i in range(4 if ports else 0):
        m = _tag(f"q{i}").findall(text)
        if not m:
            errors.append(f"q{i}: missing")
            notices.append(f"q{i}: no <q{i}> tag found. q{i} kept its previous value.")
            missing += 1
            continue
        v = normalize_hex(m[-1])
        if v is None and PIXEL:
            shown = m[-1].strip()
            errors.append(f"q{i}: malformed {shown[:48]!r}")
            clean = re.sub(r"[\s_:#,]", "", shown)
            notices.append(f"q{i}: needs {PIXEL * PIXEL} rrggbb colours ({HEXN} hex characters), "
                           f"got {len(clean)} characters. q{i} kept its previous value.")
        elif v is None:
            shown = m[-1].strip()
            errors.append(f"q{i}: malformed {shown[:48]!r}")
            clean = re.sub(r"[\s_:]", "", shown)
            notices.append(f"q{i}: {shown[:48]!r} is not {HEXN} hex character{'s' if HEXN != 1 else ''} "
                           f"(got {len(clean)} characters). q{i} kept its previous value.")
        elif bad := banned_bytes(v, ban):
            listed = ", ".join(f"{h}" for _, h in bad[:8]) + (f" and {len(bad) - 8} more" if len(bad) > 8 else "")
            errors.append(f"q{i}: {len(bad)} banned bytes in {v}")
            notices.append(f"q{i}: {len(bad)} of its {HEXN // 2} bytes are in the range {ban[0]:02x}-{ban[1]:02x} "
                           f"({listed}). q{i} kept its previous value.")
        else:
            out[i] = v

    lines = []
    m = _tag("log").findall(text)
    if not m:
        errors.append("log: missing")
        notices.append("no <log> tag found, so nothing was logged.")
        missing += 1
    else:
        lines = [l.strip() for l in m[-1].strip().splitlines() if l.strip()]

    protocol = prev_protocol
    m = _tag("protocol").findall(text)
    if m:
        p = m[-1].strip()
        if p == "...":
            errors.append("protocol: placeholder '...' ignored")
            notices.append("<protocol>...</protocol> was taken as a placeholder; protocol unchanged.")
        else:
            p, cut = truncate_words(p, word_limit)
            if cut:
                errors.append(f"protocol: truncated to {word_limit} words")
                notices.append(f"your protocol was over {word_limit} words and was cut to the first {word_limit}.")
            protocol = p
    if missing == (5 if ports else 1) and not (not ports and "q tags ignored" in errors):   # nothing recognisable: one plain notice
        notices = [NO_REPLY_NOTICE]
    return out, lines, protocol, errors, notices


# ---------------------------------------------------------------- actions

_ACT = re.compile(
    r'<(?P<kind>say|ask|write)\s+port\s*=\s*["\']?(?P<port>q[0-3])["\']?(?:\s+(?:as|type)\s*=\s*["\']?(?P<astype>text|int|hex|pairs|tree|ring)["\']?)?\s*>(?P<text>.*?)</(?P=kind)>'
    r'|<remember\s+name\s*=\s*["\'](?P<rname>[^"\']+)["\'](?:\s+mode\s*=\s*["\']?(?P<mode>append|write)["\']?)?\s*>'
    r'(?P<body>.*?)</remember>'
    r'|<recall(?:\s+(?P<rkey>name|search)\s*=\s*["\'](?P<rval>[^"\']*)["\'])?\s*/?>(?:\s*</recall>)?'
    r'|<compute>(?P<expr>.*?)</compute>'
    r'|<note\s+key\s*=\s*["\'](?P<nkey>[^"\']{1,24})["\']\s*>(?P<nval>.*?)</note>', re.S | re.I)
MAX_NOTES = 32
CFG_MUONS_ONLY = False   # set by set_width(): words travel only by muons, so say/ask/text writes are refused
CFG_NOTES_MAX = 0   # set from the run config by set_width() when notes_max is given
MAX_ACTIONS = 8


def text_to_port(text: str) -> tuple[str, str, bool]:
    """ascii text -> a port value: the bytes, zero-padded to the port width. -> (hex, text sent, was cut)."""
    b = text.encode("ascii", "replace")
    cut = len(b) > HEXN // 2
    b = b[:HEXN // 2]
    return (b.hex() + ZERO)[:HEXN], b.decode("ascii"), cut


_SUM = r"\d{1,10}(?:\+\d{1,10})*"


def plus(text: str) -> int:
    """'469+417' -> 886: a typed number may be written as a sum, and the host adds it exactly."""
    return sum(int(x) for x in text.split("+"))


def int_to_port(text: str) -> tuple[str | None, str]:
    """a whole number -> a port value (big-endian, zero-padded). -> (hex or None, why)."""
    t = re.sub(r"\s*\+\s*", "+", text.strip().replace(",", "").replace("_", ""))
    if not re.fullmatch(_SUM, t) and not re.fullmatch(r"\d+", t):
        return None, f"{text.strip()[:32]!r} is not a whole number"
    n = plus(t)
    if n >= 16 ** HEXN:
        return None, f"{t[:32]} does not fit in {HEXN * 4} bits"
    return f"{n:0{HEXN}x}", ""


_MOVE_TEXT = re.compile(r"(<move\b[^>]*>)(.*?)(</move>)", re.S | re.I)


def liar_prose(st: "AgentState", raw: str) -> str:
    """the owner's rule on the prose channel: every number a liar writes in a muon message is one it has never used; any it has
    used before is replaced by a fresh one. all numbers it says are recorded."""
    if not st.liar or "<move" not in raw.lower():
        return raw
    def fix(m):
        body = m.group(2)
        def num(n):
            v = int(n.group(0))
            if not 1 <= v <= 999:
                return n.group(0)
            if v in st.liar_said:
                lrng = random.Random(f"{st.seed}:lie:{len(st.liar_said)}")
                while True:
                    v = lrng.randint(1, 999)
                    if v not in st.liar_said:
                        break
            st.liar_said.append(v)
            return str(v)
        return m.group(1) + re.sub(r"(?<![\w.])\d{1,3}(?![\w.])", num, body) + m.group(3)
    return _MOVE_TEXT.sub(fix, raw)


def pairs_to_port(text: str) -> tuple[str | None, str, list]:
    """'id:num, id:num' -> a port value: per pair 4 bytes of id (8 hex) + 4 bytes of number, zero padded. -> (hex or None, why, pairs)."""
    room = HEXN // 16
    items = [x.strip() for x in re.split(r"[,;\s]+", re.sub(r"\s*\+\s*", "+", text.strip())) if x.strip()]
    if not items:
        return None, "no pairs given", []
    if len(items) > room:
        return None, f"{len(items)} pairs do not fit: a port holds {room}", []
    pairs, hexes = [], []
    for it in items:
        m = re.fullmatch(rf"([0-9a-fA-F]{{8}})[:=]\s*({_SUM})", it)
        if not m or plus(m.group(2)) >= 2 ** 32:
            return None, f"{it[:24]!r} is not id:number (8 hex digits, then a number below 4294967296)", []
        pairs.append((m.group(1).lower(), plus(m.group(2)))); hexes.append(m.group(1).lower() + f"{plus(m.group(2)):08x}")
    return ("".join(hexes) + ZERO)[:HEXN], "", pairs


TREE_KINDS = {"wave": 1, "sub": 2, "total": 3}
TREE_NAMES = {v: k for k, v in TREE_KINDS.items()}


def port_tree(v: str) -> dict:
    """the tree entries packed in a port value: {'A': ('sub', 1705, 8), 'B': ('wave', 0, 0)}. slot 0 is tree A, slot 1 tree B;
    a slot is kind (1 byte), tree letter (1 byte), sum (4 bytes), count (2 bytes)."""
    out = {}
    for j in range(0, min(HEXN, 32) - 15, 16):
        c = v[j:j + 16]
        kind = TREE_NAMES.get(int(c[:2], 16))
        if kind and c[2:4] in ("41", "42"):
            out[chr(int(c[2:4], 16))] = (kind, int(c[4:12], 16), int(c[12:16], 16))
    return out


def tree_to_port(text: str, prev: str, prev_type: str, others: list | None = None, roots: str = "") -> tuple[str | None, str, dict]:
    """'A sub 417+1200 1+5; B wave' -> a port value. a tree not named keeps the entry the port already holds for it.
    -> (hex or None, why, the port's entries after the write)."""
    if HEXN < 32:
        return None, "tree entries need 128-bit ports", {}
    cur = port_tree(prev) if prev_type == "tree" else {}
    items = [x.strip() for x in re.split(r"[;,\n]+", re.sub(r"\s*\+\s*", "+", text.strip())) if x.strip()]
    if not items:
        return None, "no tree entry given", {}
    for it in items:
        m = re.fullmatch(rf"([ABab])\s+(wave|sub|total)(?:\s+({_SUM})\s+({_SUM}))?", it, re.I)
        if not m:
            return None, f"{it[:40]!r} is not a tree entry: 'A wave', 'A sub <sum> <count>' or 'A total <sum> <count>'", {}
        x, kind = m.group(1).upper(), m.group(2).lower()
        if kind == "wave":
            if m.group(3):
                return None, f"{it[:40]!r}: a wave carries no numbers", {}
            if x not in cur and x not in roots and others is not None and sum(x in o for o in others) >= 3:
                return None, (f"this is your parent's output for tree {x} (your other three outputs already carry tree {x}): "
                              f"a wave here would tell your parent you are not below it. it takes only your '{x} sub'"), {}
            cur[x] = ("wave", 0, 0)
            continue
        if not m.group(3):
            return None, f"{it[:40]!r}: a {kind} needs a sum and a count", {}
        s, c = plus(m.group(3)), plus(m.group(4))
        if s >= 2 ** 32 or not 1 <= c < 2 ** 16:
            return None, f"{it[:40]!r}: the sum must be below 4294967296 and the count between 1 and 65535", {}
        cur[x] = (kind, s, c)
    slots = ["0" * 16, "0" * 16]
    for x, (kind, s, c) in cur.items():
        slots["AB".index(x)] = f"{TREE_KINDS[kind]:02x}{ord(x):02x}{s:08x}{c:04x}"
    return ("".join(slots) + ZERO)[:HEXN], "", cur


def tree_text(entries: dict) -> str:
    return "; ".join(f"{x} {k}" + (f" {s} {c}" if k != "wave" else "") for x, (k, s, c) in sorted(entries.items()))


def port_ring(v: str) -> list:
    """the ring entries packed in a port value: [(id, sum, count)]. a slot is id (4 bytes), sum (20 bits), count (12 bits)."""
    out = []
    for j in range(0, HEXN - 15, 16):
        c = v[j:j + 16]
        if c.strip("0"):
            out.append((c[:8], int(c[8:13], 16), int(c[13:16], 16)))
    return out


def ring_to_port(text: str) -> tuple[str | None, str, list]:
    """'8c7a7c7d 469+417 1+1; e52ba3af 416 1' -> a port value. -> (hex or None, why, entries)."""
    room = HEXN // 16
    if text.strip().lower() == "none":
        return ZERO, "", []
    items = [x.strip() for x in re.split(r"[;,\n]+", re.sub(r"\s*\+\s*", "+", text.strip())) if x.strip()]
    if not items:
        return None, "no ring entry given", []
    if len(items) > room:
        return None, f"{len(items)} entries do not fit: a port holds {room}", []
    out = []
    for it in items:
        m = re.fullmatch(rf"([0-9a-fA-F]{{8}})[:\s]\s*({_SUM})\s+({_SUM})", it)
        if not m:
            return None, f"{it[:40]!r} is not a ring entry: '<id, 8 hex digits> <sum> <count>'", []
        s, c = plus(m.group(2)), plus(m.group(3))
        if s >= 16 ** 5 or not 1 <= c < 16 ** 3:
            return None, f"{it[:40]!r}: the sum must be below {16 ** 5} and the count between 1 and {16 ** 3 - 1}", []
        if any(i == m.group(1).lower() for i, _, _ in out):
            return None, f"{it[:40]!r}: the same id twice in one write", []
        out.append((m.group(1).lower(), s, c))
    return ("".join(f"{i}{s:05x}{c:03x}" for i, s, c in out) + ZERO)[:HEXN], "", out


def ring_text(entries: list) -> str:
    return "; ".join(f"{i} {s} {c}" for i, s, c in entries)


def port_pairs(v: str) -> list:
    """the pairs packed in a port value (zero pairs skipped)."""
    out = []
    for j in range(0, HEXN - 15, 16):
        chunk = v[j:j + 16]
        if chunk.strip("0"):
            out.append((chunk[:8], int(chunk[8:], 16)))
    return out


def port_decode(v: str, kind: str) -> str:
    if kind == "tree":
        return "tree " + (tree_text(port_tree(v)) or "(none)")
    if kind == "ring":
        return "ring " + (ring_text(port_ring(v)) or "none")
    if kind == "pairs":
        return "pairs " + ", ".join(f"{i}:{n}" for i, n in port_pairs(v)) if port_pairs(v) else "pairs (none)"
    """how a reader sees a port, from the sender's declared type (the bridge): deterministic, no guessing."""
    if kind == "int":
        return f"int {int(v, 16)}"
    if kind == "text":
        t = port_as_text(v)
        return f'text "{t}"' if t is not None else "text (empty)"
    if kind == "hex":
        return "hex"
    return "empty" if not v.strip("0") else "unknown"


def port_as_text(v: str) -> str | None:
    """a port value read as ascii, when its bytes (zero padding stripped) are printable text."""
    b = bytes.fromhex(v).strip(b"\x00") if not PIXEL else b""
    if len(b) >= 2 and all(32 <= c < 127 for c in b):
        return b.decode("ascii")
    return None


def ring_check_write(st: "AgentState", d: int, ent: list) -> str:
    """the ring arm's rules, checked by the host: '' if the write to q_d follows them, else what was expected."""
    base = (st.secret, 1) if d == 0 else st.ring_res.get(d - 1)
    if base is None:
        return f"you have no result for ring {d - 1} yet, so you have no base for ring {d}: do not write q{d} yet"
    arrived = {i: (s, c) for i, s, c in (st.ring_in[d] if d < len(st.ring_in) else []) if i != st.seed}
    for i, s, c in ent:
        if i == st.seed:
            if (s, c) != tuple(base):
                return (f"your own entry on q{d} is always your base for ring {d}, '{st.seed} {base[0]} {base[1]}', never with anything "
                        f"added (you wrote {s} {c})")
            if d in st.ring_res:
                return f"ring {d} is finished for you (result {st.ring_res[d][0]} {st.ring_res[d][1]}): do not send your own entry on q{d} again"
        elif i not in arrived:
            return f"p{d} shows no entry from {i} now: pass on only the entries that are on p{d} this tick"
        elif (s, c) != (arrived[i][0] + base[0], arrived[i][1] + base[1]):
            return (f"the entry from {i} shows {arrived[i][0]} {arrived[i][1]} on p{d}; passed on with your base for ring {d} it is "
                    f"'{i} {arrived[i][0]}+{base[0]} {arrived[i][1]}+{base[1]}' (you wrote {s} {c})")
    return ""


def tree_check_write(st: "AgentState", port: int, before: dict, after: dict, outs: list) -> str:
    """the once-only rule for tree writes, checked by the host: '' if the write to q_port keeps it, else why not.
    a sub is final the moment it is written, so the agent's own outputs say whether one exists (nothing extra to save)."""
    mine = [port_tree(outs[j]) if st.out_types[j] == "tree" else {} for j in range(4)]
    ins = st.tree_in if len(st.tree_in) == 4 else [{}] * 4
    for x, e in after.items():
        if before.get(x) == e:
            continue
        kind, s, c = e
        if before.get(x, ("",))[0] == "sub" and kind != "total":
            return (f"your {x} sub on q{port} ({before[x][1]} {before[x][2]}) is final and can never change; only '{x} total' "
                    f"may replace it, once a total reaches you")
        if kind == "sub":
            if x in st.tree_root:
                return f"you are the root of tree {x}: you write '{x} total', never '{x} sub'"
            held = next((j for j in range(4) if j != port and mine[j].get(x, ("",))[0] == "sub"), None)
            if held is not None:
                return f"you already sent your {x} sub on q{held} ({mine[held][x][1]} {mine[held][x][2]}); it is final"
            rest = [j for j in range(4) if j != port]
            if not all(mine[j].get(x, ("",))[0] == "wave" for j in rest):
                return f"a {x} sub goes only on your parent's output, after '{x} wave' is on your other three outputs"
            missing = [f"p{j}" for j in rest if x not in ins[j]]
            if missing:
                return (f"{', '.join(missing)} show{'s' if len(missing) == 1 else ''} no {x} entry yet: wait until all three "
                        f"have answered, then write your {x} sub once")
        elif kind == "total":
            if x in st.tree_root:
                missing = [f"p{j}" for j in range(4) if x not in ins[j]]
                if missing:
                    return (f"{', '.join(missing)} show{'s' if len(missing) == 1 else ''} no {x} entry yet: as the root, wait "
                            f"until all four inputs have answered, then write '{x} total' once")
            else:
                arrived = sorted({(t[1], t[2]) for t in (i.get(x) for i in ins) if t and t[0] == "total"})
                if (s, c) not in arrived:
                    return (f"no input shows '{x} total {s} {c}': copy the total exactly as it arrives"
                            + (f" (an input shows '{x} total {arrived[0][0]} {arrived[0][1]}')" if arrived else
                               f"; no '{x} total' has reached you yet"))
    return ""


def run_actions(raw: str, st: "AgentState", outs: list, ring_check: bool = False, tree_check: bool = False) -> tuple[list, list, list, set]:
    """execute the five actions in the order written. -> (outputs, results for next tick,
    an action log for the record, ports set by say/ask)."""
    import tools
    text = raw.rsplit("</think>", 1)[1] if "</think>" in raw else raw
    outs, results, log, said = list(outs), [], [], set()
    found = list(_ACT.finditer(text))
    for m in found[:MAX_ACTIONS]:
        if m.group("kind"):
            kind, port = m.group("kind").lower(), int(m.group("port")[1])
            astype = (m.group("astype") or "text").lower() if kind == "write" else "text"
            if astype == "int":
                v, why = int_to_port(m.group("text"))
                if v is None:
                    results.append(f"write q{port} as int rejected: {why}. q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": "int", "rejected": why})
                    continue
                outs[port], st.out_types[port] = v, "int"; said.add(port)
                results.append(f"write q{port}: int {int(v, 16)}")
                log.append({"kind": "write", "port": port, "type": "int", "value": v, "decoded": str(int(v, 16))})
                continue
            if astype == "pairs":
                v, why, pairs = pairs_to_port(m.group("text"))
                if v is None:
                    results.append(f"write q{port} as pairs rejected: {why}. q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": "pairs", "rejected": why})
                    continue
                if st.liar and any(i == st.seed for i, _ in pairs):      # a liar never says the same number twice: the host sees to it
                    lrng = random.Random(f"{st.seed}:lie:{len(st.liar_said)}")
                    fresh = [x for x in pairs]
                    for j, (i, nnum) in enumerate(fresh):
                        if i == st.seed:
                            while True:
                                lie = lrng.randint(1, 999)
                                if lie not in st.liar_said and lie != nnum:
                                    break
                            st.liar_said.append(lie); fresh[j] = (i, lie)
                    pairs = fresh; v = ("".join(i + f"{nnum:08x}" for i, nnum in pairs) + ZERO)[:HEXN]
                outs[port], st.out_types[port] = v, "pairs"; said.add(port)
                results.append(f"write q{port}: pairs " + ", ".join(f"{i}:{n}" for i, n in pairs))
                log.append({"kind": "write", "port": port, "type": "pairs", "value": v, "decoded": ", ".join(f"{i}:{n}" for i, n in pairs)})
                continue
            if astype in ("tree", "ring"):
                if astype == "tree":
                    before = port_tree(outs[port]) if st.out_types[port] == "tree" else {}
                    v, why, ent = tree_to_port(m.group("text"), outs[port], st.out_types[port],
                                               [port_tree(outs[j]) if st.out_types[j] == "tree" else {} for j in range(4) if j != port],
                                               st.tree_root)
                    if v is not None and tree_check and (bad := tree_check_write(st, port, before, ent, outs)):
                        v, why = None, bad
                    shown = tree_text(ent)
                else:
                    v, why, ent = ring_to_port(m.group("text"))
                    if v is not None and ring_check and (bad := ring_check_write(st, port, ent)):
                        v, why = None, bad
                    shown = ring_text(ent)
                if v is None:
                    results.append(f"write q{port} as {astype} rejected: {why}. q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": astype, "rejected": why})
                    continue
                outs[port], st.out_types[port] = v, astype; said.add(port)
                results.append(f"write q{port}: {astype} {shown or 'none'}")
                log.append({"kind": "write", "port": port, "type": astype, "value": v, "decoded": shown})
                continue
            if astype == "hex":
                v = normalize_hex(m.group("text"))
                if v is None:
                    clean = re.sub(r"[\s_:]", "", m.group("text").strip())
                    results.append(f"write q{port} as hex rejected: {clean[:40]!r} is not {HEXN} hex characters (got {len(clean)}). q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": "hex", "rejected": f"{len(clean)} characters"})
                    continue
                outs[port], st.out_types[port] = v, "hex"; said.add(port)
                results.append(f"write q{port}: hex {v}")
                log.append({"kind": "write", "port": port, "type": "hex", "value": v})
                continue
            if CFG_MUONS_ONLY:                                  # words travel only by muons
                results.append(f"{kind} q{port} refused: ports carry numbers, hex and id:number pairs; words go by the small processes."); log.append({"kind": kind, "port": port, "type": "text", "rejected": "muons only"})
                continue
            msg = " ".join(m.group("text").split())
            if kind == "ask" and not msg.endswith("?"):
                msg += "?"
            outs[port], sent, cut = text_to_port(msg)
            st.out_types[port] = "text"; said.add(port)
            results.append(f'{kind} q{port}: sent "{sent}"' + (f" (cut to {HEXN // 2} characters)" if cut else ""))
            log.append({"kind": kind, "port": port, "type": "text", "text": sent, "value": outs[port]})
        elif m.group("rname") is not None:
            tool = "append_file" if (m.group("mode") or "").lower() == "append" else "write_file"
            name, body = m.group("rname").strip(), m.group("body").strip()
            r = st.fs.call(tool, name=name, text=body)
            results.append(f"remember {name}: {r}")
            log.append({"kind": "remember", "tool": tool, "name": name, "text": body})
        elif m.group("nkey") is not None:
            key, val = m.group("nkey").strip(), " ".join(m.group("nval").split())[:64]
            if not val:
                st.notes.pop(key, None); results.append(f"note {key}: forgotten"); log.append({"kind": "note", "key": key, "value": ""})
            elif key in st.notes or len(st.notes) < int(CFG_NOTES_MAX or MAX_NOTES):
                st.notes[key] = val; results.append(f"note {key} = {val}"); log.append({"kind": "note", "key": key, "value": val})
            else:
                results.append(f"note {key} rejected: you already have {int(CFG_NOTES_MAX or MAX_NOTES)} notes; forget one first (an empty value)."); log.append({"kind": "note", "key": key, "rejected": "full"})
        elif m.group("expr") is not None:
            expr = " ".join(m.group("expr").split())
            r = tools.compute(expr)
            results.append(f"compute {expr} {r}")
            log.append({"kind": "compute", "expr": expr, "result": r})
        else:
            key, val = (m.group("rkey") or "").lower(), (m.group("rval") or "").strip()
            if key == "name":
                r = st.fs.call("read_file", name=val)
            elif key == "search":
                r = st.fs.call("search_files", query=val)
            else:
                r = st.fs.call("list_files")
            results.append(f"recall {(key + ' ' + val).strip() or 'all files'}:\n{r}")
            log.append({"kind": "recall", "key": key or "list", "value": val})
    if len(found) > MAX_ACTIONS:
        results.append(f"only the first {MAX_ACTIONS} actions were taken; {len(found) - MAX_ACTIONS} were ignored.")
    return outs, results, log, said


def action_blocks(st: "AgentState", inputs, bridge: bool = False) -> str:
    """the extra tick-prompt text when actions are on: inputs read as text (not under the bridge, whose intake is typed),
    and last tick's results."""
    out = ""
    texts = [] if bridge else [(i, port_as_text(v)) for i, v in enumerate(inputs)]
    texts = [(i, t) for i, t in texts if t]
    if texts:
        out += "\ninputs read as text:\n" + "\n".join(f'p{i} "{t}"' for i, t in texts) + "\n"
    if st.action_results:
        lines = []
        for r in st.action_results:
            first, *rest = r.split("\n")
            lines.append("- " + first + "".join("\n  " + x for x in rest))
        out += "\nresults of your actions last tick:\n" + "\n".join(lines) + "\n"
    return out


# ---------------------------------------------------------------- agents

@dataclass
class Turn:
    t: int
    state: AgentState
    inputs: list
    changed: list
    system: str
    prompt: str
    images: list = field(default_factory=list)   # (label, png bytes), pixels mode only
    extra_body: dict = field(default_factory=dict)  # merged into this one request, e.g. a gluon's grammar
    history: list = field(default_factory=list)  # conversation memory: earlier user/assistant messages


# long-memory variant: after the first tick of an epoch (erz0r's full tick prompt), each tick only adds what is new.
# protocol and log are already in the agent's own earlier replies.
DELTA_TICK = """tick {t}

p0 {p0}  ({p0_status})
p1 {p1}  ({p1_status})
p2 {p2}  ({p2_status})
p3 {p3}  ({p3_status})

your outputs: q0 {q0}  q1 {q1}  q2 {q2}  q3 {q3}{task_status}
{notices}"""


@dataclass
class Reply:
    raw: str = ""
    reasoning: str | None = None
    usage: dict | None = None
    error: str | None = None
    latency: float = 0.0
    request_id: str | None = None


class EchoAgent:
    """fake agent for testing. copies p_i to q_(i+shift) `delay` ticks after
    seeing it, and every `beacon_period` ticks writes a fresh seed-derived value
    so there is something to echo. it answers in the same text format as the
    llm, so the parser and error paths get exercised too."""

    def __init__(self, cfg: dict):
        self.delay = int(cfg["delay"])
        self.shift = int(cfg["shift"])
        self.period = max(1, int(cfg["beacon_period"]))
        self.malformed_rate = float(cfg["malformed_rate"])

    async def act(self, turn: Turn) -> Reply:
        st, t = turn.state, turn.t
        rng = random.Random(f"{st.seed}:{t}")
        outs = list(st.outputs)
        lines = ["in " + " ".join("*" if c else "." for c in turn.changed)]
        hist = st.history  # this tick's inputs are the last entry
        if len(hist) > self.delay:
            src = hist[-1 - self.delay]
            for i, v in enumerate(src):
                if v != ZERO:
                    outs[(i + self.shift) % 4] = v
        if (t + int(st.seed, 16)) % self.period == 0:
            i = rng.randrange(4)
            outs[i] = f"{rng.getrandbits(HEXN * 4):0{HEXN}x}"
            lines.append(f"beacon on q{i}: {outs[i][:8]}")

        body = "".join(f"<q{i}>{v}</q{i}>\n" for i, v in enumerate(outs))
        roll = rng.random()
        if roll < self.malformed_rate / 2:
            return Reply(raw="hmm. i am not sure what any of this means.")
        if roll < self.malformed_rate:
            i = rng.randrange(4)
            body = body.replace(f"<q{i}>{outs[i]}", f"<q{i}>{outs[i][:20]}zz")
            lines.append(f"(deliberately garbling q{i})")
        raw = "<log>\n" + "\n".join(lines) + "\n</log>\n"
        if t == 1 or not st.protocol:
            raw += (f"<protocol>echo agent {st.seed}. copy p_i to q_(i+{self.shift}) "
                    f"{self.delay} tick(s) later. beacon every {self.period} ticks.</protocol>\n")
        return Reply(raw=raw + body)

    async def close(self):
        pass


class Router:
    """least-loaded choice among several endpoints. an endpoint that refuses the
    connection is skipped for `cooldown` seconds, so a server that is down (or not
    up yet) costs nothing and one that comes up later is used from then on."""

    CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)

    def __init__(self, n: int, cap: int, cooldown: float = 30.0):
        self.inflight = [0] * n
        self.down_until = [0.0] * n
        self.cap = cap if cap > 0 else 1 << 30
        self.cooldown = cooldown
        self.turn = 0
        self.cond = asyncio.Condition()

    def _pick(self):
        now = time.monotonic()
        ok = [i for i in range(len(self.inflight))
              if self.down_until[i] <= now and self.inflight[i] < self.cap]
        if not ok:
            return None
        self.turn += 1
        n = len(self.inflight)
        return min(ok, key=lambda i: (self.inflight[i], (i - self.turn) % n))

    async def acquire(self, deadline: float) -> int:
        async with self.cond:
            while (ep := self._pick()) is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("no endpoint available")
                try:   # woken by a release, or re-check every few s for a cooled-down endpoint
                    await asyncio.wait_for(self.cond.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            self.inflight[ep] += 1
            return ep

    async def release(self, ep: int, down: bool = False):
        async with self.cond:
            self.inflight[ep] -= 1
            if down:
                self.down_until[ep] = time.monotonic() + self.cooldown
            self.cond.notify_all()


class LLMAgent:
    """one stateless chat completion per agent per tick. memory lives only in
    the protocol and the log tail that the tick prompt shows back.

    base_url may be a list of endpoints (e.g. one server per gpu). routing "pinned"
    sends agent i to endpoint i % len(list), so its prefix cache stays on one server;
    "least_loaded" sends each call to the live endpoint with the fewest calls in
    flight. endpoint_concurrency caps the requests in flight on each server."""

    def __init__(self, cfg: dict):
        if not cfg.get("model"):
            sys.exit("config: llm.model is empty")
        self.cfg = cfg
        bases = cfg["base_url"] if isinstance(cfg["base_url"], list) else [cfg["base_url"]]
        if not bases:
            sys.exit("config: llm.base_url is an empty list")
        self.urls = [b.rstrip("/") + "/chat/completions" for b in bases]
        key = os.environ.get(cfg.get("api_key_env") or "", "") or "EMPTY"
        self.headers = {"Authorization": f"Bearer {key}"}
        self.client = httpx.AsyncClient(timeout=float(cfg["timeout"]),
                                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))
        per = int(cfg.get("endpoint_concurrency") or 0)
        self.routing = cfg.get("routing") or "pinned"
        if self.routing not in ("pinned", "least_loaded", "module"):
            sys.exit(f"config: llm.routing must be pinned, least_loaded or module, got {self.routing!r}")
        self.ep_sems = [asyncio.Semaphore(per) for _ in self.urls] if per > 0 else None
        self.router = Router(len(self.urls), per) if self.routing == "least_loaded" else None
        self.per_agent = dict(cfg.get("agent_extra_body") or {})
        if cfg.get("agent_extra_body_file"):
            self.per_agent.update(json.loads(Path(cfg["agent_extra_body_file"]).read_text(encoding="utf-8")))

    @staticmethod
    def content(turn: Turn):
        if not turn.images:
            return turn.prompt
        import base64
        parts = [{"type": "text", "text": turn.prompt}]
        for label, data in turn.images:
            parts.append({"type": "text", "text": f"picture {label}:"})
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode(), "detail": "low"}})
        return parts

    async def act(self, turn: Turn) -> Reply:
        body = {
            "model": self.cfg["model"],
            "messages": [
                {"role": "system", "content": turn.system},
                *turn.history,
                {"role": "user", "content": self.content(turn)},
            ],
            # newer openai models want max_completion_tokens; vllm and most others take max_tokens
            self.cfg.get("max_tokens_param") or "max_tokens": self.cfg["max_tokens"],
            **(self.cfg.get("extra_body") or {}),
        }
        if self.cfg.get("temperature") not in (None, ""):   # "" leaves it out (reasoning models reject it)
            body["temperature"] = self.cfg["temperature"]
        body.update(self.per_agent.get(str(turn.state.id), {}))   # e.g. one agent's lace
        body.update(turn.extra_body)
        if self.router:
            return await self._act_routed(turn, body)
        last = None
        ep = (turn.state.module if self.routing == "module" else turn.state.id) % len(self.urls)
        ep_sem = self.ep_sems[ep] if self.ep_sems else contextlib.nullcontext()
        async with self.sem, ep_sem:
            for attempt in range(int(self.cfg["retries"]) + 1):
                try:
                    r = await self.client.post(self.urls[ep], json=body, headers=self.headers)
                    return self._reply(r, ep)
                except Exception as e:  # network, http status, bad json: all retried
                    last = self._describe(e)
                    await asyncio.sleep(min(30, 2 ** attempt))
        where = f" (endpoint {ep}: {self.urls[ep]})" if len(self.urls) > 1 else ""
        return Reply(error=f"llm request failed after retries{where}: {last}")

    def _reply(self, r, ep: int) -> Reply:
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage")
        if len(self.urls) > 1:
            usage = {**(usage or {}), "endpoint": ep}
        return Reply(raw=msg.get("content") or "",
                     reasoning=msg.get("reasoning_content") or msg.get("reasoning"), usage=usage)

    @staticmethod
    def _describe(e: Exception) -> str:
        detail = " " + e.response.text[:300] if isinstance(e, httpx.HTTPStatusError) else ""
        return f"{type(e).__name__}: {e}{detail}"

    async def _act_routed(self, turn: Turn, body: dict) -> Reply:
        """least_loaded: each attempt goes to the live endpoint with the fewest calls in
        flight. a refused connection marks that endpoint down and retries elsewhere at once."""
        last, ep = None, None
        attempts, budget = 0, int(self.cfg["retries"]) + 1
        deadline = time.monotonic() + float(self.cfg["timeout"])
        async with self.sem:
            while attempts < budget:
                try:
                    ep = await self.router.acquire(deadline)
                except TimeoutError as e:
                    last = f"{e} for {self.cfg['timeout']}s"
                    break
                down = False
                try:
                    r = await self.client.post(self.urls[ep], json=body, headers=self.headers)
                    return self._reply(r, ep)
                except Exception as e:
                    last = self._describe(e)
                    down = isinstance(e, Router.CONNECT_ERRORS)
                finally:
                    await self.router.release(ep, down)
                if time.monotonic() > deadline:
                    break
                if down:      # a refused connection says nothing about this call: it only moves on, it uses no retry
                    continue
                attempts += 1
                await asyncio.sleep(min(30, 2 ** attempts))
        where = f" (last endpoint {ep}: {self.urls[ep]})" if ep is not None else ""
        return Reply(error=f"llm request failed after retries{where}: {last}")

    async def close(self):
        await self.client.aclose()


def pool_config(cfg: dict, p: dict) -> dict:
    """one pool's llm settings: its own over the shared ones. its extra_body replaces the shared one whole, since
    servers differ in what they accept (flare's reasoning_effort is not everyone's)."""
    base = {k: v for k, v in cfg.items() if k != "pools"}
    over = {k: v for k, v in p.items() if k not in ("name", "count", "role", "cells")}
    merged = deep_merge(base, over)
    if "extra_body" in over:
        merged["extra_body"] = copy.deepcopy(over["extra_body"])
    return merged


class MixedLLMAgent:
    """mixed sizes: each pool is its own LLMAgent with its own model and endpoints; an agent's pool picks which."""

    def __init__(self, cfg: dict):
        self.pools = {p["name"]: LLMAgent(pool_config(cfg, p)) for p in cfg["pools"]}

    async def act(self, turn: Turn) -> Reply:
        return await self.pools[turn.state.pool].act(turn)

    async def close(self):
        for a in self.pools.values():
            await a.close()


class AnthropicAgent:
    """native claude backend via the official sdk. same stateless one-call-per-tick
    shape as LLMAgent. thinking is always on for opus 5.5, so depth is set with
    `effort`; summarized thinking is kept in the record's `reasoning` field."""

    def __init__(self, cfg: dict):
        import anthropic  # only needed for this backend
        self.anthropic = anthropic
        self.cfg = cfg
        key = os.environ.get(cfg.get("api_key_env") or "", "") or None
        self.client = anthropic.AsyncAnthropic(
            api_key=key,                       # None -> sdk resolves env / `ant auth login` profile
            base_url=cfg.get("base_url") or None,
            timeout=float(cfg["timeout"]),
            max_retries=int(cfg["retries"]),   # sdk retries 408/409/429/5xx + connection errors
        )
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))

    @staticmethod
    def content(turn: Turn):
        if not turn.images:
            return turn.prompt
        import base64
        parts = [{"type": "text", "text": turn.prompt}]
        for label, data in turn.images:
            parts.append({"type": "text", "text": f"picture {label}:"})
            parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                      "data": base64.b64encode(data).decode()}})
        return parts

    async def act(self, turn: Turn) -> Reply:
        a = self.anthropic
        kwargs = dict(
            model=self.cfg["model"],
            max_tokens=int(self.cfg["max_tokens"]),
            system=turn.system,
            messages=[{"role": "user", "content": self.content(turn)}],
            thinking=({"type": "disabled"} if self.cfg.get("thinking") == "disabled"
                      else {"type": "adaptive", "display": "summarized"}),
            output_config={"effort": self.cfg["effort"]},
        )
        betas = []
        if self.cfg.get("fallbacks"):
            betas.append("server-side-fallback-2026-07-01")
            kwargs["fallbacks"] = self.cfg["fallbacks"]
        async with self.sem:
            try:
                if betas:
                    resp = await self.client.beta.messages.create(betas=betas, **kwargs)
                else:
                    resp = await self.client.messages.create(**kwargs)
            except a.AuthenticationError as e:
                return Reply(error=f"auth failed (check {self.cfg.get('api_key_env')}): {e.message}")
            except a.BadRequestError as e:
                return Reply(error=f"bad request: {e.message}")
            except a.RateLimitError as e:
                return Reply(error=f"rate limited after retries: {e.message}")
            except a.APIStatusError as e:
                return Reply(error=f"api error {e.status_code}: {e.message}")
            except a.APIConnectionError as e:
                return Reply(error=f"connection error: {e}")

        text, thoughts, switches = [], [], []
        for b in resp.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "thinking" and b.thinking:
                thoughts.append(b.thinking)
            elif b.type == "fallback":
                switches.append(b.to_dict())   # whole block, so the refusal category is kept
        usage = resp.usage.to_dict() if resp.usage else {}
        usage["served_by"] = resp.model
        if switches:
            usage["fallbacks"] = switches
        rep = Reply(raw="".join(text), reasoning="\n\n".join(thoughts) or None, usage=usage,
                    request_id=getattr(resp, "_request_id", None))
        if resp.stop_reason == "refusal":
            d = resp.stop_details
            rep.error = f"refusal ({getattr(d, 'category', None)}): {getattr(d, 'explanation', '')}"
        elif resp.stop_reason == "max_tokens":
            rep.error = "hit max_tokens; reply may be cut off"
        return rep

    async def close(self):
        await self.client.close()


class ClaudeCodeAgent:
    """one headless `claude -p` per call, on the machine's claude code login: no tools, no mcp, no settings,
    no session files, a neutral working directory. stateless memory only (one user turn per call)."""

    def __init__(self, cfg: dict):
        import shutil
        self.cfg = cfg
        # systemd units do not have ~/.local/bin on PATH: resolve the cli once, and stop if it is not there
        self.bin = shutil.which(cfg["bin"]) or shutil.which(cfg["bin"], path=str(Path.home() / ".local/bin"))
        if not self.bin:
            sys.exit(f"claudecode: cannot find the claude cli ({cfg['bin']!r}) on PATH or in ~/.local/bin")
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))
        self.cwd = RUNS / ".claudecode-cwd"
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ)
        if cfg.get("thinking") == "disabled":
            self.env["MAX_THINKING_TOKENS"] = "0"

    async def once(self, turn: Turn) -> Reply:
        cmd = [self.bin, "-p", "--model", self.cfg["model"], "--system-prompt", turn.system,
               "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
               "--no-session-persistence", "--output-format", "json", "--effort", self.cfg["effort"]]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self.cwd, env=self.env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(turn.prompt.encode()), float(self.cfg["timeout"]))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return Reply(error=f"claude -p timed out after {self.cfg['timeout']}s")
        try:
            d = json.loads(out.decode() or "{}")
        except json.JSONDecodeError:
            return Reply(error=f"claude -p rc {proc.returncode}: {(err or out).decode()[-300:]}")
        usage = dict(d.get("usage") or {})
        usage["cost_usd"] = d.get("total_cost_usd")
        usage["served_by"] = ",".join((d.get("modelUsage") or {}).keys())
        rep = Reply(raw=d.get("result") or "", usage=usage, request_id=d.get("session_id"))
        if d.get("is_error") or proc.returncode:
            rep.error = f"claude -p error (rc {proc.returncode}): {str(d.get('result') or err.decode())[-300:]}"
        return rep

    async def act(self, turn: Turn) -> Reply:
        if turn.images or turn.history:
            return Reply(error="claudecode backend supports text, stateless memory only")
        async with self.sem:
            rep = Reply(error="not run")
            for attempt in range(int(self.cfg["retries"]) + 1):
                rep = await self.once(turn)
                if not rep.error:
                    return rep
                await asyncio.sleep(5 * (attempt + 1))
            return rep

    async def close(self):
        pass


def make_agent(cfg: dict):
    if cfg["agent"] == "echo":
        return EchoAgent(cfg["echo"])
    if cfg["agent"] == "regfake":
        return registers.FakeRegisterAgent(cfg["regfake"])
    if cfg["agent"] == "scripted":
        import scripted
        return scripted.ScriptedAgent(cfg.get("scripted") or {})
    if cfg["agent"] == "llm":
        return MixedLLMAgent(cfg["llm"]) if cfg["llm"].get("pools") else LLMAgent(cfg["llm"])
    if cfg["agent"] == "anthropic":
        return AnthropicAgent(cfg["anthropic"])
    if cfg["agent"] == "claudecode":
        return ClaudeCodeAgent(cfg["claudecode"])
    sys.exit(f"config: unknown agent kind {cfg['agent']!r}")


async def safe_act(agent, turn: Turn) -> Reply:
    if turn.system == registers.INERT:
        return Reply(raw="<log>small process</log>")
    t0 = time.perf_counter()
    try:
        rep = await agent.act(turn)
    except Exception as e:
        rep = Reply(error=f"agent crashed: {type(e).__name__}: {e}")
    rep.latency = round(time.perf_counter() - t0, 3)
    return rep


# ---------------------------------------------------------------- tick

async def step(t: int, meta: dict, states: list[AgentState], agent) -> dict:
    cfg = meta["config"]
    world = meta["agents"]
    inputs = read_inputs(world, states, t, int(cfg.get("seam_period") or 1))   # everyone reads first
    turns, scores = [], []
    bank = registers.bank_for(meta) if registers.on(cfg) else None
    for st, inp in zip(states, inputs):
        changed = [x != y for x, y in zip(inp, st.prev_inputs)]
        st.history.append(inp)
        del st.history[:-HISTORY]
        if cfg.get("protocol_hint") == "tree" and cfg.get("say_only") and cfg.get("tree_check"):
            st.tree_in = [port_tree(v) if l and states[l["agent"]].out_types[l["port"]] == "tree" else {}
                          for v, l in zip(inp, world[st.id]["links"])]
        if cfg.get("protocol_hint") == "ring" and cfg.get("say_only"):
            st.ring_in = [port_ring(v) if l and states[l["agent"]].out_types[l["port"]] == "ring" else []
                          for v, l in zip(inp, world[st.id]["links"])]
            for d, ents in enumerate(st.ring_in):
                own = next((e for e in ents if e[0] == st.seed), None)
                if own and d not in st.ring_res:
                    st.ring_res[d] = (own[1], own[2])
        score = naming_score(inp, st.outputs[0]) if cfg.get("task") == "naming" else None
        scores.append(score)
        convo = cfg.get("memory") == "conversation"
        tpl = DELTA_TICK if (convo and st.convo) else meta["prompts"]["tick"]
        if bank and registers.is_gluon(world[st.id]) and registers.muons_only(cfg):
            turns.append(Turn(t=t, state=st, inputs=inp, changed=changed, system=registers.INERT, prompt=""))
            continue
        if bank and registers.is_gluon(world[st.id]):
            turns.append(Turn(t=t, state=st, inputs=inp, changed=changed, system=registers.gluon_system(cfg),
                              prompt=registers.gluon_tick(meta, bank, st.id, t, inp, st.outputs),
                              extra_body=registers.gluon_grammar(meta, st.id, HEXN, bank.programs[st.id])))
            continue
        turns.append(Turn(
            t=t, state=st, inputs=inp, changed=changed,
            system=system_prompt(meta, st.seed),
            history=list(st.convo) if convo else [],
            prompt=tick_prompt(tpl, t, st, inp, changed, cfg["log_lines_shown"], score,
                               intake=bridge_intake(world[st.id], states, t, int(cfg.get("seam_period") or 1)) if cfg.get("say_only") else None)
                   + (registers.tick_block(meta, bank, st.id, t) if bank else "")
                   + (bridge_state_block(st, bool(cfg.get("protocol_hint")), cfg.get("protocol_hint") == "ring") if cfg.get("say_only") else "")
                   + (action_blocks(st, inp, bridge=bool(cfg.get("say_only"))) if cfg.get("actions") else ""),
            images=port_pictures(inp, st.outputs, int(cfg.get("pixel_scale", 16)))[1] if PIXEL else [],
        ))

    replies = await asyncio.gather(*(safe_act(agent, tn) for tn in turns))

    records = []
    for tn, rep, score in zip(turns, replies, scores):          # then everyone writes
        st = tn.state
        errors = [rep.error] if rep.error else []
        alog, results = [], []
        refused = rep.error and rep.error.startswith("refusal")
        if (rep.error and not rep.raw) or refused:   # never act on a refused reply
            outs, lines, proto = list(st.outputs), [], st.protocol
            notices = [NO_REPLY_NOTICE]
        else:
            acted = registers.without_prose(rep.raw) if registers.muons_only(cfg) else rep.raw
            outs, lines, proto, perr, notices = parse_response(acted, st.outputs, st.protocol,
                                                               cfg["protocol_word_limit"], byte_ban(cfg),
                                                               ports=not cfg.get("say_only"))
            errors += perr
            if cfg.get("actions"):
                outs, results, alog, said = run_actions(acted, st, outs, ring_check=cfg.get("protocol_hint") == "ring",
                                                        tree_check=cfg.get("protocol_hint") == "tree" and bool(cfg.get("tree_check")))
                if said:
                    st.last_write_tick = t
                if said:   # a port written by say/ask needs no <qN> tag
                    errors = [e for e in errors if not any(e.startswith(f"q{i}:") for i in said)]
                    notices = [n for n in notices if not any(n.startswith(f"q{i}:") for i in said)]
                if alog and notices == [NO_REPLY_NOTICE]:
                    notices = []
        rec = {
            "agent": st.id,
            **({"pool": st.pool} if st.pool else {}),
            "inputs": tn.inputs,
            "changed": tn.changed,
            "outputs": outs,
            "log": lines,
            "protocol": proto,
            "protocol_changed": proto != st.protocol,
            "raw": rep.raw,
            "reasoning": rep.reasoning,
            "errors": errors,
            "notices": notices,          # shown to the agent at the next tick
            "latency": rep.latency,
            "usage": rep.usage,
            "request_id": rep.request_id,
        }
        if cfg.get("actions"):
            rec["actions"] = alog if not ((rep.error and not rep.raw) or refused) else []
            if cfg.get("say_only"):
                rec["out_types"] = list(st.out_types); rec["last_write_tick"] = st.last_write_tick
                rec["ledger"] = [list(map(list, x)) for x in st.ledger]; rec["notes"] = dict(st.notes)
                if st.seam_types:
                    rec["seam_types"] = {str(k): v for k, v in st.seam_types.items()}
                if st.ring_res:
                    rec["ring_res"] = {str(k): list(v) for k, v in st.ring_res.items()}
                if st.liar:
                    rec["liar_said"] = list(st.liar_said)
            rec["action_results"] = results if rec["actions"] else []
            st.action_results = rec["action_results"]
        if score is not None:
            rec["score"] = score         # naming task: earned on this tick's inputs, before writing
        if cfg.get("store_prompts") or cfg.get("memory") == "conversation":
            rec["prompt"] = tn.prompt
        if cfg.get("memory") == "conversation":
            rec["epoch_start"] = not tn.history
            ok = bool(rep.raw) and not refused
            rec["convo_appended"] = ok
            if ok:
                st.convo += [{"role": "user", "content": tn.prompt}, {"role": "assistant", "content": rep.raw}]
                u = rep.usage or {}
                st.convo_tokens = int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
            rec["convo_tokens"] = st.convo_tokens
            rec["convo_reset"] = st.convo_tokens > int(cfg.get("conversation_budget") or 24000)
            if rec["convo_reset"]:          # next tick starts a new epoch from the notebook
                st.convo, st.convo_tokens = [], 0
        records.append(rec)
        st.outputs, st.protocol, st.prev_inputs, st.notices = outs, proto, tn.inputs, notices
        st.log.extend((t, l) for l in lines)
    tick = {"type": "tick", "t": t, "time": time.time(), "agents": records}
    if bank:
        raws = [None if ((r.error and not r.raw) or (r.error and r.error.startswith("refusal"))) else liar_prose(st, r.raw or "")
                for st, r in zip(states, replies)]
        tick["reg"], res = registers.apply_tick(meta, bank, t, raws, [r["outputs"] for r in records])
        if bank.pending:
            await registers.settle_pending(meta, bank, t, agent, tick["reg"], Turn, AgentState, safe_act)
        for st, rec, r in zip(states, records, res):
            rec.update(reg_ok=r["reg_ok"], led_first=r["led_first"])
            rec["errors"] = rec["errors"] + r["errors"]
            rec["notices"] = rec["notices"] + r["notices"]
            st.notices = rec["notices"]
        if any(e["kind"] == "swap" for e in tick["reg"]["events"]):    # same state a resume would rebuild
            for st, a, rec in zip(states, world, records):
                st.module = a.get("module", 0)
                st.seam_in = {i: rec["inputs"][i] for i, l in enumerate(a["links"]) if l and l.get("seam")}
                st.seam_types = {i: states[l["agent"]].out_types[l["port"]] for i, l in enumerate(a["links"]) if l and l.get("seam")}   # the held type belongs to the new link
    return tick


# ---------------------------------------------------------------- run files

def new_run(cfg: dict, run_id: str | None) -> tuple[Path, dict]:
    if cfg.get("say_only") and not cfg.get("actions"):
        sys.exit("config: say_only needs actions = true, or nothing can write a port")
    if cfg.get("say_only") and registers.has_board(cfg):
        sys.exit("config: say_only (the bridge) and registers together give the nodes a second channel; pick one")
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{random.randrange(16**4):04x}"
    world_seed = cfg["world_seed"]
    if world_seed is None:
        world_seed = random.SystemRandom().getrandbits(48)
    rng = random.Random(world_seed)
    meta = {
        "type": "meta",
        "run_id": run_id,
        "created": time.time(),
        "world_seed": world_seed,
        "rows": cfg["rows"],
        "cols": cfg["cols"],
        "topology": cfg.get("topology", "grid"),
        "config": cfg,
        "prompts": {
            "system": (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8"),
            "tick": (ROOT / "prompts" / "tick.txt").read_text(encoding="utf-8"),
            "system_bridge": (ROOT / "prompts" / "system-bridge.txt").read_text(encoding="utf-8"),
        },
        "agents": (build_world_tesseract(int(cfg.get("side") or round(cfg["rows"] ** 0.5)), rng) if cfg.get("topology") == "tesseract"
                   else build_world(cfg["rows"], cfg["cols"], rng, torus=cfg.get("topology") == "torus")),
    }
    if cfg.get("modules"):
        assign_modules(meta["agents"], cfg["rows"], cfg["cols"], cfg["modules"])
    if cfg.get("task") == "sum":     # own rng, so the world is the same with or without the task
        srng = random.Random(f"{world_seed}:secrets")
        lo, hi = (int(x) for x in cfg.get("secret_range") or (1, 999))
        for a in meta["agents"]:
            a["secret"] = srng.randint(lo, hi)
        meta["answer"] = sum(a["secret"] for a in meta["agents"])
    if registers.on(cfg):
        registers.validate(cfg)
    registers.assign_pools(meta)
    if cfg.get("task") == "sum":
        players = [a for a in meta["agents"] if not registers.is_gluon(a)]      # gluons carry words; they are not players
        k = int(cfg.get("liars") or 0)
        if k:
            if k >= len(players):
                sys.exit(f"config: liars = {k} but there are only {len(players)} players")
            lrng = random.Random(f"{world_seed}:liars")
            for a in lrng.sample(players, k):
                a["liar"] = True
            meta["liars"] = sorted(a["seed"] for a in players if a.get("liar"))
        meta["answer"] = sum(a["secret"] for a in players if not a.get("liar"))
    if registers.muons_only(cfg) and "answer" in meta:
        meta["answer"] = sum(a["secret"] for a in meta["agents"] if not registers.is_gluon(a) and not a.get("liar"))   # liars have no number
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=False)
    path = d / "ticks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    return path, meta


def load_run(path: Path) -> tuple[dict, list[dict]]:
    """reads a ticks.jsonl, dropping (and truncating away) any half-written tail."""
    data = path.read_bytes()
    meta, ticks, good = None, [], 0
    pos = 0
    while pos < len(data):
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        line = data[pos:nl]
        try:
            rec = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            break
        if rec is not None:
            if rec.get("type") == "meta":
                meta = rec
            elif rec.get("type") == "tick":
                if ticks and rec["t"] != ticks[-1]["t"] + 1:
                    break
                ticks.append(rec)
        pos = good = nl + 1
    if good < len(data):
        print(f"note: dropping {len(data) - good} bytes of incomplete data at the end of {path}")
        with open(path, "r+b") as f:
            f.truncate(good)
    if meta is None:
        sys.exit(f"{path}: no meta record")
    return meta, ticks


def restore_states(meta: dict, ticks: list[dict]) -> list[AgentState]:
    if registers.on(meta["config"]):
        registers.restore(meta, ticks)   # also puts swapped nodes back where they were
    states = [AgentState(id=a["id"], seed=a["seed"], module=a.get("module", 0), pool=a.get("pool", ""), liar=bool(a.get("liar")))
              for a in meta["agents"]]
    for k, r in enumerate((meta["config"].get("tree_roots") or [0])[:2] if meta["config"].get("protocol_hint") == "tree" else []):
        states[int(r)].tree_root += "AB"[k]
    for st, a in zip(states, meta["agents"]):
        st.secret = int(a.get("secret") or 0)
    acting = meta["config"].get("actions")
    if acting:
        import tools
        for st in states:
            st.fs = tools.AgentFS()
    for rec in ticks:
        for st, a in zip(states, rec["agents"]):
            st.log.extend((rec["t"], l) for l in a["log"])
            st.history.append(a["inputs"])
            if acting:   # files are rebuilt by replaying every save, in order
                for act in a.get("actions", []):
                    if act["kind"] == "remember":
                        st.fs.call(act["tool"], name=act["name"], text=act["text"])
            if meta["config"].get("memory") == "conversation":
                if a.get("epoch_start"):
                    st.convo = []
                if a.get("convo_appended"):
                    st.convo += [{"role": "user", "content": a["prompt"]}, {"role": "assistant", "content": a["raw"]}]
                st.convo_tokens = a.get("convo_tokens", 0)
                if a.get("convo_reset"):
                    st.convo, st.convo_tokens = [], 0
    for st in states:
        del st.history[:-HISTORY]
    if ticks:
        for st, a in zip(states, ticks[-1]["agents"]):
            st.outputs = list(a["outputs"])
            st.protocol = a["protocol"]
            st.prev_inputs = list(a["inputs"])
            st.notices = list(a.get("notices", []))   # older runs have none
            st.action_results = list(a.get("action_results", []))
            st.out_types = list(a.get("out_types", st.out_types)); st.last_write_tick = int(a.get("last_write_tick", 0))
            st.ledger = [[tuple(x) for x in port] for port in a.get("ledger", st.ledger)]; st.notes = dict(a.get("notes", {}))
            st.liar_said = list(a.get("liar_said", []))
            st.seam_types = {int(k): v for k, v in a.get("seam_types", {}).items()}
            st.ring_res = {int(k): tuple(v) for k, v in a.get("ring_res", {}).items()}
        for st, w, a in zip(states, meta["agents"], ticks[-1]["agents"]):
            for i, l in enumerate(w["links"]):
                if l and l.get("seam"):
                    st.seam_in[i] = a["inputs"][i]
    return states


# ---------------------------------------------------------------- ratchet

_WORDS = re.compile(rb"[A-Za-z]")


def _is_text(v: str) -> str | None:
    """the same rule as experiments/quincy/ascii_text.py: printable, >= 4 chars, >= 3 letters, >= 3 distinct."""
    b = bytes.fromhex(v).strip(b"\x00")
    if len(b) < 4 or not all(32 <= c < 127 for c in b) or len(_WORDS.findall(b)) < 3:
        return None
    s = b.decode()
    return s if len(set(s.replace(" ", ""))) >= 3 else None


def interest(meta: dict, window: list[dict], before: list[dict]) -> tuple[bool, dict]:
    """is a run still interesting after this window of ticks? any one signal is enough:
    talk      >= 5% of agents put readable text on a port in the window and >= 3 new distinct messages appeared
    discovery the share of agents with a true neighbour's id in their notes rose by >= 0.05 over the window
    task      (sum task) more agents have the exact sum in their notes or reply at the window's end than at its start"""
    n = len(meta["agents"])
    seeds = [a["seed"] for a in meta["agents"]]
    nb = [{l["agent"] for l in a["links"] if l} for a in meta["agents"]]
    seen_before = {t for rec in before for a in rec["agents"] for v in a["outputs"] if (t := _is_text(v))}
    talkers, new = set(), set()
    for rec in window:
        for a in rec["agents"]:
            for v in a["outputs"]:
                t = _is_text(v)
                if t:
                    talkers.add(a["agent"])
                    if t not in seen_before:
                        new.add(t)

    def named(rec):
        return sum(any(seeds[j] in (a["protocol"] + " ".join(a["log"])).lower() for j in nb[i])
                   for i, a in enumerate(rec["agents"])) / n

    def right(rec):
        ans = str(meta.get("answer", ""))
        return sum(bool(ans) and ans in (a["protocol"] + " " + (a.get("raw") or "")) for a in rec["agents"])

    start = before[-1] if before else window[0]
    sig = {
        "talkers": len(talkers), "new_messages": len(new),
        "named_start": round(named(start), 3), "named_end": round(named(window[-1]), 3),
    }
    talk = len(talkers) >= 0.05 * n and len(new) >= 3
    discovery = sig["named_end"] - sig["named_start"] >= 0.05
    task = False
    if meta["config"].get("task") == "sum":
        sig["exact_sum_start"], sig["exact_sum_end"] = right(start), right(window[-1])
        task = sig["exact_sum_end"] > sig["exact_sum_start"]
    sig.update(talk=talk, discovery=discovery, task=task)
    reg = False
    if registers.has_board(meta["config"]):
        reg, reg_sig = registers.interesting(window, before, n)
        sig.update(reg_sig)
    return talk or discovery or task or reg, sig


def all_done(meta: dict, rec: dict) -> bool:
    """tree and ring arms: does every agent hold a typed total (a "total" entry for each tree on an output, or a ring3 note)?
    says nothing about whether the totals are right."""
    cfg = meta["config"]
    if cfg.get("protocol_hint") == "ring":
        return all("3" in (a.get("ring_res") or {}) or "ring3" in (a.get("notes") or {}) for a in rec["agents"])
    trees = set("AB"[:max(1, len((cfg.get("tree_roots") or [0])[:2]))])
    return all({x for v, k in zip(a["outputs"], a.get("out_types", [])) if k == "tree"
                for x, e in port_tree(v).items() if e[0] == "total"} >= trees for a in rec["agents"])


def max_score(meta: dict) -> int:
    """naming task: the best possible grid total, one point per live input port."""
    return sum(l is not None for a in meta["agents"] for l in a["links"])


async def run(path: Path, meta: dict, states: list[AgentState], start: int, end: int,
              ratchet: int = 0, history: list | None = None):
    """ratchet N: every N ticks of this invocation, stop the run unless interest() says it is still interesting."""
    agent = make_agent(meta["config"])
    done = list(history or [])          # earlier ticks (for "new message" and before/after comparisons)
    window: list[dict] = []
    try:
        with open(path, "a", encoding="utf-8") as f:
            for t in range(start, end + 1):
                t0 = time.perf_counter()
                rec = await step(t, meta, states, agent)
                if rec["agents"] and all(a["errors"] and not a.get("raw") for a in rec["agents"]):
                    # nobody got a reply (servers down, a lease ended): keep this tick out of the record
                    print(f"tick {t}: no agent got a reply: {rec['agents'][0]['errors'][0][:200]}. exiting 3; resume later",
                          flush=True)
                    sys.exit(3)
                window.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                bad = sum(bool(a["errors"]) for a in rec["agents"])
                sc = [a["score"] for a in rec["agents"] if "score" in a]
                extra = f"  score {sum(sc)}/{max_score(meta)}" if sc else ""
                try:  # a closed or broken console must not stop the run
                    print(f"tick {t:4d}  {time.perf_counter() - t0:6.2f}s  agents with errors: {bad}{extra}", flush=True)
                except (OSError, ValueError):
                    pass
                if meta["config"].get("stop_when_done") and meta["config"].get("protocol_hint") and all_done(meta, rec):
                    print(f"== every agent holds a total at t{t}: stopping", flush=True)
                    break
                if ratchet and len(window) >= ratchet and t < end:
                    keep, sig = interest(meta, window, done[-ratchet:])
                    verdict = {"tick": t, "window": ratchet, "keep": keep, **sig}
                    with open(path.parent / "ratchet.jsonl", "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(verdict) + "\n")
                    print(f"== ratchet at t{t}: {'interesting, continuing' if keep else 'not interesting, stopping'} {sig}",
                          flush=True)
                    if not keep:
                        break
                    done += window
                    window = []
    finally:
        await agent.close()


def open_viewer(path: Path):
    """start view.py on this run as its own process. it's detached from this
    console's ctrl-c, so stopping the sim leaves the window open for replay."""
    import subprocess
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    log = open(path.parent / "view.log", "a", encoding="utf-8")  # viewer errors land here
    subprocess.Popen([sys.executable, str(ROOT / "view.py"), str(path), "--live"],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     creationflags=flags, start_new_session=(os.name != "nt"))
    log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="toml config (default: config.toml if present)")
    ap.add_argument("--ticks", type=int, help="run until this tick number")
    ap.add_argument("--run-id", help="name for a new run")
    ap.add_argument("--resume", metavar="RUN", help="run id or path of a run to continue")
    ap.add_argument("--agent", choices=["llm", "anthropic", "claudecode", "echo", "regfake", "scripted"], help="override agent kind for a new run")
    ap.add_argument("--world-seed", type=int, help="override world seed for a new run")
    ap.add_argument("--rows", type=int, help="override grid rows for a new run")
    ap.add_argument("--cols", type=int, help="override grid cols for a new run")
    ap.add_argument("--topology", choices=["grid", "torus", "tesseract"], help="override topology for a new run (tesseract: a directed 4-torus, rows = cols = side**2)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value for a new run, e.g. --set prime='' or "
                         "--set llm.max_tokens=2048 (VALUE is read as toml, else as a string)")
    ap.add_argument("--view", action="store_true", help="also open the viewer, following this run live")
    ap.add_argument("--ratchet", type=int, default=0, metavar="N",
                    help="every N ticks, stop unless the run is still interesting (see interest())")
    args = ap.parse_args()
    load_dotenv()

    if args.resume:
        p = Path(args.resume)
        path = p if p.is_file() else (p / "ticks.jsonl" if p.is_dir() else RUNS / args.resume / "ticks.jsonl")
        meta, ticks = load_run(path)
        for kv in args.set:                 # overrides on resume too (e.g. llm.base_url when the run's server is gone)
            key, _, raw = kv.partition("=")
            try:
                val = tomllib.loads(f"v = {raw}")["v"]
            except tomllib.TOMLDecodeError:
                val = raw
            node = meta["config"]
            *kpath, last = key.strip().split(".")
            for part in kpath:
                node = node.setdefault(part, {})
            node[last] = val
        set_width(meta["config"])
        states = restore_states(meta, ticks)
        start = ticks[-1]["t"] + 1 if ticks else 1
    else:
        cfg = load_config(args.config)
        if args.agent:
            cfg["agent"] = args.agent
        if args.world_seed is not None:
            cfg["world_seed"] = args.world_seed
        for k in ("rows", "cols", "topology"):
            if getattr(args, k) is not None:
                cfg[k] = getattr(args, k)
        if args.ticks is not None:      # record the target so a later --resume runs to it too
            cfg["ticks"] = args.ticks
        for kv in args.set:
            key, _, raw = kv.partition("=")
            try:
                val = tomllib.loads(f"v = {raw}")["v"]
            except tomllib.TOMLDecodeError:
                val = raw
            node = cfg
            *path, last = key.strip().split(".")
            for part in path:
                node = node.setdefault(part, {})
            node[last] = val
        if cfg.get("topology", "grid") not in ("grid", "torus", "tesseract"):
            sys.exit(f"config: topology must be grid, torus or tesseract, got {cfg['topology']!r}")
        if cfg.get("topology") == "tesseract":
            side = int(cfg.get("side") or round(cfg["rows"] ** 0.5))
            if cfg["rows"] != side * side or cfg["cols"] != side * side:
                sys.exit(f"config: a tesseract of side {side} needs rows = cols = {side * side}, got {cfg['rows']}x{cfg['cols']}")
        set_width(cfg)
        path, meta = new_run(cfg, args.run_id)
        states = restore_states(meta, [])
        start = 1
    end = args.ticks if args.ticks is not None else meta["config"]["ticks"]

    print(f"run {meta['run_id']}  ({meta['config']['agent']})  ticks {start}..{end}  -> {path}")
    if args.view:
        open_viewer(path)
    try:
        asyncio.run(run(path, meta, states, start, end, args.ratchet,
                        ticks if args.resume else []))
    except KeyboardInterrupt:
        print(f"\nstopped. continue with: python sim.py --resume {meta['run_id']} --ticks {end}")


if __name__ == "__main__":
    main()
